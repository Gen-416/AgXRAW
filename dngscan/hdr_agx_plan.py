# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile an independent HdrAgxPlan from scene analysis and a display target.

The existing RenderPlan is used as an immutable carrier of shared scene intent. HDR copies
the relevant starting parameters into its own formation plan, derives its own white endpoint
from the reliable scene tail, and never consumes SDR pixels.

Only quantities whose meaning is already established are read. The reliable tail decides
how many stops the scene justifies, so reconstructed highlights cannot buy display range
the sensor never recorded, and the scene median is deliberately not consulted at all --
an HDR capacity must not re-expose a night scene.
"""
from __future__ import annotations

import math
from dataclasses import replace

from ._deps import np
from .constants import DARKTABLE_BASE_GAMMA, OUTPUT_REFERENCE_WHITE_STOPS
from .hdr_agx_math import (
    body_anchor_from_curve,
    compile_hdr_shoulder,
    requested_headroom_ev,
    validate_hdr_shoulder,
)
from .models import (
    Analysis,
    HdrAgxPlan,
    HdrColorGeometry,
    HdrDisplayTarget,
    HdrToneCurve,
    RenderPlan,
)


def reliable_tail_ev(plan: RenderPlan) -> float:
    """Scene EV of the highest luminance still backed by unclipped RAW evidence.

    There is deliberately no fallback to the reconstructed tail or the compiled white
    endpoint. Neither is RAW evidence, so using either would turn a missing measurement
    into positive HDR budget.
    """
    scene = getattr(plan, "scene", None)
    if scene is None:
        return float("nan")
    value = getattr(scene, "reliable_tail_ev_p9999", None)
    if value is not None and np.isfinite(value):
        return float(value)
    return float("nan")


# Prototype ceiling on channel separation. Not a calibrated value: the design lists
# Blender's HDR_purity=0.5 as a probe starting point whose semantics are not the same
# thing, so this stays a conservative cap until an EDR corpus says otherwise.
RHO_BASE = 0.5

# Project calibration policy, not values defined by darktable, AgX, Apple, ACES or
# ISO 21496-1. Naming them keeps a future EDR-corpus calibration from hiding in arithmetic.
MULTICHANNEL_CLIP_ZERO_CONFIDENCE_PCT = 10.0
P3_PRESSURE_ZERO_CONFIDENCE_PCT = 20.0
UNALIGNED_DECODER_RHO_CAP = 0.25
NORMAL_WHITE_MARGIN_EV = 0.30
SPARSE_EMITTER_WHITE_MARGIN_EV = 0.50
NORMAL_MINIMUM_WHITE_EV = 3.00
SPARSE_EMITTER_MINIMUM_WHITE_EV = 3.50
MAXIMUM_WHITE_EV = 8.50
# Where the HDR shoulder leaves the darktable body, in scene EV above mid gray. A small
# positive value keeps ordinary bright subject matter on the body's own contrast; sparse
# emitters start at the pivot because their highlights are the subject. Both are project
# latitude policy awaiting corpus calibration, not values defined by darktable or AgX.
NORMAL_SHOULDER_START_EV = 0.20
SPARSE_EMITTER_SHOULDER_START_EV = 0.00


def compile_channel_separation(
    analysis: Analysis,
    scene_decoder: str = "libraw",
    rho_base: float = RHO_BASE,
) -> float:
    """How much per-channel freedom the RAW evidence supports.

    Highlight chroma is only worth preserving where the sensor actually measured it. Each
    factor below withdraws that freedom for a different reason, and they multiply because
    any one of them alone is sufficient grounds for caution:

    - multi-channel CFA clipping means the hue up there is reconstructed, so separating
      channels would be inventing colour rather than keeping it;
    - a poor SNR ceiling means chroma in the tail is noise, and per-channel expansion
      amplifies exactly that noise;
    - heavy out-of-gamut pressure means the projector will be pulling chroma back anyway,
      so spending range on it just moves work downstream.

    The Core Image path is capped harder still. It has no aligned per-pixel CFA mask, so
    it cannot tell a clipped highlight from a bright one locally, and the design forbids
    fabricating a local rho from aggregate statistics.
    """
    # cell_ge2_of_clipped_pct is conditional on cells that clipped at all, so it reads as
    # a huge number on frames with almost no clipping -- 86 % of 0.08 % of the image. The
    # absolute share is what matters here, and cell_k_of_all_pct already carries it.
    k_all = getattr(analysis, "cell_k_of_all_pct", None) or {}
    multi_pct = sum(float(k_all.get(k, 0.0) or 0.0) for k in (2, 3, 4))
    if not k_all:
        clipped = float(getattr(analysis, "cell_union_pct", 0.0) or 0.0)
        conditional = float(getattr(analysis, "cell_ge2_of_clipped_pct", 0.0) or 0.0)
        multi_pct = clipped * conditional / 100.0
    # Multi-channel clipping is the decisive one: single-channel clipping still leaves two
    # measured channels to place the hue. 10 % of the frame is treated as total loss of
    # confidence, which is deliberately strict for a first cut.
    raw_confidence = float(
        np.clip(1.0 - multi_pct / MULTICHANNEL_CLIP_ZERO_CONFIDENCE_PCT, 0.0, 1.0)
    )

    # The design also lists an SNR confidence factor, and it is deliberately not applied
    # here. snr1_dr is only populated when analyze() runs with diagnostics=True, which a
    # plain --jpeg export does not, so including it would make the same photograph render
    # differently depending on whether --csv or --scan happened to be passed. A render
    # must not depend on an unrelated diagnostic flag. Restore this factor once SNR is
    # measured on the production path, not by inventing a fallback confidence for it.

    gamut = getattr(analysis, "gamut_out_pct", None) or {}
    out_pct = float(gamut.get("Display P3", gamut.get("P3", 0.0)) or 0.0)
    gamut_confidence = float(
        np.clip(1.0 - out_pct / P3_PRESSURE_ZERO_CONFIDENCE_PCT, 0.0, 1.0)
    )

    rho = float(rho_base) * raw_confidence * gamut_confidence
    if str(scene_decoder) != "libraw":
        rho = min(rho, UNALIGNED_DECODER_RHO_CAP)
    return float(np.clip(rho, 0.0, 1.0))


def compile_hdr_agx_plan(
    scene_plan: RenderPlan,
    target: HdrDisplayTarget | None = None,
    analysis: Analysis | None = None,
    scene_decoder: str = "libraw",
) -> HdrAgxPlan:
    """Compile scene-earned headroom into an independent native HDR AgX curve."""
    display = target if target is not None else HdrDisplayTarget()
    tail = reliable_tail_ev(scene_plan)
    scene = getattr(scene_plan, "scene", None)
    sparse = bool(getattr(scene, "sparse_emitter_tail", False))
    white_margin = SPARSE_EMITTER_WHITE_MARGIN_EV if sparse else NORMAL_WHITE_MARGIN_EV
    minimum_white = SPARSE_EMITTER_MINIMUM_WHITE_EV if sparse else NORMAL_MINIMUM_WHITE_EV
    white = (
        float(
            np.clip(
                max(tail + white_margin, minimum_white),
                minimum_white,
                MAXIMUM_WHITE_EV,
            )
        )
        if math.isfinite(tail)
        else minimum_white
    )
    headroom = float(display.display_headroom_ev)
    requested = requested_headroom_ev(tail, headroom)
    knee = SPARSE_EMITTER_SHOULDER_START_EV if sparse else NORMAL_SHOULDER_START_EV
    contrast = float(scene_plan.tone.contrast)

    # The shoulder is the only part of the curve headroom may touch, so the body is built
    # first and never revisited. HDR keeps a fixed pivot: a movable one would have to be
    # solved jointly with the shoulder anchors, and inheriting an SDR offset would move
    # the EV=0 output the whole coordinate system is anchored on.
    formation = replace(
        scene_plan.tone,
        tone_core="agx",
        white_ev=white,
        dynamic_range_ev=white - float(scene_plan.tone.black_ev),
        pivot_ev_offset=0.0,
        curve_gamma=DARKTABLE_BASE_GAMMA,
        target_white_linear=1.0,
    )

    peak_stops = OUTPUT_REFERENCE_WHITE_STOPS + requested
    # Anchor on the body that will actually render. K lands at or just past the darktable
    # curve's own shoulder transition on real plans, so the central-line closed form would
    # make the C1 join approximate rather than exact. Subdivision is enabled for the
    # authoritative plan: display headroom caps Z_peak independently of the tail-driven W
    # (H_content = min(H_display, H_signal)), so a low-headroom display with a long
    # reliable tail legitimately pushes alpha past the single-segment bound. That request
    # is not malformed -- it is an ordinary, strongly compressive shoulder -- and the
    # subdivided chain passes the same structural contract (C1 joins, pinned K tangent,
    # zero white tangent, per-piece monotonicity). Disabling HDR there would be the least
    # faithful choice on offer and a discontinuity in an otherwise continuous control.
    from .drt import c1_value_and_derivative_at_ev

    def _body_anchor(ev: float) -> tuple[float, float]:
        return c1_value_and_derivative_at_ev(ev, formation)

    segments = (
        compile_hdr_shoulder(
            knee,
            white,
            peak_stops,
            contrast,
            evaluate_body_with_derivative=_body_anchor,
            allow_subdivision=True,
        )
        if requested > 0.0
        else ()
    )
    _, knee_stops, knee_slope = body_anchor_from_curve(_body_anchor, knee)
    rendered = requested
    if segments:
        ok, reason = validate_hdr_shoulder(segments, knee_slope, peak_stops)
        if not ok:
            # A shoulder that fails its own structural contract is not rendered at reduced
            # strength; the plan reports no HDR. Degrading here would ship a curve whose
            # C1 join or monotonicity was never established.
            segments, rendered = (), 0.0
    else:
        rendered = 0.0

    # Diagnostic: the request's own normalized start tangent, before any subdivision.
    # Above 3 it records that the single-segment family could not span this geometry and
    # the compiled chain is the subdivided one; the compiler never re-reads it.
    span_e = white - knee
    span_z = peak_stops - knee_stops
    request_alpha = (
        knee_slope * span_e / span_z
        if requested > 0.0 and span_e > 0.0 and span_z > 1e-12 and math.isfinite(knee_slope)
        else float("nan")
    )

    tone = HdrToneCurve(
        black_ev=float(formation.black_ev),
        shoulder_start_ev=float(knee),
        white_ev=white,
        body_gamma=DARKTABLE_BASE_GAMMA,
        body_contrast=contrast,
        toe_power=float(formation.toe_power),
        reference_white_stops=OUTPUT_REFERENCE_WHITE_STOPS,
        display_headroom_ev=headroom,
        requested_headroom_ev=requested,
        rendered_headroom_ev=rendered,
        peak_linear=2.0 ** rendered,
        reliable_tail_ev=tail,
        white_margin_ev=float(white_margin),
        shoulder_segments=tuple(segments),
        shoulder_alpha=request_alpha,
    )
    # Compiled from RAW evidence when it is available. Without an Analysis there is no
    # evidence to justify per-channel freedom, so the answer is none rather than a guess:
    # a neutral HDR is always defensible, an invented highlight hue is not.
    rho = (
        compile_channel_separation(analysis, scene_decoder)
        if analysis is not None
        else 0.0
    )
    color = HdrColorGeometry(
        channel_separation=rho,
        raw_clip_retreat=float(scene_plan.color.raw_clip_retreat_strength)
        if getattr(scene_plan, "color", None) is not None
        else 0.0,
        # Neutral until a production-path per-channel tail SNR metric exists. Keeping this
        # explicit is safer than making render output depend on diagnostics=True.
        snr_gate=1.0,
        hue_restore=float(scene_plan.tone.hue_restore),
        primaries_preset=str(scene_plan.tone.agx_primaries),
        gamut_fit_margin=0.0,
    )
    return HdrAgxPlan(formation=formation, display=display, tone=tone, color=color)


def describe_hdr_plan(plan: HdrAgxPlan) -> str:
    """One line naming capture request, solved endpoint and display capacity."""
    tone = plan.tone
    if tone.rendered_headroom_ev <= 0.0:
        tail_text = (
            f"{tone.reliable_tail_ev:+.2f}EV"
            if math.isfinite(tone.reliable_tail_ev)
            else "不可用"
        )
        return (
            f"HDR: 无可用扩展白（可靠尾部 {tail_text}，"
            f"显示容量 +{tone.display_headroom_ev:.2f}EV）"
        )
    reduced = (
        f"，RAW 请求 +{tone.requested_headroom_ev:.2f}EV"
        if tone.rendered_headroom_ev + 1e-4 < tone.requested_headroom_ev
        else ""
    )
    count = len(tone.shoulder_segments)
    shape = "单段 shoulder" if count <= 1 else f"细分 shoulder（{count} 段）"
    return (
        f"HDR: 原生 AgX 白点 +{tone.rendered_headroom_ev:.2f}EV / "
        f"容量 +{tone.display_headroom_ev:.2f}EV{reduced}；"
        f"K {tone.shoulder_start_ev:+.2f}EV / W {tone.white_ev:+.2f}EV，"
        f"alpha {tone.shoulder_alpha:.3f}，{shape}，"
        f"可靠尾部 {tone.reliable_tail_ev:+.2f}EV"
    )
