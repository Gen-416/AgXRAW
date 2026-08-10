# SPDX-License-Identifier: GPL-3.0-or-later
"""Exposure gain and analysis-driven tone compression plans."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from ._deps import np
from . import agx as agx_engine
from . import scene_transform as scene_transform_engine
from .color import (
    apply_rgb_matrix3, clamp_float, luminance_from_rgb_space, output_gamut_space,
    rec2020_to_xyz, XYZ_TO_RGB,
)
from .constants import EPS, EV_REPORT_FLOOR, GAMUT_EPS, GRAY_EV, MIDGRAY_HEADROOM_STOPS
from . import retreat as retreat_engine
from .models import (
    Analysis, ColorGeometryPlan, RawBundle, RenderAdjustments, RenderPlan,
    SceneToneMetrics, ToneCompressionPlan,
)

TONE_CORE_CHOICES = ("gated", "agx", "lum", "neutral")
LUM_NORM_CHOICES = ("y", "power", "max")
ENDPOINT_MODE_CHOICES = ("adaptive", "evidence")


def exposure_mode_for_tone_core(tone_core: str) -> str:
    """Exposure anchor for every tone core.

    All cores including neutral share the same fixed mid-gray reference so manual/auto
    EV means the same thing when A/B-ing against AgX. Only the tone operator differs.
    """
    return "agx"


def neutral_tone_plan(target_gamut: str) -> ToneCompressionPlan:
    """Fixed Y-ratio diagnostic curve, not a production or camera-render baseline.

    Endpoints are constants, not compiled from scene body/tail statistics. The operator
    is luminance-ratio compression only; AgX inset/outset and scene C1 planning are
    skipped. It intentionally exposes what ratio-preserving color does near a narrow
    output-gamut boundary; saturated highlights may look harder or more neon than AgX.
    Shared EV anchor and delivery gamut fit still apply. CFA clip retreat applies only
    when the selected capture decoder has a spatial mask.
    """
    from .neutral import (
        NEUTRAL_BLACK_EV, NEUTRAL_CONTRAST, NEUTRAL_SHOULDER_POWER,
        NEUTRAL_TOE_POWER, NEUTRAL_WHITE_EV,
    )

    return ToneCompressionPlan(
        target_gamut=target_gamut,
        luma_p1=0.0,
        luma_p50=0.0,
        luma_p99=0.0,
        luma_p999=0.0,
        black_ev=NEUTRAL_BLACK_EV,
        white_ev=NEUTRAL_WHITE_EV,
        dynamic_range_ev=NEUTRAL_WHITE_EV - NEUTRAL_BLACK_EV,
        contrast=NEUTRAL_CONTRAST,
        toe_power=NEUTRAL_TOE_POWER,
        shoulder_power=NEUTRAL_SHOULDER_POWER,
        chroma_p95=0.0,
        negative_rgb_pct=0.0,
        over_rgb_pct=0.0,
        tone_core="neutral",
        lum_norm="y",
        use_c1_endpoints=False,
        view_brightness=1.0,
    )

def compute_exposure_gain(mode: str, ev: float) -> float:
    """Constant, content-independent exposure anchor plus manual EV compensation.

    Every tone core uses the same anchor: a nominally-exposed mid gray (~clip /
    2**headroom) maps to 0.18 scene-linear Rec.2020 at EV=0, then manual EV scales
    from there. This is a fixed scalar, never derived from scene content.
    """
    manual = 2.0 ** float(ev)
    return 0.18 * (2.0 ** MIDGRAY_HEADROOM_STOPS) * manual


def scene_rec2020_to_float(
    values: Any,
    scene_scale: float,
    gain: float = 1.0,
    *,
    contract: Any | None = None,
) -> Any:
    """Convert stored decoder RGB to intent-scene float32.

    Prefer passing ``contract`` (:class:`~dngscan.models.SceneScaleContract`) so
    callers do not have to reassemble ``storage_scale`` and ``total_render_gain``.
    The legacy ``scene_scale`` / ``gain`` pair remains for migration callers and
    must stay bit-identical to the contract path.
    """
    if contract is not None:
        scale = float(contract.storage_scale)
        gain = float(contract.total_render_gain)
    else:
        scale = float(scene_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    # A float decoder can legitimately carry scene_scale below one (for example when an
    # A/B alignment gain is above one). Clamping the divisor to one would make analysis
    # and rendering disagree exactly in that case.
    rgb = values.astype(np.float32, copy=False) / np.float32(scale)
    if gain != 1.0:
        rgb = rgb * np.float32(gain)
    return np.nan_to_num(rgb, nan=0.0, posinf=1e6, neginf=0.0)


def scene_intent_rec2020(values: Any, bundle: Any, gain: float | None = None) -> Any:
    """Storage RGB -> intent-scene float, viewed through any declared lens filter.

    The filter multiplies here — before tone metrics, HDR budgeting and both display
    formations — because glass in front of the lens is capture, not a look: film would
    have metered, exposed and clipped through it too. The mired shift acts on the
    rendered balance and is position-invariant, so no illuminant parameter is needed;
    see lens_filter.lens_filter_matrix for the coordinate semantics.
    """
    rec = scene_rec2020_to_float(
        values,
        bundle.scene_scale,
        bundle.exposure_gain if gain is None else gain,
    )
    name = getattr(bundle, "lens_filter", "none") or "none"
    if name != "none":
        from .lens_filter import apply_lens_filter_rec2020

        rec = apply_lens_filter_rec2020(rec, name)
    return rec


def subsample_step(pixel_count: int, max_samples: int = 800_000) -> int:
    return max(1, int(math.ceil(pixel_count / max_samples)))


def rank_trim_reconstructed_highlights(
    ev: Any, valid: Any, clipped_cell_pct: float
) -> Any:
    """Exclude a RAW-measured fraction of the brightest reconstructed samples.

    Core Image applies DNG warps, so LibRaw's CFA clip mask cannot be mapped to its
    pixels without a calibrated geometric transform. The aggregate clipped-cell rate is
    still valid. Removing that fraction from the top of the luminance ranking restores
    the body/tail contract without pretending that the two frames align spatially.
    """
    values = np.asarray(ev, dtype=np.float32)
    keep = np.asarray(valid, dtype=bool).copy()
    indices = np.flatnonzero(keep)
    if indices.size == 0:
        return keep
    fraction = clamp_float(float(clipped_cell_pct) / 100.0, 0.0, 1.0)
    trim_count = int(math.ceil(indices.size * fraction))
    min_keep = max(256, indices.size // 20)
    trim_count = min(trim_count, max(0, indices.size - min_keep))
    if trim_count <= 0:
        return keep
    ranked = values[indices]
    top = np.argpartition(ranked, ranked.size - trim_count)[-trim_count:]
    keep[indices[top]] = False
    return keep


def tone_plan_sample_scene_rec2020(
    bundle: RawBundle,
    max_samples: int = 800_000,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    exposure_gain: float | None = None,
) -> Any:
    flat = bundle.scene_rec2020_render.reshape(-1, bundle.scene_rec2020_render.shape[-1])
    step = subsample_step(flat.shape[0], max_samples)
    gain = bundle.exposure_gain if exposure_gain is None else exposure_gain
    rec2020 = scene_intent_rec2020(flat[::step, :3], bundle, gain)
    wb_adapt = scene_transform_engine.window_transport(bundle)
    return scene_transform_engine.apply_scene_transform_rec2020(
        rec2020, scene_transform, scene_transform_strength, wb_adapt
    )


def reliable_scene_ev_selection(
    bundle: RawBundle,
    analysis: Analysis,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    exposure_gain: float | None = None,
    max_samples: int = 800_000,
    rec2020_sample: Any | None = None,
) -> tuple[Any, Any, Any, bool]:
    """The single declared reliable-sample selection behind every scene decision.

    Returns ``(ev, body_mask, evidence_mask, evidence_ok)`` where ``ev`` is the
    subsampled intent-scene luminance in EV relative to 18% gray, ``evidence_mask``
    marks samples trustworthy enough to grant headroom (RAW-clip and floor-clamp
    excluded, no fallback), ``body_mask`` is the possibly-fallback population SDR
    body percentiles may use, and ``evidence_ok`` says whether the evidence mask
    kept enough samples to speak for a reliable tail.

    Tone planning (:func:`scene_tone_metrics`) and any observer of the planner's
    inputs — for example the GUI's scene EV histogram — must consume this one
    function so the display can never describe data the render did not see.

    Samples sitting at the representable floor are clamped values, not measurements:
    the decoder produced zero (or below one code value) and the clip above pinned them
    to EV_REPORT_FLOOR. Letting them into the body percentiles is the black-end twin of
    letting reconstructed highlights define the white point, so they are excluded while
    enough real samples remain.

    (That exclusion was written when the Core Image path left 1.78 % of pixels on the
    floor against LibRaw's 0.001 %, and the cause was misread as Apple's black handling.
    It was not: CIRAWFilter's shadowBias defaults to 5.0 and had not been zeroed, and
    once it was (see coreimage_decode) that path leaves 0.006..0.18 %, below LibRaw's
    own 0.16..1.9 %. The exclusion stays because it is right for any decoder — LibRaw
    reaches the floor too, and a clamped sample is not evidence whoever produced it —
    but it is a correctness guard, not a workaround for one back end.)
    """
    flat = bundle.scene_rec2020_render.reshape(-1, bundle.scene_rec2020_render.shape[-1])
    step = subsample_step(flat.shape[0], max_samples)
    if rec2020_sample is None:
        gain = bundle.exposure_gain if exposure_gain is None else exposure_gain
        rec = tone_plan_sample_scene_rec2020(
            bundle,
            max_samples=max_samples,
            scene_transform=scene_transform,
            scene_transform_strength=scene_transform_strength,
            exposure_gain=gain,
        )
    else:
        rec = np.asarray(rec2020_sample)
        expected = flat[::step, :3].shape
        if rec.shape != expected:
            raise ValueError(
                f"prepared tone-plan sample shape {rec.shape} does not match {expected}"
            )
    y = np.clip(rec2020_to_xyz(rec)[:, 1], 2.0 ** EV_REPORT_FLOOR, None)
    ev = np.log2(y) - GRAY_EV

    floor_ev = float(EV_REPORT_FLOOR) - GRAY_EV
    above_floor = ev > (floor_ev + 1e-3)
    reliable = above_floor.copy()
    if getattr(bundle, "clip_masks", None) is not None:
        masks = retreat_engine.clip_masks_for_shape(bundle, bundle.scene_rec2020_render.shape[:2])
        reliable &= np.max(masks.reshape(-1, 3)[::step], axis=1) < np.float32(0.10)
    elif getattr(bundle, "scene_decoder", "libraw") == "coreimage":
        # RAW 9's reconstructed highlight pixels are geometrically warped relative to
        # the CFA mosaic. Use the full-resolution RAW clipped-cell rate as a rank-domain
        # constraint so those invented values cannot compile the global white endpoint.
        reliable = rank_trim_reconstructed_highlights(ev, reliable, analysis.cell_union_pct)

    # Keep evidence authority separate from the fallback needed to compile a usable SDR
    # curve. If fewer than 5% (and at least 256) trustworthy samples remain, SDR may still
    # use the broader distribution for a defensive endpoint, but that fallback is not
    # allowed to call itself a reliable tail and grant HDR headroom.
    min_reliable = max(256, ev.size // 20)
    evidence_reliable = reliable
    evidence_ok = int(np.count_nonzero(evidence_reliable)) >= min_reliable

    body = reliable
    if int(np.count_nonzero(body)) < min_reliable:
        body = above_floor if int(np.count_nonzero(above_floor)) >= 256 else np.ones_like(above_floor)
    if int(np.count_nonzero(body)) < min_reliable:
        body = np.ones((ev.shape[0],), dtype=bool)
    return ev, body, evidence_reliable, evidence_ok


def scene_tone_metrics(
    bundle: RawBundle,
    analysis: Analysis,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    plan_exposure_gain: float | None = None,
    max_samples: int = 800_000,
    rec2020_sample: Any | None = None,
) -> SceneToneMetrics:
    """Measure the reliable scene body separately from its highlight tail.

    Reconstruction may make a clipped lamp visually plausible, but it cannot restore its
    sensor headroom. On LibRaw we therefore exclude soft CFA-clipped sites from body
    percentiles. Core Image's opcode geometry prevents spatial reuse, so that path removes
    the aggregate clipped-cell fraction from the brightest luminance ranks instead. The
    complete rendered tail remains available only for topology classification. The sample
    selection itself lives in :func:`reliable_scene_ev_selection` and is shared with the
    GUI's scene EV histogram by declaration.
    """
    ev, body_mask, evidence_reliable, evidence_ok = reliable_scene_ev_selection(
        bundle,
        analysis,
        scene_transform,
        scene_transform_strength,
        plan_exposure_gain,
        max_samples,
        rec2020_sample,
    )
    reliable_sample_pct = float(np.mean(evidence_reliable) * 100.0)
    reliable_tail_p9999 = (
        float(np.percentile(ev[evidence_reliable], 99.99))
        if evidence_ok
        else float("nan")
    )
    reliable_ev = ev[body_mask]

    p1, p5, p50, p95, p99, p999 = [
        float(v) for v in np.percentile(reliable_ev, [1.0, 5.0, 50.0, 95.0, 99.0, 99.9])
    ]
    tail_p9999 = float(np.percentile(ev, 99.99))
    tail0 = float(np.mean(ev > 0.0) * 100.0)
    tail2 = float(np.mean(ev > 2.0) * 100.0)
    extremity = tail2 / max(tail0, 1e-4)
    sparse_emitter = bool(tail0 < 3.0 and extremity > 0.12)
    return SceneToneMetrics(
        reliable_sample_pct=reliable_sample_pct,
        body_ev_p1=p1,
        body_ev_p5=p5,
        body_ev_p50=p50,
        body_ev_p95=p95,
        body_ev_p99=p99,
        body_ev_p999=p999,
        tail_ev_p9999=tail_p9999,
        tail_area_ev0_pct=tail0,
        tail_area_ev2_pct=tail2,
        tail_extremity=extremity,
        sparse_emitter_tail=sparse_emitter,
        raw_clip_union_pct=float(analysis.cell_union_pct),
        reliable_tail_ev_p9999=reliable_tail_p9999,
    )


def build_color_geometry_plan(
    analysis: Analysis, output_gamut: str, tone_core: str = "agx"
) -> ColorGeometryPlan:
    space = output_gamut_space(output_gamut)
    pressure = float(analysis.gamut_out_pct.get(space, 0.0))
    # The output fit reacts slightly sooner in the smaller sRGB container and grows its
    # adaptive-L0 safety margin as measured output-gamut pressure rises. It remains a
    # colour-only decision: no tone endpoint or contrast parameter reads this value.
    base_alpha = 0.045 if output_gamut == "p3" else 0.060
    alpha = base_alpha + 0.015 * clamp_float(pressure / 5.0, 0.0, 1.0)
    if tone_core == "gated":
        noise_floor = -12.0
        if math.isfinite(analysis.usable_dr_eff_ev):
            noise_floor = -float(analysis.usable_dr_eff_ev) - 1.0
        return ColorGeometryPlan(
            target_gamut=output_gamut,
            raw_clip_retreat_strength=0.0,
            output_gamut_pressure_pct=pressure,
            gamut_fit_alpha=alpha,
            display_highlight_chroma_retreat=0.28,
            color_path_master=1.0,
            gated_midtone_protect=0.92,
            color_path_highlight_ev_lo=0.25,
            color_path_highlight_ev_hi=2.75,
            gated_noise_ev_floor=noise_floor,
        )
    return ColorGeometryPlan(
        target_gamut=output_gamut,
        # Every non-gated core requests clip retreat, but it executes only when the
        # capture decoder supplied a spatial CFA mask. RAW 9 has aggregate evidence only.
        raw_clip_retreat_strength=1.0,
        output_gamut_pressure_pct=pressure,
        gamut_fit_alpha=alpha,
        display_highlight_chroma_retreat=0.35 if tone_core == "lum" else 0.0,
    )


def _smoothstep_f(edge0: float, edge1: float, x: float) -> float:
    """Scalar smoothstep retained for the separate AgX colour-punch gate."""
    t = clamp_float((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_tone_compression_plan(
    bundle: RawBundle,
    analysis: Analysis,
    target_gamut: str,
    ev_from_agx_inset: bool = False,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    punch_scale: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    plan_exposure_gain: float | None = None,
    scene_metrics: SceneToneMetrics | None = None,
    endpoint_mode: str = "adaptive",
    rec2020_sample: Any | None = None,
) -> ToneCompressionPlan:
    agx_primaries = agx_engine.resolve_agx_primaries(agx_primaries)
    endpoint_mode = endpoint_mode if endpoint_mode in ENDPOINT_MODE_CHOICES else "adaptive"
    if tone_core == "neutral":
        return neutral_tone_plan(target_gamut)

    plan_gain = plan_exposure_gain if plan_exposure_gain is not None else bundle.exposure_gain
    metrics = scene_metrics if scene_metrics is not None else scene_tone_metrics(
        bundle,
        analysis,
        scene_transform,
        scene_transform_strength,
        plan_gain,
    )
    rec2020 = (
        tone_plan_sample_scene_rec2020(
            bundle,
            scene_transform=scene_transform,
            scene_transform_strength=scene_transform_strength,
            exposure_gain=plan_gain,
        )
        if rec2020_sample is None
        else np.asarray(rec2020_sample)
    )
    xyz = rec2020_to_xyz(rec2020)
    y = np.clip(xyz[:, 1], 0.0, None)
    ev_p1 = metrics.body_ev_p1
    ev_p50 = metrics.body_ev_p50
    ev_p99 = metrics.body_ev_p99
    ev_p999 = metrics.body_ev_p999
    luma_p1, luma_p50, luma_p99, luma_p999 = [float(v) for v in np.percentile(y, [1.0, 50.0, 99.0, 99.9])]

    plan_dr = analysis.usable_dr_eff_ev if math.isfinite(analysis.usable_dr_eff_ev) else analysis.usable_dr_ev
    if math.isfinite(plan_dr):
        noise_limited_black = -plan_dr - 1.5
    else:
        noise_limited_black = -12.0
    black_ev = max(ev_p1 - 0.25, noise_limited_black)
    black_ev = clamp_float(black_ev, -14.0, -1.5)
    # darktable's C1 curve starts toe/shoulder at the pivot by default. Do not map a
    # dark scene's p95 directly to the shoulder: when p95 < 0 EV, that segment crosses
    # the calibrated pivot and creates exactly the dark-frame / glaring-lamp failure.
    latitude_lo_ev = 0.10
    latitude_hi_ev = 0.20 if not metrics.sparse_emitter_tail else 0.0
    toe_start_ev = -latitude_lo_ev
    shoulder_start_ev = latitude_hi_ev

    # The complete tail describes topology (for example, sparse emitters), but has no
    # authority over the global white endpoint: reconstructed/RAW-clipped values are not
    # measured scene radiometry. Only the reliable tail may set the shoulder endpoint.
    white_margin = 0.50 if metrics.sparse_emitter_tail else 0.30
    min_white_ev = 3.50 if metrics.sparse_emitter_tail else 3.00
    reliable_white_tail = metrics.reliable_tail_ev_p9999
    if not math.isfinite(reliable_white_tail):
        reliable_white_tail = metrics.tail_ev_p9999
    white_ev = max(reliable_white_tail + white_margin, min_white_ev)
    white_ev = clamp_float(white_ev, min_white_ev, 8.5)

    # Evidence endpoint mode: pin the endpoints to what the sensor measured instead of
    # where the scene percentiles happen to sit. The black endpoint follows the noise
    # floor (prior read-noise when a sensor prior exists, single-frame estimate
    # otherwise); the white endpoint may consult ONLY the reliable RAW tail — never the
    # reconstructed one — while keeping the same defensive margin/minimum-white guards
    # as adaptive so the shoulder side of the curve does not collapse onto the subject.
    # Every degradation is recorded truthfully instead of silently falling back.
    endpoint_note: str | None = None
    if endpoint_mode == "evidence":
        from .analysis import noise_floor_ev_estimate

        notes: list[str] = []
        floor_ev_est, floor_source = noise_floor_ev_estimate(analysis)
        if math.isfinite(floor_ev_est):
            black_ev = clamp_float(floor_ev_est, -14.0, -1.5)
            if floor_source == "prior":
                notes.append("黑端点=先验读出噪声底")
            else:
                notes.append("黑端点=单帧噪声底估计（无传感器先验）")
        else:
            notes.append("黑端点证据缺席，沿用自适应端点")
        if math.isfinite(metrics.reliable_tail_ev_p9999):
            white_ev = clamp_float(
                max(metrics.reliable_tail_ev_p9999 + white_margin, min_white_ev),
                min_white_ev,
                8.5,
            )
            if metrics.reliable_tail_ev_p9999 + white_margin < min_white_ev:
                notes.append(f"白端点=可靠尾部（受最低白点 +{min_white_ev:.2f}EV 保护）")
            else:
                notes.append("白端点=可靠尾部 p99.99")
        else:
            notes.append("白端点证据缺席，回退自适应白点（含重建尾部）")
        endpoint_note = "；".join(notes)

    # These are strictly tone decisions. Colour clipping and output gamut live in
    # ColorGeometryPlan and must not change either curve endpoint or pivot contrast.
    dynamic_range_ev = white_ev - black_ev
    contrast = 3.0
    dark_body = clamp_float((-metrics.body_ev_p50 - 1.5) / 3.0, 0.0, 1.0)
    toe_power = 1.50 - 0.35 * dark_body
    shoulder_power = 2.55 if metrics.sparse_emitter_tail else 2.90
    # Scene-adaptive pivot stays OFF, now for a measured reason rather than an unsolved
    # constraint. agx.curve_params can hold the EV0 -> 18% anchor while the pivot moves
    # (bisection on the pivot output, see Ev0AnchorSolverTest), but measuring both ends
    # of that trade on a -3.4 EV night frame refutes the idea itself: anchoring EV0
    # crushes the subject (output at -2 EV falls 0.024 -> 0.007) for no contrast gain,
    # while preserving subject brightness instead drives EV0 to 0.95 — nearly white.
    # A pivot move needs contrast/toe/shoulder re-solved with it (darktable relies on
    # the user for exactly that); one automatic knob cannot do it. Left compiled-in and
    # tested so the capability is ready if that 2-D solve is ever attempted.
    pivot_ev_offset = 0.0
    target_black_linear = 0.0
    shadow_quality = _smoothstep_f(5.5, 8.5, plan_dr) if math.isfinite(plan_dr) else 0.5
    view_brightness = 1.0 + 0.30 * dark_body * shadow_quality
    # Punch is a post-core chroma operator, not a tone decision: it is calculated after
    # endpoint selection and cannot feed back into pivot, toe, shoulder or exposure.
    # The luminance core deliberately stays at zero because it already retains the
    # original RGB ratio through the body; neutral is a diagnostic reference.
    if tone_core in ("agx", "gated"):
        w_bright = _smoothstep_f(-3.0, -1.2, metrics.body_ev_p50)
        w_quality = _smoothstep_f(7.5, 9.5, plan_dr) if math.isfinite(plan_dr) else 0.5
        w_dr = _smoothstep_f(6.5, 8.0, dynamic_range_ev)
        punch_strength = clamp_float(
            w_bright * w_quality * (0.55 + 0.45 * w_dr) * clamp_float(punch_scale, 0.0, 1.5),
            0.0,
            1.0,
        )
    else:
        punch_strength = 0.0

    if target_gamut == "Rec2020":
        rgb = rec2020
    else:
        rgb = apply_rgb_matrix3(xyz, XYZ_TO_RGB[target_gamut])
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1e6, neginf=-1e6)
    anchor = np.maximum(y, 0.0)
    chroma_ratio = np.max(np.abs(rgb - anchor[:, None]), axis=1) / np.maximum(anchor, EPS)
    finite_chroma = chroma_ratio[np.isfinite(chroma_ratio) & (anchor > 2.0 ** EV_REPORT_FLOOR)]
    chroma_p95 = float(np.percentile(finite_chroma, 95.0)) if finite_chroma.size else 0.0

    negative_rgb_pct = float(np.mean(np.min(rgb, axis=1) < -GAMUT_EPS) * 100.0)
    over_rgb_pct = float(np.mean(np.max(rgb, axis=1) > 1.0 + GAMUT_EPS) * 100.0)
    # The pinned darktable scene default uses Blender-like/base geometry with 60% hue
    # restoration. Its sigmoid-like smooth preset deliberately disables hue restoration.
    hue_restore = 0.0 if agx_primaries == "smooth" else 0.6

    return ToneCompressionPlan(
        target_gamut=target_gamut,
        luma_p1=luma_p1,
        luma_p50=luma_p50,
        luma_p99=luma_p99,
        luma_p999=luma_p999,
        black_ev=black_ev,
        white_ev=white_ev,
        dynamic_range_ev=dynamic_range_ev,
        contrast=contrast,
        toe_power=toe_power,
        shoulder_power=shoulder_power,
        latitude_lo_ev=latitude_lo_ev,
        latitude_hi_ev=latitude_hi_ev,
        punch_strength=punch_strength,
        chroma_p95=chroma_p95,
        negative_rgb_pct=negative_rgb_pct,
        over_rgb_pct=over_rgb_pct,
        tone_core=tone_core,
        lum_norm=lum_norm,
        pivot_ev_offset=pivot_ev_offset,
        target_black_linear=target_black_linear,
        target_white_linear=1.0,
        agx_primaries=agx_primaries,
        hue_restore=hue_restore,
        toe_start_ev=toe_start_ev,
        shoulder_start_ev=shoulder_start_ev,
        use_c1_endpoints=True,
        view_brightness=view_brightness,
        endpoint_mode=endpoint_mode,
        endpoint_note=endpoint_note,
    )


def apply_render_adjustments(
    plan: RenderPlan, adjustments: RenderAdjustments | None
) -> RenderPlan:
    """Apply restrained user biases without recompiling the scene analysis.

    The calibrated pivot and scene-compiled endpoints remain authoritative. Tone controls alter
    only the local curve shape; highlight fade is a display-side chroma control. The
    fixed neutral reference intentionally ignores all of these adjustments.
    """
    if (
        adjustments is None
        or adjustments.is_identity()
        or plan.tone.tone_core == "neutral"
    ):
        return plan

    brightness_bias = clamp_float(float(adjustments.midtone_brightness), -1.0, 1.0)
    contrast_bias = clamp_float(float(adjustments.midtone_contrast), -1.0, 1.0)
    shadow_bias = clamp_float(float(adjustments.shadow_transition), -1.0, 1.0)
    highlight_bias = clamp_float(float(adjustments.highlight_transition), -1.0, 1.0)
    fade_bias = clamp_float(float(adjustments.highlight_fade), -1.0, 1.0)
    if (
        str(getattr(plan.tone, "film_mode", "observe")) == "full"
        and str(getattr(plan.tone, "curve_preset", "none")) != "none"
    ):
        # The takeover LUT owns the highlight colour path entirely; a stale
        # highlight-fade value from before the mode switch measurably altered
        # full-mode exports (0.129 max linear channel diff on _SDI0222 +
        # Velvia) through a control the GUI shows as disabled. Forced off at
        # the compiler so no payload can smuggle it in (review batch 10).
        fade_bias = 0.0
    toe_end_bias = clamp_float(float(adjustments.toe_end_offset), -3.0, 0.5)
    shoulder_white_bias = clamp_float(float(adjustments.shoulder_white_offset), -2.0, 3.0)

    # An untouched slider must not alter the compiled plan: the clamp ranges below
    # belong to the *moved* value, and film-preset plans legitimately compile powers
    # outside them (e.g. a fitted toe_power of 3.45). Re-clamping at zero bias would
    # silently reshape such plans the moment any other slider is touched — and break
    # the toe_end_offset control's monotone gradient across its own zero point.
    def _biased(value: float, bias: float, rate: float, low: float, high: float) -> float:
        if abs(bias) <= 1e-12:
            return float(value)
        return clamp_float(float(value) * (2.0 ** (rate * bias)), low, high)

    tone = replace(
        plan.tone,
        # At 18% gray this range is approximately -0.5 to +0.4 display EV. It is a
        # darktable-style interior power, not scene exposure, so both endpoints hold.
        view_brightness=_biased(plan.tone.view_brightness, brightness_bias, 0.25, 0.65, 1.65),
        contrast=_biased(plan.tone.contrast, contrast_bias, 0.25, 1.5, 4.5),
        # Positive UI direction means a more open toe and a softer shoulder. Lower
        # endpoint powers produce those two shapes in the C1 solver.
        toe_power=_biased(plan.tone.toe_power, shadow_bias, -0.45, 0.65, 2.5),
        shoulder_power=_biased(plan.tone.shoulder_power, highlight_bias, -0.45, 1.25, 5.0),
    )
    # Shoulder-white offset: moves the scene EV at which the curve reaches near-white,
    # by re-solving shoulder_power on the otherwise-finished tone plan. Deliberately
    # NOT a latitude/shoulder-start move: with contrast 3 the display range above the
    # pivot is spent within ~1 EV, so the C1 legality clamps absorb any start move
    # before it can render — that dead control is exactly the defect this replaces.
    # Curvature has real room instead: the shoulder region spans several scene EV, so
    # redistributing the roll-off visibly shifts where highlight gradations merge.
    # Black/white endpoints, pivot anchor and the toe side are fixed inputs to the
    # solve, so nothing below the shoulder moves. Runs at plan-compile time only.
    if abs(shoulder_white_bias) > 1e-12:
        from . import drt as drt_engine

        base_white = drt_engine.compiled_curve_transitions(tone)["shoulder_white_ev"]
        if base_white is not None:
            # A None base means the compiled curve has no measurable near-white
            # crossing; there is no truthful coordinate to offset, so the control
            # honestly does nothing instead of solving against a sentinel.
            solved_shoulder = drt_engine.solve_shoulder_power_for_white_ev(
                tone, base_white + shoulder_white_bias
            )
            tone = replace(
                tone,
                shoulder_power=clamp_float(
                    solved_shoulder,
                    drt_engine.SHOULDER_POWER_SOLVE_MIN,
                    drt_engine.SHOULDER_POWER_SOLVE_MAX,
                ),
            )
    # Toe-end offset: moves the scene EV at which the curve lands at near-black, by
    # re-solving toe_power on the otherwise-finished tone plan. Deliberately NOT a
    # latitude move: extending the linear latitude downward replaces the lifted toe
    # sigmoid with the mid segment's lower line and measurably darkens deep shadows —
    # the opposite of this control's declared meaning. Black/white endpoints, pivot
    # anchor and the shoulder side are fixed inputs to the solve, so nothing above
    # the toe moves. Runs at plan-compile time only.
    if abs(toe_end_bias) > 1e-12:
        from . import drt as drt_engine

        base_toe_end = drt_engine.compiled_curve_transitions(tone)["toe_end_ev"]
        if base_toe_end is not None:
            # A None base means the compiled curve has no measurable near-black
            # crossing; there is no truthful coordinate to offset, so the control
            # honestly does nothing instead of solving against a sentinel.
            solved_power = drt_engine.solve_toe_power_for_toe_end(
                tone, base_toe_end + toe_end_bias
            )
            tone = replace(tone, toe_power=clamp_float(solved_power, 0.35, 3.5))
    color = replace(
        plan.color,
        display_highlight_chroma_retreat=clamp_float(
            float(plan.color.display_highlight_chroma_retreat) + 0.30 * fade_bias,
            -0.30,
            0.70,
        ),
        display_highlight_chroma_start=clamp_float(
            float(plan.color.display_highlight_chroma_start) - 0.12 * fade_bias,
            0.58,
            0.90,
        ),
    )
    return replace(plan, tone=tone, color=color)


def _interimage_beta_for(stock: str) -> float:
    from .film_develop import interimage_beta

    return interimage_beta(stock)


def build_render_plan(
    bundle: RawBundle,
    analysis: Analysis,
    mode: str,
    output_gamut: str = "srgb",
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    punch_scale: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    adjustments: RenderAdjustments | None = None,
    film_curve: str = "none",
    film_mode: str = "observe",
    film_crossover: str = "off",
    endpoint_mode: str = "adaptive",
    color_head_y: float = 0.0,
    color_head_m: float = 0.0,
    film_exposure_ev: float = 0.0,
    film_print_timing: str = "fixed",
    film_print_medium: str = "",
    film_print_exposure_ev: float = 0.0,
    film_development: str = "measured_default",
    film_interimage: str = "declared",
    film_dev_contrast: float = 0.0,
    film_dev_fog: float = 0.0,
    film_dev_density: float = 0.0,
    film_compression: float = 0.0,
    film_compression_knee: float = 2.0,
    film_highlight_density: float = 0.0,
    film_grain: float = 0.0,
    film_halation: float = 0.0,
    film_bloom: float = 0.0,
    film_optics_seed: int = 0,
) -> RenderPlan:
    """Compile independent scene, tone and colour plans from an immutable capture."""
    # Full-mode input-domain normalization also lives HERE so hand callers of
    # build_render_plan cannot desynchronize the plan sample from the render
    # (review batch 10; the CLI and GUI already normalize at their sources).
    from .scene_transform import effective_scene_transform

    scene_transform = effective_scene_transform(scene_transform, film_mode, film_curve)
    tone_core = tone_core if tone_core in TONE_CORE_CHOICES else "agx"
    lum_norm = lum_norm if lum_norm in LUM_NORM_CHOICES else "y"
    endpoint_mode = endpoint_mode if endpoint_mode in ENDPOINT_MODE_CHOICES else "adaptive"
    agx_primaries = agx_engine.resolve_agx_primaries(agx_primaries)
    if tone_core == "gated":
        from .guidance import ensure_raw_guidance

        ensure_raw_guidance(bundle, analysis)
    # All four current cores operate in the Rec.2020 working space; the
    # `else` stays a defensive fallback for any future output-space-native core.
    if mode == "agx" or tone_core in ("lum", "neutral", "gated"):
        target_gamut = "Rec2020"
    else:
        target_gamut = output_gamut_space(output_gamut)
    plan_gain = compute_exposure_gain(exposure_mode_for_tone_core(tone_core), 0.0)
    # The scene metrics and the tone compiler historically transformed the identical
    # deterministic <=800k sample twice.  Share that exact array: every downstream
    # operation and percentile keeps its original order and dtype, while one full
    # scene-transform pass disappears from first-WB and first-film plan compilation.
    rec2020_sample = tone_plan_sample_scene_rec2020(
        bundle,
        scene_transform=scene_transform if mode == "agx" else "none",
        scene_transform_strength=scene_transform_strength,
        exposure_gain=plan_gain,
    )
    scene = scene_tone_metrics(
        bundle, analysis, scene_transform if mode == "agx" else "none",
        scene_transform_strength, plan_gain,
        rec2020_sample=rec2020_sample,
    )
    tone = build_tone_compression_plan(
        bundle,
        analysis,
        target_gamut,
        ev_from_agx_inset=False,
        scene_transform=scene_transform if mode == "agx" else "none",
        scene_transform_strength=scene_transform_strength,
        punch_scale=punch_scale if mode == "agx" else 0.0,
        tone_core=tone_core,
        lum_norm=lum_norm,
        # RAW-gated rendering uses the same pinned darktable scene-default geometry as
        # full-frame AgX. RAW evidence changes permission to use that color path, not the
        # definition of the path itself.
        agx_primaries=agx_primaries if mode == "agx" and tone_core == "agx" else "base",
        plan_exposure_gain=plan_gain,
        scene_metrics=scene,
        endpoint_mode=endpoint_mode,
        rec2020_sample=rec2020_sample,
    )
    if film_curve != "none":
        from dataclasses import replace as _replace

        from .film_curve import apply_film_curve_preset

        # Film response is fixed: the named coordinate replaces the scene-adaptive
        # curve wholesale (whole-roll consistency is the point of choosing it). Scene
        # metrics stay untouched so HDR budgeting keeps reading the real capture, and
        # user adjustments below still stack on top of the declared coordinate.
        # A declared film coordinate also supersedes any endpoint mode: its endpoints
        # are the preset's, so the plan must not keep claiming evidence endpoints.
        tone = apply_film_curve_preset(tone, film_curve)
        tone = _replace(tone, endpoint_mode="adaptive", endpoint_note=None)
        mode_value = film_mode if film_mode in ("observe", "full") else "observe"
        if mode_value == "full" and str(tone_core) != "agx":
            # The takeover LUT replaces the AgX formation wholesale and only
            # runs in the agx pipeline slot; the render stage would otherwise
            # fall back to the requested core SILENTLY while the filename
            # still claimed filmfull (the review's measured 0.0-diff bug).
            raise ValueError(
                "胶片接管显影（full 模式）只在 AgX tone core 上运行：接管 LUT "
                "整体替换 AgX formation，lum/neutral/gated 核会静默退回普通渲染；"
                "请使用 --tone-core agx 或切回 observe 模式"
            )
        # Crossover is a declaration on the takeover LUT's neutral axis: it rides the plan only
        # alongside an active preset, defaults to "off" (byte-identical status quo)
        # and is inert outside full mode (the film-takeover LUT's variant switch).
        # It must be stamped here, with the preset itself — not inside the
        # colour-head block below, which only runs for nonzero CC values (the
        # #20/#21 merge briefly moved it there, which silently killed
        # --film-crossover for every reversal preset: reversals reject the
        # colour head, so the stamp became unreachable).
        crossover_value = (
            film_crossover if film_crossover in ("off", "datasheet") else "off"
        )
        # film v2 P2: the emulsion exposure state and the print timing are
        # FULL-mode declarations (observe has no emulsion/print model); the
        # combination fails closed rather than silently ignoring the dial.
        exposure_value = float(film_exposure_ev)
        timing_value = str(film_print_timing or "fixed")
        medium_value = str(film_print_medium or "")
        print_exposure_value = float(film_print_exposure_ev)
        if timing_value not in ("fixed", "retimed", "custom"):
            raise ValueError(
                f"未知印相 timing:{timing_value}(可选 fixed/retimed/custom)"
            )
        if mode_value != "full" and (
            exposure_value != 0.0 or timing_value != "fixed"
            or medium_value != "" or print_exposure_value != 0.0
        ):
            raise ValueError(
                "胶片曝光状态、印相 timing/介质与手动印相曝光属于接管显影"
                "(full 模式):observe 模式没有乳剂/印相模型"
            )
        development_value = str(film_development or "measured_default")
        if development_value not in ("measured_default", "editorial_custom"):
            raise ValueError(
                f"未知显影配方:{development_value}"
                "(可选 measured_default/editorial_custom)"
            )
        if mode_value != "full" and (
            development_value != "measured_default"
            or float(film_compression) != 0.0
            or float(film_highlight_density) != 0.0
        ):
            raise ValueError(
                "显影配方与 Film Compression 属于接管显影(full 模式):"
                "observe 模式没有显影/压缩模型"
            )
        if mode_value != "full" and (
            float(film_grain) != 0.0 or float(film_halation) != 0.0
            or float(film_bloom) != 0.0
        ):
            raise ValueError(
                "模拟光学(颗粒/halation/bloom)属于接管显影(full 模式):"
                "observe 模式没有密度/印相空间模型"
            )
        if timing_value != "custom" and print_exposure_value != 0.0:
            raise ValueError(
                "手动印相曝光仅在 timing=custom 下有意义;fixed/retimed 的印相"
                "由联合求解决定"
            )
        tone = _replace(
            tone,
            film_mode=mode_value,
            film_crossover=crossover_value,
            film_exposure_ev=exposure_value,
            film_print_timing=timing_value,
            film_print_medium=medium_value,
            film_print_exposure_ev=print_exposure_value,
            film_development=development_value,
            film_interimage=str(film_interimage or "declared"),
            film_interimage_beta=(
                _interimage_beta_for(film_curve)
                if str(film_interimage or "declared") == "declared" else 0.0
            ),
            film_dev_contrast=float(film_dev_contrast),
            film_dev_fog=float(film_dev_fog),
            film_dev_density=float(film_dev_density),
            film_compression=float(film_compression),
            film_compression_knee=float(film_compression_knee),
            film_highlight_density=float(film_highlight_density),
            film_grain=float(film_grain),
            film_halation=float(film_halation),
            film_bloom=float(film_bloom),
            film_optics_seed=int(film_optics_seed),
        )
    if float(color_head_y) != 0.0 or float(color_head_m) != 0.0:
        # Enlarger colour head: a declared printing decision, valid only where a
        # printing stage physically exists — a negative preset. Reversal film is
        # its own display medium (no enlarger pass), and without any film preset
        # there is no print model at all, so both are rejected rather than
        # silently ignored.
        from .film_curve import film_process, validate_color_head_cc

        head_y = validate_color_head_cc(color_head_y, "色头 Y")
        head_m = validate_color_head_cc(color_head_m, "色头 M")
        if film_curve == "none":
            raise ValueError("放大机色头需要一个负片胶片曲线预设（当前未选择胶片）")
        if film_process(film_curve) != "negative":
            raise ValueError(
                f"放大机色头仅对负片预设有效：{film_curve} 是反转片，"
                "物理上没有印相环节（幻灯片自身就是显示介质）"
            )
        if str(film_mode) == "full" and str(film_print_timing or "fixed") != "custom":
            raise ValueError(
                "full 模式的色头只在 timing=custom 下可用（P3:在 paper-layer "
                "exposure model 内转换为 B1 后的逐层 Δτ,标 modelled）;"
                "fixed/retimed 的印相由联合求解决定,不能追加滤镜"
            )
        from dataclasses import replace as _replace

        # film_mode/film_crossover are already stamped with the preset above;
        # the colour head only adds its own dial values here.
        tone = _replace(tone, color_head_y=head_y, color_head_m=head_m)
    # film v2 plan objects (FILM_PRINT_RENDERING_PLAN §4): identity defaults,
    # validated fail-closed whenever the film domain is active. P1 wires the
    # structure; exposure/development/print state grows in later stages.
    if film_curve == "none" and (
        str(film_development or "measured_default") != "measured_default"
        or float(film_dev_contrast) != 0.0 or float(film_dev_fog) != 0.0
        or float(film_dev_density) != 0.0 or float(film_compression) != 0.0
        or float(film_highlight_density) != 0.0
        or float(film_grain) != 0.0 or float(film_halation) != 0.0
        or float(film_bloom) != 0.0
    ):
        raise ValueError("显影配方、Film Compression 与模拟光学需要一个胶片曲线预设")
    film_plans = None
    if film_curve != "none":
        from .film_curve import film_process
        from .film_plans import (
            AnalogFinishPlan,
            FilmDevelopmentPlan,
            FilmExposurePlan,
            FilmPrintPlan,
            validate_film_plans,
        )

        process = film_process(film_curve)
        neutralization = (
            "datasheet" if str(getattr(tone, "film_crossover", "off")) == "datasheet"
            else "bounded"
        )
        film_plans = (
            FilmExposurePlan(
                stock_id=film_curve,
                exposure_ev=float(getattr(tone, "film_exposure_ev", 0.0)),
            ),
            FilmDevelopmentPlan(
                recipe_id=str(getattr(tone, "film_development", "measured_default")),
                contrast_delta=float(getattr(tone, "film_dev_contrast", 0.0)),
                fog_delta=float(getattr(tone, "film_dev_fog", 0.0)),
                color_density=float(getattr(tone, "film_dev_density", 0.0)),
                provenance=(
                    "editorial"
                    if str(getattr(tone, "film_development", "measured_default"))
                    == "editorial_custom" else "measured"
                ),
                interimage_mode=str(
                    getattr(tone, "film_interimage", "declared") or "declared"
                ),
                interimage_beta=(
                    _interimage_beta_for(film_curve)
                    if str(getattr(tone, "film_interimage", "declared") or "declared")
                    == "declared" else 0.0
                ),
            ),
            FilmPrintPlan(
                medium_id=(
                    "reversal_direct" if process == "reversal"
                    else (str(getattr(tone, "film_print_medium", "")) or "print_paper")
                ),
                timing_policy=str(getattr(tone, "film_print_timing", "fixed")),
                neutralization_policy=neutralization,
                printer_y_cc=float(getattr(tone, "color_head_y", 0.0)),
                printer_m_cc=float(getattr(tone, "color_head_m", 0.0)),
                print_exposure_ev=float(getattr(tone, "film_print_exposure_ev", 0.0)),
            ),
            AnalogFinishPlan(
                compression=float(getattr(tone, "film_compression", 0.0)),
                compression_knee_ev=float(getattr(tone, "film_compression_knee", 2.0)),
                highlight_color_density=float(
                    getattr(tone, "film_highlight_density", 0.0)
                ),
                grain_profile=(
                    "modelled_default"
                    if float(getattr(tone, "film_grain", 0.0)) > 0.0 else "off"
                ),
                grain_amount=float(getattr(tone, "film_grain", 0.0)),
                halation_profile=(
                    "modelled_default"
                    if float(getattr(tone, "film_halation", 0.0)) > 0.0 else "off"
                ),
                halation_amount=float(getattr(tone, "film_halation", 0.0)),
                bloom_amount=float(getattr(tone, "film_bloom", 0.0)),
                seed=int(getattr(tone, "film_optics_seed", 0)),
            ),
        )
        validate_film_plans(*film_plans)
    plan = RenderPlan(
        tone=tone,
        color=build_color_geometry_plan(analysis, output_gamut, tone_core),
        scene=scene,
        film=film_plans,
    )
    return apply_render_adjustments(plan, adjustments)


def plan_for_mode(
    bundle: RawBundle,
    analysis: Analysis,
    mode: str,
    output_gamut: str = "srgb",
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    punch_scale: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    adjustments: RenderAdjustments | None = None,
) -> ToneCompressionPlan:
    """Compatibility accessor for callers that only need the tone sub-plan."""
    return build_render_plan(
        bundle,
        analysis,
        mode,
        output_gamut,
        scene_transform,
        scene_transform_strength,
        punch_scale,
        tone_core,
        lum_norm,
        agx_primaries,
        adjustments,
    ).tone
