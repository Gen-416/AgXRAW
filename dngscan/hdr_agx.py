# SPDX-License-Identifier: GPL-3.0-or-later
"""dngscan's independent extended-white HDR AgX formation.

The HDR and SDR renditions share capture data and scene intent, then split before display
formation. HDR owns its curve, extended colour volume and scene-driven white endpoint; it
never renders or corrects against SDR pixels. The endpoint is solved inside AgX itself:

    inset -> native extended-white C1 sigmoid -> hue restore/outset

`rho` mixes only chromaticity between a reference-white AgX path and the native extended
path, both normalized to the native curve's luminance. CFA masks can withdraw that freedom
locally. The completed image receives a bounded extended-P3 neutral-axis projection. The
projector preserves linear Y and an RGB opponent direction, not a perceptual CAM hue.

The two dispatchers share AgX primitives, but their completed curves are independent and
no pixel region is required to match between SDR and HDR.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from ._deps import np
from . import _fast as fast_backend
from . import agx as agx_engine
from . import guidance as guidance_engine
from . import punch as punch_engine
from . import retreat as retreat_engine
from . import scene_transform as scene_transform_engine
from .color import encode_display_linear, fit_to_output_gamut, rec2020_to_output
from .drt import curve_params_from_plan
from .hdr_curve import HdrCurveTable, compile_hdr_curve_table_pair
from .hdr_color import (
    output_luma_weights,
    blend_native_hdr_paths,
    fit_hdr_color_volume,
    formation_luma_weights,
    raw_gated_channel_separation,
)
from .models import Analysis, HdrAgxPlan, RawBundle, RenderPlan, ToneCompressionPlan
from .render import (
    STREAM_QUANTIZE_CHUNK,
    STREAM_RENDER_CHUNK,
    STREAM_THREAD_MIN_PIXELS,
    _apply_output_color_ops,
    apply_tone_core,
    dither_quantize_u8_with_noise,
    finalize_output_linear,
    generate_dither_noise,
    plan_with_look_overrides,
    scene_intent_rec2020,
)


def _hdr_render_workers() -> int:
    """Render-pipeline width, rented from the S3 CPU budget: NumPy releases
    the GIL on large array ops, so the chunk pipeline scales past the
    historical 2 workers. Capped at 6 — beyond the performance cores the
    marginal chunk mostly contends for memory bandwidth, and each in-flight
    chunk holds its formation temporaries (~tens of MB) alive."""
    import os

    from .cpu_budget import outer_workers

    return outer_workers(min(6, max(2, (os.cpu_count() or 4) - 2)))


def _hdr_tone_plan(hdr_plan: HdrAgxPlan) -> ToneCompressionPlan:
    return replace(
        hdr_plan.formation,
        hue_restore=float(hdr_plan.color.hue_restore),
        agx_primaries=str(hdr_plan.color.primaries_preset),
    )


def _pack_peak(hdr_plan: HdrAgxPlan) -> float:
    peak = float(hdr_plan.tone.peak_linear)
    return peak * max(0.0, 1.0 - float(hdr_plan.color.gamut_fit_margin))


def _hdr_reference_needed(hdr_plan: HdrAgxPlan) -> bool:
    """Whether the reference-white chroma candidate must be compiled.

    rho blends toward native: at rho==1 the blend returns the native path, and at
    rho==0 it returns the reference-white chromaticity carried at native luminance
    — the conservative end of the dial, NOT the native path. A previous version
    treated rho==0 as a skip-identity, which inverted the design exactly where it
    matters most (zero permission — including analysis=None — silently received
    the most open native chroma). The only true identities are a unit peak or no
    rendered headroom, where the reference curve IS the native curve.
    """
    return (
        float(hdr_plan.tone.rendered_headroom_ev) > 0.0
        and abs(float(hdr_plan.tone.peak_linear) - 1.0) > 1e-12
    )


def _compile_native_hdr_plan(
    hdr_plan: HdrAgxPlan,
    hdr_tone_plan: ToneCompressionPlan,
    inset_matrix: Any,
    outset_matrix: Any,
    formation_y: Any,
    curve_tables: tuple[HdrCurveTable, HdrCurveTable],
    peak: float,
    output_gamut: str,
) -> Any | None:
    """Native HDR formation plan, or None when the NumPy path must serve.

    Same strict/fallback ladder as the SDR output finalizer: compile failures fall
    back silently in auto mode and raise NativeKernelError under DNGSCAN_FAST=1.
    """
    if not fast_backend.supports_hdr_formation(hdr_tone_plan):
        return None
    try:
        return fast_backend.compile_hdr_plan(
            hdr_plan,
            hdr_tone_plan,
            inset_matrix,
            outset_matrix,
            formation_y,
            curve_tables,
            peak,
            output_gamut,
        )
    except Exception as exc:
        if fast_backend.strict_requested():
            if isinstance(exc, fast_backend.NativeKernelError):
                raise
            raise fast_backend.NativeKernelError(str(exc)) from exc
    return None


def _form_hdr_chunk(
    intent_rec: Any,
    hdr_plan: HdrAgxPlan,
    hdr_tone_plan: ToneCompressionPlan,
    inset_matrix: Any,
    outset_matrix: Any,
    formation_y: Any,
    curve_tables: tuple[HdrCurveTable, HdrCurveTable],
    clip_masks_chunk: Any | None,
    peak: float,
    output_gamut: str,
    native_plan: Any | None = None,
) -> Any:
    """One HDR display-linear chunk from shared scene-intent Rec.2020.

    The per-plan curve tables carry the compiled body+shoulder curve (§P3); the
    analytic evaluator in hdr_curve remains the oracle they are tested against.
    A pair of aliased tables encodes "no reference candidate needed".

    With a compiled native plan the whole chain runs in the C++ kernel
    (tests/test_hdr_native.py holds the two paths to per-pixel parity); the NumPy
    body below is the reference implementation and the fallback.
    """
    if native_plan is not None:
        try:
            arr = np.ascontiguousarray(intent_rec, dtype=np.float32)
            masks = (
                np.ascontiguousarray(clip_masks_chunk, dtype=np.float32)
                if clip_masks_chunk is not None
                else None
            )
            return fast_backend.apply_hdr_formation_f32(arr, masks, native_plan)
        except fast_backend.NativeKernelError:
            if fast_backend.strict_requested():
                raise
        except Exception as exc:
            if fast_backend.strict_requested():
                raise fast_backend.NativeKernelError(str(exc)) from exc
    inset, pre_hue = agx_engine.prepare_formation(intent_rec, hdr_tone_plan, inset_matrix)
    # The HDR plan is a replace() of the film plan so curve_preset rides along;
    # the retired channel-ratio machinery no longer exists in either dispatcher,
    # and exposure-dependent film colour (the colour head) is SDR/observe-side.
    native_table, reference_table = curve_tables
    native_formation = native_table.apply(inset)

    def formation_tail(formation: Any) -> Any:
        mapped_rec = agx_engine.finish_formation(
            formation, pre_hue, hdr_tone_plan, outset_matrix
        )
        # Observe-mode film colour joins here — post-outset Rec.2020, scene
        # luminance axis — mirroring the SDR order (outset -> film colour ->
        # punch). Native kernels exclude active colour heads, so this stays a
        # Python-only operator until it earns a C++ port.
        mapped_rec = agx_engine.apply_film_color_rec2020(
            mapped_rec, intent_rec, hdr_tone_plan
        )
        mapped_rec = punch_engine.apply_punch_rec2020(
            mapped_rec, float(getattr(hdr_tone_plan, "punch_strength", 0.0))
        )
        output_linear = rec2020_to_output(mapped_rec, output_gamut)
        return np.nan_to_num(output_linear, nan=0.0, posinf=1e6, neginf=-1e6)

    if reference_table is not native_table:
        global_rho = float(hdr_plan.color.channel_separation) * float(
            hdr_plan.color.snr_gate
        )
        rho = raw_gated_channel_separation(global_rho, clip_masks_chunk)
        blended = blend_native_hdr_paths(
            reference_table.apply(inset),
            native_formation,
            rho,
            formation_y,
        )
        # The blend equalizes Y at the formation point, but hue restore, the
        # channel gain and punch are not Y-preserving, so the two chroma paths
        # drift apart again downstream (measured p95 0.186 EV at the default
        # hue_restore=0.6 before this normalization). The native branch is the
        # sole Y authority end-to-end: run both candidates through the same
        # tail and re-anchor the blend to the native branch's final Y.
        final_native = formation_tail(native_formation)
        final_blend = formation_tail(blended)
        w_out = output_luma_weights(output_gamut)
        y_native = np.tensordot(final_native, w_out, axes=([-1], [0]))
        y_blend = np.tensordot(final_blend, w_out, axes=([-1], [0]))
        scale = y_native / np.maximum(y_blend, np.float32(1e-9))
        valid = (y_native > 1e-9) & (y_blend > 1e-9)
        output_linear = np.where(
            valid[..., None], final_blend * scale[..., None], final_native
        ).astype(np.float32, copy=False)
    else:
        output_linear = formation_tail(native_formation)
    return fit_hdr_color_volume(output_linear, peak, output_gamut).astype(
        np.float32, copy=False
    )


def scene_render_to_hdr_display_linear(
    bundle: RawBundle,
    plan: ToneCompressionPlan | RenderPlan,
    hdr_plan: HdrAgxPlan,
    output_gamut: str = "p3",
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
) -> Any:
    """Scene-linear -> extended display-linear, values above 1.0 permitted.

    Display looks and filters are refused by the exporter rather than ignored: they are SDR
    operators that have not been given an independent HDR meaning.
    """
    source_tone = plan.tone if isinstance(plan, RenderPlan) else plan
    if str(getattr(source_tone, "tone_core", "agx")) != "agx":
        raise RuntimeError("HDR AgX 仅支持 tone_core=agx；不能把其他 SDR tone core 标成 HDR AgX")

    # HdrColorGeometry is the source of truth for the HDR branch. The values currently
    # start from shared scene intent, but both the tone object and geometry are HDR-owned.
    hdr_tone_plan = _hdr_tone_plan(hdr_plan)
    inset_matrix, outset_matrix = agx_engine.formation_matrices(hdr_tone_plan)
    formation_y = formation_luma_weights(outset_matrix)
    body_params = curve_params_from_plan(hdr_tone_plan)
    curve_tables = compile_hdr_curve_table_pair(
        hdr_plan.tone,
        hdr_tone_plan,
        need_reference=_hdr_reference_needed(hdr_plan),
        body_params=body_params,
    )

    scene = bundle.scene_rec2020_render
    h, w = scene.shape[:2]
    flat_scene = scene.reshape(-1, scene.shape[-1])
    out = np.empty((flat_scene.shape[0], 3), dtype=np.float32)
    chunk = 1_000_000

    clip_masks = None
    if getattr(bundle, "clip_masks", None) is not None:
        clip_masks = retreat_engine.clip_masks_for_shape(bundle, (h, w)).reshape(-1, 3)

    # The scene-authorized native curve endpoint. Display capacity is only the container's
    # outer ceiling; using it here would normalize every photograph to display peak.
    peak = _pack_peak(hdr_plan)

    native_hdr_plan = _compile_native_hdr_plan(
        hdr_plan, hdr_tone_plan, inset_matrix, outset_matrix,
        formation_y, curve_tables, peak, output_gamut,
    )

    wb_adapt = scene_transform_engine.window_transport(bundle)

    def render_hdr_chunk(start: int, end: int) -> None:
        rec = scene_intent_rec2020(flat_scene[start:end, :3], bundle)
        rec = scene_transform_engine.apply_scene_transform_rec2020(
            rec, scene_transform, scene_transform_strength, wb_adapt
        )
        retreat_strength = float(hdr_plan.color.raw_clip_retreat)
        if clip_masks is not None and retreat_strength > 0.0:
            rec = retreat_engine.apply_clip_retreat_rec2020(
                rec, clip_masks[start:end], retreat_strength
            )
        out[start:end] = _form_hdr_chunk(
            rec,
            hdr_plan,
            hdr_tone_plan,
            inset_matrix,
            outset_matrix,
            formation_y,
            curve_tables,
            clip_masks[start:end] if clip_masks is not None else None,
            peak,
            output_gamut,
            native_plan=native_hdr_plan,
        )

    ranges = [
        (start, min(start + chunk, flat_scene.shape[0]))
        for start in range(0, flat_scene.shape[0], chunk)
    ]
    if flat_scene.shape[0] < STREAM_THREAD_MIN_PIXELS or len(ranges) < 2:
        for start, end in ranges:
            render_hdr_chunk(start, end)
    else:
        # Chunks write disjoint slices of one preallocated buffer, so completion
        # order is irrelevant here (unlike the dithered pair path).
        workers = min(_hdr_render_workers(), len(ranges))
        from .cpu_budget import claim, split_for, set_inner

        # S3: the HDR formation is native-kernel heavy, so it takes the
        # native-core split (few outer workers, wide native budget).
        _outer, _inner = split_for(True)
        workers = min(workers, _outer)
        with claim(_inner), ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="dngscan-hdr",
            initializer=set_inner, initargs=(_inner,),
        ) as pool:
            list(pool.map(lambda r: render_hdr_chunk(*r), ranges))
    return out.reshape(h, w, 3)


def render_ultrahdr_film_pair(
    bundle: RawBundle,
    analysis: Analysis,
    plan: RenderPlan,
    hdr_plan: HdrAgxPlan,
    output_gamut: str = "p3",
) -> tuple[Any, Any]:
    """胶片印相 + scene HDR 扩展 (plan §10): SDR base and HDR alternate for a
    film-takeover plan.

    The SDR base IS the standalone film print — produced by the very same
    render_output_u8 call an SDR export runs, so it is byte-identical by
    construction (never a reimplementation that could drift). The HDR
    alternate multiplies the print's display-linear rendition by a C1
    luminance gain driven by the SCENE's own highlight EV: gain 1 at and
    below the print's reference-white join (film_reference_white_ev probe),
    smoothstep up to the plan's solved reliable headroom. This extends
    reliable scene highlights above reference white; it never re-develops
    the body and never claims physical film HDR. Costs a second film walk
    (an honest first version; fusing the two walks is P7 material).
    """
    from .film_develop import film_reference_white_ev
    from .film_v2_math import film_hdr_gain_log2
    from .render import _optics_band_rows, render_output_u8

    tone = plan.tone
    if (
        str(getattr(tone, "film_mode", "observe")) != "full"
        or str(getattr(tone, "curve_preset", "none")) == "none"
    ):
        raise RuntimeError("render_ultrahdr_film_pair 只服务 film full 计划")
    scene = bundle.scene_rec2020_render
    h, w = scene.shape[:2]
    # ONE walk (review batch 17): the base render captures its post-film
    # Rec.2020 pixels as it quantizes, so the film chain, scene intent and
    # spatial operators run once instead of twice.
    mapped = np.empty((h * w, 3), dtype=np.float32)
    base_u8 = render_output_u8(
        bundle, analysis, output_gamut, plan, capture_mapped=mapped
    )
    # The HDR alternate (review batch 14):
    #     hdr = decoded_base + float_print * (gain - 1)
    # Below the join gain == 1, so the body IS the decoded real base (the
    # 8-bit dithered pixels the file carries — gain >= 1 holds against the
    # actual SDR leg by construction). Above the join the INCREMENT comes
    # from the high-precision float print, so highlight detail is not
    # limited to 8-bit steps and quantization noise is not amplified by the
    # gain (the pure decoded-base construction measured p99 ~0.007 linear
    # error at a mere +0.15 EV headroom). Everything is banded float32 —
    # the previous full-frame float64 decode held ~1.4 GB at 60 MP.
    from .color import srgb_decode
    from .film_develop import film_reference_white_ev
    from .film_v2_math import film_hdr_gain_log2

    color_plan = plan.color
    gamut_alpha = float(color_plan.gamut_fit_alpha) if color_plan is not None else 0.05
    join_ev = film_reference_white_ev(tone)
    # The extension may only spend RELIABLE scene highlights above the join:
    # the solved headroom is co-compiled with the reliable tail's distance
    # from the join (review batch 14), so gain never engages content the
    # RAW cannot vouch for.
    headroom_ev = min(
        float(hdr_plan.tone.rendered_headroom_ev),
        max(float(hdr_plan.tone.reliable_tail_ev) - join_ev, 0.0),
    )
    span_ev = max(headroom_ev, 1.0) * 1.5
    flat_scene = scene.reshape(-1, scene.shape[-1])
    flat_u8 = base_u8.reshape(-1, 3)
    luma = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
    hdr_out = np.empty((flat_u8.shape[0], 3), dtype=np.float32)
    band = max(_optics_band_rows(w), 1) * w
    for s0 in range(0, flat_u8.shape[0], band):
        e0 = min(s0 + band, flat_u8.shape[0])
        decoded = srgb_decode(flat_u8[s0:e0].astype(np.float32) / 255.0)
        band_linear = np.nan_to_num(
            rec2020_to_output(mapped[s0:e0], output_gamut),
            nan=0.0, posinf=1e6, neginf=-1e6,
        )
        fitted = fit_to_output_gamut(
            band_linear, output_gamut, alpha=gamut_alpha
        ).astype(np.float32, copy=False)
        rec = scene_intent_rec2020(flat_scene[s0:e0, :3], bundle)
        ev = np.log2(
            np.maximum(np.asarray(rec, dtype=np.float32) @ luma, 1e-9)
            / np.float32(0.18)
        )
        gain = np.exp2(film_hdr_gain_log2(
            ev, headroom_ev=headroom_ev, join_ev=join_ev, span_ev=span_ev,
        )).astype(np.float32)
        hdr_out[s0:e0] = decoded + fitted * (gain[:, None] - 1.0)
    return base_u8, hdr_out.reshape(h, w, 3)


def render_ultrahdr_agx_pair(
    bundle: RawBundle,
    analysis: Analysis,
    plan: RenderPlan,
    hdr_plan: HdrAgxPlan,
    output_gamut: str = "p3",
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
) -> tuple[Any, Any]:
    """One intent walk producing SDR u8 and HDR display-linear.

    Ultrahdr previously paid for two full-resolution formations of the same scene intent.
    Scale, scene transform and clip retreat are shared; each branch then runs its own
    display formation. SDR still goes through ``apply_tone_core`` so the native AgX kernel
    and dither grouping match a standalone ``render_output_u8`` export.
    """
    if str(getattr(plan.tone, "tone_core", "agx")) != "agx":
        raise RuntimeError("Ultrahdr AgX pair 仅支持 tone_core=agx")
    if (
        str(getattr(plan.tone, "film_mode", "observe")) == "full"
        and str(getattr(plan.tone, "curve_preset", "none")) != "none"
    ):
        # Kernel-level defence (review batch 11): the SDR leg routes through
        # apply_tone_core, which under full+preset is the takeover film chain, while
        # the HDR leg below is AgX formation — refusing here keeps the pair
        # honest even for hand-built plans that bypassed export_jpeg.
        raise RuntimeError(
            "Ultrahdr AgX pair 不支持胶片接管显影（film_mode=full）："
            "SDR 腿是接管 LUT、HDR 腿是 AgX formation，两种显影不能拼进同一个 gain-map"
        )

    effective_plan = plan_with_look_overrides(plan, "none", 1.0)
    effective_tone = effective_plan.tone if isinstance(effective_plan, RenderPlan) else effective_plan
    color_plan = effective_plan.color if isinstance(effective_plan, RenderPlan) else None

    # The SDR base shares render_output_u8's fused native finalizer (matrix →
    # Oklab gamut fit → transfer → TPDF quantize). Ultrahdr forces look/filter
    # off, so only the pre-fit color ops (highlight chroma retreat) stay in the
    # render workers; the same strict/fallback ladder applies.
    gamut_alpha = float(color_plan.gamut_fit_alpha) if color_plan is not None else 0.05
    native_output_plan = None
    if fast_backend.supports_output_finalizer():
        try:
            native_output_plan = fast_backend.compile_output_plan(
                output_gamut, gamut_alpha
            )
        except Exception as exc:
            if fast_backend.strict_requested():
                raise fast_backend.NativeKernelError(str(exc)) from exc

    hdr_tone_plan = _hdr_tone_plan(hdr_plan)
    inset_matrix, outset_matrix = agx_engine.formation_matrices(hdr_tone_plan)
    formation_y = formation_luma_weights(outset_matrix)
    body_params = curve_params_from_plan(hdr_tone_plan)
    curve_tables = compile_hdr_curve_table_pair(
        hdr_plan.tone,
        hdr_tone_plan,
        need_reference=_hdr_reference_needed(hdr_plan),
        body_params=body_params,
    )
    peak = _pack_peak(hdr_plan)
    native_hdr_plan = _compile_native_hdr_plan(
        hdr_plan, hdr_tone_plan, inset_matrix, outset_matrix,
        formation_y, curve_tables, peak, output_gamut,
    )

    scene = bundle.scene_rec2020_render
    h, w = scene.shape[:2]
    flat_scene = scene.reshape(-1, scene.shape[-1])
    sdr_out = np.empty((flat_scene.shape[0], 3), dtype=np.uint8)
    hdr_out = np.empty((flat_scene.shape[0], 3), dtype=np.float32)

    quantize_chunk_size = STREAM_QUANTIZE_CHUNK
    render_chunk_size = (
        STREAM_RENDER_CHUNK
        if flat_scene.shape[0] >= STREAM_THREAD_MIN_PIXELS
        else quantize_chunk_size
    )
    if quantize_chunk_size % render_chunk_size != 0:
        raise ValueError(
            f"stream chunking misaligned: quantize {quantize_chunk_size} "
            f"must be a multiple of render {render_chunk_size}"
        )

    clip_masks = None
    raw_guidance = None
    if color_plan is not None and getattr(bundle, "clip_masks", None) is not None:
        clip_masks = retreat_engine.clip_masks_for_shape(bundle, (h, w)).reshape(-1, 3)
        if str(getattr(effective_tone, "tone_core", "agx")) == "gated":
            raw_guidance = guidance_engine.raw_guidance_for_shape(bundle, (h, w), analysis)

    wb_adapt = scene_transform_engine.window_transport(bundle)
    # Ultrahdr forces look/filter off. HDR copies retreat from the scene plan, so one
    # shared intent strength matches what each branch would have applied alone.
    shared_retreat = (
        float(color_plan.raw_clip_retreat_strength) if color_plan is not None else 0.0
    )

    def render_pair_chunk(start: int, end: int) -> tuple[Any, Any]:
        rec = scene_intent_rec2020(flat_scene[start:end, :3], bundle)
        rec = scene_transform_engine.apply_scene_transform_rec2020(
            rec, scene_transform, scene_transform_strength, wb_adapt
        )
        sample_masks = clip_masks[start:end] if clip_masks is not None else None
        if sample_masks is not None and shared_retreat > 0.0:
            rec = retreat_engine.apply_clip_retreat_rec2020(
                rec, sample_masks, shared_retreat
            )
        mapped_rec = apply_tone_core(
            rec,
            effective_tone,
            color_plan,
            sample_masks,
            guidance_engine.flatten_raw_guidance(raw_guidance, start, end)
            if raw_guidance is not None
            else None,
        )
        output_linear = rec2020_to_output(mapped_rec, output_gamut)
        output_linear = np.nan_to_num(
            output_linear, nan=0.0, posinf=1e6, neginf=-1e6
        ).astype(np.float32, copy=False)
        if native_output_plan is not None:
            sdr_final = np.ascontiguousarray(
                _apply_output_color_ops(
                    output_linear, output_gamut, "none", 1.0, color_plan
                ),
                dtype=np.float32,
            )
        else:
            finalized = finalize_output_linear(
                output_linear, output_gamut, "none", 1.0, color_plan
            )
            sdr_final = np.nan_to_num(
                finalized.astype(np.float32, copy=False),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
        hdr_final = _form_hdr_chunk(
            rec,
            hdr_plan,
            hdr_tone_plan,
            inset_matrix,
            outset_matrix,
            formation_y,
            curve_tables,
            sample_masks,
            peak,
            output_gamut,
            native_plan=native_hdr_plan,
        )
        return sdr_final, hdr_final

    ranges = [
        (start, min(start + render_chunk_size, flat_scene.shape[0]))
        for start in range(0, flat_scene.shape[0], render_chunk_size)
    ]
    rng = np.random.default_rng(0)

    def quantize_chunk(start: int, end: int, finalized: Any) -> None:
        noise_a, noise_b = generate_dither_noise(rng, finalized.shape)
        if native_output_plan is not None:
            try:
                sdr_out[start:end] = fast_backend.finalize_output_u8_f32(
                    finalized, noise_a, noise_b, native_output_plan
                )
                return
            except Exception as exc:
                if fast_backend.strict_requested():
                    if isinstance(exc, fast_backend.NativeKernelError):
                        raise
                    raise fast_backend.NativeKernelError(str(exc)) from exc
            finalized = fit_to_output_gamut(
                finalized, output_gamut, alpha=gamut_alpha
            ).astype(np.float32, copy=False)
        encoded = encode_display_linear(finalized, output_gamut)
        sdr_out[start:end] = dither_quantize_u8_with_noise(encoded, noise_a, noise_b)

    def consume_in_quantize_groups(results: Any) -> None:
        group_start = 0
        group_parts: list[Any] = []
        for start, end, sdr_final, hdr_final in results:
            hdr_out[start:end] = hdr_final
            group_parts.append(sdr_final)
            group_end = min(group_start + quantize_chunk_size, flat_scene.shape[0])
            if end == group_end:
                merged = (
                    group_parts[0]
                    if len(group_parts) == 1
                    else np.concatenate(group_parts, axis=0)
                )
                quantize_chunk(group_start, group_end, merged)
                group_start = group_end
                group_parts = []

    if flat_scene.shape[0] < STREAM_THREAD_MIN_PIXELS or len(ranges) < 2:
        consume_in_quantize_groups(
            (start, end, *render_pair_chunk(start, end)) for start, end in ranges
        )
    else:
        workers = min(_hdr_render_workers(), len(ranges))
        from .cpu_budget import claim, split_for, set_inner

        # S3: the HDR formation is native-kernel heavy, so it takes the
        # native-core split (few outer workers, wide native budget).
        _outer, _inner = split_for(True)
        workers = min(workers, _outer)
        with claim(_inner), ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="dngscan-ultrahdr",
            initializer=set_inner, initargs=(_inner,),
        ) as pool:
            pending: dict[int, Any] = {}
            submit_idx = 0
            while submit_idx < min(workers, len(ranges)):
                start, end = ranges[submit_idx]
                pending[submit_idx] = pool.submit(render_pair_chunk, start, end)
                submit_idx += 1

            def ordered_results() -> Any:
                nonlocal submit_idx
                for idx, (start, end) in enumerate(ranges):
                    sdr_final, hdr_final = pending.pop(idx).result()
                    if submit_idx < len(ranges):
                        next_start, next_end = ranges[submit_idx]
                        pending[submit_idx] = pool.submit(
                            render_pair_chunk, next_start, next_end
                        )
                        submit_idx += 1
                    yield start, end, sdr_final, hdr_final

            consume_in_quantize_groups(ordered_results())

    return sdr_out.reshape(h, w, 3), hdr_out.reshape(h, w, 3)


def to_gainmap_alternate(hdr_display_linear: Any, peak: float) -> Any:
    """Pack the HDR rendition as the float16 RGBA alternate the gain-map writer expects.

    The HDR renderer normally returns an in-volume image. Clipping remains here as a
    defensive encode-boundary guard for callers that provide their own rendition.
    """
    arr = np.clip(np.asarray(hdr_display_linear, dtype=np.float32), 0.0, float(peak))
    rgba = np.empty(arr.shape[:2] + (4,), dtype=np.float16)
    rgba[:, :, :3] = arr.astype(np.float16, copy=False)
    rgba[:, :, 3] = np.float16(1.0)
    return rgba


def achieved_headroom(hdr_display_linear: Any, percentile: float = 99.99) -> float:
    """H_actual, reported from a percentile rather than the single brightest pixel.

    One specular pixel is not evidence that a render used its headroom, and accepting it
    as such is how a capacity ceiling turns into a normalisation target.
    """
    arr = np.asarray(hdr_display_linear, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    top = float(np.percentile(arr, percentile))
    return float(np.log2(top)) if top > 1.0 else 0.0
