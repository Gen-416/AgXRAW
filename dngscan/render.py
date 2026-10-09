# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene-linear to display-linear tone mapping pipelines."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from ._deps import np
from . import agx as agx_engine
from . import display_filter as filter_engine
from . import lum as lum_engine
from . import neutral as neutral_engine
from . import look as look_engine
from . import gated_drt as gated_drt_engine
from . import guidance as guidance_engine
from . import punch as punch_engine
from . import retreat as retreat_engine
from . import scene_transform as scene_transform_engine
from . import _fast as fast_backend
from .color import (
    encode_display_linear, fit_to_output_gamut, oklab_to_output_rgb,
    rec2020_to_output, rgb_to_oklab, smoothstep,
)
from .models import Analysis, ColorGeometryPlan, RawBundle, RenderPlan, ToneCompressionPlan
from .tone import build_render_plan, scene_intent_rec2020

# Streamed-export chunking. The dither RNG consumes noise in quantize-group order, so
# render chunks must tile each quantize group exactly (quantize % render == 0) for the
# grouped flush to stay lossless and byte-identical to the legacy single-pass path.
# render_output_u8 validates this at runtime; keep the constants divisible.
STREAM_QUANTIZE_CHUNK = 1_000_000
STREAM_RENDER_CHUNK = 500_000
STREAM_COMPLEX_COLOR_CHUNK = 250_000
STREAM_THREAD_MIN_PIXELS = 2_000_000


def _stream_render_workers() -> int:
    """Bounded worker count shared by the full-res streaming pools (rented
    from the S3 CPU budget)."""
    from .cpu_budget import outer_workers

    return outer_workers(min(6, max(2, (TOTAL_CPUS) - 2)))


from .cpu_budget import TOTAL as TOTAL_CPUS  # noqa: E402


def dither_quantize_u8(encoded: Any, rng: Any) -> Any:
    """Quantize display-domain [0,1] floats to uint8 with 1-LSB TPDF dither."""
    noise_a, noise_b = generate_dither_noise(rng, encoded.shape)
    return dither_quantize_u8_with_noise(encoded, noise_a, noise_b)


def generate_dither_noise(rng: Any, shape: Any) -> tuple[Any, Any]:
    """Generate the authoritative deterministic TPDF source planes.

    Kept as a named stage so profiling can distinguish RNG/memory traffic from the
    fused output kernel without putting timers in the production render path.
    """
    return (
        rng.random(shape, dtype=np.float32),
        rng.random(shape, dtype=np.float32),
    )


def dither_quantize_u8_with_noise(
    encoded: Any, noise_a: Any, noise_b: Any
) -> Any:
    """Reference quantizer with explicit noise for deterministic native parity."""
    # Operation order is part of the pixel contract.  Keeping the two source planes
    # separate is observably different at float32 rounding boundaries from first
    # combining them as ``noise_a - noise_b`` (a handful of final channels can move by
    # one code value on a 1920px frame).  This is also the order used by the native
    # finalizer and by the uncached streamed export path.
    scaled = encoded.astype(np.float32, copy=False) * np.float32(255.0)
    quantized = scaled + np.float32(0.5)
    quantized = quantized + np.asarray(noise_a, dtype=np.float32)
    quantized = quantized - np.asarray(noise_b, dtype=np.float32)
    return np.clip(np.floor(quantized), 0, 255).astype(np.uint8)


def dither_quantize_u8_with_tpdf(encoded: Any, noise: Any) -> Any:
    """Quantize with an already-combined TPDF plane."""
    scaled = encoded.astype(np.float32, copy=False) * np.float32(255.0)
    return np.clip(np.floor(scaled + np.float32(0.5) + noise), 0, 255).astype(np.uint8)


def deterministic_dither_planes(shape: Any) -> tuple[Any, Any]:
    """Build the seed-0 TPDF source planes consumed by the streamed SDR path.

    Quantize-group ordering is part of the pixel contract, so a preview
    session can reuse immutable planes across frames without changing bytes.
    """
    dimensions = tuple(int(value) for value in shape)
    if len(dimensions) != 3 or dimensions[-1] != 3:
        raise ValueError("dither shape must be (H, W, 3)")
    pixel_count = dimensions[0] * dimensions[1]
    flat_a = np.empty((pixel_count, 3), dtype=np.float32)
    flat_b = np.empty((pixel_count, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    for start in range(0, pixel_count, STREAM_QUANTIZE_CHUNK):
        end = min(start + STREAM_QUANTIZE_CHUNK, pixel_count)
        first, second = generate_dither_noise(rng, (end - start, 3))
        flat_a[start:end] = first
        flat_b[start:end] = second
    first = flat_a.reshape(dimensions)
    second = flat_b.reshape(dimensions)
    first.setflags(write=False)
    second.setflags(write=False)
    return first, second


def deterministic_dither_plane(shape: Any) -> Any:
    """Legacy combined TPDF helper retained for external callers.

    New preview code must use :func:`deterministic_dither_planes`; combining the
    planes changes float32 addition order and is therefore not a bit-exact cache for
    the authoritative streamed quantizer.
    """
    first, second = deterministic_dither_planes(shape)
    combined = np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32)
    combined.setflags(write=False)
    return combined


def output_linear_to_u8(
    rgb_linear: Any,
    output_gamut: str = "srgb",
    look: str = "none",
    look_strength: float = 1.0,
    color_plan: ColorGeometryPlan | None = None,
) -> Any:
    return quantize_final_output_linear_to_u8(
        finalize_output_linear(rgb_linear, output_gamut, look, look_strength, color_plan),
        output_gamut,
    )


def quantize_final_output_linear_to_u8(rgb_linear: Any, output_gamut: str = "srgb") -> Any:
    """Encode finalized display-linear RGB to 8-bit JPEG code values."""
    flat = rgb_linear.reshape(-1, 3)
    out = np.empty((flat.shape[0], 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    chunk = 1_000_000
    for start in range(0, flat.shape[0], chunk):
        end = min(start + chunk, flat.shape[0])
        fitted = np.nan_to_num(flat[start:end].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
        encoded = encode_display_linear(fitted, output_gamut)
        out[start:end] = dither_quantize_u8(encoded, rng)
    return out.reshape(rgb_linear.shape[:2] + (3,))


def _apply_display_highlight_chroma_retreat(
    rgb: Any,
    output_gamut: str,
    strength: float,
    start: float = 0.75,
    end: float = 0.98,
) -> Any:
    """Gently fade only near-display-white chroma in the luminance-core path.

    This is deliberately downstream of the scene-linear clip mask: it is an aesthetic
    guard for bright, *unclipped* colours, not a claim that sensor information is lost.
    """
    if abs(strength) <= 1e-9:
        return rgb
    lab_l, lab_a, lab_b = rgb_to_oklab(rgb, output_gamut)
    amount = np.float32(strength) * smoothstep(float(start), float(end), lab_l)
    # Positive strength fades chroma toward display white. A small negative range lets
    # the user retain more highlight colour; the final gamut fit remains authoritative.
    keep = np.clip(np.float32(1.0) - amount, 0.0, 1.5)
    return oklab_to_output_rgb(lab_l, lab_a * keep, lab_b * keep, output_gamut)


def finalize_output_linear(
    rgb_linear: Any,
    output_gamut: str = "srgb",
    look: str = "none",
    look_strength: float = 1.0,
    color_plan: ColorGeometryPlan | None = None,
    *,
    out: Any | None = None,
) -> Any:
    """Apply common post-tone color operations in display-linear output RGB.

    This stage runs after agx, gated, lum, and neutral alike: optional look, optional
    display-highlight chroma retreat, then the authoritative hue-preserving gamut fit.
    """
    original_shape = rgb_linear.shape
    flat = rgb_linear.reshape(-1, 3)
    if out is None:
        out = np.empty((flat.shape[0], 3), dtype=np.float32)
    else:
        if (out.shape != original_shape or out.dtype != np.float32
                or not out.flags.c_contiguous or not out.flags.writeable):
            raise ValueError("display-linear destination must be writable contiguous float32 RGB")
        out = out.reshape(-1, 3)
    chunk = 1_000_000
    for start in range(0, flat.shape[0], chunk):
        end = min(start + chunk, flat.shape[0])
        piece = _apply_output_color_ops(
            flat[start:end], output_gamut, look, look_strength, color_plan
        )
        # Oklab hue-preserving gamut fit replaces per-channel clipping for every mode.
        alpha = float(color_plan.gamut_fit_alpha) if color_plan is not None else 0.05
        out[start:end] = fit_to_output_gamut(piece, output_gamut, alpha=alpha).astype(np.float32, copy=False)
    return out.reshape(original_shape)


def _apply_output_color_ops(
    rgb_linear: Any,
    output_gamut: str,
    look: str,
    look_strength: float,
    color_plan: ColorGeometryPlan | None,
) -> Any:
    """Operations that must remain between output conversion and gamut fitting."""
    piece = np.nan_to_num(
        np.asarray(rgb_linear).astype(np.float32, copy=False),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )
    if look != "none":
        lab_l, lab_a, lab_b = rgb_to_oklab(piece, output_gamut)
        lab_l, lab_a, lab_b = look_engine.apply_look_oklab(
            lab_l, lab_a, lab_b, look, look_strength
        )
        piece = oklab_to_output_rgb(lab_l, lab_a, lab_b, output_gamut)
    if (
        color_plan is not None
        and abs(float(color_plan.display_highlight_chroma_retreat)) > 1e-9
    ):
        piece = _apply_display_highlight_chroma_retreat(
            piece,
            output_gamut,
            float(color_plan.display_highlight_chroma_retreat),
            float(color_plan.display_highlight_chroma_start),
            float(color_plan.display_highlight_chroma_end),
        )
    return piece


def plan_with_look_overrides(
    plan: ToneCompressionPlan | RenderPlan, look: str, look_strength: float = 1.0
) -> ToneCompressionPlan | RenderPlan:
    """Apply a chromatic look's AgX-core overrides (hue restore, faded target black) to the
    tone plan. Identity when the look carries none, so renders stay byte-identical."""
    tone = plan.tone if isinstance(plan, RenderPlan) else plan
    overrides = look_engine.agx_plan_overrides(look, look_strength, float(tone.hue_restore))
    if not overrides:
        return plan
    adjusted = replace(tone, **overrides)
    return replace(plan, tone=adjusted) if isinstance(plan, RenderPlan) else adjusted


def apply_agx_core(rgb_rec2020: Any, plan: ToneCompressionPlan) -> Any:
    """AgX in Rec.2020: inset -> log2/C1 -> linearize -> hue restore -> outset.

    The inset/outset channel crosstalk is what makes this AgX rather than a per-channel
    filmic curve; the darktable-derived sigmoid supplies the curve shape, while the plan's
    black/white EV keep the log2 window anchored on the exposure we set.
    """
    if fast_backend.supports_agx(plan):
        arr = np.ascontiguousarray(rgb_rec2020, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 3 and np.isfinite(arr).all():
            try:
                native_plan = fast_backend.compile_agx_plan(plan)
                return fast_backend.apply_agx_core_f32(arr, native_plan)
            except fast_backend.NativeKernelError:
                if fast_backend.strict_requested():
                    raise
            except Exception as exc:
                if fast_backend.strict_requested():
                    raise fast_backend.NativeKernelError(str(exc)) from exc

    inset, outset = agx_engine.formation_matrices(plan)
    mapped = agx_engine.apply_core(rgb_rec2020, plan, inset, outset)
    # Scene-driven purity compensation (dngscan/punch.py). This wrapper is the single
    # convergence point for the main render AND the auto-EV probe path, so both see the
    # same transform. Strength 0 short-circuits to identity.
    return punch_engine.apply_punch_rec2020(mapped, float(getattr(plan, "punch_strength", 0.0)))


# The 256 MiB tier was REMOVED (review batch 19): measured with the real
# production path (pass A + pass B + banded render, baseline and spatial
# sampled at the SAME tier), a 6 MP frame with all three operators needs
# ~466 MiB extra — the fixed context alone (72 MiB grain integral + the
# float64 bloom-source accumulator, its spread, the integral transient and
# the stored map on a 2048-wide grid) exceeds 256 MiB before a single band
# exists. Advertising a tier the implementation cannot honour is worse than
# offering fewer tiers.
def apply_tone_core(
    rgb_rec2020: Any,
    plan: ToneCompressionPlan,
    color_plan: ColorGeometryPlan | None = None,
    clip_masks_rgb: Any | None = None,
    raw_guidance: Any | None = None,
) -> Any:
    core = str(getattr(plan, "tone_core", "agx"))
    if core == "neutral":
        return neutral_engine.apply_neutral_core(rgb_rec2020, plan)
    if core == "lum":
        return lum_engine.apply_lum_core(rgb_rec2020, plan)
    if core == "gated":
        return gated_drt_engine.apply_gated_core(
            rgb_rec2020, plan, color_plan, clip_masks_rgb, raw_guidance
        )
    return apply_agx_core(rgb_rec2020, plan)


def sensor_px_per_render_px(bundle: RawBundle, h: int, w: int) -> float:
    """Area-equivalent sensor-pixel scale; band selection also checks both axes."""
    from .sampling_geometry import sensor_px_per_render_axes
    sy, sx = sensor_px_per_render_axes(bundle, h, w)
    return float((sy * sx) ** 0.5)


def _prepare_chroma_nr_map(
    bundle: RawBundle,
    tone_plan: Any,
    color_plan: Any,
    flat_scene: Any,
    clip_masks: Any,
    h: int,
    w: int,
    scene_transform: str,
    scene_transform_strength: float,
    wb_adapt: Any,
    retreat_strength: float | None = None,
    analysis: Analysis | None = None,
) -> Any:
    """Pass 0: the chroma-NR correction map, or None when the dial is 0.

    ``retreat_strength`` overrides the clip-retreat strength read from
    ``color_plan`` — the HDR entries own theirs (hdr_plan.color / the pair's
    shared intent strength) and must build the map from the SAME
    preprocessing their chunks apply.

    Walks the scene in row bands through intent, scene transform and clip retreat,
    then area-decimates the signed result for the calibrated chroma correction.
    """
    amount = float(getattr(tone_plan, "chroma_nr", 0.0) or 0.0)
    if amount <= 0.0:
        bundle.chroma_nr_status = "disabled"
        bundle.chroma_nr_reason = None
        return None
    from .chroma_nr import chroma_correction_map
    from .spatial import area_decimate_rows, spread_grid_shape, spatial_band_rows
    from .noise_propagation import calibrated_chroma_variance

    model = getattr(analysis, "noise_model", None) or getattr(bundle, "noise_model", None)
    if model is None or getattr(model, "status", None) != "valid":
        bundle.chroma_nr_status = "skipped"
        bundle.chroma_nr_reason = "independent noise calibration unavailable"
        return None
    if scene_transform != "none" and scene_transform_strength != 0:
        bundle.chroma_nr_status = "skipped"
        bundle.chroma_nr_reason = "nonlinear scene-transform noise propagation unavailable"
        return None
    descriptor = getattr(bundle, "noise_decode", None) or {}
    if not descriptor.get("supported", False):
        bundle.chroma_nr_status = "skipped"
        bundle.chroma_nr_reason = descriptor.get("reason") or "decoder noise propagation unavailable"
        return None

    dh, dw = spread_grid_shape(h, w)
    acc = np.zeros((dh, dw, 3), dtype=np.float64)
    invalid_acc = np.zeros((dh, dw, 1), dtype=np.float64)
    band = spatial_band_rows(w)
    if retreat_strength is None:
        retreat_strength = (
            float(color_plan.raw_clip_retreat_strength)
            if color_plan is not None else 0.0
        )
    retreat_strength = float(retreat_strength)
    for y0 in range(0, h, band):
        y1 = min(y0 + band, h)
        s0, e0 = y0 * w, y1 * w
        rec = scene_intent_rec2020(flat_scene[s0:e0, :3], bundle)
        rec = scene_transform_engine.apply_scene_transform_rec2020(
            rec, scene_transform, scene_transform_strength, wb_adapt
        )
        if clip_masks is not None:
            invalid = np.any(np.asarray(clip_masks[s0:e0]) > 0, axis=-1)
            area_decimate_rows(
                invalid.astype(np.float32).reshape(-1, w, 1),
                y0, h, w, dh, dw, invalid_acc,
            )
        if clip_masks is not None and retreat_strength > 0.0:
            rec = retreat_engine.apply_clip_retreat_rec2020(
                rec, clip_masks[s0:e0], retreat_strength
            )
        area_decimate_rows(
            np.asarray(rec, dtype=np.float32).reshape(-1, w, 3),
            y0, h, w, dh, dw, acc,
        )
    # Sensor pixels per decimated cell: the render grid's own ratio times
    # the render's sensor scale (identity on a full decode).
    factor = (
        max(h, w) * sensor_px_per_render_px(bundle, h, w) / max(max(dh, dw), 1)
    )
    from .sampling_geometry import sensor_px_per_render_axes
    sy, sx = sensor_px_per_render_axes(bundle, h, w)
    axis_factors = (sy * h / dh, sx * w / dw)
    dec32 = acc.astype(np.float32)
    del acc
    propagated = calibrated_chroma_variance(bundle, model, dec32, return_validity=True)
    variance, reason = propagated[:2]
    if variance is None:
        bundle.chroma_nr_status = "skipped"
        bundle.chroma_nr_reason = reason
        return None
    bundle.chroma_nr_status = "active-approximate"
    bundle.chroma_nr_reason = reason
    return chroma_correction_map(
        dec32, amount, decimation_factor=factor, decimation_axis_factors=axis_factors,
        chroma_variance=variance,
        valid_mask=(invalid_acc[..., 0] == 0) & propagated[2],
    )


def scene_render_to_display_linear(
    bundle: RawBundle,
    plan: ToneCompressionPlan | RenderPlan,
    output_gamut: str = "srgb",
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    analysis: Analysis | None = None,
) -> Any:
    """Scene-linear -> display-linear through agx, gated, lum, or neutral.

    Named for the pipeline stage, not the AgX core specifically: `plan.tone.tone_core`
    selects which core runs (see apply_tone_core), and clip retreat + gamut fit come
    from `plan.color` regardless of core.
    """
    tone_plan = plan.tone if isinstance(plan, RenderPlan) else plan
    color_plan = plan.color if isinstance(plan, RenderPlan) else None
    scene = bundle.scene_rec2020_render
    h, w = scene.shape[:2]
    flat_scene = scene.reshape(-1, scene.shape[-1])
    out = np.empty((flat_scene.shape[0], 3), dtype=np.float32)
    chunk = 1_000_000
    clip_masks = None
    raw_guidance = None
    if color_plan is not None and getattr(bundle, "clip_masks", None) is not None:
        clip_masks = retreat_engine.clip_masks_for_render(bundle, (h, w))
        if str(getattr(tone_plan, "tone_core", "agx")) == "gated":
            # Review batch 21 item 5: same evidence as the streaming u8 path
            # below — without the analysis the gated guidance builds against
            # metadata white and no sensor-SNR prior, so a linear render that
            # ran first diverged from the u8 render of the same plan.
            raw_guidance = guidance_engine.raw_guidance_for_shape(
                bundle, (h, w), analysis
            )

    wb_adapt = scene_transform_engine.window_transport(bundle)
    chroma_map = _prepare_chroma_nr_map(
        bundle, tone_plan, color_plan, flat_scene, clip_masks, h, w,
        scene_transform, scene_transform_strength, wb_adapt,
        analysis=analysis,
    )
    for start in range(0, flat_scene.shape[0], chunk):
        end = min(start + chunk, flat_scene.shape[0])
        starte, ende = start, end
        rec = scene_intent_rec2020(flat_scene[starte:ende, :3], bundle)
        rec = scene_transform_engine.apply_scene_transform_rec2020(
            rec, scene_transform, scene_transform_strength, wb_adapt
        )
        if clip_masks is not None and float(color_plan.raw_clip_retreat_strength) > 0.0:
            rec = retreat_engine.apply_clip_retreat_rec2020(
                rec,
                clip_masks[starte:ende],
                float(color_plan.raw_clip_retreat_strength),
            )
        if chroma_map is not None:
            from .chroma_nr import apply_chroma_correction_flat

            rec = apply_chroma_correction_flat(
                rec, chroma_map, starte, ende, h, w
            )
        mapped_rec = apply_tone_core(
            rec,
            tone_plan,
            color_plan,
            clip_masks[start:end] if clip_masks is not None else None,
            guidance_engine.flatten_raw_guidance(raw_guidance, start, end) if raw_guidance is not None else None,
        )
        if display_filter != "none" and filter_strength > 0.0:
            output_linear = filter_engine.apply_display_filter_rec2020(
                mapped_rec, output_gamut, display_filter, filter_strength, scene_rec2020=rec
            )
        else:
            output_linear = rec2020_to_output(mapped_rec, output_gamut)
        output_linear = np.nan_to_num(output_linear, nan=0.0, posinf=1e6, neginf=-1e6)
        out[start:end] = output_linear.astype(np.float32, copy=False)
    return out.reshape(h, w, 3)

# Back-compat alias: the pipeline ran only AgX when this was named; kept for external
# callers. Prefer scene_render_to_display_linear.
scene_render_to_agx_linear = scene_render_to_display_linear


def scene_render_to_agx_u8(
    bundle: RawBundle,
    plan: ToneCompressionPlan | RenderPlan,
    output_gamut: str = "srgb",
    look: str = "none",
    look_strength: float = 1.0,
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    analysis: Any = None,
) -> Any:
    # Review batch 22: the legacy public entry forwards the Analysis like the
    # streaming u8 path (batch 21 item 2) — a gated plan rendered through
    # here otherwise built its guidance without the sensor-SNR evidence.
    return output_linear_to_u8(
        scene_render_to_display_linear(
            bundle,
            plan_with_look_overrides(plan, look, look_strength),
            output_gamut,
            display_filter,
            filter_strength,
            scene_transform,
            scene_transform_strength,
            analysis=analysis,
        ),
        output_gamut,
        look,
        look_strength,
        plan.color if isinstance(plan, RenderPlan) else None,
    )


def render_output_linear(
    bundle: RawBundle,
    analysis: Analysis | None,
    output_gamut: str = "srgb",
    tone_plan: ToneCompressionPlan | RenderPlan | None = None,
    look: str = "none",
    look_strength: float = 1.0,
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
) -> Any:
    if look != "none" and display_filter != "none":
        raise ValueError("色度 look 与输出滤镜不能同时启用")
    if analysis is None:
        raise ValueError("AgX 导出需要分析结果")
    plan = tone_plan if tone_plan is not None else build_render_plan(
        bundle,
        analysis,
        "agx",
        output_gamut,
        scene_transform,
        scene_transform_strength,
        tone_core=tone_core,
        lum_norm=lum_norm,
        agx_primaries=agx_primaries,
    )
    effective_plan = plan_with_look_overrides(plan, look, look_strength)
    agx_linear = scene_render_to_display_linear(
        bundle,
        effective_plan,
        output_gamut,
        display_filter,
        filter_strength,
        scene_transform,
        scene_transform_strength,
        analysis=analysis,
    )
    color_plan = effective_plan.color if isinstance(effective_plan, RenderPlan) else None
    # This raster belongs to this render. Finalize one chunk before replacing
    # it, retaining one full-size float master instead of a second RGB copy.
    return finalize_output_linear(agx_linear, output_gamut, look, look_strength,
                                  color_plan, out=agx_linear)


def render_output_encoded_float(
    bundle: RawBundle,
    analysis: Analysis | None,
    output_gamut: str = "srgb",
    tone_plan: ToneCompressionPlan | RenderPlan | None = None,
    look: str = "none",
    look_strength: float = 1.0,
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
) -> Any:
    """Finished nonlinear SDR float32 RGB, without output quantization.

    All scene/display operations match the linear renderer. The output OETF
    is applied once, in bounded chunks over the caller-owned finished master.
    The target codec owns its bit-depth quantization and dither.
    """
    rgb = render_output_linear(
        bundle, analysis, output_gamut, tone_plan, look, look_strength,
        display_filter, filter_strength, scene_transform,
        scene_transform_strength, tone_core, lum_norm, agx_primaries,
    )
    flat = rgb.reshape(-1, 3)
    for start in range(0, len(flat), STREAM_RENDER_CHUNK):
        end = min(start + STREAM_RENDER_CHUNK, len(flat))
        flat[start:end] = encode_display_linear(flat[start:end], output_gamut)
    return rgb


def render_output_u8(
    bundle: RawBundle,
    analysis: Analysis | None,
    output_gamut: str = "srgb",
    tone_plan: ToneCompressionPlan | RenderPlan | None = None,
    look: str = "none",
    look_strength: float = 1.0,
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    dither_noise: Any | None = None,
    capture_mapped: Any | None = None,
) -> Any:
    """Render SDR bytes; optionally capture pre-quantization Rec.2020 pixels."""
    if look != "none" and display_filter != "none":
        raise ValueError("色度 look 与输出滤镜不能同时启用")
    if analysis is None:
        raise ValueError("AgX 导出需要分析结果")
    plan = tone_plan if tone_plan is not None else build_render_plan(
        bundle,
        analysis,
        "agx",
        output_gamut,
        scene_transform,
        scene_transform_strength,
        tone_core=tone_core,
        lum_norm=lum_norm,
        agx_primaries=agx_primaries,
    )
    effective_plan = plan_with_look_overrides(plan, look, look_strength)
    effective_tone = effective_plan.tone if isinstance(effective_plan, RenderPlan) else effective_plan
    color_plan = effective_plan.color if isinstance(effective_plan, RenderPlan) else None
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
    has_output_color_ops = look != "none" or (
        color_plan is not None
        and abs(float(color_plan.display_highlight_chroma_retreat)) > 1e-9
    )
    native_rec2020_input = native_output_plan is not None and not has_output_color_ops and (
        display_filter == "none" or filter_strength <= 0.0
    )

    scene = bundle.scene_rec2020_render
    h, w = scene.shape[:2]
    flat_scene = scene.reshape(-1, scene.shape[-1])
    out = np.empty((flat_scene.shape[0], 3), dtype=np.uint8)
    flat_dither_noise = None
    flat_dither_noise_a = None
    flat_dither_noise_b = None
    if dither_noise is not None:
        if isinstance(dither_noise, (tuple, list)) and len(dither_noise) == 2:
            first = np.asarray(dither_noise[0], dtype=np.float32)
            second = np.asarray(dither_noise[1], dtype=np.float32)
            if first.shape != (h, w, 3) or second.shape != (h, w, 3):
                raise ValueError(
                    "dither noise plane shapes "
                    f"{first.shape}/{second.shape} do not match render {(h, w, 3)}"
                )
            flat_dither_noise_a = np.ascontiguousarray(first).reshape(-1, 3)
            flat_dither_noise_b = np.ascontiguousarray(second).reshape(-1, 3)
        else:
            # Compatibility only.  A single combined plane cannot preserve the
            # authoritative two-plane float32 operation order.
            noise = np.asarray(dither_noise, dtype=np.float32)
            if noise.shape != (h, w, 3):
                raise ValueError(
                    f"dither noise shape {noise.shape} does not match render {(h, w, 3)}"
                )
            flat_dither_noise = np.ascontiguousarray(noise).reshape(-1, 3)
    # S3: the split follows who does the heavy lifting for THIS plan (see
    # cpu_budget.split_for); the native tone core changes the answer.
    from .cpu_budget import split_for

    _native_core = (
        fast_backend.available()
        and str(getattr(effective_tone, "tone_core", "agx")) == "agx"
        and fast_backend.supports_agx(effective_tone)
    )
    stream_outer, stream_inner = split_for(_native_core)
    quantize_chunk_size = STREAM_QUANTIZE_CHUNK
    if flat_scene.shape[0] < STREAM_THREAD_MIN_PIXELS:
        render_chunk_size = quantize_chunk_size
    elif scene_transform != "none" or look != "none":
        # NumPy scene transforms/looks expose more independent work than the native
        # baseline chain. A smaller exact divisor keeps all render workers occupied;
        # quantization still joins in the historical 1M-pixel order below.
        render_chunk_size = STREAM_COMPLEX_COLOR_CHUNK
    else:
        render_chunk_size = STREAM_RENDER_CHUNK
    if quantize_chunk_size % render_chunk_size != 0:
        # The grouped flush below fires on exact boundary equality; a non-divisible
        # pairing would never flush and silently leave trailing pixels at zero.
        raise ValueError(
            f"stream chunking misaligned: quantize {quantize_chunk_size} "
            f"must be a multiple of render {render_chunk_size}"
        )

    clip_masks = None
    raw_guidance = None
    if color_plan is not None and getattr(bundle, "clip_masks", None) is not None:
        clip_masks = retreat_engine.clip_masks_for_render(bundle, (h, w))
        if str(getattr(effective_tone, "tone_core", "agx")) == "gated":
            raw_guidance = guidance_engine.raw_guidance_for_shape(bundle, (h, w), analysis)

    wb_adapt = scene_transform_engine.window_transport(bundle)
    chroma_map = _prepare_chroma_nr_map(
        bundle, effective_tone, color_plan, flat_scene, clip_masks, h, w,
        scene_transform, scene_transform_strength, wb_adapt,
        analysis=analysis,
    )
    def render_post_tone_chunk(start: int, end: int) -> Any:
        starte, ende = start, end
        rec = scene_intent_rec2020(flat_scene[starte:ende, :3], bundle)
        rec = scene_transform_engine.apply_scene_transform_rec2020(
            rec, scene_transform, scene_transform_strength, wb_adapt
        )
        sample_masks = clip_masks[starte:ende] if clip_masks is not None else None
        if (
            sample_masks is not None
            and color_plan is not None
            and float(color_plan.raw_clip_retreat_strength) > 0.0
        ):
            rec = retreat_engine.apply_clip_retreat_rec2020(
                rec, sample_masks, float(color_plan.raw_clip_retreat_strength)
            )
        if chroma_map is not None:
            from .chroma_nr import apply_chroma_correction_flat

            rec = apply_chroma_correction_flat(
                rec, chroma_map, starte, ende, h, w
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
        if capture_mapped is not None:
            capture_mapped[start:end] = mapped_rec
        if native_rec2020_input:
            return np.ascontiguousarray(mapped_rec, dtype=np.float32)
        if display_filter != "none" and filter_strength > 0.0:
            output_linear = filter_engine.apply_display_filter_rec2020(
                mapped_rec,
                output_gamut,
                display_filter,
                filter_strength,
                scene_rec2020=rec,
            )
        else:
            output_linear = rec2020_to_output(mapped_rec, output_gamut)
        output_linear = np.nan_to_num(
            output_linear, nan=0.0, posinf=1e6, neginf=-1e6
        ).astype(np.float32, copy=False)
        if native_output_plan is not None:
            return np.ascontiguousarray(
                _apply_output_color_ops(
                    output_linear,
                    output_gamut,
                    look,
                    look_strength,
                    color_plan,
                ),
                dtype=np.float32,
            )
        finalized = finalize_output_linear(
            output_linear, output_gamut, look, look_strength, color_plan
        )
        return np.nan_to_num(
            finalized.astype(np.float32, copy=False),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

    ranges = [
        (start, min(start + render_chunk_size, flat_scene.shape[0]))
        for start in range(0, flat_scene.shape[0], render_chunk_size)
    ]
    rng = np.random.default_rng(0)

    def quantize_chunk(start: int, end: int, pixels: Any, noise_pair: Any = None) -> None:
        noise = (
            flat_dither_noise[start:end]
            if flat_dither_noise is not None
            else None
        )
        if noise_pair is not None:
            noise_a, noise_b = noise_pair
        elif flat_dither_noise_a is not None:
            noise_a = flat_dither_noise_a[start:end]
            noise_b = flat_dither_noise_b[start:end]
        elif noise is None:
            noise_a, noise_b = generate_dither_noise(rng, pixels.shape)
        else:
            noise_a = noise_b = None
        if native_output_plan is not None:
            try:
                if native_rec2020_input and noise is not None:
                    encoded_u8 = fast_backend.finalize_rec2020_u8_noise_f32(
                        pixels, noise, native_output_plan
                    )
                elif native_rec2020_input:
                    encoded_u8 = fast_backend.finalize_rec2020_u8_f32(
                        pixels, noise_a, noise_b, native_output_plan
                    )
                elif noise is not None:
                    encoded_u8 = fast_backend.finalize_output_u8_noise_f32(
                        pixels, noise, native_output_plan
                    )
                else:
                    encoded_u8 = fast_backend.finalize_output_u8_f32(
                        pixels, noise_a, noise_b, native_output_plan
                    )
                out[start:end] = encoded_u8
                return
            except Exception as exc:
                if fast_backend.strict_requested():
                    if isinstance(exc, fast_backend.NativeKernelError):
                        raise
                    raise fast_backend.NativeKernelError(str(exc)) from exc

            if native_rec2020_input:
                pixels = rec2020_to_output(pixels, output_gamut)
                pixels = _apply_output_color_ops(
                    pixels, output_gamut, look, look_strength, color_plan
                )
            pixels = fit_to_output_gamut(
                pixels, output_gamut, alpha=gamut_alpha
            ).astype(np.float32, copy=False)
        encoded = encode_display_linear(pixels, output_gamut)
        if noise is not None:
            out[start:end] = dither_quantize_u8_with_tpdf(encoded, noise)
        else:
            out[start:end] = dither_quantize_u8_with_noise(
                encoded, noise_a, noise_b
            )

    def consume_in_quantize_groups(results: Any) -> None:
        group_start = 0
        group_parts: list[Any] = []
        for start, end, pixels in results:
            group_parts.append(pixels)
            group_end = min(group_start + quantize_chunk_size, flat_scene.shape[0])
            if end == group_end:
                if native_output_plan is not None and len(group_parts) > 1:
                    # Draw the complete group's A then B in the historical order;
                    # pass borrowed slices without concatenating the RGB chunks.
                    pair = (generate_dither_noise(rng, (group_end - group_start, 3))
                            if flat_dither_noise is None and flat_dither_noise_a is None
                            else None)
                    offset = group_start
                    for part in group_parts:
                        next_offset = offset + len(part)
                        part_noise = (tuple(n[offset - group_start:next_offset - group_start]
                                            for n in pair) if pair is not None else None)
                        quantize_chunk(offset, next_offset, part, part_noise)
                        offset = next_offset
                    del pair, part_noise, part
                else:
                    merged = group_parts[0] if len(group_parts) == 1 else np.concatenate(group_parts, axis=0)
                    quantize_chunk(group_start, group_end, merged)
                group_start = group_end
                group_parts = []

    if flat_scene.shape[0] < STREAM_THREAD_MIN_PIXELS or len(ranges) < 2:
        consume_in_quantize_groups(
            (start, end, render_post_tone_chunk(start, end)) for start, end in ranges
        )
    else:
        # Sized like the HDR pair pool: chunked streaming bounds each worker's
        # temporaries to chunk scale, so the old two-worker cap only left the
        # NumPy pre-tone stages under-parallelized. Worker count cannot change
        # output bytes — chunks are independent and the consumer quantizes in
        # group order with the same serial RNG sequence.
        workers = min(stream_outer, len(ranges))
        from .cpu_budget import claim, set_inner

        # S3: publish this section's inner budget so the native kernels and
        # the inner operator pools stop stacking their own full-width pools
        # on top of this one (the review measured 49 threads from that
        # nesting; the bound is now TOTAL + a small constant).
        with claim(stream_inner), ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="dngscan-render",
            initializer=set_inner, initargs=(stream_inner,),
        ) as pool:
            pending: dict[int, Any] = {}
            submit_idx = 0
            while submit_idx < min(workers, len(ranges)):
                start, end = ranges[submit_idx]
                pending[submit_idx] = pool.submit(render_post_tone_chunk, start, end)
                submit_idx += 1
            def ordered_results() -> Any:
                nonlocal submit_idx
                for idx, (start, end) in enumerate(ranges):
                    pixels = pending.pop(idx).result()
                    if submit_idx < len(ranges):
                        next_start, next_end = ranges[submit_idx]
                        pending[submit_idx] = pool.submit(
                            render_post_tone_chunk, next_start, next_end
                        )
                        submit_idx += 1
                    yield start, end, pixels

            consume_in_quantize_groups(ordered_results())
    return out.reshape(h, w, 3)
