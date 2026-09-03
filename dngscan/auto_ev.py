# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoded-scene brightness reference with final-output highlight safety."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from ._deps import np
from . import display_filter as filter_engine
from . import retreat as retreat_engine
from . import scene_transform as scene_transform_engine
from .color import RGB_TO_XYZ, output_gamut_space, rec2020_to_output
from .constants import EPS
from .models import (
    Analysis, AutoEvResult, RawBundle, RenderAdjustments, RenderPlan,
    ToneCompressionPlan,
)
from . import _fast as fast_backend
from .color import fit_to_output_gamut
from .render import (
    _apply_output_color_ops,
    apply_tone_core,
    finalize_output_linear,
    plan_with_look_overrides,
)
from .tone import build_render_plan, compute_exposure_gain, exposure_mode_for_tone_core, scene_intent_rec2020, scene_rec2020_to_float

EV_AUTO_TOKEN = "auto"


def parse_ev_value(value: str | float) -> float | str:
    if isinstance(value, str) and value.strip().lower() == EV_AUTO_TOKEN:
        return EV_AUTO_TOKEN
    return float(value)


def is_ev_auto(value: str | float) -> bool:
    return isinstance(value, str) and value.strip().lower() == EV_AUTO_TOKEN


def median_align_ev(mode: str, analysis: Analysis) -> float:
    """EV compensation that places the scene median on 18% gray after the mode anchor."""
    base_gain = compute_exposure_gain(mode, 0.0)
    return float(-analysis.median_vs_gray_ev - math.log2(max(base_gain, EPS)))


def anchored_median_ev(mode: str, analysis: Analysis, ev: float) -> float:
    gain = compute_exposure_gain(mode, ev)
    return float(analysis.median_vs_gray_ev + math.log2(max(gain, EPS)))


def scene_body_align_ev(plan: RenderPlan) -> float:
    """EV that places the decoder-specific reliable scene body median at 18% gray."""
    return float(-plan.scene.body_ev_p50)


def anchored_scene_body_ev(plan: RenderPlan, ev: float) -> float:
    """Reliable scene body median, in EV relative to 18% gray, after a manual EV."""
    return float(plan.scene.body_ev_p50 + float(ev))


NEAR_WHITE_LINEAR = 0.956


def output_highlight_stats(
    rgb_linear: Any, gamut: str, percentile_mask: Any | None = None
) -> tuple[float, float, float, float]:
    """(p99.9 luma%, p99.9 max-channel%, clipped-pixel%, near-white%) of an output-linear buffer.

    R3 item 5: `percentile_mask` restricts the two PERCENTILE figures to the
    reliable body (samples that were not already near-white at the baseline
    EV) — a scene's existing lamps and speculars are supposed to sit at the
    top and must not smuggle themselves into the body-brightness reading.
    The clip%/near-white% area figures always read the full sample: their
    baseline-relative growth budgets are how pre-existing emitters are
    excused, and masking them too would blind the search to NEW clipping
    inside the excluded set.
    """
    rgb = np.clip(np.nan_to_num(rgb_linear, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    matrix = RGB_TO_XYZ[output_gamut_space(gamut)]
    y = matrix[1, 0] * rgb[:, 0] + matrix[1, 1] * rgb[:, 1] + matrix[1, 2] * rgb[:, 2]
    y = np.clip(np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    max_channel = np.max(rgb, axis=1)
    y_pct = y
    max_pct = max_channel
    if percentile_mask is not None and int(np.count_nonzero(percentile_mask)) >= 256:
        y_pct = y[percentile_mask]
        max_pct = max_channel[percentile_mask]
    return (
        float(np.percentile(y_pct, 99.9) * 100.0),
        float(np.percentile(max_pct, 99.9) * 100.0),
        float(np.mean(np.any(rgb >= np.float32(0.999), axis=1)) * 100.0),
        float(np.mean(max_channel >= np.float32(NEAR_WHITE_LINEAR)) * 100.0),
    )


def output_highlight_margin(
    rgb_linear: Any,
    gamut: str,
    baseline: tuple[float, float, float, float] | None = None,
    percentile_mask: Any | None = None,
) -> float:
    """Positive margin means headroom before highlight risk thresholds.

    With `baseline` (the stats at the starting EV), the clip/near-white limits become
    growth budgets relative to that baseline: clipping that already exists in the capture
    (lamps, speculars — light sources are SUPPOSED to clip) does not count against the
    boost; only NEW clipping does. The luma/max-channel percentile limits stay absolute,
    but since R3 item 5 they are evaluated on the reliable body (`percentile_mask`,
    derived at the baseline EV) — a handful of pre-existing emitters used to push the
    absolute p99.9 gate past its limit at the STARTING EV, vetoing the whole search on
    exactly the dark lamp-lit scenes the brightness reference exists for."""
    if np is None:
        return 0.0
    y_p999, max_p999, clip_pct, near_pct = output_highlight_stats(
        rgb_linear, gamut, percentile_mask
    )
    clip_limit = 0.03
    near_limit = 0.25
    if baseline is not None:
        clip_limit = max(clip_limit, baseline[2] + 0.03)
        near_limit = max(near_limit, baseline[3] + 0.25)
    margin_luma = 92.0 - y_p999
    margin_rgb = 96.0 - max_p999
    margin_clip = clip_limit - clip_pct
    margin_near = near_limit - near_pct
    return float(min(margin_luma, margin_rgb, margin_clip * 10.0, margin_near))


def render_sample_linear_output(
    bundle: RawBundle,
    analysis: Analysis | None,
    gamut: str,
    ev: float,
    sample_rec2020: Any,
    tone_plan: ToneCompressionPlan | RenderPlan | None = None,
    look: str = "none",
    look_strength: float = 1.0,
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    punch_scale: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    sample_masks: Any | None = None,
    sample_raw_guidance: Any | None = None,
    adjustments: RenderAdjustments | None = None,
    spatial_shape: tuple | None = None,
) -> Any:
    from .grade import RENDER_MODE

    exposure_gain = compute_exposure_gain(exposure_mode_for_tone_core(tone_core), ev)
    ev_bundle = replace(bundle, exposure_gain=exposure_gain)
    rec = scene_intent_rec2020(sample_rec2020, bundle, exposure_gain)
    plan = tone_plan if tone_plan is not None else (
        build_render_plan(
            ev_bundle,
            analysis,
            RENDER_MODE,
            gamut,
            scene_transform,
            scene_transform_strength,
            punch_scale,
            tone_core,
            lum_norm,
            agx_primaries=agx_primaries,
            adjustments=adjustments,
        ) if analysis is not None else None
    )
    wb_adapt = scene_transform_engine.window_transport(ev_bundle)
    rec = scene_transform_engine.apply_scene_transform_rec2020(
        rec, scene_transform, scene_transform_strength, wb_adapt
    )
    color_plan = plan.color if isinstance(plan, RenderPlan) else None
    if color_plan is not None and sample_masks is not None and float(color_plan.raw_clip_retreat_strength) > 0.0:
        rec = retreat_engine.apply_clip_retreat_rec2020(
            rec, sample_masks, float(color_plan.raw_clip_retreat_strength)
        )
    effective_plan = plan_with_look_overrides(plan, look, look_strength) if plan is not None else None
    effective_tone = effective_plan.tone if isinstance(effective_plan, RenderPlan) else effective_plan
    eff_color = effective_plan.color if isinstance(effective_plan, RenderPlan) else color_plan
    if spatial_shape is not None:
        # Analog-optics probe (review batch 14): the samples are an
        # area-decimated IMAGE, so the §9 spatial operators participate in
        # the safe-EV answer at preview scale (halation and bloom are
        # low-frequency and survive decimation; grain area-averages out).
        from .film_develop import apply_film_core

        mapped_rec = apply_film_core(rec, effective_tone, spatial_shape=spatial_shape)
    else:
        mapped_rec = apply_tone_core(rec, effective_tone, eff_color, sample_masks, sample_raw_guidance)
    if display_filter != "none" and filter_strength > 0.0:
        output_linear = filter_engine.apply_display_filter_rec2020(
            mapped_rec, gamut, display_filter, filter_strength, scene_rec2020=rec
        )
    else:
        output_linear = rec2020_to_output(mapped_rec, gamut)
    return _probe_finalize_linear(output_linear, gamut, look, look_strength, color_plan)


def _probe_finalize_linear(
    output_linear: Any,
    gamut: str,
    look: str,
    look_strength: float,
    color_plan: Any,
) -> Any:
    """The probe's finalize: same color ops, gamut fit via the native kernel.

    Declared operating point (B5): the probe's bisection quantum is 1/128 EV;
    the native/NumPy fit difference (measured max 2.8e-5 in float) can at most
    flip a borderline threshold test by one quantum. Any native failure falls
    back to the exact NumPy finalize; strict mode surfaces the failure.
    """
    if fast_backend.supports_output_finalizer():
        alpha = float(color_plan.gamut_fit_alpha) if color_plan is not None else 0.05
        try:
            plan = fast_backend.compile_output_plan(gamut, alpha)
            piece = _apply_output_color_ops(
                output_linear, gamut, look, look_strength, color_plan
            )
            return fast_backend.fit_output_gamut_f32(
                np.ascontiguousarray(piece, dtype=np.float32), plan
            )
        except Exception as exc:
            if fast_backend.strict_requested():
                if isinstance(exc, fast_backend.NativeKernelError):
                    raise
                raise fast_backend.NativeKernelError(str(exc)) from exc
    return finalize_output_linear(output_linear, gamut, look, look_strength, color_plan)


def max_safe_ev(
    bundle: RawBundle,
    analysis: Analysis | None,
    gamut: str,
    from_ev: float = 0.0,
    max_samples: int = 220_000,
    search_hi: float = 3.0,
    look: str = "none",
    look_strength: float = 1.0,
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    punch_scale: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    adjustments: RenderAdjustments | None = None,
    tone_plan: RenderPlan | None = None,
    endpoint_mode: str = "adaptive",
    film_curve: str = "none",
    film_mode: str = "observe",
    film_crossover: str | None = None,
    film_exposure_ev: float = 0.0,
    film_print_timing: str = "fixed",
    film_print_medium: str = "",
    film_print_exposure_ev: float = 0.0,
    film_development: str = "measured_default",
    film_interimage: str = "declared",
    film_appearance: str = "technical",
    film_appearance_strength: float = 1.0,
    film_appearance_variant: str = "reference",
    film_richness: float = 0.0,
    film_color_density: float = 0.0,
    film_neutral_bias: float = 1.0,
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
    film_media_scatter: str = "declared",
    film_interimage_beta_dial: float | None = None,
    color_head_y: float = 0.0,
    color_head_m: float = 0.0,
    chroma_nr: float = 0.0,
    lens_filter: str | None = None,
) -> float:
    """Largest EV (>= from_ev) whose preview-scale output stays below highlight thresholds.

    The probe must consult the same compiled curve the real render will use, so the
    caller's endpoint mode and full film declaration (curve, mode, crossover, color
    head) and declared lens filter all participate; a None lens_filter keeps whatever
    the bundle already declares.

    chroma_nr rides the plan for domain validation only: the sampled probe never
    builds the chroma-NR map (it needs the full spread grid), so the highlight gates
    see the un-repaired scene. The repair is zero-luma at the scene stage and only
    ever shrinks chroma by MAD-scale amounts inside the 8-128 px band, so the probe
    is on the conservative side by far less than its 1/128 EV bisection quantum
    (math review 2026-09-03).
    """
    if np is None:
        return float(from_ev)
    if lens_filter is not None and lens_filter != getattr(bundle, "lens_filter", "none"):
        bundle = replace(bundle, lens_filter=lens_filter)
    if tone_plan is None and analysis is not None:
        from .grade import RENDER_MODE

        reference_bundle = replace(
            bundle,
            exposure_gain=compute_exposure_gain(exposure_mode_for_tone_core(tone_core), 0.0),
        )
        tone_plan = build_render_plan(
            reference_bundle,
            analysis,
            RENDER_MODE,
            gamut,
            scene_transform,
            scene_transform_strength,
            punch_scale,
            tone_core,
            lum_norm,
            agx_primaries=agx_primaries,
            adjustments=adjustments,
            film_curve=film_curve,
            film_mode=film_mode,
            film_crossover=film_crossover,
        film_exposure_ev=film_exposure_ev,
        film_print_timing=film_print_timing,
        film_print_medium=film_print_medium,
        film_print_exposure_ev=film_print_exposure_ev,
        film_development=film_development,
        film_interimage=film_interimage,
        film_appearance=film_appearance,
        film_appearance_strength=film_appearance_strength,
        film_appearance_variant=film_appearance_variant,
        film_richness=film_richness,
        film_color_density=film_color_density,
        film_neutral_bias=film_neutral_bias,
        film_dev_contrast=film_dev_contrast,
        film_dev_fog=film_dev_fog,
        film_dev_density=film_dev_density,
        film_compression=film_compression,
        film_compression_knee=film_compression_knee,
        film_highlight_density=film_highlight_density,
        film_grain=film_grain,
        film_halation=film_halation,
        film_bloom=film_bloom,
        film_optics_seed=film_optics_seed,
        film_media_scatter=film_media_scatter,
        film_interimage_beta_dial=film_interimage_beta_dial,
            color_head_y=color_head_y,
            color_head_m=color_head_m,
        chroma_nr=chroma_nr,
            endpoint_mode=endpoint_mode,
        )

    flat = bundle.scene_rec2020_render.reshape(-1, bundle.scene_rec2020_render.shape[-1])
    step = max(1, int(math.ceil(flat.shape[0] / max_samples)))
    probe_tone = tone_plan.tone if isinstance(tone_plan, RenderPlan) else tone_plan
    spatial_shape = None

    def _probe_needs_spatial(tone: Any) -> bool:
        # R4 F4: the probe's decimated image dilutes point speculars by the
        # cell area (~277x at 61 MP), so it is only worth that cost for the
        # LOOK amounts whose spread genuinely moves the safe-EV answer
        # (bloom/halation/grain, review batch 14). The scatter-only default
        # (R3: media scatter declared, all amounts 0) is a conservative
        # sub-0.02 mm redistribution that cannot rescue a clipped highlight,
        # while the decimation would blind the clip budgets to exactly the
        # pinpoint emitters they police — so it keeps the strided real-pixel
        # probe.
        if tone is None or str(getattr(tone, "film_mode", "observe")) != "full" \
                or str(getattr(tone, "curve_preset", "none")) == "none":
            return False
        return any(
            float(getattr(tone, k, 0.0) or 0.0) > 0.0
            for k in ("film_grain", "film_halation", "film_bloom")
        )

    if _probe_needs_spatial(probe_tone):
        # Decimated-image probe: strided flat samples cannot carry the
        # spatial operators, so bloom/halation silently sat out the safe-EV
        # answer (review batch 14).
        from .film_optics import area_decimate

        sh, sw = bundle.scene_rec2020_render.shape[:2]
        scale = min(1.0, (max_samples / float(sh * sw)) ** 0.5)
        dh = max(int(round(sh * scale)), 16)
        dw = max(int(round(sw * scale)), 16)
        sample_rgb = area_decimate(
            bundle.scene_rec2020_render[:, :, :3], dh, dw
        ).reshape(-1, 3)
        spatial_shape = (dh, dw)
        sample_masks = None
        sample_raw_guidance = None
        if getattr(bundle, "clip_masks", None) is not None:
            masks = retreat_engine.clip_masks_for_shape(
                bundle, (sh, sw)
            ).astype(np.float32)
            sample_masks = area_decimate(masks, dh, dw).reshape(-1, 3)
    else:
        sample_rgb = flat[::step, :3]
        sample_masks = None
    sample_raw_guidance = None
    if spatial_shape is None and getattr(bundle, "clip_masks", None) is not None:
        masks = retreat_engine.clip_masks_for_shape(bundle, bundle.scene_rec2020_render.shape[:2]).reshape(-1, 3)
        sample_masks = masks[::step]
        if tone_core == "gated":
            from .guidance import flatten_raw_guidance, raw_guidance_for_shape

            guidance = raw_guidance_for_shape(bundle, bundle.scene_rec2020_render.shape[:2], analysis)
            sample_raw_guidance = flatten_raw_guidance(guidance, 0, masks.shape[0], step=step)
    baseline_stats: tuple[float, float, float, float] | None = None
    body_percentile_mask: Any | None = None

    def margin_at(ev: float) -> float:
        rgb = render_sample_linear_output(
            bundle,
            analysis,
            gamut,
            ev,
            sample_rgb,
            tone_plan=tone_plan,
            look=look,
            look_strength=look_strength,
            display_filter=display_filter,
            filter_strength=filter_strength,
            scene_transform=scene_transform,
            scene_transform_strength=scene_transform_strength,
            punch_scale=punch_scale,
            tone_core=tone_core,
            lum_norm=lum_norm,
            agx_primaries=agx_primaries,
            sample_masks=sample_masks,
            sample_raw_guidance=sample_raw_guidance,
            adjustments=adjustments,
            spatial_shape=spatial_shape,
        )
        return output_highlight_margin(rgb, gamut, baseline_stats, body_percentile_mask)

    baseline_rgb = render_sample_linear_output(
        bundle,
        analysis,
        gamut,
        from_ev,
        sample_rgb,
        tone_plan=tone_plan,
        look=look,
        look_strength=look_strength,
        display_filter=display_filter,
        filter_strength=filter_strength,
        scene_transform=scene_transform,
        scene_transform_strength=scene_transform_strength,
        punch_scale=punch_scale,
        tone_core=tone_core,
        lum_norm=lum_norm,
        agx_primaries=agx_primaries,
        sample_masks=sample_masks,
        sample_raw_guidance=sample_raw_guidance,
        adjustments=adjustments,
        spatial_shape=spatial_shape,
    )
    # R3 item 5: the reliable body is fixed at the BASELINE — the samples not
    # already near-white before any boost. The same sample array renders at
    # every probed EV, so the mask stays aligned across the whole search.
    _base = np.clip(
        np.nan_to_num(baseline_rgb, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
    )
    _mask = np.max(_base, axis=1) < np.float32(NEAR_WHITE_LINEAR)
    if int(np.count_nonzero(_mask)) >= 256:
        body_percentile_mask = _mask
    del _base, _mask
    baseline_stats = output_highlight_stats(baseline_rgb, gamut, body_percentile_mask)
    if output_highlight_margin(
        baseline_rgb, gamut, baseline_stats, body_percentile_mask
    ) <= 0.0:
        return float(from_ev)

    low = float(from_ev)
    high = low + 0.5
    while margin_at(high) > 0.0 and high < from_ev + search_hi:
        low = high
        high += 0.5

    if margin_at(high) > 0.0:
        return float(high)

    for _ in range(6):
        mid = (low + high) * 0.5
        if margin_at(mid) > 0.0:
            low = mid
        else:
            high = mid
    return float(low)


def compute_auto_ev(
    bundle: RawBundle,
    analysis: Analysis,
    gamut: str = "p3",
    baseline_ev: float = 0.0,
    look: str = "none",
    look_strength: float = 1.0,
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    punch_scale: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    adjustments: RenderAdjustments | None = None,
    endpoint_mode: str = "adaptive",
    film_curve: str = "none",
    film_mode: str = "observe",
    film_crossover: str | None = None,
    film_exposure_ev: float = 0.0,
    film_print_timing: str = "fixed",
    film_print_medium: str = "",
    film_print_exposure_ev: float = 0.0,
    film_development: str = "measured_default",
    film_interimage: str = "declared",
    film_appearance: str = "technical",
    film_appearance_strength: float = 1.0,
    film_appearance_variant: str = "reference",
    film_richness: float = 0.0,
    film_color_density: float = 0.0,
    film_neutral_bias: float = 1.0,
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
    film_media_scatter: str = "declared",
    film_interimage_beta_dial: float | None = None,
    color_head_y: float = 0.0,
    color_head_m: float = 0.0,
    chroma_nr: float = 0.0,
    lens_filter: str | None = None,
) -> AutoEvResult:
    """Reference the reliable decoded scene body to 18% gray without changing EV 0.

    The body statistic is measured after the selected decoder and scene transform, while
    excluding unreliable RAW-clipped highlights. This makes the optional reference
    decoder-independent without introducing a hidden fixed correction. Highlight safety
    limits upward boost only; high-key scenes are never darkened toward gray.

    The internal reference plan compiles with the caller's endpoint mode, the full
    film declaration (curve preset, observe/full mode, crossover switch, enlarger
    color head) and declared lens filter, so the brightness reference and the
    highlight-safety cap are judged against the plan the real render will actually
    compile — not a simplified stand-in; a None lens_filter keeps whatever the
    bundle already declares.
    """
    from .grade import RENDER_MODE

    if lens_filter is not None and lens_filter != getattr(bundle, "lens_filter", "none"):
        bundle = replace(bundle, lens_filter=lens_filter)
    reference_bundle = replace(
        bundle,
        exposure_gain=compute_exposure_gain(exposure_mode_for_tone_core(tone_core), 0.0),
    )
    reference_plan = build_render_plan(
        reference_bundle,
        analysis,
        RENDER_MODE,
        gamut,
        scene_transform,
        scene_transform_strength,
        punch_scale,
        tone_core,
        lum_norm,
        agx_primaries=agx_primaries,
        adjustments=adjustments,
        film_curve=film_curve,
        film_mode=film_mode,
        film_crossover=film_crossover,
        film_exposure_ev=film_exposure_ev,
        film_print_timing=film_print_timing,
        film_print_medium=film_print_medium,
        film_print_exposure_ev=film_print_exposure_ev,
        film_development=film_development,
        film_interimage=film_interimage,
        film_appearance=film_appearance,
        film_appearance_strength=film_appearance_strength,
        film_appearance_variant=film_appearance_variant,
        film_richness=film_richness,
        film_color_density=film_color_density,
        film_neutral_bias=film_neutral_bias,
        film_dev_contrast=film_dev_contrast,
        film_dev_fog=film_dev_fog,
        film_dev_density=film_dev_density,
        film_compression=film_compression,
        film_compression_knee=film_compression_knee,
        film_highlight_density=film_highlight_density,
        film_grain=film_grain,
        film_halation=film_halation,
        film_bloom=film_bloom,
        film_optics_seed=film_optics_seed,
        film_media_scatter=film_media_scatter,
        film_interimage_beta_dial=film_interimage_beta_dial,
        color_head_y=color_head_y,
        color_head_m=color_head_m,
        chroma_nr=chroma_nr,
        endpoint_mode=endpoint_mode,
    )
    target = scene_body_align_ev(reference_plan)
    cap = max_safe_ev(
        bundle,
        analysis,
        gamut,
        from_ev=baseline_ev,
        look=look,
        look_strength=look_strength,
        display_filter=display_filter,
        filter_strength=filter_strength,
        scene_transform=scene_transform,
        scene_transform_strength=scene_transform_strength,
        punch_scale=punch_scale,
        tone_core=tone_core,
        lum_norm=lum_norm,
        agx_primaries=agx_primaries,
        adjustments=adjustments,
        tone_plan=reference_plan,
        endpoint_mode=endpoint_mode,
        film_curve=film_curve,
        film_mode=film_mode,
        film_crossover=film_crossover,
        film_exposure_ev=film_exposure_ev,
        film_print_timing=film_print_timing,
        film_print_medium=film_print_medium,
        film_print_exposure_ev=film_print_exposure_ev,
        film_development=film_development,
        film_interimage=film_interimage,
        film_appearance=film_appearance,
        film_appearance_strength=film_appearance_strength,
        film_appearance_variant=film_appearance_variant,
        film_richness=film_richness,
        film_color_density=film_color_density,
        film_neutral_bias=film_neutral_bias,
        film_dev_contrast=film_dev_contrast,
        film_dev_fog=film_dev_fog,
        film_dev_density=film_dev_density,
        film_compression=film_compression,
        film_compression_knee=film_compression_knee,
        film_highlight_density=film_highlight_density,
        film_grain=film_grain,
        film_halation=film_halation,
        film_bloom=film_bloom,
        film_optics_seed=film_optics_seed,
        film_media_scatter=film_media_scatter,
        film_interimage_beta_dial=film_interimage_beta_dial,
        color_head_y=color_head_y,
        color_head_m=color_head_m,
        chroma_nr=chroma_nr,
    )
    boost_target = max(target, baseline_ev)
    ev = min(boost_target, cap)
    limited = boost_target > cap + 1e-6
    return AutoEvResult(
        ev=float(ev),
        ev_median_target=float(target),
        ev_boost=float(ev - baseline_ev),
        highlight_limited=limited,
        highlight_cap_ev=float(cap),
        anchored_median_ev=anchored_scene_body_ev(reference_plan, ev),
    )


def resolve_export_ev(
    ev: str | float,
    bundle: RawBundle,
    analysis: Analysis,
    gamut: str,
    look: str = "none",
    look_strength: float = 1.0,
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    punch_scale: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    adjustments: RenderAdjustments | None = None,
    endpoint_mode: str = "adaptive",
    film_curve: str = "none",
    film_mode: str = "observe",
    film_crossover: str | None = None,
    film_exposure_ev: float = 0.0,
    film_print_timing: str = "fixed",
    film_print_medium: str = "",
    film_print_exposure_ev: float = 0.0,
    film_development: str = "measured_default",
    film_interimage: str = "declared",
    film_appearance: str = "technical",
    film_appearance_strength: float = 1.0,
    film_appearance_variant: str = "reference",
    film_richness: float = 0.0,
    film_color_density: float = 0.0,
    film_neutral_bias: float = 1.0,
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
    film_media_scatter: str = "declared",
    film_interimage_beta_dial: float | None = None,
    color_head_y: float = 0.0,
    color_head_m: float = 0.0,
    chroma_nr: float = 0.0,
    lens_filter: str | None = None,
) -> tuple[float, AutoEvResult | None]:
    if not is_ev_auto(ev):
        return float(ev), None
    result = compute_auto_ev(
        bundle,
        analysis,
        gamut,
        0.0,
        look,
        look_strength,
        display_filter,
        filter_strength,
        scene_transform,
        scene_transform_strength,
        punch_scale,
        tone_core,
        lum_norm,
        agx_primaries,
        adjustments,
        endpoint_mode=endpoint_mode,
        film_curve=film_curve,
        film_mode=film_mode,
        film_crossover=film_crossover,
        film_exposure_ev=film_exposure_ev,
        film_print_timing=film_print_timing,
        film_print_medium=film_print_medium,
        film_print_exposure_ev=film_print_exposure_ev,
        film_development=film_development,
        film_interimage=film_interimage,
        film_appearance=film_appearance,
        film_appearance_strength=film_appearance_strength,
        film_appearance_variant=film_appearance_variant,
        film_richness=film_richness,
        film_color_density=film_color_density,
        film_neutral_bias=film_neutral_bias,
        film_dev_contrast=film_dev_contrast,
        film_dev_fog=film_dev_fog,
        film_dev_density=film_dev_density,
        film_compression=film_compression,
        film_compression_knee=film_compression_knee,
        film_highlight_density=film_highlight_density,
        film_grain=film_grain,
        film_halation=film_halation,
        film_bloom=film_bloom,
        film_optics_seed=film_optics_seed,
        film_media_scatter=film_media_scatter,
        film_interimage_beta_dial=film_interimage_beta_dial,
        color_head_y=color_head_y,
        color_head_m=color_head_m,
        chroma_nr=chroma_nr,
        lens_filter=lens_filter,
    )
    return result.ev, result


def auto_ev_overlay_lines(result: AutoEvResult) -> list[str]:
    lines = [f"全图亮度参考 {result.ev_boost:+.2f} EV"]
    if result.ev_median_target < -1e-6 and result.ev_boost < 1e-6:
        lines.append("可靠主体已高于锚定 · 保持 EV 0")
    elif result.highlight_limited:
        lines.append(
            f"参考目标 {result.ev_median_target:+.2f} · 高光限制至 {result.ev:+.2f}"
        )
    else:
        lines.append(f"可靠主体参考 18% ({result.anchored_median_ev:+.2f} EV)")
    return lines
