# SPDX-License-Identifier: GPL-3.0-or-later
"""RAW-derived permission maps for gated display rendering.

RAW evidence controls how much the DRT may alter hue/chroma; scene RGB controls
appearance; output gamut controls final constraint. These weights are not style masks.
"""
from __future__ import annotations

import math
from typing import Any

from ._deps import np
from .color import OKLAB_M1, OKLAB_M2, apply_rgb_matrix3, rec2020_to_xyz, smoothstep
from .constants import EPS
from .models import Analysis, RawBundle, RawGuidanceMaps

# Discrete CFA clip classes (bit flags): R=1, G=2, B=4.
CLIP_CLASS_NONE = 0
CLIP_CLASS_R = 1
CLIP_CLASS_G = 2
CLIP_CLASS_B = 4
CLIP_CLASS_RG = 3
CLIP_CLASS_RB = 5
CLIP_CLASS_GB = 6
CLIP_CLASS_RGB = 7


def clip_class_from_masks(masks_rgb: Any, threshold: float = 0.35) -> Any:
    """Per-pixel discrete clip class from soft R/G/B CFA masks."""
    m = np.clip(np.asarray(masks_rgb, dtype=np.float32), 0.0, 1.0)
    flags = (
        (m[:, 0] > np.float32(threshold)).astype(np.int32)
        | ((m[:, 1] > np.float32(threshold)).astype(np.int32) << 1)
        | ((m[:, 2] > np.float32(threshold)).astype(np.int32) << 2)
    )
    return flags


def headroom_from_masks(masks_rgb: Any) -> Any:
    """Legacy clip-mask proxy retained for callers without RAW headroom maps."""
    m = np.clip(np.asarray(masks_rgb, dtype=np.float32), 0.0, 1.0)
    return np.clip(np.float32(1.0) - m, 0.0, 1.0)


def saturation_proximity_from_headroom(headroom_rgb: Any) -> Any:
    """Map measured remaining headroom to the same 95–99% soft clip interval."""
    headroom = np.clip(np.asarray(headroom_rgb, dtype=np.float32), 0.0, 1.0)
    return np.float32(1.0) - smoothstep(np.float32(0.01), np.float32(0.05), headroom)


def raw_color_permission(
    masks_rgb: Any | None = None,
    *,
    headroom_rgb: Any | None = None,
    clip_class: Any | None = None,
) -> Any:
    """RAW loss permission for path-to-white (0 = measured colour remains trusted)."""
    if headroom_rgb is not None:
        m = saturation_proximity_from_headroom(headroom_rgb)
    elif masks_rgb is not None:
        m = np.clip(np.asarray(masks_rgb, dtype=np.float32), 0.0, 1.0)
    else:
        raise ValueError("raw_color_permission requires masks_rgb or headroom_rgb")
    mr, mg, mb = m[:, 0], m[:, 1], m[:, 2]
    # Continuous combination (G-only weakest, multi-channel strongest).
    strength = np.float32(1.0) - (
        (np.float32(1.0) - np.float32(0.40) * mg)
        * (np.float32(1.0) - np.float32(0.55) * mr)
        * (np.float32(1.0) - np.float32(0.55) * mb)
    )
    classes = np.asarray(clip_class, dtype=np.int32) if clip_class is not None else clip_class_from_masks(m)
    # Bit flags are not ordinal: B-only is 4, while RG is 3. Count set bits instead of
    # comparing numeric class values so every single-channel clip is treated consistently.
    multi = ((classes & (classes - 1)) != 0).astype(np.float32)
    return np.clip(strength + np.float32(0.12) * multi, 0.0, 1.0)


def scene_highlight_permission(
    scene_ev: Any,
    ev_lo: float = 0.25,
    ev_hi: float = 2.75,
) -> Any:
    """Open color-path only in scene-luminance shoulder (independent of RAW clip)."""
    return smoothstep(np.float32(ev_lo), np.float32(ev_hi), np.asarray(scene_ev, dtype=np.float32))


def midtone_protection(
    scene_ev: Any,
    raw_permission: Any,
    strength: float = 0.92,
) -> Any:
    """Suppress color-path in trustworthy midtones (portraits, costumes, props)."""
    ev = np.asarray(scene_ev, dtype=np.float32)
    raw = np.clip(np.asarray(raw_permission, dtype=np.float32), 0.0, 1.0)
    body = np.float32(1.0) - smoothstep(np.float32(-2.5), np.float32(0.35), ev)
    return np.clip(np.float32(strength) * body * (np.float32(1.0) - raw), 0.0, 1.0)


def color_path_weight(
    masks_rgb: Any | None,
    scene_ev: Any,
    gamut_pressure_pct: float = 0.0,
    *,
    scene_rgb_rec2020: Any | None = None,
    noise_ev_floor: float | None = None,
    raw_headroom_rgb: Any | None = None,
    raw_clip_class: Any | None = None,
    raw_snr_confidence: Any | None = None,
    raw_permission: Any | None = None,
    midtone_protect: float = 0.92,
    highlight_ev_lo: float = 0.25,
    highlight_ev_hi: float = 2.75,
    gamut_pressure_scale: float = 0.30,
) -> Any:
    """Blend weight for AgX color geometry over luma-first DRT (per pixel, 0–1)."""
    ev = np.asarray(scene_ev, dtype=np.float32)
    if raw_permission is not None and raw_permission.shape[0] == ev.shape[0]:
        raw_perm = np.asarray(raw_permission, dtype=np.float32)
    elif raw_headroom_rgb is not None and raw_headroom_rgb.shape[0] == ev.shape[0]:
        raw_perm = raw_color_permission(headroom_rgb=raw_headroom_rgb, clip_class=raw_clip_class)
    elif masks_rgb is not None and masks_rgb.shape[0] == ev.shape[0]:
        raw_perm = raw_color_permission(masks_rgb)
    else:
        raw_perm = np.zeros_like(ev, dtype=np.float32)
    scene_perm = scene_highlight_permission(ev, highlight_ev_lo, highlight_ev_hi)
    gamut_perm = np.float32(gamut_pressure_scale) * np.float32(
        min(1.0, max(0.0, float(gamut_pressure_pct) / 8.0))
    )
    nonraw_perm = np.clip(scene_perm * np.float32(0.45) + gamut_perm, 0.0, 1.0)
    if raw_snr_confidence is not None and raw_snr_confidence.shape[0] == ev.shape[0]:
        nonraw_perm *= np.clip(np.asarray(raw_snr_confidence, dtype=np.float32), 0.0, 1.0)
    elif noise_ev_floor is not None:
        nonraw_perm *= snr_confidence_from_ev(ev, float(noise_ev_floor))
    if scene_rgb_rec2020 is not None and scene_rgb_rec2020.shape[0] == ev.shape[0]:
        # Hue policy is an aesthetic display choice. It may attenuate scene/gamut-driven
        # geometry, but must not override a measured loss of RAW channel information.
        nonraw_perm *= sector_hue_multiplier(scene_rgb_rec2020, ev)
    w = raw_perm + (np.float32(1.0) - raw_perm) * nonraw_perm
    protect = midtone_protection(ev, raw_perm, midtone_protect)
    w = np.clip(w * (np.float32(1.0) - protect), 0.0, 1.0)
    return w.astype(np.float32, copy=False)


def snr_confidence_from_ev(scene_ev: Any, noise_ev_floor: float) -> Any:
    """Trust scene color more as luminance rises above the noise floor."""
    ev = np.asarray(scene_ev, dtype=np.float32)
    lo = np.float32(noise_ev_floor + 1.5)
    hi = np.float32(noise_ev_floor + 5.5)
    return smoothstep(lo, hi, ev).astype(np.float32, copy=False)


def _rec2020_oklab_hue_deg(rgb: Any) -> Any:
    xyz = rec2020_to_xyz(rgb)
    lms = apply_rgb_matrix3(xyz, OKLAB_M1)
    lab = apply_rgb_matrix3(np.cbrt(np.maximum(lms, np.float32(EPS))), OKLAB_M2)
    return (np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % np.float32(360.0)).astype(
        np.float32, copy=False
    )


def sector_hue_multiplier(scene_rgb_rec2020: Any, scene_ev: Any) -> Any:
    """Skin midtone protect + green/cyan highlight openness (hue-path policy)."""
    rgb = np.asarray(scene_rgb_rec2020, dtype=np.float32).reshape(-1, 3)
    ev = np.asarray(scene_ev, dtype=np.float32).reshape(-1)
    hue = _rec2020_oklab_hue_deg(rgb)
    skin = smoothstep(np.float32(22.0), np.float32(38.0), hue) * (
        np.float32(1.0) - smoothstep(np.float32(68.0), np.float32(82.0), hue)
    )
    skin *= smoothstep(np.float32(-1.25), np.float32(0.35), ev) * (
        np.float32(1.0) - smoothstep(np.float32(1.35), np.float32(2.35), ev)
    )
    green = smoothstep(np.float32(128.0), np.float32(148.0), hue) * (
        np.float32(1.0) - smoothstep(np.float32(188.0), np.float32(208.0), hue)
    )
    green *= smoothstep(np.float32(0.35), np.float32(1.85), ev)
    return np.clip(
        (np.float32(1.0) - np.float32(0.32) * skin) * (np.float32(1.0) + np.float32(0.14) * green),
        np.float32(0.55),
        np.float32(1.18),
    ).astype(np.float32, copy=False)


def _bin_period_min(arr: Any, ph: int, pw: int) -> Any:
    h, w = arr.shape[:2]
    h2 = max(1, h // ph)
    w2 = max(1, w // pw)
    cropped = arr[: h2 * ph, : w2 * pw]
    return cropped.reshape(h2, ph, w2, pw, arr.shape[2]).min(axis=(1, 3))


def _cfa_bin_period(bundle: RawBundle) -> tuple[int, int]:
    """The bin that guarantees every output cell saw every channel.

    R6 item 4: a Bayer 2x2 contains all three channels, but an X-Trans 2x2
    block can lack red entirely — its neutral 1.0 fill then read as FULL
    headroom ("nothing near clip here") instead of unknown, so gated
    under-weighted local clipping exactly where evidence was thinnest. The
    honest bin is one CFA period: coarser guidance resolution on 6x6
    sensors, but every cell's min is a real measurement of every channel.
    """
    try:
        pattern = np.asarray(bundle.raw_pattern)
        ph, pw = int(pattern.shape[0]), int(pattern.shape[1])
        if ph >= 1 and pw >= 1:
            return ph, pw
    except Exception:
        pass
    return 2, 2


def _align_cfa_rgb_map(bundle: RawBundle, values: Any, target_shape: tuple[int, int]) -> Any:
    """Reduce raw-CFA RGB evidence to scene geometry without changing its values."""
    from . import raw_io

    ph, pw = _cfa_bin_period(bundle)
    binned = _bin_period_min(np.asarray(values, dtype=np.float32), ph, pw)
    oriented = raw_io._orient_like_libraw(binned, bundle.orientation_flip)
    return raw_io._resize_mask_to_shape(oriented, target_shape).astype(np.float32, copy=False)


def _raw_headroom_rgb(
    bundle: RawBundle,
    target_shape: tuple[int, int],
    analysis: Analysis | None = None,
) -> Any:
    """Actual pre-WB remaining well capacity for R/G/B CFA samples.

    R3 item 4: the fullwell reference is the ANALYSIS-resolved per-channel
    value (metadata white refined by the observed saturation pile) whenever
    an Analysis rides along — the clip masks were rebuilt on exactly that
    endpoint, and reading raw metadata here let the same pixel be "near
    clip" to the mask and "has headroom" to this map at once. Without an
    Analysis the metadata fallback stands, stated rather than implied.
    """
    from . import raw_io

    resolved = dict(getattr(analysis, "channel_fullwell", None) or {})
    raw = np.asarray(bundle.raw_image, dtype=np.float32)
    colors = np.asarray(bundle.raw_colors)
    headroom = np.ones(raw.shape + (3,), dtype=np.float32)
    for cid in np.unique(colors):
        cid_i = int(cid)
        label = raw_io.channel_label(bundle.color_desc, cid_i)
        out_idx = 0 if label.startswith("R") else 1 if label.startswith("G") else 2 if label.startswith("B") else None
        if out_idx is None:
            continue
        black = raw_io.channel_black_level(bundle.black_levels, cid_i)
        fullwell = float(
            resolved.get(cid_i)
            or raw_io.channel_fullwell(
                bundle.white_level, bundle.camera_white_levels, cid_i
            )
        )
        channel_headroom = np.clip((np.float32(fullwell) - raw) / np.float32(max(fullwell - black, 1.0)), 0.0, 1.0)
        plane = np.where(colors == cid_i, channel_headroom, np.float32(1.0))
        headroom[:, :, out_idx] = np.minimum(headroom[:, :, out_idx], plane)
    return _align_cfa_rgb_map(bundle, headroom, target_shape)


def _has_sensor_snr_prior(analysis: Analysis | None) -> bool:
    gain = getattr(analysis, "gain_e_per_dn", None)
    read_noise = getattr(analysis, "prior_read_noise_e", None)
    return bool(
        gain is not None and read_noise is not None
        and math.isfinite(gain) and math.isfinite(read_noise)
        and gain > 0.0 and read_noise >= 0.0
    )


def _raw_snr_confidence(bundle: RawBundle, analysis: Analysis | None, target_shape: tuple[int, int]) -> Any | None:
    """Conservative per-cell colour SNR confidence from RAW DN plus sensor priors."""
    if not _has_sensor_snr_prior(analysis):
        # Keep the established scene-EV fallback for cameras without calibrated electron
        # priors instead of pretending a unity confidence is a measured RAW SNR.
        return None
    gain = float(analysis.gain_e_per_dn)
    read_noise = float(analysis.prior_read_noise_e)

    from . import raw_io

    raw = np.asarray(bundle.raw_image, dtype=np.float32)
    colors = np.asarray(bundle.raw_colors)
    confidence = np.ones(raw.shape + (3,), dtype=np.float32)
    rn2 = np.float32(read_noise * read_noise)
    for cid in np.unique(colors):
        cid_i = int(cid)
        label = raw_io.channel_label(bundle.color_desc, cid_i)
        out_idx = 0 if label.startswith("R") else 1 if label.startswith("G") else 2 if label.startswith("B") else None
        if out_idx is None:
            continue
        black = raw_io.channel_black_level(bundle.black_levels, cid_i)
        electrons = np.maximum(raw - np.float32(black), 0.0) * np.float32(gain)
        snr = electrons / np.sqrt(np.maximum(electrons + rn2, np.float32(EPS)))
        plane = np.where(colors == cid_i, smoothstep(np.float32(1.0), np.float32(10.0), snr), np.float32(1.0))
        confidence[:, :, out_idx] = np.minimum(confidence[:, :, out_idx], plane)
    rgb_confidence = _align_cfa_rgb_map(bundle, confidence, target_shape)
    return np.min(rgb_confidence, axis=2).astype(np.float32, copy=False)


def build_raw_guidance_maps(
    bundle: RawBundle,
    analysis: Analysis | None = None,
) -> RawGuidanceMaps | None:
    """Build RAW-domain headroom, clip class and SNR maps aligned to clip masks."""
    masks = getattr(bundle, "clip_masks", None)
    if masks is None:
        return None
    if getattr(bundle, "raw_image", None) is None:
        # R4 F2: a cache-proxy bundle has no mosaic to measure; whatever maps
        # it carries were built from the full mosaic at entry-build time.
        return getattr(bundle, "raw_guidance", None)
    target_shape = np.asarray(masks).shape[:2]
    headroom = _raw_headroom_rgb(bundle, target_shape, analysis)
    clip_class = clip_class_from_masks(saturation_proximity_from_headroom(headroom).reshape(-1, 3)).reshape(target_shape)
    snr = _raw_snr_confidence(bundle, analysis, target_shape)
    stored_headroom = headroom.astype(np.float16, copy=False)
    stored_class = clip_class.astype(np.uint8, copy=False)
    permission = raw_color_permission(
        headroom_rgb=stored_headroom.reshape(-1, 3),
        clip_class=stored_class.reshape(-1),
    ).reshape(target_shape)
    return RawGuidanceMaps(
        headroom=stored_headroom,
        clip_class=stored_class,
        snr_confidence=snr.astype(np.float16, copy=False) if snr is not None else None,
        raw_permission=permission,
    )


def ensure_raw_guidance(bundle: RawBundle, analysis: Analysis | None = None) -> RawGuidanceMaps | None:
    wants_sensor_snr = _has_sensor_snr_prior(analysis)
    # R3 item 4: maps built before the analysis existed measured headroom
    # against metadata white; once the resolved per-channel fullwell is
    # available they must be rebuilt on it, same upgrade rule as the
    # sensor-SNR prior. R4 F2: an upgrade demands the MOSAIC — cache-proxy
    # bundles carry raw_image=None precisely because their guidance was
    # already built (with the analysis) at entry-build time, so for them the
    # cached maps ARE the upgraded maps and a forced rebuild would crash on
    # the missing mosaic.
    can_rebuild = getattr(bundle, "raw_image", None) is not None
    wants_resolved_fullwell = (
        bool(getattr(analysis, "channel_fullwell", None)) and can_rebuild
    )
    if getattr(bundle, "raw_guidance", None) is not None and (
        not wants_sensor_snr
        or getattr(bundle, "_raw_guidance_has_sensor_snr", False)
        or not can_rebuild
    ) and (
        not wants_resolved_fullwell
        or getattr(bundle, "_raw_guidance_has_resolved_fullwell", False)
    ):
        return bundle.raw_guidance
    maps = build_raw_guidance_maps(bundle, analysis)
    bundle.raw_guidance = maps
    bundle._raw_guidance_has_sensor_snr = wants_sensor_snr
    bundle._raw_guidance_has_resolved_fullwell = wants_resolved_fullwell
    bundle._raw_guidance_cache_shape = None
    bundle._raw_guidance_resized = None
    return maps


def _resize_nearest_scalar(
    values: Any,
    shape: tuple[int, int],
    crop: tuple[float, float, float, float] | None = None,
) -> Any:
    from PIL import Image

    arr = np.asarray(values, dtype=np.uint8)
    if crop is not None:
        y0, x0, y1, x1 = (float(v) for v in crop)
        interim_h = max(1, int(round(y1 - y0)))
        interim_w = max(1, int(round(x1 - x0)))
        arr = np.asarray(
            Image.fromarray(arr, mode="L").resize(
                (interim_w, interim_h), Image.Resampling.NEAREST, box=(x0, y0, x1, y1)
            ),
            dtype=np.uint8,
        )
    if arr.shape[:2] == shape:
        return arr
    return np.asarray(
        Image.fromarray(arr, mode="L").resize((shape[1], shape[0]), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )


def raw_guidance_for_shape(
    bundle: RawBundle, shape: tuple[int, int], analysis: Analysis | None = None
) -> RawGuidanceMaps | None:
    """Resize immutable RAW evidence once per output geometry."""
    maps = ensure_raw_guidance(bundle, analysis)
    if maps is None:
        return None
    if maps.headroom.shape[:2] == shape:
        return maps
    if getattr(bundle, "_raw_guidance_cache_shape", None) == shape:
        cached = getattr(bundle, "_raw_guidance_resized", None)
        if cached is not None:
            return cached
    from .retreat import resize_clip_masks

    crop = getattr(bundle, "scene_geometry_crop", None)
    resized_headroom = resize_clip_masks(maps.headroom, shape, crop=crop).astype(
        np.float16, copy=False
    )
    resized_class = _resize_nearest_scalar(maps.clip_class, shape, crop=crop)
    resized_permission = raw_color_permission(
        headroom_rgb=resized_headroom.reshape(-1, 3),
        clip_class=resized_class.reshape(-1),
    ).reshape(shape)
    resized = RawGuidanceMaps(
        headroom=resized_headroom,
        clip_class=resized_class,
        snr_confidence=(
            resize_clip_masks(maps.snr_confidence[:, :, None], shape, crop=crop)[:, :, 0].astype(
                np.float16, copy=False
            )
            if maps.snr_confidence is not None
            else None
        ),
        raw_permission=resized_permission,
    )
    bundle._raw_guidance_cache_shape = shape
    bundle._raw_guidance_resized = resized
    return resized


def flatten_raw_guidance(
    maps: RawGuidanceMaps | None, start: int, end: int, step: int = 1
) -> RawGuidanceMaps | None:
    if maps is None:
        return None
    return RawGuidanceMaps(
        headroom=np.asarray(maps.headroom).reshape(-1, 3)[start:end:step],
        clip_class=np.asarray(maps.clip_class).reshape(-1)[start:end:step],
        snr_confidence=(
            np.asarray(maps.snr_confidence).reshape(-1)[start:end:step]
            if maps.snr_confidence is not None else None
        ),
        raw_permission=(
            np.asarray(maps.raw_permission).reshape(-1)[start:end:step]
            if maps.raw_permission is not None
            else None
        ),
    )


def scene_ev_from_rec2020(rgb_rec2020: Any) -> Any:
    """Scene EV relative to 18% gray from Rec.2020 linear RGB."""
    from .color import luminance_from_rec2020

    y = np.maximum(luminance_from_rec2020(rgb_rec2020), np.float32(EPS))
    return np.log2(y / np.float32(0.18)).astype(np.float32, copy=False)
