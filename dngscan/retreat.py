# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-demosaic CFA headroom-driven chroma retreat for scene tone cores."""
from __future__ import annotations

from typing import Any

from ._deps import np
from .color import RGB_TO_XYZ, luminance_from_rec2020

REC2020_Y_ROW_SUM = float(RGB_TO_XYZ["Rec2020"][1].sum())


def resize_clip_masks(
    clip_masks: Any,
    shape: tuple[int, int],
    crop: tuple[float, float, float, float] | None = None,
) -> Any:
    """Bilinearly resize half-resolution clip masks to a render buffer shape.

    When ``crop`` is set to evidence-pixel bounds ``(y0, x0, y1, x1)``, the source is
    first cropped (fractional via PIL) and then resized. A full-frame crop is equivalent
    to a pure scale from a shared top-left origin — the Core Image ↔ LibRaw mapping.
    """
    if clip_masks is None:
        return None
    mask = np.asarray(clip_masks, dtype=np.float32)
    if crop is not None:
        y0, x0, y1, x1 = (float(v) for v in crop)
        src_h = max(y1 - y0, 1e-6)
        src_w = max(x1 - x0, 1e-6)
        # Map the crop rectangle onto a temporary grid whose aspect matches the crop,
        # then let the final resize land on ``shape``. Using an intermediate at the
        # crop's integer size keeps the path close to a pure scale when crop is full-frame.
        interim_h = max(1, int(round(src_h)))
        interim_w = max(1, int(round(src_w)))
        from PIL import Image

        cropped = np.empty((interim_h, interim_w, mask.shape[2]), dtype=np.float32)
        for idx in range(mask.shape[2]):
            im = Image.fromarray(mask[:, :, idx].astype(np.float32, copy=False), mode="F")
            # PIL crop box is (left, upper, right, lower) in pixel coords.
            box = (x0, y0, x1, y1)
            cropped[:, :, idx] = np.asarray(
                im.resize((interim_w, interim_h), Image.Resampling.BILINEAR, box=box),
                dtype=np.float32,
            )
        mask = cropped
    if mask.shape[:2] == shape:
        return np.clip(mask, 0.0, 1.0)
    from PIL import Image

    h, w = shape
    channels = []
    for idx in range(mask.shape[2]):
        im = Image.fromarray(mask[:, :, idx].astype(np.float32, copy=False), mode="F")
        im = im.resize((w, h), Image.Resampling.BILINEAR)
        channels.append(np.asarray(im, dtype=np.float32))
    return np.clip(np.stack(channels, axis=2), 0.0, 1.0)


def clip_masks_for_shape(bundle: Any, shape: tuple[int, int]) -> Any:
    """Resize bundle clip masks once per render shape (cached on the bundle)."""
    masks = getattr(bundle, "clip_masks", None)
    if masks is None:
        return None
    cache_shape = getattr(bundle, "_clip_masks_cache_shape", None)
    cached = getattr(bundle, "_clip_masks_resized", None)
    if cache_shape == shape and cached is not None:
        return cached
    crop = getattr(bundle, "scene_geometry_crop", None)
    # Only apply the evidence→scene crop when the request is the scene buffer itself (or
    # a further resize of an already scene-sized mask). Masks stay stored at evidence
    # (LibRaw) resolution; cropping is part of the mapping into scene space. This path
    # never runs for the Core Image decoder, which carries no masks at all — it exists
    # for scene buffers that are a pure scale/crop of the evidence frame.
    evidence_shape = getattr(bundle, "evidence_shape", None)
    use_crop = None
    if crop is not None and evidence_shape is not None:
        eh, ew = (int(evidence_shape[0]), int(evidence_shape[1]))
        if masks.shape[:2] == (eh, ew) or (
            abs(masks.shape[0] - eh) <= 1 and abs(masks.shape[1] - ew) <= 1
        ):
            use_crop = crop
    resized = resize_clip_masks(masks, shape, crop=use_crop)
    bundle._clip_masks_cache_shape = shape
    bundle._clip_masks_resized = resized
    return resized


def retreat_strength_from_masks(masks_rgb: Any) -> Any:
    """Continuous R/G/B clip classing: G-only < single R/B < multi-channel clip."""
    masks = np.clip(np.asarray(masks_rgb, dtype=np.float32), 0.0, 1.0)
    mr = masks[:, 0]
    mg = masks[:, 1]
    mb = masks[:, 2]
    strength = np.float32(1.0) - (
        (np.float32(1.0) - np.float32(0.35) * mg)
        * (np.float32(1.0) - np.float32(0.50) * mr)
        * (np.float32(1.0) - np.float32(0.50) * mb)
    )
    return np.clip(strength, 0.0, 1.0)


def apply_clip_retreat_rec2020(rgb_rec2020: Any, masks_rgb: Any, strength: float = 1.0) -> Any:
    """Move near/full-well chroma toward the Rec.2020 neutral axis at fixed luminance."""
    if masks_rgb is None or strength <= 0.0:
        return rgb_rec2020
    rgb = np.asarray(rgb_rec2020, dtype=np.float32)
    s = retreat_strength_from_masks(masks_rgb) * np.float32(max(0.0, float(strength)))
    if not np.any(s > 0.0):
        return rgb
    y = luminance_from_rec2020(rgb).astype(np.float32, copy=False)
    neutral = (y / np.float32(max(REC2020_Y_ROW_SUM, 1e-9)))[:, None]
    return (rgb + s[:, None] * (neutral - rgb)).astype(np.float32, copy=False)
