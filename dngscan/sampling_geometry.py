# SPDX-License-Identifier: GPL-3.0-or-later
"""Native sensor-pixel sampling of the retained, decoded scene window.

This describes sampling geometry, independently of whether a noise-transfer
model is available. It is not a local Jacobian for lens distortion.
"""
from __future__ import annotations

import math


def _shape(value):
    try:
        h, w = (float(x) for x in value)
    except (TypeError, ValueError, OverflowError):
        return None
    return (h, w) if all(math.isfinite(x) and x > 0 for x in (h, w)) else None


def sensor_window_from_recipe(recipe, scene_shape, orientation_flip):
    """Un-oriented native window after rounded crop and any final box reduction."""
    geometry = getattr(recipe, "noise_geometry", {}) or {}
    crop = geometry.get("effective_sensor_crop")
    decoded = geometry.get("decoded_crop")
    if crop is None or decoded is None or len(crop) != 4 or len(decoded) != 4:
        return None
    native, grid, output = _shape(crop[2:]), _shape(decoded[2:]), _shape(scene_shape)
    reduction = geometry.get("post_decode_reduction", 1)
    if native is None or grid is None or output is None or reduction not in (1, 2):
        return None
    if int(orientation_flip) & 4:
        output = output[::-1]
    return tuple(n * o * reduction / g for n, o, g in zip(native, output, grid))


def sensor_px_per_render_axes(bundle, h, w):
    """Native sensor pixels per render pixel, in oriented (row, column) order.

    New decodes keep their retained sensor window through proxy caching. Older
    callers can still supply raw/crop geometry or the legacy proxy_scale ruler.
    """
    target = _shape((h, w))
    if target is None:
        raise ValueError("render dimensions must be positive and finite")
    native = _shape(getattr(bundle, "scene_sensor_window_shape", None))
    if native is None:
        descriptor = getattr(bundle, "noise_decode", None) or {}
        native = _shape(descriptor.get("sensor_window_shape"))
    if native is None:
        crop = getattr(bundle, "scene_crop_sensor", None)
        if crop is not None and len(crop) == 4:
            native = _shape(crop[2:])
    if native is None:
        raw = getattr(bundle, "raw_image", None)
        if raw is not None:
            native = _shape(raw.shape[:2])
    if native is not None:
        if int(getattr(bundle, "orientation_flip", 0)) & 4:
            native = native[::-1]
        return tuple(n / t for n, t in zip(native, target))
    try:
        scale = float(getattr(bundle, "proxy_scale", 1.0) or 1.0)
    except (TypeError, ValueError, OverflowError):
        scale = 1.0
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    scene = getattr(bundle, "scene_rec2020_render", None)
    source = _shape(scene.shape[:2]) if scene is not None else target
    return tuple(scale * s / t for s, t in zip(source, target))
