# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent, spatially filtered evidence for decoders with opaque geometry."""
from __future__ import annotations

import numpy as np


def reliable_reference_samples(evidence, scene, scale, processing_loss, recipe):
    """Filter in the reference decoder's own coordinates, never Apple's pixels.

    Returns normalized RGB and retained sample percentage. Empty is a measured
    lack of reliable evidence; callers must distinguish it from unavailable (None).
    """
    from .analysis import channel_saturation_levels, detect_ceilings, resolve_fullwell
    from .dng_opcodes import Warp
    from .raw_io import build_clip_masks, _merge_processing_loss
    from .sampling import sample_indices

    ids = [int(c) for c in np.unique(evidence.raw_colors)]
    sat = channel_saturation_levels(ids, evidence.camera_white_levels, evidence.white_level)
    ceilings, _, _, spike = detect_ceilings(evidence.raw_image, evidence.raw_colors, ids, sat)
    _, _, _, fullwell = resolve_fullwell(ids, ceilings, spike, sat)
    levels = [fullwell.get(c, evidence.white_level) for c in range(max(ids) + 1)]
    masks = build_clip_masks(
        evidence.raw_image, evidence.raw_colors, evidence.color_desc,
        evidence.white_level, evidence.black_levels, levels,
        evidence.orientation_flip, scene.shape[:2], evidence.raw_pattern,
        tuple(op for op in recipe.post if isinstance(op, Warp)), recipe.crop,
        evidence.spatial_black,
    )
    _merge_processing_loss(masks, processing_loss)
    flat = np.asarray(scene).reshape(-1, 3)
    rows = sample_indices(len(flat), max_samples=200_000)
    rgb = flat[rows].astype(np.float32) / np.float32(scale)
    if masks is None:
        raise ValueError("reference has no spatial reliability mask")
    reliable = np.max(masks.reshape(-1, 3)[rows], axis=1) < 0.1
    reliable &= np.isfinite(rgb).all(axis=1)
    pct = float(np.mean(reliable) * 100.0)
    if np.count_nonzero(reliable) < 256 or pct < 5.0:
        return np.empty((0, 3), dtype=np.float32), pct
    return np.ascontiguousarray(rgb[reliable]), pct
