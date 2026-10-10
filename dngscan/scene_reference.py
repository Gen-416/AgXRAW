# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent, spatially filtered evidence for decoders with opaque geometry."""
from __future__ import annotations

import numpy as np


def reliable_reference_samples(evidence, scene, scale, processing_loss, recipe):
    """Filter in the reference decoder's own coordinates, never Apple's pixels.

    Returns normalized RGB and retained sample percentage. Empty is a measured
    lack of reliable evidence; callers must distinguish it from unavailable (None).
    """
    from .decoder_loss import support_is_untrusted

    if (getattr(recipe, "loss_support_untrusted", False)
            or support_is_untrusted(getattr(recipe, "loss_support", None))
            or support_is_untrusted(getattr(recipe, "source_loss_support", None))):
        # An unknown decoder influence range has no spatial mask. Its absence
        # does not certify the reference: retain the explicit empty result.
        return np.empty((0, 3), dtype=np.float32), 0.0

    from .sensor_summary import summarize_sensor
    from .dng_opcodes import Warp
    from .raw_io import build_clip_masks, _merge_processing_loss
    from .sampling import sample_indices

    summary = summarize_sensor(evidence.raw_image, evidence.raw_colors, evidence.white_level,
                               evidence.camera_white_levels, evidence=evidence)
    ids, fullwell = list(summary.channel_ids), dict(summary.channel_fullwell)
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
    source_exclusion = getattr(recipe, "scene_reliability_exclusion", None)
    if source_exclusion is not None:
        if np.shape(source_exclusion) != scene.shape[:2]:
            raise ValueError("reference source exclusion is not aligned to scene")
        reliable &= ~np.asarray(source_exclusion, dtype=bool).reshape(-1)[rows]
    reliable &= np.isfinite(rgb).all(axis=1)
    pct = float(np.mean(reliable) * 100.0)
    if np.count_nonzero(reliable) < 256 or pct < 5.0:
        return np.empty((0, 3), dtype=np.float32), pct
    return np.ascontiguousarray(rgb[reliable]), pct
