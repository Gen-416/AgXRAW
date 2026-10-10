# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene dependency qualification, separate from visual clipping evidence."""
from ._deps import np


def _axis_any(values, output_size, axis, filter_radius=0.):
    source = np.moveaxis(values, axis, 1)
    rows, size = source.shape
    if size == output_size:
        return values
    if filter_radius:
        scale = size / output_size
        centres = (np.arange(output_size, dtype=np.float64) + .5) * scale
        support = filter_radius * max(scale, 1.)
        # Pillow widens Lanczos when downsampling. Include its full support,
        # including zero-weight endpoints, rather than just the area cell.
        lo = np.maximum(np.floor(centres - support + .5).astype(np.int64), 0)
        hi = np.minimum(np.ceil(centres + support + .5).astype(np.int64), size)
    else:
        lo = np.arange(output_size, dtype=np.int64) * size // output_size
        hi = (np.arange(1, output_size + 1, dtype=np.int64) * size + output_size - 1) // output_size
    result = np.empty((rows, output_size), dtype=np.uint8)
    # Bounded row prefixes avoid a full-resolution integral-image allocation.
    for start in range(0, rows, 128):
        band = source[start:start + 128]
        prefix = np.empty((band.shape[0], size + 1), dtype=np.uint32)
        prefix[:, 0] = 0
        np.cumsum(band != 0, axis=1, dtype=np.uint32, out=prefix[:, 1:])
        result[start:start + 128] = prefix[:, hi] != prefix[:, lo]
    return np.moveaxis(result, 1, axis)


def resize_exclusion(values, shape, *, filter_radius=0.):
    """Retain ANY excluded source cell overlapping each destination cell.

    Coarse noise cells use area footprints. Scene proxies additionally pass
    their reconstruction filter radius (3 for Pillow Lanczos). Nearest or
    bilinear sampling can erase a small dependency and cannot qualify them.
    This array is already in final scene coordinates: no RAW crop is reapplied.
    """
    if values is None:
        return None
    arr = np.asarray(values)
    shape = tuple(int(n) for n in shape)
    if arr.ndim != 2 or len(shape) != 2 or min(*arr.shape, *shape) < 1:
        raise ValueError("dependency exclusion requires positive two-dimensional geometry")
    if not np.isfinite(filter_radius) or filter_radius < 0:
        raise ValueError("dependency resize filter radius must be nonnegative")
    if arr.shape == shape:
        return arr
    return np.ascontiguousarray(_axis_any(_axis_any(arr, shape[1], 1, filter_radius), shape[0], 0, filter_radius))


def scene_exclusion_for_shape(bundle, shape):
    return resize_exclusion(getattr(bundle, "scene_reliability_exclusion", None), shape)


def separation_masks(masks, exclusion):
    """HDR colour gating only; callers retain original masks for retreat."""
    if exclusion is None or not np.any(exclusion):
        return masks
    excluded = np.asarray(exclusion).reshape(-1, 1) != 0
    if masks is None:
        return np.broadcast_to(excluded, (excluded.size, 3))
    return np.maximum(masks, excluded)
