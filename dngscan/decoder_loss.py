# SPDX-License-Identifier: GPL-3.0-or-later
"""Processing-loss evidence at LibRaw's scaling/demosaic boundary.

This module never changes samples.  The scale contract follows the pinned
LibRaw e419de08: adjust_bl/raw2image subtract the per-plane black, then
scale_colors divides by (maximum - common black), normalizes fixed WB by its
minimum in clip mode (maximum in other highlight modes), and clips to uint16.
DefaultScale is applied later by stretch, after interpolation.
"""
from __future__ import annotations

from typing import Any

from ._deps import np


def record_wb_ceiling_loss(raw: Any, black_levels: Any, white_level: float,
                           camera_wb: Any, highlight: str,
                           loss: Any | None = None, *, colors: Any | None = None) -> Any | None:
    """Merge pre-demosaic WB saturation into a byte-per-sensel loss log.

    Pass the *working* black and LibRaw encoding maximum, including explicit
    spatial-black overrides, never LinearResponseLimit's effective white.
    This records actual overflow of LibRaw's CLIP(int(value)): values below
    65536 truncate to an allowed uint16 code.  The final camera-RGB == 65535
    check remains separately conservative.  An exact source white alone must
    not create a new pre-demosaic loss or alter automatic algorithm selection.
    Allocations/transients are bounded to 128 rows until the first event
    requires a persistent loss log.
    """
    image = np.asarray(raw.raw_image_visible)
    black = np.asarray(list(black_levels) or [0.], dtype=np.float32)
    wb = np.asarray(list(camera_wb)[:4], dtype=np.float32)
    if wb.size < 3 or not np.all(np.isfinite(wb[:3]) & (wb[:3] > 0)):
        # The camera-RGB production contract rejects this unsupported WB at
        # its existing boundary; do not invent a decoder preconditioner here.
        return loss
    if wb.size < 4:
        wb = np.append(wb, wb[1]).astype(np.float32)
    elif not np.isfinite(wb[3]) or wb[3] <= 0:
        wb[3] = wb[1]
    denominator = float(white_level) - float(np.min(black))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("invalid LibRaw scaling range for processing-loss evidence")
    normalizer = np.min(wb) if highlight == "clip" else np.max(wb)
    # LibRaw stores pre_mul and scale_mul as float, not double.
    scale = ((wb / np.float32(normalizer)) * np.float32(65535.)) / np.float32(denominator)
    colors = (np.asarray(raw.raw_colors_visible if colors is None else colors)
              if image.ndim == 2 else None)
    for y in range(0, image.shape[0], 128):
        band = image[y:y + 128]
        if image.ndim == 2:
            ids = colors[y:y + 128]
            shifted = band.astype(np.float32) - black[np.minimum(ids, black.size - 1)]
            boundary = shifted * scale[np.minimum(ids, scale.size - 1)] >= np.float32(65536.)
        else:
            n = min(3, image.shape[2])
            shifted = band[..., :n].astype(np.float32) - black[np.minimum(np.arange(n), black.size - 1)]
            boundary = shifted * scale[:n] >= np.float32(65536.)
        if not np.any(boundary):
            continue
        if loss is None:
            loss = np.zeros(image.shape if image.ndim == 2 else (*image.shape[:2], n), dtype=np.uint8)
        loss[y:y + 128] |= boundary.astype(np.uint8)
    return loss


def pool_mosaic_loss(loss: Any, colors: Any, color_desc: str) -> Any:
    """Exact Bayer half-size source footprint: both greens share one plane."""
    loss = np.asarray(loss)
    if loss.ndim == 3:
        return (loss[..., :3] != 0).astype(np.uint8)
    h, w = loss.shape
    out = np.zeros(((h + 1) // 2, (w + 1) // 2, 3), dtype=np.uint8)
    for r in range(2):
        for c in range(2):
            plane = loss[r::2, c::2]
            ids = colors[r::2, c::2]
            for cid in np.unique(ids):
                label = color_desc[int(cid):int(cid) + 1].upper()
                if label not in ("R", "G", "B"):
                    continue
                dest = out[:plane.shape[0], :plane.shape[1], "RGB".index(label)]
                np.maximum(dest, (plane != 0) & (ids == cid), out=dest)
    return out


def dilate_loss(mask: Any, radius: int) -> Any:
    """Square max support with two byte planes, independent of radius size."""
    source = np.asarray(mask, dtype=np.uint8)
    if radius <= 0:
        return source
    horizontal = source.copy()
    for d in range(1, min(radius, source.shape[1] - 1) + 1):
        np.maximum(horizontal[:, d:], source[:, :-d], out=horizontal[:, d:])
        np.maximum(horizontal[:, :-d], source[:, d:], out=horizontal[:, :-d])
    out = horizontal.copy()
    for d in range(1, min(radius, source.shape[0] - 1) + 1):
        np.maximum(out[d:], horizontal[:-d], out=out[d:])
        np.maximum(out[:-d], horizontal[d:], out=out[:-d])
    return out


def transport_default_scale(mask: Any, shape: tuple[int, int], pixel_aspect: float) -> Any:
    """Transport max evidence through pinned LibRaw stretch's two taps.

    Coordinates use the actual pixel aspect, rather than nearest resizing or
    target-size ratios.  The latter lose registration after dimension rounding.
    Including the second tap is conservative even on the pinned implementation
    whose integer frac currently makes its contribution zero.
    """
    mask = np.asarray(mask)
    aspect = float(pixel_aspect)
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("invalid pixel aspect for processing-loss evidence")
    # Dimension rounding can hide a nonidentity stretch. LibRaw still advances
    # its aspect-based source coordinates even when the raster size is equal.
    if mask.shape[:2] == shape and aspect == 1.:
        return mask
    if aspect < 1 and mask.shape[1] == shape[1]:
        coord = np.zeros(shape[0], dtype=np.float64)
        # stretch advances a double accumulator. Multiplication by arange
        # disagrees at some integer crossings (for example 10 * .1 versus
        # ten accumulated .1 steps), moving a source footprint by one pixel.
        np.cumsum(np.full(max(0, shape[0] - 1), aspect), out=coord[1:])
        lo = np.minimum(np.floor(coord).astype(np.int64), mask.shape[0] - 1)
        hi = np.minimum(lo + 1, mask.shape[0] - 1)
        return np.maximum(mask[lo], mask[hi])
    if aspect > 1 and mask.shape[0] == shape[0]:
        coord = np.zeros(shape[1], dtype=np.float64)
        np.cumsum(np.full(max(0, shape[1] - 1), 1. / aspect), out=coord[1:])
        lo = np.minimum(np.floor(coord).astype(np.int64), mask.shape[1] - 1)
        hi = np.minimum(lo + 1, mask.shape[1] - 1)
        return np.maximum(mask[:, lo], mask[:, hi])
    raise ValueError("unsupported LibRaw loss geometry before lens operations")


def propagate_mosaic_loss(loss: Any | None, colors: Any, color_desc: str,
                           shape: tuple[int, int], *, half_size: bool,
                           demosaic: Any, highlight: str,
                           is_bayer: bool, pixel_aspect: float = 1.) -> tuple[Any | None, str | None]:
    """Return float16 RGB permission evidence and its explicit support policy.

    Full-resolution DHT includes an in-place hot-pixel pass and frame-wide
    extrema clamps.  Its complete loss support has no audited local bound in
    this adapter.  Until decoder-side evidence is exposed, an actual
    pre-demosaic loss makes that frame's support uncertified.  Other unaudited
    interpolation/recovery paths use the same conservative policy; image
    formation and RAW saturation statistics are unaffected.

    Bayer half-size clip bypasses demosaic and has an audited 2x2 footprint.
    AHD's source support is bounded by five native pixels: green radius 2,
    R/B radius 3, Lab homogeneity radius 4, 3x3 homogeneity vote radius 5.
    Blend is pointwise, so it joins plane permissions at the same location.
    """
    if loss is None or not np.any(loss):
        return None, None
    source = np.asarray(loss)
    algorithm = str(getattr(demosaic, "name", demosaic) or "default").upper()
    if highlight == "reconstruct":
        return (np.ones((*shape, 3), dtype=np.float16),
                "global-conservative: pre-demosaic loss with spatial highlight reconstruction")
    if source.ndim == 3:
        mask = (source[..., :3] != 0).astype(np.uint8)
        reason = "linear-planes: no demosaic; maximum DefaultScale support"
    elif half_size and is_bayer:
        mask = pool_mosaic_loss(source, colors, color_desc)
        reason = "bayer-half: per-plane 2x2 maximum; maximum DefaultScale support"
    elif not half_size and is_bayer and algorithm == "AHD":
        spatial = dilate_loss(source != 0, 5)
        mask = np.broadcast_to(spatial[..., None], (*spatial.shape, 3))
        reason = "bayer-AHD: conservative radius-5 source support; maximum DefaultScale support"
    else:
        return (np.ones((*shape, 3), dtype=np.float16),
                f"global-conservative: pre-demosaic loss with uninstrumented {algorithm} support")
    if highlight == "blend":
        joined = np.max(mask, axis=2)
        mask = np.broadcast_to(joined[..., None], (*joined.shape, 3))
        reason += "; pointwise highlight blend joins RGB support"
    return transport_default_scale(mask, shape, pixel_aspect).astype(np.float16), reason
