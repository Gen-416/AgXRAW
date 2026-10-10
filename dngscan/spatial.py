# SPDX-License-Identifier: GPL-3.0-or-later
"""Generic area resampling and row-streamed spatial correction utilities."""
from __future__ import annotations
import math
import os
from ._deps import np

SPATIAL_GRID_RESERVE_MIB = 348


def spatial_budget_mib() -> int:
    """Bounded temporary-memory tier for streamed spatial processing."""
    try:
        value = int(os.environ.get("DNGSCAN_SPATIAL_BUDGET_MIB", "512"))
    except ValueError:
        return 512
    return value if value in (512, 1024) else 512


def spread_grid_shape(height: int, width: int) -> tuple[int, int]:
    """Aspect-preserving coarse grid used for low-frequency chroma repair."""
    limit = 2048 if spatial_budget_mib() == 1024 else 1408
    long_side = max(height, width)
    if long_side <= limit:
        return height, width
    scale = limit / long_side
    return max(int(round(height * scale)), 1), max(int(round(width * scale)), 1)


def chroma_nr_grid_shape(bundle, height: int, width: int) -> tuple[int, int]:
    """Memory-bounded NR grid with at least four native sensels per cell.

    More working memory permits a finer grid, but cannot make the independent
    Bayer-sample approximation invalid. Use the same retained sensor window as
    covariance propagation; the sampling ruler covers callers without that
    descriptor, including full/half decodes, oriented crops and compact proxies.
    The propagator still validates the resulting sample count independently.
    """
    dh, dw = spread_grid_shape(height, width)
    descriptor = getattr(bundle, "noise_decode", None) or {}
    native = descriptor.get("sensor_window_shape")
    try:
        nh, nw = (float(x) for x in native)
        if not all(math.isfinite(x) and x > 0 for x in (nh, nw)):
            raise ValueError("invalid sensor window")
    except (TypeError, ValueError, OverflowError):
        from .sampling_geometry import sensor_px_per_render_axes
        sy, sx = sensor_px_per_render_axes(bundle, height, width)
        nh, nw = sy * height, sx * width
    max_cells = nh * nw / 4.
    if dh * dw <= max_cells:
        return dh, dw
    scale = math.sqrt(max_cells / (dh * dw))
    dh, dw = max(1, math.floor(dh * scale)), max(1, math.floor(dw * scale))
    # A very thin raster can hit the one-cell minimum on one axis. Keep the
    # other axis within the area bound too, rather than rounding above it.
    if dh * dw > max_cells:
        if dw >= dh:
            dw = max(1, math.floor(max_cells / dh))
        else:
            dh = max(1, math.floor(max_cells / dw))
    return dh, dw


def spatial_band_rows(width: int) -> int:
    """Source rows per band, with room reserved for coarse noise maps."""
    budget_bytes = max(spatial_budget_mib() - SPATIAL_GRID_RESERVE_MIB, 32) * (1 << 20)
    per_row = max(width, 1) * 3 * 4 * 8
    return int(np.clip(budget_bytes // (3 * per_row), 1, 8192))


def area_resample(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Exact area-mean resample [h,w,c] -> [out_h,out_w,c] for arbitrary
    ratios via the same fractional integral-image rectangles as
    sample_field. Linear-domain energy is conserved per output cell."""
    h, w = img.shape[:2]
    if (h, w) == (out_h, out_w):
        return np.asarray(img, dtype=np.float32)
    ii = np.zeros((h + 1, w + 1, img.shape[2]), dtype=np.float64)
    np.cumsum(img, axis=0, out=ii[1:, 1:])
    np.cumsum(ii[1:, 1:], axis=1, out=ii[1:, 1:])
    ye = h * np.arange(out_h + 1) / out_h
    xe = w * np.arange(out_w + 1) / out_w

    def _ii_at(yq, xq):
        yi = np.clip(np.floor(yq).astype(int), 0, h - 1)
        xi = np.clip(np.floor(xq).astype(int), 0, w - 1)
        yf = (yq - yi)[:, None, None]
        xf = (xq - xi)[None, :, None]
        top = ii[yi][:, xi] * (1 - xf) + ii[yi][:, xi + 1] * xf
        bot = ii[yi + 1][:, xi] * (1 - xf) + ii[yi + 1][:, xi + 1] * xf
        return top * (1 - yf) + bot * yf

    s = (
        _ii_at(ye[1:], xe[1:]) - _ii_at(ye[1:], xe[:-1])
        - _ii_at(ye[:-1], xe[1:]) + _ii_at(ye[:-1], xe[:-1])
    )
    area = (ye[1:] - ye[:-1])[:, None, None] * (xe[1:] - xe[:-1])[None, :, None]
    return (s / np.maximum(area, 1e-12)).astype(np.float32)



def area_decimate(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Exact area-mean decimation [h,w,3] -> [out_h,out_w,3], structured as
    row accumulation so the renderer can stream source rows in bands and the
    full-frame oracle can pass the whole array — both produce identical
    bytes. Columns first (1-D fractional integral per row), then each source
    row scatters into the (at most two) decimated rows it overlaps."""
    img = np.asarray(img)
    h, w = img.shape[:2]
    acc = np.zeros((out_h, out_w, img.shape[2]), dtype=np.float64)
    area_decimate_rows(img, 0, h, w, out_h, out_w, acc)
    return (acc).astype(np.float32)



def area_decimate_rows(
    rows: np.ndarray,
    y0: int,
    h: int,
    w: int,
    out_h: int,
    out_w: int,
    acc: np.ndarray,
) -> None:
    """Accumulate source rows [y0, y0+rows.shape[0]) into the decimated
    accumulator (see area_decimate). Deterministic in any band split.

    Stage 2 (2026-09-15): the Rust kernel replicates this body element for
    element (sequential float64 column integral, np.add.at accumulation
    order, float32 accumulators updated as f32(f64(acc) + v)). Float32
    source views are promoted sample by sample in Rust without a band copy;
    this NumPy body is the reference (tests/test_rust_stage2.py)."""
    from . import _fast

    native = _fast.kernel("area_decimate_rows")
    if native is not None and acc.dtype in (np.float64, np.float32) and acc.flags["C_CONTIGUOUS"]:
        native(np.asarray(rows), int(y0), int(h), int(w), int(out_h), int(out_w), acc)
        return
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, w, rows.shape[-1])
    n = rows.shape[0]
    # columns: fractional integral image along x
    cs = np.zeros((n, w + 1, rows.shape[2]), dtype=np.float64)
    np.cumsum(rows, axis=1, out=cs[:, 1:])
    xe = w * np.arange(out_w + 1) / out_w
    xi = np.clip(np.floor(xe).astype(int), 0, w - 1)
    xf = xe - xi
    at = cs[:, xi] * (1 - xf)[None, :, None] + cs[:, xi + 1] * xf[None, :, None]
    col = (at[:, 1:] - at[:, :-1]) / np.maximum(
        (xe[1:] - xe[:-1])[None, :, None], 1e-12
    )
    # rows: each source row [y, y+1) overlaps at most two decimated rows
    ye = h * np.arange(out_h + 1) / out_h
    ys = np.arange(y0, y0 + n, dtype=np.float64)
    lo = np.clip(np.searchsorted(ye, ys, side="right") - 1, 0, out_h - 1)
    for shift in (0, 1):
        idx = np.clip(lo + shift, 0, out_h - 1)
        seg_lo = np.maximum(ys, ye[idx])
        seg_hi = np.minimum(ys + 1.0, ye[np.minimum(idx + 1, out_h)])
        wgt = np.maximum(seg_hi - seg_lo, 0.0) / np.maximum(
            ye[np.minimum(idx + 1, out_h)] - ye[idx], 1e-12
        )
        if shift == 1:
            wgt = np.where(idx > lo, wgt, 0.0)
        np.add.at(acc, idx, col * wgt[:, None, None])



def upsample_rows(map_dec: np.ndarray, y0: int, y1: int, height: int, width: int) -> np.ndarray:
    """Bilinear upsample of a decimated map for output rows [y0, y1): the
    row-band path samples exactly the same continuous surface the full-frame
    path does, so band seams are zero by construction."""
    from . import _fast

    native = _fast.kernel("upsample_rows")
    if native is not None:
        return native(np.asarray(map_dec, dtype=np.float32), int(y0), int(y1), int(height), int(width))
    dh, dw = map_dec.shape[:2]
    yq = (np.arange(y0, y1) + 0.5) / height * dh - 0.5
    xq = (np.arange(width) + 0.5) / width * dw - 0.5
    yi = np.clip(np.floor(yq).astype(int), 0, dh - 1)
    xi = np.clip(np.floor(xq).astype(int), 0, dw - 1)
    y1i = np.minimum(yi + 1, dh - 1)
    x1i = np.minimum(xi + 1, dw - 1)
    yf = np.clip(yq - yi, 0.0, 1.0)[:, None, None]
    xf = np.clip(xq - xi, 0.0, 1.0)[None, :, None]
    a = map_dec[yi][:, xi]
    b = map_dec[yi][:, x1i]
    c = map_dec[y1i][:, xi]
    d = map_dec[y1i][:, x1i]
    return (
        (a * (1 - xf) + b * xf) * (1 - yf) + (c * (1 - xf) + d * xf) * yf
    ).astype(np.float32)
