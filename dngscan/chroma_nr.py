# SPDX-License-Identifier: GPL-3.0-or-later
"""Model-guided chroma smoothing in a declared sensor-pixel band.

The noise scale must come from independent calibration, propagated into
this scene grid. Image details only provide evidence for preserving
structure; they never estimate the noise threshold. Missing or zero noise
models produce an exact zero correction. This is a denoising tradeoff:
weak colour structure near the noise scale can still be attenuated.

Corrections preserve scene Rec.2020 luminance by projection. Fine detail
levels and the coarsest residual pass through; neither property is a
guarantee that all real colour texture survives in the processed band.
The removed fraction is 1/(1 + (d/T)²), with the threshold derived from
calibrated noise and excess local signal energy. Only LUMINANCE, at the scene stage,
is preserved; subsequent tone/gamut operations can change display luminance.
Amount zero remains the caller's original no-context fast path.
"""
from __future__ import annotations

from ._deps import np

# Rec.2020 luma row — the projection axis. Kept local so the operator has
# no import-time dependency on the render modules that call it.
LUMA_W = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float32)

# The shrunk band, declared in FULL-RESOLUTION SENSOR PIXELS — the
# sensor coordinate, so the scale follows capture pixels rather than the
# output size. Structure finer than
# BAND_LO_PX is the kept texture — pixel speckle and the film-like fine
# coloured graininess; structure coarser than BAND_HI_PX is treated as
# real colour and passes through in the residual. Both bounds are
# modelled choices, not measurements.
BAND_LO_PX = 8.0
BAND_HI_PX = 128.0
_SQRT2 = 2.0 ** 0.5
_B3 = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0


def atrous_levels_for(decimation_factor: float, *, axis_factors=None) -> tuple[int, ...]:
    """Which à-trous levels on the decimated grid fall inside the declared
    full-resolution band. Level k shrinks detail between hole spacings
    2^k and 2^(k+1) cells, i.e. factor·2^k .. factor·2^(k+1) sensor px —
    include it while the level's geometric centre factor·2^k·√2 lies in
    [BAND_LO_PX, BAND_HI_PX]. The realized band is octave-aligned, so it
    can only be declared to within a factor √2: it always lies INSIDE
    [BAND_LO_PX/√2, BAND_HI_PX·√2] and always COVERS [BAND_LO_PX·√2,
    BAND_HI_PX/√2] (the lower edge bounded below by the grid's own cell
    when the decimation factor exceeds BAND_LO_PX·√2). (Review batch 23: the earlier any-overlap rule let the
    top level reach 2·BAND_HI_PX at some decimation factors — a memory
    tier could silently widen the declared band.) At identity grids (small
    renders) the first levels fall below BAND_LO_PX and are skipped, which
    is what keeps pixel-scale speckle untouched there."""
    if axis_factors is None:
        factors = (max(float(decimation_factor), 1.0),)
    else:
        factors = tuple(float(v) for v in axis_factors)
        if len(factors) != 2 or not np.isfinite(factors).all() or min(factors) <= 0:
            raise ValueError("sensor sampling must contain two finite positive axis scales")
    levels = []
    for k in range(13):
        centres = tuple(factor * (2.0 ** k) * _SQRT2 for factor in factors)
        if max(centres) > BAND_HI_PX:
            break
        if min(centres) >= BAND_LO_PX:
            levels.append(k)
    return tuple(levels)


def _atrous_smooth(plane: np.ndarray, level: int) -> np.ndarray:
    """Dispatch the production 2-D planes; retain the general NumPy reference."""
    from . import _fast
    limit = int(np.iinfo(np.intp).max)
    if (type(plane) is np.ndarray and plane.ndim == 2 and plane.size > 0
            and plane.dtype == np.float32 and plane.flags.aligned
            and plane.size * plane.itemsize <= limit
            and all(stride % plane.itemsize == 0 and stride != -limit - 1
                    for stride in plane.strides)
            and plane.itemsize + sum((dim - 1) * abs(stride)
                    for dim, stride in zip(plane.shape, plane.strides)) <= limit
            and isinstance(level, int) and 0 <= level < 32
            and _fast._fast_mode() != "off"
            and "atrous_smooth_f32" not in _fast._skipped_kernels()):
        ext = _fast._load_extension()
        if ext is not None:
            try:
                return ext.atrous_smooth_f32(plane, level)
            except Exception as exc:
                if _fast.strict_requested():
                    raise _fast.NativeKernelError("native B3 smoothing failed") from exc
    return _atrous_smooth_reference(plane, level)


def _atrous_smooth_reference(plane: np.ndarray, level: int) -> np.ndarray:
    """One à-trous B3 smoothing pass with hole spacing 2**level, separable,
    reflect-padded. plane is [h, w, c] float32."""
    step = 1 << level
    h, w = plane.shape[:2]
    out = plane
    for axis, n in ((0, h), (1, w)):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (2 * step, 2 * step)
        padded = np.pad(out, pad, mode="reflect")
        acc = np.zeros_like(out)
        for j, kv in enumerate(_B3):
            sl = [slice(None)] * out.ndim
            offset = j * step
            sl[axis] = slice(offset, offset + n)
            acc += np.float32(kv) * padded[tuple(sl)]
        out = acc
    return out


def chroma_correction_map(
    scene_dec: np.ndarray,
    amount: float,
    decimation_factor: float = 1.0,
    decimation_axis_factors: tuple[float, float] | None = None,
    *,
    noise_covariance: np.ndarray | None = None,
    chroma_variance: np.ndarray | None = None,
    detail_variance: dict[int, np.ndarray] | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """The zero-luma correction to ADD to the scene, on the decimated grid.

    scene_dec: [dh, dw, 3] area-decimated scene-linear Rec.2020 (signed —
    this is a colour-path repair, not a light-transport source, so it sees
    the same signed coordinates the colour path does).
    decimation_factor: full-resolution pixels per decimated cell (long-side
    ratio) — anchors the shrunk band in sensor pixels (atrous_levels_for).

    ``noise_covariance`` is a constant or per-pixel scene-RGB covariance;
    pixels must be spatially independent. ``chroma_variance`` supplies its
    already-projected diagonal instead. Correlated-noise models can supply
    ``detail_variance`` directly for each wavelet band. These are independent
    model inputs, never statistics calculated from ``scene_dec``.

    Returns [dh, dw, 3] float32 with w·map == 0 per pixel. Without noise
    evidence, returns zeros rather than falling back to image-detail MAD.
    """
    if not np.isfinite(amount) or amount <= 0.0:
        raise ValueError("chroma_correction_map is only defined for amount > 0")
    dec = np.asarray(scene_dec, dtype=np.float32)
    if dec.ndim != 3 or dec.shape[-1] != 3 or not np.isfinite(dec).all():
        raise ValueError("scene must be finite HxWx3 Rec.2020")
    if sum(x is not None for x in (noise_covariance, chroma_variance, detail_variance)) > 1:
        raise ValueError("provide one noise variance representation")
    if noise_covariance is None and chroma_variance is None and detail_variance is None:
        return np.zeros_like(dec)
    if noise_covariance is not None:
        from .noise_propagation import transform_covariance

        projection = np.eye(3) - np.ones((3, 1)) * LUMA_W.astype(np.float64)[None, :]
        cov = transform_covariance(noise_covariance, projection)
        chroma_variance = np.diagonal(cov, axis1=-2, axis2=-1)
    if chroma_variance is not None:
        variance = np.asarray(chroma_variance, dtype=np.float32)
        if variance.shape not in ((3,), dec.shape):
            raise ValueError("chroma variance must be RGB or HxWx3")
        if not np.isfinite(variance).all() or np.any(variance < 0):
            raise ValueError("noise variance must be finite and nonnegative")
        variance = np.broadcast_to(variance, dec.shape)
        if not np.any(variance):
            return np.zeros_like(dec)
    else:
        variance = None
        for value in detail_variance.values():
            value = np.asarray(value)
            if value.shape not in ((3,), dec.shape) or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError("detail variance must be finite nonnegative RGB or HxWx3")
        if not any(np.any(value) for value in detail_variance.values()):
            return np.zeros_like(dec)
    y = dec @ LUMA_W
    chroma = dec - y[..., None]
    del dec, y

    # Multiplicative domain would be ill-defined at y <= 0; the additive
    # opponent form keeps the operator linear-in-signal and lets shadows —
    # where the mottle lives — carry proportionally small absolute
    # corrections bounded by their own chroma.
    total_removed = np.zeros_like(chroma)
    max_step = max((min(chroma.shape[:2]) - 1) // 2, 1)
    included = set(atrous_levels_for(decimation_factor, axis_factors=decimation_axis_factors))
    top = max(included) if included else -1
    validity = {}
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=np.float32)
        if valid.shape != chroma.shape[:2] or not np.isfinite(valid).all():
            raise ValueError("noise validity mask must match the scene grid")
        valid = (valid > 0.999999).astype(np.float32)
        for level in range(top + 1):
            invalid = _atrous_smooth(np.float32(1) - valid, level)
            valid = (invalid == 0).astype(np.float32)
            validity[level] = valid.astype(bool)
    # The cascade always runs from level 0 — an à-trous level's hole
    # spacing only avoids aliasing on the PROGRESSIVELY smoothed image —
    # but only the in-band levels shrink; protected fine levels pass
    # through untouched inside their detail coefficients.
    #
    # Retain the per-channel cascade to bound scratch storage. Noise
    # propagation adds its own working set; old MAD-path memory figures
    # are not a measurement of this calibrated implementation.
    for c in range(3):
        smooth = chroma[..., c]
        for level in range(top + 1):
            if (1 << level) > max_step:
                break  # grid too small to carry this scale's reflect pad
            coarser = _atrous_smooth(smooth, level)
            if level in included:
                detail = smooth - coarser
                if detail_variance is not None:
                    supplied = detail_variance.get(level)
                    if supplied is None:
                        smooth = coarser
                        continue
                    noise_var = np.broadcast_to(np.asarray(supplied, dtype=np.float32), chroma.shape)[..., c]
                else:
                    from .noise_propagation import atrous_detail_variance

                    noise_var = atrous_detail_variance(variance[..., c], level)
                if level in validity:
                    noise_var = np.where(validity[level], noise_var, np.float32(0))
                # Profiled/BayesShrink principle: expected random noise is
                # known independently. Observed energy above that expectation
                # protects local structure; it cannot increase the noise model.
                local_energy = _atrous_smooth(np.square(detail), 0)
                local_noise = _atrous_smooth(noise_var, 0)
                signal_var = np.maximum(local_energy - local_noise, np.float32(0))
                floor = np.maximum(noise_var * np.float32(1e-6), np.float32(1e-30))
                t2 = np.minimum(noise_var, np.square(noise_var) / np.maximum(signal_var, floor))
                t2 *= np.float32(float(amount) ** 2 * 9.0)
                total_removed[..., c] += detail * (
                    t2 / (t2 + np.square(detail) + np.float32(1e-30))
                )
                del detail
            smooth = coarser
        del smooth

    correction = -total_removed
    # Exact zero-luma projection: whatever numerical luma the per-channel
    # shrinkage introduced is subtracted here, so Y is preserved to float
    # precision at every pixel — the grain lives in Y and must not move.
    correction -= (correction @ LUMA_W)[..., None]
    return correction.astype(np.float32, copy=False)


def apply_chroma_correction_rows(
    rgb_rows: np.ndarray,
    correction_map: np.ndarray,
    y0: int,
    y1: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Add the upsampled correction to scene rows [y0, y1) — the streaming
    row-band form; the full-frame oracle passes (0, height). Bilinear
    upsampling is the same continuous surface everywhere, so band seams are
    zero by construction (spatial.upsample_rows contract)."""
    from .spatial import upsample_rows

    rows = np.asarray(rgb_rows, dtype=np.float32).reshape(y1 - y0, width, 3)
    return (
        rows + upsample_rows(correction_map, y0, y1, height, width)
    ).reshape(-1, 3)


def apply_chroma_correction_flat(
    rgb_flat: np.ndarray,
    correction_map: np.ndarray,
    start: int,
    end: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Row-band apply for a FLAT pixel chunk [start, end) that may cut rows
    mid-way (the 1M-pixel streaming chunks): upsample the covering rows and
    slice — the same continuous bilinear surface, so chunk boundaries are
    seam-free like band boundaries."""
    from .spatial import upsample_rows

    y0 = start // width
    y1 = (end - 1) // width + 1
    up = upsample_rows(correction_map, y0, y1, height, width).reshape(-1, 3)
    flat = np.asarray(rgb_flat, dtype=np.float32).reshape(-1, 3)
    return flat + up[start - y0 * width:end - y0 * width]
