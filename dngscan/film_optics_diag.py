# SPDX-License-Identifier: GPL-3.0-or-later
"""Measurement primitives for analog optics (FILM_OPTICS_V2 §9/§10, phase P0).

Every number the V2 plan gates on is computed here, once, so that the tests
and the report tool cannot disagree about what "the halo radius" or "the
granularity" means. Nothing in the render path imports this module.

The four measurements and why each exists:

- **Radial profile / half-energy radius** — a spread operator is characterised
  by where its energy sits, not by the sigma its author typed. Reading the
  radius back off a rendered point source is the only way to catch a kernel
  whose declared mm and delivered mm differ.
- **Radial PSD and aperture RMS** — granularity is a spectrum, not a number.
  A single RMS at one aperture cannot distinguish fine grain from coarse
  mottle; the pair (RMS at 48 um, how that RMS falls with aperture) can.
  Selwyn's law says sigma * sqrt(area) is aperture-independent for a genuine
  grain process, i.e. log sigma falls with slope -1 against log aperture. A
  process that measures far from -1 is not grain at any amplitude.
- **Slanted-edge MTF** — detail cutoff, so "grain got bigger" can be told
  apart from "the picture got softer".
- **Mask isolation** — the delta an operator contributed, alone, in linear
  light, which is the only fair input to any of the above.
"""
from __future__ import annotations

import numpy as np

LUMA_REC2020 = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)


def luminance(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, dtype=np.float64) @ LUMA_REC2020


# --------------------------------------------------------------------------
# radial profile
# --------------------------------------------------------------------------

def radial_profile(
    img: np.ndarray,
    center: tuple[float, float],
    *,
    bin_px: float = 1.0,
    max_radius_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuthally averaged profile about `center`.

    Returns (radius_px at bin centres, mean value [bins, C], sample count).
    Bins with no samples are dropped, so the caller never has to guard NaN.
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w = arr.shape[:2]
    cy, cx = center
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - cy, xx - cx).ravel()
    if max_radius_px is None:
        max_radius_px = float(r.max())
    keep = r <= max_radius_px
    idx = np.floor(r[keep] / bin_px).astype(np.int64)
    n_bins = int(idx.max()) + 1 if idx.size else 0
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    flat = arr.reshape(-1, arr.shape[2])[keep]
    sums = np.stack(
        [np.bincount(idx, weights=flat[:, c], minlength=n_bins) for c in range(arr.shape[2])],
        axis=1,
    )
    live = counts > 0
    radii = (np.arange(n_bins, dtype=np.float64) + 0.5) * bin_px
    return radii[live], (sums[live] / counts[live, None]), counts[live]


def half_energy_radius(
    radii: np.ndarray, profile: np.ndarray, *, baseline: float | None = None
) -> float:
    """Radius containing half of the profile's excess energy over baseline.

    Energy, not amplitude: a wide shallow veil and a tight bright halo can
    share a peak value and differ entirely in where the light actually went.
    Weighted by 2*pi*r because a radial bin's area grows with r.
    """
    p = np.asarray(profile, dtype=np.float64).ravel()
    r = np.asarray(radii, dtype=np.float64).ravel()
    base = float(p[-1]) if baseline is None else float(baseline)
    excess = np.maximum(p - base, 0.0) * r
    total = excess.sum()
    if total <= 0.0:
        return float("nan")
    cdf = np.cumsum(excess) / total
    # cdf[i] is the energy accumulated THROUGH bin i, so it belongs to the
    # bin's upper edge, not its centre. Interpolating on centres biases the
    # radius low by half a bin — 5% on a tight halo, which is exactly the
    # scale of difference this measurement has to resolve.
    half_bin = 0.5 * float(r[1] - r[0]) if r.size > 1 else 0.0
    upper = r + half_bin
    i = int(np.searchsorted(cdf, 0.5))
    if i >= upper.size:
        return float(upper[-1])
    c0 = 0.0 if i == 0 else float(cdf[i - 1])
    lo = 0.0 if i == 0 else float(upper[i - 1])
    t = (0.5 - c0) / max(float(cdf[i]) - c0, 1e-12)
    return float(lo + t * (float(upper[i]) - lo))


def encircled_energy(
    radii: np.ndarray, profile: np.ndarray, *, baseline: float | None = None
) -> np.ndarray:
    """Cumulative fraction of excess energy inside each bin's UPPER edge."""
    p = np.asarray(profile, dtype=np.float64).ravel()
    r = np.asarray(radii, dtype=np.float64).ravel()
    base = float(p[-1]) if baseline is None else float(baseline)
    excess = np.maximum(p - base, 0.0) * r
    total = excess.sum()
    if total <= 0.0:
        return np.zeros_like(r)
    return np.cumsum(excess) / total


# --------------------------------------------------------------------------
# granularity: aperture RMS, Selwyn slope, radial PSD
# --------------------------------------------------------------------------

def aperture_rms(field: np.ndarray, n_cells: int) -> np.ndarray:
    """RMS of the field after non-overlapping n x n box averaging, per channel.

    This is the microdensitometer model: a real granularity reading is the
    standard deviation of density seen through a finite aperture, so any
    comparison to a datasheet number must average first and take the RMS
    second, never the other way round.
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    n = max(int(n_cells), 1)
    h = arr.shape[0] // n * n
    w = arr.shape[1] // n * n
    if h == 0 or w == 0:
        return np.full(arr.shape[2], np.nan)
    blk = arr[:h, :w].reshape(h // n, n, w // n, n, arr.shape[2]).mean(axis=(1, 3))
    return blk.std(axis=(0, 1))


def selwyn_slope(
    field: np.ndarray, *, apertures: tuple[int, ...] = (1, 2, 4, 8, 16)
) -> float:
    """d log(RMS) / d log(aperture), fitted over `apertures`.

    Selwyn's law fixes this at -1 for a granularity process whose correlation
    length is below the smallest aperture: doubling the aperture diameter
    halves the standard deviation. A field that measures, say, -0.3 is
    telling you its correlation length is comparable to the apertures — it is
    blotch, not grain, and no amplitude constant can turn one into the other.
    """
    arr = np.asarray(field, dtype=np.float64)
    xs, ys = [], []
    for n in apertures:
        r = aperture_rms(arr, n)
        m = float(np.mean(r))
        if not np.isfinite(m) or m <= 0.0:
            continue
        xs.append(np.log(float(n)))
        ys.append(np.log(m))
    if len(xs) < 2:
        return float("nan")
    return float(np.polyfit(np.asarray(xs), np.asarray(ys), 1)[0])


def rms_granularity(
    density_field: np.ndarray, pitch_um: float, *, aperture_um: float = 48.0
) -> np.ndarray:
    """Datasheet-comparable granularity: 1000 * sigma_D through a round-number
    aperture. Kodak quotes diffuse RMS granularity at a 48 um aperture, so a
    model claiming to match a stock has to be read the same way."""
    n = max(int(round(aperture_um / max(pitch_um, 1e-9))), 1)
    return 1000.0 * aperture_rms(density_field, n)


def autocorrelation_profile(field: np.ndarray, *, max_lag: int = 32) -> np.ndarray:
    """Normalised autocorrelation along x, lags 0..max_lag (channel mean).

    Computed by FFT on the mean-removed field, so it is the true circular
    autocorrelation of the sample — for the periodic master grain field that
    is also the exact autocorrelation of the process.
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = arr - arr.mean()
    f = np.fft.rfft2(arr)
    ac = np.fft.irfft2(f * np.conj(f), s=arr.shape)
    ac = ac / max(ac[0, 0], 1e-30)
    return ac[0, : max_lag + 1]


def correlation_length_cells(field: np.ndarray, *, max_lag: int = 64) -> float:
    """Lag at which the autocorrelation first falls below 0.5, interpolated.

    Reported in grid cells; multiply by the grid pitch for a physical blob
    size. Half-drop rather than 1/e because it is what a reader can check by
    eye against a crop.
    """
    prof = autocorrelation_profile(field, max_lag=max_lag)
    below = np.nonzero(prof < 0.5)[0]
    if below.size == 0:
        return float(max_lag)
    i = int(below[0])
    if i == 0:
        return 0.0
    p0, p1 = prof[i - 1], prof[i]
    return float((i - 1) + (p0 - 0.5) / max(p0 - p1, 1e-12))


def radial_psd(field: np.ndarray, *, bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectral density of a mean-removed field.

    Returns (frequency in cycles per cell, PSD). Normalised so that the mean
    of the full 2-D periodogram equals the field variance — i.e. the PSD is a
    variance density, and two fields with the same variance but different
    shape are separated by this curve alone.
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = arr - arr.mean()
    h, w = arr.shape
    p = np.abs(np.fft.fft2(arr)) ** 2 / float(h * w)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.hypot(np.broadcast_to(fy, (h, w)), np.broadcast_to(fx, (h, w))).ravel()
    edges = np.linspace(0.0, 0.5, bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, bins - 1)
    counts = np.bincount(idx, minlength=bins).astype(np.float64)
    sums = np.bincount(idx, weights=p.ravel(), minlength=bins)
    live = counts > 0
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres[live], (sums[live] / counts[live])


# --------------------------------------------------------------------------
# slanted-edge MTF
# --------------------------------------------------------------------------

def slanted_edge_mtf(
    plane: np.ndarray, *, oversample: int = 4, half_window_px: float = 16.0
) -> tuple[np.ndarray, np.ndarray]:
    """ISO-12233-style MTF from a slanted edge, on ONE channel.

    Returns (frequency in cycles per pixel, MTF normalised to 1 at DC).
    The edge location is fitted per row from the derivative centroid, all
    pixels are projected onto the edge normal, binned at `oversample` times
    the pixel rate, differentiated and transformed. The tilt is what supplies
    the sub-pixel phases; a straight edge cannot be measured this way.
    """
    img = np.asarray(plane, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    h, w = img.shape
    d = np.gradient(img, axis=1)
    weight = np.abs(d)
    xs = np.arange(w, dtype=np.float64)[None, :]
    # Locate each row's edge at its gradient PEAK first, then take the
    # centroid only in a narrow window around it. A whole-row centroid is
    # pulled off the edge by any wide, low-amplitude halo — a spread operator
    # puts a small gradient across hundreds of columns, and their moment arm
    # is long enough to rival the edge spike. The misfit then smears the
    # reconstructed ESF and the measurement reports the fit error as
    # resolution loss.
    peak = np.argmax(weight, axis=1)
    window = max(int(round(half_window_px)), 3)
    cols = np.arange(w, dtype=np.int64)[None, :]
    near = np.abs(cols - peak[:, None]) <= window
    weight = np.where(near, weight, 0.0)
    denom = weight.sum(axis=1)
    live = denom > 0
    if live.sum() < 4:
        raise ValueError("no detectable edge in the supplied plane")
    centroid = np.full(h, np.nan)
    centroid[live] = (weight[live] * xs).sum(axis=1) / denom[live]
    rows = np.arange(h, dtype=np.float64)[live]
    coef = np.polyfit(rows, centroid[live], 1)
    edge_x = np.polyval(coef, np.arange(h, dtype=np.float64))

    dist = (np.arange(w, dtype=np.float64)[None, :] - edge_x[:, None]).ravel()
    vals = img.ravel()
    keep = np.abs(dist) <= half_window_px
    dist, vals = dist[keep], vals[keep]
    n_bins = int(round(2.0 * half_window_px * oversample))
    idx = np.clip(
        ((dist + half_window_px) * oversample).astype(np.int64), 0, n_bins - 1
    )
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sums = np.bincount(idx, weights=vals, minlength=n_bins)
    if np.any(counts == 0):
        # Fill gaps by interpolation rather than dropping bins: a ragged ESF
        # sample grid silently rescales the frequency axis.
        good = counts > 0
        pos = np.arange(n_bins, dtype=np.float64)
        esf = np.interp(pos, pos[good], (sums[good] / counts[good]))
    else:
        esf = sums / counts

    lsf = np.gradient(esf)
    lsf = lsf - lsf[[0, -1]].mean()
    lsf *= np.hanning(lsf.size)
    spec = np.abs(np.fft.rfft(lsf))
    dc = spec[0] if spec[0] > 0 else 1.0
    mtf = spec / dc
    freq = np.fft.rfftfreq(lsf.size, d=1.0 / oversample)
    keep_f = freq <= 0.5
    return freq[keep_f], mtf[keep_f]


def mtf50(freq: np.ndarray, mtf: np.ndarray) -> float:
    """Frequency where the MTF first crosses 0.5, in cycles per pixel."""
    f = np.asarray(freq, dtype=np.float64)
    m = np.asarray(mtf, dtype=np.float64)
    below = np.nonzero(m < 0.5)[0]
    if below.size == 0:
        return float(f[-1])
    i = int(below[0])
    if i == 0:
        return 0.0
    m0, m1 = m[i - 1], m[i]
    t = (m0 - 0.5) / max(m0 - m1, 1e-12)
    return float(f[i - 1] + t * (f[i] - f[i - 1]))


# --------------------------------------------------------------------------
# isolation
# --------------------------------------------------------------------------

def isolate(with_effect: np.ndarray, without_effect: np.ndarray) -> np.ndarray:
    """The operator's own contribution, in the linear domain it was applied in.

    Kept as a named function because the mistake it prevents is common: a
    difference taken on 8-bit output or after a gamut fit measures the
    delivery chain, not the operator, and every radius read from it is wrong.
    """
    return np.asarray(with_effect, dtype=np.float64) - np.asarray(
        without_effect, dtype=np.float64
    )


def energy_ratio(delta: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Per-channel sum(delta) / sum(base): whether an operator conserves,
    adds or removes light overall. A conservative scatter must read ~0."""
    d = np.asarray(delta, dtype=np.float64).sum(axis=(0, 1))
    b = np.asarray(base, dtype=np.float64).sum(axis=(0, 1))
    return d / np.maximum(np.abs(b), 1e-30)


def chroma_luma_ratio(delta: np.ndarray) -> float:
    """RMS chroma over RMS luma of a difference image.

    The number that separates "grain" from "colour speckle": real dye-cloud
    granularity is dominated by the luminance component after the three
    layers are viewed together, so a ratio near or above 1 means the
    cross-layer covariance is wrong regardless of amplitude.
    """
    d = np.asarray(delta, dtype=np.float64)
    lum = d @ LUMA_REC2020
    chroma = d - lum[..., None]
    l_rms = float(np.sqrt(np.mean(lum ** 2)))
    c_rms = float(np.sqrt(np.mean(chroma ** 2)))
    return c_rms / max(l_rms, 1e-30)
