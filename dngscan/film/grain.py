# SPDX-License-Identifier: GPL-3.0-or-later
"""Negative-density grain for the observe film engine.

Ported from the frozen integration (AgXRAW 522c7ed, ``film_optics``: the
measured_sigma_v2 path, NumPy branch) and the AgXFilm research operator
``tools/observe_grain.py``. The physics:

    log_e   = Stage-A layer exposure of the scene pixel (+ anchor)
    D       = characteristic_c(log_e_c)                    (per layer)
    D'      = D + sigma_c(D) * amount * F(x, y) / rms48 + ln10/2 * s^2
    log_e'  = characteristic_c^-1(D')
    rgb'_c  = rgb_c * 10^(log_e'_c - log_e_c)

F is one band-limited, periodic, unit-RMS film-plane field (Philox, seed 0)
sampled by exact area integration over each pixel's footprint on a 36x24 mm
gate; ``rms48`` calibrates it so a 48 um aperture reproduces the chart sigma.
A per-RAW integer seed only shifts the field's phase. The AgX curve that
follows approximates print∘negative, so the output perturbation matches the
full chain to first order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

LOG10_2 = 0.30102999566398119521
MAX_DELTA_LOG10 = 0.6   # +/- 2 stops: rails the inversion where a table is flat
GATE_W_MM = 36.0
GATE_H_MM = 24.0


@dataclass(frozen=True)
class GrainProfile:
    pitch_um: float
    size_um: float
    layer_corr: float
    bands: tuple
    aperture_um: float
    chart_density: tuple      # ((base, max),) * 3, RGB
    sigma_density: tuple      # (((D, sigma), ...),) * 3, RGB

    @classmethod
    def from_asset(cls, raw: dict) -> "GrainProfile":
        if raw.get("model") != "measured_sigma_v2" or raw.get("channel_order") != "RGB":
            raise ValueError("film grain asset: unsupported model or channel order")
        return cls(
            pitch_um=float(raw["pitch_um"]), size_um=float(raw["size_um"]),
            layer_corr=float(raw["layer_corr"]),
            bands=tuple((float(s), float(w)) for s, w in raw["bands"]),
            aperture_um=float(raw["aperture_um"]),
            chart_density=tuple((float(a), float(b)) for a, b in raw["chart_density"]),
            sigma_density=tuple(tuple((float(d), float(s)) for d, s in t) for t in raw["sigma_density"]),
        )


# --------------------------------------------------------------------------
# film-plane geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FilmGeometry:
    """Affine from the 36x24 mm gate to the rendering's pixel grid; portrait
    images turn the gate, non-3:2 aspects letterbox centred inside it."""

    height: int
    width: int
    x0_mm: float = 0.0
    y0_mm: float = 0.0
    w_mm: float = GATE_W_MM
    h_mm: float = 0.0
    rotated: bool = False

    @classmethod
    def fit(cls, height: int, width: int) -> "FilmGeometry":
        rotated = height > width
        gate_w, gate_h = (GATE_H_MM, GATE_W_MM) if rotated else (GATE_W_MM, GATE_H_MM)
        aspect = height / max(width, 1)
        if gate_w * aspect <= gate_h:
            w_mm, h_mm = gate_w, gate_w * aspect
        else:
            h_mm, w_mm = gate_h, gate_h / aspect
        return cls(height, width, x0_mm=(gate_w - w_mm) / 2.0, y0_mm=(gate_h - h_mm) / 2.0,
                   w_mm=w_mm, h_mm=h_mm, rotated=rotated)

    def region(self) -> tuple[float, float, float, float]:
        h_mm = self.h_mm if self.h_mm > 0 else self.w_mm * self.height / max(self.width, 1)
        return (self.x0_mm, self.y0_mm, self.w_mm, h_mm)

    def rows(self, y0: int, y1: int) -> "FilmGeometry":
        _, base_y0, w_mm, h_mm = self.region()
        return FilmGeometry(
            y1 - y0, self.width, x0_mm=self.x0_mm,
            y0_mm=base_y0 + h_mm * y0 / max(self.height, 1), w_mm=w_mm,
            h_mm=h_mm * (y1 - y0) / max(self.height, 1), rotated=self.rotated,
        )


# --------------------------------------------------------------------------
# the master field
# --------------------------------------------------------------------------

def _film_grid_shape(g: GrainProfile) -> tuple[int, int]:
    pitch_mm = g.pitch_um * 1e-3
    return (int(round(GATE_H_MM / pitch_mm)), int(round(GATE_W_MM / pitch_mm)))


def _gaussian_blur_periodic(img: np.ndarray, sigma: float, slab: int = 256) -> np.ndarray:
    """Separable float32 Gaussian with wrap padding, in halo slabs (the frozen
    ``_gaussian_blur_slabbed(periodic=True)`` NumPy branch, same taps/order)."""
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    k32 = k.astype(np.float32)
    src = np.ascontiguousarray(img, dtype=np.float32)
    h, w = src.shape[:2]
    for c0 in range(0, w, slab):
        c1 = min(c0 + slab, w)
        pad = np.pad(src[:, c0:c1], ((radius, radius), (0, 0), (0, 0)), mode="wrap")
        acc = np.zeros((h, c1 - c0, src.shape[2]), dtype=np.float32)
        for i in range(k32.size):
            acc += k32[i] * pad[i:i + h]
        src[:, c0:c1] = acc
    for r0 in range(0, h, slab):
        r1 = min(r0 + slab, h)
        pad = np.pad(src[r0:r1], ((0, 0), (radius, radius), (0, 0)), mode="wrap")
        acc = np.zeros((r1 - r0, w, src.shape[2]), dtype=np.float32)
        for i in range(k32.size):
            acc += k32[i] * pad[:, i:i + w]
        src[r0:r1] = acc
    return src


def band_limited_field(g: GrainProfile, seed: int = 0) -> np.ndarray:
    """[gh, gw, 3] unit-RMS per layer; cross-layer correlation through a shared
    component; multi-band Gaussian spectrum. Philox keyed on the seed alone."""
    gh, gw = _film_grid_shape(g)
    rng = np.random.Generator(np.random.Philox(key=int(seed) & 0xFFFFFFFF))
    c = float(np.clip(g.layer_corr, 0.0, 1.0))
    bands = g.bands or ((g.size_um, 1.0),)
    field = None
    for size_um, weight in bands:
        white = rng.standard_normal((gh, gw, 3), dtype=np.float32)
        shared = rng.standard_normal((gh, gw, 1), dtype=np.float32)
        white *= np.float32(np.sqrt(1.0 - c))
        for ch in range(3):
            white[..., ch] += np.float32(np.sqrt(c)) * shared[..., 0]
        del shared
        sigma_cells = float(size_um) / g.pitch_um
        band = _gaussian_blur_periodic(white, sigma_cells) if sigma_cells > 0.55 else white
        n = band.shape[0] * band.shape[1]
        rms = np.sqrt(np.einsum("hwc,hwc->c", band, band, dtype=np.float64) / n)
        band /= np.maximum(rms, 1e-12).astype(np.float32)
        band *= np.float32(weight)
        if field is None:
            field = band
        else:
            field += band
            del band
    n = field.shape[0] * field.shape[1]
    rms = np.sqrt(np.einsum("hwc,hwc->c", field, field, dtype=np.float64) / n)
    field /= np.maximum(rms, 1e-12).astype(np.float32)
    return field


def integral_image(field: np.ndarray) -> np.ndarray:
    """float32 summed-area table [gh+1, gw+1, 3], accumulated in float64
    column slabs (frozen ``_grain_ii_for``)."""
    gh, gw = field.shape[:2]
    ii32 = np.zeros((gh + 1, gw + 1, field.shape[2]), dtype=np.float32)
    carry = np.zeros((gh + 1, 1, field.shape[2]), dtype=np.float64)
    for c0 in range(0, gw, 256):
        c1 = min(c0 + 256, gw)
        block = field[:, c0:c1].astype(np.float64)
        np.cumsum(block, axis=0, out=block)
        np.cumsum(block, axis=1, out=block)
        block += carry[1:]
        ii32[1:, c0 + 1:c1 + 1] = block
        carry[1:] = block[:, -1:]
    return ii32


def aperture_rms(ii: np.ndarray, g: GrainProfile) -> float:
    """RMS of the master field averaged over the chart's measurement aperture."""
    n = max(int(round(g.aperture_um / g.pitch_um)), 1)
    acc = 0.0
    for ch in range(3):
        c = ii[..., ch]
        box = c[n:, n:].astype(np.float64)
        box -= c[n:, :-n]
        box -= c[:-n, n:]
        box += c[:-n, :-n]
        box /= float(n * n)
        box *= box
        acc += float(box.mean())
    return float(np.sqrt(acc / 3.0))


def _splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def realization_phases(seed: int, gh: int, gw: int) -> tuple[int, int]:
    seed = int(seed)
    if seed == 0:
        return (0, 0)
    a = _splitmix64(seed)
    b = _splitmix64(a)
    return (a % max(gh, 1), b % max(gw, 1))


def sample_field(ii: np.ndarray, geometry: FilmGeometry, phase: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Exact area-integrated sampling of the periodic film-plane field onto the
    pixel grid (frozen ``sample_field`` NumPy branch)."""
    if geometry.rotated:
        ii = ii.transpose(1, 0, 2)
        gate_w_mm, gate_h_mm = GATE_H_MM, GATE_W_MM
        phase = (phase[1], phase[0])
    else:
        gate_w_mm, gate_h_mm = GATE_W_MM, GATE_H_MM
    gh, gw = ii.shape[0] - 1, ii.shape[1] - 1
    x0, y0, w_mm, h_mm = geometry.region()
    ye = np.clip((y0 + h_mm * np.arange(geometry.height + 1) / geometry.height)
                 / gate_h_mm * gh, 0.0, gh) + float(phase[0])
    xe = np.clip((x0 + w_mm * np.arange(geometry.width + 1) / geometry.width)
                 / gate_w_mm * gw, 0.0, gw) + float(phase[1])

    def _ii_at(yq, xq):
        yi = np.clip(np.floor(yq).astype(int), 0, gh - 1)
        xi = np.clip(np.floor(xq).astype(int), 0, gw - 1)
        yf = (yq - yi)[:, None, None]
        xf = (xq - xi)[None, :, None]
        top = ii[yi][:, xi] * (1 - xf) + ii[yi][:, xi + 1] * xf
        bot = ii[yi + 1][:, xi] * (1 - xf) + ii[yi + 1][:, xi + 1] * xf
        return top * (1 - yf) + bot * yf

    if phase == (0, 0):
        ye_q, xe_q = ye, xe
        straddle_row = straddle_col = None
        periodic_at = None
    else:
        ye_q = ye - gh * np.floor(ye / gh)
        xe_q = xe - gw * np.floor(xe / gw)

        def _straddler(edges, G):
            over = np.nonzero((edges[:-1] - G * np.floor(edges[:-1] / G))
                              > (edges[1:] - G * np.floor(edges[1:] / G)))[0]
            return int(over[0]) if over.size else None

        straddle_row = _straddler(ye, gh)
        straddle_col = _straddler(xe, gw)
        total = ii[gh, gw].astype(np.float64)
        row_tot = ii[gh]
        col_tot = ii[:, gw]

        def _interp1(table, q, n):
            qi = np.clip(np.floor(q).astype(int), 0, n - 1)
            qf = (q - qi)[:, None]
            return table[qi] * (1 - qf) + table[qi + 1] * qf

        def periodic_at(yq, xq):
            ky = np.floor(yq / gh).astype(np.float64)
            kx = np.floor(xq / gw).astype(np.float64)
            base = _ii_at(yq - ky * gh, xq - kx * gw)
            wrap_y = _interp1(row_tot, xq - kx * gw, gw)[None, :, :]
            wrap_x = _interp1(col_tot, yq - ky * gh, gh)[:, None, :]
            return (base + ky[:, None, None] * wrap_y + kx[None, :, None] * wrap_x
                    + (ky[:, None, None] * kx[None, :, None]) * total[None, None, :])

    out = np.empty((geometry.height, geometry.width, ii.shape[2]), dtype=np.float32)

    def _fill(rows, yq_e, ya_e, cols, xq_e, xa_e, query):
        n_rows, n_cols = rows.stop - rows.start, cols.stop - cols.start
        if n_rows <= 0 or n_cols <= 0:
            return
        area_x = np.maximum(xa_e[1:] - xa_e[:-1], 1e-12)[None, :, None]
        slab = max(1, 8_000_000 // max(n_cols, 1))
        for r0 in range(0, n_rows, slab):
            r1 = min(r0 + slab, n_rows)
            e = query(yq_e[r0:r1 + 1], xq_e)
            area = (ya_e[r0 + 1:r1 + 1] - ya_e[r0:r1])[:, None, None] * area_x
            out[rows.start + r0:rows.start + r1, cols] = (
                (e[1:, 1:] - e[1:, :-1] - e[:-1, 1:] + e[:-1, :-1]) / np.maximum(area, 1e-12))

    _fill(slice(0, geometry.height), ye_q, ye, slice(0, geometry.width), xe_q, xe, _ii_at)
    if periodic_at is not None:
        ph_ye = ye - gh * np.floor(ye[:1] / gh)
        ph_xe = xe - gw * np.floor(xe[:1] / gw)
        if straddle_row is not None:
            _fill(slice(straddle_row, straddle_row + 1), ph_ye[straddle_row:straddle_row + 2],
                  ye[straddle_row:straddle_row + 2], slice(0, geometry.width), ph_xe, xe, periodic_at)
        if straddle_col is not None:
            _fill(slice(0, geometry.height), ph_ye, ye, slice(straddle_col, straddle_col + 1),
                  ph_xe[straddle_col:straddle_col + 2], xe[straddle_col:straddle_col + 2], periodic_at)
    return out


_MASTER: dict = {}
# A band whose Gaussian sigma is below this many master cells cannot be
# represented on the film-plane grid: the frozen field rendered it as one
# independent value per 12 um cell, i.e. 2x2-pixel blocks at 6 um/px -- the
# "digital noise" look. Such bands are synthesised per output pixel instead.
SUBGRID_CELLS = 0.55
# Cross-layer correlation of the grain field by colour style. "film" keeps the
# asset's modelled value; "scan" is a declared convention (mostly luminance
# grain, as film looks after scanning/post), not a measurement.
GRAIN_CHROMA_STYLES = ("film", "scan")
SCAN_LAYER_CORR = 0.8


def split_bands(g: GrainProfile) -> tuple[tuple, tuple]:
    """(film-plane bands, per-pixel bands) of the asset's spectrum."""
    bands = g.bands or ((g.size_um, 1.0),)
    coarse = tuple(b for b in bands if b[0] / g.pitch_um > SUBGRID_CELLS)
    fine = tuple(b for b in bands if b[0] / g.pitch_um <= SUBGRID_CELLS)
    return coarse, fine


def master(g: GrainProfile, bands: tuple | None = None,
           layer_corr: float | None = None) -> tuple[np.ndarray, float]:
    """(integral image, 48 um aperture RMS) of a process-wide master field.
    ``bands`` defaults to all asset bands (the frozen realization)."""
    from dataclasses import replace

    gg = replace(g, bands=tuple(bands) if bands is not None else g.bands,
                 layer_corr=g.layer_corr if layer_corr is None else float(layer_corr))
    key = (gg.pitch_um, gg.size_um, gg.layer_corr, gg.bands, gg.aperture_um)
    got = _MASTER.get(key)
    if got is None:
        if len(_MASTER) >= 2:
            _MASTER.clear()
        ii = integral_image(band_limited_field(gg, 0))
        got = (ii, aperture_rms(ii, gg))
        _MASTER[key] = got
    return got


def _gauss_taps(sigma_px: float) -> np.ndarray:
    radius = max(int(np.ceil(3.0 * sigma_px)), 1)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_px) ** 2)
    return k / k.sum()


def _box_taps(n: float) -> np.ndarray:
    """Normalized 1-D box of (possibly fractional) width n pixels."""
    whole = int(np.floor(n))
    frac = n - whole
    taps = np.ones(whole + (1 if frac > 1e-9 else 0), dtype=np.float64)
    if frac > 1e-9:
        taps[-1] = frac
    return taps / max(n, 1e-9)


def pixel_band_rms48(sigma_px: float, aperture_px: float) -> float:
    """48 um aperture RMS of a unit-variance per-pixel band (white noise per
    pixel, Gaussian-smoothed by sigma_px). Separable, so exact in closed form:
    var = (sum (k*b)^2)^2 / (sum k^2)^2."""
    k = _gauss_taps(sigma_px) if sigma_px >= 0.3 else np.ones(1)
    b = _box_taps(max(aperture_px, 1.0))
    kb = np.convolve(k, b)
    return float(np.sum(kb ** 2) / np.sum(k ** 2))


def _row_noise(seed: int, row: int, width: int, corr: float) -> np.ndarray:
    """Unit-variance [width, 3] white noise of one output row, layers sharing
    a component with weight sqrt(corr). Keyed by (seed, row) only, so any
    chunking of the frame draws identical values."""
    # The row index goes in the counter's second word: each row owns a 2^64-block
    # stream. (Counter=row in the low word would make row r+1 a shifted copy of
    # row r, since one row consumes thousands of counter blocks.)
    rng = np.random.Generator(np.random.Philox(
        key=(int(seed) & 0xFFFFFFFF) | (0x5EED << 32),
        counter=np.array([0, int(row) & 0xFFFFFFFFFFFFFFFF, 0, 0], dtype=np.uint64)))
    white = rng.standard_normal((width, 3), dtype=np.float32)
    shared = rng.standard_normal((width, 1), dtype=np.float32)
    white *= np.float32(np.sqrt(1.0 - corr))
    white += np.float32(np.sqrt(corr)) * shared
    return white


def pixel_band(seed: int, y0: int, y1: int, height: int, width: int,
               sigma_px: float, corr: float) -> np.ndarray:
    """[y1-y0, width, 3] unit-variance per-pixel grain band: per-row white noise
    (rows mirrored at the frame edges) smoothed by a separable Gaussian."""
    if sigma_px < 0.3:
        return np.stack([_row_noise(seed, y, width, corr) for y in range(y0, y1)])
    k = _gauss_taps(sigma_px).astype(np.float32)
    r = (k.size - 1) // 2

    def mirror(y: int) -> int:
        if height == 1:
            return 0
        period = 2 * (height - 1)
        y = y % period
        return y if y < height else period - y

    rows = np.stack([_row_noise(seed, mirror(y), width, corr) for y in range(y0 - r, y1 + r)])
    # vertical pass (halo rows are real neighbours, so chunks agree exactly)
    acc = np.zeros((y1 - y0, width, 3), dtype=np.float32)
    for i in range(k.size):
        acc += k[i] * rows[i:i + (y1 - y0)]
    # horizontal pass, mirrored at the frame's left/right edges
    pad = np.pad(acc, ((0, 0), (r, r), (0, 0)), mode="reflect" if width > r else "edge")
    out = np.zeros_like(acc)
    for i in range(k.size):
        out += k[i] * pad[:, i:i + width]
    out /= np.float32(np.sum(k.astype(np.float64) ** 2))
    return out


def apply_density_grain(d: np.ndarray, field: np.ndarray, g: GrainProfile, amount: float,
                        rms48: float) -> np.ndarray:
    """D' = D + sigma(D)*amount/rms48*F + ln10/2*s^2 (mean-transmittance term)."""
    ln10 = np.float32(np.log(10.0))
    out = d.astype(np.float32, copy=True)
    for ch in range(3):
        base, _dmax = g.chart_density[ch]
        tab = np.asarray(g.sigma_density[ch], dtype=np.float64)
        s = d[:, ch].astype(np.float32) + np.float32(base)
        s = np.interp(s, tab[:, 0], tab[:, 1]).astype(np.float32)
        s *= np.float32(float(amount) / rms48)
        out[:, ch] += s * field[:, ch]
        np.square(s, out=s)
        s *= ln10 * np.float32(0.5)
        out[:, ch] += s
    return out


# --------------------------------------------------------------------------
# observe operator
# --------------------------------------------------------------------------

class StockGrain:
    """Per-stock tables for the observe grain: Stage A observer, the density
    characteristic and its per-layer inverse."""

    def __init__(self, stage_a: dict):
        self.observer = np.asarray(stage_a["observer"], dtype=np.float64)
        self.anchor_log10 = float(stage_a["anchor_ev"]) * LOG10_2
        self.le = np.asarray(stage_a["char_le"], dtype=np.float64)
        self.tab = np.asarray(stage_a["char_amounts"], dtype=np.float64)
        self.inv = []
        for c in range(3):
            dcol = self.tab[:, c]
            if dcol[-1] < dcol[0]:
                dcol, e = dcol[::-1], self.le[::-1]
            else:
                e = self.le
            keep = np.concatenate([[True], np.diff(dcol) > 1e-6])
            self.inv.append((dcol[keep], e[keep]))

    def log_exposure(self, rgb: np.ndarray) -> np.ndarray:
        """Neutral-anchored layer log exposure (frozen ``layer_log_exposure``):
        scene grey 0.18 maps to logE 0 on the profile axis, then + anchor."""
        mid = np.maximum(self.observer @ np.full(3, 0.18), 1e-12)
        e = np.asarray(rgb, dtype=np.float64) @ self.observer.T
        return np.log10(np.maximum(e, 1e-12) / mid[None, :]) + self.anchor_log10

    def density(self, log_e: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(log_e[:, c], self.le, self.tab[:, c]) for c in range(3)], axis=1)


def grain_field(start: int, n: int, height: int, width: int, g: GrainProfile,
                seed: int, chroma: str = "film") -> tuple[np.ndarray, float]:
    """(field[n, 3], 48 um RMS) for flat pixels [start, start+n) of a frame.

    Bands the film-plane grid can carry are area-sampled from the periodic
    master (phase by seed); finer bands are synthesised per output pixel at
    this rendering's own pitch. Each band is unit-variance and the mix keeps
    the asset's weights, so the total and its 48 um calibration follow the
    frozen definition at any resolution."""
    if chroma not in GRAIN_CHROMA_STYLES:
        raise ValueError(f"胶片颗粒颜色：未知 {chroma!r}（可选 {'/'.join(GRAIN_CHROMA_STYLES)}）")
    corr = g.layer_corr if chroma == "film" else SCAN_LAYER_CORR
    geometry = FilmGeometry.fit(height, width)
    y0, y1 = start // width, (start + n - 1) // width + 1
    off = start - y0 * width
    pitch_um = geometry.region()[2] * 1000.0 / max(width, 1)
    coarse, fine = split_bands(g)
    field = np.zeros(((y1 - y0) * width, 3), dtype=np.float32)
    weights_sq = 0.0
    rms_sq = 0.0
    if coarse:
        wc = float(np.sqrt(sum(w * w for _, w in coarse)))
        ii, r48 = master(g, coarse, corr)
        gh, gw = ii.shape[0] - 1, ii.shape[1] - 1
        field += wc * sample_field(ii, geometry.rows(y0, y1),
                                   realization_phases(seed, gh, gw)).reshape(-1, 3)
        weights_sq += wc * wc
        rms_sq += wc * wc * r48 * r48
    for i, (size_um, w) in enumerate(fine):
        sigma_px = size_um / pitch_um
        band = pixel_band(seed * 2654435761 + i + 1, y0, y1, height, width, sigma_px, corr)
        field += w * band.reshape(-1, 3)
        weights_sq += w * w
        rms_sq += w * w * pixel_band_rms48(sigma_px, g.aperture_um / pitch_um) ** 2
    norm = float(np.sqrt(weights_sq))
    field /= np.float32(norm)
    return field[off:off + n], float(np.sqrt(rms_sq) / norm)


def apply_observe_grain(rgb: np.ndarray, start: int, height: int, width: int,
                        stock: StockGrain, g: GrainProfile, amount: float, seed: int,
                        chroma: str = "film") -> np.ndarray:
    """Grain for flat scene pixels rgb[(N,3)] that start at flat index ``start``
    of a (height, width) frame."""
    if amount <= 0.0 or rgb.shape[0] == 0:
        return rgb
    n = rgb.shape[0]
    field, rms48 = grain_field(start, n, height, width, g, seed, chroma)
    log_e = stock.log_exposure(rgb)
    d0 = stock.density(log_e)
    d1 = apply_density_grain(d0, field, g, amount, rms48)
    out = np.array(rgb, dtype=np.float32, copy=True)
    for c in range(3):
        dd, ee = stock.inv[c]
        delta = np.interp(d1[:, c], dd, ee) - np.interp(d0[:, c], dd, ee)
        np.clip(delta, -MAX_DELTA_LOG10, MAX_DELTA_LOG10, out=delta)
        out[:, c] *= np.power(10.0, delta).astype(np.float32)
    return out
