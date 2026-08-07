# SPDX-License-Identifier: GPL-3.0-or-later
"""film v2 analog optics (FILM_PRINT_RENDERING_PLAN §9): density grain,
halation reinjection and medium bloom.

All three are MODELLED profiles in this first version (profile + amount only,
per §9.2) — no measured grain spectra or scatter profiles exist yet, and the
report says so. The contracts that ARE hard here:

- The grain random field lives in NEGATIVE FILM COORDINATES (mm on a declared
  gate), not output pixels. A fixed seed generates one band-limited field on
  the film-space grid; every rendering — preview, crop, full export — samples
  that same field by AREA INTEGRATION over each pixel's footprint. The same
  emulsion position therefore carries the same grain realization at every
  scale; texture varies with sampling rate, statistics do not.
- Halation extracts from the pre-emulsion highlight scene exposure and
  reinjects into the LAYER EXPOSURE before the characteristic curves, through
  a red-dominant backscatter kernel. It never shares a blur with bloom.
- Medium bloom is the positive medium's intrinsic scatter, applied after
  print formation (B2 output) and before delivery gamut fit, as a multi-scale
  low-frequency pyramid.
- Amount 0 is a strict identity everywhere (the caller keeps the
  chunk-stream fast path; these functions are only entered when engaged).

This module is the CPU oracle (plan §9.2/§9.3): correct first, tiled and
accelerated in the P5b batch without changing any contract here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 135 full-frame gate. Other gate sizes (8/16/65 mm) become profile data when
# per-stock measured optics land; the coordinate CONTRACT does not change.
GATE_W_MM = 36.0
GATE_H_MM = 24.0


@dataclass(frozen=True)
class OpticsProfile:
    """Modelled analog-optics profile (provenance: modelled, first version)."""

    # grain: band-limited Gaussian field on the film grid
    grain_pitch_um: float = 12.0        # film-space sample pitch
    grain_size_um: float = 18.0         # band-limit (Gaussian sigma) in film space
    grain_sigma0: float = 0.055         # density RMS at the mid-density peak
    grain_layer_corr: float = 0.35      # cross-layer covariance
    # halation: red-sensitive backscatter, exponential-tail cascade
    halation_radius_mm: float = 0.55
    halation_weights: tuple = (1.0, 0.22, 0.06)   # R, G, B layer reinjection
    halation_threshold_ev: float = 1.5            # above mid-grey, scene EV
    halation_strength: float = 0.12               # energy fraction at amount 1
    # medium bloom: intrinsic scatter of the positive medium
    bloom_levels: int = 4
    bloom_threshold: float = 0.6                  # display-linear source floor
    bloom_strength: float = 0.22                  # mixed-back fraction at amount 1


MODELLED_DEFAULT = OpticsProfile()

OPTICS_PROFILES = {"modelled_default": MODELLED_DEFAULT}


@dataclass(frozen=True)
class FilmGeometry:
    """Affine from the full negative gate to this rendering's pixel grid.

    The image covers the gate region [x0, x0+w_mm] × [y0, y0+h_mm]. Crops and
    preview scales share the ONE gate: a crop passes the sub-region it covers,
    a preview passes the same region with fewer pixels.
    """

    height: int
    width: int
    x0_mm: float = 0.0
    y0_mm: float = 0.0
    w_mm: float = GATE_W_MM
    h_mm: float = 0.0  # 0 -> derived from the image aspect over w_mm

    def region(self) -> tuple[float, float, float, float]:
        h_mm = (
            self.h_mm if self.h_mm > 0
            else self.w_mm * self.height / max(self.width, 1)
        )
        return (self.x0_mm, self.y0_mm, self.w_mm, h_mm)


# --------------------------------------------------------------------------
# separable Gaussian (oracle-grade CPU path)
# --------------------------------------------------------------------------

def _sep_axis(padded: np.ndarray, kernel: np.ndarray, n: int, axis: int) -> np.ndarray:
    acc = np.zeros(
        (n,) + padded.shape[1:] if axis == 0
        else (padded.shape[0], n, padded.shape[2]),
        dtype=np.float64,
    )
    for i in range(kernel.size):
        acc += kernel[i] * (padded[i:i + n] if axis == 0 else padded[:, i:i + n])
    return acc.astype(np.float32)


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.asarray(img, dtype=np.float32)
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = np.asarray(img, dtype=np.float32)
    pad = np.pad(out, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
    out = _sep_axis(pad, k, img.shape[0], axis=0)
    pad = np.pad(out, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    return _sep_axis(pad, k, img.shape[1], axis=1)


# --------------------------------------------------------------------------
# deterministic film-space grain field
# --------------------------------------------------------------------------

def _film_grid_shape(profile: OpticsProfile) -> tuple[int, int]:
    pitch_mm = profile.grain_pitch_um * 1e-3
    return (int(round(GATE_H_MM / pitch_mm)), int(round(GATE_W_MM / pitch_mm)))


def _band_limited_field(profile: OpticsProfile, seed: int) -> np.ndarray:
    """[gh, gw, 3]: unit-RMS per layer, cross-layer correlation via a shared
    component, Gaussian band limit at the declared grain size. Philox keyed
    on the seed alone — no shape, tile order or thread count enters the
    stream, so the realization is reproducible by contract."""
    gh, gw = _film_grid_shape(profile)
    rng = np.random.Generator(np.random.Philox(key=int(seed) & 0xFFFFFFFF))
    white = rng.standard_normal((gh, gw, 3), dtype=np.float32)
    shared = rng.standard_normal((gh, gw, 1), dtype=np.float32)
    c = float(np.clip(profile.grain_layer_corr, 0.0, 1.0))
    mixed = np.sqrt(1.0 - c) * white + np.sqrt(c) * shared
    field = _gaussian_blur(mixed, profile.grain_size_um / profile.grain_pitch_um)
    rms = np.sqrt(np.mean(np.square(field, dtype=np.float64), axis=(0, 1)))
    return (field / np.maximum(rms, 1e-12)).astype(np.float32)


_FIELD_CACHE: dict[tuple, np.ndarray] = {}


def grain_field_for(profile: OpticsProfile, seed: int) -> np.ndarray:
    key = (profile, int(seed))
    got = _FIELD_CACHE.get(key)
    if got is None:
        got = _band_limited_field(profile, seed)
        _FIELD_CACHE.clear()  # one field resident at a time (the grid is large)
        _FIELD_CACHE[key] = got
    return got


def sample_field(field: np.ndarray, geometry: FilmGeometry) -> np.ndarray:
    """Area-integrated sampling of the film-space field onto the pixel grid.

    Each output pixel averages the field over its exact mm footprint via a
    2-D integral image (bilinear interpolation of the integral image is exact
    for cell-constant fields at fractional coordinates). A half-resolution
    preview therefore equals the block mean of the full-resolution sampling
    by construction, and a crop equals the corresponding region of the full
    frame — the §9.1 shared-coordinate contract.
    """
    gh, gw = field.shape[:2]
    x0, y0, w_mm, h_mm = geometry.region()
    ii = np.zeros((gh + 1, gw + 1, field.shape[2]), dtype=np.float64)
    np.cumsum(field, axis=0, out=ii[1:, 1:])
    np.cumsum(ii[1:, 1:], axis=1, out=ii[1:, 1:])

    ye = np.clip(
        (y0 + h_mm * np.arange(geometry.height + 1) / geometry.height)
        / GATE_H_MM * gh, 0.0, gh,
    )
    xe = np.clip(
        (x0 + w_mm * np.arange(geometry.width + 1) / geometry.width)
        / GATE_W_MM * gw, 0.0, gw,
    )

    def _ii_at(yq: np.ndarray, xq: np.ndarray) -> np.ndarray:
        yi = np.clip(np.floor(yq).astype(int), 0, gh - 1)
        xi = np.clip(np.floor(xq).astype(int), 0, gw - 1)
        yf = (yq - yi)[:, None, None]
        xf = (xq - xi)[None, :, None]
        top = ii[yi][:, xi] * (1 - xf) + ii[yi][:, xi + 1] * xf
        bot = ii[yi + 1][:, xi] * (1 - xf) + ii[yi + 1][:, xi + 1] * xf
        return top * (1 - yf) + bot * yf

    s = (
        _ii_at(ye[1:], xe[1:]) - _ii_at(ye[1:], xe[:-1])
        - _ii_at(ye[:-1], xe[1:]) + _ii_at(ye[:-1], xe[:-1])
    )
    area = np.maximum(
        (ye[1:] - ye[:-1])[:, None, None] * (xe[1:] - xe[:-1])[None, :, None],
        1e-12,
    )
    return (s / area).astype(np.float32)


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------

def apply_density_grain(
    amounts: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    geometry: FilmGeometry,
    profile: OpticsProfile,
    amount: float,
    seed: int,
) -> np.ndarray:
    """D' = D + sigma_c(D) * N(x,y) (plan §9.1): the field modulates DENSITY
    before printing, so grain participates in colour and local contrast
    downstream. The RMS follows the classic mid-density peak
    sigma0 * 4*Dn*(1-Dn) — zero at film base and at Dmax, no extra knobs."""
    if amount <= 0.0:
        return amounts
    h, w = geometry.height, geometry.width
    field = sample_field(grain_field_for(profile, seed), geometry)
    a = np.asarray(amounts, dtype=np.float64).reshape(h, w, 3)
    lo64 = np.asarray(lo, dtype=np.float64)
    span = np.maximum(np.asarray(hi, dtype=np.float64) - lo64, 1e-9)
    dn = np.clip((a - lo64) / span, 0.0, 1.0)
    sigma = float(amount) * profile.grain_sigma0 * 4.0 * dn * (1.0 - dn)
    return (a + sigma * span * field.astype(np.float64)).reshape(-1, 3)


def halation_reinject(
    log_e: np.ndarray,
    scene_ev_y: np.ndarray,
    geometry: FilmGeometry,
    profile: OpticsProfile,
    amount: float,
) -> np.ndarray:
    """Red-dominant backscatter into the LAYER EXPOSURE (plan §9.2): source
    is the pre-emulsion highlight scene exposure above the declared
    threshold, spread by an exponential-tail kernel (approximated by a short
    Gaussian cascade in the oracle), added in LINEAR exposure per layer with
    the red-heavy weights, before the characteristic curves."""
    if amount <= 0.0:
        return log_e
    h, w = geometry.height, geometry.width
    _, _, w_mm, _ = geometry.region()
    px_per_mm = geometry.width / max(w_mm, 1e-9)
    src = np.maximum(
        np.exp2(np.asarray(scene_ev_y, dtype=np.float64).reshape(h, w))
        - float(np.exp2(profile.halation_threshold_ev)),
        0.0,
    )[..., None].astype(np.float32)
    r0_px = max(profile.halation_radius_mm * px_per_mm, 0.5)
    spread = np.zeros_like(src)
    for scale, wgt in ((0.5, 0.55), (1.0, 0.30), (2.0, 0.15)):
        spread += wgt * _gaussian_blur(src, r0_px * scale)
    lin = np.power(10.0, np.asarray(log_e, dtype=np.float64).reshape(h, w, 3))
    gain = float(amount) * profile.halation_strength
    for c, wc in enumerate(profile.halation_weights):
        lin[..., c] += gain * wc * spread[..., 0].astype(np.float64)
    return np.log10(np.maximum(lin, 1e-12)).reshape(-1, 3)


def medium_bloom(
    display_linear: np.ndarray,
    geometry: FilmGeometry,
    profile: OpticsProfile,
    amount: float,
) -> np.ndarray:
    """Intrinsic scatter of the positive medium (plan §9.2): multi-scale
    low-frequency spread of the print's own highlights, after print
    formation and before delivery gamut fit."""
    if amount <= 0.0:
        return display_linear
    h, w = geometry.height, geometry.width
    img = np.asarray(display_linear, dtype=np.float32).reshape(h, w, 3)
    level = np.maximum(img - profile.bloom_threshold, 0.0)
    spread = np.zeros_like(img)
    total = 0.0
    for lvl in range(profile.bloom_levels):
        sh, sw = max(level.shape[0] // 2, 1), max(level.shape[1] // 2, 1)
        level = level[: sh * 2, : sw * 2].reshape(sh, 2, sw, 2, 3).mean(axis=(1, 3))
        blurred = _gaussian_blur(level, 2.0)
        factor = 2 ** (lvl + 1)
        up = np.repeat(np.repeat(blurred, factor, axis=0), factor, axis=1)
        if up.shape[0] < h or up.shape[1] < w:
            up = np.pad(
                up,
                ((0, h - min(up.shape[0], h)), (0, w - min(up.shape[1], w)), (0, 0)),
                mode="edge",
            )
        wgt = 1.0 / (lvl + 1.0)
        total += wgt
        spread += wgt * up[:h, :w]
    spread /= max(total, 1e-9)
    return (img + float(amount) * profile.bloom_strength * spread).reshape(-1, 3)
