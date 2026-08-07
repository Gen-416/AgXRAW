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

    The image covers the gate region [x0, x0+w_mm] × [y0, y0+h_mm], in IMAGE
    axes. `rotated` declares a portrait capture: the physical gate is turned
    90° in the camera, so image x runs along the gate's 24 mm side and image
    y along the 36 mm side (the stored field is one landscape grid; rotated
    geometries sample it transposed). Crops and preview scales share the ONE
    gate: a crop passes the sub-region it covers, a preview passes the same
    region with fewer pixels.

    Use `FilmGeometry.fit(height, width)` for whole-image renderings: it
    orients the gate to the image and letterboxes non-3:2 aspects CENTERED
    inside the gate, so every pixel maps to a real emulsion position (the
    review's measured failure: portrait and 4:3 images spilled past the
    24 mm side and whole row bands sampled a zero field).
    """

    height: int
    width: int
    x0_mm: float = 0.0
    y0_mm: float = 0.0
    w_mm: float = GATE_W_MM
    h_mm: float = 0.0  # 0 -> derived from the image aspect over w_mm
    rotated: bool = False

    @classmethod
    def fit(cls, height: int, width: int) -> "FilmGeometry":
        rotated = height > width
        gate_w, gate_h = (GATE_H_MM, GATE_W_MM) if rotated else (GATE_W_MM, GATE_H_MM)
        aspect = height / max(width, 1)
        if gate_w * aspect <= gate_h:
            w_mm = gate_w
            h_mm = gate_w * aspect
        else:
            h_mm = gate_h
            w_mm = gate_h / aspect
        return cls(
            height, width,
            x0_mm=(gate_w - w_mm) / 2.0,
            y0_mm=(gate_h - h_mm) / 2.0,
            w_mm=w_mm, h_mm=h_mm, rotated=rotated,
        )

    def rows(self, y0: int, y1: int) -> "FilmGeometry":
        """The sub-geometry for output rows [y0, y1) — same gate mapping."""
        _, base_y0, w_mm, h_mm = self.region()
        return FilmGeometry(
            y1 - y0, self.width,
            x0_mm=self.x0_mm,
            y0_mm=base_y0 + h_mm * y0 / max(self.height, 1),
            w_mm=w_mm,
            h_mm=h_mm * (y1 - y0) / max(self.height, 1),
            rotated=self.rotated,
        )

    def region(self) -> tuple[float, float, float, float]:
        h_mm = (
            self.h_mm if self.h_mm > 0
            else self.w_mm * self.height / max(self.width, 1)
        )
        return (self.x0_mm, self.y0_mm, self.w_mm, h_mm)


# Spread maps (halation / bloom) are DEFINED on a decimated grid: the source
# is area-decimated in the linear domain to at most SPREAD_MAX_DIM on the
# long side, the kernels run there, and the result upsamples bilinearly.
# This is the operator's contract, not an approximation of some other truth:
# both spreads are physically low-frequency, the full-frame oracle and the
# streamed row-band path share the one definition, so band seams are exact
# and no full-resolution convolution (or halo) ever exists (§9.3).
SPREAD_MAX_DIM = 2048


def spread_grid_shape(height: int, width: int) -> tuple[int, int]:
    long_side = max(height, width)
    if long_side <= SPREAD_MAX_DIM:
        return (height, width)
    scale = SPREAD_MAX_DIM / long_side
    return (max(int(round(height * scale)), 1), max(int(round(width * scale)), 1))


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
    img = np.asarray(img, dtype=np.float64)
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
    accumulator (see area_decimate). Deterministic in any band split."""
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


# --------------------------------------------------------------------------
# separable Gaussian (oracle-grade CPU path)
# --------------------------------------------------------------------------

def _sep_axis(padded: np.ndarray, kernel: np.ndarray, n: int, axis: int) -> np.ndarray:
    acc = np.zeros(
        (n,) + padded.shape[1:] if axis == 0
        else (padded.shape[0], n, padded.shape[2]),
        dtype=np.float32,
    )
    for i in range(kernel.size):
        acc += kernel[i] * (padded[i:i + n] if axis == 0 else padded[:, i:i + n])
    return acc.astype(np.float32)


def _gaussian_blur_slabbed(img: np.ndarray, sigma: float, slab: int = 256) -> np.ndarray:
    """Separable Gaussian for LARGE arrays, processed in halo slabs so the
    transient working set stays at slab scale (review batch 13: the whole-
    array pad/accumulate chain alone spent ~390 MiB on the grain grid).
    Reflect-padded, float32, identical taps to _gaussian_blur."""
    if sigma <= 0.0:
        return np.asarray(img, dtype=np.float32)
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    k32 = k.astype(np.float32)
    src = np.ascontiguousarray(img, dtype=np.float32)
    h, w = src.shape[:2]
    # Both passes write back IN PLACE per slab (the pad copy already holds
    # the slab's context), so the whole-array footprint stays at one buffer.
    for c0 in range(0, w, slab):     # vertical pass, column slabs
        c1 = min(c0 + slab, w)
        pad = np.pad(src[:, c0:c1], ((radius, radius), (0, 0), (0, 0)),
                     mode="reflect")
        acc = np.zeros((h, c1 - c0, src.shape[2]), dtype=np.float32)
        for i in range(k32.size):
            acc += k32[i] * pad[i:i + h]
        src[:, c0:c1] = acc
    for r0 in range(0, h, slab):     # horizontal pass, row slabs
        r1 = min(r0 + slab, h)
        pad = np.pad(src[r0:r1], ((0, 0), (radius, radius), (0, 0)),
                     mode="reflect")
        acc = np.zeros((r1 - r0, w, src.shape[2]), dtype=np.float32)
        for i in range(k32.size):
            acc += k32[i] * pad[:, i:i + w]
        src[r0:r1] = acc
    return src


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
    # In-place mixing and an einsum RMS: the naive expression chain held four
    # grid-sized temporaries at once and the float64 square another two —
    # the measured ~620 MiB build transient of review batch 13.
    white *= np.float32(np.sqrt(1.0 - c))
    for ch in range(3):
        white[..., ch] += np.float32(np.sqrt(c)) * shared[..., 0]
    del shared
    field = _gaussian_blur_slabbed(
        white, profile.grain_size_um / profile.grain_pitch_um
    )
    del white
    n = field.shape[0] * field.shape[1]
    rms = np.sqrt(
        np.einsum("hwc,hwc->c", field, field, dtype=np.float64) / n
    )
    field /= np.maximum(rms, 1e-12).astype(np.float32)
    return field


_FIELD_CACHE: dict[tuple, np.ndarray] = {}


def grain_field_for(profile: OpticsProfile, seed: int) -> np.ndarray:
    """The deterministic film-space field (kept for tests/inspection; the
    sampling path uses the cached integral image below)."""
    return _band_limited_field(profile, seed)


def _grain_ii_for(profile: OpticsProfile, seed: int) -> np.ndarray:
    """The field's 2-D integral image, built ONCE per (profile, seed) and
    cached INSTEAD of the field (review batch 13): rebuilding the ~144 MB
    float64 integral every row band dominated the measured peak, and the
    field itself is never needed after the integral exists. The old entry is
    released before the replacement is built."""
    key = (profile, int(seed))
    got = _FIELD_CACHE.get(key)
    if got is None:
        _FIELD_CACHE.clear()
        field = _band_limited_field(profile, seed)
        gh, gw = field.shape[:2]
        # float64 during accumulation (the running sums reach ~1e3 x cell
        # values), stored float32: sampling differences carry ~1e-4 relative
        # noise on the unit-RMS field — orders below the grain sigma it
        # modulates — and the resident cost halves to 72 MB, which is what
        # lets the 256 MiB tier exist at all (review batch 14).
        ii = np.zeros((gh + 1, gw + 1, field.shape[2]), dtype=np.float64)
        np.cumsum(field, axis=0, out=ii[1:, 1:])
        del field
        np.cumsum(ii[1:, 1:], axis=1, out=ii[1:, 1:])
        got = ii.astype(np.float32)
        del ii
        _FIELD_CACHE[key] = got
    return got


def _as_integral(arr: np.ndarray) -> np.ndarray:
    """Accept a raw field [gh,gw,c] or a prebuilt integral image
    [gh+1,gw+1,c] (recognized by the zero first row/column of the latter)."""
    if (
        arr.dtype in (np.float32, np.float64)
        and arr.shape[0] > 1 and arr.shape[1] > 1
        and not arr[0].any() and not arr[:, 0].any()
    ):
        return arr
    ii = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1, arr.shape[2]), dtype=np.float64)
    np.cumsum(arr, axis=0, out=ii[1:, 1:])
    np.cumsum(ii[1:, 1:], axis=1, out=ii[1:, 1:])
    return ii


def sample_field(field: np.ndarray, geometry: FilmGeometry) -> np.ndarray:
    """Area-integrated sampling of the film-space field onto the pixel grid.

    Each output pixel averages the field over its exact mm footprint via a
    2-D integral image (bilinear interpolation of the integral image is exact
    for cell-constant fields at fractional coordinates). A half-resolution
    preview therefore equals the block mean of the full-resolution sampling
    by construction, and a crop equals the corresponding region of the full
    frame — the §9.1 shared-coordinate contract. Rotated (portrait)
    geometries sample the one landscape grid transposed, so the same
    emulsion position keeps the same realization in either orientation.
    Output rows are processed in slabs so the query temporaries stay bounded
    (review batch 13: the unslabbed path peaked at several output-sized
    float64 arrays); the integral image itself (~144 MB at the default
    grain grid) is accounted in the renderer's budget.
    """
    ii = _as_integral(field)
    if geometry.rotated:
        # querying the SAME integral with swapped axes samples the
        # transposed field — no data movement
        ii = ii.transpose(1, 0, 2)
        gate_w_mm, gate_h_mm = GATE_H_MM, GATE_W_MM
    else:
        gate_w_mm, gate_h_mm = GATE_W_MM, GATE_H_MM
    gh, gw = ii.shape[0] - 1, ii.shape[1] - 1
    x0, y0, w_mm, h_mm = geometry.region()

    ye = np.clip(
        (y0 + h_mm * np.arange(geometry.height + 1) / geometry.height)
        / gate_h_mm * gh, 0.0, gh,
    )
    xe = np.clip(
        (x0 + w_mm * np.arange(geometry.width + 1) / geometry.width)
        / gate_w_mm * gw, 0.0, gw,
    )

    def _ii_at(yq: np.ndarray, xq: np.ndarray) -> np.ndarray:
        yi = np.clip(np.floor(yq).astype(int), 0, gh - 1)
        xi = np.clip(np.floor(xq).astype(int), 0, gw - 1)
        yf = (yq - yi)[:, None, None]
        xf = (xq - xi)[None, :, None]
        top = ii[yi][:, xi] * (1 - xf) + ii[yi][:, xi + 1] * xf
        bot = ii[yi + 1][:, xi] * (1 - xf) + ii[yi + 1][:, xi + 1] * xf
        return top * (1 - yf) + bot * yf

    out = np.empty((geometry.height, geometry.width, field.shape[2]), dtype=np.float32)
    area_x = np.maximum(xe[1:] - xe[:-1], 1e-12)[None, :, None]
    slab = max(1, 8_000_000 // max(geometry.width, 1))
    for r0 in range(0, geometry.height, slab):
        r1 = min(r0 + slab, geometry.height)
        s00 = _ii_at(ye[r0:r1], xe[:-1])
        s01 = _ii_at(ye[r0:r1], xe[1:])
        s10 = _ii_at(ye[r0 + 1:r1 + 1], xe[:-1])
        s11 = _ii_at(ye[r0 + 1:r1 + 1], xe[1:])
        area = (ye[r0 + 1:r1 + 1] - ye[r0:r1])[:, None, None] * area_x
        out[r0:r1] = (s11 - s10 - s01 + s00) / np.maximum(area, 1e-12)
    return out


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
    field = sample_field(_grain_ii_for(profile, seed), geometry)
    a = np.asarray(amounts, dtype=np.float64).reshape(h, w, 3)
    lo64 = np.asarray(lo, dtype=np.float64)
    span = np.maximum(np.asarray(hi, dtype=np.float64) - lo64, 1e-9)
    dn = np.clip((a - lo64) / span, 0.0, 1.0)
    sigma = float(amount) * profile.grain_sigma0 * 4.0 * dn * (1.0 - dn)
    return (a + sigma * span * field.astype(np.float64)).reshape(-1, 3)


def halation_spread_map(
    ev_y_dec: np.ndarray,
    full_width: int,
    geometry_w_mm: float,
    profile: OpticsProfile,
) -> np.ndarray:
    """Halation spread on the decimated grid (§9.2, spread-grid contract):
    source is the pre-emulsion highlight LINEAR scene exposure above the
    declared threshold (area-decimated in the linear domain upstream),
    spread by the exponential-tail kernel approximated as a Gaussian
    cascade. Returned map is (dh, dw, 1), luminance-exposure units."""
    dh, dw = ev_y_dec.shape[:2]
    src = np.maximum(
        np.asarray(ev_y_dec, dtype=np.float32)
        - np.float32(np.exp2(profile.halation_threshold_ev)),
        0.0,
    ).reshape(dh, dw, 1)
    px_per_mm_dec = dw / max(geometry_w_mm, 1e-9)
    r0_px = max(profile.halation_radius_mm * px_per_mm_dec, 0.5)
    spread = np.zeros_like(src)
    for scale, wgt in ((0.5, 0.55), (1.0, 0.30), (2.0, 0.15)):
        spread += wgt * _gaussian_blur(src, r0_px * scale)
    return spread


def halation_reinject_rows(
    log_e: np.ndarray,
    spread_map: np.ndarray,
    y0: int,
    y1: int,
    height: int,
    width: int,
    profile: OpticsProfile,
    amount: float,
) -> np.ndarray:
    """Reinject the upsampled spread into the LAYER EXPOSURE for output rows
    [y0, y1), in LINEAR exposure per layer with the red-heavy weights,
    before the characteristic curves."""
    if amount <= 0.0:
        return log_e
    spread = upsample_rows(spread_map, y0, y1, height, width)[..., 0]
    lin = np.power(10.0, np.asarray(log_e, dtype=np.float64).reshape(y1 - y0, width, 3))
    gain = float(amount) * profile.halation_strength
    for c, wc in enumerate(profile.halation_weights):
        lin[..., c] += gain * wc * spread.astype(np.float64)
    return np.log10(np.maximum(lin, 1e-12)).reshape(-1, 3)


def bloom_spread_map(developed_dec: np.ndarray, profile: OpticsProfile) -> np.ndarray:
    """Medium bloom spread on the decimated grid (§9.2): multi-scale
    low-frequency pyramid over the positive medium's own highlights
    (area-decimated developed image, display-linear). (dh, dw, 3)."""
    level = np.maximum(
        np.asarray(developed_dec, dtype=np.float32) - profile.bloom_threshold, 0.0
    )
    dh, dw = level.shape[:2]
    spread = np.zeros_like(level)
    total = 0.0
    for lvl in range(profile.bloom_levels):
        if min(level.shape[0], level.shape[1]) <= 1:
            # nothing left to spread at this scale (tiny inputs / deep grids)
            break
        # Odd edges are edge-padded to even BEFORE the 2x2 block mean: the
        # bare truncation dropped the last row/column each level, so a
        # highlight on the odd edge lost its bloom entirely and sizes like
        # 5x5 crashed the reshape (review batch 13).
        pad_h = level.shape[0] % 2
        pad_w = level.shape[1] % 2
        if pad_h or pad_w:
            level = np.pad(level, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        sh, sw = level.shape[0] // 2, level.shape[1] // 2
        level = level.reshape(sh, 2, sw, 2, 3).mean(axis=(1, 3))
        blurred = _gaussian_blur(level, 2.0)
        factor = 2 ** (lvl + 1)
        # slab-wise nearest-neighbour accumulate: the double np.repeat held
        # two dec-grid-sized copies per level (review batch 13)
        col_idx = np.minimum(np.arange(dw) // factor, blurred.shape[1] - 1)
        wgt = 1.0 / (lvl + 1.0)
        total += wgt
        for r0 in range(0, dh, 256):
            r1 = min(r0 + 256, dh)
            row_idx = np.minimum(
                np.arange(r0, r1) // factor, blurred.shape[0] - 1
            )
            spread[r0:r1] += wgt * blurred[row_idx][:, col_idx]
    return spread / max(total, 1e-9)


def bloom_apply_rows(
    display_linear: np.ndarray,
    spread_map: np.ndarray,
    y0: int,
    y1: int,
    height: int,
    width: int,
    profile: OpticsProfile,
    amount: float,
) -> np.ndarray:
    """Add the upsampled bloom spread to output rows [y0, y1) — after print
    formation, before delivery gamut fit."""
    if amount <= 0.0:
        return display_linear
    img = np.asarray(display_linear, dtype=np.float32).reshape(y1 - y0, width, 3)
    spread = upsample_rows(spread_map, y0, y1, height, width)
    return (img + float(amount) * profile.bloom_strength * spread).reshape(-1, 3)
