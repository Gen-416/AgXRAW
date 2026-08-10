# SPDX-License-Identifier: GPL-3.0-or-later
"""film v2 analog optics (FILM_PRINT_RENDERING_PLAN §9): density grain,
halation reinjection and print-medium scatter.

Each operator now takes the SPECIFIC asset it implements — a GrainAsset, a
HalationAsset, a PrintScatterAsset — rather than one shared profile struct
(FILM_OPTICS_V2 §7.1, phase P1). The old single `OpticsProfile` made it easy
to read a print-medium constant as a film property and to quote a modelled
halo radius as if the whole profile were measured; the assets carry their own
provenance, and a function that only needs grain cannot see halation at all.

The contracts that ARE hard here:

- The grain random field lives in NEGATIVE FILM COORDINATES (mm on a declared
  gate), not output pixels. A fixed seed generates one band-limited field on
  the film-space grid; every rendering — preview, crop, full export — samples
  that same field by AREA INTEGRATION over each pixel's footprint. The same
  emulsion position therefore carries the same grain realization at every
  scale; texture varies with sampling rate, statistics do not.
- Halation extracts from the pre-emulsion highlight scene exposure and
  reinjects into the LAYER EXPOSURE before the characteristic curves, through
  a red-dominant backscatter kernel. It never shares a blur with bloom.
- Print-medium scatter (the control the GUI still labels "Bloom") is the
  positive medium's intrinsic scatter, applied after print formation (B2
  output) and before delivery gamut fit, as a multi-scale low-frequency
  pyramid.
- Amount 0 is a strict identity everywhere (the caller keeps the
  chunk-stream fast path; these functions are only entered when engaged).

This module is the CPU oracle (plan §9.2/§9.3): correct first, tiled and
accelerated in the P5b batch without changing any contract here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .film_optics_assets import GrainAsset, HalationAsset, PrintScatterAsset

# 135 full-frame gate. Other gate sizes (8/16/65 mm) become profile data when
# per-stock measured optics land; the coordinate CONTRACT does not change.
GATE_W_MM = 36.0
GATE_H_MM = 24.0


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

# The memory tier now GOVERNS the spread grid instead of only sizing bands.
# P3 put a second spread operator on this grid, and two operators' maps plus
# their construction temporaries do not fit a 512 MiB budget at 2048 — CI's
# independent-process gate measured 810 MiB against a 608 MiB allowance.
# Shaving buffers got it to ~620 and no further, so the choice was between
# advertising a tier the implementation cannot honour (which review batch 19
# explicitly refused) and letting the tier mean something. It now means
# something: 512 MiB renders the spread on a 1408-long grid, 1024 on 2048.
#
# What that costs: at 1408 on a 36 mm gate a cell is 25.6 um against 17.6.
# Halation's tightest component (65 um) is still 2.5 cells across and the
# capture bloom's finest diffusion (40 um) 1.6 — both above the Nyquist of
# their own kernels, which is why this is a resolution trade and not a
# different operator.
_SPREAD_DIM_BY_TIER = {512: 1408, 1024: SPREAD_MAX_DIM}


def spread_max_dim() -> int:
    from .render import _optics_budget_mib

    return _SPREAD_DIM_BY_TIER.get(_optics_budget_mib(), 1408)


def spread_grid_shape(height: int, width: int) -> tuple[int, int]:
    long_side = max(height, width)
    limit = spread_max_dim()
    if long_side <= limit:
        return (height, width)
    scale = limit / long_side
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


def _gaussian_blur_slabbed(
    img: np.ndarray, sigma: float, slab: int = 256, periodic: bool = False
) -> np.ndarray:
    """Separable Gaussian for LARGE arrays, processed in halo slabs so the
    transient working set stays at slab scale (review batch 13). float32,
    identical taps to _gaussian_blur. periodic=True wrap-pads both passes —
    the master grain field must be GENUINELY periodic for the phase
    realizations: reflect padding left a ~20-sigma first-difference seam at
    the wrap line (review batch 16)."""
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
    edge_mode = "wrap" if periodic else "reflect"
    for c0 in range(0, w, slab):     # vertical pass, column slabs
        c1 = min(c0 + slab, w)
        pad = np.pad(src[:, c0:c1], ((radius, radius), (0, 0), (0, 0)),
                     mode=edge_mode)
        acc = np.zeros((h, c1 - c0, src.shape[2]), dtype=np.float32)
        for i in range(k32.size):
            acc += k32[i] * pad[i:i + h]
        src[:, c0:c1] = acc
    for r0 in range(0, h, slab):     # horizontal pass, row slabs
        r1 = min(r0 + slab, h)
        pad = np.pad(src[r0:r1], ((0, 0), (radius, radius), (0, 0)),
                     mode=edge_mode)
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

def _blur_bounded(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian with a SLAB-bounded working set, out of place.

    `_gaussian_blur` holds a padded copy plus a full accumulator for each
    pass, so a spread grid's worth of decimated RGB costs two extra full
    buffers per call — and P3's two spread operators make eighteen such calls
    between them. CI's independent-process RSS gate measured 810 MiB against
    a 608 MiB allowance from exactly that; the slabbed kernel keeps the
    transient at slab scale instead. The copy is explicit because the slabbed
    routine works in place and `ascontiguousarray` hands back the argument
    itself when it is already float32 and contiguous.
    """
    return _gaussian_blur_slabbed(np.array(img, dtype=np.float32, copy=True), sigma)


def _film_grid_shape(grain: GrainAsset) -> tuple[int, int]:
    pitch_mm = grain.pitch_um * 1e-3
    return (int(round(GATE_H_MM / pitch_mm)), int(round(GATE_W_MM / pitch_mm)))


def _band_limited_field(grain: GrainAsset, seed: int) -> np.ndarray:
    """[gh, gw, 3]: unit-RMS per layer, cross-layer correlation via a shared
    component, Gaussian band limit at the declared grain size. Philox keyed
    on the seed alone — no shape, tile order or thread count enters the
    stream, so the realization is reproducible by contract."""
    gh, gw = _film_grid_shape(grain)
    rng = np.random.Generator(np.random.Philox(key=int(seed) & 0xFFFFFFFF))
    white = rng.standard_normal((gh, gw, 3), dtype=np.float32)
    shared = rng.standard_normal((gh, gw, 1), dtype=np.float32)
    c = float(np.clip(grain.layer_corr, 0.0, 1.0))
    # In-place mixing and an einsum RMS: the naive expression chain held four
    # grid-sized temporaries at once and the float64 square another two —
    # the measured ~620 MiB build transient of review batch 13.
    white *= np.float32(np.sqrt(1.0 - c))
    for ch in range(3):
        white[..., ch] += np.float32(np.sqrt(c)) * shared[..., 0]
    del shared
    field = _gaussian_blur_slabbed(
        white, grain.size_um / grain.pitch_um, periodic=True
    )
    del white
    n = field.shape[0] * field.shape[1]
    rms = np.sqrt(
        np.einsum("hwc,hwc->c", field, field, dtype=np.float64) / n
    )
    field /= np.maximum(rms, 1e-12).astype(np.float32)
    return field


_FIELD_CACHE: dict[tuple, np.ndarray] = {}


# The ONE master realization per profile (P2, review batch 15): the
# expensive band-limited field and its integral image are built for
# MASTER_SEED only and reused process-wide; per-RAW "randomness" is a cheap
# spatial PHASE on the periodic master (SplitMix64 below) — it changes the
# grain ARRANGEMENT a photo sees, never the size, spectrum, density
# response or cross-layer covariance. MASTER_SEED = 0 keeps the historical
# seed-0 output byte-identical (phase (0, 0)).
MASTER_SEED = 0


def _splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def realization_phases(seed: int, gh: int, gw: int) -> tuple[int, int]:
    """Per-RAW spatial phase on the periodic master grid. Seed 0 is the
    master realization itself (historical output); any other seed mixes
    through SplitMix64 into integer cell offsets. Creation is O(1)."""
    seed = int(seed)
    if seed == 0:
        return (0, 0)
    a = _splitmix64(seed)
    b = _splitmix64(a)
    return (a % max(gh, 1), b % max(gw, 1))


def grain_field_for(grain: GrainAsset, seed: int) -> np.ndarray:
    """The deterministic film-space field (kept for tests/inspection; the
    sampling path uses the cached MASTER integral image below)."""
    return _band_limited_field(grain, seed)


def _grain_ii_for(grain: GrainAsset, seed: int) -> np.ndarray:
    """The field's 2-D integral image, built ONCE per (profile, seed) and
    cached INSTEAD of the field (review batch 13): rebuilding the ~144 MB
    float64 integral every row band dominated the measured peak, and the
    field itself is never needed after the integral exists. The old entry is
    released before the replacement is built."""
    key = (grain, int(seed))
    got = _FIELD_CACHE.get(key)
    if got is None:
        _FIELD_CACHE.clear()
        field = _band_limited_field(grain, seed)
        gh, gw = field.shape[:2]
        # float64 during accumulation (the running sums reach ~1e3 x cell
        # values), stored float32: sampling differences carry ~1e-4 relative
        # noise on the unit-RMS field — orders below the grain sigma it
        # modulates. The accumulation runs in COLUMN SLABS with a float64
        # carry, written straight into the float32 store: the whole-grid
        # float64 intermediate plus its astype copy peaked ~290 MB and sank
        # the 256 MiB tier on CI (review batch 14).
        ii32 = np.zeros((gh + 1, gw + 1, field.shape[2]), dtype=np.float32)
        carry = np.zeros((gh + 1, 1, field.shape[2]), dtype=np.float64)
        slab = 256
        for c0 in range(0, gw, slab):
            c1 = min(c0 + slab, gw)
            block = field[:, c0:c1].astype(np.float64)
            np.cumsum(block, axis=0, out=block)
            np.cumsum(block, axis=1, out=block)
            block += carry[1:]
            ii32[1:, c0 + 1:c1 + 1] = block
            carry[1:] = block[:, -1:]
            del block
        del field
        _FIELD_CACHE[key] = ii32
        got = ii32
    return got


def integral_from_field(field: np.ndarray) -> np.ndarray:
    """Build the 2-D summed-area table of a field: [h,w,c] -> [h+1,w+1,c].

    EXPLICIT by contract (review batch 19): the previous helper guessed
    whether its argument was already an integral image by testing for a zero
    first row and column. A localized light source in a large frame leaves
    exactly that pattern in its spread map, so a plain map was taken for an
    integral and the bloom energy silently vanished (measured: 0.087 per
    channel lost at 640x960 and 2160x3840, with the neighbourhood gaining
    nothing, while the 64x96 test frame happened to pass). Never infer a
    data type from pixel content — callers say which one they hold.
    """
    arr = np.asarray(field)
    ii = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1, arr.shape[2]), dtype=np.float64)
    np.cumsum(arr, axis=0, out=ii[1:, 1:])
    np.cumsum(ii[1:, 1:], axis=1, out=ii[1:, 1:])
    return ii


def sample_field(
    field: np.ndarray,
    geometry: FilmGeometry,
    phase: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Area-integrated sampling of the film-space field onto the pixel grid.

    Each output pixel averages the field over its exact mm footprint via a
    2-D integral image (bilinear interpolation of the integral image is exact
    for cell-constant fields at fractional coordinates). A half-resolution
    preview therefore equals the block mean of the full-resolution sampling
    by construction, and a crop equals the corresponding region of the full
    frame — the §9.1 shared-coordinate contract. Rotated (portrait)
    geometries sample the one landscape grid transposed.

    `phase` shifts the query onto the PERIODIC extension of the master grid
    (P2, review batch 15): the summed-area table is extended analytically —
    Ĩ(y, x) = I(y%G, x%G) + ky·I(G, x%G) + kx·I(y%G, G) + ky·kx·I(G, G) —
    so no np.roll copy of the field ever exists, wrap seams cancel exactly,
    and phase (0, 0) takes the original single-lookup path unchanged.
    Output rows are processed in slabs so query temporaries stay bounded.
    """
    # `field` IS the integral image (grain callers pass the cached master;
    # tests pass integral_from_field(raw_field)) — no content sniffing.
    ii = np.asarray(field)
    if geometry.rotated:
        ii = ii.transpose(1, 0, 2)
        gate_w_mm, gate_h_mm = GATE_H_MM, GATE_W_MM
        phase = (phase[1], phase[0])
    else:
        gate_w_mm, gate_h_mm = GATE_W_MM, GATE_H_MM
    gh, gw = ii.shape[0] - 1, ii.shape[1] - 1
    x0, y0, w_mm, h_mm = geometry.region()

    ye = np.clip(
        (y0 + h_mm * np.arange(geometry.height + 1) / geometry.height)
        / gate_h_mm * gh, 0.0, gh,
    ) + float(phase[0])
    xe = np.clip(
        (x0 + w_mm * np.arange(geometry.width + 1) / geometry.width)
        / gate_w_mm * gw, 0.0, gw,
    ) + float(phase[1])

    def _ii_at(yq: np.ndarray, xq: np.ndarray) -> np.ndarray:
        yi = np.clip(np.floor(yq).astype(int), 0, gh - 1)
        xi = np.clip(np.floor(xq).astype(int), 0, gw - 1)
        yf = (yq - yi)[:, None, None]
        xf = (xq - xi)[None, :, None]
        top = ii[yi][:, xi] * (1 - xf) + ii[yi][:, xi + 1] * xf
        bot = ii[yi + 1][:, xi] * (1 - xf) + ii[yi + 1][:, xi + 1] * xf
        return top * (1 - yf) + bot * yf

    # Periodic phase (review batch 16): for any pixel whose footprint does
    # NOT straddle a wrap line, the analytic periodic extension's ky/kx/total
    # terms CANCEL in the four-corner rectangle difference — so simply
    # mapping every edge by mod G and using the original single-lookup query
    # is EXACT for those pixels, at zero extra cost. Only the (at most one)
    # straddling pixel row and column need the explicit two-piece periodic
    # sum, computed on thin strips afterwards. The master is genuinely
    # periodic (wrap-blurred), so the wrap line carries no seam.
    if phase == (0, 0):
        ye_q, xe_q = ye, xe
        straddle_row = straddle_col = None
        periodic_at = None
    else:
        ye_q = ye - gh * np.floor(ye / gh)
        xe_q = xe - gw * np.floor(xe / gw)

        def _straddler(edges: np.ndarray, G: int):
            over = np.nonzero(
                (edges[:-1] - G * np.floor(edges[:-1] / G))
                > (edges[1:] - G * np.floor(edges[1:] / G))
            )[0]
            return int(over[0]) if over.size else None

        straddle_row = _straddler(ye, gh)
        straddle_col = _straddler(xe, gw)
        total = ii[gh, gw].astype(np.float64)
        row_tot = ii[gh]
        col_tot = ii[:, gw]

        def _interp1(table: np.ndarray, q: np.ndarray, n: int) -> np.ndarray:
            qi = np.clip(np.floor(q).astype(int), 0, n - 1)
            qf = (q - qi)[:, None]
            return table[qi] * (1 - qf) + table[qi + 1] * qf

        def periodic_at(yq: np.ndarray, xq: np.ndarray) -> np.ndarray:
            ky = np.floor(yq / gh).astype(np.float64)
            kx = np.floor(xq / gw).astype(np.float64)
            ry = yq - ky * gh
            rx = xq - kx * gw
            base = _ii_at(ry, rx)
            wrap_y = _interp1(row_tot, rx, gw)[None, :, :]
            wrap_x = _interp1(col_tot, ry, gh)[:, None, :]
            return (
                base
                + ky[:, None, None] * wrap_y
                + kx[None, :, None] * wrap_x
                + (ky[:, None, None] * kx[None, :, None]) * total[None, None, :]
            )

    out = np.empty((geometry.height, geometry.width, ii.shape[2]), dtype=np.float32)

    def _fill(rows: slice, yq_e: np.ndarray, ya_e: np.ndarray,
              cols: slice, xq_e: np.ndarray, xa_e: np.ndarray, query) -> None:
        n_rows = rows.stop - rows.start
        n_cols = cols.stop - cols.start
        if n_rows <= 0 or n_cols <= 0:
            return
        area_x = np.maximum(xa_e[1:] - xa_e[:-1], 1e-12)[None, :, None]
        slab = max(1, 8_000_000 // max(n_cols, 1))
        for r0 in range(0, n_rows, slab):
            r1 = min(r0 + slab, n_rows)
            s00 = query(yq_e[r0:r1], xq_e[:-1])
            s01 = query(yq_e[r0:r1], xq_e[1:])
            s10 = query(yq_e[r0 + 1:r1 + 1], xq_e[:-1])
            s11 = query(yq_e[r0 + 1:r1 + 1], xq_e[1:])
            area = (ya_e[r0 + 1:r1 + 1] - ya_e[r0:r1])[:, None, None] * area_x
            out[rows.start + r0:rows.start + r1, cols] = (
                (s11 - s10 - s01 + s00) / np.maximum(area, 1e-12)
            )

    _fill(slice(0, geometry.height), ye_q, ye,
          slice(0, geometry.width), xe_q, xe, _ii_at)
    if periodic_at is not None:
        ph_ye = ye - gh * np.floor(ye[:1] / gh)  # shift into [0, 2G) once
        ph_xe = xe - gw * np.floor(xe[:1] / gw)
        if straddle_row is not None:
            _fill(slice(straddle_row, straddle_row + 1),
                  ph_ye[straddle_row:straddle_row + 2], ye[straddle_row:straddle_row + 2],
                  slice(0, geometry.width), ph_xe, xe, periodic_at)
        if straddle_col is not None:
            _fill(slice(0, geometry.height), ph_ye, ye,
                  slice(straddle_col, straddle_col + 1),
                  ph_xe[straddle_col:straddle_col + 2], xe[straddle_col:straddle_col + 2],
                  periodic_at)
    return out


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------

def apply_density_grain(
    amounts: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    geometry: FilmGeometry,
    grain: GrainAsset,
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
    master = _grain_ii_for(grain, MASTER_SEED)
    gh, gw = master.shape[0] - 1, master.shape[1] - 1
    field = sample_field(
        master, geometry, phase=realization_phases(seed, gh, gw)
    )
    a = np.asarray(amounts, dtype=np.float64).reshape(h, w, 3)
    lo64 = np.asarray(lo, dtype=np.float64)
    span = np.maximum(np.asarray(hi, dtype=np.float64) - lo64, 1e-9)
    dn = np.clip((a - lo64) / span, 0.0, 1.0)
    sigma = float(amount) * grain.sigma0 * 4.0 * dn * (1.0 - dn)
    return (a + sigma * span * field.astype(np.float64)).reshape(-1, 3)


def halation_layer_gate(
    e_lin: np.ndarray, e_ref: np.ndarray, gate_ev: np.ndarray
) -> np.ndarray:
    """Per-layer C1 source gate on LINEAR layer exposure.

    R1 §5.2. The old gate collapsed the scene to one photometric luminance
    and thresholded that, so a saturated blue LED — low Y, enormous blue-layer
    exposure — could not produce halation at all. Each layer now opens on its
    own exposure relative to that layer's own 18% reference, and the transfer
    matrix decides what colour comes back. Whether a source triggers belongs
    to the layers; what colour returns belongs to the matrix; one luminance
    scalar cannot do both jobs.

    smootherstep, not a step: §10.1 gate 4 requires the source gate to be at
    least C1, and a hard threshold draws a visible contour around the onset.
    """
    e = np.asarray(e_lin, dtype=np.float32)
    ref = np.asarray(e_ref, dtype=np.float32).reshape(1, 1, 3)
    gate = np.asarray(gate_ev, dtype=np.float32)
    t0 = gate[None, None, :, 0]
    span = np.maximum(gate[:, 1] - gate[:, 0], 1e-6)[None, None, :]
    # Evaluated IN PLACE. The obvious expression chain — log2, subtract,
    # divide, clip, then t*t*t*(t*(t*6-15)+10) — holds six full grids at once,
    # and this runs once per component in both the map and the pointwise term.
    # CI's RSS gate measured 810 MiB against a 608 MiB allowance largely here.
    t = np.maximum(e, 1e-20)
    t /= np.maximum(ref, 1e-20)
    np.log2(t, out=t)
    t -= t0
    t /= span
    np.clip(t, 0.0, 1.0, out=t)
    poly = t * np.float32(6.0)
    poly -= np.float32(15.0)
    poly *= t
    poly += np.float32(10.0)
    poly *= t
    poly *= t
    poly *= t
    return poly


def halation_pointwise_return(
    e_lin: np.ndarray, e_ref: np.ndarray, halation: HalationAsset
) -> np.ndarray:
    """Sum over components of A_i @ U_i, evaluated POINTWISE.

    This is the DC term of the residual form, and it is also what the apply
    side subtracts at full resolution. Evaluating it pointwise rather than
    from the decimated proxy is the same lesson review batch 16 learned on
    bloom: a bright point inside a large decimated cell contributes a real
    source at full resolution that the proxy never saw, and subtracting the
    proxy's version instead steals light from pixels that never had any.
    """
    e = np.asarray(e_lin, dtype=np.float32)
    out = np.zeros(e.shape[:-1] + (3,), dtype=np.float32)
    for comp in halation.components:
        u = halation_layer_gate(e, e_ref, comp.gate_ev)
        u *= e
        out += np.einsum("cj,...j->...c", comp.transfer.astype(np.float32), u)
        del u
    return out


def halation_spread_map(
    e_lin_dec: np.ndarray,
    e_ref: np.ndarray,
    geometry_w_mm: float,
    halation: HalationAsset,
) -> np.ndarray:
    """Per-component spread of the gated LAYER exposure, on the decimated grid.

    Returns (dh, dw, 3) in linear layer-exposure units: the sum over
    components of A_i @ (K_i * U_i). The apply side subtracts its own
    pointwise A @ U, so this map is only ever the SPREAD half of the residual
    — see halation_reinject_rows.

    Each component carries its own radius and its own non-negative layer
    transfer matrix. That is what produces a warm inner ring and a red outer
    one from a white source: the tight component returns red AND green, the
    wide one returns almost only red. A single radius with a fixed weight
    vector can only make the same colour at every distance.
    """
    e = np.asarray(e_lin_dec, dtype=np.float32)
    dh, dw = e.shape[:2]
    px_per_mm = dw / max(geometry_w_mm, 1e-9)
    out = np.zeros((dh, dw, 3), dtype=np.float32)
    for comp in halation.components:
        u = halation_layer_gate(e, e_ref, comp.gate_ev)
        u *= e
        r0 = max(comp.radius_mm * px_per_mm, 0.35)
        # Transfer FIRST, then blur. The same kernel is applied to every
        # channel, so the channel mix and the convolution commute — and doing
        # the mix on the source costs one buffer instead of one per scale.
        u = (u.reshape(-1, 3) @ comp.transfer.astype(np.float32).T).reshape(
            dh, dw, 3
        )
        # Exponential-tail cascade approximated by three Gaussians whose
        # weights sum to 1, so a uniform source spreads to itself and the
        # residual is exactly zero on a flat field.
        for scale, wgt in ((0.5, 0.55), (1.0, 0.30), (2.0, 0.15)):
            blurred = _blur_bounded(u, r0 * scale)
            blurred *= np.float32(wgt)
            out += blurred
            del blurred
        del u
    return out


def halation_reinject_rows(
    log_e: np.ndarray,
    spread_map: np.ndarray,
    e_ref: np.ndarray,
    y0: int,
    y1: int,
    height: int,
    width: int,
    halation: HalationAsset,
    amount: float,
) -> np.ndarray:
    """Reinject the SPATIAL RESIDUAL into layer exposure for rows [y0, y1).

        E' = E + amount * (upsample(spread) - A @ U(E))

    R1 §5.3, and the reason is calibration, not taste. The characteristic
    curves come from sensitometric exposure of large uniform patches, where
    the light scattered out of a patch is replaced by light scattered in from
    its neighbours. That DC gain is already inside the curve. The additive
    form added it a second time — measured at +0.95% frame-wide red energy in
    the P0 baseline, which is a warm cast and a general veiling, not a halo.

    Under the residual form a uniform field is an exact identity (the spread
    of a constant is that constant), so only contrast boundaries reinject.

    The subtraction is capped at the pixel's own exposure per layer, which
    makes non-negativity structural rather than a clamp applied afterwards:
    a highlight core can give away all of its light but not more.
    """
    if amount <= 0.0:
        return log_e
    n = y1 - y0
    lin = np.power(
        10.0, np.asarray(log_e, dtype=np.float64).reshape(n, width, 3)
    ).astype(np.float32)
    if halation.dc_mode == "residual":
        give = halation_pointwise_return(lin, e_ref, halation)
        give = np.minimum(np.float32(amount) * give, lin)
    else:
        # legacy additive branch, kept so an asset that still declares it
        # renders what it declares instead of silently getting the new maths
        give = np.zeros_like(lin)
    take = np.float32(amount) * upsample_rows(spread_map, y0, y1, height, width)
    lin = np.maximum(lin + take - give, 0.0)
    return np.log10(np.maximum(lin.astype(np.float64), 1e-12)).reshape(-1, 3)


def layer_reference_exposure(observer: np.ndarray, grey: float = 0.18) -> np.ndarray:
    """Layer exposure produced by a neutral 18% scene — the per-layer anchor
    the source gate measures EV against.

    Derived from the stock's own observer rather than declared in the asset:
    the reference has to move with the observer, or a stock whose blue layer
    is twice as sensitive would appear to trigger a stop earlier for no
    physical reason.
    """
    grey_rgb = np.full((1, 3), float(grey), dtype=np.float64)
    return (grey_rgb @ np.asarray(observer, dtype=np.float64).T).reshape(3)


def scatter_source(rgb: np.ndarray, scatter: PrintScatterAsset) -> np.ndarray:
    """The luminance-gated, RGB-proportional removable energy at a pixel:
    source = rgb * max(Y - threshold, 0) / max(Y, eps). Pointwise, so it can
    be evaluated at ANY resolution — the production path evaluates it on the
    actual full-resolution developed band (review batch 16: subtracting a
    proxy-derived source at full resolution stole light from dark pixels
    that never had any)."""
    rgb = np.asarray(rgb, dtype=np.float32)
    luma = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
    y = rgb @ luma
    excess = np.maximum(y - scatter.threshold, 0.0)
    return rgb * (excess / np.maximum(y, 1e-9))[..., None]


def scatter_spread(source: np.ndarray, scatter: PrintScatterAsset) -> np.ndarray:
    """The non-negative multi-scale scatter of a source map, per-channel
    renormalized (float64 sums) to the source's exact energy."""
    dh, dw = source.shape[:2]
    level = source
    spread = np.zeros_like(source)
    total = 0.0
    for lvl in range(scatter.levels):
        if min(level.shape[0], level.shape[1]) <= 1:
            break
        pad_h = level.shape[0] % 2
        pad_w = level.shape[1] % 2
        if pad_h or pad_w:
            level = np.pad(level, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        sh, sw = level.shape[0] // 2, level.shape[1] // 2
        level = level.reshape(sh, 2, sw, 2, 3).mean(axis=(1, 3))
        blurred = _gaussian_blur(level, 2.0)
        factor = 2 ** (lvl + 1)
        col_idx = np.minimum(np.arange(dw) // factor, blurred.shape[1] - 1)
        wgt = 1.0 / (lvl + 1.0)
        total += wgt
        for r0 in range(0, dh, 256):
            r1 = min(r0 + 256, dh)
            row_idx = np.minimum(
                np.arange(r0, r1) // factor, blurred.shape[0] - 1
            )
            spread[r0:r1] += wgt * blurred[row_idx][:, col_idx]
    spread /= max(total, 1e-9)
    src_sum = np.sum(source, axis=(0, 1), dtype=np.float64)
    spr_sum = np.sum(spread, axis=(0, 1), dtype=np.float64)
    scale = np.where(spr_sum > 0.0, src_sum / np.maximum(spr_sum, 1e-30), 0.0)
    spread *= scale.astype(np.float32)[None, None, :]
    return spread


def bloom_delta_map(developed_dec: np.ndarray, scatter: PrintScatterAsset) -> np.ndarray:
    """CONSERVATIVE medium scatter on the decimated grid (P1, review batch
    15): the positive medium REDISTRIBUTES energy, it never adds light.

        Y      = dot(rgb, luma)
        excess = max(Y - threshold, 0)
        source = rgb * excess / max(Y, eps)      # luminance-gated, RGB-
                                                 # proportional: hue intact
        spread = K(source)                       # non-negative multi-scale
        delta  = spread - source                 # signed, sums to ZERO

    K is the same multi-scale pyramid, per-channel renormalized to the
    source's exact energy (float64 sums), so sum(delta) == 0 per channel to
    float precision: highlight cores LOSE what their neighbourhoods gain.
    The frame is a declared closed system — scatter neither appears from
    nowhere nor models losses past the frame edge. A uniform field yields
    delta == 0 (blur and pyramid preserve uniformity), so flat scenes pass
    through untouched.
    """
    rgb = np.asarray(developed_dec, dtype=np.float32)
    source = scatter_source(rgb, scatter)
    return scatter_spread(source, scatter) - source



def bloom_apply_rows(
    display_linear: np.ndarray,
    spread_ii: np.ndarray,
    y0: int,
    y1: int,
    height: int,
    width: int,
    scatter: PrintScatterAsset,
    amount: float,
) -> np.ndarray:
    """Conservative scatter for output rows [y0, y1), two-term form (review
    batch 16):

        out = img - a * source_full + a * upsample(spread)

    source_full is evaluated POINTWISE on this very band of the developed
    full-resolution image — the subtraction can never remove light a pixel
    does not carry, so a bright point inside a dark decimated cell no longer
    drives its neighbours negative (a * source_full <= strength * img keeps
    the output non-negative by construction). spread comes from the
    decimated proxy, renormalized to its own source's energy; the applied
    balance therefore conserves total light to proxy accuracy — exactly, on
    a uniform field, where the two terms cancel pointwise."""
    if amount <= 0.0:
        return display_linear
    img = np.asarray(display_linear, dtype=np.float32).reshape(y1 - y0, width, 3)
    a = float(amount) * scatter.strength
    up = _sample_plain(spread_ii, y0, y1, height, width)
    source_full = scatter_source(img, scatter)
    return (img + a * (up - source_full)).reshape(-1, 3)


def _sample_plain(
    ii: np.ndarray, y0: int, y1: int, height: int, width: int
) -> np.ndarray:
    """Fractional-footprint area means of a decimated map for output rows
    [y0, y1) — the area-preserving resampler behind bloom_apply_rows."""
    gh, gw = ii.shape[0] - 1, ii.shape[1] - 1
    ye = np.arange(y0, y1 + 1) * (gh / height)
    xe = np.arange(width + 1) * (gw / width)

    def _at(yq, xq):
        yi = np.clip(np.floor(yq).astype(int), 0, gh - 1)
        xi = np.clip(np.floor(xq).astype(int), 0, gw - 1)
        yf = (yq - yi)[:, None, None]
        xf = (xq - xi)[None, :, None]
        top = ii[yi][:, xi] * (1 - xf) + ii[yi][:, xi + 1] * xf
        bot = ii[yi + 1][:, xi] * (1 - xf) + ii[yi + 1][:, xi + 1] * xf
        return top * (1 - yf) + bot * yf

    s = _at(ye[1:], xe[1:]) - _at(ye[1:], xe[:-1]) - _at(ye[:-1], xe[1:]) + _at(ye[:-1], xe[:-1])
    area = (ye[1:] - ye[:-1])[:, None, None] * (xe[1:] - xe[:-1])[None, :, None]
    return (s / np.maximum(area, 1e-12)).astype(np.float32)


# --------------------------------------------------------------------------
# editorial capture bloom (§6.1, phase P3)
# --------------------------------------------------------------------------

def capture_bloom_gate(y_over_grey: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """smootherstep source gate on scene EV above 18% grey."""
    # In place, for the same reason as halation_layer_gate: the readable
    # expression chain costs six full grids of transient on a spread grid.
    t = np.maximum(np.asarray(y_over_grey, dtype=np.float32), 1e-20)
    np.log2(t, out=t)
    t -= np.float32(t0)
    t /= np.float32(max(t1 - t0, 1e-6))
    np.clip(t, 0.0, 1.0, out=t)
    poly = t * np.float32(6.0)
    poly -= np.float32(15.0)
    poly *= t
    poly += np.float32(10.0)
    poly *= t
    poly *= t
    poly *= t
    return poly


def capture_bloom_source_rows(
    rgb: np.ndarray, bloom: "CaptureBloomAsset"
) -> np.ndarray:
    """The FINEST detection scale, evaluated at full resolution.

    §6.1 wants source SIZE and diffusion radius to be independent, which needs
    the detector to be a scale space rather than one radius. The finest scale
    has to run at full resolution or a filament lamp — one pixel, enormously
    bright — is averaged below the gate before it is ever looked at, which is
    the same non-commutation that cost the old bloom its energy (review batch
    18). The coarser scales are detected on the decimated grid, which IS a
    low-pass: area decimation to the spread grid is a box filter at roughly
    18 um, exactly the next rung of the ladder.
    """
    arr = np.asarray(rgb, dtype=np.float32)
    y = arr @ np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float32)
    y /= np.float32(0.18)
    t0, t1 = bloom.scales[0].gate_ev
    return arr * capture_bloom_gate(y, t0, t1)[..., None]


def capture_bloom_map(
    fine_source_dec: np.ndarray,
    scene_dec: np.ndarray,
    geometry_w_mm: float,
    bloom: "CaptureBloomAsset",
) -> np.ndarray:
    """Assemble the diffused glow on the decimated grid.

    Each detection scale gets its OWN diffusion kernel, so a filament, a neon
    tube and a blown window do not all receive a halo of the same size — the
    §10.1 gate 18 requirement, and the thing a single detector radius cannot
    do however its threshold is set.
    """
    dh, dw = scene_dec.shape[:2]
    px_per_mm = dw / max(geometry_w_mm, 1e-9)
    scene = np.asarray(scene_dec, dtype=np.float32)
    y = scene @ np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float32)
    out = np.zeros((dh, dw, 3), dtype=np.float32)
    for i, scale in enumerate(bloom.scales):
        if i == 0:
            src = np.asarray(fine_source_dec, dtype=np.float32)
        else:  # noqa: PLR5501 - the branches allocate differently on purpose
            # Detect on a progressively low-passed luminance: only sources
            # whose AREA survives the blur still clear the gate, which is what
            # separates "a big light" from "a bright speck".
            det_px = max(scale.detect_um * 1e-3 * px_per_mm, 0.35)
            y_lp = _blur_bounded(np.ascontiguousarray(y[..., None]), det_px)[..., 0]
            y_lp /= np.float32(0.18)
            src = scene * capture_bloom_gate(y_lp, *scale.gate_ev)[..., None]
            del y_lp
        r_px = max(scale.diffuse_um * 1e-3 * px_per_mm, 0.35)
        blurred = _blur_bounded(src, r_px)
        del src
        blurred *= np.float32(scale.weight)
        out += blurred
        del blurred
    return out


def capture_bloom_apply_rows(
    rgb: np.ndarray,
    glow_map: np.ndarray,
    y0: int,
    y1: int,
    height: int,
    width: int,
    bloom: "CaptureBloomAsset",
    amount: float,
) -> np.ndarray:
    """Add the glow to scene-linear rows [y0, y1), with C1 core protection.

    Save Lights used to be `max(G - k*S, 0)`, which is only C0 and can leave a
    hard ring or a hollow halo around a large bright area — and it conflicts
    with §10.1 gate 4's own C1 requirement. The suppression is now a smooth
    function of how much of the local glow the pixel itself contributes:

        r      = S / (G + eps)
        w_core = 1 - save_lights * smootherstep(r0, r1, r)
        delta  = G * w_core

    A pixel that IS the source has r near 1 and keeps little of the glow; a
    pixel merely near one has r near 0 and keeps all of it. Nothing is
    clipped, so the halo stays continuous everywhere.

    This runs BEFORE the emulsion by construction (§6.1): the scene is where
    six stops of overrange still exist. The old operator ran after B2, where
    P0 measured 0.73 EV of headroom against the scene's 6.00 — no strength
    could have made that look like veiling glare.
    """
    if amount <= 0.0:
        return rgb
    n = y1 - y0
    img = np.asarray(rgb, dtype=np.float32).reshape(n, width, 3)
    glow = upsample_rows(glow_map, y0, y1, height, width)
    source = capture_bloom_source_rows(img, bloom)
    lum = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float32)
    gy = glow @ lum
    sy = source @ lum
    ratio = sy / np.maximum(gy, 1e-12)
    t = np.clip(
        (ratio - np.float32(bloom.core_ratio[0]))
        / np.float32(max(bloom.core_ratio[1] - bloom.core_ratio[0], 1e-6)),
        0.0, 1.0,
    )
    smooth = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    w_core = 1.0 - np.float32(bloom.save_lights) * smooth
    delta = glow * w_core[..., None]
    if bloom.saturation != 1.0:
        # Scale the glow's chroma about its own luminance axis: a saturation
        # control must not change how much LIGHT the glow adds, only its hue
        # purity, or it becomes a second strength slider.
        dy = delta @ lum
        delta = np.maximum(
            dy[..., None] + np.float32(bloom.saturation) * (delta - dy[..., None]),
            0.0,
        )
    return np.maximum(img + np.float32(amount) * delta, 0.0).reshape(-1, 3)
