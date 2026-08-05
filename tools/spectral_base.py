# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectral primitives for the film calibration base (rebuild stages 1+2).

This module owns the physical constants of the printing chain the fitter had been
approximating away, with every convention stated rather than implied:

- The printing illuminant TH-KG3 is a 3400 K blackbody through a Schott KG3
  heat-absorbing filter, exactly as the upstream spektrafilm project constructs
  it (`model/illuminants.py` at the pinned commit). A bare 3200 K blackbody —
  the previous stand-in — differs by a typical [-0.58, +0.25, +0.33] stop per
  paper layer after removing the common exposure, which printer lights cannot
  cancel across the whole curve.
- Integration is trapezoidal on explicit wavelength grids. The printing
  integral runs on the profiles' native 380-780/5 nm range; the viewing
  colorimetry integral runs on the intersection of the medium's grid with the
  CMF support — for these profiles still 380-780 nm. Media are never
  extrapolated onto the CMF's full 360-830 nm domain.
- Out-of-range behaviour is declared per quantity: sensitivities, illuminant
  SPDs and filter transmittances fill with ZERO outside their tabulated range
  (no light, no response); dye and base densities HOLD their edge value (a dye
  stack does not become transparent at the table boundary).
- The viewing translation is flare in XYZ, a luminance-only surround term, then
  Bradford adaptation from the medium's viewing white to D65 before entering
  Rec.2020 — so printer lights are never forced to eat a D50->D65 white-point
  difference disguised as exposure.

Float64 throughout; this is offline oracle code.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KG3_CSV = (
    PROJECT_ROOT
    / "dngscan_assets"
    / "spectral"
    / "spektrafilm"
    / "filters"
    / "schott_KG3.csv"
)

TH_KG3_TEMPERATURE_K = 3400.0  # upstream spektrafilm convention, not 3200


def trapezoid(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Trapezoidal integration with the numpy-version compatibility shim."""
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return integrate(y, x=x, axis=axis)


def blackbody_spd(wavelengths_nm: np.ndarray, temp_k: float) -> np.ndarray:
    """Relative Planck spectrum, peak-normalized (shape only; scale is solved)."""
    wl_m = np.asarray(wavelengths_nm, dtype=np.float64) * 1e-9
    c2 = 1.438776877e-2
    spd = 1.0 / (np.power(wl_m, 5.0) * np.expm1(c2 / (wl_m * temp_k)))
    return spd / np.max(spd)


def load_kg3_samples(path: Path = KG3_CSV) -> tuple[np.ndarray, np.ndarray]:
    """Raw KG3 (wavelength nm, transmittance 0..1) samples, deduplicated.

    The vendored CSV carries 146 rows but only 139 unique wavelengths; duplicate
    abscissae are merged by the declared rule: MEAN transmittance per wavelength,
    then sorted ascending. np.interp silently mishandles duplicated x, so the
    merge happens here, once, and the provenance records the rule.
    """
    rows = np.loadtxt(path, delimiter=",", dtype=np.float64)
    order = np.argsort(rows[:, 0], kind="stable")
    wl, tr = rows[order, 0], rows[order, 1]
    unique_wl, inverse = np.unique(wl, return_inverse=True)
    sums = np.zeros_like(unique_wl)
    counts = np.zeros_like(unique_wl)
    np.add.at(sums, inverse, tr)
    np.add.at(counts, inverse, 1.0)
    return unique_wl, sums / counts


def kg3_transmission(wavelengths_nm: np.ndarray, path: Path = KG3_CSV) -> np.ndarray:
    """KG3 transmittance on a grid. Out of tabulated range -> ZERO (declared)."""
    wl, tr = load_kg3_samples(path)
    grid = np.asarray(wavelengths_nm, dtype=np.float64)
    out = np.interp(grid, wl, tr, left=0.0, right=0.0)
    return np.clip(out, 0.0, 1.0)


def th_kg3_spd(wavelengths_nm: np.ndarray, path: Path = KG3_CSV) -> np.ndarray:
    """The printing illuminant: 3400 K blackbody x KG3, mean-normalized to 1.

    Mean normalization over the evaluation grid is the upstream convention; the
    absolute scale is irrelevant because printer exposures are solved, but a
    stated normalization keeps the recorded printer lights comparable between
    refits.
    """
    grid = np.asarray(wavelengths_nm, dtype=np.float64)
    spd = blackbody_spd(grid, TH_KG3_TEMPERATURE_K) * kg3_transmission(grid, path)
    mean = float(np.mean(spd))
    if mean <= 0.0:
        raise ValueError("TH-KG3 SPD collapsed to zero on the requested grid")
    return spd / mean


def th_kg3_provenance(path: Path = KG3_CSV) -> dict:
    """Everything needed to reproduce the illuminant, stamped into fit outputs."""
    wl, _ = load_kg3_samples(path)
    return {
        "illuminant": "TH-KG3",
        "construction": f"blackbody({TH_KG3_TEMPERATURE_K:.0f}K) x Schott KG3",
        "kg3_file": str(path.relative_to(PROJECT_ROOT)),
        "kg3_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kg3_samples": int(wl.size),
        "dedup_rule": "duplicate wavelengths merged by mean transmittance",
        "interpolation": "linear; outside tabulated range -> 0",
        "normalization": "SPD scaled to mean 1 over the evaluation grid",
        "upstream": "spektrafilm model/illuminants.py @ 3bb2c2d2801f",
    }


def intersect_grid(medium_wl: np.ndarray, cmf_lo: float = 360.0, cmf_hi: float = 830.0) -> np.ndarray:
    """The viewing-colorimetry grid: the medium's own samples inside CMF support.

    The medium is never extended onto the CMF's full domain — colorimetry can
    only be evaluated where the medium was actually measured.
    """
    wl = np.asarray(medium_wl, dtype=np.float64)
    keep = (wl >= cmf_lo) & (wl <= cmf_hi)
    if not np.any(keep):
        raise ValueError("medium grid does not intersect the CMF support")
    return wl[keep]


# --- viewing translation ------------------------------------------------------

_BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float64,
)

# CIE D65 white (2-degree observer), Y = 1.
XYZ_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

# Rec.2020 (D65) from XYZ, BT.2020-2 primaries. Row-scaled so the D65 white maps
# to EXACTLY [1,1,1]: neutrality assertions downstream must not inherit the
# published matrix's 2e-4 rounding as a phantom cast.
_XYZ_TO_REC2020_RAW = np.array(
    [
        [1.7166511880, -0.3556707838, -0.2533662814],
        [-0.6666843518, 1.6164812366, 0.0157685458],
        [0.0176398574, -0.0427706133, 0.9421031212],
    ],
    dtype=np.float64,
)


XYZ_TO_REC2020 = (
    np.diag(1.0 / (_XYZ_TO_REC2020_RAW @ np.array([0.95047, 1.0, 1.08883])))
    @ _XYZ_TO_REC2020_RAW
)


def bradford_cat(xyz_white_src: np.ndarray, xyz_white_dst: np.ndarray = XYZ_D65) -> np.ndarray:
    """Bradford chromatic adaptation matrix mapping src white exactly onto dst."""
    src = _BRADFORD @ np.asarray(xyz_white_src, dtype=np.float64)
    dst = _BRADFORD @ np.asarray(xyz_white_dst, dtype=np.float64)
    if np.any(src <= 0.0) or np.any(dst <= 0.0):
        raise ValueError("non-physical white point for Bradford adaptation")
    return np.linalg.inv(_BRADFORD) @ np.diag(dst / src) @ _BRADFORD


def viewing_translation_rec2020(
    xyz: np.ndarray,
    xyz_white: np.ndarray,
    flare: float,
    surround_exponent: float,
) -> np.ndarray:
    """Medium XYZ -> scene-linear Rec.2020 (D65) via the declared order.

    1. Viewing flare in XYZ:      XYZ_f = (XYZ + f * XYZ_w) / (1 + f)
    2. Surround, luminance only:  scale XYZ_f by Y_f^(s-1)  (Y' = Y^s)
    3. Bradford CAT medium-white -> D65, then the Rec.2020 matrix.

    The luminance-only surround deliberately avoids per-channel exponentiation
    in any RGB basis, which would silently change saturation; and because the
    CAT carries the white-point difference, printer lights never have to.
    """
    arr = np.asarray(xyz, dtype=np.float64)
    white = np.asarray(xyz_white, dtype=np.float64) / max(float(xyz_white[1]), 1e-12)
    flared = (arr + float(flare) * white[None, :]) / (1.0 + float(flare))
    y = np.maximum(flared[:, 1], 1e-9)
    scaled = flared * np.power(y, float(surround_exponent) - 1.0)[:, None]
    m = XYZ_TO_REC2020 @ bradford_cat(white)
    return scaled @ m.T
