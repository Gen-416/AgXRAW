#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Calibrate a demo Sigma fp -> ARRI-style scene-linear pre-AgX prefilter.

This is an offline tool: it writes a small JSON preset consumed by
`dngscan.scene_transform`.  Runtime JPEG export never imports this script and
does not need scipy/colour-science.

Data notes / replacement points:
- ALEV3 SSF: default CSV is dngscan_assets/spectral/arri_alexa_alev3_ssf_digitized.csv.
  Replace it with digitized values from
  https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/23/1/art00029
- IMX410 QE: default CSV is dngscan_assets/spectral/sony_imx410_qe_zwo_asi2400mc_digitized.csv.
  Replace it with a digitized ZWO ASI2400MC RGB QE curve.  That curve is a bare
  sensor/CFA proxy, not Sigma fp's final stack.
- Sigma fp hot mirror: default CSV is a sigmoid 420..660nm model. Replace it
  with a measured transmission curve when available, or pass --ir-transmission-csv.
- Skin spectra: default CSV is still an analytic demo skin manifold.  Pass
  --skin-csv or --skin-dir for a licensed/public skin reflectance data set.
- Colour standards: pass --standard-data colour after installing colour-science
  for tabulated D55/A illuminants and CIE 1931 2-degree CMFs.

The important implementation detail is that the final region matrices are fitted
in the same domain where dngscan applies them: white-balanced scene-linear
Rec.2020.  The SSF/QE curves first build simple per-camera colourimetric profiles
from broad reflectance spectra; the ARRI-like matrix then fits the residual
IMX410-profiled Rec.2020 response toward the ALEV-profiled Rec.2020 response
inside each target material family.  A separate look gain can amplify that
residual without changing the neutral-axis constraint.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

WL = np.arange(400.0, 701.0, 10.0, dtype=np.float64)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "dngscan_assets" / "spectral"

DEFAULT_ALEXA_SSF_CSV = "arri_alexa_alev3_ssf_digitized.csv"
DEFAULT_IMX410_QE_CSV = "sony_imx410_qe_zwo_asi2400mc_digitized.csv"
DEFAULT_IR_TRANSMISSION_CSV = "sigma_fp_hot_mirror_model_420_660.csv"
DEFAULT_SKIN_CSV = "demo_skin_reflectance.csv"
DEFAULT_CYAN_CSV = "demo_cyan_reflectance.csv"


# Rough digitized-style curves on WL.  These are intentionally easy to replace:
# each row is R/G/B sensitivity, sampled at 400..700nm in 10nm increments.
BUILTIN_ALEV3_SSF = np.array(
    [
        [0.02, 0.03, 0.10, 0.22, 0.42, 0.62, 0.80, 0.89, 0.82, 0.62, 0.38, 0.18, 0.08, 0.04, 0.03, 0.04, 0.07, 0.14, 0.28, 0.48, 0.68, 0.82, 0.88, 0.84, 0.72, 0.54, 0.36, 0.22, 0.12, 0.06, 0.03],
        [0.02, 0.04, 0.08, 0.18, 0.36, 0.60, 0.82, 0.95, 1.00, 0.95, 0.86, 0.72, 0.54, 0.36, 0.22, 0.14, 0.09, 0.06, 0.04, 0.03, 0.025, 0.02, 0.016, 0.012, 0.010, 0.008, 0.006, 0.004, 0.003, 0.002, 0.001],
        [0.15, 0.34, 0.62, 0.86, 1.00, 0.94, 0.78, 0.56, 0.34, 0.18, 0.09, 0.045, 0.025, 0.016, 0.010, 0.008, 0.006, 0.005, 0.004, 0.003, 0.0025, 0.002, 0.0015, 0.001, 0.0008, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001],
    ],
    dtype=np.float64,
).T

BUILTIN_IMX410_QE = np.array(
    [
        [0.01, 0.015, 0.03, 0.07, 0.14, 0.24, 0.38, 0.52, 0.62, 0.58, 0.44, 0.28, 0.16, 0.10, 0.12, 0.20, 0.34, 0.52, 0.70, 0.84, 0.92, 0.94, 0.90, 0.82, 0.70, 0.56, 0.42, 0.30, 0.20, 0.12, 0.06],
        [0.015, 0.025, 0.05, 0.11, 0.24, 0.44, 0.68, 0.86, 0.96, 1.00, 0.96, 0.82, 0.62, 0.40, 0.23, 0.13, 0.08, 0.05, 0.035, 0.025, 0.018, 0.014, 0.010, 0.008, 0.006, 0.0045, 0.0035, 0.0025, 0.0018, 0.0012, 0.0008],
        [0.08, 0.20, 0.42, 0.68, 0.88, 0.96, 1.00, 0.86, 0.62, 0.36, 0.18, 0.08, 0.035, 0.018, 0.011, 0.008, 0.006, 0.004, 0.003, 0.002, 0.0016, 0.0012, 0.0009, 0.0007, 0.0005, 0.0004, 0.0003, 0.0002, 0.00015, 0.0001, 0.00005],
    ],
    dtype=np.float64,
).T

XYZ_TO_REC2020 = np.array(
    [
        [1.7166511880, -0.3556707838, -0.2533662814],
        [-0.6666843518, 1.6164812366, 0.0157685458],
        [0.0176398574, -0.0427706133, 0.9421031212],
    ],
    dtype=np.float64,
)
REC2020_TO_XYZ = np.linalg.inv(XYZ_TO_REC2020)
D65_XYZ = REC2020_TO_XYZ @ np.ones(3, dtype=np.float64)

BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float64,
)
BRADFORD_INV = np.linalg.inv(BRADFORD)


def sigmoid_ir_cut(wavelengths: np.ndarray, cutoff_nm: float, width_nm: float) -> np.ndarray:
    red = 1.0 / (1.0 + np.exp((wavelengths - cutoff_nm) / max(width_nm, 1e-6)))
    blue = 1.0 / (1.0 + np.exp((420.0 - wavelengths) / 8.0))
    return red * blue


def blackbody_spd(wavelengths_nm: np.ndarray, temp_k: float) -> np.ndarray:
    wl_m = wavelengths_nm * 1e-9
    c2 = 1.438776877e-2
    spd = 1.0 / (np.power(wl_m, 5.0) * np.expm1(c2 / (wl_m * temp_k)))
    return spd / np.max(spd)


def _require_colour_science():
    """Calibration-grade colorimetry is not optional (review finding: presets
    silently depended on whether colour-science happened to be installed —
    parameters moved by up to 0.43 between environments). The analytic
    approximations remain reachable only by explicit request."""
    try:
        import colour  # type: ignore[import-not-found]

        return colour
    except ImportError as exc:
        raise RuntimeError(
            "colour-science is required for calibration-grade fitting; install it "
            "with `pip install 'dngscan[calibration]'` or pass "
            "standard_data='approximate' to explicitly accept the analytic "
            "CMF/blackbody approximations (the choice is stamped into the output)."
        ) from exc


def colorimetry_provenance(standard_data: str = "auto") -> dict:
    """What produced the colorimetric numbers — stamped into fit outputs."""
    if standard_data == "approximate":
        return {"backend": "approximate", "detail": "analytic CMF + blackbody SPDs"}
    colour = _require_colour_science()
    return {
        "backend": "colour-science",
        "version": str(getattr(colour, "__version__", "unknown")),
        "cmf": "CIE 1931 2 Degree Standard Observer",
    }


def _uniform_shape(colour, wavelengths_nm: np.ndarray):
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    steps = np.diff(wl)
    if wl.size < 2 or not np.allclose(steps, steps[0]):
        raise ValueError("colorimetry grids must be uniform")
    return colour.SpectralShape(float(wl[0]), float(wl[-1]), float(steps[0]))


def illuminant_spd(
    name: str,
    wavelengths_nm: np.ndarray | None = None,
    standard_data: str = "auto",
    csv_path: Path | None = None,
) -> np.ndarray:
    grid = WL if wavelengths_nm is None else np.asarray(wavelengths_nm, dtype=np.float64)
    if csv_path is not None:
        return load_spd_csv(csv_path)
    key = name.upper()
    if standard_data != "approximate":
        colour = _require_colour_science()
        shape = _uniform_shape(colour, grid)
        if key in colour.SDS_ILLUMINANTS:
            sd = colour.SDS_ILLUMINANTS[key].copy().align(shape)
        elif key == "K75P":
            # Cine print viewing: the Kinoton 75P xenon projector lives in
            # colour's light-source set, not the illuminant set — the same
            # source upstream spektrafilm uses for its 'cine' condition.
            sd = colour.SDS_LIGHT_SOURCES["Kinoton 75P"].copy().align(shape)
        else:
            raise ValueError(f"unknown illuminant: {name}")
        values = np.asarray(sd.values, dtype=np.float64)
        return values / max(float(np.max(values)), 1e-12)
    if key == "A":
        return blackbody_spd(grid, 2856.0)
    if key == "D65":
        return blackbody_spd(grid, 6504.0)
    if key == "D55":
        return blackbody_spd(grid, 5500.0)
    raise ValueError(f"unknown illuminant: {name}")


def cie_1931_cmf(wavelengths_nm: np.ndarray, standard_data: str = "auto", csv_path: Path | None = None) -> np.ndarray:
    if csv_path is not None:
        return load_curve_csv(csv_path)
    if standard_data != "approximate":
        colour = _require_colour_science()
        shape = _uniform_shape(colour, np.asarray(wavelengths_nm, dtype=np.float64))
        cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
        return np.asarray(cmfs.values, dtype=np.float64)
    return cie_1931_cmf_approx(wavelengths_nm)


def cie_1931_cmf_approx(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Analytic CIE 1931 2-degree CMF approximation.

    This fallback keeps the demo self-contained.  Replace with colour-science
    tabulated CMFs if you need calibration-grade mask coordinates.
    """
    w = wavelengths_nm
    x_t1 = np.where(w < 442.0, (w - 442.0) * 0.0624, (w - 442.0) * 0.0374)
    x_t2 = np.where(w < 599.8, (w - 599.8) * 0.0264, (w - 599.8) * 0.0323)
    x_t3 = np.where(w < 501.1, (w - 501.1) * 0.0490, (w - 501.1) * 0.0382)
    x = 0.362 * np.exp(-0.5 * x_t1 * x_t1) + 1.056 * np.exp(-0.5 * x_t2 * x_t2) - 0.065 * np.exp(-0.5 * x_t3 * x_t3)

    y_t1 = np.where(w < 568.8, (w - 568.8) * 0.0213, (w - 568.8) * 0.0247)
    y_t2 = np.where(w < 530.9, (w - 530.9) * 0.0613, (w - 530.9) * 0.0322)
    y = 0.821 * np.exp(-0.5 * y_t1 * y_t1) + 0.286 * np.exp(-0.5 * y_t2 * y_t2)

    z_t1 = np.where(w < 437.0, (w - 437.0) * 0.0845, (w - 437.0) * 0.0278)
    z_t2 = np.where(w < 459.0, (w - 459.0) * 0.0385, (w - 459.0) * 0.0725)
    z = 1.217 * np.exp(-0.5 * z_t1 * z_t1) + 0.681 * np.exp(-0.5 * z_t2 * z_t2)
    return np.stack([x, y, z], axis=1)


def integrate_spectral(values: np.ndarray) -> np.ndarray:
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return integrate(values, WL, axis=0)


def demo_skin_spectra() -> np.ndarray:
    spectra: list[np.ndarray] = []
    for melanin in np.linspace(0.0, 1.0, 8):
        for blood in np.linspace(0.4, 1.3, 6):
            for yellow in np.linspace(-0.03, 0.05, 3):
                base = 0.20 + 0.46 / (1.0 + np.exp(-(WL - 540.0) / 58.0))
                melanin_abs = (0.30 + 0.42 * melanin) * np.exp(-(WL - 400.0) / 155.0)
                hemo = blood * (
                    0.10 * np.exp(-0.5 * ((WL - 415.0) / 18.0) ** 2)
                    + 0.055 * np.exp(-0.5 * ((WL - 542.0) / 18.0) ** 2)
                    + 0.050 * np.exp(-0.5 * ((WL - 577.0) / 16.0) ** 2)
                )
                carotene = yellow * np.exp(-0.5 * ((WL - 590.0) / 75.0) ** 2)
                spectra.append(np.clip(base - melanin_abs - hemo + carotene, 0.025, 0.82))
    return np.asarray(spectra, dtype=np.float64)


def demo_cyan_spectra() -> np.ndarray:
    spectra: list[np.ndarray] = []
    for center in np.linspace(485.0, 520.0, 5):
        for width in np.linspace(34.0, 58.0, 4):
            blue_green = 0.04 + 0.55 * np.exp(-0.5 * ((WL - center) / width) ** 2)
            spectra.append(np.clip(blue_green + 0.05 / (1.0 + np.exp((WL - 610.0) / 28.0)), 0.02, 0.70))
    return np.asarray(spectra, dtype=np.float64)


def demo_neutral_spectra() -> np.ndarray:
    spectra: list[np.ndarray] = []
    center = (WL - 550.0) / 150.0
    for level in np.linspace(0.08, 0.82, 9):
        for slope in np.linspace(-0.10, 0.10, 5):
            for curve in np.linspace(-0.035, 0.035, 3):
                spectra.append(np.clip(level * (1.0 + slope * center + curve * (center * center - 0.35)), 0.02, 0.95))
    return np.asarray(spectra, dtype=np.float64)


def demo_colour_spectra() -> np.ndarray:
    spectra: list[np.ndarray] = []
    for center in np.linspace(430.0, 660.0, 10):
        for width in (28.0, 45.0, 70.0):
            bump = 0.035 + 0.68 * np.exp(-0.5 * ((WL - center) / width) ** 2)
            spectra.append(np.clip(bump, 0.015, 0.82))
    # Add foliage-like and warm fabric/wood-like families so camera profiles are not
    # fitted only on skin/cyan samples.
    for edge in np.linspace(500.0, 560.0, 5):
        spectra.append(np.clip(0.04 + 0.50 / (1.0 + np.exp(-(WL - edge) / 24.0)), 0.02, 0.72))
    for red_bias in np.linspace(0.25, 0.60, 5):
        spectra.append(np.clip(0.10 + red_bias / (1.0 + np.exp(-(WL - 585.0) / 42.0)), 0.03, 0.80))
    return np.asarray(spectra, dtype=np.float64)


def camera_profile_spectra(skin: np.ndarray, cyan: np.ndarray) -> np.ndarray:
    return np.concatenate([demo_neutral_spectra(), demo_colour_spectra(), skin, cyan], axis=0)


def _try_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = []
        for row in csv.reader(fh):
            cleaned = [cell.strip() for cell in row]
            if not cleaned or not any(cleaned):
                continue
            if cleaned[0].startswith("#"):
                continue
            rows.append(cleaned)
    if not rows:
        raise ValueError(f"{path} contains no CSV data")
    return rows


def _row_is_numeric(row: list[str], min_cols: int = 2) -> bool:
    return sum(_try_float(cell) is not None for cell in row) >= min_cols


def _normalise_header(cell: str) -> str:
    return cell.strip().lower().replace(" ", "_").replace("-", "_")


def _find_header_index(header: list[str], candidates: tuple[str, ...]) -> int | None:
    names = [_normalise_header(cell) for cell in header]
    for candidate in candidates:
        key = _normalise_header(candidate)
        for index, name in enumerate(names):
            if name == key or name.endswith("_" + key):
                return index
    return None


def _numeric_matrix(rows: list[list[str]]) -> np.ndarray:
    numeric_rows: list[list[float]] = []
    for row in rows:
        nums = [_try_float(cell) for cell in row]
        if all(value is not None for value in nums):
            numeric_rows.append([float(value) for value in nums if value is not None])
    if not numeric_rows:
        raise ValueError("CSV contains no fully numeric rows")
    width = min(len(row) for row in numeric_rows)
    return np.asarray([row[:width] for row in numeric_rows], dtype=np.float64)


def _interpolate_curve(wavelengths: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    values = values[order]
    return np.interp(WL, wavelengths, values)


def load_curve_csv(path: Path) -> np.ndarray:
    """Load wavelength,R,G,B CSV and resample to 400..700nm/10nm."""
    rows = _read_csv_rows(path)
    header = rows[0] if not _row_is_numeric(rows[0], 4) else None
    data_rows = rows[1:] if header is not None else rows
    if header is not None:
        wl_i = _find_header_index(header, ("wavelength", "wavelength_nm", "lambda", "nm", "wl"))
        r_i = _find_header_index(header, ("r", "red"))
        g_i = _find_header_index(header, ("g", "green"))
        b_i = _find_header_index(header, ("b", "blue"))
        if None not in (wl_i, r_i, g_i, b_i):
            numeric: list[tuple[float, float, float, float]] = []
            for row in data_rows:
                try:
                    numeric.append((float(row[wl_i]), float(row[r_i]), float(row[g_i]), float(row[b_i])))  # type: ignore[index]
                except (IndexError, ValueError):
                    continue
            if len(numeric) >= 2:
                data = np.asarray(numeric, dtype=np.float64)
                return np.stack([_interpolate_curve(data[:, 0], data[:, i]) for i in (1, 2, 3)], axis=1)
    data = _numeric_matrix(data_rows)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"{path} must contain wavelength,R,G,B columns")
    return np.stack([_interpolate_curve(data[:, 0], data[:, i]) for i in (1, 2, 3)], axis=1)


def load_spd_csv(path: Path, normalize: bool = True) -> np.ndarray:
    """Load wavelength,value CSV and resample to 400..700nm/10nm."""
    rows = _read_csv_rows(path)
    header = rows[0] if not _row_is_numeric(rows[0], 2) else None
    data_rows = rows[1:] if header is not None else rows
    if header is not None:
        wl_i = _find_header_index(header, ("wavelength", "wavelength_nm", "lambda", "nm", "wl"))
        value_i = _find_header_index(header, ("value", "transmission", "reflectance", "power", "spd"))
        if wl_i is not None and value_i is not None:
            pairs: list[tuple[float, float]] = []
            for row in data_rows:
                try:
                    pairs.append((float(row[wl_i]), float(row[value_i])))
                except (IndexError, ValueError):
                    continue
            if len(pairs) >= 2:
                data = np.asarray(pairs, dtype=np.float64)
                values = _interpolate_curve(data[:, 0], data[:, 1])
                return values / max(float(np.max(values)), 1e-12) if normalize else values
    data = _numeric_matrix(data_rows)
    if data.shape[1] < 2:
        raise ValueError(f"{path} must contain wavelength,value columns")
    values = _interpolate_curve(data[:, 0], data[:, 1])
    return values / max(float(np.max(values)), 1e-12) if normalize else values


def load_spectra_csv(path: Path) -> np.ndarray:
    """Load reflectance spectra from a wide or tidy CSV.

    Accepted forms:
    - wide: wavelength,sample_a,sample_b,...
    - tidy: sample,wavelength,reflectance
    - single: wavelength,reflectance
    - numeric matrix: either rows are samples on WL, or first column is wavelength.
    """
    rows = _read_csv_rows(path)
    header = rows[0] if not _row_is_numeric(rows[0], 2) else None
    data_rows = rows[1:] if header is not None else rows
    if header is not None:
        wl_i = _find_header_index(header, ("wavelength", "wavelength_nm", "lambda", "nm", "wl"))
        value_i = _find_header_index(header, ("reflectance", "value", "transmission"))
        sample_i = _find_header_index(header, ("sample", "name", "material", "id"))
        if wl_i is not None and value_i is not None and sample_i is not None:
            groups: dict[str, list[tuple[float, float]]] = {}
            for row in data_rows:
                try:
                    groups.setdefault(row[sample_i], []).append((float(row[wl_i]), float(row[value_i])))
                except (IndexError, ValueError):
                    continue
            spectra = [_interpolate_curve(np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs]))
                       for pairs in groups.values() if len(pairs) >= 2]
            if spectra:
                return np.clip(np.asarray(spectra, dtype=np.float64), 0.0, 1.5)
        if wl_i is not None:
            sample_columns = [i for i in range(len(header)) if i != wl_i]
            numeric: list[list[float]] = []
            for row in data_rows:
                values: list[float] = []
                try:
                    values.append(float(row[wl_i]))
                    values.extend(float(row[i]) for i in sample_columns)
                except (IndexError, ValueError):
                    continue
                numeric.append(values)
            if len(numeric) >= 2 and sample_columns:
                data = np.asarray(numeric, dtype=np.float64)
                spectra = [_interpolate_curve(data[:, 0], data[:, i]) for i in range(1, data.shape[1])]
                return np.clip(np.asarray(spectra, dtype=np.float64), 0.0, 1.5)
    data = _numeric_matrix(data_rows)
    if data.ndim != 2:
        raise ValueError(f"{path} must be a 2D CSV")
    if data.shape[1] >= 2 and np.nanmin(data[:, 0]) <= 405.0 and np.nanmax(data[:, 0]) >= 695.0:
        return np.clip(np.stack([_interpolate_curve(data[:, 0], data[:, i]) for i in range(1, data.shape[1])], axis=0), 0.0, 1.5)
    if data.shape[1] == WL.size:
        return np.clip(data.astype(np.float64, copy=False), 0.0, 1.5)
    raise ValueError(f"{path} must use 31 samples at 400..700nm or include wavelength headers")


def load_spectra_dir(path: Path) -> np.ndarray:
    spectra: list[np.ndarray] = []
    skipped: list[str] = []
    for item in sorted(path.glob("*.csv")):
        try:
            loaded = load_spectra_csv(item)
        except ValueError as exc:
            skipped.append(f"{item.name}: {exc}")
            continue
        spectra.append(loaded)
    if not spectra:
        detail = "; ".join(skipped[:3])
        raise ValueError(f"{path} contains no usable .csv spectra" + (f" ({detail})" if detail else ""))
    if skipped:
        print(f"warning: skipped {len(skipped)} non-spectral CSV file(s) in {path}", file=sys.stderr)
    return np.concatenate(spectra, axis=0)


def integrate_response(reflectance: np.ndarray, illuminant: np.ndarray, ssf: np.ndarray) -> np.ndarray:
    weighted = reflectance[:, :, None] * illuminant[None, :, None] * ssf[None, :, :]
    rgb = integrate_spectral(weighted.swapaxes(0, 1))
    white = integrate_spectral(illuminant[:, None] * ssf)
    return rgb / np.maximum(white[None, :], 1e-12)


def spectra_to_rec2020(reflectance: np.ndarray, illuminant: np.ndarray, cmf: np.ndarray) -> np.ndarray:
    weighted = reflectance[:, :, None] * illuminant[None, :, None] * cmf[None, :, :]
    xyz = integrate_spectral(weighted.swapaxes(0, 1))
    white = integrate_spectral(illuminant[:, None] * cmf)
    xyz = xyz / max(white[1], 1e-12)
    src_white = white / max(white[1], 1e-12)
    xyz = chromatic_adapt_xyz(xyz, src_white, D65_XYZ)
    rgb = xyz @ XYZ_TO_REC2020.T
    return np.clip(rgb, 1e-8, None)


def normalize_chroma(rgb: np.ndarray) -> np.ndarray:
    return rgb / np.maximum(rgb[:, 1:2], 1e-12)


def constrained_row_sum_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_n = normalize_chroma(src)
    dst_n = normalize_chroma(dst)
    a = np.stack([src_n[:, 0] - src_n[:, 2], src_n[:, 1] - src_n[:, 2]], axis=1)
    matrix = np.empty((3, 3), dtype=np.float64)
    for row in range(3):
        b = dst_n[:, row] - src_n[:, 2]
        coeff, *_ = np.linalg.lstsq(a, b, rcond=None)
        matrix[row, 0] = coeff[0]
        matrix[row, 1] = coeff[1]
        matrix[row, 2] = 1.0 - coeff[0] - coeff[1]
    return matrix


def chromatic_adapt_xyz(xyz: np.ndarray, src_white: np.ndarray, dst_white: np.ndarray) -> np.ndarray:
    src_lms = BRADFORD @ src_white
    dst_lms = BRADFORD @ dst_white
    scale = np.diag(dst_lms / np.maximum(src_lms, 1e-12))
    matrix = BRADFORD_INV @ scale @ BRADFORD
    return xyz @ matrix.T


def camera_to_rec2020_matrix(camera_rgb: np.ndarray, target_rec2020: np.ndarray) -> np.ndarray:
    return constrained_row_sum_fit(camera_rgb, target_rec2020)


def camera_rec2020_response(reflectance: np.ndarray, illuminant: np.ndarray, ssf: np.ndarray, camera_to_rec2020: np.ndarray) -> np.ndarray:
    camera_rgb = integrate_response(reflectance, illuminant, ssf)
    return np.clip(camera_rgb @ camera_to_rec2020.T, 1e-8, None)


def strengthen_matrix(matrix: np.ndarray, look_gain: float) -> np.ndarray:
    out = np.eye(3, dtype=np.float64) + float(look_gain) * (matrix - np.eye(3, dtype=np.float64))
    # Numerical guard: neutral axis must remain neutral after amplification.
    row_sum = out.sum(axis=1)
    out[:, 1] += 1.0 - row_sum
    return out


def controlled_cyan_matrix(red_pull: float = 0.12, blue_push: float = 0.04) -> np.ndarray:
    """ARRI-like cyan counter-axis: reduce red in cyan/blue-green materials while
    lightly supporting blue.  This is a bounded look matrix, not an SSF residual,
    because the rough cyan spectra make the residual fit unstable."""
    red_pull = float(red_pull)
    blue_push = float(blue_push)
    return np.asarray(
        [
            [1.0 + red_pull, -red_pull, 0.0],
            [0.0, 1.0, 0.0],
            [-blue_push, 0.0, 1.0 + blue_push],
        ],
        dtype=np.float64,
    )


def controlled_cool_balance_matrix(red_pull: float = 0.35, blue_push: float = 0.12) -> np.ndarray:
    red_pull = float(red_pull)
    blue_push = float(blue_push)
    return np.asarray(
        [
            [1.0 + red_pull, -red_pull, 0.0],
            [0.0, 1.0, 0.0],
            [-blue_push, 0.0, 1.0 + blue_push],
        ],
        dtype=np.float64,
    )


def build_fixed_region(
    name: str,
    matrix: np.ndarray,
    mu_rg_bg: tuple[float, float],
    cov_rg_bg: tuple[tuple[float, float], tuple[float, float]],
    scale: float,
    region_strength: float,
) -> dict:
    return {
        "name": name,
        "matrix": [[round(float(v), 8) for v in row] for row in matrix],
        "mu_rg_bg": [float(mu_rg_bg[0]), float(mu_rg_bg[1])],
        "cov_rg_bg": [[float(v) for v in row] for row in cov_rg_bg],
        "scale": float(scale),
        "strength": float(region_strength),
    }


def mask_params(src: np.ndarray, scale: float) -> tuple[list[float], list[list[float]]]:
    chroma = np.stack([src[:, 0] / np.maximum(src[:, 1], 1e-12), src[:, 2] / np.maximum(src[:, 1], 1e-12)], axis=1)
    mu = np.mean(chroma, axis=0)
    cov = np.cov(chroma.T)
    cov += np.eye(2) * 1e-5
    return [float(mu[0]), float(mu[1])], [[float(cov[0, 0]), float(cov[0, 1])], [float(cov[1, 0]), float(cov[1, 1])]]


def build_region(
    name: str,
    spectra: np.ndarray,
    illuminant: np.ndarray,
    fp_ssf: np.ndarray,
    alexa_ssf: np.ndarray,
    fp_to_rec2020: np.ndarray,
    alexa_to_rec2020: np.ndarray,
    scale: float,
    look_gain: float,
    region_strength: float,
    matrix_override: np.ndarray | None = None,
) -> dict:
    fp = camera_rec2020_response(spectra, illuminant, fp_ssf, fp_to_rec2020)
    alexa = camera_rec2020_response(spectra, illuminant, alexa_ssf, alexa_to_rec2020)
    matrix = matrix_override if matrix_override is not None else strengthen_matrix(constrained_row_sum_fit(fp, alexa), look_gain)
    mask_rgb = fp
    mu, cov = mask_params(mask_rgb, scale)
    return {
        "name": name,
        "matrix": [[round(float(v), 8) for v in row] for row in matrix],
        "mu_rg_bg": [round(v, 8) for v in mu],
        "cov_rg_bg": [[round(v, 10) for v in row] for row in cov],
        "scale": scale,
        "strength": region_strength,
    }


def existing_region_windows(path: Path, key: str) -> dict[str, dict]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        regions = raw.get("transforms", raw)[key].get("regions", [])
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for region in regions:
        if isinstance(region, dict) and "name" in region:
            out[str(region["name"])] = region
    return out


def preserve_window(region: dict, existing: dict[str, dict]) -> dict:
    old = existing.get(str(region.get("name", "")))
    if not old:
        return region
    for field in ("mu_rg_bg", "cov_rg_bg", "scale"):
        if field in old:
            region[field] = old[field]
    return region


# --- material-aware ALEV simulation (strict mode) ---------------------------
# Layer 1 of the material prefeed: per-material Sigma->ALEV fits with an explicit
# error report (before/after divergence in Oklab) and a cross-material leakage table
# that justifies the runtime windows. Layer 2 (runtime) consumes the emitted regions:
# each window carries its own matrix and a confidence in [0,1] that scene_transform
# folds into the effective weight.

REC2020_TO_XYZ_D65 = np.array([
    [0.6369580, 0.1446169, 0.1688810],
    [0.2627002, 0.6779981, 0.0593017],
    [0.0000000, 0.0280727, 1.0609851],
])
_OK_M1 = np.array([
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070],
])
_OK_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])


def rec2020_to_oklab(rgb: np.ndarray) -> np.ndarray:
    xyz = np.maximum(rgb, 0.0) @ REC2020_TO_XYZ_D65.T
    lms = np.cbrt(np.maximum(xyz @ _OK_M1.T, 0.0))
    return lms @ _OK_M2.T


def oklab_divergence(a_rgb: np.ndarray, b_rgb: np.ndarray) -> np.ndarray:
    """Per-sample Oklab distance between two chroma-normalized Rec.2020 renders."""
    a = normalize_chroma(a_rgb) * 0.18
    b = normalize_chroma(b_rgb) * 0.18
    return np.linalg.norm(rec2020_to_oklab(a) - rec2020_to_oklab(b), axis=1)


def demo_foliage_spectra() -> np.ndarray:
    """Chlorophyll-style reflectance with an explicit red edge.

    The red edge (steep rise past ~690nm) is where the fp hot-mirror (~660nm cutoff)
    and ALEV's wider red response diverge hardest, so foliage is the class with the
    largest native camera disagreement — the per-material fit must capture it."""
    spectra: list[np.ndarray] = []
    for green_peak in np.linspace(0.12, 0.30, 5):
        for edge_nm in np.linspace(686.0, 700.0, 3):
            for dry in np.linspace(0.0, 0.35, 3):
                bump = green_peak * np.exp(-0.5 * ((WL - 552.0) / 32.0) ** 2)
                red_edge = 0.42 / (1.0 + np.exp(-(WL - edge_nm) / 7.0))
                base = 0.035 + dry * 0.10 * np.clip((WL - 520.0) / 180.0, 0.0, 1.0)
                spectra.append(np.clip(base + bump + red_edge, 0.015, 0.85))
    return np.asarray(spectra, dtype=np.float64)


def demo_magenta_spectra() -> np.ndarray:
    """Magenta/purple dye family (wigs, costume fabric): blue+red high, green trough."""
    spectra: list[np.ndarray] = []
    for blue in np.linspace(0.22, 0.55, 4):
        for red in np.linspace(0.25, 0.65, 4):
            for trough in np.linspace(500.0, 560.0, 3):
                b = blue * np.exp(-0.5 * ((WL - 445.0) / 38.0) ** 2)
                r = red / (1.0 + np.exp(-(WL - 605.0) / 22.0))
                g = 0.05 + 0.02 * np.exp(-0.5 * ((WL - trough) / 60.0) ** 2)
                spectra.append(np.clip(b + r + g, 0.02, 0.9))
    return np.asarray(spectra, dtype=np.float64)


def demo_sky_reflectance(d55: np.ndarray) -> np.ndarray:
    """Sky radiance family expressed as equivalent reflectance under D55 (SPD ratio),
    so the standard reflectance pipeline reproduces sky colours exactly. Profile- and
    report-only: sky is an emitter and gets no runtime window."""
    spectra: list[np.ndarray] = []
    for k in np.linspace(0.6, 2.2, 5):
        sky = d55 * (WL / 550.0) ** (-k)
        ratio = sky / np.maximum(d55, 1e-9)
        spectra.append(np.clip(ratio / max(float(ratio.max()), 1e-9) * 0.8, 0.02, 1.0))
    return np.asarray(spectra, dtype=np.float64)


MATERIAL_BASE_STRENGTH = {"skin": 1.0, "foliage": 0.9, "cyan": 0.9, "neutral": 0.5, "magenta": 0.85}
MATERIAL_MASK_SCALE = {"skin": 1.6, "foliage": 2.0, "cyan": 2.2, "neutral": 1.2, "magenta": 2.0}


def synthetic_reflectance_families(count_per_family: int = 60) -> np.ndarray:
    """Siragusano-style "create objects": smooth parametric reflectance families.

    The AMPAS 190 spectra cover real materials but leave chromaticity space sparse;
    FilmLight's virtual shoot-out densifies the pairing with synthetic families. These
    are physically plausible smooth reflectances (Gaussian bumps, sigmoid edges,
    slopes and two-bump combinations on [0.02, 0.95]) used ONLY for fitting the two
    observers' colorimetric profiles — per-material class fits keep their own measured
    spectra, because a synthetic bump is not evidence about skin.
    """
    rng_like = np.linspace(0.0, 1.0, count_per_family)
    spectra = []
    wl_n = (WL - WL[0]) / (WL[-1] - WL[0])
    for i, t in enumerate(rng_like):
        center = 0.1 + 0.8 * t
        width = 0.08 + 0.22 * ((i * 7) % count_per_family) / count_per_family
        bump = np.exp(-0.5 * ((wl_n - center) / width) ** 2)
        base = 0.05 + 0.25 * ((i * 3) % count_per_family) / count_per_family
        spectra.append(base + (0.9 - base) * bump)
        edge = 1.0 / (1.0 + np.exp(-(wl_n - center) / max(0.03, width / 3)))
        spectra.append(0.05 + 0.85 * edge)
        spectra.append(0.05 + 0.85 * (1.0 - edge))
        if i % 3 == 0:
            c2 = 0.9 - 0.8 * t
            bump2 = np.exp(-0.5 * ((wl_n - c2) / (width * 0.8)) ** 2)
            spectra.append(np.clip(0.08 + 0.5 * bump + 0.45 * bump2, 0.0, 1.0))
    return np.clip(np.asarray(spectra, dtype=np.float64), 0.02, 0.95)


def material_spectra_sets(args: argparse.Namespace) -> dict[str, tuple[np.ndarray, str]]:
    def from_file(fname: str, fallback: np.ndarray, label: str) -> tuple[np.ndarray, str]:
        path = default_data_path(args.data_dir, fname)
        if path is not None:
            return load_spectra_csv(path), source_label(path, label)
        return fallback, label

    skin_path = args.skin_csv if args.skin_csv is not None else (None if args.skin_dir else default_data_path(args.data_dir, DEFAULT_SKIN_CSV))
    cyan_path = args.cyan_csv if args.cyan_csv is not None else (None if args.cyan_dir else default_data_path(args.data_dir, DEFAULT_CYAN_CSV))
    skin, skin_src = load_spectra_input(skin_path, args.skin_dir, demo_skin_spectra())
    cyan, cyan_src = load_spectra_input(cyan_path, args.cyan_dir, demo_cyan_spectra())
    foliage, fol_src = from_file("foliage_reflectance.csv", demo_foliage_spectra(), "analytic foliage demo (red-edge family)")
    magenta, mag_src = from_file("magenta_reflectance.csv", demo_magenta_spectra(), "analytic magenta dye demo")
    neutral, neu_src = from_file("neutral_reflectance.csv", demo_neutral_spectra(), "analytic neutral ramps")
    return {
        "skin": (skin, skin_src),
        "foliage": (foliage, fol_src),
        "cyan": (cyan, cyan_src),
        "neutral": (neutral, neu_src),
        "magenta": (magenta, mag_src),
    }


def run_material_mode(
    args: argparse.Namespace,
    alexa_ssf: np.ndarray,
    fp_ssf: np.ndarray,
    cmf: np.ndarray,
    base_sources: dict[str, str],
) -> int:
    ill_names = [s.strip() for s in args.fit_illuminants.split(",") if s.strip()]
    if "D55" not in ill_names:
        ill_names.insert(0, "D55")
    ills = {name: illuminant_spd(name, args.standard_data) for name in ill_names}
    d55 = ills["D55"]

    materials = material_spectra_sets(args)
    sky = demo_sky_reflectance(d55)
    profile_parts = [spec for spec, _ in materials.values()] + [demo_colour_spectra(), sky, synthetic_reflectance_families()]
    if args.profile_csv is not None:
        profile_parts.append(load_spectra_csv(args.profile_csv))
    if args.profile_dir is not None:
        profile_parts.append(load_spectra_dir(args.profile_dir))
    profile = np.concatenate(profile_parts, axis=0)

    # Per-illuminant camera profiles: each camera is balanced to that illuminant,
    # matching the runtime WB convention (and the daylight-frame window transport).
    profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, spd in ills.items():
        target = spectra_to_rec2020(profile, spd, cmf)
        profiles[name] = (
            camera_to_rec2020_matrix(integrate_response(profile, spd, fp_ssf), target),
            camera_to_rec2020_matrix(integrate_response(profile, spd, alexa_ssf), target),
        )

    gain = float(args.material_look_gain)
    fits: dict[str, dict] = {}
    responses: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for mat, (spectra, source) in materials.items():
        per_ill: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in ill_names:
            spd = ills[name]
            fp2020, alexa2020 = profiles[name]
            fp = camera_rec2020_response(spectra, spd, fp_ssf, fp2020)
            alexa = camera_rec2020_response(spectra, spd, alexa_ssf, alexa2020)
            per_ill[name] = (fp, alexa)
        responses[mat] = per_ill
        src = np.concatenate([fp for fp, _ in per_ill.values()])
        dst = np.concatenate([alexa for _, alexa in per_ill.values()])
        matrix = strengthen_matrix(constrained_row_sum_fit(src, dst), gain)

        errors: dict[str, dict[str, float]] = {}
        conf_terms: list[float] = []
        for name, (fp, alexa) in per_ill.items():
            before = oklab_divergence(fp, alexa)
            after = oklab_divergence(np.clip(fp @ matrix.T, 1e-8, None), alexa)
            errors[name] = {
                "before_mean": float(before.mean()), "before_p95": float(np.percentile(before, 95.0)),
                "after_mean": float(after.mean()), "after_p95": float(np.percentile(after, 95.0)),
            }
            if before.mean() < 1e-5:
                conf_terms.append(1.0)
            else:
                conf_terms.append(max(0.0, min(1.0, 1.0 - after.mean() / before.mean())))
        confidence = float(np.mean(conf_terms))
        fits[mat] = {"matrix": matrix, "errors": errors, "confidence": confidence, "source": source}

    # Cross-material leakage under D55: applying material k's matrix to material j.
    # Large off-diagonal degradation is the physical argument for windowed application.
    leakage: dict[str, dict[str, float]] = {}
    for k, fit_k in fits.items():
        row: dict[str, float] = {}
        for j in fits:
            fp_j, alexa_j = responses[j]["D55"]
            after = oklab_divergence(np.clip(fp_j @ fit_k["matrix"].T, 1e-8, None), alexa_j)
            row[j] = float(after.mean())
        leakage[k] = row

    regions = []
    for mat, fit in fits.items():
        fp_d55, _ = responses[mat]["D55"]
        mu, cov = mask_params(fp_d55, MATERIAL_MASK_SCALE[mat])
        regions.append({
            "name": mat,
            "matrix": [[round(float(v), 8) for v in row] for row in fit["matrix"]],
            "mu_rg_bg": [round(float(v), 8) for v in mu],
            "cov_rg_bg": [[round(float(v), 10) for v in row] for row in cov],
            "scale": MATERIAL_MASK_SCALE[mat],
            "strength": MATERIAL_BASE_STRENGTH[mat],
            "confidence": round(fit["confidence"], 4),
        })

    key = args.material_key
    preset = {
        "name": key,
        "label": args.material_label,
        "illuminant": "D55",
        "working_space": "Rec2020",
        "note": (
            f"Material-aware Sigma->{args.target_name} separation: per-material constrained fits on "
            f"{'+'.join(ill_names)} samples, windows in the D55 calibration frame "
            f"(runtime von Kries transport applies), look_gain={gain:g}. Confidence "
            f"folds fit quality into the effective weight. Data quality per "
            f"dngscan_assets/spectral/README.md; regenerate after replacing CSVs."
        ),
        "sources": dict(base_sources, **{f"{m}_spectra": f["source"] for m, f in fits.items()}),
        "regions": regions,
    }

    out = args.out
    existing: dict = {}
    if out.is_file():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.setdefault("version", 1)
    existing.setdefault("transforms", {})[key] = preset
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = {
        "illuminants": ill_names,
        "look_gain": gain,
        "materials": {m: {"errors": f["errors"], "confidence": f["confidence"], "source": f["source"]} for m, f in fits.items()},
        "leakage_d55_after_mean": leakage,
    }
    report_path = args.report_json or (args.data_dir / "calibration_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote preset '{key}' -> {out}")
    print(f"wrote report -> {report_path}\n")
    print(f"{'材料':10s}{'光源':6s}{'拟合前ΔOk':>12s}{'拟合后ΔOk':>12s}{'p95后':>10s}{'置信度':>8s}")
    for mat, fit in fits.items():
        for name, err in fit["errors"].items():
            print(f"{mat:10s}{name:6s}{err['before_mean']:>12.5f}{err['after_mean']:>12.5f}{err['after_p95']:>10.5f}{fit['confidence']:>8.3f}")
    print("\n跨材料泄漏 (D55, 用k矩阵处理j材料后的平均ΔOk; 对角=本类):")
    mats = list(fits)
    print(f"{'k/j':10s}" + "".join(f"{j:>10s}" for j in mats))
    for k in mats:
        print(f"{k:10s}" + "".join(f"{leakage[k][j]:>10.5f}" for j in mats))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate dngscan scene-transform skin/cyan preset.")
    parser.add_argument("--out", type=Path, default=Path("dngscan/scene_transform_presets.json"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing calibration CSV inputs.")
    parser.add_argument("--illuminant", choices=("D55", "D65", "A"), default="D55")
    parser.add_argument("--standard-data", choices=("auto", "colour", "analytic"), default="auto",
                        help="Use colour-science standard CMF/illuminant data when available; analytic fallback keeps the tool self-contained.")
    parser.add_argument("--illuminant-csv", type=Path,
                        help="Optional wavelength,value SPD CSV overriding --illuminant/--standard-data.")
    parser.add_argument("--cmf-csv", type=Path,
                        help="Optional wavelength,X,Y,Z CSV overriding colour-science/analytic CMF.")
    parser.add_argument("--ir-cutoff", type=float, default=660.0)
    parser.add_argument("--ir-width", type=float, default=15.0)
    parser.add_argument("--ir-transmission-csv", type=Path,
                        help="Optional wavelength,transmission CSV overriding the sigmoid Sigma fp hot-mirror model.")
    parser.add_argument("--mask-scale", type=float, default=2.3)
    parser.add_argument("--skin-mask-scale", type=float)
    parser.add_argument("--cyan-mask-scale", type=float)
    parser.add_argument("--look-gain", type=float, default=2.35,
                        help="Amplify the physically fitted residual from identity while preserving neutral axis.")
    parser.add_argument("--skin-look-gain", type=float)
    parser.add_argument("--cyan-look-gain", type=float)
    parser.add_argument("--skin-region-strength", type=float, default=1.15)
    parser.add_argument("--cyan-region-strength", type=float, default=0.90)
    parser.add_argument("--cyan-mode", choices=("cool", "spectral"), default="cool",
                        help="cool=bounded ARRI-like cyan counter-axis; spectral=raw SSF residual fit.")
    parser.add_argument("--cyan-red-pull", type=float, default=0.12)
    parser.add_argument("--cyan-blue-push", type=float, default=0.04)
    parser.add_argument("--background-cool", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--background-region-strength", type=float, default=0.42)
    parser.add_argument("--background-red-pull", type=float, default=0.35)
    parser.add_argument("--background-blue-push", type=float, default=0.12)
    parser.add_argument("--preserve-existing-windows", action="store_true",
                        help="Keep existing mu/cov/scale windows in --out and update only matrices/strengths.")
    parser.add_argument("--alexa-ssf-csv", type=Path)
    parser.add_argument("--imx410-qe-csv", type=Path)
    parser.add_argument("--skin-csv", type=Path)
    parser.add_argument("--skin-dir", type=Path)
    parser.add_argument("--cyan-csv", type=Path)
    parser.add_argument("--cyan-dir", type=Path)
    parser.add_argument("--profile-csv", type=Path,
                        help="Optional extra reflectance CSV for fitting camera-to-Rec2020 profiles.")
    parser.add_argument("--profile-dir", type=Path,
                        help="Optional directory of extra reflectance CSVs for camera profile fitting.")
    parser.add_argument("--write-bootstrap-csv", type=Path,
                        help="Write the current rough built-in spectral data as replaceable CSV files and exit.")
    parser.add_argument("--preset-mode", choices=("skin", "material"), default="skin",
                        help="skin=旧版 skin/cyan 预设；material=多材料 ALEV 仿真前馈（逐类拟合+误差报告+置信度）")
    parser.add_argument("--fit-illuminants", default="D55,A",
                        help="material 模式的联合拟合光源列表（窗口恒在 D55 标定系）")
    parser.add_argument("--material-look-gain", type=float, default=1.0,
                        help="material 模式的残差增益；1.0=严格物理拟合，不做风格放大")
    parser.add_argument("--material-key", default="alev_material_d55",
                        help="material 模式写入的 preset 键名（合并写入，不清空其他 preset）")
    parser.add_argument("--material-label", default="ALEV material prefeed (D55 windows)",
                        help="material 模式 preset 的显示名（如：胶片分离 · Portra 400）")
    parser.add_argument("--target-name", default="ALEV",
                        help="拟合目标观察者的名字，用于 note/报告（如 Portra 400）")
    parser.add_argument("--report-json", type=Path,
                        help="误差/泄漏报告输出路径（默认 data-dir/calibration_report.json）")
    return parser.parse_args()


def default_data_path(data_dir: Path, filename: str) -> Path | None:
    path = data_dir / filename
    return path if path.is_file() else None


def choose_path(explicit: Path | None, data_dir: Path, filename: str) -> Path | None:
    if explicit is not None:
        return explicit
    return default_data_path(data_dir, filename)


def source_label(path: Path | None, fallback: str) -> str:
    if path is None:
        return fallback
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_spectra_input(csv_path: Path | None, dir_path: Path | None, fallback: np.ndarray) -> tuple[np.ndarray, str]:
    parts: list[np.ndarray] = []
    sources: list[str] = []
    if csv_path is not None:
        parts.append(load_spectra_csv(csv_path))
        sources.append(source_label(csv_path, str(csv_path)))
    if dir_path is not None:
        parts.append(load_spectra_dir(dir_path))
        sources.append(source_label(dir_path, str(dir_path)))
    if parts:
        return np.concatenate(parts, axis=0), ", ".join(sources)
    return fallback, "built-in analytic generator"


def write_curve_csv(path: Path, data: np.ndarray, columns: tuple[str, str, str, str] = ("wavelength_nm", "R", "G", "B")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for wl, row in zip(WL, data, strict=True):
            writer.writerow([f"{wl:.0f}", *(f"{float(v):.8g}" for v in row)])


def write_spd_csv(path: Path, values: np.ndarray, value_name: str = "value") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["wavelength_nm", value_name])
        for wl, value in zip(WL, values, strict=True):
            writer.writerow([f"{wl:.0f}", f"{float(value):.8g}"])


def write_spectra_csv(path: Path, spectra: np.ndarray, sample_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["wavelength_nm", *[f"{sample_prefix}_{i:03d}" for i in range(spectra.shape[0])]])
        for col, wl in enumerate(WL):
            writer.writerow([f"{wl:.0f}", *(f"{float(v):.8g}" for v in spectra[:, col])])


def write_bootstrap_csv_bundle(path: Path, ir_cutoff: float, ir_width: float) -> None:
    """Export current rough built-ins as editable CSV starting points."""
    path.mkdir(parents=True, exist_ok=True)
    write_curve_csv(path / DEFAULT_ALEXA_SSF_CSV, BUILTIN_ALEV3_SSF)
    write_curve_csv(path / DEFAULT_IMX410_QE_CSV, BUILTIN_IMX410_QE)
    write_spd_csv(path / DEFAULT_IR_TRANSMISSION_CSV, sigmoid_ir_cut(WL, ir_cutoff, ir_width), "transmission")
    write_spectra_csv(path / DEFAULT_SKIN_CSV, demo_skin_spectra(), "skin")
    write_spectra_csv(path / DEFAULT_CYAN_CSV, demo_cyan_spectra(), "cyan")
    (path / "README.md").write_text(
        "# dngscan spectral calibration CSVs\n\n"
        "These CSV files are replaceable calibration inputs for `tools/calibrate_skin_matrix.py`.\n"
        "They intentionally keep measured or digitized spectral data outside the algorithm.\n\n"
        "Current files are bootstrap-quality approximations, not authoritative measurements:\n\n"
        "- `arri_alexa_alev3_ssf_digitized.csv`: rough ALEV3/ALEXA SSF digitization target. Replace with points digitized from Figure 1 of Leonhardt & Brendel, CIC 2015: https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/23/1/art00029\n"
        "- `sony_imx410_qe_zwo_asi2400mc_digitized.csv`: rough IMX410 RGB QE proxy. Replace with points digitized from the ZWO ASI2400MC QE graph/specification: https://www.zwoastro.com/product/asi2400mc-pro/\n"
        "- `sigma_fp_hot_mirror_model_420_660.csv`: sigmoid hot-mirror transmission model, 420nm blue-side and 660nm red-side cutoff assumption. Replace with measured transmission if available. Kolari teardown confirms the fp conversion/filter-stack context: https://kolarivision.com/the-sigma-fp-disassembly-and-teardown/\n"
        "- `demo_skin_reflectance.csv`: analytic demo skin reflectance manifold. Replace with a licensed/public skin reflectance library such as Hyper-Skin if you have permission to use the released data: https://github.com/hyperspectral-skin/Hyper-Skin-2023\n"
        "- `demo_cyan_reflectance.csv`: analytic cyan/blue-green material manifold. Replace or augment with measured surface spectra. The MLS dataset is one open source of real measured object/illumination spectra under CC BY-SA 4.0: https://github.com/visillect/mls-dataset\n\n"
        "Accepted formats include wide `wavelength_nm,sample_1,sample_2,...`, tidy `sample,wavelength_nm,reflectance`, and single `wavelength_nm,value` CSVs.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.write_bootstrap_csv is not None:
        write_bootstrap_csv_bundle(args.write_bootstrap_csv, args.ir_cutoff, args.ir_width)
        print(f"wrote bootstrap CSV bundle to {args.write_bootstrap_csv}")
        return 0

    data_dir = args.data_dir
    alexa_path = choose_path(args.alexa_ssf_csv, data_dir, DEFAULT_ALEXA_SSF_CSV)
    imx410_path = choose_path(args.imx410_qe_csv, data_dir, DEFAULT_IMX410_QE_CSV)
    ir_path = choose_path(args.ir_transmission_csv, data_dir, DEFAULT_IR_TRANSMISSION_CSV)
    skin_path = args.skin_csv if args.skin_csv is not None else (None if args.skin_dir else default_data_path(data_dir, DEFAULT_SKIN_CSV))
    cyan_path = args.cyan_csv if args.cyan_csv is not None else (None if args.cyan_dir else default_data_path(data_dir, DEFAULT_CYAN_CSV))

    alexa_ssf = load_curve_csv(alexa_path) if alexa_path else BUILTIN_ALEV3_SSF
    imx410_qe = load_curve_csv(imx410_path) if imx410_path else BUILTIN_IMX410_QE
    transmission = load_spd_csv(ir_path, normalize=False) if ir_path else sigmoid_ir_cut(WL, args.ir_cutoff, args.ir_width)
    fp_ssf = imx410_qe * transmission[:, None]
    illum = illuminant_spd(args.illuminant, args.standard_data, args.illuminant_csv)
    cmf = cie_1931_cmf(WL, args.standard_data, args.cmf_csv)
    if args.preset_mode == "material":
        return run_material_mode(args, alexa_ssf, fp_ssf, cmf, {
            "alev3_ssf": source_label(alexa_path, "built-in rough ALEV3 SSF placeholder"),
            "imx410_qe": source_label(imx410_path, "built-in rough IMX410 QE placeholder"),
            "sigma_fp_hot_mirror": source_label(ir_path, f"sigmoid transmission, cutoff {args.ir_cutoff:g}nm width {args.ir_width:g}nm"),
            "cmf": f"CIE 1931 2° via {args.standard_data}",
        })
    skin, skin_source = load_spectra_input(skin_path, args.skin_dir, demo_skin_spectra())
    cyan, cyan_source = load_spectra_input(cyan_path, args.cyan_dir, demo_cyan_spectra())
    profile_spectra = camera_profile_spectra(skin, cyan)
    if args.profile_csv is not None:
        profile_spectra = np.concatenate([profile_spectra, load_spectra_csv(args.profile_csv)], axis=0)
    if args.profile_dir is not None:
        profile_spectra = np.concatenate([profile_spectra, load_spectra_dir(args.profile_dir)], axis=0)
    target_rec2020 = spectra_to_rec2020(profile_spectra, illum, cmf)
    fp_to_rec2020 = camera_to_rec2020_matrix(integrate_response(profile_spectra, illum, fp_ssf), target_rec2020)
    alexa_to_rec2020 = camera_to_rec2020_matrix(integrate_response(profile_spectra, illum, alexa_ssf), target_rec2020)
    skin_scale = args.skin_mask_scale if args.skin_mask_scale is not None else args.mask_scale
    cyan_scale = args.cyan_mask_scale if args.cyan_mask_scale is not None else args.mask_scale
    skin_gain = args.skin_look_gain if args.skin_look_gain is not None else args.look_gain
    cyan_gain = args.cyan_look_gain if args.cyan_look_gain is not None else args.look_gain
    key = f"arri_skin_{args.illuminant.lower()}"
    existing_windows = existing_region_windows(args.out, key) if args.preserve_existing_windows else {}
    cyan_override = controlled_cyan_matrix(args.cyan_red_pull, args.cyan_blue_push) if args.cyan_mode == "cool" else None
    skin_region = build_region(
        "skin", skin, illum, fp_ssf, alexa_ssf, fp_to_rec2020, alexa_to_rec2020,
        skin_scale, skin_gain, args.skin_region_strength,
    )
    cyan_region = build_region(
        "cyan", cyan, illum, fp_ssf, alexa_ssf, fp_to_rec2020, alexa_to_rec2020,
        cyan_scale, cyan_gain, args.cyan_region_strength, matrix_override=cyan_override,
    )
    if existing_windows:
        skin_region = preserve_window(skin_region, existing_windows)
        cyan_region = preserve_window(cyan_region, existing_windows)
    regions = [skin_region, cyan_region]
    if args.background_cool:
        background_region = build_fixed_region(
            "cool_balance",
            controlled_cool_balance_matrix(args.background_red_pull, args.background_blue_push),
            (0.78, 0.82),
            ((0.09, 0.0), (0.0, 0.16)),
            1.0,
            args.background_region_strength,
        )
        if existing_windows:
            background_region = preserve_window(background_region, existing_windows)
        regions.append(background_region)

    payload = {
        "version": 1,
        "transforms": {
            key: {
                "name": key,
                "label": f"ARRI skin prefeed ({args.illuminant})",
                "illuminant": args.illuminant,
                "working_space": "Rec2020",
                "note": (
                    "Rec.2020-domain demo spectral fit from rough ALEV3/IMX410 curves and a sigmoid Sigma fp "
                    "hot-mirror model; look_gain amplifies the fitted residual while preserving neutral axis. "
                    "Replace curves with measured CSV data for production calibration."
                ),
                "regions": regions,
            }
        },
        "sources": {
            "alev3_ssf": source_label(alexa_path, "built-in rough ALEV3 SSF placeholder"),
            "imx410_qe": source_label(imx410_path, "built-in rough IMX410 QE placeholder"),
            "sigma_fp_hot_mirror": source_label(ir_path, f"sigmoid transmission with red cutoff {args.ir_cutoff:g}nm, width {args.ir_width:g}nm."),
            "skin_spectra": skin_source,
            "cyan_spectra": cyan_source,
            "illuminant": source_label(args.illuminant_csv, f"{args.illuminant} via {args.standard_data}") if args.illuminant_csv else f"{args.illuminant} via {args.standard_data}",
            "cmf": source_label(args.cmf_csv, f"CIE 1931 2° via {args.standard_data}") if args.cmf_csv else f"CIE 1931 2° via {args.standard_data}",
            "fit_domain": "camera SSF residual fitted after per-camera constrained Rec.2020 profiling; matrices apply in scene-linear Rec.2020.",
            "look_gain": (
                f"skin={skin_gain:g}, cyan={cyan_gain:g}"
                if args.cyan_mode == "spectral"
                else f"skin={skin_gain:g}, cyan=cool(red_pull={args.cyan_red_pull:g}, blue_push={args.cyan_blue_push:g})"
            ),
            "cyan_mode": args.cyan_mode,
            "background_cool": (
                f"enabled strength={args.background_region_strength:g}, "
                f"red_pull={args.background_red_pull:g}, blue_push={args.background_blue_push:g}"
                if args.background_cool else "disabled"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
