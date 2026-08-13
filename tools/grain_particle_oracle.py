#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Particle oracle for Grain V2 (optics V2 P4): a dye-cloud Boolean model
whose statistics tell the fast render field what to reproduce.

No measured PSD for any of our stocks is public (P4 survey), so the
reference statistics come from first principles instead: developed
colour-negative grain is a Poisson field of overlapping dye clouds a few
micrometres across (Bayer 1964 random-dot; Trabka/Lawton crowded-emulsion
theory; the IPOL/TOG Boolean renderings use the same object). This tool

  1. simulates such a field at 1 um/cell,
  2. measures its aperture-RMS curve, Selwyn slope and correlation length
     — both raw and after box-averaging to the render grid's 12 um pitch
     (what a 12 um representation CAN reproduce; finer structure shows up
     there as per-cell white noise),
  3. fits the render-side mixture — per-cell white plus Gaussian bands —
     to the 12 um-averaged oracle's aperture-RMS ratios,

and prints the recommended `bands` table for the stock asset, plus the
acceptance windows the tests pin (Selwyn slope, correlation FWHM).

    python tools/grain_particle_oracle.py            # report + fit
    python tools/grain_particle_oracle.py --json OUT # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from dngscan import film_optics_diag as diag

# oracle grid: 1 um cells; 4096 um patch = plenty of 96 um apertures
ORACLE_UM = 1.0
ORACLE_N = 4096
RENDER_PITCH_UM = 12.0

# dye clouds: lognormal radius, median 1.0 um, sigma_ln 0.45 — the
# few-micrometre cloud scale the emulsion literature puts developed
# colour grain at. Density ~1.0 mean via cloud count.
CLOUD_MEDIAN_UM = 1.0
CLOUD_SIGMA_LN = 0.45
MEAN_DENSITY = 1.0


def simulate_oracle(seed: int = 0) -> np.ndarray:
    """One channel of the dye-cloud field at 1 um/cell (density units).

    Millions of clouds make per-cloud stamping unusable; instead the
    Poisson centres are binned into radius classes, each class becomes a
    weighted impulse histogram, and one periodic Gaussian blur per class
    turns the impulses into clouds — exact up to the radius binning."""
    rng = np.random.Generator(np.random.Philox(key=seed))
    # each cloud contributes amp * 2*pi*r^2 integrated density
    amp = 0.35
    mean_area = 2.0 * np.pi * (
        CLOUD_MEDIAN_UM * np.exp(CLOUD_SIGMA_LN ** 2 / 2.0)
    ) ** 2
    area = float(ORACLE_N * ORACLE_N)
    n_clouds = int(round(MEAN_DENSITY * area / (amp * mean_area)))
    radii = np.exp(rng.normal(np.log(CLOUD_MEDIAN_UM), CLOUD_SIGMA_LN, n_clouds))
    xs = rng.integers(0, ORACLE_N, n_clouds)
    ys = rng.integers(0, ORACLE_N, n_clouds)
    edges = np.quantile(radii, np.linspace(0.0, 1.0, 9))
    which = np.clip(np.searchsorted(edges, radii, side="right") - 1, 0, 7)
    field = np.zeros((ORACLE_N, ORACLE_N), dtype=np.float32)
    for b in range(8):
        m = which == b
        if not m.any():
            continue
        r = float(radii[m].mean())
        imp = np.zeros((ORACLE_N, ORACLE_N), dtype=np.float32)
        np.add.at(imp, (ys[m], xs[m]), np.float32(amp * 2.0 * np.pi * r * r))
        field += _blur(imp, r).astype(np.float32)
        del imp
    return field


def box_average(field: np.ndarray, n: int) -> np.ndarray:
    h = field.shape[0] // n * n
    w = field.shape[1] // n * n
    return field[:h, :w].reshape(h // n, n, w // n, n).mean(axis=(1, 3))


def mixture_field(weights: dict[float, float], n: int, pitch_um: float,
                  seed: int = 7) -> np.ndarray:
    """Render-side mixture at the render pitch: {size_um: weight}."""
    rng = np.random.Generator(np.random.Philox(key=seed))
    total = np.zeros((n, n), dtype=np.float64)
    for size_um, wgt in sorted(weights.items()):
        white = rng.standard_normal((n, n))
        sigma = max(size_um / pitch_um, 1e-6)
        # a blur below ~half a cell is indistinguishable from per-cell white
        band = _blur(white, sigma) if sigma > 0.55 else white
        band /= max(band.std(), 1e-12)
        total += wgt * band
    total /= max(total.std(), 1e-12)
    return total


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    # separable Gaussian via FFT-free repeated box would distort; use the
    # same slabbed Gaussian the renderer uses
    from dngscan.film_optics import _gaussian_blur_slabbed

    return _gaussian_blur_slabbed(
        np.ascontiguousarray(a[:, :, None], dtype=np.float32), sigma,
        periodic=True,
    )[:, :, 0].astype(np.float64)


def aperture_curve(field: np.ndarray, cells: tuple[int, ...]) -> list[float]:
    return [float(np.mean(diag.aperture_rms(field, n))) for n in cells]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    oracle = simulate_oracle()
    oracle = oracle - float(oracle.mean())

    # raw statistics at 1 um
    raw_slope = diag.selwyn_slope(oracle, apertures=(12, 24, 48, 96))
    # what the 12 um render grid can see
    seen = box_average(oracle, int(RENDER_PITCH_UM))
    seen /= max(seen.std(), 1e-12)
    seen_slope = diag.selwyn_slope(seen, apertures=(1, 2, 4, 8))
    seen_corr = diag.correlation_length_cells(seen[:, :, None])
    cells = (1, 2, 4, 8)
    target = np.array(aperture_curve(seen, cells))
    target /= target[0]

    # fit the render mixture: per-cell white + 18 um + 36 um bands
    sizes = (RENDER_PITCH_UM * 0.5, 18.0, 36.0)  # 0.5*pitch = per-cell white
    best = None
    for w_mid in np.linspace(0.0, 0.6, 13):
        for w_coarse in np.linspace(0.0, 0.4, 9):
            w_fine = 1.0 - w_mid - w_coarse
            if w_fine <= 0.2:
                continue
            mix = mixture_field(
                {sizes[0]: w_fine, sizes[1]: w_mid, sizes[2]: w_coarse},
                1024, RENDER_PITCH_UM,
            )
            got = np.array(aperture_curve(mix, cells))
            got /= got[0]
            err = float(np.sqrt(np.mean((np.log(got) - np.log(target)) ** 2)))
            if best is None or err < best["err"]:
                best = {
                    "err": err,
                    "weights": {sizes[0]: float(w_fine), sizes[1]: float(w_mid),
                                sizes[2]: float(w_coarse)},
                    "curve": [float(v) for v in got],
                }
    mix = mixture_field(best["weights"], 2048, RENDER_PITCH_UM)
    fit_slope = diag.selwyn_slope(mix, apertures=(1, 2, 4, 8))
    fit_corr = diag.correlation_length_cells(mix[:, :, None])

    report = {
        "oracle": {
            "cell_um": ORACLE_UM, "patch_cells": ORACLE_N,
            "cloud_median_um": CLOUD_MEDIAN_UM, "cloud_sigma_ln": CLOUD_SIGMA_LN,
            "mean_density": MEAN_DENSITY,
            "selwyn_slope_12_96um": float(raw_slope),
        },
        "seen_at_render_pitch": {
            "pitch_um": RENDER_PITCH_UM,
            "selwyn_slope_1_8cells": float(seen_slope),
            "correlation_fwhm_um": float(2.0 * seen_corr * RENDER_PITCH_UM),
            "aperture_rms_ratio_1_2_4_8": [float(v) for v in target],
        },
        "fit": {
            "bands_um_weight": {f"{k:g}": v for k, v in best["weights"].items()},
            "curve_ratio": best["curve"],
            "log_rms_error": best["err"],
            "selwyn_slope_1_8cells": float(fit_slope),
            "correlation_fwhm_um": float(2.0 * fit_corr * RENDER_PITCH_UM),
        },
    }
    print(json.dumps(report, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
