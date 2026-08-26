# SPDX-License-Identifier: GPL-3.0-or-later
"""Route-D phase-1 measurement: is the D55 illuminant assumption expensive?

The Stage A chromaticity field (and the 3x3 observer before it) is fitted
under D55. A real scene may be lit by tungsten, LED or worse; white balance
neutralizes the cast but cannot restore the SPECTRAL difference, so the
film-layer exposures of a white-balanced non-daylight scene are not the
D55-trained model's exposures. WB CCT cannot identify an SPD, so any runtime
tier choice would be an ILLUMINANT ASSUMPTION, never a measurement — this
tool decides whether such tiers are worth carrying at all.

For each stock x illuminant, on the same five folds as the Stage A CV:

    assumed   D55-trained cubic field evaluated on I-lit held-out samples
              (what the shipped runtime does today when the scene was I-lit)
    dedicated I-trained cubic field on I-lit held-out samples
              (the ceiling a declared tier could reach)
    3x3_assumed / 3x3_dedicated - the same pair for the linear observer

The decision number is assumed - dedicated: the recoverable part of the
assumption error. Output: docs/illuminant_tier_cv.json.

Illuminants: A (tungsten, the classic second calibration point), LED-B3
(CIE high-CRI phosphor LED ~4000K), LED-RGB1 (narrow tri-band pressure
case — expected to be bad for EVERY deterministic model; it bounds what a
tier cannot fix and motivates the lowered profile confidence instead).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import fit_chroma_field as fcf  # noqa: E402
import fit_film_curve as ff  # noqa: E402

OUT_PATH = PROJECT_ROOT / "docs" / "illuminant_tier_cv.json"
TIER_ILLUMINANTS = ("A", "LED-B3", "LED-RGB1")
ORDER = 3


def _heldout(xyz_fit, exp_fit, xyz_eval, exp_eval, folds_list):
    """Field trained on (xyz_fit, exp_fit) rows, evaluated on the SAME held-out
    rows of (xyz_eval, exp_eval). Same-illuminant when fit==eval; the
    assumption case trains on D55 pairs and evaluates on I-lit pairs."""
    errs = []
    n = exp_fit.shape[0]
    x, y, lum = fcf.chromaticity(xyz_eval)
    feats = fcf.poly_features(x, y, ORDER)
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        coefs = fcf.fit_field(xyz_fit, exp_fit, train, ORDER)
        pred = lum[:, None] * 2.0 ** (feats @ coefs.T)
        errs.append(
            np.abs(
                np.log2(
                    np.maximum(pred[test], 1e-15)
                    / np.maximum(exp_eval[test], 1e-15)
                )
            )
        )
    return np.concatenate([e.ravel() for e in errs])


def _heldout_3x3(rgb_fit, exp_fit, rgb_eval, exp_eval, folds_list):
    errs = []
    n = exp_fit.shape[0]
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        a = fcf.fit_3x3(rgb_fit, exp_fit, train)
        pred = rgb_eval[test] @ a.T
        errs.append(
            np.abs(
                np.log2(
                    np.maximum(pred, 1e-15)
                    / np.maximum(exp_eval[test], 1e-15)
                )
            )
        )
    return np.concatenate([e.ravel() for e in errs])


def run() -> dict:
    stocks_out = {}
    for name, stock in ff.STOCKS.items():
        xyz_d, rgb_d, exp_d, _m = fcf.stimulus_and_exposures(stock, "D55")
        folds_list = fcf.cv_folds(exp_d.shape[0])
        per_ill = {}
        for ill in TIER_ILLUMINANTS:
            xyz_i, rgb_i, exp_i, _mi = fcf.stimulus_and_exposures(stock, ill)
            assumed = _heldout(xyz_d, exp_d, xyz_i, exp_i, folds_list)
            dedicated = _heldout(xyz_i, exp_i, xyz_i, exp_i, folds_list)
            obs_assumed = _heldout_3x3(rgb_d, exp_d, rgb_i, exp_i, folds_list)
            obs_dedicated = _heldout_3x3(rgb_i, exp_i, rgb_i, exp_i, folds_list)
            per_ill[ill] = {
                "field_assumed_d55": fcf.summarize(assumed),
                "field_dedicated": fcf.summarize(dedicated),
                "3x3_assumed_d55": fcf.summarize(obs_assumed),
                "3x3_dedicated": fcf.summarize(obs_dedicated),
                "recoverable_p99_stop": round(
                    float(
                        np.percentile(assumed, 99.0)
                        - np.percentile(dedicated, 99.0)
                    ),
                    4,
                ),
            }
        stocks_out[name] = per_ill
        a_rec = per_ill["A"]["recoverable_p99_stop"]
        led_rec = per_ill["LED-B3"]["recoverable_p99_stop"]
        print(f"{name:18} recoverable p99: A {a_rec:+.3f}  LED-B3 {led_rec:+.3f}")

    med = {
        ill: round(
            float(
                np.median(
                    [s[ill]["recoverable_p99_stop"] for s in stocks_out.values()]
                )
            ),
            4,
        )
        for ill in TIER_ILLUMINANTS
    }
    return {
        "purpose": (
            "Route-D phase-1: cost of the D55 illuminant assumption per stock "
            "and per illuminant, versus the ceiling of a dedicated tier — "
            "identical folds/anchor/training discipline as the Stage A CV"
        ),
        "illuminants": {
            "A": "CIE A (tungsten 2856K)",
            "LED-B3": "CIE LED-B3 (high-CRI phosphor LED ~4000K)",
            "LED-RGB1": "CIE LED-RGB1 (narrow tri-band; pressure case)",
        },
        "wb_semantics": (
            "training white board under the SAME illuminant anchors row 0 and "
            "the Bradford CAT, mirroring a white-balanced runtime scene; a "
            "runtime tier choice would be an ILLUMINANT ASSUMPTION from "
            "WB/CCT, never a measurement"
        ),
        "folds": fcf.FOLDS,
        "fold_seed": fcf.FOLD_SEED,
        "median_recoverable_p99_stop": med,
        "stocks": stocks_out,
    }


def main() -> int:
    report = run()
    OUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_PATH}")
    print("median recoverable p99:", report["median_recoverable_p99_stop"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
