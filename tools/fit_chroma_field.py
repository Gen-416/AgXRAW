# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage-A chromaticity-field CV: measure whether E = Y * 2^P(x,y) beats the 3x3.

Route-C phase-1 measurement (two-route doctrine, 2026-08-26). The deployed
Stage A maps scene Rec.2020 to per-layer film exposure through a constrained
3x3 observer inverse (build_full_lut.observer_matrix). Its held-out layer
exposure error — observer_p99_stop, 0.47..1.13 stop across shipped stocks —
is the metamerism boundary of any LINEAR three-channel map, not a code
defect. This tool asks the next question: how much of that error is the
LINEARITY rather than the metamerism?

The candidate keeps exposure homogeneity by construction:

    E_layer(k * X) = k * E_layer(X)      because      E = Y * 2^P(x, y)

with P a per-layer polynomial in CIE chromaticity fitted in log2 domain
(ridge least squares), white-anchored exactly like the production observer
(the perfect reflector's exposure is reproduced bit-for-bit at the anchor).
Folds, training set (rawtoaces 190 reflectances + white board) and the D55
stimulus construction are IDENTICAL between baseline and candidate — the
comparison measures the model family, nothing else.

Output: docs/chroma_field_cv.json with per-stock held-out p95/p99/max for
the deployed 3x3 (refit per fold with the production NNLS + anchor) and the
polynomial field at orders 2..4. The verdict block states the adoption
criterion the owner set: held-out p95/p99 must drop significantly.

This tool changes NO runtime behaviour. Runtime adoption (asset schema,
Stage A evaluation, out-of-hull guard — a cubic extrapolates dangerously
outside the training chromaticity hull, so the runtime form must clamp or
blend back to the 3x3 there) is a separate, later decision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import build_full_lut as v1  # noqa: E402
import calibrate_skin_matrix as csm  # noqa: E402
import fit_film_curve as ff  # noqa: E402
import spectral_base as sb  # noqa: E402

OUT_PATH = PROJECT_ROOT / "docs" / "chroma_field_cv.json"
FOLDS = 5
FOLD_SEED = 20260826
RIDGE = 1e-6
ORDERS = (2, 3, 4)


def stimulus_and_exposures(stock: dict, illuminant: str = "D55"):
    """Training stimulus exactly as observer_matrix builds it (white row 0).

    ``illuminant`` generalizes the SPD (route D): the white board under the
    SAME illuminant stays row 0, and the Bradford CAT to the working white
    mirrors the runtime, where WB has already neutralized the scene — so a
    non-D55 tier answers "the scene was LIT by I and white-balanced", not
    "the pixels still carry I's cast".
    """
    neg = ff._load_spectral(stock["negative"])
    wl = neg["wl"]
    if illuminant == "D55":
        spd = v1._d55_spd(wl)
    else:
        spd = csm.illuminant_spd(illuminant, wl)
    refl = v1._training_set(wl)
    refl = np.concatenate([np.ones((wl.size, 1)), refl], axis=1)
    stim = refl * spd[:, None]
    exposures = sb.trapezoid(stim[:, :, None] * neg["sens"][:, None, :], wl, axis=0)
    cmf = csm.cie_1931_cmf(sb.intersect_grid(wl))
    keep = np.isin(wl, sb.intersect_grid(wl))
    xyz = sb.trapezoid(stim[keep][:, :, None] * cmf[:, None, :], wl[keep], axis=0)
    white_xyz = xyz[0]
    xyz = xyz / max(float(white_xyz[1]), 1e-12)
    m = sb.XYZ_TO_REC2020 @ sb.bradford_cat(
        white_xyz / max(float(white_xyz[1]), 1e-12)
    )
    rgb = xyz @ m.T
    return xyz, rgb, exposures, m


def cv_folds(n_samples: int):
    rng = np.random.default_rng(FOLD_SEED)
    idx = rng.permutation(np.arange(1, n_samples))  # white anchor never held out
    return np.array_split(idx, FOLDS)


def fit_3x3(rgb, exposures, train_idx):
    """The production procedure, whole: per-layer NNLS + white anchor."""
    rows = np.concatenate([[0], train_idx])
    a = np.stack(
        [v1._nnls_3(rgb[rows], exposures[rows, layer]) for layer in range(3)],
        axis=0,
    )
    for layer in range(3):
        a[layer] *= exposures[0, layer] / max(float(a[layer] @ rgb[0]), 1e-12)
    return a


def poly_features(x, y, order: int):
    cols = [np.ones_like(x)]
    for total in range(1, order + 1):
        for i in range(total + 1):
            cols.append(x ** (total - i) * y**i)
    return np.stack(cols, axis=1)


def chromaticity(xyz):
    s = np.maximum(xyz.sum(axis=1), 1e-12)
    return xyz[:, 0] / s, xyz[:, 1] / s, np.maximum(xyz[:, 1], 1e-12)


def fit_field(xyz, exposures, train_idx, order: int):
    """log2(E/Y) = P(x, y) per layer; ridge LS; white anchor exact."""
    rows = np.concatenate([[0], train_idx])
    x, y, lum = chromaticity(xyz)
    feats = poly_features(x, y, order)
    coefs = []
    for layer in range(3):
        t = np.log2(np.maximum(exposures[:, layer], 1e-15) / lum)
        a = feats[rows]
        b = t[rows]
        g = a.T @ a + RIDGE * np.eye(a.shape[1])
        c = np.linalg.solve(g, a.T @ b)
        c[0] += t[0] - feats[0] @ c  # white anchor: exact at the white board
        coefs.append(c)
    return np.stack(coefs, axis=0)


def heldout_errors_3x3(rgb, exposures, folds_list):
    errs = []
    n = exposures.shape[0]
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        a = fit_3x3(rgb, exposures, train)
        pred = rgb[test] @ a.T
        errs.append(
            np.abs(
                np.log2(
                    np.maximum(pred, 1e-15) / np.maximum(exposures[test], 1e-15)
                )
            )
        )
    return np.concatenate([e.ravel() for e in errs])


def heldout_errors_field(xyz, exposures, folds_list, order: int):
    errs = []
    n = exposures.shape[0]
    x, y, lum = chromaticity(xyz)
    feats = poly_features(x, y, order)
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        coefs = fit_field(xyz, exposures, train, order)
        pred = lum[:, None] * 2.0 ** (feats @ coefs.T)
        errs.append(
            np.abs(
                np.log2(
                    np.maximum(pred[test], 1e-15)
                    / np.maximum(exposures[test], 1e-15)
                )
            )
        )
    return np.concatenate([e.ravel() for e in errs])


P95_ADOPT_RATIO = 0.85


def adopts(candidate: dict, baseline: dict, ratio: float = P95_ADOPT_RATIO) -> bool:
    """The owner's adoption rule (2026-08-26), shared by the asset builder and
    the route-D tier decisions: a candidate replaces the baseline only when
    its held-out p95 drops by at least (1 - ratio) AND its p99 is no worse."""
    return bool(
        candidate["p95_stop"] <= ratio * baseline["p95_stop"]
        and candidate["p99_stop"] <= baseline["p99_stop"]
    )


def summarize(err):
    return {
        "p95_stop": round(float(np.percentile(err, 95.0)), 4),
        "p99_stop": round(float(np.percentile(err, 99.0)), 4),
        "max_stop": round(float(err.max()), 4),
    }


def run() -> dict:
    stocks = {}
    for name, stock in ff.STOCKS.items():
        xyz, rgb, exposures, _m = stimulus_and_exposures(stock)
        folds_list = cv_folds(exposures.shape[0])
        entry = {"3x3": summarize(heldout_errors_3x3(rgb, exposures, folds_list))}
        for order in ORDERS:
            entry[f"poly{order}"] = summarize(
                heldout_errors_field(xyz, exposures, folds_list, order)
            )
        stocks[name] = entry
        print(
            f"{name:18} 3x3 p99 {entry['3x3']['p99_stop']:.3f} -> "
            f"poly3 {entry['poly3']['p99_stop']:.3f}"
        )

    p99_3 = np.array([s["3x3"]["p99_stop"] for s in stocks.values()])
    p99_f = np.array([s["poly3"]["p99_stop"] for s in stocks.values()])
    p95_3 = np.array([s["3x3"]["p95_stop"] for s in stocks.values()])
    p95_f = np.array([s["poly3"]["p95_stop"] for s in stocks.values()])
    report = {
        "purpose": (
            "Route-C phase-1 measurement: held-out layer-exposure error of a "
            "Y-homogeneous chromaticity field E = Y*2^P(x,y) against the "
            "deployed 3x3 observer, identical folds/anchor/training set"
        ),
        "training_set": "rawtoaces_training_reflectance.csv (190 samples) + white board, D55",
        "folds": FOLDS,
        "fold_seed": FOLD_SEED,
        "ridge": RIDGE,
        "adoption_criterion": (
            "owner 2026-08-26: adopt only if held-out p95/p99 drop "
            "significantly; the residual after the field is the metamerism "
            "boundary proper and cannot be removed by any deterministic "
            "three-channel map"
        ),
        "median_p99_improvement_pct": round(
            float(np.median((p99_3 - p99_f) / p99_3 * 100.0)), 1
        ),
        "median_p95_improvement_pct": round(
            float(np.median((p95_3 - p95_f) / p95_3 * 100.0)), 1
        ),
        "runtime_caveat": (
            "no runtime change in this phase; a cubic extrapolates dangerously "
            "outside the training chromaticity hull, so any runtime form must "
            "clamp to the hull or blend back to the 3x3 there"
        ),
        "stocks": stocks,
    }
    return report


def main() -> int:
    report = run()
    OUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_PATH}")
    print(
        f"median improvement: p95 {report['median_p95_improvement_pct']}%  "
        f"p99 {report['median_p99_improvement_pct']}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
