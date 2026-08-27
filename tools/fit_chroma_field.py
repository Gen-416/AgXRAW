# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage-A chromaticity-field CV: measure whether E = Y * 2^P(x,y) beats the 3x3.

Route-C phase-1 measurement (two-route doctrine, 2026-08-26). The deployed
Stage A maps scene Rec.2020 to per-layer film exposure through a constrained
3x3 observer inverse (build_full_lut.observer_matrix). Its held-out layer
exposure error — observer_p99_stop, 0.47..1.13 stop across shipped stocks —
is the generalization residual of a LINEAR three-channel map on this
190-reflectance set, not a code defect. This tool asks the next question:
how much of it is the LINEARITY? (Whether what remains is the metamerism
floor of ANY deterministic three-channel map is NOT established here —
that needs same-tristimulus/different-layer-exposure spectral pairs or an
independent reflectance library; see the record's residual_caveat.)

The candidate keeps exposure homogeneity by construction:

    E_layer(k * X) = k * E_layer(X)      because      E = Y * 2^P(x, y)

with P a per-layer polynomial in CIE chromaticity fitted in log2 domain
(ridge least squares), white-anchored exactly like the production observer
(the perfect reflector's exposure is reproduced bit-for-bit at the anchor).
Folds, training set (rawtoaces 190 reflectances + white board) and the D55
stimulus construction are IDENTICAL between baseline and candidate — the
comparison measures the model family, nothing else.

RUNTIME-FAITHFUL CV (external review 2026-08-27, F1-F3). The first record
cross-validated the CONTINUOUS polynomial, but what ships is a baked LUT
(training-hull guard, blend band to the 3x3, 256^2 bilinear grid) — a
different operator, and on the training reflectances its p99 differed from
the polynomial's by 0.30 stop median. Every decision number here is now
produced by baking the LUT per fold from the training subset with the
SAME pure function the asset builder uses (bake_lut) and evaluating the
held-out rows through the SAME runtime function
(film_v2_math.chroma_field_log_exposure), so the number describes the
operator that runs. The fit itself is white-anchored BY PARAMETRIZATION
(P = P_w + sum beta_j (f_j - f_j(w)), standardized features; F2) rather
than by patching the intercept after an unconstrained solve, and the fold
split is repeated over N_SEEDS deterministic seeds (F3) so a marginal
stock is decided by adoption frequency, not by one draw.

Output: docs/chroma_field_cv.json. This tool changes no runtime behaviour;
the asset builder reads its adoption decisions from the record.
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


def cv_folds(n_samples: int, seed: int = FOLD_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(np.arange(1, n_samples))  # white anchor never held out
    return np.array_split(idx, FOLDS)


# Repeated K-fold (F3): one draw cannot decide a marginal stock. The seeds are
# deterministic and listed, the adoption is a FREQUENCY over them.
N_SEEDS = 30
SEEDS = tuple(FOLD_SEED + 1000 * k for k in range(N_SEEDS))
ADOPT_FREQUENCY = 2.0 / 3.0


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
    """Monomials WITHOUT the constant term (the anchor supplies it)."""
    cols = []
    for total in range(1, order + 1):
        for i in range(total + 1):
            cols.append(x ** (total - i) * y**i)
    return np.stack(cols, axis=1)


def chromaticity(xyz):
    s = np.maximum(xyz.sum(axis=1), 1e-12)
    return xyz[:, 0] / s, xyz[:, 1] / s, np.maximum(xyz[:, 1], 1e-12)


def fit_field(xyz, exposures, train_idx, order: int, ridge: float = RIDGE) -> dict:
    """White-anchored field by PARAMETRIZATION (F2).

        log2(E/Y) = t_w + sum_j beta_j * (f_j(x,y) - f_j(x_w,y_w)) / s_j

    t_w is the white board's exact value, the features are centred at the
    white chromaticity and scaled by their training std, and ridge acts on
    beta alone — so the anchor holds by construction and the slopes are the
    constrained least-squares slopes, not slopes optimized under a wrong
    intercept and then shifted. Returns the model dict eval_field consumes,
    plus the design condition number for the record.
    """
    x, y, lum = chromaticity(xyz)
    raw = poly_features(x, y, order)
    centre = raw[0].copy()
    g = raw - centre[None, :]
    rows = np.asarray(train_idx, dtype=np.int64)  # row 0 is exact by construction
    scale = np.std(g[rows], axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    a = g[rows] / scale[None, :]
    gram = a.T @ a
    cond = float(np.linalg.cond(gram))
    betas = []
    t_w = np.log2(np.maximum(exposures[0], 1e-15) / lum[0])
    for layer in range(3):
        t = np.log2(np.maximum(exposures[:, layer], 1e-15) / lum)
        b = t[rows] - t_w[layer]
        beta = np.linalg.solve(gram + ridge * np.eye(a.shape[1]), a.T @ b)
        betas.append(beta)
    return {
        "order": int(order),
        "t_w": t_w,
        "beta": np.stack(betas, axis=0),
        "centre": centre,
        "scale": scale,
        "white_xy": (float(x[0]), float(y[0])),
        "cond": cond,
    }


def eval_field(model: dict, x, y):
    """log2(E/Y) per layer at chromaticities (x, y)."""
    raw = poly_features(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), model["order"])
    a = (raw - model["centre"][None, :]) / model["scale"][None, :]
    return model["t_w"][None, :] + a @ model["beta"].T


# --- the deployed operator, as ONE pure function (F1) ---------------------

LUT_N = 256
# Blend band OUTSIDE the training hull: the mask is dilated by DILATE_CELLS
# before the Gaussian so the hull itself (whose vertices are training
# points) keeps the field at ~full weight, and the transition to the 3x3's
# chromaticity response happens in the region no training reflectance
# reaches. sigma 5 (was 2) after the palette fold gate caught beta-amplified
# hue folds on rings crossing the band.
BLEND_SIGMA_CELLS = 5.0
DILATE_CELLS = 10
# The field weight also tapers to ZERO within this many cells of the
# Rec.2020 triangle edge, so the LUT at the gamut boundary is exactly the
# 3x3's own response on the grid: a pixel crossing a channel through zero
# hands over from the LUT to the signed 3x3 (F4) without a step. Without
# the taper the blend band reached the cyan/magenta edges (the training
# hull sits closer to them than 25 cells) and the extrapolated field
# leaked into the hand-over.
EDGE_TAPER_CELLS = 12


def bake_lut(model: dict, xyz_train, m, observer, n: int = LUT_N,
             sigma: float = BLEND_SIGMA_CELLS, dilate: int = DILATE_CELLS):
    """Bake the runtime LUT for a fitted field: (lut[n,n,3] float32-ready,
    domain[4], xyz_from_rec2020[3,3]). Pure: same inputs, same bytes."""
    from scipy.ndimage import binary_dilation, gaussian_filter
    from scipy.spatial import Delaunay

    m_inv = np.linalg.inv(m)
    prim_xyz = np.eye(3) @ m_inv.T
    prim_s = prim_xyz.sum(axis=1)
    tri_x = prim_xyz[:, 0] / prim_s
    tri_y = prim_xyz[:, 1] / prim_s
    pad = 0.02
    x0, x1 = float(tri_x.min() - pad), float(tri_x.max() + pad)
    y0, y1 = float(tri_y.min() - pad), float(tri_y.max() + pad)
    gx = x0 + (x1 - x0) * np.arange(n) / (n - 1)
    gy = y0 + (y1 - y0) * np.arange(n) / (n - 1)
    cx, cy = np.meshgrid(gx, gy, indexing="ij")
    field = eval_field(model, cx.ravel(), cy.ravel()).reshape(n, n, 3)
    # 3x3 chromaticity response on the grid (homogeneous xyz at s=1).
    cz = 1.0 - cx - cy
    xyz_grid = np.stack([cx, cy, cz], axis=-1).reshape(-1, 3)
    rgb_grid = xyz_grid @ m.T
    e_obs = rgb_grid @ np.asarray(observer, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = np.log2(
            np.maximum(e_obs, 1e-12) / np.maximum(cy.ravel(), 1e-12)[:, None]
        ).reshape(n, n, 3)
    obs = np.nan_to_num(obs, nan=0.0, posinf=60.0, neginf=-60.0)
    tx = xyz_train[:, 0] / xyz_train.sum(axis=1)
    ty = xyz_train[:, 1] / xyz_train.sum(axis=1)
    hull = Delaunay(np.stack([tx, ty], axis=1))
    inside = (
        hull.find_simplex(np.stack([cx.ravel(), cy.ravel()], axis=1)) >= 0
    ).reshape(n, n)
    if dilate > 0:
        inside = binary_dilation(inside, iterations=int(dilate))
    w = gaussian_filter(inside.astype(np.float64), sigma=float(sigma))
    # Distance to the triangle edge in cells (barycentric coordinates of the
    # grid chromaticities w.r.t. the primaries), smoothstepped to a taper.
    tri = np.stack([tri_x, tri_y], axis=1)
    t_mat = np.array([[tri[0, 0] - tri[2, 0], tri[1, 0] - tri[2, 0]],
                      [tri[0, 1] - tri[2, 1], tri[1, 1] - tri[2, 1]]])
    rel = np.stack([cx.ravel() - tri[2, 0], cy.ravel() - tri[2, 1]], axis=0)
    lam01 = np.linalg.solve(t_mat, rel)
    lam = np.stack([lam01[0], lam01[1], 1.0 - lam01[0] - lam01[1]], axis=0)
    # cells per unit barycentric coordinate along each edge's altitude
    cell = max((x1 - x0), (y1 - y0)) / (n - 1)
    def _cross2(a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    alt = np.array([
        abs(_cross2(tri[1] - tri[2], tri[0] - tri[2])) / np.linalg.norm(tri[1] - tri[2]),
        abs(_cross2(tri[0] - tri[2], tri[1] - tri[2])) / np.linalg.norm(tri[0] - tri[2]),
        abs(_cross2(tri[0] - tri[1], tri[2] - tri[1])) / np.linalg.norm(tri[0] - tri[1]),
    ])
    dist_cells = np.min(lam * alt[:, None], axis=0) / cell
    tt = np.clip(dist_cells / float(EDGE_TAPER_CELLS), 0.0, 1.0)
    taper = (tt * tt * (3.0 - 2.0 * tt)).reshape(n, n)
    w = (w * taper)[..., None]
    # Anchor the FIELD term (w == 1 at the white chromaticity, deep inside
    # the hull) so a bilinear sample at white is exact while the pure-3x3
    # region stays the 3x3's own response — no global shift, no step at the
    # gamut edge where negative-channel inputs hand over to the signed 3x3.
    wx, wy = model["white_xy"]
    fx = (wx - x0) / (x1 - x0) * (n - 1)
    fy = (wy - y0) / (y1 - y0) * (n - 1)
    ix, iy = min(int(fx), n - 2), min(int(fy), n - 2)
    ax, ay = fx - ix, fy - iy
    at_white = (
        field[ix, iy] * (1 - ax) * (1 - ay)
        + field[ix + 1, iy] * ax * (1 - ay)
        + field[ix, iy + 1] * (1 - ax) * ay
        + field[ix + 1, iy + 1] * ax * ay
    )
    field = field + (model["t_w"] - at_white)[None, None, :]
    lut = w * field + (1.0 - w) * obs
    return lut, np.asarray([x0, x1, y0, y1]), m_inv


# --- runtime-faithful held-out evaluation --------------------------------

LOG10_2 = 0.30102999566398119521


def _runtime_stock(lut, domain, m_inv, observer) -> dict:
    return {
        "observer": np.asarray(observer, dtype=np.float64),
        "chroma_lut": np.asarray(lut, dtype=np.float32).astype(np.float64),
        "chroma_domain": np.asarray(domain, dtype=np.float64),
        "chroma_xyz_from_rec2020": np.asarray(m_inv, dtype=np.float64),
    }


def _truth_log10(exposures, observer):
    a = np.asarray(observer, dtype=np.float64)
    mid = a @ np.full(3, 0.18)
    return np.log10(np.maximum(exposures, 1e-15) / np.maximum(mid, 1e-12)[None, :])


def heldout_errors_runtime(xyz, rgb, exposures, m, folds_list, order: int = 3,
                           ridge: float = RIDGE, use_field: bool = True):
    """Held-out |stop| error of the SHIPPED operator: per fold fit the 3x3
    (and the field, baked to a LUT from the training rows only), then run
    the held-out working-space rgb through film_v2_math.stage_a_log_exposure
    exactly as the renderer does. use_field=False measures the 3x3 alone."""
    from dngscan.film_v2_math import stage_a_log_exposure

    errs = []
    n = exposures.shape[0]
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        observer = fit_3x3(rgb, exposures, train)
        if use_field:
            rows = np.concatenate([[0], train])
            model = fit_field(xyz, exposures, train, order, ridge)
            lut, domain, m_inv = bake_lut(model, xyz[rows], m, observer)
            stock = _runtime_stock(lut, domain, m_inv, observer)
        else:
            stock = {"observer": observer, "chroma_lut": None}
        pred = stage_a_log_exposure(rgb[test], stock)
        truth = _truth_log10(exposures[test], observer)
        errs.append(np.abs(pred - truth) / LOG10_2)
    return np.concatenate([e.ravel() for e in errs])


def heldout_errors_continuous(xyz, exposures, folds_list, order: int = 3,
                              ridge: float = RIDGE):
    """The polynomial itself (no LUT, no hull band) — kept for the audit
    trail and the ridge-sensitivity sweep, NOT a decision number."""
    errs = []
    n = exposures.shape[0]
    x, y, lum = chromaticity(xyz)
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        model = fit_field(xyz, exposures, train, order, ridge)
        pred = lum[:, None] * 2.0 ** eval_field(model, x, y)
        errs.append(np.abs(np.log2(
            np.maximum(pred[test], 1e-15) / np.maximum(exposures[test], 1e-15)
        )))
    return np.concatenate([e.ravel() for e in errs])


P95_ADOPT_RATIO = 0.85


def adopts_once(candidate: dict, baseline: dict, ratio: float = P95_ADOPT_RATIO) -> bool:
    """The owner's margin rule on ONE fold draw: candidate p95 down by at
    least (1 - ratio) AND p99 no worse."""
    return bool(
        candidate["p95_stop"] <= ratio * baseline["p95_stop"]
        and candidate["p99_stop"] <= baseline["p99_stop"]
    )


def adopts(entry: dict) -> bool:
    """The adoption DECISION for a stock record: the margin rule must pass
    on at least ADOPT_FREQUENCY of the repeated fold draws. Marginal stocks
    therefore keep the simpler 3x3."""
    return bool(entry["runtime"]["adopt_frequency"] >= ADOPT_FREQUENCY)


def summarize(err):
    return {
        "p95_stop": round(float(np.percentile(err, 95.0)), 4),
        "p99_stop": round(float(np.percentile(err, 99.0)), 4),
        "max_stop": round(float(err.max()), 4),
    }


def _median_summary(summaries):
    return {
        k: round(float(np.median([s[k] for s in summaries])), 4)
        for k in ("p95_stop", "p99_stop", "max_stop")
    }


def stock_record(stock: dict, seeds=SEEDS) -> dict:
    xyz, rgb, exposures, m = stimulus_and_exposures(stock)
    n = exposures.shape[0]
    field_runs, obs_runs, votes = [], [], []
    cont_runs = []
    conds = []
    for seed in seeds:
        folds_list = cv_folds(n, seed)
        f = summarize(heldout_errors_runtime(xyz, rgb, exposures, m, folds_list, use_field=True))
        o = summarize(heldout_errors_runtime(xyz, rgb, exposures, m, folds_list, use_field=False))
        field_runs.append(f)
        obs_runs.append(o)
        votes.append(adopts_once(f, o))
        cont_runs.append(summarize(heldout_errors_continuous(xyz, exposures, folds_list)))
    for seed in seeds[:3]:
        folds_list = cv_folds(n, seed)
        train = np.setdiff1d(np.arange(1, n), folds_list[0])
        conds.append(fit_field(xyz, exposures, train, 3)["cond"])
    # Ridge sensitivity (continuous evaluation, first seed): how much the
    # regularization strength moves p99 — documented, not tuned.
    folds0 = cv_folds(n, seeds[0])
    sensitivity = {
        f"{lam:g}": summarize(heldout_errors_continuous(xyz, exposures, folds0, 3, lam))["p99_stop"]
        for lam in (1e-6, 1e-4, 1e-2, 1.0)
    }
    freq = float(np.mean(votes))
    return {
        "runtime": {
            "field": _median_summary(field_runs),
            "3x3": _median_summary(obs_runs),
            "adopt_frequency": round(freq, 4),
            "adopt_votes": f"{int(sum(votes))}/{len(votes)}",
            "field_p99_iqr": [
                round(float(np.percentile([r["p99_stop"] for r in field_runs], q)), 4)
                for q in (25, 75)
            ],
        },
        "continuous_poly3": _median_summary(cont_runs),
        "design_cond_standardized": round(float(np.median(conds)), 1),
        "ridge_sensitivity_p99": sensitivity,
    }


def run(seeds=SEEDS) -> dict:
    stocks = {}
    for name, stock in ff.STOCKS.items():
        entry = stock_record(stock, seeds)
        stocks[name] = entry
        r = entry["runtime"]
        print(
            f"{name:18} runtime p99 3x3 {r['3x3']['p99_stop']:.3f} -> field "
            f"{r['field']['p99_stop']:.3f}  adopt {r['adopt_votes']}"
            f" -> {'FIELD' if adopts(entry) else '3x3'}"
        )
    p99_3 = np.array([s["runtime"]["3x3"]["p99_stop"] for s in stocks.values()])
    p99_f = np.array([s["runtime"]["field"]["p99_stop"] for s in stocks.values()])
    p95_3 = np.array([s["runtime"]["3x3"]["p95_stop"] for s in stocks.values()])
    p95_f = np.array([s["runtime"]["field"]["p95_stop"] for s in stocks.values()])
    return {
        "purpose": (
            "Route-C measurement, runtime-faithful: held-out layer-exposure "
            "error of the SHIPPED Stage A operator (per-fold baked LUT through "
            "film_v2_math.stage_a_log_exposure) against the 3x3 observer refit "
            "per fold; repeated K-fold over listed seeds"
        ),
        "training_set": "rawtoaces_training_reflectance.csv (190 samples) + white board, D55",
        "folds": FOLDS,
        "seeds": list(seeds),
        "ridge": RIDGE,
        "lut": {"n": LUT_N, "blend_sigma_cells": BLEND_SIGMA_CELLS, "dilate_cells": DILATE_CELLS,
                "edge_taper_cells": EDGE_TAPER_CELLS},
        "fit": "white anchor by parametrization; features centred at the white chromaticity and std-scaled; ridge on the slopes only",
        "adoption_rule": (
            f"margin rule (p95 <= {P95_ADOPT_RATIO} * 3x3 p95 AND p99 no worse) on each "
            f"fold draw; adopt when it passes on >= {ADOPT_FREQUENCY:.3f} of the seeds"
        ),
        "residual_caveat": (
            "the field's remaining residual is this model family's generalization "
            "residual on this 190-reflectance set; whether it is the metamerism floor "
            "of ANY deterministic three-channel map is NOT established (that needs "
            "same-tristimulus/different-layer-exposure spectral pairs or an "
            "independent reflectance library)"
        ),
        "median_p99_improvement_pct": round(float(np.median((p99_3 - p99_f) / p99_3 * 100.0)), 1),
        "median_p95_improvement_pct": round(float(np.median((p95_3 - p95_f) / p95_3 * 100.0)), 1),
        "stocks_adopting": int(sum(adopts(s) for s in stocks.values())),
        "stocks": stocks,
    }


def main() -> int:
    report = run()
    OUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_PATH}")
    print(
        f"median improvement: p95 {report['median_p95_improvement_pct']}%  "
        f"p99 {report['median_p99_improvement_pct']}%  adopting {report['stocks_adopting']}/{len(report['stocks'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
