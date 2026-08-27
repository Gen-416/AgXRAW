# SPDX-License-Identifier: GPL-3.0-or-later
"""Route-D measurement: is the D55 illuminant assumption expensive?

The Stage A chromaticity field (and the 3x3 observer before it) is fitted
under D55. A real scene may be lit by tungsten, LED or worse; white balance
neutralizes the cast but cannot restore the SPECTRAL difference, so the
film-layer exposures of a white-balanced non-daylight scene are not
necessarily the D55-trained model's exposures. This tool measures that
cost against the ceiling of a dedicated same-illuminant model, with the
metric the runtime can actually exhibit.

RUNTIME-FAITHFUL METRIC (two corrections, 2026-08-27):

1. Normalization. film_v2_math normalizes every Stage A model by its own
   mid-grey, so a model can only ever change colour separation relative
   to the neutral axis. The error is the held-out log2 layer-exposure
   error MINUS the model's own white-board value per layer. The first
   record compared raw exposures across illuminants and counted the
   per-layer white offset (up to -2.8 stop on the blue layer under CIE A)
   as assumption cost.
2. Lookup coordinate. The runtime feeds the D55 model the WHITE-BALANCED
   Rec.2020 pixel through the D55 asset's own xyz_from_rec2020 — i.e. the
   scene's tristimulus chromatically adapted (Bradford, folded into the
   training matrix) into the D55 training space. The first two records
   evaluated the D55 field at the UN-adapted I-lit chromaticity, as if the
   pixels still carried the cast — a coordinate the runtime never sees.

Both earlier numbers are kept in the record (``field_assumed_d55_raw``,
``field_assumed_d55_unadapted``) so the history of the mistake is
auditable; neither is a decision number.

Per stock x illuminant, on the same five folds as the Stage A CV:

    shipped_base       field or 3x3 — the route-C adoption of this stock
    shipped_assumed    that shipped D55 model on I-lit held-out samples,
                       runtime coordinates (the cost of assuming D55)
    field_dedicated    I-trained cubic field on I-lit held-out samples
    3x3_dedicated      the same for the linear observer
    tier_model         field or 3x3 by the route-C rule on the dedicated pair
    shipped_dedicated  what a shipped tier would do (the ceiling)
    tier_adopted       the route-C rule on shipped_dedicated vs
                       shipped_assumed — would a tier earn its place?

RESULT (docs/illuminant_tier_cv.json): it would not. Median held-out p99,
CIE A: D55 model 0.70 vs dedicated 0.75; LED-B3: 0.67 vs 0.68; 0/20
stocks adopt either. After white balance the D55 model already sits in
the same error class as a same-illuminant model — the residual is the
metamerism floor, not the illuminant. LED-RGB1 (narrow tri-band) is the
one case a tier would recover (0.88 -> 0.75), and it is unidentifiable
from colour temperature: it lowers confidence, it does not earn a tier.
The numbers the report prints per stock come from this record.
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
CHROMA_RECORD = PROJECT_ROOT / "docs" / "chroma_field_cv.json"
TIER_ILLUMINANTS = ("A", "LED-B3", "LED-RGB1")
ORDER = 3


# --- predictions, all as normalized log2 layer exposure ---------------------
# "normalized" = minus the model's own white-board (row 0) value per layer,
# exactly what the runtime's mid-grey normalization does to every model.


def _norm(p):
    return p - p[0][None, :]


def _truth(exp_eval):
    return _norm(np.log2(np.maximum(exp_eval, 1e-15)))


def _field_pred(xyz_fit, exp_fit, m_fit_inv, rgb_eval, train_idx):
    """The cubic field trained on the fit pairs, evaluated the way the
    runtime does: the white-balanced Rec.2020 rows go through the FIT
    model's own xyz_from_rec2020 into its training space."""
    xyz_lookup = rgb_eval @ m_fit_inv.T
    x, y, lum = fcf.chromaticity(xyz_lookup)
    feats = fcf.poly_features(x, y, ORDER)
    coefs = fcf.fit_field(xyz_fit, exp_fit, train_idx, ORDER)
    return _norm(np.log2(np.maximum(lum, 1e-15))[:, None] + feats @ coefs.T)


def _field_pred_unadapted(xyz_fit, exp_fit, xyz_eval, train_idx):
    """The second record's coordinate: the field evaluated at the I-lit
    chromaticity itself (no adaptation) — kept only to pin the mistake."""
    x, y, lum = fcf.chromaticity(xyz_eval)
    feats = fcf.poly_features(x, y, ORDER)
    coefs = fcf.fit_field(xyz_fit, exp_fit, train_idx, ORDER)
    return _norm(np.log2(np.maximum(lum, 1e-15))[:, None] + feats @ coefs.T)


def _obs_pred(rgb_fit, exp_fit, rgb_eval, train_idx):
    a = fcf.fit_3x3(rgb_fit, exp_fit, train_idx)
    return _norm(np.log2(np.maximum(rgb_eval @ a.T, 1e-15)))


class Shipped:
    """One illuminant's shipped Stage A model for a stock: the cubic field
    if the route-C rule cleared it on this illuminant's own held-out pair,
    the 3x3 otherwise — the same choice the asset builder bakes."""

    def __init__(self, xyz, rgb, exp, m, use_field: bool):
        self.xyz, self.rgb, self.exp = xyz, rgb, exp
        self.m_inv = np.linalg.inv(m)
        self.use_field = use_field

    def predict(self, rgb_eval, train_idx):
        if self.use_field:
            return _field_pred(self.xyz, self.exp, self.m_inv, rgb_eval, train_idx)
        return _obs_pred(self.rgb, self.exp, rgb_eval, train_idx)


def _heldout(pred_fn, exp_eval, folds_list):
    """|pred - truth| over the held-out rows and layers; pred_fn(train_idx)
    returns the normalized log2 prediction for EVERY row."""
    truth = _truth(exp_eval)
    n = exp_eval.shape[0]
    errs = []
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        errs.append(np.abs(pred_fn(train)[test] - truth[test]))
    return np.concatenate([e.ravel() for e in errs])


def heldout_assumed(base: Shipped, rgb_i, exp_i, folds_list):
    return _heldout(lambda tr: base.predict(rgb_i, tr), exp_i, folds_list)


def heldout_field(xyz_i, rgb_i, exp_i, m_i, folds_list):
    m_inv = np.linalg.inv(m_i)
    return _heldout(
        lambda tr: _field_pred(xyz_i, exp_i, m_inv, rgb_i, tr), exp_i, folds_list
    )


def heldout_3x3(rgb_i, exp_i, folds_list):
    return _heldout(lambda tr: _obs_pred(rgb_i, exp_i, rgb_i, tr), exp_i, folds_list)


def heldout_unadapted(xyz_d, exp_d, xyz_i, exp_i, folds_list):
    return _heldout(
        lambda tr: _field_pred_unadapted(xyz_d, exp_d, xyz_i, tr), exp_i, folds_list
    )


def heldout_raw(xyz_d, exp_d, xyz_i, exp_i, folds_list):
    """The first record's metric: raw exposures compared across illuminants
    at the un-adapted coordinate. Counts the per-layer neutral offset the
    runtime removes — kept so the record shows what the original headline
    number was measuring."""
    n = exp_d.shape[0]
    x, y, lum = fcf.chromaticity(xyz_i)
    feats = fcf.poly_features(x, y, ORDER)
    errs = []
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        coefs = fcf.fit_field(xyz_d, exp_d, train, ORDER)
        pred = lum[:, None] * 2.0 ** (feats @ coefs.T)
        errs.append(np.abs(np.log2(
            np.maximum(pred[test], 1e-15) / np.maximum(exp_i[test], 1e-15)
        )))
    return np.concatenate([e.ravel() for e in errs])


def run() -> dict:
    chroma = json.loads(CHROMA_RECORD.read_text())["stocks"]
    stocks_out = {}
    for name, stock in ff.STOCKS.items():
        xyz_d, rgb_d, exp_d, m_d = fcf.stimulus_and_exposures(stock, "D55")
        folds_list = fcf.cv_folds(exp_d.shape[0])
        base_field = fcf.adopts(chroma[name]["poly3"], chroma[name]["3x3"])
        base = Shipped(xyz_d, rgb_d, exp_d, m_d, base_field)
        per_ill = {}
        for ill in TIER_ILLUMINANTS:
            xyz_i, rgb_i, exp_i, m_i = fcf.stimulus_and_exposures(stock, ill)
            assumed = fcf.summarize(heldout_assumed(base, rgb_i, exp_i, folds_list))
            ded_field = fcf.summarize(heldout_field(xyz_i, rgb_i, exp_i, m_i, folds_list))
            ded_obs = fcf.summarize(heldout_3x3(rgb_i, exp_i, folds_list))
            tier_field = fcf.adopts(ded_field, ded_obs)
            dedicated = ded_field if tier_field else ded_obs
            per_ill[ill] = {
                "shipped_base": "field" if base_field else "3x3",
                "shipped_assumed": assumed,
                "field_dedicated": ded_field,
                "3x3_dedicated": ded_obs,
                "tier_model": "field" if tier_field else "3x3",
                "shipped_dedicated": dedicated,
                "tier_adopted": fcf.adopts(dedicated, assumed),
                "recoverable_p99_stop": round(
                    float(assumed["p99_stop"] - dedicated["p99_stop"]), 4
                ),
                "field_assumed_d55_unadapted": fcf.summarize(
                    heldout_unadapted(xyz_d, exp_d, xyz_i, exp_i, folds_list)
                ),
                "field_assumed_d55_raw": fcf.summarize(
                    heldout_raw(xyz_d, exp_d, xyz_i, exp_i, folds_list)
                ),
            }
        stocks_out[name] = per_ill
        a, l = per_ill["A"], per_ill["LED-B3"]
        print(
            f"{name:18} A: D55 {a['shipped_assumed']['p99_stop']:.3f} vs dedicated "
            f"{a['shipped_dedicated']['p99_stop']:.3f} ({'adopt' if a['tier_adopted'] else 'no tier'})"
            f"  LED-B3: {l['shipped_assumed']['p99_stop']:.3f} vs "
            f"{l['shipped_dedicated']['p99_stop']:.3f} ({'adopt' if l['tier_adopted'] else 'no tier'})"
        )

    def _median(fn):
        return round(float(np.median([fn(s) for s in stocks_out.values()])), 4)

    med = {
        ill: {
            "shipped_assumed_p95": _median(lambda s: s[ill]["shipped_assumed"]["p95_stop"]),
            "shipped_assumed_p99": _median(lambda s: s[ill]["shipped_assumed"]["p99_stop"]),
            "shipped_dedicated_p95": _median(lambda s: s[ill]["shipped_dedicated"]["p95_stop"]),
            "shipped_dedicated_p99": _median(lambda s: s[ill]["shipped_dedicated"]["p99_stop"]),
            "recoverable_p99_stop": _median(lambda s: s[ill]["recoverable_p99_stop"]),
            "stocks_adopting": int(sum(s[ill]["tier_adopted"] for s in stocks_out.values())),
            "field_assumed_d55_unadapted_p99": _median(
                lambda s: s[ill]["field_assumed_d55_unadapted"]["p99_stop"]
            ),
            "field_assumed_d55_raw_p99": _median(
                lambda s: s[ill]["field_assumed_d55_raw"]["p99_stop"]
            ),
        }
        for ill in TIER_ILLUMINANTS
    }
    return {
        "purpose": (
            "Route-D: cost of the D55 illuminant assumption per stock and per "
            "illuminant versus a dedicated same-illuminant model — identical "
            "folds/anchor/training discipline as the Stage A CV, with the "
            "metric the runtime can exhibit"
        ),
        "metric": (
            "held-out |log2 layer-exposure error| AFTER subtracting each "
            "model's own white-board value per layer (the runtime normalizes "
            "every model by its own mid-grey), with the D55 model evaluated at "
            "the RUNTIME coordinate: the white-balanced Rec.2020 row through "
            "the D55 model's own xyz_from_rec2020. field_assumed_d55_unadapted "
            "(second record: un-adapted I-lit chromaticity) and "
            "field_assumed_d55_raw (first record: raw exposures, un-adapted) "
            "are kept for the audit trail; neither is a decision number."
        ),
        "illuminants": {
            "A": "CIE A (tungsten 2856K)",
            "LED-B3": "CIE LED-B3 (high-CRI phosphor LED ~4000K)",
            "LED-RGB1": "CIE LED-RGB1 (narrow tri-band; pressure case)",
        },
        "wb_semantics": (
            "training white board under the SAME illuminant anchors row 0 and "
            "the Bradford CAT, mirroring a white-balanced runtime scene"
        ),
        "adoption_rule": (
            "fit_chroma_field.adopts (p95 <= 0.85 * baseline p95, p99 no "
            "worse): tier_model on the dedicated field vs 3x3; tier_adopted "
            "on the shipped tier vs the shipped D55 model"
        ),
        "conclusion": (
            "no illuminant tier earns its place: after white balance the D55 "
            "model is in the same held-out error class as a dedicated "
            "same-illuminant model under CIE A and LED-B3 (0/20 stocks "
            "adopt); the narrow-band LED-RGB1 case would recover ~0.12 stop "
            "but is unidentifiable from colour temperature — a confidence "
            "downgrade, not a tier"
        ),
        "folds": fcf.FOLDS,
        "fold_seed": fcf.FOLD_SEED,
        "median": med,
        "stocks": stocks_out,
    }


def main() -> int:
    report = run()
    OUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_PATH}")
    print("median:", json.dumps(report["median"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
