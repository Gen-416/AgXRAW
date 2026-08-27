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

RUNTIME-FAITHFUL OPERATOR (external review 2026-08-27, F1) and ONE
CANDIDATE PER CLAIM (F5): every "shipped" model here — the D55 base and a
would-be dedicated tier alike — is the operator that would actually run:
the 3x3 refit per fold, or the cubic field baked to its LUT per fold with
fit_chroma_field.bake_lut and evaluated through
film_v2_math.stage_a_log_exposure. The decision numbers compare
``shipped_assumed`` (the D55 operator on I-lit rows) with
``shipped_dedicated`` (the operator the route-C rule would ship for I);
``field_dedicated`` (the fixed cubic family regardless of the rule) is
recorded for the family question and is never mixed into the decision.
Fold splits are repeated over the same seeds as the Stage A record (F3).

RESULT: see docs/illuminant_tier_cv.json ``conclusion`` — generated from
the numbers, not written ahead of them.
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
# Family-level threshold (stocks adopting) set before the numbers were seen.
FAMILY_THRESHOLD = 5


# --- predictions, all as normalized log2 layer exposure ---------------------
# "normalized" = minus the model's own white-board (row 0) value per layer,
# exactly what the runtime's mid-grey normalization does to every model.


def _norm(p):
    return p - p[0][None, :]


def _truth(exp_eval):
    return _norm(np.log2(np.maximum(exp_eval, 1e-15)))


LOG10_2 = fcf.LOG10_2


def _field_pred(xyz_fit, exp_fit, m_fit, rgb_eval, train_idx, observer_fit):
    """The SHIPPED field: fitted on the train rows, baked to its LUT with
    the production bake, and run on the white-balanced Rec.2020 rows through
    the runtime dispatcher — the exact operator the renderer executes."""
    from dngscan.film_v2_math import stage_a_log_exposure

    rows = np.concatenate([[0], np.asarray(train_idx)])
    model = fcf.fit_field(xyz_fit, exp_fit, train_idx, ORDER)
    lut, domain, m_inv = fcf.bake_lut(model, xyz_fit[rows], m_fit, observer_fit)
    stock = fcf._runtime_stock(lut, domain, m_inv, observer_fit)
    return _norm(stage_a_log_exposure(rgb_eval, stock) / LOG10_2)


def _field_pred_unadapted(xyz_fit, exp_fit, xyz_eval, train_idx):
    """The second record's coordinate: the CONTINUOUS field evaluated at the
    I-lit chromaticity itself (no adaptation) — kept only to pin the mistake."""
    x, y, lum = fcf.chromaticity(xyz_eval)
    model = fcf.fit_field(xyz_fit, exp_fit, train_idx, ORDER)
    return _norm(np.log2(np.maximum(lum, 1e-15))[:, None] + fcf.eval_field(model, x, y))


def _obs_pred(rgb_fit, exp_fit, rgb_eval, train_idx):
    """The shipped 3x3: the signed observer product, as the runtime does."""
    from dngscan.film_v2_math import layer_log_exposure

    a = fcf.fit_3x3(rgb_fit, exp_fit, train_idx)
    return _norm(layer_log_exposure(rgb_eval, a) / LOG10_2)


class Shipped:
    """One illuminant's shipped Stage A model for a stock: the cubic field
    (baked) if the route-C rule adopted it, the 3x3 otherwise — the same
    choice the asset builder bakes."""

    def __init__(self, xyz, rgb, exp, m, use_field: bool):
        self.xyz, self.rgb, self.exp, self.m = xyz, rgb, exp, m
        self.use_field = use_field

    def predict(self, rgb_eval, train_idx):
        observer = fcf.fit_3x3(self.rgb, self.exp, train_idx)
        if self.use_field:
            return _field_pred(self.xyz, self.exp, self.m, rgb_eval, train_idx, observer)
        return _norm(__import__("dngscan.film_v2_math", fromlist=["layer_log_exposure"]).layer_log_exposure(rgb_eval, observer) / LOG10_2)


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
    return _heldout(
        lambda tr: _field_pred(xyz_i, exp_i, m_i, rgb_i, tr, fcf.fit_3x3(rgb_i, exp_i, tr)),
        exp_i, folds_list,
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
    errs = []
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        model = fcf.fit_field(xyz_d, exp_d, train, ORDER)
        pred = lum[:, None] * 2.0 ** fcf.eval_field(model, x, y)
        errs.append(np.abs(np.log2(
            np.maximum(pred[test], 1e-15) / np.maximum(exp_i[test], 1e-15)
        )))
    return np.concatenate([e.ravel() for e in errs])


def _median_summary(summaries):
    return {
        k: round(float(np.median([x[k] for x in summaries])), 4)
        for k in ("p95_stop", "p99_stop", "max_stop")
    }


def stock_record(name: str, stock: dict, chroma_entry: dict, seeds=fcf.SEEDS) -> dict:
    xyz_d, rgb_d, exp_d, m_d = fcf.stimulus_and_exposures(stock, "D55")
    n = exp_d.shape[0]
    base_field = fcf.adopts(chroma_entry)
    base = Shipped(xyz_d, rgb_d, exp_d, m_d, base_field)
    per_ill = {}
    for ill in TIER_ILLUMINANTS:
        xyz_i, rgb_i, exp_i, m_i = fcf.stimulus_and_exposures(stock, ill)
        assumed, ded_field, ded_obs, tier_votes = [], [], [], []
        for seed in seeds:
            folds_list = fcf.cv_folds(n, seed)
            assumed.append(fcf.summarize(heldout_assumed(base, rgb_i, exp_i, folds_list)))
            ded_field.append(fcf.summarize(heldout_field(xyz_i, rgb_i, exp_i, m_i, folds_list)))
            ded_obs.append(fcf.summarize(heldout_3x3(rgb_i, exp_i, folds_list)))
            tier_votes.append(fcf.adopts_once(ded_field[-1], ded_obs[-1]))
        # Two passes (third review, F3): the tier's candidate is FIXED first
        # by the vote across seeds, and only then is its adoption frequency
        # measured — every seed judges the same model, never a per-seed pick.
        tier_field = float(np.mean(tier_votes)) >= fcf.ADOPT_FREQUENCY
        fixed = ded_field if tier_field else ded_obs
        adopt_votes = [fcf.adopts_once(fx, a) for fx, a in zip(fixed, assumed)]
        dedicated = _median_summary(fixed)
        assumed_med = _median_summary(assumed)
        folds0 = fcf.cv_folds(n, seeds[0])
        per_ill[ill] = {
            "shipped_base": "field" if base_field else "3x3",
            "shipped_assumed": assumed_med,
            "field_dedicated": _median_summary(ded_field),
            "3x3_dedicated": _median_summary(ded_obs),
            "tier_model": "field" if tier_field else "3x3",
            "tier_model_votes": f"{int(sum(tier_votes))}/{len(tier_votes)}",
            "shipped_dedicated": dedicated,
            "tier_adopt_frequency": round(float(np.mean(adopt_votes)), 4),
            "tier_adopted": bool(float(np.mean(adopt_votes)) >= fcf.ADOPT_FREQUENCY),
            "recoverable_p99_stop": round(
                float(assumed_med["p99_stop"] - dedicated["p99_stop"]), 4
            ),
            "field_assumed_d55_unadapted": fcf.summarize(
                heldout_unadapted(xyz_d, exp_d, xyz_i, exp_i, folds0)
            ),
            "field_assumed_d55_raw": fcf.summarize(
                heldout_raw(xyz_d, exp_d, xyz_i, exp_i, folds0)
            ),
        }
    return per_ill


def run(seeds=fcf.SEEDS) -> dict:
    chroma = json.loads(CHROMA_RECORD.read_text())["stocks"]
    stocks_out = {}
    for name, stock in ff.STOCKS.items():
        per_ill = stock_record(name, stock, chroma[name], seeds)
        stocks_out[name] = per_ill
        a, l = per_ill["A"], per_ill["LED-B3"]
        print(
            f"{name:18} A: D55 {a['shipped_assumed']['p99_stop']:.3f} vs dedicated "
            f"{a['shipped_dedicated']['p99_stop']:.3f} ({'adopt' if a['tier_adopted'] else 'no tier'})"
            f"  LED-B3: {l['shipped_assumed']['p99_stop']:.3f} vs "
            f"{l['shipped_dedicated']['p99_stop']:.3f} ({'adopt' if l['tier_adopted'] else 'no tier'})"
        )

    # Family statistics over UNIQUE spectral responses (third review, F5):
    # push presets share their base emulsion and must not vote twice.
    uniq_names = list(fcf.unique_by_response(
        {n: {"response_id": chroma[n]["response_id"]} for n in stocks_out}
    ).keys())
    uniq = {n: stocks_out[n] for n in uniq_names}

    def _median(fn):
        return round(float(np.median([fn(s) for s in uniq.values()])), 4)

    med = {
        ill: {
            "shipped_assumed_p95": _median(lambda s: s[ill]["shipped_assumed"]["p95_stop"]),
            "shipped_assumed_p99": _median(lambda s: s[ill]["shipped_assumed"]["p99_stop"]),
            "shipped_dedicated_p95": _median(lambda s: s[ill]["shipped_dedicated"]["p95_stop"]),
            "shipped_dedicated_p99": _median(lambda s: s[ill]["shipped_dedicated"]["p99_stop"]),
            "field_dedicated_p99": _median(lambda s: s[ill]["field_dedicated"]["p99_stop"]),
            "recoverable_p99_stop": _median(lambda s: s[ill]["recoverable_p99_stop"]),
            "stocks_adopting": int(sum(s[ill]["tier_adopted"] for s in uniq.values())),
            "presets_adopting": int(sum(s[ill]["tier_adopted"] for s in stocks_out.values())),
            "field_assumed_d55_unadapted_p99": _median(
                lambda s: s[ill]["field_assumed_d55_unadapted"]["p99_stop"]
            ),
            "field_assumed_d55_raw_p99": _median(
                lambda s: s[ill]["field_assumed_d55_raw"]["p99_stop"]
            ),
        }
        for ill in TIER_ILLUMINANTS
    }
    n_stocks = len(uniq)
    verdict = []
    for ill in TIER_ILLUMINANTS:
        m = med[ill]
        verdict.append(
            f"{ill}: D55 operator p99 {m['shipped_assumed_p99']} vs would-be tier "
            f"{m['shipped_dedicated_p99']} (recoverable {m['recoverable_p99_stop']:+}), "
            f"{m['stocks_adopting']}/{n_stocks} responses adopt"
            + ("; a fixed cubic family would sit at "
               f"{m['field_dedicated_p99']} — a family number, not the shipping rule"
               "; a narrow tri-band source is UNIDENTIFIABLE from colour temperature, "
               "so this earns a profile-confidence downgrade, not a tier"
               if ill == "LED-RGB1" else "")
        )
    any_family = any(med[i]["stocks_adopting"] >= FAMILY_THRESHOLD for i in ("A", "LED-B3"))
    return {
        "purpose": (
            "Route-D: cost of the D55 illuminant assumption per stock and per "
            "illuminant versus a dedicated same-illuminant model — identical "
            "folds/anchor/training discipline as the Stage A CV, with the "
            "metric the runtime can exhibit and the operator that would ship"
        ),
        "metric": (
            "held-out |log2 layer-exposure error| AFTER subtracting each "
            "model's own white-board value per layer (the runtime normalizes "
            "every model by its own mid-grey), every model being the SHIPPED "
            "operator (3x3 refit per fold, or the field baked per fold with "
            "fit_chroma_field.bake_lut and run through "
            "film_v2_math.stage_a_log_exposure) on the white-balanced Rec.2020 "
            "rows. field_assumed_d55_unadapted (second record) and "
            "field_assumed_d55_raw (first record) are kept for the audit trail; "
            "neither is a decision number."
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
            "fit_chroma_field.adopts_once per fold draw (p95 <= 0.85 * baseline "
            "p95, p99 no worse), adoption when >= ADOPT_FREQUENCY of the seeds "
            "pass: tier_model on the dedicated field vs 3x3; tier_adopted on the "
            "shipped tier vs the shipped D55 operator; a family-level tier needs "
            f">= {FAMILY_THRESHOLD} stocks"
        ),
        "family_threshold": FAMILY_THRESHOLD,
        "unique_responses": len(uniq),
        "statistics_note": (
            "medians and adoption counts are over unique spectral responses "
            f"({len(uniq)} of {len(stocks_out)} presets); per-preset entries kept"
        ),
        "conclusion": (
            ("an illuminant tier earns its place for at least one family" if any_family
             else "no illuminant tier earns its place under the shipping rule")
            + " — " + "; ".join(verdict)
        ),
        "folds": fcf.FOLDS,
        "seeds": list(seeds),
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
    print("conclusion:", report["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
