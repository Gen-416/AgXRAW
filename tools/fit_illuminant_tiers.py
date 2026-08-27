# SPDX-License-Identifier: GPL-3.0-or-later
"""Route-D measurement: is the D55 illuminant assumption expensive, and how
should the runtime move between illuminant tiers?

The Stage A chromaticity field (and the 3x3 observer before it) is fitted
under D55. A real scene may be lit by tungsten, LED or worse; white balance
neutralizes the cast but cannot restore the SPECTRAL difference, so the
film-layer exposures of a white-balanced non-daylight scene are not the
D55-trained model's exposures. WB CCT cannot identify an SPD, so any runtime
tier choice is an ILLUMINANT ASSUMPTION, never a measurement.

METRIC (runtime-faithful, phase-1 correction 2026-08-27). The runtime
(film_v2_math) normalizes every Stage A model by its OWN mid-grey exposure,
so a tier can only ever change COLOUR SEPARATION relative to the neutral
axis — never the per-layer neutral exposure. The error that matters is
therefore the held-out log2 layer-exposure error MINUS its own white-board
value per layer. The first phase-1 record compared raw exposures across
illuminants and so counted the per-layer white offset (up to -2.8 stop on
the blue layer under CIE A) as "assumption cost"; that number is kept as
``*_raw`` for the record but is NOT a decision number.

Per stock x illuminant, on the same five folds as the Stage A CV:

    field_assumed_d55  D55-trained cubic field on I-lit held-out samples
    field_dedicated    I-trained cubic field on I-lit held-out samples
    3x3_*              the same pair for the linear observer
    shipped_assumed    what the shipped D55 base does on I-lit samples (field
                       or 3x3 per the route-C adoption of this stock)
    shipped_dedicated  what a shipped tier would do (field or 3x3 per the
                       SAME adoption rule applied to the dedicated pair)
    tier_adopted       the TAIL adoption rule (p99 -15%, p95 no worse) on
                       shipped_dedicated vs shipped_assumed — the per-stock
                       decision the builder reads. p99 decides because the
                       assumption's cost lives in the tail (saturated
                       colours), not the body.

INTERPOLATION ORACLE. The runtime resolves a declared tungsten-family WB
into a weight on the A tier (reciprocal-CCT between 2856K and 5503K) and
blends the tiers' normalized log exposures. Phase 1 only measured the
endpoints; this record measures bare Planckian radiators at 3000–4500K
against direct spectral integration, comparing d55_only / log_blend (the
runtime rule) / linear_blend / the best log weight / a dedicated field
trained at that CCT (the family ceiling). Output: docs/illuminant_tier_cv.json.
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
RUNTIME_TIERS = {"A": "a", "LED-B3": "led"}
INTERP_CCTS = (3000.0, 3200.0, 3400.0, 4000.0, 4500.0)
CCT_D55 = 5503.0
CCT_A = 2856.0
ORDER = 3
W_GRID = np.round(np.linspace(0.0, 1.0, 21), 4)


def tungsten_weight(cct: float) -> float:
    """The runtime rule (tone.build_render_plan): reciprocal-CCT position of
    the declared white between D55 and CIE A, clamped to [0, 1]."""
    w = (1.0 / cct - 1.0 / CCT_D55) / (1.0 / CCT_A - 1.0 / CCT_D55)
    return float(min(max(w, 0.0), 1.0))


# --- predictions, all as normalized log2 layer exposure ---------------------
# "normalized" = minus the model's own white-board (row 0) value per layer,
# exactly what the runtime's mid-grey normalization does to every tier.


def _truth(exp_eval):
    t = np.log2(np.maximum(exp_eval, 1e-15))
    return t - t[0][None, :]


def _field_pred(xyz_fit, exp_fit, xyz_eval, train_idx):
    x, y, lum = fcf.chromaticity(xyz_eval)
    feats = fcf.poly_features(x, y, ORDER)
    coefs = fcf.fit_field(xyz_fit, exp_fit, train_idx, ORDER)
    p = np.log2(np.maximum(lum, 1e-15))[:, None] + feats @ coefs.T
    return p - p[0][None, :]


def _obs_pred(rgb_fit, exp_fit, rgb_eval, train_idx):
    a = fcf.fit_3x3(rgb_fit, exp_fit, train_idx)
    p = np.log2(np.maximum(rgb_eval @ a.T, 1e-15))
    return p - p[0][None, :]


def _heldout(xyz_fit, exp_fit, xyz_eval, exp_eval, folds_list):
    """Cubic field trained on the fit pairs, evaluated on the held-out rows of
    the eval pairs, colour-separation error (normalized). Same-illuminant
    when fit==eval; the assumption case trains on D55 pairs and evaluates on
    I-lit pairs."""
    truth = _truth(exp_eval)
    n = exp_fit.shape[0]
    errs = []
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        pred = _field_pred(xyz_fit, exp_fit, xyz_eval, train)
        errs.append(np.abs(pred[test] - truth[test]))
    return np.concatenate([e.ravel() for e in errs])


def _heldout_3x3(rgb_fit, exp_fit, rgb_eval, exp_eval, folds_list):
    truth = _truth(exp_eval)
    n = exp_fit.shape[0]
    errs = []
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        pred = _obs_pred(rgb_fit, exp_fit, rgb_eval, train)
        errs.append(np.abs(pred[test] - truth[test]))
    return np.concatenate([e.ravel() for e in errs])


def _heldout_raw(xyz_fit, exp_fit, xyz_eval, exp_eval, folds_list):
    """The first phase-1 metric: raw exposures compared across illuminants.
    Counts the per-layer neutral offset the runtime removes — kept only so
    the record shows what the original headline number was measuring."""
    n = exp_fit.shape[0]
    x, y, lum = fcf.chromaticity(xyz_eval)
    feats = fcf.poly_features(x, y, ORDER)
    errs = []
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        coefs = fcf.fit_field(xyz_fit, exp_fit, train, ORDER)
        pred = lum[:, None] * 2.0 ** (feats @ coefs.T)
        errs.append(np.abs(np.log2(
            np.maximum(pred[test], 1e-15) / np.maximum(exp_eval[test], 1e-15)
        )))
    return np.concatenate([e.ravel() for e in errs])


class _Shipped:
    """One illuminant's shipped Stage A model for a stock: the cubic field if
    the adoption rule cleared it on this illuminant's own held-out pair, the
    3x3 otherwise — the same choice the asset builder bakes."""

    def __init__(self, xyz, rgb, exp, use_field: bool):
        self.xyz, self.rgb, self.exp, self.use_field = xyz, rgb, exp, use_field

    def predict(self, xyz_eval, rgb_eval, train_idx):
        if self.use_field:
            return _field_pred(self.xyz, self.exp, xyz_eval, train_idx)
        return _obs_pred(self.rgb, self.exp, rgb_eval, train_idx)


def _heldout_model(model: _Shipped, xyz_eval, rgb_eval, exp_eval, folds_list):
    truth = _truth(exp_eval)
    n = exp_eval.shape[0]
    errs = []
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        pred = model.predict(xyz_eval, rgb_eval, train)
        errs.append(np.abs(pred[test] - truth[test]))
    return np.concatenate([e.ravel() for e in errs])


def _blend_errors(base: _Shipped, tier: _Shipped, xyz_eval, rgb_eval, exp_eval,
                  folds_list, w: float, domain: str):
    """The runtime blend of two shipped tiers at weight w on the tier, in the
    log domain (what film_v2_math does) or the linear exposure domain."""
    truth = _truth(exp_eval)
    n = exp_eval.shape[0]
    errs = []
    for test in folds_list:
        train = np.setdiff1d(np.arange(1, n), test)
        pb = base.predict(xyz_eval, rgb_eval, train)
        pt = tier.predict(xyz_eval, rgb_eval, train)
        if domain == "log":
            pred = w * pt + (1.0 - w) * pb
        else:
            pred = np.log2(w * np.exp2(pt) + (1.0 - w) * np.exp2(pb))
        errs.append(np.abs(pred[test] - truth[test]))
    return np.concatenate([e.ravel() for e in errs])


def run() -> dict:
    chroma = json.loads(CHROMA_RECORD.read_text())["stocks"]
    stocks_out = {}
    interp_out = {}
    for name, stock in ff.STOCKS.items():
        xyz_d, rgb_d, exp_d, _m = fcf.stimulus_and_exposures(stock, "D55")
        folds_list = fcf.cv_folds(exp_d.shape[0])
        base_field = fcf.adopts(chroma[name]["poly3"], chroma[name]["3x3"])
        base = _Shipped(xyz_d, rgb_d, exp_d, base_field)
        per_ill = {}
        tiers = {}
        for ill in TIER_ILLUMINANTS:
            xyz_i, rgb_i, exp_i, _mi = fcf.stimulus_and_exposures(stock, ill)
            field_assumed = _heldout(xyz_d, exp_d, xyz_i, exp_i, folds_list)
            field_dedicated = _heldout(xyz_i, exp_i, xyz_i, exp_i, folds_list)
            obs_assumed = _heldout_3x3(rgb_d, exp_d, rgb_i, exp_i, folds_list)
            obs_dedicated = _heldout_3x3(rgb_i, exp_i, rgb_i, exp_i, folds_list)
            ded_field_s = fcf.summarize(field_dedicated)
            ded_obs_s = fcf.summarize(obs_dedicated)
            tier_field = fcf.adopts(ded_field_s, ded_obs_s)
            tier = _Shipped(xyz_i, rgb_i, exp_i, tier_field)
            tiers[ill] = tier
            shipped_assumed = fcf.summarize(
                _heldout_model(base, xyz_i, rgb_i, exp_i, folds_list)
            )
            shipped_dedicated = ded_field_s if tier_field else ded_obs_s
            per_ill[ill] = {
                "field_assumed_d55": fcf.summarize(field_assumed),
                "field_dedicated": ded_field_s,
                "3x3_assumed_d55": fcf.summarize(obs_assumed),
                "3x3_dedicated": ded_obs_s,
                "field_assumed_d55_raw": fcf.summarize(
                    _heldout_raw(xyz_d, exp_d, xyz_i, exp_i, folds_list)
                ),
                "shipped_base": "field" if base_field else "3x3",
                "shipped_assumed": shipped_assumed,
                "tier_model": "field" if tier_field else "3x3",
                "shipped_dedicated": shipped_dedicated,
                "tier_adopted": fcf.adopts_tail(shipped_dedicated, shipped_assumed),
                "recoverable_p99_stop": round(
                    float(shipped_assumed["p99_stop"] - shipped_dedicated["p99_stop"]),
                    4,
                ),
            }
        stocks_out[name] = per_ill

        # Interpolation oracle: bare Planckian radiators between the tiers.
        per_cct = {}
        tier_a = tiers["A"]
        for cct in INTERP_CCTS:
            xyz_t, rgb_t, exp_t, _mt = fcf.stimulus_and_exposures(
                stock, f"BB{int(cct)}K"
            )
            w_rule = tungsten_weight(cct)
            log_grid = {
                float(w): fcf.summarize(_blend_errors(
                    base, tier_a, xyz_t, rgb_t, exp_t, folds_list, float(w), "log"
                ))
                for w in W_GRID
            }
            w_opt = min(log_grid, key=lambda w: (log_grid[w]["p99_stop"], log_grid[w]["p95_stop"]))
            per_cct[f"{int(cct)}K"] = {
                "w_rule": round(w_rule, 4),
                "d55_only": fcf.summarize(
                    _heldout_model(base, xyz_t, rgb_t, exp_t, folds_list)
                ),
                "a_only": fcf.summarize(
                    _heldout_model(tier_a, xyz_t, rgb_t, exp_t, folds_list)
                ),
                "log_blend_rule": fcf.summarize(_blend_errors(
                    base, tier_a, xyz_t, rgb_t, exp_t, folds_list, w_rule, "log"
                )),
                "linear_blend_rule": fcf.summarize(_blend_errors(
                    base, tier_a, xyz_t, rgb_t, exp_t, folds_list, w_rule, "linear"
                )),
                "w_opt_log": round(float(w_opt), 4),
                "log_blend_opt": log_grid[w_opt],
                "dedicated_field": fcf.summarize(
                    _heldout(xyz_t, exp_t, xyz_t, exp_t, folds_list)
                ),
            }
        interp_out[name] = per_cct
        a_rec = per_ill["A"]["recoverable_p99_stop"]
        led_rec = per_ill["LED-B3"]["recoverable_p99_stop"]
        r32 = per_cct["3200K"]
        print(
            f"{name:18} recoverable p99: A {a_rec:+.3f} ({'adopt' if per_ill['A']['tier_adopted'] else 'keep D55'})"
            f"  LED-B3 {led_rec:+.3f} ({'adopt' if per_ill['LED-B3']['tier_adopted'] else 'keep D55'})"
            f" | 3200K p99: d55 {r32['d55_only']['p99_stop']:.3f}"
            f" log {r32['log_blend_rule']['p99_stop']:.3f}"
            f" lin {r32['linear_blend_rule']['p99_stop']:.3f}"
            f" opt(w={r32['w_opt_log']:.2f}) {r32['log_blend_opt']['p99_stop']:.3f}"
            f" ded {r32['dedicated_field']['p99_stop']:.3f}"
        )

    def _median(fn):
        return round(float(np.median([fn(s) for s in stocks_out.values()])), 4)

    med = {
        ill: {
            "shipped_assumed_p99": _median(lambda s: s[ill]["shipped_assumed"]["p99_stop"]),
            "shipped_dedicated_p99": _median(lambda s: s[ill]["shipped_dedicated"]["p99_stop"]),
            "recoverable_p99_stop": _median(lambda s: s[ill]["recoverable_p99_stop"]),
            "field_assumed_d55_raw_p99": _median(lambda s: s[ill]["field_assumed_d55_raw"]["p99_stop"]),
            "stocks_adopting": int(sum(s[ill]["tier_adopted"] for s in stocks_out.values())),
        }
        for ill in TIER_ILLUMINANTS
    }
    interp_med = {}
    for cct in INTERP_CCTS:
        key = f"{int(cct)}K"
        interp_med[key] = {
            m: round(float(np.median([interp_out[s][key][m]["p99_stop"] for s in interp_out])), 4)
            for m in ("d55_only", "a_only", "log_blend_rule", "linear_blend_rule", "log_blend_opt", "dedicated_field")
        }
        interp_med[key]["w_rule"] = round(tungsten_weight(cct), 4)
        interp_med[key]["w_opt_log_median"] = round(
            float(np.median([interp_out[s][key]["w_opt_log"] for s in interp_out])), 4
        )
    return {
        "purpose": (
            "Route-D: cost of the D55 illuminant assumption per stock and per "
            "illuminant versus a dedicated tier, plus the intermediate-CCT "
            "interpolation oracle — identical folds/anchor/training discipline "
            "as the Stage A CV"
        ),
        "metric": (
            "held-out |log2 layer-exposure error| AFTER subtracting each "
            "model's own white-board value per layer (colour-separation "
            "error). This is what the runtime can exhibit: film_v2_math "
            "normalizes every tier by its own mid-grey, so the per-layer "
            "neutral offset between illuminants never reaches the image. "
            "*_raw keeps the un-normalized first-phase number for the record."
        ),
        "illuminants": {
            "A": "CIE A (tungsten 2856K)",
            "LED-B3": "CIE LED-B3 (high-CRI phosphor LED ~4000K)",
            "LED-RGB1": "CIE LED-RGB1 (narrow tri-band; pressure case)",
            "BB<T>K": "bare Planckian radiator at T kelvin (interpolation oracle)",
        },
        "wb_semantics": (
            "training white board under the SAME illuminant anchors row 0 and "
            "the Bradford CAT, mirroring a white-balanced runtime scene; a "
            "runtime tier choice is an ILLUMINANT ASSUMPTION from WB/CCT, "
            "never a measurement"
        ),
        "adoption_rule": (
            "tier_model: fit_chroma_field.adopts (p95 <= 0.85 * baseline p95, "
            "p99 no worse) on the dedicated field vs the dedicated 3x3 — the "
            "route-C rule. tier_adopted: fit_chroma_field.adopts_tail (p99 <= "
            "0.85 * baseline p99, p95 no worse) on the shipped tier vs the "
            "shipped D55 base — p99 decides because the assumption's cost is a "
            "tail phenomenon (see median.A: p99 1.14 vs 0.73, p95 0.51 vs 0.40)"
        ),
        "interpolation_rule": (
            "w = clamp((1/T - 1/5503) / (1/2856 - 1/5503), 0, 1) on the A tier; "
            "tiers blend in normalized log-exposure domain (runtime)"
        ),
        "folds": fcf.FOLDS,
        "fold_seed": fcf.FOLD_SEED,
        "median": med,
        "interpolation_median_p99": interp_med,
        "stocks": stocks_out,
        "interpolation": interp_out,
    }


def main() -> int:
    report = run()
    OUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_PATH}")
    print("median:", json.dumps(report["median"], indent=1))
    print("interpolation median p99:", json.dumps(report["interpolation_median_p99"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
