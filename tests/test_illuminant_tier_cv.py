# SPDX-License-Identifier: GPL-3.0-or-later
"""Gates on the route-D illuminant record (docs/illuminant_tier_cv.json).

Same doctrine as the Stage A CV gates, plus the two corrections of the
2026-08-27 review: every model in the record is the SHIPPED operator (3x3
refit per fold, or the field baked per fold and run through the runtime
dispatcher — F1), and each claim compares ONE candidate pair (F5): the D55
operator against the tier the shipping rule would actually ship. The fixed
cubic family's number is recorded separately and never mixed into the
decision. The two earlier measurement mistakes (raw cross-illuminant scale;
un-adapted lookup coordinate) stay pinned so they cannot come back.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fit_chroma_field as fcf  # noqa: E402
import fit_film_curve as ff  # noqa: E402
import fit_illuminant_tiers as fit  # noqa: E402

sys.path.pop(0)

RECORD = ROOT / "docs" / "illuminant_tier_cv.json"
CHROMA = ROOT / "docs" / "chroma_field_cv.json"


class IlluminantTierRecordTests(unittest.TestCase):
    STOCK = "portra400"
    ILL = "A"

    @classmethod
    def setUpClass(cls) -> None:
        stock = ff.STOCKS[cls.STOCK]
        cls.xyz_d, cls.rgb_d, cls.exp_d, cls.m_d = fcf.stimulus_and_exposures(stock, "D55")
        cls.xyz_a, cls.rgb_a, cls.exp_a, cls.m_a = fcf.stimulus_and_exposures(stock, cls.ILL)
        cls.record = json.loads(RECORD.read_text())
        chroma = json.loads(CHROMA.read_text())["stocks"][cls.STOCK]
        cls.base = fit.Shipped(cls.xyz_d, cls.rgb_d, cls.exp_d, cls.m_d, fcf.adopts(chroma))

    def test_record_declares_one_candidate_per_claim(self) -> None:
        r = self.record
        self.assertEqual(tuple(r["seeds"]), fcf.SEEDS)
        self.assertEqual(r["family_threshold"], fit.FAMILY_THRESHOLD)
        for ill in fit.TIER_ILLUMINANTS:
            s = r["stocks"][self.STOCK][ill]
            # recoverable is DEFINED on the shipped pair, never on the family number
            self.assertAlmostEqual(
                s["recoverable_p99_stop"],
                s["shipped_assumed"]["p99_stop"] - s["shipped_dedicated"]["p99_stop"],
                places=4,
            )
            self.assertIn(s["tier_model"], ("field", "3x3"))
            chosen = s["field_dedicated"] if s["tier_model"] == "field" else s["3x3_dedicated"]
            self.assertEqual(s["shipped_dedicated"], chosen)
        # the conclusion is generated from the numbers
        for ill in fit.TIER_ILLUMINANTS:
            self.assertIn(f"{r['median'][ill]['stocks_adopting']}/", r["conclusion"])

    def test_stored_record_reproduces_on_the_first_seed(self) -> None:
        stored = self.record["stocks"][self.STOCK][self.ILL]
        folds0 = fcf.cv_folds(self.exp_d.shape[0], fcf.SEEDS[0])
        got = {
            "shipped_assumed": fcf.summarize(
                fit.heldout_assumed(self.base, self.rgb_a, self.exp_a, folds0)
            ),
            "field_dedicated": fcf.summarize(
                fit.heldout_field(self.xyz_a, self.rgb_a, self.exp_a, self.m_a, folds0)
            ),
            "3x3_dedicated": fcf.summarize(fit.heldout_3x3(self.rgb_a, self.exp_a, folds0)),
        }
        # one seed against a 30-seed median: within the seed spread class
        for name, summary in got.items():
            self.assertAlmostEqual(
                stored[name]["p99_stop"], summary["p99_stop"], delta=0.06, msg=name
            )
        for name in ("field_assumed_d55_unadapted", "field_assumed_d55_raw"):
            fn = fit.heldout_unadapted if "unadapted" in name else fit.heldout_raw
            fresh = fcf.summarize(fn(self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, folds0))
            self.assertAlmostEqual(stored[name]["p99_stop"], fresh["p99_stop"], delta=2e-3, msg=name)

    def test_shipped_models_are_the_runtime_operator(self) -> None:
        """The D55 base is the baked LUT run through the runtime dispatcher —
        its normalized prediction at the white board is exactly zero, and it
        differs from the continuous polynomial on I-lit rows (the two are
        different operators; F1)."""
        train = np.setdiff1d(np.arange(1, self.exp_d.shape[0]), fcf.cv_folds(self.exp_d.shape[0])[0])
        pred = self.base.predict(self.rgb_a, train)
        np.testing.assert_array_equal(pred[0], np.zeros(3))
        cont = fit._field_pred_unadapted(self.xyz_d, self.exp_d, self.rgb_a @ np.linalg.inv(self.m_d).T, train)
        self.assertGreater(float(np.abs(pred - cont).max()), 1e-4)

    def test_measurement_mistakes_stay_pinned(self) -> None:
        folds0 = fcf.cv_folds(self.exp_d.shape[0], fcf.SEEDS[0])
        runtime = np.percentile(fit.heldout_assumed(self.base, self.rgb_a, self.exp_a, folds0), 99)
        unadapted = np.percentile(
            fit.heldout_unadapted(self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, folds0), 99
        )
        raw = np.percentile(fit.heldout_raw(self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, folds0), 99)
        self.assertGreater(float(unadapted - runtime), 0.3)
        self.assertGreater(float(raw - unadapted), 0.8)
        np.testing.assert_allclose(self.rgb_a @ np.linalg.inv(self.m_a).T, self.xyz_a, atol=1e-9)

    def test_decision_follows_the_shipping_rule_only(self) -> None:
        """Whatever the conclusion is, it must be the one the shipped-pair
        numbers imply under the family threshold — no family-level number
        may decide it (F5). The LED-RGB1 family number is labelled as such."""
        med = self.record["median"]
        conclusion = self.record["conclusion"]
        implied = [
            med[ill]["stocks_adopting"] >= fit.FAMILY_THRESHOLD for ill in ("A", "LED-B3")
        ]
        if any(implied):
            self.assertIn("earns its place for", conclusion)
        else:
            self.assertIn("no illuminant tier earns its place", conclusion)
        self.assertIn("family number, not the shipping rule", conclusion)

    def test_white_board_stays_anchored_under_every_illuminant(self) -> None:
        for ill in ("D55", "A", "LED-B3", "LED-RGB1"):
            _xyz, rgb, _exp, _m = fcf.stimulus_and_exposures(ff.STOCKS[self.STOCK], ill)
            np.testing.assert_allclose(rgb[0], np.ones(3), atol=5e-7, err_msg=ill)


if __name__ == "__main__":
    unittest.main()
