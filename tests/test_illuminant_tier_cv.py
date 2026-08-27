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
            self.assertIn(f"{r['median'][ill]['responses_adopting']}/", r["conclusion"])
        # third review F3/F5: adoption is judged on the FIXED candidate, and
        # family statistics count each spectral response once
        self.assertIn("unique_responses", r)
        self.assertLess(r["unique_responses"], len(r["stocks"]))
        for ill in fit.TIER_ILLUMINANTS:
            self.assertIn("presets_adopting", r["median"][ill])

    def test_decide_tier_fixes_the_candidate_before_counting(self) -> None:
        """Pure-function gate (fourth review, F2): constructed per-seed
        summaries on which a per-seed pick and a fixed candidate DISAGREE.
        Seeds 0-1: the field wins the tier vote and beats the assumed model;
        seed 2: the field loses the tier vote but the 3x3 would beat the
        assumed model. Fixed candidate = field (2/3 votes) -> adoption 2/3;
        a per-seed pick would count seed 2's 3x3 win and report 3/3."""
        good = {"p95_stop": 0.20, "p99_stop": 0.40, "max_stop": 0.5}
        bad = {"p95_stop": 0.40, "p99_stop": 0.80, "max_stop": 1.0}
        assumed = [bad, bad, bad]
        ded_field = [good, good, bad]
        ded_obs = [bad, bad, good]
        d = fit.decide_tier(assumed, ded_field, ded_obs)
        self.assertEqual(d["tier_model"], "field")
        self.assertEqual(d["tier_model_votes"], "2/3")
        self.assertAlmostEqual(d["tier_adopt_frequency"], 2 / 3, places=4)
        # ... and the mirror case: the 3x3 wins the vote, its own losses count.
        d2 = fit.decide_tier(assumed, [bad, bad, good], [good, good, bad])
        self.assertEqual(d2["tier_model"], "3x3")
        self.assertAlmostEqual(d2["tier_adopt_frequency"], 2 / 3, places=4)

    def test_record_frequencies_are_the_fixed_candidate_ones(self) -> None:
        """The stored frequency of every (stock, illuminant) must equal
        decide_tier on the stored per-seed summaries, and the record must
        contain at least one pair where a per-seed pick would have given a
        different number — so a regression to mixing cannot hide."""
        differs = 0
        for name, per_ill in self.record["stocks"].items():
            for ill, s in per_ill.items():
                ps = s["per_seed"]
                d = fit.decide_tier(ps["assumed"], ps["field_dedicated"], ps["3x3_dedicated"])
                self.assertEqual(d["tier_model"], s["tier_model"], f"{name}/{ill}")
                self.assertAlmostEqual(
                    d["tier_adopt_frequency"], s["tier_adopt_frequency"], places=4,
                    msg=f"{name}/{ill}",
                )
                mixed = np.mean([
                    fcf.adopts_once(f if fcf.adopts_once(f, o) else o, a)
                    for a, f, o in zip(ps["assumed"], ps["field_dedicated"], ps["3x3_dedicated"])
                ])
                if abs(float(mixed) - s["tier_adopt_frequency"]) > 1e-9:
                    differs += 1
        self.assertGreater(differs, 0, "record cannot distinguish fixed from per-seed candidates")

    def test_white_board_stays_anchored_under_every_illuminant(self) -> None:
        for ill in ("D55", "A", "LED-B3", "LED-RGB1"):
            _xyz, rgb, _exp, _m = fcf.stimulus_and_exposures(ff.STOCKS[self.STOCK], ill)
            np.testing.assert_allclose(rgb[0], np.ones(3), atol=5e-7, err_msg=ill)


if __name__ == "__main__":
    unittest.main()
