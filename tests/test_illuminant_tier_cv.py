# SPDX-License-Identifier: GPL-3.0-or-later
"""Gates on the route-D illuminant-tier record (measurement + interpolation
oracle, docs/illuminant_tier_cv.json).

Same doctrine as the Stage A CV gates: the stored record must reproduce
from the current spectral assets, and the decision claims must survive any
re-fit — with the runtime-faithful metric (each model normalized by its own
white board, the way film_v2_math normalizes every tier by its own
mid-grey). The first phase-1 record compared raw exposures across
illuminants and counted the per-layer neutral offset as assumption cost;
that mistake is pinned here so it cannot come back.
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


class IlluminantTierRecordTests(unittest.TestCase):
    STOCK = "portra400"
    ILL = "A"

    @classmethod
    def setUpClass(cls) -> None:
        stock = ff.STOCKS[cls.STOCK]
        cls.xyz_d, cls.rgb_d, cls.exp_d, _ = fcf.stimulus_and_exposures(stock, "D55")
        cls.xyz_a, cls.rgb_a, cls.exp_a, _ = fcf.stimulus_and_exposures(stock, cls.ILL)
        cls.folds = fcf.cv_folds(cls.exp_d.shape[0])
        cls.record = json.loads(RECORD.read_text())

    def test_stored_record_reproduces_for_the_probe_pair(self) -> None:
        stored = self.record["stocks"][self.STOCK][self.ILL]
        assumed = fcf.summarize(
            fit._heldout(self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, self.folds)
        )
        dedicated = fcf.summarize(
            fit._heldout(self.xyz_a, self.exp_a, self.xyz_a, self.exp_a, self.folds)
        )
        raw = fcf.summarize(
            fit._heldout_raw(self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, self.folds)
        )
        for key in ("p95_stop", "p99_stop"):
            self.assertAlmostEqual(stored["field_assumed_d55"][key], assumed[key], delta=2e-3)
            self.assertAlmostEqual(stored["field_dedicated"][key], dedicated[key], delta=2e-3)
            self.assertAlmostEqual(stored["field_assumed_d55_raw"][key], raw[key], delta=2e-3)

    def test_metric_is_runtime_faithful(self) -> None:
        """The decision metric subtracts each model's own white-board value
        per layer (row 0 error is exactly zero); the raw cross-illuminant
        comparison carries the neutral offset the runtime removes — under
        CIE A that offset alone is over a stop at p99."""
        train = np.setdiff1d(np.arange(1, self.exp_d.shape[0]), self.folds[0])
        pred = fit._field_pred(self.xyz_d, self.exp_d, self.xyz_a, train)
        np.testing.assert_array_equal(pred[0], np.zeros(3))
        raw = np.percentile(
            fit._heldout_raw(self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, self.folds), 99
        )
        norm = np.percentile(
            fit._heldout(self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, self.folds), 99
        )
        self.assertGreater(float(raw - norm), 0.8)

    def test_tungsten_cost_is_a_tail_phenomenon_and_recoverable(self) -> None:
        """Decision claims of route D: assuming D55 on a tungsten-lit
        (white-balanced) scene costs about a stop at p99 in colour
        separation, a dedicated tier recovers most of it, the cost lives in
        the tail (p99 ratio > p95 ratio — why tiers adopt on p99), and
        nearly every stock adopts."""
        med = self.record["median"]["A"]
        self.assertGreater(med["shipped_assumed_p99"], 1.0)
        self.assertLess(med["shipped_dedicated_p99"], 0.8)
        self.assertGreaterEqual(med["recoverable_p99_stop"], 0.3)
        self.assertGreaterEqual(med["stocks_adopting"], 18)
        stocks = self.record["stocks"]
        r99 = np.median([
            s["A"]["shipped_assumed"]["p99_stop"] / s["A"]["shipped_dedicated"]["p99_stop"]
            for s in stocks.values()
        ])
        r95 = np.median([
            s["A"]["shipped_assumed"]["p95_stop"] / s["A"]["shipped_dedicated"]["p95_stop"]
            for s in stocks.values()
        ])
        self.assertGreater(float(r99), float(r95))
        # Reproduced, not just read: the probe stock's shipped pair.
        stored = stocks[self.STOCK]["A"]
        assumed = fit._heldout(self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, self.folds)
        self.assertAlmostEqual(
            stored["shipped_assumed"]["p99_stop"], float(np.percentile(assumed, 99)), delta=2e-3
        )
        self.assertTrue(stored["tier_adopted"])
        self.assertTrue(fcf.adopts_tail(stored["shipped_dedicated"], stored["shipped_assumed"]))

    def test_led_tier_is_not_justified(self) -> None:
        """After white balance a high-CRI phosphor LED scene is within a
        tenth of a stop of daylight for Stage A: no LED tier is carried."""
        med = self.record["median"]["LED-B3"]
        self.assertLess(med["recoverable_p99_stop"], 0.1)
        self.assertEqual(med["stocks_adopting"], 0)
        self.assertEqual(self.record["median"]["LED-RGB1"]["stocks_adopting"], 20)

    def test_interpolation_rule_validated_below_4000k_only(self) -> None:
        """The runtime's reciprocal-CCT log blend: at 3000-3400K it beats
        the plain base by >=0.04 stop, beats the linear blend, and sits
        within 0.12 stop of a dedicated field; at 4000-4500K it is WORSE
        than the plain base, which is itself within 0.04 stop of the ceiling
        — the measured reason auto stops at 4000K."""
        med = self.record["interpolation_median_p99"]
        for key in ("3000K", "3200K", "3400K"):
            with self.subTest(cct=key):
                m = med[key]
                self.assertLessEqual(m["log_blend_rule"], m["d55_only"] - 0.04)
                self.assertLessEqual(m["log_blend_rule"], m["linear_blend_rule"])
                self.assertLessEqual(m["log_blend_rule"], m["dedicated_field"] + 0.12)
        for key in ("4000K", "4500K"):
            with self.subTest(cct=key):
                m = med[key]
                self.assertLess(m["d55_only"], m["log_blend_rule"])
                self.assertLessEqual(m["d55_only"], m["dedicated_field"] + 0.04)
        self.assertAlmostEqual(fit.tungsten_weight(2856.0), 1.0)
        self.assertAlmostEqual(fit.tungsten_weight(5503.0), 0.0)
        self.assertAlmostEqual(fit.tungsten_weight(3200.0), med["3200K"]["w_rule"], places=4)

    def test_interpolation_probe_reproduces(self) -> None:
        stock = ff.STOCKS[self.STOCK]
        chroma = json.loads((ROOT / "docs" / "chroma_field_cv.json").read_text())["stocks"]
        base = fit._Shipped(
            self.xyz_d, self.rgb_d, self.exp_d,
            fcf.adopts(chroma[self.STOCK]["poly3"], chroma[self.STOCK]["3x3"]),
        )
        stored_a = self.record["stocks"][self.STOCK]["A"]
        tier = fit._Shipped(self.xyz_a, self.rgb_a, self.exp_a, stored_a["tier_model"] == "field")
        xyz_t, rgb_t, exp_t, _ = fcf.stimulus_and_exposures(stock, "BB3200K")
        got = fcf.summarize(fit._blend_errors(
            base, tier, xyz_t, rgb_t, exp_t, self.folds, fit.tungsten_weight(3200.0), "log"
        ))
        stored = self.record["interpolation"][self.STOCK]["3200K"]["log_blend_rule"]
        self.assertAlmostEqual(stored["p99_stop"], got["p99_stop"], delta=2e-3)

    def test_white_board_stays_anchored_under_every_illuminant(self) -> None:
        """Row 0 is the white board under the SAME illuminant: its working
        rgb must be exactly neutral after the CAT, or the tier would smuggle
        a cast into the anchor."""
        for ill in ("D55", "A", "LED-B3", "LED-RGB1", "BB3200K", "BB4500K"):
            _xyz, rgb, _exp, _m = fcf.stimulus_and_exposures(
                ff.STOCKS[self.STOCK], ill
            )
            np.testing.assert_allclose(
                rgb[0], np.ones(3), atol=5e-7,
                err_msg=f"white board not neutral under {ill}",
            )


if __name__ == "__main__":
    unittest.main()
