# SPDX-License-Identifier: GPL-3.0-or-later
"""Gates on the route-D illuminant record (docs/illuminant_tier_cv.json).

Same doctrine as the Stage A CV gates: the stored record must reproduce
from the current spectral assets, and the decision claim — a NEGATIVE
one: no illuminant tier earns its place — must survive any re-fit. The
two measurement mistakes that produced the opposite claim are pinned so
they cannot come back: comparing raw exposures across illuminants (the
per-layer white offset the runtime removes) and evaluating the D55 model
at the un-adapted I-lit chromaticity (a coordinate the runtime never
sees — it feeds the white-balanced pixel through the D55 asset's own
xyz_from_rec2020).
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
FAMILY_THRESHOLD = 5  # stocks adopting, set before the LED numbers were seen


class IlluminantTierRecordTests(unittest.TestCase):
    STOCK = "portra400"
    ILL = "A"

    @classmethod
    def setUpClass(cls) -> None:
        stock = ff.STOCKS[cls.STOCK]
        cls.xyz_d, cls.rgb_d, cls.exp_d, cls.m_d = fcf.stimulus_and_exposures(stock, "D55")
        cls.xyz_a, cls.rgb_a, cls.exp_a, cls.m_a = fcf.stimulus_and_exposures(stock, cls.ILL)
        cls.folds = fcf.cv_folds(cls.exp_d.shape[0])
        cls.record = json.loads(RECORD.read_text())
        chroma = json.loads(CHROMA.read_text())["stocks"][cls.STOCK]
        cls.base = fit.Shipped(
            cls.xyz_d, cls.rgb_d, cls.exp_d, cls.m_d,
            fcf.adopts(chroma["poly3"], chroma["3x3"]),
        )

    def test_stored_record_reproduces_for_the_probe_pair(self) -> None:
        stored = self.record["stocks"][self.STOCK][self.ILL]
        got = {
            "shipped_assumed": fcf.summarize(
                fit.heldout_assumed(self.base, self.rgb_a, self.exp_a, self.folds)
            ),
            "field_dedicated": fcf.summarize(
                fit.heldout_field(self.xyz_a, self.rgb_a, self.exp_a, self.m_a, self.folds)
            ),
            "3x3_dedicated": fcf.summarize(
                fit.heldout_3x3(self.rgb_a, self.exp_a, self.folds)
            ),
            "field_assumed_d55_unadapted": fcf.summarize(
                fit.heldout_unadapted(self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, self.folds)
            ),
            "field_assumed_d55_raw": fcf.summarize(
                fit.heldout_raw(self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, self.folds)
            ),
        }
        for name, summary in got.items():
            for key in ("p95_stop", "p99_stop"):
                self.assertAlmostEqual(
                    stored[name][key], summary[key], delta=2e-3, msg=f"{name}.{key}"
                )

    def test_metric_is_runtime_faithful(self) -> None:
        """Row 0 (the white board) has exactly zero error in every normalized
        prediction; the runtime coordinate (white-balanced rgb through the
        D55 model's own matrix) is a different — and far kinder — place to
        evaluate the D55 model than the un-adapted I-lit chromaticity; and
        the raw cross-illuminant comparison adds the neutral offset on top.
        Both wrong numbers exceed the runtime-faithful one by a large
        margin, which is the whole story of the withdrawn tiers."""
        train = np.setdiff1d(np.arange(1, self.exp_d.shape[0]), self.folds[0])
        pred = self.base.predict(self.rgb_a, train)
        np.testing.assert_array_equal(pred[0], np.zeros(3))
        runtime = np.percentile(
            fit.heldout_assumed(self.base, self.rgb_a, self.exp_a, self.folds), 99
        )
        unadapted = np.percentile(
            fit.heldout_unadapted(self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, self.folds), 99
        )
        raw = np.percentile(
            fit.heldout_raw(self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, self.folds), 99
        )
        self.assertGreater(float(unadapted - runtime), 0.3)
        self.assertGreater(float(raw - unadapted), 0.8)
        # Same-illuminant: the runtime coordinate IS the training XYZ.
        np.testing.assert_allclose(
            self.rgb_a @ np.linalg.inv(self.m_a).T, self.xyz_a, atol=1e-9
        )

    def test_no_illuminant_tier_earns_its_place(self) -> None:
        """The decision claim of route D, negative: after white balance the
        shipped D55 model sits in the same held-out error class as a
        dedicated same-illuminant model under tungsten and high-CRI LED —
        the adoption rule passes on fewer stocks than the family threshold
        (set at 5 before the LED numbers were seen) for either."""
        med = self.record["median"]
        self.assertLessEqual(
            med["A"]["shipped_assumed_p99"], med["A"]["shipped_dedicated_p99"] + 0.02
        )
        self.assertLessEqual(
            abs(med["LED-B3"]["shipped_assumed_p99"] - med["LED-B3"]["shipped_dedicated_p99"]),
            0.05,
        )
        for ill in ("A", "LED-B3"):
            self.assertLess(med[ill]["stocks_adopting"], FAMILY_THRESHOLD, ill)
        # Daylight error class: the assumption costs at most a tenth of a
        # stop at p99 on the probe stock, reproduced, not just read.
        stored = self.record["stocks"][self.STOCK]
        for ill in ("A", "LED-B3"):
            self.assertLess(
                stored[ill]["shipped_assumed"]["p99_stop"], stored[ill]["shipped_dedicated"]["p99_stop"] + 0.1
            )
        assumed = fit.heldout_assumed(self.base, self.rgb_a, self.exp_a, self.folds)
        self.assertAlmostEqual(
            stored["A"]["shipped_assumed"]["p99_stop"], float(np.percentile(assumed, 99)), delta=2e-3
        )
        self.assertFalse(stored["A"]["tier_adopted"])

    def test_narrow_band_is_the_real_but_unidentifiable_case(self) -> None:
        """LED-RGB1 is where a dedicated field would recover about a tenth
        of a stop — and it cannot be told from colour temperature, so it is
        the measured basis for a confidence downgrade, not a tier."""
        stocks = self.record["stocks"]
        assumed = np.median([s["LED-RGB1"]["shipped_assumed"]["p99_stop"] for s in stocks.values()])
        field = np.median([s["LED-RGB1"]["field_dedicated"]["p99_stop"] for s in stocks.values()])
        self.assertGreater(float(assumed - field), 0.08)
        self.assertGreater(
            self.record["median"]["LED-RGB1"]["shipped_assumed_p99"],
            self.record["median"]["A"]["shipped_assumed_p99"] + 0.1,
        )

    def test_white_board_stays_anchored_under_every_illuminant(self) -> None:
        """Row 0 is the white board under the SAME illuminant: its working
        rgb must be exactly neutral after the CAT, or a tier comparison
        would smuggle a cast into the anchor."""
        for ill in ("D55", "A", "LED-B3", "LED-RGB1"):
            _xyz, rgb, _exp, _m = fcf.stimulus_and_exposures(
                ff.STOCKS[self.STOCK], ill
            )
            np.testing.assert_allclose(
                rgb[0], np.ones(3), atol=5e-7,
                err_msg=f"white board not neutral under {ill}",
            )


if __name__ == "__main__":
    unittest.main()
