# SPDX-License-Identifier: GPL-3.0-or-later
"""Gates on the route-D illuminant-tier CV record (phase 1, measurement only).

Same doctrine as the Stage A CV gates: the stored record must reproduce from
the current spectral assets, and the headline decision numbers — the D55
assumption is expensive under tungsten while a dedicated tier recovers it —
must survive any re-fit.
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

    def test_stored_record_reproduces_for_the_probe_pair(self) -> None:
        stored = json.loads(RECORD.read_text())["stocks"][self.STOCK][self.ILL]
        assumed = fcf.summarize(
            fit._heldout(self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, self.folds)
        )
        dedicated = fcf.summarize(
            fit._heldout(self.xyz_a, self.exp_a, self.xyz_a, self.exp_a, self.folds)
        )
        for key in ("p95_stop", "p99_stop"):
            self.assertAlmostEqual(
                stored["field_assumed_d55"][key], assumed[key], delta=2e-3
            )
            self.assertAlmostEqual(
                stored["field_dedicated"][key], dedicated[key], delta=2e-3
            )

    def test_tungsten_assumption_is_expensive_and_recoverable(self) -> None:
        """The decision claim of route D phase 1: assuming D55 on a
        tungsten-lit (white-balanced) scene costs over a stop at p99, and a
        dedicated tier recovers it to the daylight error class."""
        assumed = fit._heldout(
            self.xyz_d, self.exp_d, self.xyz_a, self.exp_a, self.folds
        )
        dedicated = fit._heldout(
            self.xyz_a, self.exp_a, self.xyz_a, self.exp_a, self.folds
        )
        self.assertGreater(float(np.percentile(assumed, 99.0)), 1.5)
        self.assertLess(float(np.percentile(dedicated, 99.0)), 1.0)

    def test_white_board_stays_anchored_under_every_illuminant(self) -> None:
        """Row 0 is the white board under the SAME illuminant: its working
        rgb must be exactly neutral after the CAT, or the tier would smuggle
        a cast into the anchor."""
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
