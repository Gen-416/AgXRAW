# SPDX-License-Identifier: GPL-3.0-or-later
"""Optics V2 P5 data pass: digitized Kodak MTF tables and the fitted
scatter-kernel parameters (see tools/import_kodak_mtf.py)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

MTF_DIR = Path(__file__).resolve().parents[1] / "dngscan" / "data" / "mtf"


class MtfAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.a = {
            key: json.loads((MTF_DIR / f"mtf_{key}.json").read_text("utf-8"))
            for key in ("5207", "2383")
        }

    def test_provenance_and_schema(self) -> None:
        for key, tag in (("5207", "H-1-5207"), ("2383", "H-1-2383")):
            a = self.a[key]
            self.assertEqual(a["schema"], 1, key)
            self.assertIn(tag, a["source"], key)
            self.assertIn("uncertainty", a, key)
            self.assertEqual(set(a["channels"]), {"R", "G", "B"}, key)

    def test_anchors_roll_off_monotonically_past_the_bump(self) -> None:
        # beyond 15 c/mm every channel must be non-increasing (the <15
        # region may carry the documented adjacency bump)
        for key, a in self.a.items():
            for name, rows in a["channels"].items():
                tail = [r[1] for r in rows if r[0] >= 15.0]
                self.assertEqual(tail, sorted(tail, reverse=True),
                                 f"{key}/{name}")

    def test_fit_parameters_are_physical(self) -> None:
        for key, a in self.a.items():
            for name, fit in a["fit"].items():
                self.assertGreater(fit["s"], 0.1, f"{key}/{name}")
                self.assertLessEqual(fit["s"], 1.0, f"{key}/{name}")
                self.assertGreater(fit["sigma_um"], 2.0, f"{key}/{name}")
                self.assertLess(fit["sigma_um"], 16.0, f"{key}/{name}")
                self.assertLess(fit["rms_residual"], 0.06, f"{key}/{name}")

    def test_5207_layer_order_facts(self) -> None:
        # the red-sensitive layer sits at the bottom of the pack and sees
        # the most scatter: its rolloff is the deepest of the three
        fit = self.a["5207"]["fit"]
        r50 = {n: [r[1] for r in self.a["5207"]["channels"][n] if r[0] == 50][0]
               for n in ("R", "G", "B")}
        self.assertLess(r50["R"], r50["G"])
        self.assertLess(r50["R"], r50["B"])
        self.assertGreaterEqual(fit["R"]["s"], fit["G"]["s"])

    def test_2383_blue_layer_is_softest(self) -> None:
        fit = self.a["2383"]["fit"]
        self.assertGreater(fit["B"]["s"], fit["R"]["s"])
        self.assertGreater(fit["R"]["s"], fit["G"]["s"])

    def test_fit_reproduces_the_anchor_rolloff(self) -> None:
        import numpy as np

        import tools.import_kodak_mtf as m

        for key, a in self.a.items():
            for name, rows in a["channels"].items():
                f = np.array([r[0] for r in rows if r[0] >= 15.0])
                y = np.array([r[1] for r in rows if r[0] >= 15.0])
                fit = a["fit"][name]
                if a["model"] == "bi_gaussian_v1":
                    got = m.mtf_bi_gaussian(
                        f, fit["s"], fit["w"],
                        fit["sigma_um"] / 1000.0, fit["tail_sigma_um"] / 1000.0)
                else:
                    got = m.mtf_gaussian(f, fit["s"], fit["sigma_um"] / 1000.0)
                self.assertLess(float(np.max(np.abs(got - y))), 0.09,
                                f"{key}/{name}")


if __name__ == "__main__":
    unittest.main()
