# SPDX-License-Identifier: GPL-3.0-or-later
"""Grain V2 measured-data assets: structure, monotonicity and provenance
of the digitized Kodak diffuse-rms-granularity tables (P4 data pass;
see tools/import_kodak_granularity.py for the extraction method)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

GRAIN_DIR = Path(__file__).resolve().parents[1] / "dngscan" / "data" / "grain"

# (filename, source tag, loge_max, sigma window)
ASSETS = {
    "5207": ("granularity_5207.json", "H-1-5207", 5.0, (0.003, 0.02)),
    "2383": ("granularity_2383.json", "H-1-2383", 1.95, (0.002, 0.07)),
}


class GranularityAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.assets = {
            key: json.loads((GRAIN_DIR / fname).read_text(encoding="utf-8"))
            for key, (fname, *_rest) in ASSETS.items()
        }

    def test_provenance_and_schema(self) -> None:
        for key, (_, tag, *_rest) in ASSETS.items():
            a = self.assets[key]
            self.assertEqual(a["schema"], 1, key)
            self.assertIn(tag, a["source"], key)
            self.assertEqual(a["aperture_um"], 48.0, key)
            self.assertIn("uncertainty", a, key)
            self.assertIn("method", a, key)

    def test_channels_and_monotone_density(self) -> None:
        for key in ASSETS:
            a = self.assets[key]
            self.assertEqual(set(a["channels"]), {"R", "G", "B"}, key)
            for name, ch in a["channels"].items():
                dens = [row[1] for row in ch["density_loge"]]
                self.assertEqual(dens, sorted(dens), f"{key}/{name}")
                loges = [row[0] for row in ch["density_loge"]]
                self.assertEqual(loges, sorted(loges), f"{key}/{name}")
                self.assertAlmostEqual(loges[0], 0.0, msg=f"{key}/{name}")

    def test_sigma_ranges_and_shape(self) -> None:
        # every sigma stays inside the stock's plausible 48um window, and
        # the sigma curves are not flat lines (mid-exposure structure)
        for key, (_, _, _, (lo, hi)) in ASSETS.items():
            for name, ch in self.assets[key]["channels"].items():
                sig = [row[1] for row in ch["sigma_loge"]]
                self.assertTrue(all(lo <= s <= hi for s in sig),
                                f"{key}/{name}: {min(sig)}..{max(sig)}")
                self.assertGreater(max(sig) / sig[0], 1.2, f"{key}/{name}")

    def test_sigma_density_join_matches_anchors(self) -> None:
        for key, (_, _, loge_max, _) in ASSETS.items():
            for name, ch in self.assets[key]["channels"].items():
                sd = ch["sigma_density"]
                self.assertAlmostEqual(
                    sd[0][0], ch["density_loge"][0][1], places=3,
                    msg=f"{key}/{name}")
                self.assertAlmostEqual(
                    sd[0][1], ch["sigma_loge"][0][1], places=5,
                    msg=f"{key}/{name}")
                # last join row sits at the declared logE coverage limit;
                # composition is monotone-cubic (import 2026-08-25), so the
                # linear expectation here matches only to read-uncertainty
                # scale (±0.03 D declared), not float scale
                import numpy as np
                d_tab = np.array(ch["density_loge"], dtype=float)
                expect_d = float(np.interp(loge_max, d_tab[:, 0], d_tab[:, 1]))
                self.assertAlmostEqual(sd[-1][0], expect_d, delta=0.02,
                                       msg=f"{key}/{name}")

    def test_5207_chart_facts(self) -> None:
        # negative stock: B carries the orange-mask base (~1.0) and the
        # largest grain; R the smallest base
        ch = self.assets["5207"]["channels"]
        self.assertGreater(ch["B"]["density_loge"][0][1], 0.9)
        self.assertLess(ch["R"]["density_loge"][0][1], 0.3)
        self.assertGreater(ch["B"]["sigma_loge"][0][1], ch["G"]["sigma_loge"][0][1])

    def test_2383_chart_facts(self) -> None:
        # print stock: clear base (~0.08, no mask) on all channels; B is
        # by far the grainiest layer; R granularity sits ABOVE G (chart
        # labels R over G) — opposite of the negative stock
        ch = self.assets["2383"]["channels"]
        for name in ("R", "G", "B"):
            self.assertLess(ch[name]["density_loge"][0][1], 0.15, name)
        end = {n: ch[n]["sigma_loge"][-1][1] for n in ("R", "G", "B")}
        self.assertGreater(end["B"], 3 * end["G"])
        self.assertGreater(end["R"], end["G"])


if __name__ == "__main__":
    unittest.main()
