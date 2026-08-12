# SPDX-License-Identifier: GPL-3.0-or-later
"""Grain V2 measured-data asset: structure, monotonicity and provenance
of the digitized Kodak 5207 diffuse-rms-granularity tables (P4 data
pass; see tools/import_kodak_granularity.py for the extraction method)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ASSET = Path(__file__).resolve().parents[1] / "dngscan" / "data" / "grain" / "granularity_5207.json"


class GranularityAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.a = json.loads(ASSET.read_text(encoding="utf-8"))

    def test_provenance_and_schema(self) -> None:
        self.assertEqual(self.a["schema"], 1)
        self.assertIn("H-1-5207", self.a["source"])
        self.assertEqual(self.a["aperture_um"], 48.0)
        self.assertIn("uncertainty", self.a)
        self.assertIn("method", self.a)

    def test_channels_and_monotone_density(self) -> None:
        self.assertEqual(set(self.a["channels"]), {"R", "G", "B"})
        for name, ch in self.a["channels"].items():
            dens = [row[1] for row in ch["density_loge"]]
            self.assertEqual(dens, sorted(dens), name)
            loges = [row[0] for row in ch["density_loge"]]
            self.assertEqual(loges, sorted(loges), name)
            self.assertAlmostEqual(loges[0], 0.0)
            self.assertAlmostEqual(loges[-1], 5.0)

    def test_sigma_ranges_and_shape(self) -> None:
        # 5207 is a fine-grained T-grain cine stock: every sigma stays in
        # a plausible 48um window, and the mid-exposure hump exists (the
        # dashed curves are not flat lines).
        for name, ch in self.a["channels"].items():
            sig = [row[1] for row in ch["sigma_loge"]]
            self.assertTrue(all(0.003 <= s <= 0.02 for s in sig), name)
            self.assertGreater(max(sig) / sig[0], 1.2, name)

    def test_sigma_density_join_matches_anchors(self) -> None:
        # the parametric join must reproduce each channel's base density
        # and end density, and pair the base with the low-exposure sigma
        for name, ch in self.a["channels"].items():
            sd = ch["sigma_density"]
            self.assertAlmostEqual(sd[0][0], ch["density_loge"][0][1], places=3)
            self.assertAlmostEqual(sd[-1][0], ch["density_loge"][-1][1], places=3)
            self.assertAlmostEqual(sd[0][1], ch["sigma_loge"][0][1], places=5)

    def test_channel_ordering_facts(self) -> None:
        # chart facts: B carries the orange-mask base (~1.0) and the
        # largest grain; R the smallest base and finest grain at base
        ch = self.a["channels"]
        self.assertGreater(ch["B"]["density_loge"][0][1], 0.9)
        self.assertLess(ch["R"]["density_loge"][0][1], 0.3)
        self.assertGreater(ch["B"]["sigma_loge"][0][1], ch["G"]["sigma_loge"][0][1])
        self.assertGreater(ch["G"]["sigma_loge"][0][1], ch["R"]["sigma_loge"][0][1] * 0.99)


if __name__ == "__main__":
    unittest.main()
