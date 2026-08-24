# SPDX-License-Identifier: GPL-3.0-or-later
"""Priors importer gates: JPTC PTC fit, P2P bulk conversion, fallback chain.

Three tiers feed dngscan.priors.find_priors: curated entries (hand-checked),
JPTC first-party PTC fits (dngscan/data/priors/jptc/), and the P2P bulk
table (dngscan/data/priors/p2p_bulk.json). These tests pin the conversion
math against hand-computed values and gate the ordering so a bulk entry can
never shadow a curated one.
"""
from __future__ import annotations

import math
import subprocess
import sys
import unittest
from pathlib import Path

from dngscan import priors

REPO = Path(__file__).parents[1]


class TestJptcImporter(unittest.TestCase):
    def test_self_test_synthetic_sensor(self):
        """The importer must recover a known synthetic sensor's parameters."""
        proc = subprocess.run(
            [sys.executable, str(REPO / "tools" / "import_jptc.py"), "--self-test"],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("self-test: PASS", proc.stdout)

    def test_a7m5_entry_loads_as_tier2_prior(self):
        entries = priors._jptc_entries()
        hits = [e for e in entries if "ILCE-7M5" in e["model_equals"]]
        self.assertEqual(len(hits), 1)
        e = hits[0]
        # gain 4.3238 e-/DN @ISO100 -> unity_gain_ev = log2(432.38)
        self.assertAlmostEqual(e["unity_gain_ev"], math.log2(432.38), places=3)
        self.assertAlmostEqual(e["fwc_e"], 62906, delta=5)
        self.assertGreater(e["fwc_e_uncertainty"], 0)
        # Single-ISO entry: read noise extrapolates flat, PDR degrades to None.
        self.assertAlmostEqual(priors.read_noise_e(e, 100), 7.204, places=2)
        self.assertAlmostEqual(priors.read_noise_e(e, 6400), 7.204, places=2)
        self.assertIsNone(priors.pdr_ev(e, 100))


class TestP2pBulk(unittest.TestCase):
    def test_table_loads(self):
        entries = priors._bulk_entries()
        self.assertGreaterEqual(len(entries), 130)
        for e in entries:
            self.assertTrue(e["pdr_log2iso_ev"])
            self.assertTrue(e["read_noise_log2iso_log2e"])

    def test_a77v_pinned_conversion(self):
        """Hand-validated cross-check camera (vs the public P2P PDR chart)."""
        e = priors.find_priors("SONY", "SLT-A77V")
        self.assertIsNotNone(e)
        self.assertIn("p2p_bulk", e["source"])
        self.assertAlmostEqual(priors.pdr_ev(e, 100), 9.91, delta=0.02)
        self.assertAlmostEqual(priors.read_noise_e(e, 100), 4.04, delta=0.05)
        # High-ISO read noise converges to the table's published 2.7 e-.
        self.assertAlmostEqual(priors.read_noise_e(e, 6400), 2.70, delta=0.05)
        self.assertAlmostEqual(priors.gain_e_per_dn(e, 100), 1.192, delta=0.01)

    def test_sanity_ranges(self):
        for e in priors._bulk_entries():
            for _, y in e["read_noise_log2iso_log2e"]:
                self.assertTrue(0.05 <= 2.0 ** y <= 200.0, e["id"])
            for _, y in e["pdr_log2iso_ev"]:
                self.assertTrue(1.0 <= y <= 17.0, e["id"])


class TestFallbackOrdering(unittest.TestCase):
    def test_curated_shadows_bulk(self):
        """Sigma fp exists in both tiers; the curated entry must win."""
        e = priors.find_priors("SIGMA", "SIGMA FP")
        self.assertIn("dcg_switch_iso", e)  # curated-only annotation

    def test_curated_shadows_jptc(self):
        """A7M5 exists curated + JPTC; curated (with PDR curve) must win."""
        e = priors.find_priors("SONY", "ILCE-7M5")
        self.assertEqual(e["id"], "Sony ILCE-7M5 (A7 V)")
        self.assertIsNotNone(priors.pdr_ev(e, 100))

    def test_unknown_camera_none(self):
        self.assertIsNone(priors.find_priors("ACME", "FOOCAM 9000"))

    def test_a7m5_internal_consistency(self):
        """Regression pin for the corrected unity_gain_ev: the gain implied
        by unity gain at ISO 100 must reproduce fwc_e from the 14-bit range
        (the defect fixed 2026-08-24 was a 3x contradiction here)."""
        e = priors.find_priors("SONY", "ILCE-7M5")
        implied_fwc = priors.gain_e_per_dn(e, 100) * (16383 - 512)
        self.assertLess(abs(implied_fwc - e["fwc_e"]) / e["fwc_e"], 0.10)


if __name__ == "__main__":
    unittest.main()
