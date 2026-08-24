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

    def test_a7m5_entries_load_as_tier2_priors(self):
        entries = priors._jptc_entries()
        hits = [e for e in entries if "ILCE-7M5" in e["model_equals"]]
        # Mechanical + electronic measurements coexist; the ELECTRONIC one is
        # preferred because the mechanical floor (~0.4 DN) is below the PTC's
        # resolution, so its read-noise curve is empty.
        self.assertEqual(len(hits), 2)
        e = hits[0]
        self.assertEqual(e["shutter"], "electronic")
        # robust fit: gain 4.4472 e-/DN @ISO100 -> unity_gain_ev = log2(444.72)
        self.assertAlmostEqual(e["unity_gain_ev"], 8.7967, places=3)
        self.assertAlmostEqual(e["fwc_e"], 64702, delta=5)
        self.assertGreater(e["fwc_e_uncertainty"], 0)
        # Single-ISO entry: read noise extrapolates flat, PDR degrades to None.
        self.assertAlmostEqual(priors.read_noise_e(e, 100), 8.069, places=2)
        self.assertAlmostEqual(priors.read_noise_e(e, 6400), 8.069, places=2)
        self.assertIsNone(priors.pdr_ev(e, 100))
        # The mechanical entry's unresolved read noise degrades to None.
        self.assertIsNone(priors.read_noise_e(hits[1], 100))
        self.assertEqual(hits[1]["shutter"], "mechanical")

    def test_multi_measurement_preference(self):
        """A7RM6 has iso100 mech/elec + iso640 mech in the JPTC tier: lowest
        ISO wins (fwc_e semantics), then the entry with resolved read noise
        (electronic). Queried on the tier directly — in find_priors the
        curated A7RM6 entry shadows all of these by design."""
        hits = [e for e in priors._jptc_entries() if "ILCE-7RM6" in e["model_equals"]]
        self.assertEqual(len(hits), 3)
        e = hits[0]
        self.assertEqual(e["measured_iso"], 100)
        self.assertEqual(e["shutter"], "electronic")
        self.assertIsNotNone(priors.read_noise_e(e, 100))

    def test_new_makes_resolve(self):
        for make, model in [
            ("NIKON CORPORATION", "NIKON Z 7"),
            ("Panasonic", "DC-G9M2"),
            ("Panasonic", "DC-S1M2"),
        ]:
            e = priors.find_priors(make, model)
            self.assertIsNotNone(e, model)
            self.assertIn("JPTC", e["id"])
            self.assertGreater(priors.gain_e_per_dn(e, 100), 0)


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

    def test_a7rm6_internal_consistency(self):
        """Same defect class, fixed 2026-08-24: curated unity_gain_ev came
        from reciprocal extrapolation through the extended-ISO segment and
        contradicted fwc_e by ~3x; the corrected value must reproduce fwc_e
        from the 14-bit range and agree with the first-party JPTC ramps."""
        e = priors.find_priors("SONY", "ILCE-7RM6")
        self.assertNotIn("measured_iso", e)  # curated, not the JPTC tier
        implied_fwc = priors.gain_e_per_dn(e, 100) * (16383 - 512)
        self.assertLess(abs(implied_fwc - e["fwc_e"]) / e["fwc_e"], 0.10)
        jptc = [x for x in priors._jptc_entries() if "ILCE-7RM6" in x["model_equals"]]
        self.assertAlmostEqual(
            e["unity_gain_ev"], jptc[0]["unity_gain_ev"], delta=0.05
        )


if __name__ == "__main__":
    unittest.main()
