# SPDX-License-Identifier: GPL-3.0-or-later
"""Camera generations must not share physical priors through a name prefix."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from dngscan import priors
from dngscan.analysis import sensor_prior_evidence


class ExactPackagedCameraMatchingTests(unittest.TestCase):
    def setUp(self):
        # Tests of packaged data must not depend on the user's imported profiles.
        self.user = patch("dngscan.calibration.matching_prior", return_value=None)
        self.user.start()
        self.addCleanup(self.user.stop)

    def test_unknown_generation_never_borrows_a_later_model(self):
        for model in ("Canon EOS 5D", "EOS 5D", " canon   eos  5d ",
                      "EOS 5D Mark", "EOS 5D Mark I", "EOS 5D Mark II extra",
                      "EOS", "5D", "EOS R50"):
            with self.subTest(model=model):
                self.assertIsNone(priors.find_priors("Canon", model))

    def test_exact_generation_keeps_its_own_prior(self):
        for suffix in ("Mark II", "Mark III", "Mark IV", "S", "S R"):
            # 5DS and 5DS R intentionally have no space before the S suffix.
            name = "EOS 5D" + (suffix if suffix.startswith("S") else " " + suffix)
            with self.subTest(model=name):
                entry = priors.find_priors("Canon", name)
                self.assertEqual(entry["id"], "Canon " + name)
                self.assertEqual(entry["mode_match"], "bulk-model-only")

    def test_brand_prefix_case_and_whitespace_normalization(self):
        for make, model, expected in (
            (" canon inc. ", " canon   eos 5d mark ii ", "Canon EOS 5D Mark II"),
            (" sony corporation ", " sony  slt-a77v ", "Sony SLT-A77V"),
            ("SONY", "SLT-A77V", "Sony SLT-A77V"),
            ("sigma corporation", " sigma   fp ", "Sigma fp"),
            ("SIGMA", "fp", "Sigma fp"),
        ):
            with self.subTest(make=make, model=model):
                self.assertEqual(priors.find_priors(make, model)["id"], expected)

    def test_manufacturer_is_an_exact_identity(self):
        for make in ("NOT CANON", "CANONICAL", "SONY", "ACME CANON"):
            with self.subTest(make=make):
                self.assertIsNone(priors.find_priors(make, "Canon EOS 5D Mark II"))
        self.assertIsNone(priors.find_priors("CANON", "Sony SLT-A77V"))
        self.assertIsNone(priors.find_priors("SONY", "SIGMA FP"))
        self.assertIsNone(priors.find_priors("SIGMA", "SIGMA FP L"))

    def test_every_bulk_camera_accepts_its_complete_name_and_brandless_model(self):
        with patch.object(priors, "PRIOR_TABLE", []), \
             patch.object(priors, "_jptc_entries", return_value=[]):
            for expected in priors._bulk_entries():
                make, model = expected["make_model"].split(maxsplit=1)
                for query in (model, expected["make_model"]):
                    with self.subTest(model=query):
                        self.assertEqual(priors.find_priors(make, query)["id"], expected["id"])

    def test_explicit_lumix_aliases_preserve_bulk_matches(self):
        for model in ("DC-S1", "DC-S1R", "DC-S5"):
            for query in (model, "Lumix " + model, "Panasonic Lumix " + model):
                with self.subTest(model=query):
                    self.assertEqual(priors.find_priors("Panasonic", query)["id"],
                                     "Panasonic Lumix " + model)
        for model in ("DC-S", "DC-S1R II", "Lumix DC-S1 extra"):
            with self.subTest(model=model):
                self.assertIsNone(priors.find_priors("Panasonic", model))

    def test_curated_and_jptc_tiers_retain_priority_and_readout_gate(self):
        fp = priors.find_priors("SIGMA", "SIGMA FP")
        self.assertEqual(fp["mode_match"], "curated")
        self.assertEqual(fp["dcg_switch_iso"], 640)
        sony = priors.find_priors("SONY", "SONY ILCE-7M5")
        self.assertEqual(sony["id"], "Sony ILCE-7M5 (A7 V)")
        self.assertEqual(sony["mode_match"], "curated")
        nikon = priors.find_priors("Nikon Corporation", " Nikon  Z 7 ")
        self.assertIn("JPTC", nikon["id"])
        blind = priors.find_priors("Panasonic", "DC-S1M2")
        self.assertFalse(priors.prior_usability(blind)[0])
        self.assertEqual(blind["mode_match"], "model-only-ambiguous-shutter")
        exact = priors.find_priors("Panasonic", "DC-S1M2", shutter="electronic")
        self.assertTrue(priors.prior_usability(exact)[0])
        self.assertEqual(exact["mode_match"], "exact-shutter")
        mismatch = priors.find_priors("Panasonic", "DC-S1M2", shutter="other")
        self.assertFalse(priors.prior_usability(mismatch)[0])

    def test_unknown_camera_has_no_physical_evidence_even_at_compatible_dn_scale(self):
        for code_range in (4095., 4096., 3967.75, 15871.):
            with self.subTest(code_range=code_range):
                result = sensor_prior_evidence(
                    "Canon", "Canon EOS 5D", 100, nf=.001,
                    fullwell=code_range, mean_black=0., coding_range=code_range,
                )
                self.assertIsNone(result[0])
                self.assertEqual(result[4:], (None, None, None, None))

    def test_known_camera_dn_scale_and_suspect_iso_gates_remain_active(self):
        good = sensor_prior_evidence("SIGMA", "SIGMA FP", 100, nf=.001,
                                     fullwell=16383., mean_black=1024., coding_range=15359.)
        self.assertGreater(good[4], 0.)
        self.assertGreater(good[5], 0.)
        for iso, span, reason in ((100, 12000., "unmatched-dn-scale"),
                                  (40000, 15359., "suspect-iso")):
            with self.subTest(iso=iso, span=span):
                result = sensor_prior_evidence("SIGMA", "SIGMA FP", iso, nf=.001,
                                               fullwell=16383., mean_black=1024., coding_range=span)
                self.assertEqual(result[1], reason)
                self.assertEqual(result[4:], (None, None, None, None))


class UserPriorPrecedenceTests(unittest.TestCase):
    def test_user_matcher_retains_original_camera_mode_iso_and_priority(self):
        user_entry = {"id": "owner calibration", "mode_match": "user-exact-shutter"}
        with patch("dngscan.calibration.matching_prior", return_value=user_entry) as match, \
             patch.object(priors, "_jptc_entries", side_effect=AssertionError("lower tier")), \
             patch.object(priors, "_bulk_entries", side_effect=AssertionError("lower tier")):
            result = priors.find_priors(" sigma corporation ", " sigma  fp ",
                                       shutter="electronic", iso=640)
        self.assertIs(result, user_entry)
        match.assert_called_once_with(" sigma corporation ", " sigma  fp ",
                                      shutter="electronic", iso=640)


if __name__ == "__main__":
    unittest.main()
