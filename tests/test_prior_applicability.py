# SPDX-License-Identifier: GPL-3.0-or-later
"""Capture restrictions qualify every source before source priority applies."""
from __future__ import annotations

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from dngscan import priors
from dngscan.analysis import sensor_prior_evidence
from dngscan.noise_model import model_from_prior
from tests.test_noise_model import frame


def entry(name, shutter="mechanical"):
    return {"id": name, "make_contains": "SONY", "model_equals": {"ILCE-7M5"},
            "make_model": "Sony ILCE-7M5", "shutter": shutter,
            "unity_gain_ev": math.log2(400.), "reference_dn_range": 15359.,
            "fwc_e": 61436., "read_noise_log2iso_log2e": [(math.log2(100), 1.)],
            "pdr_log2iso_ev": [(math.log2(100), 10.)]}


class PriorApplicabilityTests(unittest.TestCase):
    def select(self, *, user=None, curated=(), packaged=(), bulk=(), shutter=None, readout=None):
        with patch("dngscan.calibration.matching_prior", return_value=user), \
             patch.object(priors, "PRIOR_TABLE", list(curated)), \
             patch.object(priors, "_jptc_entries", return_value=list(packaged)), \
             patch.object(priors, "_bulk_entries", return_value=list(bulk)):
            return priors.find_priors("SONY", "ILCE-7M5", shutter=shutter, iso=100,
                                      readout=readout)

    def test_all_sources_reject_unknown_and_mismatched_shutter(self):
        for source in ("user", "curated", "packaged", "bulk"):
            for shutter in (None, "electronic", "mechanical"):
                with self.subTest(source=source, shutter=shutter):
                    value = entry(source)
                    options = {source: value if source == "user" else [value]}
                    found = self.select(shutter=shutter, **options)
                    self.assertEqual(found["id"], source)
                    self.assertEqual(priors.prior_usability(found)[0], shutter == "mechanical")

    def test_lower_verified_candidate_can_replace_inapplicable_curated(self):
        found = self.select(curated=[entry("mechanical")],
                            packaged=[entry("electronic", "electronic")], shutter="electronic")
        self.assertEqual(found["id"], "electronic")
        self.assertTrue(priors.prior_usability(found)[0])
        self.assertEqual(found["shutter_match_status"], "matched")

    def test_failed_user_cannot_silently_yield_to_undeclared_generic(self):
        found = self.select(user=entry("user"), curated=[entry("generic", None)],
                            shutter="electronic")
        self.assertEqual(found["id"], "user")
        self.assertEqual(priors.prior_usability(found), (False, "file-shutter-mismatch"))
        matched = self.select(user=entry("user"),
                              packaged=[entry("electronic", "electronic")], shutter="electronic")
        self.assertEqual(matched["id"], "electronic")

    def test_file_shutter_wins_and_model_rebind_does_not_trust_previous_match(self):
        mechanical = self.select(curated=[entry("mechanical")], shutter="mechanical")
        self.assertTrue(priors.prior_usability(mechanical)[0])
        wrong_file = self.select(curated=[entry("mechanical")], shutter="mechanical",
                                 readout={"shutter": "electronic"})
        self.assertFalse(priors.prior_usability(wrong_file)[0])
        for shutter, reason in ((None, "file-shutter-unavailable"),
                                ("electronic", "file-shutter-mismatch")):
            bundle = replace(frame(), shot_iso=100, capture_readout={"shutter": shutter})
            model = model_from_prior(bundle, {i: 16383. for i in range(4)}, mechanical)
            self.assertEqual(model.status, "rejected")
            self.assertEqual(model.reason, reason)

    def test_sony_mechanical_components_and_electronic_selection_are_distinct(self):
        with patch("dngscan.calibration.matching_prior", return_value=None):
            mechanical = priors.find_priors("SONY", "ILCE-7M5", shutter="mechanical", iso=100)
            electronic = priors.find_priors("SONY", "ILCE-7M5", shutter="electronic", iso=100)
            unknown = priors.find_priors("SONY", "ILCE-7M5", iso=100)
        self.assertEqual(mechanical["mode_match"], "curated")
        self.assertEqual({record["shutter"] for record in mechanical["component_sources"].values()},
                         {"mechanical"})
        self.assertAlmostEqual(priors.gain_e_per_dn(mechanical, 100), 4.451971754176194)
        self.assertNotEqual(electronic["id"], mechanical["id"])
        self.assertEqual(electronic["shutter"], "electronic")
        self.assertAlmostEqual(priors.read_noise_e(electronic, 100), 8.80349828059152)
        self.assertFalse(priors.prior_usability(unknown)[0])
        for shutter in (None, "other"):
            evidence = sensor_prior_evidence("SONY", "ILCE-7M5", 100, nf=.001,
                                             fullwell=16383., mean_black=512.2552337646484,
                                             coding_range=15870.744766235352, shutter=shutter)
            self.assertEqual(evidence[4:], (None, None, None, None))


if __name__ == "__main__":
    unittest.main()
