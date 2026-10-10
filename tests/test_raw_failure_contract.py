# SPDX-License-Identifier: GPL-3.0-or-later
"""Real generated RAW rejects structural damage without publishing evidence."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dngscan.raw_io import load_raw
from tests.test_spectral_fallback_pipeline import write_noise_dng
from tools.validate_raw_failures import digest, make_variant, probe, run_cases


class RawFailureContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "valid.dng"
        write_noise_dng(self.source)

    def test_real_decoder_rejects_structural_damage_before_analysis(self):
        # A real successful control prevents a broken decoder installation
        # from being mistaken for successful corruption handling.
        bundle = load_raw(self.source, scene_half_size=True)
        self.assertGreater(bundle.raw_image.size, 0)
        self.assertTrue(np.isfinite(bundle.scene_rec2020_render).all())
        original_hash = digest(self.source)
        for mutation in ("invalid-header", "prefix-4096", "truncated-half",
                         "unsupported-main-cfa-compression"):
            with self.subTest(mutation=mutation):
                target = self.directory / (mutation + ".dng")
                self.assertIsNotNone(make_variant(self.source, target, mutation))
                with self.assertRaises(RuntimeError):
                    load_raw(target, scene_half_size=True)
                result = probe(target)
                self.assertEqual(result["status"], "explicit-decoder-failure")
                for field in ("decode_supported", "render_supported", "measurement_qualified",
                              "analysis_attempted", "published_output"):
                    self.assertIs(result[field], False)
        self.assertEqual(digest(self.source), original_hash)

    def test_isolated_acceptance_runner_keeps_original_and_reports_all_cases(self):
        row = run_cases(self.source, self.directory)
        self.assertTrue(row["control"]["passed"])
        self.assertTrue(row["original_unchanged"])
        self.assertEqual(len(row["cases"]), 4)
        self.assertTrue(all(case["passed"] for case in row["cases"]))
        self.assertTrue(all(case["process_returncode"] == 0 for case in row["cases"]))


if __name__ == "__main__":
    unittest.main()
