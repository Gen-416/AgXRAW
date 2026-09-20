# SPDX-License-Identifier: GPL-3.0-or-later
"""Apple fallback may reuse evidence only for the same verified source version."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import raw_io
from tests.test_pipeline_corrections import write_sensor_dng


class FallbackSourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sensor.dng"
        write_sensor_dng(self.path, signal=1000)

    def load(self):
        return raw_io.load_raw(self.path, decoder="coreimage", scene_half_size=True,
                               _defer_clip_masks=True, _analysis_luminance_only=True)

    def check(self, bundle, value):
        self.assertEqual(bundle.scene_decoder, "libraw")
        self.assertIn("Apple RAW auto", bundle.scene_decoder_fallback)
        self.assertTrue(np.all(bundle.raw_image == value))
        self.assertIsNone(bundle.xyz_render)
        self.assertIsNotNone(bundle._analysis_y_render)
        self.assertTrue(bundle._clip_masks_pending)

    def test_unchanged_source_reuses_evidence_and_opens_one_fresh_scene(self):
        with patch("dngscan.coreimage_decode.runtime_available", return_value=False), \
             patch.object(raw_io, "acquire_raw_evidence", wraps=raw_io.acquire_raw_evidence) as acquire, \
             patch.object(raw_io, "_decode_corrected_libraw", wraps=raw_io._decode_corrected_libraw) as decode:
            bundle = self.load()
        self.assertEqual(acquire.call_count, 1)
        self.assertEqual(decode.call_count, 1)
        self.check(bundle, 1000)

    def test_change_during_evidence_acquisition_disables_fallback_reuse(self):
        acquire_original = raw_io.acquire_raw_evidence
        captured = []

        def acquire(path):
            evidence = acquire_original(path)
            captured.append(evidence)
            if len(captured) == 1:
                write_sensor_dng(path, signal=1200)
            return evidence

        with patch("dngscan.coreimage_decode.runtime_available", return_value=False), \
             patch.object(raw_io, "acquire_raw_evidence", side_effect=acquire):
            bundle = self.load()
        self.assertEqual(len(captured), 2)
        self.assertIs(bundle.evidence, captured[1])
        self.check(bundle, 1200)

    def test_change_during_apple_failure_reacquires_current_evidence(self):
        def fail_apple(**_):
            write_sensor_dng(self.path, signal=1400)
            return False

        with patch("dngscan.coreimage_decode.runtime_available", side_effect=fail_apple), \
             patch.object(raw_io, "acquire_raw_evidence", wraps=raw_io.acquire_raw_evidence) as acquire:
            bundle = self.load()
        self.assertEqual(acquire.call_count, 2)
        self.check(bundle, 1400)

    def test_change_at_recursive_entry_rejects_previously_verified_evidence(self):
        original_load = raw_io.load_raw
        passed = []

        def recursive(path, *args, **kwargs):
            passed.append(kwargs["_fallback_evidence"])
            write_sensor_dng(path, signal=1600)
            return original_load(path, *args, **kwargs)

        with patch("dngscan.coreimage_decode.runtime_available", return_value=False), \
             patch.object(raw_io, "acquire_raw_evidence", wraps=raw_io.acquire_raw_evidence) as acquire, \
             patch.object(raw_io, "load_raw", side_effect=recursive):
            bundle = original_load(self.path, decoder="coreimage", scene_half_size=True,
                                   _defer_clip_masks=True, _analysis_luminance_only=True)
        self.assertEqual(len(passed), 1)
        self.assertIsNotNone(passed[0])
        self.assertEqual(acquire.call_count, 2)
        self.check(bundle, 1600)


if __name__ == "__main__":
    unittest.main()
