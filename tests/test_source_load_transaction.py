# SPDX-License-Identifier: GPL-3.0-or-later
"""A successful RAW load commits one source generation, or no bundle at all."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import raw_io
from dngscan.source_metadata import (
    SourceChangedError, SourceMetadataSession, SourceVerificationError,
)
from tests.test_pipeline_corrections import write_sensor_dng


class SourceLoadTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sensor.dng"
        write_sensor_dng(self.path, signal=1000)

    def rewrite(self, atomic):
        target = self.path.with_suffix(".new") if atomic else self.path
        write_sensor_dng(target, signal=2000)
        if atomic:
            target.replace(self.path)

    def load(self, decoder="libraw"):
        return raw_io.load_raw(self.path, decoder=decoder, scene_half_size=True,
                              _defer_clip_masks=True, _analysis_luminance_only=True)

    def test_real_libraw_cannot_publish_old_evidence_and_new_scene(self):
        original = raw_io.acquire_raw_evidence
        for atomic in (False, True):
            with self.subTest(atomic=atomic):
                write_sensor_dng(self.path, signal=1000)
                captured = []

                def acquire(path):
                    evidence = original(path)
                    captured.append(evidence)
                    self.rewrite(atomic)
                    return evidence

                with patch.object(raw_io, "acquire_raw_evidence", side_effect=acquire):
                    with self.assertRaises(SourceChangedError):
                        self.load()
                self.assertEqual(float(np.median(captured[0].raw_image)), 1000.)
                # A new outer call starts a new, coherent capture transaction.
                self.assertEqual(float(np.median(self.load().raw_image)), 2000.)

    def test_metadata_and_successful_scene_mutations_abort_commit(self):
        for stage in ("metadata", "scene"):
            owner, name = ((raw_io.dng_metadata, "read_dng_shot_info") if stage == "metadata"
                           else (raw_io, "_decode_corrected_libraw"))
            original = getattr(owner, name)
            for atomic in (False, True):
                with self.subTest(stage=stage, atomic=atomic):
                    write_sensor_dng(self.path, signal=1000)
                    changed = False

                    def mutate(*args, **kwargs):
                        nonlocal changed
                        result = original(*args, **kwargs)
                        if not changed:
                            changed = True
                            self.rewrite(atomic)
                        return result

                    with patch.object(owner, name, side_effect=mutate):
                        with self.assertRaises(SourceChangedError):
                            self.load()

    def test_apple_scene_and_reference_changes_abort_complete_load(self):
        for stage in ("apple-scene", "libraw-reference"):
            for atomic in (False, True):
                with self.subTest(stage=stage, atomic=atomic):
                    write_sensor_dng(self.path, signal=1000)
                    original = raw_io._decode_corrected_libraw

                    def apple(*args, **kwargs):
                        if stage == "apple-scene":
                            self.rewrite(atomic)
                        return np.full((64, 64, 3), .2, np.float16), {
                            "version": "9", "baseline_exposure_cleared": True,
                            "fallback_errors": [], "recipe": {},
                        }

                    def reference(*args, **kwargs):
                        result = original(*args, **kwargs)
                        if stage == "libraw-reference":
                            self.rewrite(atomic)
                        return result

                    with patch("dngscan.coreimage_decode.runtime_available", return_value=True), \
                         patch("dngscan.coreimage_decode.decode_scene_rec2020", side_effect=apple), \
                         patch.object(raw_io, "_decode_corrected_libraw", side_effect=reference):
                        with self.assertRaises(SourceChangedError):
                            self.load("coreimage")

    def test_explicit_source_change_cannot_be_swallowed_by_auto_fallback(self):
        with patch("dngscan.coreimage_decode.runtime_available", return_value=True), \
             patch("dngscan.coreimage_decode.decode_scene_rec2020",
                   side_effect=SourceChangedError(self.path, phase="decode")), \
             patch.object(raw_io, "_decode_corrected_libraw") as fallback:
            with self.assertRaises(SourceChangedError):
                self.load("coreimage")
        fallback.assert_not_called()

    def test_libraw_fallback_success_must_still_pass_the_capture_transaction(self):
        original = raw_io._decode_corrected_libraw
        for atomic in (False, True):
            with self.subTest(atomic=atomic):
                write_sensor_dng(self.path, signal=1000)

                def scene(*args, **kwargs):
                    result = original(*args, **kwargs)
                    self.rewrite(atomic)
                    return result

                with patch("dngscan.coreimage_decode.runtime_available", return_value=False), \
                     patch.object(raw_io, "_decode_corrected_libraw", side_effect=scene):
                    with self.assertRaises(SourceChangedError):
                        self.load("coreimage")

    def test_final_load_verification_io_error_does_not_become_decoder_fallback(self):
        original_scene, original_stat = raw_io._decode_corrected_libraw, Path.stat
        finished = False

        def scene(*args, **kwargs):
            nonlocal finished
            result = original_scene(*args, **kwargs)
            finished = True
            return result

        def stat(path, *args, **kwargs):
            if finished and path == self.path:
                raise OSError(5, "injected final source verification failure")
            return original_stat(path, *args, **kwargs)

        with patch.object(raw_io, "_decode_corrected_libraw", side_effect=scene), \
             patch.object(Path, "stat", stat):
            with self.assertRaises(SourceVerificationError):
                self.load()

    def test_initial_unknown_identity_preserves_basic_load_without_claiming_change(self):
        with patch("dngscan.source_metadata._source_identity", return_value=None):
            session = SourceMetadataSession(self.path)
            self.assertEqual(session.status, "unavailable")
            self.assertFalse(session.assert_consistent())
            self.assertEqual(float(np.median(self.load().raw_image)), 1000.)

    def test_known_identity_later_io_failure_is_unverifiable_not_changed(self):
        session = SourceMetadataSession(self.path)
        error = OSError(5, "injected final stat failure")
        with patch.object(Path, "stat", side_effect=error):
            self.assertFalse(session.is_current())
            self.assertEqual(session.status, "unverifiable")
            with self.assertRaises(SourceVerificationError) as caught:
                session.assert_consistent()
        self.assertIs(caught.exception.__cause__, error)
        self.assertNotIsInstance(caught.exception, SourceChangedError)
        self.assertEqual(session.status, "unverifiable")

    def test_initial_identity_acquisition_distinguishes_change_and_late_io(self):
        with patch("dngscan.source_metadata._stat_key", side_effect=[(1,) * 5, (2,) * 5]):
            with self.assertRaises(SourceChangedError):
                SourceMetadataSession(self.path)
        with patch.object(Path, "stat", side_effect=OSError(5, "final hash stat failure")):
            with self.assertRaises(SourceVerificationError):
                SourceMetadataSession(self.path)


if __name__ == "__main__":
    unittest.main()
