# SPDX-License-Identifier: GPL-3.0-or-later
"""Dependency exclusions keep source rows and ANY proxy footprints distinct."""
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import guidance
from dngscan.analysis import analyze, reanalyze_balanced_scene
from dngscan.gui import preview_cache as cache
from dngscan.models import RawGuidanceMaps
from dngscan.prepared_sample import PreparedSceneSample
from dngscan.raw_io import load_raw, rebalance_raw_bundle
from dngscan.reliability import resize_exclusion
from dngscan.sampling import sample_indices
from dngscan.tone import reliable_scene_ev_selection, scene_tone_metrics
from tests.test_odd_raw_clip_masks import write_odd_dng
from tests.test_preview_cache import _analysis, _bundle


class ReliabilityHandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "store"))
        environment.start()
        self.addCleanup(environment.stop)

    def proxy(self, bundle, analysis):
        with patch.object(cache, "PROXY_LONG_EDGE", 32):
            return cache.build_proxy_entry(bundle, analysis)

    def roundtrip(self, entry):
        path = self.root / "preview.npz"
        cache._write_disk_entry(path, entry)
        restored = cache._read_disk_entry(path, entry.bundle.path, require_guidance=False)
        self.assertIsNotNone(restored)
        return restored

    def test_real_late_crop_deferred_wb_prepared_proxy_and_disk_keep_reliability(self):
        path = self.root / "late-crop-ahd.dng"
        pixels = np.full((125, 127), 1000, np.uint16)
        pixels[124, 124] = 4095
        write_odd_dng(path, pixels, crop=(126, 124))
        bundle = load_raw(path, demosaic="ahd", _defer_clip_masks=True)
        self.assertTrue(bundle._clip_masks_pending)
        analysis, _, _ = analyze(bundle, 4)
        self.assertFalse(bundle._clip_masks_pending)
        self.assertFalse(bundle.scene_loss_support_untrusted)
        self.assertTrue(np.any(bundle.scene_reliability_exclusion))
        self.assertFalse(np.any(bundle.clip_masks))
        original = reliable_scene_ev_selection(bundle, analysis)
        self.assertFalse(np.all(original[2]))
        self.assertTrue(np.any(original[2]))
        prepared = PreparedSceneSample.from_bundle(bundle)
        expected = bundle.scene_reliability_exclusion.reshape(-1)[prepared.source_indices]
        np.testing.assert_array_equal(prepared.exclusion, expected)
        self.assertFalse(prepared.exclusion.flags.writeable)
        self.assertTrue(bundle.scene_reliability_exclusion.flags.writeable)
        bound = prepared.bind(bundle)
        self.assertIs(bound._tone_plan_sample_exclusion, prepared.exclusion)
        entry = self.proxy(bundle, analysis)
        np.testing.assert_array_equal(entry.bundle._tone_plan_sample_exclusion, expected)
        self.assertTrue(entry.source_metadata["has_scene_reliability_exclusion"])
        np.testing.assert_array_equal(
            entry.bundle.scene_reliability_exclusion,
            resize_exclusion(bundle.scene_reliability_exclusion, entry.bundle.scene_rec2020_render.shape[:2],
                             filter_radius=3),
        )
        restored = self.roundtrip(entry)
        for candidate in (bound, entry.bundle, restored.bundle):
            actual = reliable_scene_ev_selection(candidate, analysis)
            for found, wanted in zip(actual[:3], original[:3]):
                np.testing.assert_array_equal(found, wanted)
            self.assertEqual(actual[3], original[3])
            self.assertFalse(np.any(candidate.clip_masks))
        from_proxy = PreparedSceneSample.from_bundle(restored.bundle)
        self.assertIs(from_proxy.exclusion, restored.bundle._tone_plan_sample_exclusion)
        self.assertIsNone(from_proxy.source_indices)
        for mode in ("camera", "daylight"):
            balanced = rebalance_raw_bundle(restored.bundle, mode)
            balanced_analysis = reanalyze_balanced_scene(analysis, balanced)
            np.testing.assert_array_equal(balanced.scene_reliability_exclusion,
                                          entry.bundle.scene_reliability_exclusion)
            np.testing.assert_array_equal(balanced._tone_plan_sample_exclusion, expected)
            np.testing.assert_array_equal(reliable_scene_ev_selection(balanced, balanced_analysis)[2], original[2])

    def test_canonical_sample_exclusion_uses_exact_full_source_indices(self):
        shape = (801, 1001)
        exclusion = (np.arange(np.prod(shape)).reshape(shape) % 101 == 0).astype(np.uint8)
        bundle = replace(_bundle(), scene_rec2020_render=np.full(shape + (3,), 2000, np.uint16),
                         clip_masks=None, scene_reliability_exclusion=exclusion)
        indices = sample_indices(np.prod(shape))
        expected = exclusion.reshape(-1)[indices]
        prepared = PreparedSceneSample.from_bundle(bundle)
        np.testing.assert_array_equal(prepared.exclusion, expected)
        entry = self.proxy(bundle, _analysis())
        np.testing.assert_array_equal(entry.bundle._tone_plan_sample_exclusion, expected)
        # ANY-resized proxy cells can be more conservative, but must never
        # replace the exact full-source population used by tone compilation.
        self.assertTrue(np.all(entry.bundle.scene_reliability_exclusion))
        self.assertTrue(np.any(expected == 0))
        original = reliable_scene_ev_selection(bundle, _analysis())
        for candidate in (prepared.bind(bundle), entry.bundle):
            np.testing.assert_array_equal(reliable_scene_ev_selection(candidate, _analysis())[2], original[2])
            self.assertEqual(scene_tone_metrics(candidate, _analysis()).reliable_sample_pct,
                             scene_tone_metrics(bundle, _analysis()).reliable_sample_pct)

    def test_proxy_any_footprint_keeps_a_single_exclusion_without_visual_loss(self):
        exclusion = np.zeros((97, 131), np.uint8)
        exclusion[50, 73] = 1
        scene = np.full((97, 131, 3), 2000, np.uint16)
        before = cache.downsample_mean(scene, 32)
        scene[50, 73, 0] = 64000
        bundle = replace(_bundle(), clip_masks=None,
                         scene_rec2020_render=scene,
                         scene_reliability_exclusion=exclusion)
        entry = self.proxy(bundle, _analysis())
        self.assertTrue(np.any(entry.bundle.scene_reliability_exclusion))
        changed = np.any(entry.bundle.scene_rec2020_render != before, axis=2)
        self.assertTrue(np.any(changed))
        self.assertTrue(np.all(entry.bundle.scene_reliability_exclusion[changed] != 0))
        self.assertTrue(np.any(entry.bundle.scene_reliability_exclusion == 0))
        self.assertIsNone(entry.bundle.clip_masks)
        np.testing.assert_array_equal(entry.bundle._tone_plan_sample_exclusion, exclusion.reshape(-1))
        restored = self.roundtrip(entry)
        np.testing.assert_array_equal(restored.bundle.scene_reliability_exclusion,
                                      entry.bundle.scene_reliability_exclusion)
        self.assertEqual(int(np.count_nonzero(restored.bundle._tone_plan_sample_exclusion)), 1)

    def test_metadata_and_version_distinguish_presence(self):
        bundle = _bundle()
        first = cache._bundle_metadata(bundle)
        bundle.scene_reliability_exclusion = np.zeros((8, 8), np.uint8)
        second = cache._bundle_metadata(bundle)
        self.assertFalse(first["has_scene_reliability_exclusion"])
        self.assertTrue(second["has_scene_reliability_exclusion"])
        self.assertNotEqual(first, second)
        self.assertEqual(cache.PREVIEW_CACHE_VERSION, 29)

    def test_disk_rejects_missing_wrong_shape_dtype_and_inconsistent_presence(self):
        bundle = replace(_bundle(), scene_reliability_exclusion=np.eye(8, dtype=np.uint8))
        entry = self.proxy(bundle, _analysis())
        original = self.root / "original.npz"
        cache._write_disk_entry(original, entry)
        with np.load(original, allow_pickle=False) as stored:
            base = {key: np.asarray(stored[key]).copy() for key in stored.files}
        cases = ("missing-scene", "missing-sample", "scene-shape", "sample-shape",
                 "scene-dtype", "sample-dtype", "hidden-scene", "hidden-sample",
                 "source-presence", "source-type", "flag-type", "sample-without-rgb",
                 "missing-flag", "prior-version")
        for case in cases:
            with self.subTest(case=case):
                values = dict(base)
                metadata = json.loads(str(values["metadata"].item()))
                if case == "missing-scene":
                    values.pop("scene_reliability_exclusion")
                elif case == "missing-sample":
                    values.pop("tone_sample_exclusion")
                elif case == "scene-shape":
                    values["scene_reliability_exclusion"] = np.zeros((1, 1), np.uint8)
                elif case == "sample-shape":
                    values["tone_sample_exclusion"] = np.zeros((1,), np.uint8)
                elif case == "scene-dtype":
                    values["scene_reliability_exclusion"] = values["scene_reliability_exclusion"].astype(np.float32)
                elif case == "sample-dtype":
                    values["tone_sample_exclusion"] = values["tone_sample_exclusion"].astype(np.float32)
                elif case == "hidden-scene":
                    metadata["bundle"]["has_scene_reliability_exclusion"] = False
                elif case == "hidden-sample":
                    metadata["has_tone_plan_sample_exclusion"] = False
                elif case == "source-presence":
                    metadata["source_bundle"]["has_scene_reliability_exclusion"] = False
                elif case == "source-type":
                    metadata["source_bundle"] = []
                elif case == "flag-type":
                    metadata["has_tone_plan_sample_exclusion"] = 1
                elif case == "sample-without-rgb":
                    values.pop("tone_sample")
                elif case == "missing-flag":
                    metadata.pop("has_tone_plan_sample_exclusion")
                elif case == "prior-version":
                    metadata["version"] = 27
                values["metadata"] = np.asarray(json.dumps(metadata))
                path = self.root / f"{case}.npz"
                np.savez(path, **values)
                self.assertIsNone(cache._read_disk_entry(path, bundle.path, require_guidance=False))

    def test_absent_exclusions_remain_absent_in_legacy_scope(self):
        bundle = _bundle()
        prepared = PreparedSceneSample.from_bundle(bundle)
        self.assertIsNone(prepared.exclusion)
        restored = self.roundtrip(self.proxy(bundle, _analysis()))
        self.assertIsNone(restored.bundle.scene_reliability_exclusion)
        self.assertIsNone(restored.bundle._tone_plan_sample_exclusion)
        self.assertIsNone(PreparedSceneSample.from_bundle(restored.bundle).exclusion)

    def test_missing_snr_stays_missing_with_independent_proxy_and_disk_eligibility(self):
        shape = (97, 131)
        exclusion = np.zeros(shape, np.uint8)
        exclusion[50, 73] = 1
        original = RawGuidanceMaps(np.ones(shape + (3,), np.float16),
                                   np.zeros(shape, np.uint8), None,
                                   np.zeros(shape, np.float32))
        bundle = replace(_bundle(), raw_image=None, raw_colors=None, clip_masks=None,
                         scene_rec2020_render=np.full(shape + (3,), 2000, np.uint16),
                         scene_reliability_exclusion=exclusion, raw_guidance=original)
        with patch.object(cache, "PROXY_LONG_EDGE", 32):
            entry = cache.build_proxy_entry(bundle, _analysis(), include_guidance=True)
        restored = self.roundtrip(entry)
        self.assertIsNone(original.snr_confidence)
        self.assertIsNone(original.scene_eligibility)
        for candidate in (entry.bundle, restored.bundle):
            maps = candidate.raw_guidance
            self.assertIsNone(maps.snr_confidence)
            np.testing.assert_array_equal(maps.scene_eligibility,
                                          1 - candidate.scene_reliability_exclusion)
            flat = guidance.flatten_raw_guidance(maps, 0, maps.clip_class.size)
            self.assertIsNone(flat.snr_confidence)
            ev = np.full(flat.clip_class.shape, 2., np.float32)
            arguments = dict(raw_headroom_rgb=flat.headroom, raw_clip_class=flat.clip_class,
                             raw_permission=flat.raw_permission, raw_snr_confidence=flat.snr_confidence,
                             noise_ev_floor=0., midtone_protect=0.)
            baseline = guidance.color_path_weight(None, ev, 8., **arguments)
            qualified = guidance.color_path_weight(None, ev, 8., scene_eligibility=flat.scene_eligibility,
                                                   **arguments)
            allowed = flat.scene_eligibility != 0
            self.assertTrue(np.any(allowed))
            self.assertTrue(np.any(~allowed))
            self.assertTrue(np.all(baseline > 0.))
            np.testing.assert_array_equal(qualified[allowed], baseline[allowed])
            np.testing.assert_array_equal(qualified[~allowed], 0.)

    def test_disk_validates_guidance_eligibility_without_inventing_snr(self):
        bundle = _bundle()
        bundle.raw_guidance = RawGuidanceMaps(
            np.ones((8, 8, 3), np.float16), np.zeros((8, 8), np.uint8), None,
            np.zeros((8, 8), np.float32), np.eye(8, dtype=np.uint8),
        )
        entry = cache.PreviewEntry(bundle, _analysis())
        original = self.root / "guidance.npz"
        cache._write_disk_entry(original, entry)
        valid = cache._read_disk_entry(original, bundle.path, require_guidance=True)
        self.assertIsNotNone(valid)
        self.assertIsNone(valid.bundle.raw_guidance.snr_confidence)
        np.testing.assert_array_equal(valid.bundle.raw_guidance.scene_eligibility,
                                      bundle.raw_guidance.scene_eligibility)
        with np.load(original, allow_pickle=False) as stored:
            base = {key: np.asarray(stored[key]).copy() for key in stored.files}
        for case in ("missing", "missing-flag", "hidden", "shape", "nonfinite", "out-of-range"):
            with self.subTest(case=case):
                values = dict(base)
                metadata = json.loads(str(values["metadata"].item()))
                if case == "missing":
                    values.pop("guidance_scene_eligibility")
                elif case == "missing-flag":
                    metadata.pop("guidance_has_eligibility")
                elif case == "hidden":
                    metadata["guidance_has_eligibility"] = False
                elif case == "shape":
                    values["guidance_scene_eligibility"] = np.ones((1, 1), np.uint8)
                elif case == "nonfinite":
                    values["guidance_scene_eligibility"] = np.full((8, 8), np.nan, np.float16)
                elif case == "out-of-range":
                    values["guidance_scene_eligibility"] = np.full((8, 8), 2, np.uint8)
                values["metadata"] = np.asarray(json.dumps(metadata))
                path = self.root / f"eligibility-{case}.npz"
                np.savez(path, **values)
                self.assertIsNone(cache._read_disk_entry(path, bundle.path, require_guidance=True))


if __name__ == "__main__":
    unittest.main()
