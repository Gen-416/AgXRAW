# SPDX-License-Identifier: GPL-3.0-or-later
"""Source saturation has its own compact, decoder-dependent exclusion domain."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dngscan.decoder_loss import (
    sensor_saturation_sources, source_reliability_exclusion, support_is_untrusted,
)
from dngscan.scene_reference import reliable_reference_samples
from dngscan import raw_io
from dngscan.analysis import analyze
from dngscan.prepared_sample import PreparedSceneSample
from dngscan.tone import reliable_scene_ev_selection
from tests.test_scene_reference import _evidence
from tests.test_odd_raw_clip_masks import write_odd_dng
from tests.test_pipeline_corrections import write_sensor_dng


def evidence_for(raw):
    raw = np.asarray(raw, dtype=np.uint16)
    if raw.ndim == 3:
        colors = np.broadcast_to(np.arange(3, dtype=np.uint8), raw.shape)
    else:
        h, w = raw.shape
        colors = np.tile(np.array([[0, 1], [3, 2]], np.uint8),
                         ((h + 1) // 2, (w + 1) // 2))[:h, :w]
    return SimpleNamespace(raw_image=raw, raw_colors=colors, white_level=1000,
                           camera_white_levels=[1000.] * 4, raw_pattern=[[0, 1], [3, 2]])


def geometry_for(evidence, **overrides):
    geometry = dict(decoded_shape=list(evidence.raw_image.shape[:2]), half_size=False,
                    demosaic="AHD", highlight="clip", pixel_aspect=1., warp_ops=[],
                    crop=None, orientation_flip=0, reduction=1, stage1_spatial=False)
    geometry.update(overrides)
    return geometry


class SourceReliabilityTests(unittest.TestCase):
    def test_saturation_uses_physical_fullwell_and_never_changes_samples(self):
        raw = np.full((5, 7), 500, np.uint16)
        raw[4, 6], raw[3, 6] = 900, 1000
        evidence = evidence_for(raw)
        before = raw.copy()
        source = sensor_saturation_sources(evidence, {0: 900, 1: 1000, 2: 1000, 3: 1000})
        self.assertEqual(source.dtype, np.uint8)
        np.testing.assert_array_equal(np.argwhere(source), [[3, 6], [4, 6]])
        np.testing.assert_array_equal(raw, before)
        self.assertIsNone(sensor_saturation_sources(evidence, {cid: 1001 for cid in range(4)}))

    def test_full_ahd_support_precedes_crop_but_half_size_can_drop_source(self):
        raw = np.full((125, 127), 500, np.uint16)
        raw[124, 124] = 1000
        evidence = evidence_for(raw)
        fullwell = {cid: 1000 for cid in range(4)}
        full_geometry = geometry_for(evidence, crop=[0., 0., 124., 126.])
        mask, reason = source_reliability_exclusion(evidence, full_geometry, fullwell)
        self.assertEqual(mask.shape, (124, 126))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask[123, 123], 1)
        self.assertEqual(mask[16, 16], 0)
        self.assertFalse(support_is_untrusted(reason))
        half_geometry = geometry_for(evidence, half_size=True, decoded_shape=[63, 64],
                                     demosaic="half-size", crop=[0., 0., 124., 126.])
        mask, reason = source_reliability_exclusion(evidence, half_geometry, fullwell)
        self.assertIsNone(mask)
        self.assertFalse(support_is_untrusted(reason))

    def test_dht_and_reconstruction_with_saturation_are_qualified_globally(self):
        raw = np.full((16, 16), 500, np.uint16)
        evidence = evidence_for(raw)
        for demosaic, mode in (("DHT", "clip"), ("AHD", "reconstruct")):
            geometry = geometry_for(evidence, demosaic=demosaic, highlight=mode)
            mask, reason = source_reliability_exclusion(evidence, geometry, {cid: 1000 for cid in range(4)})
            self.assertIsNone(mask)
            self.assertIsNone(reason)
            raw[8, 8] = 1000
            mask, reason = source_reliability_exclusion(evidence, geometry, {cid: 1000 for cid in range(4)})
            self.assertIsNone(mask)
            self.assertTrue(support_is_untrusted(reason))
            self.assertIn("source saturation", reason)
            raw[8, 8] = 500

    def test_linear_source_tracks_any_plane_and_actual_oriented_box_support(self):
        raw = np.full((5, 7, 3), 500, np.uint16)
        raw[-1, 2, 0] = 1000
        evidence = evidence_for(raw)
        fullwell = {cid: 1000 for cid in range(3)}
        source = sensor_saturation_sources(evidence, fullwell)
        self.assertEqual(source.shape, (5, 7))
        self.assertEqual(np.count_nonzero(source), 1)
        geometry = geometry_for(evidence, demosaic="linear-planes", reduction=2)
        mask, _ = source_reliability_exclusion(evidence, geometry, fullwell)
        self.assertIsNone(mask)  # The actual box discards the last oriented row.
        geometry["orientation_flip"] = 2
        mask, _ = source_reliability_exclusion(evidence, geometry, fullwell)
        expected = np.zeros((2, 3), np.uint8)
        expected[0, 1] = 1
        np.testing.assert_array_equal(mask, expected)
        # A JSON round trip preserves the replay contract exactly.
        replay, _ = source_reliability_exclusion(evidence, json.loads(json.dumps(geometry)), fullwell)
        np.testing.assert_array_equal(replay, expected)

    def test_independent_reference_consumes_exclusion_without_visual_mask_changes(self):
        evidence = _evidence(np.full((64, 64), 500, np.uint16))
        scene = np.full((32, 32, 3), 10., np.float32)
        exclusion = np.zeros((32, 32), np.uint8)
        exclusion[-4:] = 1
        recipe = SimpleNamespace(post=(), crop=None, scene_reliability_exclusion=exclusion)
        reference, percent = reliable_reference_samples(evidence, scene, 100., None, recipe)
        self.assertEqual(percent, 87.5)
        self.assertEqual(reference.shape, (896, 3))
        exclusion[:] = 1
        reference, percent = reliable_reference_samples(evidence, scene, 100., None, recipe)
        self.assertEqual(percent, 0.)
        self.assertEqual(reference.shape, (0, 3))

    def test_load_analysis_does_not_repeat_source_propagation_with_same_fullwell(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "resolved.dng"
            raw = np.full((125, 127), 1000, np.uint16)
            raw[124, 124] = 4095
            write_odd_dng(path, raw, crop=(126, 124))
            bundle = raw_io.load_raw(path, demosaic="ahd")
            before = bundle.scene_reliability_exclusion
            with patch("dngscan.decoder_loss.source_reliability_exclusion",
                       wraps=source_reliability_exclusion) as propagate:
                analyze(bundle, 4, diagnostics=False)
            self.assertEqual(propagate.call_count, 0)
            self.assertIs(bundle.scene_reliability_exclusion, before)

    def test_fullwell_refresh_replays_source_support_independently_of_visual_crop(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "refresh.dng"
            raw = np.full((125, 127), 1000, np.uint16)
            raw[124, 124] = 3000
            write_odd_dng(path, raw, crop=(126, 124))
            for algorithm in ("ahd", "dht"):
                with self.subTest(algorithm=algorithm):
                    bundle = raw_io.load_raw(path, demosaic=algorithm)
                    scene = bundle.scene_rec2020_render.copy()
                    initial_descriptor = bundle.noise_decode
                    original = {int(cid): level for cid, level in
                                initial_descriptor["source_loss_fullwell"].items()}
                    self.assertFalse(bundle.scene_loss_support_untrusted)
                    self.assertIsNone(bundle.scene_reliability_exclusion)
                    changed = original.copy()
                    changed[0] = 3000
                    bundle.raw_guidance = object()
                    self.assertTrue(raw_io.refresh_clip_masks_from_fullwell(bundle, changed))
                    self.assertIsNone(bundle.raw_guidance)
                    self.assertIsNot(bundle.noise_decode, initial_descriptor)
                    self.assertEqual(initial_descriptor["source_loss_fullwell"]["0"], 4095)
                    self.assertFalse(np.any(bundle.clip_masks))
                    self.assertIsNone(bundle.processing_clip_masks)
                    self.assertEqual(bundle.scene_processing_loss_pct, 0.)
                    if algorithm == "ahd":
                        self.assertFalse(bundle.scene_loss_support_untrusted)
                        self.assertEqual(bundle.scene_reliability_exclusion[123, 123], 1)
                    else:
                        self.assertTrue(bundle.scene_loss_support_untrusted)
                        self.assertIsNone(bundle.scene_reliability_exclusion)
                        self.assertEqual(bundle.scene_reliability_source, "decoder-support-untrusted")
                    np.testing.assert_array_equal(bundle.scene_rec2020_render, scene)
                    self.assertTrue(raw_io.refresh_clip_masks_from_fullwell(bundle, original))
                    self.assertFalse(bundle.scene_loss_support_untrusted)
                    self.assertIsNone(bundle.scene_reliability_exclusion)
                    self.assertIsNone(bundle.scene_correction_note)
                    self.assertEqual(bundle.scene_reliability_source, "sensor-spatial")

    def test_refresh_cannot_clear_separate_unknown_processing_support(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "processing.dng"
            raw = np.full((128, 128), 1000, np.uint16)
            raw[64, 64] = 3000
            write_sensor_dng(path, signal=raw, neutral=(.5, 1., 1.))
            bundle = raw_io.load_raw(path, demosaic="dht")
            self.assertTrue(bundle.scene_loss_support_untrusted)
            self.assertIsNone(bundle.noise_decode["source_loss_support"])
            fullwell = {int(cid): 4096 for cid in bundle.noise_decode["source_loss_fullwell"]}
            raw_io.refresh_clip_masks_from_fullwell(bundle, fullwell)
            self.assertTrue(bundle.scene_loss_support_untrusted)
            self.assertTrue(support_is_untrusted(bundle.noise_decode["loss_support"]))
            self.assertIsNone(bundle.noise_decode["source_loss_support"])
            self.assertEqual(bundle.scene_reliability_source, "decoder-support-untrusted")

    def test_fullwell_refresh_invalidates_bound_sample_evidence(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample-refresh.dng"
            raw = np.full((125, 127), 1000, np.uint16)
            raw[124, 124] = 4095  # Existing, cropped-out source exclusion.
            raw[64, 65] = 3000  # Green becomes saturated at the new endpoint.
            write_odd_dng(path, raw, crop=(126, 124))
            bundle = raw_io.load_raw(path, demosaic="ahd")
            result, _, _ = analyze(bundle, 4, diagnostics=False)
            bundle = PreparedSceneSample.from_bundle(bundle).bind(bundle)
            self.assertIsNotNone(bundle._tone_plan_sample_exclusion)
            original = {int(cid): level for cid, level in
                        bundle.noise_decode["source_loss_fullwell"].items()}
            old_exclusion = bundle._tone_plan_sample_exclusion.copy()
            changed = {cid: (3000 if cid in (1, 3) else level)
                       for cid, level in original.items()}
            self.assertEqual(bundle.scene_reliability_exclusion[64, 65], 0)
            self.assertTrue(raw_io.refresh_clip_masks_from_fullwell(bundle, changed))
            self.assertEqual(bundle.scene_reliability_exclusion[64, 65], 1)
            self.assertIsNone(bundle._tone_plan_sample)
            self.assertIsNone(bundle._tone_plan_sample_masks)
            self.assertIsNone(bundle._tone_plan_sample_exclusion)
            _, _, reliable, _ = reliable_scene_ev_selection(bundle, result)
            reliable = reliable.reshape(bundle.scene_rec2020_render.shape[:2])
            self.assertFalse(reliable[64, 65])
            self.assertTrue(np.any(reliable))
            rebuilt = PreparedSceneSample.from_bundle(bundle)
            self.assertGreater(np.count_nonzero(rebuilt.exclusion),
                               np.count_nonzero(old_exclusion))

            # The visual rebuild also invalidates bound samples on older
            # bundles that do not have a replayable source descriptor.
            bundle = rebuilt.bind(bundle)
            bundle.noise_decode = dict(bundle.noise_decode)
            bundle.noise_decode.pop("source_loss_geometry")
            self.assertTrue(raw_io.refresh_clip_masks_from_fullwell(bundle, original))
            self.assertIsNone(bundle._tone_plan_sample)
            self.assertIsNone(bundle._tone_plan_sample_masks)
            self.assertIsNone(bundle._tone_plan_sample_exclusion)


if __name__ == "__main__":
    unittest.main()
