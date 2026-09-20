# SPDX-License-Identifier: GPL-3.0-or-later
"""Deferred masks must preserve the former eager-load + full-well refresh result."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import raw_io
from dngscan.dng_opcodes import Warp
from dngscan.models import RawBundle
from dngscan.spatial_black import SpatialBlack
from tests.test_pipeline_corrections import write_sensor_dng


def _bundle(kind="bayer"):
    h, w = 24, 30
    pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8)
    desc = "RGBG"
    if kind == "xtrans":
        pattern = np.array([[1, 0, 1, 1, 2, 1], [2, 1, 2, 0, 1, 0],
                            [1, 0, 1, 1, 2, 1], [1, 2, 1, 1, 0, 1],
                            [0, 1, 0, 2, 1, 2], [1, 2, 1, 1, 0, 1]], dtype=np.uint8)
    elif kind == "sparse":
        pattern = np.array([[0, 2], [2, 4]], dtype=np.uint8)
        desc = "R?G?B"
    raw = np.linspace(740, 1010, h * w).reshape(h, w).astype(np.uint16)
    colors = np.tile(pattern, (h // len(pattern), w // pattern.shape[1]))
    if kind == "linear":
        raw = np.stack([raw, np.roll(raw, 3, axis=1), np.roll(raw, 5, axis=0)], axis=2)
        colors = np.broadcast_to(np.arange(3, dtype=np.uint8), raw.shape)
        pattern = np.empty((0, 0), dtype=np.uint8)
        desc = "RGB"
    scene = np.full((12, 15, 3), 12000., dtype=np.float32)
    return RawBundle(
        path=Path("small-mask-fixture.dng"), raw_image=raw, raw_colors=colors,
        xyz_render=scene.copy(), render_scale=65535., scene_rec2020_render=scene,
        scene_scale=65535., white_level=1000, black_levels=[12., 15., 17., 13., 19.],
        camera_wb=[2., 1., 1.5, 1.], daylight_wb=[1.5, 1., 2., 1.],
        applied_wb=[2., 1., 1.5, 1.], decode_wb=[2., 1., 1.5, 1.],
        wb_xyz_to_cam=np.eye(3),
        wb_color_matrix=np.hstack([np.eye(3), np.zeros((3, 1))]),
        color_desc=desc, raw_pattern=pattern.tolist(), camera_white_levels=[1000.] * 5,
    )


def _mask(bundle, levels):
    result = raw_io.build_clip_masks(
        bundle.raw_image, bundle.raw_colors, bundle.color_desc, bundle.white_level,
        bundle.black_levels, levels, bundle.orientation_flip,
        bundle.scene_rec2020_render.shape[:2], bundle.raw_pattern,
        bundle.scene_geometry_ops, bundle.scene_crop_sensor,
        getattr(bundle.evidence, "spatial_black", None),
    )
    raw_io._merge_processing_loss(result, bundle.processing_clip_masks)
    return result


def _legacy_eager_result(bundle, fullwell):
    """Endpoint selection before deferral: retain float metadata unless a rebuild ran."""
    initial = _mask(bundle, bundle.camera_white_levels)
    if not fullwell:
        return initial, None
    ids = [int(cid) for cid in np.unique(bundle.raw_colors)]
    metadata = {cid: int(bundle.camera_white_levels[cid]
                        if cid < len(bundle.camera_white_levels)
                        and bundle.camera_white_levels[cid] > 0 else bundle.white_level)
                for cid in ids}
    resolved = {cid: int(fullwell.get(cid, metadata[cid])) for cid in ids}
    if resolved == metadata:
        return initial, None
    levels = [0.] * (max(ids) + 1)
    for cid in ids:
        levels[cid] = float(fullwell.get(cid, metadata[cid]))
    return _mask(bundle, levels), resolved


def _pending(bundle):
    return replace(bundle, clip_masks=None, _clip_masks_pending=True)


class DeferredMaskContractTests(unittest.TestCase):
    def assert_matches_eager(self, bundle, fullwell):
        expected, stamp = _legacy_eager_result(bundle, fullwell)
        deferred = _pending(bundle)
        before_raw = deferred.raw_image.copy()
        before_loss = (None if deferred.processing_clip_masks is None
                       else deferred.processing_clip_masks.copy())
        with patch.object(raw_io, "build_clip_masks", wraps=raw_io.build_clip_masks) as build:
            self.assertTrue(raw_io.refresh_clip_masks_from_fullwell(deferred, fullwell))
        self.assertEqual(build.call_count, 1)
        np.testing.assert_array_equal(deferred.clip_masks, expected)
        np.testing.assert_array_equal(deferred.raw_image, before_raw)
        if before_loss is not None:
            np.testing.assert_array_equal(deferred.processing_clip_masks, before_loss)
        self.assertEqual(deferred._clip_mask_fullwell, stamp)
        self.assertFalse(deferred._clip_masks_pending)
        with patch.object(raw_io, "build_clip_masks", side_effect=AssertionError("duplicate build")):
            self.assertFalse(raw_io.refresh_clip_masks_from_fullwell(deferred, fullwell))
        return deferred

    def test_patterns_match_eager_at_metadata_and_observed_full_wells(self):
        for kind in ("bayer", "xtrans", "linear", "sparse"):
            bundle = _bundle(kind)
            for wells in ({}, {int(cid): 1000 for cid in np.unique(bundle.raw_colors)}, {0: 925}):
                with self.subTest(kind=kind, wells=wells):
                    self.assert_matches_eager(bundle, wells)

    def test_fractional_metadata_is_not_rounded_when_no_rebuild_was_needed(self):
        bundle = _bundle()
        bundle.camera_white_levels = [1000.75, 998.625, 1001.5, 999.875]
        wells = {cid: int(value) for cid, value in enumerate(bundle.camera_white_levels)}
        result = self.assert_matches_eager(bundle, wells)
        rounded = _mask(bundle, list(wells.values()))
        self.assertTrue(np.any(result.clip_masks != rounded), "fixture must expose DN rounding")

    def test_positive_subunit_and_missing_metadata_keep_old_fallback_rules(self):
        for levels, wells in (([.75, 0., -2., 1000.75], {0: 0, 1: 1000, 2: 1000, 3: 1000}),
                              ([0., .75], {0: 940}), ([], {}), ([], {2: 920})):
            with self.subTest(levels=levels, wells=wells):
                bundle = _bundle()
                bundle.camera_white_levels = levels
                self.assert_matches_eager(bundle, wells)

    def test_spatial_black_geometry_and_processing_loss_match_eager(self):
        for kind in ("bayer", "linear"):
            with self.subTest(kind=kind):
                bundle = _bundle(kind)
                bundle.evidence = SimpleNamespace(spatial_black=SpatialBlack(
                    np.linspace(-3, 6, 30), np.linspace(-5, 5, 24),
                    np.array([[[12., 15., 17.], [13., 16., 18.]]]), (0, 0), (30., 33., 35.)))
                bundle.orientation_flip = 6
                bundle.scene_geometry_ops = (Warp(((1.02, -.08, .01, 0., .003, -.002),), .4, .6),)
                bundle.scene_crop_sensor = (2.25, 3.5, 19.5, 24.25)
                bundle.scene_rec2020_render = np.full((15, 10, 3), 12000., dtype=np.float32)
                bundle.processing_clip_masks = np.zeros((30, 20, 3), dtype=np.float16)
                bundle.processing_clip_masks[4:8, 6:10, 2] = 1.
                result = self.assert_matches_eager(bundle, {0: 960, 1: 940, 2: 950, 3: 930})
                self.assertEqual(float(result.clip_masks[2, 3, 2]), 1.)

    def test_success_invalidates_all_derived_mask_and_guidance_caches(self):
        bundle = _pending(_bundle())
        for field in ("_clip_masks_resized", "raw_guidance", "_raw_guidance_resized"):
            setattr(bundle, field, object())
        bundle._clip_masks_cache_shape = bundle._raw_guidance_cache_shape = (1, 1)
        bundle._raw_guidance_has_sensor_snr = bundle._raw_guidance_has_resolved_fullwell = True
        self.assertTrue(raw_io.refresh_clip_masks_from_fullwell(bundle, {}))
        for field in ("_clip_masks_resized", "raw_guidance", "_raw_guidance_resized",
                      "_clip_masks_cache_shape", "_raw_guidance_cache_shape"):
            self.assertIsNone(getattr(bundle, field), field)
        self.assertFalse(bundle._raw_guidance_has_sensor_snr)
        self.assertFalse(bundle._raw_guidance_has_resolved_fullwell)

    def test_build_and_merge_failures_do_not_publish_and_can_retry(self):
        for operation in ("build_clip_masks", "_merge_processing_loss"):
            with self.subTest(operation=operation):
                bundle = _pending(_bundle())
                sentinel = object()
                bundle.raw_guidance = bundle._clip_masks_resized = sentinel
                bundle._raw_guidance_has_sensor_snr = True
                with patch.object(raw_io, operation, side_effect=RuntimeError("injected mask failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected mask failure"):
                        raw_io.refresh_clip_masks_from_fullwell(bundle, {0: 930})
                self.assertTrue(bundle._clip_masks_pending)
                self.assertIsNone(bundle.clip_masks)
                self.assertIsNone(bundle._clip_mask_fullwell)
                self.assertIs(bundle.raw_guidance, sentinel)
                self.assertIs(bundle._clip_masks_resized, sentinel)
                self.assertTrue(bundle._raw_guidance_has_sensor_snr)
                expected, _ = _legacy_eager_result(bundle, {0: 930})
                self.assertTrue(raw_io.refresh_clip_masks_from_fullwell(bundle, {0: 930}))
                np.testing.assert_array_equal(bundle.clip_masks, expected)
                self.assertFalse(bundle._clip_masks_pending)

    def test_later_fullwell_changes_still_rebuild_and_merge_loss(self):
        bundle = _bundle()
        bundle.processing_clip_masks = np.zeros((12, 15, 3), dtype=np.float16)
        bundle.processing_clip_masks[1, 1, 0] = 1.
        deferred = self.assert_matches_eager(bundle, {})
        for wells in ({0: 925}, {0: 880, 3: 940}, {cid: 1000 for cid in range(4)}):
            with self.subTest(wells=wells):
                self.assertTrue(raw_io.refresh_clip_masks_from_fullwell(deferred, wells))
                expected, _ = _legacy_eager_result(bundle, wells)
                np.testing.assert_array_equal(deferred.clip_masks, expected)
                self.assertEqual(float(deferred.clip_masks[1, 1, 0]), 1.)

    def test_replace_release_and_white_balance_keep_pending_state(self):
        source = _pending(_bundle())
        copies = (replace(source, exposure_gain=2.), raw_io.release_analysis_buffers(source),
                  raw_io.rebalance_raw_bundle(source, "camera"),
                  raw_io.rebalance_raw_bundle(source, "daylight"))
        for copy in copies:
            with self.subTest(wb=copy.wb_mode, xyz=copy.xyz_render is None):
                self.assertTrue(copy._clip_masks_pending)
                self.assertIsNone(copy.clip_masks)
                self.assertTrue(raw_io.refresh_clip_masks_from_fullwell(copy, {0: 925}))
                np.testing.assert_array_equal(copy.clip_masks, _legacy_eager_result(source, {0: 925})[0])
        self.assertTrue(source._clip_masks_pending)
        self.assertIsNone(source.clip_masks)

    def test_absent_or_apple_masks_are_not_inferred_without_pending_state(self):
        for decoder, pending in (("libraw", False), ("coreimage", False), ("coreimage", True)):
            with self.subTest(decoder=decoder, pending=pending):
                bundle = replace(_bundle(), scene_decoder=decoder, _clip_masks_pending=pending)
                with patch.object(raw_io, "build_clip_masks", side_effect=AssertionError("unexpected build")):
                    self.assertFalse(raw_io.refresh_clip_masks_from_fullwell(bundle, {0: 925}))
                self.assertIsNone(bundle.clip_masks)


class DeferredMaskLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "sensor.dng"
        write_sensor_dng(self.source, spatial_black=True)

    def test_default_loader_is_eager_and_opt_in_builds_once_at_final_fullwell(self):
        with patch.object(raw_io, "build_clip_masks", wraps=raw_io.build_clip_masks) as build:
            eager = raw_io.load_raw(self.source, scene_half_size=True)
            self.assertEqual(build.call_count, 1)
            deferred = raw_io.load_raw(self.source, scene_half_size=True, _defer_clip_masks=True)
            self.assertEqual(build.call_count, 1)
            self.assertIsNotNone(eager.clip_masks)
            self.assertFalse(eager._clip_masks_pending)
            self.assertIsNone(deferred.clip_masks)
            self.assertTrue(deferred._clip_masks_pending)
            wells = {cid: 3900 for cid in np.unique(eager.raw_colors)}
            self.assertTrue(raw_io.refresh_clip_masks_from_fullwell(eager, wells))
            self.assertTrue(raw_io.refresh_clip_masks_from_fullwell(deferred, wells))
            self.assertEqual(build.call_count, 3)
        np.testing.assert_array_equal(eager.clip_masks, deferred.clip_masks)
        np.testing.assert_array_equal(eager.scene_rec2020_render, deferred.scene_rec2020_render)
        self.assertEqual(eager._clip_mask_fullwell, deferred._clip_mask_fullwell)

    def test_apple_auto_fallback_preserves_the_deferral_request(self):
        for defer in (False, True):
            with self.subTest(defer=defer), \
                 patch("dngscan.coreimage_decode.runtime_available", return_value=False), \
                 patch.object(raw_io, "build_clip_masks", wraps=raw_io.build_clip_masks) as build:
                bundle = raw_io.load_raw(self.source, decoder="coreimage", scene_half_size=True,
                                         _defer_clip_masks=defer)
                self.assertEqual(bundle.scene_decoder, "libraw")
                self.assertIn("Apple RAW auto", bundle.scene_decoder_fallback)
                self.assertEqual(bundle._clip_masks_pending, defer)
                self.assertEqual(build.call_count, int(not defer))
                self.assertEqual(bundle.clip_masks is None, defer)

    def test_successful_apple_decode_never_publishes_spatial_mask_or_pending_state(self):
        decoded = (np.full((16, 16, 3), .2, dtype=np.float32),
                   {"version": "9", "baseline_exposure_authored": 0., "baseline_exposure_cleared": True})
        with patch("dngscan.coreimage_decode.runtime_available", return_value=True), \
             patch("dngscan.coreimage_decode.decode_scene_rec2020", return_value=decoded), \
             patch("dngscan.raw_io._decode_corrected_libraw", side_effect=RuntimeError("no reference")), \
             patch.object(raw_io, "build_clip_masks", side_effect=AssertionError("Apple spatial mask")):
            bundle = raw_io.load_raw(self.source, decoder="coreimage", scene_half_size=True,
                                     _defer_clip_masks=True)
        self.assertEqual(bundle.scene_decoder, "coreimage")
        self.assertIsNone(bundle.clip_masks)
        self.assertFalse(bundle._clip_masks_pending)
        self.assertIsNotNone(bundle.evidence)


if __name__ == "__main__":
    unittest.main()
