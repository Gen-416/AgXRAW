# SPDX-License-Identifier: GPL-3.0-or-later
"""Source-scale loss and its actual LibRaw half-size/DefaultScale footprint."""
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from dngscan import decoder_loss, raw_io
from dngscan._deps import rawpy
from dngscan.analysis import analyze
from dngscan.tone import reliable_scene_ev_selection
from tests.test_pipeline_corrections import write_sensor_dng


def red_point(value=3000):
    pixels = np.full((128, 128), 1000, np.uint16)
    pixels[64, 64] = value
    return pixels


class SourceCeilingTests(unittest.TestCase):
    def test_actual_wb_source_is_recorded_before_demosaic_and_raw_is_not_changed(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "source.dng"
            for value, expected in ((2000, False), (3000, True), (3800, True)):
                write_sensor_dng(path, signal=red_point(value), neutral=(.5, 1., 1.))
                with rawpy.imread(str(path)) as raw:
                    before = raw.raw_image_visible.copy()
                    loss = decoder_loss.record_wb_ceiling_loss(raw,
                        raw.black_level_per_channel, raw.white_level,
                        raw.camera_whitebalance, "clip")
                    np.testing.assert_array_equal(raw.raw_image_visible, before)
                    self.assertEqual(loss is not None, expected)
                    if expected:
                        self.assertEqual(np.count_nonzero(loss), 1)
                        self.assertEqual(loss[64, 64], 1)
                    # Highlight modes normalize by maximum WB and do not
                    # introduce this clip-mode red scaling ceiling.
                    for mode in ("blend", "reconstruct"):
                        self.assertIsNone(decoder_loss.record_wb_ceiling_loss(raw,
                            raw.black_level_per_channel, raw.white_level,
                            raw.camera_whitebalance, mode))

    def test_override_black_and_saturation_range_and_existing_loss_are_preserved(self):
        values = np.array([[2500, 1000], [1000, 1000]], np.uint16)
        raw = SimpleNamespace(raw_image_visible=values,
            raw_colors_visible=np.array([[0, 1], [3, 2]], np.uint8))
        existing = np.array([[0, 0], [0, 1]], np.uint8)
        actual = decoder_loss.record_wb_ceiling_loss(raw,
            [512] * 4, 4095, [2, 1, 1, 0], "clip", existing)
        self.assertIs(actual, existing)
        np.testing.assert_array_equal(actual, [[1, 0], [0, 1]])
        # Explicit spatial-black normalization is already at zero/65535.
        raw.raw_image_visible = np.array([[35000, 1000], [1000, 1000]], np.uint16)
        result = decoder_loss.record_wb_ceiling_loss(raw,
            [0] * 4, 65535, [2, 1, 1, 0], "clip")
        np.testing.assert_array_equal(result, [[1, 0], [0, 0]])

    def test_linear_dng_tracks_each_plane_without_a_cfa(self):
        values = np.full((3, 5, 3), 1000, np.uint16)
        values[1, 2, 0] = 3000
        raw = SimpleNamespace(raw_image_visible=values)
        result = decoder_loss.record_wb_ceiling_loss(raw,
            [0] * 4, 4095, [2, 1, 1, 0], "clip")
        self.assertEqual(result.shape, values.shape)
        self.assertEqual(np.count_nonzero(result), 1)
        self.assertEqual(result[1, 2, 0], 1)

    def test_exact_white_is_distinct_from_integer_overflow(self):
        raw = SimpleNamespace(raw_image_visible=np.array([[65535]], np.uint16),
            raw_colors_visible=np.zeros((1, 1), np.uint8))
        self.assertIsNone(decoder_loss.record_wb_ceiling_loss(raw,
            [0] * 4, 65535, [1, 1, 1, 1], "clip"))
        raw.raw_image_visible[0, 0] = 32768
        self.assertEqual(decoder_loss.record_wb_ceiling_loss(raw,
            [0] * 4, 65535, [2, 1, 1, 1], "clip")[0, 0], 1)
        # This float product is above 65535 but truncates to 65535; it is
        # quantization at the boundary, not scale_colors integer overflow.
        raw.raw_image_visible[0, 0] = 32767
        self.assertIsNone(decoder_loss.record_wb_ceiling_loss(raw,
            [0] * 4, 65535, [2.00004, 1, 1, 1], "clip"))


class HalfSizeSupportTests(unittest.TestCase):
    def test_pooling_includes_odd_edges_and_both_greens(self):
        colors = np.tile([[0, 1], [3, 2]], (3, 4))[:5, :7].astype(np.uint8)
        loss = np.zeros(colors.shape, np.uint8)
        loss[0, 1] = loss[1, 0] = loss[4, 6] = 1
        pooled = decoder_loss.pool_mosaic_loss(loss, colors, "RGBG")
        self.assertEqual(pooled.shape, (3, 4, 3))
        self.assertEqual(pooled[0, 0, 1], 1)
        self.assertEqual(pooled[-1, -1, 0], 1)
        self.assertEqual(np.count_nonzero(pooled), 2)

    def test_actual_half_decode_dependents_are_covered_after_default_scale(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "half.dng"
            for scale in (None, (1., 2.), (2., 1.), (1., 1.3), (1.7, 1.), (1., 10.), (10., 1.)):
                with self.subTest(scale=scale):
                    scenes = []
                    for value in (2000, 3000, 3800):
                        write_sensor_dng(path, signal=red_point(value),
                            neutral=(.5, 1., 1.), scale=scale)
                        with rawpy.imread(str(path)) as raw:
                            loss = decoder_loss.record_wb_ceiling_loss(raw,
                                raw.black_level_per_channel, raw.white_level,
                                raw.camera_whitebalance, "clip")
                            pooled = (decoder_loss.pool_mosaic_loss(loss,
                                raw.raw_colors_visible, "RGBG") if loss is not None else None)
                            aspect = raw.sizes.pixel_aspect
                            camera = raw_io.render_to_scene_rec2020(raw, "clip", True,
                                raw_io.resolve_demosaic_algorithm(raw, "auto"),
                                raw_io._fixed_asshot_wb_kwargs(raw.camera_whitebalance),
                                camera_rgb=True)
                            scenes.append(camera)
                            if pooled is not None:
                                transported = decoder_loss.transport_default_scale(
                                    pooled, camera.shape[:2], aspect)
                    np.testing.assert_array_equal(scenes[1], scenes[2])
                    depends = np.any(scenes[0] != scenes[1], axis=2)
                    self.assertTrue(np.any(depends))
                    self.assertTrue(np.all(np.max(transported, axis=2)[depends] != 0))
                    # The exact source footprint remains local in half mode.
                    self.assertLess(np.count_nonzero(np.max(transported, axis=2)),
                        max(16, 4 * max(aspect, 1. / aspect) + 8))

    def test_default_scale_uses_source_taps_not_nearest_target_center(self):
        source = np.zeros((3, 7, 3), np.uint8)
        source[1, 2, 0] = 1
        out = decoder_loss.transport_default_scale(source, (3, 12), 1.7)
        coords = np.floor(np.arange(12) / 1.7).astype(int)
        expected = (coords == 2) | (coords + 1 == 2)
        np.testing.assert_array_equal(out[1, :, 0] != 0, expected)
        with self.assertRaisesRegex(ValueError, "unsupported LibRaw loss geometry"):
            decoder_loss.transport_default_scale(source, (6, 12), 1.)

    def test_default_scale_tracks_libraw_accumulated_coordinate_rounding(self):
        source = np.zeros((3, 1, 3), np.uint8)
        source[0, 0, 0] = 1
        out = decoder_loss.transport_default_scale(source, (30, 1), .1)
        # LibRaw's ten increments give .9999999999999999, so its source row
        # is still zero. floor(arange(30) * .1) would miss that dependency.
        self.assertEqual(out[10, 0, 0], 1)


class ProductionLossSupportTests(unittest.TestCase):
    def test_default_decode_rejects_lost_source_dependents_after_warp_and_scale(self):
        warp = struct.pack(">L8d", 1, .9, 0., 0., 0., 0., 0., .5, .5)
        with TemporaryDirectory() as td:
            path = Path(td) / "production.dng"
            for half in (False, True):
                for scale in (None, (1., 2.), (2., 1.)):
                    for opcodes in (None, {51022: [(1, warp)]}):
                        with self.subTest(half=half, scale=scale, warp=opcodes is not None):
                            bundles = []
                            for value in (2000, 3000, 3800):
                                write_sensor_dng(path, signal=red_point(value),
                                    neutral=(.5, 1., 1.), scale=scale, opcodes=opcodes)
                                # Compare source dependencies under one fixed
                                # AHD algorithm, rather than treating automatic
                                # algorithm selection as a sensel dependency.
                                bundles.append(raw_io.load_raw(path, scene_half_size=half,
                                    demosaic="ahd" if value == 2000 else "auto"))
                                if value == 3000:
                                    forced = raw_io.load_raw(path, scene_half_size=half, demosaic="ahd")
                                    np.testing.assert_array_equal(bundles[-1].scene_rec2020_render,
                                                                  forced.scene_rec2020_render)
                            before, clipped, upper = bundles
                            np.testing.assert_array_equal(clipped.scene_rec2020_render,
                                                          upper.scene_rec2020_render)
                            depends = np.any(before.scene_rec2020_render != clipped.scene_rec2020_render, axis=2)
                            self.assertTrue(np.any(depends))
                            self.assertIsNotNone(clipped.processing_clip_masks)
                            self.assertTrue(np.all(np.max(clipped.processing_clip_masks, axis=2)[depends] > 0))
                            result, _, _ = analyze(clipped, 4)
                            self.assertTrue(all(value == 0 for value in result.clip_pct.values()))
                            _, _, reliable, _ = reliable_scene_ev_selection(clipped, result)
                            self.assertFalse(np.any(reliable.reshape(depends.shape)[depends]))
                            np.testing.assert_array_equal(clipped.raw_image, red_point(3000))

    def test_stage2_gainmap_loss_uses_the_same_demosaic_support(self):
        # Gain only one R sensel, so no file RAW sample reaches sensor white.
        gain = (struct.pack(">4L2L2L2L", 64, 64, 65, 65, 0, 1, 1, 1, 1, 1)
                + struct.pack(">4dL", 1., 1., 0., 0., 1) + struct.pack(">f", 2.))
        with TemporaryDirectory() as td:
            path = Path(td) / "gain.dng"
            for half in (False, True):
                for scale in (None, (1., 2.), (2., 1.)):
                    with self.subTest(half=half, scale=scale):
                        scenes = []
                        for value in (1500, 3000, 3800):
                            write_sensor_dng(path, signal=red_point(value), neutral=(1., 1., 1.),
                                scale=scale, opcodes={51009: [(9, gain)]})
                            scenes.append(raw_io.load_raw(path, scene_half_size=half,
                                demosaic="ahd" if value == 1500 else "auto"))
                        before, clipped, upper = scenes
                        np.testing.assert_array_equal(clipped.scene_rec2020_render, upper.scene_rec2020_render)
                        depends = np.any(before.scene_rec2020_render != clipped.scene_rec2020_render, axis=2)
                        self.assertTrue(np.any(depends))
                        self.assertTrue(np.all(np.max(clipped.processing_clip_masks, axis=2)[depends] > 0))
                        result, _, _ = analyze(clipped, 4)
                        self.assertTrue(all(value == 0 for value in result.clip_pct.values()))

    def test_audited_ahd_radius_preserves_locality_and_covers_actual_dependencies(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "ahd.dng"
            scenes = []
            rng = np.random.default_rng(271)
            background = rng.integers(600, 1800, (128, 128), dtype=np.uint16)
            for value in (2000, 3000):
                pixels = background.copy()
                pixels[64, 64] = value
                write_sensor_dng(path, signal=pixels, neutral=(.5, 1., 1.))
                scenes.append(raw_io.load_raw(path, demosaic="ahd"))
            depends = np.any(scenes[0].scene_rec2020_render != scenes[1].scene_rec2020_render, axis=2)
            self.assertTrue(np.any(depends))
            spatial = np.max(scenes[1].processing_clip_masks, axis=2)
            self.assertTrue(np.all(spatial[depends] > 0))
            self.assertLessEqual(np.count_nonzero(spatial), 121)

    def test_auto_fallback_is_only_for_actual_overflow_and_explicit_algorithm_is_retained(self):
        ahd = getattr(rawpy.DemosaicAlgorithm, "AHD", None)
        if ahd is None or not ahd.isSupported:
            self.skipTest("LibRaw build does not support AHD")
        with TemporaryDirectory() as td:
            path = Path(td) / "selection.dng"
            for value, neutral in ((2000, (.5, 1., 1.)), (4095, (1., 1., 1.))):
                write_sensor_dng(path, signal=red_point(value), neutral=neutral)
                with rawpy.imread(str(path)) as raw:
                    expected = raw_io.resolve_demosaic_algorithm(raw, "auto")
                bundle = raw_io.load_raw(path)
                self.assertEqual(bundle.noise_decode["demosaic_algorithm"], expected.name)
                self.assertIsNone(bundle.noise_decode["loss_support"])
                if value == 4095:
                    # Exact source white is physical saturation, not a new WB
                    # overflow or reason to change automatic interpolation.
                    self.assertTrue(bundle.scene_loss_support_untrusted)
                    self.assertIn("source saturation", bundle.scene_correction_note)
                else:
                    self.assertFalse(bundle.scene_loss_support_untrusted)
                    self.assertIsNone(bundle.scene_correction_note)
            write_sensor_dng(path, signal=red_point(), neutral=(.5, 1., 1.))
            automatic = raw_io.load_raw(path)
            explicit_ahd = raw_io.load_raw(path, demosaic="ahd")
            self.assertEqual(automatic.noise_decode["demosaic_algorithm"], "AHD")
            self.assertIn("bayer-AHD", automatic.noise_decode["loss_support"])
            self.assertIn("AHD", automatic.scene_correction_note)
            self.assertLess(np.count_nonzero(np.max(automatic.processing_clip_masks, axis=2)), 256)
            np.testing.assert_array_equal(automatic.scene_rec2020_render,
                                          explicit_ahd.scene_rec2020_render)
            dht = getattr(rawpy.DemosaicAlgorithm, "DHT", None)
            if dht is not None and dht.isSupported:
                explicit = raw_io.load_raw(path, demosaic="dht")
                self.assertEqual(explicit.noise_decode["demosaic_algorithm"], "DHT")
                self.assertIn("global-conservative", explicit.noise_decode["loss_support"])
                self.assertIn("整帧保守", explicit.scene_correction_note)
                self.assertTrue(explicit.scene_loss_support_untrusted)
                self.assertEqual(explicit.scene_reliability_source, "decoder-support-untrusted")
                # Uncertified propagation denies reliability through the flag,
                # never by claiming every colour is clipped everywhere.
                np.testing.assert_array_equal(explicit.clip_masks[16, 16], 0)
                if explicit.processing_clip_masks is not None:
                    np.testing.assert_array_equal(explicit.processing_clip_masks[16, 16], 0)
                with rawpy.imread(str(path)) as raw:
                    camera = raw_io.render_to_scene_rec2020(raw, "clip", False, dht,
                        raw_io._fixed_asshot_wb_kwargs(raw.camera_whitebalance), camera_rgb=True)
                    from dngscan import dng_opcodes as ops
                    expected = ops.camera_to_rec2020(camera, ops.libraw_camera_matrix(
                        raw.color_matrix, raw.rgb_xyz_matrix, is_dng=True))
                np.testing.assert_array_equal(explicit.scene_rec2020_render, expected)

    def test_selection_without_returned_loss_still_observes_stage2_clipping(self):
        gain = (struct.pack(">4L2L2L2L", 64, 64, 65, 65, 0, 1, 1, 1, 1, 1)
                + struct.pack(">4dL", 1., 1., 0., 0., 1) + struct.pack(">f", 2.))
        with TemporaryDirectory() as td:
            path = Path(td) / "selection-without-mask.dng"
            write_sensor_dng(path, signal=red_point(), neutral=(1., 1., 1.),
                opcodes={51009: [(9, gain)]})
            production = raw_io.load_raw(path)
            with rawpy.imread(str(path)) as raw:
                scene, loss, recipe, _ = raw_io._decode_corrected_libraw(raw, path,
                    production.evidence, "clip", False,
                    raw_io.resolve_demosaic_algorithm(raw, "auto"),
                    track_loss=False, allow_loss_fallback=True)
            self.assertEqual(recipe.demosaic_algorithm, "AHD")
            self.assertIsNone(loss)
            self.assertIsNone(recipe.loss_support)
            np.testing.assert_array_equal(scene, production.scene_rec2020_render)


if __name__ == "__main__":
    unittest.main()
