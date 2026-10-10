# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical chroma-band sampling across decode, crop, orientation and proxies."""
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from dngscan.chroma_nr import atrous_levels_for
from dngscan.sampling_geometry import sensor_px_per_render_axes, sensor_window_from_recipe
from dngscan.render import sensor_px_per_render_px


class SensorSamplingGeometryTests(unittest.TestCase):
    def test_half_decode_and_further_proxy_use_one_native_window(self):
        bundle = SimpleNamespace(scene_sensor_window_shape=(4000, 6000),
                                 proxy_scale=1, orientation_flip=0)
        self.assertEqual(sensor_px_per_render_axes(bundle, 2000, 3000), (2, 2))
        self.assertEqual(sensor_px_per_render_axes(bundle, 400, 600), (10, 10))
        bundle.proxy_scale = 10
        self.assertEqual(sensor_px_per_render_px(bundle, 400, 600), 10)

    def test_crop_and_rotation_keep_sensor_axes(self):
        bundle = SimpleNamespace(scene_sensor_window_shape=(1200, 2000),
                                 orientation_flip=6)
        self.assertEqual(sensor_px_per_render_axes(bundle, 1000, 600), (2, 2))
        bundle.orientation_flip = 0
        self.assertEqual(sensor_px_per_render_axes(bundle, 600, 2000), (2, 1))

    def test_default_scale_keeps_band_inside_both_sensor_axes(self):
        bundle = SimpleNamespace(scene_sensor_window_shape=(128, 128), orientation_flip=0)
        axes = sensor_px_per_render_axes(bundle, 128, 256)
        self.assertEqual(axes, (1, .5))
        levels = atrous_levels_for(1, axis_factors=axes)
        self.assertEqual(levels, (4, 5, 6))
        for k in levels:
            centres = np.asarray(axes) * (2 ** k) * np.sqrt(2)
            self.assertTrue(np.all((centres >= 8) & (centres <= 128)))

    def test_half_and_full_decimation_select_same_sensor_band(self):
        bundle = SimpleNamespace(scene_sensor_window_shape=(4000, 6000), orientation_flip=0)
        selected = []
        for h, w in ((4000, 6000), (2000, 3000), (400, 600)):
            sy, sx = sensor_px_per_render_axes(bundle, h, w)
            axes = (sy * h / 200, sx * w / 300)
            selected.append(atrous_levels_for(1, axis_factors=axes))
        self.assertEqual(selected[0], selected[1])
        self.assertEqual(selected[1], selected[2])

    def test_odd_final_reduction_keeps_only_retained_sensor_pixels(self):
        recipe = SimpleNamespace(noise_geometry={
            'effective_sensor_crop': [10, 20, 101, 201],
            'decoded_crop': [0, 0, 101, 201], 'post_decode_reduction': 2})
        self.assertEqual(sensor_window_from_recipe(recipe, (50, 100), 0), (100, 200))
        self.assertEqual(sensor_window_from_recipe(recipe, (100, 50), 6), (100, 200))

    def test_legacy_proxy_ruler_and_released_raw_buffer(self):
        bundle = SimpleNamespace(raw_image=None, proxy_scale=5,
                                 scene_rec2020_render=np.empty((100, 150, 3)))
        self.assertEqual(sensor_px_per_render_axes(bundle, 50, 75), (10, 10))
        self.assertEqual(sensor_px_per_render_px(SimpleNamespace(proxy_scale=5.94), 10, 10), 5.94)

    def test_invalid_axis_scales_are_rejected(self):
        for axes in ((0, 1), (-1, 1), (np.nan, 1), (1,), (1, np.inf)):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                atrous_levels_for(1, axis_factors=axes)

    def test_cache_keeps_native_window_after_discarding_raw(self):
        from dngscan.gui.preview_cache import _bundle_metadata, _bundle_from_cache
        from tests.test_preview_cache import _bundle
        bundle = _bundle()
        bundle.scene_sensor_window_shape = (1200., 2000.)
        meta = _bundle_metadata(bundle)
        restored = _bundle_from_cache(bundle.path, meta, bundle.scene_rec2020_render, None, None)
        self.assertIsNone(restored.raw_image)
        self.assertEqual(restored.scene_sensor_window_shape, (1200., 2000.))
        self.assertEqual(sensor_px_per_render_axes(restored, 600, 1000), (2, 2))

    def test_real_libraw_decode_sampling_is_not_gui_proxy_metadata(self):
        from dngscan.raw_io import load_raw, release_analysis_buffers
        from tests.test_pipeline_corrections import write_sensor_dng
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'sensor.dng'
            for scale in (None, [2., 1.]):
                write_sensor_dng(path, signal=1000, scale=scale)
                for half in (False, True):
                    with self.subTest(default_scale=scale, half=half):
                        bundle = load_raw(path, scene_half_size=half)
                        self.assertEqual(bundle.scene_sensor_window_shape, (128, 128))
                        h, w = bundle.scene_rec2020_render.shape[:2]
                        self.assertEqual(sensor_px_per_render_axes(bundle, h, w), (128/h, 128/w))
                        released = release_analysis_buffers(bundle)
                        self.assertEqual(sensor_px_per_render_axes(released, h, w), (128/h, 128/w))

    def test_odd_post_orientation_reduction_keeps_native_crop_masks_and_reference(self):
        from dngscan import raw_io
        from dngscan._deps import rawpy
        from dngscan.scene_reference import reliable_reference_samples
        from tests.test_pipeline_corrections import write_sensor_dng
        y, x = np.indices((128, 128))
        pixels = (400+y*8+x*4).astype(np.uint16)
        pixels[30:50, 50:70] = 4095
        # EXIF orientation values corresponding to all eight LibRaw flip codes.
        orientations = (1, 2, 4, 3, 5, 8, 6, 7)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'odd-oriented-trim.dng'
            for flip, orientation in enumerate(orientations):
                with self.subTest(flip=flip):
                    write_sensor_dng(path, signal=pixels, opcodes={
                        51022: [(6, struct.pack('>4l', 11, 13, 120, 116))]})
                    # Replace an optional tag in the real TIFF IFD, keeping all
                    # pixel/opcode offsets intact. LibRaw must read the actual
                    # orientation rather than a mocked evidence field.
                    content = bytearray(path.read_bytes())
                    count, = struct.unpack_from('<H', content, 8)
                    for i in range(count):
                        off = 10+12*i
                        tag, = struct.unpack_from('<H', content, off)
                        if tag == 50730:
                            content[off:off+12] = struct.pack('<HHIHH', 274, 3, 1, orientation, 0)
                            break
                    else:
                        self.fail('fixture has no optional BaselineExposure tag')
                    path.write_bytes(content)
                    # Use the audited local demosaic for the nonempty reference
                    # oracle. Source-saturated DHT now withdraws it globally.
                    full = raw_io.load_raw(path, demosaic="ahd")
                    bundle = raw_io.load_raw(path, scene_half_size=True, demosaic="ahd")
                    self.assertEqual(bundle.orientation_flip, flip)
                    h, w = bundle.scene_rec2020_render.shape[:2]
                    np.testing.assert_array_equal(bundle.scene_rec2020_render,
                        full.scene_rec2020_render[:2*h, :2*w].reshape(h, 2, w, 2, 3).mean(axis=(1, 3)))
                    # The original 109x103 native crop loses one pixel per axis;
                    # mirroring determines which native edge is discarded.
                    expected_crop = (float(11+bool(flip & 2)), float(13+bool(flip & 1)), 108., 102.)
                    self.assertEqual(bundle.scene_crop_sensor, expected_crop)
                    self.assertEqual(bundle.scene_sensor_window_shape, (108., 102.))
                    axes = (108/h, 102/w) if not flip & 4 else (102/h, 108/w)
                    self.assertEqual(sensor_px_per_render_axes(bundle, h, w), axes)
                    expected_masks = raw_io.build_clip_masks(
                        bundle.raw_image, bundle.raw_colors, bundle.color_desc,
                        bundle.white_level, bundle.black_levels, bundle.camera_white_levels,
                        flip, (h, w), bundle.raw_pattern, crop_sensor=expected_crop)
                    if bundle.processing_clip_masks is not None:
                        expected_masks = np.maximum(expected_masks, bundle.processing_clip_masks)
                    np.testing.assert_array_equal(bundle.clip_masks, expected_masks)
                    with rawpy.imread(str(path)) as raw:
                        scene, loss, recipe, _ = raw_io._decode_corrected_libraw(
                            raw, path, bundle.evidence, 'clip', True, rawpy.DemosaicAlgorithm.AHD)
                    self.assertEqual(recipe.crop, expected_crop)
                    self.assertEqual(tuple(recipe.noise_geometry['effective_sensor_crop']), expected_crop)
                    self.assertEqual(tuple(recipe.noise_geometry['decoded_crop']), expected_crop)
                    self.assertEqual(sensor_window_from_recipe(recipe, scene.shape[:2], flip), (108., 102.))
                    samples, percent = reliable_reference_samples(
                        bundle.evidence, scene, bundle.scene_scale, loss, recipe)
                    reliable = np.max(bundle.clip_masks.reshape(-1, 3), axis=1) < .1
                    if bundle.scene_reliability_exclusion is not None:
                        reliable &= bundle.scene_reliability_exclusion.reshape(-1) == 0
                    expected = (scene.reshape(-1, 3)/np.float32(bundle.scene_scale))[reliable]
                    np.testing.assert_array_equal(samples, expected)
                    self.assertAlmostEqual(percent, np.mean(reliable)*100)
