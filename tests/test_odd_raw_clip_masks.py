# SPDX-License-Identifier: GPL-3.0-or-later
"""Partial CFA cells stay evidence unless the decoded window drops them."""
from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

from dngscan._deps import np, rawpy
from dngscan.analysis import analyze
from dngscan.guidance import (
    CLIP_CLASS_G, build_raw_guidance_maps, _raw_headroom_rgb_reference,
)
from dngscan.raw_io import (
    _align_sensor_cell_loss, _bin_2x2_max, _build_bayer_clip_mask_planes,
    build_clip_masks, load_raw,
)
from dngscan.spatial_black import SpatialBlack, clip_mask
from dngscan.tone import reliable_scene_ev_selection
from tests.test_pipeline_corrections import write_sensor_dng
from tests.test_readout_contract import edit_ifd


def write_odd_dng(path, pixels, *, pattern=(0, 1, 1, 2), orientation=1,
                  scale=None, warp=False, spatial_black=False, crop=None):
    """Keep known camera calibration, replace the actual TIFF sensor strip."""
    write_sensor_dng(path, signal=1000, scale=scale, opcodes={51022: [
        (1, struct.pack(">L8d", 1, 1., .03, 0., 0., 0., 0., .5, .5))
    ]} if warp else None)
    pixels = np.asarray(pixels, dtype="<u2")
    height, width = pixels.shape
    data = bytearray(path.read_bytes())
    offset = len(data)
    data.extend(pixels.tobytes())
    path.write_bytes(data)
    replacements = {
        256: (4, 1, struct.pack("<L", width)),
        257: (4, 1, struct.pack("<L", height)),
        273: (4, 1, struct.pack("<L", offset)),
        274: (3, 1, struct.pack("<H", orientation)),
        278: (4, 1, struct.pack("<L", height)),
        279: (4, 1, struct.pack("<L", pixels.nbytes)),
        33422: (1, 4, bytes(pattern)),
        50829: (4, 4, struct.pack("<LLLL", 0, 0, height, width)),
        50720: (4, 2, struct.pack("<LL", width, height)),
    }
    if spatial_black:
        deltas = np.linspace(-16., 16., width)
        replacements[50715] = (10, width, b"".join(
            struct.pack("<ll", int(round(value * 1000)), 1000) for value in deltas))
    if crop is not None:
        replacements[50720] = (4, 2, struct.pack("<LL", *crop))
    edit_ifd(path, replacements)


class PartialClipPoolTests(unittest.TestCase):
    def test_partial_pool_uses_only_present_samples_and_keeps_complete_box_contract(self):
        for height, width in ((1, 1), (1, 5), (5, 1), (5, 7), (4, 5), (5, 4), (4, 4)):
            for dtype in (np.float16, np.float32):
                with self.subTest(shape=(height, width), dtype=dtype):
                    source = np.arange(height * width * 3, dtype=dtype).reshape(height, width, 3)
                    expected = np.empty(((height + 1) // 2, (width + 1) // 2, 3), dtype=dtype)
                    for row in range(expected.shape[0]):
                        for col in range(expected.shape[1]):
                            expected[row, col] = source[2*row:2*row+2, 2*col:2*col+2].max(axis=(0, 1))
                    np.testing.assert_array_equal(_bin_2x2_max(source, include_partial=True), expected)
                    np.testing.assert_array_equal(_bin_2x2_max(source), expected[:height//2, :width//2])

    def test_actual_complete_box_reduction_does_not_include_discarded_edges(self):
        source = np.zeros((5, 7, 3), np.float16)
        source[-1, :, 0] = 1
        source[:, -1, 1] = 1
        self.assertFalse(np.any(_bin_2x2_max(source)))
        self.assertTrue(np.any(_bin_2x2_max(source, include_partial=True)))

    def test_bayer_generic_and_spatial_black_partial_cells_agree(self):
        for height, width in ((1, 1), (1, 5), (5, 1), (5, 7), (4, 7), (5, 6), (129, 131)):
            for pattern in (((0, 1), (3, 2)), ((2, 3), (1, 0)),
                            ((1, 0), (2, 3)), ((3, 2), (0, 1))):
                with self.subTest(shape=(height, width), pattern=pattern):
                    colors = np.tile(np.asarray(pattern, np.uint8),
                                     ((height+1)//2, (width+1)//2))[:height, :width]
                    pixels = np.full((height, width), 100, np.uint16)
                    pixels[-1, :] = pixels[:, -1] = 1000
                    direct = _build_bayer_clip_mask_planes(pixels, pattern, "RGBG", 1000,
                                                           [0.] * 4, [1000.] * 4)
                    soft = np.zeros((height, width, 3), np.float32)
                    for channel, ids in enumerate(((0,), (1, 3), (2,))):
                        soft[..., channel] = (pixels == 1000) & np.isin(colors, ids)
                    expected = _bin_2x2_max(soft, include_partial=True)
                    np.testing.assert_array_equal(direct, expected)
                    model = SpatialBlack(np.zeros(width), np.zeros(height),
                                         np.zeros((1, 1, 1)), (0, 0), (0.,))
                    np.testing.assert_array_equal(
                        clip_mask(pixels, colors, "RGBG", [0.] * 4, [1000.] * 4, model), expected)
                    kwargs = dict(color_desc="RGBG", white_level=1000, black_levels=[0.] * 4,
                                  camera_white_levels=[1000.] * 4, orientation_flip=0,
                                  scene_shape=(height, width))
                    np.testing.assert_array_equal(build_clip_masks(pixels, colors, **kwargs),
                                                  build_clip_masks(pixels, colors, raw_pattern=pattern, **kwargs))

    def test_spatial_black_linear_planes_keep_partial_edges(self):
        source = np.zeros((129, 131, 3), np.uint16)
        source[-1, :, 0] = source[:, -1, 2] = 1000
        model = SpatialBlack(np.zeros(131), np.zeros(129), np.zeros((1, 1, 3)), (0, 0), (0.,) * 3)
        actual = clip_mask(source, None, "RGBG", [0.] * 4, [1000.] * 4, model)
        np.testing.assert_array_equal(actual, _bin_2x2_max((source == 1000).astype(np.float32), include_partial=True))

    def test_crop_does_not_reintroduce_an_excluded_partial_sensor_cell(self):
        source = np.full((125, 127), 1000, np.uint16)
        source[-1, :] = source[:, -1] = 4095
        pattern = [[0, 1], [3, 2]]
        colors = np.tile(pattern, (63, 64))[:125, :127]
        for flip in range(8):
            shape = (126, 124) if flip & 4 else (124, 126)
            with self.subTest(flip=flip):
                masks = build_clip_masks(source, colors, "RGBG", 4095, [0.] * 4,
                                         [4095.] * 4, flip, shape, pattern,
                                         crop_sensor=(0., 0., 124., 126.))
                self.assertFalse(np.any(masks))

    def test_arbitrary_partial_cfa_period_registers_actual_cell_extent(self):
        for height, width, ph, pw in ((13, 17, 6, 6), (5, 7, 2, 3)):
            cells = np.zeros(((height+ph-1)//ph, (width+pw-1)//pw, 3), np.float32)
            cells[-1, :, 0] = cells[:, -1, 1] = 1
            expected = np.repeat(np.repeat(cells, ph, axis=0), pw, axis=1)[:height, :width]
            with self.subTest(shape=(height, width), period=(ph, pw)):
                actual = _align_sensor_cell_loss(cells, (height, width), (height, width),
                                                 0, period=(ph, pw))
                np.testing.assert_array_equal(actual, expected)
                crop = (0., 0., float(height//ph*ph), float(width//pw*pw))
                cropped = _align_sensor_cell_loss(cells, (height, width),
                                                  (int(crop[2]), int(crop[3])), 0,
                                                  crop_sensor=crop, period=(ph, pw))
                self.assertFalse(np.any(cropped))


@unittest.skipIf(rawpy is None, "rawpy unavailable")
class OddRawClipPipelineTests(unittest.TestCase):
    def pair(self, path, *, point, half, **settings):
        bundles = []
        for value in (3000, 4095):
            pixels = np.full((125, 127), 1000, np.uint16)
            pixels[point] = value
            write_odd_dng(path, pixels, **settings)
            bundles.append(load_raw(path, scene_half_size=half))
        return bundles

    def test_real_odd_bayer_residual_green_row_and_column_reach_reliability_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odd-sensor.dng"
            for half in (False, True):
                for point in ((124, 125), (123, 126)):
                    with self.subTest(half=half, point=point):
                        before, clipped = self.pair(path, point=point, half=half)
                        shape = clipped.scene_rec2020_render.shape[:2]
                        self.assertEqual(clipped.raw_image.shape, (125, 127))
                        self.assertEqual(shape, (63, 64) if half else (125, 127))
                        sample = (point[0]//2, point[1]//2) if half else point
                        self.assertTrue(np.any(clipped.scene_rec2020_render[sample]
                                               != before.scene_rec2020_render[sample]))
                        # The original counterexample has no processing mask;
                        # only the sensor mask can prevent false certification.
                        self.assertIsNone(clipped.processing_clip_masks)
                        self.assertGreater(float(np.max(clipped.clip_masks[sample])), .1)
                        result, _, _ = analyze(clipped, 4)
                        self.assertGreater(max(result.clip_pct.values()), 0.)
                        _, _, reliable, _ = reliable_scene_ev_selection(clipped, result)
                        self.assertFalse(reliable.reshape(shape)[sample])
                        maps = build_raw_guidance_maps(clipped, result)
                        self.assertEqual(float(maps.headroom[sample][1]), 0.)
                        self.assertEqual(int(maps.clip_class[sample]), CLIP_CLASS_G)
                        self.assertGreater(float(maps.raw_permission[sample]), 0.)
                        np.testing.assert_array_equal(
                            maps.headroom, _raw_headroom_rgb_reference(clipped, shape, result).astype(np.float16))

    def test_real_spatial_black_odd_edges_are_not_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odd-spatial-black.dng"
            for half in (False, True):
                for point in ((124, 125), (123, 126)):
                    with self.subTest(half=half, point=point):
                        before, clipped = self.pair(path, point=point, half=half, spatial_black=True)
                        self.assertIsNotNone(clipped.evidence.spatial_black)
                        difference = np.max(np.abs(clipped.scene_rec2020_render - before.scene_rec2020_render), axis=2)
                        sample = np.unravel_index(np.argmax(difference), difference.shape)
                        self.assertGreater(float(difference[sample]), 0.)
                        self.assertGreater(float(np.max(clipped.clip_masks[sample])), .1)
                        result, _, _ = analyze(clipped, 4)
                        _, _, reliable, _ = reliable_scene_ev_selection(clipped, result)
                        self.assertFalse(reliable.reshape(difference.shape)[sample])
                        maps = build_raw_guidance_maps(clipped, result)
                        self.assertEqual(float(maps.headroom[sample][1]), 0.)
                        self.assertEqual(int(maps.clip_class[sample]), CLIP_CLASS_G)
                        self.assertGreater(float(maps.raw_permission[sample]), 0.)

    def test_real_crop_excludes_partial_edge_from_headroom_and_clip_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odd-cropped.dng"
            for half in (False, True):
                for orientation in (1, 6):
                    with self.subTest(half=half, orientation=orientation):
                        pixels = np.full((125, 127), 1000, np.uint16)
                        pixels[-1, :] = pixels[:, -1] = 4095
                        write_odd_dng(path, pixels, crop=(126, 124), orientation=orientation)
                        bundle = load_raw(path, scene_half_size=half)
                        result, _, _ = analyze(bundle, 4)
                        maps = build_raw_guidance_maps(bundle, result)
                        self.assertGreater(max(result.clip_pct.values()), 0.)
                        self.assertFalse(np.any(bundle.clip_masks))
                        self.assertFalse(np.any(maps.clip_class))
                        self.assertFalse(np.any(maps.raw_permission))
                        self.assertTrue(np.all(maps.headroom > .7))

    def test_real_odd_edge_with_bayer_phase_orientation_scale_and_warp(self):
        configurations = (
            ((0, 1, 1, 2), 2, None, False),
            ((2, 1, 1, 0), 6, (1.03, 1.), False),
            ((1, 0, 2, 1), 4, None, True),
            ((1, 2, 0, 1), 8, (1., 1.03), True),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odd-geometry.dng"
            for half in (False, True):
                for pattern, orientation, scale, warp in configurations:
                    col = 125 if pattern[1] == 1 else 126
                    with self.subTest(half=half, pattern=pattern, orientation=orientation,
                                      scale=scale, warp=warp):
                        before, clipped = self.pair(path, point=(124, col), half=half,
                                                   pattern=pattern, orientation=orientation,
                                                   scale=scale, warp=warp)
                        difference = np.max(np.abs(clipped.scene_rec2020_render - before.scene_rec2020_render), axis=2)
                        sample = np.unravel_index(np.argmax(difference), difference.shape)
                        self.assertGreater(float(difference[sample]), 0.)
                        self.assertGreater(float(np.max(clipped.clip_masks[sample])), .1)
                        result, _, _ = analyze(clipped, 4)
                        _, _, reliable, _ = reliable_scene_ev_selection(clipped, result)
                        self.assertFalse(reliable.reshape(difference.shape)[sample])


if __name__ == "__main__":
    unittest.main()
