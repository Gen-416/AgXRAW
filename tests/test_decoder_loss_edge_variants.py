# SPDX-License-Identifier: GPL-3.0-or-later
"""Real decoder variants where raster shape alone does not identify support."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from dngscan.analysis import analyze
from dngscan.raw_io import load_raw
from dngscan.tone import reliable_scene_ev_selection
from tests.test_pipeline_corrections import write_sensor_dng


class DecoderLossEdgeVariants(unittest.TestCase):
    def test_half_green_loss_moves_even_when_default_scale_keeps_dimensions(self):
        # Both native greens share the half-size output G. One clipped green
        # averages below uint16's ceiling, so the final ceiling check cannot
        # rescue an incorrectly transported source mask.
        for scale, dependent in (((1.007, 1.), (32, 33)),
                                 ((1., 1.007), (33, 32))):
            with self.subTest(scale=scale), TemporaryDirectory() as directory:
                path = Path(directory) / "same-rounded-size.dng"
                bundles = []
                for value in (2000, 3000, 3800):
                    pixels = np.full((128, 128), 1000, np.uint16)
                    pixels[64, 65] = value
                    write_sensor_dng(path, signal=pixels, neutral=(1., .5, 1.), scale=scale)
                    bundles.append(load_raw(path, scene_half_size=True))
                before, clipped, upper = bundles
                self.assertEqual(clipped.scene_rec2020_render.shape, (64, 64, 3))
                np.testing.assert_array_equal(clipped.scene_rec2020_render,
                                              upper.scene_rec2020_render)
                depends = np.any(before.scene_rec2020_render != clipped.scene_rec2020_render,
                                 axis=2)
                np.testing.assert_array_equal(np.argwhere(depends), [dependent])
                self.assertTrue(np.all(np.max(clipped.processing_clip_masks, axis=2)[depends] > 0))
                result, _, _ = analyze(clipped, 4)
                self.assertTrue(all(value == 0 for value in result.clip_pct.values()))
                _, _, reliable, _ = reliable_scene_ev_selection(clipped, result)
                self.assertFalse(np.any(reliable.reshape(depends.shape)[depends]))

    def test_linear_dng_reports_no_demosaic_when_half_size_is_requested(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "linear.dng"
            pixels = np.full((128, 128), 1000, np.uint16)
            pixels[64, 64] = 3000
            write_sensor_dng(path, linear=True, signal=pixels, neutral=(.5, 1., 1.))
            full = load_raw(path)
            half = load_raw(path, scene_half_size=True)
            # LibRaw does not shrink already-linear camera planes.
            self.assertEqual(half.scene_rec2020_render.shape, (128, 128, 3))
            np.testing.assert_array_equal(full.scene_rec2020_render, half.scene_rec2020_render)
            np.testing.assert_array_equal(full.processing_clip_masks, half.processing_clip_masks)
            for bundle in (full, half):
                self.assertEqual(bundle.noise_decode["demosaic_algorithm"], "linear-planes")
                self.assertTrue(bundle.noise_decode["loss_support"].startswith("linear-planes:"))


if __name__ == "__main__":
    unittest.main()
