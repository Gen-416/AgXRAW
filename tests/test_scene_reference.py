# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference evidence stays in its own geometry and declared exposure units."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dngscan.models import RawBundle, RawEvidence
from dngscan.scene_reference import reliable_reference_samples
from dngscan.raw_io import rebalance_raw_bundle


def _evidence(raw):
    return RawEvidence(path=Path('reference.dng'), raw_image=raw,
        raw_colors=np.tile(np.asarray([[0, 1], [3, 2]], dtype=np.uint8),
                           (raw.shape[0] // 2, raw.shape[1] // 2)),
        white_level=1000, black_levels=[0.] * 4, camera_wb=[1.] * 4,
        daylight_wb=[1.] * 4, color_desc='RGBG', raw_pattern=[[0, 1], [3, 2]],
        camera_white_levels=[1000.] * 4, orientation_flip=0, xyz_to_cam=np.eye(3))


class IndependentReferenceTests(unittest.TestCase):
    def test_dark_loss_mask_preserves_unclipped_reference_highlights(self):
        raw = np.full((64, 64), 500, dtype=np.uint16)
        rgb = np.full((32, 32, 3), 10., dtype=np.float32)
        rgb[:2] = 1.
        rgb[-2:] = 200.
        loss = np.zeros(rgb.shape, dtype=np.float16)
        loss[:2] = 1
        reference, pct = reliable_reference_samples(
            _evidence(raw), rgb, 100., loss, SimpleNamespace(post=(), crop=None))
        self.assertEqual(pct, 93.75)
        self.assertEqual(reference.shape, (960, 3))
        self.assertEqual(float(reference.max()), 2.)
        self.assertGreater(float(reference.min()), .01)
        self.assertEqual(float(rgb.max()), 200.)  # Input scene remains untouched.

    def test_sensor_clipped_bright_area_is_removed_before_tail_measurement(self):
        raw = np.full((64, 64), 500, dtype=np.uint16)
        raw[-16:] = 1000
        rgb = np.full((32, 32, 3), 10., dtype=np.float32)
        rgb[-8:] = 200.
        reference, pct = reliable_reference_samples(
            _evidence(raw), rgb, 100., None, SimpleNamespace(post=(), crop=None))
        self.assertGreater(len(reference), 256)
        self.assertLess(pct, 100.)
        self.assertAlmostEqual(float(reference.max()), .1, places=6)

    def test_fully_clipped_reference_is_present_empty_not_unavailable(self):
        raw = np.full((64, 64), 1000, dtype=np.uint16)
        reference, pct = reliable_reference_samples(_evidence(raw),
            np.full((32, 32, 3), 200., dtype=np.float32), 100., None,
            SimpleNamespace(post=(), crop=None))
        self.assertEqual(reference.shape, (0, 3))
        self.assertEqual(pct, 0.)

    def test_hot_white_balance_transforms_reference_in_same_rgb_basis(self):
        raw = np.full((64, 64), 500, dtype=np.uint16)
        evidence = _evidence(raw)
        rgb = np.full((32, 32, 3), 20., dtype=np.float32)
        reference = np.full((1000, 3), .2, dtype=np.float32)
        bundle = RawBundle(path=evidence.path, raw_image=raw, raw_colors=evidence.raw_colors,
            xyz_render=rgb.copy(), render_scale=100., scene_rec2020_render=rgb, scene_scale=100.,
            white_level=1000, black_levels=[0.] * 4, camera_wb=[1.] * 4,
            color_desc='RGBG', raw_pattern=evidence.raw_pattern, camera_white_levels=[1000.] * 4,
            daylight_wb=[2., 1., .5, 1.], decode_wb=[1.] * 4,
            wb_xyz_to_cam=np.eye(3), scene_reliable_reference_rec2020=reference)
        # Lock a nontrivial transform to test the coordinate handoff, not the
        # unrelated calibration solver or an identity WB.
        matrix = np.diag([2., 1., .5]).astype(np.float32)
        with patch('dngscan.raw_io.hot_wb_matrix_rec2020', return_value=matrix):
            balanced = rebalance_raw_bundle(bundle, 'daylight')
        np.testing.assert_allclose(balanced.scene_reliable_reference_rec2020[0],
            balanced.scene_rec2020_render[0, 0] / 100., atol=1e-7)
        np.testing.assert_array_equal(reference, np.full((1000, 3), .2, dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
