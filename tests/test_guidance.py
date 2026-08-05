# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import unittest

import numpy as np

from dngscan.guidance import (
    CLIP_CLASS_B,
    CLIP_CLASS_G,
    CLIP_CLASS_NONE,
    CLIP_CLASS_RGB,
    build_raw_guidance_maps,
    clip_class_from_masks,
    color_path_weight,
    raw_color_permission,
    sector_hue_multiplier,
    snr_confidence_from_ev,
)
from dngscan.models import RawBundle


class GuidanceTest(unittest.TestCase):
    def test_clip_class_discrete(self) -> None:
        masks = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.6, 0.6, 0.6]], dtype=np.float32)
        classes = clip_class_from_masks(masks)
        self.assertEqual(int(classes[0]), CLIP_CLASS_NONE)
        self.assertEqual(int(classes[1]), CLIP_CLASS_G)
        self.assertEqual(int(classes[2]), CLIP_CLASS_RGB)

    def test_unclipped_midtone_low_permission(self) -> None:
        masks = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        ev = np.asarray([-1.0], dtype=np.float32)
        self.assertLess(float(raw_color_permission(masks)[0]), 0.05)
        self.assertLess(float(color_path_weight(masks, ev)[0]), 0.08)

    def test_rgb_clip_high_permission(self) -> None:
        masks = np.asarray([[0.9, 0.85, 0.95]], dtype=np.float32)
        ev = np.asarray([2.5], dtype=np.float32)
        self.assertGreater(float(color_path_weight(masks, ev)[0]), 0.5)

    def test_single_blue_clip_is_not_misclassified_as_multi_channel(self) -> None:
        red = raw_color_permission(np.asarray([[0.9, 0.0, 0.0]], dtype=np.float32))
        blue = raw_color_permission(np.asarray([[0.0, 0.0, 0.9]], dtype=np.float32))
        self.assertAlmostEqual(float(red[0]), float(blue[0]), places=6)
        self.assertEqual(int(clip_class_from_masks(np.asarray([[0.0, 0.0, 0.9]], dtype=np.float32))[0]), CLIP_CLASS_B)

    def test_guidance_uses_raw_headroom_and_sensor_snr(self) -> None:
        from pathlib import Path
        from types import SimpleNamespace

        raw = np.zeros((4, 4), dtype=np.uint16)
        colors = np.asarray(
            [[0, 1, 0, 1], [3, 2, 3, 2], [0, 1, 0, 1], [3, 2, 3, 2]], dtype=np.uint8
        )
        raw[0, 0] = 995
        raw[0, 1] = 950
        raw[1, 0] = 950
        raw[1, 1] = 900
        bundle = RawBundle(
            path=Path("test.dng"), raw_image=raw, raw_colors=colors,
            xyz_render=np.zeros((2, 2, 3), dtype=np.uint16), render_scale=1000.0,
            scene_rec2020_render=np.zeros((2, 2, 3), dtype=np.uint16), scene_scale=1000.0,
            white_level=1000, black_levels=[0.0] * 4, camera_wb=[1.0] * 4,
            color_desc="RGBG", raw_pattern=[[0, 1], [3, 2]], camera_white_levels=[1000.0] * 4,
            clip_masks=np.zeros((2, 2, 3), dtype=np.float16),
        )
        maps = build_raw_guidance_maps(bundle, SimpleNamespace(gain_e_per_dn=1.0, prior_read_noise_e=2.0))
        self.assertIsNotNone(maps)
        self.assertLess(float(maps.headroom[0, 0, 0]), 0.01)
        self.assertEqual(int(maps.clip_class[0, 0]) & 1, 1)
        self.assertLess(float(maps.snr_confidence[0, 1]), float(maps.snr_confidence[0, 0]))

    def test_skin_midtone_sector_reduces_weight(self) -> None:
        skin = sector_hue_multiplier(
            np.asarray([[0.42, 0.22, 0.18]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
        )
        neutral = sector_hue_multiplier(
            np.asarray([[0.22, 0.22, 0.22]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
        )
        self.assertLess(float(skin[0]), float(neutral[0]))

    def test_snr_confidence_rises_with_ev(self) -> None:
        low = float(snr_confidence_from_ev(np.asarray([-10.0]), -12.0)[0])
        high = float(snr_confidence_from_ev(np.asarray([0.0]), -12.0)[0])
        self.assertLess(low, high)


if __name__ == "__main__":
    unittest.main()
