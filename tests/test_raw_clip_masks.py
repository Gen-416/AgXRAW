# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import unittest
from types import SimpleNamespace

from dngscan._deps import np
from dngscan.raw_io import build_clip_masks, refresh_clip_masks_from_fullwell


def test_build_clip_masks_uses_cfa_max_binning():
    raw = np.full((4, 4), 100, dtype=np.uint16)
    colors = np.asarray(
        [
            [0, 1, 0, 1],
            [3, 2, 3, 2],
            [0, 1, 0, 1],
            [3, 2, 3, 2],
        ],
        dtype=np.uint8,
    )
    raw[0, 0] = 990
    raw[1, 1] = 970
    masks = build_clip_masks(
        raw,
        colors,
        "RGBG",
        white_level=1000,
        black_levels=[0.0, 0.0, 0.0, 0.0],
        camera_white_levels=[1000.0, 1000.0, 1000.0, 1000.0],
        orientation_flip=0,
        scene_shape=(2, 2),
    ).astype(np.float32)
    assert masks.shape == (2, 2, 3)
    assert float(masks[0, 0, 0]) > 0.2
    assert float(masks[0, 0, 2]) > 0.05
    assert float(masks[1, 1, 0]) < float(masks[0, 0, 0])


class RawClipMasksTest(unittest.TestCase):
    test_build_clip_masks_uses_cfa_max_binning = staticmethod(test_build_clip_masks_uses_cfa_max_binning)

    def test_direct_bayer_planes_match_generic_mask_path(self) -> None:
        raw = np.arange(64, dtype=np.uint16).reshape(8, 8) * 16
        colors = np.tile(
            np.asarray([[0, 1], [3, 2]], dtype=np.uint8),
            (4, 4),
        )
        kwargs = dict(
            color_desc="RGBG",
            white_level=1000,
            black_levels=[0.0, 0.0, 0.0, 0.0],
            camera_white_levels=[1000.0, 1000.0, 1000.0, 1000.0],
            orientation_flip=0,
            scene_shape=(4, 4),
        )
        generic = build_clip_masks(raw, colors, **kwargs)
        direct = build_clip_masks(raw, colors, raw_pattern=[[0, 1], [3, 2]], **kwargs)
        np.testing.assert_array_equal(direct, generic)

    def test_observed_fullwell_refreshes_the_soft_permission_mask(self) -> None:
        raw = np.full((4, 4), 100, dtype=np.uint16)
        colors = np.tile(np.asarray([[0, 1], [3, 2]], dtype=np.uint8), (2, 2))
        raw[0, 0] = 890
        initial = build_clip_masks(
            raw, colors, "RGBG", 1000, [0.0] * 4, [1000.0] * 4, 0, (2, 2),
            [[0, 1], [3, 2]],
        )
        bundle = SimpleNamespace(
            scene_decoder="libraw",
            clip_masks=initial,
            raw_image=raw,
            raw_colors=colors,
            color_desc="RGBG",
            white_level=1000,
            black_levels=[0.0] * 4,
            camera_white_levels=[1000.0] * 4,
            orientation_flip=0,
            scene_rec2020_render=np.zeros((2, 2, 3), dtype=np.uint16),
            raw_pattern=[[0, 1], [3, 2]],
            _clip_masks_cache_shape=(1, 1),
            _clip_masks_resized=np.ones((1, 1, 3), dtype=np.float16),
            raw_guidance=object(),
            _raw_guidance_cache_shape=(1, 1),
            _raw_guidance_resized=object(),
            _raw_guidance_has_sensor_snr=True,
        )
        changed = refresh_clip_masks_from_fullwell(
            bundle, {0: 900, 1: 1000, 2: 1000, 3: 1000}
        )
        self.assertTrue(changed)
        self.assertGreater(float(bundle.clip_masks[0, 0, 0]), float(initial[0, 0, 0]))
        self.assertIsNone(bundle._clip_masks_resized)
        self.assertIsNone(bundle.raw_guidance)
        self.assertFalse(
            refresh_clip_masks_from_fullwell(
                bundle, {0: 900, 1: 1000, 2: 1000, 3: 1000}
            )
        )


if __name__ == "__main__":
    unittest.main()
