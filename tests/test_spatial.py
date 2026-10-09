# SPDX-License-Identifier: GPL-3.0-or-later
"""Generic spatial primitives stay independent of optional film research."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from dngscan import _fast
from dngscan.spatial import (
    SPATIAL_GRID_RESERVE_MIB, area_decimate, area_decimate_rows,
    area_resample, spatial_band_rows, spread_grid_shape,
    upsample_rows,
)


class SpatialContractTests(unittest.TestCase):
    def test_fractional_area_mean_conserves_energy(self) -> None:
        src = np.random.default_rng(7).normal(size=(43, 61, 3)).astype(np.float32)
        with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "0"}):
            out = area_decimate(src, 17, 23)
            integral_reference = area_resample(src, 17, 23)
        np.testing.assert_allclose(out, integral_reference, rtol=0, atol=2e-7)
        np.testing.assert_allclose(
            out.mean(axis=(0, 1), dtype=np.float64),
            src.mean(axis=(0, 1), dtype=np.float64), rtol=0, atol=2e-8,
        )

    def test_streamed_decimation_matches_whole_frame(self) -> None:
        src = np.random.default_rng(8).normal(size=(45, 63, 3)).astype(np.float32)
        for mode in ("0", "auto"):
            with self.subTest(mode=mode), mock.patch.dict(os.environ, {"DNGSCAN_FAST": mode}):
                expected = area_decimate(src, 19, 27)
                acc = np.zeros((19, 27, 3), dtype=np.float64)
                for y0 in range(0, 45, 7):
                    area_decimate_rows(src[y0:y0 + 7], y0, 45, 63, 19, 27, acc)
                np.testing.assert_array_equal(acc.astype(np.float32), expected)

    def test_upsample_row_seams_are_exact(self) -> None:
        src = np.random.default_rng(9).normal(size=(17, 23, 3)).astype(np.float32)
        expected = upsample_rows(src, 0, 45, 45, 63)
        bands = [upsample_rows(src, y0, min(y0 + 7, 45), 45, 63)
                 for y0 in range(0, 45, 7)]
        np.testing.assert_array_equal(np.concatenate(bands), expected)

    def test_memory_tier_preserves_grid_policy(self) -> None:
        for tier, limit in (("512", 1408), ("1024", 2048), ("invalid", 1408)):
            with self.subTest(tier=tier), mock.patch.dict(
                os.environ, {"DNGSCAN_SPATIAL_BUDGET_MIB": tier}
            ):
                self.assertEqual(spread_grid_shape(4000, 6000), (round(4000 * limit / 6000), limit))

    def test_wide_image_bands_fit_after_resident_grid_reserve(self) -> None:
        for tier in (512, 1024):
            with mock.patch.dict(os.environ, {"DNGSCAN_SPATIAL_BUDGET_MIB": str(tier)}):
                for width in (512, 8192, 12000, 100000):
                    rows = spatial_band_rows(width)
                    estimated_band_bytes = rows * 3 * width * 3 * 4 * 8
                    available_bytes = (tier - SPATIAL_GRID_RESERVE_MIB) * (1 << 20)
                    self.assertLessEqual(estimated_band_bytes, available_bytes)


@unittest.skipUnless(_fast.available(), "native extension not built")
class NativeSpatialParityTests(unittest.TestCase):
    def test_area_decimation_matches_numpy_for_borrowed_source_types(self) -> None:
        source = np.random.default_rng(10).normal(size=(61, 47, 3))
        for dtype in (np.float32, np.float64):
            # Borrow a strided view, as the production decoder can do.
            src = source.astype(dtype)[::2, ::2]
            for acc_dtype in (np.float32, np.float64):
                outputs = []
                for mode in ("0", "1"):
                    with mock.patch.dict(os.environ, {"DNGSCAN_FAST": mode}):
                        acc = np.zeros((13, 11, 3), dtype=acc_dtype)
                        area_decimate_rows(src, 0, *src.shape[:2], 13, 11, acc)
                        outputs.append(acc)
                np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_upsample_matches_numpy(self) -> None:
        src = np.random.default_rng(11).normal(size=(17, 23, 3)).astype(np.float32)
        outputs = []
        for mode in ("0", "1"):
            with mock.patch.dict(os.environ, {"DNGSCAN_FAST": mode}):
                outputs.append(upsample_rows(src, 7, 38, 45, 63))
        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_native_extension_has_only_core_rendering_entrypoints(self) -> None:
        ext = _fast._require_extension()
        for name in ("area_decimate_rows", "upsample_rows", "apply_rgb_matrix3",
                     "apply_agx_core_f32", "apply_hdr_formation_f32"):
            self.assertTrue(hasattr(ext, name), name)
        for name in ("film_appearance_apply_f32", "layer_log_exposure", "density_grain_v1",
                     "halation_reinject_rows", "capture_bloom_apply_rows"):
            self.assertFalse(hasattr(ext, name), name)


if __name__ == "__main__":
    unittest.main()
