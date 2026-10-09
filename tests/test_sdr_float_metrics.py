# SPDX-License-Identifier: GPL-3.0-or-later
"""Fractional SDR code errors survive the high-bit-depth delivery gate."""
import os
import unittest
from unittest import mock

import numpy as np

from dngscan import gainmap
from dngscan.auto_encode import coding_metrics, coding_metrics_float
from dngscan.local_detail import local_detail_loss, local_detail_loss_sdr_float


class SdrFloatMetricsTests(unittest.TestCase):
    def test_fractional_code_error_is_measured_without_integer_dispatch(self):
        intended = np.full((16, 24, 3), .5, np.float32)
        decoded = intended + np.float32(.25 / 255.)
        np.testing.assert_array_equal(np.rint(decoded * 255).astype(np.uint8),
                                      np.rint(intended * 255).astype(np.uint8))
        with mock.patch.dict(os.environ, DNGSCAN_FAST="1", DNGSCAN_FAST_STRICT="1"), \
             mock.patch("dngscan._fast.kernel", side_effect=AssertionError("float used uint8 native path")):
            metrics = gainmap._base_and_coding_metrics_float(decoded, intended)
        for key in ("base_mean_code_error", "base_p99_code_error", "base_max_code_error",
                    "base_channel_bias_code_error", "base_block_p99_code_error",
                    "coding_luma_rmse", "coding_local_luma_p99"):
            self.assertAlmostEqual(metrics[key], .25, delta=1e-5, msg=key)
        self.assertEqual(metrics["base_local_detail_loss"], 0.)
        self.assertEqual(metrics["coding_local_detail_loss"], 0.)

    def test_identical_inputs_have_exact_zero_errors_and_preserve_strided_owners(self):
        rng = np.random.default_rng(48)
        owner = rng.random((19, 27, 4), dtype=np.float32)
        rgb = owner[::-1, ::-1, :3]
        rgb.flags.writeable = False
        before = owner.tobytes()
        metrics = gainmap._base_and_coding_metrics_float(rgb, rgb)
        self.assertTrue(all(value == 0. for value in metrics.values()), metrics)
        self.assertEqual(before, owner.tobytes())

    def test_code_units_and_eight_code_detail_floor_match_u8_reference(self):
        rng = np.random.default_rng(49)
        a = rng.integers(0, 256, (137, 29, 3), np.uint8)
        e = rng.integers(0, 256, a.shape, np.uint8)
        af, ef = a.astype(np.float32) / 255., e.astype(np.float32) / 255.
        expected = coding_metrics(a, e)
        actual = coding_metrics_float(af, ef)
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key], delta=2e-5, msg=key)
        self.assertAlmostEqual(local_detail_loss_sdr_float(af, ef),
                               local_detail_loss(a, e), delta=2e-7)

    def test_localized_texture_loss_is_rejected_in_float_sdr(self):
        intended = np.full((256, 256, 3), .25, np.float32)
        detail = np.where(np.indices((32, 32)).sum(axis=0) % 2, .375, .125).astype(np.float32)
        intended[96:128, 96:128] = detail[..., None]
        decoded = np.full_like(intended, .25)
        metrics = gainmap._base_and_coding_metrics_float(decoded, intended)
        self.assertEqual(metrics["base_local_detail_loss"], 1.)
        self.assertEqual(metrics["coding_local_detail_loss"], 1.)
        self.assertFalse(gainmap._base_roundtrip_is_acceptable(metrics))

    def test_finite_codec_overshoot_is_measured_without_clipping(self):
        intended = np.ones((8, 8, 3), np.float32)
        decoded = np.full_like(intended, 1.001)
        metrics = gainmap._base_and_coding_metrics_float(decoded, intended)
        self.assertAlmostEqual(metrics["base_mean_code_error"], .255, delta=2e-5)
        self.assertGreater(metrics["coding_luma_rmse"], .25)
        decoded.fill(-.001)
        intended.fill(0.)
        self.assertGreater(gainmap._base_and_coding_metrics_float(decoded, intended)["base_mean_code_error"], .25)

    def test_invalid_domain_format_and_arithmetic_cannot_pass(self):
        good = np.full((8, 8, 3), .5, np.float32)
        for value in (np.nan, np.inf, -.01, 1.01):
            bad = good.copy()
            bad[-1, -1, 2] = value
            with self.subTest(master=value), self.assertRaises(ValueError):
                gainmap._base_and_coding_metrics_float(good, bad)
            self.assertEqual(local_detail_loss_sdr_float(good, bad), float('inf'))
        for bad in (good.astype(np.uint8), good[:0], good[:, :, :2],
                    good.astype('>f4'), np.full_like(good, np.nan),
                    np.full_like(good, np.finfo(np.float32).max)):
            with self.subTest(dtype=bad.dtype, shape=bad.shape), self.assertRaises(ValueError):
                gainmap._base_and_coding_metrics_float(bad, good)
            self.assertEqual(local_detail_loss_sdr_float(bad, good), float('inf'))


if __name__ == '__main__':
    unittest.main()
