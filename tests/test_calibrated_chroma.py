# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent-noise inputs, texture counterexamples and propagation units."""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dngscan.chroma_nr import LUMA_W, _atrous_smooth, chroma_correction_map
from dngscan.noise_model import NoiseModel
from dngscan.noise_propagation import (
    _warp_points, area_variance, atrous_detail_variance,
    calibrated_chroma_variance, coarse_spatial_moments, transform_covariance,
)


def _texture(size=256, amplitude=.05, period=32):
    wave = amplitude * np.sin(2 * np.pi * np.arange(size) / period)
    scene = np.full((size, size, 3), .2, np.float32)
    direction = np.asarray([1, -LUMA_W[0] / LUMA_W[1], 0], np.float32)
    scene += wave[None, :, None] * direction
    return scene, wave


class CalibratedOperatorTests(unittest.TestCase):
    def test_no_model_is_identity_even_on_dense_texture(self):
        scene, _ = _texture()
        np.testing.assert_array_equal(chroma_correction_map(scene, 1, 8), np.zeros_like(scene))

    def test_known_zero_noise_preserves_dense_texture_at_every_strength(self):
        for amplitude in (.005, .05, .15):
            scene, _ = _texture(512, amplitude, 64)
            for amount in (.25, .5, 1):
                correction = chroma_correction_map(
                    scene, amount, 1, noise_covariance=np.zeros((3, 3)))
                np.testing.assert_array_equal(correction, np.zeros_like(scene))

    def test_calibrated_noise_improves_error_and_preserves_real_colour(self):
        clean, wave = _texture()
        sigma = .02
        noisy = clean + np.random.default_rng(42).normal(0, sigma, clean.shape).astype(np.float32)
        correction = chroma_correction_map(
            noisy, 1, 8, noise_covariance=np.eye(3) * sigma ** 2)
        result = noisy + correction
        self.assertLess(float(np.mean((result - clean) ** 2) / np.mean((noisy - clean) ** 2)), .65)
        amplitude = np.sum((result[..., 0] - .2) * wave[None, :]) / np.sum(np.broadcast_to(wave, clean.shape[:2]) ** 2)
        self.assertGreater(float(amplitude), .9)
        self.assertLess(float(np.max(np.abs(correction @ LUMA_W))), 1e-7)

    def test_noise_scale_is_independent_of_texture_amplitude(self):
        # Increasing genuine texture should protect it more, rather than
        # increase a threshold estimated from that same texture.
        fractions = []
        for amplitude in (.01, .1):
            scene, _ = _texture(amplitude=amplitude)
            correction = chroma_correction_map(
                scene, 1, 8, noise_covariance=np.eye(3) * 1e-6)
            fractions.append(float(np.linalg.norm(correction) / np.linalg.norm(scene - .2)))
        self.assertLess(fractions[1], fractions[0] / 2)

    def test_untrusted_pixels_do_not_authorize_neighbouring_shrinkage(self):
        scene = np.random.default_rng(8).normal(.2, .01, (64, 64, 3)).astype(np.float32)
        valid = np.ones((64, 64), bool)
        valid[32, 32] = False
        correction = chroma_correction_map(
            scene, 1, 8, chroma_variance=np.full(3, .0001), valid_mask=valid)
        np.testing.assert_array_equal(correction[30:35, 30:35], np.zeros((5, 5, 3), np.float32))

    def test_invalid_noise_cannot_silently_trigger_mad_fallback(self):
        scene, _ = _texture(64)
        for covariance in (np.eye(3) * -1, np.eye(3) * np.nan):
            with self.assertRaises(ValueError):
                chroma_correction_map(scene, 1, 8, noise_covariance=covariance)

    def test_native_and_numpy_share_the_calibrated_operator(self):
        from dngscan import _fast

        if not _fast.available():
            self.skipTest("native extension unavailable")
        scene, _ = _texture(128)
        scene += np.random.default_rng(10).normal(0, .01, scene.shape).astype(np.float32)
        with patch.dict(os.environ, {"DNGSCAN_FAST": "off"}):
            reference = chroma_correction_map(scene, 1, 8, noise_covariance=np.eye(3) * .0001)
        with patch.dict(os.environ, {"DNGSCAN_FAST": "1"}):
            native = chroma_correction_map(scene, 1, 8, noise_covariance=np.eye(3) * .0001)
        np.testing.assert_array_equal(native, reference)


class VariancePropagationTests(unittest.TestCase):
    def test_channel_covariance_follows_matrix_and_squared_gain(self):
        matrix = np.asarray([[2., 1., 0.], [0., .5, 1.], [1., 0., 1.]])
        covariance = np.diag([1., 4., 9.])
        actual = transform_covariance(covariance, matrix)
        np.testing.assert_allclose(actual, matrix @ covariance @ matrix.T)
        np.testing.assert_allclose(transform_covariance(covariance, 3 * matrix), 9 * actual)

    def test_fractional_area_uses_squared_weights(self):
        # Three independent samples averaged into two cells: weights 2/3,
        # 1/3 in each cell, hence variance (4+1)/9 rather than 1/1.5.
        actual = area_variance(np.ones((1, 3)), 1, 2)
        np.testing.assert_allclose(actual, [[5 / 9, 5 / 9]])
        np.testing.assert_allclose(area_variance(np.ones((4, 4)), 2, 2), .25)

    def test_atrous_predicted_variance_matches_independent_noise(self):
        rng = np.random.default_rng(19)
        noise = rng.standard_normal((512, 512)).astype(np.float32)
        smooth = noise
        for level in range(3):
            coarse = _atrous_smooth(smooth, level)
            predicted = atrous_detail_variance(np.ones_like(noise), level)
            use = predicted > 0
            ratio = float(np.mean(np.square(smooth - coarse)[use]) / np.mean(predicted[use]))
            self.assertGreater(ratio, .94)
            self.assertLess(ratio, 1.06)
            smooth = coarse

    def test_spatially_varying_variance_propagation(self):
        # An impulse's variance is multiplied by the squared filter
        # coefficient, independently of every neighbouring sample.
        variance = np.zeros((64, 64), np.float32)
        variance[32, 32] = 2
        actual = atrous_detail_variance(variance, 0)
        coefficient = 1 - (6 / 16) ** 2
        self.assertAlmostEqual(float(actual[32, 32]), 2 * coefficient ** 2, places=6)

    def _bundle(self):
        return SimpleNamespace(
            raw_image=np.zeros((512, 512), np.uint16),
            raw_pattern=[[0, 1], [3, 2]], color_desc="RGBG",
            scene_decoder="libraw", wb_mode="camera", exposure_gain=2,
            lens_filter="none", lens_shading=None, scene_geometry_ops=(),
            noise_decode={"supported": True, "normalized_raw_to_scene": np.eye(3)},
        )

    def test_cfa_count_and_exposure_scale_are_used(self):
        bundle = self._bundle()
        model = NoiseModel(status="valid", channel_variance={c: (0., .004) for c in "RGB"})
        variance, reason = calibrated_chroma_variance(bundle, model, np.full((64, 64, 3), .2))
        # 64 sensels/cell: R/B have 16 independent samples, green 32.
        projection = np.eye(3) - np.ones((3, 1)) * LUMA_W[None, :]
        expected = np.diag(projection @ np.diag([.004 / 16, .004 / 32, .004 / 16]) @ projection.T) * 4
        np.testing.assert_allclose(variance, np.broadcast_to(expected, variance.shape), rtol=2e-7)
        self.assertIn("approximate", reason)

    def test_compact_proxy_uses_recorded_sensor_window_and_hot_wb(self):
        bundle = self._bundle()
        model = NoiseModel(status="valid", channel_variance={c: (0., .004) for c in "RGB"})
        scene = np.full((64, 64, 3), .2)
        expected, _ = calibrated_chroma_variance(bundle, model, scene)
        bundle.raw_image = None
        bundle.wb_mode = "5500"
        bundle.noise_decode.update(sensor_window_shape=(512, 512), wb_mode="5500")
        actual, _ = calibrated_chroma_variance(bundle, model, scene)
        np.testing.assert_array_equal(actual, expected)

    def test_unknown_spatial_operations_fail_closed(self):
        bundle = self._bundle()
        bundle.lens_shading = "gainmap"
        model = NoiseModel(status="valid", channel_variance={c: (0., .004) for c in "RGB"})
        variance, reason = calibrated_chroma_variance(bundle, model, np.full((64, 64, 3), .2))
        self.assertIsNone(variance)
        self.assertIn("shading", reason)

    def _spatial_descriptor(self):
        return {"full_sensor_shape": [512, 512], "sensor_crop": [0, 0, 512, 512],
                "orientation_flip": 0}

    def _gain_map(self, gains):
        return {"top": 0, "left": 0, "bottom": 512, "right": 512,
                "plane": 0, "planes": 1, "row_pitch": 1, "col_pitch": 1,
                "spacing_v": 1., "spacing_h": 1., "origin_v": 0., "origin_h": 0.,
                "gains": np.asarray(gains)[..., None].tolist()}

    def test_combined_gain_maps_are_squared_after_multiplication(self):
        descriptor = self._spatial_descriptor()
        descriptor["gain_maps"] = [self._gain_map(np.full((2, 2), value)) for value in (2, 3)]
        mean, second, area, valid = coarse_spatial_moments(descriptor, (32, 32), list("RGGB"))
        np.testing.assert_allclose(mean, 6)
        np.testing.assert_allclose(second, 36)
        np.testing.assert_allclose(area, 1)
        self.assertTrue(valid.all())

    def test_linear_gain_quadrature_matches_analytic_second_moment(self):
        descriptor = self._spatial_descriptor()
        descriptor["gain_maps"] = [self._gain_map([[1, 2], [1, 2]])]
        mean, second, _, _ = coarse_spatial_moments(descriptor, (8, 8), list("RGGB"))
        left = 1 + np.arange(8) / 8
        right = left + 1 / 8
        expected = (left ** 2 + left * right + right ** 2) / 3
        np.testing.assert_allclose(mean[..., 0], np.broadcast_to((left + right) / 2, (8, 8)))
        np.testing.assert_allclose(second[..., 0], np.broadcast_to(expected, (8, 8)))

    def test_half_size_decoded_crop_maps_back_to_native_gain_positions(self):
        full = self._spatial_descriptor()
        full.update(sensor_crop=[20, 20, 400, 400], decoded_crop=[20, 20, 400, 400],
                    pre_crop_shape=[512, 512], gain_maps=[self._gain_map([[1, 2], [1, 2]])])
        half = dict(full, decoded_crop=[10, 10, 200, 200], pre_crop_shape=[256, 256])
        expected = coarse_spatial_moments(full, (8, 8), list("RGGB"))
        actual = coarse_spatial_moments(half, (8, 8), list("RGGB"))
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want)

    def test_rotated_gain_map_keeps_the_same_sensor_window(self):
        from dngscan.raw_io import _orient_like_libraw

        descriptor = self._spatial_descriptor()
        descriptor["gain_maps"] = [self._gain_map([[1, 2], [1, 2]])]
        expected = coarse_spatial_moments(descriptor, (8, 16), list("RGGB"))
        descriptor["orientation_flip"] = 6
        actual = coarse_spatial_moments(descriptor, (16, 8), list("RGGB"))
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, _orient_like_libraw(want, 6))

    def test_phase_gain_maps_only_change_the_declared_cfa_phase(self):
        descriptor = self._spatial_descriptor()
        gain = self._gain_map(np.full((2, 2), 2.))
        gain.update(row_pitch=2, col_pitch=2)
        descriptor["gain_maps"] = [gain]
        mean, second, _, _ = coarse_spatial_moments(descriptor, (8, 8), list("RGGB"))
        np.testing.assert_allclose(mean[..., 0], 2)
        np.testing.assert_allclose(second[..., 0], 4)
        np.testing.assert_allclose(mean[..., 1:], 1)

    def test_vector_warp_matches_the_renderer_coordinate_contract(self):
        from dngscan.dng_opcodes import Warp, _coordinates

        warp = Warp(coefficients=((1.01, -.03, .001, .004, .0001, -.0002),), cx=.48, cy=.51)
        expected_y, expected_x = _coordinates(warp, 80, 120, 0, 80, 0)
        y, x = np.mgrid[:80, :120]
        actual_y, actual_x = _warp_points(warp, y, x, 80, 120, 0)
        np.testing.assert_array_equal(actual_y, expected_y)
        np.testing.assert_array_equal(actual_x, expected_x)

    def test_warp_source_area_and_no_antialias_expansion_limit(self):
        descriptor = self._spatial_descriptor()
        for scale in (.8, 1.2):
            descriptor["warp_ops"] = [{"coefficients": [[scale, 0, 0, 0, 0, 0]], "cx": .5, "cy": .5}]
            _, _, area, valid = coarse_spatial_moments(descriptor, (32, 32), list("RGGB"))
            np.testing.assert_allclose(area[valid], min(scale ** 2, 1), atol=1e-12)
            self.assertGreater(int(valid.sum()), 100)

    def test_folded_or_extrapolated_warp_does_not_authorize_denoising(self):
        descriptor = self._spatial_descriptor()
        descriptor["warp_ops"] = [{"coefficients": [[0, 0, 0, 0, 0, 0]], "cx": .5, "cy": .5}]
        _, _, _, valid = coarse_spatial_moments(descriptor, (16, 16), list("RGGB"))
        self.assertFalse(valid.any())


if __name__ == "__main__":
    unittest.main()
