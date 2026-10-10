# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed-kernel spectral variance checks, independent of image structure."""
import unittest

import numpy as np

from dngscan.noise_propagation import atrous_detail_variance
from dngscan.spectral_variance import (
    one_sided_power_mass, sensor_frequencies, separable_atrous_detail_variance,
    measured_phase_detail_variance, project_independent_phase_bands,
    separable_linear_probe_variance,
)


def periodogram(values, *, axis=-1):
    n = values.shape[axis]
    shifted = values - values.mean(axis=axis, keepdims=True)
    return np.abs(np.fft.rfft(shifted, axis=axis)) ** 2 / n ** 2


def periodic_smooth(image, level):
    step = 2 ** level
    weights = np.asarray([1., 4., 6., 4., 1.]) / 16.
    out = sum(weight * np.roll(image, (j - 2) * step, axis=1) for j, weight in enumerate(weights))
    return sum(weight * np.roll(out, (j - 2) * step, axis=0) for j, weight in enumerate(weights))


def ar_spectrum(n, rho):
    frequency = np.fft.fftfreq(n)
    return (1 - rho ** 2) / (1 + rho ** 2 - 2 * rho * np.cos(2 * np.pi * frequency))


def shaped_plane(n, horizontal, vertical, rng):
    white = rng.normal(size=(n, n))
    return np.fft.ifft2(np.fft.fft2(white) * np.sqrt(vertical[:, None] * horizontal[None, :])).real


def b3_detail_kernel(level):
    previous = np.asarray([1.])
    for current in range(level + 1):
        step = 2 ** current
        smooth = np.zeros(4 * step + 1)
        smooth[::step] = np.asarray([1., 4., 6., 4., 1.]) / 16.
        following = np.convolve(previous, smooth)
        if current == level:
            pad = (following.size - previous.size) // 2
            previous = np.pad(previous, (pad, pad))
            return np.outer(previous, previous) - np.outer(following, following)
        previous = following


class SpectralVariancePrototypeTests(unittest.TestCase):
    def test_rectangular_parseval_handles_even_odd_dc_and_nyquist(self):
        rng = np.random.default_rng(175)
        for n in (127, 128):
            values = rng.normal(0, 3, n)
            mass = one_sided_power_mass(periodogram(values), n)
            self.assertAlmostEqual(float(mass.sum()), float(values.var()), places=12)
        self.assertEqual(float(one_sided_power_mass([4., 0., 9.], 4).sum()), 13.)

    def test_difference_spectrum_restores_one_exposure_variance(self):
        rng = np.random.default_rng(271)
        difference = rng.normal(size=256) - rng.normal(size=256)
        mass = one_sided_power_mass(periodogram(difference), 256, difference=True)
        self.assertAlmostEqual(float(mass.sum()), float(difference.var()) / 2., places=13)

    def test_bayer_frequency_conversion_matches_sampled_sensor_sinusoid(self):
        sensor = np.sin(2 * np.pi * np.arange(256) / 16.)
        channel = sensor[::2]
        peak = np.argmax(np.abs(np.fft.rfft(channel)))
        f_plane = peak / channel.size
        self.assertEqual(f_plane, .125)
        self.assertEqual(float(sensor_frequencies(f_plane)), 1. / 16.)

    def test_white_spectrum_matches_existing_independent_atrous_variance(self):
        n = 256
        p = np.full(n // 2 + 1, 1. / n)
        predicted = separable_atrous_detail_variance(
            p, p, length_h=n, length_v=n, difference=False, assume_separable=True)
        for level, actual in predicted.items():
            reference = atrous_detail_variance(np.ones((129, 129)), level)[64, 64]
            self.assertEqual(np.float32(actual), reference)

    def test_known_correlated_noise_matches_fixed_kernel_monte_carlo(self):
        n = 512
        freq = np.fft.fftfreq(n)
        spectra = [(1 - rho ** 2) / (1 + rho ** 2 - 2 * rho * np.cos(2 * np.pi * freq))
                   for rho in (.75, .35)]
        noise = np.random.default_rng(415).normal(size=(n, n))
        shaped = np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(spectra[1][:, None] * spectra[0][None, :])).real
        predicted = separable_atrous_detail_variance(
            spectra[0][:n // 2 + 1] / n, spectra[1][:n // 2 + 1] / n,
            length_h=n, length_v=n, difference=False, assume_separable=True)
        for level in range(3):
            smooth = periodic_smooth(shaped, level)
            measured = float(np.mean((shaped - smooth) ** 2))
            self.assertAlmostEqual(measured / predicted[level], 1., delta=.08)
            shaped = smooth

    def test_white_and_separately_row_column_correlated_noise_match_observed_bands(self):
        n = 512
        for horizontal, vertical in ((0., 0.), (.85, 0.), (0., .85)):
            sx, sy = ar_spectrum(n, horizontal), ar_spectrum(n, vertical)
            shaped = shaped_plane(n, sx, sy, np.random.default_rng(1907))
            predicted = separable_atrous_detail_variance(
                sx[:257] / n, sy[:257] / n, length_h=n, length_v=n,
                difference=False, assume_separable=True)
            for level in range(3):
                smooth = periodic_smooth(shaped, level)
                observed = float(np.mean((shaped - smooth) ** 2))
                with self.subTest(horizontal=horizontal, vertical=vertical, level=level):
                    self.assertAlmostEqual(observed / predicted[level], 1., delta=.035)
                shaped = smooth

    def test_narrow_band_prediction_matches_analytic_response(self):
        n = 128
        ph, pv = np.zeros(65), np.zeros(65)
        ph[5], pv[9] = .5, .5
        actual = separable_atrous_detail_variance(
            ph, pv, length_h=n, length_v=n, difference=False, assume_separable=True)
        previous = 1.
        for level in range(3):
            current = previous * np.cos(np.pi * (2 ** level) * 5 / n) ** 4 * np.cos(np.pi * (2 ** level) * 9 / n) ** 4
            self.assertAlmostEqual(actual[level], (previous - current) ** 2, places=13)
            previous = current

    def test_narrow_peak_and_low_frequency_structure_match_observed_fixed_bands(self):
        n = 256
        yy, xx = np.indices((n, n))
        for kh, kv in ((17, 29), (1, 2)):
            image = 2 * np.cos(2 * np.pi * kh * xx / n + .3) * np.cos(2 * np.pi * kv * yy / n + .7)
            ph, pv = np.zeros(129), np.zeros(129)
            ph[kh], pv[kv] = .5, .5  # Each complete marginal integrates to 1.
            predicted = separable_atrous_detail_variance(
                ph, pv, length_h=n, length_v=n, levels=(0, 1, 2, 3, 4),
                difference=False, assume_separable=True)
            for level in range(5):
                smooth = periodic_smooth(image, level)
                observed = float(np.mean((image - smooth) ** 2))
                with self.subTest(peaks=(kh, kv), level=level):
                    self.assertAlmostEqual(observed, predicted[level], delta=max(1e-14, predicted[level] * 1e-10))
                image = smooth

    def test_pure_and_crossed_low_frequency_banding_cannot_recover_removed_line_power(self):
        from tests.test_complete_noise_spectrum import line_spectrum
        n = 128
        yy, xx = np.indices((n, n))
        stripe = np.sin(2 * np.pi * yy / 32.)
        crossed = stripe + np.cos(2 * np.pi * xx / 32.)
        for image in (stripe, crossed):
            ph, _ = line_spectrum(image, 1)
            pv, _ = line_spectrum(image, 0)
            with self.subTest(crossed=image is crossed), self.assertRaisesRegex(ValueError, "missing covariance power"):
                separable_atrous_detail_variance(
                    ph, pv, length_h=n, length_v=n, difference=False,
                    assume_separable=True, variance=float(image.var()))

    def test_uniform_gain_and_linear_scaling_square_the_observed_band_variance(self):
        n = 512
        sx, sy = ar_spectrum(n, .7), ar_spectrum(n, .3)
        noise = shaped_plane(n, sx, sy, np.random.default_rng(411))
        original = separable_atrous_detail_variance(
            sx[:257] / n, sy[:257] / n, length_h=n, length_v=n,
            difference=False, assume_separable=True)
        for gain, linear_scale in ((1., 1.), (2.3, .25), (.65, 4.)):
            scaled = noise * gain * linear_scale
            for level in range(3):
                predicted = separable_linear_probe_variance(
                    sx[:257] / n, sy[:257] / n, b3_detail_kernel(level),
                    length_h=n, length_v=n, difference=False, assume_separable=True,
                    gain_patch=gain * linear_scale)
                smooth = periodic_smooth(scaled, level)
                observed = float(np.mean((scaled - smooth) ** 2))
                with self.subTest(gain=gain, scale=linear_scale, level=level):
                    self.assertAlmostEqual(predicted, original[level] * (gain * linear_scale) ** 2, places=13)
                    self.assertAlmostEqual(observed / predicted, 1., delta=.035)
                scaled = smooth

    def test_spatial_gainmap_and_fixed_resampling_match_local_covariance_monte_carlo(self):
        n, draws = 512, 80_000
        sx, sy = ar_spectrum(n, .7), ar_spectrum(n, .35)
        weights = {
            "bilinear": np.asarray([[.12, .28], [.18, .42]]),
            "area-average": np.full((3, 4), 1. / 12.),
            "detail": b3_detail_kernel(0),
        }
        rng = np.random.default_rng(776)
        for name, kernel in weights.items():
            height, width = kernel.shape
            cy = .35 ** np.abs(np.arange(height)[:, None] - np.arange(height)[None, :])
            cx = .7 ** np.abs(np.arange(width)[:, None] - np.arange(width)[None, :])
            white = rng.normal(size=(draws, height, width))
            source = np.einsum("ij,njk,lk->nil", np.linalg.cholesky(cy), white, np.linalg.cholesky(cx))
            yy, xx = np.indices(kernel.shape)
            gain = .8 + .35 * yy + .2 * xx  # Known nonstationary pre-kernel map.
            for label, patch in (("before-gain", np.ones_like(gain)), ("after-gain", gain)):
                predicted = separable_linear_probe_variance(
                    sx[:257] / n, sy[:257] / n, kernel, length_h=n, length_v=n,
                    difference=False, assume_separable=True, gain_patch=patch)
                observed = float(np.var(np.einsum("nij,ij->n", source, kernel * patch)))
                with self.subTest(kernel=name, stage=label):
                    self.assertAlmostEqual(observed / predicted, 1., delta=.012)
            if name == "detail":
                after = separable_linear_probe_variance(
                    sx[:257] / n, sy[:257] / n, kernel, length_h=n, length_v=n,
                    difference=False, assume_separable=True, gain_patch=gain)
                naive = separable_linear_probe_variance(
                    sx[:257] / n, sy[:257] / n, kernel, length_h=n, length_v=n,
                    difference=False, assume_separable=True) * gain[2, 2] ** 2
                self.assertGreater(abs(after / naive - 1.), .01)

    def test_fixed_bilinear_cfa_with_unequal_phase_gains_matches_predicted_rgb_variance(self):
        from dngscan.sensor_research import signed_bilinear_bayer
        n = 256
        sx, sy = ar_spectrum(n, .7), ar_spectrum(n, .35)
        variance = np.asarray([1., 4., 9., 16.])
        gains = np.asarray([2., 1.5, .75, 1.25])
        mosaic = np.empty((2 * n, 2 * n))
        rng = np.random.default_rng(788)
        for i in range(4):
            plane = shaped_plane(n, sx, sy, rng) * np.sqrt(variance[i])
            mosaic[i // 2::2, i % 2::2] = plane
        rendered = signed_bilinear_bayer(mosaic, "RGGB", phase_gains=gains)
        # Fixed bilinear weights at a red lattice site, in each phase grid.
        kernels = [np.ones((1, 1)), np.full((1, 2), .25),
                   np.full((2, 1), .25), np.full((2, 2), .25)]
        phase_predicted = [separable_linear_probe_variance(
            sx[:129] * v / n, sy[:129] * v / n, kernel, length_h=n, length_v=n,
            difference=False, assume_separable=True, gain_patch=g)
            for kernel, v, g in zip(kernels, variance, gains)]
        expected = np.asarray([phase_predicted[0], sum(phase_predicted[1:3]), phase_predicted[3]])
        observed = np.var(rendered[8:-8:2, 8:-8:2], axis=(0, 1))
        np.testing.assert_allclose(observed, expected, rtol=.04)

    def test_linear_probe_rejects_unqualified_or_unbounded_operators(self):
        p = np.full(65, 1. / 128)
        with self.assertRaisesRegex(ValueError, "explicit"):
            separable_linear_probe_variance(p, p, [[1.]], length_h=128, length_v=128)
        for kernel, gain in ((np.ones((66, 1)), None), (np.ones((2, 2)), np.ones((3, 3))),
                             (np.ones((2, 2)), 0.), (np.asarray([[np.nan]]), None)):
            with self.subTest(kernel_shape=kernel.shape, gain=gain), self.assertRaises(ValueError):
                separable_linear_probe_variance(p, p, kernel, length_h=128, length_v=128,
                    assume_separable=True, gain_patch=gain)

    def test_missing_covariance_and_unreconciled_variance_fail_closed(self):
        p = np.full(65, 1. / 128)
        with self.assertRaisesRegex(ValueError, "explicit"):
            separable_atrous_detail_variance(p, p, length_h=128, length_v=128)
        with self.assertRaisesRegex(ValueError, "disagree"):
            separable_atrous_detail_variance(p, p * 2, length_h=128, length_v=128,
                                             assume_separable=True)
        with self.assertRaisesRegex(ValueError, "missing"):
            separable_atrous_detail_variance(p, np.zeros(65), length_h=128, length_v=128,
                                             assume_separable=True)

    def test_zero_noise_and_invalid_input(self):
        actual = separable_atrous_detail_variance(
            np.zeros(65), np.zeros(65), length_h=128, length_v=128, assume_separable=True)
        self.assertEqual(actual, {0: 0., 1: 0., 2: 0.})
        for values in ([1., -1., 0.], [0., np.nan, 0.]):
            with self.assertRaises(ValueError):
                one_sided_power_mass(values, 4)

    def test_retained_measured_psd_can_supply_fixed_linear_chroma_detail_bands(self):
        from dngscan.chroma_nr import chroma_correction_map, LUMA_W
        from tests.test_complete_noise_spectrum import measured_fixture

        spectrum, _, _, _ = measured_fixture()
        phase_bands = {phase: measured_phase_detail_variance(
            spectrum, phase, 200, assume_separable=True) for phase in ("C00", "C01", "C10", "C11")}
        # Deliberately explicit toy decoder: co-sited R,(G1+G2)/2,B,
        # then projection onto Rec.2020 zero-luma chroma and fixed DN scale.
        camera = np.asarray([[1., 0., 0., 0.], [0., .5, .5, 0.], [0., 0., 0., 1.]]) / 1000.
        chroma_projection = np.eye(3) - np.ones((3, 1)) * LUMA_W[None, :]
        projection = chroma_projection @ camera
        supplied = project_independent_phase_bands(phase_bands, projection, assume_independent=True)
        self.assertEqual(set(supplied), {0, 1, 2})
        self.assertTrue(all(v.shape == (3,) and np.all(v > 0) for v in supplied.values()))

        rng = np.random.default_rng(780)
        phase_noise = rng.normal(0., 3., (128, 128, 4))
        scene = np.float32(.25) + (phase_noise @ projection.T).astype(np.float32)
        valid = np.ones((128, 128), dtype=np.float32)
        valid[[0, -1], :] = 0
        valid[:, [0, -1]] = 0
        correction = chroma_correction_map(scene, .5, decimation_factor=8.,
                                           detail_variance=supplied, valid_mask=valid)
        self.assertGreater(float(np.max(np.abs(correction))), 0.)
        np.testing.assert_allclose(correction @ LUMA_W, 0., atol=2e-10)
        np.testing.assert_array_equal(correction[0], 0.)
        np.testing.assert_array_equal(correction[:, 0], 0.)
        # The same measurement is independent of the image; a constant image
        # does not invent a different noise scale or a nonzero correction.
        flat = np.full_like(scene, .25)
        np.testing.assert_array_equal(chroma_correction_map(
            flat, .5, decimation_factor=8., detail_variance=supplied, valid_mask=valid), 0.)

    def test_fixed_linear_projection_matches_monte_carlo_and_requires_independence(self):
        variances = np.array([1., 4., 9., 16.])
        phase_bands = {phase: {0: variance} for phase, variance in
                       zip(("C00", "C01", "C10", "C11"), variances)}
        matrix = np.asarray([[1., -.5, .25, 0.], [0., .5, .5, 0.], [-.1, 0., 0., 1.]])
        with self.assertRaisesRegex(ValueError, "independence"):
            project_independent_phase_bands(phase_bands, matrix)
        expected = project_independent_phase_bands(phase_bands, matrix, assume_independent=True)[0]
        noise = np.random.default_rng(806).normal(size=(200_000, 4)) * np.sqrt(variances)
        actual = np.var(noise @ matrix.T, axis=0)
        np.testing.assert_allclose(actual, expected, rtol=.012)
        with self.assertRaisesRegex(ValueError, "same levels"):
            project_independent_phase_bands({"C00": {0: 1.}, "C01": {1: 2.}},
                                             np.ones((3, 2)), assume_independent=True)


if __name__ == '__main__':
    unittest.main()
