# SPDX-License-Identifier: GPL-3.0-or-later
"""Product choices do not silently authorize fine or unsupported processing."""
import unittest
import numpy as np

from dngscan.chroma_policy import CONSERVATIVE_AMOUNT, resolve_chroma_policy
from dngscan.chroma_nr import LUMA_W, chroma_correction_map
from dngscan.chroma_research import joint_multiscale_correction, fixed_phase_psd_models


class ChromaPolicyTests(unittest.TestCase):
    def test_old_numeric_recipes_keep_custom_strength_including_zero(self):
        for amount in (0, .25, .5, 1):
            policy = resolve_chroma_policy(custom_amount=amount)
            self.assertEqual(policy.mode, "custom")
            self.assertEqual(policy.amount, amount)
            self.assertEqual(policy.fine_chroma, "experimental-unavailable")

    def test_conservative_is_explicit_conditional_coarse_permission(self):
        off = resolve_chroma_policy("off", custom_amount=.8)
        selected = resolve_chroma_policy("conservative", custom_amount=.8)
        self.assertEqual(off.amount, 0)
        self.assertEqual(off.coarse_chroma, "disabled")
        self.assertEqual(selected.amount, CONSERVATIVE_AMOUNT)
        self.assertEqual(selected.coarse_chroma, "conditional")
        self.assertNotEqual(off.cache_identity, selected.cache_identity)
        self.assertIn("real-camera-validation-pending", selected.to_dict()["evidence"])

    def test_invalid_custom_and_unimplemented_modes_are_rejected(self):
        for amount in (float("nan"), float("inf"), -.1, 1.1, "bad"):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                resolve_chroma_policy("custom", custom_amount=amount)
        for mode in ("fine", "joint", "auto", ""):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                resolve_chroma_policy(mode)


class JointBandCandidateTests(unittest.TestCase):
    def setUp(self):
        self.scene = np.random.default_rng(38).normal(.2, .01, (64, 64, 3)).astype(np.float32)
        self.covariance = np.asarray([[1., .2, -.1], [.2, 1.2, .1], [-.1, .1, .8]]) * .0001

    def test_zero_strength_missing_model_and_zero_model_are_exact_identity(self):
        for kwargs in ({}, {"noise_covariance": np.zeros((3, 3))}, {"detail_covariance": {0: np.zeros((3, 3))}}):
            np.testing.assert_array_equal(joint_multiscale_correction(self.scene, .5, 8, **kwargs), np.zeros_like(self.scene))
        np.testing.assert_array_equal(joint_multiscale_correction(self.scene, 0, 8, noise_covariance=self.covariance), np.zeros_like(self.scene))

    def test_constant_colour_and_zero_noise_colour_texture_are_preserved(self):
        constant = np.full((64, 64, 3), [.1, .2, .3], np.float32)
        correction = joint_multiscale_correction(constant, .5, 8, noise_covariance=self.covariance)
        np.testing.assert_array_equal(correction, np.zeros_like(constant))
        yy, xx = np.indices((64, 64))
        weave = constant + (np.sin(xx * 2 * np.pi / 8) + np.sin(yy * 2 * np.pi / 8))[..., None] * np.asarray([.1, -.04, .01])
        np.testing.assert_array_equal(joint_multiscale_correction(weave, .5, 8, noise_covariance=np.zeros((3, 3))), np.zeros_like(constant))

    def test_scene_Y_constraint_and_gain_squared_contract(self):
        correction = joint_multiscale_correction(self.scene, .1, 8, noise_covariance=self.covariance)
        # Float32 sums/projected corrections: two ulp at the scene magnitude,
        # not comparison of quantized display pixels.
        tolerance = 2 * np.finfo(np.float32).eps * np.max(np.abs(self.scene))
        self.assertLess(float(np.abs(correction @ LUMA_W).max()), tolerance)
        doubled = joint_multiscale_correction(self.scene * 2, .1, 8, noise_covariance=self.covariance * 4)
        np.testing.assert_allclose(doubled, 2 * correction, rtol=1e-6, atol=1e-8)

    def test_non_PSD_singular_asymmetric_and_ill_conditioned_models_reject(self):
        for covariance in (np.diag([1., -1., 1.]), np.diag([1., 0., 0.]),
                           np.asarray([[1., .5, 0.], [0., 1., 0.], [0., 0., 1.]]),
                           np.eye(3) * np.nan):
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                joint_multiscale_correction(self.scene, .1, 8, noise_covariance=covariance)
        with self.assertRaises(ValueError):
            joint_multiscale_correction(self.scene, .1, 8, noise_covariance=self.covariance, max_condition=1)

    def test_invalid_source_support_and_missing_bands_do_not_shrink(self):
        valid = np.ones((64, 64), bool)
        valid[32, 32] = False
        result = joint_multiscale_correction(self.scene, .5, 8, noise_covariance=self.covariance, valid_mask=valid)
        np.testing.assert_array_equal(result[30:35, 30:35], np.zeros((5, 5, 3)))
        missing = joint_multiscale_correction(self.scene, .5, 8, detail_covariance={12: self.covariance})
        np.testing.assert_array_equal(missing, np.zeros_like(missing))


class FixedPsdBridgeTests(unittest.TestCase):
    def test_complete_psd_reaches_both_operators_without_authorizing_decoder(self):
        from tests.test_complete_noise_spectrum import measured_fixture
        spectrum, _, _, _ = measured_fixture()
        phases = ("C00", "C01", "C10", "C11")
        matrix = np.asarray([[1., 0., 0., 0.], [0., .5, .5, 0.], [0., 0., 0., 1.]]) * .001
        model = fixed_phase_psd_models(spectrum, 200, matrix, phases=phases,
                                     assume_separable=True, assume_independent_phases=True)
        self.assertFalse(model["decoder_authorized"])
        scene = np.random.default_rng(39).normal(.2, .01, (64, 64, 3)).astype(np.float32)
        legacy = chroma_correction_map(scene, .1, 8, detail_variance=model["detail_variance"])
        joint = joint_multiscale_correction(scene, .1, 8, detail_covariance=model["detail_covariance"])
        self.assertGreater(float(np.abs(legacy).max()), 0)
        self.assertGreater(float(np.abs(joint).max()), 0)
        self.assertLess(float(np.abs(legacy @ LUMA_W).max()), 1e-8)
        self.assertLess(float(np.abs(joint @ LUMA_W).max()), 1e-8)
        doubled = fixed_phase_psd_models(spectrum, 200, matrix * 2, phases=phases,
                                       assume_separable=True, assume_independent_phases=True)
        for level in model["detail_covariance"]:
            np.testing.assert_allclose(doubled["detail_covariance"][level], 4 * model["detail_covariance"][level])

    def test_missing_axis_unmeasured_ISO_and_implicit_assumptions_reject(self):
        from tests.test_complete_noise_spectrum import measured_fixture
        from copy import deepcopy
        spectrum, _, _, _ = measured_fixture()
        kwargs = {"phases": ("C00", "C01", "C10", "C11"),
                  "assume_separable": True, "assume_independent_phases": True}
        matrix = np.asarray([[1, 0, 0, 0], [0, .5, .5, 0], [0, 0, 0, 1]])
        for change in ({"assume_separable": False}, {"assume_independent_phases": False}):
            with self.assertRaises(ValueError):
                fixed_phase_psd_models(spectrum, 200, matrix, **{**kwargs, **change})
        with self.assertRaises(ValueError):
            fixed_phase_psd_models(spectrum, 300, matrix, **kwargs)
        bad = deepcopy(spectrum)
        del bad["axes"]["v"]
        with self.assertRaises(ValueError):
            fixed_phase_psd_models(bad, 200, matrix, **kwargs)


if __name__ == "__main__":
    unittest.main()
