# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent signal experiments and explicit failure domains."""
from dataclasses import replace
from types import SimpleNamespace
from types import MappingProxyType
import unittest

import numpy as np

from dngscan.sensor_research import (
    LUMA, PHASES, PersonalBlackCalibration, signed_black_correct,
    signed_bilinear_bayer, dark_stack_products, joint_chroma_shrink,
    quantized_gain_chain,
)
from dngscan.spatial_black import SpatialBlack, apply_to_working


def personal():
    return PersonalBlackCalibration('TEST', 'Bayer', 'body-1', 'readout-1', 100,
        (.01, .02), dict(zip(PHASES, (512.25, 512.5, 513.75, 514.))),
        dict.fromkeys(PHASES, .1), 'sha256:test', temperature_c=(18., 22.),
        firmware='1.0', valid_until_utc='2027-01-01T00:00:00Z')


def capture():
    return {'make': 'TEST', 'model': 'Bayer', 'body_serial': 'body-1',
            'readout_identity': 'readout-1', 'iso': 100, 'exposure_seconds': .015,
            'temperature_c': 20., 'firmware': '1.0', 'evaluation_utc': '2026-10-09T12:00:00Z'}


class PersonalBlackTests(unittest.TestCase):
    def test_absolute_replaces_baseline_once_and_retains_spatial_metadata(self):
        model = personal()
        phase = np.asarray(list(model.phase_dn.values())).reshape(2, 2)
        gradient = np.arange(16)[None, :] * .01
        black = 512 + np.broadcast_to(gradient, (16, 16))
        raw = np.tile(phase, (8, 8)) + gradient + .2
        original = raw.copy()
        result, evidence = signed_black_correct(raw, black, model, capture(),
                                               metadata_phase_black=dict.fromkeys(PHASES, 512.))
        np.testing.assert_allclose(result, .2, atol=1e-12)
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(evidence['phase_uncertainty_dn'], dict.fromkeys(PHASES, .1))
        residual = replace(model, mode='residual', phase_dn={p: model.phase_dn[p] - 512 for p in PHASES})
        corrected, _ = signed_black_correct(raw, black, residual, capture())
        np.testing.assert_array_equal(corrected, result)

    def test_mismatch_missing_stale_conditions_reject(self):
        for key, value in (('body_serial', 'another'), ('readout_identity', None), ('iso', 200),
                           ('exposure_seconds', .03), ('temperature_c', None), ('firmware', '2'),
                           ('evaluation_utc', '2028-01-01T00:00:00Z')):
            with self.subTest(key=key):
                info = capture(); info[key] = value
                with self.assertRaises(ValueError):
                    signed_black_correct(np.zeros((8, 8)), 0, personal(), info,
                                         metadata_phase_black=dict.fromkeys(PHASES, 512.))

    def test_identity_includes_values_uncertainty_and_applicability(self):
        base = personal()
        for change in ({'phase_dn': dict.fromkeys(PHASES, 512.)},
                       {'phase_uncertainty_dn': dict.fromkeys(PHASES, .2)},
                       {'source_hash': 'sha256:other'}, {'body_serial': 'body-2'},
                       {'exposure_seconds': (.01, .03)}, {'mode': 'residual'}):
            self.assertNotEqual(base.identity(), replace(base, **change).identity())

    def test_incomplete_phase_or_wrong_order_is_not_invented(self):
        for model in (replace(personal(), phase_dn={'C01': 512.}),
                      replace(personal(), domain='post-stage2'),
                      replace(personal(), phase_uncertainty_dn=dict.fromkeys(PHASES, -.1))):
            with self.assertRaises(ValueError):
                model.validate()

    def test_required_capture_scope_and_malformed_measurements_explicitly_reject(self):
        changes = ({'exposure_seconds':None}, {'exposure_seconds':(.01,)},
                   {'exposure_seconds':('0.01','0.02')}, {'exposure_seconds':(True,True)},
                   {'iso':None}, {'iso':'100'}, {'iso':True}, {'iso':float('nan')},
                   {'phase_dn':None}, {'phase_dn':list(PHASES)},
                   {'phase_dn':dict.fromkeys(PHASES,[512.])},
                   {'phase_uncertainty_dn':dict.fromkeys(PHASES,[.1])},
                   {'firmware':''}, {'valid_until_utc':'not-a-date'},
                   {'valid_until_utc':'2027-01-01T00:00:00'}, {'readout_identity':' '})
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                replace(personal(),**change).validate()

    def test_malformed_capture_is_not_an_incidental_type_error_or_a_match(self):
        for key in ('iso','exposure_seconds','temperature_c'):
            for value in (None,'0.015',True,[],float('nan')):
                info = capture(); info[key] = value
                with self.subTest(key=key,value=value):
                    self.assertFalse(personal().match(info)[0])
                    with self.assertRaises(ValueError):
                        signed_black_correct(np.zeros((8,8)),0,personal(),info,
                            metadata_phase_black=dict.fromkeys(PHASES,512.))
        with self.assertRaisesRegex(ValueError,'mapping'):
            personal().match(None)

    def test_readonly_phase_mapping_and_numpy_scalar_identity_are_supported(self):
        model = personal()
        readonly = replace(model,iso=np.float64(100),
            phase_dn=MappingProxyType({key:np.float64(v) for key,v in model.phase_dn.items()}),
            phase_uncertainty_dn=MappingProxyType(dict(model.phase_uncertainty_dn)))
        self.assertEqual(readonly.identity(),model.identity())

    def test_absolute_metadata_baselines_need_four_scalar_values(self):
        for baseline in (list(PHASES), {'C00':512.}, dict.fromkeys(PHASES,[512.])):
            with self.subTest(baseline=baseline),self.assertRaises(ValueError):
                signed_black_correct(np.zeros((8,8)),0,personal(),capture(),
                                     metadata_phase_black=baseline)


class SignedPrecisionTests(unittest.TestCase):
    def test_real_spatial_black_zero_clip_bias_and_signed_reference(self):
        rng = np.random.default_rng(816)
        black, sigma, white = 512.25, 4., 16383.
        raw = np.rint(black + rng.normal(0, sigma, (512, 512))).astype(np.uint16)
        residual = raw.astype(float) - black
        handle = SimpleNamespace(raw_image_visible=raw.copy())
        spatial = SpatialBlack(np.zeros(512), np.zeros(512), np.full((1, 1, 1), black),
                               (0, 0), (black,))
        loss = np.zeros(raw.shape, np.uint8)
        apply_to_working(handle, spatial, [black] * 4, white, loss)
        equivalent = handle.raw_image_visible.astype(float) * (white - black) / 65535.
        self.assertLess(abs(float(residual.mean())), .03)
        self.assertGreater(float(equivalent.mean()), 1.5)
        self.assertLess(float(equivalent.var()), .36 * float(residual.var()))
        self.assertFalse(np.any(loss))  # Existing mask is deliberately upper-loss evidence.
        signed = signed_bilinear_bayer(residual, 'RGGB')
        self.assertLess(abs(float(signed[4:-4, 4:-4].mean())), .04)
        self.assertLess(float(signed.min()), 0)

    def test_signed_cfa_preserves_constants_phase_gains_and_all_bayer_orders(self):
        for pattern in ('RGGB', 'BGGR', 'GBRG', 'GRBG'):
            raw = np.full((16, 20), -.25)
            result = signed_bilinear_bayer(raw, pattern)
            np.testing.assert_array_equal(result, -.25)
        gains = [2, 1, 3, 4]
        result = signed_bilinear_bayer(np.ones((16, 20)), 'RGGB', phase_gains=gains)
        np.testing.assert_array_equal(result[2:-2, 2:-2], [2, 2, 4] * np.ones((12, 16, 3)))

    def test_repeated_fractional_gain_truncation_has_directional_bias(self):
        raw = np.linspace(1, 1000, 10000)
        gains = [1.003, .997, 1.007, .993]
        exact = raw * np.prod(gains)
        errors = {p: quantized_gain_chain(raw, gains, white=4095, policy=p) - exact
                  for p in ('truncate', 'nearest', 'deferred')}
        self.assertLess(float(errors['truncate'].mean()), -1.8)
        self.assertLess(abs(float(errors['nearest'].mean())), .04)
        self.assertLess(abs(float(errors['deferred'].mean())), .02)
        self.assertLess(float(np.mean(errors['deferred'] ** 2)), float(np.mean(errors['nearest'] ** 2)))

    def test_quantization_contract_preserves_upper_clip_and_rejects_invalid_gain(self):
        for policy in ('truncate', 'nearest', 'deferred'):
            np.testing.assert_array_equal(quantized_gain_chain([4090, 10], [2., .5], white=4095, policy=policy),
                                          [2047 if policy == 'truncate' else 2048, 10])
        with self.assertRaises(ValueError):
            quantized_gain_chain([0], [-1], white=4095, policy='nearest')


class DarkStackTests(unittest.TestCase):
    def test_heldout_map_reduces_fixed_bias_without_fitting_random_noise_or_scene(self):
        rng = np.random.default_rng(447)
        h, w, n, sigma = 64, 80, 64, 2.
        fixed = 3 * np.sin(2 * np.pi * np.arange(h)[:, None] / 16) + np.zeros((h, w))
        fixed[21, 31] += 40
        train = 512 + fixed + rng.normal(0, sigma, (n, h, w))
        product = dark_stack_products(train, calibration_identity='same-body-iso-time-temp')
        self.assertAlmostEqual(float(product['temporal_variance_dn2'].mean()), sigma ** 2, delta=.08)
        self.assertAlmostEqual(float(product['pair_variance_dn2'].mean()), sigma ** 2, delta=.08)
        self.assertTrue(product['hot_pixel_candidates'][21, 31])
        weak_scene = np.broadcast_to(.2 * np.cos(np.arange(w))[None, :], (h, w))
        heldout = 512 + fixed + weak_scene + rng.normal(0, sigma, (32, h, w))
        corrected = heldout - product['mean_bias_map_dn']
        baseline = heldout - 512
        self.assertLess(float(np.mean((corrected.mean(0) - weak_scene) ** 2)),
                        float(np.mean((baseline.mean(0) - weak_scene) ** 2)) / 20)
        np.testing.assert_allclose(corrected.var(0), heldout.var(0), rtol=1e-12)
        self.assertFalse(product['correction_authorized'])

    def test_insufficient_or_unidentified_stack_rejected(self):
        for frames, identity in ((np.zeros((2, 8, 8)), 'a'), (np.zeros((4, 8, 8)), '')):
            with self.assertRaises(ValueError):
                dark_stack_products(frames, calibration_identity=identity)

    def test_empty_or_malformed_raster_and_nontext_identity_reject(self):
        for frames in (np.zeros((4,0,8)),np.zeros((4,8,0)),np.zeros((4,0,0)),
                       np.zeros((4,8)),np.full((4,2,2),np.nan),[[{},{}]]*4):
            with self.subTest(shape=np.shape(frames)),self.assertRaises(ValueError):
                dark_stack_products(frames,calibration_identity='measurement')
        for identity in (None,True,1,['measurement'],{},' '):
            with self.subTest(identity=identity),self.assertRaises(ValueError):
                dark_stack_products(np.zeros((4,2,2)),calibration_identity=identity)

    def test_single_pixel_sensor_stack_is_valid_without_claiming_bayer_calibration(self):
        product = dark_stack_products(np.arange(5.).reshape(5,1,1),calibration_identity='mono-measurement')
        self.assertEqual(product['frame_count'],5)
        self.assertEqual(product['mean_bias_map_dn'].shape,(1,1))
        self.assertFalse(product['correction_authorized'])


class JointChromaTests(unittest.TestCase):
    def test_full_covariance_joint_shrink_preserves_luma_and_vector_direction(self):
        directions = np.random.default_rng(8).normal(0, .1, (100, 3))
        covariance = np.asarray([[.0004, .0001, 0], [.0001, .0002, -.00005], [0, -.00005, .0008]])
        result, _ = joint_chroma_shrink(directions, covariance, strength=.5)
        np.testing.assert_allclose(result @ LUMA, directions @ LUMA, rtol=1e-13, atol=1e-16)
        before = directions - (directions @ LUMA)[:, None]
        after = result - (result @ LUMA)[:, None]
        np.testing.assert_allclose(np.cross(before, after), 0, atol=1e-16)
        identity, _ = joint_chroma_shrink(directions, covariance, strength=0)
        np.testing.assert_allclose(identity, directions, rtol=1e-13, atol=1e-16)

    def test_singular_ill_conditioned_or_nonphysical_models_rejected(self):
        for cov in (np.zeros((3, 3)), np.diag([1, 1e-12, 1e-12]), -np.eye(3),
                    np.asarray([[1, .5, 0], [0, 1, 0], [0, 0, 1]])):
            with self.assertRaises(ValueError):
                joint_chroma_shrink(np.ones((4, 3)), cov, max_condition=1e4)

    def test_covariance_scale_changes_shrink_and_weak_structure_can_be_lost(self):
        direction = np.array([1, -LUMA[0] / LUMA[1], 0])
        low = direction * .001
        high = direction * .1
        out, _ = joint_chroma_shrink(np.array([low, high]), np.eye(3) * .0001)
        np.testing.assert_allclose(out[0], 0, atol=1e-17)
        self.assertGreater(float(np.linalg.norm(out[1]) / np.linalg.norm(high)), .8)


if __name__ == '__main__':
    unittest.main()
