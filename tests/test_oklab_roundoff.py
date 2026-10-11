# SPDX-License-Identifier: GPL-3.0-or-later
"""Signed Oklab precision near LMS cancellation; unchanged parity budgets."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import _fast
from dngscan.color import (
    _oklab_lms_with_roundoff_bound, apply_rgb_matrix3, encode_display_linear,
    fit_to_output_gamut, output_gamut_space, rgb_to_oklab,
)
from dngscan.constants import OKLAB_M1, RGB_TO_XYZ


PROBE = np.asarray([
    [.1406678855419159, -.01818111352622509, -.011180703528225422],
    [.1406678706407547, -.018181087449193, -.01118070911616087],
], np.float32)


class OklabRoundoffTests(unittest.TestCase):
    def test_original_independent_RGB_probe_stays_within_existing_budget(self):
        with patch.dict(os.environ, {"DNGSCAN_FAST": "off"}):
            encoded = encode_display_linear(fit_to_output_gamut(PROBE, "p3", alpha=.045), "p3")
            lms, bound = _oklab_lms_with_roundoff_bound(PROBE, "p3")
        np.testing.assert_array_equal(lms[:, 2], 0)
        self.assertLess(float(np.max(np.abs(encoded[1] - encoded[0]))), 1e-4)
        self.assertTrue(np.all(bound[:, 2] > 0))

    def test_positive_and_negative_ULP_neighbours_share_the_cancellation_convention(self):
        rows = []
        for base in PROBE:
            for channel in range(3):
                for sign in (-np.inf, np.inf):
                    row = base.copy()
                    for _ in range(2):
                        row[channel] = np.nextafter(row[channel], np.float32(sign))
                    rows.append(row)
        rows = np.asarray(rows, np.float32)
        with patch.dict(os.environ, {"DNGSCAN_FAST": "off"}):
            matrix = RGB_TO_XYZ[output_gamut_space("p3")]
            raw_lms = apply_rgb_matrix3(apply_rgb_matrix3(rows, matrix), OKLAB_M1)
            lms, bound = _oklab_lms_with_roundoff_bound(rows, "p3")
            encoded = encode_display_linear(fit_to_output_gamut(rows, "p3", alpha=.045), "p3")
        self.assertTrue(np.any(raw_lms[:, 2] < 0))
        self.assertTrue(np.any(raw_lms[:, 2] > 0))
        ambiguous = np.abs(raw_lms) <= bound
        np.testing.assert_array_equal(lms[ambiguous], 0)
        np.testing.assert_array_equal(lms[~ambiguous], raw_lms[~ambiguous])
        # An ULP perturbation outside the error interval is not declared
        # equivalent to zero: signed cbrt is inherently ill-conditioned near
        # zero. Only indistinguishable cancellation components share the
        # canonical value; the other assertions retain the true signed input.
        cancelled = ambiguous[:, 2]
        self.assertGreater(int(cancelled.sum()), 1)
        self.assertLess(float(np.max(np.ptp(encoded[cancelled], axis=0))), 1e-4)

    def test_meaningful_negative_LMS_is_preserved_not_positive_clipped(self):
        for gamut in ("srgb", "p3"):
            matrix = OKLAB_M1 @ RGB_TO_XYZ[output_gamut_space(gamut)]
            target = np.asarray([[.03, .02, -.001], [.03, .02, .001]])
            rgb = (target @ np.linalg.inv(matrix).T).astype(np.float32)
            with patch.dict(os.environ, {"DNGSCAN_FAST": "off"}):
                lms, bound = _oklab_lms_with_roundoff_bound(rgb, gamut)
                lab = np.stack(rgb_to_oklab(rgb, gamut), axis=-1)
            self.assertLess(float(lms[0, 2]), 0)
            self.assertGreater(float(lms[1, 2]), 0)
            self.assertTrue(np.all(np.abs(lms[:, 2]) > bound[:, 2] * 1000))
            self.assertFalse(np.array_equal(lab[0], lab[1]))

    def test_zero_convention_and_bound_scale_with_global_gain(self):
        with patch.dict(os.environ, {"DNGSCAN_FAST": "off"}):
            reference, bound = _oklab_lms_with_roundoff_bound(PROBE, "p3")
            for gain in (2. ** -12, .5, 2., 2. ** 12):
                actual, actual_bound = _oklab_lms_with_roundoff_bound(PROBE * gain, "p3")
                np.testing.assert_array_equal(actual, reference * gain)
                np.testing.assert_allclose(actual_bound, bound * gain, rtol=1e-14, atol=0)
        np.testing.assert_array_equal(_oklab_lms_with_roundoff_bound(np.zeros((2, 3)), "p3")[0], 0)

    def test_in_gamut_values_remain_exact_and_source_is_unchanged(self):
        rgb = np.asarray([[0., 0., 0.], [.001, .02, .3], [.5, .2, .9], [1., 1., 1.]], np.float32)
        before = rgb.copy()
        for gamut in ("srgb", "p3"):
            np.testing.assert_array_equal(fit_to_output_gamut(rgb, gamut), rgb)
        np.testing.assert_array_equal(rgb, before)

    def test_overflow_and_nonfinite_LMS_are_not_reclassified_as_zero(self):
        # Relative forward bounds apply to finite arithmetic. In particular,
        # inf <= inf must not hide an existing matrix overflow as cancellation.
        huge = np.finfo(np.float32).max
        rgb = np.asarray([[huge, huge, huge], [huge, -huge, huge],
                          [np.inf, 0, 0], [np.nan, 0, 0]], np.float32)
        before = rgb.copy()
        for gamut in ("srgb", "p3"):
            with patch.dict(os.environ, {"DNGSCAN_FAST": "off"}), np.errstate(over="ignore", invalid="ignore"):
                matrix = RGB_TO_XYZ[output_gamut_space(gamut)]
                raw_lms = apply_rgb_matrix3(apply_rgb_matrix3(rgb, matrix), OKLAB_M1)
                actual, bound = _oklab_lms_with_roundoff_bound(rgb, gamut)
            outside_contract = ~np.isfinite(raw_lms) | ~np.isfinite(bound)
            self.assertTrue(np.any(outside_contract))
            np.testing.assert_array_equal(actual[outside_contract], raw_lms[outside_contract])
        np.testing.assert_array_equal(rgb, before)

    def test_native_gamut_fit_uses_the_same_roundoff_contract(self):
        if not _fast.available():
            self.skipTest("native extension unavailable")
        rng = np.random.default_rng(6024)
        rgb = np.concatenate([PROBE, rng.uniform(-.05, 1.2, (4096, 3)).astype(np.float32)])
        for gamut in ("srgb", "p3"):
            with patch.dict(os.environ, {"DNGSCAN_FAST": "off"}):
                reference = fit_to_output_gamut(rgb, gamut, alpha=.045)
            with patch.dict(os.environ, {"DNGSCAN_FAST": "1"}):
                actual = _fast.fit_output_gamut_f32(rgb, _fast.compile_output_plan(gamut, .045))
            self.assertLess(float(np.max(np.abs(actual - reference))), 1e-4)
            if gamut == "p3":
                self.assertLess(float(np.max(np.abs(actual[1] - actual[0]))), 1e-4)


if __name__ == "__main__":
    unittest.main()
