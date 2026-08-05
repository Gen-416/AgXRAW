# SPDX-License-Identifier: GPL-3.0-or-later
"""Declared lens filter gates: published mired shifts, physical direction, homogeneity."""
from __future__ import annotations

import unittest

import numpy as np

from dngscan.constants import RGB_TO_XYZ
from dngscan.lens_filter import (
    LENS_FILTERS,
    apply_lens_filter_rec2020,
    lens_filter_matrix,
    shifted_cct,
    validate_lens_filter,
)
from dngscan.wb import cct_to_xy


def _cct_of_rec2020(rgb: np.ndarray) -> float:
    """McCamy CCT approximation from a Rec.2020 triplet, good to tens of Kelvin."""
    xyz = np.array(RGB_TO_XYZ["Rec2020"], dtype=np.float64) @ np.asarray(rgb, dtype=np.float64)
    x = xyz[0] / xyz.sum()
    y = xyz[1] / xyz.sum()
    n = (x - 0.3320) / (0.1858 - y)
    return 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33


class MiredArithmeticTests(unittest.TestCase):
    def test_published_conversion_shifts_land_on_their_targets(self) -> None:
        self.assertAlmostEqual(shifted_cct(5500.0, +131.0), 3200.0, delta=25.0)
        self.assertAlmostEqual(shifted_cct(5500.0, +112.0), 3400.0, delta=35.0)
        self.assertAlmostEqual(shifted_cct(3200.0, -131.0), 5500.0, delta=60.0)

    def test_unknown_filter_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            validate_lens_filter("nd8")


class FilterMatrixTests(unittest.TestCase):
    def test_85b_shifts_the_rendered_neutral_by_its_mireds(self) -> None:
        """The cast lands at the working white's CCT shifted by +131 mired.

        "5500K light becomes 3200K" describes light entering the filter; the rendered
        neutral in a D65-anchored working space picks up D65 (+131 mired) ~= 3512K.
        Mired invariance is the whole reason filters are catalogued in mireds.
        """
        filtered = apply_lens_filter_rec2020(np.ones((1, 3), dtype=np.float32), "85b")[0]
        self.assertGreater(filtered[0], filtered[2])  # warm: R above B
        expected = shifted_cct(6504.0, +131.0)
        # Gate is mired-scale correctness (~5%), not colorimetric exactness:
        # McCamy and daylight/Planck family mixing are both worth ~200K here.
        self.assertAlmostEqual(_cct_of_rec2020(filtered), expected, delta=260.0)

    def test_80a_cools_in_the_blue_direction(self) -> None:
        """-131 mired from a D65 anchor runs far up the locus; assert direction and
        magnitude in mired space via the roundtrip test below, chromaticity here."""
        filtered = apply_lens_filter_rec2020(np.ones((1, 3), dtype=np.float32), "80a")[0]
        self.assertGreater(filtered[2], filtered[0])

    def test_neutral_luminance_is_preserved(self) -> None:
        """The published filter factor is exposure bookkeeping, normalized out."""
        weights = np.array(RGB_TO_XYZ["Rec2020"], dtype=np.float64)[1]
        for name in LENS_FILTERS:
            with self.subTest(filter=name):
                filtered = apply_lens_filter_rec2020(
                    np.ones((1, 3), dtype=np.float32), name
                )[0]
                self.assertAlmostEqual(
                    float(weights @ filtered.astype(np.float64)),
                    float(weights @ np.ones(3)),
                    delta=5e-5,
                )

    def test_exposure_homogeneity_is_exact(self) -> None:
        """Glass is linear: doubling the light doubles the filtered result."""
        rng = np.random.default_rng(7)
        rgb = rng.uniform(0.01, 4.0, size=(64, 3)).astype(np.float32)
        one = apply_lens_filter_rec2020(rgb, "85b")
        two = apply_lens_filter_rec2020(rgb * np.float32(2.0), "85b")
        np.testing.assert_allclose(two, one * 2.0, rtol=1e-6)

    def test_opposite_filters_roughly_cancel(self) -> None:
        """+131 then -131 mired returns close to the original balance."""
        m_warm = lens_filter_matrix("85b")
        m_cool = lens_filter_matrix("80a")
        roundtrip = m_cool.astype(np.float64) @ m_warm.astype(np.float64)
        # Symmetric anchoring makes equal-and-opposite pairs invert exactly in
        # chromaticity; a scalar remains from the two independent Y-normalizations
        # (each matrix normalizes its own neutral, not the other's cast).
        np.testing.assert_allclose(roundtrip / roundtrip[1, 1], np.eye(3), atol=1e-4)

    def test_none_is_identity_passthrough(self) -> None:
        rgb = np.full((4, 3), 0.5, dtype=np.float32)
        self.assertIs(apply_lens_filter_rec2020(rgb, "none"), rgb)


class IntentPlumbingTests(unittest.TestCase):
    def test_bundle_filter_reaches_scene_intent(self) -> None:
        from types import SimpleNamespace

        from dngscan.tone import scene_intent_rec2020

        flat = np.full((8, 3), 1000.0, dtype=np.float32)
        base = SimpleNamespace(
            scene_scale=1000.0, exposure_gain=1.0, lens_filter="none", wb_mode="5500k"
        )
        warm = SimpleNamespace(
            scene_scale=1000.0, exposure_gain=1.0, lens_filter="85b", wb_mode="5500k"
        )
        plain = scene_intent_rec2020(flat, base)
        filtered = scene_intent_rec2020(flat, warm)
        np.testing.assert_allclose(plain, np.ones((8, 3)), rtol=1e-5)
        self.assertGreater(float(filtered[0, 0]), float(filtered[0, 2]))


if __name__ == "__main__":
    unittest.main()
