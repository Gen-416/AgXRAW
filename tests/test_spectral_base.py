# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed numeric regressions for the spectral calibration base.

These pin the physical conventions of the printing chain (TH-KG3 illuminant,
grids, integration, viewing translation) so a refit can never silently drift
because a primitive changed underneath it.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import spectral_base as sb


class Kg3Tests(unittest.TestCase):
    def test_duplicate_wavelengths_merge_to_139_by_mean(self) -> None:
        wl, tr = sb.load_kg3_samples()
        self.assertEqual(wl.size, 139)  # 146 rows, 7 duplicated abscissae
        self.assertTrue(np.all(np.diff(wl) > 0.0))
        self.assertTrue(np.all((tr >= 0.0) & (tr <= 1.0)))

    def test_outside_tabulated_range_is_zero(self) -> None:
        tr = sb.kg3_transmission(np.array([200.0, 550.0, 1200.0]))
        self.assertEqual(tr[0], 0.0)
        self.assertEqual(tr[2], 0.0)
        self.assertGreater(tr[1], 0.5)


class ThKg3Tests(unittest.TestCase):
    def test_fixed_regression_on_the_printing_grid(self) -> None:
        grid = np.arange(380.0, 781.0, 5.0)
        spd = sb.th_kg3_spd(grid)
        self.assertAlmostEqual(float(spd.mean()), 1.0, places=12)
        np.testing.assert_allclose(
            spd[:3], [0.270900, 0.292990, 0.315243], atol=2e-5
        )
        np.testing.assert_allclose(
            spd[-3:], [0.487207, 0.449722, 0.414612], atol=2e-5
        )

    def test_heat_filter_suppresses_the_red_end_vs_bare_blackbody(self) -> None:
        grid = np.arange(380.0, 781.0, 5.0)
        spd = sb.th_kg3_spd(grid)
        bare = sb.blackbody_spd(grid, 3200.0)
        bare = bare / bare.mean()
        ratio = float(spd[-17:].sum() / bare[-17:].sum())
        self.assertLess(ratio, 0.5)  # this is why 3200K-bare was not a detail

    def test_provenance_names_the_rules(self) -> None:
        prov = sb.th_kg3_provenance()
        self.assertEqual(prov["kg3_samples"], 139)
        for key in ("kg3_sha256", "dedup_rule", "interpolation", "normalization"):
            self.assertIn(key, prov)


class GridAndIntegrationTests(unittest.TestCase):
    def test_viewing_grid_is_the_intersection_not_an_extension(self) -> None:
        grid = sb.intersect_grid(np.arange(380.0, 781.0, 5.0))
        self.assertEqual(float(grid[0]), 380.0)
        self.assertEqual(float(grid[-1]), 780.0)
        self.assertEqual(grid.size, 81)

    def test_trapezoid_matches_the_analytic_integral(self) -> None:
        x = np.linspace(0.0, np.pi, 20001)
        self.assertAlmostEqual(float(sb.trapezoid(np.sin(x), x)), 2.0, places=7)


class ViewingTranslationTests(unittest.TestCase):
    D50 = np.array([0.9642, 1.0, 0.8249])

    def test_bradford_maps_the_source_white_exactly_onto_d65(self) -> None:
        cat = sb.bradford_cat(self.D50)
        np.testing.assert_allclose(cat @ self.D50, sb.XYZ_D65, atol=1e-12)

    def test_d65_white_is_exactly_rec2020_ones(self) -> None:
        np.testing.assert_allclose(
            sb.XYZ_TO_REC2020 @ sb.XYZ_D65, np.ones(3), atol=1e-12
        )

    def test_medium_white_lands_neutral_after_the_full_translation(self) -> None:
        # A D50-viewed medium white must come out neutral in Rec.2020(D65):
        # the CAT carries the white difference so printer lights never have to.
        rgb = sb.viewing_translation_rec2020(
            self.D50[None, :], self.D50, flare=0.01, surround_exponent=1.0
        )
        np.testing.assert_allclose(rgb[0], np.full(3, rgb[0, 1]), rtol=1e-12)

    def test_surround_touches_luminance_only(self) -> None:
        xyz = np.array([[0.20, 0.18, 0.30]])
        base = sb.viewing_translation_rec2020(xyz, self.D50, 0.0, 1.0)
        dim = sb.viewing_translation_rec2020(xyz, self.D50, 0.0, 1.2)
        # Chromaticity (per-channel ratios) unchanged; only the scale moved.
        np.testing.assert_allclose(
            dim[0] / dim[0, 1], base[0] / base[0, 1], rtol=1e-10
        )
        self.assertNotAlmostEqual(float(dim[0, 1]), float(base[0, 1]))


if __name__ == "__main__":
    unittest.main()
