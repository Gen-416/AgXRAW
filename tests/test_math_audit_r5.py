# SPDX-License-Identifier: GPL-3.0-or-later
"""Math audit R5 gates (five-agent numerical verification, 2026-08-18).

Each gate pins a finding from the independent re-derivation audit: the RWG
matrix transcription fix, the pivot-path underflow guard, and the shipped
characteristic tables' bounded cube-floor excursion. The audit's PASS
coverage lives in its probe scripts; these tests keep the fixed points
fixed."""
from __future__ import annotations

import unittest

import numpy as np


class RwgMatrixTests(unittest.TestCase):
    def test_rwg_matrix_reproduces_d65_exactly(self) -> None:
        """The [2][2]=1.516745 transcription error put the white point at
        xy (0.31263, 0.32893); the corrected matrix must map RWG (1,1,1)
        to D65 within float64 rounding of the primaries derivation."""
        from dngscan.log_encode import _RWG_TO_XYZ

        white = np.asarray(_RWG_TO_XYZ) @ np.ones(3)
        x = white[0] / white.sum()
        y = white[1] / white.sum()
        self.assertAlmostEqual(float(x), 0.3127, places=4)
        self.assertAlmostEqual(float(y), 0.3290, places=4)
        # And the derivation itself: rebuild from RED's published primaries.
        prim = np.array(
            [[0.780308, 0.304253], [0.121595, 1.493994], [0.095612, -0.084589]]
        )
        wp = np.array([0.3127, 0.3290])

        def xy_to_xyz(xy):
            return np.array([xy[0] / xy[1], 1.0, (1 - xy[0] - xy[1]) / xy[1]])

        p = np.stack([xy_to_xyz(v) for v in prim], axis=1)
        m = p * np.linalg.solve(p, xy_to_xyz(wp))
        np.testing.assert_allclose(np.asarray(_RWG_TO_XYZ), m, atol=5.5e-7)


class PivotUnderflowGuardTests(unittest.TestCase):
    def test_lifted_black_pivot_plan_compiles(self) -> None:
        """Audit repro: pivot_ev_offset in the declared range plus a film
        black floor drove tx**power to underflow and curve_params to
        ZeroDivisionError. The guard must compile a finite, monotone curve."""
        from dngscan import agx

        params = agx.curve_params(
            -10.0, 2.0, 4.5, 0.5, 0.7, 0.0, 0.0,
            pivot_ev_offset=-1.5, target_black_linear=0.02,
        )
        for v in params.values():
            if isinstance(v, float):
                self.assertTrue(np.isfinite(v), f"non-finite param {v}")
        xs = np.linspace(0.0, 1.0, 4096, dtype=np.float64)
        ys = np.asarray(agx.apply_curve(xs, params), dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(ys)))
        self.assertTrue(np.all(np.diff(ys) >= -1e-9), "curve must stay monotone")


class CubeFloorExcursionGateTests(unittest.TestCase):
    def test_shipped_tables_excurse_below_the_floor_only_boundedly(self) -> None:
        """The corrected contract: shipped characteristic tables MAY dip
        below the Stage B cube floor (fit residual vs the physical
        amount>=0 floor) by a bounded amount — audit worst -1.56e-3, gate
        at 2e-3. Beyond that is a corrupt or mis-built asset. Above-ceiling
        excursion is not expected at all."""
        from dngscan.film_develop import _load_v2
        from dngscan.film_curve import FILM_CURVE_PRESETS

        checked = 0
        for preset in FILM_CURVE_PRESETS:
            try:
                stock, _media = _load_v2(preset)
            except Exception:
                continue  # presets without v2 assets are not in scope
            below = np.maximum(stock["lo"][None, :] - stock["char_amounts"], 0.0)
            above = np.maximum(stock["char_amounts"] - stock["hi"][None, :], 0.0)
            self.assertLess(
                float(below.max()), 2e-3,
                f"{preset}: cube-floor excursion beyond the audited bound",
            )
            self.assertEqual(
                float(above.max()), 0.0,
                f"{preset}: table exceeds the cube ceiling",
            )
            checked += 1
        self.assertGreater(checked, 10, "gate must actually cover the assets")


if __name__ == "__main__":
    unittest.main()
