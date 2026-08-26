# SPDX-License-Identifier: GPL-3.0-or-later
"""Route-C phase-2 runtime contracts: the chromaticity-field Stage A.

The dispatcher (film_v2_math.stage_a_log_exposure) routes CV-adopted stocks
through E = Y * 2^L(x,y) and everything else through the 3x3 observer. The
contracts pinned here are the ones the phase-2 design document promised:
exposure homogeneity survives the LUT, grey ramps stay on the observer's
logE axis (white re-anchor), degenerate inputs fall back to the exact 3x3,
and the non-adopted stock is bit-identical to the pure observer path.
"""
from __future__ import annotations

import unittest

import numpy as np

from dngscan.film_develop import _load_v2
from dngscan.film_v2_math import (
    chroma_field_log_exposure,
    layer_log_exposure,
    stage_a_log_exposure,
)

ADOPTED = "portra400"
RETAINED = "pro400h"


def _stock(name: str) -> dict:
    return _load_v2(name)[0]


class DispatchTests(unittest.TestCase):
    def test_adoption_matches_the_cv_selection(self) -> None:
        self.assertIsNotNone(_stock(ADOPTED)["chroma_lut"])
        self.assertIsNone(_stock(RETAINED)["chroma_lut"])

    def test_retained_stock_is_bit_identical_to_the_observer_path(self) -> None:
        stock = _stock(RETAINED)
        rng = np.random.default_rng(20260826)
        rgb = 0.18 * np.exp2(rng.uniform(-9.0, 5.0, (256, 3)))
        np.testing.assert_array_equal(
            stage_a_log_exposure(rgb, stock),
            layer_log_exposure(rgb, stock["observer"]),
        )

    def test_adopted_stock_actually_diverges_from_the_observer(self) -> None:
        stock = _stock(ADOPTED)
        rng = np.random.default_rng(20260826)
        rgb = 0.18 * np.exp2(rng.uniform(-4.0, 3.0, (256, 3)))
        field = stage_a_log_exposure(rgb, stock)
        obs = layer_log_exposure(rgb, stock["observer"])
        self.assertGreater(float(np.abs(field - obs).max()), 1e-3)


class FieldContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stock = _stock(ADOPTED)

    def test_exposure_homogeneity_through_the_lut(self) -> None:
        rng = np.random.default_rng(7)
        rgb = 0.18 * np.exp2(rng.uniform(-6.0, 4.0, (128, 3)))
        base = stage_a_log_exposure(rgb, self.stock)
        for k in (0.125, 2.0, 11.3):
            scaled = stage_a_log_exposure(rgb * k, self.stock)
            np.testing.assert_allclose(
                scaled - base,
                np.full_like(base, np.log10(k)),
                atol=1e-9,
                err_msg=f"E(k*x) != k*E(x) at k={k}",
            )

    def test_grey_ramp_stays_on_the_observer_logE_axis(self) -> None:
        """The white re-anchor promise: neutral inputs must land where the
        3x3 lands them (the characteristic tables' axis), to bilinear
        exactness — otherwise every neutral gate downstream would drift."""
        g = 0.18 * np.exp2(np.linspace(-8.0, 4.0, 49))
        grey = np.repeat(g[:, None], 3, axis=1)
        field = stage_a_log_exposure(grey, self.stock)
        obs = layer_log_exposure(grey, self.stock["observer"])
        np.testing.assert_allclose(field, obs, atol=5e-7)

    def test_degenerate_inputs_take_the_exact_observer_path(self) -> None:
        rgb = np.asarray(
            [
                [0.0, 0.0, 0.0],          # channel clamp -> neutral, still fine
                [-1.0, -1.0, -1.0],       # all-negative clamps to neutral floor
                [np.nan, 0.2, 0.3],       # non-finite falls back
            ],
            dtype=np.float64,
        )
        out = chroma_field_log_exposure(
            rgb,
            self.stock["chroma_lut"],
            self.stock["chroma_domain"],
            self.stock["chroma_xyz_from_rec2020"],
            self.stock["observer"],
        )
        obs = layer_log_exposure(rgb, self.stock["observer"])
        # Rows 0/1 clamp to the neutral floor -> chromaticity is white, both
        # paths agree to bilinear exactness; row 2 must be the exact fallback.
        np.testing.assert_allclose(out[:2], obs[:2], atol=5e-7)
        np.testing.assert_array_equal(out[2], obs[2])

    def test_runtime_chromaticities_cannot_leave_the_lut_domain(self) -> None:
        """Channel-clamped positive Rec.2020 keeps chromaticity inside the
        primary triangle, whose bounding box (plus pad) is the LUT domain —
        so saturated primaries still ride the LUT, not the fallback."""
        prim = np.eye(3) * 3.7 + 1e-9
        out = stage_a_log_exposure(prim, self.stock)
        self.assertTrue(bool(np.isfinite(out).all()))
        x0, x1, y0, y1 = self.stock["chroma_domain"]
        m = self.stock["chroma_xyz_from_rec2020"]
        xyz = np.maximum(prim, 1e-9) @ m.T
        s = xyz.sum(axis=1)
        self.assertTrue(bool(np.all((xyz[:, 0] / s >= x0) & (xyz[:, 0] / s <= x1))))
        self.assertTrue(bool(np.all((xyz[:, 1] / s >= y0) & (xyz[:, 1] / s <= y1))))


if __name__ == "__main__":
    unittest.main()
