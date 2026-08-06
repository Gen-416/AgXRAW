# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the project-authored chromatic look layer."""

from __future__ import annotations

import unittest

import numpy as np

from dngscan.look import (
    LOOK_FIELDS,
    _apply_look_oklab_reference,
    _hue_in_arc,
    apply_look_oklab,
)


class LookLayerTests(unittest.TestCase):
    def test_parallel_hot_path_is_bit_exact(self) -> None:
        rng = np.random.default_rng(20260804)
        lightness = rng.uniform(-0.25, 1.5, 80_000).astype(np.float32)
        a = rng.normal(0.0, 0.35, lightness.size).astype(np.float32)
        b = rng.normal(0.0, 0.35, lightness.size).astype(np.float32)
        lightness[:5] = [0.0, 1.0, np.nan, np.inf, -np.inf]
        a[:5] = [0.0, -0.0, np.nan, np.inf, -np.inf]
        b[:5] = [-0.0, 0.0, np.inf, np.nan, -np.inf]
        with np.errstate(all="ignore"):
            expected = _apply_look_oklab_reference(
                lightness, a, b, "optic_warm_cyan", 1.0
            )
            actual = apply_look_oklab(
                lightness, a, b, "optic_warm_cyan", 1.0
            )
        for optimized, reference in zip(actual, expected):
            np.testing.assert_array_equal(optimized, reference)

    def test_public_registry_contains_only_project_look(self) -> None:
        self.assertEqual(set(LOOK_FIELDS), {"optic_warm_cyan"})

    def test_strength_zero_is_identity(self) -> None:
        L = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        a = np.array([0.08, 0.12, 0.05], dtype=np.float32)
        b = np.array([0.04, -0.06, 0.10], dtype=np.float32)
        for look in LOOK_FIELDS:
            L2, a2, b2 = apply_look_oklab(L, a, b, look, strength=0.0)
            np.testing.assert_allclose(L2, L, rtol=0, atol=1e-6)
            np.testing.assert_allclose(a2, a, rtol=0, atol=1e-6)
            np.testing.assert_allclose(b2, b, rtol=0, atol=1e-6)

    def test_luminance_unchanged(self) -> None:
        L = np.linspace(0.15, 0.85, 64, dtype=np.float32)
        a = np.full_like(L, 0.10)
        b = np.full_like(L, 0.05)
        for look in LOOK_FIELDS:
            L2, _, _ = apply_look_oklab(L, a, b, look, strength=1.0)
            np.testing.assert_allclose(L2, L, rtol=0, atol=1e-7)

    def test_hue_in_arc_peaks_at_center(self) -> None:
        field = LOOK_FIELDS["optic_warm_cyan"]
        hues = np.array([20.0, 30.0, 40.0, 50.0, 64.0], dtype=np.float32)
        w = _hue_in_arc(hues, field.skin_hue_lo, field.skin_hue_hi)
        np.testing.assert_allclose(w, [0.0, 1.0, 1.0, 1.0, 0.0], atol=1e-5)

    def test_green_sector_rotates_toward_cyan(self) -> None:
        field = LOOK_FIELDS["optic_warm_cyan"]
        self.assertLess(field.hue_rotation_deg[1], 0.0)
        L = np.array([0.55], dtype=np.float32)
        hue_deg = 45.0
        c = 0.10
        rad = np.radians(hue_deg)
        a = np.array([c * np.cos(rad)], dtype=np.float32)
        b = np.array([c * np.sin(rad)], dtype=np.float32)
        _, a2, b2 = apply_look_oklab(L, a, b, "optic_warm_cyan", 1.0)
        h_out = float(np.degrees(np.arctan2(b2.item(), a2.item())) % 360.0)
        delta = (h_out - hue_deg + 180.0) % 360.0 - 180.0
        self.assertLess(delta, -0.5)

    def test_optic_warm_cyan_cools_low_chroma_mids(self) -> None:
        L = np.array([0.45], dtype=np.float32)
        a = np.array([0.0], dtype=np.float32)
        b = np.array([0.0], dtype=np.float32)
        _, a2, b2 = apply_look_oklab(L, a, b, "optic_warm_cyan", 1.0)
        self.assertLess(a2.item(), -0.002)
        self.assertLess(b2.item(), -0.003)

    def test_optic_warm_cyan_warms_skin_arc(self) -> None:
        L = np.array([0.56], dtype=np.float32)
        hue_deg = 42.0
        c = 0.08
        rad = np.radians(hue_deg)
        a = np.array([c * np.cos(rad)], dtype=np.float32)
        b = np.array([c * np.sin(rad)], dtype=np.float32)
        _, a2, b2 = apply_look_oklab(L, a, b, "optic_warm_cyan", 1.0)
        self.assertGreater(a2.item(), a.item())
        self.assertGreater(b2.item(), b.item())

    def test_optic_warm_cyan_tames_non_skin_magenta(self) -> None:
        L = np.array([0.50], dtype=np.float32)
        hue_deg = 315.0
        c = 0.12
        rad = np.radians(hue_deg)
        a = np.array([c * np.cos(rad)], dtype=np.float32)
        b = np.array([c * np.sin(rad)], dtype=np.float32)
        _, a2, b2 = apply_look_oklab(L, a, b, "optic_warm_cyan", 1.0)
        c_out = float(np.hypot(a2.item(), b2.item()))
        self.assertLess(c_out, c * 0.95)


if __name__ == "__main__":
    unittest.main()


class PeriodicInterpPortabilityTests(unittest.TestCase):
    def test_nan_hue_is_nan_out_without_platform_defined_indexing(self) -> None:
        # float->int of NaN is platform-defined (INT32_MIN on x86_64 Linux, an
        # out-of-bounds index); the interp must route NaN explicitly. The
        # ubuntu CI leg used to be the gate; since the platform converged on
        # macOS this test is the only one pinning the contract.
        from dngscan.look import _periodic_interp

        hue = np.array([np.nan, np.inf, -np.inf, 45.0], dtype=np.float32)
        with np.errstate(all="ignore"):
            out = _periodic_interp(tuple(float(i) for i in range(12)), hue)
        self.assertTrue(np.all(np.isnan(out[:3])))
        self.assertTrue(np.isfinite(out[3]))
