# SPDX-License-Identifier: GPL-3.0-or-later
"""Appearance E3 gates: the native palette kernel (plan §16 P6).

The NumPy kernel is the correctness ORACLE; the C++ port must match it
elementwise. What this file pins:

- parity across every shipped recipe (all five assets), custom deltas,
  non-default strength, and a synthetic neutral-bias table — on the palette
  volume AND a seeded random HDR-ish volume;
- the clamp counters agree between paths;
- the strict/off dispatch contract (DNGSCAN_FAST=0 must never touch the
  extension; =1 must refuse to run without it).

Tolerance: the two paths reorder float32 arithmetic (fused scalar ops vs
NumPy temporaries; cbrt/atan2 differ in the last ulps) and the x^3 opponent
reconstruction amplifies input noise threefold, so bytes differ. Measured
max |Δ| 1.7e-5 on a 0..4 HDR volume; the gate is 5e-5 in linear units —
below one 16-bit step at the 4.0 range top and ~1/3600 of mid grey.
"""
from __future__ import annotations

import unittest

import numpy as np

from dngscan import film_appearance as fa
from dngscan import film_palette_diag as pal

RECIPES = (
    ("portra400", "kodak_portra_endura__translated", "reference"),
    ("ektar100", "kodak_portra_endura__translated", "reference"),
    ("vision3250d", "kodak_2383__translated", "reference"),
    ("vision3250d", "kodak_2383__translated", "extended"),
    ("velvia100", "direct__velvia100", "reference"),
)


def _native_available() -> bool:
    from dngscan import _fast

    return _fast._load_extension() is not None


def _volumes():
    vol, _ = pal.palette_volume()
    rng = np.random.default_rng(416)
    rand = rng.uniform(0.0, 4.0, size=(50_000, 3)).astype(np.float32)
    return [np.asarray(vol, np.float32).reshape(-1, 3), rand]


def _both_paths(rgb, plan):
    """Run the oracle and the native kernel on identical inputs."""
    e = np.log2(
        np.maximum(rgb @ np.array([0.2627, 0.6780, 0.0593], np.float32),
                   np.float32(1e-9)) / np.float32(0.18)
    ).astype(np.float32)
    native = fa._native_apply(rgb, e, plan)
    if native is None:
        raise unittest.SkipTest("native extension unavailable")
    # Oracle: force the NumPy path by patching availability off.
    from dngscan import _fast
    import unittest.mock as mock
    with mock.patch.object(_fast, "available", return_value=False), \
         mock.patch.object(_fast, "strict_requested", return_value=False):
        oracle = fa.apply_film_appearance(rgb, plan, e)
    return np.asarray(oracle, np.float64), np.asarray(native, np.float64)


class ParityTests(unittest.TestCase):
    def test_every_shipped_recipe_matches_the_oracle(self) -> None:
        if not _native_available():
            self.skipTest("native extension unavailable")
        for stock, medium, variant in RECIPES:
            plan = fa.compile_appearance_plan(
                "reference", 1.0, stock_id=stock, medium_id=medium,
                variant=variant,
            )
            for vi, rgb in enumerate(_volumes()):
                with self.subTest(stock=stock, variant=variant, volume=vi):
                    oracle, native = _both_paths(rgb, plan)
                    self.assertLessEqual(
                        float(np.abs(oracle - native).max()), 5e-5
                    )

    def test_custom_deltas_strength_and_neutral_bias_match(self) -> None:
        if not _native_available():
            self.skipTest("native extension unavailable")
        plan = fa.compile_appearance_plan(
            "custom", 1.35, stock_id="ektar100",
            medium_id="kodak_portra_endura__translated",
            richness_delta=0.4, color_density_delta=-0.25,
            neutral_bias_strength=1.5,
        )
        # graft a nonzero neutral-bias table so that branch is exercised
        recipe = dict(plan.recipe)
        nb = np.zeros((len(fa.EV_KNOTS), 2), np.float32)
        nb[1:4, 0] = 0.004
        nb[1:4, 1] = -0.003
        recipe["neutral_bias_ab"] = nb
        import dataclasses
        plan = dataclasses.replace(plan, recipe=recipe)
        for vi, rgb in enumerate(_volumes()):
            with self.subTest(volume=vi):
                oracle, native = _both_paths(rgb, plan)
                self.assertLessEqual(
                    float(np.abs(oracle - native).max()), 5e-5
                )

    def test_clamp_counters_agree(self) -> None:
        if not _native_available():
            self.skipTest("native extension unavailable")
        plan = fa.compile_appearance_plan(
            "reference", 1.5, stock_id="ektar100",
            medium_id="kodak_portra_endura__translated",
        )
        rgb = _volumes()[1]
        e = np.zeros(rgb.shape[0], np.float32)
        from dngscan import _fast
        import unittest.mock as mock
        fa.clamp_stats_reset()
        with mock.patch.object(_fast, "available", return_value=False), \
             mock.patch.object(_fast, "strict_requested", return_value=False):
            fa.apply_film_appearance(rgb, plan, e)
        rows_py, neg_py = fa.clamp_stats()
        fa.clamp_stats_reset()
        if fa._native_apply(rgb, e, plan) is None:
            self.skipTest("native path disabled (DNGSCAN_FAST=0)")
        rows_nat, neg_nat = fa.clamp_stats()
        self.assertEqual(rows_py, rows_nat)
        self.assertEqual(neg_py, neg_nat)


class DispatchContractTests(unittest.TestCase):
    def test_fast_off_never_touches_the_extension(self) -> None:
        import os
        import unittest.mock as mock
        with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "0"}):
            plan = fa.compile_appearance_plan(
                "reference", 1.0, stock_id="portra400",
                medium_id="kodak_portra_endura__translated",
            )
            rgb = np.full((16, 3), 0.2, np.float32)
            self.assertIsNone(
                fa._native_apply(rgb, np.zeros(16, np.float32), plan)
            )

    def test_fast_strict_requires_the_extension(self) -> None:
        import os
        import unittest.mock as mock
        from dngscan import _fast

        plan = fa.compile_appearance_plan(
            "reference", 1.0, stock_id="portra400",
            medium_id="kodak_portra_endura__translated",
        )
        rgb = np.full((16, 3), 0.2, np.float32)
        with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "1"}), \
             mock.patch.object(_fast, "_load_extension", return_value=None):
            with self.assertRaises(_fast.NativeKernelError):
                fa._native_apply(rgb, np.zeros(16, np.float32), plan)


if __name__ == "__main__":
    unittest.main()
