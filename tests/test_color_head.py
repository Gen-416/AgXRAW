# SPDX-License-Identifier: GPL-3.0-or-later
"""Enlarger colour head (Y/M CC filtration) — unit and formation-level tests.

The colour head is a declared physical control: real darkroom units (CC filter
density, detents of 5, 0-200), spectrally derived per-preset response fields
(tools/fit_film_curve.py), negative presets only (reversal film has no printing
stage). Direction contract, in darkroom mnemonic form: add the colour the print
has too much of — +Y removes yellow (raises displayed blue), +M removes magenta
(raises displayed green). Zero dials must be the byte-exact status quo.
"""
from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from dngscan import agx as agx_engine
from dngscan.film_curve import (
    FILM_CURVE_PRESETS,
    apply_film_curve_preset,
    color_head_gain_curves,
    color_head_supported,
    film_process,
    validate_color_head_cc,
)
from dngscan.models import ToneCompressionPlan


def _base_plan() -> ToneCompressionPlan:
    return ToneCompressionPlan(
        target_gamut="Rec2020",
        luma_p1=0.001,
        luma_p50=0.18,
        luma_p99=0.9,
        luma_p999=0.99,
        black_ev=-8.0,
        white_ev=4.0,
        dynamic_range_ev=12.0,
        contrast=2.0,
        toe_power=1.5,
        shoulder_power=2.0,
        chroma_p95=0.3,
        negative_rgb_pct=0.0,
        over_rgb_pct=0.0,
    )


def _film_plan(name: str = "portra400", **overrides) -> ToneCompressionPlan:
    plan = apply_film_curve_preset(_base_plan(), name)
    return dataclasses.replace(plan, **overrides) if overrides else plan


class TestProcessClassification(unittest.TestCase):
    REVERSALS = {"provia100f", "velvia100", "ektachrome100", "kodachrome64"}

    def test_every_preset_is_classified(self):
        for key in FILM_CURVE_PRESETS:
            self.assertIn(film_process(key), ("negative", "reversal"), key)

    def test_reversal_set_matches_physics(self):
        """E-6/K-14 stocks are reversal; C-41/ECN-2 print-through stocks negative."""
        for key in FILM_CURVE_PRESETS:
            expected = "reversal" if key in self.REVERSALS else "negative"
            self.assertEqual(film_process(key), expected, key)

    def test_color_head_field_negatives_only(self):
        for key in FILM_CURVE_PRESETS:
            self.assertEqual(
                color_head_supported(key), film_process(key) == "negative", key
            )

    def test_unknown_preset(self):
        self.assertIsNone(film_process("nope"))
        self.assertFalse(color_head_supported("nope"))


class TestDialValidation(unittest.TestCase):
    def test_valid_detents(self):
        for v in (0, 5, 30, 125, 200):
            self.assertEqual(validate_color_head_cc(v, "t"), float(v))

    def test_rejects_off_detent(self):
        for v in (7, 2.5, 199):
            with self.assertRaises(ValueError):
                validate_color_head_cc(v, "t")

    def test_rejects_out_of_range(self):
        for v in (-5, 205, float("nan"), float("inf"), "x", None):
            with self.assertRaises(ValueError):
                validate_color_head_cc(v, "t")


class TestGainCurves(unittest.TestCase):
    def _mid(self, name: str, y: float, m: float) -> list[float]:
        ev, g = color_head_gain_curves(name, y, m)
        return [float(np.interp(0.0, ev, g[:, c])) for c in range(3)]

    def test_zero_is_none(self):
        self.assertIsNone(color_head_gain_curves("portra400", 0.0, 0.0))

    def test_reversal_is_none(self):
        self.assertIsNone(color_head_gain_curves("velvia100", 30.0, 0.0))

    def test_direction_darkroom_mnemonic(self):
        """+Y raises displayed blue at mid (print loses yellow), +M raises green."""
        for key in FILM_CURVE_PRESETS:
            if film_process(key) != "negative":
                continue
            y_mid = self._mid(key, 30.0, 0.0)
            self.assertGreater(y_mid[2], 1.0, key)
            m_mid = self._mid(key, 0.0, 30.0)
            self.assertGreater(m_mid[1], 1.0, key)

    def test_retimed_mid_luminance_is_preserved(self):
        """The darkroom re-timing: filtration changes colour, not mid luminance."""
        # v2 fields are Rec.2020-basis; mid-luminance preservation is judged in
        # the calibration basis, not BT.709.
        luma = np.array([0.2627, 0.6780, 0.0593])
        for y, m in ((30.0, 0.0), (0.0, 60.0), (120.0, 0.0)):
            mid = np.array(self._mid("portra400", y, m))
            self.assertLess(abs(float(mid @ luma) - 1.0), 0.02, (y, m))

    def test_monotone_in_cc_including_interpolated_detents(self):
        """Strictly monotone through 120 CC; at 200 CC the paper's blue layer
        sits on its Dmin and the response honestly saturates (allowed to
        plateau within numerical tolerance, never to reverse materially)."""
        mids = [self._mid("portra400", cc, 0.0)[2] for cc in (5, 15, 30, 60, 120)]
        for lo, hi in zip(mids, mids[1:]):
            self.assertLess(lo, hi)
        self.assertGreater(mids[0], 1.0)
        m200 = self._mid("portra400", 200.0, 0.0)[2]
        self.assertGreater(m200, mids[-1] * 0.995)

    def test_separable_combination(self):
        """Y and M compose multiplicatively (the declared approximation).

        HONEST LIMIT (review finding): this asserts the implementation's own
        definition, not physical truth — the joint spectral re-time refutes
        separability at ~0.35 stop RMS on Portra 30Y+30M (0.44 on Superia).
        The real acceptance test is a joint Y x M solve as oracle; it arrives
        with the two-dimensional colour-head field (spectral rebuild stage 3).
        Until then this test only pins that the runtime matches the published
        approximation it claims to implement.
        """
        ev, g_y = color_head_gain_curves("portra400", 30.0, 0.0)
        _, g_m = color_head_gain_curves("portra400", 0.0, 30.0)
        _, g_ym = color_head_gain_curves("portra400", 30.0, 30.0)
        np.testing.assert_allclose(g_ym, g_y * g_m, rtol=1e-5)

    def test_magnitude_matches_cc_density_physics(self):
        """30 CC = one stop of the blue layer's PRINT exposure; through the
        paper's gamma that is materially more than one stop of displayed blue —
        the response must be well clear of a no-op and keep compounding toward
        the paper's saturation at the next stop."""
        m30 = self._mid("portra400", 30.0, 0.0)[2]
        m60 = self._mid("portra400", 60.0, 0.0)[2]
        self.assertGreater(m30, 1.5)
        self.assertGreater(m60 / m30, 1.3)


class TestFormation(unittest.TestCase):
    def _neutral(self) -> np.ndarray:
        return np.full((64, 3), 0.18, dtype=np.float32)

    def test_zero_dials_bitwise_identical(self):
        plan0 = _film_plan()
        plan_zero = _film_plan(color_head_y=0.0, color_head_m=0.0)
        inset, outset = agx_engine.formation_matrices(plan0)
        rgb = self._neutral()
        out0 = agx_engine.apply_core(rgb, plan0, inset, outset)
        out1 = agx_engine.apply_core(rgb, plan_zero, inset, outset)
        self.assertTrue(np.array_equal(out0, out1))

    def test_direction_at_render_formation_level(self):
        plan0 = _film_plan()
        inset, outset = agx_engine.formation_matrices(plan0)
        rgb = self._neutral()
        out0 = agx_engine.apply_core(rgb, plan0, inset, outset)
        out_y = agx_engine.apply_core(rgb, _film_plan(color_head_y=30.0), inset, outset)
        out_m = agx_engine.apply_core(rgb, _film_plan(color_head_m=30.0), inset, outset)
        # +Y de-yellows: blue rises relative to red+green.
        self.assertGreater(
            out_y[0, 2] / out_y[0, :2].mean(), out0[0, 2] / out0[0, :2].mean()
        )
        # +M de-magentas: green rises relative to red+blue.
        self.assertGreater(
            out_m[0, 1] / out_m[0, ::2].mean(), out0[0, 1] / out0[0, ::2].mean()
        )

    def test_full_mode_film_core_consumes_the_head(self):
        from dngscan.film_develop import apply_film_core

        rgb = self._neutral()
        out0 = apply_film_core(rgb, _film_plan(film_mode="full"))
        out_y = apply_film_core(rgb, _film_plan(film_mode="full", color_head_y=30.0))
        self.assertFalse(np.array_equal(out0, out_y))
        self.assertGreater(out_y[0, 2] / out_y[0, :2].mean(),
                           out0[0, 2] / out0[0, :2].mean())

    def test_native_kernel_excluded_only_when_active(self):
        from dngscan import _fast

        self.assertFalse(_fast.supports_agx(_film_plan(color_head_y=30.0)))
        self.assertFalse(
            _fast.supports_hdr_formation(_film_plan(color_head_m=5.0))
        )
        # Zero dials keep whatever the plain plan's dispatch was: the exclusion
        # clause itself must not trigger.
        plan = _film_plan()
        self.assertEqual(
            _fast.supports_agx(plan),
            _fast.supports_agx(dataclasses.replace(plan, color_head_y=0.0)),
        )


class TestPlanGuards(unittest.TestCase):
    def test_reversal_preset_rejects_head_in_gui_parser(self):
        from dngscan.gui.service import parse_film_params

        with self.assertRaises(ValueError):
            parse_film_params({"filmCurve": "velvia100", "colorHeadY": 30})
        with self.assertRaises(ValueError):
            parse_film_params({"filmCurve": "none", "colorHeadM": 5})
        lens, curve, _mode, _xover, y, m = parse_film_params(
            {"filmCurve": "portra400", "colorHeadY": 30, "colorHeadM": 5}
        )
        self.assertEqual((curve, y, m), ("portra400", 30.0, 5.0))
        # Reversal preset with zero dials stays valid — the control is absent,
        # not the preset.
        lens, curve, _mode, _xover, y, m = parse_film_params({"filmCurve": "velvia100"})
        self.assertEqual((curve, y, m), ("velvia100", 0.0, 0.0))

    def test_gui_parser_rejects_off_detent(self):
        from dngscan.gui.service import parse_film_params

        with self.assertRaises(ValueError):
            parse_film_params({"filmCurve": "portra400", "colorHeadY": 7})


if __name__ == "__main__":
    unittest.main()
