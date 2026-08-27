# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end self-review (2026-08-27) — regression pins for the confirmed
P2/P3 fixes that no existing suite covered. The two P1 hot-WB items live in
tests/test_wb.py (CoreImageAlignedScaleTests, the rung rewrites)."""
from __future__ import annotations

import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from dngscan._deps import np
from dngscan import policy
from dngscan.agx import look_brightness_power
from dngscan.color import luminance_from_rec2020
from dngscan.film_develop import _compression_knee
from dngscan.film_optics import light_source
from dngscan.lum import apply_lum_core
from dngscan.models import ColorGeometryPlan, RenderAdjustments, RenderPlan, ToneCompressionPlan
from dngscan.priors import gain_e_per_dn
from dngscan.tone import apply_render_adjustments


def _tone(**overrides) -> ToneCompressionPlan:
    base = dict(
        target_gamut="Rec2020",
        luma_p1=0.01,
        luma_p50=0.18,
        luma_p99=1.0,
        luma_p999=2.0,
        black_ev=-8.0,
        white_ev=4.0,
        dynamic_range_ev=12.0,
        contrast=3.0,
        toe_power=1.5,
        shoulder_power=2.9,
        chroma_p95=0.0,
        negative_rgb_pct=0.0,
        over_rgb_pct=0.0,
        use_c1_endpoints=True,
    )
    base.update(overrides)
    return ToneCompressionPlan(**base)


def _render_plan(tone: ToneCompressionPlan) -> RenderPlan:
    color = ColorGeometryPlan(
        target_gamut="p3", raw_clip_retreat_strength=1.0, output_gamut_pressure_pct=0.0
    )
    return RenderPlan(tone=tone, color=color, scene=None)  # type: ignore[arg-type]


class BiasedSliderWindowTests(unittest.TestCase):
    """P2: the slider window must contain the compiled value, so the declared
    direction survives film presets whose powers sit outside the scene window."""

    def _toe_after(self, toe_power: float, bias: float) -> float:
        plan = _render_plan(_tone(toe_power=toe_power))
        out = apply_render_adjustments(plan, RenderAdjustments(shadow_transition=bias))
        return float(out.tone.toe_power)

    def test_direction_holds_outside_the_scene_window(self) -> None:
        compiled = 3.45  # a fitted film toe power, above the 2.5 window edge
        self.assertLess(self._toe_after(compiled, +1.0), compiled)
        self.assertGreater(self._toe_after(compiled, -1.0), compiled)
        # full +-0.45 stop span is live around an out-of-window compiled value
        self.assertAlmostEqual(self._toe_after(compiled, +1.0), compiled * 2.0 ** (-0.45), places=6)
        self.assertAlmostEqual(self._toe_after(compiled, -1.0), compiled * 2.0 ** (0.45), places=6)
        # an in-window value near the edge keeps the historical clamp
        self.assertAlmostEqual(self._toe_after(2.4, -1.0), 2.5, places=9)

    def test_direction_inside_the_window_is_unchanged(self) -> None:
        self.assertAlmostEqual(self._toe_after(1.5, +1.0), 1.5 * 2.0 ** (-0.45), places=6)
        self.assertAlmostEqual(self._toe_after(1.5, -1.0), 1.5 * 2.0 ** (0.45), places=6)

    def test_zero_bias_is_exact_identity(self) -> None:
        self.assertEqual(self._toe_after(3.45, 0.0), 3.45)


class LumViewBrightnessParityTests(unittest.TestCase):
    """P3: the lum core uses the same brightness->power map as the agx core."""

    def test_lum_core_uses_the_agx_brightness_map(self) -> None:
        grey = np.full((5, 3), 0.18, dtype=np.float32) * np.asarray(
            [[0.25], [0.5], [1.0], [2.0], [4.0]], dtype=np.float32
        )
        base = _tone(tone_core="lum", lum_norm="y")
        y_unit = luminance_from_rec2020(apply_lum_core(grey, base))
        for vb in (0.84, 1.2):
            y_vb = luminance_from_rec2020(apply_lum_core(grey, replace(base, view_brightness=vb)))
            expect = np.power(np.maximum(y_unit, 0.0), look_brightness_power(vb))
            np.testing.assert_allclose(y_vb, expect, rtol=2e-5, atol=1e-6)
        # the declared numbers: 18% grey at 0.84 darkens ~-0.22 EV, not -0.47
        y84 = luminance_from_rec2020(apply_lum_core(grey[2:3], replace(base, view_brightness=0.84)))
        shift = math.log2(float(y84[0]) / float(y_unit[2]))
        self.assertGreater(shift, -0.30)
        self.assertLess(shift, -0.12)


class NoiseFloorPolicyTests(unittest.TestCase):
    def test_noise_floor_offsets_are_registered_at_v5(self) -> None:
        names = {e.name for e in policy.ENTRIES}
        self.assertIn("BLACK_BELOW_NOISE_FLOOR_EV", names)
        self.assertIn("GATED_BELOW_NOISE_FLOOR_EV", names)
        self.assertGreaterEqual(policy.POLICY_VERSION, 5)


class FilmBoundaryTests(unittest.TestCase):
    def test_light_source_keeps_only_light(self) -> None:
        out = light_source([[-0.5, 0.2, 1.0]])
        np.testing.assert_array_equal(out, np.asarray([[0.0, 0.2, 1.0]], dtype=np.float32))
        self.assertEqual(out.dtype, np.float32)

    def test_zero_knee_is_legal(self) -> None:
        self.assertEqual(_compression_knee(SimpleNamespace(film_compression_knee=0.0)), 0.0)
        self.assertEqual(_compression_knee(SimpleNamespace(film_compression_knee=None)), 2.0)
        self.assertEqual(_compression_knee(SimpleNamespace()), 2.0)


class PaperIsotonicTests(unittest.TestCase):
    """Paper amount tables are [n_logE, n_layers]: each layer column must be
    monotone along the exposure axis for the B1 log2-exposure inversion."""

    def test_columns_become_monotone_and_monotone_columns_are_untouched(self) -> None:
        from tools.build_film_v2_assets import _isotonic_rows

        mono = np.asarray([[0.0, 2.0], [0.5, 1.5], [1.0, 1.0], [1.5, 0.5]], dtype=np.float64)
        np.testing.assert_array_equal(_isotonic_rows(mono), mono)
        wobble = np.asarray([[0.0, 1.0], [0.6, 0.9], [0.4, 0.8], [1.5, 2.0]], dtype=np.float64)
        fixed = _isotonic_rows(wobble)
        self.assertTrue(bool(np.all(np.diff(fixed, axis=0) >= 0.0)))
        np.testing.assert_allclose(fixed[:, 0], [0.0, 0.5, 0.5, 1.5], atol=1e-12)
        # pool-adjacent-violators preserves the column mean (L2 projection)
        np.testing.assert_allclose(fixed.mean(axis=0), wobble.mean(axis=0), atol=1e-12)


class PriorsBaseIsoTests(unittest.TestCase):
    def test_reciprocal_gain_floors_at_base_iso(self) -> None:
        priors = {"unity_gain_ev": 10.0, "base_iso": 100}
        self.assertEqual(gain_e_per_dn(priors, 50), gain_e_per_dn(priors, 100))
        self.assertAlmostEqual(gain_e_per_dn(priors, 200), gain_e_per_dn(priors, 100) / 2.0)

    def test_base_iso_falls_back_to_the_read_noise_curve_start(self) -> None:
        priors = {"unity_gain_ev": 10.0, "read_noise_log2iso_log2e": [[math.log2(100), 1.0], [math.log2(1600), 0.5]]}
        self.assertAlmostEqual(gain_e_per_dn(priors, 64), gain_e_per_dn(priors, 100), places=9)


class GuiPunchFailClosedTests(unittest.TestCase):
    def test_out_of_range_or_non_numeric_punch_raises(self) -> None:
        from dngscan.gui.service import parse_punch

        self.assertEqual(parse_punch({"punch": "1.2"}), 1.2)
        self.assertEqual(parse_punch({}), 1.0)
        for bad in (3, -0.1, "abc", float("nan"), None):
            with self.assertRaises(ValueError):
                parse_punch({"punch": bad})


class CliSentinelTests(unittest.TestCase):
    def test_defaults_resolve_when_no_combo(self) -> None:
        from dngscan.cli import parse_args

        args = parse_args(["photo.dng", "--jpeg", "out.jpg"])
        self.assertEqual(args.wb, "camera")
        self.assertEqual(args.scene_transform, "none")
        self.assertEqual(args.film_curve, "none")

    def test_explicit_flags_override_the_film_combo(self) -> None:
        from dngscan import cli

        combos = getattr(cli, "FILM_COMBOS", None) or getattr(cli, "FILM_PRESET_COMBOS", None)
        if not combos:
            self.skipTest("no film combo table exposed")
        name = next((k for k, v in combos.items() if str(v.get("scene_transform", "none")) != "none"), None)
        if name is None:
            self.skipTest("no combo declares a scene transform")
        filled = cli.parse_args(["photo.dng", "--jpeg", "out.jpg", "--film", name])
        self.assertEqual(filled.scene_transform, combos[name]["scene_transform"])
        forced = cli.parse_args(["photo.dng", "--jpeg", "out.jpg", "--film", name, "--scene-transform", "none"])
        self.assertEqual(forced.scene_transform, "none")


class OpticsFreezeCheckTests(unittest.TestCase):
    def test_check_fails_on_a_missing_fixture(self) -> None:
        import tempfile
        from pathlib import Path
        from unittest import mock

        from tools import regen_optics_freeze as rof

        with tempfile.TemporaryDirectory() as tmp:
            fake = SimpleNamespace(path=Path(tmp) / "missing.npz", stem="missing")
            with mock.patch.object(rof, "iter_cases", return_value=[fake]), mock.patch.object(
                rof, "render_case", return_value=(np.zeros((2, 2, 3), np.float32), np.zeros((2, 2, 3), np.uint8))
            ), mock.patch.object(rof, "FREEZE_DIR", Path(tmp)):
                self.assertEqual(rof.regen(check=True), 1)


if __name__ == "__main__":
    unittest.main()
