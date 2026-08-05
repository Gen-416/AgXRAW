# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import unittest

from dngscan._deps import np
from dngscan.color import rgb_to_oklab
from dngscan.drt import apply_c1_endpoints
from dngscan.gui.service import parse_render_adjustments
from dngscan.models import (
    ColorGeometryPlan, RenderAdjustments, RenderPlan, ToneCompressionPlan,
)
from dngscan.render import _apply_display_highlight_chroma_retreat
from dngscan.tone import apply_render_adjustments


def _plan(core: str = "agx") -> RenderPlan:
    tone = ToneCompressionPlan(
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
        tone_core=core,
        pivot_ev_offset=-1.25,
        view_brightness=1.1,
    )
    color = ColorGeometryPlan(
        target_gamut="p3",
        raw_clip_retreat_strength=1.0,
        output_gamut_pressure_pct=0.0,
        display_highlight_chroma_retreat=0.0,
    )
    return RenderPlan(tone=tone, color=color, scene=None)  # type: ignore[arg-type]


class RenderAdjustmentPlanTest(unittest.TestCase):
    def test_zero_adjustments_are_exact_identity(self) -> None:
        plan = _plan()
        self.assertIs(apply_render_adjustments(plan, None), plan)
        self.assertIs(apply_render_adjustments(plan, RenderAdjustments()), plan)

    def test_tone_controls_preserve_automatic_anchors(self) -> None:
        plan = _plan()
        adjusted = apply_render_adjustments(
            plan,
            RenderAdjustments(
                midtone_brightness=1.0,
                midtone_contrast=1.0,
                shadow_transition=1.0,
                highlight_transition=1.0,
            ),
        )
        self.assertEqual(adjusted.tone.black_ev, plan.tone.black_ev)
        self.assertEqual(adjusted.tone.white_ev, plan.tone.white_ev)
        self.assertEqual(adjusted.tone.pivot_ev_offset, plan.tone.pivot_ev_offset)
        self.assertGreater(adjusted.tone.view_brightness, plan.tone.view_brightness)
        self.assertGreater(adjusted.tone.contrast, plan.tone.contrast)
        self.assertLess(adjusted.tone.toe_power, plan.tone.toe_power)
        self.assertLess(adjusted.tone.shoulder_power, plan.tone.shoulder_power)

    def test_highlight_fade_is_color_only(self) -> None:
        plan = _plan()
        adjusted = apply_render_adjustments(
            plan, RenderAdjustments(highlight_fade=1.0)
        )
        self.assertEqual(adjusted.tone, plan.tone)
        self.assertGreater(
            adjusted.color.display_highlight_chroma_retreat,
            plan.color.display_highlight_chroma_retreat,
        )
        self.assertLess(
            adjusted.color.display_highlight_chroma_start,
            plan.color.display_highlight_chroma_start,
        )

    def test_neutral_reference_ignores_adjustments(self) -> None:
        plan = _plan("neutral")
        self.assertIs(
            apply_render_adjustments(
                plan,
                RenderAdjustments(
                    midtone_brightness=1.0,
                    midtone_contrast=1.0,
                    shadow_transition=1.0,
                    highlight_transition=1.0,
                    highlight_fade=1.0,
                ),
            ),
            plan,
        )


def _c1_plan(
    black_ev: float, white_ev: float, toe_power: float, shoulder_power: float,
    latitude_hi_ev: float,
) -> RenderPlan:
    """A production-shaped plan (fixed 2.2 gamma path, pivot at 0) for curve renders."""
    tone = ToneCompressionPlan(
        target_gamut="Rec2020",
        luma_p1=0.01,
        luma_p50=0.18,
        luma_p99=1.0,
        luma_p999=2.0,
        black_ev=black_ev,
        white_ev=white_ev,
        dynamic_range_ev=white_ev - black_ev,
        contrast=3.0,
        toe_power=toe_power,
        shoulder_power=shoulder_power,
        latitude_lo_ev=0.10,
        latitude_hi_ev=latitude_hi_ev,
        chroma_p95=0.0,
        negative_rgb_pct=0.0,
        over_rgb_pct=0.0,
        tone_core="agx",
        pivot_ev_offset=0.0,
        view_brightness=1.0,
        use_c1_endpoints=True,
    )
    color = ColorGeometryPlan(
        target_gamut="srgb",
        raw_clip_retreat_strength=1.0,
        output_gamut_pressure_pct=0.0,
    )
    return RenderPlan(tone=tone, color=color, scene=None)  # type: ignore[arg-type]


def _curve_code_values(tone: ToneCompressionPlan, ev: np.ndarray) -> np.ndarray:
    linear = np.clip(apply_c1_endpoints(ev, tone), 0.0, 1.0)
    encoded = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return encoded * 255.0


class TransitionSliderMagnitudeTest(unittest.TestCase):
    """Pin the documented full-swing magnitudes of the transition trims.

    EDITING_TUTORIAL / USER_GUIDE state these sliders are deliberately restrained:
    full swing peaks at a few code values, confined to the toe/shoulder regions, and
    the shoulder trim loses all room on plans whose shoulder is slope-pinned. If the
    bias rate is ever re-tuned, these bounds fail and flag the docs for update.
    """

    def _delta(self, plan: RenderPlan, adjustments: RenderAdjustments, ev: np.ndarray) -> np.ndarray:
        adjusted = apply_render_adjustments(plan, adjustments)
        return _curve_code_values(adjusted.tone, ev) - _curve_code_values(plan.tone, ev)

    def test_shadow_transition_full_swing_is_a_restrained_shadow_trim(self) -> None:
        plan = _c1_plan(-8.0, 4.0, 1.5, 2.55, latitude_hi_ev=0.0)
        ev = np.linspace(-8.0, 4.0, 2001, dtype=np.float32)
        for bias in (-1.0, 1.0):
            delta = self._delta(plan, RenderAdjustments(shadow_transition=bias), ev)
            peak = float(np.max(np.abs(delta)))
            self.assertGreater(peak, 2.0)
            self.assertLess(peak, 12.0)
            # The trim reshapes the toe only: at and above mid gray nothing moves.
            self.assertLess(float(np.max(np.abs(delta[ev >= 0.0]))), 0.5)

    def test_highlight_transition_full_swing_peaks_at_a_few_code_values(self) -> None:
        plan = _c1_plan(-8.0, 4.0, 1.5, 2.55, latitude_hi_ev=0.0)
        ev = np.linspace(-8.0, 4.0, 2001, dtype=np.float32)
        for bias in (-1.0, 1.0):
            delta = self._delta(plan, RenderAdjustments(highlight_transition=bias), ev)
            peak = float(np.max(np.abs(delta)))
            self.assertGreater(peak, 1.0)
            self.assertLess(peak, 6.0)
            # Shoulder-only: below mid gray nothing moves.
            self.assertLess(float(np.max(np.abs(delta[ev <= 0.0]))), 0.5)

    def test_highlight_transition_is_flat_when_the_shoulder_is_slope_pinned(self) -> None:
        # DR 10 with the normal 0.2 EV upper latitude: the shoulder segment's
        # required average slope nearly equals the linear slope, leaving the
        # sigmoid no curvature freedom — the documented "此路不通" case.
        plan = _c1_plan(-7.0, 3.0, 1.3, 2.90, latitude_hi_ev=0.20)
        ev = np.linspace(-7.0, 3.0, 2001, dtype=np.float32)
        for bias in (-1.0, 1.0):
            delta = self._delta(plan, RenderAdjustments(highlight_transition=bias), ev)
            self.assertLess(float(np.max(np.abs(delta))), 1.0)


class HighlightFadeTest(unittest.TestCase):
    def test_signed_control_changes_chroma_without_moving_lightness(self) -> None:
        rgb = np.asarray([[1.0, 0.68, 0.22]], dtype=np.float32)
        faded = _apply_display_highlight_chroma_retreat(rgb, "p3", 0.3)
        retained = _apply_display_highlight_chroma_retreat(rgb, "p3", -0.3)
        before = rgb_to_oklab(rgb, "p3")
        after_fade = rgb_to_oklab(faded, "p3")
        after_retain = rgb_to_oklab(retained, "p3")
        c0 = float(np.hypot(before[1][0], before[2][0]))
        c_fade = float(np.hypot(after_fade[1][0], after_fade[2][0]))
        c_retain = float(np.hypot(after_retain[1][0], after_retain[2][0]))
        self.assertAlmostEqual(float(before[0][0]), float(after_fade[0][0]), places=5)
        self.assertAlmostEqual(float(before[0][0]), float(after_retain[0][0]), places=5)
        self.assertLess(c_fade, c0)
        self.assertGreater(c_retain, c0)

    def test_neutral_highlights_are_untouched(self) -> None:
        # The docs promise the fade slider is invisible on achromatic content:
        # a neutral ramp through the retreat must come back unchanged.
        gray = np.repeat(
            np.linspace(0.05, 1.0, 32, dtype=np.float32)[:, None], 3, axis=1
        )
        for strength, start in ((0.30, 0.63), (-0.30, 0.87)):
            out = _apply_display_highlight_chroma_retreat(gray, "srgb", strength, start, 0.98)
            self.assertLess(float(np.max(np.abs(out - gray))), 1e-3)


class RenderAdjustmentParserTest(unittest.TestCase):
    def test_gui_names_parse_to_model(self) -> None:
        parsed = parse_render_adjustments(
            {
                "midtoneBrightness": 0.25,
                "midtoneContrast": -0.1,
                "shadowTransition": 0.4,
                "highlightTransition": 0.5,
                "highlightFade": -0.3,
            }
        )
        self.assertEqual(parsed.midtone_brightness, 0.25)
        self.assertEqual(parsed.highlight_fade, -0.3)

    def test_out_of_range_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_render_adjustments({"highlightFade": 1.01})


if __name__ == "__main__":
    unittest.main()
