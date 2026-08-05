# SPDX-License-Identifier: GPL-3.0-or-later
"""Exposure & range phase 1: evidence endpoints and bounded toe/shoulder offsets.

Covers the three contracts this feature adds:

1. `endpoint_mode="evidence"` compiles the black endpoint from the measured noise
   floor (prior read-noise when a sensor prior exists, single-frame estimate
   otherwise) and the white endpoint from the reliable RAW tail only, with truthful
   degradation notes when evidence is absent.
2. `toe_end_offset` / `shoulder_white_offset` are bounded plan-level adjustments:
   they move the compiled toe-end / shoulder-white crossings (by re-solving
   toe_power / shoulder_power) without moving the black/white endpoints, and every
   request — including out-of-range ones — still compiles to a monotone curve
   inside the display range.
3. CLI flags, GUI service parsing, cache keys and the page contract carry the new
   parameters end to end.
"""
from __future__ import annotations

import contextlib
import io
import math
import unittest
from dataclasses import replace

from dngscan._deps import np
from dngscan.analysis import noise_floor_ev_estimate
from dngscan.constants import MIDGRAY_HEADROOM_STOPS
from dngscan.drt import (
    SHOULDER_POWER_SOLVE_MAX,
    SHOULDER_POWER_SOLVE_MIN,
    TOE_POWER_SOLVE_MAX,
    TOE_POWER_SOLVE_MIN,
    apply_c1_endpoints,
    compiled_curve_transitions,
)
from dngscan.gui.service import (
    _adjustment_key,
    parse_endpoint_mode,
    parse_render_adjustments,
)
from dngscan.models import (
    ColorGeometryPlan, RenderAdjustments, RenderPlan, ToneCompressionPlan,
)
from dngscan.tone import apply_render_adjustments, build_tone_compression_plan

from tests.golden_support import _analysis_for, _bundle_from_scene, _scene_metrics


def _scene_bundle():
    scene = np.full((32, 32, 3), 6553, dtype=np.uint16)
    return _bundle_from_scene(scene)


def _tone_plan(**overrides) -> ToneCompressionPlan:
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
        latitude_lo_ev=0.1,
        latitude_hi_ev=0.2,
        toe_start_ev=-0.1,
        shoulder_start_ev=0.2,
        use_c1_endpoints=True,
    )
    base.update(overrides)
    return ToneCompressionPlan(**base)


def _render_plan(tone: ToneCompressionPlan) -> RenderPlan:
    color = ColorGeometryPlan(
        target_gamut="p3",
        raw_clip_retreat_strength=1.0,
        output_gamut_pressure_pct=0.0,
    )
    return RenderPlan(tone=tone, color=color, scene=None)  # type: ignore[arg-type]


def _assert_monotone(test: unittest.TestCase, tone: ToneCompressionPlan) -> None:
    ev = np.linspace(tone.black_ev - 1.0, tone.white_ev + 1.0, 512, dtype=np.float32)
    out = apply_c1_endpoints(ev, tone)
    test.assertTrue(bool(np.all(np.diff(out) >= -1e-6)), "curve must stay monotone")
    test.assertTrue(bool(np.all(out >= -1e-6)))
    test.assertTrue(bool(np.all(out <= float(tone.target_white_linear) + 1e-4)))


class NoiseFloorEvEstimateTests(unittest.TestCase):
    def test_prior_read_noise_floor(self) -> None:
        analysis = _analysis_for(
            median_vs_gray_ev=0.0, ev_median=0.0, ev_p99=2.0, ev_p999=3.0,
            usable_dr_ev=9.0,
        )
        # fullwell_e = noise_floor_e / noise_floor = 2.0 / 0.002 = 1000 e-
        expected = MIDGRAY_HEADROOM_STOPS + math.log2(3.0 / 1000.0)
        value, source = noise_floor_ev_estimate(analysis)
        self.assertEqual(source, "prior")
        self.assertAlmostEqual(value, expected, places=9)

    def test_single_frame_fallback_without_prior(self) -> None:
        analysis = replace(
            _analysis_for(
                median_vs_gray_ev=0.0, ev_median=0.0, ev_p99=2.0, ev_p999=3.0,
                usable_dr_ev=9.0,
            ),
            noise_floor_e=None,
            prior_read_noise_e=None,
        )
        value, source = noise_floor_ev_estimate(analysis)
        self.assertEqual(source, "frame")
        self.assertAlmostEqual(value, MIDGRAY_HEADROOM_STOPS - 9.0, places=9)

    def test_no_estimate_at_all(self) -> None:
        analysis = replace(
            _analysis_for(
                median_vs_gray_ev=0.0, ev_median=0.0, ev_p99=2.0, ev_p999=3.0,
                usable_dr_ev=9.0,
            ),
            noise_floor_e=None,
            prior_read_noise_e=None,
            usable_dr_ev=float("nan"),
        )
        value, source = noise_floor_ev_estimate(analysis)
        self.assertEqual(source, "none")
        self.assertTrue(math.isnan(value))


class EvidenceEndpointCompileTests(unittest.TestCase):
    def _compile(self, analysis, metrics, endpoint_mode):
        return build_tone_compression_plan(
            _scene_bundle(),
            analysis,
            "Rec2020",
            endpoint_mode=endpoint_mode,
            scene_metrics=metrics,
        )

    def test_default_stays_adaptive_with_no_note(self) -> None:
        analysis = _analysis_for(
            median_vs_gray_ev=0.0, ev_median=0.0, ev_p99=2.0, ev_p999=3.0,
            usable_dr_ev=9.0,
        )
        metrics = _scene_metrics(-1.0, tail_ev_p9999=2.0)
        plan = build_tone_compression_plan(
            _scene_bundle(), analysis, "Rec2020", scene_metrics=metrics
        )
        self.assertEqual(plan.endpoint_mode, "adaptive")
        self.assertIsNone(plan.endpoint_note)

    def test_evidence_black_uses_prior_read_noise_floor(self) -> None:
        analysis = _analysis_for(
            median_vs_gray_ev=0.0, ev_median=0.0, ev_p99=2.0, ev_p999=3.0,
            usable_dr_ev=9.0,
        )
        metrics = _scene_metrics(-1.0, tail_ev_p9999=2.0)
        adaptive = self._compile(analysis, metrics, "adaptive")
        evidence = self._compile(analysis, metrics, "evidence")
        expected_black = MIDGRAY_HEADROOM_STOPS + math.log2(3.0 / 1000.0)
        self.assertAlmostEqual(evidence.black_ev, expected_black, places=6)
        self.assertNotAlmostEqual(evidence.black_ev, adaptive.black_ev, places=2)
        self.assertEqual(evidence.endpoint_mode, "evidence")
        self.assertIn("先验读出噪声底", evidence.endpoint_note)
        self.assertAlmostEqual(
            evidence.dynamic_range_ev, evidence.white_ev - evidence.black_ev, places=6
        )

    def test_evidence_black_single_frame_note_without_prior(self) -> None:
        analysis = replace(
            _analysis_for(
                median_vs_gray_ev=0.0, ev_median=0.0, ev_p99=2.0, ev_p999=3.0,
                usable_dr_ev=9.0,
            ),
            noise_floor_e=None,
            prior_read_noise_e=None,
        )
        metrics = _scene_metrics(-1.0, tail_ev_p9999=2.0)
        evidence = self._compile(analysis, metrics, "evidence")
        self.assertAlmostEqual(
            evidence.black_ev, MIDGRAY_HEADROOM_STOPS - 9.0, places=6
        )
        self.assertIn("单帧噪声底估计", evidence.endpoint_note)

    def test_evidence_black_degrades_truthfully_without_any_estimate(self) -> None:
        analysis = replace(
            _analysis_for(
                median_vs_gray_ev=0.0, ev_median=0.0, ev_p99=2.0, ev_p999=3.0,
                usable_dr_ev=9.0,
            ),
            noise_floor_e=None,
            prior_read_noise_e=None,
            usable_dr_ev=float("nan"),
            usable_dr_eff_ev=float("nan"),
        )
        metrics = _scene_metrics(-1.0, tail_ev_p9999=2.0)
        adaptive = self._compile(analysis, metrics, "adaptive")
        evidence = self._compile(analysis, metrics, "evidence")
        self.assertEqual(evidence.black_ev, adaptive.black_ev)
        self.assertIn("黑端点证据缺席", evidence.endpoint_note)

    def test_evidence_white_follows_reliable_tail(self) -> None:
        analysis = _analysis_for(
            median_vs_gray_ev=0.0, ev_median=0.0, ev_p99=2.0, ev_p999=3.0,
            usable_dr_ev=9.0,
        )
        metrics = _scene_metrics(-1.0, tail_ev_p9999=4.4)
        evidence = self._compile(analysis, metrics, "evidence")
        # non-sparse margin 0.30, minimum white +3.00
        self.assertAlmostEqual(evidence.white_ev, 4.7, places=6)
        self.assertIn("白端点=可靠尾部", evidence.endpoint_note)

    def test_evidence_white_falls_back_when_reliable_tail_missing(self) -> None:
        analysis = _analysis_for(
            median_vs_gray_ev=0.0, ev_median=0.0, ev_p99=2.0, ev_p999=3.0,
            usable_dr_ev=9.0,
        )
        metrics = replace(
            _scene_metrics(-1.0, tail_ev_p9999=6.0),
            reliable_tail_ev_p9999=float("nan"),
        )
        adaptive = self._compile(analysis, metrics, "adaptive")
        evidence = self._compile(analysis, metrics, "evidence")
        self.assertEqual(evidence.white_ev, adaptive.white_ev)
        self.assertIn("白端点证据缺席", evidence.endpoint_note)

    def test_evidence_endpoints_pass_curve_legality(self) -> None:
        analysis = _analysis_for(
            median_vs_gray_ev=0.0, ev_median=0.0, ev_p99=2.0, ev_p999=3.0,
            usable_dr_ev=12.0,
        )
        metrics = _scene_metrics(-1.0, tail_ev_p9999=4.0)
        evidence = self._compile(analysis, metrics, "evidence")
        _assert_monotone(self, evidence)


class ToeEndOffsetTests(unittest.TestCase):
    def test_zero_offsets_are_exact_identity(self) -> None:
        plan = _render_plan(_tone_plan())
        self.assertTrue(RenderAdjustments().is_identity())
        self.assertIs(apply_render_adjustments(plan, RenderAdjustments()), plan)
        self.assertFalse(RenderAdjustments(toe_end_offset=-1.0).is_identity())
        self.assertFalse(RenderAdjustments(shoulder_white_offset=1.0).is_identity())

    def test_negative_offset_moves_toe_end_deeper_and_lifts_shadows(self) -> None:
        plan = _render_plan(_tone_plan())
        base_end = compiled_curve_transitions(plan.tone)["toe_end_ev"]
        adjusted = apply_render_adjustments(
            plan, RenderAdjustments(toe_end_offset=-1.5)
        )
        new_end = compiled_curve_transitions(adjusted.tone)["toe_end_ev"]
        self.assertLess(new_end, base_end - 0.5)
        self.assertAlmostEqual(new_end, base_end - 1.5, delta=0.15)
        # endpoints and shoulder side must not move
        self.assertEqual(adjusted.tone.black_ev, plan.tone.black_ev)
        self.assertEqual(adjusted.tone.white_ev, plan.tone.white_ev)
        self.assertEqual(adjusted.tone.latitude_hi_ev, plan.tone.latitude_hi_ev)
        # deep shadows lift, upper mids/highlights stay put
        probe_deep = float(apply_c1_endpoints(np.asarray([-4.0]), adjusted.tone)[0])
        base_deep = float(apply_c1_endpoints(np.asarray([-4.0]), plan.tone)[0])
        self.assertGreater(probe_deep, base_deep)
        probe_high = float(apply_c1_endpoints(np.asarray([2.0]), adjusted.tone)[0])
        base_high = float(apply_c1_endpoints(np.asarray([2.0]), plan.tone)[0])
        self.assertAlmostEqual(probe_high, base_high, places=5)
        _assert_monotone(self, adjusted.tone)

    def test_positive_offset_tightens_the_toe(self) -> None:
        plan = _render_plan(_tone_plan())
        base_end = compiled_curve_transitions(plan.tone)["toe_end_ev"]
        adjusted = apply_render_adjustments(
            plan, RenderAdjustments(toe_end_offset=0.5)
        )
        new_end = compiled_curve_transitions(adjusted.tone)["toe_end_ev"]
        self.assertGreaterEqual(new_end, base_end - 0.02)
        _assert_monotone(self, adjusted.tone)

    def test_out_of_reach_request_clamps_to_legal_toe_power(self) -> None:
        plan = _render_plan(_tone_plan(black_ev=-4.0, dynamic_range_ev=8.0))
        adjusted = apply_render_adjustments(
            plan, RenderAdjustments(toe_end_offset=-3.0)
        )
        self.assertGreaterEqual(adjusted.tone.toe_power, TOE_POWER_SOLVE_MIN - 1e-9)
        self.assertLessEqual(adjusted.tone.toe_power, TOE_POWER_SOLVE_MAX + 1e-9)
        _assert_monotone(self, adjusted.tone)

    def test_out_of_range_value_is_clamped_not_amplified(self) -> None:
        plan = _render_plan(_tone_plan())
        wild = apply_render_adjustments(plan, RenderAdjustments(toe_end_offset=-99.0))
        bounded = apply_render_adjustments(plan, RenderAdjustments(toe_end_offset=-3.0))
        self.assertAlmostEqual(wild.tone.toe_power, bounded.tone.toe_power, places=6)


class LiftedBlackToeEndTests(unittest.TestCase):
    """Floor-relative toe-end semantics on lifted-black (film paper Dmax) plans.

    The near-black reference is TOE_END_DISPLAY_LINEAR ABOVE the compiled
    ``target_black_linear`` floor: identical to the absolute 0.002 level for
    zero-floor plans, and the only definition under which lifted-floor plans keep a
    real, monotone measurement. The old code returned the black endpoint as a
    sentinel and the solver mistook it for a crossing, so every offset — either
    sign — solved to the hardest legal toe (3.5), the reverse of the declared
    control direction.
    """

    PRESETS = ("portra400", "kodachrome64", "vision3250d_theatrical")
    OFFSETS = (-3.0, -1.0, -0.05, 0.0, 0.5)

    @staticmethod
    def _film_plan(preset: str) -> RenderPlan:
        from dngscan.film_curve import apply_film_curve_preset

        return _render_plan(apply_film_curve_preset(_tone_plan(), preset))

    def test_offsets_keep_a_monotone_gradient_with_the_declared_direction(self) -> None:
        for preset in self.PRESETS:
            with self.subTest(preset=preset):
                plan = self._film_plan(preset)
                base_power = float(plan.tone.toe_power)
                base_end = compiled_curve_transitions(plan.tone)["toe_end_ev"]
                self.assertIsNotNone(base_end)
                powers, ends = [], []
                for offset in self.OFFSETS:
                    adjusted = apply_render_adjustments(
                        plan, RenderAdjustments(toe_end_offset=offset)
                    )
                    end = compiled_curve_transitions(adjusted.tone)["toe_end_ev"]
                    self.assertIsNotNone(end)
                    powers.append(float(adjusted.tone.toe_power))
                    ends.append(float(end))
                    if offset < 0.0:
                        # More open (or clamped open), never harder than base.
                        self.assertLessEqual(adjusted.tone.toe_power, base_power + 1e-9)
                        self.assertLessEqual(end, base_end + 1e-9)
                    elif offset > 0.0:
                        self.assertGreaterEqual(adjusted.tone.toe_power, base_power - 1e-9)
                        self.assertGreaterEqual(end, base_end - 1e-9)
                for a, b in zip(powers, powers[1:]):
                    self.assertLessEqual(a, b + 1e-9)
                for a, b in zip(ends, ends[1:]):
                    self.assertLessEqual(a, b + 1e-9)

    def test_lifted_floor_toe_end_is_a_real_crossing_not_the_black_endpoint(self) -> None:
        from dngscan.drt import TOE_END_DISPLAY_LINEAR, _value_at_ev, curve_params_from_plan

        for preset in ("portra400", "kodachrome64"):
            with self.subTest(preset=preset):
                tone = self._film_plan(preset).tone
                end = compiled_curve_transitions(tone)["toe_end_ev"]
                self.assertIsNotNone(end)
                self.assertGreater(end, float(tone.black_ev) + 0.05)
                params = curve_params_from_plan(tone)
                floor = float(params["target_black"]) ** float(params["gamma"])
                self.assertAlmostEqual(
                    _value_at_ev(float(end), params),
                    floor + TOE_END_DISPLAY_LINEAR,
                    places=4,
                )

    def test_zero_floor_measurement_is_unchanged_absolute_reference(self) -> None:
        from dngscan.drt import TOE_END_DISPLAY_LINEAR, _value_at_ev, curve_params_from_plan

        tone = _tone_plan()
        end = compiled_curve_transitions(tone)["toe_end_ev"]
        self.assertIsNotNone(end)
        params = curve_params_from_plan(tone)
        self.assertEqual(float(params["target_black"]), 0.0)
        self.assertAlmostEqual(
            _value_at_ev(float(end), params), TOE_END_DISPLAY_LINEAR, places=4
        )

    def test_unmeasurable_crossing_reports_none_and_the_solver_refuses_to_move(self) -> None:
        from unittest import mock

        from dngscan import drt

        tone = _tone_plan()
        with mock.patch.object(drt, "toe_end_ev_from_params", return_value=None):
            self.assertIsNone(compiled_curve_transitions(tone)["toe_end_ev"])
            solved = drt.solve_toe_power_for_toe_end(tone, -2.0)
            self.assertEqual(solved, float(tone.toe_power))
            plan = _render_plan(tone)
            adjusted = apply_render_adjustments(
                plan, RenderAdjustments(toe_end_offset=-1.0)
            )
            self.assertEqual(adjusted.tone.toe_power, tone.toe_power)

    def test_unmeasurable_toe_end_serializes_to_null_for_the_page(self) -> None:
        from dngscan.gui.service import _finite_or_none

        self.assertIsNone(_finite_or_none(None))

    def test_untouched_sliders_do_not_reclamp_film_preset_powers(self) -> None:
        # vision3250d_theatrical compiles toe_power 3.45, outside the shadow
        # slider's own clamp range; a zero shadow bias must leave it alone even
        # when another slider is active.
        plan = self._film_plan("vision3250d_theatrical")
        self.assertGreater(float(plan.tone.toe_power), 2.5)
        adjusted = apply_render_adjustments(
            plan, RenderAdjustments(midtone_brightness=0.5)
        )
        self.assertEqual(adjusted.tone.toe_power, plan.tone.toe_power)
        self.assertEqual(adjusted.tone.shoulder_power, plan.tone.shoulder_power)
        moved = apply_render_adjustments(
            plan, RenderAdjustments(shadow_transition=0.5)
        )
        self.assertLess(float(moved.tone.toe_power), float(plan.tone.toe_power))


class ShoulderWhiteOffsetTests(unittest.TestCase):
    """Shoulder-white semantics: the slider moves the compiled near-white crossing.

    The former shoulder_start_offset moved the latitude anchor instead; measured on
    nine real frames it was a dead control in BOTH directions — with contrast 3 the
    display range above the pivot is spent within ~1 EV, so the C1 legality clamps
    absorbed the whole start move before it could render (compiled shoulder_start_ev
    moved 0.2 -> 2.2 EV while bright-region output medians changed <= 1 code value).
    The replacement re-solves shoulder_power for a target near-white crossing, which
    has real geometric room: the shoulder region spans several scene EV.
    """

    def test_positive_offset_postpones_the_white_point(self) -> None:
        plan = _render_plan(_tone_plan())
        base_white = compiled_curve_transitions(plan.tone)["shoulder_white_ev"]
        self.assertIsNotNone(base_white)
        adjusted = apply_render_adjustments(
            plan, RenderAdjustments(shoulder_white_offset=0.25)
        )
        new_white = compiled_curve_transitions(adjusted.tone)["shoulder_white_ev"]
        self.assertGreater(new_white, base_white + 0.2)
        self.assertAlmostEqual(new_white, base_white + 0.25, delta=0.05)
        # softer shoulder = lower power; endpoints, latitude and toe must not move
        self.assertLess(adjusted.tone.shoulder_power, plan.tone.shoulder_power)
        self.assertEqual(adjusted.tone.white_ev, plan.tone.white_ev)
        self.assertEqual(adjusted.tone.black_ev, plan.tone.black_ev)
        self.assertEqual(adjusted.tone.latitude_hi_ev, plan.tone.latitude_hi_ev)
        self.assertEqual(adjusted.tone.toe_power, plan.tone.toe_power)
        self.assertEqual(
            compiled_curve_transitions(adjusted.tone)["shoulder_start_ev"],
            compiled_curve_transitions(plan.tone)["shoulder_start_ev"],
        )
        # softer roll-off darkens the low highlights, deep shadows stay put
        probe_high = float(apply_c1_endpoints(np.asarray([2.0]), adjusted.tone)[0])
        base_high = float(apply_c1_endpoints(np.asarray([2.0]), plan.tone)[0])
        self.assertLess(probe_high, base_high)
        probe_deep = float(apply_c1_endpoints(np.asarray([-4.0]), adjusted.tone)[0])
        base_deep = float(apply_c1_endpoints(np.asarray([-4.0]), plan.tone)[0])
        self.assertAlmostEqual(probe_deep, base_deep, places=5)
        _assert_monotone(self, adjusted.tone)

    def test_negative_offset_hardens_the_shoulder(self) -> None:
        plan = _render_plan(_tone_plan())
        base_white = compiled_curve_transitions(plan.tone)["shoulder_white_ev"]
        adjusted = apply_render_adjustments(
            plan, RenderAdjustments(shoulder_white_offset=-0.4)
        )
        new_white = compiled_curve_transitions(adjusted.tone)["shoulder_white_ev"]
        self.assertLess(new_white, base_white - 0.2)
        self.assertGreater(adjusted.tone.shoulder_power, plan.tone.shoulder_power)
        _assert_monotone(self, adjusted.tone)

    def test_offset_sweep_is_monotone_in_crossing_and_power(self) -> None:
        plan = _render_plan(_tone_plan())
        crossings, powers = [], []
        for offset in (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0):
            adjusted = apply_render_adjustments(
                plan, RenderAdjustments(shoulder_white_offset=offset)
            )
            crossing = compiled_curve_transitions(adjusted.tone)["shoulder_white_ev"]
            self.assertIsNotNone(crossing)
            crossings.append(float(crossing))
            powers.append(float(adjusted.tone.shoulder_power))
            _assert_monotone(self, adjusted.tone)
        for a, b in zip(crossings, crossings[1:]):
            self.assertLessEqual(a, b + 1e-9)
        for a, b in zip(powers, powers[1:]):
            self.assertGreaterEqual(a, b - 1e-9)

    def test_out_of_reach_request_clamps_to_legal_shoulder_power(self) -> None:
        # +3 EV against a +4 EV white endpoint saturates at the softest legal
        # shoulder; the compiled fact reports the crossing actually achieved.
        plan = _render_plan(_tone_plan())
        adjusted = apply_render_adjustments(
            plan, RenderAdjustments(shoulder_white_offset=3.0)
        )
        self.assertGreaterEqual(
            adjusted.tone.shoulder_power, SHOULDER_POWER_SOLVE_MIN - 1e-9
        )
        self.assertLessEqual(
            adjusted.tone.shoulder_power, SHOULDER_POWER_SOLVE_MAX + 1e-9
        )
        compiled = compiled_curve_transitions(adjusted.tone)["shoulder_white_ev"]
        self.assertLess(compiled, plan.tone.white_ev)
        _assert_monotone(self, adjusted.tone)

    def test_out_of_range_value_is_clamped_not_amplified(self) -> None:
        plan = _render_plan(_tone_plan())
        wild = apply_render_adjustments(
            plan, RenderAdjustments(shoulder_white_offset=99.0)
        )
        bounded = apply_render_adjustments(
            plan, RenderAdjustments(shoulder_white_offset=3.0)
        )
        self.assertAlmostEqual(
            wild.tone.shoulder_power, bounded.tone.shoulder_power, places=6
        )

    def test_unmeasurable_crossing_reports_none_and_the_solver_refuses_to_move(self) -> None:
        from unittest import mock

        from dngscan import drt

        tone = _tone_plan()
        with mock.patch.object(drt, "shoulder_white_ev_from_params", return_value=None):
            self.assertIsNone(compiled_curve_transitions(tone)["shoulder_white_ev"])
            solved = drt.solve_shoulder_power_for_white_ev(tone, 3.0)
            self.assertEqual(solved, float(tone.shoulder_power))
            plan = _render_plan(tone)
            adjusted = apply_render_adjustments(
                plan, RenderAdjustments(shoulder_white_offset=1.0)
            )
            self.assertEqual(adjusted.tone.shoulder_power, tone.shoulder_power)

    def test_neutral_reference_still_ignores_all_adjustments(self) -> None:
        plan = _render_plan(_tone_plan(tone_core="neutral"))
        self.assertIs(
            apply_render_adjustments(
                plan,
                RenderAdjustments(toe_end_offset=-2.0, shoulder_white_offset=2.0),
            ),
            plan,
        )


class FilmPresetShoulderWhiteTests(unittest.TestCase):
    """Span-relative shoulder-white semantics on film-preset plans.

    The near-white reference is SHOULDER_WHITE_DISPLAY_RATIO of the way from the
    compiled black floor to the compiled white target: identical to the absolute
    0.90 level for floor-0 / white-1 plans, and the only definition under which
    lifted-floor (paper Dmax) or faded-white plans keep a real, monotone
    measurement — the mirror of the toe fix's floor-relative near-black reference.
    kodachrome64 additionally compiles shoulder_power 9.1, outside the highlight
    slider's own clamp range, so these plans also exercise the no-reclamp-at-zero
    contract and the solve bounds' hard-side margin.
    """

    PRESETS = ("portra400", "kodachrome64", "vision3250d_theatrical")
    OFFSETS = (-2.0, -1.0, -0.05, 0.0, 0.05, 1.0, 3.0)

    @staticmethod
    def _film_plan(preset: str) -> RenderPlan:
        from dngscan.film_curve import apply_film_curve_preset

        return _render_plan(apply_film_curve_preset(_tone_plan(), preset))

    def test_offsets_keep_a_monotone_gradient_with_the_declared_direction(self) -> None:
        for preset in self.PRESETS:
            with self.subTest(preset=preset):
                plan = self._film_plan(preset)
                base_power = float(plan.tone.shoulder_power)
                base_white = compiled_curve_transitions(plan.tone)["shoulder_white_ev"]
                self.assertIsNotNone(base_white)
                powers, whites = [], []
                for offset in self.OFFSETS:
                    adjusted = apply_render_adjustments(
                        plan, RenderAdjustments(shoulder_white_offset=offset)
                    )
                    white = compiled_curve_transitions(adjusted.tone)["shoulder_white_ev"]
                    self.assertIsNotNone(white)
                    powers.append(float(adjusted.tone.shoulder_power))
                    whites.append(float(white))
                    if offset < 0.0:
                        # Harder (or clamped hard), never softer than base.
                        self.assertGreaterEqual(
                            adjusted.tone.shoulder_power, base_power - 1e-9
                        )
                        self.assertLessEqual(white, base_white + 1e-9)
                    elif offset > 0.0:
                        self.assertLessEqual(
                            adjusted.tone.shoulder_power, base_power + 1e-9
                        )
                        self.assertGreaterEqual(white, base_white - 1e-9)
                    _assert_monotone(self, adjusted.tone)
                for a, b in zip(powers, powers[1:]):
                    self.assertGreaterEqual(a, b - 1e-9)
                for a, b in zip(whites, whites[1:]):
                    self.assertLessEqual(a, b + 1e-9)

    def test_lifted_floor_white_point_is_the_span_relative_crossing(self) -> None:
        from dngscan.drt import (
            SHOULDER_WHITE_DISPLAY_RATIO,
            _value_at_ev,
            curve_params_from_plan,
        )

        for preset in ("portra400", "kodachrome64"):
            with self.subTest(preset=preset):
                tone = self._film_plan(preset).tone
                white = compiled_curve_transitions(tone)["shoulder_white_ev"]
                self.assertIsNotNone(white)
                self.assertLess(white, float(tone.white_ev))
                params = curve_params_from_plan(tone)
                gamma = float(params["gamma"])
                floor = float(params["target_black"]) ** gamma
                top = float(params["target_white"]) ** gamma
                self.assertAlmostEqual(
                    _value_at_ev(float(white), params),
                    floor + SHOULDER_WHITE_DISPLAY_RATIO * (top - floor),
                    places=4,
                )

    def test_zero_floor_measurement_is_the_absolute_ratio_reference(self) -> None:
        from dngscan.drt import (
            SHOULDER_WHITE_DISPLAY_RATIO,
            _value_at_ev,
            curve_params_from_plan,
        )

        tone = _tone_plan()
        white = compiled_curve_transitions(tone)["shoulder_white_ev"]
        self.assertIsNotNone(white)
        params = curve_params_from_plan(tone)
        self.assertEqual(float(params["target_black"]), 0.0)
        self.assertAlmostEqual(
            _value_at_ev(float(white), params), SHOULDER_WHITE_DISPLAY_RATIO, places=4
        )

    def test_untouched_sliders_do_not_reclamp_film_preset_shoulder_powers(self) -> None:
        # kodachrome64 compiles shoulder_power 9.102, outside the highlight
        # slider's own clamp range; a zero shoulder-white bias must leave it
        # alone even when another slider is active.
        plan = self._film_plan("kodachrome64")
        self.assertGreater(float(plan.tone.shoulder_power), 5.0)
        adjusted = apply_render_adjustments(
            plan, RenderAdjustments(midtone_brightness=0.5)
        )
        self.assertEqual(adjusted.tone.shoulder_power, plan.tone.shoulder_power)


class ShoulderWhiteRenderEfficacyTests(unittest.TestCase):
    """Render-level efficacy: the slider must visibly move real output pixels.

    THIS is the test class whose absence let the original defect ship: the old
    shoulder_start_offset had unit tests proving the compiled shoulder_start_ev
    anchor moved (0.2 -> 2.2 EV), yet rendered bright-region medians moved <= 1
    code value on nine real frames — the compiled fact moved, the picture did not.
    A plan-level assertion can never catch that class of bug; only pushing a
    synthetic gradient through the FULL production pipeline (plan compile ->
    apply_render_adjustments -> render_output_u8) and asserting a monotone,
    visibly large response in the output codes can.
    """

    THRESHOLD = 180  # bright region: output max-channel above this code value
    OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)

    @classmethod
    def setUpClass(cls) -> None:
        from dngscan.render import render_output_u8
        from dngscan.tone import build_render_plan

        # Neutral-gray column gradient spanning deep shadow to a bright reliable
        # tail (+6 EV -> compiled white endpoint ~+6.3 EV, a wide-DR daylight
        # window): every EV band is present, so the bright-region median, the
        # untouched-shadow check and the shoulder's several-EV working room all
        # read real pipeline output. The default 65535 storage scale clips scene
        # linear at 1.0 (+2.5 EV over mid gray) and would silently shrink the
        # shoulder's room to nothing, so the bundle stores at scale 4096 (up to
        # +6.5 EV of real scene headroom).
        scale = 4096.0
        ev = np.linspace(-8.0, 6.0, 120, dtype=np.float32)
        linear = 0.18 * np.power(2.0, ev)
        scene = np.repeat(linear[np.newaxis, :], 64, axis=0)[..., np.newaxis]
        scene = np.repeat(scene, 3, axis=2)
        stored = np.clip(scene * scale, 0.0, 65535.0).astype(np.uint16)
        bundle = replace(
            _bundle_from_scene(stored), scene_scale=scale, render_scale=scale
        )
        analysis = _analysis_for(
            median_vs_gray_ev=0.0, ev_median=0.0, ev_p99=5.0, ev_p999=5.8,
            usable_dr_ev=13.0,
        )
        cls._renders = {}
        for offset in cls.OFFSETS:
            plan = build_render_plan(
                bundle, analysis, "agx", "srgb",
                adjustments=RenderAdjustments(shoulder_white_offset=offset),
            )
            cls._renders[offset] = render_output_u8(
                bundle, analysis, "srgb", tone_plan=plan
            )

    def _bright_median(self, offset: float) -> float:
        out = self._renders[offset]
        base = self._renders[0.0]
        mask = base.max(axis=2) > self.THRESHOLD
        self.assertGreater(int(mask.sum()), 100, "gradient must contain a bright region")
        return float(np.median(out[mask]))

    def test_bright_region_median_responds_monotonically_and_visibly(self) -> None:
        medians = [self._bright_median(offset) for offset in self.OFFSETS]
        # Declared direction: right (positive) = softer = later white = darker
        # bright-region output. Monotone non-increasing across the whole sweep.
        for a, b in zip(medians, medians[1:]):
            self.assertGreaterEqual(a, b - 0.51)
        # Visibly large at strong settings on both sides.
        self.assertGreaterEqual(medians[0] - medians[2], 8.0)
        self.assertGreaterEqual(medians[2] - medians[-1], 8.0)

    def test_shadows_and_midgray_do_not_move(self) -> None:
        base = self._renders[0.0]
        for offset in (-1.0, 1.0):
            out = self._renders[offset]
            dark = base.max(axis=2) < 100
            self.assertGreater(int(dark.sum()), 100)
            delta = np.abs(
                out[dark].astype(np.int16) - base[dark].astype(np.int16)
            )
            self.assertLessEqual(int(delta.max()), 1)


class ServiceParsingTests(unittest.TestCase):
    def test_offsets_parse_with_their_own_ranges(self) -> None:
        parsed = parse_render_adjustments(
            {"toeEndOffset": -2.5, "shoulderWhiteOffset": 2.5}
        )
        self.assertEqual(parsed.toe_end_offset, -2.5)
        self.assertEqual(parsed.shoulder_white_offset, 2.5)

    def test_legacy_shoulder_key_still_parses_as_an_alias(self) -> None:
        # Persisted settings and older callers used the shoulder-start names; they
        # must keep driving the renamed control, and the new key wins when both
        # are present.
        parsed = parse_render_adjustments({"shoulderStartOffset": 1.5})
        self.assertEqual(parsed.shoulder_white_offset, 1.5)
        parsed = parse_render_adjustments({"shoulder_start_offset": -0.5})
        self.assertEqual(parsed.shoulder_white_offset, -0.5)
        parsed = parse_render_adjustments(
            {"shoulderWhiteOffset": 2.0, "shoulderStartOffset": 1.0}
        )
        self.assertEqual(parsed.shoulder_white_offset, 2.0)

    def test_offset_range_rejections(self) -> None:
        for payload in (
            {"toeEndOffset": -3.01},
            {"toeEndOffset": 0.51},
            {"shoulderWhiteOffset": -2.01},
            {"shoulderWhiteOffset": 3.01},
            {"shoulderStartOffset": 3.01},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    parse_render_adjustments(payload)

    def test_endpoint_mode_parses_and_rejects(self) -> None:
        self.assertEqual(parse_endpoint_mode({}), "adaptive")
        self.assertEqual(parse_endpoint_mode({"endpointMode": "evidence"}), "evidence")
        with self.assertRaises(ValueError):
            parse_endpoint_mode({"endpointMode": "percentile"})

    def test_adjustment_cache_key_carries_the_offsets(self) -> None:
        base = _adjustment_key(RenderAdjustments())
        self.assertEqual(len(base), 7)
        moved = _adjustment_key(RenderAdjustments(toe_end_offset=-1.0))
        self.assertNotEqual(base, moved)
        self.assertEqual(_adjustment_key(None), base)


class CliParsingTests(unittest.TestCase):
    def test_flags_parse(self) -> None:
        from dngscan.cli import parse_args

        args = parse_args(
            [
                "x.dng",
                "--endpoint-mode", "evidence",
                "--toe-end-offset", "-2",
                "--shoulder-white-offset", "1.5",
            ]
        )
        self.assertEqual(args.endpoint_mode, "evidence")
        self.assertEqual(args.toe_end_offset, -2.0)
        self.assertEqual(args.shoulder_white_offset, 1.5)

    def test_legacy_shoulder_flag_still_parses_as_an_alias(self) -> None:
        from dngscan.cli import parse_args

        args = parse_args(["x.dng", "--shoulder-start-offset", "1.5"])
        self.assertEqual(args.shoulder_white_offset, 1.5)

    def test_default_is_adaptive_and_zero(self) -> None:
        from dngscan.cli import parse_args

        args = parse_args(["x.dng"])
        self.assertEqual(args.endpoint_mode, "adaptive")
        self.assertEqual(args.toe_end_offset, 0.0)
        self.assertEqual(args.shoulder_white_offset, 0.0)

    def test_range_errors(self) -> None:
        from dngscan.cli import parse_args

        for argv in (
            ["x.dng", "--toe-end-offset", "-3.1"],
            ["x.dng", "--toe-end-offset", "0.6"],
            ["x.dng", "--shoulder-white-offset", "-2.1"],
            ["x.dng", "--shoulder-white-offset", "3.1"],
            ["x.dng", "--shoulder-start-offset", "3.1"],
            ["x.dng", "--endpoint-mode", "percentile"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    parse_args(argv)


class GuiPageContractTests(unittest.TestCase):
    def test_controls_exist_in_tone_adjust_card(self) -> None:
        from dngscan.gui.page import PAGE

        card = PAGE[
            PAGE.index('<div class="card" id="toneAdjustCard"'):
            PAGE.index('<div class="card"', PAGE.index('<div class="card" id="toneAdjustCard"') + 1)
        ]
        self.assertIn('id="endpointMode"', card)
        self.assertIn('value="adaptive"', card)
        self.assertIn('value="evidence"', card)
        self.assertIn('id="toeEndOffset" min="-3" max="0.5"', card)
        self.assertIn('id="shoulderWhiteOffset" min="-2" max="3"', card)
        # New declared semantics: label and title describe the white point, not
        # the start anchor.
        self.assertIn("肩部收白", card)
        self.assertIn("近白参考", card)
        self.assertNotIn("肩部起点", card)

    def test_payload_carries_the_new_parameters(self) -> None:
        from dngscan.gui.page import PAGE

        body = PAGE[PAGE.index("function payload()"):]
        body = body[: body.index("\n}")]
        self.assertIn('endpointMode:$("#endpointMode").value', body)
        self.assertIn('toeEndOffset:+$("#toeEndOffset").value', body)
        self.assertIn('shoulderWhiteOffset:+$("#shoulderWhiteOffset").value', body)

    def test_tone_fact_reports_compiled_transitions(self) -> None:
        from dngscan.gui.page import PAGE

        renderer = PAGE[PAGE.index("function renderDetectedParams(") :]
        renderer = renderer[: renderer.index("\n}")]
        self.assertIn("趾部收黑", renderer)
        self.assertIn("肩部收白", renderer)
        self.assertIn("d.toe_end_ev", renderer)
        self.assertIn("d.shoulder_white_ev", renderer)
        self.assertIn("endpoint_note", renderer)

    def test_new_controls_schedule_preview_and_refresh_facts(self) -> None:
        from dngscan.gui.page import PAGE

        wiring = PAGE[PAGE.index('"toeEndOffset","shoulderWhiteOffset"') :]
        wiring = wiring[: wiring.index("restoreSettings();")]
        self.assertIn("scheduleLivePreview()", wiring)
        self.assertIn("preparePreview()", wiring)
        self.assertIn('$("#endpointMode").addEventListener("change"', wiring)

    def test_settings_persist_the_new_controls(self) -> None:
        from dngscan.gui.page import PAGE

        save = PAGE[PAGE.index("function saveSettings()"):]
        save = save[: save.index("\n}")]
        self.assertIn("endpointMode", save)
        self.assertIn("toeEndOffset", save)
        self.assertIn("shoulderWhiteOffset", save)


if __name__ == "__main__":
    unittest.main()
