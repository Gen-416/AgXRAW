# SPDX-License-Identifier: GPL-3.0-or-later
"""HDR AgX v2 tone math gates. No DNG, no image, no pixels."""
from __future__ import annotations

import math
import unittest

from dngscan.constants import (
    AGX_REFERENCE_RANGE_EV,
    DARKTABLE_BASE_GAMMA,
    OUTPUT_REFERENCE_WHITE_STOPS,
    SCENE_MIDGRAY,
)
from dngscan.hdr_agx_math import (
    MAX_SINGLE_SEGMENT_ALPHA,
    achieved_headroom_ev,
    body_anchor_at_ev,
    body_anchor_from_curve,
    body_encoded_slope,
    compile_hdr_shoulder,
    compile_hdr_shoulder_from_anchor,
    evaluate_hdr_shoulder,
    requested_headroom_ev,
    validate_hdr_shoulder,
)
from dngscan.models import HdrShoulderSegment
from dngscan.hdr_agx_plan import (
    MAXIMUM_WHITE_EV,
    NORMAL_MINIMUM_WHITE_EV,
    NORMAL_SHOULDER_START_EV,
    NORMAL_WHITE_MARGIN_EV,
    SPARSE_EMITTER_MINIMUM_WHITE_EV,
    SPARSE_EMITTER_SHOULDER_START_EV,
    SPARSE_EMITTER_WHITE_MARGIN_EV,
)

CONTRAST = 3.0
KNEE = 0.20


def _shoulder(peak_stops: float, knee_ev: float = KNEE, white_ev: float = 4.138):
    return compile_hdr_shoulder(knee_ev, white_ev, peak_stops, CONTRAST)


def _linear(scene_ev: float, segments) -> float:
    return SCENE_MIDGRAY * 2.0 ** evaluate_hdr_shoulder(scene_ev, segments)


class CoordinateConstantTests(unittest.TestCase):
    def test_output_reference_white_is_derived_not_stored(self) -> None:
        """One source of truth: the stop count follows from mid gray, nothing else."""
        self.assertEqual(SCENE_MIDGRAY, 0.18)
        self.assertAlmostEqual(
            OUTPUT_REFERENCE_WHITE_STOPS, math.log2(1.0 / SCENE_MIDGRAY), places=15
        )
        self.assertAlmostEqual(OUTPUT_REFERENCE_WHITE_STOPS, 2.473931188332412, places=12)

    def test_contrast_is_not_the_encoded_slope(self) -> None:
        self.assertAlmostEqual(
            body_encoded_slope(3.0), 3.0 / AGX_REFERENCE_RANGE_EV, places=15
        )
        self.assertNotAlmostEqual(body_encoded_slope(3.0), 3.0, places=3)

    def test_body_anchor_matches_its_closed_form(self) -> None:
        """Anchors are computed from midgray/gamma/contrast, never stored separately."""
        q0 = SCENE_MIDGRAY ** (1.0 / DARKTABLE_BASE_GAMMA)
        self.assertAlmostEqual(q0, 0.4586564468643811, places=15)
        value, stops, slope = body_anchor_at_ev(0.0, CONTRAST)
        self.assertAlmostEqual(value, SCENE_MIDGRAY, places=15)
        self.assertAlmostEqual(stops, 0.0, places=15)
        # dz/de at the pivot follows from 0.18 / 2.2 / 3.0 / 16.5 alone.
        self.assertAlmostEqual(slope, 1.2581923143145526, places=12)

    def test_knee_anchor_matches_the_designed_example(self) -> None:
        value, stops, slope = body_anchor_at_ev(KNEE, CONTRAST)
        self.assertAlmostEqual(value, 0.21289732342815634, places=15)
        self.assertAlmostEqual(stops, 0.24216090560632872, places=15)
        self.assertAlmostEqual(slope, 1.1657668767547156, places=15)

    def test_production_anchor_consumes_value_and_analytic_derivative(self) -> None:
        value = 0.2129
        slope_t = 0.1725
        anchor = body_anchor_from_curve(lambda _ev: (value, slope_t), KNEE)
        self.assertEqual(anchor[0], value)
        self.assertAlmostEqual(anchor[1], math.log2(value / SCENE_MIDGRAY), places=15)
        self.assertAlmostEqual(
            anchor[2], slope_t / (math.log(2.0) * value), places=15
        )


class RequestedHeadroomTests(unittest.TestCase):
    def test_tail_at_reference_white_earns_nothing(self) -> None:
        self.assertEqual(
            requested_headroom_ev(OUTPUT_REFERENCE_WHITE_STOPS, 3.0), 0.0
        )

    def test_display_capacity_is_only_a_ceiling(self) -> None:
        self.assertAlmostEqual(
            requested_headroom_ev(OUTPUT_REFERENCE_WHITE_STOPS + 9.0, 2.0), 2.0, places=12
        )

    def test_missing_tail_earns_nothing(self) -> None:
        """An absent measurement must not read as unlimited signal."""
        self.assertEqual(requested_headroom_ev(float("nan"), 3.0), 0.0)
        self.assertEqual(requested_headroom_ev(float("inf"), 3.0), 0.0)
        self.assertEqual(requested_headroom_ev(5.0, float("nan")), 0.0)


class ShoulderStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.peak_stops = OUTPUT_REFERENCE_WHITE_STOPS + 1.3641
        self.segments = _shoulder(self.peak_stops)
        _, self.knee_stops, self.knee_slope = body_anchor_at_ev(KNEE, CONTRAST)

    def test_designed_example_needs_one_segment(self) -> None:
        self.assertEqual(len(self.segments), 1)
        self.assertAlmostEqual(self.segments[0].alpha, 1.2767, places=3)
        self.assertLessEqual(self.segments[0].alpha, MAX_SINGLE_SEGMENT_ALPHA)

    def test_joins_the_body_with_matching_value_and_slope(self) -> None:
        """The C1 join at K is the whole point; a limiter that rescaled m0 would break it."""
        self.assertAlmostEqual(
            evaluate_hdr_shoulder(KNEE, self.segments), self.knee_stops, places=15
        )
        step = 1e-7
        numeric = (
            evaluate_hdr_shoulder(KNEE + step, self.segments)
            - evaluate_hdr_shoulder(KNEE, self.segments)
        ) / step
        self.assertAlmostEqual(numeric, self.knee_slope, places=5)

    def test_white_endpoint_is_reached_with_zero_slope(self) -> None:
        white = 4.138
        self.assertAlmostEqual(
            evaluate_hdr_shoulder(white, self.segments), self.peak_stops, places=12
        )
        step = 1e-7
        inner = (
            evaluate_hdr_shoulder(white, self.segments)
            - evaluate_hdr_shoulder(white - step, self.segments)
        ) / step
        self.assertLess(abs(inner), 1e-4)
        # Outside the window the curve clamps, so the outer derivative is exactly zero.
        self.assertEqual(
            evaluate_hdr_shoulder(white + 1.0, self.segments),
            evaluate_hdr_shoulder(white, self.segments),
        )

    def test_monotone_across_the_window(self) -> None:
        samples = [
            evaluate_hdr_shoulder(KNEE + (4.138 - KNEE) * i / 20000.0, self.segments)
            for i in range(20001)
        ]
        self.assertGreaterEqual(min(b - a for a, b in zip(samples, samples[1:])), -1e-12)

    def test_validation_rejects_a_rescaled_start_tangent(self) -> None:
        """Guards the failure mode a generic PCHIP limiter would introduce."""
        broken = (
            HdrShoulderSegment(
                e0=KNEE, e1=4.138,
                z0=self.knee_stops, z1=self.peak_stops,
                m0=self.knee_slope * 0.5, m1=0.0,
            ),
        )
        ok, reason = validate_hdr_shoulder(broken, self.knee_slope, self.peak_stops)
        self.assertFalse(ok)
        self.assertIn("C1", reason)

    def test_validation_accepts_the_compiled_shoulder(self) -> None:
        ok, reason = validate_hdr_shoulder(
            self.segments, self.knee_slope, self.peak_stops
        )
        self.assertTrue(ok, msg=reason)


class BodyInvarianceTests(unittest.TestCase):
    """The invariant v1 could not offer, and the reason v2 exists.

    v1 bought its peak by raising the whole curve's gamma, so more headroom darkened the
    shadows: measured -1.5 EV at -4 EV on a real frame. In v2 nothing below K is a function
    of H at all, so this holds structurally rather than numerically.
    """

    def test_headroom_cannot_reach_below_the_knee(self) -> None:
        shoulders = {
            h: _shoulder(OUTPUT_REFERENCE_WHITE_STOPS + h) for h in (0.5, 1.0, 2.0, 3.0)
        }
        for ev in (-6.0, -5.0, -4.0, -3.0, -1.0, 0.0, KNEE):
            body = body_anchor_at_ev(ev, CONTRAST)[0]
            for h, segments in shoulders.items():
                with self.subTest(ev=ev, headroom=h):
                    if ev < KNEE:
                        # Below K the shoulder is not consulted at all.
                        self.assertEqual(
                            evaluate_hdr_shoulder(ev, segments), segments[0].z0
                        )
                    self.assertAlmostEqual(
                        body_anchor_at_ev(ev, CONTRAST)[0], body, places=15
                    )

    def test_knee_value_is_identical_across_headrooms(self) -> None:
        values = {
            h: _linear(KNEE, _shoulder(OUTPUT_REFERENCE_WHITE_STOPS + h))
            for h in (0.25, 1.0, 3.0)
        }
        reference = body_anchor_at_ev(KNEE, CONTRAST)[0]
        for h, value in values.items():
            with self.subTest(headroom=h):
                self.assertAlmostEqual(value, reference, delta=1e-6)


class SingleSegmentFeasibilityTests(unittest.TestCase):
    def test_steep_request_without_opt_in_compiles_nothing(self) -> None:
        """Without the subdivision opt-in a steep request yields no shoulder at all.

        The math layer never silently selects another curve family; callers state the
        contract they want. The authoritative plan compiler passes allow_subdivision=True
        and receives a validated monotone chain instead (covered below and at plan level).
        """
        knee = 0.0
        _, knee_stops, knee_slope = body_anchor_at_ev(knee, CONTRAST)
        white = knee + 1.0
        peak_stops = knee_stops + 0.30
        single_alpha = knee_slope * (white - knee) / (peak_stops - knee_stops)
        self.assertGreater(single_alpha, MAX_SINGLE_SEGMENT_ALPHA)
        self.assertEqual(
            compile_hdr_shoulder(knee, white, peak_stops, CONTRAST), ()
        )

    def test_subdivision_requires_explicit_opt_in(self) -> None:
        """Subdivided compiles keep the full structural contract when opted into."""
        knee = 0.0
        _, knee_stops, knee_slope = body_anchor_at_ev(knee, CONTRAST)
        white = knee + 1.0
        peak_stops = knee_stops + 0.30
        segments = compile_hdr_shoulder_from_anchor(
            knee_ev=knee,
            white_ev=white,
            knee_stops=knee_stops,
            knee_slope=knee_slope,
            peak_stops=peak_stops,
            allow_subdivision=True,
        )
        self.assertGreater(len(segments), 1)
        ok, reason = validate_hdr_shoulder(segments, knee_slope, peak_stops)
        self.assertTrue(ok, msg=reason)
        samples = [
            evaluate_hdr_shoulder(knee + (white - knee) * i / 5000.0, segments)
            for i in range(5001)
        ]
        self.assertGreaterEqual(min(b - a for a, b in zip(samples, samples[1:])), -1e-12)

    def test_full_contrast_and_tail_policy_domain_stays_single_segment(self) -> None:
        """Pin automatic and user-adjustable bounds so a retune forces design review."""
        policies = (
            (
                "normal",
                NORMAL_SHOULDER_START_EV,
                NORMAL_WHITE_MARGIN_EV,
                NORMAL_MINIMUM_WHITE_EV,
                1.8494437932436134,
                2.7357483714059394,
            ),
            (
                "sparse",
                SPARSE_EMITTER_SHOULDER_START_EV,
                SPARSE_EMITTER_WHITE_MARGIN_EV,
                SPARSE_EMITTER_MINIMUM_WHITE_EV,
                1.9537393335285478,
                2.9306090002928213,
            ),
        )
        # apply_render_adjustments clamps the user-facing midtone contrast to this range.
        contrast_values = (1.5, 3.0, 4.5)
        for (
            label,
            knee,
            margin,
            minimum_white,
            expected_default,
            expected_full_domain,
        ) in policies:
            with self.subTest(policy=label):
                worst_by_contrast = {}
                for contrast in contrast_values:
                    _, knee_stops, knee_slope = body_anchor_at_ev(knee, contrast)
                    worst = (-1.0, 0.0, 0.0)
                    for step in range(0, 12001):
                        tail = step / 1000.0
                        headroom = requested_headroom_ev(tail, 3.0)
                        if headroom <= 0.0:
                            continue
                        white = min(max(tail + margin, minimum_white), MAXIMUM_WHITE_EV)
                        peak_stops = OUTPUT_REFERENCE_WHITE_STOPS + headroom
                        alpha = knee_slope * (white - knee) / (peak_stops - knee_stops)
                        if alpha > worst[0]:
                            worst = (alpha, white, peak_stops)
                    worst_by_contrast[contrast] = worst[0]
                    self.assertLess(worst[0], MAX_SINGLE_SEGMENT_ALPHA)
                    self.assertEqual(
                        len(
                            compile_hdr_shoulder(
                                knee, worst[1], worst[2], contrast
                            )
                        ),
                        1,
                    )
                self.assertAlmostEqual(
                    worst_by_contrast[3.0], expected_default, places=9
                )
                self.assertAlmostEqual(
                    max(worst_by_contrast.values()), expected_full_domain, places=9
                )

    def test_display_headroom_axis_always_compiles_a_valid_shoulder(self) -> None:
        """Every well-posed (contrast, tail, headroom) request compiles and validates.

        The scan above fixes display headroom at 3.0 EV, which is where the original
        "alpha < 3 over the whole policy domain" claim came from -- and that claim is
        false once the headroom control drops: H_content = min(H_display, H_signal), so a
        low-headroom display caps Z_peak while the tail keeps pushing W out, and alpha
        grows without bound. The contract is therefore not "single segment everywhere"
        but "a validated monotone C1 shoulder everywhere": single where alpha <= 3,
        subdivided beyond it, never absent for a well-posed request.
        """
        policies = (
            ("normal", NORMAL_SHOULDER_START_EV, NORMAL_WHITE_MARGIN_EV,
             NORMAL_MINIMUM_WHITE_EV),
            ("sparse", SPARSE_EMITTER_SHOULDER_START_EV, SPARSE_EMITTER_WHITE_MARGIN_EV,
             SPARSE_EMITTER_MINIMUM_WHITE_EV),
        )
        saw_subdivided = False
        for label, knee, margin, minimum_white in policies:
            for contrast in (1.5, 3.0, 4.5):
                _, knee_stops, knee_slope = body_anchor_at_ev(knee, contrast)
                for headroom_tenths in range(5, 54, 4):
                    headroom = headroom_tenths / 10.0
                    for tail_tenths in range(25, 121, 5):
                        tail = tail_tenths / 10.0
                        requested = requested_headroom_ev(tail, headroom)
                        if requested <= 0.0:
                            continue
                        white = min(
                            max(tail + margin, minimum_white), MAXIMUM_WHITE_EV
                        )
                        peak_stops = OUTPUT_REFERENCE_WHITE_STOPS + requested
                        alpha = knee_slope * (white - knee) / (peak_stops - knee_stops)
                        segments = compile_hdr_shoulder(
                            knee, white, peak_stops, contrast, allow_subdivision=True
                        )
                        with self.subTest(
                            policy=label, contrast=contrast,
                            headroom=headroom, tail=tail,
                        ):
                            ok, reason = validate_hdr_shoulder(
                                segments, knee_slope, peak_stops
                            )
                            self.assertTrue(ok, msg=reason)
                            if alpha <= MAX_SINGLE_SEGMENT_ALPHA:
                                # Subdivision must never be gratuitous.
                                self.assertEqual(len(segments), 1)
                            else:
                                self.assertGreater(len(segments), 1)
                                saw_subdivided = True
        # The sweep must actually exercise the low-headroom subdivided region.
        self.assertTrue(saw_subdivided)


class DegenerateRequestTests(unittest.TestCase):
    """Malformed input must produce no shoulder, never something that merely renders."""

    def test_zero_or_negative_window(self) -> None:
        peak = OUTPUT_REFERENCE_WHITE_STOPS + 1.0
        self.assertEqual(compile_hdr_shoulder(KNEE, KNEE, peak, CONTRAST), ())
        self.assertEqual(compile_hdr_shoulder(KNEE, KNEE - 1.0, peak, CONTRAST), ())

    def test_peak_at_or_below_the_knee(self) -> None:
        _, knee_stops, _ = body_anchor_at_ev(KNEE, CONTRAST)
        self.assertEqual(compile_hdr_shoulder(KNEE, 4.0, knee_stops, CONTRAST), ())
        self.assertEqual(
            compile_hdr_shoulder(KNEE, 4.0, knee_stops - 0.5, CONTRAST), ()
        )

    def test_empty_shoulder_fails_validation(self) -> None:
        ok, reason = validate_hdr_shoulder((), 1.0, 1.0)
        self.assertFalse(ok)
        self.assertIn("空", reason)


class AchievedHeadroomTests(unittest.TestCase):
    def test_reports_zero_when_nothing_exceeds_reference_white(self) -> None:
        self.assertEqual(achieved_headroom_ev([[0.2, 0.5, 1.0]]), 0.0)

    def test_reports_the_reached_peak(self) -> None:
        self.assertAlmostEqual(achieved_headroom_ev([[1.0, 4.0]]), 2.0, places=12)


if __name__ == "__main__":
    unittest.main()
