# SPDX-License-Identifier: GPL-3.0-or-later
"""Contracts for the endpoint-normalized C1 DRT."""
from __future__ import annotations

import unittest

from dngscan._deps import np
from dngscan.drt import (
    apply_c1_endpoints,
    c1_value_and_derivative_at_ev,
    curve_params_from_plan,
)
from dngscan.models import ToneCompressionPlan


def _plan() -> ToneCompressionPlan:
    return ToneCompressionPlan(
        target_gamut="Rec2020",
        luma_p1=0.01,
        luma_p50=0.18,
        luma_p99=1.0,
        luma_p999=2.0,
        black_ev=-7.0,
        white_ev=4.5,
        dynamic_range_ev=11.5,
        contrast=3.0,
        toe_power=1.5,
        shoulder_power=3.3,
        chroma_p95=0.0,
        negative_rgb_pct=0.0,
        over_rgb_pct=0.0,
        toe_start_ev=-3.0,
        shoulder_start_ev=1.25,
        use_c1_endpoints=True,
    )


class C1EndpointDrtTest(unittest.TestCase):
    def test_endpoint_normalization_keeps_calibrated_pivot(self) -> None:
        plan = _plan()
        self.assertAlmostEqual(float(apply_c1_endpoints(np.asarray([0.0]), plan)[0]), 0.18, places=5)

        wider = ToneCompressionPlan(**{**plan.__dict__, "white_ev": 6.0})
        self.assertAlmostEqual(float(apply_c1_endpoints(np.asarray([0.0]), wider)[0]), 0.18, places=5)

    def test_endpoint_values_and_derivatives_are_continuous(self) -> None:
        plan = _plan()
        delta = np.float32(1e-3)
        params = curve_params_from_plan(plan)
        for transition_x in (float(params["toe_transition_x"]), float(params["shoulder_transition_x"])):
            endpoint = plan.black_ev + transition_x * float(params["range_ev"])
            samples = np.asarray([endpoint - delta, endpoint, endpoint + delta], dtype=np.float32)
            mapped = apply_c1_endpoints(samples, plan)
            self.assertAlmostEqual(
                float((mapped[1] - mapped[0]) / delta),
                float((mapped[2] - mapped[1]) / delta),
                delta=0.01,
            )

    def test_authoritative_anchor_uses_runtime_value_and_analytic_piece_tangent(self) -> None:
        plan = ToneCompressionPlan(
            **{**_plan().__dict__, "latitude_lo_ev": 0.8, "latitude_hi_ev": 0.2}
        )
        params = curve_params_from_plan(plan)
        transition_evs = (
            float(params["black_ev"])
            + float(params["toe_transition_x"]) * float(params["range_ev"]),
            float(params["black_ev"])
            + float(params["shoulder_transition_x"]) * float(params["range_ev"]),
        )
        probes = (
            transition_evs[0] - 0.3,
            transition_evs[0],
            0.5 * (transition_evs[0] + transition_evs[1]),
            transition_evs[1],
            transition_evs[1] + 0.3,
        )
        step = 1e-3
        for ev in probes:
            with self.subTest(ev=ev):
                value, derivative = c1_value_and_derivative_at_ev(ev, plan)
                runtime = float(
                    apply_c1_endpoints(np.asarray([ev], dtype=np.float32), plan)[0]
                )
                self.assertEqual(value, runtime)
                lo = float(
                    apply_c1_endpoints(np.asarray([ev - step], dtype=np.float32), plan)[0]
                )
                hi = float(
                    apply_c1_endpoints(np.asarray([ev + step], dtype=np.float32), plan)[0]
                )
                numeric = (hi - lo) / (2.0 * step)
                self.assertAlmostEqual(derivative, numeric, delta=2e-4)

    def test_curve_is_monotone_and_clamped_at_endpoints(self) -> None:
        plan = _plan()
        ev = np.linspace(plan.black_ev - 2.0, plan.white_ev + 2.0, 10001, dtype=np.float32)
        mapped = apply_c1_endpoints(ev, plan)
        self.assertGreaterEqual(float(np.diff(mapped).min()), -1e-6)
        self.assertLess(float(mapped[0]), 1e-6)
        self.assertAlmostEqual(float(mapped[-1]), 1.0, places=6)

    def test_view_brightness_lifts_interior_without_moving_endpoints(self) -> None:
        """Audit R11: the old body computed the power law on its own numbers
        — a mathematical identity that passed even if the runtime dropped
        view_brightness entirely. This drives the REAL tone path (agx
        formation consumes plan.view_brightness) and asserts the lifted
        render differs exactly as promised."""
        from dngscan.agx import apply_core, formation_matrices

        plan = _plan()
        lifted = ToneCompressionPlan(**{**plan.__dict__, "view_brightness": 1.25})
        rgb = np.asarray(
            [[2.0 ** plan.black_ev * 0.18] * 3,
             [0.02, 0.03, 0.025], [0.18, 0.18, 0.18],
             [2.0 ** plan.white_ev * 0.18] * 3], dtype=np.float32)
        inset, outset = formation_matrices(plan)
        base = apply_core(rgb, plan, inset, outset)
        out = apply_core(rgb, lifted, inset, outset)
        # endpoints pinned...
        self.assertLess(float(np.max(np.abs(out[0] - base[0]))), 1e-5)
        self.assertLess(float(np.max(np.abs(out[-1] - base[-1]))), 1e-4)
        # ...and the interior actually lifted through the runtime
        self.assertGreater(float(np.mean(out[1])), float(np.mean(base[1])))
        self.assertGreater(float(np.mean(out[2])), float(np.mean(base[2])))


class LookOverrideC1InteractionTest(unittest.TestCase):
    """A chromatic look's AgX-core overrides must reach the C1 endpoint path."""

    def test_none_look_is_identity_object(self) -> None:
        from dngscan.render import plan_with_look_overrides

        plan = _plan()
        self.assertIs(plan_with_look_overrides(plan, "none"), plan)

    def test_target_black_override_lifts_c1_black(self) -> None:
        # Faded-film target black flows through curve_params_from_plan into the C1 curve.
        plan = _plan()
        faded = ToneCompressionPlan(**{**plan.__dict__, "target_black_linear": 0.05})
        deep_ev = np.asarray([plan.black_ev - 1.0, plan.black_ev], dtype=np.float32)
        base = apply_c1_endpoints(deep_ev, plan)
        lifted = apply_c1_endpoints(deep_ev, faded)
        self.assertLess(float(base[0]), 1e-4)
        self.assertGreater(float(lifted[0]), float(base[0]) + 1e-3)

    def test_pivot_offset_flows_through_c1(self) -> None:
        # The offset must reach the compiled curve and reallocate contrast, while the
        # EV0 anchor holds. Slope is deliberately NOT the witness: the anchor solver
        # runs at the uncompensated base slope, so an offset alone leaves it unchanged;
        # what moves is the curve geometry (transitions) and the subject's rendering.
        plan = _plan()
        shifted = ToneCompressionPlan(**{**plan.__dict__, "pivot_ev_offset": -2.0})
        p0 = curve_params_from_plan(plan)
        p1 = curve_params_from_plan(shifted)
        self.assertEqual(float(p0["black_ev"]), float(p1["black_ev"]))
        self.assertNotAlmostEqual(
            float(p0["toe_transition_x"]), float(p1["toe_transition_x"]), places=3
        )
        subject = np.asarray([-2.0], dtype=np.float32)
        self.assertNotAlmostEqual(
            float(apply_c1_endpoints(subject, plan)[0]),
            float(apply_c1_endpoints(subject, shifted)[0]),
            places=3,
        )
        anchored = float(apply_c1_endpoints(np.asarray([0.0], dtype=np.float32), shifted)[0])
        self.assertAlmostEqual(anchored, 0.18, delta=0.006)

    def test_target_white_fades_shoulder_in_c1(self) -> None:
        plan = _plan()
        faded = ToneCompressionPlan(**{**plan.__dict__, "target_white_linear": 0.85})
        hi_ev = np.asarray([plan.white_ev - 0.5, plan.white_ev], dtype=np.float32)
        full = apply_c1_endpoints(hi_ev, plan)
        milky = apply_c1_endpoints(hi_ev, faded)
        self.assertGreater(float(full[-1]), float(milky[-1]))
        self.assertLess(float(milky[-1]), 0.92)

    def test_target_white_extends_c1_endpoint_above_sdr_white(self) -> None:
        plan = _plan()
        extended = ToneCompressionPlan(**{**plan.__dict__, "target_white_linear": 8.0})
        white_ev = np.asarray([plan.white_ev], dtype=np.float32)
        sdr = apply_c1_endpoints(white_ev, plan)
        hdr = apply_c1_endpoints(white_ev, extended)
        self.assertAlmostEqual(float(sdr[0]), 1.0, places=5)
        self.assertAlmostEqual(float(hdr[0]), 8.0, places=4)

    def test_hue_restore_override_changes_agx_core_output(self) -> None:
        from dngscan.agx import AGX_INSET_REC2020, AGX_OUTSET_REC2020, apply_core

        plan = _plan()
        low_restore = ToneCompressionPlan(**{**plan.__dict__, "hue_restore": 0.0})
        high_restore = ToneCompressionPlan(**{**plan.__dict__, "hue_restore": 1.0})
        # A near-primary saturated stimulus maximizes the per-channel "notorious six"
        # skew that hue restore controls, so the override's effect is unambiguous.
        rgb = np.asarray([[0.80, 0.04, 0.02]], dtype=np.float32)
        a = apply_core(rgb, low_restore, AGX_INSET_REC2020, AGX_OUTSET_REC2020)
        b = apply_core(rgb, high_restore, AGX_INSET_REC2020, AGX_OUTSET_REC2020)
        self.assertGreater(float(np.max(np.abs(a - b))), 1e-3)


if __name__ == "__main__":
    unittest.main()
