# SPDX-License-Identifier: GPL-3.0-or-later
"""HDR curve runtime: body-once dual shoulder and curve-params cache."""
from __future__ import annotations

import unittest

import numpy as np

from dngscan.constants import SCENE_MIDGRAY
from dngscan.drt import curve_params_from_plan
from dngscan.hdr_agx_math import compile_hdr_shoulder
from dngscan.hdr_curve import HdrCurveTable, apply_hdr_curve, apply_hdr_curve_pair
from dngscan.models import HdrToneCurve, ToneCompressionPlan


def _formation() -> ToneCompressionPlan:
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
        shoulder_start_ev=0.2,
        use_c1_endpoints=True,
    )


def _tone(peak_linear: float = 4.0) -> HdrToneCurve:
    peak_stops = float(np.log2(peak_linear / SCENE_MIDGRAY))
    segments = compile_hdr_shoulder(0.2, 4.138, peak_stops, 3.0)
    rendered = float(np.log2(peak_linear))
    return HdrToneCurve(
        black_ev=-7.0,
        shoulder_start_ev=0.2,
        white_ev=4.138,
        body_gamma=2.4,
        body_contrast=3.0,
        toe_power=1.5,
        reference_white_stops=float(np.log2(1.0 / SCENE_MIDGRAY)),
        display_headroom_ev=3.0,
        requested_headroom_ev=rendered,
        rendered_headroom_ev=rendered,
        peak_linear=peak_linear,
        reliable_tail_ev=4.0,
        white_margin_ev=0.5,
        shoulder_segments=segments,
        shoulder_alpha=float(segments[0].alpha) if segments else float("nan"),
    )


class HdrCurvePairTests(unittest.TestCase):
    def test_body_once_pair_matches_two_separate_calls(self) -> None:
        formation = _formation()
        tone = _tone(4.0)
        rgb = np.array(
            [
                [0.02, 0.02, 0.02],
                [0.18, 0.18, 0.18],
                [1.5, 0.8, 0.4],
                [8.0, 6.0, 4.0],
            ],
            dtype=np.float32,
        )
        params = curve_params_from_plan(formation)
        native_sep = apply_hdr_curve(rgb, tone, formation, body_params=params)
        ref_sep = apply_hdr_curve(
            rgb, tone, formation, peak_linear=1.0, body_params=params
        )
        native, reference = apply_hdr_curve_pair(
            rgb, tone, formation, need_reference=True, body_params=params
        )
        np.testing.assert_array_equal(native, native_sep)
        np.testing.assert_array_equal(reference, ref_sep)

    def test_rho_skip_reuses_native_without_second_shoulder(self) -> None:
        formation = _formation()
        tone = _tone(4.0)
        rgb = np.linspace(0.05, 6.0, 32, dtype=np.float32).reshape(32, 1)
        rgb = np.repeat(rgb, 3, axis=1)
        params = curve_params_from_plan(formation)
        native, reference = apply_hdr_curve_pair(
            rgb, tone, formation, need_reference=False, body_params=params
        )
        self.assertIs(native, reference)
        expected = apply_hdr_curve(rgb, tone, formation, body_params=params)
        np.testing.assert_array_equal(native, expected)

    def test_curve_params_cache_is_stable(self) -> None:
        formation = _formation()
        a = curve_params_from_plan(formation)
        b = curve_params_from_plan(formation)
        self.assertIs(a, b)


class HdrCurveTableTests(unittest.TestCase):
    """§12.3 gates: the runtime table against the analytic evaluator it replaces.

    max absolute linear error <= 2e-5, max output-stop error <= 1e-3 EV outside the
    near-zero region, and the pivot/endpoints must be exact to well below the image
    contracts that consume them (EV0 == 0.18 within 1e-5).
    """

    def _dense_ev_probe(self, tone: HdrToneCurve) -> np.ndarray:
        # Off-grid probe: dense uniform sweep plus jitter so samples fall between
        # table nodes, covering toe, body, knee neighbourhood and shoulder-to-peak.
        rng = np.random.default_rng(7)
        ev = np.linspace(tone.black_ev, tone.white_ev + 1.0, 200_001)
        ev = ev + rng.uniform(-4e-4, 4e-4, ev.shape)
        return ev.astype(np.float32)

    def test_non_finite_ev_contract_is_platform_independent(self) -> None:
        table = HdrCurveTable(
            -1.0,
            1.0,
            np.asarray([0.1, 0.5, 0.9], dtype=np.float32),
        )
        out = table.apply_to_ev(
            np.asarray([np.nan, -np.inf, np.inf], dtype=np.float32)
        )
        self.assertTrue(np.isnan(out[0]))
        self.assertEqual(out[1], table.values[0])
        self.assertEqual(out[2], table.values[-1])

    def _assert_table_matches_oracle(self, peak_linear: float) -> None:
        from dngscan.hdr_curve import compile_hdr_curve_table

        formation = _formation()
        tone = _tone(peak_linear)
        params = curve_params_from_plan(formation)
        table = compile_hdr_curve_table(tone, formation, body_params=params)
        ev = self._dense_ev_probe(tone)
        rgb = (SCENE_MIDGRAY * np.exp2(ev, dtype=np.float64)).astype(np.float32)
        oracle = apply_hdr_curve(
            rgb.reshape(-1, 1), tone, formation, body_params=params
        ).reshape(-1)
        approx = table.apply(rgb.reshape(-1, 1)).reshape(-1)

        linear_error = float(np.max(np.abs(approx - oracle)))
        self.assertLessEqual(linear_error, 2e-5)

        meaningful = oracle > 1e-4
        stop_error = float(
            np.max(
                np.abs(
                    np.log2(np.maximum(approx[meaningful], 1e-12))
                    - np.log2(np.maximum(oracle[meaningful], 1e-12))
                )
            )
        )
        self.assertLessEqual(stop_error, 1e-3)

    def test_native_table_matches_analytic_oracle(self) -> None:
        self._assert_table_matches_oracle(4.0)

    def test_reference_table_matches_analytic_oracle(self) -> None:
        from dngscan.hdr_curve import compile_hdr_curve_table

        formation = _formation()
        tone = _tone(4.0)
        params = curve_params_from_plan(formation)
        table = compile_hdr_curve_table(
            tone, formation, peak_linear=1.0, body_params=params
        )
        ev = self._dense_ev_probe(tone)
        rgb = (SCENE_MIDGRAY * np.exp2(ev, dtype=np.float64)).astype(np.float32)
        oracle = apply_hdr_curve(
            rgb.reshape(-1, 1), tone, formation, peak_linear=1.0, body_params=params
        ).reshape(-1)
        approx = table.apply(rgb.reshape(-1, 1)).reshape(-1)
        self.assertLessEqual(float(np.max(np.abs(approx - oracle))), 2e-5)

    def test_pivot_and_endpoints_are_exact_enough(self) -> None:
        from dngscan.hdr_curve import compile_hdr_curve_table

        formation = _formation()
        tone = _tone(4.0)
        params = curve_params_from_plan(formation)
        table = compile_hdr_curve_table(tone, formation, body_params=params)
        probe = np.array(
            [[SCENE_MIDGRAY] * 3, [tone.peak_linear * 4.0] * 3], dtype=np.float32
        )
        out = table.apply(probe)
        oracle = apply_hdr_curve(probe, tone, formation, body_params=params)
        self.assertLess(abs(float(out[0, 0]) - float(oracle[0, 0])), 1e-5)
        # Above W the curve clamps at the content peak; edge clamping must be exact.
        self.assertEqual(float(out[1, 0]), float(oracle[1, 0]))

    def test_pair_tables_alias_when_reference_cannot_contribute(self) -> None:
        from dngscan.hdr_curve import compile_hdr_curve_table_pair

        formation = _formation()
        tone = _tone(4.0)
        params = curve_params_from_plan(formation)
        native, reference = compile_hdr_curve_table_pair(
            tone, formation, need_reference=False, body_params=params
        )
        self.assertIs(native, reference)
        native2, reference2 = compile_hdr_curve_table_pair(
            tone, formation, need_reference=True, body_params=params
        )
        self.assertIsNot(native2, reference2)


if __name__ == "__main__":
    unittest.main()
