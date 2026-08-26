# SPDX-License-Identifier: GPL-3.0-or-later
"""Route-A sensor SNR coordinates and the evidence-mode toe binding.

EV_SNR1/10/20 turn the prior's electron-domain gain and read noise into scene
EV coordinates (analysis.snr_ev_coordinates); the evidence endpoint mode then
re-solves the toe so the compiled curve's near-black crossing lands on
EV_SNR10 — the toe rolls off across the sensor's own noisy band and SNR>10
shadows sit on the body. Adaptive mode is untouched by construction.
"""
from __future__ import annotations

import dataclasses
import math
import unittest

from dngscan.analysis import snr_ev_coordinates
from dngscan.constants import MIDGRAY_HEADROOM_STOPS


def _analysis_with_prior(base, *, read_e: float, fullwell_e: float):
    nf = 1e-4
    return dataclasses.replace(
        base,
        noise_floor=nf,
        noise_floor_e=fullwell_e * nf,
        prior_read_noise_e=read_e,
    )


class SnrCoordinateMathTests(unittest.TestCase):
    def test_electron_levels_solve_the_snr_equation(self) -> None:
        for r in (1.5, 3.0, 8.0):
            for s in (1.0, 10.0, 20.0):
                n = 0.5 * (s * s + s * math.sqrt(s * s + 4.0 * r * r))
                self.assertAlmostEqual(n / math.sqrt(n + r * r), s, places=9)

    def test_coordinates_are_ordered_and_follow_the_convention(self) -> None:
        from tests.golden_support import build_daylight_wide_dr

        base = build_daylight_wide_dr().analysis
        analysis = _analysis_with_prior(base, read_e=3.0, fullwell_e=60000.0)
        coords = snr_ev_coordinates(analysis)
        self.assertIsNotNone(coords)
        self.assertLess(coords["snr1"], coords["snr10"])
        self.assertLess(coords["snr10"], coords["snr20"])
        r2 = 9.0
        n10 = 0.5 * (100.0 + 10.0 * math.sqrt(100.0 + 4.0 * r2))
        expected = MIDGRAY_HEADROOM_STOPS + math.log2(n10 / 60000.0)
        self.assertAlmostEqual(coords["snr10"], expected, places=9)

    def test_fails_closed_without_prior_evidence(self) -> None:
        from tests.golden_support import build_daylight_wide_dr

        base = build_daylight_wide_dr().analysis
        analysis = dataclasses.replace(base, prior_read_noise_e=None)
        self.assertIsNone(snr_ev_coordinates(analysis))

    def test_fails_closed_when_a_tier_reaches_the_full_well(self) -> None:
        from tests.golden_support import build_daylight_wide_dr

        base = build_daylight_wide_dr().analysis
        # Full well of 300 e- with 20 e- read noise: N_20 ~ 610 e- > full well.
        analysis = _analysis_with_prior(base, read_e=20.0, fullwell_e=300.0)
        self.assertIsNone(snr_ev_coordinates(analysis))


class EvidenceModeToeBindingTests(unittest.TestCase):
    READ_E = 3.0
    FULLWELL_E = 60000.0

    def _plans(self):
        from tests.golden_support import build_daylight_wide_dr
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        analysis = _analysis_with_prior(
            scene.analysis, read_e=self.READ_E, fullwell_e=self.FULLWELL_E
        )
        evidence = build_render_plan(
            scene.bundle, analysis, "agx", "srgb", endpoint_mode="evidence"
        )
        adaptive = build_render_plan(scene.bundle, analysis, "agx", "srgb")
        return analysis, evidence, adaptive

    def test_evidence_toe_end_lands_on_snr10(self) -> None:
        from dngscan import drt

        analysis, evidence, _ = self._plans()
        coords = snr_ev_coordinates(analysis)
        self.assertIsNotNone(coords)
        crossing = drt.compiled_curve_transitions(evidence.tone)["toe_end_ev"]
        self.assertIsNotNone(crossing)
        # The solver clamps its target into the compiled curve's reachable
        # window; assert against the same clamped target it was given.
        target = min(
            -0.25, max(float(evidence.tone.black_ev) + 0.05, coords["snr10"])
        )
        self.assertLess(
            abs(float(crossing) - target), 0.05,
            f"toe end {crossing} did not land on the SNR10 target {target}",
        )
        self.assertIn("EV_SNR10", evidence.tone.endpoint_note or "")

    def test_adaptive_mode_is_untouched(self) -> None:
        _, evidence, adaptive = self._plans()
        self.assertIsNone(adaptive.tone.endpoint_note)
        self.assertNotEqual(
            float(adaptive.tone.toe_power), float(evidence.tone.toe_power)
        )

    def test_missing_coordinates_are_declared_not_guessed(self) -> None:
        from tests.golden_support import build_daylight_wide_dr
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        analysis = dataclasses.replace(scene.analysis, prior_read_noise_e=None)
        plan = build_render_plan(
            scene.bundle, analysis, "agx", "srgb", endpoint_mode="evidence"
        )
        self.assertIn("SNR 坐标证据缺席", plan.tone.endpoint_note or "")


if __name__ == "__main__":
    unittest.main()
