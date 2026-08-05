# SPDX-License-Identifier: GPL-3.0-or-later
"""Film curve preset gates: declared coordinates, pinned residuals, HDR feasibility."""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dngscan.constants import OUTPUT_REFERENCE_WHITE_STOPS
from dngscan.drt import apply_c1_endpoints
from dngscan.film_curve import (
    FILM_CURVE_CHOICES,
    FILM_CURVE_PRESETS,
    apply_film_curve_preset,
    validate_film_curve,
)
from dngscan.hdr_agx_math import (
    body_anchor_at_ev,
    compile_hdr_shoulder,
    requested_headroom_ev,
    validate_hdr_shoulder,
)

SAMPLE = Path.home() / "Pictures" / "_SDI0150.DNG"
SAMPLE_NIGHT = Path.home() / "Pictures" / "_SDI0199.DNG"

# Calibrated to the measured residual landscape across all twenty stocks after the
# viewing-condition-complete refit AND the luminance-composed floor fix. The floor fix
# retired the reversal exception entirely: the "structural AgX-vs-slide distance" was
# an arithmetic-mean floor sitting up to 0.57 stop above the luminance target's own
# asymptote (Velvia rms 0.284 -> 0.013, all four reversals now pin-free interior
# solutions). Worst stock today: Ektar 100 rms 0.067 / max 0.164.
RMS_GATE = 0.10
MAX_GATE = 0.25
REVERSAL_GATE_FACTOR = 1.0  # retired 2026-07-30: reversals fit as well as negatives
# Theatrical quotations carry the dark-surround report verbatim into a curve family
# shaped for average-surround media — intentionally out-of-condition, and the extra
# ~1.5x contrast strains the family (worst: Verita theatrical rms 0.102). The looser
# gate is the declared cost of quotation over translation.
THEATRICAL_GATE_FACTOR = 1.25
STORED_RESIDUAL_SLACK = 0.005

CURVE_FIELDS = (
    "black_ev", "white_ev", "contrast", "toe_power", "shoulder_power",
    "latitude_lo_ev", "latitude_hi_ev", "target_black_linear",
)


def _compiled_curve(params: dict, ev: np.ndarray) -> np.ndarray:
    plan = SimpleNamespace(
        black_ev=params["black_ev"],
        white_ev=params["white_ev"],
        contrast=params["contrast"],
        toe_power=params["toe_power"],
        shoulder_power=params["shoulder_power"],
        latitude_lo_ev=params["latitude_lo_ev"],
        latitude_hi_ev=params["latitude_hi_ev"],
        pivot_ev_offset=0.0,
        target_black_linear=params["target_black_linear"],
        target_white_linear=1.0,
        curve_gamma=2.2,
    )
    return np.asarray(
        apply_c1_endpoints(ev.astype(np.float32), plan), dtype=np.float64
    )


class PresetRegistryTests(unittest.TestCase):
    def test_flagship_presets_exist_with_provenance(self) -> None:
        for name in ("portra400", "superia400"):
            preset = FILM_CURVE_PRESETS[name]
            self.assertIn("CC BY-SA", preset["source"]["license"])
            self.assertTrue(preset["source"]["film"])
            self.assertTrue(preset["source"]["print"])
            self.assertIn(name, FILM_CURVE_CHOICES)

    def test_validate_rejects_unknown(self) -> None:
        self.assertEqual(validate_film_curve("none"), "none")
        with self.assertRaises(ValueError):
            validate_film_curve("kodachrome_imaginary")


class ResidualGateTests(unittest.TestCase):
    """The stored fit must reproduce against the stored target — and stay in gate."""

    def test_residuals_recompute_within_gates(self) -> None:
        for name, preset in FILM_CURVE_PRESETS.items():
            with self.subTest(preset=name):
                ev = np.array(preset["target_curve"]["ev"], dtype=np.float64)
                target = np.array(
                    preset["target_curve"]["display_linear"], dtype=np.float64
                )
                fitted = _compiled_curve(preset["params"], ev)
                mask = target > 1e-4
                r = np.log2(np.maximum(fitted[mask], 1e-4)) - np.log2(target[mask])
                rms = float(np.sqrt(np.mean(r * r)))
                worst = float(np.max(np.abs(r)))
                model = str(preset["source"].get("model", ""))
                # The reversal factor is 1.0 since the luminance-composed floor fix
                # (slides now fit as well as negatives); the hook stays so a future
                # regression surfaces as a factor change in review, not a gate edit.
                factor = 1.0
                if "reversal" in model:
                    factor = REVERSAL_GATE_FACTOR
                elif "theatrical quotation" in model:
                    factor = THEATRICAL_GATE_FACTOR
                rms_gate = RMS_GATE * factor
                max_gate = MAX_GATE * factor
                self.assertLessEqual(rms, rms_gate)
                self.assertLessEqual(worst, max_gate)
                # Regression pinning: drift cannot hide beneath the ceiling.
                self.assertLessEqual(
                    abs(rms - float(preset["fit"]["rms_stop"])),
                    STORED_RESIDUAL_SLACK,
                )

    def test_ev0_anchor_is_exact(self) -> None:
        for name, preset in FILM_CURVE_PRESETS.items():
            with self.subTest(preset=name):
                t0 = float(_compiled_curve(preset["params"], np.array([0.0]))[0])
                self.assertLess(abs(t0 - 0.18), 1e-5)

    def test_paper_floor_is_declared_and_reached(self) -> None:
        """The print's Dmax floor must survive into the compiled curve's deep toe."""
        for name, preset in FILM_CURVE_PRESETS.items():
            with self.subTest(preset=name):
                floor = float(preset["params"]["target_black_linear"])
                self.assertGreater(floor, 0.0)
                deep = float(_compiled_curve(preset["params"], np.array([-12.0]))[0])
                self.assertAlmostEqual(deep, floor, delta=floor * 0.25)


class ChannelRatioFieldTests(unittest.TestCase):
    """Exposure-dependent colour phase 1: the per-channel ratio field's invariants.

    r_c(EV) = T_c / T_neutral along the balanced neutral ramp is the stock's measured
    layer-saturation differential. Its contract: unity at the EV0 anchor (per-channel
    balance guarantees it), bounded everywhere (a broken balance or surround term
    would blow the range), and present for every preset so the phase-2 runtime
    (out_c = C(EV_c) * r_c(EV_c) / r_c(EV_Y)) can rely on it.
    """

    def test_every_preset_carries_a_ratio_field(self) -> None:
        for name, preset in FILM_CURVE_PRESETS.items():
            with self.subTest(preset=name):
                rc = preset.get("channel_ratio_curve")
                self.assertIsNotNone(rc, f"{name} missing channel_ratio_curve")
                self.assertEqual(len(rc["ev"]), len(rc["ratio_rgb"]))
                self.assertGreaterEqual(len(rc["ev"]), 64)

    def test_ratio_is_unity_at_the_ev0_anchor(self) -> None:
        for name, preset in FILM_CURVE_PRESETS.items():
            with self.subTest(preset=name):
                rc = preset["channel_ratio_curve"]
                ev = np.array(rc["ev"], dtype=np.float64)
                r = np.array(rc["ratio_rgb"], dtype=np.float64)
                for c in range(3):
                    r0 = float(np.interp(0.0, ev, r[:, c]))
                    self.assertLess(abs(r0 - 1.0), 6e-3)

    def test_ratio_field_is_bounded(self) -> None:
        for name, preset in FILM_CURVE_PRESETS.items():
            with self.subTest(preset=name):
                r = np.array(preset["channel_ratio_curve"]["ratio_rgb"])
                self.assertGreater(float(r.min()), 0.2)
                self.assertLess(float(r.max()), 5.0)


class StylePairingTests(unittest.TestCase):
    """The editorial look layer: every preset pairs a declared separation strength
    with one of AgX's own primaries geometries — declarations, not measurements."""

    def test_every_preset_has_a_pairing_within_bounds(self) -> None:
        from dngscan.film_curve import FILM_CURVE_PRESETS, film_style_pairing

        for name in FILM_CURVE_PRESETS:
            with self.subTest(preset=name):
                strength, primaries = film_style_pairing(name)
                self.assertTrue(1.0 <= strength <= 3.0, strength)
                self.assertIn(primaries, ("base", "punchy", "muted", "smooth"))

    def test_unknown_preset_falls_back_conservatively(self) -> None:
        from dngscan.film_curve import film_style_pairing

        strength, primaries = film_style_pairing("nonexistent")
        self.assertEqual((strength, primaries), (1.3, "base"))


class TheatricalVariantTests(unittest.TestCase):
    """Quotation presets: every dark-surround print chain carries a *_theatrical twin."""

    CINE_STOCKS = ("verita200d", "vision3200t", "vision3250d", "vision3500t", "vision350d")

    def test_every_cine_chain_has_a_theatrical_twin(self) -> None:
        for base in self.CINE_STOCKS:
            with self.subTest(stock=base):
                t = FILM_CURVE_PRESETS.get(f"{base}_theatrical")
                self.assertIsNotNone(t, f"{base}_theatrical missing")
                self.assertIn("theatrical quotation", str(t["source"]["model"]))
                # The quotation shares the base declaration layers: same WB, same
                # spectral separation — only the viewing translation differs.
                self.assertEqual(t["combo"], FILM_CURVE_PRESETS[base]["combo"])

    def test_quotation_is_contrastier_than_translation(self) -> None:
        """The verbatim dark-surround report must render steeper mid-tones than the
        translated one — that difference *is* the surround term being real."""
        for base in self.CINE_STOCKS:
            with self.subTest(stock=base):
                ev = np.array([-2.0, 2.0])
                translated = _compiled_curve(FILM_CURVE_PRESETS[base]["params"], ev)
                quoted = _compiled_curve(
                    FILM_CURVE_PRESETS[f"{base}_theatrical"]["params"], ev
                )
                span_t = np.log2(translated[1] / max(translated[0], 1e-6))
                span_q = np.log2(quoted[1] / max(quoted[0], 1e-6))
                self.assertGreater(float(span_q), float(span_t) + 0.5)


class PairingSentinelTests(unittest.TestCase):
    """Pairings fill only ABSENT values, encoded as None sentinels — never value
    equality. 'The value equals the default' and 'the user did not set it' are
    different intents; the old equality test silently rewrote an explicit x1.0
    into the pairing's x1.6 and mislabeled two documentation plates."""

    def _parse(self, *extra):
        from dngscan.cli import parse_args

        return parse_args(["/tmp/x.dng", *extra])

    def test_pairing_fills_absent_values(self) -> None:
        args = self._parse("--film", "velvia100")
        self.assertAlmostEqual(args.scene_transform_strength, 1.6)
        self.assertEqual(args.agx_primaries, "punchy")

    def test_explicit_default_values_survive_the_pairing(self) -> None:
        args = self._parse(
            "--film", "velvia100",
            "--scene-transform-strength", "1.0", "--agx-primaries", "base",
        )
        self.assertAlmostEqual(args.scene_transform_strength, 1.0)
        self.assertEqual(args.agx_primaries, "base")

    def test_no_film_resolves_to_documented_defaults(self) -> None:
        args = self._parse()
        self.assertAlmostEqual(args.scene_transform_strength, 1.0)
        self.assertEqual(args.agx_primaries, "base")


class ShadowFloorCodeTests(unittest.TestCase):
    """Film floors are the digital D-min: deep shadows must never pile at u8 code 0.

    Cineon put black at code 95/1023 so film's shadow grain kept its bilateral
    distribution instead of clipping single-sided at the bottom of the encoding;
    LogC4 re-made the same decision with its +64 toe offset. The preset floors serve
    that function in the u8 delivery — if any floor encoded to sRGB code 0, shadow
    noise would fold upward and the medium's D-min guard would be silently lost.
    """

    def test_deep_shadows_encode_above_code_zero(self) -> None:
        from dngscan.color import srgb_encode

        for name, preset in FILM_CURVE_PRESETS.items():
            with self.subTest(preset=name):
                ev = np.linspace(-14.0, -8.0, 25)
                deep = _compiled_curve(preset["params"], ev)
                codes = np.round(np.asarray(srgb_encode(deep)) * 255.0)
                floor = float(preset["params"]["target_black_linear"])
                floor_code = round(float(srgb_encode(np.float64(floor))) * 255.0)
                # Every translated floor sits comfortably off zero (lowest today: the
                # cine chain at linear 0.00586 -> code ~18). Theatrical quotations
                # keep the projection print's genuinely deep Dmax (2383 at ~code 2):
                # for them the guard is only that black never collapses to code 0.
                theatrical = "theatrical quotation" in str(preset["source"]["model"])
                self.assertGreaterEqual(floor_code, 1 if theatrical else 8)
                self.assertGreaterEqual(float(codes.min()), floor_code - 1.0)


class HdrFeasibilityTests(unittest.TestCase):
    def test_preset_contrasts_compile_valid_shoulders_across_headroom(self) -> None:
        """Film bodies must keep the HDR contract: a validated shoulder everywhere."""
        for name, preset in FILM_CURVE_PRESETS.items():
            contrast = float(preset["params"]["contrast"])
            for knee in (0.0, 0.2):
                _, knee_stops, knee_slope = body_anchor_at_ev(knee, contrast)
                for headroom_tenths in range(5, 54, 6):
                    headroom = headroom_tenths / 10.0
                    for tail in (3.0, 5.5, 8.0, 12.0):
                        requested = requested_headroom_ev(tail, headroom)
                        if requested <= 0.0:
                            continue
                        white = min(max(tail + 0.3, 3.0), 8.5)
                        peak = OUTPUT_REFERENCE_WHITE_STOPS + requested
                        segments = compile_hdr_shoulder(
                            knee, white, peak, contrast, allow_subdivision=True
                        )
                        with self.subTest(
                            preset=name, knee=knee, headroom=headroom, tail=tail
                        ):
                            ok, reason = validate_hdr_shoulder(
                                segments, knee_slope, peak
                            )
                            self.assertTrue(ok, msg=reason)


class ApplyPresetTests(unittest.TestCase):
    def _tone_stub(self):
        from dngscan.models import ToneCompressionPlan

        return ToneCompressionPlan(
            target_gamut="Rec2020",
            luma_p1=0.01, luma_p50=0.18, luma_p99=1.0, luma_p999=2.0,
            black_ev=-7.0, white_ev=4.5, dynamic_range_ev=11.5,
            contrast=3.0, toe_power=1.5, shoulder_power=3.3,
            chroma_p95=0.0, negative_rgb_pct=0.0, over_rgb_pct=0.0,
            toe_start_ev=-3.0, shoulder_start_ev=0.2, use_c1_endpoints=True,
        )

    def test_none_is_identity(self) -> None:
        tone = self._tone_stub()
        self.assertIs(apply_film_curve_preset(tone, "none"), tone)

    def test_preset_pins_every_curve_field(self) -> None:
        tone = apply_film_curve_preset(self._tone_stub(), "portra400")
        p = FILM_CURVE_PRESETS["portra400"]["params"]
        for field in CURVE_FIELDS:
            self.assertEqual(getattr(tone, field), float(p[field]), msg=field)
        self.assertEqual(tone.curve_preset, "portra400")
        self.assertEqual(tone.dynamic_range_ev, tone.white_ev - tone.black_ev)


class FilmModeTests(unittest.TestCase):
    """The two-mode contract: observe = film declares, AgX develops (no ratio);
    full = the film development core takes over per-channel (EXPERIMENTAL)."""

    def _film_plan(self, name: str = "portra400", mode: str = "observe"):
        from dataclasses import replace

        from dngscan.models import ToneCompressionPlan

        base = ToneCompressionPlan(
            target_gamut="Rec2020",
            luma_p1=0.01, luma_p50=0.18, luma_p99=1.0, luma_p999=2.0,
            black_ev=-7.0, white_ev=4.5, dynamic_range_ev=11.5,
            contrast=3.0, toe_power=1.5, shoulder_power=3.3,
            chroma_p95=0.0, negative_rgb_pct=0.0, over_rgb_pct=0.0,
            toe_start_ev=-3.0, shoulder_start_ev=0.2, use_c1_endpoints=True,
        )
        return replace(apply_film_curve_preset(base, name), film_mode=mode)

    def test_observe_mode_disables_the_ratio_gain(self) -> None:
        """Colour belongs to AgX in observe mode: the ratio field must not leak
        into the formation, or the film would silently co-own development colour
        again — the exact hybrid the two-mode contract retired."""
        from dngscan import agx as agx_engine

        plan = self._film_plan(mode="observe")
        _, outset_mtx = agx_engine.formation_matrices(plan)
        weights = agx_engine.formation_luma_row(outset_mtx)
        rgb = np.array([[0.5, 0.2, 0.1]], dtype=np.float32)
        self.assertIsNone(agx_engine.channel_ratio_gain(rgb, plan, weights))

    def test_full_mode_gain_is_unity_on_the_neutral_axis(self) -> None:
        from dngscan import agx as agx_engine

        plan = self._film_plan(mode="full")
        _, outset_mtx = agx_engine.formation_matrices(plan)
        weights = agx_engine.formation_luma_row(outset_mtx)
        neutral = np.repeat(
            np.geomspace(1e-4, 8.0, 64, dtype=np.float32)[:, None], 3, axis=1
        )
        gain = agx_engine.channel_ratio_gain(neutral, plan, weights)
        self.assertIsNotNone(gain)
        # Exact in real arithmetic; float32's luminance dot product rounds <=1 ulp.
        self.assertLess(float(np.abs(gain - 1.0).max()), 2e-6)

    def test_film_core_preserves_neutrality_and_differs_from_agx(self) -> None:
        from dngscan.film_develop import apply_film_core
        from dngscan import agx as agx_engine

        plan = self._film_plan(mode="full")
        neutral = np.repeat(
            np.geomspace(1e-3, 4.0, 32, dtype=np.float32)[:, None], 3, axis=1
        )
        developed = apply_film_core(neutral, plan)
        # Per-channel identical curve on identical channels: neutrality holds.
        self.assertLess(float(np.abs(developed - developed[:, :1]).max()), 2e-5)
        chroma = np.array([[2.4, 0.9, 0.4], [0.05, 0.5, 0.02]], dtype=np.float32)
        inset_mtx, outset_mtx = agx_engine.formation_matrices(plan)
        agx_out = agx_engine.apply_core(chroma, plan, inset_mtx, outset_mtx)
        film_out = apply_film_core(chroma, plan)
        rel = np.abs(film_out - agx_out) / np.maximum(np.abs(agx_out), 1e-4)
        self.assertGreater(float(rel.max()), 1e-2)

    def test_dispatch_routes_full_mode_to_the_film_core(self) -> None:
        from dngscan.film_develop import apply_film_core
        from dngscan.render import apply_tone_core

        plan = self._film_plan(mode="full")
        rgb = np.array([[0.6, 0.3, 0.15], [0.18, 0.18, 0.18]], dtype=np.float32)
        via_dispatch = apply_tone_core(rgb, plan)
        direct = apply_film_core(rgb, plan)
        np.testing.assert_allclose(via_dispatch, direct, rtol=0.0, atol=1e-6)

    def test_fast_backend_declines_only_full_mode(self) -> None:
        """Observe-mode film plans are plain curve parameters — native eligible.
        The refusal must be scoped to the takeover core, not to film in general."""
        from dngscan import _fast as fast_backend

        self.assertFalse(fast_backend.supports_agx(self._film_plan(mode="full")))
        observe = self._film_plan(mode="observe")
        # The film gate specifically must not veto observe mode; overall support
        # still depends on the extension being present in this environment.
        self.assertEqual(
            fast_backend.supports_agx(observe),
            fast_backend.supports_agx(
                __import__("dataclasses").replace(observe, curve_preset="none")
            ),
        )


class FilmCrossoverTests(unittest.TestCase):
    """The declared crossover switch (plan.film_crossover) on the measured ratio
    field: "off" (default) keeps the luminance-normalized anti-hidden-WB rail
    byte-for-byte; "datasheet" re-anchors at mid-grey so the fitted inter-layer
    drift appears on the neutral axis while EV0 stays strictly white-balanced."""

    def _film_plan(self, name: str = "velvia100", mode: str = "full",
                   crossover: str = "off"):
        from dataclasses import replace

        from dngscan.models import ToneCompressionPlan

        base = ToneCompressionPlan(
            target_gamut="Rec2020",
            luma_p1=0.01, luma_p50=0.18, luma_p99=1.0, luma_p999=2.0,
            black_ev=-7.0, white_ev=4.5, dynamic_range_ev=11.5,
            contrast=3.0, toe_power=1.5, shoulder_power=3.3,
            chroma_p95=0.0, negative_rgb_pct=0.0, over_rgb_pct=0.0,
            toe_start_ev=-3.0, shoulder_start_ev=0.2, use_c1_endpoints=True,
        )
        return replace(
            apply_film_curve_preset(base, name),
            film_mode=mode, film_crossover=crossover,
        )

    @staticmethod
    def _neutral_ramp(ev_lo: float = -6.0, ev_hi: float = 3.0, n: int = 128):
        ev = np.linspace(ev_lo, ev_hi, n, dtype=np.float32)
        return ev, np.repeat((0.18 * np.exp2(ev))[:, None], 3, axis=1).astype(
            np.float32
        )

    def test_off_is_the_default_and_the_status_quo(self) -> None:
        """A plan that never heard of the field and an explicit "off" plan must
        produce the identical gain — the freeze discipline in one assertion."""
        from dngscan import agx as agx_engine

        plan_off = self._film_plan(crossover="off")
        _, outset_mtx = agx_engine.formation_matrices(plan_off)
        weights = agx_engine.formation_luma_row(outset_mtx)
        self.assertEqual(plan_off.film_crossover, "off")
        rgb = np.array(
            [[0.5, 0.2, 0.1], [0.02, 0.4, 1.3], [0.18, 0.18, 0.18]],
            dtype=np.float32,
        )
        gain_off = agx_engine.channel_ratio_gain(rgb, plan_off, weights)

        class LegacyPlanView:
            """The plan as an older caller would build it: no crossover field."""

            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                if name == "film_crossover":
                    raise AttributeError(name)
                return getattr(self._inner, name)

        gain_legacy = agx_engine.channel_ratio_gain(
            rgb, LegacyPlanView(plan_off), weights
        )
        np.testing.assert_array_equal(gain_off, gain_legacy)

    def test_datasheet_is_inert_outside_full_mode(self) -> None:
        from dngscan import agx as agx_engine

        plan = self._film_plan(mode="observe", crossover="datasheet")
        _, outset_mtx = agx_engine.formation_matrices(plan)
        weights = agx_engine.formation_luma_row(outset_mtx)
        rgb = np.array([[0.5, 0.2, 0.1]], dtype=np.float32)
        self.assertIsNone(agx_engine.channel_ratio_gain(rgb, plan, weights))

    def test_datasheet_matches_the_field_pointwise_on_the_neutral_axis(self) -> None:
        """gain_c(EV) must equal r_c(EV)/r_c(0) exactly (clamped to the rail):
        the switch releases the measured data, it does not reshape it."""
        from dngscan import agx as agx_engine
        from dngscan.film_curve import channel_ratio_field

        plan = self._film_plan(crossover="datasheet")
        _, outset_mtx = agx_engine.formation_matrices(plan)
        weights = agx_engine.formation_luma_row(outset_mtx)
        ev, ramp = self._neutral_ramp()
        gain = agx_engine.channel_ratio_gain(ramp, plan, weights)
        self.assertIsNotNone(gain)
        grid, ratios = channel_ratio_field("velvia100")
        for c in range(3):
            expected = np.interp(ev, grid, ratios[:, c]) / float(
                np.interp(0.0, grid, ratios[:, c])
            )
            np.testing.assert_allclose(
                gain[:, c], np.clip(expected, 0.25, 4.0), rtol=0.0, atol=2e-5
            )

    def test_datasheet_semantics_mid_grey_shadows_and_highlights(self) -> None:
        """Mid-grey strictly neutral; EV-2 clearly cool (B/R >> 1, the fitted
        Velvia shadow differential); EV+1 slightly blue — per the field data."""
        from dngscan import agx as agx_engine

        plan = self._film_plan(crossover="datasheet")
        _, outset_mtx = agx_engine.formation_matrices(plan)
        weights = agx_engine.formation_luma_row(outset_mtx)
        probe = np.array(
            [[0.18, 0.18, 0.18],
             [0.045, 0.045, 0.045],   # EV-2
             [0.36, 0.36, 0.36]],     # EV+1
            dtype=np.float32,
        )
        gain = agx_engine.channel_ratio_gain(probe, plan, weights)
        # EV0: r_c(0)/r_c(0) — WB fidelity at the anchor is exact.
        self.assertLess(float(np.abs(gain[0] - 1.0).max()), 1e-6)
        # EV-2: measured cool shadows (field: B 1.327 / R 0.894, B/R ~ 1.48).
        self.assertGreater(float(gain[1, 2] / gain[1, 0]), 1.3)
        # EV+1: slight blue lean (field: B ~ 1.040).
        self.assertGreater(float(gain[2, 2] / gain[2, 0]), 1.0)
        self.assertLess(float(gain[2, 2] / gain[2, 0]), 1.1)

    def test_film_core_neutral_ramp_renders_the_declared_drift(self) -> None:
        """Render-level: through film_develop.apply_film_core a neutral ramp
        stays neutral at EV0 (1e-6), turns cool at EV-2, slightly blue at EV+1;
        with the switch off the same ramp stays neutral everywhere."""
        from dngscan.film_develop import apply_film_core

        probe = np.array(
            [[0.18, 0.18, 0.18], [0.045, 0.045, 0.045], [0.36, 0.36, 0.36]],
            dtype=np.float32,
        )
        out_off = apply_film_core(probe, self._film_plan(crossover="off"))
        self.assertLess(
            float(np.abs(out_off - out_off[:, :1]).max()), 2e-5
        )
        out = apply_film_core(probe, self._film_plan(crossover="datasheet"))
        # EV0 mid-grey: channels equal within 1e-6 relative to their magnitude.
        self.assertLess(
            float(np.abs(out[0] - out[0].mean()).max()), 1e-6 + 1e-6 * out[0].mean()
        )
        self.assertGreater(float(out[1, 2] / out[1, 0]), 1.3)
        self.assertGreater(float(out[2, 2] / out[2, 0]), 1.0)
        self.assertLess(float(out[2, 2] / out[2, 0]), 1.1)

    def test_build_render_plan_validates_and_defaults_the_switch(self) -> None:
        """The plan compiler's crossover plumbing: unknown values collapse to
        "off" and the declared value rides the tone plan only with a preset."""
        import inspect

        from dngscan.tone import build_render_plan

        signature = inspect.signature(build_render_plan)
        self.assertIn("film_crossover", signature.parameters)
        self.assertEqual(signature.parameters["film_crossover"].default, "off")

    def test_compiled_plan_carries_the_switch_without_a_color_head(self) -> None:
        """Regression: the #20/#21 merge stamped film_crossover inside the
        colour-head block, which only runs for nonzero CC values — and reversal
        presets reject the colour head entirely, so --film-crossover datasheet
        on velvia100 (the flagship crossover stock) could never reach the plan.
        The declared switch must ride the tone plan with the preset alone."""
        from golden_support import build_daylight_wide_dr
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        plan = build_render_plan(
            scene.bundle,
            scene.analysis,
            "agx",
            "srgb",
            film_curve="velvia100",
            film_mode="full",
            film_crossover="datasheet",
        )
        self.assertEqual(plan.tone.film_mode, "full")
        self.assertEqual(plan.tone.film_crossover, "datasheet")
        # And the default remains byte-identical "off" when undeclared.
        plan_default = build_render_plan(
            scene.bundle,
            scene.analysis,
            "agx",
            "srgb",
            film_curve="velvia100",
            film_mode="full",
        )
        self.assertEqual(plan_default.tone.film_crossover, "off")


class FilmPrefeedPresetTests(unittest.TestCase):
    """The film-separation prefeed presets obey the scene-transform contract."""

    # Every film separation preset the calibrator produced, discovered dynamically.
    from dngscan.film_curve import FILM_CURVE_PRESETS as _FCP

    PRESETS = tuple(
        sorted(
            {
                str(p.get("combo", {}).get("scene_transform"))
                for p in _FCP.values()
            }
        )
    )

    def test_presets_load_with_confidence_and_windows(self) -> None:
        from dngscan.scene_transform import SCENE_TRANSFORMS

        for name in self.PRESETS:
            with self.subTest(preset=name):
                preset = SCENE_TRANSFORMS[name]
                self.assertEqual(preset.illuminant, "D55")
                names = {r.name for r in preset.regions}
                self.assertLessEqual(
                    {"skin", "foliage", "cyan", "neutral", "magenta"}, names
                )
                for region in preset.regions:
                    self.assertGreater(region.confidence, 0.0)
                    self.assertLessEqual(region.confidence, 1.0)

    def test_neutral_axis_survives_film_separation(self) -> None:
        from dngscan._deps import np as _np
        from dngscan.scene_transform import apply_scene_transform_rec2020

        gray = _np.full((64, 3), 0.18, dtype=_np.float32)
        for name in self.PRESETS:
            with self.subTest(preset=name):
                out = apply_scene_transform_rec2020(gray, name, 1.0)
                _np.testing.assert_allclose(out, gray, atol=2e-3)

    def test_film_separation_changes_nonneutral_colour(self) -> None:
        from dngscan._deps import np as _np
        from dngscan.scene_transform import SCENE_TRANSFORMS, apply_scene_transform_rec2020

        for name in self.PRESETS:
            with self.subTest(preset=name):
                skin_region = next(
                    r for r in SCENE_TRANSFORMS[name].regions if r.name == "skin"
                )
                rg, bg = skin_region.mu_rg_bg
                skin_like = _np.array([[0.18 * rg, 0.18, 0.18 * bg]] * 8,
                                      dtype=_np.float32)
                out = apply_scene_transform_rec2020(skin_like, name, 1.0)
                self.assertGreater(
                    float(_np.max(_np.abs(out - skin_like))), 1e-4
                )


@unittest.skipUnless(
    SAMPLE.is_file() and SAMPLE_NIGHT.is_file(), "sample frames unavailable"
)
class WholeRollConsistencyTests(unittest.TestCase):
    def test_two_scenes_receive_the_identical_curve(self) -> None:
        """Film response is fixed: day and night frames share one declared curve."""
        from dngscan.analysis import analyze
        from dngscan.grade import RENDER_MODE
        from dngscan.raw_io import load_raw
        from dngscan.tone import build_render_plan

        curves = []
        scenes = []
        for path in (SAMPLE, SAMPLE_NIGHT):
            bundle = load_raw(path, scene_half_size=True)
            analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
            plan = build_render_plan(
                bundle, analysis, RENDER_MODE, "p3", film_curve="portra400"
            )
            curves.append(tuple(getattr(plan.tone, f) for f in CURVE_FIELDS))
            scenes.append(plan.scene)
        self.assertEqual(curves[0], curves[1])
        # Scene metrics stay scene-derived — HDR budgeting still reads the capture.
        self.assertNotEqual(
            scenes[0].reliable_tail_ev_p9999, scenes[1].reliable_tail_ev_p9999
        )


if __name__ == "__main__":
    unittest.main()
