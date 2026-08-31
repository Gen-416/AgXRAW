# SPDX-License-Identifier: GPL-3.0-or-later
"""External review batch 21 (2026-08-31): pins for each verified finding.

1. Ultrahdr lost view_brightness on the HDR leg: the SDR path raises its
   rendition to look_brightness_power(view_brightness) while the HDR body
   ran the bare C1 curve, so a gain map encoded the whole lift as "HDR
   renders darker" (the dark-scene auto reaches ~1.3).
2. Gated guidance evidence: the linear render path built raw guidance
   without the Analysis while the streaming u8 path passed it — the same
   plan could render differently by output route.
3. Export fingerprint: the input file was not a fingerprinted parameter, so
   two RAWs sharing a stem overwrote each other; 6 hex was also inside
   birthday-collision range.
4. Audit plan medium: hardcoded "print_paper" where the runtime resolves
   the stock's default_medium — EVERY negative stock's default differs.
5. film_mode=full without a preset silently rendered plain.
6. An unbaked --film-print-medium survived plan compile and export naming,
   failing only mid-render.
7. auto_ev defaulted film_crossover to "off" where the compiler treats
   None as "let the appearance recipe declare".
8. ToneCompressionPlan was mutable while cached and shared.
9. scene_render_to_hdr_display_linear accepted film-takeover plans the
   pair entry refuses (the SDR developer would differ from the HDR one).
10. The preview pixel cache key omitted the optics budget tier the export
    fingerprint already carries.
"""
from __future__ import annotations

import dataclasses
import inspect
import unittest
from types import SimpleNamespace

import numpy as np

from dngscan.constants import SCENE_MIDGRAY
from dngscan.drt import c1_value_and_derivative_at_ev
from dngscan.hdr_agx_math import OUTPUT_REFERENCE_WHITE_STOPS, compile_hdr_shoulder
from dngscan.hdr_curve import apply_hdr_curve, body_brightness_power
from dngscan.models import HdrToneCurve, ToneCompressionPlan


def _formation(view_brightness: float = 1.0) -> ToneCompressionPlan:
    return ToneCompressionPlan(
        target_gamut="Rec2020", luma_p1=0.01, luma_p50=0.18, luma_p99=1.0,
        luma_p999=2.0, black_ev=-7.0, white_ev=4.5, dynamic_range_ev=11.5,
        contrast=3.0, toe_power=1.5, shoulder_power=3.3, chroma_p95=0.0,
        negative_rgb_pct=0.0, over_rgb_pct=0.0, toe_start_ev=-3.0,
        shoulder_start_ev=0.2, use_c1_endpoints=True,
        view_brightness=view_brightness,
    )


def _tone_for(form: ToneCompressionPlan, peak_linear: float = 4.0) -> HdrToneCurve:
    """Solve the shoulder against the SAME composed body the compiler uses."""
    knee, white = 0.2, 4.138
    power = body_brightness_power(form)

    def anchor(ev: float) -> tuple[float, float]:
        value, slope = c1_value_and_derivative_at_ev(ev, form)
        if power != 1.0:
            base = max(float(value), 1e-12)
            value = base ** power
            slope = float(slope) * power * base ** (power - 1.0)
        return value, slope

    peak_stops = OUTPUT_REFERENCE_WHITE_STOPS + float(np.log2(peak_linear))
    segments = compile_hdr_shoulder(
        knee, white, peak_stops, 3.0,
        evaluate_body_with_derivative=anchor, allow_subdivision=True,
    )
    return HdrToneCurve(
        black_ev=-7.0, shoulder_start_ev=knee, white_ev=white,
        body_gamma=2.2, body_contrast=3.0, toe_power=1.5,
        reference_white_stops=OUTPUT_REFERENCE_WHITE_STOPS,
        display_headroom_ev=2.0, requested_headroom_ev=2.0,
        rendered_headroom_ev=2.0, peak_linear=peak_linear,
        reliable_tail_ev=4.0, white_margin_ev=0.5,
        shoulder_segments=tuple(segments), shoulder_alpha=1.0,
    )


class HdrViewBrightnessTests(unittest.TestCase):
    def test_hdr_body_equals_the_sdr_rendition(self) -> None:
        """Below the join the pair's gain must be 1: the HDR body carries the
        same brightness power the SDR formation applies."""
        from dngscan import agx

        for vb in (1.0, 1.3, 0.84):
            with self.subTest(view_brightness=vb):
                form = _formation(vb)
                tone = _tone_for(form)
                ev = np.linspace(-6.5, 0.1, 240, dtype=np.float32)
                rgb = (SCENE_MIDGRAY * np.exp2(ev))[:, None].repeat(3, axis=1)
                hdr = apply_hdr_curve(rgb, tone, form)
                sdr = agx.apply_formation_curve(rgb.astype(np.float32), form)
                np.testing.assert_allclose(hdr, sdr, atol=1e-7)

    def test_shoulder_join_stays_c1_under_brightness(self) -> None:
        """The knee anchor composes the power via the chain rule, so the C1
        join must be as clean at brightness 1.3 as at 1.0."""
        def slope_jump(vb: float) -> float:
            form = _formation(vb)
            tone = _tone_for(form)
            d = 1e-3
            evs = np.array([0.2 - 2 * d, 0.2 - d, 0.2 + d, 0.2 + 2 * d])
            out = apply_hdr_curve(
                (SCENE_MIDGRAY * np.exp2(evs))[:, None].repeat(3, axis=1),
                tone, form,
            )[:, 0]
            below = (out[1] - out[0]) / d
            above = (out[3] - out[2]) / d
            return abs(above / below - 1.0)

        baseline = slope_jump(1.0)
        for vb in (1.3, 0.84):
            self.assertLess(
                slope_jump(vb), baseline + 0.01,
                f"brightness {vb} must not introduce a knee kink",
            )

    def test_brightness_one_keeps_the_bare_body(self) -> None:
        self.assertEqual(body_brightness_power(_formation(1.0)), 1.0)
        self.assertEqual(body_brightness_power(_formation(1.0 + 5e-7)), 1.0)
        self.assertNotEqual(body_brightness_power(_formation(1.3)), 1.0)

    def test_compiler_anchor_composes_the_power(self) -> None:
        src = inspect.getsource(__import__(
            "dngscan.hdr_agx_plan", fromlist=["compile_hdr_agx_plan"]
        ).compile_hdr_agx_plan)
        self.assertIn("body_brightness_power", src,
                      "the shoulder solve must anchor on the composed body")


class GatedGuidanceEvidenceTests(unittest.TestCase):
    def test_linear_path_passes_the_analysis(self) -> None:
        from dngscan.render import render_output_linear, scene_render_to_display_linear

        self.assertIn(
            "analysis",
            inspect.signature(scene_render_to_display_linear).parameters,
        )
        src = inspect.getsource(scene_render_to_display_linear)
        call = src[src.find("raw_guidance_for_shape("):][:160]
        self.assertIn("analysis", call,
                      "the linear path must build guidance on the same "
                      "evidence as the streaming u8 path")
        src = inspect.getsource(render_output_linear)
        self.assertIn("analysis=analysis", src)


class ExportNamingTests(unittest.TestCase):
    def test_fingerprint_carries_the_input_identity(self) -> None:
        from dngscan.gui import service

        src = inspect.getsource(service.run_export)
        call = src[src.find("fingerprint = export_plan_fingerprint("):]
        call = call[:call.find(")\n")]
        for needle in ("input_path", "input_size"):
            self.assertIn(
                needle, call,
                "two RAWs sharing a stem must not share an output path",
            )


class FilmPlanContractTests(unittest.TestCase):
    def _scene(self):
        from tests.golden_support import build_daylight_wide_dr

        return build_daylight_wide_dr()

    def test_audit_medium_is_the_stock_default(self) -> None:
        from dngscan.tone import build_render_plan, _default_medium_for

        scene = self._scene()
        plan = build_render_plan(
            scene.bundle, scene.analysis, "agx", "srgb",
            film_curve="portra400", film_mode="full", film_crossover="datasheet",
        )
        print_plan = plan.film[2]
        expected = _default_medium_for("portra400")
        self.assertNotEqual(expected, "print_paper",
                            "portra400's baked default must differ or this "
                            "test pins nothing")
        self.assertEqual(print_plan.medium_id, expected)

    def test_full_mode_without_a_preset_refuses(self) -> None:
        from dngscan.tone import build_render_plan

        scene = self._scene()
        with self.assertRaisesRegex(ValueError, "预设"):
            build_render_plan(
                scene.bundle, scene.analysis, "agx", "srgb",
                film_curve="none", film_mode="full",
            )

    def test_unbaked_print_medium_refuses_at_compile(self) -> None:
        from dngscan.tone import build_render_plan

        scene = self._scene()
        with self.assertRaisesRegex(ValueError, "未烘焙印相介质"):
            build_render_plan(
                scene.bundle, scene.analysis, "agx", "srgb",
                film_curve="portra400", film_mode="full",
                film_crossover="datasheet",
                film_print_medium="no_such_paper",
            )

    def test_auto_ev_lets_the_compiler_resolve_crossover(self) -> None:
        from dngscan import auto_ev

        for name, fn in inspect.getmembers(auto_ev, inspect.isfunction):
            params = inspect.signature(fn).parameters
            if "film_crossover" in params:
                self.assertIsNone(
                    params["film_crossover"].default,
                    f"{name} must pass None through so reference-appearance "
                    "plans keep their declared neutralization",
                )


class PlanImmutabilityTests(unittest.TestCase):
    def test_tone_plan_is_frozen(self) -> None:
        self.assertTrue(ToneCompressionPlan.__dataclass_params__.frozen)
        plan = _formation()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.contrast = 2.0  # type: ignore[misc]


class HdrFilmGuardTests(unittest.TestCase):
    def test_hdr_formation_refuses_takeover_plans(self) -> None:
        from dngscan.hdr_agx import scene_render_to_hdr_display_linear

        plan = SimpleNamespace(
            tone_core="agx", film_mode="full", curve_preset="portra400"
        )
        with self.assertRaisesRegex(RuntimeError, "胶片接管"):
            scene_render_to_hdr_display_linear(None, plan, None)


class PreviewCacheKeyTests(unittest.TestCase):
    def test_pixel_key_carries_the_optics_budget_tier(self) -> None:
        import os
        from unittest import mock

        from dngscan.gui import service

        bundle = SimpleNamespace(scene_scale=1.0, scene_decoder_runtime="")

        def key() -> tuple:
            return service._preview_pixel_key(
                bundle, "srgb", 0.0, "none", 1.0, "none", 1.0, "none", 1.0,
                1.0, "agx", "y", "base", "none", "none", None,
            )

        with mock.patch.dict(os.environ, {"DNGSCAN_OPTICS_BUDGET_MIB": "512"}):
            low = key()
        with mock.patch.dict(os.environ, {"DNGSCAN_OPTICS_BUDGET_MIB": "1024"}):
            high = key()
        self.assertNotEqual(
            low, high,
            "a preview cached under one spread-grid tier must not serve "
            "the other",
        )


if __name__ == "__main__":
    unittest.main()
