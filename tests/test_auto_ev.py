# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dngscan.auto_ev import (
    anchored_median_ev,
    compute_auto_ev,
    is_ev_auto,
    median_align_ev,
    parse_ev_value,
    render_sample_linear_output,
    resolve_export_ev,
    anchored_scene_body_ev,
    scene_body_align_ev,
)
from dngscan._deps import np
from dngscan.models import Analysis, RawBundle, ToneCompressionPlan
from dngscan.tone import compute_exposure_gain


def _minimal_analysis(median_vs_gray_ev: float) -> Analysis:
    return Analysis(
        channel_ids=[0, 1, 2, 3],
        labels={0: "R", 1: "G1", 2: "B", 3: "G2"},
        ceilings={},
        ceil_spike_counts={},
        ceil_near_counts={},
        ceil_spike_ok={},
        fullwell_channel_ids=[0, 1, 3],
        fullwell_note="",
        saturation_levels={},
        channel_fullwell={},
        channel_thresholds={},
        fullwell=16000,
        threshold=15996,
        clip_pct={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        cfa_cell_supported=True,
        cell_union_pct=0.0,
        cell_ge2_of_clipped_pct=0.0,
        cell_k_of_clipped_pct={},
        cell_k_of_all_pct={},
        ev_p1=-8.0,
        ev_raw_p1=-8.0,
        ev_median=-2.0,
        ev_p99=-0.5,
        ev_p999=-0.2,
        ev_dr_p1_p999=7.8,
        ev_floor_hit_pct=0.0,
        median_vs_gray_ev=median_vs_gray_ev,
        median_y=0.05,
        noise_floor=0.002,
        usable_dr_ev=8.5,
        snr_curves={},
        snr1_dr={},
        snr1_stop={},
        gamut_out_pct={"sRGB": 0.0, "P3": 0.0, "Rec2020": 0.0},
        bright_pixel_pct=0.5,
        survivor_channel="B",
        container_bits_est=14,
        prior_id=None,
        gain_e_per_dn=None,
        noise_floor_e=None,
        prior_read_noise_e=None,
        prior_pdr_ev=None,
        usable_dr_eff_ev=8.5,
        health_lag1_corr=0.0,
        health_hist_empty_pct=0.0,
    )


def test_parse_ev_auto_token():
    assert parse_ev_value("auto") == "auto"
    assert parse_ev_value("AUTO") == "auto"
    assert parse_ev_value("1.25") == 1.25
    assert is_ev_auto("auto")


def test_median_align_ev_agx():
    analysis = _minimal_analysis(-1.51)
    ev = median_align_ev("agx", analysis)
    base = compute_exposure_gain("agx", 0.0)
    assert abs(ev - (-analysis.median_vs_gray_ev - math.log2(base))) < 1e-4
    assert abs(anchored_median_ev("agx", analysis, ev)) < 1e-4


def test_median_align_ev_neutral():
    analysis = _minimal_analysis(-1.0)
    ev = median_align_ev("agx", analysis)
    assert abs(ev - (-analysis.median_vs_gray_ev - math.log2(compute_exposure_gain("agx", 0.0)))) < 1e-4
    assert abs(anchored_median_ev("agx", analysis, ev)) < 1e-4


def test_decoded_scene_body_reference_uses_plan_metric():
    plan = SimpleNamespace(scene=SimpleNamespace(body_ev_p50=-1.75))
    assert scene_body_align_ev(plan) == 1.75
    assert anchored_scene_body_ev(plan, 1.25) == -0.5


def test_resolve_export_ev_manual():
    bundle = RawBundle(
        path=__file__,
        raw_image=None,
        raw_colors=None,
        xyz_render=None,
        render_scale=1.0,
        scene_rec2020_render=None,
        scene_scale=1.0,
        white_level=16383,
        black_levels=[1024.0, 1024.0, 1024.0, 1024.0],
        camera_wb=[1.0, 1.0, 1.0, 1.0],
        color_desc="RGBG",
        raw_pattern=[[0, 1], [1, 2]],
        camera_white_levels=[16383.0, 16383.0, 16383.0, 16383.0],
    )
    analysis = _minimal_analysis(-1.0)
    ev, auto = resolve_export_ev(0.5, bundle, analysis, "p3")
    assert ev == 0.5
    assert auto is None


def _minimal_bundle() -> RawBundle:
    return RawBundle(
        path=__file__,
        raw_image=None,
        raw_colors=None,
        xyz_render=None,
        render_scale=1.0,
        scene_rec2020_render=None,
        scene_scale=1.0,
        white_level=16383,
        black_levels=[1024.0, 1024.0, 1024.0, 1024.0],
        camera_wb=[1.0, 1.0, 1.0, 1.0],
        color_desc="RGBG",
        raw_pattern=[[0, 1], [1, 2]],
        camera_white_levels=[16383.0, 16383.0, 16383.0, 16383.0],
    )


def _minimal_plan() -> ToneCompressionPlan:
    return ToneCompressionPlan(
        target_gamut="Rec2020",
        luma_p1=0.01,
        luma_p50=0.18,
        luma_p99=0.75,
        luma_p999=0.9,
        black_ev=-8.0,
        white_ev=4.0,
        dynamic_range_ev=12.0,
        contrast=3.0,
        toe_power=1.5,
        shoulder_power=3.0,
        chroma_p95=0.0,
        negative_rgb_pct=0.0,
        over_rgb_pct=0.0,
    )


def test_render_sample_output_does_not_mutate_bundle_gain():
    bundle = _minimal_bundle()
    bundle.exposure_gain = 7.0
    sample = np.full((8, 3), 0.25, dtype=np.float32)

    render_sample_linear_output(bundle, None, "p3", 1.0, sample, tone_plan=_minimal_plan())

    assert bundle.exposure_gain == 7.0


def test_compute_auto_ev_boost_only_high_key():
    analysis = _minimal_analysis(+1.5)
    bundle = _minimal_bundle()
    plan = SimpleNamespace(scene=SimpleNamespace(body_ev_p50=1.5))
    with patch("dngscan.auto_ev.build_render_plan", return_value=plan), patch(
        "dngscan.auto_ev.max_safe_ev", return_value=3.0
    ) as safe:
        result = compute_auto_ev(bundle, analysis, "p3")
    assert result.ev_median_target < 0
    assert result.ev == 0.0
    assert result.ev_boost == 0.0
    assert result.highlight_limited is False
    assert safe.call_args.kwargs["tone_plan"] is plan


def test_compute_auto_ev_caps_upward_boost():
    analysis = _minimal_analysis(-2.0)
    bundle = _minimal_bundle()
    plan = SimpleNamespace(scene=SimpleNamespace(body_ev_p50=-2.0))
    with patch("dngscan.auto_ev.build_render_plan", return_value=plan), patch(
        "dngscan.auto_ev.max_safe_ev", return_value=0.5
    ) as safe:
        result = compute_auto_ev(bundle, analysis, "p3")
    assert result.ev == 0.5
    assert result.highlight_limited is True
    assert result.ev_median_target > 0.5
    assert safe.call_args.kwargs["tone_plan"] is plan


def test_compute_auto_ev_reference_plan_carries_the_full_film_declaration():
    """The reference plan must be the plan the real render will compile.

    A reference that omits the film mode, crossover switch or enlarger color head
    judges brightness and the highlight cap against a different curve; measured on a
    daylight fixture, that reads +0.469 EV where the true full-film + color-head plan
    reads +1.008 EV.
    """
    analysis = _minimal_analysis(-2.0)
    bundle = _minimal_bundle()
    plan = SimpleNamespace(scene=SimpleNamespace(body_ev_p50=-2.0))
    with patch(
        "dngscan.auto_ev.build_render_plan", return_value=plan
    ) as build, patch("dngscan.auto_ev.max_safe_ev", return_value=0.5) as safe:
        compute_auto_ev(
            bundle,
            analysis,
            "p3",
            film_curve="velvia100",
            film_mode="full",
            film_crossover="datasheet",
            color_head_y=10.0,
            color_head_m=5.0,
        )
    build_kwargs = build.call_args.kwargs
    assert build_kwargs["film_curve"] == "velvia100"
    assert build_kwargs["film_mode"] == "full"
    assert build_kwargs["film_crossover"] == "datasheet"
    assert build_kwargs["color_head_y"] == 10.0
    assert build_kwargs["color_head_m"] == 5.0
    safe_kwargs = safe.call_args.kwargs
    assert safe_kwargs["tone_plan"] is plan
    assert safe_kwargs["film_mode"] == "full"
    assert safe_kwargs["film_crossover"] == "datasheet"
    assert safe_kwargs["color_head_y"] == 10.0
    assert safe_kwargs["color_head_m"] == 5.0


def test_resolve_export_ev_forwards_the_full_film_declaration():
    analysis = _minimal_analysis(-2.0)
    bundle = _minimal_bundle()
    with patch("dngscan.auto_ev.compute_auto_ev") as compute:
        compute.return_value = SimpleNamespace(ev=0.25)
        resolve_export_ev(
            "auto",
            bundle,
            analysis,
            "p3",
            film_curve="portra400",
            film_mode="full",
            film_crossover="datasheet",
            color_head_y=30.0,
            color_head_m=10.0,
        )
    kwargs = compute.call_args.kwargs
    assert kwargs["film_mode"] == "full"
    assert kwargs["film_crossover"] == "datasheet"
    assert kwargs["color_head_y"] == 30.0
    assert kwargs["color_head_m"] == 10.0


class AutoEvTest(unittest.TestCase):
    test_parse_ev_auto_token = staticmethod(test_parse_ev_auto_token)
    test_median_align_ev_agx = staticmethod(test_median_align_ev_agx)
    test_median_align_ev_neutral = staticmethod(test_median_align_ev_neutral)
    test_decoded_scene_body_reference_uses_plan_metric = staticmethod(test_decoded_scene_body_reference_uses_plan_metric)
    test_resolve_export_ev_manual = staticmethod(test_resolve_export_ev_manual)
    test_render_sample_output_does_not_mutate_bundle_gain = staticmethod(test_render_sample_output_does_not_mutate_bundle_gain)
    test_compute_auto_ev_boost_only_high_key = staticmethod(test_compute_auto_ev_boost_only_high_key)
    test_compute_auto_ev_caps_upward_boost = staticmethod(test_compute_auto_ev_caps_upward_boost)
    test_compute_auto_ev_reference_plan_carries_the_full_film_declaration = staticmethod(
        test_compute_auto_ev_reference_plan_carries_the_full_film_declaration
    )
    test_resolve_export_ev_forwards_the_full_film_declaration = staticmethod(
        test_resolve_export_ev_forwards_the_full_film_declaration
    )


if __name__ == "__main__":
    unittest.main()


class ProbeNativeFinalizeTests(unittest.TestCase):
    """B5: the headroom probe's gamut fit routes native, within declared bounds."""

    def test_probe_finalize_matches_numpy_within_the_declared_envelope(self) -> None:
        import numpy as np
        from dngscan import _fast as fast_backend
        from dngscan.auto_ev import _probe_finalize_linear
        from dngscan.render import finalize_output_linear

        if not fast_backend.supports_output_finalizer():
            self.skipTest("native output finalizer unavailable")
        rng = np.random.default_rng(5)
        sample = (rng.random((120_000, 3), dtype=np.float32) * 1.6 - 0.1)
        native = _probe_finalize_linear(sample, "p3", "none", 1.0, None)
        reference = finalize_output_linear(sample, "p3", "none", 1.0, None)
        diff = np.abs(np.asarray(native, dtype=np.float32) - reference)
        # Declared envelope: measured max 2.8e-5; gate at 1e-4 for headroom.
        self.assertLess(float(diff.max()), 1e-4)

    def test_native_failure_falls_back_to_the_exact_numpy_path(self) -> None:
        import os
        import numpy as np
        from unittest import mock
        from dngscan import _fast as fast_backend
        from dngscan.auto_ev import _probe_finalize_linear
        from dngscan.render import finalize_output_linear

        sample = np.linspace(-0.1, 1.4, 300, dtype=np.float32).reshape(-1, 3)
        # The silent fallback is the auto-mode contract; strict mode instead
        # surfaces the failure (see _probe_finalize_linear's ladder). Pin the
        # dispatch predicate so this failure-injection test does not depend on
        # whether the optional extension was already built in the environment.
        with mock.patch.object(
            fast_backend, "supports_output_finalizer", return_value=True
        ):
            with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "auto"}):
                with mock.patch.object(
                    fast_backend,
                    "compile_output_plan",
                    side_effect=RuntimeError("boom"),
                ):
                    out = _probe_finalize_linear(sample, "p3", "none", 1.0, None)
        reference = finalize_output_linear(sample, "p3", "none", 1.0, None)
        self.assertTrue(np.array_equal(out, reference))
        with mock.patch.object(
            fast_backend, "supports_output_finalizer", return_value=True
        ):
            with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "1"}):
                with mock.patch.object(
                    fast_backend,
                    "compile_output_plan",
                    side_effect=RuntimeError("boom"),
                ):
                    with self.assertRaises(fast_backend.NativeKernelError):
                        _probe_finalize_linear(sample, "p3", "none", 1.0, None)


class AutoEvReferencePlanTests(unittest.TestCase):
    """The auto-EV reference plan must be the plan the real render will use.

    Previously compute_auto_ev / max_safe_ev compiled their internal reference plan
    without endpoint_mode / film_curve / lens_filter, so the brightness reference and
    the highlight-safety cap were judged against an adaptive curve even when the
    actual render used evidence endpoints, a film preset, or declared front glass.
    """

    @staticmethod
    def _scene():
        from tests.golden_support import build_daylight_wide_dr

        return build_daylight_wide_dr()

    @staticmethod
    def _capture_reference_plan(**kwargs):
        import dngscan.auto_ev as auto_ev_mod
        from dngscan.tone import build_render_plan as real_build

        scene = AutoEvReferencePlanTests._scene()
        captured = []

        def recording(*args, **kw):
            plan = real_build(*args, **kw)
            captured.append((args, kw, plan))
            return plan

        with patch.object(auto_ev_mod, "build_render_plan", side_effect=recording):
            auto_ev_mod.compute_auto_ev(scene.bundle, scene.analysis, "p3", **kwargs)
        return captured[0]

    def test_endpoint_mode_reaches_the_reference_plan_and_moves_black_ev(self) -> None:
        _, kw_adaptive, plan_adaptive = self._capture_reference_plan()
        _, kw_evidence, plan_evidence = self._capture_reference_plan(
            endpoint_mode="evidence"
        )
        self.assertEqual(kw_adaptive.get("endpoint_mode", "adaptive"), "adaptive")
        self.assertEqual(kw_evidence["endpoint_mode"], "evidence")
        self.assertEqual(plan_evidence.tone.endpoint_mode, "evidence")
        self.assertNotAlmostEqual(
            plan_evidence.tone.black_ev, plan_adaptive.tone.black_ev, places=2
        )

    def test_film_curve_and_lens_filter_reach_the_reference_plan(self) -> None:
        args, kw, plan = self._capture_reference_plan(
            film_curve="portra400", lens_filter="85b"
        )
        self.assertEqual(kw["film_curve"], "portra400")
        self.assertEqual(plan.tone.curve_preset, "portra400")
        from dngscan.film_curve import FILM_CURVE_PRESETS

        self.assertAlmostEqual(
            plan.tone.target_black_linear,
            float(FILM_CURVE_PRESETS["portra400"]["params"]["target_black_linear"]),
            places=6,
        )
        # The reference bundle the plan compiles from must see the declared glass.
        self.assertEqual(args[0].lens_filter, "85b")

    def test_max_safe_ev_compiles_its_own_plan_with_the_parameters(self) -> None:
        import dngscan.auto_ev as auto_ev_mod
        from dngscan.tone import build_render_plan as real_build

        scene = self._scene()
        captured = []

        def recording(*args, **kw):
            plan = real_build(*args, **kw)
            captured.append((args, kw, plan))
            return plan

        with patch.object(auto_ev_mod, "build_render_plan", side_effect=recording):
            auto_ev_mod.max_safe_ev(
                scene.bundle,
                scene.analysis,
                "p3",
                endpoint_mode="evidence",
                film_curve="portra400",
                lens_filter="85b",
            )
        args, kw, plan = captured[0]
        self.assertEqual(kw["endpoint_mode"], "evidence")
        self.assertEqual(kw["film_curve"], "portra400")
        self.assertEqual(plan.tone.endpoint_mode, "adaptive")  # film preset supersedes
        self.assertEqual(plan.tone.curve_preset, "portra400")
        self.assertEqual(args[0].lens_filter, "85b")
