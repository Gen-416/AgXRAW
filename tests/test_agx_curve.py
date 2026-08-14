# SPDX-License-Identifier: GPL-3.0-or-later
"""AgX curve: inversion protection, adaptive pivot/gamma, target black, outset presets."""
from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from dngscan.agx import (
    AGX_HUE_RESTORE, AGX_INSET_REC2020, AGX_OUTSET_REC2020, AGX_PRIMARIES_PRESETS, MIN_SEGMENT_X,
    apply_core, apply_curve, compute_pivot_ev_offset, curve_params, formation_matrices,
    look_brightness_power, matrices_for_preset,
)
from dngscan.models import Analysis, ToneCompressionPlan


# The X-T2 greycard scene that originally collapsed the shoulder: narrow highlights,
# pivot far right, latitude pushing the transition past the window edge.
NARROW_SCENE = dict(
    black_ev=-5.96, white_ev=1.82, contrast=3.03,
    toe_power=1.23, shoulder_power=3.30,
    latitude_lo_ev=0.0, latitude_hi_ev=1.94,
)


def _compile_flat_plan(agx_primaries: str = "base") -> ToneCompressionPlan:
    """Compile a real plan from a synthetic flat scene.

    Anything asserting on a *default* has to come through the compiler, because the
    compiler overrides several dataclass defaults per preset. A hand-built stub would
    only re-assert whatever the stub itself declared.
    """
    from pathlib import Path

    from dngscan.models import RawBundle
    from dngscan.tone import build_tone_compression_plan, compute_exposure_gain

    bundle = RawBundle(
        path=Path("x.dng"),
        raw_image=np.zeros((8, 8), dtype=np.uint16),
        raw_colors=np.zeros((8, 8), dtype=np.uint8),
        xyz_render=np.zeros((8, 8, 3), dtype=np.float32),
        render_scale=65535.0,
        scene_rec2020_render=np.full((8, 8, 3), 0.18, dtype=np.float32),
        scene_scale=1.0,
        white_level=16383,
        black_levels=[1000.0, 1000.0, 1000.0],
        camera_wb=[1.0, 1.0, 1.0, 0.0],
        color_desc="RGB",
        raw_pattern=[[0, 1], [1, 2]],
        camera_white_levels=[16383, 16383, 16383],
        exposure_gain=compute_exposure_gain("agx", 0.0),
    )
    return build_tone_compression_plan(
        bundle, _flat_analysis(), "Rec2020", agx_primaries=agx_primaries
    )


def _flat_analysis() -> Analysis:
    return Analysis(
        channel_ids=[0, 1, 2], labels={0: "R", 1: "G", 2: "B"},
        ceilings={0: 16383, 1: 16383, 2: 16383},
        ceil_spike_counts={0: 0, 1: 0, 2: 0}, ceil_near_counts={0: 0, 1: 0, 2: 0},
        ceil_spike_ok={0: True, 1: True, 2: True}, fullwell_channel_ids=[0, 1, 2],
        fullwell_note="", saturation_levels={0: 16383, 1: 16383, 2: 16383},
        channel_fullwell={0: 16383, 1: 16383, 2: 16383},
        channel_thresholds={0: 16379, 1: 16379, 2: 16379},
        fullwell=16383, threshold=16379,
        clip_pct={0: 0.0, 1: 0.0, 2: 0.0}, cfa_cell_supported=True,
        cell_union_pct=0.0, cell_ge2_of_clipped_pct=0.0,
        cell_k_of_clipped_pct={}, cell_k_of_all_pct={},
        ev_p1=-4.0, ev_raw_p1=-4.0, ev_median=0.0, ev_p99=2.0, ev_p999=2.5,
        ev_dr_p1_p999=6.5, ev_floor_hit_pct=0.0, median_vs_gray_ev=0.0, median_y=0.18,
        noise_floor=0.002, usable_dr_ev=8.0, snr_curves={}, snr1_dr={}, snr1_stop={},
        gamut_out_pct={"sRGB": 0.0, "Display P3": 0.0, "Rec2020": 0.0},
        bright_pixel_pct=0.0, survivor_channel="R", container_bits_est=14,
        usable_dr_eff_ev=8.0,
    )


class _PlanStub:
    black_ev = -6.5
    white_ev = 4.0
    contrast = 3.0
    toe_power = 1.5
    shoulder_power = 3.3
    latitude_lo_ev = 0.0
    latitude_hi_ev = 1.0
    punch_strength = 0.0
    tone_core = "agx"
    agx_primaries = "base"
    hue_restore = 0.6


class CurveInversionTest(unittest.TestCase):
    def test_shoulder_keeps_minimum_run(self) -> None:
        p = curve_params(**NARROW_SCENE)
        self.assertLessEqual(float(p["shoulder_transition_x"]), 1.0 - MIN_SEGMENT_X + 1e-9)
        # transition y must sit on the linear segment (consistent x/y clamping)
        expected_y = float(p["slope"]) * float(p["shoulder_transition_x"]) + float(p["intercept"])
        self.assertAlmostEqual(float(p["shoulder_transition_y"]), expected_y, places=5)

    def test_curve_reaches_white_and_black(self) -> None:
        for kwargs in (NARROW_SCENE, dict(black_ev=-10.0, white_ev=6.5, contrast=3.0,
                                          toe_power=1.5, shoulder_power=3.3)):
            p = curve_params(**kwargs)
            x = np.linspace(0.0, 1.0, 2001, dtype=np.float32)
            y = apply_curve(x, p)
            self.assertGreater(float(y[-1]), 0.985, msg=str(kwargs))
            self.assertLess(float(y[0]), float(p["target_black"]) + 0.02, msg=str(kwargs))

    def test_curve_monotone_no_jump(self) -> None:
        p = curve_params(**NARROW_SCENE)
        x = np.linspace(0.0, 1.0, 4001, dtype=np.float32)
        y = apply_curve(x, p)
        dy = np.diff(y.astype(np.float64))
        self.assertGreaterEqual(float(dy.min()), -1e-6)
        # no near-discontinuity: largest step bounded (the old collapsed shoulder
        # jumped ~0.2 across one sample)
        self.assertLess(float(dy.max()), 0.01)

    def test_adaptive_gamma_puts_pivot_near_diagonal(self) -> None:
        p = curve_params(black_ev=-10.0, white_ev=6.5, contrast=3.0, toe_power=1.5, shoulder_power=3.3)
        pivot_x = -(-10.0) / 16.5
        pivot_y = float(p["slope"]) * pivot_x + float(p["intercept"])
        self.assertLess(abs(pivot_y - pivot_x), 0.02)
        # mid gray still maps to 0.18 linear at the pivot
        self.assertAlmostEqual(pivot_y ** float(p["gamma"]), 0.18, places=3)


class AdaptivePivotTest(unittest.TestCase):
    def test_zero_offset_unchanged_reference(self) -> None:
        a = curve_params(-8.0, 4.0, 3.0, 1.5, 3.3)
        b = curve_params(-8.0, 4.0, 3.0, 1.5, 3.3, pivot_ev_offset=0.0)
        self.assertEqual(a, b)

    def test_shifted_pivot_holds_ev0_not_subject_brightness(self) -> None:
        # The two constraints (subject brightness preserved / EV0 anchored) share one
        # degree of freedom; the EV0 anchor is the hard one. The subject's own output
        # is allowed to move — that shift IS the contrast reallocation — while
        # calibrated EV 0 must keep rendering ~0.18 linear (see Ev0AnchorSolverTest).
        offset = -0.9
        shifted = curve_params(-8.0, 4.0, 3.0, 1.5, 3.3, pivot_ev_offset=offset)
        x0 = np.asarray([8.0 / 12.0], dtype=np.float32)
        y0 = float(apply_curve(x0, shifted)[0]) ** float(shifted["gamma"])
        self.assertAlmostEqual(y0, 0.18, delta=0.006)

    def test_shifted_pivot_raises_contrast_at_subject(self) -> None:
        offset = -1.2
        base = curve_params(-8.0, 4.0, 3.0, 1.5, 3.3)
        shifted = curve_params(-8.0, 4.0, 3.0, 1.5, 3.3, pivot_ev_offset=offset)
        x0 = (offset + 8.0) / 12.0
        xs = np.asarray([x0 - 0.01, x0 + 0.01], dtype=np.float32)
        def linear_slope(p):
            ys = apply_curve(xs, p).astype(np.float64) ** float(p["gamma"])
            return (ys[1] - ys[0]) / 0.02
        self.assertGreater(linear_slope(shifted), linear_slope(base))


class ViewBrightnessTest(unittest.TestCase):
    def test_darktable_piecewise_power(self) -> None:
        self.assertAlmostEqual(look_brightness_power(1.25), 0.8)
        self.assertAlmostEqual(look_brightness_power(0.64), 1.25)


class TargetBlackTest(unittest.TestCase):
    def test_target_black_lifts_floor(self) -> None:
        p = curve_params(-8.0, 4.0, 3.0, 1.5, 3.3, target_black_linear=0.03)
        x = np.linspace(0.0, 1.0, 501, dtype=np.float32)
        y_linear = apply_curve(x, p).astype(np.float64) ** float(p["gamma"])
        self.assertGreater(float(y_linear.min()), 0.02)
        self.assertGreater(float(y_linear[-1]), 0.95)


    def test_target_white_lowers_shoulder(self) -> None:
        p_full = curve_params(-8.0, 4.0, 3.0, 1.5, 3.3, target_white_linear=1.0)
        p_fade = curve_params(-8.0, 4.0, 3.0, 1.5, 3.3, target_white_linear=0.85)
        x = np.linspace(0.0, 1.0, 501, dtype=np.float32)
        y_full = apply_curve(x, p_full).astype(np.float64) ** float(p_full["gamma"])
        y_fade = apply_curve(x, p_fade).astype(np.float64) ** float(p_fade["gamma"])
        self.assertGreater(float(y_full[-1]), float(y_fade[-1]))
        self.assertLess(float(y_fade[-1]), 0.92)

    def test_target_white_accepts_extended_linear_white(self) -> None:
        p_sdr = curve_params(-8.0, 4.0, 3.0, 1.5, 3.3, target_white_linear=1.0)
        p_hdr = curve_params(-8.0, 4.0, 3.0, 1.5, 3.3, target_white_linear=8.0)
        self.assertNotEqual(p_sdr, p_hdr)
        self.assertGreater(float(p_hdr["target_white"]), 1.0)

        x = np.linspace(0.0, 1.0, 4001, dtype=np.float32)
        y_hdr = apply_curve(x, p_hdr).astype(np.float64) ** float(p_hdr["gamma"])
        self.assertAlmostEqual(float(y_hdr[-1]), 8.0, places=4)
        self.assertGreaterEqual(float(np.diff(y_hdr).min()), -1e-6)

    def test_target_white_rejects_nonfinite_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_white_linear must be finite"):
            curve_params(-8.0, 4.0, 3.0, 1.5, 3.3, target_white_linear=float("inf"))


class OutsetPresetTest(unittest.TestCase):
    def test_base_preset_matches_default_geometry(self) -> None:
        inset, outset = matrices_for_preset("base")
        self.assertTrue(np.allclose(inset, AGX_INSET_REC2020))
        self.assertTrue(np.allclose(outset, AGX_OUTSET_REC2020))

    def test_base_matrices_match_pinned_darktable_rec2020_geometry(self) -> None:
        inset, outset = matrices_for_preset("base")
        np.testing.assert_allclose(
            inset,
            np.asarray(
                [
                    [0.85655585, 0.09506743, 0.04837672],
                    [0.13706229, 0.76123601, 0.10170170],
                    [0.10984838, 0.07674446, 0.81340716],
                ]
            ),
            rtol=0.0,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            outset,
            np.asarray(
                [
                    [1.12818374, -0.11152949, -0.01665424],
                    [-0.13973488, 1.15638912, -0.01665424],
                    [-0.13973488, -0.11152949, 1.25126437],
                ]
            ),
            rtol=0.0,
            atol=2e-7,
        )
        np.testing.assert_allclose(inset.sum(axis=1), 1.0, rtol=0.0, atol=2e-7)
        np.testing.assert_allclose(outset.sum(axis=1), 1.0, rtol=0.0, atol=2e-7)

    def test_preset_purity_ordering_end_to_end(self) -> None:
        # Directional contract through the REAL pipeline path (left-multiply M @ v via
        # apply_core), not `linear @ M` which silently tests the transpose: punchy must
        # render more chroma than base, muted less. This is the ordering the GUI labels
        # (鲜明 / 柔和) promise.
        from dngscan.color import apply_rgb_matrix3
        from dngscan.constants import OKLAB_M1, OKLAB_M2, RGB_TO_XYZ

        def mean_chroma(rgb):
            xyz = apply_rgb_matrix3(rgb.astype(np.float32), RGB_TO_XYZ["Rec2020"])
            lab = apply_rgb_matrix3(np.cbrt(np.maximum(apply_rgb_matrix3(xyz, OKLAB_M1), 0.0)), OKLAB_M2)
            return float(np.hypot(lab[:, 1], lab[:, 2]).mean())

        test = np.asarray(
            [[0.5, 0.10, 0.05], [0.06, 0.30, 0.10], [0.08, 0.12, 0.45], [0.35, 0.25, 0.06]],
            dtype=np.float32,
        )
        chroma = {}
        for name in ("base", "punchy", "muted"):
            plan = _PlanStub()
            plan.agx_primaries = name
            inset, outset = formation_matrices(plan)
            chroma[name] = mean_chroma(apply_core(test, plan, inset, outset))
        self.assertGreater(chroma["punchy"], chroma["base"] + 1e-3)
        self.assertLess(chroma["muted"], chroma["base"] - 1e-3)

    def test_muted_differs_from_base(self) -> None:
        rgb = np.asarray([[0.30, 0.12, 0.06], [0.05, 0.20, 0.35]], dtype=np.float32)
        plan_b = _PlanStub()
        plan_b.agx_primaries = "base"
        inset_b, outset_b = formation_matrices(plan_b)
        plan_s = _PlanStub()
        plan_s.agx_primaries = "muted"
        inset_s, outset_s = formation_matrices(plan_s)
        base = apply_core(rgb, plan_b, inset_b, outset_b)
        muted = apply_core(rgb, plan_s, inset_s, outset_s)
        self.assertGreater(float(np.abs(base - muted).max()), 1e-4)

    def test_smooth_uses_different_inset(self) -> None:
        rgb = np.asarray([[0.30, 0.12, 0.06], [0.05, 0.20, 0.35]], dtype=np.float32)
        plan_b = _PlanStub()
        plan_b.agx_primaries = "base"
        plan_s = _PlanStub()
        plan_s.agx_primaries = "smooth"
        inset_b, outset_b = formation_matrices(plan_b)
        inset_s, outset_s = formation_matrices(plan_s)
        self.assertGreater(float(np.abs(inset_b - inset_s).max()), 1e-3)
        base = apply_core(rgb, plan_b, inset_b, outset_b)
        smooth = apply_core(rgb, plan_s, inset_s, outset_s)
        self.assertGreater(float(np.abs(base - smooth).max()), 1e-4)

    def test_neutral_axis_preserved_by_presets(self) -> None:
        gray = np.asarray([[0.18, 0.18, 0.18]], dtype=np.float32)
        for name in ("base", "punchy"):
            plan = _PlanStub()
            plan.agx_primaries = name
            inset, outset = formation_matrices(plan)
            out = apply_core(gray, plan, inset, outset)[0]
            self.assertLess(float(out.max() - out.min()), 1e-2, msg=name)


class HueRestoreTest(unittest.TestCase):
    def test_hue_restore_extremes_differ(self) -> None:
        rgb = np.asarray([[0.45, 0.08, 0.04]], dtype=np.float32)
        plan_lo = _PlanStub()
        plan_lo.hue_restore = 0.0
        plan_hi = _PlanStub()
        plan_hi.hue_restore = 1.0
        lo = apply_core(rgb, plan_lo, AGX_INSET_REC2020, AGX_OUTSET_REC2020)
        hi = apply_core(rgb, plan_hi, AGX_INSET_REC2020, AGX_OUTSET_REC2020)
        self.assertGreater(float(np.abs(lo - hi).max()), 1e-4)

    def test_compiled_default_renders_as_explicit_darktable_hue_restore(self) -> None:
        """Compare a compiled plan against an explicit 0.6, not a stub against itself.

        `_PlanStub` declares hue_restore = 0.6 of its own, so stub-vs-stub would pass
        whatever the shipped default became. Driving one side through the compiler makes
        the assertion depend on the value that actually renders.
        """
        rgb = np.asarray([[0.45, 0.08, 0.04]], dtype=np.float32)
        compiled = _compile_flat_plan(agx_primaries="base")
        explicit = dataclasses.replace(compiled, hue_restore=0.6)
        inset, outset = formation_matrices(compiled)
        a = apply_core(rgb, compiled, inset, outset)
        b = apply_core(rgb, explicit, inset, outset)
        self.assertTrue(np.array_equal(a, b))
        # And a different restore really does move the result, so equality means something.
        other = dataclasses.replace(compiled, hue_restore=0.0)
        self.assertGreater(
            float(np.abs(apply_core(rgb, other, inset, outset) - a).max()), 1e-6
        )


class LookOverrideTest(unittest.TestCase):
    def test_plan_overrides_from_look_fields(self) -> None:
        from dngscan import look as look_engine

        field = look_engine.LOOK_FIELDS["optic_warm_cyan"]
        self.assertIsNone(field.agx_hue_restore)
        optic = look_engine.agx_plan_overrides("optic_warm_cyan")
        self.assertAlmostEqual(optic["hue_restore"], 0.52)
        self.assertEqual(look_engine.agx_plan_overrides("does_not_exist"), {})

        import dataclasses

        faded = dataclasses.replace(field, agx_hue_restore=0.6, agx_target_black=0.025)
        look_engine.LOOK_FIELDS["_test_faded"] = faded
        try:
            overrides = look_engine.agx_plan_overrides("_test_faded")
            self.assertAlmostEqual(overrides["hue_restore"], 0.6)
            self.assertAlmostEqual(overrides["target_black_linear"], 0.025)
        finally:
            del look_engine.LOOK_FIELDS["_test_faded"]


class PivotAutomationTest(unittest.TestCase):
    def test_dark_body_pulls_pivot_negative(self) -> None:
        offset = compute_pivot_ev_offset(-3.5, -8.0, 4.0)
        self.assertLess(offset, -1.0)
        self.assertGreater(offset, -4.0)

    def test_bright_body_keeps_zero_offset(self) -> None:
        self.assertEqual(compute_pivot_ev_offset(0.5, -8.0, 4.0), 0.0)


class HueRestoreAnchorTest(unittest.TestCase):
    """Pin hue_restore at every layer that can decide it.

    Three places carry the value and only one reaches a normal render, so a test on the
    wrong one reads like a guard without being one. `_plan_hue_restore` prefers
    `plan.hue_restore`, falls back to `1 - plan.hue_keep` for pre-rename objects, and
    only then reaches `AGX_HUE_RESTORE` — which a real `ToneCompressionPlan` never does.
    Setting the module constant to 0.61 leaves all 112 golden cases byte-identical, so
    asserting on it alone would let the shipped default drift unnoticed.
    """

    def test_dataclass_default_is_darktable_restore(self) -> None:
        self.assertEqual(ToneCompressionPlan.hue_restore, 0.6)

    def test_compiled_plans_carry_the_per_preset_value(self) -> None:
        """The compiler overrides the dataclass default per preset, so pin that too:
        pinned darktable restores 60 % on its scene default and disables restoration
        entirely on the sigmoid-like smooth geometry."""
        for primaries, expected in (
            ("base", 0.6),
            ("punchy", 0.6),
            ("muted", 0.6),
            ("smooth", 0.0),
        ):
            with self.subTest(primaries=primaries):
                plan = _compile_flat_plan(agx_primaries=primaries)
                self.assertEqual(plan.agx_primaries, primaries)
                self.assertAlmostEqual(plan.hue_restore, expected, places=9)

    def test_constant_is_only_the_pre_rename_fallback(self) -> None:
        """Objects predating the rename still resolve, and only they see the constant."""
        from dngscan.agx import _plan_hue_restore

        class _NoField:
            pass

        class _OldKeep:
            hue_keep = 0.25

        self.assertEqual(AGX_HUE_RESTORE, 0.6)
        self.assertAlmostEqual(_plan_hue_restore(_NoField()), AGX_HUE_RESTORE, places=9)
        # The rename inverted the meaning, so a legacy 0.25 "keep" is 0.75 "restore".
        self.assertAlmostEqual(_plan_hue_restore(_OldKeep()), 0.75, places=9)
        self.assertAlmostEqual(_plan_hue_restore(_compile_flat_plan()), 0.6, places=9)


class TonePlanPivotTest(unittest.TestCase):
    def test_dark_median_does_not_reanchor_pivot(self) -> None:
        from dngscan.models import Analysis, RawBundle, ToneCompressionPlan
        from dngscan.tone import build_tone_compression_plan
        from pathlib import Path

        analysis = Analysis(
            channel_ids=[0, 1, 2],
            labels={0: "R", 1: "G", 2: "B"},
            ceilings={0: 1000, 1: 1000, 2: 1000},
            ceil_spike_counts={0: 0, 1: 0, 2: 0},
            ceil_near_counts={0: 0, 1: 0, 2: 0},
            ceil_spike_ok={0: False, 1: False, 2: False},
            fullwell_channel_ids=[0, 1, 2],
            fullwell_note="test",
            saturation_levels={0: 1000, 1: 1000, 2: 1000},
            channel_fullwell={0: 1000, 1: 1000, 2: 1000},
            channel_thresholds={0: 996, 1: 996, 2: 996},
            fullwell=1000,
            threshold=996,
            clip_pct={0: 0.0, 1: 0.0, 2: 0.0},
            cfa_cell_supported=True,
            cell_union_pct=0.0,
            cell_ge2_of_clipped_pct=0.0,
            cell_k_of_clipped_pct={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
            cell_k_of_all_pct={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
            ev_p1=-8.0,
            ev_raw_p1=-8.0,
            ev_median=-4.5,
            ev_p99=-1.0,
            ev_p999=-0.5,
            ev_dr_p1_p999=7.5,
            ev_floor_hit_pct=0.0,
            median_vs_gray_ev=-2.0,
            median_y=0.04,
            noise_floor=0.002,
            usable_dr_ev=8.0,
            snr_curves={},
            snr1_dr={},
            snr1_stop={},
            gamut_out_pct={"sRGB": 0.0, "Display P3": 0.0, "Rec2020": 0.0},
            bright_pixel_pct=0.0,
            survivor_channel="R",
            container_bits_est=14,
            usable_dr_eff_ev=8.0,
        )
        bundle = RawBundle(
            path=Path("x.dng"),
            raw_image=np.zeros((4, 4), dtype=np.uint16),
            raw_colors=np.zeros((4, 4), dtype=np.uint8),
            xyz_render=np.zeros((2, 2, 3), dtype=np.float32),
            render_scale=65535.0,
            scene_rec2020_render=np.full((2, 2, 3), 0.04, dtype=np.float32),
            scene_scale=65535.0,
            white_level=16383,
            black_levels=[1000.0, 1000.0, 1000.0],
            camera_wb=[1.0, 1.0, 1.0, 0.0],
            color_desc="RGB",
            raw_pattern=[[0, 1], [1, 2]],
            camera_white_levels=[16383, 16383, 16383],
            exposure_gain=1.0,
        )
        from dngscan.models import SceneToneMetrics

        bright_metrics = SceneToneMetrics(
            reliable_sample_pct=95.0,
            body_ev_p1=-2.0,
            body_ev_p5=-1.0,
            body_ev_p50=0.0,
            body_ev_p95=2.0,
            body_ev_p99=3.0,
            body_ev_p999=4.0,
            tail_ev_p9999=5.0,
            tail_area_ev0_pct=0.0,
            tail_area_ev2_pct=0.0,
            tail_extremity=0.0,
            sparse_emitter_tail=False,
            raw_clip_union_pct=0.0,
            reliable_tail_ev_p9999=3.0,
        )
        plan = build_tone_compression_plan(
            bundle, analysis, "Rec2020", scene_metrics=bright_metrics,
        )
        # Bright body near mid gray: contrast pivot stays at calibrated 0 EV.
        self.assertEqual(plan.pivot_ev_offset, 0.0)
        self.assertAlmostEqual(plan.white_ev, 3.3, places=5)


class DarkSceneTonePlanTest(unittest.TestCase):
    def _dark_analysis(self) -> "Analysis":
        from dngscan.models import Analysis

        return Analysis(
            channel_ids=[0, 1, 2],
            labels={0: "R", 1: "G", 2: "B"},
            ceilings={0: 1000, 1: 1000, 2: 1000},
            ceil_spike_counts={0: 0, 1: 0, 2: 0},
            ceil_near_counts={0: 0, 1: 0, 2: 0},
            ceil_spike_ok={0: False, 1: False, 2: False},
            fullwell_channel_ids=[0, 1, 2],
            fullwell_note="test",
            saturation_levels={0: 1000, 1: 1000, 2: 1000},
            channel_fullwell={0: 1000, 1: 1000, 2: 1000},
            channel_thresholds={0: 996, 1: 996, 2: 996},
            fullwell=1000,
            threshold=996,
            clip_pct={0: 0.0, 1: 0.0, 2: 0.0},
            cfa_cell_supported=True,
            cell_union_pct=0.0,
            cell_ge2_of_clipped_pct=0.0,
            cell_k_of_clipped_pct={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
            cell_k_of_all_pct={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
            ev_p1=-8.0,
            ev_raw_p1=-8.0,
            ev_median=-4.5,
            ev_p99=-1.0,
            ev_p999=-0.5,
            ev_dr_p1_p999=7.5,
            ev_floor_hit_pct=0.0,
            median_vs_gray_ev=-2.0,
            median_y=0.04,
            noise_floor=0.002,
            usable_dr_ev=6.0,
            snr_curves={},
            snr1_dr={},
            snr1_stop={},
            gamut_out_pct={"sRGB": 0.0, "Display P3": 0.0, "Rec2020": 0.0},
            bright_pixel_pct=0.0,
            survivor_channel="R",
            container_bits_est=14,
            usable_dr_eff_ev=6.0,
        )

    def test_dark_scene_uses_a_separate_toe_without_lifting_black(self) -> None:
        from dngscan.models import RawBundle
        from dngscan.tone import build_tone_compression_plan, compute_exposure_gain
        from pathlib import Path

        bundle = RawBundle(
            path=Path("x.nef"),
            raw_image=np.zeros((4, 4), dtype=np.uint16),
            raw_colors=np.zeros((4, 4), dtype=np.uint8),
            xyz_render=np.zeros((2, 2, 3), dtype=np.float32),
            render_scale=65535.0,
            scene_rec2020_render=np.full((2, 2, 3), 0.012, dtype=np.float32),
            scene_scale=1.0,
            white_level=16383,
            black_levels=[1000.0, 1000.0, 1000.0],
            camera_wb=[1.0, 1.0, 1.0, 0.0],
            color_desc="RGB",
            raw_pattern=[[0, 1], [1, 2]],
            camera_white_levels=[16383, 16383, 16383],
            exposure_gain=compute_exposure_gain("agx", 0.0),
        )
        plan = build_tone_compression_plan(bundle, self._dark_analysis(), "Rec2020")
        # The automatic path keeps the pivot at mid gray: measured on a real night frame,
        # relocating it either crushes the subject (EV0 anchored) or blows EV0 to white
        # (subject brightness preserved). See the pivot comment in tone.py.
        self.assertEqual(plan.pivot_ev_offset, 0.0)
        from dngscan.drt import apply_c1_endpoints

        self.assertAlmostEqual(float(apply_c1_endpoints(np.asarray([0.0]), plan)[0]), 0.18, places=5)
        self.assertLess(plan.view_brightness, 1.08)
        self.assertLess(plan.black_ev, -2.0)
        self.assertEqual(plan.target_black_linear, 0.0)
        self.assertGreater(plan.toe_start_ev, plan.black_ev)

    def test_dark_indoor_scene_not_fully_black_after_agx(self) -> None:
        from dngscan.models import RawBundle
        from dngscan.tone import build_tone_compression_plan, compute_exposure_gain
        from pathlib import Path

        gain = compute_exposure_gain("agx", 2.0)
        scene = np.full((64, 3), 0.025, dtype=np.float32)
        bundle = RawBundle(
            path=Path("x.nef"),
            raw_image=np.zeros((8, 8), dtype=np.uint16),
            raw_colors=np.zeros((8, 8), dtype=np.uint8),
            xyz_render=np.zeros((2, 2, 3), dtype=np.float32),
            render_scale=65535.0,
            scene_rec2020_render=scene.reshape(8, 8, 3),
            scene_scale=1.0,
            white_level=16383,
            black_levels=[1000.0, 1000.0, 1000.0],
            camera_wb=[1.0, 1.0, 1.0, 0.0],
            color_desc="RGB",
            raw_pattern=[[0, 1], [1, 2]],
            camera_white_levels=[16383, 16383, 16383],
            exposure_gain=gain,
        )
        plan = build_tone_compression_plan(bundle, self._dark_analysis(), "Rec2020")
        inset, outset = formation_matrices(plan)
        boosted = (scene * gain).astype(np.float32)
        out = apply_core(boosted, plan, inset, outset)
        self.assertGreater(float(np.median(out)), 0.02)
        self.assertGreater(float(np.percentile(out, 1)), 0.005)


class AgxPlanStabilityTest(unittest.TestCase):
    def test_manual_ev_does_not_reshape_tone_plan(self) -> None:
        from dngscan.models import Analysis, RawBundle
        from dngscan.tone import build_tone_compression_plan, compute_exposure_gain, plan_for_mode
        from pathlib import Path

        analysis = Analysis(
            channel_ids=[0, 1, 2],
            labels={0: "R", 1: "G", 2: "B"},
            ceilings={0: 1000, 1: 1000, 2: 1000},
            ceil_spike_counts={0: 0, 1: 0, 2: 0},
            ceil_near_counts={0: 0, 1: 0, 2: 0},
            ceil_spike_ok={0: False, 1: False, 2: False},
            fullwell_channel_ids=[0, 1, 2],
            fullwell_note="test",
            saturation_levels={0: 1000, 1: 1000, 2: 1000},
            channel_fullwell={0: 1000, 1: 1000, 2: 1000},
            channel_thresholds={0: 996, 1: 996, 2: 996},
            fullwell=1000,
            threshold=996,
            clip_pct={0: 0.0, 1: 0.0, 2: 0.0},
            cfa_cell_supported=True,
            cell_union_pct=0.0,
            cell_ge2_of_clipped_pct=0.0,
            cell_k_of_clipped_pct={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
            cell_k_of_all_pct={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
            ev_p1=-6.0,
            ev_raw_p1=-6.0,
            ev_median=-1.5,
            ev_p99=0.5,
            ev_p999=1.0,
            ev_dr_p1_p999=7.0,
            ev_floor_hit_pct=0.0,
            median_vs_gray_ev=-1.5,
            median_y=0.06,
            noise_floor=0.002,
            usable_dr_ev=9.0,
            snr_curves={},
            snr1_dr={},
            snr1_stop={},
            gamut_out_pct={"sRGB": 0.0, "Display P3": 0.0, "Rec2020": 0.0},
            bright_pixel_pct=0.0,
            survivor_channel="R",
            container_bits_est=14,
            usable_dr_eff_ev=9.0,
        )
        scene = np.full((8, 8, 3), 0.08, dtype=np.float32)
        base = RawBundle(
            path=Path("x.dng"),
            raw_image=np.zeros((8, 8), dtype=np.uint16),
            raw_colors=np.zeros((8, 8), dtype=np.uint8),
            xyz_render=np.zeros((2, 2, 3), dtype=np.float32),
            render_scale=65535.0,
            scene_rec2020_render=scene,
            scene_scale=1.0,
            white_level=16383,
            black_levels=[1000.0, 1000.0, 1000.0],
            camera_wb=[1.0, 1.0, 1.0, 0.0],
            color_desc="RGB",
            raw_pattern=[[0, 1], [1, 2]],
            camera_white_levels=[16383, 16383, 16383],
            exposure_gain=compute_exposure_gain("agx", 0.0),
        )
        boosted = RawBundle(
            **{**base.__dict__, "exposure_gain": compute_exposure_gain("agx", 2.5)}
        )
        plan0 = plan_for_mode(base, analysis, "agx", "srgb")
        plan_boost = plan_for_mode(boosted, analysis, "agx", "srgb")
        self.assertAlmostEqual(plan0.black_ev, plan_boost.black_ev, places=4)
        self.assertAlmostEqual(plan0.white_ev, plan_boost.white_ev, places=4)
        self.assertAlmostEqual(plan0.pivot_ev_offset, plan_boost.pivot_ev_offset, places=4)
        self.assertAlmostEqual(plan0.contrast, plan_boost.contrast, places=4)
        self.assertAlmostEqual(plan0.target_black_linear, plan_boost.target_black_linear, places=5)


if __name__ == "__main__":
    unittest.main()


class Ev0AnchorSolverTest(unittest.TestCase):
    def test_shifted_pivot_holds_ev0_anchor(self) -> None:
        # The contrast pivot may move onto the subject, but calibrated EV 0 must keep
        # rendering to 0.18 linear — the solver's whole contract.
        for offset in (-0.8, -1.5, -2.5):
            p = curve_params(-8.0, 4.0, 3.0, 1.5, 3.3, pivot_ev_offset=offset)
            x0 = (0.0 - float(p["black_ev"])) / float(p["range_ev"])
            y0 = float(apply_curve(np.asarray([x0], dtype=np.float32), p)[0]) ** float(p["gamma"])
            self.assertAlmostEqual(y0, 0.18, delta=0.006, msg=f"offset={offset}")

    def test_zero_offset_bitwise_unchanged(self) -> None:
        a = curve_params(-8.0, 4.0, 3.0, 1.5, 3.3)
        b = curve_params(-8.0, 4.0, 3.0, 1.5, 3.3, pivot_ev_offset=0.0)
        self.assertEqual(a, b)
