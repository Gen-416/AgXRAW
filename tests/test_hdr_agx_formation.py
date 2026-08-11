# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent HDR AgX formation and scene-intent gates."""
from __future__ import annotations

import dataclasses
import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dngscan.analysis import analyze
from dngscan.color import luminance_from_rgb_space
from dngscan.drt import c1_value_and_derivative_at_ev, curve_params_from_plan
from dngscan.grade import RENDER_MODE
from dngscan.hdr_agx import (
    achieved_headroom,
    scene_render_to_hdr_display_linear,
)
from dngscan.hdr_agx_plan import compile_hdr_agx_plan, describe_hdr_plan, reliable_tail_ev
from dngscan.models import HdrDisplayTarget
from dngscan.raw_io import load_raw
from dngscan.render import finalize_output_linear, scene_render_to_display_linear
from dngscan.tone import build_render_plan

PICTURES = Path.home() / "Pictures"
# Daylight, two night frames and a phone capture: the night frames are the ones that
# would expose an HDR capacity quietly re-exposing a dark scene.
FRAMES = {
    "daylight": PICTURES / "AgXRAW样张" / "_SDI0150.DNG",
    "night_stage": PICTURES / "AgXRAW样张" / "_SDI0199.DNG",
    "night_bar": PICTURES / "AgXRAW样张" / "_SDI0133.DNG",
}


def _render_pair(path: Path):
    bundle = load_raw(path, scene_half_size=True)
    analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
    plan = build_render_plan(bundle, analysis, RENDER_MODE, "p3")
    hdr_plan = compile_hdr_agx_plan(plan, analysis=analysis)
    sdr = scene_render_to_display_linear(bundle, plan, "p3")
    hdr = scene_render_to_hdr_display_linear(bundle, plan, hdr_plan, "p3")
    return bundle, plan, hdr_plan, sdr, hdr


def _p3_luminance(rgb: np.ndarray) -> np.ndarray:
    return luminance_from_rgb_space(rgb.reshape(-1, 3), "p3").reshape(rgb.shape[:-1])


@unittest.skipUnless(FRAMES["daylight"].is_file(), "sample frames unavailable")
class FormationExitConditionTests(unittest.TestCase):
    """Image-level conditions the independent HDR DRT must satisfy."""

    def test_zero_budget_still_runs_the_hdr_dispatcher(self) -> None:
        """H=0 disables HDR allocation; it does not turn HDR into an SDR function call."""
        bundle, plan, hdr_plan, _, _ = _render_pair(FRAMES["daylight"])
        zeroed = dataclasses.replace(
            hdr_plan,
            tone=dataclasses.replace(
                hdr_plan.tone,
                requested_headroom_ev=0.0,
                rendered_headroom_ev=0.0,
                peak_linear=1.0,
                shoulder_segments=(),
            ),
        )
        with mock.patch(
            "dngscan.render.scene_render_to_display_linear",
            side_effect=AssertionError("HDR must not dispatch through SDR"),
        ):
            rendered = scene_render_to_hdr_display_linear(bundle, plan, zeroed, "p3")
        self.assertTrue(bool(np.all(np.isfinite(rendered))))
        self.assertGreaterEqual(float(np.min(rendered)), 0.0)
        self.assertLessEqual(float(np.max(rendered)), 1.0 + 1e-6)

    def test_neutral_pixels_stay_neutral(self) -> None:
        """Grey must not acquire a cast, at any rho.

        Replaces a blanket "chromaticity never changes" check that only described Phase 2.
        Channel separation exists precisely to move chromaticity in the highlights, so
        asserting it never moves would now be asserting the feature does nothing.

        Neutrality is judged on the *scene*, not on the SDR output. Those are different
        questions: AgX converges channels toward white in the shoulder, so a coloured
        highlight can leave the SDR render neutral, and separation legitimately restores
        some of that colour. Only a scene that was neutral has nothing to separate.
        """
        for name, path in FRAMES.items():
            if not path.is_file():
                continue
            with self.subTest(frame=name):
                bundle, _, _, _, hdr = _render_pair(path)
                scene = (
                    np.asarray(bundle.scene_rec2020_render, dtype=np.float32)
                    / np.float32(bundle.scene_scale)
                    * np.float32(bundle.exposure_gain)
                )
                span = scene.max(axis=-1) - scene.min(axis=-1)
                neutral = (span < 1e-4) & (scene.max(axis=-1) > 0.05)
                if not bool(np.any(neutral)):
                    continue
                out = hdr[neutral]
                self.assertLess(float(np.max(out.max(axis=-1) - out.min(axis=-1))), 1e-3)

    def test_night_scene_keeps_its_exposure_intent(self) -> None:
        """Independent formation may move the body, but must not turn night into day."""
        for name, path in FRAMES.items():
            if not path.is_file():
                continue
            with self.subTest(frame=name):
                _, plan, _, sdr, hdr = _render_pair(path)
                sdr_final = finalize_output_linear(sdr, "p3", color_plan=plan.color)
                y_sdr, y_hdr = _p3_luminance(sdr_final), _p3_luminance(hdr)
                body = (y_sdr > 0.02) & (y_sdr < 0.5)
                self.assertTrue(bool(np.any(body)))
                delta = float(np.log2(np.median(y_hdr[body]) / np.median(y_sdr[body])))
                self.assertLess(abs(delta), 0.5)

    def test_achieved_headroom_stays_within_budget(self) -> None:
        """H_actual <= H_budget <= H_display, all three separately observable."""
        for name, path in FRAMES.items():
            if not path.is_file():
                continue
            with self.subTest(frame=name):
                _, _, hdr_plan, _, hdr = _render_pair(path)
                actual = achieved_headroom(hdr)
                self.assertLessEqual(actual, hdr_plan.tone.rendered_headroom_ev + 1e-6)
                self.assertLessEqual(
                    hdr_plan.tone.rendered_headroom_ev, hdr_plan.tone.display_headroom_ev + 1e-9
                )

    def test_hdr_plan_owns_a_distinct_formation_object(self) -> None:
        _, plan, hdr_plan, _, _ = _render_pair(FRAMES["daylight"])
        self.assertIsNot(hdr_plan.formation, plan.tone)
        self.assertEqual(hdr_plan.formation.tone_core, "agx")
        # v2 keeps the extended peak entirely in the shoulder. The formation object stays
        # a reference-white body: if target_white_linear or curve_gamma ever moved with
        # headroom again, that would be v1's toe-rewriting mechanism returning.
        self.assertEqual(hdr_plan.formation.target_white_linear, 1.0)
        self.assertEqual(hdr_plan.formation.curve_gamma, hdr_plan.tone.body_gamma)
        self.assertEqual(hdr_plan.tone.body_gamma, 2.2)
        self.assertGreater(len(hdr_plan.tone.shoulder_segments), 0)
        self.assertAlmostEqual(
            float(np.log2(hdr_plan.tone.peak_linear)),
            hdr_plan.tone.rendered_headroom_ev,
            places=9,
        )


@unittest.skipUnless(FRAMES["daylight"].is_file(), "sample frames unavailable")
class PlanCompilationTests(unittest.TestCase):
    def test_shoulder_starts_on_the_authoritative_body_value_and_tangent(self) -> None:
        bundle = load_raw(FRAMES["daylight"], scene_half_size=True)
        analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
        plan = build_render_plan(bundle, analysis, RENDER_MODE, "p3")
        hdr_plan = compile_hdr_agx_plan(plan, analysis=analysis)
        first = hdr_plan.tone.shoulder_segments[0]
        value, slope_t = c1_value_and_derivative_at_ev(
            hdr_plan.tone.shoulder_start_ev, hdr_plan.formation
        )
        expected_z = math.log2(value / 0.18)
        expected_m = slope_t / (math.log(2.0) * value)
        self.assertEqual(first.z0, expected_z)
        self.assertEqual(first.m0, expected_m)

    def test_headrooms_are_reported_separately(self) -> None:
        _, _, hdr_plan, _, _ = _render_pair(FRAMES["daylight"])
        text = describe_hdr_plan(hdr_plan)
        self.assertIn("白点", text)
        self.assertIn("容量", text)

    def test_low_headroom_compiles_a_subdivided_shoulder_not_no_hdr(self) -> None:
        """A capped display keeps earned HDR range through a validated monotone chain.

        H_content = min(H_display, H_signal) caps Z_peak while the tail keeps pushing W
        out, so alpha exceeds the single-segment bound. That request is well-posed, and
        turning HDR off there would be both the least faithful rendering available and a
        cliff in an otherwise continuous headroom control.
        """
        bundle = load_raw(FRAMES["daylight"], scene_half_size=True)
        analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
        plan = build_render_plan(bundle, analysis, RENDER_MODE, "p3")
        steep_tone = dataclasses.replace(plan.tone, contrast=4.5)
        scene = dataclasses.replace(plan.scene, reliable_tail_ev_p9999=8.0)
        steep = dataclasses.replace(plan, tone=steep_tone, scene=scene)
        low = compile_hdr_agx_plan(
            steep, HdrDisplayTarget(peak_nits=280.0), analysis=analysis
        )
        self.assertGreater(low.tone.shoulder_alpha, 3.0)
        self.assertGreater(low.tone.rendered_headroom_ev, 0.0)
        self.assertEqual(
            low.tone.rendered_headroom_ev, low.tone.requested_headroom_ev
        )
        self.assertGreater(len(low.tone.shoulder_segments), 1)
        self.assertIn("细分", describe_hdr_plan(low))

        # The chain honours the same structural contract as a single segment: the body
        # anchor at K is untouched and every join is C1.
        first = low.tone.shoulder_segments[0]
        value, slope_t = c1_value_and_derivative_at_ev(
            low.tone.shoulder_start_ev, low.formation
        )
        self.assertEqual(first.z0, math.log2(value / 0.18))
        self.assertEqual(first.m0, slope_t / (math.log(2.0) * value))
        self.assertEqual(low.tone.shoulder_segments[-1].m1, 0.0)

        # Continuity of the control: a display one notch brighter still compiles, and the
        # curve below K is bit-identical between the two (headroom cannot reach the body).
        high = compile_hdr_agx_plan(
            steep, HdrDisplayTarget(peak_nits=800.0), analysis=analysis
        )
        self.assertGreater(high.tone.rendered_headroom_ev, 0.0)
        self.assertEqual(low.formation, high.formation)

    def test_bigger_display_target_never_moves_scene_white_endpoint(self) -> None:
        bundle = load_raw(FRAMES["daylight"], scene_half_size=True)
        analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
        plan = build_render_plan(bundle, analysis, RENDER_MODE, "p3")
        small = compile_hdr_agx_plan(plan, HdrDisplayTarget(peak_nits=400.0))
        large = compile_hdr_agx_plan(plan, HdrDisplayTarget(peak_nits=4000.0))
        self.assertEqual(small.tone.white_ev, large.tone.white_ev)
        self.assertGreaterEqual(large.tone.rendered_headroom_ev, small.tone.rendered_headroom_ev)

    def test_absent_tail_measurement_does_not_grant_headroom(self) -> None:
        """A missing measurement must not read as an unlimited tail."""
        bundle = load_raw(FRAMES["daylight"], scene_half_size=True)
        analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
        plan = build_render_plan(bundle, analysis, RENDER_MODE, "p3")
        stripped = dataclasses.replace(plan, scene=None)
        self.assertTrue(np.isnan(reliable_tail_ev(stripped)))
        compiled = compile_hdr_agx_plan(stripped, analysis=analysis)
        self.assertEqual(compiled.tone.rendered_headroom_ev, 0.0)

    def test_unfiltered_tail_cannot_replace_missing_reliable_tail(self) -> None:
        bundle = load_raw(FRAMES["daylight"], scene_half_size=True)
        analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
        plan = build_render_plan(bundle, analysis, RENDER_MODE, "p3")
        scene = dataclasses.replace(
            plan.scene, reliable_tail_ev_p9999=float("nan"), tail_ev_p9999=12.0
        )
        compiled = compile_hdr_agx_plan(dataclasses.replace(plan, scene=scene), analysis=analysis)
        self.assertEqual(compiled.tone.rendered_headroom_ev, 0.0)

    def test_sdr_plan_is_not_mutated(self) -> None:
        bundle = load_raw(FRAMES["daylight"], scene_half_size=True)
        analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
        plan = build_render_plan(bundle, analysis, RENDER_MODE, "p3")
        before = dataclasses.asdict(plan.tone)
        compile_hdr_agx_plan(plan, HdrDisplayTarget(peak_nits=4000.0))
        self.assertEqual(dataclasses.asdict(plan.tone), before)

    def test_hdr_white_endpoint_is_not_read_from_sdr_tone(self) -> None:
        bundle = load_raw(FRAMES["daylight"], scene_half_size=True)
        analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
        plan = build_render_plan(bundle, analysis, RENDER_MODE, "p3")
        changed_tone = dataclasses.replace(
            plan.tone,
            white_ev=8.5,
            dynamic_range_ev=8.5 - float(plan.tone.black_ev),
        )
        changed = dataclasses.replace(plan, tone=changed_tone)
        original_hdr = compile_hdr_agx_plan(plan, analysis=analysis)
        changed_hdr = compile_hdr_agx_plan(changed, analysis=analysis)
        self.assertEqual(original_hdr.tone.white_ev, changed_hdr.tone.white_ev)
        self.assertEqual(original_hdr.formation.white_ev, changed_hdr.formation.white_ev)


def _coreimage_available() -> bool:
    try:
        from dngscan import coreimage_decode

        return coreimage_decode.runtime_available()
    except Exception:
        return False


@unittest.skipUnless(
    FRAMES["daylight"].is_file() and _coreimage_available(),
    "sample frames or Core Image decoder unavailable",
)
class Raw9HdrCouplingTests(unittest.TestCase):
    """RAW 9 scene-referred data must couple into the same HDR AgX contract.

    The Core Image decoder is a separate pipeline: warped geometry, no per-pixel CFA
    masks, aggregate-only RAW evidence. These gates pin the couplings that keep its HDR
    output honest — capped chroma freedom, rank-trimmed reliable tail, mask-free
    formation — against the LibRaw path on the same frame.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.pairs = {}
        for decoder in ("libraw", "coreimage"):
            bundle = load_raw(
                FRAMES["daylight"], scene_half_size=True, decoder=decoder
            )
            analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
            plan = build_render_plan(bundle, analysis, RENDER_MODE, "p3")
            hdr_plan = compile_hdr_agx_plan(
                plan, analysis=analysis, scene_decoder=bundle.scene_decoder
            )
            cls.pairs[decoder] = (bundle, analysis, plan, hdr_plan)

    def test_masks_are_dropped_not_faked(self) -> None:
        bundle, _, _, _ = self.pairs["coreimage"]
        self.assertEqual(bundle.scene_decoder, "coreimage")
        self.assertIsNone(bundle.clip_masks)

    def test_chroma_freedom_is_capped_without_aligned_evidence(self) -> None:
        from dngscan.hdr_agx_plan import (
            UNALIGNED_DECODER_RHO_CAP,
            compile_channel_separation,
        )

        _, analysis, _, hdr_plan = self.pairs["coreimage"]
        rho = compile_channel_separation(analysis, "coreimage")
        self.assertLessEqual(rho, UNALIGNED_DECODER_RHO_CAP + 1e-9)
        self.assertLessEqual(
            hdr_plan.color.channel_separation, UNALIGNED_DECODER_RHO_CAP + 1e-9
        )
        # The same frame under LibRaw earns more freedom, proving the cap binds here
        # rather than the scene being colour-poor.
        libraw_rho = compile_channel_separation(
            self.pairs["libraw"][1], "libraw"
        )
        self.assertGreater(libraw_rho, rho)

    def test_reliable_tail_agrees_with_libraw_within_policy_margin(self) -> None:
        """Rank-trimmed RAW9 tail must track the CFA-masked LibRaw measurement.

        The two decoders measure the same scene through different highlight machinery;
        if the rank-domain constraint works, their reliable tails differ by decode
        variance, not by reconstruction fabricating a brighter white endpoint. 0.3 EV is
        both looser than measured (0.09 EV) and tighter than the smallest policy step.
        """
        libraw_tail = self.pairs["libraw"][3].tone.reliable_tail_ev
        raw9_tail = self.pairs["coreimage"][3].tone.reliable_tail_ev
        self.assertTrue(math.isfinite(libraw_tail) and math.isfinite(raw9_tail))
        self.assertLess(abs(raw9_tail - libraw_tail), 0.3)

    def test_mask_free_hdr_formation_renders_in_volume(self) -> None:
        bundle, _, plan, hdr_plan = self.pairs["coreimage"]
        self.assertGreater(hdr_plan.tone.rendered_headroom_ev, 0.0)
        rendered = scene_render_to_hdr_display_linear(bundle, plan, hdr_plan, "p3")
        self.assertTrue(bool(np.all(np.isfinite(rendered))))
        self.assertGreaterEqual(float(np.min(rendered)), 0.0)
        self.assertLessEqual(
            float(np.max(rendered)), float(hdr_plan.tone.peak_linear) + 1e-5
        )


if __name__ == "__main__":
    unittest.main()
