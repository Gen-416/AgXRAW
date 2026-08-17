# SPDX-License-Identifier: GPL-3.0-or-later
"""Realtime preview histogram contracts.

The scene EV histogram must consume the tone planner's own reliable-sample
selection (one declared sampling, tone.reliable_scene_ev_selection) so the GUI
never describes data the render did not see; the display histogram must
describe the exact rendered u8 frame. Payloads must stay small, integer and
JSON-safe, and evidence that is absent must be absent (None), never invented.
"""
from __future__ import annotations

import json
import math
import unittest
from types import SimpleNamespace

import numpy as np

from dngscan.constants import OUTPUT_REFERENCE_WHITE_STOPS
from dngscan.gui.histogram import (
    DISPLAY_HIST_SAMPLE_TARGET,
    HISTOGRAM_BINS,
    SCENE_EV_MAX,
    SCENE_EV_MIN,
    _scene_counts,
    display_histogram,
    hdr_earned_ev,
    scene_ev_base,
    scene_ev_histogram,
)
from dngscan.tone import (
    compute_exposure_gain,
    reliable_scene_ev_selection,
    scene_tone_metrics,
)


def _plan(black=-8.0, white=3.5, tail=2.5):
    return SimpleNamespace(
        tone=SimpleNamespace(black_ev=black, white_ev=white),
        scene=SimpleNamespace(reliable_tail_ev_p9999=tail),
    )


def _bundle(scene, clip_masks=None):
    return SimpleNamespace(
        scene_rec2020_render=scene,
        scene_scale=1.0,
        exposure_gain=1.0,
        wb_mode="camera",
        camera_wb=None,
        applied_wb=None,
        daylight_wb=None,
        clip_masks=clip_masks,
        scene_decoder="libraw",
    )


class SceneCountsTests(unittest.TestCase):
    def test_bin_edges_are_half_open_over_declared_range(self) -> None:
        width = (SCENE_EV_MAX - SCENE_EV_MIN) / HISTOGRAM_BINS
        ev = np.asarray(
            [
                SCENE_EV_MIN,            # first bin, inclusive lower edge
                SCENE_EV_MIN - 1e-3,     # below range: dropped
                SCENE_EV_MAX,            # right edge: dropped (half-open)
                SCENE_EV_MAX - width / 2,  # last bin
                0.0,
            ],
            dtype=np.float32,
        )
        counts = _scene_counts(ev)
        self.assertEqual(counts.shape[0], HISTOGRAM_BINS)
        self.assertEqual(int(counts.sum()), 3)
        self.assertEqual(int(counts[0]), 1)
        self.assertEqual(int(counts[HISTOGRAM_BINS - 1]), 1)
        zero_bin = int((0.0 - SCENE_EV_MIN) / width)
        self.assertEqual(int(counts[zero_bin]), 1)

    def test_matches_numpy_histogram_modulo_float32_edge_rounding(self) -> None:
        # The fast manual binning computes indices in float32; samples within one
        # float32 ulp of a bin edge may land one bin apart from np.histogram's
        # edge-corrected assignment. Totals and everything off-edge must agree.
        rng = np.random.default_rng(7)
        ev = (rng.random(50_000, dtype=np.float32) * 20.0 - 13.0).astype(np.float32)
        expected, _ = np.histogram(ev, bins=HISTOGRAM_BINS, range=(SCENE_EV_MIN, SCENE_EV_MAX))
        actual = _scene_counts(ev)
        self.assertEqual(int(actual.sum()), int(expected.sum()))
        self.assertLessEqual(int(np.abs(actual - expected).max()), 1)
        # Off-edge samples agree exactly: re-binning only the samples that sit
        # clearly inside a bin reproduces numpy bin-for-bin.
        width = (SCENE_EV_MAX - SCENE_EV_MIN) / HISTOGRAM_BINS
        frac = (ev - SCENE_EV_MIN) / width
        interior = np.abs(frac - np.round(frac)) > 1e-3
        expected_int, _ = np.histogram(
            ev[interior], bins=HISTOGRAM_BINS, range=(SCENE_EV_MIN, SCENE_EV_MAX)
        )
        np.testing.assert_array_equal(_scene_counts(ev[interior]), expected_int)

    def test_extreme_values_do_not_break_the_truncation_offset(self) -> None:
        ev = np.asarray([-40.0, 40.0, SCENE_EV_MIN + 0.01], dtype=np.float32)
        counts = _scene_counts(ev)
        self.assertEqual(int(counts.sum()), 1)


class SceneSelectionConsistencyTests(unittest.TestCase):
    """The declared-consistency proof: histogram and tone plan share one selection."""

    def _scene(self) -> np.ndarray:
        rng = np.random.default_rng(3)
        values = rng.uniform(0.005, 1.5, size=(100, 100, 1)).astype(np.float32)
        return np.repeat(values, 3, axis=2)

    def test_tone_metrics_tail_is_the_selection_evidence_percentile(self) -> None:
        scene = self._scene()
        masks = np.zeros_like(scene)
        masks[:20, :, :] = 1.0  # 20% RAW-clipped
        bundle = _bundle(scene, clip_masks=masks)
        analysis = SimpleNamespace(cell_union_pct=0.0)
        gain0 = compute_exposure_gain("agx", 0.0)

        ev, body, evidence, evidence_ok = reliable_scene_ev_selection(
            bundle, analysis, exposure_gain=gain0
        )
        metrics = scene_tone_metrics(bundle, analysis, plan_exposure_gain=gain0)

        self.assertTrue(evidence_ok)
        self.assertAlmostEqual(
            metrics.reliable_tail_ev_p9999,
            float(np.percentile(ev[evidence], 99.99)),
            places=10,
        )
        self.assertAlmostEqual(
            metrics.body_ev_p50, float(np.percentile(ev[body], 50.0)), places=10
        )

    def test_scene_base_population_is_the_selection_body(self) -> None:
        scene = self._scene()
        masks = np.zeros_like(scene)
        masks[:30, :, :] = 1.0
        bundle = _bundle(scene, clip_masks=masks)
        analysis = SimpleNamespace(cell_union_pct=0.0)
        gain0 = compute_exposure_gain("agx", 0.0)

        base = scene_ev_base(bundle, analysis)
        ev, body, _, _ = reliable_scene_ev_selection(bundle, analysis, exposure_gain=gain0)

        np.testing.assert_array_equal(base["ev_body"], ev[body].astype(np.float32))
        self.assertEqual(base["sample_count"], int(np.count_nonzero(body)))
        # RAW-clipped samples were actually removed from the histogram population.
        self.assertEqual(base["sample_count"], 7000)

    def test_base_ignores_the_bundles_intent_exposure(self) -> None:
        scene = self._scene()
        analysis = SimpleNamespace(cell_union_pct=0.0)
        one = scene_ev_base(_bundle(scene), analysis)
        boosted_bundle = _bundle(scene)
        boosted_bundle.exposure_gain = 123.0
        boosted = scene_ev_base(boosted_bundle, analysis)
        np.testing.assert_array_equal(one["ev_body"], boosted["ev_body"])


class SceneHistogramPayloadTests(unittest.TestCase):
    def test_user_ev_shifts_population_and_tail_but_not_endpoints(self) -> None:
        ev_body = np.linspace(-6.0, 2.0, 4096, dtype=np.float32)
        base = {"ev_body": ev_body, "sample_count": ev_body.shape[0]}
        at0 = scene_ev_histogram(base, _plan(tail=2.5), 0.0)
        at1 = scene_ev_histogram(base, _plan(tail=2.5), 1.0)

        expected1 = _scene_counts(ev_body + np.float32(1.0))
        np.testing.assert_array_equal(np.asarray(at1["counts"]), expected1)
        self.assertEqual(at0["black_ev"], -8.0)
        self.assertEqual(at1["black_ev"], -8.0)
        self.assertEqual(at1["white_ev"], 3.5)
        self.assertEqual(at0["pivot_ev"], 0.0)
        self.assertAlmostEqual(at0["reliable_tail_ev"], 2.5)
        self.assertAlmostEqual(at1["reliable_tail_ev"], 3.5)

    def test_absent_evidence_is_none_not_a_number(self) -> None:
        base = {"ev_body": np.zeros(16, dtype=np.float32), "sample_count": 16}
        payload = scene_ev_histogram(base, _plan(tail=float("nan")), 0.5)
        self.assertIsNone(payload["reliable_tail_ev"])

    def test_payload_is_json_safe_and_small(self) -> None:
        rng = np.random.default_rng(0)
        base = {
            "ev_body": (rng.random(100_000, dtype=np.float32) * 14 - 10),
            "sample_count": 100_000,
        }
        payload = scene_ev_histogram(base, _plan(), 0.25)
        encoded = json.dumps(payload, allow_nan=False)
        self.assertLess(len(encoded), 4096)
        self.assertEqual(len(payload["counts"]), HISTOGRAM_BINS)
        self.assertTrue(all(isinstance(v, int) for v in payload["counts"]))


class DisplayHistogramTests(unittest.TestCase):
    def test_channels_and_luma_bins_map_code_values_correctly(self) -> None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[0, 0] = (255, 255, 255)
        frame[0, 1] = (255, 0, 0)
        payload = display_histogram(frame)
        self.assertEqual(payload["bins"], HISTOGRAM_BINS)
        self.assertEqual(payload["sample_count"], 100)
        self.assertEqual(payload["r"][127], 2)
        self.assertEqual(payload["r"][0], 98)
        self.assertEqual(payload["g"][127], 1)
        self.assertEqual(payload["b"][127], 1)
        # Pure white keeps luma in the top bin; pure red lands at 54*255>>9 = 26.
        self.assertEqual(payload["luma"][127], 1)
        self.assertEqual(payload["luma"][(54 * 255) >> 9], 1)
        for name in ("r", "g", "b", "luma"):
            self.assertEqual(sum(payload[name]), payload["sample_count"])

    def test_large_frames_use_the_declared_deterministic_stride(self) -> None:
        h, w = 1280, 1920
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        payload = display_histogram(frame)
        step = math.ceil(h * w / DISPLAY_HIST_SAMPLE_TARGET)
        self.assertGreater(step, 1)
        self.assertEqual(payload["sample_count"], len(range(0, h * w, step)))

    def test_payload_is_json_safe(self) -> None:
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        payload = display_histogram(frame)
        json.dumps(payload, allow_nan=False)


class PreviewPayloadFieldTests(unittest.TestCase):
    def test_preview_response_carries_both_histograms_and_the_hdr_scalar(self) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        import dngscan as dg
        from dngscan.gui.preview_cache import PreviewEntry
        from dngscan.gui.service import export_preview_jpeg

        bundle = MagicMock()
        bundle.exposure_gain = 1.0
        bundle.scene_scale = 1.0
        bundle.scene_decoder = "libraw"
        bundle.scene_decoder_runtime = "test"
        bundle.scene_rec2020_render = dg.np.full((12, 18, 3), 400, dtype=dg.np.uint16)
        bundle.lens_filter = "none"
        bundle.wb_mode = "camera"
        bundle.camera_wb = None
        bundle.applied_wb = None
        bundle.daylight_wb = None
        bundle.clip_masks = None
        # R2 item 20: real proxy bundles declare these; a MagicMock would
        # hand the sampler a truthy child mock instead.
        bundle._tone_plan_sample = None
        bundle._tone_plan_sample_masks = None
        entry = PreviewEntry(bundle=bundle, analysis=MagicMock())
        pixels = dg.np.zeros((12, 18, 3), dtype=dg.np.uint8)
        plan = _plan(black=-8.0, white=3.5, tail=float(OUTPUT_REFERENCE_WHITE_STOPS) + 0.5)
        with patch(
            "dngscan.gui.service.dg.with_intent_exposure", return_value=bundle
        ), patch(
            "dngscan.gui.service._cached_render_plan", return_value=plan
        ), patch(
            "dngscan.gui.service.dg.render_output_u8", return_value=pixels
        ), patch(
            "dngscan.gui.service.dg.output_icc_profile_bytes", return_value=None
        ):
            payload = export_preview_jpeg(
                Path("synthetic.dng"), "clip", "srgb", 0.0, 95, cached=entry
            )

        scene = payload["scene_histogram"]
        display = payload["display_histogram"]
        self.assertEqual(scene["kind"], "scene_ev")
        self.assertEqual(len(scene["counts"]), HISTOGRAM_BINS)
        self.assertEqual(scene["black_ev"], -8.0)
        self.assertEqual(scene["white_ev"], 3.5)
        self.assertEqual(display["kind"], "display")
        for name in ("r", "g", "b", "luma"):
            self.assertEqual(len(display[name]), HISTOGRAM_BINS)
        self.assertAlmostEqual(payload["hdr_earned_ev"], 0.5)
        json.dumps(payload["scene_histogram"], allow_nan=False)
        json.dumps(payload["display_histogram"], allow_nan=False)


class HdrEarnedTests(unittest.TestCase):
    def test_earned_headroom_matches_detected_scene_params_definition(self) -> None:
        tail = float(OUTPUT_REFERENCE_WHITE_STOPS) + 1.25
        self.assertAlmostEqual(hdr_earned_ev(_plan(tail=tail)), 1.25)
        self.assertEqual(hdr_earned_ev(_plan(tail=float(OUTPUT_REFERENCE_WHITE_STOPS) - 1.0)), 0.0)

    def test_absent_tail_is_none(self) -> None:
        self.assertIsNone(hdr_earned_ev(_plan(tail=float("nan"))))


if __name__ == "__main__":
    unittest.main()
