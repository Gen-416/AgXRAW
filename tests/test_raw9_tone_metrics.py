# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from dngscan.tone import scene_tone_metrics, neutral_tone_plan
from dngscan.hdr_agx_plan import compile_hdr_agx_plan, describe_hdr_plan


def _bundle(scene, **kw):
    return SimpleNamespace(
        scene_rec2020_render=scene, scene_scale=1.0, exposure_gain=1.0,
        wb_mode="camera", camera_wb=None, applied_wb=None, daylight_wb=None,
        clip_masks=None, scene_decoder="coreimage", **kw,
    )


def _plan(metrics):
    return SimpleNamespace(tone=neutral_tone_plan("p3"), color=None, scene=metrics)


class ReliableTailAuthorityTests(unittest.TestCase):
    def test_dark_processing_losses_cannot_delete_valid_highlights(self) -> None:
        ev = np.zeros(20_000, dtype=np.float32)
        ev[:300] = -5.0
        ev[-200:] = 4.0
        scene = np.repeat((0.18 * np.exp2(ev))[:, None], 3, axis=1).reshape(100, 200, 3)
        # Independent reference excluded only the dark lost pixels. Both
        # decoders see the valid bright pixels; no spatial mapping is implied.
        ref = scene.reshape(-1, 3)[300:].copy()
        for loss in (0.0, 1.5, 100.0, None, float("nan")):
            with self.subTest(loss=loss):
                bundle = _bundle(scene, scene_processing_loss_pct=loss,
                                 scene_reliable_reference_rec2020=ref,
                                 scene_reliable_reference_pct=98.5)
                metrics = scene_tone_metrics(bundle, SimpleNamespace(cell_union_pct=0.0))
                self.assertAlmostEqual(metrics.reliable_tail_ev_p9999, 4.0, places=4)
                self.assertEqual(metrics.reliability_source, "sensor-reference")
                self.assertGreater(compile_hdr_agx_plan(_plan(metrics)).tone.rendered_headroom_ev, 1.0)

    def test_reconstructed_bright_pixels_cannot_override_sensor_reference(self) -> None:
        scene = np.full((100, 100, 3), 0.18, dtype=np.float32)
        scene[:10] *= 32.0  # A reconstructed highlight plateau in Apple RGB.
        reference = np.full((9000, 3), 0.18, dtype=np.float32)
        bundle = _bundle(scene, scene_reliable_reference_rec2020=reference,
                         scene_reliable_reference_pct=90.0)
        metrics = scene_tone_metrics(bundle, SimpleNamespace(cell_union_pct=10.0))
        self.assertAlmostEqual(metrics.reliable_tail_ev_p9999, 0.0, places=4)
        self.assertEqual(compile_hdr_agx_plan(_plan(metrics)).tone.rendered_headroom_ev, 0.0)

    def test_empty_successful_reference_does_not_fallback_to_image_estimate(self) -> None:
        bundle = _bundle(np.full((100, 100, 3), 4.0, dtype=np.float32),
                         scene_reliable_reference_rec2020=np.empty((0, 3), dtype=np.float32))
        metrics = scene_tone_metrics(bundle, SimpleNamespace(cell_union_pct=0.0))
        self.assertTrue(math.isnan(metrics.reliable_tail_ev_p9999))
        self.assertEqual(metrics.reliability_source, "sensor-reference")
        self.assertEqual(compile_hdr_agx_plan(_plan(metrics)).tone.rendered_headroom_ev, 0.0)

    def test_reference_failure_is_bounded_labelled_estimate_not_false_zero(self) -> None:
        bundle = _bundle(np.full((100, 100, 3), 4.0, dtype=np.float32),
                         scene_processing_loss_pct=None)
        analysis = SimpleNamespace(cell_union_pct=0.0,
                                   color_clip_k_of_all_pct={1: 0., 2: 0., 3: 0.},
                                   gamut_out_pct={})
        metrics = scene_tone_metrics(bundle, analysis)
        hdr = compile_hdr_agx_plan(_plan(metrics), analysis=analysis, scene_decoder="coreimage")
        self.assertEqual(metrics.reliability_source, "decoded-image-estimate")
        self.assertGreater(hdr.tone.rendered_headroom_ev, 0.0)
        self.assertLessEqual(hdr.tone.rendered_headroom_ev, 1.0)
        self.assertEqual(hdr.color.channel_separation, 0.0)
        self.assertIn("非传感器实测", describe_hdr_plan(hdr))

    def test_apple_only_unknown_sensor_stats_are_not_zero_clip_measurements(self) -> None:
        bundle = _bundle(np.full((100, 100, 3), 4.0, dtype=np.float32))
        metrics = scene_tone_metrics(bundle, SimpleNamespace(cell_union_pct=float("nan")))
        self.assertTrue(math.isnan(metrics.raw_clip_union_pct))
        self.assertEqual(metrics.reliability_source, "decoded-image-estimate")
        self.assertGreater(compile_hdr_agx_plan(_plan(metrics)).tone.rendered_headroom_ev, 0.0)

    def test_known_absence_of_reliable_sensor_support_vetoes_estimate(self) -> None:
        scene = np.full((100, 100, 3), 4.0, dtype=np.float32)
        for rate in (95.0, 96.0, 99.0, 100.0):
            with self.subTest(rate=rate):
                metrics = scene_tone_metrics(_bundle(scene), SimpleNamespace(cell_union_pct=rate))
                self.assertTrue(math.isnan(metrics.reliable_tail_ev_p9999))
                self.assertTrue(math.isfinite(metrics.body_ev_p50))

    def test_reference_normalized_units_ignore_storage_scale_and_follow_intent_ev(self) -> None:
        scene = np.full((100, 100, 3), 32.0, dtype=np.float32)
        bundle = _bundle(scene, scene_reliable_reference_rec2020=np.full((1000, 3), 2.0, dtype=np.float32))
        bundle.scene_scale = 16.0
        first = scene_tone_metrics(bundle, SimpleNamespace(cell_union_pct=0.0))
        second = scene_tone_metrics(bundle, SimpleNamespace(cell_union_pct=0.0), plan_exposure_gain=2.0)
        self.assertAlmostEqual(first.reliable_tail_ev_p9999, math.log2(2.0 / .18), places=4)
        self.assertAlmostEqual(second.reliable_tail_ev_p9999 - first.reliable_tail_ev_p9999, 1.0, places=5)

    def test_sdr_fallback_does_not_become_hdr_evidence(self) -> None:
        scene = np.ones((100, 100, 3), dtype=np.float32)
        bundle = _bundle(scene)
        bundle.clip_masks = np.ones_like(scene)
        bundle.scene_decoder = "libraw"
        metrics = scene_tone_metrics(bundle, SimpleNamespace(cell_union_pct=100.0))
        self.assertEqual(metrics.reliable_sample_pct, 0.0)
        self.assertTrue(math.isnan(metrics.reliable_tail_ev_p9999))
        self.assertTrue(math.isfinite(metrics.body_ev_p50))


if __name__ == "__main__":
    unittest.main()
