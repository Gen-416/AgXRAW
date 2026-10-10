# SPDX-License-Identifier: GPL-3.0-or-later
"""Located sensor dependencies withdraw authority without inventing clipping."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from dngscan import guidance, hdr_agx, tone
from dngscan.noise_model import NoiseModel
from dngscan.noise_propagation import calibrated_chroma_variance
from dngscan.reliability import resize_exclusion
from tests.test_preview_cache import _analysis, _bundle


class ReliabilityConsumerTests(unittest.TestCase):
    def test_any_resize_matches_area_footprint_oracle(self):
        rng = np.random.default_rng(981)
        for source_shape, target_shape in (((7, 11), (3, 4)), ((3, 4), (7, 11)),
                                            ((5, 9), (5, 4)), ((8, 7), (3, 7))):
            for _ in range(8):
                source = (rng.random(source_shape) < .12).astype(np.uint8)
                expected = np.zeros(target_shape, np.uint8)
                h, w = source_shape
                dh, dw = target_shape
                for y in range(dh):
                    for x in range(dw):
                        expected[y, x] = np.any(source[y*h//dh:((y+1)*h+dh-1)//dh,
                                                      x*w//dw:((x+1)*w+dw-1)//dw])
                np.testing.assert_array_equal(resize_exclusion(source, target_shape), expected)

    def test_tone_selection_excludes_dependency_but_keeps_unaffected_rows(self):
        exclusion = np.zeros((8, 8), np.uint8)
        exclusion[3, 4] = 1
        bundle = replace(_bundle(), scene_reliability_exclusion=exclusion,
                         clip_masks=np.zeros((8, 8, 3), np.float16),
                         scene_geometry_crop=(2., 2., 6., 6.))
        # This final-scene mask must not inherit the visual evidence crop.
        baseline = tone.reliable_scene_ev_selection(replace(bundle, scene_reliability_exclusion=None), _analysis())[2]
        actual = tone.reliable_scene_ev_selection(bundle, _analysis())[2]
        expected = baseline.copy()
        expected[3*8+4] = False
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(bundle.clip_masks, 0)

    def test_guidance_only_closes_scene_permission_and_retains_raw_measurements(self):
        exclusion = np.zeros((8, 8), np.uint8)
        exclusion[3, 4] = 1
        maps = guidance.RawGuidanceMaps(np.ones((8, 8, 3), np.float16),
                                       np.zeros((8, 8), np.uint8), None,
                                       np.zeros((8, 8), np.float32))
        bundle = replace(_bundle(), raw_image=None, raw_colors=None,
                         raw_guidance=maps, scene_reliability_exclusion=exclusion)
        qualified = guidance.raw_guidance_for_shape(bundle, (8, 8), _analysis())
        for field in ("headroom", "clip_class", "raw_permission"):
            np.testing.assert_array_equal(getattr(qualified, field), getattr(maps, field))
        self.assertIsNone(qualified.snr_confidence)
        np.testing.assert_array_equal(qualified.scene_eligibility, 1-exclusion)
        ev = np.full(64, -4., np.float32)
        before = guidance.color_path_weight(None, ev, 8., noise_ev_floor=-6.)
        after = guidance.color_path_weight(None, ev, 8., noise_ev_floor=-6.,
                                          scene_eligibility=qualified.scene_eligibility.reshape(-1))
        np.testing.assert_array_equal(after[exclusion.reshape(-1) == 0], before[exclusion.reshape(-1) == 0])
        self.assertEqual(float(after[3*8+4]), 0.)
        resized = guidance.raw_guidance_for_shape(bundle, (3, 3), _analysis())
        self.assertIsNone(resized.snr_confidence)
        np.testing.assert_array_equal(resized.scene_eligibility[resize_exclusion(exclusion, (3, 3)) != 0], 0)
        self.assertTrue(np.any(resized.scene_eligibility > 0))

    def test_coarse_noise_cell_with_any_dependency_has_no_authority(self):
        exclusion = np.zeros((127, 129), np.uint8)
        exclusion[63, 64] = 1
        bundle = SimpleNamespace(
            raw_image=None, raw_pattern=[[0, 1], [3, 2]], color_desc="RGBG",
            scene_decoder="libraw", wb_mode="camera", exposure_gain=1.,
            lens_filter="none", lens_shading=None, scene_geometry_ops=(),
            scene_reliability_exclusion=exclusion,
            noise_decode={"supported": True, "normalized_raw_to_scene": np.eye(3),
                          "sensor_window_shape": [508, 516]},
        )
        model = NoiseModel(status="valid", channel_variance={c: (0., .004) for c in "RGB"})
        variance, _, valid = calibrated_chroma_variance(
            bundle, model, np.full((16, 17, 3), .2), return_validity=True)
        expected = resize_exclusion(exclusion, (16, 17)) == 0
        np.testing.assert_array_equal(valid, expected)
        np.testing.assert_array_equal(variance[~valid], 0)
        self.assertTrue(np.all(variance[valid] > 0))

    def test_auto_ev_external_sample_must_have_matching_dependency_qualification(self):
        from unittest.mock import patch
        from dngscan import auto_ev

        bundle = replace(_bundle(), scene_reliability_exclusion=np.eye(8, dtype=np.uint8))
        analysis = _analysis()
        plan = tone.build_render_plan(bundle, analysis, "agx", "p3", tone_core="gated")
        rows = np.full((2, 3), 4000., np.float32)
        maps = guidance.RawGuidanceMaps(np.ones((2, 3), np.float16), np.zeros(2, np.uint8),
                                       np.ones(2, np.float16), np.zeros(2, np.float32))
        observed = []
        original = auto_ev.apply_tone_core

        def capture(rgb, effective, color, masks, qualified):
            observed.append(qualified)
            return original(rgb, effective, color, masks, qualified)

        with patch.object(auto_ev, "apply_tone_core", side_effect=capture):
            for exclusion in (None, np.asarray([1, 0], np.uint8)):
                auto_ev.render_sample_linear_output(bundle, analysis, "p3", 0., rows,
                                                    tone_plan=plan, sample_raw_guidance=maps,
                                                    sample_exclusion=exclusion)
        np.testing.assert_array_equal(observed[0].snr_confidence, 0.)
        np.testing.assert_array_equal(observed[1].snr_confidence, [1., 1.])
        np.testing.assert_array_equal(observed[1].scene_eligibility, [0., 1.])
        np.testing.assert_array_equal(maps.snr_confidence, 1.)

    def test_optional_chroma_correction_does_not_touch_excluded_sources(self):
        from dngscan.render import _prepare_chroma_nr_map

        rng = np.random.default_rng(93)
        shape = (192, 192)
        scene = (.03 + rng.normal(0., .003, shape + (3,))).astype(np.float32)
        exclusion = np.zeros(shape, np.uint8)
        exclusion[95, 95] = 1
        model = NoiseModel(status="valid", channel_variance={c: (0., .001) for c in "RGB"})
        bundle = replace(_bundle(), scene_rec2020_render=scene, scene_scale=1., render_scale=1.,
                         raw_image=None, scene_reliability_exclusion=exclusion,
                         scene_sensor_window_shape=(768., 768.), noise_model=model,
                         noise_decode={"supported": True, "sensor_window_shape": [768, 768],
                                       "normalized_raw_to_scene": np.eye(3)})
        correction = _prepare_chroma_nr_map(bundle, SimpleNamespace(chroma_nr=1.), None,
                                            scene.reshape(-1, 3), None, *shape,
                                            "none", 0., None)
        self.assertEqual(bundle.chroma_nr_status, "active-approximate")
        np.testing.assert_array_equal(correction[95, 95], 0.)
        self.assertGreater(float(np.max(np.abs(correction))), 0.)

    def test_hdr_dependency_uses_conservative_colour_without_changing_sdr(self):
        from tests.golden_support import build_daylight_wide_dr
        from dngscan.hdr_agx_plan import compile_hdr_agx_plan
        from dngscan.render import render_output_u8

        scene = build_daylight_wide_dr()
        bundle, analysis = scene.bundle, scene.analysis
        plan = tone.build_render_plan(bundle, analysis, "agx", "p3")
        hdr_plan = compile_hdr_agx_plan(plan, analysis=analysis)
        hdr_plan = replace(hdr_plan, color=replace(hdr_plan.color, channel_separation=1., snr_gate=1.))
        conservative = replace(hdr_plan, color=replace(hdr_plan.color, channel_separation=0.))
        baseline_sdr, baseline_hdr = hdr_agx.render_ultrahdr_agx_pair(bundle, analysis, plan, hdr_plan)
        reference = hdr_agx.scene_render_to_hdr_display_linear(bundle, plan, conservative, analysis=analysis)
        exclusion = np.zeros(bundle.scene_rec2020_render.shape[:2], np.uint8)
        exclusion[20:30, 20:30] = 1
        qualified = replace(bundle, scene_reliability_exclusion=exclusion)
        sdr, hdr = hdr_agx.render_ultrahdr_agx_pair(qualified, analysis, plan, hdr_plan)
        standalone = hdr_agx.scene_render_to_hdr_display_linear(qualified, plan, hdr_plan, analysis=analysis)
        bad = exclusion != 0
        np.testing.assert_array_equal(sdr, baseline_sdr)
        np.testing.assert_array_equal(sdr, render_output_u8(qualified, analysis, "p3", plan))
        np.testing.assert_allclose(hdr[bad], reference[bad], atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(hdr[~bad], baseline_hdr[~bad])
        np.testing.assert_allclose(hdr, standalone, atol=2e-5, rtol=2e-5)
        self.assertGreater(float(np.max(np.abs(baseline_hdr[bad]-reference[bad]))), 1e-5)

if __name__ == "__main__":
    unittest.main()
