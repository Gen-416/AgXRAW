# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoder distrust survives handoffs without becoming fabricated RGB loss."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import auto_ev, guidance, raw_io, render, tone
from dngscan._deps import rawpy
from dngscan.analysis import analyze, reanalyze_balanced_scene
from dngscan.gui import preview_cache as cache
from dngscan.models import RawGuidanceMaps
from dngscan.noise_model import NoiseModel
from dngscan.noise_propagation import calibrated_chroma_variance
from dngscan.prepared_sample import PreparedSceneSample
from tests.test_pipeline_corrections import write_sensor_dng
from tests.test_preview_cache import _analysis, _bundle


def _maps(shape):
    headroom = np.ones(shape + (3,), np.float16)
    headroom.reshape(-1, 3)[0, 0] = 0.
    classes = np.zeros(shape, np.uint8)
    classes.reshape(-1)[0] = 1  # Only the measured red channel clipped.
    permission = guidance.raw_color_permission(
        headroom_rgb=headroom.reshape(-1, 3), clip_class=classes.reshape(-1),
    ).reshape(shape)
    return RawGuidanceMaps(headroom, classes, np.ones(shape, np.float16), permission)


class LossQualificationHandoffTests(unittest.TestCase):
    def assert_no_nonraw_permission(self, maps):
        flat = guidance.flatten_raw_guidance(maps, 0, int(np.prod(maps.headroom.shape[:-1])))
        size = flat.headroom.shape[0]
        # Bright scene and maximum gamut pressure would otherwise open the
        # scene-driven path. Its weight must contain only measured RAW loss.
        actual = guidance.color_path_weight(
            None, np.full(size, 3., np.float32), 8.,
            raw_headroom_rgb=flat.headroom, raw_clip_class=flat.clip_class,
            raw_snr_confidence=flat.snr_confidence, raw_permission=flat.raw_permission,
            midtone_protect=0.,
        )
        np.testing.assert_array_equal(actual, flat.raw_permission)
        np.testing.assert_array_equal(flat.snr_confidence, 0.)

    def test_real_deferred_decode_proxy_disk_and_wb_keep_qualification(self):
        dht = getattr(rawpy.DemosaicAlgorithm, "DHT", None)
        if dht is None or not dht.isSupported:
            self.skipTest("LibRaw build does not support DHT")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "local-overflow.dng"
            pixels = np.full((128, 128), 1000, np.uint16)
            pixels[64, 64] = 3000
            write_sensor_dng(path, signal=pixels, neutral=(.5, 1., 1.))
            bundle = raw_io.load_raw(path, demosaic="dht", _defer_clip_masks=True)
            self.assertTrue(bundle.scene_loss_support_untrusted)
            self.assertTrue(bundle._clip_masks_pending)
            self.assertIsNone(bundle.clip_masks)
            analysis, _, _ = analyze(bundle, 4)
            self.assertTrue(bundle.scene_loss_support_untrusted)
            self.assertFalse(bundle._clip_masks_pending)
            np.testing.assert_array_equal(bundle.clip_masks[16, 16], 0.)
            with patch.object(cache, "PROXY_LONG_EDGE", 64):
                entry = cache.build_proxy_entry(bundle, analysis, include_guidance=True)
            self.assertTrue(entry.bundle.scene_loss_support_untrusted)
            self.assertTrue(entry.source_metadata["scene_loss_support_untrusted"])
            self.assertIsNone(entry.bundle.raw_image)
            np.testing.assert_array_equal(entry.bundle._tone_plan_sample_masks[16*128+16], 0.)
            self.assert_no_nonraw_permission(entry.bundle.raw_guidance)
            disk = Path(directory) / "entry.npz"
            cache._write_disk_entry(disk, entry)
            restored = cache._read_disk_entry(disk, path, require_guidance=True)
            self.assertIsNotNone(restored)
            self.assertTrue(restored.bundle.scene_loss_support_untrusted)
            self.assertTrue(restored.source_metadata["scene_loss_support_untrusted"])
            np.testing.assert_array_equal(restored.bundle._tone_plan_sample_masks,
                                          entry.bundle._tone_plan_sample_masks)
            for wb in ("camera", "daylight"):
                balanced = raw_io.rebalance_raw_bundle(restored.bundle, wb)
                balanced_analysis = reanalyze_balanced_scene(restored.analysis, balanced)
                self.assertTrue(balanced.scene_loss_support_untrusted)
                selected = tone.reliable_scene_ev_selection(balanced, balanced_analysis)
                self.assertFalse(selected[3])
                self.assertFalse(np.any(selected[2]))
                self.assertEqual(tone.scene_tone_metrics(balanced, balanced_analysis).reliability_source,
                                 "decoder-support-untrusted")
                self.assert_no_nonraw_permission(guidance.raw_guidance_for_shape(
                    balanced, balanced.scene_rec2020_render.shape[:2], balanced_analysis))

    def test_cache_source_identity_distinguishes_qualification(self):
        trusted = _bundle()
        untrusted = replace(trusted, scene_loss_support_untrusted=True)
        first, second = cache._bundle_metadata(trusted), cache._bundle_metadata(untrusted)
        self.assertNotEqual(first, second)
        self.assertFalse(first["scene_loss_support_untrusted"])
        self.assertTrue(second["scene_loss_support_untrusted"])
        self.assertGreaterEqual(cache.PREVIEW_CACHE_VERSION, 27)

    def test_prepared_sample_and_private_ev_copy_keep_local_masks_and_distrust(self):
        bundle = replace(_bundle(), scene_loss_support_untrusted=True,
                         clip_masks=np.zeros((8, 8, 3), np.float16))
        prepared = PreparedSceneSample.from_bundle(bundle)
        private = prepared.bind(replace(bundle, exposure_gain=2.))
        self.assertTrue(private.scene_loss_support_untrusted)
        np.testing.assert_array_equal(prepared.masks, 0.)
        selection = tone.reliable_scene_ev_selection(private, _analysis())
        self.assertFalse(selection[3])
        self.assertFalse(np.any(selection[2]))
        self.assertEqual(tone.scene_tone_metrics(private, _analysis()).reliability_source,
                         "decoder-support-untrusted")

    def test_nonempty_coreimage_reference_cannot_restore_revoked_authority(self):
        # The current Core Image producer does not create this combination,
        # but consumer qualification must remain authoritative after a handoff.
        bundle = replace(
            _bundle(), scene_decoder="coreimage", scene_scale=1.,
            scene_rec2020_render=np.full((64, 64, 3), 32., np.float32),
            scene_reliable_reference_rec2020=np.full((4096, 3), 32., np.float32),
            scene_reliable_reference_pct=100., scene_reliability_source="sensor-reference",
        )
        trusted = tone.scene_tone_metrics(bundle, _analysis())
        self.assertEqual(trusted.reliability_source, "sensor-reference")
        self.assertTrue(np.isfinite(trusted.reliable_tail_ev_p9999))
        self.assertEqual(trusted.reliable_sample_pct, 100.)
        bundle.scene_loss_support_untrusted = True
        selection = tone.reliable_scene_ev_selection(bundle, _analysis())
        self.assertFalse(selection[3])
        self.assertFalse(np.any(selection[2]))
        metrics = tone.scene_tone_metrics(bundle, _analysis())
        self.assertEqual(metrics.reliability_source, "decoder-support-untrusted")
        self.assertTrue(np.isnan(metrics.reliable_tail_ev_p9999))
        self.assertEqual(metrics.reliable_sample_pct, 0.)

    def test_cached_guidance_keeps_measured_red_loss_without_scene_permission(self):
        original = _maps((4, 5))
        bundle = replace(_bundle(), raw_image=None, raw_colors=None,
                         raw_guidance=original, scene_loss_support_untrusted=True)
        actual = guidance.raw_guidance_for_shape(bundle, (4, 5), _analysis())
        for field in ("headroom", "clip_class", "raw_permission"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(original, field))
        self.assertEqual(int(actual.clip_class[0, 0]), 1)
        self.assertGreater(float(actual.raw_permission[0, 0]), 0.)
        np.testing.assert_array_equal(original.snr_confidence, 1.)
        self.assert_no_nonraw_permission(actual)

    def test_resized_cached_and_uncached_guidance_remain_qualified(self):
        source, cached = _maps((4, 5)), _maps((9, 11))
        bundle = replace(_bundle(), raw_image=None, raw_colors=None,
                         raw_guidance=source, scene_loss_support_untrusted=True,
                         _raw_guidance_cache_shape=(9, 11), _raw_guidance_resized=cached)
        actual = guidance.raw_guidance_for_shape(bundle, (9, 11), _analysis())
        np.testing.assert_array_equal(actual.raw_permission, cached.raw_permission)
        np.testing.assert_array_equal(actual.clip_class, cached.clip_class)
        np.testing.assert_array_equal(cached.snr_confidence, 1.)
        self.assert_no_nonraw_permission(actual)
        bundle._raw_guidance_cache_shape = None
        bundle._raw_guidance_resized = None
        resized = guidance.raw_guidance_for_shape(bundle, (9, 11), _analysis())
        self.assertGreater(float(resized.raw_permission[0, 0]), 0.)
        self.assertEqual(int(resized.clip_class[0, 0]), 1)
        self.assert_no_nonraw_permission(resized)

    def test_absent_spatial_guidance_does_not_reopen_scene_permission(self):
        bundle = replace(_bundle(), raw_image=None, raw_colors=None,
                         clip_masks=None, scene_loss_support_untrusted=True)
        actual = guidance.raw_guidance_for_shape(bundle, (8, 8), _analysis())
        self.assertIsNotNone(actual)
        np.testing.assert_array_equal(actual.headroom, 1.)
        np.testing.assert_array_equal(actual.clip_class, 0)
        np.testing.assert_array_equal(actual.raw_permission, 0.)
        self.assert_no_nonraw_permission(actual)

    def test_auto_ev_explicit_sample_guidance_cannot_bypass_qualification(self):
        bundle = replace(_bundle(), scene_loss_support_untrusted=True)
        analysis = _analysis()
        plan = tone.build_render_plan(bundle, analysis, "agx", "p3", tone_core="gated")
        sample = np.asarray([[16000., 4000., 2000.], [4000., 16000., 2000.]], np.float32)
        supplied = _maps((2,))
        observed = []
        original = auto_ev.apply_tone_core

        def capture(rgb, effective, color, masks, maps):
            observed.append(maps)
            return original(rgb, effective, color, masks, maps)

        with patch.object(auto_ev, "apply_tone_core", side_effect=capture):
            output = auto_ev.render_sample_linear_output(
                bundle, analysis, "p3", 0., sample, tone_plan=plan,
                sample_raw_guidance=supplied,
            )
            auto_ev.render_sample_linear_output(bundle, analysis, "p3", 0., sample, tone_plan=plan)
        self.assertTrue(np.isfinite(output).all())
        self.assertEqual(len(observed), 2)
        for field in ("headroom", "clip_class", "raw_permission"):
            np.testing.assert_array_equal(getattr(observed[0], field), getattr(supplied, field))
        self.assert_no_nonraw_permission(observed[0])
        self.assert_no_nonraw_permission(observed[1])
        np.testing.assert_array_equal(supplied.snr_confidence, 1.)

    def test_public_sdr_entrypoints_qualify_bare_gated_tone_plan(self):
        bundle = replace(_bundle(), raw_image=None, raw_colors=None, clip_masks=None,
                         scene_loss_support_untrusted=True)
        analysis = _analysis()
        plan = tone.build_render_plan(bundle, analysis, "agx", "p3", tone_core="gated").tone
        observed = []
        original = render.apply_tone_core

        def capture(rgb, effective, color, masks, maps):
            self.assertIsNone(color)  # The supported bare ToneCompressionPlan API.
            observed.append(maps)
            return original(rgb, effective, color, masks, maps)

        with patch.object(render, "apply_tone_core", side_effect=capture):
            floating = render.scene_render_to_display_linear(bundle, plan, "p3", analysis=analysis)
            encoded = render.render_output_u8(bundle, analysis, "p3", plan)
        self.assertTrue(np.isfinite(floating).all())
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertEqual(len(observed), 2)
        for maps in observed:
            self.assertIsNotNone(maps)
            np.testing.assert_array_equal(maps.headroom, 1.)
            np.testing.assert_array_equal(maps.clip_class, 0)
            self.assert_no_nonraw_permission(maps)

    def test_direct_noise_propagation_stays_closed_for_both_contract_fields(self):
        model = NoiseModel(status="valid", channel_variance={"R": (.001, .00001)})
        for global_flag, descriptor_flag in ((True, False), (False, True)):
            bundle = replace(_bundle(), scene_loss_support_untrusted=global_flag,
                             noise_decode={"supported": True, "loss_support_untrusted": descriptor_flag})
            result = calibrated_chroma_variance(bundle, model, np.ones((4, 4, 3), np.float32),
                                                 return_validity=True)
            self.assertIsNone(result[0])
            self.assertIn("uncertified", result[1])


if __name__ == "__main__":
    unittest.main()
