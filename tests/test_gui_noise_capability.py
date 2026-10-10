# SPDX-License-Identifier: GPL-3.0-or-later
"""GUI prerequisites quote the same optional-NR gates as the renderer."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan.analysis import analyze
from dngscan.gui.preview_cache import build_proxy_entry, _read_disk_entry, _write_disk_entry
from dngscan.gui.service import chroma_nr_capability, detected_scene_params
from dngscan.noise_propagation import chroma_nr_skip_reason, calibrated_chroma_variance
from dngscan.raw_io import load_raw
from dngscan.render import render_output_encoded_float
from dngscan.tone import build_render_plan
from tests.test_general_raw_fallback import write_uncalibrated_dng


class GuiNoiseCapabilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "calibration"))
        environment.start()
        self.addCleanup(environment.stop)

    def decoded(self, *, profile=True):
        path = self.root / "uncalibrated.dng"
        write_uncalibrated_dng(path, profile=profile)
        bundle = load_raw(path, scene_half_size=True)
        analysis, _, _ = analyze(bundle, 4, diagnostics=False)
        return bundle, analysis

    def rendered(self, bundle, analysis, *, scene_transform="none", strength=0.):
        plan = build_render_plan(bundle, analysis, "agx", "p3", chroma_nr=1.)
        render_output_encoded_float(
            bundle, analysis, "p3", plan,
            scene_transform=scene_transform, scene_transform_strength=strength,
        )
        return plan

    def test_valid_file_model_is_available_even_when_last_render_was_disabled(self):
        bundle, analysis = self.decoded()
        capability = chroma_nr_capability(bundle, analysis)
        self.assertTrue(capability["available"])
        self.assertIsNone(capability["reason"])
        self.assertTrue(capability["approximate"])
        self.assertEqual(bundle.chroma_nr_status, "disabled")
        plan = self.rendered(bundle, analysis)
        self.assertEqual(bundle.chroma_nr_status, "active-approximate")
        detected = detected_scene_params(bundle, analysis, plan)
        self.assertEqual(detected["chroma_nr_capability"], capability)
        self.assertEqual(detected["demosaic_actual"], bundle.noise_decode["demosaic_algorithm"])
        self.assertEqual(detected["capture_readout"], bundle.capture_readout or {})

    def test_normal_missing_model_does_not_claim_optional_nr_available(self):
        bundle, analysis = self.decoded(profile=False)
        capability = chroma_nr_capability(bundle, analysis)
        self.assertFalse(capability["available"])
        self.assertEqual(capability["reason"], "independent noise calibration unavailable")
        self.rendered(bundle, analysis)
        self.assertEqual(bundle.chroma_nr_status, "skipped")
        self.assertEqual(bundle.chroma_nr_reason, capability["reason"])

    def test_decoder_and_correction_restrictions_match_actual_renderer(self):
        bundle, analysis = self.decoded()
        for reason, decoder in (
            ("opaque-decoder-noise-transfer", "coreimage"),
            ("non-Bayer-noise-transfer-unavailable", "libraw"),
            ("noise-transfer-for-spatial-or-point-corrections-unavailable", "libraw"),
        ):
            with self.subTest(reason=reason):
                source = replace(bundle, scene_decoder=decoder,
                                 noise_decode={"supported": False, "reason": reason})
                capability = chroma_nr_capability(source, analysis)
                self.assertFalse(capability["available"])
                self.assertEqual(capability["reason"], reason)
                plan = self.rendered(source, analysis)
                self.assertEqual(source.chroma_nr_status, "skipped")
                self.assertEqual(source.chroma_nr_reason, reason)
                if decoder != "libraw":
                    # LibRaw reference metadata must not pretend to describe
                    # Apple's own demosaicing algorithm.
                    source.noise_decode["demosaic_algorithm"] = "DHT"
                    self.assertIsNone(detected_scene_params(source, analysis, plan)["demosaic_actual"])

    def test_spectral_restriction_and_variance_gate_share_the_same_reason(self):
        bundle, analysis = self.decoded()
        analysis.noise_model = replace(analysis.noise_model,
                                       correlation="measured-spectral-imbalance",
                                       spectral_ratios={"h": .1})
        capability = chroma_nr_capability(bundle, analysis)
        self.assertFalse(capability["available"])
        variance, reason = calibrated_chroma_variance(
            bundle, analysis.noise_model, bundle.scene_rec2020_render,
        )
        self.assertIsNone(variance)
        self.assertEqual(reason, capability["reason"])
        self.rendered(bundle, analysis)
        self.assertEqual(bundle.chroma_nr_reason, reason)

    def test_scene_transform_context_tracks_current_settings(self):
        bundle, analysis = self.decoded()
        blocked = chroma_nr_capability(bundle, analysis,
                                      scene_transform="alev_material_d55",
                                      scene_transform_strength=1.)
        self.assertFalse(blocked["available"])
        plan = self.rendered(bundle, analysis, scene_transform="alev_material_d55", strength=1.)
        self.assertEqual(bundle.chroma_nr_reason, blocked["reason"])
        detected = detected_scene_params(
            bundle, analysis, plan, scene_transform="alev_material_d55", scene_transform_strength=1.,
        )
        self.assertEqual(detected["chroma_nr_capability"], blocked)
        self.assertTrue(chroma_nr_capability(bundle, analysis,
                                           scene_transform="alev_material_d55",
                                           scene_transform_strength=0.)["available"])

    def test_compact_disk_proxy_retains_capability_after_releasing_raw(self):
        bundle, analysis = self.decoded()
        expected = chroma_nr_capability(bundle, analysis)
        self.assertTrue(expected["available"])
        entry = build_proxy_entry(bundle, analysis, include_guidance=True)
        cache_path = self.root / "preview.npz"
        _write_disk_entry(cache_path, entry)
        restored = _read_disk_entry(cache_path, bundle.path, require_guidance=True)
        self.assertIsNotNone(restored)
        self.assertIsNone(restored.bundle.raw_image)
        self.assertEqual(chroma_nr_capability(restored.bundle, restored.analysis), expected)
        self.rendered(restored.bundle, restored.analysis)
        self.assertEqual(restored.bundle.chroma_nr_status, "active-approximate")

    def test_coarse_grid_requirements_are_checked_without_a_render(self):
        bundle, analysis = self.decoded()
        reason = chroma_nr_skip_reason(bundle, analysis.noise_model, (128, 128))
        self.assertEqual(reason, "coarse noise approximation requires at least four sensels per cell")
        variance, actual = calibrated_chroma_variance(
            bundle, analysis.noise_model, np.zeros((128, 128, 3), np.float32),
        )
        self.assertIsNone(variance)
        self.assertEqual(actual, reason)


if __name__ == "__main__":
    unittest.main()
