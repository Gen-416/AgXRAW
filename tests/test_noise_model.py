# SPDX-License-Identifier: GPL-3.0-or-later
"""Known-noise counterexamples through the real analysis/HDR seam."""
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import struct
from unittest.mock import patch

import numpy as np

from dngscan.analysis import analyze, noise_floor_ev_estimate, snr_ev_coordinates
from dngscan.calibration import import_calibration
from dngscan.constants import MIDGRAY_HEADROOM_STOPS
from dngscan.hdr_agx_plan import compile_tail_snr_gate
from dngscan.noise_model import NoiseModel, model_snr_curves, resolve_noise_model
from tests.golden_support import _bundle_from_scene
from tests.test_user_calibration import single_profile, collect_profile, unresolved_collect_profile


def frame(texture=False, *, known=True):
    rng = np.random.default_rng(740)
    y, x = np.indices((512, 512))
    signal = np.full((512, 512), 2400.)
    if texture:
        signal += 800 * np.sin(2 * np.pi * x / 128)
    raw = np.rint(signal + rng.normal(0, 1, signal.shape)).astype(np.uint16)
    pattern = np.array([[0, 1], [3, 2]], np.uint8)
    colours = np.tile(pattern, (256, 256))
    bundle = _bundle_from_scene(np.full((256, 256, 3), 8000, np.uint16),
                                raw_image=raw, raw_colors=colours)
    return replace(bundle, path=Path("noise-model-fixture.dng"),
                   white_level=16383, camera_white_levels=[16383.] * 4,
                   black_levels=[1024.] * 4,
                   shot_make="SIGMA" if known else "unmeasured",
                   shot_model="SIGMA fp" if known else "unmeasured", shot_iso=640)


class AnalysisNoiseModelTests(unittest.TestCase):
    def test_texture_cannot_revoke_calibration_or_change_model_snr(self):
        flat, _, _ = analyze(frame(), 4, diagnostics=True)
        textured_bundle = frame(True)
        textured, _, _ = analyze(textured_bundle, 4, diagnostics=True)
        self.assertGreater(textured.health_lag1_corr, .2)
        self.assertEqual(textured.noise_correlation_status, "spatial-residual-clue")
        self.assertEqual(textured.noise_evidence_status, "model-valid")
        self.assertEqual(flat.noise_model, textured.noise_model)
        self.assertIs(textured_bundle.noise_model, textured.noise_model)
        self.assertEqual(flat.gain_e_per_dn, textured.gain_e_per_dn)
        self.assertIsNotNone(textured.gain_e_per_dn)
        self.assertEqual(flat.noise_floor, textured.noise_floor)
        for group in flat.snr_curves:
            np.testing.assert_array_equal(flat.snr_curves[group]["snr_db"],
                                          textured.snr_curves[group]["snr_db"])
            self.assertEqual(textured.snr_curves[group]["kind"], "model")
            self.assertEqual(int(textured.snr_curves[group]["count"].sum()), 0)
        self.assertEqual(compile_tail_snr_gate(flat), compile_tail_snr_gate(textured))

    def test_diagnostics_do_not_change_rendering_evidence(self):
        normal, _, _ = analyze(frame(True), 4, diagnostics=False)
        diagnostic, _, _ = analyze(frame(True), 4, diagnostics=True)
        self.assertEqual(normal.noise_model, diagnostic.noise_model)
        self.assertEqual(normal.noise_floor, diagnostic.noise_floor)
        self.assertEqual(compile_tail_snr_gate(normal), compile_tail_snr_gate(diagnostic))
        self.assertTrue(np.isnan(normal.health_lag1_corr))

    def test_no_calibration_does_not_promote_texture_to_noise(self):
        result, _, _ = analyze(frame(True, known=False), 4)
        self.assertEqual(result.noise_evidence_status, "unavailable")
        self.assertTrue(np.isnan(result.noise_floor))
        self.assertIsNone(result.gain_e_per_dn)
        for curve in result.snr_curves.values():
            self.assertTrue(np.isnan(curve["snr_db"]).all())

    def test_rejected_evidence_differs_from_missing(self):
        missing = SimpleNamespace(noise_model=NoiseModel(), snr_curves={})
        rejected = SimpleNamespace(noise_model=NoiseModel(status="rejected"), snr_curves={})
        self.assertEqual(compile_tail_snr_gate(missing), 1.)
        self.assertEqual(compile_tail_snr_gate(rejected), 0.)
        unresolved = SimpleNamespace(noise_model=NoiseModel(status="unresolved"), snr_curves={})
        self.assertEqual(compile_tail_snr_gate(unresolved), 0.)

    def test_noise_curve_matches_independent_physics(self):
        a, b = 1e-4, 1e-8
        model = NoiseModel(status="valid", channel_variance={"R": (a, b)})
        curves, _, _ = model_snr_curves(model, [0], {0: "R"})
        mu = np.exp2(curves["R"]["stops"])
        expected = 20 * np.log10(mu / np.sqrt(a * mu + b))
        np.testing.assert_allclose(curves["R"]["snr_db"], expected, atol=4e-6)


class FileNoiseProfileTests(unittest.TestCase):
    def resolve(self, values, extra=None):
        bundle = frame(known=False)
        tags = {262: [32803], 51041: values, 50710: [2, 1, 0], **(extra or {})}
        with patch("dngscan.spatial_black.sensor_tags", return_value=tags):
            return resolve_noise_model(bundle, {i: 16383 for i in range(4)})

    def test_cfa_plane_order_is_respected(self):
        model = self.resolve([3e-4, 3e-8, 2e-4, 2e-8, 1e-4, 1e-8])
        self.assertEqual(model.status, "valid")
        self.assertEqual(model.coefficients("R"), (1e-4, 1e-8))
        self.assertEqual(model.coefficients("G1"), (2e-4, 2e-8))
        self.assertEqual(model.coefficients("B"), (3e-4, 3e-8))

    def test_malformed_models_are_rejected(self):
        for values in ([float("nan"), 0], [-1, 0], [1, -1], [1, 0, 1]):
            with self.subTest(values=values):
                self.assertEqual(self.resolve(values).status, "rejected")

    def test_declared_processing_cannot_be_independent_raw_model(self):
        with patch("dngscan.spatial_black.sensor_tags", return_value={
                262: [32803], 51041: [1e-4, 1e-8], 50935: [.5]}):
            model = resolve_noise_model(frame(known=False), {i: 16383 for i in range(4)})
        self.assertEqual(model.status, "rejected")

    def test_processing_declaration_also_restricts_external_calibration(self):
        for profile in (None, [1e-4, 1e-8]):
            tags = {262: [32803], 50935: [.5]}
            if profile is not None:
                tags[51041] = profile
            with patch("dngscan.spatial_black.sensor_tags", return_value=tags):
                model = resolve_noise_model(frame(), {i: 16383 for i in range(4)})
            self.assertEqual(model.status, "rejected")
            self.assertEqual(model.reason, "raw-noise-reduction-declared")

    def test_optional_metadata_errors_do_not_break_analysis(self):
        for extra in ({262: []}, {50935: ["bad"]}, {50935: [float("nan")]},
                      {50935: []}, {50935: [2.]}, {50710: [0, 1, 1]}):
            with self.subTest(extra=extra):
                self.assertEqual(self.resolve([1e-4, 1e-8], extra).status, "rejected")
        with patch("dngscan.spatial_black.sensor_tags", side_effect=struct.error("truncated IFD")):
            model = resolve_noise_model(frame(known=False), {i: 16383 for i in range(4)})
        self.assertEqual(model.status, "unavailable")


class ImportedNoisePipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        env = patch.dict(os.environ, {"DNGSCAN_CALIBRATION_DIR": str(self.directory / "store")})
        env.start()
        self.addCleanup(env.stop)

    def install(self, profile):
        path = self.directory / "profile.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        return import_calibration(path)

    def bundle(self):
        bundle = frame(True)
        bundle.shot_iso = 100
        bundle.shot_shutter = "electronic"
        bundle.capture_readout = {"shutter": "electronic"}
        bundle.noise_decode = {"supported": True, "normalized_raw_to_scene": np.eye(3),
                               "sensor_window_shape": [512, 512]}
        return bundle

    def test_imported_units_reach_snr_endpoint_and_chroma_variance(self):
        from dngscan.noise_propagation import calibrated_chroma_variance
        self.install(single_profile())
        bundle = self.bundle()
        analysis, _, _ = analyze(bundle, 4)
        model = analysis.noise_model
        a, b = 1 / (4 * 15359), (3 / (4 * 15359)) ** 2
        self.assertEqual(model.coefficients("R"), (a, b))
        self.assertEqual(analysis.gain_e_per_dn, 4)
        self.assertEqual(analysis.prior_read_noise_e, 3)
        self.assertIsNone(analysis.prior_pdr_ev)
        floor, source = noise_floor_ev_estimate(analysis)
        self.assertEqual(source, "model")
        self.assertAlmostEqual(floor, MIDGRAY_HEADROOM_STOPS + math.log2(math.sqrt(b)))
        for snr in (1, 10, 20):
            mu = 2 ** (snr_ev_coordinates(analysis)[f"snr{snr}"] - MIDGRAY_HEADROOM_STOPS)
            self.assertAlmostEqual(mu / math.sqrt(a * mu + b), snr, places=10)
        variance, _ = calibrated_chroma_variance(bundle, model, np.full((64, 64, 3), .2))
        self.assertIsNotNone(variance)
        self.assertTrue(np.all(variance > 0))
        self.assertEqual(compile_tail_snr_gate(analysis), 1.)

    def test_failed_read_noise_is_not_a_valid_model_or_missing_calibration(self):
        from dngscan.render import _prepare_chroma_nr_map
        from dngscan.noise_propagation import calibrated_chroma_variance
        self.install(unresolved_collect_profile())
        for iso in (150, 200, 300):
            with self.subTest(iso=iso):
                bundle = self.bundle()
                bundle.shot_iso = iso
                analysis, _, _ = analyze(bundle, 4)
                self.assertEqual(analysis.noise_model.status, "unresolved")
                self.assertEqual(analysis.noise_evidence_status, "model-unresolved")
                self.assertEqual(analysis.noise_model.reason, "read-noise-unresolved" if iso == 200
                                 else "read-noise-unresolved-interval")
                self.assertAlmostEqual(analysis.gain_e_per_dn, 400/iso)
                self.assertIsNone(analysis.prior_read_noise_e)
                self.assertTrue(math.isnan(analysis.noise_floor))
                self.assertIsNone(analysis.noise_model.coefficients("G1"))
                self.assertEqual(compile_tail_snr_gate(analysis), 0.)
                variance, _ = calibrated_chroma_variance(bundle, analysis.noise_model,
                                                         np.full((64, 64, 3), .2))
                self.assertIsNone(variance)
                self.assertIsNone(_prepare_chroma_nr_map(
                    bundle, SimpleNamespace(chroma_nr=1), None, None, None,
                    256, 256, "none", 0., None, analysis=analysis))
                self.assertEqual(bundle.chroma_nr_status, "skipped")

    def test_independent_file_profile_replacement_retains_unresolved_evidence(self):
        from dngscan.calibration import calibration_diagnostics
        self.install(unresolved_collect_profile())
        bundle = self.bundle()
        bundle.shot_iso = 200
        tags = {262: [32803], 51041: [1e-4, 1e-8]}
        with patch("dngscan.spatial_black.sensor_tags", return_value=tags):
            analysis, _, _ = analyze(bundle, 4)
        model = analysis.noise_model
        self.assertEqual(model.status, "valid")
        self.assertEqual(model.source, "DNG NoiseProfile")
        self.assertEqual(model.reason, "file-declared-model-after-unresolved-prior")
        self.assertEqual(model.fallback_reason, "read-noise-unresolved")
        self.assertIn("User JPTC calibration", model.fallback_source)
        self.assertEqual(analysis.gain_e_per_dn, 2)
        self.assertIsNone(analysis.prior_read_noise_e)
        self.assertEqual(calibration_diagnostics("SIGMA", "fp", "electronic", 200)[0]["status"], "gain-only")
        self.assertEqual(compile_tail_snr_gate(analysis), 1.)
        from dngscan.gui.preview_cache import _analysis_from_json, _analysis_to_json
        from dngscan.report import noise_model_line_cn
        restored = _analysis_from_json(json.loads(json.dumps(_analysis_to_json(analysis))))
        self.assertEqual(restored.noise_model, model)
        self.assertIn("原标定=", noise_model_line_cn(restored))
        self.assertIn("read-noise-unresolved", noise_model_line_cn(restored))

    def test_ordinary_missing_noise_is_not_promoted_to_failed_measurement(self):
        profile = collect_profile()
        profile["read_noise_log2iso_log2e"] = []
        self.install(profile)
        analysis, _, _ = analyze(self.bundle(), 4)
        self.assertEqual(analysis.noise_model.status, "unavailable")
        self.assertEqual(analysis.noise_model.reason, "read-noise-unavailable")
        self.assertEqual(compile_tail_snr_gate(analysis), 1.)

    def test_file_fallback_preserves_independent_spectrum_when_read_noise_is_missing(self):
        from dngscan.gui.preview_cache import _analysis_from_json, _analysis_to_json
        from dngscan.noise_propagation import calibrated_chroma_variance
        for unresolved in (False, True):
            for ratio, correlation, gate in ((.1, "measured-spectral-imbalance", 0.),
                                              (0., "measured-spectral-imbalance", 0.),
                                              (3., "measured-spectral-imbalance", 0.),
                                              (1., "measured-spectrum-summary", 1.)):
                with self.subTest(unresolved=unresolved, ratio=ratio):
                    profile = unresolved_collect_profile() if unresolved else collect_profile()
                    if not unresolved:
                        profile["read_noise_log2iso_log2e"] = []
                    profile["noise_whiteness_h_log2iso"] = [[math.log2(200), ratio]]
                    profile["noise_whiteness_v_log2iso"] = [[math.log2(200), 1.25]]
                    self.install(profile)
                    bundle = self.bundle()
                    bundle.shot_iso = 200
                    with patch("dngscan.spatial_black.sensor_tags", return_value={
                            262: [32803], 51041: [1e-4, 1e-8]}):
                        analysis, _, _ = analyze(bundle, 4)
                    model = analysis.noise_model
                    self.assertEqual(model.status, "valid")
                    self.assertEqual(model.source, "DNG NoiseProfile")
                    self.assertEqual(model.coefficients("G1"), (1e-4, 1e-8))
                    self.assertEqual(model.spectral_ratios, {"h": ratio, "v": 1.25})
                    self.assertEqual(model.correlation, correlation)
                    self.assertIn("User JPTC calibration", model.fallback_source)
                    self.assertEqual(model.fallback_reason, "read-noise-unresolved" if unresolved
                                     else "read-noise-unavailable")
                    self.assertEqual(compile_tail_snr_gate(analysis), gate)
                    variance, reason = calibrated_chroma_variance(
                        bundle, model, np.full((64, 64, 3), .2))
                    if gate == 0:
                        self.assertIsNone(variance)
                        self.assertIn("correlated-noise", reason)
                    else:
                        self.assertIsNotNone(variance)
                    restored = _analysis_from_json(json.loads(json.dumps(_analysis_to_json(analysis))))
                    self.assertEqual(restored.noise_model, model)

    def test_missing_variance_without_replacement_keeps_negative_spectral_evidence(self):
        profile = collect_profile()
        profile["read_noise_log2iso_log2e"] = []
        profile["noise_whiteness_h_log2iso"] = [[math.log2(100), .1]]
        self.install(profile)
        for tags in ({262: [32803]}, {262: [2], 51041: [1e-4, 1e-8]}):
            with self.subTest(tags=tags), patch("dngscan.spatial_black.sensor_tags", return_value=tags):
                analysis, _, _ = analyze(self.bundle(), 4)
            self.assertEqual(analysis.noise_model.status, "unavailable")
            self.assertEqual(analysis.noise_model.spectral_ratios, {"h": .1})
            self.assertEqual(compile_tail_snr_gate(analysis), 0.)

    def test_incompatible_file_dn_scale_cannot_donate_spectrum(self):
        for read_state in ("measured", "unresolved", "missing"):
            with self.subTest(read_state=read_state):
                profile = unresolved_collect_profile() if read_state == "unresolved" else collect_profile()
                if read_state == "missing":
                    profile["read_noise_log2iso_log2e"] = []
                profile["noise_whiteness_h_log2iso"] = [[math.log2(200), .1]]
                self.install(profile)
                bundle = self.bundle()
                bundle.shot_iso = 200
                bundle.white_level = 12000
                bundle.camera_white_levels = [12000.] * 4
                with patch("dngscan.spatial_black.sensor_tags", return_value={
                        262: [32803], 51041: [1e-4, 1e-8]}):
                    analysis, _, _ = analyze(bundle, 4)
                self.assertEqual(analysis.noise_model.source, "DNG NoiseProfile")
                self.assertEqual(analysis.noise_model.status, "valid")
                self.assertEqual(analysis.noise_model.spectral_ratios, {})
                self.assertEqual(analysis.noise_model.correlation, "unknown")

    def test_inapplicable_user_measurement_cannot_donate_spectrum(self):
        profile = unresolved_collect_profile()
        profile["noise_whiteness_h_log2iso"] = [[math.log2(200), .1]]
        self.install(profile)
        for changes in ({"shot_make": "unmeasured"}, {"shot_model": "unmeasured"},
                        {"shot_shutter": "mechanical"}, {"shot_shutter": None},
                        {"shot_iso": 900}, {"shot_iso": 600}):
            with self.subTest(changes=changes):
                bundle = self.bundle()
                bundle.shot_iso = 200
                for key, value in changes.items():
                    setattr(bundle, key, value)
                if "shot_shutter" in changes:
                    bundle.capture_readout = {"shutter": changes["shot_shutter"]}
                with patch("dngscan.spatial_black.sensor_tags", return_value={
                        262: [32803], 51041: [1e-4, 1e-8]}):
                    analysis, _, _ = analyze(bundle, 4)
                self.assertEqual(analysis.noise_model.spectral_ratios, {})
                self.assertEqual(analysis.noise_model.correlation, "unknown")

    def test_nonfinite_file_scale_does_not_promote_spectrum_or_crash_fallback(self):
        profile = unresolved_collect_profile()
        profile["noise_whiteness_h_log2iso"] = [[math.log2(200), .1]]
        self.install(profile)
        for black in (float("nan"), float("inf"), 16383., 17000.):
            with self.subTest(black=black):
                bundle = self.bundle()
                bundle.shot_iso = 200
                bundle.black_levels = [black] * 4
                with patch("dngscan.spatial_black.sensor_tags", return_value={
                        262: [32803], 51041: [1e-4, 1e-8]}):
                    model = resolve_noise_model(bundle, {i: 16383 for i in range(4)})
                self.assertEqual(model.status, "valid")
                self.assertEqual(model.source, "DNG NoiseProfile")
                self.assertEqual(model.spectral_ratios, {})

    def test_raw_processing_declaration_still_overrides_spectral_fallback(self):
        profile = unresolved_collect_profile()
        profile["noise_whiteness_h_log2iso"] = [[math.log2(200), .1]]
        self.install(profile)
        for reduction in (.5, "invalid"):
            with self.subTest(reduction=reduction):
                bundle = self.bundle()
                bundle.shot_iso = 200
                with patch("dngscan.spatial_black.sensor_tags", return_value={
                        262: [32803], 51041: [1e-4, 1e-8], 50935: [reduction]}):
                    analysis, _, _ = analyze(bundle, 4)
                self.assertEqual(analysis.noise_model.status, "rejected")
                self.assertEqual(analysis.noise_model.source, "DNG NoiseReductionApplied")
                self.assertEqual(compile_tail_snr_gate(analysis), 0.)

    def test_measured_nonwhite_noise_restricts_processing_without_revoking_gain(self):
        from dngscan.noise_propagation import calibrated_chroma_variance
        profile = collect_profile()
        profile["noise_whiteness_h_log2iso"] = [[math.log2(100), .1]]
        self.install(profile)
        bundle = self.bundle()
        analysis, _, _ = analyze(bundle, 4)
        self.assertEqual(analysis.noise_model.status, "valid")
        self.assertEqual(analysis.noise_model.spectral_ratios, {"h": .1})
        self.assertEqual(analysis.noise_correlation_status, "measured-spectral-imbalance")
        self.assertEqual(analysis.gain_e_per_dn, 4)
        self.assertEqual(compile_tail_snr_gate(analysis), 0.)
        variance, reason = calibrated_chroma_variance(bundle, analysis.noise_model, np.full((64, 64, 3), .2))
        self.assertIsNone(variance)
        self.assertIn("correlated-noise", reason)

    def test_packaged_measurements_retain_both_spectrum_directions(self):
        from dngscan import priors
        path = Path(priors.__file__).parent / "data/priors/jptc_collect/a7rm6-mech-20260818.json"
        source = json.loads(path.read_text(encoding="utf-8"))
        entry = next(p for p in priors._jptc_entries() if p["id"] == source["id"])
        self.assertEqual(entry["noise_whiteness_h_log2iso"], source["noise_whiteness_h_log2iso"])
        self.assertEqual(entry["noise_whiteness_v_log2iso"], source["noise_whiteness_v_log2iso"])

    def test_legacy_rounded_spectrum_iso_is_consumed_without_extrapolation(self):
        profile = collect_profile()
        profile["noise_whiteness_v_log2iso"] = [[round(math.log2(100), 4), .1]]
        self.install(profile)
        bundle = self.bundle()
        analysis, _, _ = analyze(bundle, 4)
        self.assertEqual(analysis.noise_model.spectral_ratios, {"v": .1})
        self.assertEqual(compile_tail_snr_gate(analysis), 0.)
        bundle.shot_iso = 200
        analysis, _, _ = analyze(bundle, 4)
        self.assertEqual(analysis.noise_model.spectral_ratios, {})

    def test_zero_high_band_power_is_valid_negative_evidence(self):
        profile = collect_profile()
        profile["noise_whiteness_h_log2iso"] = [[math.log2(100), 0.]]
        self.install(profile)
        analysis, _, _ = analyze(self.bundle(), 4)
        self.assertEqual(analysis.noise_model.spectral_ratios, {"h": 0.})
        self.assertEqual(compile_tail_snr_gate(analysis), 0.)


if __name__ == "__main__":
    unittest.main()
