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
from tests.test_user_calibration import single_profile, collect_profile


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
