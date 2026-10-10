# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import lens_transmittance as LT
from dngscan.spectral_color_research import (
    Spectrum, CameraSpectralResponse, camera_responses,
    neutral_white_balance_suggestion, fit_white_constrained_matrix,
    propagate_color_covariance, color_error_report, experiment_identity, match_camera_response,
)


def example():
    wl = np.arange(380., 781., 5.)
    rgb = np.column_stack([np.exp(-((wl-610)/55)**2), np.exp(-((wl-540)/45)**2),
                           np.exp(-((wl-450)/35)**2)])
    camera = CameraSpectralResponse("test-camera", wl, rgb, "energy", "common",
                                    "reference", False, "synthetic-test")
    light = Spectrum("light", wl, 1 + .2 * np.sin(wl / 100), "energy_spd")
    reference = Spectrum("reference", wl, np.ones(len(wl)), "relative_spectral_transmittance")
    target = Spectrum("target", wl, .3 + (wl-wl[0]) / 800, "relative_spectral_transmittance")
    material = Spectrum("material", wl, .3 + .2 * np.cos(wl / 50), "reflectance")
    return wl, camera, light, reference, target, material


class LensMetadataContractTests(unittest.TestCase):
    def test_migrated_measurement_semantics_and_combined_filter(self):
        entry = LT.get_metadata("0612-contax-g35")
        self.assertEqual(entry["quantity"], "relative_spectral_transmittance")
        self.assertAlmostEqual(entry["normalization"]["relative_max_before_scaling"], 1.12528465)
        self.assertFalse(entry["absolute_throughput_available"])
        uv = LT.get_metadata("0703-3-apo-lanthar-50")
        self.assertFalse(uv["is_bare_lens"])
        self.assertIn("Hoya", uv["optical_stack"]["filters"][0])
        self.assertEqual(LT.match_exact("APO Lanthar 50")["status"], "incompatible")
        match = LT.match_exact("APO Lanthar 50", is_bare_lens=False, optical_stack=uv["optical_stack"])
        self.assertEqual(match["status"], "matched")
        self.assertFalse(match["color_correction_available"])

    def test_unknown_bare_lens_and_camera_equipment_are_not_guessed(self):
        self.assertIsNone(LT.get_metadata("0612-contax-g35")["is_bare_lens"])
        self.assertEqual(LT.match_exact("Contax G35")["status"], "incompatible")
        self.assertEqual(LT.match_exact("Sigma 35mm F2 DG DN")["status"], "unmeasured")
        self.assertEqual(LT.match_exact("Contax G")["status"], "unmeasured")

    def test_duplicate_names_keep_measurements_and_synthetic_not_matched(self):
        values = {"one": {"id": "one", "name": "same", "category": "Lens", "is_bare_lens": True},
                  "two": {"id": "two", "name": "same", "category": "Lens", "is_bare_lens": True},
                  "ref": {"id": "ref", "name": "reference", "category": "Reference",
                          "measurement_kind": "synthetic-reference"}}
        with patch.object(LT, "_CACHE", values):
            self.assertEqual(LT.match_exact("same")["status"], "ambiguous")
            selected = LT.match_exact("same", measurement_id="two")
            self.assertEqual(selected["measurement"]["measurement_id"], "two")
            self.assertEqual(LT.match_exact("reference", category="Reference")["status"], "incompatible")
            out = LT.get_metadata("one")
            out["name"] = "changed"
            self.assertEqual(LT.get_metadata("one")["name"], "same")

    def test_importer_preserves_normalization_and_requires_optical_index_identity(self):
        spec = importlib.util.spec_from_file_location("import_lens", Path(__file__).parents[1] / "tools/import_lens_transmittance.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "renamed.json"
            raw = {"id": "x", "name": "lens", "category": "Lens", "sampleCount": 3,
                   "wavelengthStart": 400, "wavelengthEnd": 402, "stepNm": 1,
                   "relativeMaxBeforeScaling": 1.2, "relativeMinBeforeScaling": .6,
                   "data": [50, 80, 100]}
            path.write_text(json.dumps(raw))
            entry = module.convert(path, {"id": "x", "name": "lens", "category": "Lens", "isBareLens": False,
                                         "measurementNote": "UV"}, source_commit="fixed")
            self.assertEqual(entry["normalization"]["relative_max_before_scaling"], 1.2)
            self.assertEqual(entry["optical_stack"]["filters"], ["UV"])
            self.assertEqual(entry["source_commit"], "fixed")
            with self.assertRaises(ValueError):
                module.convert(path, {"id": "x", "name": "different", "category": "Lens"})


class SpectralColorContractTests(unittest.TestCase):
    def test_camera_response_requires_exact_body_identity(self):
        _, camera, _, _, _, _ = example()
        self.assertEqual(match_camera_response(camera.camera_id, [camera])["status"], "matched-research")
        self.assertEqual(match_camera_response("SIGMA fp", [camera])["status"], "unmeasured")
        self.assertEqual(match_camera_response(camera.camera_id, [camera, camera])["status"], "ambiguous")
        unknown = CameraSpectralResponse(**{**camera.__dict__, "channel_normalization": "unknown"})
        self.assertEqual(match_camera_response(camera.camera_id, [unknown])["status"], "unqualified")

    def test_relative_curve_scale_has_no_normalized_color_or_exposure_effect(self):
        wl, camera, light, reference, target, material = example()
        expected, evidence = camera_responses(camera, light, [material], target)
        scaled = Spectrum("scaled", wl, np.asarray(target.values) * 37, target.quantity)
        actual, _ = camera_responses(camera, light, [material], scaled)
        np.testing.assert_allclose(actual, expected, rtol=2e-15)
        self.assertFalse(evidence["absolute_throughput_available"])
        self.assertFalse(evidence["pixel_correction_applied"])

    def test_identity_optics_and_full_camera_reference_not_double_counted(self):
        wl, camera, light, reference, target, material = example()
        actual, _ = camera_responses(camera, light, [material], reference)
        full = CameraSpectralResponse(camera.camera_id, wl, np.asarray(camera.values_rgb) * np.asarray(target.values)[:, None],
                                      "energy", "common", "target", True, camera.source_id)
        replaced, _ = camera_responses(full, light, [material], reference, reference_optics=target)
        # Full SSF includes the arbitrary relative scale of its measured optics.
        # Its common exposure scale cancels in color ratios.
        np.testing.assert_allclose(replaced / replaced[:, 1:2], actual / actual[:, 1:2], rtol=2e-15)
        identity = neutral_white_balance_suggestion(full, light, target, target, existing_wb_state="reference-optics")
        np.testing.assert_allclose(identity["relative_rgb_gains"], 1, atol=2e-15)

    def test_existing_target_wb_prevents_second_optical_neutral_correction(self):
        _, camera, light, reference, target, _ = example()
        for state in ("as-shot-target", "gray-card-target", "profile-includes-target"):
            result = neutral_white_balance_suggestion(camera, light, target, reference, existing_wb_state=state)
            self.assertEqual(result["status"], "already-compensated")
            self.assertEqual(result["relative_rgb_gains"], [1, 1, 1])
        unknown = neutral_white_balance_suggestion(camera, light, target, reference, existing_wb_state="unknown")
        self.assertIsNone(unknown["relative_rgb_gains"])
        suggested = neutral_white_balance_suggestion(camera, light, target, reference, existing_wb_state="reference-optics")
        self.assertEqual(suggested["status"], "model-suggestion")
        self.assertFalse(suggested["automatic_application"])

    def test_unknown_prescale_reference_or_response_convention_is_rejected(self):
        wl, camera, light, reference, target, material = example()
        for kwargs in ({"channel_normalization": "unknown"}, {"reference_optical_stack": None},
                       {"response_convention": "unknown"}, {"includes_reference_optics": None},
                       {"channel_normalization": "known-prescale", "channel_prescale": (1, 0, 2)}):
            values = dict(camera.__dict__); values.update(kwargs)
            with self.assertRaises(ValueError):
                camera_responses(CameraSpectralResponse(**values), light, [material], target)
        # Explicit factors undo already-applied channel prescaling.
        factors = np.array([2., 3., .5])
        known = CameraSpectralResponse(camera.camera_id, wl, np.asarray(camera.values_rgb)*factors,
                                       "energy", "known-prescale", "reference", False, "known", tuple(factors))
        a, _ = camera_responses(known, light, [material], target)
        b, _ = camera_responses(camera, light, [material], target)
        np.testing.assert_allclose(a, b, rtol=2e-15)

    def test_incomplete_wavelength_coverage_requires_explicit_restricted_experiment(self):
        wl, camera, light, reference, target, material = example()
        partial = Spectrum("partial", wl[:-5], np.asarray(target.values)[:-5], target.quantity)
        with self.assertRaisesRegex(ValueError, "coverage"):
            camera_responses(camera, light, [material], partial)
        _, evidence = camera_responses(camera, light, [material], partial, wavelength_range_nm=(380, 755))
        self.assertTrue(evidence["range_was_explicitly_restricted"])
        self.assertEqual(evidence["wavelength_range_nm"], [380, 755])

    def test_energy_photon_convention_conversion_is_explicit(self):
        wl, camera, light, reference, target, material = example()
        photon_camera = CameraSpectralResponse(camera.camera_id, wl, camera.values_rgb, "photon", "common", "reference", False, "photon")
        photon_light = Spectrum("photon-light", wl, np.asarray(light.values) * wl / 555, "photon_spd")
        energy_result, _ = camera_responses(photon_camera, light, [material], target)
        photon_result, _ = camera_responses(photon_camera, photon_light, [material], target)
        np.testing.assert_allclose(energy_result / energy_result[:, 1:2], photon_result / photon_result[:, 1:2], rtol=2e-15)

    def test_white_constrained_matrix_and_independent_heldout_report(self):
        rng = np.random.default_rng(81)
        x = rng.uniform(.1, 1, (30, 3)); m = np.array([[1.1, -.1, 0], [.02, .95, .03], [0, -.05, 1.05]])
        fitted = fit_white_constrained_matrix(x, x @ m.T, source_white=np.ones(3), target_white=m @ np.ones(3))
        np.testing.assert_allclose(fitted, m, atol=2e-15)
        held = rng.uniform(.1, 1, (16, 3))
        report = color_error_report(held @ fitted.T, held @ m.T, white_xyz=np.ones(3),
                                    sample_ids=[f"held-{n}" for n in range(16)], illuminant_id="not-training-light")
        self.assertLess(report["max_delta_e76"], 1e-12)
        self.assertEqual(len(report["hue_error_degrees"]), 16)
        with self.assertRaises(ValueError):
            fit_white_constrained_matrix(np.ones((8, 3)), np.ones((8, 3)), source_white=np.ones(3), target_white=np.ones(3))

    def test_matrix_covariance_matches_monte_carlo(self):
        rng = np.random.default_rng(12)
        matrix = np.array([[1.1, -.15, .02], [.02, .97, .02], [0, -.04, 1.07]])
        covariance = np.array([[.04, .012, -.003], [.012, .03, .005], [-.003, .005, .02]])
        expected = propagate_color_covariance(matrix, covariance)
        samples = rng.multivariate_normal(np.zeros(3), covariance, size=250000) @ matrix.T
        actual = np.cov(samples, rowvar=False)
        np.testing.assert_allclose(actual, expected, rtol=.025, atol=.00025)
        for invalid in (np.diag([1, -1, 1]), np.ones((2, 2))):
            with self.assertRaises(ValueError):
                propagate_color_covariance(matrix, invalid)

    def test_identity_changes_with_measurement_or_algorithm_contract(self):
        self.assertEqual(experiment_identity({"x": 1, "y": 2}), experiment_identity({"y": 2, "x": 1}))
        self.assertNotEqual(experiment_identity({"spectrum": [1, 2]}), experiment_identity({"spectrum": [1, 2.1]}))


class SpectralOfflineExperimentTests(unittest.TestCase):
    @staticmethod
    def tool():
        spec = importlib.util.spec_from_file_location("validate_spectral", Path(__file__).parents[1] / "tools/validate_spectral_color.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_independent_lights_expose_nonuniversal_matrix(self):
        tool = self.tool()
        dataset = tool.demo_dataset()
        result = tool.run_experiment(dataset)
        self.assertFalse(result["production_eligible"])
        self.assertFalse(result["pixel_correction_applied"])
        self.assertLess(result["white_constraint_error"], 1e-12)
        self.assertLess(result["covariance_monte_carlo"]["max_error_relative_to_max_variance"], .02)
        self.assertGreater(len(result["validation"]), 1)
        # The LED holdout is deliberately preserved even though the matrix is worse.
        led = result["validation"][0]
        self.assertGreater(led["candidate"]["max_delta_e76"], led["baseline"]["max_delta_e76"])
        self.assertTrue(set(result["training_reflectance_ids"]).isdisjoint(led["candidate"]["sample_ids"]))

    def test_renamed_training_samples_cannot_become_holdout(self):
        tool = self.tool()
        dataset = tool.demo_dataset()
        held = dict(dataset["training_reflectances"][0])
        held["measurement_id"] = "renamed-training"
        dataset["validation_reflectances"][0] = held
        with self.assertRaisesRegex(ValueError, "reuse"):
            tool.run_experiment(dataset)

    def test_zero_reflectance_has_zero_response(self):
        wl, camera, light, reference, target, _ = example()
        black = Spectrum("black", wl, np.zeros(len(wl)), "reflectance")
        actual, _ = camera_responses(camera, light, [black], target)
        np.testing.assert_array_equal(actual, np.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main()
