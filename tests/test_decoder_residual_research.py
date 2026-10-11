# SPDX-License-Identifier: GPL-3.0-or-later
import copy
import unittest

import numpy as np

from dngscan import decoder_residual_research as research


def contract(kind="dark"):
    return {"schema": research.SCHEMA, "domain": research.DOMAIN,
        "frame_independence": "separate-captures", "acquisition_kind": kind,
        "decoder_recipe": {"configuration_id": "fixed-test-configuration", "actual_decoder_token": "9.test",
            "runtime": {"os": "synthetic"}, "geometry": {"output_shape": [64, 64]},
            "output": {"space": "linear-rec2020"}, "controls": {}},
        "capture_conditions": {"make": "synthetic", "model": "test", "body_serial": "synthetic-body", "iso": 100, "readout_id": "test",
            "exposure_seconds": .01, "temperature_c": 20},
        "reference_provenance": {"source_id": "synthetic-known-zero", "domain": research.DOMAIN,
            "independent_of_captures": True, "uncertainty_status": "known-synthetic-truth"}}


def records(count, offset=0):
    return [{"source_sha256": f"{i+offset:064x}", "capture_id": f"independent-{i+offset}",
        "decoder_configuration_id": "fixed-test-configuration",
        "decoder_recipe": copy.deepcopy(contract()["decoder_recipe"]),
        "capture_conditions": contract()["capture_conditions"]} for i in range(count)]


class DecoderResidualResearchTests(unittest.TestCase):
    def test_independent_diff_covariance_factor_two_and_zero_y_basis(self):
        rng = np.random.default_rng(924)
        expected = np.array([[4., 1., .25], [1., 2., .1], [.25, .1, 1.]])*.0001
        frames = rng.multivariate_normal(np.zeros(3), expected, size=(16, 64, 64))
        stats = research.repeat_residual_statistics(frames, records(16), contract=contract())
        row = stats["regions"]["full"]
        actual = np.array(row["within_region_temporal_covariance_rgb_3x3"])
        np.testing.assert_allclose(actual, expected, rtol=.035, atol=8e-6)
        basis = np.array(stats["zero_y_basis_rgb_columns"])
        np.testing.assert_allclose(research.LUMA @ basis, 0, atol=1e-15)
        np.testing.assert_allclose(row["zero_y_temporal_covariance_2x2"], basis.T @ actual @ basis)
        summed = np.sum([b["zero_y_covariance_power_2x2"] for b in row["bands"]], axis=0)
        np.testing.assert_allclose(summed, np.array(row["zero_y_temporal_covariance_2x2"])*(4095/4096), atol=1e-14)
        self.assertFalse(stats["production_applicable"])
        self.assertEqual(row["fixed_pattern_status"], "unavailable-scene-structure-is-not-FPN")

    def test_common_offset_is_reported_separately_from_covariance(self):
        frames = np.zeros((4, 64, 64, 3))
        frames[::2] = [.01, .02, .03]
        stats = research.repeat_residual_statistics(frames, records(4), contract=contract(), reference_mean=np.zeros(3))
        row = stats["regions"]["full"]
        np.testing.assert_allclose(row["within_region_temporal_covariance_rgb_3x3"], 0, atol=1e-25)
        np.testing.assert_allclose(row["pair_difference_mean_rgb"][0], [.01, .02, .03])
        self.assertGreater(np.array(row["pair_difference_raw_second_moment_rgb_3x3"]).max(), 0)
        np.testing.assert_allclose(row["mean_bias_rgb"], [.005, .01, .015])

    def test_same_raw_rerender_or_capture_id_cannot_be_independent(self):
        frames = np.zeros((4, 64, 64, 3))
        for field in ("source_sha256", "capture_id"):
            rows = records(4); rows[1][field] = rows[0][field]
            with self.assertRaisesRegex(ValueError, "same RAW"):
                research.repeat_residual_statistics(frames, rows, contract=contract())

    def test_mismatched_configuration_and_conditions_rejected(self):
        frames = np.zeros((4, 64, 64, 3))
        rows = records(4); rows[0]["decoder_configuration_id"] = "raw8"
        with self.assertRaisesRegex(ValueError, "configuration"):
            research.repeat_residual_statistics(frames, rows, contract=contract())
        rows = records(4); rows[0]["capture_conditions"] = {"iso": 200}
        with self.assertRaisesRegex(ValueError, "conditions"):
            research.repeat_residual_statistics(frames, rows, contract=contract())

    def test_ordinary_photograph_and_wrong_units_rejected(self):
        frames = np.zeros((4, 64, 64, 3))
        for field, value in (("acquisition_kind", "photograph"), ("domain", "sensor-dn"),
                             ("frame_independence", "rerenders")):
            cfg = contract(); cfg[field] = value
            with self.assertRaises(ValueError):
                research.repeat_residual_statistics(frames, records(4), contract=cfg)

    def test_stable_config_id_does_not_hide_effective_output_changes(self):
        frames = np.zeros((4,64,64,3))
        for field,value in (("controls", {"neutralTemperature": {"effective": {"value": 7000}}}),
                            ("project_handoff", {"alignment_factor": 1.1}),
                            ("geometry", {"output_shape": [32,32]})):
            rows = records(4); rows[0]["decoder_recipe"][field] = value
            with self.assertRaisesRegex(ValueError, "per-frame output"):
                research.repeat_residual_statistics(frames, rows, contract=contract())

    def test_integer_image_cannot_masquerade_as_signed_decoder_output(self):
        with self.assertRaisesRegex(ValueError, "signed floating"):
            research.repeat_residual_statistics(np.zeros((4,64,64,3),dtype=np.uint16),records(4),contract=contract())

    def test_cropped_statistics_do_not_masquerade_as_full_decoder_grid(self):
        with self.assertRaisesRegex(ValueError, "output_shape"):
            research.repeat_residual_statistics(np.zeros((4,32,32,3)),records(4),contract=contract())
        report = research.repeat_residual_statistics(np.zeros((4,64,64,3)),records(4),contract=contract(),
            regions={"explicit-crop": [0,0,32,32]})
        self.assertEqual(report["regions"]["explicit-crop"]["shape"],[32,32,3])

    def test_cropped_bias_cannot_be_bound_to_a_larger_decoder_recipe(self):
        frames = np.zeros((4,32,32,3))
        with self.assertRaisesRegex(ValueError, "output_shape"):
            research.build_output_bias_candidate(frames, records(4), frames, records(4,100),
                contract=contract(), reference_mean=np.zeros(3))

    def test_missing_reference_evidence_is_not_accepted(self):
        frames = np.zeros((4, 64, 64, 3))
        cfg = contract(); cfg.pop("reference_provenance")
        with self.assertRaisesRegex(ValueError, "reference provenance"):
            research.repeat_residual_statistics(frames, records(4), contract=cfg, reference_mean=np.zeros(3))

    def test_regions_are_exact_output_coordinates_and_finite(self):
        frames = np.zeros((4, 64, 64, 3))
        report = research.repeat_residual_statistics(frames, records(4), contract=contract(),
            regions={"center": [16, 16, 32, 32]})
        self.assertEqual(report["regions"]["center"]["shape"], [16, 16, 3])
        for bounds in ([0, 0, 65, 64], [0, 0, 4, 4]):
            with self.assertRaises(ValueError):
                research.repeat_residual_statistics(frames, records(4), contract=contract(), regions={"bad": bounds})
        frames[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError): research.repeat_residual_statistics(frames, records(4), contract=contract())

    def test_output_bias_is_holdout_experiment_not_sensor_correction(self):
        rng = np.random.default_rng(1)
        bias = np.sin(np.arange(64)[None, :, None]/8)*np.array([.02, .01, -.01])[None, None, :]
        train = bias+rng.normal(0, .0001, (8,64,64,3))
        test = bias+rng.normal(0, .0001, (8,64,64,3))
        product = research.build_output_bias_candidate(train, records(8), test, records(8,100),
            contract=contract(), reference_mean=np.zeros(3))
        manifest = product["manifest"]
        before = np.mean(manifest["heldout_before"]["regions"]["full"]["ensemble_mean_residual_mse_rgb"])
        after = np.mean(manifest["heldout_after"]["regions"]["full"]["ensemble_mean_residual_mse_rgb"])
        self.assertLess(after, before/1000)
        self.assertFalse(manifest["predecoder_raw_correction_equivalent"])
        self.assertFalse(manifest["automatic_application"])
        self.assertEqual(product["output_bias_rgb"].dtype, np.float32)
        np.testing.assert_allclose(manifest["heldout_before"]["regions"]["full"]["within_region_temporal_covariance_rgb_3x3"],
            manifest["heldout_after"]["regions"]["full"]["within_region_temporal_covariance_rgb_3x3"], atol=1e-18)

    def test_bias_training_holdout_reuse_and_flat_rejected(self):
        frames = np.zeros((4,64,64,3))
        with self.assertRaisesRegex(ValueError, "training and heldout"):
            research.build_output_bias_candidate(frames, records(4), frames, records(4),
                contract=contract(), reference_mean=np.zeros(3))
        with self.assertRaisesRegex(ValueError, "dark captures"):
            research.build_output_bias_candidate(frames, records(4), frames, records(4,100),
                contract=contract("flat"), reference_mean=np.zeros(3))


if __name__ == "__main__": unittest.main()
