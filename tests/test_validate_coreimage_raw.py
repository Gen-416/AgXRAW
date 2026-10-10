# SPDX-License-Identifier: GPL-3.0-or-later
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

path = Path(__file__).resolve().parents[1]/"tools/validate_coreimage_raw.py"
spec = importlib.util.spec_from_file_location("validate_coreimage_raw_test_module", path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class CoreImageMatrixToolTests(unittest.TestCase):
    def test_decoder_anchor_asset_is_part_of_processing_identity(self):
        with patch.object(tool, "implementation_identity", return_value={"code": "unchanged"}), \
             patch.object(tool, "digest", return_value="unchanged-tool"), \
             patch.object(tool, "decoder_anchor_fingerprint", return_value="a"*64):
            first = tool._implementation_digest()
        with patch.object(tool, "implementation_identity", return_value={"code": "unchanged"}), \
             patch.object(tool, "digest", return_value="unchanged-tool"), \
             patch.object(tool, "decoder_anchor_fingerprint", return_value="b"*64):
            second = tool._implementation_digest()
        self.assertNotEqual(first, second)

    def test_decoder_window_matching_record_is_retained_separately(self):
        window = {"status": "matched", "profile_id": "specific-window", "asset_sha256": "a"*64,
                  "reason": "exact-decoder-runtime-and-configuration", "constraints": {"iso": [100, 400]}}
        with patch.object(tool, "decoder_anchor_fingerprint", return_value="a"*64):
            row = tool._window_provenance(SimpleNamespace(scene_window_calibration=window))
            missing = tool._window_provenance(SimpleNamespace())
        self.assertEqual(row["scene_window_calibration"], window)
        self.assertEqual(row["decoder_anchor_asset_sha256"], "a"*64)
        self.assertEqual(missing["scene_window_calibration"], {})
        window["constraints"]["iso"].append(800)
        self.assertEqual(row["scene_window_calibration"]["constraints"]["iso"], [100, 400])

    def test_matrix_axes_keep_native_preview_resize_and_scale_distinct(self):
        cases = tool.build_cases(tool.MATRIX, tool.SIZES, ("native", "aligned"))
        self.assertEqual(len(cases), 21)
        self.assertEqual(len({c["id"] for c in cases}), 21)
        self.assertEqual(sum(c["decoder"] == "libraw" for c in cases), 3)
        self.assertTrue(any(c["id"] == "raw9/native-preview/native" for c in cases))
        self.assertTrue(any(c["id"] == "raw9/resized-full/native" for c in cases))

    def test_fixed_version_cannot_pass_fallback(self):
        case = {"decoder": "raw9"}
        recipe = {"configuration_id": "fixed-recipe", "actual_decoder_token": "9.raw",
                  "linear_access": {"status":"parameter-state-verified"}}
        for decoder,token,fallback in (("coreimage", "8.raw", None), ("libraw", None, "Apple failed"),
                                       ("coreimage", "9.raw", "earlier version failed")):
            self.assertEqual(tool.version_outcome(case, actual_decoder=decoder,
                actual_token=token, fallback=fallback, recipe=recipe)[0], "failed")
        self.assertEqual(tool.version_outcome(case, actual_decoder="coreimage", actual_token="9.raw",
            fallback=None, recipe=recipe)[0], "passed")

    def test_auto_records_actual_result_without_fixed_claim(self):
        status,reason = tool.version_outcome({"decoder": "auto"}, actual_decoder="libraw",
            actual_token=None, fallback="Apple failed", recipe={"configuration_id": "libraw"})
        self.assertEqual((status,reason), ("passed", "actual-auto-libraw-fallback-not-apple-version-pass"))
        self.assertEqual(tool.version_outcome({"decoder":"libraw"}, actual_decoder="libraw",
            actual_token=None, fallback=None, recipe={}), ("passed", "actual-libraw-result"))

    def test_missing_recipe_does_not_claim_success(self):
        self.assertEqual(tool.version_outcome({"decoder": "raw9"}, actual_decoder="coreimage",
            actual_token="9.raw", fallback=None, recipe={})[1], "decoder-recipe-missing")

    def test_requested_token_cannot_substitute_for_missing_actual_getter(self):
        recipe={"configuration_id":"id","actual_decoder_token":None,
                "linear_access":{"status":"parameter-state-verified"}}
        self.assertEqual(tool.version_outcome({"decoder":"raw9"},actual_decoder="coreimage",
            actual_token="9",fallback=None,recipe=recipe)[1],"actual-decoder-token-unverified")
        recipe["actual_decoder_token"]="9"
        recipe["linear_access"]["status"]="parameter-state-unverified"
        self.assertEqual(tool.version_outcome({"decoder":"raw9"},actual_decoder="coreimage",
            actual_token="9",fallback=None,recipe=recipe)[1],"linear-parameter-state-unverified")

    def test_native_preview_target_is_independent_of_resized_full_parameter(self):
        cases=tool.build_cases(["raw9"],["native-preview","resized-full"],["native"],1600)
        self.assertEqual(cases[0]["native_preview_target_long_edge"],1280)
        self.assertIsNone(cases[0]["resized_full_max_edge"])
        self.assertEqual(cases[1]["resized_full_max_edge"],1600)
        self.assertIsNone(cases[1]["native_preview_target_long_edge"])

    def test_area_preview_preserves_signed_values_and_partial_edges(self):
        image = np.full((17,19,3),[-2, .5, 4], dtype=np.float32)
        resized,recipe = tool.resize_full(image, 8)
        self.assertEqual(resized.shape, (6,7,3))
        np.testing.assert_array_equal(resized, np.broadcast_to([-2,.5,4],resized.shape))
        self.assertEqual(recipe["integer_factor"], 3)
        self.assertFalse(recipe["clipping"])

    def test_output_counts_nonfinite_separately(self):
        image = np.array([[[-1., 2., np.nan]]],dtype=np.float32)
        summary = tool.output_summary(image)
        self.assertEqual(summary["finite_fraction"], 2/3)
        self.assertEqual(summary["negative_fraction"], 1/3)
        self.assertEqual(summary["above_one_fraction"], 1/3)
        self.assertIsNone(summary["min_rgb"])

    def test_single_image_colour_structure_is_never_labeled_noise(self):
        xx=np.arange(64)[None,:,None]
        image=np.broadcast_to(.2+np.sin(xx/8)*np.array([.01,0.,-.01]),(64,64,3)).copy()
        stats=tool.scene_structure_summary(image)
        self.assertFalse(stats["noise_estimate"])
        self.assertEqual(stats["status"],"descriptive-scene-plus-noise")
        self.assertGreater(np.trace(stats["zero_y_spatial_covariance_2x2"]),0.)
        summed=np.sum([b["zero_y_structure_covariance_power_2x2"] for b in stats["bands"]],axis=0)
        np.testing.assert_allclose(summed,stats["zero_y_spatial_covariance_2x2"],atol=1e-15)

    def test_cross_case_distribution_shift_does_not_claim_pixel_damage(self):
        def row(size,value):
            return {"case":{"id":"raw9/"+size+"/native","decoder":"raw9","size":size,"scale":"native"},
                "status":"passed","file":"synthetic.NEF","sha256":"a"*64,
                "scene_structure":tool.scene_structure_summary(np.full((32,32,3),value))}
        before,after=row("native-preview",.2),row("resized-full",.4)
        result=tool.case_comparisons([before,after])
        self.assertEqual(len(result),1)
        self.assertAlmostEqual(result[0]["Y_median_EV_difference"],1.)
        self.assertFalse(result[0]["pixel_comparison_qualified"])
        self.assertEqual(result[0]["purpose"],"native-preview-vs-full-downsample")

    def test_unsupported_advertised_version_is_unavailable_not_fallback(self):
        with patch.object(tool.sys,"platform","darwin"), patch.object(tool.coreimage_decode,"available",return_value=True), \
             patch.object(tool.coreimage_decode,"supported_versions",return_value=("8.raw",)), \
             patch.object(tool.coreimage_decode,"runtime_available") as runtime:
            result = tool._preflight(Path("synthetic.NEF"), {"decoder":"raw9","size":"full"})
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "fixed-version-not-offered-for-file")
        runtime.assert_not_called()

    def test_context_failure_is_not_claimed_file_unsupported(self):
        with patch.object(tool.sys,"platform","darwin"), patch.object(tool.coreimage_decode,"available",return_value=True), \
             patch.object(tool.coreimage_decode,"supported_versions",return_value=("9.raw",)), \
             patch.object(tool.coreimage_decode,"runtime_available",return_value=False):
            result = tool._preflight(Path("synthetic.NEF"), {"decoder":"raw9","size":"full"})
        self.assertEqual(result["qualification"], "execution-context-unavailable-not-file-unsupported")

    def test_unavailable_case_does_not_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"synthetic.NEF"; path.write_bytes(b"not a real raw")
            with patch.object(tool,"_preflight", return_value={"status":"unavailable","reason":"test-context"}), \
                 patch.object(tool,"load_raw") as load:
                row = tool.run_case(path, {"decoder":"raw9","size":"full","scale":"native"})
            self.assertEqual(row["status"], "unavailable")
            self.assertEqual(len(row["sha256"]),64)
            load.assert_not_called()


if __name__ == "__main__": unittest.main()
