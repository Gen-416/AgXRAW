# SPDX-License-Identifier: GPL-3.0-or-later
"""Recipe provenance, cache migration and transient fallback control flow.

These fixtures do not measure Apple rendering or real-camera image quality.
"""
from contextlib import ExitStack
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dngscan.decoder_identity import decoder_output_identity, mapping_identity, bind_project_handoff
from dngscan.coreimage_decode import COREIMAGE_RECIPE_VERSION
from dngscan.gui import preview_cache as pc
from dngscan.gui.service import (_cached_full_analysis, _load_export_scene,
                                 _preview_analysis_envelope)
from tests.test_preview_cache import _analysis, _bundle


def apple_bundle():
    return replace(_bundle(), scene_decoder="coreimage", scene_decoder_version="9",
                   scene_decoder_recipe={"schema": COREIMAGE_RECIPE_VERSION,
                       "actual_decoder_token": "9", "configuration_id": "a" * 64,
                       "requested_version": "auto", "controls": {
                           "color_noise": {"applicable": False, "actual": 0.0}},
                       "builtin_joint_denoise": "active"},
                   scene_window_calibration={"status": "unmeasured",
                       "reason": "no-qualified-profile", "asset_sha256": "b" * 64})


class RecipeCacheTests(unittest.TestCase):
    def test_actual_hot_wb_is_recorded_without_changing_decoder_configuration(self):
        from dngscan.raw_io import rebalance_raw_bundle
        from dngscan.decoder_residual_research import output_transform_identity
        original = rebalance_raw_bundle(apple_bundle(), "camera")
        balanced = rebalance_raw_bundle(original, "daylight")
        self.assertEqual(balanced.scene_decoder_recipe["configuration_id"],
                         original.scene_decoder_recipe["configuration_id"])
        self.assertNotEqual(output_transform_identity(original.scene_decoder_recipe),
                            output_transform_identity(balanced.scene_decoder_recipe))
        handoff = balanced.scene_decoder_recipe["project_handoff"]["project_balance"]
        self.assertEqual(handoff["effective_mode"], "daylight")
        self.assertEqual(handoff["target_multipliers"], balanced.applied_wb)
        self.assertEqual(original.scene_decoder_recipe["project_handoff"]["project_balance"]["effective_mode"], "camera")

    def test_output_scope_includes_project_scale_and_wb_strategy_not_image_exposure(self):
        recipe = apple_bundle().scene_decoder_recipe
        handoff = {"scale_policy": "aligned", "hot_wb_implementation": "project-v1",
                   "decode_wb": "as-shot", "alignment_factor": 1.2,
                   "alignment_reference": {"decoder": "libraw", "version": "test",
                                           "highlight_mode": "reconstruct", "status": "measured"}}
        aligned = bind_project_handoff(recipe, handoff)
        self.assertEqual(aligned["coreimage_configuration_id"], recipe["configuration_id"])
        self.assertEqual(aligned, bind_project_handoff(aligned, handoff))
        native = bind_project_handoff(recipe, dict(handoff, scale_policy="unity"))
        self.assertNotEqual(aligned["configuration_id"], native["configuration_id"])
        other_image = bind_project_handoff(recipe, dict(handoff, alignment_factor=2.4, scene_scale=4.0))
        self.assertEqual(aligned["configuration_id"], other_image["configuration_id"])
        other_wb = bind_project_handoff(recipe, dict(handoff, hot_wb_implementation="new-native-WB"))
        self.assertNotEqual(aligned["configuration_id"], other_wb["configuration_id"])
        self.assertIsNone(bind_project_handoff({}, handoff)["configuration_id"])

    def test_cli_report_keeps_recipe_and_does_not_call_zero_parameters_noise_free(self):
        from dngscan.report import summary_lines, csv_row, processing_evidence_summary
        bundle, analysis = apple_bundle(), _analysis()
        summary = "\n".join(summary_lines(bundle, analysis))
        self.assertIn("不表示关闭内建处理", summary)
        self.assertIn("未测量，保持未校正窗口", summary)
        row = csv_row(bundle, analysis, None)
        self.assertEqual(json.loads(row["scene_decoder_recipe"]), bundle.scene_decoder_recipe)
        report = processing_evidence_summary(bundle, analysis)
        report["decoder_recipe"]["configuration_id"] = "changed"
        self.assertEqual(bundle.scene_decoder_recipe["configuration_id"], "a" * 64)

    def test_disk_roundtrip_and_envelope_keep_independent_recipe(self):
        bundle = apple_bundle()
        entry = pc.build_proxy_entry(bundle, _analysis())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cache.npz"
            pc._write_disk_entry(path, entry)
            restored = pc._read_disk_entry(path, bundle.path, False)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.bundle.scene_decoder_recipe, bundle.scene_decoder_recipe)
        self.assertEqual(restored.bundle.scene_window_calibration, bundle.scene_window_calibration)
        restored.bundle.scene_decoder_recipe["controls"]["color_noise"]["actual"] = 1.0
        self.assertEqual(bundle.scene_decoder_recipe["controls"]["color_noise"]["actual"], 0.0)
        self.assertEqual(entry.source_metadata["scene_decoder_recipe"]["controls"]["color_noise"]["actual"], 0.0)

    def test_full_analysis_reuse_requires_same_effective_recipe_and_window(self):
        with tempfile.TemporaryDirectory() as td, patch.object(pc, "_cache_dir", return_value=Path(td)):
            path = Path(td) / "source.dng"
            path.write_bytes(b"fixture")
            bundle = replace(apple_bundle(), path=path)
            entry = pc.build_proxy_entry(bundle, _analysis())
            _, entry.cache_digest = pc._cache_identity(path, "reconstruct", "camera", "coreimage", "auto")
            envelope = _preview_analysis_envelope(entry)
            def lookup(candidate):
                return _cached_full_analysis(path, "reconstruct", "camera", "coreimage", "auto", "auto",
                                             decoded_bundle=candidate, envelope=envelope)
            self.assertIsNotNone(lookup(bundle))
            pinned = copy.deepcopy(bundle)
            pinned.scene_decoder_recipe["requested_version"] = "9"
            pinned.scene_decoder_recipe["requested_decoder_token"] = "9"
            pinned.scene_decoder_recipe["offered_tokens"] = ["9", "8"]
            pinned.scene_decoder_recipe["fallback"] = {"used": False, "failed_attempts": []}
            self.assertIsNotNone(lookup(pinned))
            changed = copy.deepcopy(bundle)
            changed.scene_decoder_recipe["configuration_id"] = "c" * 64
            self.assertIsNone(lookup(changed))
            changed = copy.deepcopy(bundle)
            changed.scene_window_calibration["profile_id"] = "different-qualified-profile"
            self.assertIsNone(lookup(changed))
            pc._write_disk_entry(Path(td) / (entry.cache_digest + ".npz"), entry)
            self.assertIsNotNone(_cached_full_analysis(path, "reconstruct", "camera", "coreimage", "auto", "auto",
                                                      decoded_bundle=bundle))

    def test_recipe_schema_and_anchor_library_invalidate_source_cache(self):
        with tempfile.NamedTemporaryFile(suffix=".dng") as source:
            path = Path(source.name)
            with patch("dngscan.scene_transform.decoder_anchor_fingerprint", return_value="first"):
                first = pc._cache_identity(path, "reconstruct", "camera", "coreimage", "9")
            with patch("dngscan.scene_transform.decoder_anchor_fingerprint", return_value="second"):
                second = pc._cache_identity(path, "reconstruct", "camera", "coreimage", "9")
            with patch("dngscan.coreimage_decode.COREIMAGE_RECIPE_VERSION", "new-contract"):
                third = pc._cache_identity(path, "reconstruct", "camera", "coreimage", "9")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_output_identity_binds_actual_controls_and_calibration(self):
        bundle = apple_bundle()
        first = decoder_output_identity(bundle)
        changed = copy.deepcopy(bundle)
        changed.scene_decoder_recipe["controls"]["color_noise"]["actual"] = 1.0
        self.assertNotEqual(first, decoder_output_identity(changed))
        changed = copy.deepcopy(bundle)
        changed.scene_window_calibration["profile_id"] = "new"
        self.assertNotEqual(first, decoder_output_identity(changed))
        self.assertEqual(mapping_identity({"a": 1, "b": 2}), mapping_identity({"b": 2, "a": 1}))
        self.assertNotEqual(mapping_identity({"bad": float("nan")}), mapping_identity({"bad": float("nan")}))

    def test_auto_fallback_retries_after_grace_period_and_keeps_displayed_preview(self):
        failed = replace(apple_bundle(), scene_decoder_version="8", scene_decoder_fallback="RAW9 attempt failed")
        entry = pc.build_proxy_entry(failed, _analysis())
        entry.created_monotonic = 100.0
        fresh = pc.build_proxy_entry(apple_bundle(), _analysis())
        cache = pc.PreviewCache()
        with tempfile.NamedTemporaryFile(suffix=".dng") as source, ExitStack() as stack:
            path = Path(source.name)
            stack.enter_context(patch.object(pc, "_read_disk_entry", side_effect=[entry, fresh]))
            stack.enter_context(patch.object(pc, "_write_disk_entry"))
            clock = stack.enter_context(patch.object(pc.time, "monotonic", return_value=100.0))
            first = cache.get(path, "reconstruct", "camera", decoder="coreimage")
            clock.return_value = 159.0
            self.assertIs(first, cache.get(path, "reconstruct", "camera", decoder="coreimage"))
            clock.return_value = 161.0
            # Export can pin the already displayed algorithm. A new get retries.
            self.assertIs(first, cache.peek(path, "reconstruct", "camera", decoder="coreimage"))
            self.assertIs(fresh, cache.get(path, "reconstruct", "camera", decoder="coreimage"))

    def test_fallback_is_not_persisted_as_permanent_capability(self):
        bundle = replace(apple_bundle(), scene_decoder_fallback="transient RAW9 failure")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cache.npz"
            pc._write_disk_entry(path, pc.build_proxy_entry(bundle, _analysis()))
            self.assertFalse(path.exists())
            # Reject an older writer's fallback even with the current schema.
            import numpy as np
            metadata = {"version": pc.PREVIEW_CACHE_VERSION,
                        "bundle": {"scene_decoder_fallback": "old failure"}}
            np.savez(path, metadata=np.asarray(json.dumps(metadata)))
            self.assertIsNone(pc._read_disk_entry(path, bundle.path, False))

    def test_export_pins_effective_configuration_but_not_auto_request_bookkeeping(self):
        preview = apple_bundle()
        fixed = copy.deepcopy(preview)
        fixed.scene_decoder_recipe["requested_version"] = "9"
        with patch("dngscan.gui.service.dg.load_raw", return_value=fixed) as load:
            result = _load_export_scene(preview.path, "reconstruct", "camera", "coreimage", "auto", "auto", "aligned", preview)
        self.assertIs(result, fixed)
        self.assertEqual(load.call_args.kwargs["coreimage_version"], "9")
        fixed.scene_decoder_recipe["configuration_id"] = "d" * 64
        with patch("dngscan.gui.service.dg.load_raw", return_value=fixed):
            with self.assertRaisesRegex(RuntimeError, "配方与预览不同"):
                _load_export_scene(preview.path, "reconstruct", "camera", "coreimage", "auto", "auto", "aligned", preview)


if __name__ == "__main__":
    unittest.main()
