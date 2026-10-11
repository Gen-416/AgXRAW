# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact RAW9 anchor qualification; synthetic contracts are not camera calibration."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import scene_transform as scene
from tools import calibrate_raw9_anchors as calibrator


def raw9_bundle(**changes):
    fields = {"scene_decoder": "coreimage", "scene_decoder_fallback": None,
        "shot_make": "SIGMA", "shot_model": "fp", "scene_decoder_version": "9", "scene_decoder_runtime": "test OS build",
        "scene_decoder_recipe": {"schema": "coreimage-decode-recipe-1", "actual_decoder_token": "9",
                                 "runtime": "test OS build", "configuration_id": "a" * 64,
                                 "linear_access": {"status": "parameter-state-verified"}, "fallback": {"used": False}}}
    fields.update(changes)
    return SimpleNamespace(**fields)


def qualified_asset(bundle):
    inputs = [{"sha256": hashlib.sha256(str(index).encode()).hexdigest(),
               "split": "train" if index < 2 else "holdout"} for index in range(4)]
    def fit(ratios):
        return {"status": "qualified", "ratio_rg_bg": ratios, "independent_frames": {"train": 2, "holdout": 2},
                "holdout_median_relative_error": [0., 0.], "holdout_p95_relative_error": [0., 0.]}
    profile = {"id": "test-profile", "status": "qualified", "scope": scene.decoder_anchor_scope(bundle),
        "validation": {"status": "passed", "independent_holdout": True, "spatial_support": "flat-target-rois"},
        "source": {"inputs": inputs}, "transport": {"global": fit([.95, 1.1]),
            "per_class": {"skin": fit([.9, 1.2])}}}
    return {"schema": scene.DECODER_ANCHOR_SCHEMA, "version": 3, "profiles": [profile]}


class Raw9AnchorScopeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "anchors.json"
        binding = patch.object(scene, "DECODER_ANCHOR_TRANSPORT_JSON", self.path)
        binding.start()
        self.addCleanup(binding.stop)
        self.bundle = raw9_bundle()
        self.asset = qualified_asset(self.bundle)
        self.write_asset()

    def write_asset(self):
        self.path.write_text(json.dumps(self.asset))

    def ratios(self, bundle, region="skin"):
        return scene.decoder_window_ratios(scene.window_transport_tag(bundle), region)

    def test_exact_scope_is_qualified_and_reports_content_identities(self):
        self.assertEqual(self.ratios(self.bundle), (.9, 1.2))
        self.assertEqual(self.ratios(self.bundle, "unknown-class"), (.95, 1.1))
        report = scene.decoder_window_calibration(self.bundle)
        self.assertEqual(report["status"], "qualified")
        self.assertEqual(report["profile_id"], "test-profile")
        self.assertEqual(report["asset_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(len(report["profile_sha256"]), 64)

    def test_camera_make_model_token_runtime_and_configuration_are_exact(self):
        variants = [raw9_bundle(shot_make="Another maker"), raw9_bundle(shot_model="fp L")]
        for field, value in (("actual_decoder_token", "8"), ("actual_decoder_token", "9.dng"),
                             ("runtime", "different build"), ("configuration_id", "b" * 64)):
            recipe = {**self.bundle.scene_decoder_recipe, field: value}
            variants.append(raw9_bundle(scene_decoder_recipe=recipe))
        for bundle in variants:
            with self.subTest(bundle=vars(bundle)):
                self.assertIsNone(self.ratios(bundle))
                self.assertEqual(scene.decoder_window_calibration(bundle)["status"], "unmeasured")

    def test_raw8_fallback_libraw_and_missing_recipe_cannot_use_raw9_anchors(self):
        for bundle in (raw9_bundle(scene_decoder="libraw"),
                       raw9_bundle(scene_decoder_fallback="RAW9 failed, RAW8 used"),
                       raw9_bundle(scene_decoder_recipe={}), raw9_bundle(scene_decoder_recipe=None)):
            with self.subTest(bundle=vars(bundle)):
                self.assertIsNone(self.ratios(bundle))
        self.assertEqual(scene.decoder_window_calibration(raw9_bundle(scene_decoder="libraw"))["status"], "identity")
        for field in ("actual_decoder_token", "runtime", "configuration_id"):
            recipe = dict(self.bundle.scene_decoder_recipe)
            recipe.pop(field)
            self.assertIsNone(self.ratios(raw9_bundle(scene_decoder_recipe=recipe)))

    def test_recipe_parameter_state_and_flat_bundle_identity_must_agree(self):
        for changes in ({"scene_decoder_version": "8"}, {"scene_decoder_runtime": "another build"},
                        {"scene_decoder_version": None}, {"scene_decoder_runtime": None}):
            self.assertIsNone(self.ratios(raw9_bundle(**changes)))
        for key, value in (("linear_access", {"status": "parameter-state-unverified"}),
                           ("fallback", {"used": True})):
            recipe = {**self.bundle.scene_decoder_recipe, key: value}
            self.assertIsNone(self.ratios(raw9_bundle(scene_decoder_recipe=recipe)))

    def test_legacy_default_never_matches_any_camera(self):
        self.asset = {"version": 1, "coreimage": {"default": {"global_ratio_rg_bg": [.9, 1.2]}}}
        self.write_asset()
        self.assertIsNone(self.ratios(self.bundle))
        for token in ("coreimage", "coreimage|fp", "coreimage|Some Future Camera", "libraw"):
            self.assertIsNone(scene.decoder_window_ratios(token, "skin"))

    def test_asset_change_invalidates_old_tag_and_window_cache_identity(self):
        tag = scene.window_transport_tag(self.bundle)
        first = scene.decoder_anchor_fingerprint()
        np.testing.assert_allclose(scene._compose_transport((1.1, .9, tag), "skin"), (.99, 1.08), rtol=1e-15)
        self.asset["profiles"][0]["transport"]["per_class"]["skin"]["ratio_rg_bg"] = [1.1, .8]
        self.write_asset()
        self.assertNotEqual(scene.decoder_anchor_fingerprint(), first)
        self.assertNotEqual(scene.window_transport_tag(self.bundle), tag)
        self.assertIsNone(scene.decoder_window_ratios(tag, "skin"))
        self.assertEqual(self.ratios(self.bundle), (1.1, .8))

    def test_ambiguous_unvalidated_or_nonfinite_entries_are_unmeasured(self):
        original = copy.deepcopy(self.asset)
        cases = []
        duplicate = copy.deepcopy(original)
        duplicate["profiles"].append(copy.deepcopy(duplicate["profiles"][0]))
        cases.append(duplicate)
        invalid = copy.deepcopy(original)
        invalid["profiles"][0]["validation"]["independent_holdout"] = False
        cases.append(invalid)
        invalid = copy.deepcopy(original)
        invalid["profiles"][0]["transport"] = {"global": {"status": "qualified", "ratio_rg_bg": [float("nan"), 1.]}}
        cases.append(invalid)
        for asset in cases:
            with self.subTest(asset=asset):
                self.asset = asset
                self.write_asset()
                self.assertIsNone(self.ratios(self.bundle))
                self.assertEqual(scene.decoder_window_calibration(self.bundle)["status"], "unmeasured")

    def test_no_qualified_global_means_unmeasured_class_has_no_fallback(self):
        self.asset["profiles"][0]["transport"]["global"]["status"] = "unmeasured"
        self.write_asset()
        self.assertEqual(self.ratios(self.bundle), (.9, 1.2))
        self.assertIsNone(self.ratios(self.bundle, "unknown-class"))


class Raw9AnchorCalibratorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = {"schema": calibrator.MANIFEST_SCHEMA,
            "target_kind": "dedicated-flat-color-target", "actual_decoder_token": "9", "scale_mode": "unity",
            "spatial_support": {"kind": "flat-target-rois", "verified": True,
                                "verification_note": "Synthetic contract test; known corresponding flat patch."},
            "frames": []}
        for index in range(4):
            name = f"target{index}.dng"
            (self.root / name).write_bytes(f"independent synthetic fixture {index}".encode())
            self.manifest["frames"].append({"path": name, "split": "train" if index < 2 else "holdout",
                "rois": [{"id": "same-physical-patch", "libraw": [0, 0, 32, 32], "coreimage": [0, 0, 32, 32]}]})
        self.calls = []
        self.mutate = None

    def loader(self, path, **options):
        self.calls.append(options)
        color = np.asarray([.30, .20, .16], dtype=np.float32)
        if options["decoder"] == "coreimage":
            color *= [1.05, 1., .97]
        bundle = raw9_bundle(scene_decoder=options["decoder"], scene_decoder_version="test-libraw" if options["decoder"] == "libraw" else "9",
            scene_rec2020_render=np.broadcast_to(color, (32, 32, 3)).copy(), scene_scale=1., exposure_gain=1.,
            scene_processing_loss_pct=0., scene_loss_support_untrusted=False, wb_degradation=None,
            scene_reliability_exclusion=None)
        if self.mutate:
            self.mutate(bundle, path, options)
        return bundle

    @staticmethod
    def analyzer(bundle, **options):
        return SimpleNamespace(cell_union_pct=0.), None, None

    def measure(self):
        return calibrator.measure_candidate(self.manifest, self.root, loader=self.loader, analyzer=self.analyzer)

    def test_candidate_requires_explicit_install_and_hashes_actual_inputs(self):
        result = self.measure()
        self.assertEqual(result["status"], "qualified-candidate", result)
        self.assertFalse((self.root / "anchors.json").exists())
        global_fit = result["profile"]["transport"]["global"]
        np.testing.assert_allclose(global_fit["ratio_rg_bg"], [1.05, .97], rtol=1e-6)
        self.assertEqual(global_fit["independent_frames"], {"train": 2, "holdout": 2})
        for call in self.calls:
            self.assertEqual(call["coreimage_version"], "9")
            self.assertEqual(call["coreimage_scale"], "unity")
        destination = self.root / "anchors.json"
        calibrator.install_candidate(result, destination)
        asset = json.loads(destination.read_text())
        self.assertEqual(asset["profiles"][0]["scope"], scene.decoder_anchor_scope(raw9_bundle()))

    def test_mixed_runtime_config_camera_and_wrong_actual_token_are_rejected(self):
        def modify(field, value):
            def apply(bundle, path, options):
                if options["decoder"] == "coreimage" and path.name == "target1.dng":
                    if field in ("shot_make", "shot_model"):
                        setattr(bundle, field, value)
                    else:
                        bundle.scene_decoder_recipe = {**bundle.scene_decoder_recipe, field: value}
            return apply
        for field, value in (("runtime", "new OS build"), ("configuration_id", "b" * 64),
                             ("actual_decoder_token", "8"), ("runtime", None), ("shot_model", "fp L")):
            with self.subTest(field=field, value=value):
                self.mutate = modify(field, value)
                result = self.measure()
                self.assertEqual(result["status"], "unmeasured")
                self.assertIsNone(result["profile"])

    def test_actual_libraw_or_raw8_fallback_is_not_raw9_measurement(self):
        def fallback(bundle, path, options):
            if options["decoder"] == "coreimage":
                bundle.scene_decoder = "libraw"
                bundle.scene_decoder_fallback = "unsupported RAW9"
        self.mutate = fallback
        result = self.measure()
        self.assertIn("actual-decoder-fallback-not-calibration-evidence", result["failures"])

    def test_natural_images_without_verified_flat_support_are_not_calibrated(self):
        self.manifest.pop("spatial_support")
        result = self.measure()
        self.assertEqual(result["status"], "unmeasured")
        self.assertEqual(self.calls, [])
        self.assertIn("verified-flat-target-spatial-support-required", result["failures"])

    def test_structure_roi_geometry_and_known_processing_loss_are_rejected(self):
        def structure(bundle, path, options):
            bundle.scene_rec2020_render[::2, :, 0] *= .5
        def loss(bundle, path, options):
            bundle.scene_processing_loss_pct = .01
        for mutator in (structure, loss):
            with self.subTest(mutator=mutator):
                self.mutate = mutator
                self.assertEqual(self.measure()["status"], "unmeasured")
        self.mutate = None
        self.manifest["frames"][0]["rois"][0]["coreimage"] = [0, 0, 40, 32]
        self.assertIn("roi-outside-actual-decoded-geometry", self.measure()["failures"])

    def test_independent_holdout_failure_and_out_of_bounds_never_clamp(self):
        def vary(bundle, path, options):
            if options["decoder"] == "coreimage" and path.name in ("target2.dng", "target3.dng"):
                bundle.scene_rec2020_render[..., 0] *= 1.2
        self.mutate = vary
        result = self.measure()
        self.assertEqual(result["status"], "unmeasured")
        self.assertEqual(result["profile"]["transport"]["global"]["reason"], "independent-holdout-disagreement")
        def large(bundle, path, options):
            if options["decoder"] == "coreimage":
                bundle.scene_rec2020_render[..., 0] *= 1.5
        self.mutate = large
        result = self.measure()
        record = result["profile"]["transport"]["global"]
        self.assertEqual(record["status"], "rejected")
        self.assertGreater(record["ratio_rg_bg"][0], 1.3)
        self.assertIn("no-clamping", record["reason"])

    def test_duplicate_content_is_not_independent_and_changed_inputs_cannot_install(self):
        result = self.measure()
        (self.root / "target0.dng").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "changed-since"):
            calibrator.install_candidate(result, self.root / "anchors.json")
        (self.root / "target1.dng").write_bytes(b"modified")
        self.assertIn("duplicate-file-content-not-independent-support", self.measure()["failures"])


if __name__ == "__main__":
    unittest.main()
