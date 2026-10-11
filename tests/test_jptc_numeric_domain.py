# SPDX-License-Identifier: GPL-3.0-or-later
"""Single-point CLI -> JSON installation -> real LibRaw numeric-domain checks."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import calibration, priors
from dngscan.analysis import analyze
from dngscan.measurement_domain import DN, DN2, SINGLE_SCHEMA
from dngscan.raw_io import load_raw
from tests import test_jptc_single_readout as single_readout
from tests.test_readout_contract import edit_ifd
from tests.test_spectral_fallback_pipeline import write_noise_dng

REPO = Path(__file__).resolve().parents[1]


class SingleNumericDomainTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "store"))
        environment.start(); self.addCleanup(environment.stop)

    def csv(self, headers=None):
        path = single_readout.SingleJptcReadoutTests.csv(self)
        if headers is not None and "Format" in headers:
            path.write_text("\n".join(line for line in path.read_text().splitlines()
                                      if not line.startswith("#Format:")) + "\n")
        with path.open("a", encoding="utf-8") as stream:
            for key, value in (headers or {}).items():
                if value is not None:
                    stream.write(f"\n#{key}: {value}\n")
        return path

    def convert(self, headers=None, *, rebuild=None, restore_only=False):
        path = self.csv(headers)
        output = self.root / "single.json"
        args = [sys.executable, str(REPO / "tools/import_jptc.py"), str(path),
                "--brand", "SIGMA", "--model", "fp", "--iso", "200", "--shutter", "electronic",
                "--out", str(output)]
        if rebuild:
            args.extend(("--rebuild-prior", str(rebuild)))
        if restore_only:
            args.append("--restore-contract-only")
        process = subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        return output, json.loads(output.read_text(encoding="utf-8"))

    def install(self, path):
        process = subprocess.run([sys.executable, "-m", "dngscan", "calibration", "import", str(path)],
                                 cwd=REPO, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        return json.loads(process.stdout)

    def raw(self, *, file_profile=False, companded=False):
        path = self.root / "capture.dng"
        write_noise_dng(path, file_profile=file_profile)
        if companded:
            curve = np.minimum(np.arange(16384, dtype=np.uint32) * 2, 65535).astype("<u2")
            edit_ifd(path, {50712: (3, 16384, curve.tobytes())})
        return path

    def run_pipeline(self, headers=None, *, file_profile=False, companded=False):
        path = self.raw(file_profile=file_profile, companded=companded)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        before = load_raw(path, scene_half_size=True).scene_rec2020_render.copy()
        output, payload = self.convert(headers)
        summary = self.install(output)
        bundle = load_raw(path, scene_half_size=True)
        result, _, _ = analyze(bundle, 4)
        selected = priors.find_priors(bundle.shot_make, bundle.shot_model,
            shutter=bundle.shot_shutter, iso=bundle.shot_iso, readout=bundle.capture_readout)
        np.testing.assert_array_equal(bundle.scene_rec2020_render, before)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(result.prior_id, payload["id"])
        return payload, summary, bundle, selected, result

    def assert_no_external_model(self, result):
        self.assertIsNone(result.gain_e_per_dn)
        self.assertIsNone(result.prior_read_noise_e)
        self.assertIsNone(result.prior_pdr_ev)

    def test_explicit_unsupported_declarations_survive_both_tools_and_block_model(self):
        cases = (({"DNDomain": "electrons"}, "measurement-dn-domain-unsupported"),
                 ({"VarianceDomain": "electron-squared"}, "measurement-variance-domain-unsupported"),
                 ({"Decoder": "foreign-decoder-output"}, "measurement-decoder-domain-unverified"),
                 ({"LinearisationCurve": "companded"}, "measurement-linearisation-domain-unverified"))
        for index, (headers, reason) in enumerate(cases):
            with self.subTest(headers=headers), patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / f"store-{index}")):
                payload, summary, _, selected, result = self.run_pipeline(headers)
                self.assertEqual(payload["single_point_measurement"]["schema"], SINGLE_SCHEMA)
                self.assertIn(next(iter(headers.values())), payload["single_point_measurement"]["declared"].values())
                self.assertEqual(selected["readout_match_status"], "unverified")
                self.assertIn(reason, summary["warnings"])
                self.assertEqual(result.noise_model.status, "rejected")
                self.assertEqual(result.noise_model.reason, reason)
                self.assertIsNone(result.noise_model.coefficients("G1"))
                self.assert_no_external_model(result)

    def test_each_invalid_source_keeps_independent_noiseprofile_fallback(self):
        for index, headers in enumerate(({"DNDomain": "electrons"}, {"VarianceDomain": "electron-squared"},
                                         {"Decoder": "foreign-output"}, {"LinearisationCurve": "unknown"})):
            with self.subTest(headers=headers), patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / f"fallback-{index}")):
                _, _, _, _, result = self.run_pipeline(headers, file_profile=True)
                self.assertEqual(result.noise_model.status, "valid")
                self.assertEqual(result.noise_model.source, "DNG NoiseProfile")
                self.assertEqual(result.noise_model.coefficients("G1"), (1e-4, 1e-8))
                self.assertTrue(result.noise_model.fallback_reason.startswith("measurement-"))
                self.assert_no_external_model(result)

    def test_supported_aliases_versions_and_optional_headers_keep_limited_single_scope(self):
        for index, headers in enumerate(({}, {"DNDomain": "raw-DN", "VarianceDomain": "DN^2",
                "Decoder": "LibRaw 0.23.0-Release", "LinearisationCurve": "identity"})):
            with self.subTest(headers=headers), patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / f"valid-{index}")):
                payload, _, _, selected, result = self.run_pipeline(headers)
                self.assertEqual(payload["acquisition_contract"]["dn_domain"], DN)
                self.assertEqual(payload["acquisition_contract"]["variance_domain"], DN2)
                self.assertEqual(result.noise_model.status, "valid")
                self.assertAlmostEqual(result.gain_e_per_dn, 4., places=6)
                self.assertAlmostEqual(result.prior_read_noise_e, 3., places=6)
                match = selected["measurement_domain_match"]
                self.assertEqual(match["status"], "matched")
                self.assertIn("unmeasured", match["adc_evidence"])
                self.assertEqual(selected["single_point_measurement"]["paired_dark"], "not-established-by-single-frame-PTC")
                self.assertNotIn("adc_step", payload["acquisition_contract"])
                self.assertNotIn("stored_dark_variance_dn2_log2iso", payload)

    def test_actual_nonidentity_lut_is_not_licensed_by_missing_or_identity_header(self):
        for index, headers in enumerate(({}, {"LinearisationCurve": "identity"})):
            with self.subTest(headers=headers), patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / f"lut-{index}")):
                _, _, bundle, _, result = self.run_pipeline(headers, companded=True, file_profile=True)
                self.assertEqual(bundle.capture_readout["sample_domain"]["linearisation"], "companded")
                self.assertEqual(result.noise_model.source, "DNG NoiseProfile")
                self.assertEqual(result.noise_model.fallback_reason, "file-linearisation-domain-companded")
                self.assert_no_external_model(result)

    def test_explicit_adc_is_checked_without_manufacturing_a_missing_measurement(self):
        _, _, _, _, result = self.run_pipeline({"AdcStep": "0.5"})
        self.assertEqual(result.noise_model.reason, "measurement-adc-step-noninteger")
        self.assert_no_external_model(result)

    def test_retained_declarations_cannot_be_overridden_by_an_edited_contract(self):
        _, payload = self.convert({"DNDomain": "electrons"})
        payload["acquisition_contract"]["dn_domain"] = DN
        prior = calibration._validated_prior(payload)
        self.assertIn("single-point-declaration-contract-conflict", prior["readout_declaration_issues"])

    def test_legacy_generated_record_is_retained_but_requires_hash_matched_reconversion(self):
        output, payload = self.convert()
        payload.pop("single_point_measurement")
        payload["acquisition_contract"] = {"geometry_domain": "camera-jpeg-output"}
        old = self.root / "old.json"; old.write_text(json.dumps(payload))
        summary = self.install(old)
        self.assertTrue(Path(summary["path"]).exists())
        self.assertIn("single-point-domain-reimport-required", summary["warnings"])
        path = self.raw(file_profile=True)
        result, _, _ = analyze(load_raw(path, scene_half_size=True), 4)
        self.assertEqual(result.noise_model.source, "DNG NoiseProfile")
        self.assertEqual(result.noise_model.fallback_reason, "single-point-domain-reimport-required")
        self.assert_no_external_model(result)
        rebuilt, restored = self.convert(rebuild=old)
        self.assertEqual(restored["source"]["input_sha256"], payload["source"]["input_sha256"])
        self.install(rebuilt)
        result, _, _ = analyze(load_raw(path, scene_half_size=True), 4)
        self.assertAlmostEqual(result.gain_e_per_dn, 4., places=6)
        self.assertEqual(result.noise_model.reason, "matched-shot-read-model")
        changed = self.csv({"DNDomain": "electrons"})
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            calibration.rebuild_single_prior(payload, changed)

    def test_reconversion_recovers_lost_invalid_headers_instead_of_granting_a_default(self):
        _, payload = self.convert({"Decoder": "foreign-output"})
        payload.pop("single_point_measurement"); payload["acquisition_contract"] = {}
        old = self.root / "old-invalid.json"; old.write_text(json.dumps(payload))
        rebuilt, restored = self.convert({"Decoder": "foreign-output"}, rebuild=old)
        self.assertEqual(restored["acquisition_contract"]["decoder_identity"], "foreign-output")
        self.install(rebuilt)
        result, _, _ = analyze(load_raw(self.raw(file_profile=False), scene_half_size=True), 4)
        self.assertEqual(result.noise_model.reason, "measurement-decoder-domain-unverified")
        self.assert_no_external_model(result)

    def test_hand_curated_json_and_complete_collect_are_outside_single_migration(self):
        from tests.test_user_calibration import single_profile, collect_profile
        for item in (single_profile(), collect_profile()):
            before = copy.deepcopy(item)
            prior = calibration._validated_prior(item)
            self.assertNotIn("single-point-domain-reimport-required", prior["readout_declaration_issues"])
            self.assertEqual(item, before)

    def test_conflicting_numeric_headers_cannot_hide_an_unsupported_declaration(self):
        path = self.csv({"DNDomain": "electrons"})
        with path.open("a") as stream:
            stream.write("#DNDomain: DN\n")
        with self.assertRaisesRegex(ValueError, "conflicting JPTC declaration: DNDomain"):
            calibration.import_csv(path, "SIGMA", "fp", 200, None, "electronic")

    def test_packaged_single_loader_keeps_numeric_rejection_and_legacy_boundary(self):
        for legacy, reason in ((False, "measurement-decoder-domain-unverified"),
                               (True, "single-point-domain-reimport-required")):
            with self.subTest(legacy=legacy):
                output, payload = self.convert({"Decoder": "foreign-output"})
                if legacy:
                    payload.pop("single_point_measurement"); payload["acquisition_contract"] = {}
                    output.write_text(json.dumps(payload))
                packaged = Path(priors.__file__).parent / "data/priors/jptc"
                collect = packaged.parent / "jptc_collect"
                original_glob = Path.glob
                def files(path, pattern):
                    if path == packaged:
                        return iter([output])
                    if path == collect:
                        return iter([])
                    return original_glob(path, pattern)
                bundle = load_raw(self.raw(file_profile=True), scene_half_size=True)
                with patch.object(Path, "glob", files), patch.object(priors, "_JPTC_CACHE", None), \
                        patch.object(priors, "PRIOR_TABLE", []):
                    result, _, _ = analyze(bundle, 4)
                self.assertEqual(result.noise_model.source, "DNG NoiseProfile")
                self.assertEqual(result.noise_model.fallback_reason, reason)
                self.assert_no_external_model(result)

    def test_legacy_column_contract_remains_limited_and_unknown_explicit_format_is_rejected(self):
        _, _, _, selected, result = self.run_pipeline({"Format": None})
        self.assertEqual(result.noise_model.status, "valid")
        self.assertEqual(selected["single_point_measurement"]["format_source"], "legacy-single-frame-column-contract")
        self.assertEqual(selected["single_point_measurement"]["source_lut"], "unrecorded")
        with patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "unknown-format")):
            _, _, _, _, result = self.run_pipeline({"Format": "JPTC/999"})
            self.assertEqual(result.noise_model.reason, "single-point-format-domain-unverified")
            self.assert_no_external_model(result)

    def test_header_restoration_preserves_fit_scope_and_recovers_unsupported_declarations(self):
        _, payload = self.convert({"DNDomain": "electrons"})
        original = self.root / "ptc-iso200.csv"
        payload.pop("single_point_measurement"); payload["acquisition_contract"] = {}
        old = self.root / "old-fit.json"; old.write_text(json.dumps(payload))
        _, restored = self.convert({"DNDomain": "electrons"}, rebuild=old, restore_only=True)
        for field in ("gain_e_per_dn", "read_noise_e", "fwc_e", "quality", "geometry", "readout_contract"):
            self.assertEqual(restored[field], payload[field])
        prior = calibration._validated_prior(restored)
        self.assertIn("measurement-dn-domain-unsupported", prior["readout_declaration_issues"])
        with original.open("a") as stream:
            stream.write("\n#Decoder: foreign-output\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            calibration.restore_single_contract(payload, original)

    def test_nine_published_assets_have_recovered_source_headers_with_limited_scope(self):
        files = sorted((REPO / "dngscan/data/priors/jptc").glob("*.json"))
        self.assertEqual(len(files), 9)
        for path in files:
            with self.subTest(path=path.name):
                item = json.loads(path.read_text())
                prior = calibration._validated_prior(item)
                record = prior["single_point_measurement"]
                self.assertEqual(record["input_sha256"], item["source"]["input_sha256"])
                self.assertEqual(record["status"], "limited-single-frame-model")
                self.assertEqual(record["source_lut"], "unrecorded")
                self.assertEqual(record["adc"], "unmeasured")
                self.assertEqual(record["paired_dark"], "not-established-by-single-frame-PTC")
                self.assertIn(item["source"]["upstream_commit"], item["source"]["upstream_url"])
                self.assertNotIn("single-point-domain-reimport-required", prior["readout_declaration_issues"])


if __name__ == "__main__":
    unittest.main()
