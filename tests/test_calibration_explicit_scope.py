# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit profile restrictions survive dependency views and JSON reloads.

CSV permissions come from the fixed official Collect writer fixture. These
tests add applicability restrictions, never manufacture qualification or
replace the real DNG decoder/noise-model selection.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from dngscan import calibration, priors, readout
from dngscan.analysis import analyze
from dngscan.calibration_capabilities import (PERMISSION_SEMANTICS_VERSION,
                                            PRODUCTS, capability_prior, stages)
from dngscan.collect_manifest import prepare_sources
from dngscan.raw_io import load_raw
from tests.test_readout_contract import edit_ifd
from tests.test_spectral_fallback_pipeline import write_noise_dng


FIXTURE = Path(__file__).parent / "fixtures/collect-writer/official.zip"


def _without(content: bytes, field: str) -> bytes:
    prefix = ("#" + field + ":").encode()
    return b"\n".join(line for line in content.splitlines()
                       if not line.startswith(prefix)) + b"\n"


class ExplicitCalibrationScopeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ,
            DNGSCAN_CALIBRATION_DIR=str(self.root / "store"))
        environment.start()
        self.addCleanup(environment.stop)
        self.counter = 0

    def payload(self, *, psd_body=None, csv_shutter="fixture"):
        self.counter += 1
        folder = self.root / f"measurements-{self.counter}"
        folder.mkdir()
        with zipfile.ZipFile(FIXTURE) as archive:
            for index, name in enumerate(archive.namelist()):
                content = _without(archive.read(name), "Firmware")
                content = content.replace(b"6000x4000", b"128x128")
                if csv_shutter != "fixture":
                    content = _without(content, "ShutterType")
                    if csv_shutter is not None:
                        content += ("\n#ShutterType: " + csv_shutter + "\n").encode()
                if psd_body and b"#Axis: horizontal" in content:
                    content += ("\n#BodySerialNumber: " + psd_body + "\n").encode()
                (folder / f"measurement-{index}.csv").write_bytes(content)
        manifest, payloads = prepare_sources(folder)
        self.assertEqual(len(payloads), 1, manifest)
        self.last_measurements = folder
        payload = next(iter(payloads.values()))
        # This is an actual parser result, including per-table acquisition
        # records. Import validation must recalculate their permissions.
        self.assertTrue(payload["measurement_acquisition"]["measurements"])
        return payload

    def install(self, payload):
        self.counter += 1
        path = self.root / f"profile-{self.counter}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        original = path.read_bytes()
        installed = calibration.import_calibration(path)
        self.assertEqual(path.read_bytes(), original)
        # Force the public store reader to validate the installed JSON again.
        rows = calibration.list_calibrations()
        self.assertTrue(any(row["id"] == installed["id"] and not row.get("error")
                            for row in rows), rows)
        return installed

    def bundle(self, *, body="BODY-A", file_profile=False, shutter="mechanical"):
        self.counter += 1
        path = self.root / f"capture-{self.counter}.dng"
        write_noise_dng(path, file_profile=file_profile)
        fields = {
            271: (2, 6, b"NIKON\0"), 272: (2, 11, b"TestCamera\0"),
            34855: (3, 1, struct.pack("<H", 100)),
            50714: (5, 1, struct.pack("<LL", 100, 1)),
            50717: (4, 1, struct.pack("<L", 4095)),
        }
        if body is not None:
            value = body.encode("ascii") + b"\0"
            fields[50735] = (2, len(value), value)
        edit_ifd(path, fields)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        bundle = load_raw(path, scene_half_size=True,
            capture_declaration={"version": 1, "shutter": shutter} if shutter else None)
        self.assertEqual(bundle.scene_decoder, "libraw")
        self.assertEqual(bundle.capture_readout.get("body_serial"), body)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        return bundle, digest

    @staticmethod
    def select(capture):
        return priors.find_priors("NIKON", "TestCamera", shutter=capture.get("shutter"),
                                 iso=100, readout=capture)

    def test_public_override_narrows_any_csv_for_actual_dng_analysis(self):
        bundles = {mode: self.bundle(shutter=mode) for mode in ("mechanical", "electronic", "efcs", None)}
        for restriction in (None, "mechanical", "electronic", "efcs"):
            self.payload(csv_shutter="any")
            installed = calibration.import_calibration(self.last_measurements, shutter_override=restriction)
            for mode, (bundle, digest) in bundles.items():
                with self.subTest(restriction=restriction, actual=mode):
                    selected = self.select(bundle.capture_readout)
                    allowed = restriction is None or mode == restriction
                    result, _, _ = analyze(bundle, 4, diagnostics=False)
                    self.assertEqual(result.noise_model.status == "valid", allowed)
                    if allowed:
                        self.assertAlmostEqual(result.gain_e_per_dn, 2.)
                        self.assertEqual(selected["shutter"], restriction or "any")
                    else:
                        self.assertIsNone(result.gain_e_per_dn)
                        self.assertIsNone(result.prior_read_noise_e)
                    self.assertEqual(hashlib.sha256(bundle.path.read_bytes()).hexdigest(), digest)
            calibration.remove_calibration(installed["id"])

    def test_json_explicit_shutter_survives_reload_and_rebinding(self):
        payload = self.payload(csv_shutter="any")
        payload["readout_scope"]["explicit_shutter"] = "mechanical"
        installed = self.install(payload)
        calibration.import_calibration(installed["path"])
        bundle, _ = self.bundle()
        first = self.select(bundle.capture_readout)
        original = copy.deepcopy(first)
        for mode in ("electronic", "efcs", None):
            with self.subTest(actual=mode):
                changed = priors.with_readout(first, {**bundle.capture_readout, "shutter": mode}, shutter=mode)
                self.assertEqual(changed["shutter"], "mechanical")
                self.assertIsNone(priors.gain_e_per_dn(changed, 100))
                self.assertEqual(changed["shutter_match_status"], "unverified" if mode is None else "mismatch")
                for name in PRODUCTS:
                    self.assertFalse(stages(changed, iso=100)[name]["applicable"])
                    self.assertEqual(capability_prior(changed, name)["shutter"], "mechanical")
                restored = priors.with_readout(changed, bundle.capture_readout, shutter="mechanical")
                self.assertAlmostEqual(priors.gain_e_per_dn(restored, 100), 2.)
                self.assertEqual(restored["shutter_match_status"], "matched")
        self.assertEqual(first, original)

    def test_any_override_does_not_widen_specific_csv_and_conflict_is_rejected(self):
        for restriction in ("mechanical", "any", "electronic"):
            self.payload(csv_shutter="mechanical")
            installed = calibration.import_calibration(self.last_measurements, shutter_override=restriction)
            for mode in ("mechanical", "electronic", None):
                with self.subTest(restriction=restriction, actual=mode):
                    bundle, _ = self.bundle(shutter=mode)
                    result, _, _ = analyze(bundle, 4, diagnostics=False)
                    allowed = restriction != "electronic" and mode == "mechanical"
                    self.assertEqual(result.noise_model.status == "valid", allowed)
                    if allowed:
                        self.assertAlmostEqual(result.gain_e_per_dn, 2.)
                        self.assertEqual(self.select(bundle.capture_readout)["shutter"], "mechanical")
            calibration.remove_calibration(installed["id"])

    def test_override_cannot_qualify_csv_without_shutter_measurement(self):
        self.payload(csv_shutter=None)
        calibration.import_calibration(self.last_measurements, shutter_override="mechanical")
        bundle, _ = self.bundle()
        result, _, _ = analyze(bundle, 4, diagnostics=False)
        self.assertNotEqual(result.noise_model.status, "valid")
        self.assertIsNone(result.gain_e_per_dn)

    def test_shutter_rejection_uses_file_noiseprofile_and_preserves_scene(self):
        self.payload(csv_shutter="any")
        calibration.import_calibration(self.last_measurements, shutter_override="mechanical")
        bundle, digest = self.bundle(shutter="electronic", file_profile=True)
        before = bundle.scene_rec2020_render.copy()
        result, _, _ = analyze(bundle, 4, diagnostics=False)
        self.assertEqual(result.noise_model.source, "DNG NoiseProfile")
        self.assertEqual(result.noise_model.status, "valid")
        diagnostics = calibration.calibration_diagnostics("NIKON", "TestCamera", "electronic", 100,
                                                        bundle.capture_readout, analysis=result)
        self.assertEqual(diagnostics[0]["reason"], "readout-mode-mismatch")
        self.assertIsNone(result.gain_e_per_dn)
        np.testing.assert_array_equal(bundle.scene_rec2020_render, before)
        self.assertEqual(hashlib.sha256(bundle.path.read_bytes()).hexdigest(), digest)

    def assert_blocked(self, selected, field):
        self.assertIsNotNone(selected)
        self.assertIsNone(priors.gain_e_per_dn(selected, 100))
        self.assertIsNone(priors.read_noise_e(selected, 100))
        self.assertFalse(priors.prior_usability(selected)[0])
        for name in PRODUCTS:
            state = stages(selected, iso=100)[name]
            self.assertFalse(state["applicable"], (name, state))
            reason = selected["capability_matches"][name]["reason"]
            # A dependent PSD with a missing per-table body declaration is
            # already unqualified. Preserve that original failure rather
            # than demanding the later file mismatch replace its reason.
            self.assertIn("body" if field == "body_serial" else field.replace("_", "-"), reason)
            self.assertIn(field, capability_prior(selected, name)["readout_contract"])

    def test_explicit_body_wrong_or_missing_blocks_real_dng_model(self):
        payload = self.payload()
        payload["readout_contract"]["body_serial"] = "BODY-A"
        self.install(payload)
        for body in ("BODY-B", None):
            with self.subTest(body=body):
                bundle, digest = self.bundle(body=body)
                self.assert_blocked(self.select(bundle.capture_readout), "body_serial")
                result, _, _ = analyze(bundle, 4, diagnostics=False)
                self.assertNotEqual(result.noise_model.status, "valid")
                self.assertIsNone(result.gain_e_per_dn)
                self.assertIsNone(result.prior_read_noise_e)
                self.assertEqual(hashlib.sha256(bundle.path.read_bytes()).hexdigest(), digest)

    def test_matching_explicit_body_keeps_all_measured_capabilities(self):
        payload = self.payload()
        payload["readout_contract"]["body_serial"] = "BODY-A"
        installed = self.install(payload)
        bundle, _ = self.bundle()
        selected = self.select(bundle.capture_readout)
        self.assertEqual(selected["calibration_id"], installed["id"])
        self.assertAlmostEqual(priors.gain_e_per_dn(selected, 100), 2.)
        self.assertIsNotNone(priors.read_noise_e(selected, 100))
        for name, state in stages(selected, iso=100).items():
            self.assertTrue(state["applicable"], (name, state))
            view = capability_prior(selected, name)
            self.assertEqual(view["readout_contract"]["body_serial"], "BODY-A")
            self.assertEqual(readout.match(view, bundle.capture_readout)[0], "matched")
        result, _, _ = analyze(bundle, 4, diagnostics=False)
        self.assertEqual(result.noise_model.status, "valid")

    def test_explicit_missing_firmware_readout_and_capture_kind_are_not_filled(self):
        bundle, _ = self.bundle()
        for field, value in (("firmware", "1.00"), ("readout_id", "full-still"),
                             ("capture_kind", "single-still")):
            with self.subTest(field=field):
                payload = self.payload()
                payload["readout_contract"][field] = value
                self.install(payload)
                capture = dict(bundle.capture_readout)
                capture[field] = None
                selected = self.select(capture)
                self.assert_blocked(selected, field)
                self.assertIsNone(capture[field])
                view = capability_prior(selected, "absolute_gain")
                self.assertEqual(readout.match(view, capture),
                    ("unverified", f"file-{field.replace('_', '-')}-unavailable"))
                matched = priors.with_readout(selected, {**capture, field: value}, shutter="mechanical")
                self.assertAlmostEqual(priors.gain_e_per_dn(matched, 100), 2.)
                wrong = priors.with_readout(matched, {**capture, field: "other"}, shutter="mechanical")
                self.assert_blocked(wrong, field)
                calibration.remove_calibration(selected["calibration_id"])

    def test_json_reimport_and_capture_rebind_retain_explicit_body(self):
        payload = self.payload()
        payload["readout_contract"]["body_serial"] = "BODY-A"
        installed = self.install(payload)
        # A stored wrapper is a supported public reimport input, not a private
        # validated-prior shortcut or a cached in-memory permission object.
        repeated = calibration.import_calibration(installed["path"])
        self.assertEqual(repeated["id"], installed["id"])
        bundle, _ = self.bundle()
        first = self.select(bundle.capture_readout)
        original = copy.deepcopy(first)
        changed = priors.with_readout(first, {**bundle.capture_readout, "body_serial": "BODY-B"},
                                     shutter="mechanical")
        self.assertEqual(first, original)
        self.assert_blocked(changed, "body_serial")
        restored = priors.with_readout(changed, bundle.capture_readout, shutter="mechanical")
        self.assertAlmostEqual(priors.gain_e_per_dn(restored, 100), 2.)
        self.assertEqual(capability_prior(restored, "absolute_gain")["readout_contract"]["body_serial"], "BODY-A")

    def test_body_rejection_uses_dng_noiseprofile_without_changing_scene_or_source(self):
        bundle, digest = self.bundle(body="BODY-B", file_profile=True)
        before = bundle.scene_rec2020_render.copy()
        payload = self.payload()
        payload["readout_contract"]["body_serial"] = "BODY-A"
        self.install(payload)
        result, _, _ = analyze(bundle, 4, diagnostics=False)
        self.assertEqual(result.noise_model.source, "DNG NoiseProfile")
        self.assertEqual(result.noise_model.status, "valid")
        self.assertEqual(result.noise_model.coefficients("G1"), (1e-4, 1e-8))
        self.assertIn("body-serial", result.noise_model.fallback_reason)
        self.assertIsNone(result.gain_e_per_dn)
        self.assertIsNone(result.prior_read_noise_e)
        np.testing.assert_array_equal(bundle.scene_rec2020_render, before)
        self.assertEqual(hashlib.sha256(bundle.path.read_bytes()).hexdigest(), digest)

    def test_optional_psd_body_does_not_restrict_independent_gain(self):
        payload = self.payload(psd_body="PSD-BODY")
        self.install(payload)
        bundle, _ = self.bundle(body="BODY-A")
        selected = self.select(bundle.capture_readout)
        self.assertAlmostEqual(priors.gain_e_per_dn(selected, 100), 2.)
        state = stages(selected, iso=100)
        self.assertTrue(state["absolute_gain"]["applicable"])
        self.assertFalse(state["spectrum_h"]["applicable"])
        self.assertTrue(state["spectrum_v"]["applicable"])
        self.assertNotIn("body_serial", capability_prior(selected, "absolute_gain")["readout_contract"])
        self.assertIn("measurement-body-incomplete", state["spectrum_h"]["reasons"])
        records = selected["measurement_acquisition"]["measurements"].values()
        self.assertTrue(any(record["kind"] == "spectrum_h" and
                            record["declared"].get("body") == "PSD-BODY" for record in records))

    def test_explicit_body_equal_to_optional_psd_body_restricts_every_product(self):
        payload = self.payload(psd_body="PSD-BODY")
        payload.setdefault("readout_scope", {"version": 1})["explicit"] = {
            "version": 1, "body_serial": "PSD-BODY"}
        self.install(payload)
        bundle, _ = self.bundle(body="BODY-A")
        self.assert_blocked(self.select(bundle.capture_readout), "body_serial")

    def test_explicit_geometry_cannot_be_replaced_by_csv_geometry(self):
        payload = self.payload()
        payload["readout_contract"]["libraw_raw_geometry"] = [256, 128]
        self.install(payload)
        bundle, _ = self.bundle()
        selected = self.select(bundle.capture_readout)
        self.assertIsNone(priors.gain_e_per_dn(selected, 100))
        self.assertFalse(stages(selected, iso=100)["absolute_gain"]["applicable"])
        view = capability_prior(selected, "absolute_gain")
        self.assertTrue(any("geometry" in issue and "conflict" in issue
                            for issue in view["readout_declaration_issues"]), view)

    def test_conflicting_storage_restriction_is_not_silently_dropped(self):
        payload = self.payload()
        payload["readout_contract"]["storage_lossless"] = False
        with self.assertRaisesRegex(ValueError, "storage_lossless|compression|conflict"):
            self.install(payload)

    def test_bound_runtime_readout_edit_remains_an_explicit_restriction(self):
        self.install(self.payload())
        bundle, _ = self.bundle(body="BODY-B")
        selected = self.select(bundle.capture_readout)
        self.assertAlmostEqual(priors.gain_e_per_dn(selected, 100), 2.)
        selected["readout_contract"]["body_serial"] = "BODY-A"
        rebound = priors.with_readout(selected, bundle.capture_readout, shutter="mechanical")
        self.assert_blocked(rebound, "body_serial")

    def test_legacy_optional_body_ambiguity_requires_reimport_not_permission(self):
        payload = self.payload(psd_body="PSD-BODY")
        folder = self.last_measurements
        payload.pop("readout_scope", None)
        self.install(payload)
        bundle, _ = self.bundle(body="BODY-A")
        selected = self.select(bundle.capture_readout)
        self.assertIsNone(priors.gain_e_per_dn(selected, 100))
        self.assertFalse(stages(selected, iso=100)["absolute_gain"]["applicable"])
        self.assertIn("readout-scope-reimport-or-explicit-scope-required",
                      selected["capability_matches"]["absolute_gain"]["reason"])
        # The original CSVs resolve the ambiguity. Reimporting them is the
        # supported migration; no guessed permissions are added to old JSON.
        calibration.import_calibration(folder)
        fresh = self.select(bundle.capture_readout)
        self.assertAlmostEqual(priors.gain_e_per_dn(fresh, 100), 2.)
        self.assertTrue(stages(fresh, iso=100)["absolute_gain"]["applicable"])

    def test_legacy_optional_body_can_be_made_explicit_without_broadening_scope(self):
        payload = self.payload(psd_body="PSD-BODY")
        payload.pop("readout_scope", None)
        payload["readout_scope"] = {"version": 1, "explicit": {
            "version": 1, "body_serial": "PSD-BODY"}}
        self.install(payload)
        wrong, _ = self.bundle(body="BODY-A")
        self.assert_blocked(self.select(wrong.capture_readout), "body_serial")
        correct, _ = self.bundle(body="PSD-BODY")
        self.assertAlmostEqual(priors.gain_e_per_dn(self.select(correct.capture_readout), 100), 2.)

    def test_legacy_complete_csv_constraints_remain_usable(self):
        payload = self.payload()
        payload.pop("readout_scope", None)
        self.install(payload)
        bundle, _ = self.bundle()
        selected = self.select(bundle.capture_readout)
        self.assertAlmostEqual(priors.gain_e_per_dn(selected, 100), 2.)
        self.assertTrue(stages(selected, iso=100)["absolute_gain"]["applicable"])

    def test_permission_semantics_invalidate_cache_without_rewriting_measurements(self):
        payload = self.payload()
        folder = self.last_measurements
        installed = self.install(payload)
        files = [*folder.glob("*.csv"), Path(installed["path"])]
        original = {path: path.read_bytes() for path in files}
        fingerprint = calibration.calibration_fingerprint()
        with patch("dngscan.calibration_capabilities.PERMISSION_SEMANTICS_VERSION",
                   PERMISSION_SEMANTICS_VERSION + 1):
            self.assertNotEqual(calibration.calibration_fingerprint(), fingerprint)
        self.assertEqual(calibration.calibration_fingerprint(), fingerprint)
        self.assertEqual({path: path.read_bytes() for path in files}, original)

    def test_public_scope_redacts_body_and_lens_serial_without_changing_internal_match(self):
        body, lens = "PRIVATE-BODY-SCOPE-019", "PRIVATE-LENS-SCOPE-527"
        payload = self.payload(psd_body=body)
        payload["readout_scope"]["explicit"] = {
            "version": 1, "body_serial": body, "lens_serial": lens}
        installed = self.install(payload)
        rows = [installed, *calibration.list_calibrations()]
        for row in rows:
            public = json.dumps(row, ensure_ascii=False)
            self.assertNotIn(body, public)
            self.assertNotIn(lens, public)
            scope = row["readout_scope"]
            self.assertTrue(scope["automatic"]["body_serial_redacted"])
            self.assertTrue(scope["explicit"]["body_serial_redacted"])
            self.assertTrue(scope["explicit"]["lens_serial_redacted"])
        stored = json.loads(Path(installed["path"]).read_text())
        internal = calibration._validated_prior(stored["payload"])
        view = capability_prior(internal, "absolute_gain")
        self.assertEqual(view["readout_contract"]["body_serial"], body)
        self.assertEqual(view["readout_contract"]["lens_serial"], lens)
        self.assertEqual(internal["readout_scope"]["automatic"]["body_serial"], body)


if __name__ == "__main__":
    unittest.main()
