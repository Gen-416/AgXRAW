# SPDX-License-Identifier: GPL-3.0-or-later
"""Official CSV contracts remain attached to measurements through runtime use."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from dngscan import calibration, priors, readout
from dngscan.analysis import analyze
from dngscan.collect_manifest import prepare_sources
from dngscan.raw_io import load_raw
from tests.test_readout_contract import edit_ifd
from tests.test_spectral_fallback_pipeline import write_noise_dng

FIXTURE = Path(__file__).parent / "fixtures/collect-writer/official.zip"


def without(content, field):
    return b"\n".join(line for line in content.splitlines()
                       if not line.startswith(("#" + field + ":").encode())) + b"\n"


class CollectAcquisitionContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "store"))
        environment.start(); self.addCleanup(environment.stop)
        with zipfile.ZipFile(FIXTURE) as archive:
            self.entries = [(item.filename, archive.read(item)) for item in archive.infolist()]

    def folder(self, name, entries):
        path = self.root / name; path.mkdir()
        for index, (_, content) in enumerate(entries):
            (path / f"measurement-{index}.csv").write_bytes(content)
        return path

    def test_official_package_preserves_each_source_and_qualified_runtime_scope(self):
        report, payloads = prepare_sources(FIXTURE)
        group = report["groups"][0]; payload = payloads[group["id"]]
        acquisition = payload["measurement_acquisition"]
        self.assertEqual(group["status"], "ready")
        self.assertEqual(acquisition["issues"], [])
        self.assertEqual(set(acquisition["measurements"]), {hashlib.sha256(c).hexdigest() for _, c in self.entries})
        self.assertTrue(all(row["status"] == "qualified" for row in acquisition["capability_qualification"].values()))
        self.assertEqual(group["runtime_contract"]["shutter"], payload["shutter"])
        self.assertEqual(payload["acquisition_contract"]["linearisation_curve"], "identity")
        self.assertEqual(payload["acquisition_contract"]["adc_step"], 1.)
        # Renaming and reparsing cannot change physical qualification or source identity.
        renamed, renamed_payloads = prepare_sources(self.folder("renamed", self.entries))
        other = renamed_payloads[renamed["groups"][0]["id"]]
        self.assertEqual(other["measurement_acquisition"], acquisition)

    def test_dark_missing_conditions_are_not_proven_by_gain_or_ptc(self):
        fields = {"ShutterType": ("shutter", "mechanical"), "Compression": ("compression", "lossless"),
                  "LinearisationCurve": ("linearisation", "identity"), "AdcStep": ("adc_step", "1")}
        for field, (normalized, value) in fields.items():
            for only in (b"JPTC-ISOGAIN/1", b"JPTC/2"):
                with self.subTest(field=field, declared_only_in=only.decode()):
                    entries = [(name, without(content, field)) for name, content in self.entries]
                    index = next(i for i, (_, content) in enumerate(entries) if b"#Format: " + only + b"\n" in content)
                    name, content = entries[index]
                    entries[index] = (name, content + f"\n#{field}: {value}\n".encode())
                    report, payloads = prepare_sources(self.folder(field + only.decode().replace("/", "-") , entries))
                    group = report["groups"][0]; payload = payloads[group["id"]]
                    acquisition = payload["measurement_acquisition"]
                    reason = "measurement-" + normalized.replace("_", "-") + "-incomplete"
                    self.assertIn(reason, acquisition["issues"])
                    self.assertNotIn(normalized, acquisition["common_contract"])
                    self.assertEqual(len(acquisition["field_coverage"][normalized]["declared_by"]), 1)
                    self.assertEqual(group["status"], "partial")
                    self.assertFalse(any(group["capabilities"].values()))
                    self.assertTrue(all(row["status"] == "unverified" for row in group["capability_qualification"].values()))
                    prior = calibration._validated_prior(payload)
                    status, why = readout.match(prior, {"storage_lossless": True, "libraw_raw_geometry": [6000, 4000], "firmware": "1.00"})
                    self.assertEqual(status, "unverified")
                    self.assertIn("measurement-", why)
                    self.assertEqual(group["runtime_contract"]["issues"], prior["readout_declaration_issues"])
                    # Qualification summaries are recomputed, not trusted from JSON.
                    edited = copy.deepcopy(payload)
                    edited["measurement_acquisition"]["issues"] = []
                    self.assertIn(reason, calibration._validated_prior(edited)["readout_declaration_issues"])

    def test_pre_contract_manifest_install_requires_reimport_without_blocking_legacy_json(self):
        report, payloads = prepare_sources(FIXTURE)
        old = copy.deepcopy(payloads[report["groups"][0]["id"]])
        old.pop("measurement_acquisition")
        path = self.root / "old-install.json"; path.write_text(json.dumps(old))
        installed = calibration.import_calibration(path)
        self.assertIn("measurement-acquisition-reimport-required", installed["warnings"])
        self.assertFalse(any(installed["capabilities"].values()))
        capture = {"storage_lossless": True, "libraw_raw_geometry": [6000, 4000], "firmware": "1.00"}
        prior = calibration.matching_prior("NIKON", "TestCamera", shutter="mechanical", iso=100, readout=capture)
        self.assertEqual(prior["readout_match_status"], "unverified")
        self.assertEqual(prior["readout_match_reason"], "measurement-acquisition-reimport-required")
        new = calibration.import_calibrations(FIXTURE)["profiles"][0]
        self.assertNotEqual(new["id"], installed["id"])
        prior = calibration.matching_prior("NIKON", "TestCamera", shutter="mechanical", iso=100, readout=capture)
        self.assertEqual(prior["readout_match_status"], "matched")
        self.assertEqual(prior["calibration_id"], new["id"])
        from tests.test_user_calibration import collect_profile
        legacy = calibration._validated_prior(collect_profile())
        self.assertNotIn("measurement-acquisition-reimport-required", legacy["readout_declaration_issues"])
        public = copy.deepcopy(old)
        public["source"]["kind"] = "JPTC collect set (first-party, credit-based grant, NOTICE.md)"
        self.assertNotIn("measurement-acquisition-reimport-required", calibration._validated_prior(public)["readout_declaration_issues"])

    def test_contradictory_conditions_never_form_one_ready_profile(self):
        for field, before, after in (("ShutterType", b"mechanical", b"electronic"),
                                     ("Compression", b"lossless", b"lossy"),
                                     ("LinearisationCurve", b"identity", b"companded"),
                                     ("AdcStep", b"1", b"4")):
            with self.subTest(field=field):
                entries = []
                for name, content in self.entries:
                    if b"#Format: JPTC-ISOGAIN/1" in content:
                        content = content.replace(("#" + field + ": ").encode() + before + b"\n",
                                                  ("#" + field + ": ").encode() + after + b"\n")
                    entries.append((name, content))
                report, payloads = prepare_sources(self.folder("conflict-" + field, entries))
                self.assertGreater(len(report["groups"]), 1)
                self.assertFalse(any(len(p["measurement_acquisition"]["measurements"]) == len(entries)
                                     for p in payloads.values()))

    def test_real_csv_dark_spectrum_and_absolute_gain_packages_compose_in_pipeline(self):
        # Controlled synthetic measurements emitted by the pinned official
        # writer. Supplemental known endpoints are explicit CSV declarations;
        # the importer must never invent them from another package or bit depth.
        gain, dark = [], []
        for name, content in self.entries:
            content = without(content, "Firmware").replace(b"6000x4000", b"128x128")
            if b"#Format: JPTC/2" in content:
                gain.append((name, content + b"\n#BlackLevel: 100,100,100,100\n"))
            elif b"#Format: JPTC-ISOGAIN/1" not in content:
                if b"#Format: JPTC-DARK/1" in content:
                    content += b"\n#SensorMaximum: 4095\n"
                dark.append((name, content))
        gain_result = calibration.import_calibrations(self.folder("gain-only", gain))
        dark_result = calibration.import_calibrations(self.folder("dark-spectrum-only", dark))
        self.assertTrue(gain_result["profiles"][0]["capabilities"]["absolute_gain"])
        self.assertFalse(gain_result["profiles"][0]["capabilities"]["read_noise"])
        self.assertTrue(dark_result["profiles"][0]["capabilities"]["read_noise"])
        self.assertTrue(dark_result["profiles"][0]["capabilities"]["spectrum_h"])
        path = self.root / "matched.dng"; write_noise_dng(path, file_profile=False)
        edit_ifd(path, {271: (2, 6, b"NIKON\0"), 272: (2, 11, b"TestCamera\0"),
                       34855: (3, 1, struct.pack("<H", 100)),
                       50714: (5, 1, struct.pack("<LL", 100, 1)),
                       50717: (4, 1, struct.pack("<L", 4095))})
        bundle = load_raw(path, scene_half_size=True, capture_declaration={"version": 1, "shutter": "mechanical"})
        selected = priors.find_priors("NIKON", "TestCamera", shutter="mechanical", iso=100, readout=bundle.capture_readout)
        self.assertEqual(selected["source"], "Compatible capability composition")
        self.assertEqual(set(selected["capability_sources"]), {"gain", "read_noise", "stored_dark_variance"})
        self.assertNotEqual(selected["capability_sources"]["gain"], selected["capability_sources"]["read_noise"])
        self.assertTrue(selected["capability_measurements"]["gain"])
        result, _, _ = analyze(bundle, 4, diagnostics=False)
        self.assertEqual(result.noise_model.status, "valid")
        self.assertAlmostEqual(result.gain_e_per_dn, 2., places=5)
        self.assertAlmostEqual(result.noise_model.coefficients("G1")[1], 4. / 3995.**2, places=14)
        self.assertTrue(result.noise_model.spectral_sources)
        self.assertEqual(result.noise_model.capability_sources, selected["capability_sources"])
        # A failed physical condition cannot be rescued by the complete gain package.
        _, dark_prior = calibration._read_record(Path(dark_result["profiles"][0]["path"]))
        bad = copy.deepcopy(dark_prior)
        bad["acquisition_contract"]["adc_step"] = 4.
        from dngscan.calibration_registry import compose_capabilities
        _, gain_prior = calibration._read_record(Path(gain_result["profiles"][0]["path"]))
        gain_prior.update(readout_match_status="matched", source="gain")
        bad.update(readout_match_status="matched", source="bad dark")
        self.assertNotIn("capability_sources", compose_capabilities(gain_prior, [bad], 100))


if __name__ == "__main__":
    unittest.main()
