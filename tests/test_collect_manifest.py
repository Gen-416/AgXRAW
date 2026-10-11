# SPDX-License-Identifier: GPL-3.0-or-later
"""Pinned actual upstream writer output reaches shared import, not just glob fixtures."""
import base64
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from dngscan import calibration, readout
from dngscan.collect_manifest import inspect_entries, inspect_sources
from dngscan.gui import calibration_service

FIXTURE = Path(__file__).parent / "fixtures/collect-writer/official.zip"


class CollectManifestTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        env = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "store"))
        env.start(); self.addCleanup(env.stop)
        with zipfile.ZipFile(FIXTURE) as archive:
            self.entries = [{"name": item.filename, "content": archive.read(item)} for item in archive.infolist()]

    def test_actual_writer_zip_directory_renames_and_data_identity(self):
        zipped = calibration.import_calibrations(FIXTURE)
        self.assertEqual(zipped["manifest"]["import_status"], "imported")
        self.assertEqual(len(zipped["profiles"]), 1)
        profile = zipped["profiles"][0]
        self.assertEqual(profile["capabilities"], {"absolute_gain": True, "read_noise": True,
            "ptc_gain": True, "relative_gain": True, "read_noise_dn": True, "read_noise_e": True,
            "stored_dark_variance": True, "spectrum_h": True, "spectrum_v": True, "spectrum_2d": False})
        folder = self.root / "unrelated-directory-name"; folder.mkdir()
        for index, item in enumerate(self.entries):
            (folder / f"renamed-{index}.csv").write_bytes(item["content"])
        renamed = calibration.import_calibrations(folder)
        self.assertEqual(renamed["profiles"][0]["id"], profile["id"])
        self.assertEqual(renamed["manifest"]["content_id"], zipped["manifest"]["content_id"])
        record, prior = calibration._read_record(Path(profile["path"]))
        self.assertEqual(len(record["payload"]["ptc_anchors"]), 2)
        self.assertAlmostEqual(2**calibration.curve_value(prior, "gain_log2iso_log2epd", 100), 2., places=5)
        self.assertAlmostEqual(2**calibration.curve_value(prior, "gain_log2iso_log2epd", 200), 1., places=5)
        changed = [dict(item) for item in self.entries]
        changed[0]["content"] += b"\n#Revision: changed measurement\n"
        self.assertNotEqual(inspect_entries(changed)["content_id"], zipped["manifest"]["content_id"])

    def test_two_cameras_modes_and_grids_form_groups(self):
        entries = list(self.entries)
        entries += [{"name": "other/" + item["name"], "content": item["content"].replace(
            b"NIKON TestCamera", b"SONY TestCamera").replace(b"mechanical", b"electronic").replace(b"6000x4000", b"4000x3000")}
            for item in self.entries]
        report = inspect_entries(entries)
        self.assertEqual(len(report["groups"]), 2)
        self.assertTrue(all(g["status"] == "ready" for g in report["groups"]))

    def test_group_mode_body_and_firmware_constraints_reach_runtime_selection(self):
        entries = []
        for mode in ("A", "B"):
            for item in self.entries:
                content = item["content"] + ("\n#ReadoutMode: " + mode
                    + "\n#DriveMode: still\n#BodySerialNumber: test-serial\n").encode()
                entries.append({"name": mode + "/" + item["name"], "content": content})
        folder = self.root / "modes"; folder.mkdir()
        for i, item in enumerate(entries):
            (folder / f"measurement-{i}.csv").write_bytes(item["content"])
        result = calibration.import_calibrations(folder)
        self.assertEqual(len(result["profiles"]), 2)
        capture = {"storage_lossless": True, "libraw_raw_geometry": [6000, 4000],
                   "readout_id": "a", "capture_kind": "still", "body_serial": "test-serial",
                   "firmware": "1.00"}
        from tests.test_measurement_domain import sample_domain_fixture
        capture["sample_domain"] = sample_domain_fixture()
        selected = calibration.matching_prior("NIKON", "TestCamera", shutter="mechanical", iso=100, readout=capture)
        self.assertEqual(selected["readout_match_status"], "matched")
        self.assertEqual(selected["readout_contract"]["readout_id"], "A")
        for field in ("readout_id", "body_serial", "firmware"):
            incomplete = dict(capture); incomplete.pop(field)
            diagnostics = calibration.calibration_diagnostics("NIKON", "TestCamera", "mechanical", 100, incomplete)
            self.assertTrue(all(row["readout_match_status"] != "matched" for row in diagnostics))
        _, prior = calibration._read_record(Path(result["profiles"][0]["path"]))
        self.assertEqual(prior["readout_contract"]["capture_kind"].casefold(), "still")

    def test_constraints_only_in_gain_table_are_not_dropped_and_alias_conflict_rejects(self):
        entries = [dict(item) for item in self.entries]
        gain = next(item for item in entries if b"JPTC-ISOGAIN/1" in item["content"])
        gain["content"] += b"\n#Readout: single-shot\n#DriveMode: still\n"
        folder = self.root / "partial-metadata"; folder.mkdir()
        for index, item in enumerate(entries):
            (folder / f"table-{index}.csv").write_bytes(item["content"])
        result = calibration.import_calibrations(folder)
        _, prior = calibration._read_record(Path(result["profiles"][0]["path"]))
        acquisition = prior["measurement_acquisition"]
        gain = next(record for record in acquisition["measurements"].values() if record["kind"] == "gain")
        self.assertEqual(gain["declared"]["readout"].casefold(), "single-shot")
        self.assertEqual(gain["declared"]["capture_kind"].casefold(), "still")
        # The declaration survives on its source; a gain CSV cannot silently
        # fill missing conditions on the dark/PTC inputs of that gain ladder.
        for name in ("absolute_gain", "relative_gain"):
            self.assertEqual(acquisition["capability_qualification"][name]["status"], "unverified")
            self.assertIn("measurement-readout-incomplete", acquisition["capability_qualification"][name]["reasons"])
        self.assertEqual(acquisition["capability_qualification"]["ptc_gain"]["status"], "qualified")
        self.assertNotIn("readout_id", prior["readout_contract"])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            readout.collect_fields({"Readout": "A", "ReadoutMode": "B"})
        with self.assertRaises(ValueError):
            readout.normalize_capture_declaration({"version": 1, "body_serial": "other"})

    def test_single_iso_official_ptc_name_is_content_recognised(self):
        ptc = next(item for item in self.entries if b"#ISO: 100\n" in item["content"] and b"#Format: JPTC/2" in item["content"])
        selected = [{"name": "official-single_ptc.csv", "content": ptc["content"]}]
        report = inspect_entries(selected)
        self.assertEqual(report["files"][0]["kind"], "ptc")
        self.assertEqual(report["files"][0]["iso_points"], [100])
        # Official PTC exports explicitly leave the black reference to the
        # dark set: recognise the table, but do not invent its missing zero.
        self.assertFalse(report["groups"][0]["capabilities"]["absolute_gain"])
        dark = next(item for item in self.entries if b"JPTC-DARK/1" in item["content"])
        complete = inspect_entries(selected + [dark])
        self.assertTrue(complete["groups"][0]["capabilities"]["absolute_gain"])

    def test_missing_duplicate_and_invalid_tables_are_visible(self):
        dark = next(item for item in self.entries if b"JPTC-DARK/1" in item["content"])
        report = inspect_entries([dark])
        self.assertFalse(report["groups"][0]["capabilities"]["absolute_gain"])
        duplicate = {"name": "second.csv", "content": dark["content"].replace(b"#Generated:", b"#Revision: repeat\n#Generated:")}
        report = inspect_entries([dark, duplicate])
        self.assertEqual(report["groups"][0]["status"], "ambiguous")
        self.assertEqual(len(report["ambiguous_files"]), 2)
        for content in (dark["content"].replace(b"JPTC-DARK/1", b"JPTC-DARK/999"),
                        dark["content"].replace(b"StdDiffClipped", b"DamagedColumn"),
                        b"#Format: JPTC/2\n#ISO: 100\nISO,G1_Mean,G1_Std\n200,300,5\n"):
            with self.subTest(content=content[:60]):
                result = inspect_entries([{"name": "bad.csv", "content": content}])
                self.assertTrue(result["unused_files"])
                self.assertTrue(result["files"][0]["reason"])
        spectrum = next(item for item in self.entries if b"JPTC-SPECTRUM/1" in item["content"])
        bad = {"name": spectrum["name"], "content": b"\n".join(line for line in spectrum["content"].splitlines() if not line.startswith(b"#FreqUnit:"))}
        self.assertIn("unit", inspect_entries([bad])["files"][0]["reason"])

    def test_zip_safety_inflated_limits_and_symlinks(self):
        def archive(entries):
            data = io.BytesIO()
            with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for name, content in entries: z.writestr(name, content)
            return [{"name": "selected.zip", "content": data.getvalue()}]
        for name in ("../x.csv", "/tmp/x.csv", "C:/x.csv", "a\\x.csv", "a//x.csv"):
            with self.subTest(name=name), self.assertRaises(ValueError): inspect_entries(archive([(name, b"x")]))
        link = zipfile.ZipInfo("link.csv"); link.create_system = 3; link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(ValueError): inspect_entries(archive([(link, b"../../x")]))
        with self.assertRaises(ValueError): inspect_entries(archive([("x.csv", b"a"), ("x.csv", b"b")]))
        with patch("dngscan.collect_manifest.MAX_TOTAL_BYTES", 32), self.assertRaises(ValueError):
            inspect_entries(archive([("x.csv", b"x" * 1000)]))
        with patch("dngscan.collect_manifest.MAX_FILES", 1), self.assertRaises(ValueError):
            inspect_entries(archive([("x.csv", b"x"), ("y.csv", b"y")]))

    def test_gui_preview_does_not_install_and_import_reparses_selected_bytes(self):
        params = {"files": [{"name": "official.zip", "base64": base64.b64encode(FIXTURE.read_bytes()).decode()}]}
        report = calibration_service.inspect_calibration(params)
        self.assertEqual(calibration.list_calibrations(), [])
        group = report["manifest"]["groups"][0]["id"]
        with patch.object(calibration_service, "_invalidate_preview"):
            result = calibration_service.import_calibration({**params, "groupIds": [group]})
        self.assertEqual(len(result["imported"]), 1)
        self.assertEqual(result["imported"][0]["apply_status"], "not-applied")
        self.assertTrue(calibration.list_calibrations()[0]["capabilities"]["absolute_gain"])

    def test_offline_converter_uses_official_zip_without_installing_profiles(self):
        target = self.root / "converted.json"
        result = subprocess.run([sys.executable, "tools/import_jptc_collect.py", str(FIXTURE),
            "--out", str(target)], cwd=Path(__file__).resolve().parents[1],
            text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(target.read_text())
        self.assertEqual(len(payload["ptc_anchors"]), 2)
        self.assertTrue(payload["gain_log2iso_log2epd"])
        self.assertEqual(len(payload["collect_manifest"]["groups"]), 1)
        self.assertEqual(calibration.list_calibrations(), [])


if __name__ == "__main__": unittest.main()
