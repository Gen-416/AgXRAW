# SPDX-License-Identifier: GPL-3.0-or-later
"""Identity facts travel from file bytes through evidence into prior selection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from dngscan import calibration, priors, readout
from dngscan.analysis import analyze
from dngscan.capture_identity import read, public_capture
from dngscan.raw_io import load_raw
from tests.test_readout_contract import edit_ifd
from tests.test_spectral_fallback_pipeline import write_noise_dng
from tests.test_user_calibration import collect_profile


def tiff_bytes(fields, *, endian="<"):
    """Small original fixture writer; external payload offsets have TIFF origin."""
    payload_offset = 8 + 2 + 12 * len(fields) + 4
    entries, payload = [], bytearray()
    for tag, typ, count, value in sorted(fields):
        pointer = value.ljust(4, b"\0") if len(value) <= 4 else struct.pack(endian + "L", payload_offset + len(payload))
        entries.append(struct.pack(endian + "HHL", tag, typ, count) + pointer)
        if len(value) > 4:
            payload.extend(value)
    return ((b"II" if endian == "<" else b"MM") + struct.pack(endian + "HLH", 42, 8, len(fields))
            + b"".join(entries) + bytes(4) + payload)


def ascii_field(tag, value):
    blob = value.encode("ascii") + b"\0"
    return tag, 2, len(blob), blob


def native_nikon(path, *, firmware="Ver.1.10", serial="BODY-123", endian="<", standard_serial=None, compression=3):
    note_endian = ">" if endian == "<" else "<"
    note_fields = [ascii_field(0x1d, serial), (1, 7, 4, b"0234")]
    if compression is not None:
        note_fields.append((0x93, 3, 1, struct.pack(note_endian + "H", compression)))
    note = b"Nikon\0\x02\x11\0\0" + tiff_bytes(note_fields, endian=note_endian)
    # Build EXIF separately, relocate its payload into the outer TIFF container.
    fields = [(37500, 7, len(note), note)]
    if standard_serial is not None:
        fields.append(ascii_field(0xa431, standard_serial))
    exif = bytearray(tiff_bytes(fields, endian=endian))
    root_fields = [ascii_field(271, "NIKON CORPORATION"), ascii_field(272, "NIKON D750"),
                   ascii_field(305, firmware), (34665, 4, 1, bytes(4))]
    root = bytearray(tiff_bytes(root_fields, endian=endian))
    # Preserve the EXIF fixture's header too: its IFD and value pointers use
    # the parent TIFF origin after relocating by the appended header offset.
    shift = len(root)
    count = struct.unpack_from(endian + "H", exif, 8)[0]
    for index in range(count):
        pos = 10 + 12 * index
        if struct.unpack_from(endian + "L", exif, pos + 4)[0] > 4:
            old = struct.unpack_from(endian + "L", exif, pos + 8)[0]
            struct.pack_into(endian + "L", exif, pos + 8, old + shift)
    for index in range(len(root_fields)):
        pos = 10 + 12 * index
        if struct.unpack_from(endian + "H", root, pos)[0] == 34665:
            struct.pack_into(endian + "L", root, pos + 8, shift + 8)
    path.write_bytes(root + exif)


class CaptureIdentityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        env = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "store"))
        env.start()
        self.addCleanup(env.stop)

    def test_native_nikon_mixed_byte_orders_and_makernote_version_is_not_firmware(self):
        path = self.root / "camera.nef"
        for endian in ("<", ">"):
            native_nikon(path, endian=endian)
            identity = read(path)
            self.assertEqual(identity["body_serial"], "BODY-123")
            self.assertEqual(identity["firmware"], "1.10")
            self.assertEqual(identity["identity_fields"]["body_serial"]["status"], "verified")
            self.assertEqual(readout.match({"readout_contract": {"firmware": "1.10"}}, identity)[0], "matched")
            self.assertEqual(readout.match({"readout_contract": {"firmware": "1.11"}}, identity),
                             ("mismatch", "file-firmware-mismatch"))
            native_nikon(path, endian=endian, firmware="Adobe Camera Raw 1.10")
            self.assertIsNone(read(path)["firmware"])

    def test_serial_conflict_not_silently_prioritized(self):
        path = self.root / "camera.nef"
        native_nikon(path, standard_serial="OTHER-BODY")
        identity = read(path)
        self.assertIsNone(identity["body_serial"])
        self.assertEqual(identity["identity_fields"]["body_serial"]["status"], "conflict")
        self.assertEqual(readout.match({"readout_contract": {"body_serial": "BODY-123"}}, identity)[0], "unverified")

    def test_nikon_lossless_requires_specific_maker_declaration_not_nef_suffix(self):
        path = self.root / "camera.nef"
        tags = {259: [34713]}
        for declaration, expected in ((3, True), (1, None), (13, None), (14, None), (None, None)):
            native_nikon(path, compression=declaration)
            self.assertIs(readout._native_compression(path, tags, "NIKON CORPORATION")[1], expected)

    def test_native_sony_requires_exact_model_software_and_never_converter_dng(self):
        path = self.root / "camera.arw"
        fields = [ascii_field(271, "SONY"), ascii_field(272, "ILCE-7M5")]
        for software, expected in (("ILCE-7M5 v2.00", "2.00"), ("ILCE-7M4 v2.00", None),
                                   ("Adobe 2.00", None), ("v2.00", None)):
            path.write_bytes(tiff_bytes(fields + [ascii_field(305, software)]))
            self.assertEqual(read(path)["firmware"], expected)
        path.write_bytes(tiff_bytes(fields + [ascii_field(305, "ILCE-7M5 v2.00"), (50706, 1, 4, bytes([1, 4, 0, 0]))]))
        self.assertIsNone(read(path)["firmware"])

    def dng(self, serial):
        path = self.root / "capture.dng"
        write_noise_dng(path, file_profile=False)
        value = serial.encode() + b"\0"
        edit_ifd(path, {50735: (2, len(value), value)})
        return path

    def test_file_evidence_to_matching_prior_and_analyze_and_other_body(self):
        profile = collect_profile()
        profile["readout_contract"] = {"version": 1, "body_serial": "DNG-BODY-1"}
        path = self.root / "profile.json"
        path.write_text(json.dumps(profile))
        imported = calibration.import_calibration(path)
        raw = self.dng("DNG-BODY-1")
        bundle = load_raw(raw, scene_half_size=True)
        candidate = calibration.matching_prior(bundle.shot_make, bundle.shot_model, shutter=bundle.shot_shutter,
                                              iso=bundle.shot_iso, readout=bundle.capture_readout)
        self.assertEqual(candidate["calibration_id"], imported["id"])
        self.assertEqual(candidate["readout_match_status"], "matched")
        self.assertTrue(priors.prior_usability(candidate)[0])
        result, _, _ = analyze(bundle, 4)
        self.assertEqual(result.noise_model.status, "valid")
        diagnostics = calibration.calibration_diagnostics(bundle.shot_make, bundle.shot_model,
            bundle.shot_shutter, bundle.shot_iso, bundle.capture_readout, analysis=result)
        self.assertNotIn("DNG-BODY-1", json.dumps(diagnostics))
        self.assertEqual(bundle.capture_readout["body_serial"], "DNG-BODY-1")
        bundle = load_raw(self.dng("DNG-BODY-2"), scene_half_size=True)
        candidate = calibration.matching_prior(bundle.shot_make, bundle.shot_model, shutter=bundle.shot_shutter,
                                              iso=bundle.shot_iso, readout=bundle.capture_readout)
        self.assertEqual(candidate["readout_match_status"], "mismatch")
        self.assertFalse(priors.prior_usability(candidate)[0])

    def test_missing_firmware_stays_unverified_not_copied_from_profile(self):
        bundle = load_raw(self.dng("DNG-BODY-1"), scene_half_size=True)
        self.assertIsNone(bundle.capture_readout["firmware"])
        self.assertEqual(readout.match({"readout_contract": {"firmware": "1.00"}}, bundle.capture_readout),
                         ("unverified", "file-firmware-unavailable"))
        with self.assertRaises(ValueError):
            readout.with_capture_declaration(bundle.capture_readout, {"version": 1, "firmware": "1.00"})

    def test_invalid_type_bounds_duplicate_and_truncated_metadata(self):
        path = self.root / "broken.dng"
        for fields in ([(50735, 7, 5, b"1234\0")],
                       [ascii_field(50735, "ONE"), ascii_field(50735, "TWO")],
                       [(50735, 2, 0xffffffff, bytes(4))]):
            path.write_bytes(tiff_bytes(fields))
            self.assertEqual(read(path)["identity_fields"]["body_serial"]["status"], "invalid")
            self.assertIsNone(read(path)["body_serial"])
        path.write_bytes(b"II*\0\x08\0\0\0\xff\xff")
        self.assertEqual(read(path)["identity_metadata_status"], "unavailable")

    def test_public_report_redacts_serial_without_mutating_matching_evidence(self):
        source = read(self.dng("PRIVATE-BODY"))
        public = public_capture(source)
        self.assertNotIn("PRIVATE-BODY", json.dumps(public))
        self.assertEqual(source["body_serial"], "PRIVATE-BODY")
        self.assertTrue(public["body_serial_redacted"])


if __name__ == "__main__":
    unittest.main()
