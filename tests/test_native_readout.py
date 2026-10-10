# SPDX-License-Identifier: GPL-3.0-or-later
"""Native container identity does not establish sensor acquisition identity."""
from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

from dngscan import readout


def native_tiff(path, *, make="SONY", compression=32766, count=80, bits=14, frames=1):
    """Metadata-only fixture, not a valid proprietary compressed RAW stream."""
    data = bytearray(b"II*\0" + bytes(4))
    data.extend(bytes(max(count, 0)))
    def ifd(fields):
        offset = len(data)
        data.extend(bytes(2 + 12*len(fields) + 4))
        struct.pack_into("<H", data, offset, len(fields))
        for index, (tag, (kind, values)) in enumerate(sorted(fields.items())):
            payload = (values.encode("ascii") + b"\0" if kind == 2
                       else struct.pack("<" + {3:"H", 4:"L"}[kind]*len(values), *values))
            number = len(payload) if kind == 2 else len(values)
            reference = payload.ljust(4, b"\0") if len(payload) <= 4 else struct.pack("<L", len(data))
            struct.pack_into("<HHL4s", data, offset+2+12*index, tag, kind, number, reference)
            if len(payload) > 4:
                data.extend(payload)
        return offset
    raw = [ifd({254:(4,[0]), 256:(4,[12]), 257:(4,[10]), 258:(3,[bits]),
                259:(3,[compression]), 262:(3,[32803]), 273:(4,[8]), 277:(3,[1]),
                279:(4,[count]), 50720:(4,[10,8])}) for _ in range(frames)]
    root = ifd({254:(4,[1]), 256:(4,[4]), 257:(4,[3]), 258:(3,[8]), 259:(3,[7]),
                262:(3,[2]), 271:(2,make), 330:(4,raw)})
    struct.pack_into("<L", data, 4, root)
    path.write_bytes(data)


class NativeReadoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "metadata.arw"

    def test_arw6_codec_and_native_geometry_do_not_borrow_preview_or_adc(self):
        native_tiff(self.path)
        capture = readout.read(self.path)
        self.assertEqual(capture["source"], "native-main-raw-ifd")
        self.assertEqual(capture["raw_geometry"], [12,10])
        self.assertEqual(capture["default_crop"], [10.,8.])
        self.assertEqual(capture["sample_bits"], 14)
        self.assertEqual(capture["codec_id"], "sony-arw6-llvc3")
        self.assertFalse(capture["storage_lossless"])
        for key in ("shutter", "sensor_bits", "sensor_binning", "readout_id", "capture_kind"):
            self.assertIsNone(capture[key])
        self.assertEqual(readout.match({}, capture)[1],
                         "lossy-raw-storage-not-supported-by-external-prior")

    def test_sony_32767_requires_pinned_dispatch_size_not_code_alone(self):
        for count, codec, lossless in ((120, "sony-arw2", False),
                                        (240, "sony-unpacked", True), (80, None, None)):
            with self.subTest(count=count):
                native_tiff(self.path, compression=32767, count=count)
                capture = readout.read(self.path)
                self.assertEqual(capture["codec_id"], codec)
                self.assertIs(capture["storage_lossless"], lossless)

    def test_nikon_container_code_does_not_guess_lossless_or_he(self):
        native_tiff(self.path, make="NIKON CORPORATION", compression=34713)
        capture = readout.read(self.path)
        self.assertEqual(capture["codec_id"], "nikon-nef-compression-34713")
        self.assertIsNone(capture["storage_lossless"])
        self.assertEqual(readout.match({}, capture),
                         ("unverified", "file-storage-lossless-unverified"))
        self.assertIsNone(capture["readout_id"])

    def test_wrong_manufacturer_cannot_borrow_sony_codec_semantics(self):
        native_tiff(self.path, make="NOT SONY")
        capture = readout.read(self.path)
        self.assertIsNone(capture["codec_id"])
        self.assertIsNone(capture["storage_lossless"])

    def test_multiple_native_main_raw_frames_remain_unqualified(self):
        native_tiff(self.path, frames=2)
        capture = readout.read(self.path)
        self.assertEqual(capture["source"], "unavailable")
        self.assertIsNone(capture["raw_geometry"])
        self.assertEqual(readout.match({}, capture),
                         ("unverified", "file-storage-lossless-unverified"))


if __name__ == "__main__":
    unittest.main()
