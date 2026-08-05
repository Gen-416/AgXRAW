# SPDX-License-Identifier: GPL-3.0-or-later
"""Shot-info reader: TIFF/DNG path and Fujifilm RAF path."""
from __future__ import annotations

import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dngscan import metadata


def _tiff_with_shot_info(make: bytes, model: bytes, iso: int) -> bytes:
    """Minimal little-endian TIFF: IFD0 with Make/Model/ISO (ISO inline in IFD0).

    Make/Model must be >3 bytes so their values go through the offset path."""
    # Header (8 bytes) + IFD0 right after.
    entry_count = 3
    ifd0_off = 8
    data_off = ifd0_off + 2 + entry_count * 12 + 4  # after entry table + next-IFD ptr

    make_field = make + b"\x00"
    model_field = model + b"\x00"

    entries = b""
    # Make (tag 271, ASCII)
    entries += struct.pack("<HHL", 271, 2, len(make_field)) + struct.pack("<L", data_off)
    # Model (tag 272, ASCII)
    entries += struct.pack("<HHL", 272, 2, len(model_field)) + struct.pack(
        "<L", data_off + len(make_field)
    )
    # ISO (tag 34855, SHORT, inline)
    entries += struct.pack("<HHL", 34855, 3, 1) + struct.pack("<HH", iso, 0)

    out = b"II" + struct.pack("<H", 42) + struct.pack("<L", ifd0_off)
    out += struct.pack("<H", entry_count) + entries + struct.pack("<L", 0)
    out += make_field + model_field
    return out


def _tiff_with_dng_version() -> bytes:
    """Minimal little-endian TIFF whose IFD0 carries only DNGVersion (tag 50706)."""
    ifd0_off = 8
    # DNGVersion: 4 BYTEs, inline (1.4.0.0)
    entry = struct.pack("<HHL", 50706, 1, 4) + bytes((1, 4, 0, 0))
    out = b"II" + struct.pack("<H", 42) + struct.pack("<L", ifd0_off)
    out += struct.pack("<H", 1) + entry + struct.pack("<L", 0)
    return out


def _jpeg_with_exif(tiff: bytes) -> bytes:
    app1_payload = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(app1_payload) + 2) + app1_payload
    return b"\xff\xd8" + app1 + b"\xff\xd9"


def _raf_bytes(model: bytes, jpeg: bytes) -> bytes:
    header = bytearray(0x64)
    header[0:16] = b"FUJIFILMCCD-RAW "
    header[16:20] = b"0201"
    header[20:28] = b"FF129502"
    header[0x1C : 0x1C + len(model)] = model
    jpeg_off = len(header)
    header[0x54:0x58] = struct.pack(">L", jpeg_off)
    header[0x58:0x5C] = struct.pack(">L", len(jpeg))
    return bytes(header) + jpeg


class MetadataTest(unittest.TestCase):
    def test_tiff_dng_path(self) -> None:
        tiff = _tiff_with_shot_info(b"SIGMA", b"fp L", 640)
        with TemporaryDirectory() as td:
            p = Path(td) / "shot.dng"
            p.write_bytes(tiff)
            info = metadata.read_dng_shot_info(p)
        self.assertEqual(info.make, "SIGMA")
        self.assertEqual(info.model, "fp L")
        self.assertEqual(info.iso, 640)

    def test_tiff_nef_path(self) -> None:
        tiff = _tiff_with_shot_info(b"NIKON CORPORATION", b"Z 6_2", 3200)
        with TemporaryDirectory() as td:
            p = Path(td) / "shot.nef"
            p.write_bytes(tiff)
            info = metadata.read_dng_shot_info(p)
        self.assertEqual(info.make, "NIKON CORPORATION")
        self.assertEqual(info.model, "Z 6_2")
        self.assertEqual(info.iso, 3200)

    def test_raf_path(self) -> None:
        exif_tiff = _tiff_with_shot_info(b"FUJIFILM", b"X-T5", 1250)
        raf = _raf_bytes(b"X-T5", _jpeg_with_exif(exif_tiff))
        with TemporaryDirectory() as td:
            p = Path(td) / "shot.raf"
            p.write_bytes(raf)
            info = metadata.read_dng_shot_info(p)
        self.assertEqual(info.make, "FUJIFILM")
        self.assertEqual(info.model, "X-T5")
        self.assertEqual(info.iso, 1250)

    def test_raf_without_exif_still_gets_model(self) -> None:
        # Embedded JPEG missing / truncated: header-derived make+model must survive.
        raf = _raf_bytes(b"X100V", b"\xff\xd8\xff\xd9")
        with TemporaryDirectory() as td:
            p = Path(td) / "shot.raf"
            p.write_bytes(raf)
            info = metadata.read_dng_shot_info(p)
        self.assertEqual(info.make, "FUJIFILM")
        self.assertEqual(info.model, "X100V")
        self.assertIsNone(info.iso)

    def test_dng_container_probe_requires_the_dng_version_tag(self) -> None:
        # A plain TIFF (NEF-style, no DNGVersion) must not read as a DNG container,
        # while adding tag 50706 to IFD0 must.  This is what LibRaw's identify.cpp
        # records as dng_version — the rung-2 cmatrix adoption gate keys on it.
        plain = _tiff_with_shot_info(b"NIKON CORPORATION", b"Z 6_2", 3200)
        dng = _tiff_with_dng_version()
        with TemporaryDirectory() as td:
            p_plain = Path(td) / "shot.nef"
            p_plain.write_bytes(plain)
            p_dng = Path(td) / "shot.dng"
            p_dng.write_bytes(dng)
            p_junk = Path(td) / "junk.bin"
            p_junk.write_bytes(b"not a raw file at all")
            self.assertFalse(metadata.is_dng_container(p_plain))
            self.assertTrue(metadata.is_dng_container(p_dng))
            self.assertFalse(metadata.is_dng_container(p_junk))
            self.assertFalse(metadata.is_dng_container(Path(td) / "missing.dng"))

    def test_garbage_file_returns_empty(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "junk.raf"
            p.write_bytes(b"not a raw file at all")
            info = metadata.read_dng_shot_info(p)
        self.assertIsNone(info.make)
        self.assertIsNone(info.model)
        self.assertIsNone(info.iso)


if __name__ == "__main__":
    unittest.main()
