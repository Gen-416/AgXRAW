# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded file identity adapters, independent of a calibration's claims.

EXIF BodySerialNumber (0xa431), DNG CameraSerialNumber (0xc62f) and Nikon
type-2 MakerNote SerialNumber (0x001d) are separate observations. Nikon
MakerNoteVersion (0x0001) is *not* camera firmware. Native Nikon/Sony camera
Software strings are accepted only under their explicit manufacturer syntax;
converted DNG and generic editing-software versions never establish firmware.
See docs/reviews/PR177_PIPELINE_FIXES_2026-10-10.zh-CN.md for sources and coverage.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
from pathlib import Path
import re
import struct

from .source_metadata import cached_source_metadata


IDENTITY_ADAPTER_VERSION = 1
_TEXT_LIMIT = 160
_NOTE_LIMIT = 2 * 1024 * 1024


class _Tiff:
    def __init__(self, stream):
        self.stream = stream
        stream.seek(0, 2)
        self.size = stream.tell()
        stream.seek(0)
        header = stream.read(8)
        if len(header) != 8 or header[:2] not in (b"II", b"MM"):
            raise ValueError("not a classic TIFF identity container")
        self.endian = "<" if header[:2] == b"II" else ">"
        if struct.unpack_from(self.endian + "H", header, 2)[0] != 42:
            raise ValueError("unsupported TIFF identity container")
        self.first = struct.unpack_from(self.endian + "L", header, 4)[0]

    def take(self, offset, count):
        if offset < 0 or count < 0 or offset > self.size or count > self.size - offset:
            raise ValueError("identity metadata outside container")
        self.stream.seek(offset)
        value = self.stream.read(count)
        if len(value) != count:
            raise ValueError("truncated identity metadata")
        return value

    def ifd(self, offset):
        count, = struct.unpack(self.endian + "H", self.take(offset, 2))
        if count > 512:
            raise ValueError("identity IFD entry limit")
        data = self.take(offset + 2, count * 12 + 4)
        entries = {}
        for index in range(count):
            tag, typ, length = struct.unpack_from(self.endian + "HHL", data, index * 12)
            entries.setdefault(tag, []).append((typ, length, data[index * 12 + 8:index * 12 + 12]))
        return entries

    def blob(self, entries, tag, *, typ, limit):
        values = entries.get(tag, [])
        if not values:
            return None
        if len(values) != 1:
            raise ValueError("duplicate identity tag")
        kind, count, inline = values[0]
        if kind != typ or not 0 < count <= limit:
            raise ValueError("invalid identity tag type or size")
        return inline[:count] if count <= 4 else self.take(struct.unpack(self.endian + "L", inline)[0], count)

    def text(self, entries, tag):
        blob = self.blob(entries, tag, typ=2, limit=_TEXT_LIMIT + 1)
        if blob is None:
            return None
        text = blob.rstrip(b"\0").decode("ascii", "strict").strip()
        if not text or len(text) > _TEXT_LIMIT or any(ord(char) < 32 or ord(char) == 127 for char in text):
            raise ValueError("invalid identity text")
        return text

    def pointer(self, entries, tag):
        values = entries.get(tag, [])
        if not values:
            return None
        if len(values) != 1 or values[0][:2] != (4, 1):
            raise ValueError("invalid identity IFD pointer")
        return struct.unpack(self.endian + "L", values[0][2])[0]


@cached_source_metadata
def read(path: Path) -> dict:
    """Return values plus independent validity/provenance; never fill from priors."""
    result = {"identity_adapter_version": IDENTITY_ADAPTER_VERSION,
              "body_serial": None, "firmware": None, "identity_fields": {}}
    observations = {key: [] for key in ("body_serial", "firmware")}
    invalid = {key: [] for key in observations}

    def observe(reader, entries, tag, field, source):
        try:
            value = reader.text(entries, tag)
            if value is not None:
                observations[field].append((value, source))
        except (ValueError, UnicodeError, struct.error) as error:
            invalid[field].append({"source": source, "reason": str(error)})

    try:
        with Path(path).open("rb") as stream:
            tiff = _Tiff(stream)
            root = tiff.ifd(tiff.first)
            observe(tiff, root, 0xc62f, "body_serial", "DNG:IFD0.CameraSerialNumber")
            exif = tiff.ifd(offset) if (offset := tiff.pointer(root, 34665)) else {}
            observe(tiff, exif, 0xa431, "body_serial", "EXIF:BodySerialNumber")
            def identity_context(tag):
                try:
                    return tiff.text(root, tag) or ""
                except (ValueError, UnicodeError, struct.error):
                    return ""
            make = identity_context(271).upper()
            model = identity_context(272)
            software = identity_context(305)
            native = 50706 not in root
            if make in ("NIKON", "NIKON CORPORATION"):
                # Type 2 has its own TIFF origin and byte order, ten bytes
                # into the note. Do not treat its version bytes as firmware.
                try:
                    note = tiff.blob(exif, 37500, typ=7, limit=_NOTE_LIMIT)
                    if note and note.startswith(b"Nikon\0\x02"):
                        maker = _Tiff(io.BytesIO(note[10:]))
                        fields = maker.ifd(maker.first)
                        observe(maker, fields, 0x001d, "body_serial", "Nikon:type2.SerialNumber")
                        compression = fields.get(0x0093, [])
                        if (native and Path(path).suffix.lower() == ".nef" and len(compression) == 1
                                and compression[0][:2] == (3, 1)):
                            result["nikon_nef_compression"] = struct.unpack(maker.endian + "H", compression[0][2][:2])[0]
                            result["nikon_nef_compression_source"] = "Nikon:type2.NEFCompression"
                        if native and Path(path).suffix.lower() == ".nef" and model.upper().startswith("NIKON "):
                            found = re.fullmatch(r"Ver\.(\d+\.\d{2})", software)
                            if found:
                                observations["firmware"].append((found[1], "Nikon:native-NEF.Software.Ver"))
                except (ValueError, UnicodeError, struct.error) as error:
                    invalid["body_serial"].append({"source": "Nikon:type2.SerialNumber", "reason": str(error)})
            elif (make in ("SONY", "SONY CORPORATION") and native
                  and Path(path).suffix.lower() == ".arw" and model):
                found = re.fullmatch(re.escape(model) + r" v(\d+\.\d{2})", software)
                if found:
                    observations["firmware"].append((found[1], "Sony:native-ARW.Software.model-version"))
    except (OSError, ValueError, UnicodeError, struct.error) as error:
        result["identity_metadata_status"] = "unavailable"
        result["identity_metadata_reason"] = type(error).__name__
    for field, values in observations.items():
        distinct = {" ".join(value.split()).casefold() for value, _ in values}
        state = ("conflict" if len(distinct) > 1 else "invalid" if invalid[field]
                 else "verified" if values else "missing")
        result["identity_fields"][field] = {
            "status": state, "sources": [source for _, source in values],
            "invalid_observations": invalid[field],
        }
        if state == "verified":
            result[field] = values[0][0]
            result[field + "_source"] = "+".join(source for _, source in values)
    return result


def public_capture(capture: dict | None) -> dict:
    """Publish matching evidence without publishing a user's body serial."""
    result = deepcopy(capture or {})
    serial = result.pop("body_serial", None)
    if serial is not None:
        result["body_serial_sha256"] = hashlib.sha256(str(serial).encode()).hexdigest()
        result["body_serial_redacted"] = True
    return result
