# SPDX-License-Identifier: GPL-3.0-or-later
"""File-backed capture constraints, distinct from rendering crops and RAW DN scale.

Unknown ADC precision, binning and capture modes stay unknown. In particular,
TIFF BitsPerSample describes stored samples, not the sensor's ADC, and a JPEG
output size is not a sensor raster size.
"""
from __future__ import annotations

import io
import math
from pathlib import Path
import re
import struct

from . import metadata as md
from .source_metadata import cached_source_metadata


READOUT_VERSION = 1
# Exact-model capability, not a proprietary file-tag or a complete readout ID.
# SIGMA's fp specifications name only an electronic shutter. This says nothing
# about ADC precision, still/video timing, cropping or binning.
_FP_SHUTTER_SOURCE = "https://www.sigma-global.com/en/cameras/fp/?local=table&tab=support&table_id=11934"
_PAIR_FIELDS = {"raw_geometry", "active_geometry", "default_crop", "libraw_raw_geometry", "sensor_binning"}
_BIT_FIELDS = {"sample_bits", "sensor_bits"}
_TEXT_FIELDS = {"readout_id", "capture_kind"}
_FIELDS = _PAIR_FIELDS | _BIT_FIELDS | _TEXT_FIELDS | {"storage_lossless"}


def _pair(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain two positive integers")
    result = []
    for item in value:
        number = float(item)
        if isinstance(item, bool) or not math.isfinite(number) or number <= 0 or number != int(number):
            raise ValueError(f"{label} must contain two positive integers")
        result.append(int(number))
    return result


def _crop_pair(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain two positive numbers")
    result = [float(number) for number in value]
    if any(isinstance(number, bool) for number in value) or any(not math.isfinite(number) or number <= 0 for number in result):
        raise ValueError(f"{label} must contain two positive numbers")
    return result


def measurement_fields(item: dict) -> dict:
    """Validate explicit constraints; never interpret overloaded mode names."""
    supplied = item.get("readout_contract")
    undeclared = supplied is None
    if undeclared:
        supplied = {}
    if not isinstance(supplied, dict) or any(k not in _FIELDS | {"version"} for k in supplied):
        raise ValueError("unsupported readout_contract field")
    if not undeclared and (type(supplied.get("version")) is not int
                           or supplied["version"] != READOUT_VERSION):
        raise ValueError("readout_contract requires version 1")
    constraints = {}
    for key, value in supplied.items():
        if key == "version":
            continue
        if key in _PAIR_FIELDS:
            constraints[key] = _crop_pair(value, key) if key == "default_crop" else _pair(value, key)
        elif key in _BIT_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 64:
                raise ValueError(f"{key} must be an integer in 1..64")
            constraints[key] = value
        elif key == "storage_lossless":
            if not isinstance(value, bool):
                raise ValueError("storage_lossless must be true or false")
            constraints[key] = value
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be nonempty text")
        else:
            constraints[key] = value.strip()
    issues, informational = [], {}
    if (item.get("acquisition_contract") or {}).get("raw_geometry_complete") is False:
        issues.append("measurement-raw-geometry-incomplete")
    compression = item.get("compression")
    if compression:
        text = " ".join(compression.lower().split())
        # These declarations establish preservation of stored codes; codec
        # differences between lossless JPEG and uncompressed TIFF do not
        # themselves establish a different sensor readout.
        if text in {"lossless", "lossless compression", "uncompressed", "none", "无损压缩", "无压缩"}:
            if constraints.get("storage_lossless") is False:
                raise ValueError("compression and storage_lossless disagree")
            constraints["storage_lossless"] = True
        else:
            issues.append("measurement-compression-declaration-unverified")
    geometry = item.get("geometry")
    if geometry and any(v not in (None, "") for v in geometry):
        pair = _pair(geometry, "geometry")
        domain = (item.get("acquisition_contract") or {}).get("geometry_domain")
        key = {"raw-ifd": "raw_geometry", "active-area": "active_geometry",
               "default-crop": "default_crop", "libraw-raw-mosaic": "libraw_raw_geometry"}.get(domain)
        if key:
            if key in constraints and constraints[key] != pair:
                raise ValueError("geometry disagrees with readout_contract")
            constraints[key] = pair
        elif domain == "camera-jpeg-output":
            # JPTC records an operator-declared JPEG output size separately
            # from RawSize. It is retained, never compared to visible CFA,
            # RAW IFD or DefaultCrop as though those layers were equivalent.
            informational["camera_jpeg_geometry"] = pair
            if not any(key in constraints for key in ("raw_geometry", "libraw_raw_geometry", "active_geometry")):
                issues.append("measurement-raw-geometry-unavailable")
        else:
            issues.append("measurement-geometry-domain-unavailable")
    return {"readout_contract": {"version": READOUT_VERSION, **constraints},
            "readout_declaration_issues": issues, "readout_informational": informational}


def match(entry: dict, capture: dict | None) -> tuple[str, str]:
    """Only declared constraints can match; missing evidence is not equality."""
    capture = capture or {}
    if capture.get("storage_lossless") is False:
        return "mismatch", "lossy-raw-storage-not-supported-by-external-prior"
    if capture.get("source") in ("dng-main-raw-ifd", "native-main-raw-ifd", "unavailable") and capture.get("storage_lossless") is None:
        return "unverified", "file-storage-lossless-unverified"
    issues = entry.get("readout_declaration_issues") or []
    if issues:
        return "unverified", issues[0]
    constraints = entry.get("readout_contract") or {}
    fields = [(key, value) for key, value in constraints.items() if key != "version"]
    if not fields:
        return "not-declared", "sub-readout-not-declared"
    for key, expected in fields:
        actual = capture.get(key)
        if actual is None:
            return "unverified", f"file-{key.replace('_', '-')}-unavailable"
        if actual != expected:
            return "mismatch", f"file-{key.replace('_', '-')}-mismatch"
    return "matched", "declared-readout-constraints-matched"


def collect_fields(header: dict) -> dict:
    """JPTC RawSize is LibRaw's full mosaic, not ImageWidth/Height.

    JPTC/2 and JPTC-DARK/1 share this acquisition declaration. Contract
    verified against JiangtherapeeTesterView 57567edf, entryCsv.mjs,
    darkCsv.mjs and binding.cc raw_width/raw_height. The JPEG dimensions
    are operator declarations and remain informational.
    """
    raw_size = header.get("RawSize")
    contract = {"geometry_domain": "camera-jpeg-output"}
    if not raw_size:
        return {"acquisition_contract_fields": contract}
    found = re.fullmatch(r"\s*(\d+)\s*[x×]\s*(\d+)\s*", str(raw_size))
    if found is None:
        raise ValueError("JPTC RawSize must contain full mosaic width x height")
    size = _pair([int(found[1]), int(found[2])], "RawSize")
    return {"acquisition_contract_fields": contract,
            "readout_contract": {"version": READOUT_VERSION, "libraw_raw_geometry": size}}


def _jpeg_process(source, offset, byte_count):
    """Inspect only the declared JPEG block's bounded header, not its entropy."""
    if (not isinstance(offset, int) or isinstance(offset, bool) or offset < 0
            or not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 2):
        return None
    try:
        source.seek(0, 2)
        file_size = source.tell()
        if offset > file_size or byte_count > file_size - offset:
            return None
        source.seek(offset)
    except (OSError, ValueError, OverflowError):
        return None
    consumed = 0
    limit = min(byte_count, 65536)

    def take(size):
        nonlocal consumed
        if size > limit - consumed:
            return b""
        value = source.read(size)
        consumed += len(value)
        return value

    if take(2) != b"\xff\xd8":
        return None
    lossless_frame = False
    frame_components = None
    while consumed < limit:
        if take(1) != b"\xff":
            return None
        marker = take(1)
        while marker == b"\xff":
            marker = take(1)
        if len(marker) != 1 or marker[0] in (0, 0xd9):
            return None
        length = take(2)
        if len(length) != 2:
            return None
        size = struct.unpack(">H", length)[0]
        if size < 2 or size - 2 > limit - consumed:
            return None
        if marker[0] in (0xc0, 0xc1, 0xc2, 0xc3):
            frame = take(size - 2)
            if len(frame) != size - 2 or len(frame) < 6 or size != 8 + 3 * frame[5]:
                return None
            if not 1 <= frame[0] <= 16 or not 1 <= frame[5] <= 4 or min(struct.unpack(">HH", frame[1:5])) <= 0:
                return None
            if marker[0] != 0xc3:
                return "lossy-jpeg-dct"
            lossless_frame = True
            frame_components = [frame[6 + 3*index] for index in range(frame[5])]
            if len(set(frame_components)) != len(frame_components):
                return None
        elif marker[0] == 0xda:
            scan = take(size - 2)
            if (not lossless_frame or len(scan) != size - 2 or len(scan) < 6
                    or size != 6 + 2 * scan[0] or not 1 <= scan[0] <= 4
                    or not 1 <= scan[-3] <= 7 or scan[-2] != 0 or scan[-1] >> 4):
                return None
            if sorted(scan[1 + 2*index] for index in range(scan[0])) != sorted(frame_components):
                # A partial-component first scan cannot establish the point
                # transforms of later scans without walking entropy data.
                return None
            # ITU T.81 A.4/K.7: a lossless JPEG point transform discards
            # low bits before predictive coding. SOF3 alone is insufficient.
            return "lossless-jpeg-sof3-pt0" if scan[-1] == 0 else "lossy-jpeg-point-transform"
        else:
            source.seek(size - 2, 1)
            consumed += size - 2
    return None


def _compression(path, tags):
    values = tags.get(259) or []
    if len(values) != 1:
        return None, None, "compression-unavailable"
    code = int(values[0])
    if code == 1:
        return code, True, "uncompressed-tiff"
    if code in (8, 32946):
        return code, True, "lossless-deflate"
    if code == 34892:
        return code, False, "dng-lossy-jpeg"
    if code == 7:
        tiled = 324 in tags or 325 in tags
        stripped = 273 in tags or 279 in tags
        if tiled and stripped:
            return code, None, "jpeg-process-unavailable"
        offsets, byte_counts = ((tags.get(324), tags.get(325)) if tiled
                                else (tags.get(273), tags.get(279)))
        if (not isinstance(offsets, (list, tuple)) or not isinstance(byte_counts, (list, tuple))
                or not offsets or len(offsets) > 4096 or len(offsets) != len(byte_counts)
                or any(not isinstance(value, int) or isinstance(value, bool) or value < minimum
                       for values, minimum in ((offsets, 0), (byte_counts, 2)) for value in values)):
            return code, None, "jpeg-process-unavailable"
        try:
            with path.open("rb") as source:
                source.seek(0, 2)
                file_size = source.tell()
                if any(offset > file_size or count > file_size - offset
                       for offset, count in zip(offsets, byte_counts)):
                    return code, None, "jpeg-process-unavailable"
                processes = {_jpeg_process(source, offset, count)
                             for offset, count in zip(offsets, byte_counts)}
        except (OSError, ValueError, OverflowError):
            return code, None, "jpeg-process-unavailable"
        if processes == {"lossless-jpeg-sof3-pt0"}:
            return code, True, "lossless-jpeg-sof3-pt0"
        if "lossy-jpeg-dct" in processes or "lossy-jpeg-point-transform" in processes:
            return code, False, "lossy-jpeg-process"
        return code, None, "jpeg-process-unavailable"
    return code, None, "compression-process-unverified"


def _native_cfa_tags(path):
    """Read a native TIFF's main CFA IFD, independently of DNG semantics.

    Preview fields cannot supply RAW geometry or codec declarations. Select the
    a unique main CFA frame; multiple native RAW frames, unsupported containers,
    LinearRAW and proprietary MakerNote readout modes stay unknown.
    """
    wanted = {254, 256, 257, 258, 259, 262, 273, 277, 279, 324, 325, 330, 50720}
    with path.open("rb") as source:
        head = source.read(8)
        if len(head) != 8 or head[:2] not in (b"II", b"MM"):
            return {}, None
        endian = "<" if head[:2] == b"II" else ">"
        if struct.unpack(endian+"H", head[2:4])[0] != 42:
            return {}, None
        first, = struct.unpack(endian+"L", head[4:])
        root = md._read_ifd_entries(source, first, endian)
        if any(tag == md.TAG_DNG_VERSION for tag, *_ in root):
            return {}, None
        make = next((values[0] for tag, typ, count, data in root
                     if tag == md.TAG_MAKE and typ == 2 and 0 < count <= 128
                     for values in [md._entry_values(source, typ, count, data, endian)]
                     if values), "")
        visited = set()
        candidates = []

        def visit(offset, entries=None):
            if offset in visited or len(visited) >= 64:
                return
            visited.add(offset)
            entries = entries if entries is not None else md._read_ifd_entries(source, offset, endian)
            tags = {}
            for tag, typ, count, data in entries:
                if tag in wanted:
                    if count * md._TYPE_SIZES.get(typ, 1) > 65536:
                        raise ValueError("native readout metadata too large")
                    tags[tag] = md._entry_values(source, typ, count, data, endian)
            yield tags
            for sub in tags.get(330, [])[:64]:
                yield from visit(int(sub))

        offset = first
        while offset and offset not in visited and len(visited) < 64:
            for tags in visit(offset, root if offset == first else None):
                width, height = (tags.get(256) or [0])[0], (tags.get(257) or [0])[0]
                if ((tags.get(254) or [0])[0] == 0 and (tags.get(262) or [0])[0] == 32803
                        and (tags.get(277) or [1])[0] == 1
                        and 0 < width < 65536 and 0 < height < 65536):
                    candidates.append(tags)
            source.seek(offset)
            count_raw = source.read(2)
            if len(count_raw) != 2:
                break
            count, = struct.unpack(endian+"H", count_raw)
            if count > 4096:
                break
            source.seek(offset+2+12*count)
            next_raw = source.read(4)
            offset = struct.unpack(endian+"L", next_raw)[0] if len(next_raw) == 4 else 0
    return (candidates[0], make) if len(candidates) == 1 else ({}, None)


def _native_compression(path, tags, make):
    """Codec identity is evidence about storage, never ADC/readout identity.

    Dispatch predicates mirror the project's pinned LibRaw tiff.cpp. They do
    not identify Sony's camera-menu Compressed/HQ option or Nikon HE/HE* mode.
    """
    code, lossless, process = _compression(path, tags)
    brand = " ".join(str(make or "").upper().split())
    codec = None
    if brand in ("SONY", "SONY CORPORATION"):
        if code == 32766 and (tags.get(258) or [None])[0] in (12, 14):
            codec, lossless, process = "sony-arw6-llvc3", False, "sony-arw6-quantized-curve"
        elif code == 32767:
            counts = tags.get(279) or []
            size = int(tags[256][0]) * int(tags[257][0])
            if len(counts) == 1 and counts[0] == size:
                codec, lossless, process = "sony-arw2", False, "sony-arw2-lossy"
            elif len(counts) == 1 and counts[0] == 2*size:
                codec, lossless, process = "sony-unpacked", True, "sony-unpacked"
        elif code in (6, 7):
            # Sony also uses Compression=6 for predictive lossless JPEG.
            _, lossless, process = _compression(path, {**tags, 259: [7]})
            codec = "sony-predictive-jpeg" if lossless is True else None
    elif brand in ("NIKON", "NIKON CORPORATION") and code == 34713:
        codec = "nikon-nef-compression-34713"
        # This container code alone does not distinguish legacy lossless,
        # lossy, packed or HE variants. Keep the model unqualified.
    return code, lossless, process, codec


def _fujifilm_shutter(path):
    """MakerNote 0x1050 is independent of the optional embedded lens profile."""
    with path.open("rb") as source:
        head = source.read(100)
        if not head.startswith(md._RAF_MAGIC) or len(head) < 92:
            return None
        offset, count = struct.unpack_from(">LL", head, 84)
        source.seek(offset)
        tiff = md._exif_tiff_from_jpeg(source.read(min(count, 128000)))
    if not tiff:
        return None
    stream = io.BytesIO(tiff)
    endian = "<" if tiff[:2] == b"II" else ">"
    def values(offset, wanted):
        return {tag: md._entry_values(stream, typ, count, data, endian)
                for tag, typ, count, data in md._read_ifd_entries(stream, offset, endian)
                if tag in wanted and count * md._TYPE_SIZES.get(typ, 1) <= 128000}
    root = values(struct.unpack_from(endian+"L", tiff, 4)[0], {34665})
    exif = values(int(root[34665][0]), {37500}) if root.get(34665) else {}
    note = (exif.get(37500) or [b""])[0]
    if not isinstance(note, bytes) or not note.startswith(b"FUJIFILM"):
        return None
    stream, endian = io.BytesIO(note), "<"
    maker = values(struct.unpack_from("<L", note, 8)[0], {0x1050})
    return {0: "mechanical", 1: "electronic", 2: "electronic", 3: "efcs"}.get((maker.get(0x1050) or [None])[0])


@cached_source_metadata
def read(path: Path) -> dict:
    """Acquire supported file declarations without a lens or scene decoder."""
    capture = {"version": READOUT_VERSION, "source": "unavailable", "shutter": None,
               "shutter_source": None, **{field: None for field in _FIELDS}}
    path = Path(path)
    try:
        from .spatial_black import sensor_tags
        tags = sensor_tags(path, {258, 259, 273, 279, 324, 325, 50829, 50719, 50720})
        native_make = None
        if not tags:
            tags, native_make = _native_cfa_tags(path)
        if tags:
            capture["source"] = "native-main-raw-ifd" if native_make is not None else "dng-main-raw-ifd"
            width, height = int(tags[256][0]), int(tags[257][0])
            capture["raw_geometry"] = _pair([width, height], "RAW IFD geometry")
            bits = tags.get(258) or []
            if bits and len(set(bits)) == 1 and 1 <= int(bits[0]) <= 64:
                capture["sample_bits"] = int(bits[0])
            area = tags.get(50829)
            if area and len(area) == 4 and 0 <= area[0] < area[2] <= height and 0 <= area[1] < area[3] <= width:
                capture["active_geometry"] = _pair([area[3]-area[1], area[2]-area[0]], "ActiveArea")
            if tags.get(50720):
                try:
                    capture["default_crop"] = _crop_pair(tags[50720], "DefaultCropSize")
                except (ValueError, TypeError):
                    capture["default_crop_status"] = "invalid-default-crop"
            if native_make is not None:
                code, lossless, process, codec = _native_compression(path, tags, native_make)
                capture.update(codec_id=codec, codec_source="main-raw-ifd+pinned-libraw-dispatch" if codec else None)
            else:
                code, lossless, process = _compression(path, tags)
            capture.update(compression_code=code, storage_lossless=lossless, compression_process=process)
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, struct.error):
        capture["metadata_status"] = "readout-metadata-unreadable"
    # A malformed optional raster field must not suppress independently
    # available shutter evidence or an exact-model manufacturer capability.
    try:
        shutter = _fujifilm_shutter(path)
        if shutter is not None:
            capture.update(shutter=shutter, shutter_source="Fujifilm MakerNote 0x1050")
        shot = md.read_dng_shot_info(path)
        make = " ".join((shot.make or "").upper().split())
        model = " ".join((shot.model or "").upper().split())
        if make in ("SIGMA", "SIGMA CORPORATION") and model in ("FP", "SIGMA FP"):
            capture.update(shutter="electronic", shutter_source="manufacturer-capability:SIGMA-fp",
                           shutter_source_url=_FP_SHUTTER_SOURCE)
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, struct.error):
        capture["shutter_status"] = "shutter-metadata-unreadable"
    return capture
