# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal RAW metadata reader for camera identification and shot metadata.

Reads only what the priors layer needs (Make, Model, ISO, AsShotNeutral) with
targeted seeks — no external dependency, no full-file load. Handles TIFF-based
containers (DNG/NEF/ARW/CR2/...) natively and Fujifilm RAF via its proprietary
header plus the EXIF block of the embedded JPEG preview. Returns None fields
on any parse trouble; callers must treat everything as best-effort.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from pathlib import Path

TAG_MAKE = 271
TAG_MODEL = 272
TAG_EXIF_IFD = 34665
TAG_ISO = 34855
TAG_AS_SHOT_NEUTRAL = 50728
TAG_SUB_IFDS = 330
# DNG 1.7 §5: log2 baseline rendering compensation applied on top of raw data.
# It may be a stable camera baseline (Sigma fp writes 1.0 on the tested frames) or part
# of a per-image rendering recipe (ProRAW can vary it with scene dynamic range). It is
# not the shutter/aperture/ISO measurement and does not imply content normalization.
TAG_BASELINE_EXPOSURE = 50730
# DNG colour calibration: XYZ -> camera matrices measured under two illuminants, plus
# the EXIF LightSource codes naming those illuminants. The pair is what makes an
# arbitrary-CCT white balance a calibrated interpolation instead of a guess.
TAG_COLOR_MATRIX_1 = 50721
TAG_COLOR_MATRIX_2 = 50722
TAG_CALIBRATION_ILLUMINANT_1 = 50778
TAG_CALIBRATION_ILLUMINANT_2 = 50779
# DNG §4: presence of DNGVersion in IFD0 is what makes a TIFF container a DNG.
# LibRaw keys its embedded-cmatrix adoption on the same fact (identify.cpp sets
# dng_version from this tag), so the hot-WB rung-2 gate must test it too.
TAG_DNG_VERSION = 50706

# EXIF LightSource code -> correlated colour temperature (K). Only codes that name a
# concrete illuminant are mapped; anything else leaves the matrix unpaired.
_LIGHT_SOURCE_CCT = {
    1: 5500.0,   # Daylight
    2: 4200.0,   # Fluorescent (nominal)
    3: 2856.0,   # Tungsten
    4: 5500.0,   # Flash (nominal)
    9: 5500.0,   # Fine weather
    10: 6500.0,  # Cloudy
    11: 7500.0,  # Shade
    17: 2856.0,  # Standard light A
    18: 4874.0,  # Standard light B
    19: 6774.0,  # Standard light C
    20: 5503.0,  # D55
    21: 6504.0,  # D65
    22: 7504.0,  # D75
    23: 5003.0,  # D50
    24: 3200.0,  # ISO studio tungsten
}

_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


@dataclass
class DngShotInfo:
    make: str | None = None
    model: str | None = None
    iso: int | None = None
    as_shot_neutral: tuple[float, float, float] | None = None
    # None when the file carries no BaselineExposure, which is not the same as 0.0.
    baseline_exposure: float | None = None


def _read_ifd_entries(fh, offset: int, endian: str) -> list[tuple[int, int, int, bytes]]:
    fh.seek(offset)
    count_raw = fh.read(2)
    if len(count_raw) < 2:
        return []
    (count,) = struct.unpack(endian + "H", count_raw)
    if count > 4096:
        return []
    data = fh.read(count * 12)
    entries = []
    for i in range(count):
        tag, typ, num = struct.unpack(endian + "HHL", data[i * 12 : i * 12 + 8])
        entries.append((tag, typ, num, data[i * 12 + 8 : i * 12 + 12]))
    return entries


def _entry_values(fh, typ: int, num: int, raw: bytes, endian: str) -> list:
    size = _TYPE_SIZES.get(typ)
    if size is None:
        return []
    total = size * num
    if total <= 4:
        buf = raw[:total]
    else:
        (off,) = struct.unpack(endian + "L", raw)
        fh.seek(off)
        buf = fh.read(total)
        if len(buf) < total:
            return []
    if typ == 2:  # ASCII
        return [buf.split(b"\x00")[0].decode("ascii", errors="replace").strip()]
    if typ == 7:  # UNDEFINED: opaque bytes (opcode lists live here)
        return [buf]
    fmt = {1: "B", 3: "H", 4: "L", 8: "h", 9: "l", 11: "f", 12: "d"}.get(typ)
    if fmt:
        return list(struct.unpack(endian + fmt * num, buf))
    if typ in (5, 10):  # RATIONAL / SRATIONAL
        sub = "l" if typ == 10 else "L"
        parts = struct.unpack(endian + sub * (2 * num), buf)
        return [parts[2 * i] / parts[2 * i + 1] if parts[2 * i + 1] else 0.0 for i in range(num)]
    return []


def _parse_tiff_shot_info(fh, info: DngShotInfo) -> None:
    """Fill `info` from a TIFF stream. `fh` must be seekable and positioned so that
    offset 0 is the TIFF header ('II'/'MM')."""
    head = fh.read(8)
    if len(head) < 8 or head[:2] not in (b"II", b"MM"):
        return
    endian = "<" if head[:2] == b"II" else ">"
    (magic,) = struct.unpack(endian + "H", head[2:4])
    if magic != 42:
        return
    (ifd0_off,) = struct.unpack(endian + "L", head[4:8])
    exif_off = None
    sub_ifd_offsets: list[int] = []
    for tag, typ, num, raw in _read_ifd_entries(fh, ifd0_off, endian):
        if tag == TAG_BASELINE_EXPOSURE:
            vals = _entry_values(fh, typ, num, raw, endian)
            if vals:
                info.baseline_exposure = float(vals[0])
        elif tag == TAG_SUB_IFDS:
            vals = _entry_values(fh, typ, num, raw, endian)
            sub_ifd_offsets = [int(v) for v in vals]
        if tag == TAG_MAKE:
            vals = _entry_values(fh, typ, num, raw, endian)
            info.make = vals[0] if vals else None
        elif tag == TAG_MODEL:
            vals = _entry_values(fh, typ, num, raw, endian)
            info.model = vals[0] if vals else None
        elif tag == TAG_ISO and info.iso is None:
            vals = _entry_values(fh, typ, num, raw, endian)
            info.iso = int(vals[0]) if vals else None
        elif tag == TAG_EXIF_IFD:
            vals = _entry_values(fh, typ, num, raw, endian)
            exif_off = int(vals[0]) if vals else None
        elif tag == TAG_AS_SHOT_NEUTRAL:
            vals = _entry_values(fh, typ, num, raw, endian)
            if len(vals) >= 3:
                info.as_shot_neutral = (float(vals[0]), float(vals[1]), float(vals[2]))
    if info.iso is None and exif_off:
        for tag, typ, num, raw in _read_ifd_entries(fh, exif_off, endian):
            if tag == TAG_ISO:
                vals = _entry_values(fh, typ, num, raw, endian)
                info.iso = int(vals[0]) if vals else None
                break
    # Writers may put BaselineExposure on the raw SubIFD rather than IFD0.
    if info.baseline_exposure is None:
        for sub_off in sub_ifd_offsets[:4]:
            for tag, typ, num, raw in _read_ifd_entries(fh, sub_off, endian):
                if tag == TAG_BASELINE_EXPOSURE:
                    vals = _entry_values(fh, typ, num, raw, endian)
                    if vals:
                        info.baseline_exposure = float(vals[0])
                    break
            if info.baseline_exposure is not None:
                break


_RAF_MAGIC = b"FUJIFILMCCD-RAW "
_RAF_MODEL_OFFSET = 0x1C
_RAF_MODEL_LENGTH = 0x20
_RAF_JPEG_OFFSET_FIELD = 0x54  # uint32 BE pair: embedded JPEG offset, length


def _exif_tiff_from_jpeg(data: bytes) -> bytes | None:
    """Extract the TIFF payload of the Exif APP1 segment from JPEG bytes."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            return None
        marker = data[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        if marker == 0xDA:  # start of scan: no EXIF past this point
            return None
        (seg_len,) = struct.unpack(">H", data[pos + 2 : pos + 4])
        seg_start = pos + 4
        seg_end = pos + 2 + seg_len
        if marker == 0xE1 and data[seg_start : seg_start + 6] == b"Exif\x00\x00":
            return data[seg_start + 6 : seg_end]
        pos = seg_end
    return None


def _parse_raf_shot_info(fh, info: DngShotInfo) -> None:
    """Fujifilm RAF: model string lives in the proprietary header; Make/ISO come
    from the EXIF block of the embedded JPEG preview."""
    info.make = "FUJIFILM"
    fh.seek(_RAF_MODEL_OFFSET)
    model_raw = fh.read(_RAF_MODEL_LENGTH)
    model = model_raw.split(b"\x00")[0].decode("ascii", errors="replace").strip()
    if model:
        info.model = model

    fh.seek(_RAF_JPEG_OFFSET_FIELD)
    field = fh.read(8)
    if len(field) < 8:
        return
    jpeg_off, jpeg_len = struct.unpack(">LL", field)
    if jpeg_off <= 0 or jpeg_len <= 0:
        return
    fh.seek(jpeg_off)
    # EXIF sits in the first APP1 segment; 128 KiB comfortably covers it without
    # pulling the whole multi-megabyte preview.
    head = fh.read(min(jpeg_len, 128 * 1024))
    tiff = _exif_tiff_from_jpeg(head)
    if tiff:
        _parse_tiff_shot_info(io.BytesIO(tiff), info)
        info.make = info.make or "FUJIFILM"
        if info.model is None and model:
            info.model = model


def read_dng_shot_info(path: Path) -> DngShotInfo:
    info = DngShotInfo()
    try:
        with open(path, "rb") as fh:
            magic = fh.read(len(_RAF_MAGIC))
            fh.seek(0)
            if magic == _RAF_MAGIC:
                _parse_raf_shot_info(fh, info)
            else:
                _parse_tiff_shot_info(fh, info)
    except (OSError, struct.error):
        pass
    return info


@dataclass
class DngColorCalibration:
    """XYZ->camera matrices under one or two named illuminants (DNG ColorMatrix1/2).

    ``matrix*`` are row-major 3x3 (RGB planes); ``cct*`` are the CCTs of the
    calibration illuminants. ``matrix2`` may be None (single-calibration files).
    """

    matrix1: tuple[tuple[float, float, float], ...]
    cct1: float
    matrix2: tuple[tuple[float, float, float], ...] | None = None
    cct2: float | None = None


def _matrix_from_values(vals: list) -> tuple[tuple[float, float, float], ...] | None:
    if len(vals) < 9:
        return None
    rows = tuple(
        (float(vals[r * 3]), float(vals[r * 3 + 1]), float(vals[r * 3 + 2]))
        for r in range(3)
    )
    if all(abs(v) < 1e-12 for row in rows for v in row):
        return None
    return rows


def is_dng_container(path: Path) -> bool:
    """True when the file is a TIFF container carrying a DNGVersion tag in IFD0.

    Mirrors what LibRaw's ``identify.cpp`` records as ``dng_version`` — the fact the
    rung-2 embedded-matrix adoption gate keys on.  Best-effort: any parse trouble is
    reported as "not a DNG" (the callers then fall to lower ladder rungs).
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) < 8 or head[:2] not in (b"II", b"MM"):
                return False
            endian = "<" if head[:2] == b"II" else ">"
            (magic,) = struct.unpack(endian + "H", head[2:4])
            if magic != 42:
                return False
            (ifd0_off,) = struct.unpack(endian + "L", head[4:8])
            for tag, _typ, _num, _raw in _read_ifd_entries(fh, ifd0_off, endian):
                if tag == TAG_DNG_VERSION:
                    return True
    except (OSError, struct.error):
        return False
    return False


def read_dng_color_calibration(path: Path) -> DngColorCalibration | None:
    """Best-effort DNG dual-illuminant colour calibration; None when unavailable."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) < 8 or head[:2] not in (b"II", b"MM"):
                return None
            endian = "<" if head[:2] == b"II" else ">"
            (magic,) = struct.unpack(endian + "H", head[2:4])
            if magic != 42:
                return None
            (ifd0_off,) = struct.unpack(endian + "L", head[4:8])
            matrices: dict[int, tuple[tuple[float, float, float], ...]] = {}
            illuminants: dict[int, float] = {}
            for tag, typ, num, raw in _read_ifd_entries(fh, ifd0_off, endian):
                if tag in (TAG_COLOR_MATRIX_1, TAG_COLOR_MATRIX_2):
                    matrix = _matrix_from_values(_entry_values(fh, typ, num, raw, endian))
                    if matrix is not None:
                        matrices[1 if tag == TAG_COLOR_MATRIX_1 else 2] = matrix
                elif tag in (TAG_CALIBRATION_ILLUMINANT_1, TAG_CALIBRATION_ILLUMINANT_2):
                    vals = _entry_values(fh, typ, num, raw, endian)
                    if vals:
                        cct = _LIGHT_SOURCE_CCT.get(int(vals[0]))
                        if cct is not None:
                            illuminants[1 if tag == TAG_CALIBRATION_ILLUMINANT_1 else 2] = cct
    except (OSError, struct.error):
        return None
    if 1 in matrices and 1 in illuminants:
        return DngColorCalibration(
            matrix1=matrices[1],
            cct1=illuminants[1],
            matrix2=matrices.get(2),
            cct2=illuminants.get(2),
        )
    if 2 in matrices and 2 in illuminants:
        return DngColorCalibration(matrix1=matrices[2], cct1=illuminants[2])
    return None


TAG_OPCODE_LIST_2 = 51009  # applied to the mosaic before demosaic (GainMap lives here)


@dataclass
class DngGainMap:
    """One GainMap opcode: per-CFA-site lens shading gains (DNG 1.4 spec ch.6).

    Opcode lists are always big-endian regardless of the TIFF byte order. Coordinates
    are active-area rows/cols with a pitch stride selecting the CFA site; the gain grid
    samples at normalized positions origin + index * spacing.
    """

    top: int
    left: int
    bottom: int
    right: int
    row_pitch: int
    col_pitch: int
    points_v: int
    points_h: int
    spacing_v: float
    spacing_h: float
    origin_v: float
    origin_h: float
    map_planes: int
    gains: object  # (points_v, points_h, map_planes) float32


def _parse_gain_map_payload(data: bytes) -> DngGainMap | None:
    import numpy as _np

    if len(data) < 76:
        return None
    head = struct.unpack(">4L2L2L2L", data[:40])
    top, left, bottom, right, _plane, _planes, row_pitch, col_pitch, pv, ph = head
    sv, sh, ov, oh = struct.unpack(">4d", data[40:72])
    (mp,) = struct.unpack(">L", data[72:76])
    n = pv * ph * mp
    if n <= 0 or len(data) < 76 + 4 * n or pv > 4096 or ph > 4096 or mp > 4:
        return None
    gains = _np.frombuffer(data, dtype=">f4", count=n, offset=76).astype(_np.float32)
    return DngGainMap(
        top=top, left=left, bottom=bottom, right=right,
        row_pitch=max(1, row_pitch), col_pitch=max(1, col_pitch),
        points_v=pv, points_h=ph,
        spacing_v=sv, spacing_h=sh, origin_v=ov, origin_h=oh,
        map_planes=mp, gains=gains.reshape(pv, ph, mp),
    )


def read_dng_gain_maps(path: Path) -> list[DngGainMap]:
    """All GainMap opcodes from OpcodeList2, searched across IFD0 and SubIFDs."""
    maps: list[DngGainMap] = []
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) < 8 or head[:2] not in (b"II", b"MM"):
                return []
            endian = "<" if head[:2] == b"II" else ">"
            (magic,) = struct.unpack(endian + "H", head[2:4])
            if magic != 42:
                return []
            (ifd0_off,) = struct.unpack(endian + "L", head[4:8])
            offsets = [ifd0_off]
            for tag, typ, num, raw in _read_ifd_entries(fh, ifd0_off, endian):
                if tag == TAG_SUB_IFDS:
                    offsets.extend(int(v) for v in _entry_values(fh, typ, num, raw, endian))
            blobs: list[bytes] = []
            for off in offsets:
                for tag, typ, num, raw in _read_ifd_entries(fh, off, endian):
                    if tag == TAG_OPCODE_LIST_2 and typ == 7:
                        vals = _entry_values(fh, typ, num, raw, endian)
                        if vals:
                            blobs.append(vals[0])
            for blob in blobs:
                if len(blob) < 4:
                    continue
                (count,) = struct.unpack(">L", blob[:4])
                pos = 4
                for _ in range(min(count, 64)):
                    if pos + 16 > len(blob):
                        break
                    opcode_id, _ver, _flags, size = struct.unpack(">4L", blob[pos:pos + 16])
                    pos += 16
                    payload = blob[pos:pos + size]
                    pos += size
                    if opcode_id == 9:  # GainMap
                        parsed = _parse_gain_map_payload(payload)
                        if parsed is not None:
                            maps.append(parsed)
    except (OSError, struct.error):
        return []
    return maps


TAG_OPCODE_LIST_3 = 51022  # applied after demosaic (FixVignetteRadial, Warp...)


@dataclass
class DngVignetteRadial:
    """FixVignetteRadial opcode: radial gain g = 1 + sum k_i * (r/m)^(2(i+1))."""

    k: tuple[float, float, float, float, float]
    cx_hat: float
    cy_hat: float


def read_dng_shading_ops(path: Path) -> dict:
    """Pre-demosaic GainMaps (OpcodeList2) and post-demosaic radial vignette
    (OpcodeList3 id 3). Warp opcodes are deliberately ignored on this path: geometry
    changes would break CFA-mask alignment, the LibRaw path's defining property."""
    gain_maps = read_dng_gain_maps(path)
    vignette = None
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) < 8 or head[:2] not in (b"II", b"MM"):
                return {"gain_maps": gain_maps, "vignette": None}
            endian = "<" if head[:2] == b"II" else ">"
            (ifd0_off,) = struct.unpack(endian + "L", head[4:8])
            offsets = [ifd0_off]
            for tag, typ, num, raw in _read_ifd_entries(fh, ifd0_off, endian):
                if tag == TAG_SUB_IFDS:
                    offsets.extend(int(v) for v in _entry_values(fh, typ, num, raw, endian))
            for off in offsets:
                for tag, typ, num, raw in _read_ifd_entries(fh, off, endian):
                    if tag == TAG_OPCODE_LIST_3 and typ == 7:
                        vals = _entry_values(fh, typ, num, raw, endian)
                        if not vals or len(vals[0]) < 4:
                            continue
                        blob = vals[0]
                        (count,) = struct.unpack(">L", blob[:4])
                        pos = 4
                        for _ in range(min(count, 64)):
                            if pos + 16 > len(blob):
                                break
                            oid, _v, _f, size = struct.unpack(">4L", blob[pos:pos + 16])
                            pos += 16
                            payload = blob[pos:pos + size]
                            pos += size
                            if oid == 3 and size >= 56:
                                vals7 = struct.unpack(">7d", payload[:56])
                                vignette = DngVignetteRadial(
                                    k=tuple(vals7[:5]), cx_hat=vals7[5], cy_hat=vals7[6]
                                )
    except (OSError, struct.error):
        pass
    return {"gain_maps": gain_maps, "vignette": vignette}
