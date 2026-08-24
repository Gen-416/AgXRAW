#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Evidence shells: strip a TIFF-family RAW to its container structure.

Idea credited to y-g-jiang's DNGSHL1 ("dngshell") test-corpus format; this is
an independent implementation with our own layout. A shell keeps EVERY byte of
the file except the bulk pixel regions (strip/tile data and embedded JPEG
previews), so the METADATA/EVIDENCE parsers — is_dng_container, DngShotInfo,
read_dng_color_calibration, read_dng_stage1_flags, read_dng_gain_maps — parse
a materialized shell exactly as they parse the original. Pixel decoding is out
of scope by design: shells gate the parsers on CI where RAW corpora cannot
live.

Format (JSON + gzip payload, single .evshell file):
    {"format": "dngscan-evshell-1", "source_name": ..., "source_sha256": ...,
     "source_bytes": N, "license": ..., "source_url": ...,
     "kept": [[off, len], ...]}\\n
    <gzip of the concatenated kept ranges>

Materialization writes the kept ranges into a sparse file of the original
size (stripped regions read as zeros). Round-trip contract: every kept byte
identical; the manifest records the source hash for provenance.

Usage:
    python tools/make_evidence_shell.py input.dng out.evshell \\
        [--license CC0] [--source-url URL]
    python tools/make_evidence_shell.py --materialize shell.evshell out.dng
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import sys
from pathlib import Path

# Bulk-data tags whose referenced regions are stripped from the shell.
TAG_STRIP_OFFSETS = 273
TAG_STRIP_BYTE_COUNTS = 279
TAG_TILE_OFFSETS = 324
TAG_TILE_BYTE_COUNTS = 325
TAG_JPEG_IF = 513          # JPEGInterchangeFormat (embedded preview offset)
TAG_JPEG_IF_LENGTH = 514
TAG_SUB_IFDS = 330
TAG_EXIF_IFD = 34665

_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8,
               11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8}

# Keep tiny data blobs even when referenced by a bulk tag: stripping a
# 4-byte "strip" saves nothing and risks clipping inline values.
MIN_STRIP_BYTES = 4096


def _read_entries(buf: bytes, off: int, endian: str):
    if off + 2 > len(buf):
        return [], None
    (count,) = struct.unpack_from(endian + "H", buf, off)
    if count > 4096:
        return [], None
    entries = []
    base = off + 2
    for i in range(count):
        e = base + i * 12
        if e + 12 > len(buf):
            return entries, None
        tag, typ, num = struct.unpack_from(endian + "HHL", buf, e)
        entries.append((tag, typ, num, buf[e + 8: e + 12]))
    nxt_off = base + count * 12
    if nxt_off + 4 > len(buf):
        return entries, None
    (nxt,) = struct.unpack_from(endian + "L", buf, nxt_off)
    return entries, (nxt or None)


def _values(buf: bytes, typ: int, num: int, raw: bytes, endian: str):
    size = _TYPE_SIZES.get(typ)
    if size is None:
        return []
    total = size * num
    if total <= 4:
        data = raw[:total]
    else:
        (off,) = struct.unpack(endian + "L", raw)
        data = buf[off: off + total]
        if len(data) < total:
            return []
    fmt = {1: "B", 3: "H", 4: "L", 8: "h", 9: "l"}.get(typ)
    if fmt:
        return list(struct.unpack(endian + fmt * num, data))
    return []


def _collect_pixel_ranges(buf: bytes) -> list[tuple[int, int]]:
    """(offset, length) of every bulk pixel region across the whole IFD tree."""
    if len(buf) < 8 or buf[:2] not in (b"II", b"MM"):
        raise ValueError("not a TIFF-family container")
    endian = "<" if buf[:2] == b"II" else ">"
    (magic,) = struct.unpack_from(endian + "H", buf, 2)
    if magic != 42:
        raise ValueError(f"unsupported TIFF magic {magic}")
    (ifd0,) = struct.unpack_from(endian + "L", buf, 4)

    ranges: list[tuple[int, int]] = []
    seen: set[int] = set()
    queue = [ifd0]
    while queue:
        off = queue.pop()
        if off in seen or not off:
            continue
        seen.add(off)
        entries, nxt = _read_entries(buf, off, endian)
        if nxt:
            queue.append(nxt)
        tags = {}
        for tag, typ, num, raw in entries:
            tags[tag] = (typ, num, raw)
            if tag in (TAG_SUB_IFDS, TAG_EXIF_IFD):
                for v in _values(buf, typ, num, raw, endian):
                    queue.append(int(v))
        for off_tag, len_tag in (
            (TAG_STRIP_OFFSETS, TAG_STRIP_BYTE_COUNTS),
            (TAG_TILE_OFFSETS, TAG_TILE_BYTE_COUNTS),
            (TAG_JPEG_IF, TAG_JPEG_IF_LENGTH),
        ):
            if off_tag in tags and len_tag in tags:
                offs = _values(buf, *tags[off_tag], endian)
                lens = _values(buf, *tags[len_tag], endian)
                for o, n in zip(offs, lens):
                    if n >= MIN_STRIP_BYTES and 0 < o < len(buf):
                        ranges.append((int(o), int(min(n, len(buf) - o))))
    return ranges


def _kept_ranges(total: int, pixel: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Complement of the pixel ranges over [0, total)."""
    merged: list[list[int]] = []
    for o, n in sorted(pixel):
        if merged and o <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], o + n)
        else:
            merged.append([o, o + n])
    kept = []
    cur = 0
    for a, b in merged:
        if a > cur:
            kept.append((cur, a - cur))
        cur = max(cur, b)
    if cur < total:
        kept.append((cur, total - cur))
    return kept


def make_shell(
    src: Path, dst: Path, license_id: str = "", source_url: str = ""
) -> dict:
    buf = src.read_bytes()
    pixel = _collect_pixel_ranges(buf)
    if not pixel:
        raise ValueError(
            "no bulk pixel regions found — refusing to shell a file whose "
            "structure was not understood (the shell would just be a copy)"
        )
    kept = _kept_ranges(len(buf), pixel)
    payload = b"".join(buf[o: o + n] for o, n in kept)
    manifest = {
        "format": "dngscan-evshell-1",
        "source_name": src.name,
        "source_sha256": hashlib.sha256(buf).hexdigest(),
        "source_bytes": len(buf),
        "license": license_id,
        "source_url": source_url,
        "kept": [[o, n] for o, n in kept],
    }
    with dst.open("wb") as fh:
        fh.write(json.dumps(manifest, separators=(",", ":")).encode() + b"\n")
        fh.write(gzip.compress(payload, 9))
    return {
        "shell": str(dst),
        "source_bytes": len(buf),
        "shell_bytes": dst.stat().st_size,
        "kept_bytes": sum(n for _, n in kept),
        "ratio": dst.stat().st_size / len(buf),
    }


def materialize(shell: Path, out: Path) -> dict:
    with shell.open("rb") as fh:
        manifest = json.loads(fh.readline())
        payload = gzip.decompress(fh.read())
    if manifest.get("format") != "dngscan-evshell-1":
        raise ValueError("not a dngscan evidence shell")
    with out.open("wb") as fh:
        fh.truncate(int(manifest["source_bytes"]))
        pos = 0
        for o, n in manifest["kept"]:
            fh.seek(int(o))
            fh.write(payload[pos: pos + int(n)])
            pos += int(n)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--license", default="")
    ap.add_argument("--source-url", default="")
    args = ap.parse_args()
    if args.materialize:
        m = materialize(args.input, args.output)
        print(json.dumps({k: m[k] for k in ("source_name", "source_sha256")}))
    else:
        info = make_shell(args.input, args.output, args.license, args.source_url)
        print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
