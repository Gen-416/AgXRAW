#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert a y-g-jiang DNGSHL1 shell into a dngscan evidence shell.

DNGSHL1 is the upstream format of the dngshell corpus
(https://y-g-jiang.github.io/shells/): magic ``DNGSHL1\\x00``, u32-LE
manifest length, JSON manifest, then the uncompressed concatenation of the
kept byte ranges. Removed ranges are zero-filled on materialization (the
upstream catalog's validations attest removedBytesZero).

This importer materializes the sparse file and re-strips it with our own
walker (tools/make_evidence_shell.py), so the result is a normal
``dngscan-evshell-1`` shell and the test harness stays single-format. The
manifest gains an ``upstream`` block carrying the DNGSHL1 provenance: the
original file's SHA-256 (which we cannot recompute — we never see the
pixels), its size, and the shell URL. Note the outer ``source_sha256``
therefore hashes the ZERO-FILLED reconstruction, not the true original;
the true original's hash lives in ``upstream.source_sha256``.

Licensing: the featured shells are first-party captures by the corpus
author. Permission granted 2026-08-25: no formal license, use permitted
with credit (see NOTICE.md). Pass the exact status text via --license.

Usage:
    python tools/import_dngshell.py SONY_ILCE-7M5.dngshell \\
        tests/data/evidence_shells/sony_ilce7m5.evshell \\
        --license "first-party capture by y-g-jiang; permitted with credit (NOTICE.md)" \\
        --source-url https://y-g-jiang.github.io/shells/SONY_ILCE-7M5.dngshell
"""
from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_evidence_shell import make_shell  # noqa: E402


def read_dngshl1(path: Path) -> tuple[dict, bytes]:
    buf = path.read_bytes()
    if buf[:8] != b"DNGSHL1\x00":
        raise ValueError(f"{path.name}: not a DNGSHL1 shell")
    (mlen,) = struct.unpack("<I", buf[8:12])
    manifest = json.loads(buf[12: 12 + mlen])
    payload = buf[12 + mlen:]
    if len(payload) != int(manifest["shellPayloadBytes"]):
        raise ValueError(f"{path.name}: payload length mismatch")
    return manifest, payload


def materialize_dngshl1(manifest: dict, payload: bytes, out: Path) -> None:
    with out.open("wb") as fh:
        fh.truncate(int(manifest["originalSize"]))
        for r in manifest["keptRanges"]:
            fh.seek(int(r["offset"]))
            off = int(r["blobOffset"])
            fh.write(payload[off: off + int(r["length"])])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--license", required=True)
    ap.add_argument("--source-url", required=True)
    args = ap.parse_args()
    up, payload = read_dngshl1(args.input)
    with tempfile.TemporaryDirectory() as td:
        sparse = Path(td) / (up.get("sourceFileName") or "materialized.dng")
        materialize_dngshl1(up, payload, sparse)
        stats = make_shell(sparse, args.output, args.license, args.source_url)
    # Graft the upstream provenance into the manifest line.
    with args.output.open("rb") as fh:
        manifest = json.loads(fh.readline())
        body = fh.read()
    manifest["upstream"] = {
        "format": "DNGSHL1",
        "shell_name": args.input.name,
        "source_name": up.get("sourceFileName"),
        "source_sha256": up.get("sourceSha256"),
        "source_bytes": up.get("originalSize"),
        "make": up.get("make"),
        "model": up.get("model"),
    }
    with args.output.open("wb") as fh:
        fh.write(json.dumps(manifest, separators=(",", ":")).encode() + b"\n")
        fh.write(body)
    stats["shell_bytes"] = args.output.stat().st_size
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
