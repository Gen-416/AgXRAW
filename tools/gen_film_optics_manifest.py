#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pin the analog-optics assets by content hash (FILM_OPTICS_V2 §7.1).

The loader verifies these digests before it parses anything, so an asset that
was edited without running this tool fails closed rather than rendering with
values nobody reviewed.

    python tools/gen_film_optics_manifest.py
    python tools/gen_film_optics_manifest.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "dngscan" / "data" / "film_optics"
MANIFEST_PATH = ASSET_DIR / "MANIFEST.json"
SCHEMA = 1


def build() -> dict:
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(ASSET_DIR.glob("*.json"))
        if path.name != "MANIFEST.json"
    }
    return {"schema": SCHEMA, "count": len(files), "files": files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    built = build()
    if args.check:
        if not MANIFEST_PATH.is_file():
            print("missing MANIFEST.json", file=sys.stderr)
            return 1
        stored = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if stored != built:
            print("optics manifest is stale", file=sys.stderr)
            return 1
        print(f"{built['count']} assets pinned, manifest current")
        return 0
    MANIFEST_PATH.write_text(
        json.dumps(built, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {MANIFEST_PATH} ({built['count']} assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
