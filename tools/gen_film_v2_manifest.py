#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pin the published film v2 asset family by content hash (review batch 13).

Writes dngscan/data/film_v2/MANIFEST.json: schema, file count and sha256
per npz. tests/test_film_v2_assets.py verifies the shipped files against it,
so a stale, truncated or tampered asset shows up as a manifest diff in
review — never as silently different rendering. Regenerate ONLY together
with an intentional rebake.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "dngscan" / "data" / "film_v2"


def main() -> None:
    files = sorted(p for p in ASSET_DIR.glob("*.npz"))
    manifest = {
        "schema": 5,
        "count": len(files),
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    out = ASSET_DIR / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"{out}: {len(files)} assets pinned")


if __name__ == "__main__":
    main()
