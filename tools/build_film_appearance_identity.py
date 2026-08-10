#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the P1 identity appearance recipes and pin the manifest.

Identity recipes exist so the P1 wiring can be exercised end to end —
compile, hash-verify, apply — while provably changing nothing: every field
is zero, and the loader's P1 gate refuses anything else. P4 replaces these
with authored palettes and removes the gate.

    python tools/build_film_appearance_identity.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from dngscan.film_appearance import (
    APPEARANCE_DIR,
    APPEARANCE_SCHEMA,
    EV_KNOTS,
    HUE_KNOT_COUNT,
    MANIFEST_PATH,
)

# The P4 first pair (plan §10): Portra and Ektar on Endura, authored jointly.
# P1 ships their identity placeholders so the whole pipeline is walkable.
RECIPES = (
    ("portra400", "kodak_portra_endura__translated"),
    ("ektar100", "kodak_portra_endura__translated"),
)


def build(stock_id: str, medium_id: str) -> Path:
    from dngscan.film_appearance import medium_family

    rid = f"{stock_id}__{medium_family(medium_id)}_reference_v1"
    k, h = len(EV_KNOTS), HUE_KNOT_COUNT
    meta = {
        "schema": APPEARANCE_SCHEMA,
        "recipe_id": rid,
        "stock_id": stock_id,
        "medium_id": medium_id,
        "process_space": "display-linear-rec2020/oklab+scene-ev",
        "provenance": "editorial-authored",
        "chroma_knee": 0.18,
        "chroma_power": 2.0,
        "neutral_chroma_c0": 0.03,
        "note": (
            "P1 identity placeholder: every field zero by construction. "
            "P4 authors the real palette; the loader refuses non-identity "
            "fields until the P2 kernel exists."
        ),
    }
    path = APPEARANCE_DIR / f"{rid}.npz"
    np.savez_compressed(
        path,
        meta=np.asarray(json.dumps(meta, sort_keys=True)),
        ev_knots=np.asarray(EV_KNOTS, dtype=np.float32),
        hue_knots_deg=(np.arange(h) * (360.0 / h)).astype(np.float32),
        hue_delta_deg=np.zeros((k, h), dtype=np.float32),
        log_chroma_gain=np.zeros((k, h), dtype=np.float32),
        density_ev=np.zeros((k, h), dtype=np.float32),
        neutral_bias_ab=np.zeros((k, 2), dtype=np.float32),
    )
    return path


def main() -> int:
    APPEARANCE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [build(stock, medium) for stock, medium in RECIPES]
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(APPEARANCE_DIR.glob("*.npz"))
    }
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "schema": APPEARANCE_SCHEMA,
                "count": len(files),
                "files": files,
                "policy": (
                    "Every appearance recipe is hash-pinned; the loader "
                    "refuses an asset whose bytes drift from this manifest."
                ),
            },
            indent=1, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    for p in paths:
        print(f"wrote {p.name}")
    print(f"wrote {MANIFEST_PATH.name} ({len(files)} assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
