#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Import y-g-jiang's first-party lens transmittance measurements.

Upstream: https://y-g-jiang.github.io/lens-transmittance-data/ — one JSON
per measurement sheet, spectral transmittance in percent on a uniform
380-755 nm @ 1 nm grid (376 samples), covering lenses and filters. These
are the author's own bench measurements (same licensing posture as the
JPTC data: permission granted 2026-08-25, no formal license, use
permitted with credit, see NOTICE.md).

The importer validates every file (uniform grid, finite values, plausible
percent range) and bundles them into ONE excisable data file
(dngscan/data/lens_transmittance.json) with provenance. Access API:
dngscan.lens_transmittance.

Usage:
    python tools/import_lens_transmittance.py <dir-of-upstream-jsons> \\
        --out dngscan/data/lens_transmittance.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def convert(path: Path) -> dict:
    j = json.loads(path.read_text())
    n = int(j["sampleCount"])
    data = [float(v) for v in j["data"]]
    if len(data) != n:
        raise ValueError(f"{path.name}: sampleCount {n} != len(data) {len(data)}")
    start, end, step = int(j["wavelengthStart"]), int(j["wavelengthEnd"]), int(j["stepNm"])
    if start + step * (n - 1) != end:
        raise ValueError(f"{path.name}: grid {start}..{end}@{step} inconsistent with {n} samples")
    if not all(math.isfinite(v) and 0.0 < v <= 150.0 for v in data):
        raise ValueError(f"{path.name}: transmittance outside (0, 150]%")
    return {
        "id": str(j["id"]),
        "name": str(j["name"]),
        "category": str(j["category"]),
        "sheet": j.get("sheet"),
        "wavelength_start_nm": start,
        "step_nm": step,
        "transmittance_percent": data,  # verbatim upstream precision
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    entries = []
    for f in sorted(args.input_dir.glob("*.json")):
        entries.append(convert(f))
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids in input set")
    out = {
        "format": "dngscan-lens-transmittance-1",
        "provenance": {
            "source": "https://y-g-jiang.github.io/lens-transmittance-data/",
            "nature": "first-party bench measurements by y-g-jiang "
                      "(spectral transmittance, percent, 1 nm grid)",
            "license": "permission granted by the author 2026-08-25: "
                       "no formal license, use permitted with credit "
                       "(NOTICE.md); this single file is the entire "
                       "footprint",
        },
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    cats: dict[str, int] = {}
    for e in entries:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
    print(json.dumps({"entries": len(entries), "categories": cats,
                      "bytes": args.out.stat().st_size}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
