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
import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path


def extract_jtjs_index(path: Path, source_commit: str) -> dict:
    """Extract only the pinned bundle's literal measurement array; never eval JS.

    Future layouts fail closed and need a reviewed extractor or data-only JSON
    index. The synthetic reference is generated elsewhere and is not measured.
    """
    if source_commit != "504e119bcfb9ff2b58fc9e47c204ddb77eee5879":
        raise ValueError("JTJS bundle extraction requires the audited source commit")
    text = path.read_text(encoding="utf-8")
    start, stop = text.find("KK=["), text.find("],YK=")
    if start < 0 or stop <= start:
        raise ValueError("unsupported JTJS bundle measurement index layout")
    literal = text[start + 3:stop + 1]
    literal = re.sub(r"([\{,])([A-Za-z][A-Za-z0-9]*):", r'\1"\2":', literal)
    literal = literal.replace("!1", "false").replace("!0", "true")
    literal = re.sub(r"([:,])\.(\d)", r"\g<1>0.\2", literal)
    literal = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'",
                     lambda match: json.dumps(ast.literal_eval(match.group(0))), literal)
    entries = json.loads(literal)
    if len(entries) != 118 or any("dataPath" not in entry for entry in entries):
        raise ValueError("unexpected pinned JTJS measurement index content")
    return {"schema": "agxraw-jtjs-measurement-index-1", "entries": entries,
            "source_commit": source_commit, "source_path": "assets/index-Du5lx1zq.js",
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_index(path: Path | None, *, supplied: dict | None = None) -> tuple[dict, dict]:
    """Read an audited, data-only extraction of the fixed JTJS measurement index.

    The public bundle is JavaScript, not executed by the importer. A maintainer
    supplies its JSON index with the bundle hash and source commit. Missing bare
    lens information remains unknown; omission must never be interpreted true.
    """
    if path is None and supplied is None:
        return {}, {}
    index = supplied if supplied is not None else json.loads(path.read_text(encoding="utf-8"))
    if index.get("schema") != "agxraw-jtjs-measurement-index-1":
        raise ValueError("unsupported JTJS index schema")
    by_id = {}
    for entry in index.get("entries", []):
        key = str(entry["id"])
        if key in by_id:
            raise ValueError(f"duplicate measurement id in index: {key}")
        by_id[key] = entry
    provenance = {k: index.get(k) for k in ("source_commit", "source_path", "source_sha256")}
    provenance["index_sha256"] = hashlib.sha256(json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return by_id, provenance


def convert(path: Path, index_entry: dict | None = None, *, source_commit: str | None = None) -> dict:
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
    index_entry = index_entry or {}
    if index_entry and (str(index_entry.get("id")) != str(j["id"])
                        or index_entry.get("name") != j["name"]
                        or index_entry.get("category") != j["category"]):
        raise ValueError(f"{path.name}: index/measurement identity conflict")
    normalization = {"method": "peak-to-100", "domain": "reported-wavelength-range",
                     "absolute_scale_known": False}
    for source, target in (("relativeMaxBeforeScaling", "relative_max_before_scaling"),
                           ("relativeMinBeforeScaling", "relative_min_before_scaling")):
        value = j.get(source)
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{path.name}: invalid {source}")
            if source in index_entry and not math.isclose(value, float(index_entry[source]), rel_tol=1e-12):
                raise ValueError(f"{path.name}: index/measurement {source} conflict")
        normalization[target] = value
    peak = max(data)
    if not math.isclose(peak, 100.0, rel_tol=1e-9):
        raise ValueError(f"{path.name}: declared peak-relative data do not peak at 100")
    bare = index_entry.get("isBareLens")
    if bare is not None and not isinstance(bare, bool):
        raise ValueError(f"{path.name}: isBareLens is not a boolean")
    note = index_entry.get("measurementNote")
    synthetic = j["category"] == "Reference" or j.get("synthetic") is True
    return {
        "id": str(j["id"]),
        "measurement_id": str(j["id"]),
        "semantic_schema": "relative-spectrum-1",
        "quantity": "relative_spectral_transmittance",
        "normalization": normalization,
        "source_commit": source_commit,
        "measurement_kind": "synthetic-reference" if synthetic else "reported-bench-measurement",
        "is_bare_lens": bare if j["category"] == "Lens" else None,
        "measurement_note": note,
        "optical_stack": {"lens": j["name"] if j["category"] == "Lens" else None,
                          "filters": ([str(note)] if bare is False and note else
                                      ([] if bare is True else None)) if j["category"] == "Lens" else [j["name"]]},
        "measurement_conditions": {"instrument": "unknown", "reference_optical_path": "unknown",
                                   "sample_identity": "measurement-id-only"},
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
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
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--index", type=Path, help="data-only JTJS index including optical-stack annotations")
    group.add_argument("--jtjs-bundle", type=Path, help="audited JTJS bundle; extracts literal index without executing JS")
    ap.add_argument("--source-commit", help="fixed upstream commit; never inferred from a live URL")
    args = ap.parse_args()
    supplied = extract_jtjs_index(args.jtjs_bundle, args.source_commit) if args.jtjs_bundle else None
    index, index_provenance = read_index(args.index, supplied=supplied)
    source_commit = args.source_commit or index_provenance.get("source_commit")
    if args.source_commit and index_provenance.get("source_commit") not in (None, args.source_commit):
        raise SystemExit("index/source commit conflict")
    entries = []
    for f in sorted(args.input_dir.glob("*.json")):
        raw_id = str(json.loads(f.read_text(encoding="utf-8"))["id"])
        if index and raw_id not in index:
            raise SystemExit(f"measurement missing from supplied index: {raw_id}")
        entries.append(convert(f, index.get(raw_id), source_commit=source_commit))
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids in input set")
    out = {
        "format": "dngscan-lens-transmittance-1",
        "semantic_schema": "relative-spectrum-1",
        "provenance": {
            "source_commit": source_commit,
            "measurement_index": index_provenance,
            "quantity": "relative_spectral_transmittance",
            "normalization": "each measured spectrum normalized to a peak of 100; not absolute throughput",
            "absolute_throughput_available": False,
            "source": "https://y-g-jiang.github.io/lens-transmittance-data/",
            "nature": "first-party bench measurements by y-g-jiang "
                      "(relative spectral shape, peak=100, 1 nm grid)",
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
