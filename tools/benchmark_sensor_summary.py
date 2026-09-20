#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure sensor-summary reuse with every native kernel held enabled.

Each invocation loads a full-resolution RAW and forms default AgX SDR/HDR
masters in a fresh process. --reference bypasses only summary caching. No
encoding is timed. --compare requires bit-identical buffers and decisions.
Alternate reference/native order across at least three pairs of processes.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--decoder", choices=("libraw", "coreimage"), default="coreimage")
    p.add_argument("--reference", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--compare", type=Path)
    args = p.parse_args()
    if args.out.exists() or args.out.is_symlink() or not args.source.is_file():
        p.error("output must be new and source must exist")
    if not (args.repo / "dngscan" / "raw_io.py").is_file():
        p.error("invalid --repo")
    previous = None
    if args.compare:
        try:
            previous = json.loads(args.compare.read_text())
            if not isinstance(previous, dict) or not isinstance(previous.get("identity"), dict):
                raise ValueError("comparison report must contain an identity object")
        except (OSError, ValueError) as exc:
            p.error(str(exc))

    os.environ["DNGSCAN_FAST"] = "1"
    os.environ["DNGSCAN_FAST_SKIP"] = ""
    sys.path.insert(0, str(args.repo.resolve()))
    from dngscan import _fast, analysis
    from tools.benchmark_loss_pipeline import pipeline, json_value
    import numpy as np

    ext = _fast._load_extension()
    if ext is None:
        raise RuntimeError("matching native extension is required on both sides")
    try:
        summary = importlib.import_module("dngscan.sensor_summary")
    except ModuleNotFoundError as exc:
        if not args.reference or exc.name != "dngscan.sensor_summary":
            raise
        summary = None  # An unchanged pre-summary checkout is also a reference.
    commit = subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=True).stdout.strip()
    record = {"reference": args.reference, "summary_available": summary is not None,
              "source": str(args.source), "decoder": args.decoder, "commit": commit,
              "python": platform.python_version(), "numpy": np.__version__,
              "platform": platform.platform(), "native_abi": ext.native_abi_version(),
              "cpus": os.cpu_count()}
    calls = {}

    def measured(name, fn):
        def invoke(*a, **kw):
            start = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                stat = calls.setdefault(name, {"count": 0, "wall_s": 0.})
                stat["count"] += 1
                stat["wall_s"] += time.perf_counter() - start
        return invoke

    with ExitStack() as stack:
        for name in ("channel_saturation_levels", "detect_ceilings", "resolve_fullwell"):
            stack.enter_context(patch.object(analysis, name, measured(name, getattr(analysis, name))))
        if summary is not None:
            stack.enter_context(patch.object(summary, "summarize_sensor",
                                             measured("summarize_sensor", summary.summarize_sensor)))
            if args.reference:
                stack.enter_context(patch.object(summary, "_cache_signature", return_value=None))
        pipeline(args, record)
    record["sensor_calls"] = calls
    with args.source.open("rb") as source:
        # A bounded source hash outside the measured pipeline. Compatible with 3.10.
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    record["source_sha256"] = digest.hexdigest()
    record["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 20 if sys.platform == "darwin" else 1024)
    matches = True
    if previous is not None:
        matches = record["identity"] == previous["identity"]
        # Older loss-pipeline reports do not carry an input hash.
        if "source_sha256" in previous:
            matches = matches and record["source_sha256"] == previous["source_sha256"]
        record["comparison"] = {"report": str(args.compare), "exact": matches}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(json.dumps(json_value(record), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k in (
        "reference", "sensor_calls", "pipeline_sdr_s", "pipeline_hdr_s", "peak_rss_mib", "comparison")}), flush=True)
    return 0 if matches else 2


if __name__ == "__main__":
    raise SystemExit(main())
