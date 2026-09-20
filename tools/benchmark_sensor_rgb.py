#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure exact sensor RGB clip counts with every other native kernel enabled.

--reference disables only sensor_rgb_clip_counts_u16. Both real RAW modes use
deferred clip masks and retain SensorSummary reuse. --source forms default AgX
SDR/HDR masters without encoding; --synthetic WIDTH HEIGHT measures only RGB
clip statistics for Bayer, X-Trans and linear camera RGB. Input generation and
hashing are outside timed regions. Run each side in a fresh process, alternating
reference/native order. --compare requires the complete identity to match.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from unittest.mock import patch


KERNEL = "sensor_rgb_clip_counts_u16"


def synthetic(args, record, analysis, array_record):
    import gc
    import numpy as np

    width, height = args.synthetic
    cases, identity = {}, {}
    patterns = {
        "bayer": [[0, 1], [3, 2]],
        "xtrans": [[1, 0, 1, 1, 2, 1], [2, 1, 2, 0, 1, 0],
                   [1, 0, 1, 1, 2, 1], [1, 2, 1, 1, 0, 1],
                   [0, 1, 0, 2, 1, 2], [1, 2, 1, 1, 0, 1]],
        "linear_rgb": [[0, 1, 2]],
    }
    for name, pattern in patterns.items():
        linear = name == "linear_rgb"
        shape = (height, width, 3) if linear else (height, width)
        raw = np.random.default_rng(371).integers(0, 16384, shape, dtype=np.uint16)
        if linear:
            colors = np.broadcast_to(np.array([0, 1, 2], dtype=np.uint8), shape)
        else:
            tile = np.asarray(pattern, dtype=np.uint8)
            ph, pw = tile.shape
            colors = np.tile(tile, ((height + ph - 1) // ph,
                                    (width + pw - 1) // pw))[:height, :width]
        labels = {0: "R", 1: "G1", 2: "B"}
        thresholds = {0: 16256, 1: 16270, 2: 16240}
        if name == "bayer":
            labels[3], thresholds[3] = "G2", 16280
        inputs = {"raw": array_record(raw), "colors": array_record(colors),
                  "color_strides": list(colors.strides), "pattern": pattern,
                  "thresholds": thresholds, "labels": labels, "seed": 371}
        elapsed, result = [], None
        for _ in range(args.repeats):
            start = time.perf_counter()
            current = analysis.compute_color_clip_metrics(raw, colors, thresholds, labels, pattern)
            elapsed.append(time.perf_counter() - start)
            if result is not None and current != result:
                raise RuntimeError("RGB clip statistics changed between identical repetitions")
            result = current
        cases[name] = {"wall_s": elapsed, "median_s": statistics.median(elapsed)}
        identity[name] = {"inputs": inputs, "rgb_clip_pct": result}
        del raw, colors
        gc.collect()
    record.update(identity=identity, synthetic_cases=cases,
                  synthetic_size=[width, height], repeats=args.repeats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path)
    mode.add_argument("--synthetic", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--decoder", choices=("libraw", "coreimage"), default="libraw")
    p.add_argument("--reference", action="store_true")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--compare", type=Path)
    args = p.parse_args()
    if args.out.exists() or args.out.is_symlink():
        p.error("output must be new")
    if args.source is not None and not args.source.is_file():
        p.error("source must exist")
    if args.repeats < 1 or (args.synthetic is not None and min(args.synthetic) < 1):
        p.error("dimensions and repeats must be positive")
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
    os.environ["DNGSCAN_FAST_SKIP"] = KERNEL if args.reference else ""
    sys.path.insert(0, str(args.repo.resolve()))
    from dngscan import _fast, analysis, raw_io
    from tools.benchmark_loss_pipeline import array_record, json_value, pipeline
    import numpy as np

    ext = _fast._load_extension()
    if ext is None:
        raise RuntimeError("matching native extension is required on both sides")
    if not args.reference and _fast.kernel(KERNEL) is None:
        raise RuntimeError("native RGB clip kernel is required outside --reference")
    commit = subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=True).stdout.strip()
    record = {"reference": args.reference, "source": str(args.source) if args.source else None,
              "decoder": args.decoder, "commit": commit,
              "python": platform.python_version(), "numpy": np.__version__,
              "platform": platform.platform(), "native_abi": ext.native_abi_version(),
              "cpus": os.cpu_count(),
              "environment": {name: os.environ[name] for name in
                              ("DNGSCAN_FAST", "DNGSCAN_FAST_SKIP")}}
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

    kernel, load_raw = _fast.kernel, raw_io.load_raw

    def observed_kernel(name):
        fn = kernel(name)
        return measured("native." + name, fn) if name == KERNEL and fn is not None else fn

    def observed_load(*a, **kw):
        # Equal handoff on both sides, including recursive decoder fallback.
        kw.setdefault("_defer_clip_masks", True)
        return load_raw(*a, **kw)

    with ExitStack() as stack:
        stack.enter_context(patch.object(analysis, "compute_color_clip_metrics", measured(
            "compute_color_clip_metrics", analysis.compute_color_clip_metrics)))
        stack.enter_context(patch.object(_fast, "kernel", observed_kernel))
        stack.enter_context(patch.object(raw_io, "load_raw", observed_load))
        if args.synthetic is not None:
            synthetic(args, record, analysis, array_record)
        else:
            pipeline(args, record)
    record["sensor_rgb_calls"] = calls
    if args.source is not None:
        digest = hashlib.sha256()
        with args.source.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        record["source_sha256"] = digest.hexdigest()
    record["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        2 ** 20 if sys.platform == "darwin" else 1024)
    record = json_value(record)
    matches = True
    if previous is not None:
        matches = record["identity"] == previous["identity"]
        if "source_sha256" in previous:
            matches = matches and record.get("source_sha256") == previous["source_sha256"]
        record["comparison"] = {"report": str(args.compare), "exact": matches}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k in (
        "reference", "sensor_rgb_calls", "synthetic_cases", "pipeline_sdr_s",
        "pipeline_hdr_s", "peak_rss_mib", "comparison")}), flush=True)
    return 0 if matches else 2


if __name__ == "__main__":
    raise SystemExit(main())
