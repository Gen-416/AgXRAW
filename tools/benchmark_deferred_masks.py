#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure deferred clip-mask construction with every native kernel enabled.

Each invocation forms default full-resolution AgX SDR/HDR masters in a fresh
process without encoding. --reference uses the public eager load contract;
otherwise only load_raw's private deferred-mask handoff is enabled. --compare
requires exact buffers and decisions except identity.masks_loaded, whose
intentional absence before analyze is recorded separately, never fabricated.
Alternate reference/deferred order across at least three pairs of processes.
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
import subprocess
import sys
import time
from unittest.mock import patch


def mask_state(bundle):
    """Record the actual handoff, including old bundles without a pending field."""
    return {"decoder": getattr(bundle, "scene_decoder", None),
            "pending": bool(getattr(bundle, "_clip_masks_pending", False)),
            "mask_present": getattr(bundle, "clip_masks", None) is not None}


def comparable_identity(identity):
    return {key: value for key, value in identity.items() if key != "masks_loaded"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--decoder", choices=("libraw", "coreimage"), default="libraw")
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
    from dngscan import _fast, raw_io
    from tools.benchmark_loss_pipeline import pipeline, json_value
    import numpy as np

    ext = _fast._load_extension()
    if ext is None:
        raise RuntimeError("matching native extension is required on both sides")
    commit = subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=True).stdout.strip()
    record = {"reference": args.reference, "source": str(args.source),
              "decoder": args.decoder, "commit": commit,
              "python": platform.python_version(), "numpy": np.__version__,
              "platform": platform.platform(), "native_abi": ext.native_abi_version(),
              "cpus": os.cpu_count()}
    calls, loaded, refreshed = {}, [], []

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

    load_raw = raw_io.load_raw
    refresh = raw_io.refresh_clip_masks_from_fullwell
    load_depth = 0

    def observed_load(*a, **kw):
        nonlocal load_depth
        # Reference supplies no new keyword, so unchanged older checkouts work.
        # Recursive auto fallback comes through this wrapper too: never overwrite
        # a supplied flag, and in particular preserve an existing True value.
        if not args.reference:
            kw.setdefault("_defer_clip_masks", True)
        depth = load_depth
        load_depth += 1
        try:
            bundle = load_raw(*a, **kw)
        finally:
            load_depth -= 1
        loaded.append({"depth": depth, "state": mask_state(bundle)})
        return bundle

    def observed_refresh(bundle, *a, **kw):
        event = {"before": mask_state(bundle), "completed": False}
        try:
            result = refresh(bundle, *a, **kw)
            event.update(completed=True, rebuilt=bool(result))
            return result
        finally:
            event["after"] = mask_state(bundle)
            refreshed.append(event)

    with ExitStack() as stack:
        stack.enter_context(patch.object(raw_io, "build_clip_masks",
                                         measured("build_clip_masks", raw_io.build_clip_masks)))
        stack.enter_context(patch.object(raw_io, "refresh_clip_masks_from_fullwell",
                                         measured("refresh_clip_masks_from_fullwell", observed_refresh)))
        stack.enter_context(patch.object(raw_io, "load_raw", observed_load))
        pipeline(args, record)
    record.update(mask_calls=calls, mask_load_states=loaded, mask_refresh_states=refreshed)
    with args.source.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    record["source_sha256"] = digest.hexdigest()
    record["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 20 if sys.platform == "darwin" else 1024)
    matches = True
    if previous is not None:
        matches = comparable_identity(record["identity"]) == comparable_identity(previous["identity"])
        if "source_sha256" in previous:
            matches = matches and record["source_sha256"] == previous["source_sha256"]
        record["comparison"] = {"report": str(args.compare), "exact": matches,
                                "excluded_fields": ["identity.masks_loaded"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(json.dumps(json_value(record), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k in (
        "reference", "mask_calls", "mask_load_states", "mask_refresh_states", "pipeline_sdr_s",
        "pipeline_hdr_s", "peak_rss_mib", "comparison")}), flush=True)
    return 0 if matches else 2


if __name__ == "__main__":
    raise SystemExit(main())
