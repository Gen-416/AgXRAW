#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in real preview cache accounting profile; never unittest discovery.

Decode and warm one camera-WB proxy, then render unique EV frames with normal
cache notifications or with notifications suppressed as a diagnostic ablation.
The ablation is local to this process and is not a production cache policy.
Frame/pixel LRUs are cleared between modes; warmed plan and dither stay shared.
Only outer trim calls are timed, avoiding per-node profiling overhead.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from unittest import mock


def distribution(values):
    values = sorted(values)
    return {"median": statistics.median(values), "min": min(values),
            "max": max(values), "sum": sum(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--modes", nargs="+", choices=("active", "notify-off"),
                        default=("active", "notify-off", "notify-off", "active"))
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink() or not args.source.is_file() or args.frames < 2:
        parser.error("source must exist, output must be new, and frames must be at least two")
    os.environ["DNGSCAN_FAST"] = "1"
    os.environ["DNGSCAN_FAST_SKIP"] = ""
    sys.path.insert(0, str(args.repo.resolve()))
    from dngscan import _fast
    from dngscan.gui import preview_cache as pc, service

    ext = _fast._load_extension()
    if ext is None:
        raise RuntimeError("matching native extension is required")
    source_before = args.source.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    identity_before = tuple(getattr(source_before, name) for name in fields)
    state = {"mode": "active", "tag": "outside", "trims": [], "notifications": []}
    original_trim, original_changed = pc.PreviewCache._trim_memory, pc.PreviewEntry._changed

    def trim(store):
        start = time.perf_counter()
        try:
            return original_trim(store)
        finally:
            state["trims"].append({"reason": state["tag"],
                                    "seconds": time.perf_counter() - start})

    def changed(entry):
        state["notifications"].append(state["tag"])
        if state["mode"] == "active":
            return original_changed(entry)

    def tagged(tag, function):
        def call(*a, **kw):
            previous = state["tag"]
            state["tag"] = tag
            try:
                return function(*a, **kw)
            finally:
                state["tag"] = previous
        return call

    common = {"input": str(args.source), "decoder": "libraw", "wb": "camera",
              "gamut": "p3", "format": "sdr", "evAuto": False,
              "previewClient": "frame-profile", "selectionEpoch": 1,
              "previewSession": "frame-profile:1", "includeMetrics": False}
    generation, rounds = 0, []

    def preview(ev):
        nonlocal generation
        generation += 1
        result = service.run_preview({**common, "ev": ev, "generation": generation})
        if not result.get("ok") or result.get("superseded"):
            raise RuntimeError(f"preview failed: {result}")
        return result

    with tempfile.TemporaryDirectory(prefix="agx-frame-profile-") as directory, ExitStack() as patches:
        patches.enter_context(mock.patch.dict(os.environ, {"DNGSCAN_PREVIEW_CACHE_DIR": directory}))
        patches.enter_context(mock.patch.object(pc.PreviewCache, "_trim_memory", trim))
        patches.enter_context(mock.patch.object(pc.PreviewEntry, "_changed", changed))
        for name, tag in (("put_pixels", "pixels"), ("put_frame", "frame"),
                          ("get_or_build_dither_noise", "dither")):
            patches.enter_context(mock.patch.object(pc.PreviewEntry, name,
                                                    tagged(tag, getattr(pc.PreviewEntry, name))))
        result = service.prepare_preview(common)
        if not result.get("ok") or result.get("superseded"):
            raise RuntimeError(f"prepare failed: {result}")
        preview(0.)
        if not pc.DISK_WRITER.flush(30):
            raise RuntimeError("disk writer did not drain")
        store = service.PREVIEW_STORE
        with store.lock:
            entry = next(iter(store.entries.values()))
        for mode in args.modes:
            state["mode"] = mode
            with entry._runtime_cache_lock:
                entry._frame_cache.clear()
                entry._pixel_cache.clear()
            frames = []
            for index in range(args.frames):
                state["trims"], state["notifications"] = [], []
                # Every frame has a different exact frame key; keep the same
                # exposure sequence in every mode for decoded-byte comparison.
                ev = 1.5 * (index + 1) / args.frames
                start = time.perf_counter()
                result = preview(ev)
                elapsed = time.perf_counter() - start
                with entry._runtime_cache_lock:
                    lengths = {"frames": len(entry._frame_cache), "pixels": len(entry._pixel_cache)}
                    base_mask_cached = getattr(entry.bundle, "_clip_masks_resized", None) is not None
                frames.append({"index": index, "ev": ev, "seconds": elapsed,
                               "trim_seconds": sum(t["seconds"] for t in state["trims"]),
                               "trim_calls": list(state["trims"]),
                               "notifications": list(state["notifications"]),
                               "cache_lengths": lengths, "base_mask_cached": base_mask_cached,
                               "preview_sha256": hashlib.sha256(result["preview"].encode("ascii")).hexdigest()})
            rounds.append({"mode": mode, "frames": frames,
                           "wall_s": distribution([f["seconds"] for f in frames]),
                           "trim_s": distribution([f["trim_seconds"] for f in frames]),
                           "cache": store.memory_snapshot()})
            print(json.dumps({"mode": mode, "wall_s": rounds[-1]["wall_s"],
                              "trim_s": rounds[-1]["trim_s"]}), flush=True)
        hashes = [f["preview_sha256"] for f in rounds[0]["frames"]]
        if any([f["preview_sha256"] for f in r["frames"]] != hashes for r in rounds):
            raise RuntimeError("preview hashes changed between notification modes")

    source_after = args.source.stat()
    if tuple(getattr(source_after, name) for name in fields) != identity_before:
        raise RuntimeError("source changed during profile")
    report = {"repo": str(args.repo.resolve()), "source": str(args.source.resolve()),
              "commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
              "native_abi": ext.native_abi_version(), "cache_version": pc.PREVIEW_CACHE_VERSION,
              "scope": "single client, camera WB, warm plan/dither; real preview rendering; notification ablation only",
              "hashes_exact": True, "rounds": rounds}
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
