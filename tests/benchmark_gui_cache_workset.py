#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in real RAW GUI cache workset probe; never part of unittest discovery.

Use a fresh process per repository and source. Runs seven white balances twice,
then clears only memory and reloads the disk proxy. Measures GUI API work without
HTTP transport. Tracked cache bytes are not process RSS; RSS is reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from unittest import mock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--decoder", choices=("libraw", "coreimage"), default="libraw")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink() or not args.source.is_file():
        parser.error("source must exist and output must be new")
    os.environ["DNGSCAN_FAST"] = "1"
    os.environ["DNGSCAN_FAST_SKIP"] = ""
    sys.path.insert(0, str(args.repo.resolve()))
    import dngscan as dg
    from dngscan.gui import preview_cache as pc, service
    from dngscan import _fast

    source_before = args.source.stat()
    identity_before = (source_before.st_dev, source_before.st_ino, source_before.st_size,
                       source_before.st_mtime_ns, source_before.st_ctime_ns)
    ext = _fast._load_extension()
    if ext is None:
        raise RuntimeError("matching native extension is required in the selected repository")
    reports, generation = [], 0
    calls = dict.fromkeys(("load", "analyze", "plan", "balance", "dither", "auto_ev"), 0)
    def counted(name, function):
        # Do not use Mock(wraps=...): call_args would retain every full RAW.
        def run(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)
        return run
    common = {"input": str(args.source), "decoder": args.decoder, "wb": "camera",
              "gamut": "p3", "format": "sdr", "ev": 0., "evAuto": False,
              "previewClient": "cache-workset", "selectionEpoch": 1,
              "previewSession": "cache-workset:1", "includeMetrics": False}

    with tempfile.TemporaryDirectory(prefix="agx-gui-workset-") as directory, \
         mock.patch.dict(os.environ, {"DNGSCAN_PREVIEW_CACHE_DIR": directory}), \
         mock.patch.object(dg, "load_raw", counted("load", dg.load_raw)), \
         mock.patch.object(dg, "analyze", counted("analyze", dg.analyze)), \
         mock.patch.object(dg, "build_render_plan", counted("plan", dg.build_render_plan)), \
         mock.patch.object(dg, "compute_auto_ev", counted("auto_ev", dg.compute_auto_ev)), \
         mock.patch.object(dg, "deterministic_dither_planes", counted("dither", dg.deterministic_dither_planes)), \
         mock.patch.object(pc.PreviewCache, "_build_balance", staticmethod(counted("balance", pc.PreviewCache._build_balance))):
        def counts():
            return dict(calls)

        def snapshot():
            store = service.PREVIEW_STORE
            result = store.memory_snapshot() if hasattr(store, "memory_snapshot") else {"entries": len(store.entries)}
            result["balance_children"] = sum(len(e._balance_cache) for e in store.entries.values())
            result["scheduler"] = service.SCHEDULER.snapshot()
            if hasattr(pc, "DISK_WRITER"):
                result["disk_writer"] = pc.DISK_WRITER.snapshot()
            return result

        def stage(name, operation):
            before = counts()
            wall, cpu = time.perf_counter(), time.process_time()
            result = operation()
            elapsed, consumed = time.perf_counter() - wall, time.process_time() - cpu
            if not result.get("ok") or result.get("superseded"):
                raise RuntimeError(f"{name} did not finish: {result}")
            report = {"name": name, "wall_s": elapsed, "cpu_s": consumed,
                      "calls": {k: v - before[k] for k, v in counts().items()}, "cache": snapshot()}
            if "preview" in result:
                report["preview_sha256"] = hashlib.sha256(result["preview"].encode("ascii")).hexdigest()
            reports.append(report)

        def preview(wb, auto=False):
            nonlocal generation
            generation += 1
            return service.run_preview({**common, "wb": wb, "generation": generation, "evAuto": auto})

        stage("cold_prepare", lambda: service.prepare_preview(common))
        stage("camera_first", lambda: preview("camera"))
        stage("camera_auto_first", lambda: preview("camera", auto=True))
        stage("camera_auto_repeat", lambda: preview("camera", auto=True))
        for round_no in range(2):
            for wb in dg.WB_CHOICES:
                stage(f"wb{round_no + 1}:{wb}", lambda wb=wb: preview(wb))
        if hasattr(pc, "DISK_WRITER") and not pc.DISK_WRITER.flush(30):
            raise RuntimeError("disk writer did not drain")
        totals = counts()
        service.PREVIEW_STORE.clear_memory()
        stage("disk_prepare", lambda: service.prepare_preview(common))
        stage("disk_camera", lambda: preview("camera"))
        if hasattr(pc, "DISK_WRITER") and not pc.DISK_WRITER.flush(30):
            raise RuntimeError("disk writer did not drain")

    source_after = args.source.stat()
    identity_after = (source_after.st_dev, source_after.st_ino, source_after.st_size,
                      source_after.st_mtime_ns, source_after.st_ctime_ns)
    if identity_before != identity_after:
        raise RuntimeError("source changed during benchmark")
    result = {"repo": str(args.repo), "source": str(args.source), "decoder": args.decoder,
              "commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
              "native_abi": ext.native_abi_version() if ext is not None else None,
              "cache_version": pc.PREVIEW_CACHE_VERSION,
              "workset_calls": totals, "stages": reports,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024)}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "workset_calls": totals, "peak_rss_mib": result["peak_rss_mib"]}))


if __name__ == "__main__":
    main()
