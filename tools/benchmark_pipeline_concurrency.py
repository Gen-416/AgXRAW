#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure real GUI prepare/preview/isolated-export overlap in a fresh process.

Warm source A, then concurrently prepare a different cold source B, render A's
exposure changes from 0 to +1.5 EV, and export A with default AgX/automatic
exposure and encoding. More frames use finer unique steps across the same EV
range, keeping long runs from turning into a sequence of clipped-white frames.
An external stdlib-only monitor samples the coordinator and its descendant
processes; the GUI's existing class quotas and inner budgets are unchanged.
Use --repo separately for baseline/development checkouts and --compare for the
second run. Alternate run order when measuring more than one pair.

RSS is a sampled sum of resident pages (shared pages can be counted twice), not
physical footprint or a guaranteed high-water mark. Samples and actual sampling
intervals are retained. Input hashes, response hashes and output hashes are
outside service-call timers. This benchmarks service paths, not HTTP/browser
paint latency. The comparison retains every service response field except the
two cache-hit flags; data URLs/files become hashes and temporary paths become
relative paths. It does not replace full RAW/master-array identity benchmarks.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time


SCHEMA = 2


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {"count": len(values), "min": values[0],
            "median": statistics.median(values),
            "p95": values[max(0, math.ceil(.95 * len(values)) - 1)], "max": values[-1]}


def file_record(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        before = os.fstat(source.fileno())
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(source.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise RuntimeError(f"file changed while hashing: {path}")
    return {"bytes": after.st_size, "sha256": digest.hexdigest()}


class _ProcTaskInfo(ctypes.Structure):
    # macOS sys/proc_info.h, PROC_PIDTASKINFO. All 18 fields are included so
    # proc_pidinfo's reported structure length can be checked exactly.
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("virtual_size", "resident_size", "total_user", "total_system",
                 "threads_user", "threads_system")] + [
                    (name, ctypes.c_int32) for name in
                    ("policy", "faults", "pageins", "cow_faults", "messages_sent",
                     "messages_received", "syscalls_mach", "syscalls_unix", "csw",
                     "threadnum", "numrunning", "priority")]


class ProcessSampler:
    """No subprocess launches or GUI imports in the sampling loop."""
    def __init__(self):
        if sys.platform == "darwin":
            self.backend = "libproc:proc_listchildpids+PROC_PIDTASKINFO"
            self.lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
            self.lib.proc_listchildpids.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
            self.lib.proc_listchildpids.restype = ctypes.c_int
            self.lib.proc_pidinfo.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                            ctypes.c_void_p, ctypes.c_int)
            self.lib.proc_pidinfo.restype = ctypes.c_int
        elif sys.platform.startswith("linux"):
            self.backend = "procfs:task/children+status"
        else:
            raise RuntimeError("process-tree sampling supports macOS and Linux")

    def children(self, pid):
        if sys.platform == "darwin":
            capacity = 64
            while capacity <= 65536:
                result = (ctypes.c_int * capacity)()
                length = self.lib.proc_listchildpids(pid, result, ctypes.sizeof(result))
                if length < 0:
                    return []
                # Unlike proc_listpids, this convenience API returns a PID
                # count; its buffer-size argument is still measured in bytes.
                if length < capacity:
                    return [int(n) for n in result[:length] if n > 0]
                capacity *= 2
            raise RuntimeError("process tree exceeds sampler capacity")
        found = set()
        # Linux children belong to the thread that spawned them, not necessarily
        # the thread-group leader (export is launched by a service worker).
        try:
            for task in Path(f"/proc/{pid}/task").iterdir():
                try:
                    found.update(map(int, (task / "children").read_text().split()))
                except (OSError, ValueError):
                    pass
        except OSError:
            pass
        return sorted(found)

    def process(self, pid):
        if sys.platform == "darwin":
            info = _ProcTaskInfo()
            length = self.lib.proc_pidinfo(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
            if length != ctypes.sizeof(info):
                return None
            return {"pid": pid, "rss_bytes": info.resident_size, "threads": info.threadnum}
        try:
            fields = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines())
            return {"pid": pid, "rss_bytes": int(fields["VmRSS"].split()[0]) * 1024,
                    "threads": int(fields["Threads"])}
        except (OSError, ValueError, KeyError):
            return None

    def sample(self, root_pid):
        started = time.monotonic()
        remaining, seen, processes, missing = [root_pid], set(), [], []
        while remaining:
            pid = remaining.pop()
            if pid in seen:
                continue
            seen.add(pid)
            remaining.extend(self.children(pid))
            value = self.process(pid)
            if value is None:
                missing.append(pid)
            else:
                processes.append(value)
        return {"monotonic_s": started, "read_s": time.monotonic() - started,
                "rss_bytes": sum(p["rss_bytes"] for p in processes),
                "threads": sum(p["threads"] for p in processes),
                "processes": sorted(processes, key=lambda p: p["pid"]), "missing_pids": missing}


def response_identity(value, workdir):
    """Keep all output/decision fields, explicitly separating cache telemetry."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in ("cache_hit", "pixel_cache_hit"):
                continue
            # GUI previews normally contain bare base64, not a data URL.
            # Keep the same exact bytes identity without a >100 MiB report
            # for a 72-frame run. This still runs after the timed window.
            if key == "preview" and isinstance(item, str) and not item.startswith("data:"):
                data = base64.b64decode(item, validate=True)
                result[str(key)] = {"encoding": "base64", "bytes": len(data),
                                    "sha256": hashlib.sha256(data).hexdigest()}
            else:
                result[str(key)] = response_identity(item, workdir)
        return result
    if isinstance(value, (list, tuple)):
        return [response_identity(item, workdir) for item in value]
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value:
            kind, payload = value.split(",", 1)
            data = base64.b64decode(payload, validate=True)
            return {"data_url_type": kind, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return value.replace(str(workdir) + os.sep, "<temporary>/")
    return value


def _checked(result, job):
    if not result.get("ok") or result.get("superseded"):
        raise RuntimeError(f"{job} failed or was superseded: {result}")
    return result


def _worker(args):
    # The parent remains independent of either checkout. Spawned export workers
    # inherit this sys.path and environment before importing their service target.
    sys.path.insert(0, str(args.repo))
    from dngscan import _fast, cpu_budget
    from dngscan.gui import preview_cache as cache
    from dngscan.gui import service
    from dngscan.gui import scheduler
    import numpy as np

    extension = _fast._load_extension()
    if extension is None:
        raise RuntimeError("matching native extension is required in the selected repo")
    workdir = args.workdir
    original_stats = {path: path.stat() for path in (args.source_a, args.source_b)}
    identity = {"source_a": file_record(args.source_a), "source_b": file_record(args.source_b),
                "settings": {"frames": args.frames, "preview_period_ms": args.preview_period_ms,
                             "decoder": args.decoder, "format": args.format, "seed": 371,
                             "exposure_values": [1.5 * (i + 1) / args.frames for i in range(args.frames)]}}
    events, event_lock, job_local = [], threading.Lock(), threading.local()
    original_slot = service.SCHEDULER.slot

    @contextmanager
    def measured_slot(kind):
        queued = time.monotonic()
        with original_slot(kind):
            acquired = time.monotonic()
            try:
                yield
            finally:
                finished = time.monotonic()
                with event_lock:
                    events.append({"kind": kind, "job": getattr(job_local, "name", "unknown"),
                                   "queued_s": queued, "started_s": acquired, "finished_s": finished,
                                   "queue_s": acquired - queued, "execute_s": finished - acquired})

    service.SCHEDULER.slot = measured_slot
    cache.PREVIEW_STORE.clear_memory()

    def snapshots():
        value = {"scheduler": service.SCHEDULER.snapshot()}
        if hasattr(cache.PREVIEW_STORE, "memory_snapshot"):
            value["proxy_memory"] = cache.PREVIEW_STORE.memory_snapshot()
        if hasattr(cache, "DISK_WRITER"):
            value["disk_writer"] = cache.DISK_WRITER.snapshot()
        return value

    def flush():
        if hasattr(cache, "DISK_WRITER") and not cache.DISK_WRITER.flush(timeout=60.):
            raise RuntimeError("preview disk writer did not drain within 60 seconds")

    def call(name, fn, params):
        job_local.name = name
        started = time.monotonic()
        result = _checked(fn(params), name)
        finished = time.monotonic()
        return result, {"started_s": started, "finished_s": finished, "wall_s": finished - started}

    common = {"input": str(args.source_a), "decoder": args.decoder, "toneCore": "agx",
              "filmOpticsSeed": 371, "includeMetrics": False,
              "previewSession": "concurrency-a", "previewClient": "concurrency-a",
              "selectionEpoch": 1, "format": args.format}
    warm_prepare, warm_prepare_time = call("warm_prepare_a", service.prepare_preview, common)
    warm_preview, warm_preview_time = call("warm_preview_a", service.run_preview,
                                           {**common, "generation": 1, "ev": 0.0})
    flush()
    identity["warm_prepare"] = response_identity(warm_prepare, workdir)
    identity["warm_preview"] = response_identity(warm_preview, workdir)
    del warm_prepare, warm_preview
    before = snapshots()
    gate = threading.Barrier(4)
    workload_start = [None]

    def cold_prepare():
        gate.wait()
        return call("cold_prepare_b", service.prepare_preview,
                    {**common, "input": str(args.source_b), "previewSession": "concurrency-b",
                     "previewClient": "concurrency-b"})

    def previews():
        gate.wait()
        responses, timings = [], []
        for i, ev in enumerate(identity["settings"]["exposure_values"]):
            scheduled = workload_start[0] + i * args.preview_period_ms / 1000.
            delay = scheduled - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            response, timing = call(f"preview_a_{i}", service.run_preview,
                                    {**common, "generation": i + 2, "ev": ev})
            timing.update(index=i, ev=ev, scheduled_s=scheduled,
                          lateness_s=max(0., timing["started_s"] - scheduled),
                          cache_hit=response.get("cache_hit"),
                          pixel_cache_hit=response.get("pixel_cache_hit"))
            # Hold only the small service payloads until all three jobs finish;
            # hashing/serialization never runs in another job's timed window.
            responses.append(response)
            timings.append(timing)
        return responses, timings

    def export():
        gate.wait()
        return call("export_a", service.run_export_isolated,
                    {**common, "outdir": str(workdir / "export")})

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="concurrency-bench") as pool:
        futures = [pool.submit(fn) for fn in (cold_prepare, previews, export)]
        workload_start[0] = time.monotonic()
        gate.wait()
        # Executor exit joins every task even if result() raises.
        (cold, cold_time), (frames, frame_times), (exported, export_time) = [f.result() for f in futures]
    workload_end = time.monotonic()
    after = snapshots()
    flush()
    identity.update(cold_prepare=response_identity(cold, workdir),
                    previews=[response_identity(frame, workdir) for frame in frames],
                    export=response_identity(exported, workdir),
                    artifacts=[{"relative_path": str(Path(path).relative_to(workdir)), **file_record(path)}
                               for path in exported.get("saved", [])])
    if not identity["artifacts"]:
        raise RuntimeError("export returned no saved artifact")
    for path, before_stat in original_stats.items():
        after_stat = path.stat()
        if any(getattr(before_stat, key) != getattr(after_stat, key) for key in
               ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
            raise RuntimeError(f"input changed during benchmark: {path}")
    git = lambda *params: subprocess.check_output(["git", "-C", str(args.repo), *params], text=True).strip()
    overlap_slots = {}
    for kind in scheduler._QUOTAS:
        selected = [event for event in events if event["kind"] == kind and
                    event["queued_s"] >= workload_start[0]]
        overlap_slots[kind] = {"calls": len(selected),
                               "queue_s": sum(e["queue_s"] for e in selected),
                               "execute_s": sum(e["execute_s"] for e in selected)}
    record = {"schema": SCHEMA, "repo": str(args.repo), "commit": git("rev-parse", "HEAD"),
              "git_status": git("status", "--short"),
              "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
              "sources": {"a": str(args.source_a), "b": str(args.source_b)},
              "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
              "native_abi": int(extension.native_abi_version()), "cpus": os.cpu_count(),
              "cpu_budget_total": cpu_budget.TOTAL, "class_quotas": dict(scheduler._QUOTAS),
              "environment": {key: val for key, val in os.environ.items()
                              if key.startswith("DNGSCAN_") or key in
                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
              "identity": identity, "warm": {"prepare": warm_prepare_time, "preview": warm_preview_time},
              "window": {"started_s": workload_start[0], "finished_s": workload_end,
                         "wall_s": workload_end - workload_start[0]},
              "jobs": {"cold_prepare": cold_time, "export": export_time, "previews": frame_times},
              "preview_wall_s": distribution([v["wall_s"] for v in frame_times]),
              "preview_lateness_s": distribution([v["lateness_s"] for v in frame_times]),
              "overlap_slots": overlap_slots,
              "slot_events": sorted(events, key=lambda value: value["queued_s"]),
              "snapshots": {"before_overlap": before, "after_overlap": after, "after_flush": snapshots()}}
    with (workdir / "worker.json").open("x") as output:
        json.dump(record, output, indent=2, allow_nan=False)
    return 0


def _sample_summary(samples, window):
    active = [row for row in samples if window["started_s"] <= row["monotonic_s"] <= window["finished_s"]]
    return {"samples": len(active), "tree_rss_peak_bytes": max((r["rss_bytes"] for r in active), default=None),
            "tree_threads_peak": max((r["threads"] for r in active), default=None),
            "processes_peak": max((len(r["processes"]) for r in active), default=None),
            "interval_s": distribution([b["monotonic_s"] - a["monotonic_s"] for a, b in zip(active, active[1:])]),
            "read_s": distribution([r["read_s"] for r in active]),
            "missing_pid_observations": sum(len(r["missing_pids"]) for r in active)}


def _performance_comparison(record, previous):
    def metrics(report):
        return {"window_s": report["window"]["wall_s"],
                "cold_prepare_s": report["jobs"]["cold_prepare"]["wall_s"],
                "export_s": report["jobs"]["export"]["wall_s"],
                "preview_p50_s": report["preview_wall_s"]["median"],
                "preview_p95_s": report["preview_wall_s"]["p95"],
                "tree_rss_peak_bytes": report["sampler"]["overlap"]["tree_rss_peak_bytes"],
                "tree_threads_peak": report["sampler"]["overlap"]["tree_threads_peak"]}
    before, after = metrics(previous), metrics(record)
    return {key: {"baseline": before[key], "current": after[key],
                  "change_pct": ((after[key] / before[key] - 1.) * 100.
                                 if before[key] and after[key] is not None else None)}
            for key in before}


def _parent(args, previous):
    sampler = ProcessSampler()
    with tempfile.TemporaryDirectory(prefix="agxraw-concurrency-") as directory:
        workdir = Path(directory)
        (workdir / "export").mkdir()
        env = dict(os.environ, DNGSCAN_FAST="1", DNGSCAN_FAST_SKIP="",
                   DNGSCAN_PREVIEW_CACHE_DIR=str(workdir / "cache"))
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--workdir", str(workdir),
                   "--repo", str(args.repo), "--source-a", str(args.source_a), "--source-b", str(args.source_b),
                   "--out", str(args.out), "--decoder", args.decoder, "--format", args.format,
                   "--frames", str(args.frames), "--preview-period-ms", str(args.preview_period_ms)]
        samples = []
        with (workdir / "worker.log").open("w") as log:
            process = subprocess.Popen(command, cwd=args.repo, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + args.timeout
            try:
                while process.poll() is None:
                    tick = time.monotonic()
                    samples.append(sampler.sample(process.pid))
                    if tick > deadline:
                        raise TimeoutError(f"benchmark exceeded {args.timeout:g} seconds")
                    time.sleep(max(0., args.sample_ms / 1000. - (time.monotonic() - tick)))
                if process.returncode:
                    raise RuntimeError(f"benchmark worker exited {process.returncode}")
            except BaseException as exc:
                # This process group belongs exclusively to this fresh benchmark.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
                if isinstance(exc, Exception):
                    log.flush()
                    tail = (workdir / "worker.log").read_text(errors="replace")[-10000:]
                    raise RuntimeError(f"{exc}\n{tail}") from exc
                raise
        record = json.loads((workdir / "worker.json").read_text())
        record["sampler"] = {"backend": sampler.backend, "requested_interval_ms": args.sample_ms,
                             "scope": "coordinator and recursive descendants; monitor excluded",
                             "rss_semantics": "sampled sum; shared resident pages may be counted more than once",
                             "overlap": _sample_summary(samples, record["window"]),
                             "whole_run": _sample_summary(samples, {"started_s": -math.inf, "finished_s": math.inf}),
                             "samples": samples}
        if previous is not None:
            record["comparison"] = {"report": str(args.compare), "commit": previous.get("commit"),
                                    "identity_equal": record["identity"] == previous["identity"],
                                    "performance": _performance_comparison(record, previous)}
        with args.out.open("x") as output:
            json.dump(record, output, indent=2, allow_nan=False)
        summary = {"out": str(args.out), "commit": record["commit"], "window_s": record["window"]["wall_s"],
                   "cold_prepare_s": record["jobs"]["cold_prepare"]["wall_s"],
                   "export_s": record["jobs"]["export"]["wall_s"], "preview_wall_s": record["preview_wall_s"],
                   "overlap_slots": record["overlap_slots"],
                   "sampled_overlap": record["sampler"]["overlap"], "comparison": record.get("comparison")}
        print(json.dumps(summary, sort_keys=True))
        return 2 if previous is not None and not record["comparison"]["identity_equal"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-a", type=Path, required=True, help="warm preview/export RAW (e.g. Sigma)")
    parser.add_argument("--source-b", type=Path, required=True, help="different cold prepare RAW (e.g. Sony)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--decoder", choices=("libraw", "coreimage"), default="libraw")
    parser.add_argument("--format", choices=("sdr", "sdr-heic", "ultrahdr", "ultrahdr-heic"), default="sdr")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--preview-period-ms", type=float, default=100.)
    parser.add_argument("--sample-ms", type=float, default=30.)
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--workdir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.frames < 1 or not math.isfinite(args.preview_period_ms) or args.preview_period_ms < 0 or
            not math.isfinite(args.sample_ms) or args.sample_ms < 20 or
            not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("frames/timeout must be positive, cadence nonnegative and sample interval at least 20 ms")
    for name in ("repo", "source_a", "source_b", "out"):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    if not (args.repo / "dngscan/gui/service.py").is_file():
        parser.error("invalid --repo")
    if not args.source_a.is_file() or not args.source_b.is_file():
        parser.error("both source files must exist")
    if args.source_a.samefile(args.source_b):
        parser.error("source A and B must be different files")
    if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
        parser.error("output must be a new file in an existing directory")
    previous = None
    if args.compare:
        try:
            previous = json.loads(args.compare.read_text())
            if previous.get("schema") != SCHEMA or not isinstance(previous.get("identity"), dict):
                raise ValueError("comparison must be a concurrency benchmark report with matching schema")
            _performance_comparison(previous, previous)
        except (OSError, ValueError, AttributeError, KeyError, TypeError) as exc:
            parser.error(str(exc))
    if args.worker:
        if args.workdir is None or not args.workdir.is_dir():
            parser.error("internal worker requires a work directory")
        return _worker(args)
    return _parent(args, previous)


if __name__ == "__main__":
    raise SystemExit(main())
