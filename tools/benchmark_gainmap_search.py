#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure HDR HEIF auto search against immutable .npy formation masters.

Run each checkout in a fresh process with the same --base/--hdr/--headroom.
--reference-report accepts this tool's JSON or the earlier heif-profile.json;
--reference-file compares compressed rendition identity as well as file SHA.
--require-match fails on changed selections, decisions, metrics or payloads.
Imports, input hashing, the small capability probe and comparison are excluded
from search timing. --dry-run validates the small array headers and arguments
without loading a codec, hashing full masters, or encoding anything.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from unittest.mock import patch


def json_safe(value):
    """Preserve byte digests when serializing the structured HEIF signature."""
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path, signature_fn):
    path = Path(path)
    signature = json_safe(signature_fn(path, "heic"))
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return {
        "path": str(path.resolve()), "bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "payload_signature_schema": "encoded-content-signature-json-v1",
        "payload_signature_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "payload_signature": signature,
    }


class CallTrace:
    """Subtract directly nested timed calls; keep a separate stack per thread."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.events = []
        self.local = threading.local()
        self.lock = threading.Lock()
        self.counter = 0

    def wrap(self, function, name, category):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            with self.lock:
                sequence = self.counter
                self.counter += 1
            event = {"sequence": sequence, "function": name, "category": category,
                     "thread_id": threading.get_ident(), "children_s": 0.0}
            if category == "encoder":
                event.update(
                    category="auxiliary_encode" if kwargs.get("auxiliary") else "primary_encode",
                    quality=args[2] if len(args) > 2 else kwargs.get("quality"),
                    chroma=args[3] if len(args) > 3 else kwargs.get("chroma", "420"),
                    shape=list(args[0].shape) if args else None,
                    bit_depth=kwargs.get("bit_depth", 10),
                    preset=kwargs.get("preset", "slow"), tune=kwargs.get("tune", "ssim"),
                )
            stack = getattr(self.local, "stack", None)
            if stack is None:
                stack = self.local.stack = []
            start = self.clock()
            stack.append(event)
            try:
                return function(*args, **kwargs)
            except BaseException as exc:
                event["error"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                elapsed = self.clock() - start
                stack.pop()
                event.update(wall_s=elapsed, self_s=elapsed - event.pop("children_s"))
                if stack:
                    stack[-1]["children_s"] += elapsed
                with self.lock:
                    self.events.append(event)
        return measured

    def summary(self):
        stages = defaultdict(lambda: {"count": 0, "wall_s": 0.0, "self_s": 0.0})
        for event in self.events:
            stage = stages[event["category"]]
            stage["count"] += 1
            for key in ("wall_s", "self_s"):
                stage[key] += event[key]
        return dict(stages)


def install_trace(stack, trace, gainmap, heif_encoder, heif_gainmap, auto_encode):
    hooks = (
        (heif_encoder, "encode", "encoder"),
        (heif_encoder, "read_rgb_item", "auxiliary_readback"),
        (gainmap, "read_primary_rgb_u8", "sdr_readback"),
        (gainmap, "_read_expanded_hdr_rgba_half", "hdr_readback"),
        (gainmap, "_base_roundtrip_error_arrays", "sdr_metrics"),
        (gainmap, "_roundtrip_error_arrays", "hdr_metrics"),
        (auto_encode, "coding_metrics", "coding_metrics"),
        (gainmap, "coding_metrics", "coding_metrics"),
        (gainmap, "inspect_gainmap_file", "inspect_container"),
        (heif_gainmap, "replace_primary", "replace_primary"),
        (heif_gainmap, "replace_image_item", "replace_image_item"),
        (heif_gainmap, "iso_gainmap_item", "find_auxiliary"),
    )
    installed = []
    for module, name, category in hooks:
        if hasattr(module, name):
            stack.enter_context(patch.object(module, name, trace.wrap(getattr(module, name), name, category)))
            installed.append(f"{module.__name__}.{name}")
    return installed


SELECTED_KEYS = ("delivery_quality", "delivery_chroma_requested", "gainmap_encoding_quality")


def selected(delivery):
    return {key: delivery.get(key) for key in SELECTED_KEYS}


def compare_reports(current, reference):
    """Exact selected/decision/common-metric checks; early reject omissions allowed.

    Rejected candidates may omit HDR metrics/bytes after an equivalent earlier
    gate, but accepted candidates must retain every reference metric. Rejection
    prose and absolute output paths are diagnostics, not rendition identity.
    """
    left, right = current["delivery"], reference["delivery"]
    failures, omitted = [], []
    if "headroom_ev" in reference and current.get("headroom_ev") != reference["headroom_ev"]:
        failures.append("input headroom changed")
    for name, item in reference.get("inputs", {}).items():
        for key in ("shape", "dtype", "file_sha256"):
            if key in item and current.get("inputs", {}).get(name, {}).get(key) != item[key]:
                failures.append(f"input {name} {key} changed")
    signature = reference.get("artifact", {}).get("payload_signature")
    if signature is not None and current.get("artifact", {}).get("payload_signature") != signature:
        failures.append("compressed rendition payload/properties changed")
    if selected(left) != selected(right):
        failures.append("selected encoding changed")
    for key, expected in right.items():
        if key.startswith(("coding_", "base_", "block_", "highlight_")) or key in (
            "chroma_error", "relative_error", "median_relative_error", "p95_relative_error",
            "p99_relative_error", "p999_relative_error", "file_headroom_ev", "headroom",
            "width", "height", "bit_depth", "gainmap_width", "gainmap_height", "profile",
            "gainmap_pixel_format", "chroma_subsampling",
        ):
            if left.get(key) != expected:
                failures.append(f"selected {key} changed")
    actual_attempts, expected_attempts = left.get("auto_attempts", []), right.get("auto_attempts", [])
    if len(actual_attempts) != len(expected_attempts):
        failures.append("attempt count changed")
    for index, (actual, expected) in enumerate(zip(actual_attempts, expected_attempts)):
        for key in ("quality", "chroma", "gainmap_quality", "accepted"):
            if actual.get(key) != expected.get(key):
                failures.append(f"attempt {index} {key} changed")
        if "bytes" in expected:
            if "bytes" not in actual:
                if expected.get("accepted"):
                    failures.append(f"attempt {index} bytes missing")
            elif actual["bytes"] != expected["bytes"]:
                failures.append(f"attempt {index} bytes changed")
        for key, value in expected.get("metrics", {}).items():
            if (key not in actual.get("metrics", {}) and not expected.get("accepted")
                    and key in ("chroma_error", "block_p95_luma_error", "highlight_max_luma_error")):
                omitted.append({"attempt": index, "metric": key})
            elif actual.get("metrics", {}).get(key) != value:
                failures.append(f"attempt {index} metric {key} changed")
    return {"matches": not failures, "failures": failures, "early_reject_omitted_metrics": omitted}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    result.add_argument("--base", type=Path, required=True, help="HxWx3 uint8 SDR .npy")
    result.add_argument("--hdr", type=Path, required=True, help="HxWx4 float16 HDR .npy")
    result.add_argument("--out", type=Path, required=True, help="new .heic/.heif candidate output")
    result.add_argument("--headroom", type=float, required=True, help="display capacity in EV, not linear gain")
    result.add_argument("--report", type=Path, help="defaults to OUT.benchmark.json")
    result.add_argument("--label", default="")
    result.add_argument("--native-mode", choices=("0", "1", "auto"), default="1")
    result.add_argument("--reference-report", type=Path)
    result.add_argument("--reference-file", type=Path)
    result.add_argument("--require-match", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def validate(args):
    if not math.isfinite(args.headroom) or args.headroom <= 0:
        raise ValueError("--headroom must be positive finite EV")
    args.repo = args.repo.resolve()
    if not (args.repo / "dngscan" / "gainmap.py").is_file():
        raise ValueError("--repo must contain dngscan/gainmap.py")
    if args.out.suffix.lower() not in (".heic", ".heif"):
        raise ValueError("--out must end in .heic or .heif")
    args.out = args.out.resolve()
    args.report = (args.report or args.out.with_suffix(".benchmark.json")).resolve()
    inputs = [args.base, args.hdr, args.reference_report, args.reference_file]
    for path in inputs:
        if path is not None and not path.is_file():
            raise ValueError(f"input does not exist: {path}")
        if path is not None and path.resolve() in (args.out, args.report):
            raise ValueError("outputs must not overwrite masters or reference artifacts")
    if args.out == args.report:
        raise ValueError("report and encoded output must be different paths")
    if not args.overwrite and any(p.exists() for p in (args.out, args.report)):
        raise ValueError("output/report already exists; select new paths or pass --overwrite")
    if args.require_match and not (args.reference_file or args.reference_report):
        raise ValueError("--require-match needs a reference report or file")


def git_info(repo):
    def run(*arguments):
        result = subprocess.run(["git", "-C", str(repo), *arguments], text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def main(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    try:
        validate(args)
    except ValueError as exc:
        arg_parser.error(str(exc))
    os.environ["DNGSCAN_FAST"] = args.native_mode
    sys.path.insert(0, str(args.repo))
    import numpy as np

    base = np.load(args.base, mmap_mode="r", allow_pickle=False)
    hdr = np.load(args.hdr, mmap_mode="r", allow_pickle=False)
    if base.dtype != np.uint8 or base.ndim != 3 or base.shape[-1] != 3 or min(base.shape[:2]) < 1:
        arg_parser.error("base must be nonempty HxWx3 uint8")
    if hdr.dtype != np.float16 or hdr.shape != base.shape[:2] + (4,):
        arg_parser.error("hdr must be matching HxWx4 float16")
    reference_report = None
    if args.reference_report:
        try:
            reference_report = json.loads(args.reference_report.read_text())
            if not isinstance(reference_report.get("delivery"), dict):
                raise ValueError("reference report must contain a delivery object")
        except (OSError, ValueError, AttributeError) as exc:
            arg_parser.error(str(exc))
    inputs = {name: {"path": str(path.resolve()), "shape": list(array.shape), "dtype": str(array.dtype)}
              for name, path, array in (("base", args.base, base), ("hdr", args.hdr, hdr))}
    if args.dry_run:
        print(json.dumps({"dry_run": True, "repo": str(args.repo), "inputs": inputs,
                          "out": str(args.out), "report": str(args.report), "headroom_ev": args.headroom}))
        return 0

    from dngscan import _fast, auto_encode, gainmap, heif_encoder, heif_gainmap
    from dngscan.delivery import resolve_delivery_profile
    from dngscan.delivery_integrity import encoded_content_signature
    if Path(gainmap.__file__).resolve().parent.parent != args.repo:
        raise RuntimeError("a different dngscan checkout is already imported; use a fresh process")
    record = {
        "schema_version": 1, "label": args.label, "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs, "headroom_ev": args.headroom,
        "environment": {"repo": str(args.repo), **git_info(args.repo), "python": sys.version,
                        "executable": sys.executable, "platform": platform.platform(),
                        "numpy": np.__version__, "cpus": os.cpu_count(),
                        "native_mode": args.native_mode, "native_available": _fast.available(),
                        "env": {key: os.environ.get(key) for key in (
                            "DNGSCAN_FAST_SKIP", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
        "timing_scope": "Only auto search; excludes imports, master hashes, capability warmup, artifact comparison. "
                        "Self times subtract nested same-thread hooks; concurrent self totals are not wall-time shares.",
    }
    ext = _fast._load_extension()
    record["environment"]["native_abi"] = ext.native_abi_version() if ext else None
    for name, path in (("base", args.base), ("hdr", args.hdr)):
        inputs[name].update(file_sha256=file_sha256(path), bytes=path.stat().st_size)
    trace = CallTrace()
    code = 0
    try:
        if args.native_mode == "1" and ext is None:
            raise RuntimeError("strict native benchmark requires the matching compiled extension")
        started = time.perf_counter()
        ok, reason = gainmap.apple_gainmap_backend_status()
        record["capability"] = {"available": ok, "reason": reason, "wall_s": time.perf_counter() - started}
        if not ok:
            raise RuntimeError(reason)
        if not heif_encoder.available():
            raise RuntimeError("tunable x265 HEIF backend unavailable; auto search would not be comparable")
        with ExitStack() as stack:
            record["trace_hooks"] = install_trace(stack, trace, gainmap, heif_encoder, heif_gainmap, auto_encode)
            start, cpu = time.perf_counter(), time.process_time()
            try:
                record["delivery"] = gainmap.write_apple_gainmap_file(
                    base, hdr, args.out, args.headroom,
                    delivery=resolve_delivery_profile("auto", container="heic"))
            finally:
                record.update(total_s=time.perf_counter() - start, cpu_s=time.process_time() - cpu)
        record["selected"] = selected(record["delivery"])
        record["attempts"] = record["delivery"].get("auto_attempts", [])
        record["artifact"] = artifact_record(args.out, encoded_content_signature)
        comparison = {}
        if reference_report is not None:
            comparison["report"] = compare_reports(record, reference_report)
        if args.reference_file:
            reference = artifact_record(args.reference_file, encoded_content_signature)
            comparison["file"] = {
                "reference": reference,
                "payload_matches": reference["payload_signature"] == record["artifact"]["payload_signature"],
                "file_sha_matches": reference["file_sha256"] == record["artifact"]["file_sha256"],
                "bytes_match": reference["bytes"] == record["artifact"]["bytes"],
            }
        record["comparison"] = comparison
        if args.require_match and (
            not comparison.get("report", {}).get("matches", True)
            or not comparison.get("file", {}).get("payload_matches", True)
        ):
            code = 2
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        code = 1
    finally:
        record["events"] = sorted(trace.events, key=lambda event: event["sequence"])
        record["stages"] = trace.summary()
        record["unattributed_s"] = record.get("total_s", 0.0) - sum(v["self_s"] for v in record["stages"].values())
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(json_safe(record), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(json_safe({key: record.get(key) for key in (
        "selected", "total_s", "cpu_s", "stages", "comparison", "error")}), ensure_ascii=False))
    print(f"report: {args.report}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
