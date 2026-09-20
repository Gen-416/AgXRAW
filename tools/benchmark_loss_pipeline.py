#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare loss-kernel ablation with all other native kernels held constant.

Each invocation is a fresh process. --reference disables only the two loss
kernels. --source runs full-resolution default AgX SDR/HDR without encoding;
--synthetic WIDTH HEIGHT measures crop and in-place merge separately. Timing
excludes hashes and input generation. RSS is the process high-water mark, not
live allocations. Supply --compare REPORT to require exact buffer/decision
identity. Alternate reference/native order across at least three repetitions.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

LOSS_KERNELS = ("crop_loss_footprint", "merge_processing_loss_f16_inplace")


def array_record(array):
    import numpy as np
    if array is None:
        return None
    array = np.asarray(array)
    digest = hashlib.sha256()
    # A bounded copy for transposed/strided arrays; no full-raster diagnostic copy.
    if array.ndim == 0:
        digest.update(array.tobytes())
    else:
        for start in range(0, len(array), 64):
            digest.update(np.ascontiguousarray(array[start:start + 64]).tobytes())
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": digest.hexdigest()}


def json_value(value):
    import numpy as np
    if dataclasses.is_dataclass(value):
        return {f.name: json_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return array_record(value)
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def pipeline(args, record):
    from dngscan import _fast, dng_opcodes, raw_io
    from dngscan.analysis import analyze
    from dngscan.auto_ev import compute_auto_ev
    from dngscan.scene_scale import with_intent_exposure
    from dngscan.tone import build_render_plan
    from dngscan.render import render_output_u8
    from dngscan.hdr_agx import render_ultrahdr_agx_pair
    from dngscan.hdr_agx_plan import compile_hdr_agx_plan
    from dngscan.models import HdrDisplayTarget

    stages, calls = {}, {}
    def measured(name, fn):
        def invoke(*a, **kw):
            start = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                count = calls.setdefault(name, {"count": 0, "wall_s": 0.})
                count["count"] += 1
                count["wall_s"] += time.perf_counter() - start
        return invoke
    kernel = _fast.kernel
    def observed_kernel(name):
        fn = kernel(name)
        return measured("native." + name, fn) if fn is not None and name in LOSS_KERNELS else fn
    def stage(name, fn):
        start = time.perf_counter()
        result = fn()
        stages[name] = time.perf_counter() - start
        return result

    with ExitStack() as stack:
        for module, name in ((dng_opcodes, "_fractional_crop_loss"), (raw_io, "_merge_processing_loss")):
            stack.enter_context(patch.object(module, name, measured(name, getattr(module, name))))
        stack.enter_context(patch.object(_fast, "kernel", observed_kernel))
        bundle = stage("load_raw", lambda: raw_io.load_raw(
            args.source, scene_half_size=False, scene_highlight_mode="clip", wb_mode="camera",
            decoder=args.decoder, coreimage_version="auto"))
        masks_loaded = array_record(bundle.clip_masks)
        processing = array_record(bundle.processing_clip_masks)
        analysis, _, _ = stage("analyze", lambda: analyze(bundle, 4, diagnostics=False, gamut_names=("P3",)))
        masks_analyzed = array_record(bundle.clip_masks)
        ev = stage("auto_ev", lambda: compute_auto_ev(bundle, analysis, tone_core="agx", gamut="p3"))
        bundle = with_intent_exposure(bundle, user_ev=ev.ev)
        plan = stage("render_plan", lambda: build_render_plan(bundle, analysis, "agx", "p3"))
        hdr_plan = stage("hdr_plan", lambda: compile_hdr_agx_plan(
            plan, HdrDisplayTarget(peak_nits=800), analysis=analysis, scene_decoder=bundle.scene_decoder))
        decisions = json_value({"analysis": analysis, "auto_ev": ev, "plan": plan,
                               "hdr_headroom_ev": hdr_plan.tone.rendered_headroom_ev,
                               "hdr_rho": hdr_plan.color.channel_separation,
                               "decoder": bundle.scene_decoder, "decoder_version": bundle.scene_decoder_version,
                               "scene_scale": bundle.scene_scale,
                               "processing_loss_pct": bundle.scene_processing_loss_pct})
        reference_samples = array_record(bundle.scene_reliable_reference_rec2020)
        bundle = raw_io.release_analysis_buffers(bundle)
        sdr = stage("render_sdr", lambda: render_output_u8(bundle, analysis, "p3", plan))
        sdr_record = array_record(sdr)
        del sdr
        base, hdr = stage("render_hdr_pair", lambda: render_ultrahdr_agx_pair(bundle, analysis, plan, hdr_plan, "p3"))
        record["identity"] = {"masks_loaded": masks_loaded, "masks_analyzed": masks_analyzed,
                              "processing_loss": processing, "reference_samples": reference_samples,
                              "decisions": decisions, "sdr": sdr_record,
                              "hdr_base": array_record(base), "hdr_alternate": array_record(hdr)}
    record.update(stages=stages, calls=calls)
    common = sum(stages[k] for k in ("load_raw", "analyze", "auto_ev", "render_plan"))
    record["pipeline_sdr_s"] = common + stages["render_sdr"]
    record["pipeline_hdr_s"] = common + stages["hdr_plan"] + stages["render_hdr_pair"]


def synthetic(args, record):
    import gc
    import numpy as np
    from dngscan.dng_opcodes import _fractional_crop_loss
    from dngscan.raw_io import _merge_processing_loss
    width, height = args.synthetic
    cases, identity = {}, {}
    for dtype in (np.float16, np.float32):
        for operation in ("crop", "merge_same", "merge_resized"):
            shape = (height, width, 3) if operation == "merge_same" else (max(1, height // 2), max(1, width // 2), 3)
            # Exactly representable fractions, including ordinary positive zero.
            source = np.random.default_rng(371).integers(0, 1025, shape, dtype=np.uint16).astype(dtype)
            source *= dtype(1 / 1024)
            times, result = [], None
            for _ in range(args.repeats):
                if result is not None:
                    del result
                if operation != "crop":
                    target = np.full((height, width, 3), .25, np.float16)
                start = time.perf_counter()
                if operation == "crop":
                    result = _fractional_crop_loss(source, (1., 1., height - 2., width - 2.),
                                                   (height, width), (height, width))
                else:
                    result = _merge_processing_loss(target, source)
                    assert result is target, "merge lost caller-owned in-place output"
                times.append(time.perf_counter() - start)
            key = operation + "_" + np.dtype(dtype).name
            cases[key] = {"seconds": times, "median_s": statistics.median(times)}
            identity[key] = array_record(result)
            del result, source
            if operation != "crop":
                del target
            gc.collect()
    record.update(cases=cases, identity=identity)


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
    if args.out.exists() or args.out.is_symlink() or (args.source and not args.source.is_file()):
        p.error("output must be new and source must exist")
    if args.repeats < 1 or (args.synthetic and min(args.synthetic) < 4):
        p.error("positive repeats and synthetic dimensions >= 4 required")
    if not (args.repo / "dngscan" / "raw_io.py").is_file():
        p.error("invalid --repo")
    reference = None
    if args.compare:
        try:
            reference = json.loads(args.compare.read_text())
            if not isinstance(reference, dict) or not isinstance(reference.get("identity"), dict):
                raise ValueError("comparison report must contain an identity object")
        except (OSError, ValueError) as exc:
            p.error(str(exc))
    os.environ["DNGSCAN_FAST"] = "1"
    os.environ["DNGSCAN_FAST_SKIP"] = ",".join(LOSS_KERNELS) if args.reference else ""
    sys.path.insert(0, str(args.repo.resolve()))
    from dngscan import _fast
    import numpy as np
    ext = _fast._load_extension()
    if ext is None:
        raise RuntimeError("matching native extension is required on both sides")
    commit = subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=True).stdout.strip()
    record = {"reference": args.reference, "source": str(args.source) if args.source else None,
              "synthetic": args.synthetic, "decoder": args.decoder, "commit": commit,
              "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
              "native_abi": ext.native_abi_version(), "cpus": os.cpu_count()}
    if args.source:
        pipeline(args, record)
    else:
        synthetic(args, record)
    # macOS reports bytes, Linux KiB. High water includes input setup and hashing.
    record["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 20 if sys.platform == "darwin" else 1024)
    matches = True
    if reference is not None:
        matches = record["identity"] == reference["identity"]
        record["comparison"] = {"report": str(args.compare), "exact": matches}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(json.dumps(json_value(record), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k in (
        "reference", "pipeline_sdr_s", "pipeline_hdr_s", "calls", "cases", "peak_rss_mib", "comparison")}), flush=True)
    return 0 if matches else 2


if __name__ == "__main__":
    raise SystemExit(main())
