#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fresh-process exact noise/SNR/health comparison on deterministic u16 Bayer RAW.

--reference calls the three original public computations independently. The new
path builds one bounded phase workspace, then calls the same noise/SNR/health
consumers with that workspace. One invocation measures one fresh process; run
alternating pairs. Input generation/hashes and result hashing are outside timers.
RSS is the process high-water mark including imports/input generation/hashing,
not a live-allocation delta. No decode, scene RGB, render or codec is included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import sys
import time
from types import SimpleNamespace


def _scalar_bits(value, path=""):
    """Supplement JSON scalar equality with exact float64 bits, including NaNs."""
    if isinstance(value, float):
        return {path: struct.pack("!d", value).hex()}
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(_scalar_bits(item, f"{path}/{key}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.update(_scalar_bits(item, f"{path}/{index}"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--size", type=int, nargs=2, default=(6000, 4000), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args(argv)
    args.repo = args.repo.expanduser().absolute()
    if min(args.size) < 2 or not (args.repo / "dngscan/phase_statistics.py").is_file():
        parser.error("Bayer dimensions must be at least 2; repo must contain phase_statistics.py")
    if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
        parser.error("output must be a new file in an existing directory")
    previous = None
    if args.compare:
        try:
            previous = json.loads(args.compare.read_text())
            if (not isinstance(previous, dict) or previous.get("schema") != 1
                    or previous.get("size") != list(args.size)
                    or not isinstance(previous.get("identity"), dict)):
                raise ValueError("comparison must be a phase report for the same dimensions")
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    os.environ["DNGSCAN_FAST"] = "1"
    os.environ["DNGSCAN_FAST_SKIP"] = ""
    sys.path.insert(0, str(args.repo))
    import numpy as np
    from dngscan import _fast, analysis, phase_statistics
    from tools.benchmark_loss_pipeline import array_record, json_value

    extension = _fast._load_extension()
    if extension is None:
        raise RuntimeError("matching native extension required for the declared production environment")
    width, height = args.size
    seed = 731
    pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8)
    raw = np.random.default_rng(seed).integers(0, 16384, (height, width), dtype=np.uint16)
    colors = np.empty(raw.shape, dtype=np.uint8)
    for y in range(2):
        for x in range(2):
            colors[y::2, x::2] = pattern[y, x]
    raw.flags.writeable = colors.flags.writeable = False
    ids = [0, 1, 2, 3]
    labels = analysis.channel_labels("RGBG", ids)
    fullwell = {0: 16383, 1: 16271, 2: 16351, 3: 16231}
    black = [63., 65.25, 61.5, 64.75]
    bundle = SimpleNamespace(raw_image=raw, raw_colors=colors, raw_pattern=pattern,
                             white_level=16383, black_levels=black, evidence=None)
    identity = {"input": {"raw": array_record(raw), "colors": array_record(colors),
                           "pattern": pattern.tolist(), "seed": seed,
                           "distribution": "uniform integer [0,16384); synthetic, not a camera noise model",
                           "black_levels": black, "fullwell": fullwell, "channel_ids": ids, "labels": labels}}
    before_highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stages = {}

    def measured(name, function):
        start = time.perf_counter()
        result = function()
        stages[name] = time.perf_counter() - start
        return result

    cpu_start, wall_start = time.process_time(), time.perf_counter()
    if args.reference:
        noise = measured("noise", lambda: analysis.estimate_raw_noise_floor(bundle, fullwell))
        snr = measured("snr", lambda: analysis.compute_snr_curves(bundle, ids, labels, fullwell))
        health = measured("health", lambda: analysis.raw_health_metrics(bundle, ids, labels))
    else:
        prepared = measured("workspace", lambda: phase_statistics.build_phase_statistics(bundle, ids, labels))
        if prepared is None:
            raise RuntimeError("synthetic layout unexpectedly fell back to the reference")
        noise = measured("noise_consumer", lambda: analysis.estimate_raw_noise_floor(
            bundle, fullwell, _prepared_phases=prepared))
        snr = measured("snr_consumer", lambda: analysis.compute_snr_curves(
            bundle, ids, labels, fullwell, _phase_stats=prepared.snr))
        health = measured("health_consumer", lambda: analysis.raw_health_metrics(
            bundle, ids, labels, _prepared_phases=prepared))
        del prepared
    elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start
    measured_highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    values = {"noise_floor": noise, "snr_curves": snr[0], "snr1_dr": snr[1], "snr1_stop": snr[2],
              "health_lag1_corr": health[0], "health_hist_empty_pct": health[1]}
    identity.update(outputs=json_value(values), scalar_float64_bits=_scalar_bits(values))
    identity = json_value(identity)
    source_paths = [Path(analysis.__file__), Path(phase_statistics.__file__),
                    args.repo / "dngscan/spatial_black.py", Path(__file__).resolve()]
    report = {"schema": 1, "size": list(args.size), "reference": args.reference, "repo": str(args.repo),
              "commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
              "implementation_sha256": {str(path.relative_to(args.repo)) if path.is_relative_to(args.repo) else path.name:
                                         hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform(), "cpus": os.cpu_count(),
                              "native_abi": int(extension.native_abi_version()),
                              "DNGSCAN_FAST": os.environ["DNGSCAN_FAST"], "DNGSCAN_FAST_SKIP": os.environ["DNGSCAN_FAST_SKIP"]},
              "identity": identity, "stages_s": stages, "total_wall_s": elapsed, "total_cpu_s": cpu_elapsed,
              "input_owned_bytes": raw.nbytes + colors.nbytes,
              "peak_rss_before_mib": before_highwater / (2**20 if sys.platform == "darwin" else 1024),
              "peak_rss_after_computation_mib": measured_highwater / (2**20 if sys.platform == "darwin" else 1024),
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024)}
    exact = previous is None or identity == previous["identity"]
    if previous is not None:
        report["comparison"] = {"report": str(args.compare), "identity_exact": exact}
    with args.out.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: report[key] for key in ("size", "reference", "total_wall_s", "total_cpu_s", "peak_rss_mib")}
                     | {"exact": exact, "out": str(args.out)}))
    return 0 if exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
