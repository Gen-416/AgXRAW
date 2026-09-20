#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare a concatenated 1M-pixel finalize with two 500k-pixel finalizes.

Both paths receive the SAME complete A noise plane followed by the SAME complete
B noise plane, generated outside timing; slicing must never interleave RNG calls.
Each mode/budget runs in a fresh process with three repetitions by default.
An external process samples RSS/threads. Full input/output SHA256 and allocation
counts accompany timings; sampled RSS is not a guaranteed peak. No production
dispatcher is modified. Budget 1/5/8 corresponds to serial/streamed/wide kernels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time


def _hash(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _worker(args):
    sys.path.insert(0, str(args.repo))
    import numpy as np
    from dngscan import _fast, cpu_budget, render

    ext = _fast._load_extension()
    if ext is None:
        raise RuntimeError("matching native extension required")
    rng = np.random.default_rng(719)
    split = (args.pixels + 1) // 2
    chunks = [rng.uniform(-.05, 1.5, (length, 3)).astype(np.float32)
              for length in (split, args.pixels - split)]
    # Full A then full B is the production quantize-group sequence. No RNG is
    # called in either measured path, and both calls receive simple row views.
    noise_a, noise_b = render.generate_dither_noise(np.random.default_rng(0), (args.pixels, 3))
    output = np.empty((args.pixels, 3), dtype=np.uint8)
    output.fill(0)  # Touch final destination pages identically before timing.
    plan = _fast.compile_output_plan(args.gamut, .05)
    identity = {"chunks_sha256": [_hash(chunk) for chunk in chunks],
                "noise_a_sha256": _hash(noise_a), "noise_b_sha256": _hash(noise_b),
                "pixels": args.pixels, "split": split, "gamut": args.gamut,
                "input_seed": 719, "noise_seed": 0}

    def concatenated():
        joined = np.concatenate(chunks, axis=0)
        output[:] = _fast.finalize_rec2020_u8_f32(joined, noise_a, noise_b, plan)

    def sliced():
        start = 0
        for chunk in chunks:
            end = start + len(chunk)
            output[start:end] = _fast.finalize_rec2020_u8_f32(
                chunk, noise_a[start:end], noise_b[start:end], plan)
            start = end

    function = concatenated if args.mode == "concat" else sliced
    samples, hashes = [], []
    with cpu_budget.claim(args.budget):
        for _ in range(args.repeats):
            started = time.monotonic()
            function()
            finished = time.monotonic()
            samples.append({"started_s": started, "finished_s": finished, "wall_s": finished - started})
            hashes.append(_hash(output))
    if len(set(hashes)) != 1:
        raise RuntimeError("output changed across identical repetitions")
    identity["output_sha256"] = hashes[0]
    stored = sum(chunk.nbytes for chunk in chunks) + noise_a.nbytes + noise_b.nbytes + output.nbytes
    largest_call = args.pixels if args.mode == "concat" else split
    report = {"mode": args.mode, "budget": args.budget, "repeats": args.repeats,
              "repo": str(args.repo), "native_abi": int(ext.native_abi_version()),
              "commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
              "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
              "cpu_count": os.cpu_count(), "budget_total": cpu_budget.TOTAL,
              "environment": {key: value for key, value in os.environ.items() if key.startswith("DNGSCAN_")},
              "identity": identity, "timings": samples,
              "median_s": statistics.median(row["wall_s"] for row in samples),
              "allocation_bytes": {"retained_inputs_noise_destination": stored,
                                   "additional_concat": args.pixels * 12 if args.mode == "concat" else 0,
                                   "largest_temporary_u8_result": largest_call * 3},
              "process_highwater_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                                                  (1 if sys.platform == "darwin" else 1024))}
    with args.worker_out.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
    return 0


def _parent(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.benchmark_pipeline_concurrency import ProcessSampler, _sample_summary
    sampler = ProcessSampler()
    reports = []
    with tempfile.TemporaryDirectory(prefix="agxraw-quantize-") as directory:
        folder = Path(directory)
        for index, budget in enumerate(args.budgets):
            for mode in (("concat", "slices") if index % 2 == 0 else ("slices", "concat")):
                worker_out = folder / f"{budget}-{mode}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--repo", str(args.repo),
                           "--out", str(args.out), "--pixels", str(args.pixels), "--gamut", args.gamut,
                           "--repeats", str(args.repeats), "--worker", "--mode", mode,
                           "--budget", str(budget), "--worker-out", str(worker_out)]
                with (folder / "worker.log").open("w") as log:
                    process = subprocess.Popen(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, DNGSCAN_FAST="1", DNGSCAN_FAST_SKIP=""))
                    samples = []
                    try:
                        while process.poll() is None:
                            tick = time.monotonic()
                            samples.append(sampler.sample(process.pid))
                            time.sleep(max(0., args.sample_ms / 1000. - (time.monotonic() - tick)))
                        if process.returncode:
                            raise RuntimeError((folder / "worker.log").read_text(errors="replace")[-10000:])
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait()
                report = json.loads(worker_out.read_text())
                active = [row for row in samples if any(t["started_s"] <= row["monotonic_s"] <= t["finished_s"]
                                                        for t in report["timings"])]
                report["sampler"] = {"backend": sampler.backend, "requested_interval_ms": args.sample_ms,
                                      "timed_calls": _sample_summary(active, {"started_s": -float("inf"), "finished_s": float("inf")}),
                                      "samples": samples}
                reports.append(report)
    identities = [report["identity"] for report in reports]
    exact = all(identity == identities[0] for identity in identities[1:])
    comparison = {}
    for budget in args.budgets:
        old = next(r for r in reports if r["budget"] == budget and r["mode"] == "concat")
        new = next(r for r in reports if r["budget"] == budget and r["mode"] == "slices")
        comparison[str(budget)] = {"concat_s": old["median_s"], "slices_s": new["median_s"],
                                  "change_pct": (new["median_s"] / old["median_s"] - 1.) * 100.,
                                  "concat_sampled_rss_bytes": old["sampler"]["timed_calls"]["tree_rss_peak_bytes"],
                                  "slices_sampled_rss_bytes": new["sampler"]["timed_calls"]["tree_rss_peak_bytes"]}
    with args.out.open("x") as output:
        json.dump({"exact": exact, "comparisons": comparison, "runs": reports}, output, indent=2, allow_nan=False)
    print(json.dumps({"out": str(args.out), "exact": exact, "comparisons": comparison}))
    return 0 if exact else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pixels", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 5, 8])
    parser.add_argument("--sample-ms", type=float, default=20.)
    parser.add_argument("--gamut", choices=("srgb", "p3"), default="p3")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("concat", "slices"), help=argparse.SUPPRESS)
    parser.add_argument("--budget", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.pixels < 2 or args.repeats < 1 or min(args.budgets) < 1 or
            len(args.budgets) != len(set(args.budgets)) or not 20 <= args.sample_ms <= 1000):
        parser.error("pixels >= 2, positive repeats/unique budgets, sample interval in [20,1000] ms required")
    args.repo, args.out = args.repo.expanduser().absolute(), args.out.expanduser().absolute()
    if not (args.repo / "dngscan/_fast.py").is_file():
        parser.error("invalid repo")
    if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
        parser.error("output must be new, in an existing directory")
    if args.worker:
        if args.mode is None or args.budget is None or args.budget < 1 or args.worker_out is None:
            parser.error("internal worker arguments missing")
        return _worker(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
