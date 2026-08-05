#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark native vs Python AgX core."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dngscan import _fast as fast_backend
from dngscan.agx import apply_core, formation_matrices
from dngscan.fast_plan import compile_agx_plan
from dngscan.models import ToneCompressionPlan
from dngscan.punch import apply_punch_rec2020
from dngscan.render import apply_agx_core, render_output_u8


def _plan() -> ToneCompressionPlan:
    return ToneCompressionPlan(
        target_gamut="Rec2020",
        luma_p1=0.01,
        luma_p50=0.18,
        luma_p99=1.0,
        luma_p999=2.0,
        black_ev=-8.0,
        white_ev=5.0,
        dynamic_range_ev=13.0,
        contrast=3.0,
        toe_power=1.5,
        shoulder_power=2.9,
        chroma_p95=0.5,
        negative_rgb_pct=0.0,
        over_rgb_pct=0.0,
        tone_core="agx",
        use_c1_endpoints=True,
    )


def _median_seconds(fn, repeats: int = 3) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(statistics.median(samples))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dng", nargs="?", help="Optional DNG for full render benchmark")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    import numpy as np

    rng = np.random.default_rng(0)
    rgb = rng.uniform(0.0, 1.2, size=(6_000_000, 3)).astype(np.float32)
    plan = _plan()

    def python_core() -> None:
        prev = os.environ.get("DNGSCAN_FAST")
        os.environ["DNGSCAN_FAST"] = "0"
        try:
            inset, outset = formation_matrices(plan)
            mapped = apply_core(rgb, plan, inset, outset)
            apply_punch_rec2020(mapped, 0.0)
        finally:
            if prev is None:
                os.environ.pop("DNGSCAN_FAST", None)
            else:
                os.environ["DNGSCAN_FAST"] = prev

    def native_core() -> None:
        prev = os.environ.get("DNGSCAN_FAST")
        os.environ["DNGSCAN_FAST"] = "1"
        try:
            native_plan = compile_agx_plan(plan)
            fast_backend.apply_agx_core_f32(rgb, native_plan)
        finally:
            if prev is None:
                os.environ.pop("DNGSCAN_FAST", None)
            else:
                os.environ["DNGSCAN_FAST"] = prev

    py_s = _median_seconds(python_core, args.repeats)
    print(f"Python AgX core ({rgb.shape[0]} px): {py_s:.3f}s")
    native_ready = fast_backend.available()
    if native_ready:
        nat_s = _median_seconds(native_core, args.repeats)
        speedup = py_s / max(nat_s, 1e-9)
        print(f"Native AgX core ({rgb.shape[0]} px): {nat_s:.3f}s ({speedup:.2f}x)")
    else:
        print("Native backend: unavailable (build with tools/build_native.sh)")

    if args.dng:
        print("Full DNG benchmark: load bundle via CLI workflow locally; synthetic core timing above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
