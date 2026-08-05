#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pipeline impact matrix: measure what every choice actually does to the image.

For one frame, render a baseline (pure CLI defaults) and a battery of single-axis
variants, then measure each variant's pixel delta against the baseline in 8-bit
output codes — total (mean / p99 / max) and split into a luma component and a
chroma remainder, so tone effects and colour effects attribute separately.

The point is architectural: every stage CLAIMS a role and a rough magnitude.
Measuring the real magnitudes turns those claims into checkable statements —
a stage that claims importance but measures ~0 is dead wiring; a stage that
claims subtlety but measures huge is a leak; a pair that should be identical
(native kernel vs NumPy reference) measuring above one dither step is a defect.

Special pairs measured against each other rather than the baseline:
  film full-mode  vs observe-mode   -> the ratio field's real contribution
  theatrical      vs translated     -> the surround term's real contribution
  DNGSCAN_FAST=0  vs baseline       -> native/NumPy parity

Metric caveats (learned on the first indoor-frame run, 2026-07-30):
  * The luma/chroma split is ADDITIVE: a ratio-preserving luminance change
    registers partly as "chroma" in code space (the lum core's chroma-dominant
    reading is that artifact). Good for screening, not for strict attribution.
  * The decoder axis is geometry-confounded: RAW9 applies DNG warps, so its
    pixel diff measures interpretation PLUS misalignment — do not compare its
    magnitude against the other axes (this is why scale alignment uses medians).
  * Parity pair: measured 99.93% identical pixels; the 0.026% above one code
    sit in 8x8 clusters — single-code dither flips amplified by JPEG DCT,
    not kernel divergence. Judge parity on the ==0/==1 shares, not the max.

Usage:
  .venv/bin/python tools/pipeline_impact.py <frame.dng> [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# (name, extra CLI args, claimed-role note). Order is the report order.
VARIANTS: list[tuple[str, list[str], str]] = [
    ("decoder=coreimage", ["--decoder", "coreimage"], "相机诠释差异：应为中大"),
    ("wb=5500k", ["--wb", "5500k"], "声明色温：随场景光源偏离而大"),
    ("wb=3200k", ["--wb", "3200k"], "声明色温：同上"),
    # NOTE: the CLI default highlight mode IS clip — a clip variant would measure
    # itself against itself (the first run's null row). Blend/reconstruct are the
    # real axes.
    ("highlight=blend", ["--highlight-mode", "blend"], "仅高光区：局部"),
    ("highlight=reconstruct", ["--highlight-mode", "reconstruct"], "仅高光区：局部"),
    ("demosaic=vng", ["--demosaic", "vng"], "插值差异：小，高频处"),
    ("tone-core=lum", ["--tone-core", "lum"], "无逐通道/AgX 几何：大"),
    ("tone-core=neutral", ["--tone-core", "neutral"], "诊断曲线：大"),
    ("tone-core=gated", ["--tone-core", "gated"], "证据门控色彩：中"),
    ("primaries=punchy", ["--agx-primaries", "punchy"], "纯度几何：中"),
    ("primaries=muted", ["--agx-primaries", "muted"], "纯度几何：中"),
    ("punch=1.5", ["--punch", "1.5"], "场景自适应纯度：小（暗场景趋零）"),
    ("prefeed=portra_d55", ["--scene-transform", "portra400_d55"], "分材料分离：小（有界）"),
    ("prefeed x2.2", ["--scene-transform", "portra400_d55",
                      "--scene-transform-strength", "2.2"], "分离超驱动：小-中"),
    ("lens-filter=85b", ["--lens-filter", "85b"], "+131 mired：大（全局暖移）"),
    ("film-curve=portra400", ["--film-curve", "portra400"], "音调签名：中-大"),
    ("film=portra400 (observe)", ["--film", "portra400"], "组合=WB+分离+曲线：大"),
    ("film=portra400 (full)", ["--film", "portra400", "--film-mode", "full"],
     "胶片接管显影：大（实验）"),
    ("film=vision3250d", ["--film", "vision3250d"], "电影链翻译：大"),
    ("film=v3250d theatrical", ["--film", "vision3250d_theatrical"],
     "引用原文：大于翻译"),
]

PAIRS = [
    ("ratio field (full vs observe)", "film=portra400 (full)",
     "film=portra400 (observe)", "比率场净贡献：小-中（无锚重建）"),
    ("surround term (theatrical vs translated)", "film=v3250d theatrical",
     "film=vision3250d", "surround 1.5 净贡献：中"),
]


def render(frame: Path, out: Path, extra: list[str], env: dict | None = None) -> None:
    cmd = [PY, "-m", "dngscan", str(frame), "--jpeg", str(out),
           "--output-format", "sdr", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(PROJECT_ROOT), env=env)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"render failed for {out.name}: {proc.stderr[-400:]}")


def measure(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.shape != b.shape:  # decoder variants may differ by a few px of geometry
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    d = a.astype(np.float64) - b.astype(np.float64)
    abs_d = np.abs(d)
    luma_d = d @ _LUMA
    chroma_d = d - luma_d[..., None]
    return {
        "mean": float(abs_d.mean()),
        "p99": float(np.percentile(abs_d, 99)),
        "max": float(abs_d.max()),
        "luma_mean": float(np.abs(luma_d).mean()),
        "chroma_mean": float(np.abs(chroma_d).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("frame", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--keep", type=Path, default=None,
                    help="keep renders in this directory instead of a temp dir")
    args = ap.parse_args()

    workdir = args.keep or Path(tempfile.mkdtemp(prefix="impact_"))
    workdir.mkdir(parents=True, exist_ok=True)
    images: dict[str, np.ndarray] = {}

    def load(name: str, extra: list[str], env: dict | None = None) -> np.ndarray:
        out = workdir / (name.replace("/", "_").replace(" ", "_") + ".jpg")
        if not out.exists():
            render(args.frame, out, extra, env)
            print(f"rendered {name}", flush=True)
        return np.asarray(Image.open(out))

    baseline = load("baseline", [])
    rows = []
    for name, extra, claim in VARIANTS:
        try:
            img = load(name, extra)
        except RuntimeError as exc:
            rows.append({"name": name, "claim": claim, "error": str(exc)[:160]})
            continue
        images[name] = img
        rows.append({"name": name, "claim": claim, **measure(img, baseline)})

    # Native vs NumPy parity: same args, kernel forced off.
    env = dict(os.environ, DNGSCAN_FAST="0")
    numpy_baseline = load("baseline_numpy", [], env)
    rows.append({"name": "native vs numpy (FAST=0)",
                 "claim": "算术奇偶性：应 ≤1 抖动步",
                 **measure(numpy_baseline, baseline)})

    for name, a_key, b_key, claim in PAIRS:
        if a_key in images and b_key in images:
            rows.append({"name": name, "claim": claim,
                         **measure(images[a_key], images[b_key])})

    print(f"\n{'axis':34s} {'mean':>6s} {'p99':>6s} {'max':>5s} "
          f"{'luma':>6s} {'chroma':>6s}  claim")
    for r in rows:
        if "error" in r:
            print(f"{r['name']:34s} ERROR: {r['error']}")
            continue
        print(f"{r['name']:34s} {r['mean']:6.2f} {r['p99']:6.1f} {r['max']:5.0f} "
              f"{r['luma_mean']:6.2f} {r['chroma_mean']:6.2f}  {r['claim']}")

    if args.out:
        args.out.write_text(json.dumps(
            {"frame": str(args.frame), "rows": rows}, ensure_ascii=False, indent=1))
        print(f"\nwrote {args.out}")
    print(f"renders kept in {workdir}" if args.keep else f"temp renders: {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
