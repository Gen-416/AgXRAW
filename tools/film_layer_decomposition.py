#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Film v2 P0: per-layer contribution measurement on real sample classes.

FILM_PRINT_RENDERING_PLAN §12 P0: render the six-way ladder on the five
declared sample classes and record what each layer contributes to Y, Oklab
L/C, hue path and pixel difference — the baseline that stops later stages
from manufacturing differences through undeclared layers.

Samples live outside the repo (photographer's cleared set); missing files are
reported and skipped so the tool stays runnable on any machine. Output:
docs/film_v2_p0_decomposition.json plus a printed summary the P0 report is
written from. Renders use the same in-process API as the freeze
(build_render_plan + render_output_u8), decode once per sample.

Usage: PYTHONPATH=. python tools/film_layer_decomposition.py [--stock portra400]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "docs" / "film_v2_p0_decomposition.json"

# The five declared sample classes (plan §12 P0). 钨丝人像 is stood in for by
# the tungsten bar scene — the cleared sample set has no tungsten portrait;
# declared here rather than papered over.
SAMPLES = [
    ("daylight_skin", "日光肤色 · 正午合照", Path.home() / "Pictures/_SDI0238.DNG"),
    ("overcast_foliage", "阴天植物 · 公园树荫", Path.home() / "Downloads/DSC00225.ARW"),
    ("tungsten_scene", "钨丝场景 · 猫吧灯带（钨丝人像的替代样张）", Path.home() / "Pictures/_SDI0016.DNG"),
    ("neon_highlights", "霓虹高光 · 夜景", Path.home() / "Pictures/_SDI0199.DNG"),
    ("deep_shadow_interior", "深影室内 · live house 后台", Path.home() / "Pictures/_SDI0173.DNG"),
]

TRANSITIONS = [
    ("curve_layer", "agx_baseline", "curve_only", "曲线层（胶片明暗坐标）"),
    ("prefeed_layer", "agx_baseline", "prefeed_only", "前馈层（感色分离 @1.0）"),
    ("observe_total", "agx_baseline", "observe_combo", "observe 组合（曲线+配对强度前馈+配对原色）"),
    ("takeover_delta", "observe_combo", "full_bounded", "full 接管对 observe（有界中性化）"),
    ("crossover_delta", "full_bounded", "full_datasheet", "层间漂移（datasheet 对 bounded）"),
]


def srgb_u8_to_oklab(u8):
    import numpy as np

    x = u8.astype(np.float64) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    m1 = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ])
    lms = lin @ m1.T
    lms_ = np.cbrt(lms)
    m2 = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ])
    return lms_ @ m2.T


def transition_stats(a_u8, b_u8) -> dict:
    import numpy as np

    a = a_u8.reshape(-1, 3).astype(np.int16)
    b = b_u8.reshape(-1, 3).astype(np.int16)
    d = np.abs(a - b).max(axis=1)
    lab_a = srgb_u8_to_oklab(a_u8.reshape(-1, 3).astype(np.uint8))
    lab_b = srgb_u8_to_oklab(b_u8.reshape(-1, 3).astype(np.uint8))
    dl = lab_b[:, 0] - lab_a[:, 0]
    ca = np.hypot(lab_a[:, 1], lab_a[:, 2])
    cb = np.hypot(lab_b[:, 1], lab_b[:, 2])
    dc = cb - ca
    # Hue path on pixels chromatic enough for hue to mean anything.
    chroma_mask = ca > 0.04
    if chroma_mask.any():
        ha = np.arctan2(lab_a[chroma_mask, 2], lab_a[chroma_mask, 1])
        hb = np.arctan2(lab_b[chroma_mask, 2], lab_b[chroma_mask, 1])
        dh = np.degrees(np.angle(np.exp(1j * (hb - ha))))
        hue_median_abs = float(np.median(np.abs(dh)))
        hue_p95_abs = float(np.percentile(np.abs(dh), 95))
    else:
        hue_median_abs = hue_p95_abs = 0.0
    luma_a = a @ np.array([0.2627, 0.678, 0.0593])
    luma_b = b @ np.array([0.2627, 0.678, 0.0593])
    dy = (luma_b - luma_a) / 255.0
    return {
        "visible_pct_gt8": round(float((d > 8).mean() * 100.0), 1),
        "mean_abs_u8": round(float(d.mean()), 2),
        "dY_mean": round(float(dy.mean()), 4),
        "dY_p95_abs": round(float(np.percentile(np.abs(dy), 95)), 4),
        "dL_mean": round(float(dl.mean()), 4),
        "dC_mean": round(float(dc.mean()), 4),
        "dC_p95_abs": round(float(np.percentile(np.abs(dc), 95)), 4),
        "hue_median_abs_deg": round(hue_median_abs, 2),
        "hue_p95_abs_deg": round(hue_p95_abs, 2),
        "chromatic_share_pct": round(float(chroma_mask.mean() * 100.0), 1),
    }


def render_ladder(path: Path, stock: str) -> dict:
    import dngscan as dg
    from tools.regen_film_freeze import BASELINE, freeze_configs, render_case

    class _Scene:
        def __init__(self, bundle, analysis):
            self.bundle = bundle
            self.analysis = analysis

    bundle = dg.load_raw(path)
    analysis, _y, _ev = dg.analyze(bundle, margin=16, diagnostics=False)
    scene = _Scene(bundle, analysis)
    outputs = {"agx_baseline": render_case(scene, BASELINE)}
    for config, params in freeze_configs(stock).items():
        outputs[config] = render_case(scene, params)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", default="portra400")
    args = parser.parse_args()
    os.environ.setdefault("DNGSCAN_FAST", "1")

    results: dict = {"stock": args.stock, "samples": {}}
    for sample_id, label, path in SAMPLES:
        if not path.is_file():
            print(f"skip {sample_id}: {path} 不存在", flush=True)
            results["samples"][sample_id] = {"label": label, "status": "missing"}
            continue
        print(f"render {sample_id} ({label}) …", flush=True)
        ladder = render_ladder(path, args.stock)
        rows = {}
        for tid, src, dst, tlabel in TRANSITIONS:
            rows[tid] = {"label": tlabel, **transition_stats(ladder[src], ladder[dst])}
            print(f"  {tid}: {rows[tid]['visible_pct_gt8']}% >8/255, "
                  f"hue~{rows[tid]['hue_median_abs_deg']}°", flush=True)
        results["samples"][sample_id] = {
            "label": label, "status": "ok", "path": str(path), "transitions": rows,
        }
    OUT_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
