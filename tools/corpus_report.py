#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only corpus report: how the automatic gates behave across a folder of RAWs.

The punch gate and the gated color-path window were tuned on a small set of
photographs. This tool renders nothing to disk and changes nothing; it decodes each
RAW at half size, compiles the same RenderPlan the exporter would use, and records the
gate inputs and outputs plus proxy-render Oklab statistics into a CSV. The printed
summary groups scenes by (ISO band x median-EV band) and flags suspected misfires so
threshold edits can be argued from data. Any threshold change that follows must re-run
the golden set and attach this summary to the commit.

Usage:
    python tools/corpus_report.py --dir ~/Pictures/corpus --csv corpus.csv
    python tools/corpus_report.py --dir ~/Pictures --glob "*.RAF" --limit 40
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import dngscan.core as dg
from dngscan.color import apply_rgb_matrix3
from dngscan.constants import OKLAB_M1, OKLAB_M2, RGB_TO_XYZ
from dngscan.tone import build_render_plan, compute_exposure_gain, scene_rec2020_to_float

RAW_SUFFIXES = {".dng", ".raf", ".nef", ".nrw", ".arw", ".cr2", ".cr3", ".rw2", ".orf"}

FIELDS = [
    "file", "model", "iso",
    "body_ev_p50", "body_ev_p99", "tail_ev_p9999", "sparse_emitter",
    "usable_dr_eff_ev", "black_ev", "white_ev", "window_dr_ev",
    "punch_strength", "view_brightness", "pivot_ev_offset",
    "render_chroma_mean", "render_chroma_p90", "render_luma_p10", "render_luma_p50",
    "flags",
]


def _oklab_stats(rgb_linear: np.ndarray) -> dict[str, float]:
    rgb = np.clip(np.nan_to_num(rgb_linear, nan=0.0), 0.0, 1.0).astype(np.float32)
    xyz = apply_rgb_matrix3(rgb, RGB_TO_XYZ["Rec2020"])
    lab = apply_rgb_matrix3(np.cbrt(np.maximum(apply_rgb_matrix3(xyz, OKLAB_M1), 0.0)), OKLAB_M2)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    return {
        "render_chroma_mean": float(chroma.mean()),
        "render_chroma_p90": float(np.percentile(chroma, 90.0)),
        "render_luma_p10": float(np.percentile(lab[:, 0], 10.0)),
        "render_luma_p50": float(np.percentile(lab[:, 0], 50.0)),
    }


def _misfire_flags(row: dict) -> list[str]:
    flags: list[str] = []
    # Punch engaging on a scene whose body median is deep below gray contradicts the
    # bright-scene gate's intent.
    if row["punch_strength"] > 0.5 and row["body_ev_p50"] < -2.5:
        flags.append("punch_on_dark_body")
    # View brightness lifting a noisy sensor window amplifies shadow noise.
    if row["view_brightness"] > 1.2 and row["usable_dr_eff_ev"] < 6.0:
        flags.append("brightness_on_noisy_dr")
    # A sparse-emitter scene with strong punch usually means the tail classifier and
    # the bright gate disagree about what the frame is.
    if row["sparse_emitter"] and row["punch_strength"] > 0.3:
        flags.append("punch_on_sparse_emitter")
    return flags


def analyze_one(path: Path) -> dict | None:
    try:
        bundle = dg.load_raw(path, "reconstruct", scene_half_size=True)
        bundle.exposure_gain = compute_exposure_gain("agx", 0.0)
        analysis, _, _ = dg.analyze(bundle, 4, diagnostics=False, gamut_names=("P3",))
        plan = build_render_plan(bundle, analysis, "agx", "p3")
    except Exception as exc:
        print(f"  跳过 {path.name}: {exc}", file=sys.stderr)
        return None
    tone, scene = plan.tone, plan.scene

    flat = bundle.scene_rec2020_render.reshape(-1, 3)
    step = max(1, flat.shape[0] // 200_000)
    rec = scene_rec2020_to_float(flat[::step, :3], bundle.scene_scale, bundle.exposure_gain)
    from dngscan.render import apply_tone_core
    from dngscan.color import rec2020_to_output

    mapped = apply_tone_core(np.ascontiguousarray(rec, dtype=np.float32), tone, plan.color)
    stats = _oklab_stats(rec2020_to_output(mapped, "p3"))

    row: dict = {
        "file": path.name,
        "model": getattr(analysis, "prior_id", "") or "",
        "iso": getattr(bundle, "shot_iso", "") or "",
        "body_ev_p50": round(scene.body_ev_p50, 3),
        "body_ev_p99": round(scene.body_ev_p99, 3),
        "tail_ev_p9999": round(scene.tail_ev_p9999, 3),
        "sparse_emitter": bool(scene.sparse_emitter_tail),
        "usable_dr_eff_ev": round(analysis.usable_dr_eff_ev, 2)
        if math.isfinite(analysis.usable_dr_eff_ev) else "",
        "black_ev": round(tone.black_ev, 2),
        "white_ev": round(tone.white_ev, 2),
        "window_dr_ev": round(tone.dynamic_range_ev, 2),
        "punch_strength": round(tone.punch_strength, 3),
        "view_brightness": round(tone.view_brightness, 3),
        "pivot_ev_offset": round(float(getattr(tone, "pivot_ev_offset", 0.0)), 3),
    }
    row.update({k: round(v, 4) for k, v in stats.items()})
    row["flags"] = ";".join(_misfire_flags(row))
    return row


def _band(value: float, edges: list[float], labels: list[str]) -> str:
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def print_summary(rows: list[dict]) -> None:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        iso = row["iso"] if isinstance(row["iso"], int) else 0
        iso_band = _band(float(iso or 0), [400, 1600, 6400], ["ISO<400", "ISO<1600", "ISO<6400", "ISO≥6400"])
        ev_band = _band(float(row["body_ev_p50"]), [-3.0, -1.5], ["中位<-3EV", "中位-3..-1.5", "中位>-1.5EV"])
        groups.setdefault((iso_band, ev_band), []).append(row)
    print(f"\n{'组':26s}{'张数':>5s}{'punch 中位':>11s}{'punch 范围':>16s}{'view_b 中位':>12s}")
    for key in sorted(groups):
        g = groups[key]
        punches = sorted(r["punch_strength"] for r in g)
        vbs = sorted(r["view_brightness"] for r in g)
        mid = punches[len(punches) // 2]
        print(f"{key[0]+'/'+key[1]:26s}{len(g):>5d}{mid:>11.2f}"
              f"{f'{punches[0]:.2f}..{punches[-1]:.2f}':>16s}{vbs[len(vbs)//2]:>12.2f}")
    flagged = [r for r in rows if r["flags"]]
    if flagged:
        print(f"\n疑似误触发 {len(flagged)} 张:")
        for r in flagged:
            print(f"  {r['file']}: {r['flags']} (punch={r['punch_strength']}, "
                  f"中位={r['body_ev_p50']}EV, DR={r['usable_dr_eff_ev']})")
    else:
        print("\n无疑似误触发。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True, help="RAW 目录")
    parser.add_argument("--glob", default=None, help="可选文件名过滤,如 '*.RAF'")
    parser.add_argument("--csv", type=Path, default=None, help="CSV 输出路径")
    parser.add_argument("--limit", type=int, default=0, help="最多处理张数(0=不限)")
    args = parser.parse_args()

    if args.glob:
        files = sorted(args.dir.glob(args.glob))
    else:
        files = sorted(p for p in args.dir.iterdir()
                       if p.suffix.lower() in RAW_SUFFIXES and p.is_file())
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("目录里没有可识别的 RAW。", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        row = analyze_one(path)
        if row is not None:
            rows.append(row)
    if not rows:
        return 1

    if args.csv is not None:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV -> {args.csv}")
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
