#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Attribute a decoder A/B to the decode or to the tone plan it produced.

Comparing --decoder libraw against --decoder coreimage at a fixed --ev does not isolate
the decoder. Each buffer compiles its own RenderPlan, and Core Image also executes DNG
geometry and highlight reconstruction. RAW 9's reconstructed tail is constrained by the
full-resolution CFA clipped-cell rate before it can set the white endpoint, but the two
plans can still differ. A side-by-side at matched exposure therefore needs both the
buffer and the plan attributed separately.

This tool runs the 2x2: each buffer rendered through each plan.

    A  libraw buffer  + libraw plan     native baseline
    B  coreimage      + coreimage       native, what an ordinary export gives
    C  coreimage      + libraw plan     decode isolated (plan held fixed)
    D  libraw         + coreimage plan  plan isolated (buffer held fixed)

Two of those comparisons are exact and two are not, which the report states rather than
blurring. A vs D and B vs C hold the buffer fixed, so they are per-pixel comparable and
the plan's effect is measured directly. A vs C holds the plan fixed but changes the
buffer, and Core Image executes the file's DNG opcodes (WarpRectilinear), so its frame
is a nonlinear warp of LibRaw's -- those are compared as distributions only.

This is a diagnostic, not a way to make pictures. A plan compiled from LibRaw applied to
a Core Image buffer puts the white endpoint in the wrong place and clips the highlight
roll-off Apple preserved; that ugliness is the point, since it marks where the two
buffers disagree about how much information they hold.

Usage:
    python tools/decode_ab.py photo.dng
    python tools/decode_ab.py photo.dng --write-jpegs out/ --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from dngscan.analysis import analyze
from dngscan.grade import RENDER_MODE
from dngscan.raw_io import load_raw
from dngscan.render import render_output_u8
from dngscan.tone import build_render_plan

LUMA_REC2020 = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)

# Display luma below which a pixel's saturation is noise rather than colour. Calibrated
# on a mostly-black ISO-high neon frame (33 % of it renders fully black): at this floor
# the sign of the LibRaw/RAW 9 saturation difference stops depending on how much black
# the frame contains, and it still holds if the floor is raised to 0.30. See _stats.
SATURATION_LUMA_FLOOR = 0.15


def _stats(rgb_u8: np.ndarray) -> dict[str, float]:
    """Distribution summary that survives a change of frame geometry."""
    arr = np.asarray(rgb_u8, dtype=np.float32) / 255.0
    luma = arr @ LUMA_REC2020
    largest = arr.max(axis=2)
    smallest = arr.min(axis=2)
    saturation = np.where(largest > 1e-6, (largest - smallest) / np.maximum(largest, 1e-6), 0.0)
    # Saturation is only meaningful where there is signal. Near black the ratio is driven
    # by chroma noise, so on a dark frame a whole-image median compares noise rather than
    # colour — and it inverts the answer. On a neon frame the unmasked median read LibRaw
    # 0.580 against RAW 9's 0.385, which attributed -0.333 to the decode and looked like
    # the ML path desaturating the scene; above the floor the same pair reads 0.333
    # against 0.492, attributing +0.148, and the rendered images agree with the second
    # reading. The unmasked number was measuring RAW 9 removing LibRaw's chroma noise.
    # Coverage is reported alongside so a threshold that leaves few pixels is visible.
    lit = luma > SATURATION_LUMA_FLOOR
    lit_pct = float(100.0 * np.count_nonzero(lit) / luma.size)
    return {
        "luma_p1": float(np.percentile(luma, 1)),
        "luma_p50": float(np.median(luma)),
        "luma_p99": float(np.percentile(luma, 99)),
        "sat_p50": float(np.median(saturation[lit])) if np.any(lit) else float("nan"),
        "sat_coverage_pct": lit_pct,
        "black_pct": float(100.0 * np.count_nonzero(luma < 1.0 / 255.0) / luma.size),
        "white_pct": float(100.0 * np.count_nonzero(luma > 254.0 / 255.0) / luma.size),
    }


def _pixelwise(a: np.ndarray, b: np.ndarray) -> dict[str, float] | None:
    """Exact difference, or None when the two frames are not the same geometry."""
    if a.shape != b.shape:
        return None
    diff = np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)) / 255.0
    return {
        "diff_p50": float(np.median(diff)),
        "diff_max": float(diff.max()),
        "changed_pct": float(
            100.0 * np.count_nonzero(diff.max(axis=2) > 1.0 / 255.0) / diff[:, :, 0].size
        ),
    }


def _ev(ratio: float) -> float:
    return float(np.log2(max(ratio, 1e-9)))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="RAW/DNG file")
    parser.add_argument(
        "--full",
        action="store_true",
        help="decode at full resolution (default: half size, much faster and enough for attribution)",
    )
    parser.add_argument(
        "--output-gamut", choices=("srgb", "p3"), default="p3",
        help="output space for the four renders (default p3)",
    )
    parser.add_argument("--tone-core", default="agx", help="tone core for both paths (default agx)")
    parser.add_argument(
        "--write-jpegs", type=Path, default=None,
        help="optional directory to write the four renders as a_*.jpg .. d_*.jpg",
    )
    args = parser.parse_args(argv)

    if not args.path.is_file():
        parser.error(f"not a file: {args.path}")

    half = not args.full
    bundles: dict[str, object] = {}
    analyses: dict[str, object] = {}
    plans: dict[str, object] = {}
    for decoder in ("libraw", "coreimage"):
        try:
            bundle = load_raw(args.path, scene_half_size=half, decoder=decoder)
        except RuntimeError as exc:
            print(f"{decoder} 解码不可用：{exc}", file=sys.stderr)
            print("本工具需要两条管线都能解码同一个文件才能做归因。", file=sys.stderr)
            return 2
        analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
        bundles[decoder] = bundle
        analyses[decoder] = analysis
        plans[decoder] = build_render_plan(
            bundle, analysis, RENDER_MODE, args.output_gamut, tone_core=args.tone_core
        )

    print(f"{args.path.name}  ({'full' if args.full else 'half'} size, gamut={args.output_gamut})")
    print("\n编译出的计划（各自缓冲）")
    for decoder in ("libraw", "coreimage"):
        tone = plans[decoder].tone
        print(
            f"  {decoder:9s} black={tone.black_ev:+6.2f} white={tone.white_ev:+5.2f} "
            f"DR={tone.white_ev - tone.black_ev:5.2f} punch={getattr(tone, 'punch_strength', float('nan')):.3f}"
        )

    cells = {
        "A": ("libraw", "libraw", "原生基准"),
        "B": ("coreimage", "coreimage", "原生 = 平常导出所得"),
        "C": ("coreimage", "libraw", "钉住计划 → 只剩解码差异"),
        "D": ("libraw", "coreimage", "钉住缓冲 → 只剩计划差异"),
    }
    renders: dict[str, np.ndarray] = {}
    print("\n2x2 渲染")
    for key, (buf, plan, note) in cells.items():
        renders[key] = render_output_u8(
            bundles[buf], analyses[buf], args.output_gamut,
            tone_plan=plans[plan], tone_core=args.tone_core,
        )
        s = _stats(renders[key])
        print(
            f"  {key}  缓冲={buf:9s} 计划={plan:9s} "
            f"亮度 p1/p50/p99={s['luma_p1']:.4f}/{s['luma_p50']:.4f}/{s['luma_p99']:.4f} "
            f"饱和={s['sat_p50']:.4f}(亮部{s['sat_coverage_pct']:4.1f}%) "
            f"纯黑={s['black_pct']:5.2f}% 纯白={s['white_pct']:5.2f}%   {note}"
        )

    # Attributing on median luma alone is misleading: the plan's effect is concentrated
    # in the highlights, where a narrower window clips what a wider one rolls off, and
    # the median cannot see it. Report the midtones, the highlight tail and chroma
    # separately so a near-zero contribution in one is not read as "no effect".
    print("\n归因（每个统计量各自分解；比值取 log2，饱和与纯白为绝对差）")
    stats = {k: _stats(v) for k, v in renders.items()}
    rows = (
        ("中间调 luma p50", "luma_p50", "ev"),
        ("高光   luma p99", "luma_p99", "ev"),
        ("饱和 sat p50(亮部)", "sat_p50", "abs"),
        ("纯白像素 %", "white_pct", "abs"),
    )
    print(f"  {'统计量':18s} {'原生 A→B':>11s} {'解码 A→C':>11s} {'计划 A→D':>11s} {'交互残差':>11s}")
    for label, field, kind in rows:
        a, b, c, d = (stats[k][field] for k in "ABCD")
        if kind == "ev":
            total, dec, pln = _ev(b / a), _ev(c / a), _ev(d / a)
            unit = "EV"
        else:
            total, dec, pln = b - a, c - a, d - a
            unit = ""
        print(
            f"  {label:18s} {total:+10.4f}{unit:>2s} {dec:+10.4f}{unit:>2s} "
            f"{pln:+10.4f}{unit:>2s} {total - dec - pln:+10.4f}{unit:>2s}"
        )
    print(
        "  解码列在计划固定时测得，计划列在缓冲固定时测得；两者相加不等于原生总差异的部分"
        "落在交互残差里。"
    )

    print("\n逐像素比较（仅同缓冲的一对有效）")
    for label, x, y in (("A vs D  同 libraw 缓冲，两套计划", "A", "D"),
                        ("B vs C  同 coreimage 缓冲，两套计划", "B", "C"),
                        ("A vs C  同计划，两个缓冲", "A", "C")):
        px = _pixelwise(renders[x], renders[y])
        if px is None:
            print(f"  {label}: 几何不同（Core Image 执行了 DNG opcode），只能按分布比较")
        else:
            print(
                f"  {label}: 中位差={px['diff_p50']:.6f} 最大={px['diff_max']:.4f} "
                f"差异像素={px['changed_pct']:.2f}%"
            )

    if args.write_jpegs is not None:
        from PIL import Image

        args.write_jpegs.mkdir(parents=True, exist_ok=True)
        for key, (buf, plan, _note) in cells.items():
            out = args.write_jpegs / f"{key.lower()}_{buf}_buf__{plan}_plan.jpg"
            Image.fromarray(renders[key]).save(out, quality=95)
            print(f"  写出 {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
