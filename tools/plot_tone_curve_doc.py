#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Draw the two curve diagrams used by docs/EDITING_TUTORIAL.zh-CN.md.

The curves are not sketches: each line is the tone curve dngscan actually compiles for
the tutorial's own sample frame, evaluated on a neutral grey ramp through the AgX core.
Output is display-encoded (sRGB transfer) so the plot reads as screen brightness.

    python tools/plot_tone_curve_doc.py --bridge ~/Pictures/AgXRAW样张/DSCF0214.RAF \
        --lamp "~/Pictures/Original RAW 26-07-11 202403093.dng"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "assets" / "editing-tutorial"


def _curve(bundle, analysis, **kw):
    from dngscan.color import luminance_from_rec2020
    from dngscan.models import RenderAdjustments
    from dngscan.render import apply_agx_core
    from dngscan.tone import build_render_plan

    endpoint = kw.pop("endpoint_mode", "adaptive")
    plan = build_render_plan(
        bundle, analysis, "agx", "srgb", endpoint_mode=endpoint,
        adjustments=RenderAdjustments(**kw),
    )
    ev = np.linspace(-12.0, 7.0, 761)
    ramp = (0.18 * np.exp2(ev))[:, None].repeat(3, axis=1).astype(np.float32)
    y = np.clip(luminance_from_rec2020(apply_agx_core(ramp, plan.tone)), 0.0, 1.0)
    enc = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return ev, 100.0 * enc, plan.tone


def _load(path: Path):
    from dngscan.analysis import analyze
    from dngscan.raw_io import load_raw

    bundle = load_raw(path, scene_half_size=True)
    analysis, _y, _ev = analyze(bundle, margin=4, diagnostics=False)
    return bundle, analysis


def _style(ax, title: str) -> None:
    ax.set_title(title, fontsize=13, loc="left")
    ax.set_xlabel("场景亮度（档，0 = 中灰）")
    ax.set_ylabel("屏幕上的亮度（%，0 = 黑，100 = 白）")
    ax.set_ylim(-2, 104)
    ax.grid(alpha=0.25)
    ax.axvline(0, color="#888", lw=0.8, ls=":")


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Heiti SC", "Arial Unicode MS", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", type=Path, required=True)
    ap.add_argument("--lamp", type=Path, required=True)
    args = ap.parse_args()

    bundle, analysis = _load(args.bridge.expanduser())
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=125)
    for kw, label, color in (
        ({}, "默认（场景自适应）", "#777777"),
        ({"endpoint_mode": "evidence"}, "传感器实测范围", "#2a7fdb"),
        ({"endpoint_mode": "evidence", "toe_end_offset": -2.0}, "传感器实测范围 + 暗部收黑 −2", "#e0742a"),
    ):
        ev, enc, tone = _curve(bundle, analysis, **dict(kw))
        ax.plot(ev, enc, lw=2.2, color=color, label=f"{label}（黑点 {tone.black_ev:+.1f} 档）")
    ax.set_xlim(-11, 5)
    ax.axvspan(-11, -4, color="#000", alpha=0.05)
    ax.text(-10.7, 96, "暗部（曲线的趾部）", fontsize=11, color="#333")
    ax.text(-3.4, 96, "中间调", fontsize=11, color="#333")
    ax.text(2.3, 96, "高光（肩部）", fontsize=11, color="#333")
    _style(ax, "样张一（逆光钢桥）：暗部怎样沉进黑")
    ax.legend(loc="center left", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "curve_toe.png")
    plt.close(fig)

    bundle, analysis = _load(args.lamp.expanduser())
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=125)
    for off, label, color in ((-1.0, "高光收白 −1（更硬）", "#c0392b"), (0.0, "高光收白 0（默认）", "#777777"), (2.0, "高光收白 +2（更柔）", "#2a7fdb")):
        ev, enc, tone = _curve(bundle, analysis, shoulder_white_offset=off)
        ax.plot(ev, enc, lw=2.2, color=color, label=label)
    ax.set_xlim(-7, 7)
    ax.axvspan(2, 7, color="#000", alpha=0.05)
    ax.text(2.3, 6, "高光（曲线的肩部）", fontsize=11, color="#333")
    _style(ax, f"样张三（手办与灯）：高光怎样升到白（白点 {tone.white_ev:+.1f} 档）")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "curve_shoulder.png")
    plt.close(fig)
    print("wrote", OUT / "curve_toe.png", "and", OUT / "curve_shoulder.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
