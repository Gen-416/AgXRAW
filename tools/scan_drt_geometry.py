#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scan DRT geometry over synthetic EV × hue × chroma samples (gated vs agx vs lum)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dngscan.color import luminance_from_rec2020, rgb_to_oklab
from dngscan.gated_drt import apply_gated_core
from dngscan.models import ColorGeometryPlan, ToneCompressionPlan
from dngscan.render import apply_agx_core, apply_tone_core


def _base_plan(**kwargs) -> ToneCompressionPlan:
    defaults = dict(
        target_gamut="Rec2020",
        luma_p1=0.01,
        luma_p50=0.18,
        luma_p99=1.0,
        luma_p999=2.0,
        black_ev=-7.0,
        white_ev=4.5,
        dynamic_range_ev=11.5,
        contrast=3.0,
        toe_power=1.5,
        shoulder_power=3.3,
        chroma_p95=0.0,
        negative_rgb_pct=0.0,
        over_rgb_pct=0.0,
        use_c1_endpoints=True,
    )
    defaults.update(kwargs)
    return ToneCompressionPlan(**defaults)


def _hsv_sample(hue_deg: float, chroma: float, value_ev: float) -> np.ndarray:
    h = hue_deg / 60.0
    c = chroma
    x = c * (1 - abs(h % 2 - 1))
    if h < 1:
        rgb = (c, x, 0.0)
    elif h < 2:
        rgb = (x, c, 0.0)
    elif h < 3:
        rgb = (0.0, c, x)
    elif h < 4:
        rgb = (0.0, x, c)
    elif h < 5:
        rgb = (x, 0.0, c)
    else:
        rgb = (c, 0.0, x)
    rgb = np.clip(np.array(rgb, dtype=np.float64), 0.0, None)
    y_target = 0.18 * (2.0 ** value_ev)
    y = float(luminance_from_rec2020(rgb.reshape(1, 3))[0])
    if y > 1e-9:
        rgb = rgb * (y_target / y)
    return rgb.astype(np.float32)


def _mean_chroma(rgb: np.ndarray) -> float:
    _, a, b = rgb_to_oklab(rgb.reshape(-1, 3), "srgb")
    return float(np.mean(np.hypot(a, b)))


def scan(
    hue_steps: int = 12,
    chroma_steps: int = 5,
    ev_steps: int = 7,
    ev_lo: float = -1.0,
    ev_hi: float = 3.0,
) -> list[dict[str, float | str]]:
    color = ColorGeometryPlan("srgb", 0.0, 0.0)
    masks = np.zeros((1, 3), dtype=np.float32)
    hues = np.linspace(0.0, 330.0, hue_steps, endpoint=False)
    chromas = np.linspace(0.08, 0.55, chroma_steps)
    evs = np.linspace(ev_lo, ev_hi, ev_steps)
    rows: list[dict[str, float | str]] = []
    for hue in hues:
        for chroma in chromas:
            for ev in evs:
                rgb = _hsv_sample(float(hue), float(chroma), float(ev)).reshape(1, 3)
                gated = apply_gated_core(
                    rgb, _base_plan(tone_core="gated", agx_primaries="smooth"), color, masks
                )
                agx = apply_agx_core(rgb, _base_plan(tone_core="agx", agx_primaries="base"))
                lum = apply_tone_core(rgb, _base_plan(tone_core="lum"))
                rows.append(
                    {
                        "hue_deg": round(float(hue), 2),
                        "chroma": round(float(chroma), 4),
                        "ev": round(float(ev), 3),
                        "chroma_gated": round(_mean_chroma(gated), 6),
                        "chroma_agx": round(_mean_chroma(agx), 6),
                        "chroma_lum": round(_mean_chroma(lum), 6),
                        "delta_gated_agx": round(_mean_chroma(gated) - _mean_chroma(agx), 6),
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--hue-steps", type=int, default=12)
    parser.add_argument("--chroma-steps", type=int, default=5)
    parser.add_argument("--ev-steps", type=int, default=7)
    parser.add_argument("--ev-lo", type=float, default=-1.0)
    parser.add_argument("--ev-hi", type=float, default=3.0)
    args = parser.parse_args(argv)
    rows = scan(args.hue_steps, args.chroma_steps, args.ev_steps, args.ev_lo, args.ev_hi)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} samples to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
