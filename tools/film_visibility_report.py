#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real-photo visibility report for the appearance layer (A13 item 4).

The synthetic-wheel gates hold the recipes SAFE (neutral axis, caps,
directional structure) but bound nothing about how much of a real
photograph visibly changes. This tool renders real RAWs through
isolated pairs and reports honest per-pixel numbers in the review's own
units (mean |ΔL|/255, mean |ΔC|/255) plus the share of pixels whose
chroma moves past visibility thresholds — the evidence base for any
strength decision. Measurement only: no pipeline change, no verdict.

    python tools/film_visibility_report.py <raw> <stock> [...]
    python tools/film_visibility_report.py --default-matrix
"""
from __future__ import annotations

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

SAMPLES = Path.home() / "Pictures/AgXRAW样张"
DEFAULT_MATRIX = (
    ("_SDI0094.DNG", "ektar100"),
    ("_SDI0094.DNG", "portra400"),
    ("_SDI0173.DNG", "vision3250d"),
    ("_SDI0231.DNG", "velvia100"),
)


def _render(bundle, analysis, stock, **kw):
    plan = build_render_plan(
        bundle, analysis, RENDER_MODE, "srgb",
        film_curve=stock, film_mode="full", **kw,
    )
    return np.asarray(
        render_output_u8(bundle, analysis, "srgb", plan), np.float64
    )


def _delta(a, b):
    """Luma/chroma deltas in /255 units (the review's convention)."""
    luma = np.array([0.2126, 0.7152, 0.0722])
    dl = np.abs((a - b) @ luma)
    # chroma = the residual after removing the luma-matched grey component
    ga = a - (a @ luma)[..., None] * np.ones(3)
    gb = b - (b @ luma)[..., None] * np.ones(3)
    dc = np.linalg.norm(ga - gb, axis=-1)
    return dl, dc


def report(stem: str, stock: str) -> None:
    path = SAMPLES / stem
    bundle = load_raw(path, "clip")
    analysis, _, _ = analyze(bundle, 4, diagnostics=False)
    base_pb = _render(bundle, analysis, stock,
                      film_crossover="print", film_appearance="technical")
    ref = _render(bundle, analysis, stock, film_appearance="reference")
    ref15 = _render(bundle, analysis, stock, film_appearance="reference",
                    film_appearance_strength=1.5)
    ref30 = _render(bundle, analysis, stock, film_appearance="reference",
                    film_appearance_strength=3.0)
    print(f"\n== {stem} × {stock}")
    for label, img in (("reference 1.0", ref), ("reference 1.5", ref15),
                       ("reference 3.0", ref30)):
        dl, dc = _delta(img, base_pb)
        line = (
            f"  {label:14s} 对 print-balanced technical: "
            f"|ΔL| mean {dl.mean():5.2f}/255  "
            f"|ΔC| mean {dc.mean():5.2f}/255  "
        )
        for th in (2.0, 5.0, 10.0):
            line += f"ΔC>{th:g}: {float((dc > th).mean() * 100):5.1f}%  "
        print(line)


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] == "--default-matrix":
        matrix = DEFAULT_MATRIX
    else:
        matrix = tuple(zip(args[0::2], args[1::2]))
    for stem, stock in matrix:
        report(stem, stock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
