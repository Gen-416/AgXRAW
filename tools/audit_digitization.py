#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sampling-adequacy audit for chart-digitized curve assets.

Question answered: are the recorded anchors dense enough that the choice of
interpolant no longer matters? This is INTERPOLATION-DENSITY adequacy, not
absolute digitization accuracy: it cannot detect a systematic offset of the
whole anchor set from the printed curve (external review 4.5 — holdout
anchors and a cross-renderer repeat are the recorded follow-ups for that). Metric: the maximum disagreement between the
two reasonable reconstructions of the same anchors — piecewise-linear
(np.interp, what the runtime uses) and monotone cubic (PCHIP). The truth
lies between them, so this "sampling ambiguity" bounds the information loss
attributable to anchor density alone; it must stay comfortably below the
declared read-off uncertainty (gate: ambiguity <= uncertainty), otherwise
the digitization needs more anchors, not a fancier interpolant.

Domain edges are excluded (5% each side): no interpolant has information
beyond the first/last anchor, and toe endpoints divide by near-zero values.

Introduced with the 2026-08-25 precision audit, which found (and fixed via
second-pass crossing-scan anchors) mtf5207.R knee at 7.3% vs the declared
±5%, and 2383 sigma composition ambiguity up to 5.5%.

Usage:
    python tools/audit_digitization.py          # audit all known assets
    (also imported by tests/test_digitization_precision.py as the gate)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sampling_ambiguity(points, edge_frac: float = 0.05):
    """(interior_max_rel, interior_max_abs) between linear and PCHIP."""
    from scipy.interpolate import PchipInterpolator

    pts = np.asarray(points, float)
    x, y = pts[:, 0], pts[:, 1]
    dense = np.linspace(x[0], x[-1], 2000)
    lin = np.interp(dense, x, y)
    pch = PchipInterpolator(x, y)(dense)
    d = np.abs(lin - pch)
    lo = x[0] + edge_frac * (x[-1] - x[0])
    hi = x[-1] - edge_frac * (x[-1] - x[0])
    m = (dense > lo) & (dense < hi)
    rel = d / np.maximum(np.abs(pch), 1e-9)
    return float(rel[m].max()), float(d[m].max())


# (asset path, curve path in the JSON, gate kind, gate value)
# Gates mirror each asset's declared read-off uncertainty.
CHECKS = [
    ("dngscan/data/grain/granularity_5207.json", "sigma_density", "rel", 0.05),
    ("dngscan/data/grain/granularity_2383.json", "sigma_density", "rel", 0.05),
    ("dngscan/data/grain/granularity_5207.json", "density_loge", "abs", 0.03),
    ("dngscan/data/grain/granularity_2383.json", "density_loge", "abs", 0.03),
    # MTF's declared uncertainty is "±5% response" in response units
    # (absolute), so the gate is absolute; a relative gate would re-measure
    # read noise at the low-response tail once anchors are dense.
    ("dngscan/data/mtf/mtf_5207.json", None, "abs", 0.05),
    ("dngscan/data/mtf/mtf_2383.json", None, "abs", 0.05),
]


def run() -> list[tuple[str, float, float, bool]]:
    results = []
    for path, curve, kind, gate in CHECKS:
        raw = json.loads((ROOT / path).read_text())
        for ch, node in raw["channels"].items():
            pts = node[curve] if curve else node
            rel, absd = sampling_ambiguity(pts)
            value = rel if kind == "rel" else absd
            name = f"{Path(path).stem}.{ch}" + (f".{curve}" if curve else "")
            results.append((name, value, gate, value <= gate))
    return results


def main() -> int:
    ok = True
    for name, value, gate, passed in run():
        ok &= passed
        print(f"{name:44s} ambiguity {value:.4f} gate {gate} "
              f"[{'ok' if passed else 'FAIL'}]")
    print("audit:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
