#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Import a JPTC collect set (y-g-jiang first-party bench measurements).

A collect set directory (https://y-g-jiang.github.io/data/collect/<id>/)
holds up to four instruments for one camera+mode:

  ptc-iso*.csv    JPTC/2       exposure ramp  -> absolute gain / FWC anchor
  gain-levels.csv JPTC-ISOGAIN/1 per-ISO means -> RELATIVE gain vs ISO
                  (g1/g2 = (t1/t2)*(M2-BL2)/(M1-BL1), the file's own formula)
  dark-scalars.csv JPTC-DARK/1  paired darks   -> temporal read noise per ISO
                  (StdDiff is the std of A-B verbatim; /sqrt(2) for one
                  frame, /sqrt(ClipVarianceFactor) undoes the declared
                  sigma clip), plus row/column banding decomposition
  spectrum-h/v.csv JPTC-SPECTRUM/1 noise power spectra -> whiteness metric
                  (high-band over mid-band mean power of the pair-difference
                  spectrum: ~1 = white/clean, <1 = spatial filtering baked
                  into the RAW, >1 = sharpening)

Derived entry (format dngscan-jptc-collect-1): absolute gain curve
gain(iso) anchored at the PTC fit, read-noise curves in DN and electrons
(Sheppard-corrected, unresolved when the correction floors), plateau
gain-jump candidates (symmetric test; extended-ISO boundary vs DCG left
undistinguished), fwc as ADC code capacity with estimator spread,
noise-whiteness evidence, and raw within-row/col metrics (semantics
unconfirmed upstream). Licensing: credit-based grant 2026-08-25
(NOTICE.md).

Usage:
    python tools/import_jptc_collect.py <set-dir> --out dngscan/data/priors/jptc_collect/<id>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dngscan.calibration import (  # noqa: E402
    _anchor_evidence, _find, _parse_rows, _sha256, build, ptc_anchor, read_dark, read_isogain, read_whiteness, fit_ptc, sanitize_json,
)

GREEN_INDICES = {1, 3}          # LibRaw colour indices for G/G2
CLIP_FRAC_MAX = 0.01


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets-json", type=Path, default=None,
                    help="sets.json for tester/measuredAt metadata")
    args = ap.parse_args()
    meta = None
    if args.sets_json and args.sets_json.exists():
        for s in json.loads(args.sets_json.read_text()).get("sets", []):
            if s.get("id") == args.set_dir.name:
                meta = s
    entry = build(args.set_dir, meta)
    entry = sanitize_json(entry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entry, ensure_ascii=False, indent=1,
                                   allow_nan=False) + "\n")
    haves = ", ".join(entry["source"]["formats"])
    print(f"wrote {args.out.name}: {entry['camera']} [{haves}] "
          f"rn_pts={len(entry.get('read_noise_dn_log2iso', []))} "
          f"gain_pts={len(entry.get('gain_log2iso_log2epd', []))} "
          f"jumps={entry.get('gain_jump_isos')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
