#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate committed golden-render fixtures (NumPy reference path only)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["DNGSCAN_FAST"] = "0"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from dngscan.render import render_output_u8
from tests.golden_support import (
    GOLDEN_DIR,
    GoldenCase,
    all_scenes,
    iter_cases,
    oklab_stats,
    render_plan_for_case,
)


def render_case(case: GoldenCase) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    scene = all_scenes()[case.scene_id]
    plan = render_plan_for_case(scene, case)
    u8 = render_output_u8(
        scene.bundle,
        scene.analysis,
        "srgb",
        plan,
        tone_core=case.tone_core,
        agx_primaries=case.agx_primaries if case.tone_core == "agx" else "base",
    )
    stats = {name: oklab_stats(u8, mask) for name, mask in scene.rois.items()}
    return u8.reshape(scene.bundle.scene_rec2020_render.shape), stats


def _delta_table(old: np.ndarray, new: np.ndarray) -> str:
  diff = np.abs(old.astype(np.int16) - new.astype(np.int16))
  changed = int(np.count_nonzero(diff))
  return f"changed_px={changed}/{old.size} max_abs={int(diff.max()) if changed else 0}"


def regen_all(compare: bool = True) -> int:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    cases = list(iter_cases())
    rows: list[str] = []
    for case in cases:
        u8, stats = render_case(case)
        path = case.fixture_path
        if compare and path.is_file():
            prev = np.load(path)["u8"]
            rows.append(f"{case.fixture_name}: {_delta_table(prev, u8)}")
        # JSON-encoded stats keep the fixture loadable with allow_pickle=False: a
        # pickled object array in a committed fixture would be an arbitrary-code
        # execution vector for anyone running tests on a contributed PR.
        np.savez_compressed(path, u8=u8, stats=np.asarray(json.dumps(stats)))
    if rows:
        print("Golden delta summary:")
        for row in rows:
            print(f"  {row}")
    print(f"Wrote {len(cases)} fixtures to {GOLDEN_DIR}")
    return 0


def export_dng_crop(dng: Path, out: Path, *, size: tuple[int, int] = (96, 64)) -> None:
    from dngscan.analysis import analyze
    from dngscan.raw_io import load_raw

    bundle = load_raw(dng, scene_half_size=True)
    scene = bundle.scene_rec2020_render
    h, w = scene.shape[:2]
    th, tw = size
    y0 = max(0, (h - th) // 2)
    x0 = max(0, (w - tw) // 2)
    crop = scene[y0 : y0 + th, x0 : x0 + tw].copy()
    analysis, _, _ = analyze(bundle, 4, diagnostics=False)
    np.savez_compressed(
        out,
        scene=crop,
        source=str(dng),
        origin=(y0, x0),
        analysis_median_ev=float(analysis.ev_median),
    )
    print(f"Wrote crop {crop.shape} from {dng.name} -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-dng", type=Path, help="Optional DNG to export a scene-linear crop fixture")
    parser.add_argument("--crop-out", type=Path, help="Output .npz for --from-dng")
    args = parser.parse_args()
    if args.from_dng is not None:
        out = args.crop_out or GOLDEN_DIR / f"crop__{args.from_dng.stem}.npz"
        export_dng_crop(args.from_dng, out)
        return 0
    return regen_all()


if __name__ == "__main__":
    raise SystemExit(main())
