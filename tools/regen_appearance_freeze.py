#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Freeze film `technical` behaviour before the appearance layer exists
(FILM_APPEARANCE_RECIPE_PLAN §16 P0).

Three artefacts:

1. **Probe freeze** — the §14.1 palette volume as mapped Rec.2020, per stock,
   in both `observe` and `full`. Exact float32: 743 samples is small enough
   that there is no reason to quantise, and this is the surface every later
   colour claim is measured against.
2. **Render freeze** — u8 and linear output of full technical on real and
   synthetic scenes. The probe cannot catch a regression in exposure, gamut
   fit or delivery; rendered bytes can.
3. **BASELINE.json** — `tools/film_palette_probe.py`'s report, which is what
   makes "full looks weak" a claim about specific hues and exposures.

The manifest also pins the `tests/golden` tree hash: P1's exit gate is that
`technical` is byte-identical to today, and the golden tree is where that is
actually enforced.

    python tools/regen_appearance_freeze.py
    python tools/regen_appearance_freeze.py --check
"""
from __future__ import annotations

import argparse
import hashlib
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

from dngscan import film_palette_diag as pal
from dngscan.render import render_output_linear, render_output_u8
from dngscan.tone import build_render_plan
from tests.golden_support import GOLDEN_DIR, all_scenes
from tools.film_palette_probe import PROBE_STOCKS, build_report, render_probe

FREEZE_DIR = ROOT / "tests" / "appearance_freeze"
MANIFEST_PATH = FREEZE_DIR / "MANIFEST.json"
BASELINE_PATH = FREEZE_DIR / "BASELINE.json"

RENDER_SCENES = ("crop__SDI0150", "crop__SDI0238", "night_sparse_lamps", "high_key")
RENDER_STOCK = "portra400"


def golden_tree_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(GOLDEN_DIR.glob("*.npz")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def probe_path(stock: str, mode: str) -> Path:
    return FREEZE_DIR / f"probe__{stock}__{mode}.npz"


def render_path(scene_id: str) -> Path:
    return FREEZE_DIR / f"render__{scene_id}__{RENDER_STOCK}__technical.npz"


def render_technical(scene_id: str) -> tuple[np.ndarray, np.ndarray]:
    scene = all_scenes()[scene_id]
    plan = build_render_plan(
        scene.bundle, scene.analysis, "agx", "srgb",
        film_curve=RENDER_STOCK, film_mode="full", film_crossover="datasheet",
    )
    linear = render_output_linear(scene.bundle, scene.analysis, "srgb", plan)
    u8 = render_output_u8(scene.bundle, scene.analysis, "srgb", plan)
    return np.asarray(linear, dtype=np.float32), np.asarray(u8, dtype=np.uint8)


def regen(check: bool = False) -> int:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    volume, _ = pal.palette_volume()
    drift = 0

    for stock in PROBE_STOCKS:
        for mode in ("observe", "full"):
            mapped = render_probe(volume, stock, mode).astype(np.float32)
            path = probe_path(stock, mode)
            if path.is_file():
                prev = np.load(path, allow_pickle=False)["mapped"]
                if not np.array_equal(prev, mapped):
                    drift += 1
                    print(
                        f"{path.name}: max_abs="
                        f"{float(np.max(np.abs(prev - mapped))):.6g}"
                    )
            if not check:
                np.savez_compressed(path, mapped=mapped)
                print(f"wrote {path.name}")

    for scene_id in RENDER_SCENES:
        if scene_id not in all_scenes():
            continue
        linear, u8 = render_technical(scene_id)
        path = render_path(scene_id)
        if path.is_file():
            prev = np.load(path, allow_pickle=False)
            changed = int(np.count_nonzero(prev["u8"] != u8))
            max_abs = float(np.max(np.abs(prev["linear"].astype(np.float32) - linear)))
            if changed or max_abs > float(np.finfo(np.float16).eps) * 4:
                drift += 1
                print(f"{path.name}: u8_changed={changed} linear_max_abs={max_abs:.6g}")
        if not check:
            np.savez_compressed(
                path, linear=linear.astype(np.float16), u8=u8,
                meta=np.asarray(json.dumps(
                    {"scene_id": scene_id, "stock": RENDER_STOCK, "mode": "full"}
                )),
            )
            print(f"wrote {path.name}")

    if not check:
        BASELINE_PATH.write_text(
            json.dumps(build_report(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {BASELINE_PATH.name}")
        fixtures = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(FREEZE_DIR.glob("*.npz"))
        }
        MANIFEST_PATH.write_text(
            json.dumps(
                {
                    "phase": "appearance_p0",
                    "purpose": (
                        "Film `technical` behaviour frozen before the "
                        "reference-print appearance layer exists."
                    ),
                    "probe_stocks": list(PROBE_STOCKS),
                    "render_scenes": list(RENDER_SCENES),
                    "render_stock": RENDER_STOCK,
                    "golden_tree_sha256": golden_tree_digest(),
                    "golden_npz_count": len(list(GOLDEN_DIR.glob("*.npz"))),
                    "fixture_sha256": fixtures,
                    "policy": (
                        "The appearance layer must leave `technical` unchanged. "
                        "Regenerate only with a recorded decision."
                    ),
                },
                indent=2, sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {MANIFEST_PATH.name}")
    print(f"{drift} drifted")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    return regen(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
