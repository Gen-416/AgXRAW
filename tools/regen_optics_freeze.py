#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate the standing analog-optics drift freeze and its measured baseline.

Two artefacts, one command:

1. **Byte freeze** — the rendered u8 and linear output of every freeze scene at
   the GUI's `light` and `standard` optics tiers. First written before P1 as a
   pin on the legacy bytes; P1-P7 landed and re-pinned it deliberately, so it
   now gates drift in the CURRENT output. A change that intends to move these
   bytes has to say so and regenerate deliberately.

2. **Measured baseline** — `BASELINE.json`, the numbers from
   `tools/film_optics_report.py`. This is what turns "the grain looks bad" into
   a value with a unit — the standing baseline each phase P2-P4 was judged
   against rather than admired.

Do not run this to make a failing test pass. A freeze that gets regenerated
whenever it complains is not a freeze.

    python tools/regen_optics_freeze.py            # regenerate, report drift
    python tools/regen_optics_freeze.py --check    # verify only, no writes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["DNGSCAN_FAST"] = "0"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from dngscan.render import render_output_linear, render_output_u8
from dngscan.tone import build_render_plan
from tests.golden_support import all_scenes
from tools.film_optics_report import TIERS, build_report

FREEZE_DIR = ROOT / "tests" / "optics_freeze"
MANIFEST_PATH = FREEZE_DIR / "MANIFEST.json"
BASELINE_PATH = FREEZE_DIR / "BASELINE.json"

# Scenes chosen for what they can prove, not for coverage. The two real crops
# are the only photographic material in the repo; the synthetics supply the
# failure modes the crops happen not to contain (isolated night sources, a
# blown high key). §13 P0 asks for five real RAWs — the repo carries two, and
# the report says so rather than passing synthetics off as photographs.
FREEZE_SCENE_IDS = (
    "night_sparse_lamps",
    "daylight_wide_dr",
    "high_key",
    "crop__SDI0150",
    "crop__SDI0238",
)
FREEZE_STOCK = "portra400"
FREEZE_SEED = 0


@dataclass(frozen=True)
class OpticsFreezeCase:
    scene_id: str
    tier: str
    stock: str = FREEZE_STOCK
    seed: int = FREEZE_SEED

    @property
    def stem(self) -> str:
        return f"{self.scene_id}__{self.stock}__{self.tier}"

    @property
    def path(self) -> Path:
        return FREEZE_DIR / f"{self.stem}.npz"


def iter_cases() -> list[OpticsFreezeCase]:
    scenes = all_scenes()
    return [
        OpticsFreezeCase(scene_id=sid, tier=tier)
        for sid in FREEZE_SCENE_IDS
        if sid in scenes
        for tier in TIERS
    ]


def render_case(case: OpticsFreezeCase) -> tuple[np.ndarray, np.ndarray]:
    scene = all_scenes()[case.scene_id]
    grain, halation, bloom = TIERS[case.tier]
    plan = build_render_plan(
        scene.bundle,
        scene.analysis,
        "agx",
        "srgb",
        film_curve=case.stock,
        film_mode="full",
        film_crossover="datasheet",
        film_grain=grain,
        film_halation=halation,
        film_bloom=bloom,
        film_optics_seed=case.seed,
    )
    linear = render_output_linear(scene.bundle, scene.analysis, "srgb", plan)
    u8 = render_output_u8(scene.bundle, scene.analysis, "srgb", plan)
    return np.asarray(linear, dtype=np.float32), np.asarray(u8, dtype=np.uint8)


def write_manifest(cases: list[OpticsFreezeCase]) -> dict:
    manifest = {
        "phase": "film_optics_v2",
        "purpose": (
            "Standing freeze of the analog-optics render. First written "
            "before P1 as a legacy byte pin; P1-P7 landed and re-pinned it "
            "deliberately, so it now gates drift in the CURRENT output."
        ),
        "stock": FREEZE_STOCK,
        "seed": FREEZE_SEED,
        "tiers": {k: list(v) for k, v in TIERS.items()},
        "cases": [asdict(c) for c in cases],
        "fixture_sha256": {
            c.stem: hashlib.sha256(c.path.read_bytes()).hexdigest() for c in cases
        },
        "policy": (
            "Regenerate only with an explicit decision recorded in "
            "docs/FILM_OPTICS_V2_PLAN.zh-CN.md. Never to silence a test."
        ),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def regen(check: bool = False) -> int:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    cases = iter_cases()
    if not cases:
        print("no freeze scenes available", file=sys.stderr)
        return 1
    drift = 0
    for case in cases:
        linear, u8 = render_case(case)
        if case.path.is_file():
            prev = np.load(case.path, allow_pickle=False)
            changed = int(np.count_nonzero(prev["u8"] != u8))
            max_abs = float(
                np.max(np.abs(prev["linear"].astype(np.float32) - linear))
            )
            # The stored linear plane is float16; comparing it to a fresh
            # float32 render at exact equality flags every case as drifted
            # every time and trains the reader to ignore this line.
            if changed or max_abs > float(np.finfo(np.float16).eps) * 4:
                drift += 1
                print(f"{case.stem}: u8_changed={changed} linear_max_abs={max_abs:.6g}")
        if check:
            continue
        np.savez_compressed(
            case.path,
            linear=linear.astype(np.float16),
            u8=u8,
            meta=np.asarray(json.dumps(asdict(case))),
        )
        print(f"wrote {case.path.name}")
    if not check:
        write_manifest(cases)
        report = build_report(FREEZE_STOCK)
        BASELINE_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {BASELINE_PATH.name}")
    print(f"{len(cases)} cases, {drift} drifted")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify without writing")
    args = ap.parse_args()
    return regen(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
