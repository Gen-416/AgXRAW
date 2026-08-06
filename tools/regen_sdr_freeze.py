#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate the SDR freeze fixtures.

These fixtures pin Display-P3 SDR linear + u8 outputs that HDR work must not change.
Do not regenerate casually: any intentional update needs an explicit review gate
(see docs/HDR_AGX_V2_IMPLEMENTATION_PLAN.zh-CN.md).
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
from dngscan.tone import build_render_plan, compute_exposure_gain, exposure_mode_for_tone_core
from tests.golden_support import GOLDEN_DIR, all_scenes

FREEZE_DIR = ROOT / "tests" / "sdr_freeze"
MANIFEST_PATH = FREEZE_DIR / "MANIFEST.json"

# Minimal matrix: enough to freeze HDR-adjacent P3 SDR behaviour without exploding size.
FREEZE_SCENE_IDS = (
    "daylight_wide_dr",
    "night_sparse_lamps",
    "crop__SDI0150",
    "crop__SDI0238",
)
FREEZE_EVS = (0.0, 1.0)


@dataclass(frozen=True)
class FreezeCase:
    scene_id: str
    ev: float
    decoder: str = "libraw"
    tone_core: str = "agx"
    agx_primaries: str = "smooth"
    output_gamut: str = "p3"

    @property
    def stem(self) -> str:
        ev_tag = f"ev{self.ev:+.2f}".replace(".", "p").replace("+", "p").replace("-", "m")
        return (
            f"{self.scene_id}__{self.decoder}__{self.tone_core}__"
            f"{self.agx_primaries}__{self.output_gamut}__{ev_tag}"
        )

    @property
    def path(self) -> Path:
        return FREEZE_DIR / f"{self.stem}.npz"


def golden_tree_digest(golden_dir: Path = GOLDEN_DIR) -> str:
    digest = hashlib.sha256()
    for path in sorted(golden_dir.glob("*.npz")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def iter_freeze_cases() -> list[FreezeCase]:
    scenes = all_scenes()
    cases: list[FreezeCase] = []
    for scene_id in FREEZE_SCENE_IDS:
        if scene_id not in scenes:
            continue
        for ev in FREEZE_EVS:
            cases.append(FreezeCase(scene_id=scene_id, ev=float(ev)))
    return cases


def render_freeze_case(case: FreezeCase) -> tuple[np.ndarray, np.ndarray]:
    scene = all_scenes()[case.scene_id]
    bundle = scene.bundle
    bundle.exposure_gain = compute_exposure_gain(
        exposure_mode_for_tone_core(case.tone_core), case.ev
    )
    plan = build_render_plan(
        bundle,
        scene.analysis,
        "agx",
        case.output_gamut,
        tone_core=case.tone_core,
        agx_primaries=case.agx_primaries,
    )
    linear = render_output_linear(
        bundle,
        scene.analysis,
        case.output_gamut,
        plan,
        tone_core=case.tone_core,
        agx_primaries=case.agx_primaries,
    )
    u8 = render_output_u8(
        bundle,
        scene.analysis,
        case.output_gamut,
        plan,
        tone_core=case.tone_core,
        agx_primaries=case.agx_primaries,
    )
    return np.asarray(linear, dtype=np.float32), np.asarray(u8, dtype=np.uint8)


def write_manifest(cases: list[FreezeCase], *, note: str) -> dict:
    fixtures: dict[str, str] = {}
    for case in cases:
        payload = case.path.read_bytes()
        fixtures[case.stem] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "phase": 0,
        "purpose": "SDR freeze baseline for ACES 2 HDR dual-rendition work",
        "note": note,
        "golden_tree_sha256": golden_tree_digest(),
        "golden_npz_count": len(list(GOLDEN_DIR.glob("*.npz"))),
        "freeze_cases": [asdict(case) for case in cases],
        "fixture_sha256": fixtures,
        "policy": (
            "Any intentional change to golden/ or sdr_freeze/ requires an explicit "
            "review gate. Do not run regen tools to silence a failing HDR phase."
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def regen(compare: bool = True) -> int:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    cases = iter_freeze_cases()
    if not cases:
        print("No freeze scenes available.", file=sys.stderr)
        return 1
    rows: list[str] = []
    for case in cases:
        linear, u8 = render_freeze_case(case)
        if compare and case.path.is_file():
            prev = np.load(case.path, allow_pickle=False)
            du8 = int(np.count_nonzero(prev["u8"] != u8))
            dlin = float(np.max(np.abs(prev["linear"].astype(np.float32) - linear)))
            rows.append(f"{case.stem}: u8_changed={du8} linear_max_abs={dlin:.6g}")
        np.savez_compressed(
            case.path,
            linear=linear.astype(np.float16),
            u8=u8,
            meta=np.asarray(
                json.dumps(
                    {
                        "scene_id": case.scene_id,
                        "ev": case.ev,
                        "decoder": case.decoder,
                        "tone_core": case.tone_core,
                        "agx_primaries": case.agx_primaries,
                        "output_gamut": case.output_gamut,
                        "exposure_gain": float(
                            compute_exposure_gain(
                                exposure_mode_for_tone_core(case.tone_core), case.ev
                            )
                        ),
                    }
                )
            ),
        )
        print(f"wrote {case.path.name}")
    manifest = write_manifest(
        cases,
        note="Generated by tools/regen_sdr_freeze.py; HDR phases must keep these bytes.",
    )
    if rows:
        print("Freeze delta summary:")
        for row in rows:
            print(f"  {row}")
    print(
        f"Manifest {MANIFEST_PATH.name}: golden_tree={manifest['golden_tree_sha256'][:16]}… "
        f"fixtures={len(cases)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip comparing against existing freeze fixtures while writing",
    )
    args = parser.parse_args(argv)
    return regen(compare=not args.no_compare)


if __name__ == "__main__":
    raise SystemExit(main())
