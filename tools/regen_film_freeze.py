#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Film v2 P0: byte/hash freeze of the CURRENT film rendering surface.

FILM_PRINT_RENDERING_PLAN §12 P0: before any v2 work, the present behaviour of
`none` / `observe` / `full` is pinned so the two-stage kernel (P1) and every
later stage can prove what they changed and what they did not. Each golden
scene renders the six-way decomposition ladder per reference stock:

    agx_baseline    plain AgX, no film layer at all
    curve_only      the film's tone coordinate alone (film_curve)
    prefeed_only    the film's spectral separation alone (scene_transform @ 1.0)
    observe_combo   curve + prefeed at the declared pairing strength/primaries
                    (the WB layer cannot apply to synthetic golden bundles and
                    is deliberately absent — declared here, not hidden)
    full_bounded    takeover LUT, bounded grey-scale neutralization (off)
    full_datasheet  takeover LUT, datasheet drift

The manifest stores a SHA-256 of the u8 output bytes plus coarse stats. Do not
regenerate casually: any intentional change needs an explicit review gate; the
old single-LUT backend keeps THIS freeze while v2 validates against the
direct-chain oracle (plan §7.2).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FREEZE_DIR = REPO / "tests" / "film_freeze"
MANIFEST_PATH = FREEZE_DIR / "MANIFEST.json"

STOCKS = ("portra400", "velvia100")
SCHEMA = 1


def freeze_configs(stock: str) -> dict[str, dict]:
    """The declared parameter set of every ladder rung for one stock."""
    from dngscan.film_curve import film_style_pairing

    strength, primaries = film_style_pairing(stock)
    return {
        "curve_only": dict(
            film_curve=stock, film_mode="observe", film_crossover="off",
            scene_transform="none", scene_transform_strength=1.0,
            agx_primaries="base",
        ),
        "prefeed_only": dict(
            film_curve="none", film_mode="observe", film_crossover="off",
            scene_transform=f"{stock}_d55", scene_transform_strength=1.0,
            agx_primaries="base",
        ),
        "observe_combo": dict(
            film_curve=stock, film_mode="observe", film_crossover="off",
            scene_transform=f"{stock}_d55",
            scene_transform_strength=float(strength),
            agx_primaries=str(primaries),
        ),
        "full_bounded": dict(
            film_curve=stock, film_mode="full", film_crossover="off",
            scene_transform="none", scene_transform_strength=1.0,
            agx_primaries="base",
        ),
        "full_datasheet": dict(
            film_curve=stock, film_mode="full", film_crossover="datasheet",
            scene_transform="none", scene_transform_strength=1.0,
            agx_primaries="base",
        ),
    }


BASELINE = dict(
    film_curve="none", film_mode="observe", film_crossover="off",
    scene_transform="none", scene_transform_strength=1.0,
    agx_primaries="base",
)


def render_case(scene, params: dict):
    from dngscan.render import render_output_u8
    from dngscan.tone import build_render_plan

    plan = build_render_plan(
        scene.bundle,
        scene.analysis,
        "agx",
        "p3",
        film_curve=params["film_curve"],
        film_mode=params["film_mode"],
        film_crossover=params["film_crossover"],
    )
    return render_output_u8(
        scene.bundle,
        scene.analysis,
        "p3",
        tone_plan=plan,
        scene_transform=params["scene_transform"],
        scene_transform_strength=params["scene_transform_strength"],
        tone_core="agx",
        agx_primaries=params["agx_primaries"],
    )


def case_entry(scene_id: str, stock: str, config: str, params: dict, u8) -> dict:
    import numpy as np

    arr = np.ascontiguousarray(u8)
    return {
        "scene": scene_id,
        "stock": stock,
        "config": config,
        "params": params,
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "shape": list(arr.shape),
        "mean_u8": round(float(arr.mean()), 4),
        "p50_u8": float(np.percentile(arr, 50)),
    }


def main() -> int:
    import os

    os.environ.setdefault("DNGSCAN_FAST", "0")  # NumPy reference path
    from tests.golden_support import all_scenes

    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for scene_id, scene in sorted(all_scenes().items()):
        u8 = render_case(scene, BASELINE)
        entries.append(case_entry(scene_id, "none", "agx_baseline", BASELINE, u8))
        for stock in STOCKS:
            for config, params in freeze_configs(stock).items():
                u8 = render_case(scene, params)
                entries.append(case_entry(scene_id, stock, config, params, u8))
        print(f"{scene_id}: {1 + len(STOCKS) * 5} cases", flush=True)
    manifest = {
        "schema": SCHEMA,
        "note": (
            "film v2 P0 freeze (plan §12 P0). Regeneration requires an "
            "explicit review gate; see tools/regen_film_freeze.py."
        ),
        "cases": entries,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {MANIFEST_PATH} ({len(entries)} cases)")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
