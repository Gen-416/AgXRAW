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

The manifest pins only these appearance fixtures.  The unrelated general
golden tree has its own tests; hashing it here would make adding an unrelated
scene look like a film-technical regression.

    python tools/regen_appearance_freeze.py
    python tools/regen_appearance_freeze.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from tests.golden_support import all_scenes
from tools.film_palette_probe import PROBE_STOCKS, build_report, render_probe

FREEZE_DIR = ROOT / "tests" / "appearance_freeze"
MANIFEST_PATH = FREEZE_DIR / "MANIFEST.json"
BASELINE_PATH = FREEZE_DIR / "BASELINE.json"

RENDER_SCENES = ("crop__SDI0150", "crop__SDI0238", "night_sparse_lamps", "high_key")
RENDER_STOCK = "portra400"
REPORT_RTOL = 1e-4
# NumPy/native mapped RGB differs by at most 1e-6, but hue around the nearly
# neutral highlight tail amplifies that to 0.00268 degrees on the frozen probe.
# Five thousandths is still orders below every appearance gate while covering
# the measured backend portability bound with margin.
REPORT_ATOL = 5e-3


def probe_path(stock: str, mode: str) -> Path:
    return FREEZE_DIR / f"probe__{stock}__{mode}.npz"


def render_path(scene_id: str) -> Path:
    return FREEZE_DIR / f"render__{scene_id}__{RENDER_STOCK}__technical.npz"


def expected_fixture_paths() -> tuple[Path, ...]:
    probes = tuple(
        probe_path(stock, mode)
        for stock in PROBE_STOCKS
        for mode in ("observe", "full")
    )
    renders = tuple(render_path(scene_id) for scene_id in RENDER_SCENES)
    return probes + renders


def render_technical(scene_id: str) -> tuple[np.ndarray, np.ndarray]:
    scene = all_scenes()[scene_id]
    plan = build_render_plan(
        scene.bundle, scene.analysis, "agx", "srgb",
        film_curve=RENDER_STOCK, film_mode="full", film_crossover="off",
    )
    linear = render_output_linear(scene.bundle, scene.analysis, "srgb", plan)
    u8 = render_output_u8(scene.bundle, scene.analysis, "srgb", plan)
    return np.asarray(linear, dtype=np.float32), np.asarray(u8, dtype=np.uint8)


def _reports_close(expected: object, actual: object) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(expected) == set(actual)
            and all(_reports_close(expected[k], actual[k]) for k in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(_reports_close(x, y) for x, y in zip(expected, actual))
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            x, y = float(expected), float(actual)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        if math.isnan(x):
            return math.isnan(y)
        if math.isinf(x):
            return y == x
        # A <=1e-6 backend perturbation in mapped RGB may be amplified by hue
        # and log-ratio coordinates near the neutral threshold. The report
        # tolerance is measured separately from the stricter pixel fixtures.
        return math.isclose(x, y, rel_tol=REPORT_RTOL, abs_tol=REPORT_ATOL)
    return expected == actual


def _beta_table() -> dict:
    from dngscan.film_develop import INTERIMAGE_BETA

    return dict(sorted(INTERIMAGE_BETA.items()))


def regen(check: bool = False) -> int:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    volume, _ = pal.palette_volume()
    drift = 0
    scenes = all_scenes()
    missing_scenes = [scene_id for scene_id in RENDER_SCENES if scene_id not in scenes]
    if missing_scenes:
        print(f"missing source scenes: {', '.join(missing_scenes)}")
        return 1

    for stock in PROBE_STOCKS:
        for mode in ("observe", "full"):
            mapped = render_probe(volume, stock, mode).astype(np.float32)
            path = probe_path(stock, mode)
            if path.is_file():
                prev = np.load(path, allow_pickle=False)["mapped"]
                delta = float(np.max(np.abs(prev - mapped)))
                if delta > 1e-6:
                    drift += 1
                    print(f"{path.name}: max_abs={delta:.6g}")
            elif check:
                drift += 1
                print(f"missing {path.name}")
            if not check:
                np.savez_compressed(path, mapped=mapped)
                print(f"wrote {path.name}")

    for scene_id in RENDER_SCENES:
        linear, u8 = render_technical(scene_id)
        path = render_path(scene_id)
        if path.is_file():
            prev = np.load(path, allow_pickle=False)
            changed = int(np.count_nonzero(prev["u8"] != u8))
            max_abs = float(np.max(np.abs(prev["linear"].astype(np.float32) - linear)))
            if changed or max_abs > 1e-6:
                drift += 1
                print(f"{path.name}: u8_changed={changed} linear_max_abs={max_abs:.6g}")
        elif check:
            drift += 1
            print(f"missing {path.name}")
        if not check:
            np.savez_compressed(
                path, linear=linear.astype(np.float32), u8=u8,
                meta=np.asarray(json.dumps(
                    {
                        "scene_id": scene_id,
                        "stock": RENDER_STOCK,
                        "mode": "full",
                        "neutralization": "bounded",
                    }
                )),
            )
            print(f"wrote {path.name}")

    live_report = build_report()
    if check:
        if not BASELINE_PATH.is_file():
            drift += 1
            print(f"missing {BASELINE_PATH.name}")
        else:
            stored_report = json.loads(BASELINE_PATH.read_text("utf-8"))
            if not _reports_close(stored_report, live_report):
                drift += 1
                print(f"{BASELINE_PATH.name}: report drifted")
        if not MANIFEST_PATH.is_file():
            drift += 1
            print(f"missing {MANIFEST_PATH.name}")
        else:
            manifest = json.loads(MANIFEST_PATH.read_text("utf-8"))
            expected = {path.name for path in expected_fixture_paths()}
            pinned = manifest.get("fixture_sha256", {})
            manifest_ok = (
                manifest.get("phase") == "appearance_p0"
                and manifest.get("interimage", {}).get("beta_table") == _beta_table()
                and set(pinned) == expected
                and all(
                    path.is_file()
                    and pinned[path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in expected_fixture_paths()
                )
            )
            if not manifest_ok:
                drift += 1
                print(f"{MANIFEST_PATH.name}: manifest drifted")
    else:
        BASELINE_PATH.write_text(
            json.dumps(live_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {BASELINE_PATH.name}")
        fixtures = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in expected_fixture_paths()
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
                    "interimage": {
                        "mode": "declared",
                        "form": "rail-preserving bounded, refit 2026-08-11",
                        "beta_table": _beta_table(),
                    },
                    "technical_definition": {
                        "tone_core": "agx",
                        "film_mode": "full",
                        "neutralization": "bounded",
                        "legacy_film_crossover": "off",
                    },
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
    return 1 if check and drift else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    return regen(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
