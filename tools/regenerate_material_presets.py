#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate every material-mode scene-transform preset from its recorded sources.

Each material preset in dngscan/scene_transform_presets.json records the target
observer SSF it was fitted toward. This driver re-runs tools/calibrate_skin_matrix.py
--preset-mode material once per preset with that target, the measured A7 III camera
SSF and the measured AMPAS reflectance set (which supplies the foliage/cyan/magenta
class spectra and their windows), so a change to the calibrator reaches every
material preset the same way. The skin-mode preset (arri_skin_d55) is not
touched; its photo-fitted skin window is what the material presets reuse.

    python tools/regenerate_material_presets.py            # rewrite the shipped file
    python tools/regenerate_material_presets.py --out /tmp/presets.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESETS = PROJECT_ROOT / "dngscan" / "scene_transform_presets.json"
SPECTRAL = PROJECT_ROOT / "dngscan_assets" / "spectral"
REAL_SET = SPECTRAL / "rawtoaces_training_reflectance.csv"
CAMERA_SSF = SPECTRAL / "sony_a7m3_ssf_weta_measured.csv"
UNIT_TRANSMISSION = SPECTRAL / "unit_transmission.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PRESETS)
    ap.add_argument("--report-dir", type=Path, default=SPECTRAL)
    ap.add_argument("--only", nargs="*", default=None, help="preset keys to regenerate")
    args = ap.parse_args()
    if args.out != PRESETS:
        shutil.copyfile(PRESETS, args.out)
    data = json.loads(PRESETS.read_text(encoding="utf-8"))
    done = []
    for key, preset in data["transforms"].items():
        note = str(preset.get("note", ""))
        if not note.startswith("Material-aware"):
            continue
        if args.only and key not in args.only:
            continue
        src = preset["sources"]
        target = re.search(r"Sigma->(.*?) separation", note)
        gain = re.search(r"look_gain=([0-9.eE+-]+)", note)
        stem = key[: -len("_d55")] if key.endswith("_d55") else key
        cmd = [
            sys.executable, str(PROJECT_ROOT / "tools" / "calibrate_skin_matrix.py"),
            "--preset-mode", "material",
            "--alexa-ssf-csv", str(PROJECT_ROOT / src["alev3_ssf"]),
            # The camera side is the MEASURED full-camera SSF the spectral README
            # recommends (Sony A7 III, same IMX410 colour sensor as the fp; Weta /
            # AMPAS rawtoaces-data) with unit transmission — it already includes a
            # filter stack, so the hot-mirror model must not be multiplied on top.
            "--imx410-qe-csv", str(CAMERA_SSF),
            "--ir-transmission-csv", str(UNIT_TRANSMISSION),
            "--profile-csv", str(REAL_SET),
            "--material-key", key,
            "--material-label", str(preset.get("label", key)),
            "--target-name", target.group(1) if target else key,
            "--material-look-gain", gain.group(1).rstrip(".") if gain else "1",
            "--report-json", str(args.report_dir / f"film_calibration_{stem}.json"),
            "--out", str(args.out),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        done.append(key)
    print(f"regenerated {len(done)} material presets -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
