#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export film spectral sensitivities from spektrafilm profiles as SSF CSVs.

The prefeed calibrator treats its fitting target as "another camera": a wavelength,R,G,B
sensitivity CSV. A film stock is exactly that observer — its layer sensitivities are the
spectral response that separated colours onto the negative. This tool converts the
log10 sensitivities stored in the CC BY-SA spektrafilm profiles into linear, per-channel
peak-normalized SSF CSVs on the calibrator's 400-700nm/10nm grid, so
tools/calibrate_skin_matrix.py --preset-mode material can fit "fp -> film" the same way
it fits "fp -> ALEV", with zero calibrator changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = PROJECT_ROOT / "dngscan_assets" / "spectral" / "spektrafilm"
OUT_DIR = PROJECT_ROOT / "dngscan_assets" / "spectral"
GRID = np.arange(400.0, 701.0, 10.0)

def _discover() -> dict[str, str]:
    import json as _json

    films: dict[str, str] = {}
    for path in sorted(PROFILE_DIR.glob("*.json")):
        info = _json.loads(path.read_text(encoding="utf-8")).get("info", {})
        if str(info.get("stage")) != "filming" or "push" in path.stem:
            continue
        key = path.stem
        if key == "fujifilm_xtra_400":
            films["superia400"] = key
            continue
        for prefix in ("kodak_", "fujifilm_"):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        films[key.replace("_", "")] = path.stem
    return films


FILMS = _discover()


def export(key: str, profile_name: str) -> Path:
    profile = json.load(open(PROFILE_DIR / f"{profile_name}.json"))
    wl = np.asarray(profile["data"]["wavelengths"], dtype=np.float64)
    log_sens = np.asarray(profile["data"]["log_sensitivity"], dtype=np.float64)
    out = OUT_DIR / f"film_ssf_{key}.csv"
    columns = []
    for c in range(3):
        col = log_sens[:, c]
        keep = np.isfinite(col)
        linear = np.power(10.0, np.interp(GRID, wl[keep], col[keep]))
        columns.append(linear / np.max(linear))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# Film SSF exported from spektrafilm profile "
                 f"{profile_name}.json (CC BY-SA 4.0, see spektrafilm/README.md)\n")
        fh.write("wavelength_nm,R,G,B\n")
        for i, w in enumerate(GRID):
            fh.write(f"{w:.0f},{columns[0][i]:.6f},{columns[1][i]:.6f},{columns[2][i]:.6f}\n")
    print(f"wrote {out}")
    return out


def main() -> int:
    for key, profile_name in FILMS.items():
        export(key, profile_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
