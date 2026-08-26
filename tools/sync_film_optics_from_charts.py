#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile the chart-derived blocks of the film-optics assets from source.

The rendering runtime reads dngscan/data/film_optics/*.json — NOT the chart
digitizations in dngscan/data/{grain,mtf}/ directly. Until 2026-08-25 that
propagation was a one-off manual edit (optics R1, #104), so three rounds of
digitization precision work updated the SOURCE tables while the compiled
assets silently kept the old ones — found by the owner asking why the
rendered output had not changed (the asset claimed "digitized from
granularity_5207.json" while no longer matching it: 声明失实).

This tool is the standing compiler for exactly the chart-derived blocks:

  stock__modelled_default.json
      grain.channels.{R,G,B}.{chart_density, sigma_density}  <- granularity_5207
      emulsion_scatter.channels.{R,G,B}                      <- mtf_5207 fit
  print__modelled_default.json
      positive_grain.channels.{R,G,B}.{chart_density, sigma_density}
                                                             <- granularity_2383
      formation_scatter.channels.{R,G,B}                     <- mtf_2383 fit

Everything else in the assets (modelled halation, anti-halation, field
geometry, editorial variants) is untouched. Source strings gain a sync date
and the input files' SHA-256. tests/test_film_optics_chart_sync.py gates
compiled == source permanently.

Usage:
    python tools/sync_film_optics_from_charts.py          # rewrite
    python tools/sync_film_optics_from_charts.py --check  # verify only
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPTICS = ROOT / "dngscan" / "data" / "film_optics"
GRAIN = ROOT / "dngscan" / "data" / "grain"
MTF = ROOT / "dngscan" / "data" / "mtf"

# (asset file, grain block key, granularity file, scatter block key, mtf file,
#  scatter fields kept per channel)
MAPPING = [
    ("stock__modelled_default.json", "grain", "granularity_5207.json",
     "emulsion_scatter", "mtf_5207.json", ("s", "sigma_um", "w", "lambda_um")),
    ("print__modelled_default.json", "positive_grain", "granularity_2383.json",
     "formation_scatter", "mtf_2383.json", ("s", "sigma_um")),
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compiled_blocks(granularity: dict, mtf: dict, scatter_fields) -> tuple[dict, dict]:
    grain_channels = {}
    for ch in ("R", "G", "B"):
        node = granularity["channels"][ch]
        dens = node["density_loge"]
        grain_channels[ch] = {
            "chart_density": [dens[0][1], dens[-1][1]],
            "sigma_density": [list(map(float, row)) for row in node["sigma_density"]],
        }
    scatter_channels = {}
    for ch, fit in mtf["fit"].items():
        scatter_channels[ch] = {k: float(fit[k]) for k in scatter_fields if k in fit}
    return grain_channels, scatter_channels


def sync(check: bool) -> int:
    drift = 0
    for asset_name, gkey, gfile, skey, mfile, sfields in MAPPING:
        asset_path = OPTICS / asset_name
        asset = json.loads(asset_path.read_text())
        granularity = json.loads((GRAIN / gfile).read_text())
        mtf = json.loads((MTF / mfile).read_text())
        grain_ch, scatter_ch = compiled_blocks(granularity, mtf, sfields)
        stale = (asset[gkey].get("channels") != grain_ch
                 or asset[skey].get("channels") != scatter_ch)
        if stale:
            drift += 1
            print(f"{asset_name}: {gkey}/{skey} out of sync with {gfile}/{mfile}")
        if check or not stale:
            continue
        asset[gkey]["channels"] = grain_ch
        asset[gkey]["source"] = (
            f"sigma(D) digitized from Kodak charts "
            f"(dngscan/data/grain/{gfile}, tools/import_kodak_granularity.py); "
            f"synced by tools/sync_film_optics_from_charts.py, "
            f"input sha256 {_sha(GRAIN / gfile)[:16]}")
        asset[skey]["channels"] = scatter_ch
        asset[skey]["source"] = (
            f"scatter kernel fitted from the digitized MTF "
            f"(dngscan/data/mtf/{mfile}, tools/import_kodak_mtf.py); "
            f"synced by tools/sync_film_optics_from_charts.py, "
            f"input sha256 {_sha(MTF / mfile)[:16]}")
        asset_path.write_text(json.dumps(asset, ensure_ascii=False, indent=1,
                                         allow_nan=False) + "\n")
        print(f"wrote {asset_name}")
    if check:
        print(f"chart-sync check: {drift} stale" if drift else "chart-sync check: in sync")
        return 1 if drift else 0
    return 0


if __name__ == "__main__":
    sys.exit(sync(check="--check" in sys.argv))
