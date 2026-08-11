#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Author the P4 reference recipes (FILM_APPEARANCE_RECIPE_PLAN §10).

The first pair, built exactly under the A4 joint contract:

    recipe(stock) = ENDURA COMMON BASE + per-stock RESIDUAL

The common base carries what the paper interpretation shares (a restrained
richness lift and deep-colour density — the print reading of the Endura
family); the residuals carry ONLY differential hue paths and differential
colour density. No differential chroma gain: the first round may not use the
richness axis for within-family separation — that is the purity axis the
inter-image beta already owns, and duplicating it would make the identity
increment unattributable.

Field values are EDITORIAL DECLARATIONS from stock reputation, expressed on
the §6.6 grid (5 EV x 24 hue knots) with a midtone-centred EV profile that
tapers to ZERO at ±6 EV — the shoulder's path to white and the deep toe are
not the palette's to bend (P0: the +6 EV high-chroma tail is where the two
chains already disagree by 28+ dE00; a recipe must not pile onto it).

Owner look review decides these numbers; this file is the single source so a
review verdict is a value edit here plus a rebuild.

    python tools/build_film_appearance_recipes.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from dngscan.film_appearance import (
    APPEARANCE_DIR,
    APPEARANCE_SCHEMA,
    EV_KNOTS,
    HUE_KNOT_COUNT,
    MANIFEST_PATH,
    medium_family,
)

H = HUE_KNOT_COUNT
K = len(EV_KNOTS)
HUES = np.arange(H) * (360.0 / H)

# Midtone-centred EV envelope over knots (-6, -3, 0, +3, +6): zero at both
# extremes by declaration.
EV_PROFILE = np.array([0.0, 0.75, 1.0, 0.7, 0.0])


def band(center: float, width: float, amount: float) -> np.ndarray:
    """A smooth periodic hue bump: raised-cosine of `width` degrees FWHM
    centred at `center`, scaled by `amount`. Sums of these are C1 on the
    periodic hue axis by construction."""
    d = np.abs((HUES - center + 180.0) % 360.0 - 180.0)
    x = np.clip(d / max(width, 1e-6), 0.0, 1.0)
    return amount * 0.5 * (1.0 + np.cos(np.pi * x))


def sheet(*bands: np.ndarray) -> np.ndarray:
    """Hue profile -> [K, H] field through the EV envelope."""
    hue_profile = np.sum(bands, axis=0) if bands else np.zeros(H)
    return (EV_PROFILE[:, None] * hue_profile[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Endura common base — the shared print interpretation. Restraint is the
# point: it must read as "a print", not as a look.
# ---------------------------------------------------------------------------
COMMON = {
    # gentle warmth through the skin-to-yellow arc
    "hue_delta_deg": sheet(band(40.0, 70.0, 1.5)),
    # a restrained COMMON richness lift (allowed: common, not differential)
    "log_chroma_gain": sheet(band(0.0, 360.0, 0.08)),
    # deep-colour density: the print's blacks hold colour instead of fading
    "density_ev": sheet(band(0.0, 360.0, 0.06)),
}

# ---------------------------------------------------------------------------
# Residuals: differential hue paths + differential colour density ONLY.
# Amplitudes respect the §15.2 per-patch hue cap (12 deg) with margin.
# ---------------------------------------------------------------------------
RESIDUALS = {
    "portra400": {
        # the soft portrait negative: warm open skin, olive-yellow greens,
        # quiet blues, magenta pulled gently toward red
        "hue_delta_deg": (
            sheet(band(45.0, 60.0, 3.0))       # skin arc: warm
            + sheet(band(140.0, 80.0, -8.0))   # greens: toward yellow-olive
            + sheet(band(235.0, 60.0, -2.0))   # blues: a breath toward cyan
            + sheet(band(300.0, 60.0, 4.0))    # magenta: toward red
        ),
        "density_ev": (
            sheet(band(45.0, 55.0, -0.08))     # skin: airy, slightly open
            + sheet(band(140.0, 80.0, 0.10))   # greens: a little weight
            + sheet(band(235.0, 60.0, 0.05))   # blues: quiet weight
            + sheet(band(320.0, 55.0, -0.05))  # magenta: kept light
        ),
    },
    "ektar100": {
        # the vivid landscape negative: crimson reds, emerald greens, deep
        # blues, and its honestly-reputed ruddier skin
        "hue_delta_deg": (
            sheet(band(25.0, 50.0, -4.0))      # reds: toward crimson
            + sheet(band(45.0, 45.0, -3.0))    # skin arc rides toward red too
            + sheet(band(140.0, 80.0, 7.0))    # greens: toward emerald
            + sheet(band(235.0, 60.0, 6.0))    # blues: deeper
            + sheet(band(300.0, 60.0, -2.0))   # magenta: toward blue
        ),
        "density_ev": (
            sheet(band(25.0, 50.0, 0.24))      # reds: dense (the signature)
            + sheet(band(75.0, 40.0, 0.12))    # yellows
            + sheet(band(140.0, 80.0, 0.16))   # greens
            + sheet(band(235.0, 60.0, 0.22))   # blues
            + sheet(band(320.0, 55.0, 0.14))   # magenta: dense like its reds
        ),
    },
}

RECIPES = (
    ("portra400", "kodak_portra_endura__translated"),
    ("ektar100", "kodak_portra_endura__translated"),
)


def build(stock_id: str, medium_id: str) -> Path:
    rid = f"{stock_id}__{medium_family(medium_id)}_reference_v1"
    res = RESIDUALS[stock_id]
    fields = {
        "hue_delta_deg": COMMON["hue_delta_deg"] + res.get(
            "hue_delta_deg", np.zeros((K, H), np.float32)
        ),
        "log_chroma_gain": COMMON["log_chroma_gain"],   # NO differential
        "density_ev": COMMON["density_ev"] + res.get(
            "density_ev", np.zeros((K, H), np.float32)
        ),
        "neutral_bias_ab": np.zeros((K, 2), np.float32),
    }
    cap = float(np.abs(fields["hue_delta_deg"]).max())
    assert cap <= 12.0, f"{rid}: hue cap {cap:.1f} > 12 deg (§15.2)"
    meta = {
        "schema": APPEARANCE_SCHEMA,
        "recipe_id": rid,
        "stock_id": stock_id,
        "medium_id": medium_id,
        "process_space": "display-linear-rec2020/oklab+scene-ev",
        "provenance": "editorial-authored",
        "neutralization_policy": "print-balanced",
        "chroma_knee": 0.18,
        "chroma_power": 2.0,
        "neutral_chroma_c0": 0.03,
        "note": (
            "P4 v1 draft: Endura common base + stock residual, authored "
            "jointly with its pair. Differential axes are hue path and "
            "colour density only; owner look review pending."
        ),
    }
    path = APPEARANCE_DIR / f"{rid}.npz"
    np.savez_compressed(
        path,
        meta=np.asarray(json.dumps(meta, sort_keys=True)),
        ev_knots=np.asarray(EV_KNOTS, dtype=np.float32),
        hue_knots_deg=HUES.astype(np.float32),
        **{k: np.asarray(v, np.float32) for k, v in fields.items()},
    )
    return path


def main() -> int:
    APPEARANCE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [build(stock, medium) for stock, medium in RECIPES]
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(APPEARANCE_DIR.glob("*.npz"))
    }
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "schema": APPEARANCE_SCHEMA,
                "count": len(files),
                "files": files,
                "policy": (
                    "Every appearance recipe is hash-pinned; the loader "
                    "refuses an asset whose bytes drift from this manifest."
                ),
            },
            indent=1, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    for p in paths:
        print(f"wrote {p.name}")
    print(f"wrote {MANIFEST_PATH.name} ({len(files)} assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
