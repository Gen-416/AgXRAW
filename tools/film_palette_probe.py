#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure where the film chain's colour identity actually lives
(FILM_APPEARANCE_RECIPE_PLAN §14.1, phase P0).

The appearance plan starts from a complaint — "full is mathematically right
but looks weak next to a commercial plugin" — and its first job is to turn
that into coordinates: which hues, at which exposures, at which purity. This
tool answers that by pushing the §14.1 probe volume through the real render
path in three configurations and reporting the differences decomposed.

    observe(stock)      AgX formation + the stock's prefeed / primaries pairing
    technical(stock)    the full v2 factorized chain
    technical(other)    the same chain on a different stock

The third comparison is the one the plan's acceptance gate actually turns on
(§15.2: two stocks must differ by dE00 >= 2 in their own target regions). If
`technical` cannot separate Portra from Velvia today, no recipe strength will
fix that by itself — and that is a measurement, not an impression.

    python tools/film_palette_probe.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DNGSCAN_FAST", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from dngscan import film_palette_diag as pal
from dngscan import scene_transform as scene_transform_engine
from dngscan.film_curve import FILM_CURVE_PRESETS, film_style_pairing
from dngscan.render import apply_tone_core
from dngscan.scene_transform import SCENE_TRANSFORMS
from dngscan.tone import build_render_plan

# §10's first targets plus one high-chroma negative for contrast. Ektar is not
# a recipe target; it is here because a probe that only sees the stocks a
# recipe was authored against cannot tell "the chain separates stocks" from
# "the recipe separates stocks".
PROBE_STOCKS = ("portra400", "velvia100", "vision3250d", "ektar100")

# Regions each recipe makes a claim about (§10), as (name, hue window in
# degrees, minimum chroma fraction). Windows are wide because the point is to
# ask whether ANYTHING happens there, not to grade a fit.
TARGET_REGIONS: dict[str, tuple[float, float, float]] = {
    "skin_warm": (20.0, 70.0, 0.2),
    "foliage_green": (110.0, 170.0, 0.2),
    "sky_cyan": (200.0, 260.0, 0.2),
    "magenta": (300.0, 360.0, 0.2),
}


def resolve_observe(stock: str) -> tuple[str, float, str]:
    """The three declarations `--film <stock>` expands to in observe mode.

    Duplicated from the CLI's combo resolution rather than imported because
    the CLI does it inside argument parsing. Kept in one function so the drift
    is visible if that resolution ever changes.
    """
    combo = FILM_CURVE_PRESETS.get(stock, {}).get("combo", {})
    transform = str(combo.get("scene_transform", "none"))
    if transform not in SCENE_TRANSFORMS:
        transform = "none"
    strength, primaries = film_style_pairing(stock)
    return transform, float(strength), str(primaries)


def reference_plan(
    stock: str, film_mode: str, film_interimage: str = "declared",
    film_appearance: str = "technical",
):
    """Compile a render plan for one (stock, mode).

    The plan is compiled from a FIXED reference scene, not from the probe
    volume. The probe deliberately contains +6 EV samples and a full hue
    wheel; letting the auto compiler see that would make the observe and full
    plans disagree about exposure and tone endpoints, and every colour
    difference measured afterwards would be contaminated by a tone difference
    nobody asked about.
    """
    from tests.golden_support import all_scenes

    scene = all_scenes()["daylight_wide_dr"]
    transform, strength, primaries = resolve_observe(stock)
    if film_mode == "full":
        # Full mode's input-domain contract: the observer inverse is fitted on
        # plain Rec.2020, so the prefeed must be off everywhere.
        transform, strength, primaries = "none", 1.0, "base"
    plan = build_render_plan(
        scene.bundle,
        scene.analysis,
        "agx",
        "srgb",
        scene_transform=transform,
        scene_transform_strength=strength,
        agx_primaries=primaries,
        film_curve=stock,
        film_mode=film_mode,
        # `technical` means the current user-visible full default.  That is
        # bounded neutralization (`off` in the legacy internal enum), not the
        # opt-in datasheet/native branch.
        film_crossover="off",
        film_interimage=film_interimage,
        film_appearance=film_appearance,
    )
    return plan, scene.bundle, transform, strength


def render_probe(
    volume: np.ndarray, stock: str, film_mode: str,
    film_interimage: str = "declared",
    film_appearance: str = "technical",
) -> np.ndarray:
    """Probe volume -> mapped Rec.2020, the common reference space.

    This reproduces the render loop's own three lines (prefeed, then tone
    core) and stops BEFORE the output gamut fit, which is where the plan
    requires appearance to be compared: sRGB and P3 must not be able to
    disagree about a recipe. `scene_intent_rec2020` is skipped deliberately —
    it is a pure scale from storage integers, and the probe already carries
    the post-intent values it wants to test.
    """
    plan, bundle, transform, strength = reference_plan(
        stock, film_mode, film_interimage, film_appearance
    )
    # Own the probe buffer. The current transforms are pure, but a diagnostic
    # must not let a future in-place optimization make later stock/mode passes
    # depend on iteration order.
    rec = np.array(volume, dtype=np.float32, copy=True).reshape(-1, 3)
    if film_mode != "full":
        wb_adapt = scene_transform_engine.window_transport(bundle)
        rec = scene_transform_engine.apply_scene_transform_rec2020(
            rec, transform, strength, wb_adapt
        )
    return np.asarray(
        apply_tone_core(rec, plan.tone, plan.color), dtype=np.float64
    )


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _region_mask(index: pal.ProbeIndex, region: str) -> np.ndarray:
    lo, hi, min_c = TARGET_REGIONS[region]
    hue = index.hue_deg
    inside = (hue >= lo) & (hue < hi) if lo < hi else (hue >= lo) | (hue < hi)
    return (
        (index.kind == "wheel")
        & np.nan_to_num(inside, nan=False).astype(bool)
        & (index.chroma_frac >= min_c)
    )


def _named_mask(index: pal.ProbeIndex, name: str) -> np.ndarray:
    prefix = f"{name}_ev"
    return np.fromiter(
        (kind == "named" and label.startswith(prefix)
         for kind, label in zip(index.kind, index.label)),
        dtype=bool,
        count=len(index.label),
    )


def _by_axis(values: np.ndarray, index: pal.ProbeIndex) -> dict:
    """Break a metric down by EV, by hue octant and by chroma level.

    The whole point of P0 is locating the weakness, so a single median is not
    an answer. Neutral samples are excluded from the hue and chroma splits
    because their hue is undefined.
    """
    wheel = index.kind == "wheel"
    out: dict = {"overall": pal.summarize(values[wheel])}
    out["by_ev"] = {
        f"{ev:+g}": pal.summarize(values[wheel & (index.ev == ev)])
        for ev in pal.PROBE_EVS
    }
    out["by_chroma"] = {
        f"{c:g}": pal.summarize(values[wheel & (index.chroma_frac == c)])
        for c in pal.CHROMA_LEVELS
    }
    out["by_hue_octant"] = {}
    for k in range(8):
        lo, hi = k * 45.0, (k + 1) * 45.0
        m = wheel & (index.hue_deg >= lo) & (index.hue_deg < hi)
        out["by_hue_octant"][f"{int(lo)}-{int(hi)}"] = pal.summarize(values[m])
    return out


def compare_pair(a: np.ndarray, b: np.ndarray, index: pal.ProbeIndex) -> dict:
    d = pal.compare(a, b)
    report = {
        "delta_e00": _by_axis(d["delta_e00"], index),
        "delta_e_ok": _by_axis(d["delta_e_ok"], index),
        "abs_hue_deg": _by_axis(np.abs(d["d_hue_deg"]), index),
        "log2_colorfulness_ratio": _by_axis(
            d["log2_colorfulness_ratio"], index
        ),
        "log2_saturation_ratio": _by_axis(d["log2_saturation_ratio"], index),
        "abs_d_L": _by_axis(np.abs(d["d_L"]), index),
        "d_output_ev": _by_axis(d["d_output_ev"], index),
        "cie_valid_fraction": float(np.mean(d["cie_valid"])),
    }
    report["by_region"] = {
        name: {
            "delta_e00": pal.summarize(d["delta_e00"][_region_mask(index, name)]),
            "abs_hue_deg": pal.summarize(
                np.abs(d["d_hue_deg"])[_region_mask(index, name)]
            ),
            "log2_colorfulness_ratio": pal.summarize(
                d["log2_colorfulness_ratio"][_region_mask(index, name)]
            ),
            "log2_saturation_ratio": pal.summarize(
                d["log2_saturation_ratio"][_region_mask(index, name)]
            ),
        }
        for name in TARGET_REGIONS
    }
    report["by_named_patch"] = {
        name: {
            "delta_e00": pal.summarize(d["delta_e00"][_named_mask(index, name)]),
            "delta_e_ok": pal.summarize(d["delta_e_ok"][_named_mask(index, name)]),
            "abs_hue_deg": pal.summarize(
                np.abs(d["d_hue_deg"])[_named_mask(index, name)]
            ),
            "log2_saturation_ratio": pal.summarize(
                d["log2_saturation_ratio"][_named_mask(index, name)]
            ),
        }
        for name in pal.NAMED_PATCHES
    }
    neutral = index.kind == "neutral"
    report["neutral"] = {
        "delta_e00": pal.summarize(d["delta_e00"][neutral]),
        "max_oklab_chroma_a": float(pal.decompose(a)["C"][neutral].max()),
        "max_oklab_chroma_b": float(pal.decompose(b)["C"][neutral].max()),
    }
    # Risk 2 (§17): a recipe whose gain is the same everywhere is a saturation
    # slider wearing a costume. Report the SPREAD of the exposure-invariant
    # saturation gain across
    # hue, not just its level — a selective palette has a wide one.
    wheel = index.kind == "wheel"
    ratios = d["log2_saturation_ratio"][wheel]
    ratios = ratios[np.isfinite(ratios)]
    report["selectivity"] = {
        "saturation_gain_iqr": float(
            np.percentile(ratios, 75) - np.percentile(ratios, 25)
        ) if ratios.size else float("nan"),
        "hue_rotation_p95_deg": float(
            np.percentile(np.abs(d["d_hue_deg"][wheel][
                np.isfinite(d["d_hue_deg"][wheel])
            ]), 95)
        ),
    }
    return report


def build_report(stocks: tuple[str, ...] = PROBE_STOCKS) -> dict:
    volume, index = pal.palette_volume()
    rendered: dict[str, dict[str, np.ndarray]] = {}
    for stock in stocks:
        rendered[stock] = {
            mode: render_probe(volume, stock, mode)
            for mode in ("observe", "full")
        }

    report: dict = {
        "probe": {
            "samples": int(volume.shape[0]),
            "hue_count": pal.HUE_COUNT,
            "chroma_levels": list(pal.CHROMA_LEVELS),
            "evs": list(pal.PROBE_EVS),
            "named_patches": sorted(pal.NAMED_PATCHES),
        },
        "technical_definition": {
            "tone_core": "agx",
            "film_mode": "full",
            "neutralization": "bounded",
            "legacy_film_crossover": "off",
            "plan_output_gamut": "srgb",
            "measurement_space": "pre-gamut-fit-linear-rec2020",
        },
        "stocks": list(stocks),
        "observe_pairing": {
            s: dict(zip(("scene_transform", "strength", "agx_primaries"),
                        resolve_observe(s)))
            for s in stocks
        },
        "observe_vs_technical": {},
        "gamut_pressure": {},
        "stock_identity": {},
    }
    for stock in stocks:
        report["observe_vs_technical"][stock] = compare_pair(
            rendered[stock]["observe"], rendered[stock]["full"], index
        )
        report["gamut_pressure"][stock] = {
            gamut: {
                mode: pal.gamut_pressure(rendered[stock][mode], gamut)
                for mode in ("observe", "full")
            }
            for gamut in ("srgb", "p3")
        }

    # Stock identity: how far apart two stocks land in the SAME mode. This is
    # the number §15.2 gates on, and it is measured for both modes so the
    # eventual recipe can be shown to add separation the chain does not
    # already provide.
    for i, a in enumerate(stocks):
        for b in stocks[i + 1:]:
            key = f"{a}__vs__{b}"
            report["stock_identity"][key] = {
                mode: compare_pair(rendered[a][mode], rendered[b][mode], index)
                for mode in ("observe", "full")
            }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--stocks", nargs="*", default=list(PROBE_STOCKS))
    args = ap.parse_args()
    report = build_report(tuple(args.stocks))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
