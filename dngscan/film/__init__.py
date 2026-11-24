# SPDX-License-Identifier: GPL-3.0-or-later
"""Observe film simulation: a film stock as a fixed, data-driven AgX rendering.

One versioned asset (``data/film/observe_v1.json``, exported by AgXFilm's
``tools/export_agxraw_film_asset.py``) carries everything per stock:

* the AgX C1 curve refitted to the stock's print/slide neutral scale under
  native viewing with luminance black-point compensation (EV0 -> 0.18);
* reversal stocks' ISO 2240 metered mid grey, as a scene exposure offset;
* per-hue Oklab rendition fitted to the full spectral chain;
* the grey scale's crossover, print-balanced;
* negative-density grain (5207-derived sigma(D), per-stock datasheet scale).

A film choice is a transformation of a compiled tone plan
(:func:`apply_film_plan`) plus a tone-core wrapper (:func:`film_tone_core`):
metered offset -> grain -> AgX -> hue rendition -> crossover. The auto-EV probe
path goes through the same wrapper without spatial context, so it sees the
same curve and colour, and no grain (grain is mean-preserving by construction).

Research record: AgXFilm docs/FILM_BPC_PRINT_VIEW.zh-CN.md.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

ASSET_PATH = Path(__file__).resolve().parents[1] / "data" / "film" / "observe_v1.json"
ASSET_SCHEMA = "agxraw-film-observe/1"
FILM_NONE = "none"
DEFAULT_FILM_GRAIN = 1.0
FILM_GRAIN_MAX = 3.0


@lru_cache(maxsize=1)
def _asset() -> dict:
    raw = ASSET_PATH.read_bytes()
    data = json.loads(raw)
    if data.get("schema") != ASSET_SCHEMA:
        raise RuntimeError(f"胶片资产 schema 不符：{data.get('schema')!r}（需要 {ASSET_SCHEMA}）")
    data["_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def film_choices() -> tuple[str, ...]:
    return (FILM_NONE,) + tuple(_asset()["stocks"])


def film_label(stock: str) -> str:
    return str(_asset()["stocks"][stock]["label"])


def film_asset_sha256() -> str:
    return str(_asset()["_sha256"])


def validate_film(stock: str) -> str:
    stock = str(stock or FILM_NONE)
    if stock == FILM_NONE or stock in _asset()["stocks"]:
        return stock
    raise ValueError(f"未知胶片：{stock}（可选：{'/'.join(film_choices())}）")


def validate_film_grain(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("胶片颗粒强度必须是数字") from exc
    if not np.isfinite(v) or not 0.0 <= v <= FILM_GRAIN_MAX:
        raise ValueError(f"胶片颗粒强度需在 0-{FILM_GRAIN_MAX:g} 之间")
    return v


def film_seed_for(identity: str) -> int:
    """Grain phase seed from a capture identity: the same RAW always gets the
    same grain arrangement (preview == export, re-export == export)."""
    digest = hashlib.sha256(str(identity).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") | 1


def film_seed_for_file(path: Any) -> int:
    """Seed from the RAW's bytes (size + first/last MiB): path-independent,
    cheap, and stable across preview, export and re-export of one capture."""
    p = Path(path)
    size = p.stat().st_size
    with p.open("rb") as fh:
        head = fh.read(1 << 20)
        fh.seek(max(size - (1 << 20), 0))
        tail = fh.read(1 << 20)
    return film_seed_for(f"{size}:" + hashlib.sha256(head + tail).hexdigest())


GRAIN_CHROMA_CHOICES = ("film", "scan")
DEFAULT_GRAIN_CHROMA = "film"


def validate_grain_chroma(value: Any) -> str:
    v = str(value or DEFAULT_GRAIN_CHROMA)
    if v not in GRAIN_CHROMA_CHOICES:
        raise ValueError(f"胶片颗粒颜色：未知 {v!r}（可选 {'/'.join(GRAIN_CHROMA_CHOICES)}）")
    return v


@dataclass(frozen=True)
class FilmSelection:
    stock: str = FILM_NONE
    grain: float = DEFAULT_FILM_GRAIN
    seed: int = 1
    # "film": the asset's modelled cross-layer correlation (colour grain as the
    # three emulsion layers form it); "scan": a declared mostly-luminance
    # convention, like film after scanning/post. Neither is a measurement.
    grain_chroma: str = DEFAULT_GRAIN_CHROMA

    @property
    def active(self) -> bool:
        return self.stock != FILM_NONE


def apply_film_plan(tone_plan: Any, film: FilmSelection | None) -> Any:
    """Replace the compiled curve by the stock's fixed coordinate.

    Scene-adaptive editorial compensations (auto punch, dark-scene view
    brightness) are neutralized — the same preset must render the same way on
    every photo — and the AgX primaries are pinned to base (the hue rendition
    replaces any primaries look). Scene metrics stay untouched.
    """
    if film is None or not film.active:
        return tone_plan
    stock = _asset()["stocks"][validate_film(film.stock)]
    if str(getattr(tone_plan, "tone_core", "agx")) != "agx":
        raise ValueError("胶片模拟只在 AgX tone 核上定义（--tone-core agx）")
    p = stock["curve"]
    return replace(
        tone_plan,
        black_ev=float(p["black_ev"]),
        white_ev=float(p["white_ev"]),
        dynamic_range_ev=float(p["white_ev"]) - float(p["black_ev"]),
        contrast=float(p["contrast"]),
        toe_power=float(p["toe_power"]),
        shoulder_power=float(p["shoulder_power"]),
        latitude_lo_ev=float(p["latitude_lo_ev"]),
        latitude_hi_ev=float(p["latitude_hi_ev"]),
        toe_start_ev=-float(p["latitude_lo_ev"]),
        shoulder_start_ev=float(p["latitude_hi_ev"]),
        target_black_linear=0.0,
        pivot_ev_offset=0.0,
        punch_strength=0.0,
        view_brightness=1.0,
        agx_primaries="base",
        film_stock=str(film.stock),
        film_grain=validate_film_grain(film.grain),
        film_seed=int(film.seed),
        film_grain_chroma=validate_grain_chroma(film.grain_chroma),
    )


class _StockEngine:
    def __init__(self, key: str):
        from .color import GreyCrossover, HueRendition
        from .grain import StockGrain

        a = _asset()
        s = a["stocks"][key]
        self.offset_gain = float(2.0 ** float(s["exposure_offset_ev"]))
        self.grain_scale = float(s["grain_scale"])
        self.hue = HueRendition(a["hue"], s["hue"])
        self.crossover = GreyCrossover(s["crossover"])
        self.grain = StockGrain(s["stage_a"])


@lru_cache(maxsize=None)
def _engine(key: str) -> _StockEngine:
    return _StockEngine(key)


@lru_cache(maxsize=1)
def _grain_profile():
    from .grain import GrainProfile

    return GrainProfile.from_asset(_asset()["grain"])


@dataclass(frozen=True)
class FilmSpatial:
    """Where a flat pixel run sits in its frame: grain needs film-plane
    positions. Probe paths pass none and render grain-free."""

    start: int
    height: int
    width: int


def film_tone_core(
    rgb_rec2020: Any,
    plan: Any,
    agx_core: Callable[[Any, Any], Any],
    spatial: FilmSpatial | None = None,
) -> Any:
    """metered offset -> grain -> AgX -> hue rendition -> grey crossover."""
    eng = _engine(str(plan.film_stock))
    rgb = np.asarray(rgb_rec2020, dtype=np.float32)
    if eng.offset_gain != 1.0:
        rgb = rgb * np.float32(eng.offset_gain)
    scene = rgb
    amount = float(getattr(plan, "film_grain", 0.0)) * eng.grain_scale
    if spatial is not None and amount > 0.0:
        from .grain import apply_observe_grain

        rgb = apply_observe_grain(rgb, spatial.start, spatial.height, spatial.width,
                                  eng.grain, _grain_profile(), amount, int(plan.film_seed),
                                  str(getattr(plan, "film_grain_chroma", DEFAULT_GRAIN_CHROMA)))
    out = agx_core(rgb, plan)
    out = eng.hue.apply(out)
    return eng.crossover.apply(out, scene)
