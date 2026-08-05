# SPDX-License-Identifier: GPL-3.0-or-later
"""Film curve presets: named coordinates in AgX's parameter space.

A preset pins the complete curve (endpoints included) to values solved offline from a
stock's published characteristic curves (tools/fit_film_curve.py). Scene-adaptive tone
compilation is deliberately bypassed while a preset is active: film's response is fixed
— the same scene always receives the same curve — and that whole-roll consistency is
exactly what the user selected. The EV0 -> 0.18 anchor survives by construction (the
fit target is balanced to 18% at mid-scale and AgX's pivot is immovable), and the
paper-Dmax shadow floor rides in through target_black_linear as a declared, measured
part of the look. User tone adjustments still apply on top, reported as departures
from the named coordinate.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ._deps import np

FILM_CURVE_PRESETS_JSON = Path(__file__).with_name("film_curve_presets.json")


_SCHEMA_ERROR: str | None = None


def _load() -> dict[str, dict[str, Any]]:
    global _SCHEMA_ERROR
    try:
        raw = json.loads(FILM_CURVE_PRESETS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    version = int(raw.get("version", 0) or 0)
    if version != 2:
        # v1 presets were calibrated in linear sRGB and consumed as diagonal
        # gains in other bases — the exact coordinate error the spectral rebuild
        # removes. The refusal is LAZY: importing dngscan (including the fit
        # tool that regenerates this very file) must not die on stale data;
        # any actual film request does, loudly.
        _SCHEMA_ERROR = (
            f"film_curve_presets.json schema v{version} is not supported; "
            "regenerate with tools/fit_film_curve.py (schema v2, Rec.2020 basis)"
        )
        return {}
    presets = raw.get("presets", {})
    return presets if isinstance(presets, dict) else {}


FILM_CURVE_PRESETS: dict[str, dict[str, Any]] = _load()
FILM_CURVE_CHOICES = ("none",) + tuple(FILM_CURVE_PRESETS)


def film_curve_label(name: str) -> str:
    if name == "none":
        return "无"
    preset = FILM_CURVE_PRESETS.get(name)
    return str(preset.get("label", name)) if preset else name


def validate_film_curve(name: str) -> str:
    if name != "none" and _SCHEMA_ERROR:
        raise RuntimeError(_SCHEMA_ERROR)
    if name == "none" or name in FILM_CURVE_PRESETS:
        return name
    raise ValueError(
        f"未知胶片曲线预设：{name}（可选：{'/'.join(FILM_CURVE_CHOICES)}）"
    )


# Editorial style pairings for the observe mode — the declared "look layer" on the
# validated base (the FilmLight-shaped architecture: a stable house DRT plus a
# separate look). Each entry pairs a prefeed separation over-drive with one of
# AgX's own validated primaries geometries. These are EDITORIAL DECLARATIONS by
# stock reputation, not measurements — first-drafted 2026-07-30, deliberately
# conservative, and any explicitly given layer value overrides the pairing (the
# combo rule: nothing is baked). Keys fall back to _DEFAULT_STYLE.
FILM_STYLE_PAIRINGS: dict[str, tuple[float, str]] = {
    # Kodak negatives: gentle portrait separation; Ektar is the vivid outlier.
    "portra160": (1.3, "base"),
    "portra400": (1.3, "base"),
    "portra800": (1.3, "base"),
    "portra800push1": (1.35, "base"),
    "portra800push2": (1.4, "base"),
    "ektar100": (1.4, "punchy"),
    "gold200": (1.4, "base"),
    "ultramax400": (1.4, "base"),
    # Fuji negatives: consumer crispness; Pro 400H's airy softness.
    "superia400": (1.5, "base"),
    "c200": (1.4, "base"),
    "pro400h": (1.3, "muted"),
    # Reversals: Velvia is THE saturated slide; Kodachrome's dense restraint.
    "provia100f": (1.4, "base"),
    "velvia100": (1.6, "punchy"),
    "ektachrome100": (1.4, "base"),
    "kodachrome64": (1.4, "muted"),
    # Cine negatives: flat wide-latitude scan look; theatrical quotes get punch.
    "vision350d": (1.2, "muted"),
    "vision3250d": (1.2, "muted"),
    "vision3200t": (1.2, "muted"),
    "vision3500t": (1.2, "muted"),
    "verita200d": (1.2, "muted"),
    "vision350d_theatrical": (1.4, "punchy"),
    "vision3250d_theatrical": (1.4, "punchy"),
    "vision3200t_theatrical": (1.4, "punchy"),
    "vision3500t_theatrical": (1.4, "punchy"),
    "verita200d_theatrical": (1.4, "punchy"),
}
_DEFAULT_STYLE = (1.3, "base")


def film_style_pairing(name: str) -> tuple[float, str]:
    """(separation strength, agx primaries) declared for a film preset."""
    return FILM_STYLE_PAIRINGS.get(str(name), _DEFAULT_STYLE)


_RATIO_FIELD_CACHE: dict[str, tuple[Any, Any] | None] = {}


def channel_ratio_field(name: str) -> tuple[Any, Any] | None:
    """Measured per-channel ratio field r_c(EV) of a preset; None when absent.

    r_c(EV) = neutral_gain_rec2020 = neutral_rgb_rec2020(EV) / target_Y(EV) along
    the stock's balanced neutral ramp — the exposure-dependent neutral colour
    solved by tools/fit_film_curve.py in the same Rec.2020 basis and viewing
    translation as the tone target (schema v2). Returns
    (ev_grid, ratios[N, 3]) as read-only float32 arrays for np.interp consumption;
    the grid is ascending and covers the fit domain, and interpolation clamps at the
    ends by construction (deep white ratios approach 1, deep shadow ratios approach
    the dye-floor differential).
    """
    key = str(name)
    if key in _RATIO_FIELD_CACHE:
        return _RATIO_FIELD_CACHE[key]
    preset = FILM_CURVE_PRESETS.get(key)
    raw = preset.get("neutral_curve") if isinstance(preset, dict) else None
    field: tuple[Any, Any] | None = None
    if isinstance(raw, dict) and raw.get("ev") and raw.get("neutral_gain_rec2020"):
        ev = np.asarray(raw["ev"], dtype=np.float32)
        ratios = np.asarray(raw["neutral_gain_rec2020"], dtype=np.float32)
        if ev.ndim == 1 and ratios.shape == (ev.size, 3) and ev.size >= 2:
            # De-duplicate the stored grid (the fitter's index subsample can repeat
            # rows); np.interp requires strictly usable ascending x.
            keep = np.concatenate(([True], np.diff(ev) > 0))
            ev, ratios = ev[keep], ratios[keep]
            ev.setflags(write=False)
            ratios.setflags(write=False)
            field = (ev, ratios)
    _RATIO_FIELD_CACHE[key] = field
    return field


def film_process(name: str) -> str | None:
    """Physical process class of a preset: "negative" (print-through C-41/ECN-2),
    "reversal" (E-6/K-14 slide — its own display medium), or None when unknown.

    Read from the preset's declared classification (tools/fit_film_curve.py derives
    it from whether the stock's profile carries a target print); consumers must not
    guess from the name.
    """
    preset = FILM_CURVE_PRESETS.get(str(name))
    if not isinstance(preset, dict):
        return None
    process = preset.get("process")
    return str(process) if process in ("negative", "reversal") else None


# --- Enlarger colour head -----------------------------------------------------
# Real darkroom units: CC filter density steps. 30 CC = 0.30 optical density on
# the named paper layer's printing exposure (~one stop). Steps of 5 mirror the
# detented dial of a real colour head.
COLOR_HEAD_CC_MAX = 200.0
COLOR_HEAD_CC_STEP = 5.0


def color_head_supported(name: str) -> bool:
    """Whether a preset carries a colour-head field (negative presets only)."""
    preset = FILM_CURVE_PRESETS.get(str(name))
    return isinstance(preset, dict) and isinstance(preset.get("color_head"), dict)


def validate_color_head_cc(value: Any, label: str) -> float:
    """One colour-head dial: 0-200 CC in detents of 5, like the physical dial."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数字") from exc
    if not np.isfinite(v) or not 0.0 <= v <= COLOR_HEAD_CC_MAX:
        raise ValueError(f"{label} 需在 0-{COLOR_HEAD_CC_MAX:g} CC 之间")
    if abs(v / COLOR_HEAD_CC_STEP - round(v / COLOR_HEAD_CC_STEP)) > 1e-6:
        raise ValueError(
            f"{label} 按真实色头档位取值：{COLOR_HEAD_CC_STEP:g} CC 步进（收到 {v:g}）"
        )
    return v


_COLOR_HEAD_FIELD_CACHE: dict[str, tuple[Any, Any, dict[str, Any]] | None] = {}


def _color_head_field(name: str) -> tuple[Any, Any, dict[str, Any]] | None:
    """(ev_grid, cc_grid, {filter: gains[ncc, nev, 3]}) as float arrays, or None."""
    key = str(name)
    if key in _COLOR_HEAD_FIELD_CACHE:
        return _COLOR_HEAD_FIELD_CACHE[key]
    preset = FILM_CURVE_PRESETS.get(key)
    raw = preset.get("color_head") if isinstance(preset, dict) else None
    field: tuple[Any, Any, dict[str, Any]] | None = None
    if isinstance(raw, dict) and raw.get("ev") and raw.get("cc_grid"):
        try:
            ev = np.asarray(raw["ev"], dtype=np.float32)
            cc = np.asarray(raw["cc_grid"], dtype=np.float64)
            gains = {
                f: np.asarray(raw[f], dtype=np.float32) for f in ("y", "m")
            }
            valid = (
                ev.ndim == 1 and ev.size >= 2 and np.all(np.diff(ev) > 0)
                and all(
                    g.shape == (cc.size, ev.size, 3) for g in gains.values()
                )
            )
        except (TypeError, ValueError):
            valid = False
        if valid:
            ev.setflags(write=False)
            for g in gains.values():
                g.setflags(write=False)
            field = (ev, cc, gains)
    _COLOR_HEAD_FIELD_CACHE[key] = field
    return field


_COLOR_HEAD_GAIN_CACHE: dict[tuple[str, float, float], tuple[Any, Any] | None] = {}


def color_head_gain_curves(
    name: str, y_cc: float, m_cc: float
) -> tuple[Any, Any] | None:
    """Combined Y+M colour-head gain curves g_c(EV) for a dialled setting.

    The published field samples each filter at a small CC grid; a dialled value is
    interpolated between grid points in log-gain (filter densities compose
    multiplicatively, so log-gain is the linear-in-density domain), and the two
    filters combine multiplicatively — the fitter's declared separable
    approximation. Returns (ev_grid, gains[N, 3]) float32 read-only, or None when
    the preset has no field or both dials are at zero. np.interp end clamping IS
    the out-of-domain semantics: beyond the visible fit domain the print sits on
    its own endpoints where filtration physically stops mattering.
    """
    y_cc, m_cc = float(y_cc), float(m_cc)
    if y_cc <= 0.0 and m_cc <= 0.0:
        return None
    key = (str(name), y_cc, m_cc)
    if key in _COLOR_HEAD_GAIN_CACHE:
        return _COLOR_HEAD_GAIN_CACHE[key]
    field = _color_head_field(str(name))
    result: tuple[Any, Any] | None = None
    if field is not None:
        ev, cc_grid, gains = field
        log_total = np.zeros((ev.size, 3), dtype=np.float64)
        for filter_name, cc in (("y", y_cc), ("m", m_cc)):
            if cc <= 0.0:
                continue
            # log-gain per grid density, with the implicit identity row at 0 CC.
            log_g = np.log(np.maximum(gains[filter_name].astype(np.float64), 1e-9))
            grid = np.concatenate(([0.0], cc_grid))
            table = np.concatenate((np.zeros((1, ev.size, 3)), log_g), axis=0)
            hi = int(np.searchsorted(grid, min(cc, grid[-1]), side="left"))
            hi = max(1, min(hi, grid.size - 1))
            lo = hi - 1
            t = (min(cc, grid[-1]) - grid[lo]) / (grid[hi] - grid[lo])
            log_total += (1.0 - t) * table[lo] + t * table[hi]
        combined = np.exp(log_total).astype(np.float32)
        combined.setflags(write=False)
        result = (ev, combined)
    _COLOR_HEAD_GAIN_CACHE[key] = result
    return result


def apply_film_curve_preset(tone_plan: Any, name: str) -> Any:
    """Replace the scene-compiled curve with the preset's fixed coordinate.

    Only curve-shape fields move; everything else on the tone plan (scene metrics,
    colour policy inputs, tone_core) is untouched, so HDR budgeting still reads the
    real scene while the SDR body renders the declared film curve.
    """
    if name == "none":
        return tone_plan
    preset = FILM_CURVE_PRESETS.get(name)
    if preset is None:
        raise ValueError(f"未知胶片曲线预设：{name}")
    p = preset["params"]
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
        target_black_linear=float(p.get("target_black_linear", 0.0)),
        curve_preset=str(name),
    )
