# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional project-authored chromatic look layer applied after AgX, in Oklab."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields as dataclass_fields
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

# User-extendable local look registry. The file is ignored by Git so users can keep
# private experiments without accidentally redistributing third-party colour assets.
LOOK_FIELDS_JSON = Path(__file__).resolve().parents[1] / "dngscan_assets" / "look_fields.json"
_LOOK_POOL = ThreadPoolExecutor(
    max_workers=min(8, max(2, (os.cpu_count() or 4) - 2)),
    thread_name_prefix="dngscan-look",
)
_LOOK_PARALLEL_MIN_PIXELS = 64 * 1024


@dataclass(frozen=True)
class LookField:
    """Measured chromatic field relative to the default dngscan AgX rendering."""

    hue_rotation_deg: tuple[float, ...]  # 12 sectors, Oklab hue
    chroma_ratio: tuple[float, ...]  # per-sector C ratio at mid-L
    mid_chroma_ratio: float
    shadow_chroma_ratio: float
    highlight_chroma_ratio: float
    shadow_cool_a: float = 0.0
    shadow_cool_b: float = 0.0
    shadow_l_lo: float = 0.10
    shadow_l_hi: float = 0.35
    highlight_l_lo: float = 0.75
    highlight_l_hi: float = 0.92
    sat_knee_c: float = 0.18
    sat_knee_relief: float = 1.0  # >1 = less chroma trim above knee (soft rolloff)
    skin_hue_lo: float = 20.0
    skin_hue_hi: float = 60.0
    skin_hue_center: float = 40.0
    skin_hue_pull: float = 0.0  # fraction of (center-hue) arc to close
    skin_chroma_scale: float = 1.0  # extra chroma trim in skin band vs mid
    skin_warm_a: float = 0.0
    skin_warm_b: float = 0.0
    neutral_cool_a: float = 0.0
    neutral_cool_b: float = 0.0
    neutral_cool_l_lo: float = 0.16
    neutral_cool_l_hi: float = 0.68
    neutral_cool_l_falloff: float = 0.92
    neutral_cool_chroma_hi: float = 0.075
    highlight_warm_a: float = 0.0
    highlight_warm_b: float = 0.0
    highlight_warm_l_lo: float = 0.58
    highlight_warm_l_hi: float = 0.92
    highlight_warm_chroma_hi: float = 0.12
    magenta_hue_lo: float = 300.0
    magenta_hue_hi: float = 18.0
    magenta_hue_center: float = 8.0
    magenta_hue_pull: float = 0.0
    magenta_chroma_scale: float = 1.0
    # Optional AgX-core overrides carried by the look (applied to the tone plan before
    # the curve runs, unlike the Oklab field above which is post-AgX):
    #   agx_hue_restore — fraction of pre-curve hue restored (None = plan default);
    #   agx_target_black — linear output floor, >0 lifts blacks for faded film looks;
    #   agx_target_white — linear output ceiling, <1 fades whites (milky/print top).
    agx_hue_restore: float | None = None
    agx_target_black: float | None = None
    agx_target_white: float | None = None


LOOK_FIELDS: dict[str, LookField] = {
    "optic_warm_cyan": LookField(
        hue_rotation_deg=(-1.0, -2.5, 1.5, 4.8, 5.2, 2.4, 0.8, -0.6, -1.4, -1.0, 0.4, 0.8),
        chroma_ratio=(1.04, 1.07, 1.00, 0.96, 0.94, 0.98, 1.02, 1.04, 0.98, 0.88, 0.86, 0.94),
        mid_chroma_ratio=1.0,
        shadow_chroma_ratio=0.92,
        highlight_chroma_ratio=1.02,
        shadow_cool_a=-0.0012,
        shadow_cool_b=-0.0020,
        shadow_l_lo=0.10,
        shadow_l_hi=0.22,
        highlight_l_lo=0.74,
        highlight_l_hi=0.92,
        sat_knee_c=0.22,
        sat_knee_relief=1.04,
        skin_hue_lo=20.0,
        skin_hue_hi=64.0,
        skin_hue_center=44.0,
        skin_hue_pull=0.11,
        skin_chroma_scale=1.045,
        skin_warm_a=0.0025,
        skin_warm_b=0.0065,
        neutral_cool_a=-0.0045,
        neutral_cool_b=-0.0060,
        neutral_cool_l_lo=0.16,
        neutral_cool_l_hi=0.66,
        neutral_cool_l_falloff=0.92,
        neutral_cool_chroma_hi=0.078,
        highlight_warm_a=0.0010,
        highlight_warm_b=0.0038,
        highlight_warm_l_lo=0.58,
        highlight_warm_l_hi=0.92,
        highlight_warm_chroma_hi=0.13,
        magenta_hue_lo=292.0,
        magenta_hue_hi=18.0,
        magenta_hue_center=8.0,
        magenta_hue_pull=0.10,
        magenta_chroma_scale=0.78,
    ),
}


def _load_json_fields() -> None:
    """Merge user-measured looks from dngscan_assets/look_fields.json into the registry.

    JSON entries win over same-named built-ins.
    Names reserved for display filters are skipped. Bad entries are skipped, never fatal."""
    try:
        raw = json.loads(LOOK_FIELDS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(raw, dict):
        return
    from .display_filter import DISPLAY_FILTERS

    allowed = {f.name for f in dataclass_fields(LookField)}
    for name, params in raw.items():
        if not isinstance(name, str) or not isinstance(params, dict) or name == "none":
            continue
        if name in DISPLAY_FILTERS:
            continue
        try:
            kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items() if k in allowed}
            LOOK_FIELDS[name] = LookField(**kwargs)
        except (TypeError, ValueError):
            continue


_load_json_fields()

LOOK_CHOICES = ("none",) + tuple(LOOK_FIELDS)

# Per-look AgX-core defaults when LookField leaves agx_* unset (JSON-measured fields
# win when explicitly set). Lower hue_restore preserves more per-channel skew
# (sunset/orange);
# target_black lifts the curve floor for faded film sims.
AGX_LOOK_DEFAULTS: dict[str, dict[str, float]] = {
    "optic_warm_cyan": {"agx_hue_restore": 0.52},
}


def _look_agx_scalar(look: str, key: str) -> float | None:
    field = LOOK_FIELDS.get(look)
    if field is not None:
        val = getattr(field, key, None)
        if val is not None:
            return float(val)
    defaults = AGX_LOOK_DEFAULTS.get(look, {})
    if key in defaults:
        return float(defaults[key])
    return None


def agx_plan_overrides(
    look: str, strength: float = 1.0, base_hue_restore: float = 0.6
) -> dict[str, float]:
    """AgX-core overrides carried by a look (hue restore, faded target black).

    Returned keys match ToneCompressionPlan field names so callers can apply them with
    dataclasses.replace. Strength scales hue restore from the caller's compiled primary
    preset, so gradeStrength < 1 eases back toward that base AgX."""
    if look == "none":
        return {}
    s = max(0.0, min(1.5, float(strength)))
    out: dict[str, float] = {}
    hue = _look_agx_scalar(look, "agx_hue_restore")
    if hue is not None:
        base = float(min(1.0, max(0.0, base_hue_restore)))
        target = float(min(1.0, max(0.0, hue)))
        out["hue_restore"] = base + s * (target - base)
    black = _look_agx_scalar(look, "agx_target_black")
    if black is not None:
        out["target_black_linear"] = s * float(min(0.15, max(0.0, black)))
    white = _look_agx_scalar(look, "agx_target_white")
    if white is not None:
        target = float(min(1.0, max(0.2, white)))
        out["target_white_linear"] = 1.0 - s * (1.0 - target)
    return out


def _smoothstep(edge0: float, edge1: float, x: Any) -> Any:
    denom = edge1 - edge0
    if abs(denom) < 1e-9:
        return np.zeros_like(np.asarray(x, dtype=np.float32))
    t = np.clip((x - np.float32(edge0)) / np.float32(denom), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@lru_cache(maxsize=64)
def _periodic_table(table: tuple[float, ...]) -> Any:
    values = np.asarray(table + (table[0],), dtype=np.float32)
    values.setflags(write=False)
    return values


def _periodic_interp(table: tuple[float, ...], hue_deg: Any) -> Any:
    """Circular linear interpolation of a 12-sector table over hue."""
    n = len(table)
    vals = _periodic_table(table)
    pos = (hue_deg - 15.0) % 360.0 / 30.0
    pos = np.clip(pos, 0.0, float(n) - 1e-5)
    idx_f = np.floor(pos)
    # A NaN hue (non-finite Oklab input) must not reach the integer cast: the
    # float->int conversion of NaN is platform-defined — 0 on arm64 macOS but
    # INT32_MIN on x86_64 Linux, an out-of-bounds index. Route NaN to slot 0;
    # frac stays NaN, so the interpolated result is NaN on every platform —
    # bit-identical to the finite path and to the historical macOS behaviour.
    idx_f = np.where(np.isfinite(idx_f), idx_f, 0.0)
    idx = idx_f.astype(np.int32)
    frac = (pos - idx_f).astype(np.float32)
    return vals[idx] * (1.0 - frac) + vals[idx + 1] * frac


def _hue_in_arc(hue_deg: Any, lo: float, hi: float) -> Any:
    """Weight 1 in the hue arc interior, soft falloff to 0 within 6° of lo/hi."""
    h = hue_deg % 360.0
    if lo <= hi:
        inside = (h >= lo) & (h <= hi)
        edge = np.minimum(h - lo, hi - h)
    else:
        inside = (h >= lo) | (h <= hi)
        d_lo = np.where(h >= lo, h - lo, 360.0 - lo + h)
        d_hi = np.where(h <= hi, hi - h, 360.0 - h + hi)
        edge = np.minimum(d_lo, d_hi)
    return inside.astype(np.float32) * _smoothstep(0.0, 6.0, edge)


def apply_look_oklab(lab_l: Any, lab_a: Any, lab_b: Any, look: str, strength: float = 1.0) -> tuple[Any, Any, Any]:
    return _apply_look_oklab(
        lab_l, lab_a, lab_b, look, strength, parallel=True
    )


def _apply_look_oklab_reference(
    lab_l: Any, lab_a: Any, lab_b: Any, look: str, strength: float = 1.0
) -> tuple[Any, Any, Any]:
    """Serial oracle retained for exact optimized-path validation."""
    return _apply_look_oklab(
        lab_l, lab_a, lab_b, look, strength, parallel=False
    )


def _apply_look_oklab(
    lab_l: Any,
    lab_a: Any,
    lab_b: Any,
    look: str,
    strength: float,
    *,
    parallel: bool,
) -> tuple[Any, Any, Any]:
    """Apply the measured chromatic field on Oklab coordinates.

    L is untouched. Four operator families:
    1) sector hue rotation (e.g. green toward cyan),
    2) L-dependent chroma trim (shadow / highlight ramps),
    3) high-saturation soft knee (extra relief above sat_knee_c),
    4) skin-band hue convergence + chroma damp.
    """
    field = LOOK_FIELDS[look]
    s = np.float32(max(0.0, strength))
    from .cpu_budget import current_inner

    # S3 budget (review batch 18): the look pool used to ignore the budget
    # entirely and stack its own submits inside every outer chunk worker.
    use_parallel = (
        parallel
        and np.asarray(lab_l).size >= _LOOK_PARALLEL_MIN_PIXELS
        and current_inner() > 1
    )
    if use_parallel:
        chroma_future = _LOOK_POOL.submit(np.hypot, lab_a, lab_b)
        hue = np.degrees(np.arctan2(lab_b, lab_a)) % 360.0
        chroma = chroma_future.result()
    else:
        chroma = np.hypot(lab_a, lab_b)
        hue = np.degrees(np.arctan2(lab_b, lab_a)) % 360.0

    chroma_w = _smoothstep(0.005, 0.03, chroma)
    rot = np.radians(_periodic_interp(field.hue_rotation_deg, hue)) * s * chroma_w
    if use_parallel:
        sin_future = _LOOK_POOL.submit(np.sin, rot)
        cos_r = np.cos(rot)
        sin_r = sin_future.result()
    else:
        cos_r = np.cos(rot)
        sin_r = np.sin(rot)
    a2 = lab_a * cos_r - lab_b * sin_r
    b2 = lab_a * sin_r + lab_b * cos_r
    hue = np.degrees(np.arctan2(b2, a2)) % 360.0

    skin_w = _hue_in_arc(hue, field.skin_hue_lo, field.skin_hue_hi) * chroma_w
    if field.skin_hue_pull > 0.0:
        delta = (field.skin_hue_center - hue + 180.0) % 360.0 - 180.0
        pull = np.radians(delta) * np.float32(field.skin_hue_pull) * s * skin_w
        if use_parallel:
            sin_future = _LOOK_POOL.submit(np.sin, pull)
            cos_p = np.cos(pull)
            sin_p = sin_future.result()
        else:
            cos_p = np.cos(pull)
            sin_p = np.sin(pull)
        a3 = a2 * cos_p - b2 * sin_p
        b3 = a2 * sin_p + b2 * cos_p
        a2, b2 = a3, b3
        hue = np.degrees(np.arctan2(b2, a2)) % 360.0

    magenta_w = _hue_in_arc(hue, field.magenta_hue_lo, field.magenta_hue_hi) * chroma_w * (1.0 - skin_w)
    if field.magenta_hue_pull > 0.0:
        delta = (field.magenta_hue_center - hue + 180.0) % 360.0 - 180.0
        pull = np.radians(delta) * np.float32(field.magenta_hue_pull) * s * magenta_w
        if use_parallel:
            sin_future = _LOOK_POOL.submit(np.sin, pull)
            cos_m = np.cos(pull)
            sin_m = sin_future.result()
        else:
            cos_m = np.cos(pull)
            sin_m = np.sin(pull)
        a3 = a2 * cos_m - b2 * sin_m
        b3 = a2 * sin_m + b2 * cos_m
        a2, b2 = a3, b3

    scale = _periodic_interp(field.chroma_ratio, hue).astype(np.float32)
    shadow_extra = field.shadow_chroma_ratio / field.mid_chroma_ratio
    highlight_extra = field.highlight_chroma_ratio / field.mid_chroma_ratio
    scale = scale * (1.0 + (shadow_extra - 1.0) * _smoothstep(field.shadow_l_hi, field.shadow_l_lo, lab_l))
    scale = scale * (1.0 + (highlight_extra - 1.0) * _smoothstep(field.highlight_l_lo, field.highlight_l_hi, lab_l))
    if field.sat_knee_relief != 1.0:
        knee_w = _smoothstep(field.sat_knee_c, field.sat_knee_c + 0.08, chroma)
        scale = scale * (1.0 + (np.float32(field.sat_knee_relief) - 1.0) * knee_w)
    if field.skin_chroma_scale != 1.0:
        scale = scale * (1.0 + (np.float32(field.skin_chroma_scale) - 1.0) * skin_w)
    if field.magenta_chroma_scale != 1.0:
        scale = scale * (1.0 + (np.float32(field.magenta_chroma_scale) - 1.0) * magenta_w)
    scale = 1.0 + (scale - 1.0) * s
    a2 = a2 * scale
    b2 = b2 * scale

    cool_w = s * _smoothstep(field.shadow_l_hi, field.shadow_l_lo, lab_l) * (1.0 - chroma_w)
    if field.shadow_cool_a != 0.0:
        a2 = a2 + np.float32(field.shadow_cool_a) * cool_w
    if field.shadow_cool_b != 0.0:
        b2 = b2 + np.float32(field.shadow_cool_b) * cool_w

    neutral_chroma_w = 1.0 - _smoothstep(
        field.neutral_cool_chroma_hi * 0.45, field.neutral_cool_chroma_hi, chroma
    )
    neutral_l_w = _smoothstep(field.neutral_cool_l_lo, field.neutral_cool_l_hi, lab_l)
    neutral_l_w = neutral_l_w * (1.0 - _smoothstep(field.neutral_cool_l_hi, field.neutral_cool_l_falloff, lab_l))
    neutral_w = s * neutral_chroma_w * neutral_l_w * (1.0 - skin_w)
    if field.neutral_cool_a != 0.0:
        a2 = a2 + np.float32(field.neutral_cool_a) * neutral_w
    if field.neutral_cool_b != 0.0:
        b2 = b2 + np.float32(field.neutral_cool_b) * neutral_w

    skin_l_w = _smoothstep(0.22, 0.58, lab_l) * (1.0 - _smoothstep(0.88, 0.98, lab_l))
    skin_tint_w = s * skin_w * skin_l_w
    if field.skin_warm_a != 0.0:
        a2 = a2 + np.float32(field.skin_warm_a) * skin_tint_w
    if field.skin_warm_b != 0.0:
        b2 = b2 + np.float32(field.skin_warm_b) * skin_tint_w

    highlight_chroma_w = 1.0 - _smoothstep(
        field.highlight_warm_chroma_hi * 0.50, field.highlight_warm_chroma_hi, chroma
    )
    highlight_w = s * highlight_chroma_w * _smoothstep(field.highlight_warm_l_lo, field.highlight_warm_l_hi, lab_l)
    if field.highlight_warm_a != 0.0:
        a2 = a2 + np.float32(field.highlight_warm_a) * highlight_w
    if field.highlight_warm_b != 0.0:
        b2 = b2 + np.float32(field.highlight_warm_b) * highlight_w

    return lab_l, a2, b2
