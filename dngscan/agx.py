# SPDX-License-Identifier: GPL-3.0-or-later
"""AgX view-transform core used by dngscan's JPEG export pipeline."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, NamedTuple

try:
    import numpy as np
except Exception:  # pragma: no cover - handled by dngscan.core import checks
    np = None  # type: ignore[assignment]

EPS = 1e-12
_FORMATION_POOL = ThreadPoolExecutor(
    max_workers=min(8, max(2, (os.cpu_count() or 4) - 2)),
    thread_name_prefix="dngscan-agx-formation",
)
_PARALLEL_FORMATION_MIN_PIXELS = 64 * 1024

# Linear Rec.2020 as represented by darktable's ICC profile. LittleCMS adapts its
# D65 primaries to the D50 PCS before darktable constructs the AgX custom primaries.
# These values were read from that profile at the pinned upstream revision documented
# in dngscan_assets/README.md; using the unadapted D65 coordinates produces different
# formation matrices even though the RGB buffer itself remains linear Rec.2020.
_WORK_PRIMARIES_XY = (
    (0.7084870354494747, 0.29354694789182006),
    (0.19020836048704062, 0.7753681035836013),
    (0.12924758043987472, 0.04714454388899011),
)
_WORK_WHITE_XY = (0.345702914918791, 0.3585385966799326)
_XYZ_TO_REC2020 = (
    np.array(
        [
            [1.64723243, -0.39361249, -0.23596681],
            [-0.68261733, 1.64761237, 0.01281035],
            [0.02968148, -0.06294926, 1.25388577],
        ],
        dtype=np.float64,
    )
    if np is not None
    else None
)

# Fraction of the pre-curve hue restored after the curve. This follows darktable's
# public parameter exactly: 0 keeps the per-channel curve shift, 1 restores input hue.
AGX_HUE_RESTORE = 0.6

# Internal y-axis encoding the curve was originally parameterized with. Kept as the
# reference for the contrast (derivative) compensation when the adaptive gamma moves
# the pivot toward the diagonal (darktable's "keep the pivot on the diagonal").
DEFAULT_CURVE_GAMMA = 2.2

# Minimum x-run reserved for toe/shoulder segments. darktable allows latitude to collapse
# toward ε and warns in the GUI; our headless pipeline forbids that mathematically.
MIN_SEGMENT_X = 0.06

# Search bounds for the EV0-anchor solve on the relocated pivot's linear output. The
# lower bound also defines how far the contrast pivot can travel before the anchor
# becomes unreachable (see the feasibility clamp in curve_params).
PIVOT_Y_SOLVE_MIN = 0.002
PIVOT_Y_SOLVE_MAX = 0.60


class PrimariesGeometry(NamedTuple):
    """darktable-style per-channel inset/outset geometry on the work profile."""

    inset: tuple[float, float, float]
    rotation: tuple[float, float, float]
    outset: tuple[float, float, float]
    unrotation: tuple[float, float, float]
    master_outset_ratio: float
    master_unrotation_ratio: float


# darktable _set_blenderlike_primaries on Rec.2020 (agx.c).
_BLENDER_GEOMETRY = PrimariesGeometry(
    inset=(0.29462451, 0.25861925, 0.14641371),
    rotation=(0.03540329, -0.02108586, -0.06305724),
    outset=(0.290776401758, 0.263155400753, 0.045810721815),
    unrotation=(0.03540329, -0.02108586, -0.06305724),
    master_outset_ratio=1.0,
    master_unrotation_ratio=0.0,
)

# Outset direction (darktable semantics): the outward matrix is the INVERSE of an
# inset built from ratio*outset amounts, so a LARGER master_outset_ratio insets those
# primaries further and its inverse expands purity MORE. ratio > 1 boosts purity above
# Blender's reference (dt slider range 0..2); ratio < 1 mutes it (dt smooth uses 0).
# Verified end-to-end: mean Oklab chroma punchy(1.35)=0.194 > base(1.0)=0.172 >
# muted(0.60)=0.164 on a saturated probe set.
_PUNCHY_GEOMETRY = PrimariesGeometry(
    inset=_BLENDER_GEOMETRY.inset,
    rotation=_BLENDER_GEOMETRY.rotation,
    outset=_BLENDER_GEOMETRY.outset,
    unrotation=_BLENDER_GEOMETRY.unrotation,
    master_outset_ratio=1.35,
    master_unrotation_ratio=0.0,
)

# muted: reduced purity restoration plus full rotation reversal (the softening comes
# from both), keeping the Blender inset character rather than switching to dt smooth.
_MUTED_GEOMETRY = PrimariesGeometry(
    inset=_BLENDER_GEOMETRY.inset,
    rotation=_BLENDER_GEOMETRY.rotation,
    outset=_BLENDER_GEOMETRY.outset,
    unrotation=_BLENDER_GEOMETRY.unrotation,
    master_outset_ratio=0.60,
    master_unrotation_ratio=1.0,
)

# darktable sigmoid smooth on the pipe work profile (_set_smooth_primaries).
_SMOOTH_GEOMETRY = PrimariesGeometry(
    inset=(0.1, 0.1, 0.15),
    rotation=(math.radians(2.0), math.radians(-1.0), math.radians(-3.0)),
    outset=(0.1, 0.1, 0.15),
    unrotation=(math.radians(2.0), math.radians(-1.0), math.radians(-3.0)),
    master_outset_ratio=0.0,
    master_unrotation_ratio=1.0,
)

AGX_PRIMARIES_PRESETS: dict[str, PrimariesGeometry] = {
    "base": _BLENDER_GEOMETRY,
    "punchy": _PUNCHY_GEOMETRY,
    "muted": _MUTED_GEOMETRY,
    "smooth": _SMOOTH_GEOMETRY,
}
AGX_PRIMARIES_CHOICES = tuple(AGX_PRIMARIES_PRESETS.keys())
# Human-readable aliases (CLI/GUI accept these; they resolve to canonical preset keys).
AGX_PRIMARIES_ALIASES = {
    "agx_blender_strong": "base",
    "agx_blender_punchy": "punchy",
    "agx_blender_soft_outset": "muted",
    "agx_dt_smooth": "smooth",
}
AGX_PRIMARIES_CLI_CHOICES = tuple(AGX_PRIMARIES_PRESETS.keys()) + tuple(AGX_PRIMARIES_ALIASES.keys())


def resolve_agx_primaries(name: str) -> str:
    """Map CLI/GUI preset name (including aliases) to a canonical AgX primaries key."""
    key = (name or "base").strip().lower()
    resolved = AGX_PRIMARIES_ALIASES.get(key, key)
    if resolved not in AGX_PRIMARIES_PRESETS:
        return "base"
    return resolved


def _det2(a: float, b: float, c: float, d: float) -> float:
    return a * d - b * c


def _intersect_line_segments(
    x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float,
) -> float:
    den = _det2(x1 - x2, x3 - x4, y1 - y2, y3 - y4)
    if abs(den) < 1e-10:
        return float("inf")
    t = _det2(x1 - x3, x3 - x4, y1 - y3, y3 - y4) / den
    return t if t >= 0.0 else float("inf")


def _find_distance_to_edge(cos_angle: float, sin_angle: float) -> float:
    wx, wy = _WORK_WHITE_XY
    x2, y2 = wx + cos_angle, wy + sin_angle
    best = float("inf")
    for i in range(3):
        j = (i + 1) % 3
        x3, y3 = _WORK_PRIMARIES_XY[i]
        x4, y4 = _WORK_PRIMARIES_XY[j]
        best = min(best, _intersect_line_segments(wx, wy, x2, y2, x3, y3, x4, y4))
    return best


def _xy_to_xyz(xy: tuple[float, float]) -> Any:
    x, y = xy
    yy = 1.0
    return np.array([x * yy / y, yy, (1.0 - x - y) * yy / y], dtype=np.float64)


def _rotate_and_scale_primary(index: int, scaling: float, rotation_rad: float) -> tuple[float, float]:
    px, py = _WORK_PRIMARIES_XY[index]
    wx, wy = _WORK_WHITE_XY
    angle = math.atan2(py - wy, px - wx) + rotation_rad
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dist = _find_distance_to_edge(cos_a, sin_a)
    return (wx + scaling * dist * cos_a, wy + scaling * dist * sin_a)


def _rgb_to_xyz_from_primaries_xy(primaries_xy: tuple[tuple[float, float], ...]) -> Any:
    columns = [_xy_to_xyz(xy) for xy in primaries_xy]
    prim = np.column_stack(columns)
    white = _xy_to_xyz(_WORK_WHITE_XY)
    scale = np.linalg.solve(prim, white)
    return prim @ np.diag(scale)


@lru_cache(maxsize=16)
def _formation_matrices_cached(spec: PrimariesGeometry) -> tuple[Any, Any]:
    """Port of darktable _create_matrices (custom_primaries.c + agx.c)."""
    inset_xy = tuple(
        _rotate_and_scale_primary(i, 1.0 - spec.inset[i], spec.rotation[i]) for i in range(3)
    )
    # darktable stores matrices transposed. Its
    #   dt_colormatrix_mul(custom_to_xyz_T, xyz_to_base_T)
    # therefore applies xyz_to_base @ custom_to_xyz in column-vector notation.
    # Reversing this order fails to preserve the neutral axis.
    inset = _XYZ_TO_REC2020 @ _rgb_to_xyz_from_primaries_xy(inset_xy)
    outset_xy = tuple(
        _rotate_and_scale_primary(
            i,
            1.0 - spec.master_outset_ratio * spec.outset[i],
            spec.master_unrotation_ratio * spec.unrotation[i],
        )
        for i in range(3)
    )
    tmp = _XYZ_TO_REC2020 @ _rgb_to_xyz_from_primaries_xy(outset_xy)
    return inset, np.linalg.inv(tmp)


def matrices_for_preset(preset_name: str) -> tuple[Any, Any]:
    spec = AGX_PRIMARIES_PRESETS.get(preset_name, _BLENDER_GEOMETRY)
    return _formation_matrices_cached(spec)


# Pinned darktable's scene-referred default uses the Blender-like Rec.2020 geometry.
AGX_INSET_REC2020, AGX_OUTSET_REC2020 = (
    matrices_for_preset("base") if np is not None else (None, None)
)


def formation_matrices(plan: Any) -> tuple[Any, Any]:
    """Inset/outset for one tone plan's primaries preset."""
    return matrices_for_preset(str(getattr(plan, "agx_primaries", "base")))


def compute_pivot_ev_offset(body_ev_p50: float, black_ev: float, white_ev: float) -> float:
    """Move max-contrast pivot toward the scene body (darktable picker workflow).

    Wired into build_tone_compression_plan for the AgX-family cores. curve_params holds
    the calibrated EV0 -> 18% anchor via a bisection on the pivot output, so a negative
    body_ev_p50 pulls the steep part of the curve onto the subject without changing the
    frame's overall brightness mapping.
    """
    if body_ev_p50 >= -0.25:
        return 0.0
    weight = min(1.0, max(0.0, (-0.25 - body_ev_p50) / 3.75))
    offset = body_ev_p50 * weight
    range_ev = max(1.0, white_ev - black_ev)
    margin = MIN_SEGMENT_X * range_ev
    lo = black_ev + margin
    hi = min(0.0, white_ev - margin)
    # Policy cap: relocating the pivot more than 2 EV turns "put the steep part on the
    # subject" into "re-expose the frame", which is not this knob's job. curve_params
    # additionally enforces the hard anchor-feasibility bound, which is contrast
    # dependent and may be tighter than this.
    lo = max(lo, -2.0)
    return max(lo, min(hi, offset))


def _clamp_float(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def _apply_matrix3(rgb: Any, matrix: Any) -> Any:
    out = np.empty((rgb.shape[0], 3), dtype=np.float32)
    out[:, 0] = matrix[0, 0] * rgb[:, 0] + matrix[0, 1] * rgb[:, 1] + matrix[0, 2] * rgb[:, 2]
    out[:, 1] = matrix[1, 0] * rgb[:, 0] + matrix[1, 1] * rgb[:, 1] + matrix[1, 2] * rgb[:, 2]
    out[:, 2] = matrix[2, 0] * rgb[:, 0] + matrix[2, 1] * rgb[:, 1] + matrix[2, 2] * rgb[:, 2]
    return out


def _build_curve_params(
    black_ev: float,
    white_ev: float,
    contrast: float,
    toe_power: float,
    shoulder_power: float,
    latitude_lo_ev: float,
    latitude_hi_ev: float,
    pivot_x: float,
    pivot_y_linear: float,
    gamma: float,
    target_black_linear: float,
    target_white_linear: float = 1.0,
    compensate_slope_for_pivot: bool = True,
) -> dict[str, float | bool]:
    # Derived from darktable's GPLv3 AgX implementation:
    # https://github.com/darktable-org/darktable/blob/master/src/iop/agx.c
    # and its OpenCL kernel:
    # https://github.com/darktable-org/darktable/blob/master/data/kernels/agx.cl
    range_ev = max(1.0, white_ev - black_ev)
    pivot_x = _clamp_float(pivot_x, EPS, 1.0 - EPS)
    pivot_y = max(EPS, pivot_y_linear) ** (1.0 / gamma)
    target_black = _clamp_float(target_black_linear, 0.0, 0.15) ** (1.0 / gamma) if target_black_linear > 0.0 else 0.0
    # darktable's curve_target_display_white_ratio: <1 makes the shoulder converge to a
    # faded (sub-display-white) top instead of pure white. The upstream UI is SDR-bound,
    # but the curve itself is display-linear and can represent an extended-white target.
    # Keep the useful lower guard while deliberately avoiding an SDR upper clamp: silently
    # turning an extended request such as 8.0 into 1.0 makes the parameter contract false.
    # The v2 HDR compiler no longer consumes this endpoint directly — it pins the body at
    # target white 1.0 and compiles its own shoulder above the knee (hdr_agx_math) — but
    # the parameter stays honest for any caller that does request an extended target.
    target_white_linear = float(target_white_linear)
    if not math.isfinite(target_white_linear):
        raise ValueError("target_white_linear must be finite")
    target_white = max(0.2, target_white_linear) ** (1.0 / gamma)
    range_adjusted_slope = contrast * (range_ev / 16.5)
    # Contrast compensation (darktable): keep the pivot's slope in LINEAR output terms
    # constant when gamma / pivot_y move, so "contrast" means the same thing whether the
    # adaptive gamma engaged or not. The EV0-anchor solver disables it: coupling slope
    # to pivot_y makes the anchored output non-monotone in pivot_y (U-shaped, target
    # unreachable); a constant encoded slope keeps the solve monotone and gives the
    # relocated pivot the full base contrast.
    if compensate_slope_for_pivot:
        pivot_y_default = 0.18 ** (1.0 / DEFAULT_CURVE_GAMMA)
        derivative_current = gamma * max(EPS, pivot_y) ** (gamma - 1.0)
        derivative_default = DEFAULT_CURVE_GAMMA * pivot_y_default ** (DEFAULT_CURVE_GAMMA - 1.0)
        slope = range_adjusted_slope / (derivative_current / derivative_default)
    else:
        slope = range_adjusted_slope

    # Latitude: a linear mid segment through the pivot. With zero latitude the curve is
    # Troy's pure sigmoid (toe meets shoulder at mid gray) — which converges channels and
    # washes chroma from mid gray UP. Scene-driven latitude pushes the shoulder start
    # above the subject's colorful range in bright wide-DR scenes. Clamps reserve
    # MIN_SEGMENT_X of x-run for both toe and shoulder AND keep the transition y inside
    # the display range, using the SAME clamped latitude for x and y so the transitions
    # stay on the linear segment.
    lat_lo_x = _clamp_float(max(0.0, latitude_lo_ev) / range_ev, 0.0, max(0.0, pivot_x - MIN_SEGMENT_X))
    lat_hi_x = _clamp_float(max(0.0, latitude_hi_ev) / range_ev, 0.0, max(0.0, 1.0 - pivot_x - MIN_SEGMENT_X))
    if slope > EPS:
        lat_lo_x = min(lat_lo_x, max(0.0, (pivot_y - target_black - 0.02) / slope))
        lat_hi_x = min(lat_hi_x, max(0.0, (min(0.95, target_white) - 0.02 - pivot_y) / slope))

    toe_transition_x = max(EPS, pivot_x - lat_lo_x)
    toe_transition_y = max(EPS, pivot_y - slope * lat_lo_x)
    inverse_toe_limit_x = 1.0
    inverse_toe_limit_y = 1.0 - target_black
    inverse_toe_transition_x = 1.0 - toe_transition_x
    inverse_toe_transition_y = 1.0 - toe_transition_y
    toe_scale = -scale(
        inverse_toe_limit_x,
        inverse_toe_limit_y,
        inverse_toe_transition_x,
        inverse_toe_transition_y,
        slope,
        toe_power,
    )
    toe_length_x = toe_transition_x
    toe_dy = max(EPS, toe_transition_y - target_black)
    toe_slope_to_limit = toe_dy / toe_length_x
    need_convex_toe = toe_slope_to_limit > slope
    toe_fallback_power = slope * toe_length_x / toe_dy
    toe_fallback_coefficient = toe_dy / max(EPS, toe_length_x) ** toe_fallback_power

    shoulder_transition_x = min(1.0 - MIN_SEGMENT_X, pivot_x + lat_hi_x)
    shoulder_transition_y = min(target_white - EPS, pivot_y + slope * (shoulder_transition_x - pivot_x))
    shoulder_scale = scale(1.0, target_white, shoulder_transition_x, shoulder_transition_y, slope, shoulder_power)
    shoulder_length_x = 1.0 - shoulder_transition_x
    shoulder_dy = max(EPS, target_white - shoulder_transition_y)
    shoulder_slope_to_limit = shoulder_dy / shoulder_length_x
    need_concave_shoulder = shoulder_slope_to_limit > slope
    shoulder_fallback_power = slope * shoulder_length_x / shoulder_dy
    shoulder_fallback_coefficient = shoulder_dy / max(EPS, shoulder_length_x) ** shoulder_fallback_power
    return {
        "black_ev": black_ev,
        "range_ev": range_ev,
        "gamma": gamma,
        "target_black": target_black,
        "target_white": target_white,
        "toe_power": toe_power,
        "toe_transition_x": toe_transition_x,
        "toe_transition_y": toe_transition_y,
        "toe_scale": toe_scale,
        "need_convex_toe": need_convex_toe,
        "toe_fallback_power": toe_fallback_power,
        "toe_fallback_coefficient": toe_fallback_coefficient,
        "slope": slope,
        "intercept": pivot_y - slope * pivot_x,
        "shoulder_power": shoulder_power,
        "shoulder_transition_x": shoulder_transition_x,
        "shoulder_transition_y": shoulder_transition_y,
        "shoulder_scale": shoulder_scale,
        "need_concave_shoulder": need_concave_shoulder,
        "shoulder_fallback_power": shoulder_fallback_power,
        "shoulder_fallback_coefficient": shoulder_fallback_coefficient,
    }


@lru_cache(maxsize=32)
def curve_params(
    black_ev: float = -10.0,
    white_ev: float = 6.5,
    contrast: float = 3.0,
    toe_power: float = 1.5,
    shoulder_power: float = 3.3,
    latitude_lo_ev: float = 0.0,
    latitude_hi_ev: float = 0.0,
    pivot_ev_offset: float = 0.0,
    target_black_linear: float = 0.0,
    target_white_linear: float = 1.0,
    keep_pivot_diagonal: bool = True,
    curve_gamma: float = DEFAULT_CURVE_GAMMA,
) -> dict[str, float | bool]:
    """AgX curve parameterization with scene-adaptive pivot and adaptive gamma.

    pivot_ev_offset moves the point of maximum contrast (in EV relative to mid gray)
    toward the subject. Calibrated EV 0 keeps mapping to 0.18 linear: the pivot's own
    output is solved for that constraint, so the shift reallocates CONTRAST without
    moving the frame's brightness anchor. The internal y gamma is otherwise solved to
    put the pivot on the curve diagonal (darktable's "keep the pivot on the diagonal"),
    which keeps the curve S-shaped and the toe/shoulder powers effective across
    narrow-DR and dark-scene windows that previously degenerated into fallback curves.
    """
    black_ev = float(black_ev)
    white_ev = float(white_ev)
    range_ev = max(1.0, white_ev - black_ev)

    # Reference curve: unshifted pivot at mid gray, original fixed gamma. Used to read
    # the brightness-preserving output for a shifted pivot.
    pivot_x0 = _clamp_float(-black_ev / range_ev, EPS, 1.0 - EPS)
    pivot_ev_offset = _clamp_float(pivot_ev_offset, black_ev + MIN_SEGMENT_X * range_ev, white_ev - MIN_SEGMENT_X * range_ev)
    # Anchor feasibility (hard math constraint, independent of any caller's policy):
    # under the solver's constant encoded slope, EV 0 sits |offset| EV above the pivot
    # and therefore gains contrast*|offset|/16.5 of encoded rise no matter what the
    # pivot output is. Once that rise alone exceeds the EV0 target minus the lowest
    # usable pivot output, NO pivot output can restore the anchor. Clamp the request to
    # the reachable region instead of silently rendering an unanchored curve.
    if pivot_ev_offset < -1e-6:
        target_encoded = 0.18 ** (1.0 / DEFAULT_CURVE_GAMMA)
        floor_encoded = PIVOT_Y_SOLVE_MIN ** (1.0 / DEFAULT_CURVE_GAMMA)
        max_offset_mag = 16.5 * max(0.0, target_encoded - floor_encoded) / max(contrast, EPS)
        pivot_ev_offset = max(pivot_ev_offset, -0.95 * max_offset_mag)
    pivot_x = _clamp_float((pivot_ev_offset - black_ev) / range_ev, 0.10, 0.90)

    if abs(pivot_ev_offset) > 1e-6:
        reference = _build_curve_params(
            black_ev, white_ev, contrast, toe_power, shoulder_power,
            latitude_lo_ev, latitude_hi_ev,
            pivot_x0, 0.18, DEFAULT_CURVE_GAMMA, target_black_linear, target_white_linear,
        )
        y_encoded = float(apply_curve(np.asarray([pivot_x], dtype=np.float32), reference)[0])
        pivot_y_linear = _clamp_float(y_encoded ** DEFAULT_CURVE_GAMMA, 0.02, 0.50)
    else:
        pivot_y_linear = 0.18

    # darktable exposes this as "keep the pivot on the diagonal". Its scene-referred
    # default keeps the historical 2.2 curve gamma; callers selecting the automatic
    # option retain the older dngscan behavior.
    def _gamma_for(py_linear: float) -> float:
        if keep_pivot_diagonal and pivot_x < 1.0 - EPS and 0.0 < py_linear < 1.0:
            return _clamp_float(float(np.log(py_linear) / np.log(pivot_x)), 1.5, 5.0)
        return _clamp_float(curve_gamma, 0.01, 100.0)

    def _build(py_linear: float) -> dict[str, float | bool]:
        return _build_curve_params(
            black_ev, white_ev, contrast, toe_power, shoulder_power,
            latitude_lo_ev, latitude_hi_ev,
            pivot_x, py_linear, _gamma_for(py_linear), target_black_linear, target_white_linear,
        )

    params = _build(pivot_y_linear)

    if abs(pivot_ev_offset) > 1e-6:
        # EV0 anchor constraint: moving the contrast pivot toward the subject must not
        # drift the calibrated mid-gray mapping. Diagonal-gamma coupling makes the EV0
        # output NON-monotone in pivot_y (measured U-shape whose minimum can sit above
        # 0.18), so the solve fixes gamma at the historical 2.2 — there the output is
        # strictly monotone in pivot_y and the anchor is reachable. Bisect pivot_y
        # until scene EV 0 renders back to 0.18 linear. Runs at plan-compile time
        # (< 40 curve builds), never per pixel.
        def _build_fixed_gamma(py_linear: float) -> dict[str, float | bool]:
            return _build_curve_params(
                black_ev, white_ev, contrast, toe_power, shoulder_power,
                latitude_lo_ev, latitude_hi_ev,
                pivot_x, py_linear, DEFAULT_CURVE_GAMMA,
                target_black_linear, target_white_linear,
                compensate_slope_for_pivot=False,
            )

        x_ev0 = _clamp_float((0.0 - black_ev) / range_ev, EPS, 1.0 - EPS)

        def _ev0_linear(p: dict[str, float | bool]) -> float:
            encoded = float(apply_curve(np.asarray([x_ev0], dtype=np.float32), p)[0])
            return max(0.0, encoded) ** float(p["gamma"])

        tol = 0.005
        lo, hi = PIVOT_Y_SOLVE_MIN, PIVOT_Y_SOLVE_MAX
        params = _build_fixed_gamma(pivot_y_linear)
        if abs(_ev0_linear(params) - 0.18) > tol:
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                candidate = _build_fixed_gamma(mid)
                out = _ev0_linear(candidate)
                if abs(out - 0.18) <= tol:
                    params = candidate
                    break
                if out < 0.18:
                    lo = mid
                else:
                    hi = mid
            else:
                params = _build_fixed_gamma(0.5 * (lo + hi))

    return params


def scale(limit_x: float, limit_y: float, transition_x: float, transition_y: float, slope: float, power: float) -> float:
    projected_rise = slope * max(EPS, limit_x - transition_x)
    actual_rise = max(EPS, limit_y - transition_y)
    base = max(EPS, actual_rise ** (-power) - projected_rise ** (-power))
    return min(1e9, base ** (-1.0 / power))


def sigmoid(x: Any, power: float) -> Any:
    return x / np.power(1.0 + np.power(x, power), 1.0 / power)


def scaled_sigmoid(x: Any, scale_value: float, slope: float, power: float, transition_x: float, transition_y: float) -> Any:
    return scale_value * sigmoid(slope * (x - transition_x) / scale_value, power) + transition_y


def apply_curve(x: Any, params: dict[str, float | bool]) -> Any:
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    # Toe below, shoulder above, and a linear latitude segment through the pivot between
    # them (empty when latitude is zero — then this degenerates to Troy's pure sigmoid).
    # All three pieces share value and slope at the transitions, so the curve stays C1.
    toe = x < float(params["toe_transition_x"])
    shoulder = x > float(params["shoulder_transition_x"])
    mid = ~(toe | shoulder)
    if np.any(mid):
        out[mid] = float(params["slope"]) * x[mid] + float(params["intercept"])
    if np.any(toe):
        if bool(params["need_convex_toe"]):
            out[toe] = float(params["target_black"]) + np.maximum(
                0.0,
                float(params["toe_fallback_coefficient"]) * np.power(np.maximum(x[toe], 0.0), float(params["toe_fallback_power"])),
            )
        else:
            out[toe] = scaled_sigmoid(
                x[toe],
                float(params["toe_scale"]),
                float(params["slope"]),
                float(params["toe_power"]),
                float(params["toe_transition_x"]),
                float(params["toe_transition_y"]),
            )
    if np.any(shoulder):
        if bool(params["need_concave_shoulder"]):
            out[shoulder] = float(params["target_white"]) - np.maximum(
                0.0,
                float(params["shoulder_fallback_coefficient"])
                * np.power(np.maximum(1.0 - x[shoulder], 0.0), float(params["shoulder_fallback_power"])),
            )
        else:
            out[shoulder] = scaled_sigmoid(
                x[shoulder],
                float(params["shoulder_scale"]),
                float(params["slope"]),
                float(params["shoulder_power"]),
                float(params["shoulder_transition_x"]),
                float(params["shoulder_transition_y"]),
            )
    return np.clip(out, float(params["target_black"]), float(params["target_white"]))


def curve_derivative(x: float, params: dict[str, float | bool]) -> float:
    """Analytic derivative of the encoded AgX curve with respect to normalized x.

    `apply_curve` deliberately renders in float32, which makes a small finite-difference
    derivative unstable near a transition. The piece equations are C1 and already carry
    every parameter needed for their exact tangent, so plan compilation should use this
    derivative instead of sampling neighbouring float32 values.
    """
    x = float(x)
    if x <= 0.0 or x >= 1.0:
        return 0.0

    toe_transition = float(params["toe_transition_x"])
    shoulder_transition = float(params["shoulder_transition_x"])
    slope = float(params["slope"])

    if x < toe_transition:
        if bool(params["need_convex_toe"]):
            power = float(params["toe_fallback_power"])
            coefficient = float(params["toe_fallback_coefficient"])
            return coefficient * power * max(x, 0.0) ** (power - 1.0)
        scale_value = float(params["toe_scale"])
        power = float(params["toe_power"])
        t = slope * (x - toe_transition) / scale_value
        return slope * (1.0 + max(t, 0.0) ** power) ** (-1.0 / power - 1.0)

    if x > shoulder_transition:
        if bool(params["need_concave_shoulder"]):
            power = float(params["shoulder_fallback_power"])
            coefficient = float(params["shoulder_fallback_coefficient"])
            return coefficient * power * max(1.0 - x, 0.0) ** (power - 1.0)
        scale_value = float(params["shoulder_scale"])
        power = float(params["shoulder_power"])
        t = slope * (x - shoulder_transition) / scale_value
        return slope * (1.0 + max(t, 0.0) ** power) ** (-1.0 / power - 1.0)

    return slope


def compress_into_gamut(rgb: Any) -> Any:
    # AgX opponent-luminance constants from the pinned darktable implementation. They
    # intentionally differ from standard Rec.2020 Y and belong only to this negative-RGB
    # guard; scene analysis and the luminance core continue to use true Rec.2020 Y.
    coeff = _AGX_OPPONENT_LUMA
    # Malformed NaN/Inf inputs are sanitized by the caller after this reference step;
    # suppress only the expected IEEE invalid-operation warnings so the Python and C++
    # fallback contracts can be tested without polluting normal command output.
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        input_y = coeff[0] * rgb[:, 0] + coeff[1] * rgb[:, 1] + coeff[2] * rgb[:, 2]
        max_rgb = np.max(rgb, axis=1)
        opponent = max_rgb[:, None] - rgb
        opponent_y = coeff[0] * opponent[:, 0] + coeff[1] * opponent[:, 1] + coeff[2] * opponent[:, 2]
        max_opponent = np.max(opponent, axis=1)
        y_compensate_negative = max_opponent - opponent_y + input_y
        offset = np.maximum(-np.min(rgb, axis=1), 0.0)
        rgb_offset = rgb + offset[:, None]
        max_offset = np.max(rgb_offset, axis=1)
        # ``opponent`` is dead after y_compensate_negative. Reusing it here removes
        # one full RGB temporary while preserving the exact ufunc calls and order.
        np.subtract(max_offset[:, None], rgb_offset, out=opponent)
        max_inverse = np.max(opponent, axis=1)
        y_inverse = coeff[0] * opponent[:, 0] + coeff[1] * opponent[:, 1] + coeff[2] * opponent[:, 2]
        y_new = coeff[0] * rgb_offset[:, 0] + coeff[1] * rgb_offset[:, 1] + coeff[2] * rgb_offset[:, 2]
        y_new = max_inverse - y_inverse + y_new
        ratio = np.ones_like(y_new)
        mask = (y_new > y_compensate_negative) & (y_new > EPS)
        ratio[mask] = y_compensate_negative[mask] / y_new[mask]
        np.multiply(rgb_offset, ratio[:, None], out=rgb_offset)
        return rgb_offset


_AGX_OPPONENT_LUMA = np.asarray(
    [0.2658180370250449, 0.59846986045365, 0.1357121025213052],
    dtype=np.float32,
)


def _compress_into_gamut_reference(rgb: Any) -> Any:
    """Allocation-heavy oracle retained for bit-exact hot-path tests."""
    coeff = _AGX_OPPONENT_LUMA
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        input_y = coeff[0] * rgb[:, 0] + coeff[1] * rgb[:, 1] + coeff[2] * rgb[:, 2]
        max_rgb = np.max(rgb, axis=1)
        opponent = max_rgb[:, None] - rgb
        opponent_y = coeff[0] * opponent[:, 0] + coeff[1] * opponent[:, 1] + coeff[2] * opponent[:, 2]
        max_opponent = np.max(opponent, axis=1)
        y_compensate_negative = max_opponent - opponent_y + input_y
        offset = np.maximum(-np.min(rgb, axis=1), 0.0)
        rgb_offset = rgb + offset[:, None]
        max_offset = np.max(rgb_offset, axis=1)
        opponent_offset = max_offset[:, None] - rgb_offset
        max_inverse = np.max(opponent_offset, axis=1)
        y_inverse = coeff[0] * opponent_offset[:, 0] + coeff[1] * opponent_offset[:, 1] + coeff[2] * opponent_offset[:, 2]
        y_new = coeff[0] * rgb_offset[:, 0] + coeff[1] * rgb_offset[:, 1] + coeff[2] * rgb_offset[:, 2]
        y_new = max_inverse - y_inverse + y_new
        ratio = np.ones_like(y_new)
        mask = (y_new > y_compensate_negative) & (y_new > EPS)
        ratio[mask] = y_compensate_negative[mask] / y_new[mask]
        return rgb_offset * ratio[:, None]


def _rgb_to_hsv(rgb: Any) -> Any:
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    delta = maxc - minc
    h = np.zeros_like(maxc)
    mask = delta > EPS
    rmask = mask & (maxc == r)
    gmask = mask & (maxc == g) & ~rmask
    bmask = mask & ~rmask & ~gmask
    safe_delta = np.where(mask, delta, np.float32(1.0))
    np.copyto(h, ((g - b) / safe_delta) % 6.0, where=rmask)
    np.copyto(h, (b - r) / safe_delta + 2.0, where=gmask)
    np.copyto(h, (r - g) / safe_delta + 4.0, where=bmask)
    h = (h / 6.0) % 1.0
    s = np.zeros_like(maxc)
    positive = maxc > EPS
    np.divide(delta, maxc, out=s, where=positive)
    out = np.empty((rgb.shape[0], 3), dtype=np.float32)
    out[:, 0] = h
    out[:, 1] = s
    out[:, 2] = maxc
    return out


def _rgb_to_hsv_reference(rgb: Any) -> Any:
    """Fancy-indexing oracle retained for bit-exact hot-path tests."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    delta = maxc - minc
    h = np.zeros_like(maxc)
    mask = delta > EPS
    rmask = mask & (maxc == r)
    gmask = mask & (maxc == g) & ~rmask
    bmask = mask & ~rmask & ~gmask
    h[rmask] = ((g[rmask] - b[rmask]) / delta[rmask]) % 6.0
    h[gmask] = (b[gmask] - r[gmask]) / delta[gmask] + 2.0
    h[bmask] = (r[bmask] - g[bmask]) / delta[bmask] + 4.0
    h = (h / 6.0) % 1.0
    s = np.zeros_like(maxc)
    positive = maxc > EPS
    s[positive] = delta[positive] / maxc[positive]
    return np.stack([h, s, maxc], axis=1)


def _hsv_to_rgb(hsv: Any) -> Any:
    h = (hsv[:, 0] % 1.0) * 6.0
    s = np.clip(hsv[:, 1], 0.0, None)
    v = hsv[:, 2]
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    out = np.empty((hsv.shape[0], 3), dtype=np.float32)
    for idx, (cr, cg, cb) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        m = i == idx
        if np.any(m):
            np.copyto(out[:, 0], cr, where=m)
            np.copyto(out[:, 1], cg, where=m)
            np.copyto(out[:, 2], cb, where=m)
    return out


def _hsv_to_rgb_reference(hsv: Any) -> Any:
    """Fancy-indexing oracle retained for bit-exact hot-path tests."""
    h = (hsv[:, 0] % 1.0) * 6.0
    s = np.clip(hsv[:, 1], 0.0, None)
    v = hsv[:, 2]
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    out = np.empty((hsv.shape[0], 3), dtype=np.float32)
    for idx, (cr, cg, cb) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        m = i == idx
        if np.any(m):
            out[m, 0] = cr[m]
            out[m, 1] = cg[m]
            out[m, 2] = cb[m]
    return out


def _mix_hue(rgb_linear: Any, pre_hue: Any, restore: float) -> Any:
    """Restore a fraction of pre-curve hue along the shortest arc.

    darktable semantics: 0 keeps processed hue, 1 restores pre-curve hue.
    """
    hsv = _rgb_to_hsv(rgb_linear)
    delta = hsv[:, 0] - pre_hue
    delta -= np.rint(delta)
    hsv[:, 0] = (pre_hue + np.float32(1.0 - restore) * delta) % 1.0
    return _hsv_to_rgb(hsv)


def look_brightness_power(brightness: float) -> float:
    """darktable's UI-brightness to display-power conversion."""
    value = max(EPS, float(brightness))
    return 1.0 / math.sqrt(value) if value < 1.0 else 1.0 / value


def _plan_hue_restore(plan: Any) -> float:
    if hasattr(plan, "hue_restore"):
        return _clamp_float(float(plan.hue_restore), 0.0, 1.0)
    # Compatibility for external callers compiled against the old, inverse parameter.
    if hasattr(plan, "hue_keep"):
        return 1.0 - _clamp_float(float(plan.hue_keep), 0.0, 1.0)
    return AGX_HUE_RESTORE


def prepare_formation(rgb_rec2020: Any, plan: Any, inset_matrix: Any) -> tuple[Any, Any | None]:
    """Apply the guard rail and inset, retaining the hue used by darktable restore.

    HDR needs access to the interval between the per-channel curve and hue restore/outset.
    Keeping that interval explicit avoids reconstructing it from the completed SDR image.
    """
    hue_restore = _plan_hue_restore(plan)
    rgb = compress_into_gamut(rgb_rec2020.astype(np.float32, copy=False))
    inset = _apply_matrix3(rgb, inset_matrix)
    pre_hue = _rgb_to_hsv(np.maximum(inset, 0.0))[:, 0] if hue_restore > 1e-6 else None
    return inset, pre_hue


def apply_formation_curve(inset: Any, plan: Any) -> Any:
    """Map inset scene channels through the existing endpoint-normalized AgX curve."""
    if bool(getattr(plan, "use_c1_endpoints", False)):
        from .drt import apply_c1_endpoints

        linear = apply_c1_endpoints(np.log2(np.maximum(inset / 0.18, EPS)), plan)
    else:
        params = curve_params(
            round(plan.black_ev, 3),
            round(plan.white_ev, 3),
            round(plan.contrast, 3),
            round(plan.toe_power, 3),
            round(plan.shoulder_power, 3),
            round(float(getattr(plan, "latitude_lo_ev", 0.0)), 3),
            round(float(getattr(plan, "latitude_hi_ev", 0.0)), 3),
            round(float(getattr(plan, "pivot_ev_offset", 0.0)), 3),
            round(float(getattr(plan, "target_black_linear", 0.0)), 4),
            round(float(getattr(plan, "target_white_linear", 1.0)), 4),
            curve_gamma=round(
                float(getattr(plan, "curve_gamma", DEFAULT_CURVE_GAMMA)), 4
            ),
        )
        log_encoded = (
            np.log2(np.maximum(inset / 0.18, EPS)) - float(params["black_ev"])
        ) / float(params["range_ev"])
        log_encoded = np.clip(log_encoded, 0.0, 1.0)
        curved = apply_curve(log_encoded, params)
        brightness = max(EPS, float(getattr(plan, "view_brightness", 1.0)))
        if abs(brightness - 1.0) > 1e-6:
            curved = np.power(np.maximum(curved, 0.0), look_brightness_power(brightness))
        linear = np.power(np.maximum(curved, 0.0), float(params["gamma"]))
    brightness = max(EPS, float(getattr(plan, "view_brightness", 1.0)))
    if bool(getattr(plan, "use_c1_endpoints", False)) and abs(brightness - 1.0) > 1e-6:
        linear = np.power(np.maximum(linear, 0.0), look_brightness_power(brightness))
    return linear.astype(np.float32, copy=False)


REC2020_LUMA_WEIGHTS = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float32)

# Bradford LMS sandwich for the joint colour-head field (derived from the
# calibration-base matrices: _BRADFORD @ inv(XYZ_TO_REC2020_white_exact); the
# field's diagonal gains are only meaningful in this basis).
REC2020_TO_LMS = np.asarray(
    [
        [0.6401768878, 0.3055303801, -0.0042787330],
        [-0.0277113829, 1.0542746845, 0.0138541654],
        [0.0067832589, -0.0119134710, 1.0946628631],
    ],
    dtype=np.float32,
)
LMS_TO_REC2020 = np.asarray(
    [
        [1.5425987843, -0.4469153832, 0.0116857969],
        [0.0406666745, 0.9366019925, -0.0116947761],
        [-0.0091163827, 0.0129626425, 0.9133235816],
    ],
    dtype=np.float32,
)


def apply_film_color_rec2020(mapped_rec: Any, scene_rec2020: Any, plan: Any) -> Any:
    """Observe-mode film colour, applied where it was calibrated (stage C).

    The v2 colour-head field is a Rec.2020-basis measurement along the neutral
    exposure axis, so it multiplies post-outset Rec.2020 pixels — a diagonal
    gain does not commute with the outset or any output matrix, which is exactly
    how the v1 sRGB-basis numbers went wrong. The lookup exposure is the SINGLE
    luminance axis EV_Y = log2(Y_scene/0.18) taken from the scene-linear input
    BEFORE tone mapping: a display value has been through the shoulder and no
    longer corresponds to the calibration exposure, and the field was only ever
    measured along the neutral axis — one exposure, one gain triple, honestly a
    neutral-axis generalization rather than three imagined emulsion exposures.

    The gains are diagonal in Bradford LMS (stage 3's joint field — stable at
    extreme filtration where Rec.2020 components may cross zero), so the pixel
    makes a round trip through the fixed LMS sandwich. Renders with both dials
    at zero keep the byte-exact fast path.
    """
    if str(getattr(plan, "film_mode", "observe")) == "full":
        # Full mode refuses the head at plan compile; this guard keeps the
        # operator honest even for hand-built plans.
        return mapped_rec
    y_cc = float(getattr(plan, "color_head_y", 0.0))
    m_cc = float(getattr(plan, "color_head_m", 0.0))
    if y_cc <= 0.0 and m_cc <= 0.0:
        return mapped_rec
    from .film_curve import color_head_gain_lms

    curves = color_head_gain_lms(
        str(getattr(plan, "curve_preset", "") or ""), y_cc, m_cc
    )
    if curves is None:
        return mapped_rec
    ev_grid, gains = curves
    scene = np.maximum(np.asarray(scene_rec2020, dtype=np.float32), 0.0)
    ev_y = np.log2(np.maximum((scene @ REC2020_LUMA_WEIGHTS) / np.float32(0.18), EPS))
    gain = np.stack(
        [np.interp(ev_y, ev_grid, gains[:, c]).astype(np.float32) for c in range(3)],
        axis=-1,
    )
    # Bradford LMS is a mathematical basis, not a physical gamut: saturated
    # Rec.2020 primaries legally map to negative LMS components, and clamping
    # them here was an UNDECLARED nonlinear projection (measured 0.026 max
    # channel error at unit gains, where the operator must be exact identity).
    # Output-boundary duty stays with the downstream gamut fit.
    lms = mapped_rec @ REC2020_TO_LMS.T
    return ((lms * gain) @ LMS_TO_REC2020.T).astype(np.float32, copy=False)


def finish_formation(
    linear: Any,
    pre_hue: Any | None,
    plan: Any,
    outset_matrix: Any,
    channel_gain: Any | None = None,
) -> Any:
    """Apply darktable hue restore, the film channel gain, and the outset.

    Order: hue restore first, channel gain second. The gain is the medium's measured
    report (layer-saturation differential), not a curve-origin hue swing — hue
    restore exists to moderate the latter and must not dilute the former.
    """
    hue_restore = _plan_hue_restore(plan)
    if pre_hue is not None:
        linear = _mix_hue(linear, pre_hue, hue_restore)
    if channel_gain is not None:
        linear = linear * channel_gain
    return _apply_matrix3(linear, outset_matrix).astype(np.float32)


def apply_core(rgb_rec2020: Any, plan: Any, inset_matrix: Any, outset_matrix: Any) -> Any:
    """AgX's shared formation order in the Rec.2020 working space:

    guard rail -> inset (rotation+attenuation) -> log2 window -> sigmoid ->
    linearize -> hue restore (darktable semantics) -> outset in LINEAR light
    (negative presets with colour-head dials then apply the LMS gain field
    post-outset via apply_film_color_rec2020).

    Deviations from the reference, all deliberate: the endpoint-normalized log2 window
    and C1 sigmoid parameters come from the scene plan while EV=0 remains the calibrated
    mid-gray pivot; the scene DRT uses darktable's default fixed internal gamma, whereas
    the legacy branch retains optional diagonal-pivot gamma. Call formation_matrices(plan)
    for preset-specific inset/outset before invoking this function.
    """
    inset, pre_hue = prepare_formation(rgb_rec2020, plan, inset_matrix)
    linear = apply_formation_curve(inset, plan)
    mapped = finish_formation(linear, pre_hue, plan, outset_matrix)
    return apply_film_color_rec2020(mapped, rgb_rec2020, plan)


def apply_core_parallel(
    rgb_rec2020: Any, plan: Any, inset_matrix: Any, outset_matrix: Any
) -> Any:
    """Exact formation with independent hue capture overlapped with the curve.

    The two branches consume the same immutable inset and are joined before the
    original hue restore/outset sequence. No arithmetic is reordered within either
    branch, so this is byte-for-byte identical to :func:`apply_core`.
    """
    if np.asarray(rgb_rec2020).shape[0] < _PARALLEL_FORMATION_MIN_PIXELS:
        return apply_core(rgb_rec2020, plan, inset_matrix, outset_matrix)
    hue_restore = _plan_hue_restore(plan)
    rgb = compress_into_gamut(rgb_rec2020.astype(np.float32, copy=False))
    inset = _apply_matrix3(rgb, inset_matrix)
    pre_hue_future = None
    if hue_restore > 1e-6:
        nonnegative = np.maximum(inset, 0.0)
        pre_hue_future = _FORMATION_POOL.submit(_rgb_to_hsv, nonnegative)
    linear = apply_formation_curve(inset, plan)
    pre_hue = (
        pre_hue_future.result()[:, 0] if pre_hue_future is not None else None
    )
    mapped = finish_formation(linear, pre_hue, plan, outset_matrix)
    return apply_film_color_rec2020(mapped, rgb_rec2020, plan)
