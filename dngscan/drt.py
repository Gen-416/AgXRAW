# SPDX-License-Identifier: GPL-3.0-or-later
"""Endpoint-normalized C1 DRT using darktable's AgX curve construction.

Black/white endpoints are scene-derived, but the calibrated 0 EV pivot stays at 18%
output. This avoids the failure mode of attaching an endpoint segment across the pivot:
that makes sparse lights glare while the rest of a dark frame stays unreadable.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from ._deps import np
from . import agx

EPS = 1e-6


def _curve_params_key(plan: Any) -> tuple:
    pivot = round(float(getattr(plan, "pivot_ev_offset", 0.0)), 3)
    return (
        round(float(getattr(plan, "black_ev", -10.0)), 3),
        round(float(getattr(plan, "white_ev", 6.5)), 3),
        round(float(getattr(plan, "contrast", 3.0)), 3),
        round(float(getattr(plan, "toe_power", 1.5)), 3),
        round(float(getattr(plan, "shoulder_power", 3.3)), 3),
        round(float(getattr(plan, "latitude_lo_ev", 0.0)), 3),
        round(float(getattr(plan, "latitude_hi_ev", 0.0)), 3),
        pivot,
        float(getattr(plan, "target_black_linear", 0.0)),
        float(getattr(plan, "target_white_linear", 1.0)),
        abs(pivot) > 1e-6,
        float(getattr(plan, "curve_gamma", agx.DEFAULT_CURVE_GAMMA)),
    )


@lru_cache(maxsize=128)
def _curve_params_cached(key: tuple) -> dict[str, float | bool]:
    (
        black_ev,
        white_ev,
        contrast,
        toe_power,
        shoulder_power,
        latitude_lo_ev,
        latitude_hi_ev,
        pivot,
        target_black_linear,
        target_white_linear,
        keep_pivot_diagonal,
        curve_gamma,
    ) = key
    return agx.curve_params(
        black_ev,
        white_ev,
        contrast,
        toe_power,
        shoulder_power,
        latitude_lo_ev,
        latitude_hi_ev,
        pivot_ev_offset=pivot,
        target_black_linear=target_black_linear,
        target_white_linear=target_white_linear,
        keep_pivot_diagonal=keep_pivot_diagonal,
        curve_gamma=curve_gamma,
    )


def curve_params_from_plan(plan: Any) -> dict[str, float | bool]:
    """Compile the darktable-style C1 curve for one scene plan.

    Endpoints stay scene-derived and EV=0 remains the calibrated mid-gray anchor for
    exposure. When pivot_ev_offset is non-zero the contrast pivot moves toward the
    scene body (brightness-preserving shifted pivot + adaptive gamma).

    Results are cached by the rounded parameter tuple: the hot path rebuilds the same
    plan for every chunk and every HDR candidate.
    """
    return _curve_params_cached(_curve_params_key(plan))


def apply_c1_endpoints(
    ev: Any, plan: Any, params: dict[str, float | bool] | None = None
) -> Any:
    """Apply darktable-style C1 sigmoid segments in the shared scene-EV domain."""
    e = np.asarray(ev, dtype=np.float32)
    resolved = params if params is not None else curve_params_from_plan(plan)
    x = (e - float(resolved["black_ev"])) / float(resolved["range_ev"])
    encoded = agx.apply_curve(np.clip(x, 0.0, 1.0), resolved)
    return np.power(np.maximum(encoded, 0.0), float(resolved["gamma"])).astype(
        np.float32, copy=False
    )


# Display-linear reference for the compiled "toe end": the level at which output is
# practically black on an SDR delivery (~sRGB code 12/255). The toe-end scene EV is
# where the curve crosses this level coming up from the black endpoint. It is a
# measurement coordinate for reporting and for the bounded toe_end_offset adjustment,
# not a curve parameter: view brightness and display looks apply after it.
#
# Declared semantics (floor-relative): the crossing is measured at the curve's
# compiled black floor (``target_black_linear``, e.g. a film paper Dmax) PLUS this
# offset. For the common zero-floor plans the reference is exactly the absolute
# 0.002 level as before, byte for byte. For lifted-black plans an absolute 0.002 is
# unreachable by construction — the curve never falls below its floor — so the
# floor-relative reference is the only definition under which "toe end" remains a
# real, monotone measurement and the toe_end_offset control keeps its declared
# direction. When even this crossing does not exist the measurement is None; report
# layers must state "not reached" or omit the fact rather than print a fake number.
TOE_END_DISPLAY_LINEAR = 0.002

# Legality bounds for the re-solved toe power. The lower bound keeps the sigmoid toe
# meaningfully shaped (below ~0.35 the toe flattens into a near-plateau whose crossing
# becomes numerically flat, so the solve loses conditioning); the upper bound matches
# the hardest toe the existing shadow-transition bias can reach with margin. Requests
# whose target crossing is unreachable inside these bounds clamp to the bound and the
# compiled toe-end fact reports the value actually achieved.
TOE_POWER_SOLVE_MIN = 0.35
TOE_POWER_SOLVE_MAX = 3.5

# Display-linear reference for the compiled "shoulder white point": the scene EV at
# which the curve has risen this fraction of the way from its compiled black floor
# (``target_black_linear``) to its compiled white target (``target_white_linear``).
# It is a measurement coordinate for reporting and for the bounded
# shoulder_white_offset adjustment, not a curve parameter: view brightness and
# display looks apply after it.
#
# Declared semantics (span-relative, the shoulder mirror of the toe fix's
# floor-relative reference): for the common floor-0 / white-1 plans the reference is
# exactly display-linear 0.90 (~sRGB code 243); for film presets with a lifted paper
# floor or a faded white the same 90%-of-span point remains a real, monotone
# measurement by construction — the curve always reaches its white target at the
# white endpoint, so an absolute 0.90 could be unreachable while the relative point
# never is. When the crossing still cannot be measured (degenerate curves whose mid
# gray already sits at the reference) the measurement is None; report layers must
# state "not reached" or omit the fact rather than print a fake number.
SHOULDER_WHITE_DISPLAY_RATIO = 0.90

# Legality bounds for the re-solved shoulder power. The lower bound keeps the sigmoid
# shoulder meaningfully shaped (below ~0.7 the roll-off flattens toward its asymptote
# so slowly that the near-white crossing pins to the white endpoint and the solve
# loses conditioning); the upper bound must admit the hardest compiled film-preset
# shoulder (kodachrome64 fits 9.1) with margin, so a zero offset never has to fight
# its own bounds. Requests whose target crossing is unreachable inside these bounds
# clamp to the bound — softest for "later", hardest for "earlier" — and the compiled
# shoulder-white fact reports the value actually achieved.
SHOULDER_POWER_SOLVE_MIN = 0.7
SHOULDER_POWER_SOLVE_MAX = 12.0


def _value_at_ev(ev: float, params: dict[str, float | bool]) -> float:
    x = (ev - float(params["black_ev"])) / float(params["range_ev"])
    x = min(1.0, max(0.0, x))
    encoded = float(agx.apply_curve(np.asarray([x], dtype=np.float32), params)[0])
    return max(0.0, encoded) ** float(params["gamma"])


def toe_end_ev_from_params(
    params: dict[str, float | bool], level: float = TOE_END_DISPLAY_LINEAR
) -> float | None:
    """Scene EV where the compiled curve rises ``level`` above its black floor.

    The reference is floor-relative (see TOE_END_DISPLAY_LINEAR): for a zero
    ``target_black_linear`` floor it is the absolute ``level``, unchanged; for a
    lifted floor it is ``floor + level``, which the curve always leaves upward from
    its black endpoint. The curve is monotone in EV, so a plain bisection is exact
    enough. Returns None when no crossing exists (the curve already sits at or above
    the reference at its black endpoint) — never a fabricated coordinate — and 0.0
    in the degenerate case where even mid gray sits below the reference.
    """
    black = float(params["black_ev"])
    floor_linear = float(params["target_black"]) ** float(params["gamma"])
    effective_level = floor_linear + float(level)
    lo, hi = black + 1e-3, 0.0
    if _value_at_ev(lo, params) >= effective_level:
        return None
    if _value_at_ev(hi, params) < effective_level:
        return 0.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if _value_at_ev(mid, params) < effective_level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def shoulder_white_ev_from_params(
    params: dict[str, float | bool], ratio: float = SHOULDER_WHITE_DISPLAY_RATIO
) -> float | None:
    """Scene EV where the compiled curve rises to ``ratio`` of its display span.

    The reference is span-relative (see SHOULDER_WHITE_DISPLAY_RATIO):
    ``floor + ratio * (white - floor)`` in display-linear terms, which equals the
    absolute ``ratio`` level for the common floor-0 / white-1 plans and stays a real
    crossing for lifted-floor or faded-white film presets. The curve is monotone in
    EV and reaches its white target at the white endpoint by construction, so a plain
    bisection between mid gray and the white endpoint is exact enough. Returns None
    when no crossing exists above mid gray (the curve already sits at or above the
    reference at scene EV 0 — a degenerate window with no honest "white point" to
    report) — never a fabricated coordinate.
    """
    gamma = float(params["gamma"])
    floor_linear = float(params["target_black"]) ** gamma
    white_linear = float(params["target_white"]) ** gamma
    level = floor_linear + float(ratio) * (white_linear - floor_linear)
    white_ev = float(params["black_ev"]) + float(params["range_ev"])
    lo, hi = 0.0, white_ev
    if _value_at_ev(lo, params) >= level:
        return None
    if _value_at_ev(hi, params) < level:
        return None
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if _value_at_ev(mid, params) < level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compiled_curve_transitions(plan: Any) -> dict[str, float | None]:
    """Measured facts of the compiled curve, after every clamp and guard.

    ``toe_end_ev`` is the near-black crossing and ``shoulder_white_ev`` the
    near-white crossing defined above; either is None when the compiled curve has
    no such crossing (report layers must not print a number for it in that case).
    ``toe_start_ev`` and
    ``shoulder_start_ev`` are the actual latitude transition anchors the solver kept
    after reserving its minimum segment runs and display-range clamps — the values a
    report may print as truth, as opposed to the requested plan fields.
    """
    params = curve_params_from_plan(plan)
    black = float(params["black_ev"])
    range_ev = float(params["range_ev"])
    return {
        "toe_end_ev": toe_end_ev_from_params(params),
        "toe_start_ev": black + float(params["toe_transition_x"]) * range_ev,
        "shoulder_start_ev": black + float(params["shoulder_transition_x"]) * range_ev,
        "shoulder_white_ev": shoulder_white_ev_from_params(params),
    }


def _params_for_toe_power(key: tuple, toe_power: float) -> dict[str, float | bool]:
    """Uncached curve build for the bisection: keep solver probes out of the caches."""
    (
        black_ev, white_ev, contrast, _toe_power, shoulder_power,
        latitude_lo_ev, latitude_hi_ev, pivot,
        target_black_linear, target_white_linear, keep_pivot_diagonal, curve_gamma,
    ) = key
    return agx.curve_params.__wrapped__(
        black_ev, white_ev, contrast, float(toe_power), shoulder_power,
        latitude_lo_ev, latitude_hi_ev,
        pivot_ev_offset=pivot,
        target_black_linear=target_black_linear,
        target_white_linear=target_white_linear,
        keep_pivot_diagonal=keep_pivot_diagonal,
        curve_gamma=curve_gamma,
    )


def solve_toe_power_for_toe_end(plan: Any, target_toe_end_ev: float) -> float:
    """Toe power whose compiled curve crosses near-black at the requested scene EV.

    The crossing is monotone in toe power (a lower power opens the toe, so the same
    display level is reached deeper in EV). The solve runs at plan-compile time only
    and touches nothing but ``toe_power``: black/white endpoints, pivot anchor,
    latitude anchors and the shoulder are all fixed inputs, so the sky-side of the
    curve cannot move. Targets deeper than the most open toe can reach clamp to the
    open bound (never the hard one); a plan whose crossing is unmeasurable at either
    bound keeps its current toe power — no solve may be driven by a sentinel.
    """
    key = _curve_params_key(plan)
    black = float(getattr(plan, "black_ev", -10.0))
    target = min(-0.25, max(black + 0.05, float(target_toe_end_ev)))
    lo, hi = TOE_POWER_SOLVE_MIN, TOE_POWER_SOLVE_MAX
    # A lower toe power lifts the toe -> deeper crossing. Check reachability first.
    crossing_lo = toe_end_ev_from_params(_params_for_toe_power(key, lo))
    crossing_hi = toe_end_ev_from_params(_params_for_toe_power(key, hi))
    if crossing_lo is None or crossing_hi is None:
        # No measurable crossing: refuse to move rather than fake a solve.
        return float(getattr(plan, "toe_power", key[3]))
    if crossing_lo > target:
        # Requested deeper than even the most open toe reaches: open, do not harden.
        return lo
    if crossing_hi < target:
        return hi
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        crossing = toe_end_ev_from_params(_params_for_toe_power(key, mid))
        if crossing is None:
            # Interior sentinel cannot happen for a monotone family whose bounds
            # both crossed; bail out conservatively if it ever does.
            return float(getattr(plan, "toe_power", key[3]))
        if abs(crossing - target) <= 0.01:
            return mid
        if crossing < target:
            # crossing too deep -> toe too open -> raise power
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _params_for_shoulder_power(key: tuple, shoulder_power: float) -> dict[str, float | bool]:
    """Uncached curve build for the bisection: keep solver probes out of the caches."""
    (
        black_ev, white_ev, contrast, toe_power, _shoulder_power,
        latitude_lo_ev, latitude_hi_ev, pivot,
        target_black_linear, target_white_linear, keep_pivot_diagonal, curve_gamma,
    ) = key
    return agx.curve_params.__wrapped__(
        black_ev, white_ev, contrast, toe_power, float(shoulder_power),
        latitude_lo_ev, latitude_hi_ev,
        pivot_ev_offset=pivot,
        target_black_linear=target_black_linear,
        target_white_linear=target_white_linear,
        keep_pivot_diagonal=keep_pivot_diagonal,
        curve_gamma=curve_gamma,
    )


def solve_shoulder_power_for_white_ev(plan: Any, target_white_ev: float) -> float:
    """Shoulder power whose compiled curve reaches near-white at the requested scene EV.

    The crossing is monotone in shoulder power (a lower power rolls off sooner and
    approaches white more gradually, so the same near-white reference is reached
    later in EV). This is the shoulder's real geometric degree of freedom: moving the
    shoulder *start* is not — with contrast 3 the display range above the pivot is
    spent within ~1 EV, so the C1 legality clamps eat any start move before it can
    render. The solve runs at plan-compile time only and touches nothing but
    ``shoulder_power``: black/white endpoints, pivot anchor, latitude anchors and the
    toe are all fixed inputs, so nothing below the shoulder can move. Targets later
    than the softest legal shoulder reaches clamp to the soft bound (never the hard
    one); a plan whose crossing is unmeasurable at either bound keeps its current
    shoulder power — no solve may be driven by a sentinel.

    Known degenerate geometry, measured and provable: the shoulder's total encoded
    y-freedom from the pivot is ``contrast * white_ev / 16.5 - (1 - 0.18**(1/2.2))``
    (the mid slope's projected rise minus the rise actually required), which vanishes
    at white_ev ~= 2.98 for contrast 3 — exactly the defensive min-white floor. Plans
    clamped to that floor therefore compile a tangent, line-like shoulder that NO
    legal member of the C1 family can reshape in either direction; both power bounds
    land on nearly the same crossing, the solve clamps, and the compiled
    shoulder_white fact truthfully reports the (non-)movement. Restoring freedom on
    such scenes would require touching contrast or the white endpoint, both outside
    this control's declared invariants.
    """
    key = _curve_params_key(plan)
    white = float(getattr(plan, "white_ev", 6.5))
    target = max(0.25, min(white - 0.05, float(target_white_ev)))
    lo, hi = SHOULDER_POWER_SOLVE_MIN, SHOULDER_POWER_SOLVE_MAX
    # A lower shoulder power softens the roll-off -> later crossing.
    crossing_lo = shoulder_white_ev_from_params(_params_for_shoulder_power(key, lo))
    crossing_hi = shoulder_white_ev_from_params(_params_for_shoulder_power(key, hi))
    if crossing_lo is None or crossing_hi is None:
        # No measurable crossing: refuse to move rather than fake a solve.
        return float(getattr(plan, "shoulder_power", key[4]))
    if crossing_lo < target:
        # Requested later than even the softest shoulder reaches: soften, do not harden.
        return lo
    if crossing_hi > target:
        return hi
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        crossing = shoulder_white_ev_from_params(_params_for_shoulder_power(key, mid))
        if crossing is None:
            # Interior sentinel cannot happen for a monotone family whose bounds
            # both crossed; bail out conservatively if it ever does.
            return float(getattr(plan, "shoulder_power", key[4]))
        if abs(crossing - target) <= 0.01:
            return mid
        if crossing > target:
            # crossing too late -> shoulder too soft -> raise power
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def c1_value_and_derivative_at_ev(ev: float, plan: Any) -> tuple[float, float]:
    """Rendered body value and analytic dT/de at one scene-EV coordinate.

    The value follows the production float32 path exactly. Its tangent is evaluated from
    the same rounded curve parameters and piece equation, avoiding finite differences on
    a float32 renderer. This is the authoritative attachment point for the HDR shoulder.
    """
    params = curve_params_from_plan(plan)
    sample = np.asarray([ev], dtype=np.float32)
    x = (sample - float(params["black_ev"])) / float(params["range_ev"])
    x = np.clip(x, 0.0, 1.0)
    encoded = agx.apply_curve(x, params)
    gamma = float(params["gamma"])
    value = float(
        np.power(np.maximum(encoded, 0.0), gamma).astype(np.float32, copy=False)[0]
    )

    encoded_value = float(encoded[0])
    if value <= 0.0 or encoded_value <= 0.0:
        return value, 0.0
    encoded_slope_x = agx.curve_derivative(float(x[0]), params)
    slope_t_ev = (
        gamma
        * encoded_value ** (gamma - 1.0)
        * encoded_slope_x
        / float(params["range_ev"])
    )
    return value, float(slope_t_ev)
