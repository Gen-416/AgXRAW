# SPDX-License-Identifier: GPL-3.0-or-later
"""HDR AgX v2 tone math: a log-stop shoulder that cannot reach into the toe.

v1 bought its extended white by raising the whole curve's encoding gamma. That works
arithmetically -- the encoded shoulder does reach 2^H -- but gamma is a coordinate of the
*entire* curve, so raising it to 4.7 also rewrote the toe. Measured on real frames, mid
gray and its local derivative held while -4 EV fell about 1.5 EV and the deepest shadows
lost nearly three stops. Headroom was silently buying darkness.

v2 puts the degree of freedom where the change belongs. Below the shoulder start K the
curve is the darktable body at its own fixed gamma, and nothing in this module can touch
it. Above K a cubic Hermite runs in output-stop coordinates

    z(e) = log2(T(e) / 0.18)

from (K, Z_K) with the body's own slope M_K to (W, Z_peak) with slope exactly zero. The
structural consequence is the property v1 could not offer: changing H changes only the
segment above K, so `T(e)` for `e <= K` is bit-identical across headrooms.

Output stops are the right coordinate because a one-stop scene change is a doubling in
linear T. Demanding a "decelerating" curve in linear T mistakes ordinary exposure
proportionality for runaway growth; in z it is just slope 1, and a shoulder is honestly a
reduction of dz/de toward zero.

Monotonicity is decided by a stated condition rather than inspection: with the white
tangent pinned at zero, a single cubic Hermite is monotone exactly when the normalized
start tangent alpha lies in [0, 3]. Alpha above 3 does not mean the request is malformed:
display headroom caps Z_peak independently of the tail-driven W, so a low-headroom
display with a long reliable tail legitimately asks for a shoulder more compressive than
one cubic can carry. There the compiler subdivides into a monotone Hermite chain that
keeps the same structural contract -- the K tangent, the zero white tangent and C1 joins
between pieces -- so behaviour stays continuous in the headroom control instead of
snapping to "no HDR". Fail-closed remains for genuinely degenerate input (empty window,
non-positive rise, non-finite anchors). Subdivision stays an explicit opt-in argument so
auxiliary callers state which contract they want.

Float64 and image-free: this is the oracle the float32 runtime is checked against.
"""
from __future__ import annotations

import math

from .constants import (
    AGX_REFERENCE_RANGE_EV,
    DARKTABLE_BASE_GAMMA,
    OUTPUT_REFERENCE_WHITE_STOPS,
    SCENE_MIDGRAY,
)
from .models import HdrShoulderSegment

# A single cubic Hermite with a zero end tangent is monotone iff its normalized start
# tangent is within this bound (exact for beta = 0: the derivative's interior minimum
# reaches zero at (alpha-3)^2 = 0). The same value doubles as the radius of the
# Fritsch-Carlson disc used as _segment_is_monotone's fast pass; interior chain
# segments with beta > 0 fall through to the exact derivative check instead.
MAX_SINGLE_SEGMENT_ALPHA = 3.0

# Subdivision ceiling for monotone Hermite chains (authoritative low-headroom shoulders
# and the reference-white chroma candidate alike). Each split roughly halves alpha, so the
# reachable range grows exponentially; a request needing more than this is malformed
# rather than demanding.
MAX_SHOULDER_SEGMENTS = 16

_EPS = 1e-12


def requested_headroom_ev(
    reliable_tail_ev: float,
    display_headroom_ev: float,
    reference_white_stops: float = OUTPUT_REFERENCE_WHITE_STOPS,
) -> float:
    """Stops above output reference white that the trustworthy capture tail justifies.

    The scene median is deliberately absent: HDR capacity must never act as an automatic
    exposure. A missing or non-finite tail yields exactly zero, so an absent measurement
    cannot be read as unlimited signal.
    """
    if not math.isfinite(float(reliable_tail_ev)) or not math.isfinite(
        float(display_headroom_ev)
    ):
        return 0.0
    signal = max(0.0, float(reliable_tail_ev) - float(reference_white_stops))
    return min(max(0.0, float(display_headroom_ev)), signal)


def body_encoded_slope(contrast: float, reference_range_ev: float = AGX_REFERENCE_RANGE_EV) -> float:
    """Encoded-curve slope per scene EV on the body's central line: `contrast / 16.5`.

    Contrast is quoted against AgX's historical -10..+6.5 EV range, so it is not itself a
    slope. Deriving this rather than storing it keeps one source of truth.
    """
    return float(contrast) / float(reference_range_ev)


def body_anchor_at_ev(
    ev: float,
    contrast: float,
    body_gamma: float = DARKTABLE_BASE_GAMMA,
    midgray: float = SCENE_MIDGRAY,
    reference_range_ev: float = AGX_REFERENCE_RANGE_EV,
) -> tuple[float, float, float]:
    """Body value, output stops and output-stop slope at one scene EV.

    Returns `(T, z, dz/de)` on the body's central encoded line, which is where the
    shoulder must attach. Everything is derived from midgray / gamma / contrast / range,
    so no anchor value is stored as an independent constant that could drift from them.
    """
    q0 = float(midgray) ** (1.0 / float(body_gamma))
    slope_q = body_encoded_slope(contrast, reference_range_ev)
    q = q0 + slope_q * float(ev)
    if q <= _EPS:
        return 0.0, -math.inf, math.inf
    value = q ** float(body_gamma)
    stops = math.log2(value / float(midgray))
    # dz/de = (dT/de) / (ln2 * T), and dT/de = g*q^(g-1)*a_q, so T cancels to g*a_q/(ln2*q).
    slope_z = float(body_gamma) * slope_q / (math.log(2.0) * q)
    return value, stops, slope_z


def body_anchor_from_curve(
    evaluate_body_with_derivative, ev: float, midgray: float = SCENE_MIDGRAY
) -> tuple[float, float, float]:
    """Anchor taken from the body curve that will actually render, not a closed form.

    `body_anchor_at_ev` assumes K sits on the central encoded line. In production it does
    not reliably: the darktable body's own shoulder transition lands within a few 1e-4 of
    K on real plans and past it on some, because the plan's latitude happens to be close
    to the chosen K. The closed form is then off by ~4e-5 in T, which is small but makes
    the C1 join approximate -- and that join is the property the whole design rests on.

    The callback returns the actual production value and its analytic `dT/de` from the
    same compiled curve parameters. This makes the join independent of which segment K
    falls in. A finite difference is deliberately not used: the production body renders
    in float32, and steps small enough to localize the transition lose derivative precision.
    """
    value, slope_t = evaluate_body_with_derivative(ev)
    value = float(value)
    slope_t = float(slope_t)
    if value <= _EPS:
        return 0.0, -math.inf, math.inf
    stops = math.log2(value / float(midgray))
    slope_z = slope_t / (math.log(2.0) * value)
    return value, stops, slope_z


def _hermite(u: float, z0: float, z1: float, m0: float, m1: float, span_e: float) -> float:
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return h00 * z0 + h10 * span_e * m0 + h01 * z1 + h11 * span_e * m1


def evaluate_hdr_shoulder(scene_ev: float, segments: tuple[HdrShoulderSegment, ...]) -> float:
    """Output stops at one scene EV, clamped to the endpoints outside the shoulder."""
    if not segments:
        return 0.0
    if scene_ev <= segments[0].e0:
        return segments[0].z0
    if scene_ev >= segments[-1].e1:
        return segments[-1].z1
    for seg in segments:
        if scene_ev <= seg.e1:
            span = seg.e1 - seg.e0
            if span <= 0.0:
                return seg.z1
            u = (scene_ev - seg.e0) / span
            return _hermite(u, seg.z0, seg.z1, seg.m0, seg.m1, span)
    return segments[-1].z1


def _segment_chain(
    knots_e: list[float], knots_z: list[float], knots_m: list[float]
) -> tuple[HdrShoulderSegment, ...]:
    return tuple(
        HdrShoulderSegment(
            e0=knots_e[i], e1=knots_e[i + 1],
            z0=knots_z[i], z1=knots_z[i + 1],
            m0=knots_m[i], m1=knots_m[i + 1],
        )
        for i in range(len(knots_e) - 1)
    )


def adaptive_monotone_segments(
    knee_ev: float,
    white_ev: float,
    knee_stops: float,
    peak_stops: float,
    knee_slope: float,
) -> tuple[HdrShoulderSegment, ...]:
    """Build a monotone Hermite chain while holding both end tangents fixed.

    Used when a single segment cannot span the requested geometry: the authoritative
    curve on low-headroom displays (Z_peak capped while the tail pushes W out), and the
    reference-white chroma candidate, whose endpoint is intentionally decoupled from the
    scene-derived W/H pair. A general PCHIP limiter would rescale the first tangent and
    break the C1 body join; subdivision instead adds free interior knots while preserving
    `knee_slope` and the zero tangent at W. Interior tangents use the Fritsch-Carlson
    harmonic mean.
    """
    span_e = float(white_ev) - float(knee_ev)
    span_z = float(peak_stops) - float(knee_stops)
    if span_e <= 0.0 or span_z <= _EPS:
        return ()

    for count in range(1, MAX_SHOULDER_SEGMENTS + 1):
        knots_e = [knee_ev + span_e * i / count for i in range(count + 1)]
        # Interpolate the stop targets along a shape that already decelerates, so the
        # secants fall monotonically and interior tangents inherit that ordering.
        fractions = [i / count for i in range(count + 1)]
        knots_z = [
            knee_stops + span_z * (1.0 - (1.0 - f) ** 2) for f in fractions
        ]
        knots_z[-1] = float(peak_stops)

        secants = [
            (knots_z[i + 1] - knots_z[i]) / (knots_e[i + 1] - knots_e[i])
            for i in range(count)
        ]
        knots_m = [float(knee_slope)]
        for i in range(1, count):
            left, right = secants[i - 1], secants[i]
            if left <= 0.0 or right <= 0.0:
                knots_m.append(0.0)
            else:
                knots_m.append(2.0 * left * right / (left + right))
        knots_m.append(0.0)

        segments = _segment_chain(knots_e, knots_z, knots_m)
        if all(_segment_is_monotone(seg) for seg in segments):
            return segments
    return ()


def _segment_is_monotone(seg: HdrShoulderSegment) -> bool:
    """Exact cubic-Hermite monotonicity via the derivative quadratic.

    The Fritsch-Carlson radius-3 disc is kept as a fast sufficient pass, but it
    is not necessary: interior chain segments carry harmonic-mean end tangents
    (beta > 0), and the disc rejects segments the curve itself proves monotone
    — even the classic [0,3]x[0,3] box lies partly outside it. Measured on the
    uniform-knot chain, the disc admits only global alpha <= 5.5025 while the
    derivative admits ~7.7493; production policies in that gap were silently
    zeroing rendered headroom or degrading the reference candidate to native.
    Outside the disc, minimize q(t) = 3(a+b-2)t^2 + 2(3-2a-b)t + a on [0,1]
    exactly: q(0) = alpha and q(1) = beta are already non-negative, so only an
    interior vertex of an upward parabola can go negative.
    """
    alpha, beta = seg.alpha, seg.beta
    if not (math.isfinite(alpha) and math.isfinite(beta)):
        return False
    if alpha < 0.0 or beta < 0.0:
        return False
    if alpha * alpha + beta * beta <= MAX_SINGLE_SEGMENT_ALPHA ** 2 + 1e-12:
        return True
    a = 3.0 * (alpha + beta - 2.0)
    b = 2.0 * (3.0 - 2.0 * alpha - beta)
    if a <= 0.0:
        # Concave or linear derivative: the minimum sits at an endpoint.
        return True
    vertex = -b / (2.0 * a)
    if vertex <= 0.0 or vertex >= 1.0:
        return True
    return alpha - b * b / (4.0 * a) >= -1e-9


def compile_hdr_shoulder_from_anchor(
    knee_ev: float,
    white_ev: float,
    knee_stops: float,
    knee_slope: float,
    peak_stops: float,
    allow_subdivision: bool = False,
) -> tuple[HdrShoulderSegment, ...]:
    """Same solve, but with the knee anchor supplied rather than derived.

    Lets a second endpoint reuse one compiled anchor, so two candidate curves provably
    leave the body at the same value and slope and differ only above K. Subdivision is an
    explicit argument so each caller states which contract it wants; the chain keeps the
    supplied knee tangent and the zero white tangent either way.
    """
    span_e = float(white_ev) - float(knee_ev)
    span_z = float(peak_stops) - float(knee_stops)
    if span_e <= 0.0 or span_z <= _EPS:
        return ()
    single = HdrShoulderSegment(
        e0=float(knee_ev), e1=float(white_ev),
        z0=float(knee_stops), z1=float(peak_stops),
        m0=float(knee_slope), m1=0.0,
    )
    if _segment_is_monotone(single):
        return (single,)
    if allow_subdivision:
        return adaptive_monotone_segments(
            knee_ev, white_ev, knee_stops, peak_stops, knee_slope
        )
    return ()


def compile_hdr_shoulder(
    knee_ev: float,
    white_ev: float,
    peak_stops: float,
    contrast: float,
    body_gamma: float = DARKTABLE_BASE_GAMMA,
    midgray: float = SCENE_MIDGRAY,
    reference_range_ev: float = AGX_REFERENCE_RANGE_EV,
    evaluate_body_with_derivative=None,
    allow_subdivision: bool = False,
) -> tuple[HdrShoulderSegment, ...]:
    """Build the shoulder joining the body at K to the content peak at W.

    Pass `evaluate_body_with_derivative` to anchor on the real rendering curve; without it
    the central-line closed form is used, which is exact only when K lies on the linear
    latitude segment. Production always passes an actual value plus analytic derivative,
    because on real plans K lands at or just past the body's own shoulder transition.

    A single monotone segment is preferred and is always used when alpha <= 3. W couples
    to the tail, but Z_peak is additionally capped by display headroom, so low-headroom
    requests can exceed the single-segment bound while remaining well-posed; with
    `allow_subdivision=True` (the authoritative plan compiler's setting) those compile to
    a monotone Hermite chain under the same structural contract. With subdivision off the
    result is an empty tuple, which callers must treat as "no shoulder", never as licence
    to enter an unvalidated shape.
    """
    if evaluate_body_with_derivative is not None:
        _, knee_stops, knee_slope = body_anchor_from_curve(
            evaluate_body_with_derivative, knee_ev, midgray=midgray
        )
    else:
        _, knee_stops, knee_slope = body_anchor_at_ev(
            knee_ev, contrast, body_gamma, midgray, reference_range_ev
        )
    if not math.isfinite(knee_stops) or not math.isfinite(knee_slope):
        return ()
    span_e = float(white_ev) - float(knee_ev)
    span_z = float(peak_stops) - float(knee_stops)
    if span_e <= 0.0 or span_z <= _EPS:
        return ()

    single = HdrShoulderSegment(
        e0=float(knee_ev), e1=float(white_ev),
        z0=float(knee_stops), z1=float(peak_stops),
        m0=float(knee_slope), m1=0.0,
    )
    if _segment_is_monotone(single):
        return (single,)
    if allow_subdivision:
        return adaptive_monotone_segments(
            knee_ev, white_ev, knee_stops, peak_stops, knee_slope
        )
    return ()


def validate_hdr_shoulder(
    segments: tuple[HdrShoulderSegment, ...],
    knee_slope: float,
    peak_stops: float,
) -> tuple[bool, str]:
    """Check the structural contract: C1 joins, pinned tangents, monotone pieces."""
    if not segments:
        return False, "shoulder 为空"
    if abs(segments[0].m0 - float(knee_slope)) > 1e-9:
        return False, "起点导数被修改，K 处不再与 body C1"
    if abs(segments[-1].m1) > 1e-12:
        return False, "白端导数非零"
    if abs(segments[-1].z1 - float(peak_stops)) > 1e-9:
        return False, "白端未到达 Z_peak"
    for i, seg in enumerate(segments):
        if not _segment_is_monotone(seg):
            return False, f"segment {i} 不满足单调条件 (alpha={seg.alpha:.4f})"
        if i and abs(seg.z0 - segments[i - 1].z1) > 1e-12:
            return False, f"segment {i} 与前段值不连续"
        if i and abs(seg.m0 - segments[i - 1].m1) > 1e-12:
            return False, f"segment {i} 与前段导数不连续"
    return True, ""


def achieved_headroom_ev(hdr_response) -> float:
    """Actual headroom reached by rendered pixels, distinct from the curve's endpoint."""
    from ._deps import np

    arr = np.asarray(hdr_response, dtype=np.float64)
    peak = float(np.max(arr)) if arr.size else 0.0
    return float(math.log2(peak)) if peak > 1.0 else 0.0
