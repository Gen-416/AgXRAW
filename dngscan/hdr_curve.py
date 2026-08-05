# SPDX-License-Identifier: GPL-3.0-or-later
"""Evaluate the HDR v2 curve over image data: body below K, log-stop shoulder above.

The split is the design. Below `shoulder_start_ev` this delegates to the same darktable
C1 body the plan compiled, and no headroom value reaches that call -- which is why more
HDR range cannot darken shadows the way v1's global gamma did. Above K the compiled
shoulder runs in output stops and clamps at the content peak: a single Hermite segment
when the request's alpha allows it, or the subdivided monotone chain the plan compiler
produced for strongly compressive low-headroom requests. The same tuple-shaped carrier
serves the explicitly subdivided reference-white chroma candidate, which is normalized
to native Y and never becomes a tone authority.

Vectorised because it runs per channel on full frames, but it is only a restatement of
`hdr_agx_math`; that module stays the float64 oracle this is checked against.
"""
from __future__ import annotations

from typing import Any

from ._deps import np
from .constants import SCENE_MIDGRAY
from .drt import apply_c1_endpoints, curve_params_from_plan
from .models import HdrShoulderSegment, HdrToneCurve

_EPS = np.float32(1e-12)


def _hermite_stops(ev: Any, segments: tuple[HdrShoulderSegment, ...]) -> Any:
    """Output stops for every sample, selecting the owning segment per element."""
    e = np.asarray(ev, dtype=np.float32)
    out = np.empty_like(e)
    # Below the first knot the shoulder is not defined; callers mask those samples away,
    # but seeding with z0 keeps any stray sample continuous rather than undefined.
    out[...] = np.float32(segments[0].z0)
    for seg in segments:
        span = np.float32(seg.e1 - seg.e0)
        if span <= 0.0:
            continue
        inside = (e > np.float32(seg.e0)) & (e <= np.float32(seg.e1))
        if not np.any(inside):
            continue
        u = (e[inside] - np.float32(seg.e0)) / span
        u2 = u * u
        u3 = u2 * u
        out[inside] = (
            (np.float32(2.0) * u3 - np.float32(3.0) * u2 + np.float32(1.0)) * np.float32(seg.z0)
            + (u3 - np.float32(2.0) * u2 + u) * span * np.float32(seg.m0)
            + (np.float32(-2.0) * u3 + np.float32(3.0) * u2) * np.float32(seg.z1)
            + (u3 - u2) * span * np.float32(seg.m1)
        )
    out[e > np.float32(segments[-1].e1)] = np.float32(segments[-1].z1)
    return out


def _apply_shoulder_above_knee(
    body: Any,
    ev: Any,
    knee: float,
    segments: tuple[HdrShoulderSegment, ...],
) -> Any:
    above = ev > np.float32(knee)
    if not np.any(above):
        return np.asarray(body, dtype=np.float32)
    out = np.asarray(body, dtype=np.float32).copy()
    stops = _hermite_stops(ev[above], segments)
    out[above] = np.float32(SCENE_MIDGRAY) * np.exp2(stops)
    return out


def apply_hdr_curve(
    scene_rgb: Any,
    tone: HdrToneCurve,
    formation: Any,
    peak_linear: float | None = None,
    *,
    body_params: dict[str, float | bool] | None = None,
) -> Any:
    """Scene-linear channel values -> display-linear HDR output.

    `peak_linear` overrides the plan's endpoint so the conservative reference-white
    chroma candidate can reuse this same primitive at endpoint 1.0. Passing a different
    peak changes only the shoulder; the body below K is identical either way, which is
    what lets the two candidates be compared without a tone difference confounding them.
    """
    rgb = np.asarray(scene_rgb, dtype=np.float32)
    ev = np.log2(np.maximum(rgb, _EPS) / np.float32(SCENE_MIDGRAY))
    params = body_params if body_params is not None else curve_params_from_plan(formation)
    body = apply_c1_endpoints(ev, formation, params=params)

    segments = tone.shoulder_segments
    if not segments:
        return np.asarray(body, dtype=np.float32)

    if peak_linear is not None and abs(float(peak_linear) - float(tone.peak_linear)) > 1e-12:
        segments = _rescaled_segments(tone, float(peak_linear))
        if not segments:
            return np.asarray(body, dtype=np.float32)

    return _apply_shoulder_above_knee(body, ev, float(tone.shoulder_start_ev), segments)


def apply_hdr_curve_pair(
    scene_rgb: Any,
    tone: HdrToneCurve,
    formation: Any,
    *,
    need_reference: bool,
    body_params: dict[str, float | bool] | None = None,
) -> tuple[Any, Any]:
    """Native extended curve plus optional reference-white chroma candidate.

    The body below K is evaluated once. When ``need_reference`` is false the reference
    alias is the native array (no second Hermite). This is an exact identity when rho is
    zero: blend would return native anyway.
    """
    rgb = np.asarray(scene_rgb, dtype=np.float32)
    ev = np.log2(np.maximum(rgb, _EPS) / np.float32(SCENE_MIDGRAY))
    params = body_params if body_params is not None else curve_params_from_plan(formation)
    body = apply_c1_endpoints(ev, formation, params=params)

    segments = tone.shoulder_segments
    if not segments:
        native = np.asarray(body, dtype=np.float32)
        return native, native

    native = _apply_shoulder_above_knee(
        body, ev, float(tone.shoulder_start_ev), segments
    )
    if not need_reference:
        return native, native

    if abs(float(tone.peak_linear) - 1.0) <= 1e-12:
        return native, native

    reference_segments = _rescaled_segments(tone, 1.0)
    if not reference_segments:
        return native, native
    reference = _apply_shoulder_above_knee(
        body, ev, float(tone.shoulder_start_ev), reference_segments
    )
    return native, reference


class HdrCurveTable:
    """One compiled scene-EV -> display-linear curve as a uniform 1D table.

    The design contract (§P3/§12.3) allows the runtime to evaluate the HDR curve
    through a per-plan table instead of per-pixel piecewise math, with the analytic
    evaluator kept as the oracle the table is checked against. The grid spans
    [black_ev, white_ev]; outside it the analytic curve is constant (the body clips
    its normalized input to [0, 1] and the shoulder clamps at the content peak), so
    edge clamping is exact rather than an approximation.
    """

    __slots__ = ("ev_start", "inv_step", "values")

    def __init__(self, ev_start: float, inv_step: float, values: Any) -> None:
        self.ev_start = np.float32(ev_start)
        self.inv_step = np.float32(inv_step)
        self.values = values

    def apply_to_ev(self, ev: Any) -> Any:
        """Linear interpolation on the uniform grid, clamped at both ends."""
        v = self.values
        u = (np.asarray(ev, dtype=np.float32) - self.ev_start) * self.inv_step
        u = np.clip(u, np.float32(0.0), np.float32(v.size - 1))
        # np.clip deliberately preserves NaN.  Keep that public/native contract,
        # but do not pass NaN through the platform-dependent float->int cast used
        # for indexing: Linux/NumPy may produce INT_MIN while macOS produces zero.
        # A temporary zero index is safe because the interpolated result is restored
        # to NaN below before it can leave this function.
        nan_mask = np.isnan(u)
        has_nan = bool(np.any(nan_mask))
        index_u = np.where(nan_mask, np.float32(0.0), u) if has_nan else u
        idx = index_u.astype(np.int32)
        np.minimum(idx, np.int32(v.size - 2), out=idx)
        frac = index_u - idx.astype(np.float32)
        lo = v[idx]
        hi = v[idx + 1]
        out = lo + (hi - lo) * frac
        if has_nan:
            out = np.where(nan_mask, np.float32(np.nan), out)
        return out

    def apply(self, scene_rgb: Any) -> Any:
        rgb = np.asarray(scene_rgb, dtype=np.float32)
        ev = np.log2(np.maximum(rgb, _EPS) / np.float32(SCENE_MIDGRAY))
        return self.apply_to_ev(ev).astype(np.float32, copy=False)


HDR_CURVE_TABLE_POINTS = 8192


def compile_hdr_curve_table(
    tone: HdrToneCurve,
    formation: Any,
    *,
    peak_linear: float | None = None,
    body_params: dict[str, float | bool] | None = None,
    points: int = HDR_CURVE_TABLE_POINTS,
) -> HdrCurveTable:
    """Bake the analytic curve (body below K, shoulder above) onto a uniform EV grid.

    Sampling goes through the very functions the table replaces, so the table cannot
    encode a different curve family than the oracle; only interpolation error remains,
    and the §12.3 gates pin that below 2e-5 linear / 1e-3 output stops.
    """
    e0 = float(tone.black_ev)
    e1 = float(tone.white_ev)
    if not (e1 > e0):
        raise ValueError(f"HDR curve table needs white_ev > black_ev, got {e0}..{e1}")
    n = max(16, int(points))
    grid_ev = np.linspace(e0, e1, n, dtype=np.float64).astype(np.float32)
    grid_rgb = np.float32(SCENE_MIDGRAY) * np.exp2(grid_ev)
    values = apply_hdr_curve(
        grid_rgb.reshape(-1, 1),
        tone,
        formation,
        peak_linear,
        body_params=body_params,
    ).reshape(-1)
    step = (e1 - e0) / (n - 1)
    return HdrCurveTable(e0, 1.0 / step, values.astype(np.float32, copy=False))


def compile_hdr_curve_table_pair(
    tone: HdrToneCurve,
    formation: Any,
    *,
    need_reference: bool,
    body_params: dict[str, float | bool] | None = None,
    points: int = HDR_CURVE_TABLE_POINTS,
) -> tuple[HdrCurveTable, HdrCurveTable]:
    """Native table plus the reference-white chroma candidate's table.

    Mirrors ``apply_hdr_curve_pair``: when the reference is not needed (or would be
    identical) the native table is aliased, so downstream identity checks keep working.
    """
    native = compile_hdr_curve_table(
        tone, formation, body_params=body_params, points=points
    )
    if (
        not need_reference
        or not tone.shoulder_segments
        or abs(float(tone.peak_linear) - 1.0) <= 1e-12
        or not _rescaled_segments(tone, 1.0)
    ):
        # Same fallbacks as apply_hdr_curve_pair: whenever the analytic pair would
        # return the native curve twice, the tables alias too.
        return native, native
    reference = compile_hdr_curve_table(
        tone, formation, peak_linear=1.0, body_params=body_params, points=points
    )
    return native, reference


def _rescaled_segments(
    tone: HdrToneCurve, peak_linear: float
) -> tuple[HdrShoulderSegment, ...]:
    """Recompile the shoulder for a different endpoint, holding K's anchor fixed.

    Used only for the conservative chroma candidate. Its fixed 1.0 endpoint is not coupled
    to scene W the way the native H endpoint is, so alpha can exceed the single-segment
    bound on high-W plans and subdivision must be available here. The result is normalized
    to native luminance before mixing and therefore cannot become a second tone curve.
    """
    from .hdr_agx_math import compile_hdr_shoulder_from_anchor
    import math

    if peak_linear <= 0.0:
        return ()
    peak_stops = math.log2(peak_linear / SCENE_MIDGRAY)
    first = tone.shoulder_segments[0]
    return compile_hdr_shoulder_from_anchor(
        knee_ev=float(tone.shoulder_start_ev),
        white_ev=float(tone.white_ev),
        knee_stops=float(first.z0),
        knee_slope=float(first.m0),
        peak_stops=peak_stops,
        allow_subdivision=True,
    )
