# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile ToneCompressionPlan values into immutable native AgX parameters."""

from __future__ import annotations

import math
from typing import Any

from ._deps import np
from . import agx as agx_engine
from . import drt as drt_engine
from .constants import (
    OKLAB_M1,
    OKLAB_M1_INV,
    OKLAB_M2,
    OKLAB_M2_INV,
    OUTPUT_GAMUT_SPACES,
    RGB_TO_XYZ,
    XYZ_TO_RGB,
)
from .models import ToneCompressionPlan

# v8 (R2 item 6): the SDR output plan carries the exact two-stage matrices
# (rec2020_to_xyz + xyz_to_output, float64) instead of the pre-merged
# rec2020_to_output that dropped one float32 rounding.
# v9 (#136, HDR per-pixel confidence): raw_gated_channel_separation gains
# the peak-proximity path (y_native, peak) — clip-compromised pixels lose
# chroma authority continuously toward the content peak — so the native HDR
# formation plan/kernel interface changed shape with it.
# v10 (review batch 25, R-P2-6): the HDR plan's output stage carries the same
# exact float64 two-stage matrices as the SDR output plan (v8); the kernel
# reads rec2020_to_xyz/xyz_to_output as float64 for that stage.
NATIVE_ABI_VERSION = 10
NATIVE_OUTPUT_GAMUT_FIT_ITERS = 16
NATIVE_OUTPUT_GAMUT_TOLERANCE = 1e-4

# Compiled plans are tiny, but every distinct scene compiles a distinct plan, so an
# unbounded dict grows for the lifetime of a GUI server session. FIFO-evict beyond this.
_PLAN_CACHE_MAX = 64
_plan_cache: dict[tuple[Any, ...], Any] = {}
_OUTPUT_PLAN_CACHE_MAX = 16
_output_plan_cache: dict[tuple[str, float], Any] = {}


def _flat_matrix(matrix: Any) -> tuple[float, ...]:
    return tuple(float(matrix[i, j]) for i in range(3) for j in range(3))


def _curve_key(params: dict[str, float | bool]) -> tuple[Any, ...]:
    return tuple(params[k] for k in sorted(params))


def _plan_cache_key(plan: ToneCompressionPlan) -> tuple[Any, ...]:
    inset, outset = agx_engine.formation_matrices(plan)
    curve = drt_engine.curve_params_from_plan(plan)
    return (
        _flat_matrix(inset),
        _flat_matrix(outset),
        _curve_key(curve),
        float(plan.hue_restore),
        float(plan.view_brightness),
        float(plan.punch_strength),
        _flat_matrix(RGB_TO_XYZ["Rec2020"]),
        _flat_matrix(XYZ_TO_RGB["Rec2020"]),
        _flat_matrix(OKLAB_M1),
        _flat_matrix(OKLAB_M2),
        _flat_matrix(OKLAB_M1_INV),
        _flat_matrix(OKLAB_M2_INV),
    )


def _finite_plan(plan: ToneCompressionPlan) -> bool:
    for value in _plan_cache_key(plan):
        if isinstance(value, tuple):
            for item in value:
                if isinstance(item, float) and not math.isfinite(item):
                    return False
                if isinstance(item, bool):
                    continue
        elif isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def _build_native_plan(plan: ToneCompressionPlan) -> Any:
    from types import SimpleNamespace

    try:
        from . import _dngscan_fast as ext
    except ImportError as exc:
        raise RuntimeError(str(exc)) from exc
    inset, outset = agx_engine.formation_matrices(plan)
    curve_py = drt_engine.curve_params_from_plan(plan)
    return SimpleNamespace(
        inset=_flat_matrix(inset),
        outset=_flat_matrix(outset),
        curve=SimpleNamespace(**curve_py),
        hue_restore=float(plan.hue_restore),
        view_brightness=max(1e-12, float(plan.view_brightness)),
        punch_strength=float(plan.punch_strength),
        rec2020_to_xyz=_flat_matrix(RGB_TO_XYZ["Rec2020"]),
        xyz_to_rec2020=_flat_matrix(XYZ_TO_RGB["Rec2020"]),
        oklab_m1=_flat_matrix(OKLAB_M1),
        oklab_m2=_flat_matrix(OKLAB_M2),
        oklab_m1_inv=_flat_matrix(OKLAB_M1_INV),
        oklab_m2_inv=_flat_matrix(OKLAB_M2_INV),
    )


def compile_agx_plan(plan: ToneCompressionPlan) -> Any:
    """Return a cached immutable native plan for one tone plan."""
    if not _finite_plan(plan):
        raise ValueError("tone plan contains non-finite parameters")
    key = _plan_cache_key(plan)
    cached = _plan_cache.get(key)
    if cached is None:
        if len(_plan_cache) >= _PLAN_CACHE_MAX:
            _plan_cache.pop(next(iter(_plan_cache)))
        cached = _build_native_plan(plan)
        _plan_cache[key] = cached
    return cached


def _build_output_plan(output_gamut: str, alpha: float) -> Any:
    from types import SimpleNamespace

    space = OUTPUT_GAMUT_SPACES[output_gamut]
    # R2 item 6: NO pre-merge on the Rec.2020 -> output conversion — the
    # kernel applies the two stages exactly as the NumPy graph does (float64
    # accumulate, float32 materialization per stage), so in-gamut pixels are
    # bit-exact against the NumPy path. The gamut fit's own matrices below
    # REMAIN pre-merged: that path only runs on out-of-gamut pixels and is
    # gated by its documented 1e-4 float tolerance, not the exact contract;
    # unmerging it belongs to the full reference-mode program.
    output_to_lms = OKLAB_M1 @ RGB_TO_XYZ[space]
    lms_to_output = XYZ_TO_RGB[space] @ OKLAB_M1_INV
    return SimpleNamespace(
        rec2020_to_xyz=_flat_matrix(RGB_TO_XYZ["Rec2020"]),
        xyz_to_output=_flat_matrix(XYZ_TO_RGB[space]),
        output_to_lms=_flat_matrix(output_to_lms),
        lms_to_output=_flat_matrix(lms_to_output),
        oklab_m2=_flat_matrix(OKLAB_M2),
        oklab_m2_inv=_flat_matrix(OKLAB_M2_INV),
        alpha=float(alpha),
        gamut_fit_iters=NATIVE_OUTPUT_GAMUT_FIT_ITERS,
        gamut_tolerance=NATIVE_OUTPUT_GAMUT_TOLERANCE,
    )


def _luma3(values: Any) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError("luminance weights must have 3 elements")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _hdr_table_namespace(table: Any) -> Any:
    from types import SimpleNamespace

    values = np.ascontiguousarray(np.asarray(table.values, dtype=np.float32))
    if values.ndim != 1 or values.size < 2:
        raise ValueError("HDR curve table must be a 1-D array of >= 2 samples")
    if not np.isfinite(values).all():
        raise ValueError("HDR curve table contains non-finite samples")
    ev_start = float(table.ev_start)
    inv_step = float(table.inv_step)
    if not (math.isfinite(ev_start) and math.isfinite(inv_step)) or inv_step <= 0.0:
        raise ValueError("HDR curve table grid is malformed")
    return SimpleNamespace(ev_start=ev_start, inv_step=inv_step, values=values)


def compile_hdr_formation_plan(
    hdr_plan: Any,
    formation_plan: Any,
    inset_matrix: Any,
    outset_matrix: Any,
    formation_luma: Any,
    curve_tables: tuple[Any, Any],
    peak: float,
    output_gamut: str,
) -> Any:
    """Immutable native plan for one HDR formation chain (dngscan/hdr_agx.py).

    Carries exactly what _form_hdr_chunk consumes: formation matrices, the compiled
    curve-table pair (aliased tables encode "no reference candidate"), the blend and
    fit luminance rows, and the scene-authorized peak. Not cached: it is compiled
    once per render and holds per-plan table arrays.
    """
    from types import SimpleNamespace

    from .hdr_color import output_luma_weights

    if output_gamut not in OUTPUT_GAMUT_SPACES:
        raise ValueError(f"unknown output gamut: {output_gamut}")
    space = OUTPUT_GAMUT_SPACES[output_gamut]

    native_table, reference_table = curve_tables
    has_reference = reference_table is not native_table

    global_rho = float(hdr_plan.color.channel_separation) * float(
        hdr_plan.color.snr_gate
    )
    hue_restore = float(agx_engine._plan_hue_restore(formation_plan))
    punch_strength = float(getattr(formation_plan, "punch_strength", 0.0))
    peak = float(peak)
    for name, value in (
        ("global_rho", global_rho),
        ("hue_restore", hue_restore),
        ("punch_strength", punch_strength),
        ("peak", peak),
    ):
        if not math.isfinite(value):
            raise ValueError(f"HDR plan {name} is not finite")
    if peak <= 0.0:
        raise ValueError("HDR plan peak must be positive")

    plan = SimpleNamespace(
        inset=_flat_matrix(np.asarray(inset_matrix, dtype=np.float64)),
        outset=_flat_matrix(np.asarray(outset_matrix, dtype=np.float64)),
        rec2020_to_xyz=_flat_matrix(RGB_TO_XYZ["Rec2020"]),
        xyz_to_rec2020=_flat_matrix(XYZ_TO_RGB["Rec2020"]),
        xyz_to_output=_flat_matrix(XYZ_TO_RGB[space]),
        oklab_m1=_flat_matrix(OKLAB_M1),
        oklab_m2=_flat_matrix(OKLAB_M2),
        oklab_m1_inv=_flat_matrix(OKLAB_M1_INV),
        oklab_m2_inv=_flat_matrix(OKLAB_M2_INV),
        formation_luma=_luma3(formation_luma),
        output_luma=_luma3(output_luma_weights(output_gamut)),
        hue_restore=hue_restore,
        punch_strength=punch_strength,
        global_rho=global_rho,
        peak=peak,
        native_table=_hdr_table_namespace(native_table),
        reference_table=_hdr_table_namespace(reference_table)
        if has_reference
        else None,
        has_reference=has_reference,
    )
    for value in plan.inset + plan.outset:
        if not math.isfinite(value):
            raise ValueError("HDR formation matrices contain non-finite values")
    return plan


def compile_output_plan(output_gamut: str, alpha: float = 0.05) -> Any:
    """Return immutable precombined matrices for the fused SDR finalizer."""
    if output_gamut not in OUTPUT_GAMUT_SPACES:
        raise ValueError(f"unknown output gamut: {output_gamut}")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("gamut fit alpha must be finite and non-negative")
    key = (output_gamut, alpha)
    cached = _output_plan_cache.get(key)
    if cached is None:
        if len(_output_plan_cache) >= _OUTPUT_PLAN_CACHE_MAX:
            _output_plan_cache.pop(next(iter(_output_plan_cache)))
        cached = _build_output_plan(output_gamut, alpha)
        _output_plan_cache[key] = cached
    return cached
