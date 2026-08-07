# SPDX-License-Identifier: GPL-3.0-or-later
"""film v2 Stage A: the analytic scene->density front (plan §3, §5.2).

Pure math shared by the asset builder, the runtime and the tests — no asset
IO here. Stage A is: plain scene-linear Rec.2020 -> film exposure offset ->
observer inverse (three layer exposures) -> three 1-D characteristic curves
-> per-layer dye amounts, bounded by the profile's declared amount domain.
The spatial operators (halation before the curves, grain after them) insert
between these steps in later stages; their off-state must leave this math
bit-identical.

Density semantics: the stock profiles publish per-layer DYE AMOUNTS along the
logE axis; spectral density is amounts @ dye spectra + base. Stage B's cube is
therefore indexed in amount space — "density domain" in the plan's sense —
normalized per channel by the profile's declared bounds.
"""
from __future__ import annotations

from typing import Any

import numpy as np

LOG10_2 = 0.30102999566398119521
SCENE_MID = 0.18


def layer_log_exposure(rgb_rec2020: Any, observer: Any) -> Any:
    """Per-layer log10 exposure, neutral-anchored (grey ramps map exactly to
    the profile's logE axis). [N,3] scene-linear Rec.2020 -> [N,3] logE."""
    a = np.asarray(observer, dtype=np.float64)
    mid = a @ np.full(3, SCENE_MID)
    e = np.maximum(np.asarray(rgb_rec2020, dtype=np.float64), 1e-9) @ a.T
    return np.log10(np.maximum(e, 1e-12) / np.maximum(mid, 1e-12)[None, :])


def characteristic_amounts(
    log_e: Any,
    le_axis: Any,
    amounts_table: Any,
    ev_offset: float = 0.0,
) -> Any:
    """Three 1-D characteristic curves: logE [N,3] -> dye amounts [N,3].

    ev_offset carries BOTH the reversal exposure anchor and the film exposure
    state (plan §5.2: they share the layer-exposure slot; film_exposure_ev
    enters as ev_offset in EV, converted to log10 here). np.interp clamps to
    the table's ends — the declared out-of-range semantics (film-base black /
    Dmax shoulder are flat beyond the measured domain).
    """
    le = np.asarray(le_axis, dtype=np.float64)
    table = np.asarray(amounts_table, dtype=np.float64)
    x = np.asarray(log_e, dtype=np.float64) + float(ev_offset) * LOG10_2
    return np.stack(
        [np.interp(x[:, c], le, table[:, c]) for c in range(3)], axis=1
    )


def amounts_to_unit(amounts: Any, lo: Any, hi: Any) -> Any:
    """Normalize dye amounts into the Stage B cube's [0,1] coordinates.

    Values outside the declared domain are clamped (the LUT contract's
    declared edge semantics); the CALLER counts and reports clamped samples —
    silent excursion is forbidden (plan §3)."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    span = np.maximum(hi - lo, 1e-12)
    return np.clip((np.asarray(amounts, dtype=np.float64) - lo[None, :]) / span[None, :], 0.0, 1.0)


def out_of_domain_share(amounts: Any, lo: Any, hi: Any, tol: float = 1e-9) -> float:
    """Fraction of samples with any channel outside the declared domain."""
    a = np.asarray(amounts, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)[None, :]
    hi = np.asarray(hi, dtype=np.float64)[None, :]
    outside = (a < lo - tol) | (a > hi + tol)
    return float(np.any(outside, axis=1).mean())


def stage_a_amounts(
    rgb_rec2020: Any,
    observer: Any,
    le_axis: Any,
    amounts_table: Any,
    *,
    film_exposure_ev: float = 0.0,
    anchor_ev_offset: float = 0.0,
) -> Any:
    """The complete Stage A front: scene Rec.2020 -> dye amounts [N,3]."""
    log_e = layer_log_exposure(rgb_rec2020, observer)
    return characteristic_amounts(
        log_e, le_axis, amounts_table,
        ev_offset=float(film_exposure_ev) + float(anchor_ev_offset),
    )
