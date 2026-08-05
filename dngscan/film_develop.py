# SPDX-License-Identifier: GPL-3.0-or-later
"""Film-takeover development core (film_mode="full") — EXPERIMENTAL.

The two-mode contract (docs/FILM_OBSERVATION_PLAN): in "observe" mode the film
declares what the observer saw (WB, separation, tone signature) and AgX develops
it — colour rendering stays with the pipeline's own validated preferred rendering.
This module is the other pole: the film's development model takes over and AgX
yields everything but delivery-side gamut safety.

What "the film's development model" means with today's data: the fitted scalar
curve applied PER CHANNEL at each channel's own exposure, times the measured
per-channel ratio field r_c(EV_c)/r_c(EV_Y) — i.e. the medium's per-channel
transfer as reconstructed by the spectral print fitter. No AgX inset/outset, no
hue restore, no punch: colour behaviour comes (only) from the film data.

Honesty label: the tone dimension of that data is externally validated (DiVERE
cross-check); the COLOUR dimension is an unanchored reconstruction with no
external oracle yet — which is why this core is opt-in and marked experimental,
and why "observe" is the default. SDR only for now: the HDR formation's colour
machinery is AgX-owned and has no film-takeover counterpart.
"""
from __future__ import annotations

from typing import Any

from ._deps import np
from .color import EPS

REC2020_LUMA = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float32)


def apply_film_core(rgb_rec2020: Any, plan: Any) -> Any:
    """Per-channel film development of scene-linear Rec.2020. [N,3] -> [N,3]."""
    from . import agx as agx_engine
    from .drt import apply_c1_endpoints, curve_params_from_plan

    rgb = np.maximum(np.asarray(rgb_rec2020, dtype=np.float32), 0.0)
    params = curve_params_from_plan(plan)
    ev = np.log2(np.maximum(rgb / np.float32(0.18), EPS))
    developed = apply_c1_endpoints(ev, plan, params=params)
    # Combined film gain: the measured ratio field (this mode's per-channel
    # transfer) times the user's declared enlarger colour-head filtration — the
    # same composition as the observe-mode formation (agx.film_channel_gain).
    # The plan.film_crossover anchor lives inside channel_ratio_gain (see its docstring).
    gain = agx_engine.film_channel_gain(rgb, plan, REC2020_LUMA)
    if gain is not None:
        developed = developed * gain
    return np.maximum(developed, 0.0).astype(np.float32, copy=False)
