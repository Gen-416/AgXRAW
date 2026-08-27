# SPDX-License-Identifier: GPL-3.0-or-later
"""Histogram payloads for the realtime preview.

Two histograms, two declared data contracts, each attached to the control it
informs (the project's data-next-to-function rule):

* **Scene EV** (exposure card): the same reliable-sample selection the tone
  planner compiles its percentiles from (:func:`dngscan.tone.
  reliable_scene_ev_selection`), measured on the exporter's full-resolution
  tone-plan sample rows carried by the cache-proxy bundle (R2 item 20; proxy
  pixels only for pre-v14 entries) and viewed through intent exposure. Moving the EV slider moves this population against
  the plan's fixed compiled endpoints, so the user can watch the body cross the
  black/white points the render actually uses.
* **Display code values** (preview card): the exact rendered 1920px u8 frame
  the browser is about to show, before any auto-EV annotation overlay.

Both are 128-bin integer count arrays; the log vertical scale is a display
choice made by the page, so counts stay raw and JSON-safe here.

Exposure-shift identity (why the per-frame cost is one shift + one bincount):
intent gain is exactly ``g0 * 2**user_ev`` with ``g0 = compute_exposure_gain
(mode, 0.0)`` (see scene_scale.scene_scale_contract_from_bundle), so every
sample's EV at slider position ``e`` equals its EV at 0 plus ``e``. The EV0
selection is therefore computed once per proxy/transform combination and only
shifted per frame. The one divergence from a full recompute is the
EV_REPORT_FLOOR clamp: samples pinned to the floor at EV0 stay excluded at
every slider position instead of un-pinning under large positive EV. Those
samples sit below -11.5 EV at EV0, i.e. below -8.5 EV even at +3, under the
noise floor and largely outside the drawn -10..+4 axis — a declared
approximation, not silent error.
"""
from __future__ import annotations

import math
from typing import Any

import dngscan as dg
from dngscan.constants import OUTPUT_REFERENCE_WHITE_STOPS
from dngscan.tone import (
    compute_exposure_gain,
    exposure_mode_for_tone_core,
    reliable_scene_ev_selection,
)

HISTOGRAM_BINS = 128
SCENE_EV_MIN = -10.0
SCENE_EV_MAX = 4.0
# Deterministic pixel stride for the display histogram, in the D10 declared-
# sampling style: ~410k of the ~2.5M proxy pixels keep the whole per-frame
# histogram increment under the 5 ms hot-path budget (measured on the 1920px
# Apple Silicon reference: scene 1.8 ms + display 2.2 ms ≈ 4 ms combined).
DISPLAY_HIST_SAMPLE_TARGET = 450_000
# Truncation offset so ``astype(int)`` (round toward zero) acts as floor for
# every representable EV sample; see _scene_counts.
_TRUNC_OFFSET = 2048.0


def _finite_or_none(value: object) -> float | None:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def scene_ev_base(
    bundle: Any,
    analysis: Any,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
) -> dict[str, Any]:
    """Compile the EV0 (plan-space) reliable population once per proxy state.

    Uses the identical selection and the identical EV0 anchor gain as
    build_render_plan, ignoring whatever intent exposure the bundle currently
    carries — user EV is applied per frame as a pure shift.
    """
    np = dg.np
    gain0 = compute_exposure_gain(exposure_mode_for_tone_core("agx"), 0.0)
    ev, body, _evidence, _evidence_ok = reliable_scene_ev_selection(
        bundle,
        analysis,
        scene_transform,
        scene_transform_strength,
        exposure_gain=gain0,
    )
    ev_body = np.ascontiguousarray(ev[body], dtype=np.float32)
    return {"ev_body": ev_body, "sample_count": int(ev_body.shape[0])}


def _scene_counts(ev: Any) -> Any:
    """128 half-open uniform bins over [SCENE_EV_MIN, SCENE_EV_MAX)."""
    np = dg.np
    scale = np.float32(HISTOGRAM_BINS / (SCENE_EV_MAX - SCENE_EV_MIN))
    # ``astype`` truncates toward zero; the large positive offset makes that a
    # floor for anything above the representable EV floor, then out-of-range
    # bins are clipped into two sacrificial slots and dropped.
    idx = ((ev - np.float32(SCENE_EV_MIN)) * scale + np.float32(_TRUNC_OFFSET)).astype(
        np.int32
    )
    idx -= int(_TRUNC_OFFSET)
    np.clip(idx, -1, HISTOGRAM_BINS, out=idx)
    return np.bincount(idx + 1, minlength=HISTOGRAM_BINS + 2)[1 : HISTOGRAM_BINS + 1]


def scene_ev_histogram(base: dict[str, Any], plan: Any, user_ev: float) -> dict[str, Any]:
    """Per-frame scene EV histogram payload from the cached EV0 population.

    Annotation lines come straight from the compiled plan the render consumes:
    black/white endpoints and the 18%-gray pivot are fixed in intent-EV space,
    while the reliable tail (p99.99) is the plan's own evidence value shifted by
    the user's EV — evidence absent (NaN) is honestly absent, not substituted.
    """
    np = dg.np
    ev = base["ev_body"] + np.float32(user_ev)
    counts = _scene_counts(ev)
    tone = plan.tone
    tail = _finite_or_none(getattr(plan.scene, "reliable_tail_ev_p9999", None))
    return {
        "kind": "scene_ev",
        "bins": HISTOGRAM_BINS,
        "ev_min": SCENE_EV_MIN,
        "ev_max": SCENE_EV_MAX,
        "counts": [int(v) for v in counts],
        "sample_count": int(base["sample_count"]),
        "black_ev": _finite_or_none(tone.black_ev),
        "white_ev": _finite_or_none(tone.white_ev),
        "pivot_ev": 0.0,
        "reliable_tail_ev": None if tail is None else tail + float(user_ev),
    }


def display_histogram(rgb_u8: Any) -> dict[str, Any]:
    """R/G/B + luma 128-bin histograms of the rendered 1920px u8 frame.

    Bins pair adjacent code values ([0,1], [2,3], .. [254,255]). Luma is the
    integer Rec.709 weighting of the *encoded* code values ((54R+183G+19B)/256)
    — the conventional display-referred luma histogram, deliberately not a
    linearized scene quantity (the scene histogram owns that role).
    """
    np = dg.np
    flat = np.asarray(rgb_u8, dtype=np.uint8).reshape(-1, 3)
    step = max(1, math.ceil(flat.shape[0] / DISPLAY_HIST_SAMPLE_TARGET))
    if step > 1:
        flat = flat[::step]
    channels = np.ascontiguousarray(flat.T)
    payload: dict[str, Any] = {
        "kind": "display",
        "bins": HISTOGRAM_BINS,
        "sample_count": int(flat.shape[0]),
    }
    for name, row in zip(("r", "g", "b"), channels):
        counts = np.bincount(row, minlength=256).reshape(HISTOGRAM_BINS, 2).sum(axis=1)
        payload[name] = [int(v) for v in counts]
    # 54+183+19 == 256, so the weighted sum fits uint16 (max 65280) and >>9
    # lands exactly in 0..127 — one shift covers both /256 and the bin pairing.
    luma = (
        np.uint16(54) * channels[0].astype(np.uint16)
        + np.uint16(183) * channels[1]
        + np.uint16(19) * channels[2]
    ) >> 9
    payload["luma"] = [
        int(v) for v in np.bincount(luma, minlength=HISTOGRAM_BINS)[:HISTOGRAM_BINS]
    ]
    return payload


def hdr_earned_ev(plan: Any) -> float | None:
    """Scene-earned HDR headroom above diffuse white, from the compiled plan.

    Same definition as service.detected_scene_params: reliable tail p99.99 minus
    the output reference white. None whenever the evidence tail is absent — the
    page then draws nothing rather than inventing a number.
    """
    tail = _finite_or_none(getattr(plan.scene, "reliable_tail_ev_p9999", None))
    if tail is None:
        return None
    return max(0.0, tail - float(OUTPUT_REFERENCE_WHITE_STOPS))
