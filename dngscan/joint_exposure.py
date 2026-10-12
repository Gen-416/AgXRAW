# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen, bounded AgX exposure/white-endpoint/shoulder selection.

Promoted from the 2026-10-12 comparison after user acceptance. Luminance
populations constrain response changes; they do not recognize night scenes,
subjects or light sources. A fallback is not a claim that the baseline passes.
"""
from dataclasses import replace

from ._deps import np
from .color import RGB_TO_XYZ, output_gamut_space
from .drt import apply_c1_endpoints, curve_params_from_plan
from .sampling import sample_indices

JOINT_POLICY_ID = "bounded-agx-2026-10-12-v1"
JOINT_MAX_EXTRA_EV = 0.25
JOINT_REMAINING_FRACTION = 0.10
JOINT_EV_CANDIDATES = 5
JOINT_SHOULDER_REDUCTION = 1.5
JOINT_SHOULDER_MIN = 0.7
JOINT_SHOULDER_CANDIDATES = 9
JOINT_MIN_POPULATION = 64
JOINT_SHADOW_MAX_LIFT_EV = 0.35
JOINT_DARK_MAX_DROP_EV = 0.15
JOINT_SEPARATION_MAX_DROP_EV = 0.35
JOINT_POPULATION_FLOOR = 0.0001
JOINT_SHADOW_CEILING = 0.03
JOINT_DARK_CEILING = 0.005
JOINT_BRIGHT_FLOOR = 0.5


def select_joint_plan(base, anchor, body_target, gamut, sample_count, probe):
    """Return (plan, EV, scalar/list diagnostics), using canonical source rows.

    ``probe(plan, ev, indices)`` must render the same rows, source masks and
    declared colour operations as the real render. The caller owns those
    buffers; no arrays or closures escape into the returned diagnostics.
    Manual controls already in ``base`` remain in force, including brightness.
    """
    from .auto_ev import (
        NEAR_WHITE_LINEAR, output_highlight_stats, _highlight_margin_from_stats,
    )

    target = anchor + min(JOINT_MAX_EXTRA_EV,
                          JOINT_REMAINING_FRACTION * max(0., body_target - anchor))
    diagnostic = {"policy": JOINT_POLICY_ID, "anchor_ev": float(anchor),
                  "soft_target_ev": float(target), "canonical_samples": sample_count,
                  "status": "identity", "selected": None, "candidates": []}
    if anchor == 0. and body_target == 0.:
        diagnostic["reason"] = "zero-target identity; no gate-pass claim"
        return base, float(anchor), diagnostic

    small = sample_indices(sample_count, max_samples=220_000)
    weights = RGB_TO_XYZ[output_gamut_space(gamut)][1].astype(np.float64)
    zero = np.asarray(probe(base, 0., None), dtype=np.float64)
    gate = np.max(np.clip(zero, 0., 1.), axis=1) < NEAR_WHITE_LINEAR
    baseline = output_highlight_stats(zero, gamut, gate)
    small_baseline = output_highlight_stats(zero[small], gamut, gate[small])
    reference_y = np.asarray(probe(base, anchor, None), dtype=np.float64) @ weights
    populations = {
        "shadow": (reference_y >= JOINT_POPULATION_FLOOR) & (reference_y < JOINT_SHADOW_CEILING),
        "dark": (reference_y >= JOINT_POPULATION_FLOOR) & (reference_y < JOINT_DARK_CEILING),
        "bright": reference_y >= JOINT_BRIGHT_FLOOR,
    }

    def relations(value, indices):
        y = value @ weights
        ref = reference_y if indices is None else reference_y[indices]
        logs = np.log2(np.maximum(y, 1e-10) / np.maximum(ref, 1e-10))
        result = {}
        for name, mask in populations.items():
            if indices is not None:
                mask = mask[indices]
            count = int(np.count_nonzero(mask))
            result[name + "_count"] = count
            result[name + "_delta_EV_p05_p50_p95"] = (
                np.percentile(logs[mask], [5, 50, 95]).tolist()
                if count >= JOINT_MIN_POPULATION else None
            )
        shadow = result["shadow_delta_EV_p05_p50_p95"]
        dark = result["dark_delta_EV_p05_p50_p95"]
        bright = result["bright_delta_EV_p05_p50_p95"]
        separation = bright[1] - shadow[1] if bright is not None and shadow is not None else None
        result["separation_change_EV"] = separation
        result["pass"] = bool(
            (shadow is None or shadow[1] <= JOINT_SHADOW_MAX_LIFT_EV)
            and (dark is None or dark[1] >= -JOINT_DARK_MAX_DROP_EV)
            and (separation is None or separation >= -JOINT_SEPARATION_MAX_DROP_EV)
        )
        return result

    powers = np.linspace(base.tone.shoulder_power,
                         max(JOINT_SHOULDER_MIN, base.tone.shoulder_power - JOINT_SHOULDER_REDUCTION),
                         JOINT_SHOULDER_CANDIDATES)
    # Descending EV, least shoulder change first. No independent shadow fit:
    # that experiment was diagnostic and never affected the accepted variant.
    for ev in np.linspace(target, anchor, JOINT_EV_CANDIDATES):
        for power in powers:
            tone = replace(base.tone, white_ev=base.tone.white_ev + float(ev),
                           shoulder_power=float(power))
            tone = replace(tone, dynamic_range_ev=tone.white_ev - tone.black_ev)
            plan = replace(base, tone=tone)
            params = curve_params_from_plan(tone)
            ramp = apply_c1_endpoints(np.linspace(tone.black_ev, tone.white_ev, 4097, dtype=np.float32), tone)
            legal = bool(np.isfinite(ramp).all() and np.diff(ramp).min() >= -2e-6
                         and not params["need_concave_shoulder"])
            record = {"ev": float(ev), "white_ev": tone.white_ev,
                      "black_ev": tone.black_ev, "shoulder": float(power),
                      "toe": tone.toe_power, "legal": legal, "pass": False}
            diagnostic["candidates"].append(record)
            if not legal:
                continue
            value = np.asarray(probe(plan, float(ev), small), dtype=np.float64)
            margin = _highlight_margin_from_stats(output_highlight_stats(value, gamut, gate[small]), small_baseline)
            rel = relations(value, small)
            record.update(proposal_margin=margin, proposal_relations=rel)
            if margin < 0. or not rel["pass"]:
                continue
            value = np.asarray(probe(plan, float(ev), None), dtype=np.float64)
            stats = output_highlight_stats(value, gamut, gate)
            margin = _highlight_margin_from_stats(stats, baseline)
            rel = relations(value, None)
            record.update(canonical_margin=margin, canonical_stats=stats,
                          canonical_relations=rel, canonical_samples=sample_count)
            record["pass"] = bool(margin >= 0. and rel["pass"])
            if record["pass"]:
                diagnostic.update(status="selected", reason="least shoulder softening at chosen EV", selected=record)
                return plan, float(ev), diagnostic
    diagnostic.update(status="fallback", reason="no constrained candidate passes; neutral anchor retained")
    return base, float(anchor), diagnostic
