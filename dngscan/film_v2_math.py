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

# The declared editorial developer domain (plan §6). SINGLE SOURCE: the plan
# validator (film_plans.validate_film_plans), the runtime guard in
# developer_perturbation below and the asset builder's shaper-domain bake
# (tools/build_film_v2_assets.py) all read these — the baked amount_lo/hi
# cover EXACTLY this envelope, so any wider recipe would clip silently at the
# Stage B rails (mainline A2 measured up to 12% excursion at contrast +1,
# density +1, 230/743 probe samples clamped).
EDITORIAL_CONTRAST_LIMIT = 0.5
EDITORIAL_DENSITY_LIMIT = 0.5
EDITORIAL_FOG_MAX = 0.3


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


def developer_perturbation(
    char_le: Any,
    char_amounts: Any,
    *,
    contrast_delta: float = 0.0,
    fog_delta: float = 0.0,
    color_density: float = 0.0,
) -> Any:
    """editorial_custom developer recipe (plan §6): a BOUNDED perturbation of
    the three characteristic curves, applied analytically at Stage A — no
    volume rebuild, because the curves never left the analytic front.

        H'_c(x) = mid_c + (1 + color_density) * (H_c(a*(x-x0)+x0) - mid_c)
                  + fog_delta

    with a = 1 + contrast_delta and x0 the logE of scene mid-grey (0). The
    mid-grey amount is unchanged by CONSTRUCTION for the contrast and
    colour-density terms (both scale about the anchor); fog is a uniform
    density addition and deliberately moves everything including mid — that
    is what chemical fog does, and hiding it would be a lie. Monotonicity is
    preserved for a > 0 and any color_density > -1. Amounts clamp at zero
    (density cannot be negative); provenance: editorial, never measured.

    The declared editorial domain is enforced HERE, not only at plan
    validation: the baked shaper domains cover exactly this envelope, so a
    caller that bypasses validate_film_plans (mainline A2 probed contrast +1,
    density +1 through a raw plan object) would otherwise push densities past
    the Stage B rails and have amounts_to_unit clip them silently — the exact
    excursion class that function's contract forbids. Fail closed instead.
    """
    le = np.asarray(char_le, dtype=np.float64)
    table = np.asarray(char_amounts, dtype=np.float64)
    if not abs(float(contrast_delta)) <= EDITORIAL_CONTRAST_LIMIT:
        raise ValueError(
            f"显影对比扰动域为 [-{EDITORIAL_CONTRAST_LIMIT}, "
            f"{EDITORIAL_CONTRAST_LIMIT}]（收到 {float(contrast_delta)}；"
            "烘焙 shaper 域只覆盖声明包络,超域会在 Stage B 轨道静默裁剪）"
        )
    if not abs(float(color_density)) <= EDITORIAL_DENSITY_LIMIT:
        raise ValueError(
            f"显影色密度扰动域为 [-{EDITORIAL_DENSITY_LIMIT}, "
            f"{EDITORIAL_DENSITY_LIMIT}]（收到 {float(color_density)}）"
        )
    if not 0.0 <= float(fog_delta) <= EDITORIAL_FOG_MAX:
        raise ValueError(
            f"显影 fog 域为 [0, {EDITORIAL_FOG_MAX}]（收到 {float(fog_delta)}）"
        )
    a = 1.0 + float(contrast_delta)
    cd = 1.0 + float(color_density)
    if a == 1.0 and cd == 1.0 and float(fog_delta) == 0.0:
        return table
    x0 = 0.0  # scene mid-grey on the neutral-anchored logE axis
    warped_x = a * (le - x0) + x0
    out = np.empty_like(table)
    for c in range(3):
        mid = float(np.interp(x0, le, table[:, c]))
        resampled = np.interp(warped_x, le, table[:, c])
        out[:, c] = mid + cd * (resampled - mid) + float(fog_delta)
    return np.maximum(out, 0.0)


def film_compression_ev(
    rgb_rec2020: Any,
    *,
    impact: float,
    knee_ev: float,
    width_ev: float = 2.0,
    highlight_color_density: float = 0.0,
) -> Any:
    """Film Compression (plan §8): an OPTIONAL editorial bridge from the
    sensor's hard highlight distribution toward negative latitude, applied to
    scene-linear Rec.2020 BEFORE the film exposure model. C1 at the knee:

        x_f = k + w*(1 - exp(-(x-k)/w))   for x > k, identity below
        x'  = (1-impact)*x + impact*x_f

    operating on the LUMINANCE EV so hue never rotates; the compression
    amount d = x - x' additionally drives highlight colour density
    C' = C * exp(-rho*d) as a chroma scale toward the (luminance-preserved)
    neutral — no per-channel clamps. impact = 0 is a strict identity (the
    caller keeps the fast path). CFA-clipped channels were already handled
    by decode-side reconstruction; this operator claims nothing about lost
    information.
    """
    rgb = np.asarray(rgb_rec2020, dtype=np.float64)
    impact = float(impact)
    if impact <= 0.0:
        return rgb
    luma = np.array([0.2627, 0.6780, 0.0593])
    y = np.maximum(rgb @ luma, 1e-9)
    x = np.log2(y / SCENE_MID)
    k = float(knee_ev)
    w = max(float(width_ev), 1e-3)
    over = x > k
    x_f = np.where(over, k + w * (1.0 - np.exp(-(x - k) / w)), x)
    x_new = (1.0 - impact) * x + impact * x_f
    gain = np.exp2(x_new - x)
    out = rgb * gain[:, None]
    d = np.maximum(x - x_new, 0.0)
    rho = float(highlight_color_density)
    if rho > 0.0:
        y_new = np.maximum(out @ luma, 1e-9)
        chroma_scale = np.exp(-rho * d)[:, None]
        out = y_new[:, None] * ((out / y_new[:, None] - 1.0) * chroma_scale + 1.0)
    return np.maximum(out, 0.0)


def film_hdr_gain_log2(
    scene_ev: Any,
    *,
    headroom_ev: float,
    join_ev: float,
    span_ev: float,
) -> Any:
    """P6 (plan §10) "胶片印相 + scene HDR 扩展" gain, in log2 units.

    Zero (gain 1) at and below the print's reference-white join, rising along
    a smoothstep of the SCENE's own highlight EV to the solved reliable
    headroom. Smoothstep is C1 at both ends: the SDR body below the join is
    untouched exactly, the ceiling is approached without a slope break, gain
    never drops below 1 and never exceeds the headroom — the §10 acceptance
    set by construction. This scales the film print's display-linear output;
    it is a scene-highlight extension, NOT a claim of physical film HDR.
    """
    ev = np.asarray(scene_ev, dtype=np.float64)
    h = max(float(headroom_ev), 0.0)
    span = max(float(span_ev), 1e-6)
    t = np.clip((ev - float(join_ev)) / span, 0.0, 1.0)
    return h * t * t * (3.0 - 2.0 * t)
