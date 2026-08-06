# SPDX-License-Identifier: GPL-3.0-or-later
"""Film-takeover development core (film_mode="full") — spectral-LUT edition.

The two-mode contract (docs/FILM_OBSERVATION_PLAN): in "observe" mode the film
declares what the observer saw and AgX develops it. This module is the other
pole — and since the stage-4 rebuild it no longer feeds Rec.2020 channels to
per-channel curves (the RGB heuristic the review refused to call a
reconstruction). It samples an offline-baked LUT of the honest chain:

    post-prefeed Rec.2020 -> constrained observer inverse (fitted over the
    rawtoaces training reflectances under D55) -> per-layer exposures ->
    characteristic curves -> negative spectral density -> TH-KG3 print chain
    (or the slide viewed directly) -> XYZ -> CAT -> Rec.2020.

Honesty label: this is a TRISTIMULUS reconstruction CONSTRAINED by spectral
data — three numbers cannot recover a spectrum, and the observer inverse's
metamer residual is measured and stamped into every LUT (observer_p99_stop).
DIR couplers / interlayer effects remain absent from the data and therefore
from the chain. The plan.film_crossover switch selects how the neutral axis
is served: "datasheet" is the baked chain verbatim; "neutralized" divides the
sampled output per pixel by a BOUNDED neutral-cast curve shipped inside the
npz, indexed at the pixel's LUMINANCE exposure EV_Y (the same single-axis
declaration as the colour head — a per-channel-exposure divisor re-imported
the retired channels-as-layer-exposures reading and blew up off-axis on hard
reversals). Bounded means the luma-normalized chroma correction is clipped to
[0.25, 4] per channel and then luma-renormalized: grays are strictly neutral
wherever the medium's own gray sits within two stops of neutral per channel,
tone follows the chain everywhere, and deeper tint — Kodachrome's floor above
all — is kept as medium character rather than chased with near-singular gains.
Why the division is at runtime and not a second baked volume: with a bounded
divisor evaluated exactly per pixel, the quotient's visible-stop error equals
the datasheet volume's own; baking the composite instead put the EV_Y kink
diagonal to the grid and measured 1.73 EV worst off-axis (full argument at
the cast_b computation in tools/build_full_lut.py).
SDR only; the enlarger colour head is REFUSED in full mode — appending a
neutral-axis LMS field to a baked chain would contradict the chain's own
physics, so the plan compiler, CLI and GUI reject the combination until
filtration is itself baked into LUT variants.

The LUT grid lives in per-channel log2 exposure,

    u_c = (log2(E_c/0.18) - ev_min) / (ev_max - ev_min),

sampled with tetrahedral interpolation; outside the domain u clamps (beyond
the top the print sits on Dmin/Dmax, below the bottom is film-base black).
"""
from __future__ import annotations

from pathlib import Path as _Path
from typing import Any

from ._deps import np
from .color import EPS

REC2020_LUMA = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float32)
_LUT_DIR = _Path(__file__).with_name("data") / "full_lut"
_LUT_CACHE: dict[str, tuple | None] = {}


def _load_lut(name: str):
    key = str(name)
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
    path = _LUT_DIR / f"{key}.npz"
    entry = None
    try:
        with np.load(path, allow_pickle=False) as payload:
            lut = np.asarray(payload["lut_datasheet"], dtype=np.float32)
            n = int(payload["n"])
            entry = (
                lut,
                np.asarray(payload["cast_ev"], dtype=np.float32),
                np.asarray(payload["cast_bounded"], dtype=np.float32),
                float(payload["ev_min"]),
                float(payload["ev_max"]),
                n,
            )
            if lut.shape != (n, n, n, 3):
                entry = None
    except (OSError, KeyError, ValueError):
        entry = None
    if entry is None:
        raise RuntimeError(
            f"film-takeover LUT for '{key}' is missing or unreadable at {path}; "
            "regenerate with tools/build_full_lut.py"
        )
    _LUT_CACHE[key] = entry
    return entry


def _tetrahedral(lut: Any, u: Any, n: int) -> Any:
    """Vectorized tetrahedral interpolation on a cubic lattice. [N,3] -> [N,3]."""
    g = np.clip(u, 0.0, 1.0) * (n - 1)
    i0 = np.minimum(g.astype(np.int32), n - 2)
    f = (g - i0).astype(np.float32)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

    def at(dx: Any, dy: Any, dz: Any) -> Any:
        return lut[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]

    c000 = at(0, 0, 0)
    c111 = at(1, 1, 1)
    out = np.empty_like(c000)
    # Six tetrahedra of the unit cube, keyed by the ordering of (fx, fy, fz).
    orders = (
        (fx >= fy) & (fy >= fz),
        (fx >= fz) & (fz > fy),
        (fz > fx) & (fx >= fy),
        (fy > fx) & (fx >= fz),
        (fy >= fz) & (fz > fx),
        (fz > fy) & (fy > fx),
    )
    corners = (
        ((1, 0, 0), (1, 1, 0)),
        ((1, 0, 0), (1, 0, 1)),
        ((0, 0, 1), (1, 0, 1)),
        ((0, 1, 0), (1, 1, 0)),
        ((0, 1, 0), (0, 1, 1)),
        ((0, 0, 1), (0, 1, 1)),
    )
    axis_f = {"x": fx, "y": fy, "z": fz}
    weights = (
        ("x", "y", "z"),
        ("x", "z", "y"),
        ("z", "x", "y"),
        ("y", "x", "z"),
        ("y", "z", "x"),
        ("z", "y", "x"),
    )
    for mask, (cA, cB), (a1, a2, a3) in zip(orders, corners, weights):
        if not bool(np.any(mask)):
            continue
        f1, f2, f3 = axis_f[a1][mask], axis_f[a2][mask], axis_f[a3][mask]
        idx = np.nonzero(mask)[0]
        pA = lut[i0[idx, 0] + cA[0], i0[idx, 1] + cA[1], i0[idx, 2] + cA[2]]
        pB = lut[i0[idx, 0] + cB[0], i0[idx, 1] + cB[1], i0[idx, 2] + cB[2]]
        out[idx] = (
            (1.0 - f1)[:, None] * c000[idx]
            + (f1 - f2)[:, None] * pA
            + (f2 - f3)[:, None] * pB
            + f3[:, None] * c111[idx]
        )
    return out


def apply_film_core(rgb_rec2020: Any, plan: Any) -> Any:
    """Film-takeover development: sample the baked spectral chain. [N,3]->[N,3]."""
    preset = str(getattr(plan, "curve_preset", "") or "")
    lut, cast_ev, cast_bounded, ev_min, ev_max, n = _load_lut(preset)
    rgb = np.maximum(np.asarray(rgb_rec2020, dtype=np.float32), 0.0)
    ev = np.log2(np.maximum(rgb / np.float32(0.18), EPS))
    u = (ev - ev_min) / (ev_max - ev_min)
    developed = _tetrahedral(lut, u, n)
    if str(getattr(plan, "film_crossover", "off")) != "datasheet":
        # Neutralized variant: per-pixel division by the bounded cast at the
        # pixel's luminance exposure. The curve is sampled on the LUT's own
        # axis and interpolated linearly, so on the neutral axis the quotient
        # is a mediant of node-exact values (no overshoot), and off-axis the
        # quotient's visible-stop error equals the datasheet volume's own —
        # see the architecture note in tools/build_full_lut.py.
        ev_y = np.log2(np.maximum(rgb @ REC2020_LUMA, EPS) / np.float32(0.18))
        for c in range(3):
            developed[:, c] /= np.interp(ev_y, cast_ev, cast_bounded[:, c])
    return np.maximum(developed, 0.0).astype(np.float32, copy=False)
