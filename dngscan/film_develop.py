# SPDX-License-Identifier: GPL-3.0-or-later
"""Film-takeover development core (film_mode="full") — two-stage edition.

film v2 (FILM_PRINT_RENDERING_PLAN §3): the default backend is the TWO-STAGE
composite — Stage A runs the analytic scene->density front per pixel
(observer inverse -> three 1-D characteristic curves -> dye amounts, with the
film exposure state and the reversal anchor sharing the layer-exposure slot),
Stage B samples a density-domain 65^3 volume of the same solved chain (fixed
timing q(0)). Halation and grain will insert between A and B in later stages;
their off-state leaves this path bit-identical. The v1 single scene-EV LUT
remains available as the HIDDEN LEGACY TEST BACKEND
(DNGSCAN_FILM_LEGACY_LUT=1) whose bytes the P0 freeze pins; v2 validates
against the direct-chain oracle shipped inside every schema-5 asset instead
(plan §7.2 migration semantics).

The v1 narrative below still describes the shared chain and the
neutralization contract; only the sampling topology changed.


The two-mode contract (docs/FILM_OBSERVATION_PLAN): in "observe" mode the film
declares what the observer saw and AgX develops it. This module is the other
pole — and since the stage-4 rebuild it no longer feeds Rec.2020 channels to
per-channel curves (the RGB heuristic the review refused to call a
reconstruction). It samples an offline-baked LUT of the honest chain:

    plain scene Rec.2020 -> constrained observer inverse (fitted over the
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
reversals). Bounded means the correction multiplier walks the straight line
h(t) = 1 + t*(1/cast - 1) from identity toward full neutralization, at the
largest t in [0,1] keeping every channel inside [0.25, 4] — every point on
that line preserves the neutral axis' luminance exactly, so grays are
strictly neutral wherever the medium's own gray sits within two stops of
neutral per channel, tone follows the chain everywhere, and deeper tint —
Kodachrome's floor above all — is kept as medium character rather than
half-chased with near-singular gains (a clip-then-renormalize cut was
measured shipping a 0.081 divisor, +3.6 EV, outside its own claimed bound).
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
            cast_ev = np.asarray(payload["cast_ev"], dtype=np.float32)
            cast = np.asarray(payload["cast_bounded"], dtype=np.float32)
            # Hard loading contract (review batch 7): schema, declared input
            # space and value sanity fail CLOSED — a stale or corrupted LUT
            # must never be silently sampled.
            if int(payload["schema"]) != 3:
                raise ValueError(f"full-LUT schema {int(payload['schema'])}, expected 3")
            input_space = str(np.asarray(payload["input_space"]))
            if input_space != "scene_rec2020":
                raise ValueError(f"full-LUT input_space {input_space!r}")
            if not bool(np.isfinite(lut).all()) or float(lut.min()) < 0.0:
                raise ValueError("full-LUT volume contains non-finite or negative values")
            if not bool(np.isfinite(cast).all()) or \
                    float(cast.min()) < 0.25 - 1e-4 or float(cast.max()) > 4.0 + 1e-4:
                raise ValueError("bounded cast curve outside its declared [0.25, 4] bound")
            # Structural contract (review batch 8): a structurally broken asset
            # must fail HERE, not deep inside interpolation or normalization.
            ev_min_v = float(payload["ev_min"])
            ev_max_v = float(payload["ev_max"])
            if n < 2:
                raise ValueError(f"full-LUT grid n={n} < 2")
            if not (np.isfinite(ev_min_v) and np.isfinite(ev_max_v)) or \
                    not ev_max_v > ev_min_v:
                raise ValueError(f"full-LUT EV domain [{ev_min_v}, {ev_max_v}] is degenerate")
            if cast_ev.ndim != 1 or cast.ndim != 2 or cast.shape != (cast_ev.size, 3):
                raise ValueError("cast curve arrays are mis-shaped")
            if cast_ev.size < 2 or not bool(np.all(np.diff(cast_ev) > 0)):
                raise ValueError("cast_ev axis is not strictly increasing")
            if not bool(np.isfinite(cast_ev).all()) or \
                    abs(float(cast_ev[0]) - ev_min_v) > 1e-3 or \
                    abs(float(cast_ev[-1]) - ev_max_v) > 1e-3:
                raise ValueError(
                    "cast_ev axis does not span the LUT's declared EV domain"
                )
            entry = (
                lut,
                cast_ev,
                cast,
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


_V2_DIR = _Path(__file__).with_name("data") / "film_v2"
_V2_CACHE: dict[str, tuple | None] = {}


def _use_legacy_backend() -> bool:
    """Hidden legacy test backend (plan §7.2): the v1 single scene-EV LUT,
    kept solely so the P0 freeze can pin its bytes. Never a user surface."""
    import os

    return os.environ.get("DNGSCAN_FILM_LEGACY_LUT") == "1"


def _npz(path):
    return np.load(path, allow_pickle=False)


def _check_common(z, kind: str, path) -> None:
    if int(z["schema"]) != 5:
        raise ValueError(f"{path.name}: schema {int(z['schema'])}, expected 5")
    if str(np.asarray(z["kind"])) != kind:
        raise ValueError(f"{path.name}: kind {np.asarray(z['kind'])}, expected {kind}")


def _load_v2(name: str):
    """Assemble the modular §7.1 asset family for one preset, fail closed.

    Returns (stock dict, {medium_id: (print_state dict | None, b2 dict)}).
    """
    key = str(name)
    if key in _V2_CACHE:
        return _V2_CACHE[key]
    stock_path = _V2_DIR / f"{key}.npz"
    entry = None
    try:
        with _npz(stock_path) as z:
            _check_common(z, "stock", stock_path)
            observer = np.asarray(z["observer"], dtype=np.float64)
            char_le = np.asarray(z["char_le"], dtype=np.float64)
            char_amounts = np.asarray(z["char_amounts"], dtype=np.float64)
            lo = np.asarray(z["amount_lo"], dtype=np.float64)
            hi = np.asarray(z["amount_hi"], dtype=np.float64)
            if observer.shape != (3, 3) or not bool(np.isfinite(observer).all()):
                raise ValueError("observer mis-shaped")
            if char_amounts.shape != (char_le.size, 3) or char_le.size < 2:
                raise ValueError("characteristic tables mis-shaped")
            if not bool(np.all(np.diff(char_le) > 0)):
                raise ValueError("logE axis not strictly increasing")
            if not bool(np.all(hi > lo)):
                raise ValueError("amount domain degenerate")
            stock = {
                "observer": observer,
                "char_le": char_le,
                "char_amounts": char_amounts,
                "lo": lo,
                "hi": hi,
                "anchor": float(z["anchor_ev_offset"]),
                "exp_lo": float(z["exposure_ev_min"]),
                "exp_hi": float(z["exposure_ev_max"]),
                "reversal": bool(z["reversal"]),
                "default_medium": str(np.asarray(z["default_medium"])),
                "media": [str(m) for m in z["media"]],
            }
            if stock["reversal"]:
                cast_ev = np.asarray(z["cast_ev"], dtype=np.float32)
                cast = np.asarray(z["cast_bounded"], dtype=np.float32)
                if cast.shape != (cast_ev.size, 3) or cast_ev.size < 2 or \
                        not bool(np.all(np.diff(cast_ev) > 0)):
                    raise ValueError("reversal cast arrays mis-shaped")
                if not bool(np.isfinite(cast).all()) or \
                        float(cast.min()) < 0.25 - 1e-4 or float(cast.max()) > 4.0 + 1e-4:
                    raise ValueError("reversal cast outside its declared bound")
                stock["cast_ev"] = cast_ev
                stock["cast_bounded"] = cast
            if not stock["exp_hi"] > stock["exp_lo"]:
                raise ValueError("exposure domain degenerate")
        media = {}
        for medium in stock["media"]:
            b2_path = _V2_DIR / f"b2__{medium}.npz"
            with _npz(b2_path) as z:
                _check_common(z, "b2", b2_path)
                n = int(z["n"])
                vol = np.asarray(z["volume"], dtype=np.float32)
                if vol.shape != (n, n, n, 3) or n < 2:
                    raise ValueError(f"{b2_path.name}: volume mis-shaped")
                if not bool(np.isfinite(vol).all()) or float(vol.min()) < 0.0:
                    raise ValueError(f"{b2_path.name}: volume non-finite/negative")
                dye_lo = np.asarray(z["dye_lo"], dtype=np.float64)
                dye_hi = np.asarray(z["dye_hi"], dtype=np.float64)
                if not bool(np.all(dye_hi > dye_lo)):
                    raise ValueError(f"{b2_path.name}: dye domain degenerate")
                b2 = {
                    "volume": vol, "n": n,
                    "paper_le2": np.asarray(z["paper_le2"], dtype=np.float64),
                    "paper_amounts": np.asarray(z["paper_amounts"], dtype=np.float64),
                    "dye_lo": dye_lo, "dye_hi": dye_hi,
                }
            ps = None
            if not stock["reversal"]:
                ps_path = _V2_DIR / f"print__{key}__{medium}.npz"
                with _npz(ps_path) as z:
                    _check_common(z, "print_state", ps_path)
                    n = int(z["n"])
                    b1 = np.asarray(z["b1_volume"], dtype=np.float32)
                    if b1.shape != (n, n, n, 3):
                        raise ValueError(f"{ps_path.name}: B1 mis-shaped")
                    tau_nodes = np.asarray(z["tau_nodes"], dtype=np.float64)
                    tau = np.asarray(z["tau"], dtype=np.float64)
                    if tau.shape != (tau_nodes.size, 3) or tau_nodes.size < 2 or \
                            not bool(np.all(np.diff(tau_nodes) > 0)):
                        raise ValueError(f"{ps_path.name}: tau table mis-shaped")
                    cast_ev = np.asarray(z["cast_ev"], dtype=np.float32)
                    casts = np.asarray(z["casts"], dtype=np.float32)
                    if casts.shape != (tau_nodes.size, cast_ev.size, 3):
                        raise ValueError(f"{ps_path.name}: casts mis-shaped")
                    if not bool(np.isfinite(casts).all()) or \
                            float(casts.min()) < 0.25 - 1e-4 or \
                            float(casts.max()) > 4.0 + 1e-4:
                        raise ValueError(f"{ps_path.name}: cast outside bound")
                    ps = {
                        "b1": b1, "n": n, "tau_nodes": tau_nodes, "tau": tau,
                        "cast_ev": cast_ev, "casts": casts,
                        "retimed_ev_min": float(z["retimed_ev_min"]),
                        "retimed_ev_max": float(z["retimed_ev_max"]),
                    }
            media[medium] = (ps, b2)
        entry = (stock, media)
    except (OSError, KeyError, ValueError):
        entry = None
    if entry is None:
        raise RuntimeError(
            f"film v2 asset family for '{key}' is missing or unreadable under "
            f"{_V2_DIR}; regenerate with tools/build_film_v2_assets.py"
        )
    _V2_CACHE[key] = entry
    return entry


# Colour-head CC -> per-layer tau attenuation, valid ONLY inside the current
# paper-layer exposure model (§7.2, modelled): a Y filter of d = CC/100
# optical density attenuates the blue-sensitive layer's exposure by 10^-d,
# i.e. delta_tau = -d / log10(2) log2 units on that layer. Real filter
# spectra would change B1's integral density-dependently and must rebuild it.
_CC_LAYER = {"y": 2, "m": 1}
_LOG10_2 = 0.30102999566398119521


def _custom_delta_tau(color_head_y: float, color_head_m: float,
                      print_exposure_ev: float) -> Any:
    delta = np.full(3, float(print_exposure_ev), dtype=np.float64)
    delta[_CC_LAYER["y"]] -= (float(color_head_y) / 100.0) / _LOG10_2
    delta[_CC_LAYER["m"]] -= (float(color_head_m) / 100.0) / _LOG10_2
    return delta


def _apply_film_core_v2(rgb: Any, plan: Any, preset: str) -> Any:
    from .film_v2_math import amounts_to_unit, stage_a_amounts

    stock, media = _load_v2(preset)
    exposure_ev = float(getattr(plan, "film_exposure_ev", 0.0) or 0.0)
    timing = str(getattr(plan, "film_print_timing", "fixed") or "fixed")
    medium = str(getattr(plan, "film_print_medium", "") or "") or stock["default_medium"]
    if medium not in media:
        raise ValueError(
            f"'{preset}' 未烘焙印相介质 '{medium}'（可用：{'/'.join(stock['media'])}）"
        )
    if not stock["exp_lo"] <= exposure_ev <= stock["exp_hi"]:
        raise ValueError(
            f"film_exposure_ev={exposure_ev} 超出 '{preset}' 资产声明域 "
            f"[{stock['exp_lo']}, {stock['exp_hi']}]"
        )
    ps, b2 = media[medium]
    amounts = stage_a_amounts(
        rgb, stock["observer"], stock["char_le"], stock["char_amounts"],
        film_exposure_ev=exposure_ev, anchor_ev_offset=stock["anchor"],
    )
    bounded = str(getattr(plan, "film_crossover", "off")) != "datasheet"
    if stock["reversal"]:
        if timing != "fixed":
            raise ValueError("reversal_direct 无印相环节：timing 只能是 fixed")
        u = amounts_to_unit(amounts, b2["dye_lo"], b2["dye_hi"])
        developed = _tetrahedral(b2["volume"], u.astype(np.float32), b2["n"])
        if bounded:
            ev_y = np.log2(np.maximum(rgb @ REC2020_LUMA, EPS) / np.float32(0.18))
            for c in range(3):
                developed[:, c] /= np.interp(
                    ev_y, stock["cast_ev"], stock["cast_bounded"][:, c]
                )
        return np.maximum(developed, 0.0).astype(np.float32, copy=False)
    # Negative: B1 -> +tau -> paper development -> B2 (ratified §5.4).
    u1 = amounts_to_unit(amounts, stock["lo"], stock["hi"])
    lep2 = _tetrahedral(ps["b1"], u1.astype(np.float32), ps["n"]).astype(np.float64)
    if timing == "retimed":
        if not ps["retimed_ev_min"] <= exposure_ev <= ps["retimed_ev_max"]:
            # The pairing's paper rails bound the re-timable span; beyond it
            # the mid-grey target is physically unreachable — declared, not
            # fabricated (§5.3 hard-reject convention).
            raise ValueError(
                f"'{preset}' × '{medium}' 的 retimed 可达域为 "
                f"[{ps['retimed_ev_min']:+.2f}, {ps['retimed_ev_max']:+.2f}] EV"
                f"（相纸轨道的物理界）；exposure={exposure_ev:+.2f} 超出。"
                "可改用 fixed timing 或缩小胶片曝光"
            )
        tau = np.array([
            np.interp(exposure_ev, ps["tau_nodes"], ps["tau"][:, c]) for c in range(3)
        ])
        cast_key_ev = exposure_ev
    elif timing == "custom":
        tau = ps["tau"][int(np.argmin(np.abs(ps["tau_nodes"])))].copy()
        tau = tau + _custom_delta_tau(
            float(getattr(plan, "color_head_y", 0.0) or 0.0),
            float(getattr(plan, "color_head_m", 0.0) or 0.0),
            float(getattr(plan, "film_print_exposure_ev", 0.0) or 0.0),
        )
        cast_key_ev = 0.0
    else:
        tau = ps["tau"][int(np.argmin(np.abs(ps["tau_nodes"])))]
        cast_key_ev = 0.0
    dye = np.stack([
        np.interp(lep2[:, c] + tau[c], b2["paper_le2"], b2["paper_amounts"][:, c])
        for c in range(3)
    ], axis=1)
    u2 = amounts_to_unit(dye, b2["dye_lo"], b2["dye_hi"])
    developed = _tetrahedral(b2["volume"], u2.astype(np.float32), b2["n"])
    if bounded:
        if timing == "custom":
            raise ValueError(
                "custom timing 与有界灰阶中性化互斥：手动印相的意义是保留"
                "印出的样子；请配 --film-neutralization datasheet"
            )
        ev_y = np.log2(np.maximum(rgb @ REC2020_LUMA, EPS) / np.float32(0.18))
        nodes = ps["tau_nodes"]
        i_hi = int(np.searchsorted(nodes, cast_key_ev, side="left").clip(1, nodes.size - 1))
        i_lo = i_hi - 1
        t = (cast_key_ev - nodes[i_lo]) / (nodes[i_hi] - nodes[i_lo])
        cast_e = (1.0 - t) * ps["casts"][i_lo] + t * ps["casts"][i_hi]
        for c in range(3):
            developed[:, c] /= np.interp(ev_y, ps["cast_ev"], cast_e[:, c])
    return np.maximum(developed, 0.0).astype(np.float32, copy=False)


def apply_film_core(rgb_rec2020: Any, plan: Any) -> Any:
    """Film-takeover development. [N,3] -> [N,3]; two-stage v2 by default."""
    preset = str(getattr(plan, "curve_preset", "") or "")
    rgb = np.maximum(np.asarray(rgb_rec2020, dtype=np.float32), 0.0)
    if not _use_legacy_backend():
        return _apply_film_core_v2(rgb, plan, preset)
    lut, cast_ev, cast_bounded, ev_min, ev_max, n = _load_lut(preset)
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
