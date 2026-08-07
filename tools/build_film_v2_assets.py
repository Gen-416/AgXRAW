#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""film v2 P1: build schema-v5 two-stage assets (plan §3, §7.1).

Per stock, one npz under dngscan/data/film_v2/ carrying:

  Stage A (analytic; NOT a 3D LUT):
    observer          3x3 observer inverse (constrained NNLS + anchor)
    char_le           logE axis of the characteristic curves
    char_amounts      per-layer dye amounts along char_le  [K,3]
    amount_lo/hi      declared per-channel amount domain (cube bounds)
    anchor_ev_offset  reversal exposure anchor (0 for print-through negatives)
    exposure_ev_min/max  public film-exposure domain (plan §5.3)

  Stage B (density-domain cube, fixed timing q(0)):
    volume            65^3 x 3 float16, amounts-unit-cube -> viewed Rec.2020
    cast_ev/cast_bounded  bounded neutral-cast curve (same semantics as v1;
                      indexed by scene luminance EV at runtime)

  Provenance / gates (schema v5, plan §7.1):
    source SHA-256 of the negative/print profile JSONs, builder commit,
    observer residuals, f16 quantization error, and shipped oracle fixtures
    with float64 direct-chain truth for both neutralization variants PLUS the
    two-stage reproduction so the runtime test compares deployed bytes
    against the offline chain.

v1 equivalence (plan §12 P1): Stage A is the exact same math the v1 baker
used; Stage B is the same develop_amounts chain sampled over the amount cube
instead of the scene-EV cube. The oracle gate proves the two-stage composite
reproduces the direct chain within tolerance; the P0 freeze pins v1 while it
remains the shipping path.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import build_full_lut as v1  # noqa: E402
import fit_film_curve as ff  # noqa: E402
import spectral_base as sb  # noqa: E402
from dngscan.film_v2_math import (  # noqa: E402
    LOG10_2,
    SCENE_MID,
    amounts_to_unit,
    out_of_domain_share,
    stage_a_amounts,
)

OUT_DIR = PROJECT_ROOT / "dngscan" / "data" / "film_v2"
GRID_N = 65
SCHEMA = 5
EXPOSURE_EV_MIN, EXPOSURE_EV_MAX = -2.0, 2.0
PILOT_STOCKS = ("portra400", "velvia100", "vision3250d")
# P2 retimed pilot (plan §12 P2): negatives only — reversals have no print
# stage to re-time, enforced fail-closed everywhere.
RETIMED_PILOT = ("portra400", "vision3250d")
# Node history, all measured: 3-node OUTPUT-volume premix p99 0.36-0.73 stop,
# 5-node 0.13-0.22 (gate 0.03) — refuted; the factorized B1/paper/B2 chain
# reduced the E-dependence to the three smooth q(E) scalars, whose 1 EV
# linear interpolation then carried the whole residual (volumes 0.003 stop,
# q-interp 0.037 — decomposed with q_true). q solves are cheap, so q and the
# per-node casts sample at 0.25 EV steps; the oracle sits OFF that grid.
RETIMED_NODES = tuple(round(-2.0 + 0.25 * i, 4) for i in range(17))
MIDPOINT_ORACLE_EVS = (-1.875, -0.625, 0.375, 1.625)

_LUMA = np.array([0.2627, 0.6780, 0.0593])


def solve_q_at_exposure(chain, exposure_ev: float) -> np.ndarray:
    """Retimed printer solve (plan §5.3): the same Newton as the EV0 joint
    solve, with the mid-grey anchor moved to the neutral ramp point that a
    film exposed at `exposure_ev` puts mid-grey on. q(0) reproduces the
    stock's shipped q exactly."""
    develop, ev = chain.chain.develop, chain.chain.ev

    def mid_rgb(q):
        rgb0 = develop(q)
        return np.array([
            float(np.interp(exposure_ev, ev, rgb0[:, c])) for c in range(3)
        ])

    q = np.asarray(chain.chain.q, dtype=np.float64).copy()
    for _ in range(30):
        f = np.log(np.maximum(mid_rgb(q), 1e-9) / SCENE_MID)
        if float(np.max(np.abs(f))) < 1e-11:
            break
        jac = np.empty((3, 3))
        h = 1e-5
        for c in range(3):
            dq = q.copy(); dq[c] += h
            jac[:, c] = (np.log(np.maximum(mid_rgb(dq), 1e-9) / SCENE_MID) - f) / h
        q = q - np.linalg.solve(jac, f)
    residual = float(np.max(np.abs(np.log(np.maximum(mid_rgb(q), 1e-9) / SCENE_MID))))
    if residual > 1e-8:
        raise RuntimeError(f"retimed solve at {exposure_ev:+.1f} EV: residual {residual:.2e}")
    return q


def develop_amounts_q(chain, neg_amounts: np.ndarray, q: np.ndarray) -> np.ndarray:
    """chain.develop_amounts generalized to explicit printer exposures q
    (negatives only; the reversal path has no q)."""
    t_neg = ff._stack_reflectance(chain.neg, neg_amounts)
    log_ep = np.log10(np.maximum(
        sb.trapezoid(
            t_neg[:, :, None] * chain.print_weight[None, :, :], chain.wl, axis=1
        ),
        1e-12,
    ))
    dye = np.stack([
        np.interp(
            log_ep[:, c] + q[c],
            chain.paper["le"],
            chain.paper["amounts"][:, c],
        )
        for c in range(3)
    ], axis=1)
    reflect = ff._stack_reflectance(chain.paper, dye)
    return np.maximum(
        ff._display_rec2020(
            reflect, chain.paper_white, chain.wl, chain.paper["viewing"],
            chain.chain.flare, chain.chain.exp,
        ),
        1e-7,
    )


def _source_sha(name: str) -> str:
    path = ff.PROFILE_DIR / f"{name}.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _builder_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _oklab_from_rec2020(rgb: np.ndarray) -> np.ndarray:
    """Display-linear Rec.2020 -> Oklab (D65), builder-local."""
    m = np.array([
        [0.6369580483, 0.1446169036, 0.1688809752],
        [0.2627002120, 0.6779980715, 0.0593017165],
        [0.0000000000, 0.0280726930, 1.0609850577],
    ])
    xyz = np.maximum(rgb, 0.0) @ m.T
    m1 = np.array([
        [0.8189330101, 0.3618667424, -0.1288597137],
        [0.0329845436, 0.9293118715, 0.0361456387],
        [0.0482003018, 0.2643662691, 0.6338517070],
    ])
    lms = np.cbrt(np.maximum(xyz @ m1.T, 1e-12))
    m2 = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ])
    return lms @ m2.T


def _rec2020_from_oklab(lab: np.ndarray) -> np.ndarray:
    m2i = np.linalg.inv(np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]))
    m1i = np.linalg.inv(np.array([
        [0.8189330101, 0.3618667424, -0.1288597137],
        [0.0329845436, 0.9293118715, 0.0361456387],
        [0.0482003018, 0.2643662691, 0.6338517070],
    ]))
    lms = np.maximum(lab @ m2i.T, 0.0) ** 3
    xyz = lms @ m1i.T
    mi = np.linalg.inv(np.array([
        [0.6369580483, 0.1446169036, 0.1688809752],
        [0.2627002120, 0.6779980715, 0.0593017165],
        [0.0000000000, 0.0280726930, 1.0609850577],
    ]))
    return np.maximum(xyz @ mi.T, 0.0)


def _premix(a: np.ndarray, b: np.ndarray, t: float, domain: str) -> np.ndarray:
    """Blend two display-linear Rec.2020 sample sets in the declared domain."""
    if domain == "display_linear":
        return (1.0 - t) * a + t * b
    if domain == "xyz_logy_xy":
        m = np.array([
            [0.6369580483, 0.1446169036, 0.1688809752],
            [0.2627002120, 0.6779980715, 0.0593017165],
            [0.0000000000, 0.0280726930, 1.0609850577],
        ])
        xa = np.maximum(a, 1e-9) @ m.T
        xb = np.maximum(b, 1e-9) @ m.T
        sa = np.maximum(xa.sum(axis=-1, keepdims=True), 1e-12)
        sb_ = np.maximum(xb.sum(axis=-1, keepdims=True), 1e-12)
        xya, xyb = xa[..., :2] / sa, xb[..., :2] / sb_
        ya, yb = xa[..., 1:2], xb[..., 1:2]
        y = np.exp((1.0 - t) * np.log(np.maximum(ya, 1e-12))
                   + t * np.log(np.maximum(yb, 1e-12)))
        xy = (1.0 - t) * xya + t * xyb
        x_ = np.clip(xy[..., 0:1], 1e-6, 1 - 1e-6)
        y_ = np.clip(xy[..., 1:2], 1e-6, 1 - 1e-6)
        big_x = x_ / y_ * y
        big_z = (1.0 - x_ - y_) / y_ * y
        xyz = np.concatenate([big_x, y, big_z], axis=-1)
        return np.maximum(xyz @ np.linalg.inv(m).T, 0.0)
    if domain == "oklab":
        return _rec2020_from_oklab(
            (1.0 - t) * _oklab_from_rec2020(a) + t * _oklab_from_rec2020(b)
        )
    raise ValueError(f"unknown premix domain {domain!r}")


def _de00_approx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Oklab-distance stand-in for DeltaE00 (scaled x100, same order)."""
    return np.linalg.norm(
        _oklab_from_rec2020(a) - _oklab_from_rec2020(b), axis=-1
    ) * 100.0


def _bake_retimed(stock, chain, observer, char_le, char_amounts,
                  amount_lo, amount_hi, grid_amounts) -> dict:
    """Retimed exposure state for a pilot negative — FACTORIZED Stage B.

    The premix design was refuted by its own gates: blending retimed OUTPUT
    volumes measured best-domain p99 0.36-0.73 stop at three nodes and
    0.13-0.22 at five (gate: 0.03) — the print re-timing shifts the paper
    curves' input, and no output-space blend represents that shift (the same
    failure class as baking the cast composite). The chain's own structure
    gives the exact answer instead: q enters BETWEEN the printing integral
    and the paper curves, so Stage B factorizes into

        B1: dye amounts -> log paper exposure   (q-independent 65^3 volume)
        +q(E): three smooth numbers, interpolated over solved nodes
        paper: three 1-D development curves     (analytic tables)
        B2: paper dye amounts -> viewed Rec.2020 (q-independent 65^3 volume)

    Every exposure state is then EXACT in q up to the q(E) node interpolation
    (three scalars, smooth) plus two smooth-volume lookups. The refutation
    numbers are recorded below; the domain study is kept as evidence, not
    shipped machinery.
    """
    from dngscan.film_v2_math import stage_a_amounts

    node_q = np.stack([solve_q_at_exposure(chain, e) for e in RETIMED_NODES])

    # B1: printing integral over the negative-amount cube (no q).
    t_neg = ff._stack_reflectance(chain.neg, grid_amounts)
    log_ep = np.log10(np.maximum(
        sb.trapezoid(
            t_neg[:, :, None] * chain.print_weight[None, :, :], chain.wl, axis=1
        ),
        1e-12,
    )).astype(np.float32)
    b1 = log_ep.reshape(GRID_N, GRID_N, GRID_N, 3)

    # Paper development tables + dye domain.
    paper_le = np.asarray(chain.paper["le"], dtype=np.float64)
    paper_amounts = np.asarray(chain.paper["amounts"], dtype=np.float64)
    pd_lo = paper_amounts.min(axis=0)
    pd_hi = paper_amounts.max(axis=0)

    # B2: paper dye cube -> viewed Rec.2020 (no q).
    axis = np.arange(GRID_N) / (GRID_N - 1)
    grid_u = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    dye_grid = pd_lo[None, :] + grid_u * (pd_hi - pd_lo)[None, :]
    reflect = ff._stack_reflectance(chain.paper, dye_grid)
    b2 = np.maximum(
        ff._display_rec2020(
            reflect, chain.paper_white, chain.wl, chain.paper["viewing"],
            chain.chain.flare, chain.chain.exp,
        ),
        1e-7,
    ).astype(np.float32).reshape(GRID_N, GRID_N, GRID_N, 3)

    # Per-node bounded casts (neutral ramp at E through q(E)); runtime
    # interpolates the cast linearly in E — exact at nodes, declared
    # approximation between them.
    ramp_ev = np.linspace(v1.EV_MIN, v1.EV_MAX, 257)
    step = (ramp_ev.size - 1) // (GRID_N - 1)
    casts = []
    for i, e in enumerate(RETIMED_NODES):
        ramp_rgb = SCENE_MID * np.exp2(ramp_ev)[:, None].repeat(3, axis=1)
        amounts = stage_a_amounts(
            ramp_rgb, observer, char_le, char_amounts,
            film_exposure_ev=e, anchor_ev_offset=0.0,
        )
        out = develop_amounts_q(chain, amounts, node_q[i])
        y = out @ _LUMA
        cast = out / np.maximum(y, 1e-9)[:, None]
        h_star = 1.0 / np.maximum(cast, 1e-9)
        t_hi = np.where(h_star > 4.0, 3.0 / np.maximum(h_star - 1.0, 1e-9), 1.0)
        t_lo = np.where(h_star < 0.25, 0.75 / np.maximum(1.0 - h_star, 1e-9), 1.0)
        t_ev = np.clip(np.min(np.minimum(t_hi, t_lo), axis=1), 0.0, 1.0)
        h = 1.0 + t_ev[:, None] * (h_star - 1.0)
        casts.append((1.0 / np.clip(h, 0.25, 4.0))[::step].astype(np.float32))

    # Midpoint oracle: truth = direct chain with q solved AT the midpoint;
    # candidate = the deployed factorized runtime (f16 volumes, q interp).
    from dngscan.film_develop import _tetrahedral
    from dngscan.film_v2_math import amounts_to_unit

    rng = np.random.default_rng(20260808)
    mid_ev = rng.uniform(-9.0, 5.0, (96, 3))
    mid_rgb = SCENE_MID * np.exp2(mid_ev)
    worst_p99 = worst_de95 = worst_demax = 0.0
    b1_f16 = b1.astype(np.float16).astype(np.float32)
    b2_f16 = b2.astype(np.float16).astype(np.float32)
    for e_mid in MIDPOINT_ORACLE_EVS:
        q_true = solve_q_at_exposure(chain, e_mid)
        amounts_mid = stage_a_amounts(
            mid_rgb, observer, char_le, char_amounts,
            film_exposure_ev=e_mid, anchor_ev_offset=0.0,
        )
        truth = develop_amounts_q(chain, amounts_mid, q_true)
        # Deployed path:
        u1 = amounts_to_unit(amounts_mid, amount_lo, amount_hi)
        lep = _tetrahedral(b1_f16, u1.astype(np.float32), GRID_N).astype(np.float64)
        q_interp = np.stack([
            np.interp(e_mid, RETIMED_NODES, node_q[:, c]) for c in range(3)
        ])
        dye = np.stack([
            np.interp(lep[:, c] + q_interp[c], paper_le, paper_amounts[:, c])
            for c in range(3)
        ], axis=1)
        u2 = amounts_to_unit(dye, pd_lo, pd_hi)
        got = _tetrahedral(b2_f16, u2.astype(np.float32), GRID_N)
        vis = truth > 5e-3
        stop = np.abs(np.log2(
            np.maximum(got[vis], 1e-9) / np.maximum(truth[vis], 1e-9)
        ))
        de = _de00_approx(got, truth)
        worst_p99 = max(worst_p99, float(np.percentile(stop, 99)))
        worst_de95 = max(worst_de95, float(np.percentile(de, 95)))
        worst_demax = max(worst_demax, float(de.max()))
    print(f"  retimed factorized: p99 {worst_p99:.4f} stop, "
          f"dE95 {worst_de95:.2f}, dE max {worst_demax:.2f}")

    # Shipped oracle fixtures: midpoint truths so the runtime test compares
    # the DEPLOYED bytes against the offline chain, node truths for the EV0
    # neutrality gate.
    oracle_truths = []
    for e_mid in MIDPOINT_ORACLE_EVS:
        q_true = solve_q_at_exposure(chain, e_mid)
        amounts_mid = stage_a_amounts(
            mid_rgb, observer, char_le, char_amounts,
            film_exposure_ev=e_mid, anchor_ev_offset=0.0,
        )
        oracle_truths.append(develop_amounts_q(chain, amounts_mid, q_true))

    return {
        "retimed_oracle_ev": mid_ev.astype(np.float32),
        "retimed_oracle_exposures": np.asarray(MIDPOINT_ORACLE_EVS, dtype=np.float32),
        "retimed_oracle_truth": np.stack(oracle_truths).astype(np.float32),
        "retimed_nodes": np.asarray(RETIMED_NODES, dtype=np.float32),
        "retimed_q": node_q.astype(np.float64),
        "retimed_b1_logep": b1.astype(np.float16),
        "retimed_b2_volume": b2.astype(np.float16),
        "retimed_paper_le": paper_le,
        "retimed_paper_amounts": paper_amounts,
        "retimed_paper_lo": pd_lo,
        "retimed_paper_hi": pd_hi,
        "retimed_casts": np.stack(casts).astype(np.float32),
        "retimed_oracle_p99_stop": np.float32(worst_p99),
        "retimed_oracle_de95": np.float32(worst_de95),
        "retimed_oracle_de_max": np.float32(worst_demax),
        # Refutation record (plan §5.4 amendment): the premix design's
        # best-domain midpoint errors, kept as evidence.
        "premix_refuted_p99_stop": np.float32(0.1346),
        "premix_refuted_note": np.asarray(
            "output-volume premix refuted: 3-node best-domain p99 0.36-0.73 "
            "stop, 5-node 0.13-0.22 (gate 0.03); factorized B1/paper/B2 "
            "replaces it"
        ),
    }


def build_stock(stock_key: str) -> dict:
    theatrical = stock_key.endswith("_theatrical")
    stock = ff.STOCKS[stock_key.removesuffix("_theatrical")]
    chain = v1._Chain(stock, theatrical=theatrical)
    observer, obs_p99, obs_cv_p99 = v1.observer_matrix(stock)

    neg = chain.neg
    char_le = np.asarray(neg["le"], dtype=np.float64)
    char_amounts = np.asarray(neg["amounts"], dtype=np.float64)
    amount_lo = char_amounts.min(axis=0)
    amount_hi = char_amounts.max(axis=0)
    anchor_ev = float(chain.e0) if chain.reversal else 0.0

    # Stage B: the amount cube -> viewed Rec.2020 through the SAME solved
    # chain (fixed timing q(0)). develop_amounts is defined for arbitrary
    # stacks, so off-curve combinations are the chain's honest answer, not an
    # extrapolation of the LUT.
    axis = np.arange(GRID_N) / (GRID_N - 1)
    grid_u = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    grid_amounts = amount_lo[None, :] + grid_u * (amount_hi - amount_lo)[None, :]
    volume = chain.develop_amounts(grid_amounts).astype(np.float32)
    volume = volume.reshape(GRID_N, GRID_N, GRID_N, 3)

    # Neutral ramp / bounded cast: identical construction to v1 (full
    # precision direct chain, node-matched sampling on the STAGE-B-relevant
    # scene axis — the runtime keys the cast on scene luminance EV, which is
    # unchanged by the A/B split).
    ramp_ev = np.linspace(v1.EV_MIN, v1.EV_MAX, 257)
    ramp_rgb = SCENE_MID * np.exp2(ramp_ev)[:, None].repeat(3, axis=1)
    ramp_out = v1.chain_eval(stock, chain, observer, ramp_rgb)
    ramp_y = ramp_out @ _LUMA
    cast = ramp_out / np.maximum(ramp_y, 1e-9)[:, None]
    h_star = 1.0 / np.maximum(cast, 1e-9)
    t_hi = np.where(h_star > 4.0, 3.0 / np.maximum(h_star - 1.0, 1e-9), 1.0)
    t_lo = np.where(h_star < 0.25, 0.75 / np.maximum(1.0 - h_star, 1e-9), 1.0)
    t_ev = np.clip(np.min(np.minimum(t_hi, t_lo), axis=1), 0.0, 1.0)
    h = 1.0 + t_ev[:, None] * (h_star - 1.0)
    cast_b = 1.0 / np.clip(h, 0.25, 4.0)
    step = (ramp_ev.size - 1) // (GRID_N - 1)
    cast_ev = ramp_ev[::step].astype(np.float32)
    cast_b = cast_b[::step].astype(np.float32)

    # Oracle fixtures: direct-chain float64 truth vs the two-stage composite
    # exactly as deployed (Stage A analytic + f16 volume tetrahedral).
    rng = np.random.default_rng(20260807)
    oracle_ev = rng.uniform(-9.0, 5.0, (96, 3))
    oracle_rgb = SCENE_MID * np.exp2(oracle_ev)
    oracle_truth = v1.chain_eval(stock, chain, observer, oracle_rgb)

    from dngscan.film_develop import _tetrahedral

    amounts = stage_a_amounts(
        oracle_rgb, observer, char_le, char_amounts,
        film_exposure_ev=0.0, anchor_ev_offset=anchor_ev,
    )
    ood = out_of_domain_share(amounts, amount_lo, amount_hi)
    u = amounts_to_unit(amounts, amount_lo, amount_hi)
    two_stage = _tetrahedral(
        volume.astype(np.float16).astype(np.float32), u.astype(np.float32), GRID_N
    )
    vis = oracle_truth > 5e-3
    err = np.abs(np.log2(
        np.maximum(two_stage[vis], 1e-9) / np.maximum(oracle_truth[vis], 1e-9)
    ))
    oracle_p99 = float(np.percentile(err, 99)) if err.size else 0.0
    oracle_max = float(err.max()) if err.size else 0.0

    q = volume.astype(np.float16).astype(np.float32)
    qvis = volume > 5e-3
    quant_err = float(np.abs(np.log2(
        np.maximum(q[qvis], 1e-9) / np.maximum(volume[qvis], 1e-9)
    )).max()) if np.any(qvis) else 0.0

    sources = {"negative": stock["negative"]}
    if not chain.reversal:
        sources["print"] = stock["print"]

    # --- P2: retimed exposure nodes (pilot negatives only) -----------------
    retimed: dict = {}
    if (not chain.reversal) and stock_key in RETIMED_PILOT:
        retimed = _bake_retimed(
            stock, chain, observer, char_le, char_amounts,
            amount_lo, amount_hi, grid_amounts,
        )

    return {
        **retimed,
        "observer": observer.astype(np.float64),
        "observer_p99_stop": np.float32(obs_p99),
        "observer_cv_p99_stop": np.float32(obs_cv_p99),
        "char_le": char_le.astype(np.float64),
        "char_amounts": char_amounts.astype(np.float64),
        "amount_lo": amount_lo.astype(np.float64),
        "amount_hi": amount_hi.astype(np.float64),
        "anchor_ev_offset": np.float64(anchor_ev),
        "exposure_ev_min": np.float32(EXPOSURE_EV_MIN),
        "exposure_ev_max": np.float32(EXPOSURE_EV_MAX),
        "volume": volume.astype(np.float16),
        "cast_ev": cast_ev,
        "cast_bounded": cast_b,
        "n": np.int32(GRID_N),
        "reversal": np.bool_(chain.reversal),
        "oracle_ev": oracle_ev.astype(np.float32),
        "oracle_truth": oracle_truth.astype(np.float32),
        "oracle_two_stage": two_stage.astype(np.float32),
        "oracle_p99_stop": np.float32(oracle_p99),
        "oracle_max_stop": np.float32(oracle_max),
        "oracle_out_of_domain_share": np.float32(ood),
        "quant_err_stop": np.float32(quant_err),
        "source_names": np.asarray(sorted(sources.values())),
        "source_sha256": np.asarray([
            _source_sha(sources[k]) for k in sorted(sources)
        ]),
        "builder_commit": np.asarray(_builder_commit()),
        "input_space": np.asarray("scene_rec2020_via_amounts"),
        "timing_policy": np.asarray("fixed_q0"),
        "schema": np.int32(SCHEMA),
    }


def write_stock(stock_key: str) -> None:
    data = build_stock(stock_key)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / f"{stock_key}.npz", **data)
    size = (OUT_DIR / f"{stock_key}.npz").stat().st_size / 1024
    print(
        f"{stock_key}: oracle p99 {float(data['oracle_p99_stop']):.4f} / "
        f"max {float(data['oracle_max_stop']):.4f} stop; "
        f"quant {float(data['quant_err_stop']):.4f}; "
        f"ood {float(data['oracle_out_of_domain_share']) * 100:.1f}%; "
        f"{size:.0f} KiB"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", nargs="*", default=None,
                    help="默认:全部卷(含影院引用变体);或显式列出")
    args = ap.parse_args()
    if args.stocks:
        keys = args.stocks
    else:
        keys = []
        for key in ff.STOCKS:
            keys.append(key)
            stock = ff.STOCKS[key]
            if (not stock.get("positive")
                    and ff.PRINT_SURROUND.get(str(stock.get("print"))) == "dark"):
                keys.append(f"{key}_theatrical")
    for key in keys:
        write_stock(key)
