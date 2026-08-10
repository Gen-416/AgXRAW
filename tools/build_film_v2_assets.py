#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""film v2 P3: build the modular schema-v5 asset family (plan §3, §7.1).

Three asset kinds under dngscan/data/film_v2/, exactly the ratified §7.1
split — Stage B is factorized and MUST NOT collapse back into one LUT:

  <stock>.npz                    kind=stock — Stage A only: observer inverse,
                                 characteristic tables, amount domain,
                                 reversal anchor, exposure domain, and the
                                 stock's default medium/pairing references.
  print__<stock>__<medium>.npz   kind=print_state (negatives) — B1 volume
                                 (negative density -> log2 paper-layer
                                 exposure; per stock dye stack x print
                                 sensitometry, NO tau), the 0.25 EV
                                 timing_table tau(E)=log2(q(E)) with per-node
                                 bounded casts, midpoint oracle residuals and
                                 the output-premix refutation record.
  b2__<medium>.npz               kind=b2 — positive-medium density -> viewed
                                 Rec.2020 (65^3) plus the medium's paper
                                 development tables (log2 axis). Keyed by
                                 print medium x viewing condition; REUSED
                                 across stocks. Reversals get their own
                                 direct__<stock> medium and skip B1/tau/paper.

Units: log2 everywhere per the ratified formalism (tau_j = log2 q_j; B1
outputs log2 paper exposure; the paper axis ships in log2). The exactness of
the analytic timing is scoped to the current paper-layer exposure model —
real Y/M filter spectra would alter B1's integral density-dependently and
must rebuild/parameterize B1 instead (§7.2).

Node history, all measured (§5.4): 3-node OUTPUT premix p99 0.36-0.73 stop,
5-node 0.13-0.22 (gate 0.03) — refuted; factorized residuals decompose to
volumes ~0.003 stop with q-interp carrying the rest, so tau/cast sample at
0.25 EV and the midpoint oracle sits OFF that grid.
"""
from __future__ import annotations

import hashlib
import json
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
    EDITORIAL_DENSITY_LIMIT,
    EDITORIAL_FOG_MAX,
    LOG10_2,
    SCENE_MID,
    amounts_to_unit,
    out_of_domain_share,
    stage_a_amounts,
)

OUT_DIR = PROJECT_ROOT / "dngscan" / "data" / "film_v2"
GRID_N = 65
SCHEMA = 6  # 6: identity fields (stock/medium) are part of the ABI
EXPOSURE_EV_MIN, EXPOSURE_EV_MAX = -2.0, 2.0
TAU_NODES = tuple(round(-2.0 + 0.25 * i, 4) for i in range(17))
MIDPOINT_ORACLE_EVS = (-1.875, -0.625, 0.375, 1.625)
PREMIX_REFUTATION = (
    "output-volume premix refuted: 3-node best-domain p99 0.36-0.73 stop, "
    "5-node 0.13-0.22 (gate 0.03); factorized B1/tau/paper/B2 replaces it "
    "(volumes ~0.003 stop, tau interp at 0.25 EV nodes)"
)
# Extra pairings beyond each stock's default (P3 cross-medium verification).
EXTRA_PAIRINGS = {"portra400": ("kodak_supra_endura",), "vision3250d": ("kodak_2393",)}

_LUMA = np.array([0.2627, 0.6780, 0.0593])


def _source_sha(name: str) -> str:
    return hashlib.sha256((ff.PROFILE_DIR / f"{name}.json").read_bytes()).hexdigest()


def _builder_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _medium_id(stock: dict, theatrical: bool, print_name: str | None = None) -> str:
    if stock.get("positive"):
        return f"direct__{stock['key']}"
    paper = print_name or stock["print"]
    return f"{paper}__{'native' if theatrical else 'translated'}"


class _PrintChain:
    """The negative->paper chain for an arbitrary stock x paper pairing,
    solved at EV0 (tau(0)) — generalizes v1._Chain to non-default papers."""

    def __init__(self, stock: dict, paper_name: str, theatrical: bool):
        self.neg = ff._load_spectral(stock["negative"])
        self.wl = self.neg["wl"]
        paper = ff._regrid(ff._load_spectral(paper_name), self.wl)
        if paper["sens"] is None:
            raise RuntimeError(f"{paper_name}: no log_sensitivity")
        self.paper = paper
        enlarger = sb.th_kg3_spd(self.wl)
        self.print_weight = paper["sens"] * enlarger[:, None]
        self.paper_white = ff._stack_reflectance(
            paper, np.nanmin(paper["amounts"], axis=0)[None, :]
        )[0]
        self.exp = 1.0 if theatrical else ff.surround_exponent(
            ff.PRINT_SURROUND.get(paper_name, "average")
        )
        self.flare = 0.0
        self.ev = self.neg["le"] / ff.LOG10_2
        # tau(0) joint solve (Newton on log(mid/0.18)), identical to
        # _solved_print_chain's construction.
        t_neg = ff._stack_reflectance(self.neg, self.neg["amounts"])
        self._ramp_logep = np.log10(np.maximum(
            sb.trapezoid(t_neg[:, :, None] * self.print_weight[None, :, :], self.wl, axis=1),
            1e-12,
        ))
        q = np.array([
            float(np.interp(
                0.5 * (paper["amounts"][:, c].min() + paper["amounts"][:, c].max()),
                paper["amounts"][:, c], paper["le"],
            )) - float(np.interp(0.0, self.ev, self._ramp_logep[:, c]))
            for c in range(3)
        ])
        self.q0 = self._solve_q(q, 0.0)

    def _develop_ramp(self, q: np.ndarray) -> np.ndarray:
        dye = np.stack([
            np.interp(self._ramp_logep[:, c] + q[c],
                      self.paper["le"], self.paper["amounts"][:, c])
            for c in range(3)
        ], axis=1)
        reflect = ff._stack_reflectance(self.paper, dye)
        return ff._display_rec2020(
            reflect, self.paper_white, self.wl, self.paper["viewing"],
            self.flare, self.exp,
        )

    def _solve_q(self, q0: np.ndarray, exposure_ev: float) -> np.ndarray:
        """Damped Newton with lstsq fallback. Raises RuntimeError when the
        target is physically unreachable (paper rails make the Jacobian
        singular and the residual stalls) — the caller DECLARES the reachable
        span instead of fabricating a solution."""
        def mid_rgb(q):
            rgb0 = self._develop_ramp(q)
            return np.array([
                float(np.interp(exposure_ev, self.ev, rgb0[:, c])) for c in range(3)
            ])

        q = np.asarray(q0, dtype=np.float64).copy()
        f = np.log(np.maximum(mid_rgb(q), 1e-9) / SCENE_MID)
        for _ in range(60):
            if float(np.max(np.abs(f))) < 1e-11:
                break
            jac = np.empty((3, 3))
            h = 1e-5
            for c in range(3):
                dq = q.copy(); dq[c] += h
                jac[:, c] = (np.log(np.maximum(mid_rgb(dq), 1e-9) / SCENE_MID) - f) / h
            try:
                step = np.linalg.solve(jac, f)
            except np.linalg.LinAlgError:
                step, *_ = np.linalg.lstsq(jac, f, rcond=1e-10)
            # Backtracking line search: accept the largest damping that
            # reduces the residual; a stalled search means the rail is real.
            best = None
            for damp in (1.0, 0.5, 0.25, 0.1, 0.05):
                cand = q - damp * step
                fc = np.log(np.maximum(mid_rgb(cand), 1e-9) / SCENE_MID)
                if float(np.max(np.abs(fc))) < float(np.max(np.abs(f))) - 1e-14:
                    best = (cand, fc)
                    break
            if best is None:
                break
            q, f = best
        residual = float(np.max(np.abs(f)))
        if residual > 1e-8:
            raise RuntimeError(
                f"printer solve at {exposure_ev:+.2f} EV: residual {residual:.2e}"
            )
        return q

    def solve_q(self, exposure_ev: float) -> np.ndarray:
        return self._solve_q(self.q0, exposure_ev)

    def b1_logep2(self, neg_amounts: np.ndarray) -> np.ndarray:
        """Negative dye amounts -> log2 paper-layer exposure (no tau)."""
        t_neg = ff._stack_reflectance(self.neg, neg_amounts)
        log_ep = np.log10(np.maximum(
            sb.trapezoid(t_neg[:, :, None] * self.print_weight[None, :, :], self.wl, axis=1),
            1e-12,
        ))
        return log_ep / LOG10_2

    def develop_q(self, neg_amounts: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Direct chain with explicit printer exposures q (log10 units)."""
        lep = self.b1_logep2(neg_amounts) * LOG10_2
        dye = np.stack([
            np.interp(lep[:, c] + q[c], self.paper["le"], self.paper["amounts"][:, c])
            for c in range(3)
        ], axis=1)
        reflect = ff._stack_reflectance(self.paper, dye)
        return np.maximum(
            ff._display_rec2020(
                reflect, self.paper_white, self.wl, self.paper["viewing"],
                self.flare, self.exp,
            ),
            1e-7,
        )


# Editorial developer envelope: colour density scales amounts about the
# mid-grey anchor by up to (1 + EDITORIAL_DENSITY_LIMIT)x, fog adds up to
# EDITORIAL_FOG_MAX uniform density (contrast warps the logE axis and cannot
# leave the measured amount range). The B1 / reversal-B2 computational shaper
# domains must COVER this envelope — review batch 13 measured 35-47% of
# perturbed curve nodes silently clamped against the bare measured envelope.
# The bounds are IMPORTED from dngscan.film_v2_math, the same constants
# validate_film_plans and the runtime guard in developer_perturbation
# enforce (mainline A2 follow-up: a locally duplicated 1.5/0.3 here is how
# the bake and the declaration drift apart). The spectral chain evaluates the
# extension by the same Beer-Lambert stacking (declared extrapolation beyond
# the measured Dmin/Dmax).
EDITORIAL_DENSITY_SCALE = 1.0 + EDITORIAL_DENSITY_LIMIT


def _stage_a_tables(stock: dict):
    neg = ff._load_spectral(stock["negative"])
    char_le = np.asarray(neg["le"], dtype=np.float64)
    char_amounts = np.asarray(neg["amounts"], dtype=np.float64)
    mid = np.array([
        np.interp(0.0, char_le, char_amounts[:, c]) for c in range(3)
    ])
    meas_lo = char_amounts.min(axis=0)
    meas_hi = char_amounts.max(axis=0)
    lo = np.maximum(mid - EDITORIAL_DENSITY_SCALE * (mid - meas_lo), 0.0)
    hi = mid + EDITORIAL_DENSITY_SCALE * (meas_hi - mid) + EDITORIAL_FOG_MAX
    return neg, char_le, char_amounts, lo, hi


def _observer(stock: dict):
    return v1.observer_matrix(stock)


def _reversal_anchor(stock: dict, chain: "v1._Chain") -> float:
    return float(chain.e0) if chain.reversal else 0.0


def _grid_u() -> np.ndarray:
    axis = np.arange(GRID_N) / (GRID_N - 1)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def build_b2_negative(paper_name: str, theatrical: bool) -> dict:
    """Positive-medium density cube -> viewed Rec.2020 for a print paper."""
    medium_id = f"{paper_name}__{'native' if theatrical else 'translated'}"
    probe = ff._load_spectral(paper_name)
    wl = np.asarray(probe["wl"], dtype=np.float64)
    paper = probe
    white = ff._stack_reflectance(paper, np.nanmin(paper["amounts"], axis=0)[None, :])[0]
    exp = 1.0 if theatrical else ff.surround_exponent(
        ff.PRINT_SURROUND.get(paper_name, "average")
    )
    pd_lo = np.asarray(paper["amounts"], dtype=np.float64).min(axis=0)
    pd_hi = np.asarray(paper["amounts"], dtype=np.float64).max(axis=0)
    dye_grid = pd_lo[None, :] + _grid_u() * (pd_hi - pd_lo)[None, :]
    reflect = ff._stack_reflectance(paper, dye_grid)
    volume = np.maximum(
        ff._display_rec2020(reflect, white, wl, paper["viewing"], 0.0, exp),
        1e-7,
    ).astype(np.float32).reshape(GRID_N, GRID_N, GRID_N, 3)
    return {
        "kind": np.asarray("b2"),
        "medium": np.asarray(medium_id),
        "volume": volume.astype(np.float16),
        "paper_le2": (np.asarray(paper["le"], dtype=np.float64) / LOG10_2),
        "paper_amounts": np.asarray(paper["amounts"], dtype=np.float64),
        "dye_lo": pd_lo,
        "dye_hi": pd_hi,
        "surround_exp": np.float64(exp),
        "viewing": np.asarray(str(paper["viewing"])),
        "source_names": np.asarray([paper_name]),
        "source_sha256": np.asarray([_source_sha(paper_name)]),
        "builder_commit": np.asarray(_builder_commit()),
        "input_space": np.asarray("positive_density"),
        "schema": np.int32(SCHEMA),
        "n": np.int32(GRID_N),
    }


def build_b2_reversal(stock_key: str) -> dict:
    """Slide dye cube -> viewed Rec.2020 (the reversal's direct medium)."""
    medium_id = f"direct__{stock_key}"
    stock = ff.STOCKS[stock_key]
    chain = v1._Chain(stock, theatrical=False)
    _, _, _, lo, hi = _stage_a_tables(stock)
    dye_grid = lo[None, :] + _grid_u() * (hi - lo)[None, :]
    volume = chain.develop_amounts(dye_grid).astype(np.float32)
    return {
        "kind": np.asarray("b2"),
        "medium": np.asarray(medium_id),
        "volume": volume.astype(np.float16).reshape(GRID_N, GRID_N, GRID_N, 3),
        "paper_le2": np.zeros(2),   # no print development stage
        "paper_amounts": np.zeros((2, 3)),
        "dye_lo": lo,
        "dye_hi": hi,
        "surround_exp": np.float64(ff.surround_exponent("dark")),
        "viewing": np.asarray(str(chain.neg["viewing"])),
        "source_names": np.asarray([stock["negative"]]),
        "source_sha256": np.asarray([_source_sha(stock["negative"])]),
        "builder_commit": np.asarray(_builder_commit()),
        "input_space": np.asarray("positive_density"),
        "schema": np.int32(SCHEMA),
        "n": np.int32(GRID_N),
    }


def build_print_state(stock_key: str, paper_name: str, theatrical: bool) -> dict:
    """B1 volume + tau/cast timing table + oracles for one stock x paper."""
    stock = dict(ff.STOCKS[stock_key.removesuffix("_theatrical")])
    stock["key"] = stock_key
    chain = _PrintChain(stock, paper_name, theatrical)
    observer, obs_p99, obs_cv = _observer(stock)
    neg, char_le, char_amounts, lo, hi = _stage_a_tables(stock)

    grid_amounts = lo[None, :] + _grid_u() * (hi - lo)[None, :]
    b1 = chain.b1_logep2(grid_amounts).astype(np.float32).reshape(GRID_N, GRID_N, GRID_N, 3)

    # Solve outward from EV0; where the paper's rails make a node
    # unreachable the pairing DECLARES its reachable retimed span instead of
    # fabricating (fail-closed at runtime beyond it). Warm-start each node
    # from its neighbour for robustness.
    node_arr = np.asarray(TAU_NODES, dtype=np.float64)
    zero_i = int(np.argmin(np.abs(node_arr)))
    q_by_node: dict[int, np.ndarray] = {zero_i: chain.solve_q(0.0)}
    lo_i = hi_i = zero_i
    for i in range(zero_i + 1, node_arr.size):
        try:
            q_by_node[i] = chain._solve_q(q_by_node[i - 1], float(node_arr[i]))
            hi_i = i
        except RuntimeError:
            break
    for i in range(zero_i - 1, -1, -1):
        try:
            q_by_node[i] = chain._solve_q(q_by_node[i + 1], float(node_arr[i]))
            lo_i = i
        except RuntimeError:
            break
    kept = list(range(lo_i, hi_i + 1))
    nodes_kept = node_arr[kept]
    tau = np.stack([q_by_node[i] for i in kept]) / LOG10_2  # log2
    retimed_ev_min = float(nodes_kept[0])
    retimed_ev_max = float(nodes_kept[-1])
    # Per-node bounded casts over the scene-EV axis.
    ramp_ev = np.linspace(v1.EV_MIN, v1.EV_MAX, 257)
    step = (ramp_ev.size - 1) // (GRID_N - 1)
    casts = []
    for i, e in enumerate(nodes_kept):
        ramp_rgb = SCENE_MID * np.exp2(ramp_ev)[:, None].repeat(3, axis=1)
        amounts = stage_a_amounts(
            ramp_rgb, observer, char_le, char_amounts, film_exposure_ev=e,
        )
        out = chain.develop_q(amounts, tau[i] * LOG10_2)
        y = out @ _LUMA
        cast = out / np.maximum(y, 1e-9)[:, None]
        h_star = 1.0 / np.maximum(cast, 1e-9)
        t_hi = np.where(h_star > 4.0, 3.0 / np.maximum(h_star - 1.0, 1e-9), 1.0)
        t_lo = np.where(h_star < 0.25, 0.75 / np.maximum(1.0 - h_star, 1e-9), 1.0)
        t_ev = np.clip(np.min(np.minimum(t_hi, t_lo), axis=1), 0.0, 1.0)
        h = 1.0 + t_ev[:, None] * (h_star - 1.0)
        casts.append((1.0 / np.clip(h, 0.25, 4.0))[::step].astype(np.float32))
    cast_ev = ramp_ev[::step].astype(np.float32)

    # Midpoint oracle: direct chain with q solved AT the midpoint vs the
    # deployed factorized composite (f16 B1/B2, tau interp, log2 paper axis).
    from dngscan.film_develop import _tetrahedral

    b2d = build_b2_negative(paper_name, theatrical)
    b2_vol = b2d["volume"].astype(np.float32)
    paper_le2 = b2d["paper_le2"]
    paper_am = b2d["paper_amounts"]
    pd_lo, pd_hi = b2d["dye_lo"], b2d["dye_hi"]

    rng = np.random.default_rng(20260808)
    mid_ev = rng.uniform(-9.0, 5.0, (96, 3))
    mid_rgb = SCENE_MID * np.exp2(mid_ev)
    b1_f16 = b1.astype(np.float16).astype(np.float32)
    worst_p99 = worst_stop_max = 0.0
    truths = []
    for e_mid in MIDPOINT_ORACLE_EVS:
        if not retimed_ev_min <= e_mid <= retimed_ev_max:
            continue
        q_true = chain.solve_q(e_mid)
        amounts_mid = stage_a_amounts(
            mid_rgb, observer, char_le, char_amounts, film_exposure_ev=e_mid,
        )
        truth = chain.develop_q(amounts_mid, q_true)
        truths.append(truth)
        u1 = amounts_to_unit(amounts_mid, lo, hi)
        lep2 = _tetrahedral(b1_f16, u1.astype(np.float32), GRID_N).astype(np.float64)
        tau_i = np.array([np.interp(e_mid, nodes_kept, tau[:, c]) for c in range(3)])
        dye = np.stack([
            np.interp(lep2[:, c] + tau_i[c], paper_le2, paper_am[:, c])
            for c in range(3)
        ], axis=1)
        u2 = amounts_to_unit(dye, pd_lo, pd_hi)
        got = _tetrahedral(b2_vol, u2.astype(np.float32), GRID_N)
        vis = truth > 5e-3
        err = np.abs(np.log2(
            np.maximum(got[vis], 1e-9) / np.maximum(truth[vis], 1e-9)
        ))
        worst_p99 = max(worst_p99, float(np.percentile(err, 99)))
        worst_stop_max = max(worst_stop_max, float(err.max()))

    return {
        "kind": np.asarray("print_state"),
        "stock": np.asarray(stock_key),
        "medium": np.asarray(_medium_id(stock, theatrical, paper_name)),
        "b1_volume": b1.astype(np.float16),
        "tau_nodes": nodes_kept,
        "retimed_ev_min": np.float64(retimed_ev_min),
        "retimed_ev_max": np.float64(retimed_ev_max),
        "tau": tau.astype(np.float64),
        "cast_ev": cast_ev,
        "casts": np.stack(casts).astype(np.float32),
        "oracle_ev": mid_ev.astype(np.float32),
        "oracle_exposures": np.asarray(MIDPOINT_ORACLE_EVS, dtype=np.float32),
        "oracle_truth": np.stack(truths).astype(np.float32),
        "oracle_p99_stop": np.float32(worst_p99),
        "oracle_max_stop": np.float32(worst_stop_max),
        "premix_refuted_note": np.asarray(PREMIX_REFUTATION),
        "source_names": np.asarray(sorted([stock["negative"], paper_name])),
        "source_sha256": np.asarray([
            _source_sha(n) for n in sorted([stock["negative"], paper_name])
        ]),
        "builder_commit": np.asarray(_builder_commit()),
        "input_space": np.asarray("negative_density"),
        "schema": np.int32(SCHEMA),
        "n": np.int32(GRID_N),
    }


def build_stock_asset(stock_key: str) -> dict:
    """Stage A tables + references (no volumes)."""
    theatrical = stock_key.endswith("_theatrical")
    stock = dict(ff.STOCKS[stock_key.removesuffix("_theatrical")])
    stock["key"] = stock_key
    chain = v1._Chain(ff.STOCKS[stock_key.removesuffix("_theatrical")], theatrical)
    observer, obs_p99, obs_cv = _observer(stock)
    neg, char_le, char_amounts, lo, hi = _stage_a_tables(stock)
    reversal = bool(stock.get("positive"))
    default_medium = _medium_id(stock, theatrical)
    media = [default_medium]
    if not reversal:
        base = stock_key.removesuffix("_theatrical")
        for extra in EXTRA_PAIRINGS.get(base, ()):
            media.append(_medium_id(stock, theatrical, extra))
    # Scene-side oracle for the fixed path (tau(0) / reversal direct).
    rng = np.random.default_rng(20260806)
    oracle_ev = rng.uniform(-9.0, 5.0, (96, 3))
    oracle_rgb = SCENE_MID * np.exp2(oracle_ev)
    a = observer
    truth = v1.chain_eval(ff.STOCKS[stock_key.removesuffix("_theatrical")], chain, a, oracle_rgb)
    entry = {
        "kind": np.asarray("stock"),
        "stock": np.asarray(stock_key),
        "observer": observer.astype(np.float64),
        "observer_p99_stop": np.float32(obs_p99),
        "observer_cv_p99_stop": np.float32(obs_cv),
        "char_le": char_le,
        "char_amounts": char_amounts,
        "amount_lo": lo,
        "amount_hi": hi,
        "anchor_ev_offset": np.float64(_reversal_anchor(stock, chain)),
        "exposure_ev_min": np.float32(EXPOSURE_EV_MIN),
        "exposure_ev_max": np.float32(EXPOSURE_EV_MAX),
        "reversal": np.bool_(reversal),
        "default_medium": np.asarray(default_medium),
        "media": np.asarray(media),
        "oracle_ev": oracle_ev.astype(np.float32),
        "oracle_truth": truth.astype(np.float32),
        "source_names": np.asarray([stock["negative"]]),
        "source_sha256": np.asarray([_source_sha(stock["negative"])]),
        "builder_commit": np.asarray(_builder_commit()),
        "input_space": np.asarray("scene_rec2020"),
        "schema": np.int32(SCHEMA),
    }
    if reversal:
        # Reversal cast (fixed path) stays with the stock.
        ramp_ev = np.linspace(v1.EV_MIN, v1.EV_MAX, 257)
        ramp_rgb = SCENE_MID * np.exp2(ramp_ev)[:, None].repeat(3, axis=1)
        ramp_out = v1.chain_eval(ff.STOCKS[stock_key], chain, a, ramp_rgb)
        y = ramp_out @ _LUMA
        cast = ramp_out / np.maximum(y, 1e-9)[:, None]
        h_star = 1.0 / np.maximum(cast, 1e-9)
        t_hi = np.where(h_star > 4.0, 3.0 / np.maximum(h_star - 1.0, 1e-9), 1.0)
        t_lo = np.where(h_star < 0.25, 0.75 / np.maximum(1.0 - h_star, 1e-9), 1.0)
        t_ev = np.clip(np.min(np.minimum(t_hi, t_lo), axis=1), 0.0, 1.0)
        h = 1.0 + t_ev[:, None] * (h_star - 1.0)
        step = (ramp_ev.size - 1) // (GRID_N - 1)
        entry["cast_ev"] = ramp_ev[::step].astype(np.float32)
        entry["cast_bounded"] = (1.0 / np.clip(h, 0.25, 4.0))[::step].astype(np.float32)
    return entry


def _write(path: Path, data: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def build_all(stocks: list[str] | None = None) -> None:
    keys = stocks
    if not keys:
        keys = []
        for key in ff.STOCKS:
            keys.append(key)
            stock = ff.STOCKS[key]
            if (not stock.get("positive")
                    and ff.PRINT_SURROUND.get(str(stock.get("print"))) == "dark"):
                keys.append(f"{key}_theatrical")
    b2_done: set[str] = set()
    for key in keys:
        theatrical = key.endswith("_theatrical")
        base = key.removesuffix("_theatrical")
        stock = dict(ff.STOCKS[base]); stock["key"] = key
        entry = build_stock_asset(key)
        _write(OUT_DIR / f"{key}.npz", entry)
        reversal = bool(stock.get("positive"))
        if reversal:
            mid = _medium_id(stock, False)
            if mid not in b2_done:
                _write(OUT_DIR / f"b2__{mid}.npz", build_b2_reversal(base))
                b2_done.add(mid)
            print(f"{key}: stock + {mid}", flush=True)
            continue
        papers = [stock["print"], *EXTRA_PAIRINGS.get(base, ())]
        for paper in papers:
            mid = _medium_id(stock, theatrical, paper)
            if mid not in b2_done:
                _write(OUT_DIR / f"b2__{mid}.npz", build_b2_negative(paper, theatrical))
                b2_done.add(mid)
            ps = build_print_state(key, paper, theatrical)
            _write(OUT_DIR / f"print__{key}__{mid}.npz", ps)
            print(
                f"{key} x {mid}: oracle p99 {float(ps['oracle_p99_stop']):.4f} / "
                f"max {float(ps['oracle_max_stop']):.4f} stop", flush=True,
            )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", nargs="*", default=None)
    args = ap.parse_args()
    build_all(args.stocks)
