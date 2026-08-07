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

_LUMA = np.array([0.2627, 0.6780, 0.0593])


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


def build_stock(stock_key: str) -> dict:
    stock = ff.STOCKS[stock_key]
    chain = v1._Chain(stock, theatrical=False)
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

    return {
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
    ap.add_argument("--stocks", nargs="*", default=list(PILOT_STOCKS))
    args = ap.parse_args()
    for key in args.stocks:
        write_stock(key)
