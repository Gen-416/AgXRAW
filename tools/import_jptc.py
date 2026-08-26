#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Import JPTC/2 first-party sensor measurements into dngscan priors.

JPTC/2 is the CSV format written by y-g-jiang's "JPTC Collect" acquisition
tool (and served by his Jiangtherapee Online Table). Its discipline matches
ours: the collector records only raw per-frame statistics (four-channel
mean/std per exposure, per-channel black level in the header) and leaves
every derived quantity to the analysis side. This importer IS an analysis
side: it fits gain / read noise / full well from the raw points and writes a
dngscan priors entry with the aperture and fit residuals declared.

Method (standard photon-transfer analysis, G1 channel):
  - black level: from the CSV header (collector-measured, per channel);
  - saturation S_sat: the clip plateau (max mean at the declared white);
  - shot-noise fit: PRIMARY = linear-prnu-corrected (trimmed OLS over
    S < 0.10*S_sat on var - prnu_top^2*S^2, iterated); when the ramp has
    fewer than 3 unsaturated top points the correction is UNRESOLVED and
    the effective path is plain linear — declared via prnu_status and
    fit_model_effective, never implied. linear-0.10 and quadratic-0.35
    are recorded as alternatives; gain_estimator_spread_rel is their
    range over the primary (an estimator spread, NOT a statistical
    uncertainty — the estimators share the data and the model set is
    not exhaustive);
  - fwc_e = (white - black) * g: ADC code-saturation capacity (exact
    given the fitted gain). No clip-onset field is published: JPTC/2
    records no per-step exposure, so the scene-exposure onset is not
    recoverable from this input (only last_unsaturated_signal_e is).

Usage:
    python tools/import_jptc.py measurement.csv --brand Sony --model "A7 V" \\
        --iso 100 --out dngscan/data/priors/jptc/<id>.json
    python tools/import_jptc.py --self-test     # synthetic-sensor gate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def parse_jptc_csv(path: Path) -> dict:
    header: dict = {}
    rows = []
    cols: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            header[key.strip()] = value.strip()
            continue
        if not cols:
            cols = [c.strip() for c in line.split(",")]
            continue
        parts = line.split(",")
        rows.append(dict(zip(cols, parts)))
    if "BlackLevel" not in header:
        raise ValueError("JPTC/2 header missing BlackLevel")
    black = [float(v) for v in header["BlackLevel"].split(",")]
    return {"header": header, "black": black, "rows": rows}


def fit_ptc(
    means: np.ndarray,
    stds: np.ndarray,
    black: float,
    white: float,
) -> dict:
    """Photon-transfer fit on one channel's per-frame (mean, std) points."""
    signal = means - black
    var = stds**2
    sat_plateau = white - black
    # Clip plateau detection: points whose mean sits within 0.5% of white are
    # saturated (their std collapses); S_sat is the highest UNSATURATED mean.
    unsat = means < white * 0.995
    if int(unsat.sum()) < 6:
        raise ValueError("not enough unsaturated exposure steps for a PTC fit")
    s_last = float(signal[unsat].max())
    # The true clip plateau sits between the last unsaturated step and the
    # plateau itself; a stepped exposure ramp cannot see inside that bracket,
    # so report the midpoint WITH the half-bracket as declared uncertainty
    # instead of quoting the lower bound as if it were exact.
    # No midpoint estimand survives (R10 item 2): the old s_sat midpoint was
    # a function of the exposure-step density and silently steered the fit
    # windows (S5M2 gain moved -2.7% between window conventions). All
    # windows now reference the ADC code capacity (white - black), which is
    # a property of the camera, not of the ramp; saturation exclusion stays
    # with the unsat mask.
    # Model set (external review 2026-08-25, revised after re-validation):
    # the review falsified the old "PRNU two decades below shot noise"
    # comment — at 0.10*S_sat the PRNU term reaches ~4.5-8.4% of the shot
    # term (R = prnu^2*g*S), biasing a plain linear fit's gain low by a few
    # percent. Three estimators are computed and DECLARED:
    #   A "linear-0.10"   : trimmed OLS over S<0.10*S_sat (legacy);
    #   B "quadratic-0.35": constrained var=a+bS+cS^2 (c>=0) over S<0.35 —
    #     best residuals, but c absorbs any mid-ramp variance structure
    #     (R6 II: 29% gain swing) and it WORSENED the A7RM6 cross-instrument
    #     check from 3.7% to 7.5%, so it is not primary;
    #   C "linear-prnu-corrected" (PRIMARY): iterate {linear fit over
    #     S<0.10*S_sat on y - prnu_top^2*S^2; re-estimate prnu_top at the
    #     ramp top with the new gain} — removes the KNOWN bias term without
    #     granting the fit freedom to absorb unrelated structure.
    # model_sensitivity = (max-min)/primary across the three gains.
    def _trimmed_linear(xa, ya, min_keep=4):
        keep = np.ones(len(xa), dtype=bool)
        slope = intercept = 0.0
        for _ in range(3):
            slope, intercept = np.polyfit(xa[keep], ya[keep], 1)
            r = ya - np.polyval([slope, intercept], xa)
            rms = float(np.sqrt(np.mean(r[keep] ** 2)))
            new_keep = np.abs(r) <= 2.5 * rms
            if int(new_keep.sum()) < min_keep or bool((new_keep == keep).all()):
                break
            keep = new_keep
        resid = float(np.sqrt(np.mean(
            (np.polyval([slope, intercept], xa[keep]) - ya[keep]) ** 2))
            / max(np.mean(ya[keep]), 1e-9))
        return slope, intercept, keep, resid

    lo_mask = unsat & (signal > 0) & (signal < 0.10 * sat_plateau)
    if int(lo_mask.sum()) < 4:
        lo_mask = unsat & (signal > 0) & (signal < 0.35 * sat_plateau)
    x = signal[lo_mask]
    y = var[lo_mask]
    if int(lo_mask.sum()) < 4:
        raise ValueError("not enough points for a PTC fit")

    def _prnu_top(g, vr):
        top_m = unsat & (signal > 0.5 * sat_plateau)
        if int(top_m.sum()) < 3:
            return 0.0
        excess = var[top_m] - signal[top_m] / g - vr
        with np.errstate(invalid="ignore"):
            pts = np.sqrt(np.clip(excess, 0, None)) / signal[top_m]
        return float(np.median(pts))

    # A: plain linear on the low decade
    slope_a, icpt_a, _, resid_a = _trimmed_linear(x, y)
    if slope_a <= 0:
        raise ValueError("non-physical PTC slope; measurement unusable")
    gain_a = 1.0 / float(slope_a)
    # C: PRNU-corrected linear (primary), iterated to convergence (max 16;
    # the review found 3 rounds left A7M5 0.011% short of its own gate)
    n_top = int((unsat & (signal > 0.5 * sat_plateau)).sum())
    gain = gain_a
    var_read = max(float(icpt_a), 0.0)
    prnu = _prnu_top(gain, var_read)
    keep = np.ones(len(x), dtype=bool)
    resid = resid_a
    prnu_iterations = 0
    prnu_converged = False
    prnu_final_delta = 0.0
    if n_top >= 3:
        for prnu_iterations in range(1, 17):
            y_corr = y - (prnu * x) ** 2
            slope, icpt, keep, resid = _trimmed_linear(x, y_corr)
            if slope <= 0:
                raise ValueError("non-physical PTC slope; measurement unusable")
            gain_new = 1.0 / float(slope)
            var_read = max(float(icpt), 0.0)
            prnu_new = _prnu_top(gain_new, var_read)
            prnu_final_delta = abs(gain_new - gain) / gain
            converged = prnu_final_delta < 1e-4 and abs(prnu_new - prnu) < 1e-5
            gain, prnu = gain_new, prnu_new
            if converged:
                prnu_converged = True
                break
    if n_top < 3:
        # top of ramp not sampled -> the correction cannot run at all
        prnu_status = "unresolved"
        fit_model_effective = "linear-0.10 (prnu unresolved -> plain linear)"
    elif not prnu_converged:
        # 16 rounds without meeting the gate: fail closed (R10 item 5) —
        # an unconverged correction must not be labelled corrected
        prnu_status = "unconverged"
        fit_model_effective = "linear-prnu-corrected (UNCONVERGED)"
    elif prnu == 0.0:
        prnu_status = "zero"
        fit_model_effective = "linear-prnu-corrected (correction = 0)"
    else:
        prnu_status = "corrected"
        fit_model_effective = "linear-prnu-corrected"
    # B: constrained quadratic over the wide range (recorded, not primary)
    wide = unsat & (signal > 0) & (signal < 0.35 * sat_plateau)
    xw, yw = signal[wide], var[wide]
    gain_q = float("nan")
    prnu_q = float("nan")
    resid_q = float("nan")
    if int(wide.sum()) >= 5:
        A = np.stack([np.ones_like(xw), xw, xw * xw], axis=1)
        cq, *_ = np.linalg.lstsq(A, yw, rcond=None)
        if cq[2] < 0:
            c1 = np.polyfit(xw, yw, 1)
            cq = np.array([c1[1], c1[0], 0.0])
        if cq[1] > 0:
            gain_q = 1.0 / float(cq[1])
            prnu_q = float(np.sqrt(max(cq[2], 0.0)))
            predw = cq[0] + cq[1] * xw + cq[2] * xw * xw
            resid_q = float(np.sqrt(np.mean((predw - yw) ** 2))
                            / max(np.mean(yw), 1e-9))
    gains = [g for g in (gain, gain_a, gain_q) if math.isfinite(g)]
    # estimator RANGE over the primary — an estimator spread, not a
    # statistical uncertainty (review P2-3): the estimators are correlated,
    # share the data, and the model set is not exhaustive.
    gain_estimator_spread_rel = (max(gains) - min(gains)) / gain
    # PRNU cross-estimate from the top of the ramp (median excess variance),
    # kept alongside the quadratic-coefficient estimate as a consistency
    # check between apertures.
    top = unsat & (signal > 0.5 * sat_plateau)
    prnu = float("nan")
    if int(top.sum()) >= 3:
        excess = var[top] - signal[top] / gain - var_read
        with np.errstate(invalid="ignore"):
            prnu_pts = np.sqrt(np.clip(excess, 0, None)) / signal[top]
        prnu = float(np.median(prnu_pts))
    # Capacity semantics (external review 2026-08-25): the exposure-step
    # bracket bounds the CLIP-ONSET scene exposure, not the code-white
    # capacity. fwc_e is therefore the ADC code-saturation capacity,
    #     fwc_e = (white - black) * gain,
    # exact given the fitted gain (no bracket needed); whether the PHYSICAL
    # full well or the ADC clips first is not knowable from clipped codes,
    # so no physical-full-well claim is made. The clip-onset bracket stays
    # available as s_sat_dn +/- s_sat_dn_uncertainty.
    read_noise_e = math.sqrt(var_read) * gain if var_read > 0 else None
    return {
        "gain_e_per_dn": gain,
        # key name "fit_model", NOT "model": import_csv merges this dict
        # with **fit and a "model" key would overwrite the CAMERA model
        "fit_model": "linear-prnu-corrected over S<0.10*S_sat (primary)",
        "fit_model_effective": fit_model_effective,
        "gain_alternatives": {"linear-0.10": gain_a,
                              "quadratic-0.35": gain_q},
        "gain_estimator_spread_rel": gain_estimator_spread_rel,
        "prnu_status": prnu_status,
        "prnu_iterations": prnu_iterations,
        "prnu_converged": prnu_converged,
        "prnu_final_delta": prnu_final_delta,
        "read_noise_dn": math.sqrt(var_read),
        "read_noise_e": read_noise_e if read_noise_e is not None else 0.0,
        "read_noise_status": "measured" if var_read > 0 else "below-resolution",
        "fwc_e": (white - black) * gain,
        "fwc_model_spread_e": (white - black) * gain * gain_estimator_spread_rel,
        "fwc_semantics": "ADC code-saturation capacity (white-black)*gain; "
                         "fwc_model_spread_e = capacity x estimator spread "
                         "(model-choice spread, not a statistical "
                         "uncertainty); physical full well not claimed",
        # No clip-onset field: JPTC/2 has no per-step exposure column, so
        # the scene-exposure onset is unrecoverable (review P1-2). The last
        # unsaturated observation is published as a lower bound only.
        "last_unsaturated_signal_e": s_last * gain,
        "prnu": (prnu if math.isfinite(prnu) and prnu_status != "unresolved" else None),
        "prnu_quadratic_fit": prnu_q if math.isfinite(prnu_q) else None,
        "fit_relative_rms": resid,
        "fit_relative_rms_alternatives": {"linear-0.10": resid_a,
                                          "quadratic-0.35": resid_q},
        "fit_points": int(keep.sum()),
        "fit_points_excluded": int((~keep).sum()),
        "sat_plateau_dn": sat_plateau,
        "quality": ("high-residual" if resid > 0.05
                    else "unconverged" if prnu_status == "unconverged"
                    else "ok"),
    }


def sanitize_json(obj):
    """NaN/Inf -> None recursively: RFC 8259 has no NaN literal, and
    Python's default json.dumps writes one anyway (review P2-2); dumps are
    paired with allow_nan=False so a regression fails loudly."""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def infer_white(means: np.ndarray, stds: np.ndarray) -> float | None:
    """Clip level from the data itself. Cameras differ (Sony 16383, Nikon
    full-scale 14-bit, Panasonic RW2 scaled to ~65430 with dither), so a
    hardcoded white silently skews the FWC bracket. A clip plateau shows as
    repeated top means: the exposure ramp is geometric (adjacent steps >=5%
    apart), so >=2 frames agreeing within 0.2% of the maximum can only be
    saturation. Zero spatial std at the top is accepted as a plateau too
    (hard clip without dither)."""
    top = float(means.max())
    plateau = means >= top * 0.998
    if int(plateau.sum()) >= 2 or bool((stds[plateau] == 0.0).any()):
        return top
    return None


def import_csv(
    path: Path, brand: str, model: str, iso: int, white: float | None,
    shutter: str = "",
) -> dict:
    parsed = parse_jptc_csv(path)
    rows = parsed["rows"]
    g1_mean = np.asarray([float(r["G1_Mean"]) for r in rows])
    g1_std = np.asarray([float(r["G1_Std"]) for r in rows])
    black_g1 = parsed["black"][1] if len(parsed["black"]) >= 2 else parsed["black"][0]
    if white is None:
        white = infer_white(g1_mean, g1_std)
        if white is None:
            raise ValueError(
                f"{path.name}: no saturated frames to infer the clip level "
                "from; pass --white explicitly"
            )
    fit = fit_ptc(g1_mean, g1_std, black_g1, white)
    return {
        "format": "dngscan-jptc-prior-1",
        "id": f"{brand} {model} (JPTC)",
        "brand": brand,
        "model": model,
        "iso": int(iso),
        "shutter": shutter or None,
        "white_level_used": white,
        "channel": "G1",
        "black_level_g1": black_g1,
        "noise_aperture": "single-frame spatial std (includes PRNU; fit "
                          "restricted to the shot-noise decades)",
        "source": {
            "kind": "JPTC/2 first-party measurement",
            "file": path.name,
            "input_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "mode": parsed["header"].get("Mode"),
            "geometry": [
                parsed["header"].get("ImageWidth"),
                parsed["header"].get("ImageHeight"),
            ],
        },
        **fit,
    }


def self_test() -> int:
    """Synthetic-sensor gate: known parameters must be recovered.

    Truth definitions fixed per the 2026-08-25 external review: the earlier
    version compared the bracket midpoint against an arbitrary 0.97*white
    "truth" — circular. Now: fwc_e (code-saturation capacity) has the exact
    truth (white-black)*gain; the clip-onset bracket must CONTAIN the true
    onset (which for a hard clip at white is the bracket's upper edge)."""
    gain_true, rn_e_true, black, white = 0.42, 3.1, 512.0, 16383.0
    prnu_true = 0.006
    fwc_true = (white - black) * gain_true          # code-saturation capacity
    means, stds = [], []
    for step in np.geomspace(2.0, (white - black) * 1.15, 40):
        s_e = step * gain_true
        var_e = s_e + rn_e_true**2 + (prnu_true * s_e) ** 2
        mean_dn = min(black + step, white)
        std_dn = 0.0 if mean_dn >= white else math.sqrt(var_e) / gain_true
        means.append(mean_dn)
        stds.append(std_dn)
    fit = fit_ptc(np.asarray(means), np.asarray(stds), black, white)
    checks = [
        ("gain", fit["gain_e_per_dn"], gain_true, 0.03),
        ("read_noise_e", fit["read_noise_e"], rn_e_true, 0.15),
        ("prnu(top-of-ramp)", fit["prnu"], prnu_true, 0.30),
        ("prnu(quadratic)", fit["prnu_quadratic_fit"], prnu_true, 0.30),
        ("fwc_e(code capacity)", fit["fwc_e"], fwc_true, 0.03),
    ]
    ok = True
    for name, got, want, tol in checks:
        rel = abs(got - want) / want
        status = "ok" if rel <= tol else "FAIL"
        ok &= rel <= tol
        print(f"  {name}: got {got:.4g} want {want:.4g} rel {rel:.3f} [{status}]")
    # No clip-onset assertion: the field was removed (unrecoverable from
    # JPTC/2 inputs). The last unsaturated observation must be a strict
    # lower bound on capacity, and the prnu path must actually run here.
    ok &= 0 < fit["last_unsaturated_signal_e"] <= fit["fwc_e"]
    ok &= fit["prnu_status"] == "corrected" and fit["prnu_converged"]
    print(f"  last_unsaturated_signal_e: {fit['last_unsaturated_signal_e']:.4g} "
          f"<= fwc {fit['fwc_e']:.4g} [ok]; prnu_status={fit['prnu_status']} "
          f"iterations={fit['prnu_iterations']} converged={fit['prnu_converged']}")
    alts = fit["gain_alternatives"]
    print(f"  gain_estimator_spread_rel: {fit['gain_estimator_spread_rel']:.4f} "
          f"(primary {fit['gain_e_per_dn']:.4g}, linear-0.10 "
          f"{alts['linear-0.10']:.4g}, quad-0.35 {alts['quadratic-0.35']:.4g})")
    # R10 item 2: the gain estimate must be invariant to the exposure-step
    # density of the ramp (the old midpoint-referenced windows were not —
    # S5M2 moved -2.7% between conventions).
    def _ramp(n_steps):
        ms, sds = [], []
        for step in np.geomspace(2.0, (white - black) * 1.15, n_steps):
            s_e = step * gain_true
            var_e = s_e + rn_e_true**2 + (prnu_true * s_e) ** 2
            mean_dn = min(black + step, white)
            sd = 0.0 if mean_dn >= white else math.sqrt(var_e) / gain_true
            ms.append(mean_dn); sds.append(sd)
        return fit_ptc(np.asarray(ms), np.asarray(sds), black, white)
    f40, f80 = _ramp(40), _ramp(80)
    dens_inv = (abs(f40["gain_e_per_dn"] - f80["gain_e_per_dn"])
                / f80["gain_e_per_dn"] < 0.005)
    ok &= dens_inv
    print(f"  ramp-density invariance: gain(40) {f40['gain_e_per_dn']:.4f} vs "
          f"gain(80) {f80['gain_e_per_dn']:.4f} [{'ok' if dens_inv else 'FAIL'}]")
    # a ramp too sparse to sample the top must DECLARE the fallback, not
    # silently blend paths (this is the fail-closed contract, not a bug)
    f30 = _ramp(30)
    sparse_ok = f30["prnu_status"] == "unresolved"
    ok &= sparse_ok
    print(f"  sparse-ramp declaration: prnu_status={f30['prnu_status']} "
          f"[{'ok' if sparse_ok else 'FAIL'}]")
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", type=Path)
    ap.add_argument("--brand", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--iso", type=int, default=0)
    ap.add_argument("--shutter", default="", help="mechanical / electronic (for tier dedup preference)")
    ap.add_argument("--white", type=float, default=None,
                    help="clip level in DN; default: inferred from zero-std saturated frames")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.csv is None:
        ap.error("csv path required (or --self-test)")
    entry = import_csv(args.csv, args.brand, args.model, args.iso, args.white, args.shutter)
    entry = sanitize_json(entry)
    text = json.dumps(entry, indent=1, ensure_ascii=False, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
