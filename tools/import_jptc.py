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
  - shot-noise fit: in the unsaturated region, var(DN) = S/g + var_read
    -> weighted linear fit of variance vs signal gives gain g (e-/DN) and
    the dark-end intercept gives read noise; single-frame stds include
    PRNU, which biases the top of the ramp, so the fit uses the lower
    signal decades and reports the residual;
  - FWC = (S_sat - black) * g;  read_noise_e = sqrt(var_read) * g.

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
    s_upper = float(min(sat_plateau, signal.max())) if bool((~unsat).any()) else s_last
    s_sat = 0.5 * (s_last + s_upper)
    s_sat_unc = 0.5 * (s_upper - s_last)
    # Shot-noise region: below 10% of saturation the PRNU term (prnu^2 * S^2)
    # is at least two decades under the shot term, so a LINEAR var-vs-signal
    # fit stays honest with single-frame stds. Dark end included for the
    # intercept.
    fit_mask = unsat & (signal > 0) & (signal < 0.10 * s_sat)
    if int(fit_mask.sum()) < 4:
        fit_mask = unsat & (signal > 0) & (signal < 0.35 * s_sat)
    x = signal[fit_mask]
    y = var[fit_mask]
    slope, intercept = np.polyfit(x, y, 1)
    if slope <= 0:
        raise ValueError("non-physical PTC slope; measurement unusable")
    gain = 1.0 / float(slope)              # e- per DN
    var_read = max(float(intercept), 0.0)
    resid = float(
        np.sqrt(np.mean((np.polyval([slope, intercept], x) - y) ** 2))
        / max(np.mean(y), 1e-9)
    )
    # PRNU from the top of the ramp: var_excess = var - S/g - var_read,
    # prnu = sqrt(var_excess)/S evaluated over the upper unsaturated decade.
    top = unsat & (signal > 0.5 * s_sat)
    prnu = float("nan")
    if int(top.sum()) >= 3:
        excess = var[top] - signal[top] / gain - var_read
        with np.errstate(invalid="ignore"):
            prnu_pts = np.sqrt(np.clip(excess, 0, None)) / signal[top]
        prnu = float(np.median(prnu_pts))
    return {
        "gain_e_per_dn": gain,
        "read_noise_dn": math.sqrt(var_read),
        "read_noise_e": math.sqrt(var_read) * gain,
        "s_sat_dn": s_sat,
        "s_sat_dn_uncertainty": s_sat_unc,
        "fwc_e": s_sat * gain,
        "fwc_e_uncertainty": s_sat_unc * gain,
        "prnu": prnu,
        "fit_relative_rms": resid,
        "fit_points": int(fit_mask.sum()),
        "sat_plateau_dn": sat_plateau,
    }


def import_csv(path: Path, brand: str, model: str, iso: int, white: float) -> dict:
    parsed = parse_jptc_csv(path)
    rows = parsed["rows"]
    g1_mean = np.asarray([float(r["G1_Mean"]) for r in rows])
    g1_std = np.asarray([float(r["G1_Std"]) for r in rows])
    black_g1 = parsed["black"][1] if len(parsed["black"]) >= 2 else parsed["black"][0]
    fit = fit_ptc(g1_mean, g1_std, black_g1, white)
    return {
        "format": "dngscan-jptc-prior-1",
        "id": f"{brand} {model} (JPTC)",
        "brand": brand,
        "model": model,
        "iso": int(iso),
        "channel": "G1",
        "black_level_g1": black_g1,
        "noise_aperture": "single-frame spatial std (includes PRNU; fit "
                          "restricted to the shot-noise decades)",
        "source": {
            "kind": "JPTC/2 first-party measurement",
            "file": path.name,
            "mode": parsed["header"].get("Mode"),
            "geometry": [
                parsed["header"].get("ImageWidth"),
                parsed["header"].get("ImageHeight"),
            ],
        },
        **fit,
    }


def self_test() -> int:
    """Synthetic-sensor gate: known gain/read-noise/FWC must be recovered."""
    rng = np.random.default_rng(7)
    gain_true, rn_e_true, black, white = 0.42, 3.1, 512.0, 16383.0
    fwc_true = (white * 0.97 - black) / (1.0 / gain_true)  # e- at the plateau
    prnu_true = 0.006
    means, stds = [], []
    for step in np.geomspace(2.0, (white - black) * 1.15, 40):
        s_e = step / (1.0 / gain_true)  # electrons at this exposure
        var_e = s_e + rn_e_true**2 + (prnu_true * s_e) ** 2
        mean_dn = min(black + step, white)
        std_dn = 0.0 if mean_dn >= white else math.sqrt(var_e) / gain_true
        means.append(mean_dn)
        stds.append(std_dn)
    fit = fit_ptc(np.asarray(means), np.asarray(stds), black, white)
    checks = [
        ("gain", fit["gain_e_per_dn"], gain_true, 0.05),
        ("read_noise_e", fit["read_noise_e"], rn_e_true, 0.15),
        ("prnu", fit["prnu"], prnu_true, 0.30),
    ]
    ok = True
    for name, got, want, tol in checks:
        rel = abs(got - want) / want
        status = "ok" if rel <= tol else "FAIL"
        ok &= rel <= tol
        print(f"  {name}: got {got:.4g} want {want:.4g} rel {rel:.3f} [{status}]")
    # FWC contract: the truth must sit inside the declared bracket.
    lo = fit["fwc_e"] - fit["fwc_e_uncertainty"] - 1e-9
    hi = fit["fwc_e"] + fit["fwc_e_uncertainty"] + 1e-9
    in_bracket = lo <= fwc_true <= hi
    ok &= in_bracket
    print(f"  fwc_e: got {fit['fwc_e']:.4g} ± {fit['fwc_e_uncertainty']:.3g} "
          f"want {fwc_true:.4g} [{'ok' if in_bracket else 'FAIL'}]")
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", type=Path)
    ap.add_argument("--brand", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--iso", type=int, default=0)
    ap.add_argument("--white", type=float, default=16383.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.csv is None:
        ap.error("csv path required (or --self-test)")
    entry = import_csv(args.csv, args.brand, args.model, args.iso, args.white)
    text = json.dumps(entry, indent=1, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
