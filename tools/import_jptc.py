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

RawSize is retained as a LibRaw full-mosaic readout constraint. Compression
declarations are retained for runtime verification; unfamiliar text stays
unverified. ImageWidth/ImageHeight describe the camera's JPEG output only.
This single-frame PTC does not establish Collect's paired-dark total variance.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dngscan.calibration import (  # noqa: E402
    fit_ptc, import_csv, infer_white, parse_jptc_csv, sanitize_json,
)


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
    lin_key = next(k for k in alts if k.startswith("linear-"))
    print(f"  gain_estimator_spread_rel: {fit['gain_estimator_spread_rel']:.4f} "
          f"(primary {fit['gain_e_per_dn']:.4g}, {lin_key} "
          f"{alts[lin_key]:.4g}, quad-0.35 {alts['quadratic-0.35']:.4g})")
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
    ap.add_argument("--shutter", default="", help="declared calibration shutter mode: mechanical / electronic / efcs")
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
