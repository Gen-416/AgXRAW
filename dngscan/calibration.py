# SPDX-License-Identifier: GPL-3.0-or-later
"""User-imported JPTC calibration and canonical measurement conversion.

Profiles are local user data, never written into the installed package. A
measured profile is usable only for its exact camera, declared readout mode
and measured ISO domain. Missing evidence is explicit; scalar green-channel
measurements are not promoted to RGB or to a fixed-pattern correction map.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

GREEN_INDICES = {1, 3}
CLIP_FRAC_MAX = 0.01

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
    # saturated (their std collapses). s_last, the highest UNSATURATED signal,
    # is published as a lower bound only; the fit windows reference
    # sat_plateau = white - black, not any ramp statistic.
    unsat = means < white * 0.995
    if int(unsat.sum()) < 6:
        raise ValueError("not enough unsaturated exposure steps for a PTC fit")
    s_last = float(signal[unsat].max())
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
    # gain_estimator_spread_rel = (max-min)/primary across the three gains.
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
        # The returned slope must be the one fitted on the returned keep set:
        # when all three rounds changed the set, the loop used to exit with a
        # slope from the previous set (self-review 2026-08-27).
        slope, intercept = np.polyfit(xa[keep], ya[keep], 1)
        resid = float(np.sqrt(np.mean(
            (np.polyval([slope, intercept], xa[keep]) - ya[keep]) ** 2))
            / max(np.mean(ya[keep]), 1e-9))
        return slope, intercept, keep, resid

    lo_mask = unsat & (signal > 0) & (signal < 0.10 * sat_plateau)
    fit_window_frac = 0.10
    if int(lo_mask.sum()) < 4:
        # Sparse ramp: widen to 0.35 of capacity and SAY SO — the published
        # fit_model used to claim the 0.10 window regardless (self-review
        # 2026-08-27).
        lo_mask = unsat & (signal > 0) & (signal < 0.35 * sat_plateau)
        fit_window_frac = 0.35
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
        fit_model_effective = f"linear-{fit_window_frac:.2f} (prnu unresolved -> plain linear)"
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
    # so no physical-full-well claim is made. No clip-onset bracket is
    # published; last_unsaturated_signal_e stands as the lower bound.
    read_noise_e = math.sqrt(var_read) * gain if var_read > 0 else None
    return {
        "gain_e_per_dn": gain,
        # key name "fit_model", NOT "model": import_csv merges this dict
        # with **fit and a "model" key would overwrite the CAMERA model
        "fit_model": (
            f"linear-prnu-corrected over S<{fit_window_frac:.2f}*S_sat (primary)"
        ),
        # The window the fit ACTUALLY used (0.10 nominal; 0.35 when the ramp is
        # too sparse below 0.10 — self-review 2026-08-27).
        "fit_window_frac": fit_window_frac,
        "fit_model_effective": fit_model_effective,
        # keyed by the window ACTUALLY fitted (review R5 item 4: a sparse ramp
        # widens to 0.35 and the alternative must not still be called 0.10)
        "gain_alternatives": {f"linear-{fit_window_frac:.2f}": gain_a,
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
        "fit_relative_rms_alternatives": {f"linear-{fit_window_frac:.2f}": resid_a,
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

def _parse_rows(path: Path) -> tuple[dict, list[dict]]:
    header: dict = {}
    rows: list[dict] = []
    cols: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, sep, value = line[1:].partition(":")
            if sep and " " not in key.strip():
                header[key.strip()] = value.strip().rstrip(",")
            continue
        if not cols:
            cols = [c.strip() for c in line.split(",")]
            continue
        rows.append(dict(zip(cols, line.split(","))))
    return header, rows

def read_dark(path: Path) -> tuple[dict, dict]:
    """{iso: {"bl": mean green BL, "rn_dn": temporal read noise,
              "row_var": ..., "col_var": ..., "total_var": ...}}"""
    header, rows = _parse_rows(path)
    clip_factor_status = "declared" if header.get("ClipVarianceFactor") else "unresolved"
    try:
        factor = float(header.get("ClipVarianceFactor", 1.0))
    except ValueError:
        # old collector wrote 'undefined': the sigma-clip bias is NOT undone
        # and every product must say so (review P2-5), not claim it was
        factor = 1.0
        clip_factor_status = "unresolved"
    if not math.isfinite(factor) or factor <= 0 or factor > 1:
        raise ValueError("ClipVarianceFactor must be in (0,1]")
    header["_clip_factor_status"] = clip_factor_status
    try:
        adc_step = float(header.get("AdcStep", 0.0))
    except ValueError:
        adc_step = 0.0
    if not math.isfinite(adc_step) or adc_step < 0:
        raise ValueError("AdcStep must be nonnegative and finite")
    per_iso: dict[int, dict] = {}
    greens_present = any(int(r["ColorIndex"]) in GREEN_INDICES for r in rows)
    for r in rows:
        # monochrome sensors (e.g. M11 Monochrom) use a single colour index;
        # fall back to every plane when no green-indexed rows exist
        if greens_present and int(r["ColorIndex"]) not in GREEN_INDICES:
            continue
        iso = int(r["ISO"])
        if iso <= 0 or int(r["ColorIndex"]) not in range(4):
            raise ValueError("dark rows require positive ISO and CFA index 0..3")
        for key in ("BlackA", "BlackB", "StdDiffClipped"):
            if not math.isfinite(float(r[key])) or (key == "StdDiffClipped" and float(r[key]) < 0):
                raise ValueError(f"invalid dark statistic: {key}")
        d = per_iso.setdefault(iso, {"bl": [], "rn": [], "row": [], "col": [], "tot": [], "qfrac": []})
        d["bl"].append(0.5 * (float(r["BlackA"]) + float(r["BlackB"])))
        # temporal read noise: undo the declared sigma clip, halve the
        # difference variance, and apply Sheppard's quantisation correction
        # (the collector leaves it recomputable by design; each frame adds
        # step^2/12 of quantisation variance, valid where the linearisation
        # is step-uniform, which holds at black level for every set here).
        var_diff = (float(r["StdDiffClipped"]) ** 2) / factor
        var_t = var_diff / 2.0
        qfrac = 0.0
        if adc_step > 0 and var_t > 0:
            q = (adc_step ** 2) / 12.0
            qfrac = q / var_t
            var_t -= q
        # a correction that floors the variance means the read noise is
        # UNRESOLVED at this aperture, not zero (external review 4.8)
        d["rn"].append(math.sqrt(var_t) if var_t > 0 else None)
        d["qfrac"].append(qfrac)
        if r.get("WithinRowVarDiff", "").strip():
            d["row"].append(float(r["WithinRowVarDiff"]) / 2.0)
            d["col"].append(float(r["WithinColVarDiff"]) / 2.0)
        d["tot"].append((float(r["StdDiffClipped"]) ** 2) / 2.0 / factor)
    out = {}
    for iso, d in per_iso.items():
        rns = [v for v in d["rn"] if v is not None]
        out[iso] = {"bl": float(np.mean(d["bl"])),
                    "rn_dn": float(np.mean(rns)) if len(rns) == len(d["rn"]) else None,
                    "quant_frac": float(np.mean(d["qfrac"])),
                    "row_var": float(np.mean(d["row"])) if d["row"] else None,
                    "col_var": float(np.mean(d["col"])) if d["col"] else None,
                    "total_var": float(np.mean(d["tot"]))}
    return header, out

def read_isogain(path: Path, dark: dict) -> dict:
    """{iso: relative gain, normalised to the lowest usable ISO}."""
    _, rows = _parse_rows(path)
    per_iso: dict[int, list[float]] = {}
    greens_present = any(int(r["ColorIndex"]) in GREEN_INDICES for r in rows)
    for r in rows:
        if greens_present and int(r["ColorIndex"]) not in GREEN_INDICES:
            continue
        if (not all(math.isfinite(float(r[k])) for k in ("ClipFrac", "ShutterSec", "Mean"))
                or not 0 <= float(r["ClipFrac"]) <= 1 or float(r["ShutterSec"]) <= 0):
            raise ValueError("invalid gain-ladder statistics")
        if float(r["ClipFrac"]) > CLIP_FRAC_MAX:
            continue
        iso = int(r["ISO"])
        if iso not in dark:
            continue
        dn = float(r["Mean"]) - dark[iso]["bl"]
        if dn <= 0:
            continue
        per_iso.setdefault(iso, []).append(float(r["ShutterSec"]) / dn)
    if not per_iso:
        return {}
    rel = {iso: float(np.mean(v)) for iso, v in per_iso.items()}
    base = rel[min(rel)]
    return {iso: v / base for iso, v in sorted(rel.items())}

def read_whiteness(path: Path, lo=(0.05, 0.20), hi=(0.35, 0.499)) -> dict:
    """{iso: high/mid mean power ratio of the green diff spectra}."""
    _, rows = _parse_rows(path)
    if not rows:
        return {}
    cols = [c for c in rows[0] if c.endswith("_diff")]
    freqs = np.asarray([float(r["freq"]) for r in rows])
    if not np.all(np.isfinite(freqs)) or np.any(freqs < 0) or np.any(freqs > .5):
        raise ValueError("spectrum frequencies must be finite and within [0,.5]")
    out: dict[int, list[float]] = {}
    for c in cols:
        # iso50_C01_diff -> iso 50, channel C01
        stem = c.split("_")
        iso = int(stem[0][3:])
        ch = stem[1]
        if ch not in ("C01", "C10") and any(
                c2.split("_")[1] in ("C01", "C10") for c2 in cols):
            continue
        p = np.asarray([float(r[c]) for r in rows])
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("spectrum power must be finite and nonnegative")
        m_lo = (freqs >= lo[0]) & (freqs <= lo[1])
        m_hi = (freqs >= hi[0]) & (freqs <= hi[1])
        if not (m_lo.any() and m_hi.any()):
            continue
        out.setdefault(iso, []).append(float(p[m_hi].mean() / max(p[m_lo].mean(), 1e-30)))
    return {iso: float(np.mean(v)) for iso, v in sorted(out.items())}

def ptc_anchor(set_dir: Path, dark: dict) -> tuple[int, dict] | None:
    cands = sorted(p for p in set_dir.glob("*ptc-iso*.csv")
                   if "unusable" not in p.name)
    if not cands:
        return None
    path = cands[0]
    iso = int("".join(ch for ch in path.stem.split("iso")[1] if ch.isdigit()))
    header, rows = _parse_rows(path)
    g1_mean = np.asarray([float(r["G1_Mean"]) for r in rows])
    g1_std = np.asarray([float(r["G1_Std"]) for r in rows])
    if not np.all(np.isfinite(g1_mean)) or not np.all(np.isfinite(g1_std)) or np.any(g1_std < 0):
        raise ValueError("PTC statistics must be finite and standard deviations nonnegative")
    black = None
    raw_bl = header.get("BlackLevel", "")
    vals = [v for v in raw_bl.split(",") if v.strip()]
    if len(vals) >= 2:
        black = float(vals[1])
    elif iso in dark:
        # the collect design keeps the black level in the dark set
        black = dark[iso]["bl"]
    if black is None:
        return None
    white = infer_white(g1_mean, g1_std)
    if white is None:
        return None
    return iso, fit_ptc(g1_mean, g1_std, black, white)

def _anchor_evidence(a_iso: int, fit: dict) -> dict:
    """Complete estimator evidence for the anchor (review P2-2): all three
    gains, both residual sets, both PRNU estimates, statuses — the model
    dispute must be reconstructible from the asset alone."""
    keys = ("gain_e_per_dn", "fit_model", "fit_model_effective",
            "gain_alternatives", "gain_estimator_spread_rel",
            "prnu_status", "prnu_iterations", "prnu_converged",
            "read_noise_e", "read_noise_status", "fwc_e",
            "fwc_model_spread_e", "fwc_semantics",
            "last_unsaturated_signal_e", "prnu", "prnu_quadratic_fit",
            "fit_relative_rms", "fit_relative_rms_alternatives",
            "fit_points", "fit_points_excluded", "quality")
    out = {"iso": a_iso}
    out.update({k: fit.get(k) for k in keys})
    return out

def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _find(set_dir: Path, *patterns: str) -> Path | None:
    """Standard name first, then the long descriptive-name variant."""
    for pat in patterns:
        hits = sorted(set_dir.glob(pat))
        if hits:
            return hits[0]
    return None

def build(set_dir: Path, meta: dict | None) -> dict:
    dark_path = _find(set_dir, "dark-scalars.csv", "*dark*scalars.csv")
    if dark_path is None:
        raise SystemExit(f"{set_dir.name}: no dark scalars file — set unusable")
    header, dark = read_dark(dark_path)
    input_hashes = {dark_path.name: _sha256(dark_path)}
    camera = header.get("Camera", "")
    make = camera.split()[0] if camera.split() else ""
    model_rest = camera[len(make):].strip()
    entry: dict = {
        "format": "dngscan-jptc-collect-1",
        "id": f"{camera} ({set_dir.name})",
        "camera": camera,
        "make": make,
        "model_candidates": sorted({model_rest, camera}),
        "shutter": header.get("ShutterType"),
        "compression": header.get("Compression"),
        "geometry": [header.get("ImageWidth"), header.get("ImageHeight")],
        "source": {
            "kind": "JPTC collect set (first-party, credit-based grant, NOTICE.md)",
            "set": set_dir.name,
            "url_base": f"https://y-g-jiang.github.io/data/collect/{set_dir.name}/",
            "url_note": "per-FILE urls are url_base + input file name (the "
                        "directory itself is not a servable page)",
            "tester": (meta or {}).get("tester"),
            "measured_at": (meta or {}).get("measuredAt"),
            "formats": [],
            "inputs": {},
        },
    }
    entry["source"]["formats"].append("JPTC-DARK/1")
    entry["source"]["inputs"] = input_hashes
    clip_status = header.get("_clip_factor_status", "declared")
    entry["acquisition_contract"] = {
        "adc_step": header.get("AdcStep"),
        "linearisation_curve": header.get("LinearisationCurve"),
        "clip_variance_factor": header.get("ClipVarianceFactor"),
        "sigma_clip_correction": ("applied" if clip_status == "declared"
                                  else "unresolved"),
        "sheppard_assumption": "uniform quantisation step at black level; "
                               "NOT verified against the companded "
                               "linearisation curve (declared limitation)",
        "stored_dark_variance_measurement": "paired-frame-pre-sheppard",
        "stored_dark_variance_domain": "linearized-raw-dn",
    }
    # The formed RAW contains quantisation as well as physical read noise.
    # Preserve the measured temporal variance before the optional Sheppard
    # subtraction; never reconstruct this measurement from an assumed ADC step
    # when it is already available. Failed physical-RN fits remain separate.
    entry["stored_dark_variance_dn2_log2iso"] = [
        [math.log2(iso), d["total_var"]] for iso, d in sorted(dark.items())]
    if clip_status == "declared":
        entry["noise_aperture"] = _LEGACY_SHEPPARD_APERTURE
    else:
        entry["noise_aperture"] = (
            "paired-frame temporal std (FPN excluded); ClipVarianceFactor "
            "undeclared upstream so the sigma-clip bias is NOT undone "
            "(sigma_clip_correction=unresolved); Sheppard step^2/12 applied")
    rn_dn_curve = [[math.log2(iso), d["rn_dn"]]
                   for iso, d in sorted(dark.items()) if d["rn_dn"] is not None]
    entry_unresolved = sorted(iso for iso, d in dark.items() if d["rn_dn"] is None)
    entry["read_noise_dn_log2iso"] = rn_dn_curve
    if entry_unresolved:
        entry["read_noise_unresolved_isos"] = entry_unresolved
    entry["quantization_fraction_log2iso"] = [
        [math.log2(iso), round(d["quant_frac"], 5)]
        for iso, d in sorted(dark.items())]
    # External review 4.4: WithinRow/ColVarDiff systematically EXCEEDS the
    # clipped total variance (ratios up to 1.61 across the corpus), so these
    # are NOT banding components and no fraction is published. The raw
    # within-metrics are kept verbatim with their semantics declared
    # unconfirmed until the JPTC-DARK/1 definition is settled with upstream.
    entry["within_var_raw_log2iso"] = {
        "semantics": "UNCONFIRMED — WithinRow/ColVarDiff halved, verbatim; "
                     "not a banding fraction (values may exceed the clipped "
                     "total variance; upstream definition being confirmed)",
        "rows": [
            [round(math.log2(iso), 4), d["row_var"], d["col_var"], d["total_var"]]
            for iso, d in sorted(dark.items()) if d["row_var"] is not None],
    }

    gain_path = _find(set_dir, "gain-levels.csv", "*gain-levels*.csv")
    if gain_path:
        input_hashes[gain_path.name] = _sha256(gain_path)
    rel = read_isogain(gain_path, dark) if gain_path else {}
    anchor = ptc_anchor(set_dir, dark)
    ptc_file = next((c for c in sorted(set_dir.glob("*ptc-iso*.csv"))
                     if "unusable" not in c.name), None)
    if ptc_file is not None:
        input_hashes[ptc_file.name] = _sha256(ptc_file)
    if anchor is not None and not rel:
        a_iso, fit = anchor
        entry["source"]["formats"].append("JPTC/2 (ptc anchor)")
        entry["ptc_anchor"] = _anchor_evidence(a_iso, fit)
        entry["unity_gain_ev"] = round(math.log2(a_iso * fit["gain_e_per_dn"]), 4)
        entry["fwc_e"] = fit["fwc_e"]
        entry["fwc_model_spread_e"] = fit["fwc_model_spread_e"]
        if a_iso in dark and dark[a_iso]["rn_dn"] is not None:
            entry["read_noise_log2iso_log2e"] = [
                [math.log2(a_iso),
                 math.log2(max(dark[a_iso]["rn_dn"] * fit["gain_e_per_dn"], 1e-6))]]
    if rel and anchor is not None:
        entry["source"]["formats"] += ["JPTC-ISOGAIN/1", "JPTC/2 (ptc anchor)"]
        a_iso, fit = anchor
        if a_iso not in rel:
            # anchor ISO missing from the gain ladder: monotone-cubic
            # interpolation in log-log; INTERPOLATION ONLY — an anchor
            # outside the ladder domain is rejected rather than silently
            # extrapolated (external review 4.7)
            if not (min(rel) <= a_iso <= max(rel)):
                raise SystemExit(
                    f"{set_dir.name}: PTC anchor ISO {a_iso} outside the "
                    f"gain-ladder domain [{min(rel)}, {max(rel)}]")
            xs = np.log2(np.asarray(sorted(rel)))
            ys = np.log2(np.asarray([rel[i] for i in sorted(rel)]))
            rel_at_anchor = float(2.0 ** _pchip(xs, ys, math.log2(a_iso)))
        else:
            rel_at_anchor = rel[a_iso]
        scale = fit["gain_e_per_dn"] / rel_at_anchor
        gain_curve = {iso: r * scale for iso, r in rel.items()}
        entry["ptc_anchor"] = _anchor_evidence(a_iso, fit)
        entry["gain_log2iso_log2epd"] = [
            [math.log2(i), math.log2(g)] for i, g in gain_curve.items()]
        entry["read_noise_log2iso_log2e"] = [
            [math.log2(i), math.log2(dark[i]["rn_dn"] * gain_curve[i])]
            for i in sorted(gain_curve)
            if i in dark and dark[i]["rn_dn"] is not None
            and dark[i]["rn_dn"] * gain_curve[i] > 0]
        # Gain-jump candidates: gain*iso is constant under the reciprocal
        # law, so the ladder is expected to sit on flat plateaus.
        # Extended-ISO segments make u rise from the very start (flat gain),
        # so an upward jump (>15%) only counts when both neighbours are
        # plateau-like (adjacent ratio < 1.08 on each side). A
        # plateau-to-plateau jump is EITHER a conversion-gain switch OR the
        # extended-to-native-base boundary; the ladder alone cannot tell
        # them apart, so the field claims neither — it lists every jump and
        # leaves the semantics to curation (声明失实才是缺陷).
        isos = sorted(gain_curve)
        u = [gain_curve[i] * i for i in isos]
        if any(v <= 0 for v in u):
            raise SystemExit(f"{set_dir.name}: non-positive gain*iso")

        def _flat(a, b):
            # symmetric plateau test (external review 4.6): a one-sided
            # ratio<1.08 lets a 50% DROP count as flat
            return abs(math.log(b / a)) < math.log(1.08)

        jumps = []
        for k in range(2, len(u) - 1):
            if (_flat(u[k - 2], u[k - 1]) and _flat(u[k], u[k + 1])
                    and u[k] / u[k - 1] > 1.15):
                jumps.append(isos[k])
        entry["gain_jump_isos"] = jumps
        entry["unity_gain_ev"] = round(math.log2(a_iso * fit["gain_e_per_dn"]), 4)
        entry["fwc_e"] = fit["fwc_e"]
        entry["fwc_model_spread_e"] = fit["fwc_model_spread_e"]
    for ax in ("h", "v"):
        sp = _find(set_dir, f"spectrum-{ax}.csv", f"*spectrum-{ax}.csv")
        if sp is not None:
            input_hashes[sp.name] = _sha256(sp)
            w = read_whiteness(sp)
            if w:
                entry["source"]["formats"].append(f"JPTC-SPECTRUM/1 ({ax})")
                entry[f"noise_whiteness_{ax}_log2iso"] = [
                    [math.log2(i), round(v, 4)] for i, v in w.items()]
    return entry


def _pchip(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    """Scalar shape-preserving cubic interpolation, no SciPy dependency.

    The harmonic-mean slopes and endpoint limiter are the PCHIP algorithm
    previously used by the offline importer. Extrapolation is never allowed.
    """
    if len(xs) < 2 or x < xs[0] or x > xs[-1]:
        raise ValueError("PTC anchor outside the gain ladder")
    h = np.diff(xs)
    m = np.diff(ys) / h
    d = np.zeros(len(xs), dtype=np.float64)
    if len(xs) == 2:
        d[:] = m[0]
    else:
        for i in range(1, len(xs) - 1):
            if m[i - 1] * m[i] > 0:
                w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
                d[i] = (w1 + w2) / (w1 / m[i - 1] + w2 / m[i])
        def endpoint(h0, h1, m0, m1):
            value = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
            if np.sign(value) != np.sign(m0):
                return 0.0
            return 3 * m0 if np.sign(m0) != np.sign(m1) and abs(value) > 3 * abs(m0) else value
        d[0] = endpoint(h[0], h[1], m[0], m[1])
        d[-1] = endpoint(h[-1], h[-2], m[-1], m[-2])
    i = min(max(int(np.searchsorted(xs, x, side="right")) - 1, 0), len(xs) - 2)
    t = (x - xs[i]) / h[i]
    return float((2*t**3 - 3*t**2 + 1)*ys[i] + (t**3 - 2*t**2 + t)*h[i]*d[i]
                 + (-2*t**3 + 3*t**2)*ys[i+1] + (t**3 - t**2)*h[i]*d[i+1])


_SUPPORTED_FORMATS = {"dngscan-jptc-collect-1", "dngscan-jptc-prior-1"}
_USER_FORMAT = "dngscan-user-calibration-1"
_MAX_JSON_BYTES = 16 * 1024 * 1024


def data_path() -> Path:
    """User data directory, including for wheel installations."""
    raw = os.environ.get("DNGSCAN_CALIBRATION_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "dngscan" / "calibrations"


def _normalise(value: str) -> str:
    return " ".join(str(value).upper().split())


def normalise_make(value: str) -> str:
    value = _normalise(value)
    # Explicit manufacturer spellings, not substring matching.
    return {"NIKON CORPORATION": "NIKON", "SONY CORPORATION": "SONY",
            "CANON INC.": "CANON", "SIGMA CORPORATION": "SIGMA",
            "RICOH IMAGING COMPANY, LTD.": "RICOH"}.get(value, value)


def normalise_model(make: str, model: str) -> str:
    value, prefix = _normalise(model), normalise_make(make) + " "
    return value[len(prefix):] if value.startswith(prefix) else value


def _number(value: Any, label: str, *, positive: bool = True) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}: expected a finite number")
    try:
        out = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label}: expected a finite number") from exc
    if not math.isfinite(out) or (out <= 0 if positive else out < 0):
        raise ValueError(f"{label}: expected a {'positive' if positive else 'nonnegative'} finite number")
    return out


def _curve(value: Any, label: str, *, logarithmic: bool = True,
           allow_zero: bool = False) -> list[list[float]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label}: expected a list")
    out = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{label}: expected [log2 ISO, value] pairs")
        x = _number(point[0], label + " ISO", positive=False)
        try:
            y = float(point[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label}: invalid value") from exc
        if not math.isfinite(y) or (not logarithmic and (y < 0 or (y == 0 and not allow_zero))):
            raise ValueError(f"{label}: invalid value")
        if x > 24 or (logarithmic and (y < -30 or y > 40)):
            raise ValueError(f"{label}: unsupported numeric scale")
        if out and x <= out[-1][0]:
            raise ValueError(f"{label}: ISO points must be strictly increasing")
        out.append([x, y])
    return out


def _read_noise_unresolved_isos(value: Any, *curves) -> list[float]:
    """Retain failed measurements as barriers, rather than missing samples."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("read_noise_unresolved_isos must be a list")
    result = sorted({_number(v, "unresolved read-noise ISO") for v in value})
    for iso in result:
        if not 1 <= iso <= 2**24:
            raise ValueError("unresolved read-noise ISO: unsupported numeric scale")
        if any(abs(math.log2(iso)-x) < 1e-7 for curve in curves for x, _ in curve):
            raise ValueError(f"read-noise ISO {iso:g} is both resolved and unresolved")
    return result


def read_noise_issue(entry: dict, iso: float, key: str = "read_noise_log2iso_log2e") -> str | None:
    """Identify unresolved samples and intervals that would cross one.

    Gain is independent evidence and is deliberately not constrained here.
    An unresolved point is not a zero-noise measurement or a license to
    interpolate through its neighbours.
    """
    if not iso or iso <= 0 or not math.isfinite(float(iso)):
        return None
    failed = entry.get("read_noise_unresolved_isos") or []
    x = math.log2(iso)
    if any(abs(x-math.log2(i)) < 1e-7 for i in failed):
        return "read-noise-unresolved"
    curve = entry.get(key) or []
    # Resolved points on either side remain valid measurements themselves.
    if any(abs(x-px) < 1e-7 for px, _ in curve):
        return None
    if curve and (x < curve[0][0] and any(x <= math.log2(i) < curve[0][0] for i in failed)
                  or x > curve[-1][0] and any(curve[-1][0] < math.log2(i) <= x for i in failed)):
        return "read-noise-unresolved-interval"
    for (x0, _), (x1, _) in zip(curve, curve[1:]):
        if x0 < x < x1 and any(x0 < math.log2(i) < x1 for i in failed):
            return "read-noise-unresolved-interval"
    return None


def read_noise_status(entry: dict, iso: float | None) -> str:
    """Report measured, interpolated, unavailable and failed evidence separately."""
    issue = read_noise_issue(entry, iso)
    if issue is not None:
        return issue
    value = curve_value(entry, "read_noise_log2iso_log2e", iso)
    if value is None:
        return "unavailable"
    x = math.log2(iso)
    return ("measured" if any(abs(x-px) < 1e-7 for px, _ in
                              entry.get("read_noise_log2iso_log2e", [])) else "interpolated")


def _shutter(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return {"机械快门": "mechanical", "电子快门": "electronic",
            "mech": "mechanical", "elec": "electronic",
            "electronic first curtain": "efcs", "电子前帘": "efcs",
            "电子前帘快门": "efcs"}.get(str(value), str(value).lower().strip())


def _quality(item: dict) -> dict:
    q = item.get("quality", "ok")
    evidence = q if isinstance(q, dict) else {}
    if isinstance(q, dict):
        q = q.get("status", "ok")
    out = {"status": str(q), "fit_model": item.get("fit_model_effective") or item.get("fit_model"),
           "prnu_status": item.get("prnu_status")}
    for src, dst in (("fit_relative_rms", "fit_relative_rms"),
                     ("gain_estimator_spread_rel", "estimator_spread")):
        value = item.get(src, evidence.get(dst))
        if value is not None:
            out[dst] = _number(value, src, positive=False)
    if out.get("fit_relative_rms", 0) > .05:
        out["status"] = "high-residual"
    if out.get("prnu_status") == "unconverged":
        out["status"] = "unconverged"
    return out


def _validated_prior(item: dict) -> dict:
    if not isinstance(item, dict) or item.get("format") not in _SUPPORTED_FORMATS:
        raise ValueError("unsupported calibration format; expected dngscan-jptc-collect-1 or dngscan-jptc-prior-1")
    single = item["format"] == "dngscan-jptc-prior-1"
    make = item.get("brand") if single else item.get("make")
    models = [item.get("model")] if single else item.get("model_candidates")
    if not isinstance(make, str) or not make.strip():
        raise ValueError("calibration requires a camera make")
    if not isinstance(models, list) or not models or not all(isinstance(m, str) and m.strip() for m in models):
        raise ValueError("calibration requires explicit camera model(s)")
    applicability = item.get("user_applicability") or {}
    if not isinstance(applicability, dict):
        raise ValueError("user_applicability must be an object")
    shutter = _shutter(applicability.get("shutter_override", item.get("shutter")))
    if shutter not in (None, "mechanical", "electronic", "efcs", "any"):
        raise ValueError("unsupported shutter mode; use mechanical, electronic, efcs, or explicit any")
    contract = item.get("acquisition_contract") or {}
    if not isinstance(contract, dict):
        raise ValueError("acquisition_contract must be an object")
    if item.get("compression") is not None and not isinstance(item["compression"], str):
        raise ValueError("compression must be text")
    geometry = item.get("geometry")
    if geometry is not None:
        if not isinstance(geometry, list) or len(geometry) != 2:
            raise ValueError("geometry must be [width, height]")
        for value in geometry:
            if value not in (None, ""):
                _number(value, "geometry dimension")
    label = item.get("id") or f"{make} {models[0]}"
    if not isinstance(label, str) or not label.strip():
        raise ValueError("calibration id must be text")
    entry = {"id": label, "make_contains": make, "make_equals": normalise_make(make),
             "model_equals": sorted({normalise_model(make, m) for m in models}),
             "model_labels": models, "shutter": shutter, "pdr_log2iso_ev": [],
             "user_calibration": True, "noise_model_channels": "scalar-green",
             "noise_aperture": item.get("noise_aperture"),
             "acquisition_contract": contract,
             "source_shutter": item.get("shutter"), "user_applicability": applicability,
             "gain_jump_isos": [], "source_format": item["format"],
             "mode_scope": "shutter-and-dn-scale",
             "unverified_readout_fields": {k: item[k] for k in ("compression", "geometry")
                                           if item.get(k) not in (None, [], [None, None], ["", ""])}}
    entry.update(stored_dark_variance_fields(item))
    if single:
        iso = _number(item.get("iso"), "iso")
        gain = _number(item.get("gain_e_per_dn"), "gain_e_per_dn")
        rn = _number(item.get("read_noise_e", 0), "read_noise_e", positive=False)
        x = math.log2(iso)
        failed = _read_noise_unresolved_isos(item.get("read_noise_unresolved_isos"))
        if rn == 0:
            failed = sorted(set(failed + [iso]))
        entry.update(measured_iso=iso, unity_gain_ev=math.log2(iso*gain),
                     gain_log2iso_log2epd=[[x, math.log2(gain)]],
                     read_noise_log2iso_log2e=[[x, math.log2(rn)]] if rn > 0 else [],
                     read_noise_unresolved_isos=failed,
                     quality=_quality(item))
        _read_noise_unresolved_isos(failed, entry["read_noise_log2iso_log2e"])
        fwc = _number(item.get("fwc_e"), "fwc_e")
        white, black = item.get("white_level_used"), item.get("black_level_g1")
        if white is not None and black is not None:
            span = _number(float(white) - float(black), "reference DN range")
            if abs(span * gain / fwc - 1.0) > 0.05:
                raise ValueError("FWC, gain and declared DN range disagree")
        else:
            span = fwc/gain
        entry.update(fwc_e=fwc, reference_dn_range=span)
    else:
        anchor = item.get("ptc_anchor") or {}
        if not isinstance(anchor, dict):
            raise ValueError("ptc_anchor must be an object")
        gains = _curve(item.get("gain_log2iso_log2epd"), "gain curve")
        if anchor:
            iso = _number(anchor.get("iso"), "anchor ISO")
            gain = _number(anchor.get("gain_e_per_dn"), "anchor gain")
            entry["measured_iso"] = iso
            if not gains:
                gains = [[math.log2(iso), math.log2(gain)]]
        entry["gain_log2iso_log2epd"] = gains
        entry["read_noise_log2iso_log2e"] = _curve(item.get("read_noise_log2iso_log2e"), "read-noise curve")
        entry["read_noise_dn_log2iso"] = _curve(item.get("read_noise_dn_log2iso"), "DN read-noise curve", logarithmic=False)
        entry["read_noise_unresolved_isos"] = _read_noise_unresolved_isos(
            item.get("read_noise_unresolved_isos"), entry["read_noise_log2iso_log2e"],
            entry["read_noise_dn_log2iso"])
        entry["quality"] = _quality(anchor)
        if item.get("fwc_e") is not None:
            entry["fwc_e"] = _number(item["fwc_e"], "fwc_e")
        jumps = item.get("gain_jump_isos") or []
        if not isinstance(jumps, list):
            raise ValueError("gain_jump_isos must be a list")
        entry["gain_jump_isos"] = sorted({_number(i, "gain jump ISO") for i in jumps})
        if anchor and entry.get("fwc_e"):
            entry["reference_dn_range"] = entry["fwc_e"] / gain
            entry["unity_gain_ev"] = math.log2(iso*gain)
            at_anchor = curve_value(entry, "gain_log2iso_log2epd", iso)
            if at_anchor is None or abs(2**at_anchor/gain - 1) > .05:
                raise ValueError("PTC anchor disagrees with measured gain curve")
        for axis in ("h", "v"):
            key = f"noise_whiteness_{axis}_log2iso"
            if item.get(key) is not None:
                entry[key] = _curve(item[key], key, logarithmic=False, allow_zero=True)
        if gains and any(not gains[0][0]+1e-7 < math.log2(j) <= gains[-1][0]+1e-7 for j in entry["gain_jump_isos"]):
            raise ValueError("gain jump outside measured ISO domain")
    # Apply the same finite scale/order checks to synthesized single-ISO
    # points as to supplied ladders.
    entry["gain_log2iso_log2epd"] = _curve(entry.get("gain_log2iso_log2epd"), "gain curve")
    entry["read_noise_log2iso_log2e"] = _curve(entry.get("read_noise_log2iso_log2e"), "read-noise curve")
    if not single:
        # The Collect exporter derives electron noise from the *same* DN
        # dark measurement and gain at that measured ISO. Therefore their
        # units must agree. Use measured intersections only: interpolation
        # (especially across gain jumps) is not an independent validation.
        # Five percent is the existing DN-scale transport tolerance in
        # priors.gain_for_file; exported curves normally agree to rounding.
        gains = entry["gain_log2iso_log2epd"]
        electron_noise = entry["read_noise_log2iso_log2e"]
        for x, rn_dn in entry.get("read_noise_dn_log2iso", []):
            gain_point = next((y for px,y in gains if abs(px-x) < 1e-7), None)
            noise_point = next((y for px,y in electron_noise if abs(px-x) < 1e-7), None)
            if gain_point is None or noise_point is None:
                continue
            expected_log = math.log2(rn_dn) + gain_point
            if abs(noise_point - expected_log) > math.log2(1.05):
                raise ValueError(f"read noise DN/electron units disagree at ISO {2**x:g}")
        for x, variance in entry.get("stored_dark_variance_dn2_log2iso", []):
            if _stored_variance_below_physical_read(entry, 2**x, variance, measured_only=True):
                raise ValueError(f"stored dark variance is below physical read variance at ISO {2**x:g}")
    if not entry.get("gain_log2iso_log2epd") and not entry.get("read_noise_dn_log2iso"):
        raise ValueError("calibration contains no usable gain or dark-noise measurements")
    if item.get("fwc_model_spread_e") is not None:
        entry["fwc_model_spread_e"] = _number(item["fwc_model_spread_e"], "fwc model spread", positive=False)
    return entry


def stored_dark_variance_fields(item: dict) -> dict:
    """Carry an explicitly typed Collect measurement into every runtime tier."""
    contract = dict(item.get("acquisition_contract") or {})
    curve = _curve(item.get("stored_dark_variance_dn2_log2iso"),
                   "stored dark variance", logarithmic=False, allow_zero=True)
    if curve and (item.get("format") != "dngscan-jptc-collect-1"
                  or contract.get("stored_dark_variance_measurement") != "paired-frame-pre-sheppard"
                  or contract.get("stored_dark_variance_domain") != "linearized-raw-dn"):
        raise ValueError("stored dark variance requires a Collect paired-frame linearized RAW DN contract")
    return {"source_format": item.get("format"), "noise_aperture": item.get("noise_aperture"),
            "acquisition_contract": contract,
            "stored_dark_variance_dn2_log2iso": curve}


_LEGACY_SHEPPARD_APERTURE = (
    "paired-frame temporal std (FPN excluded); sigma clip undone via "
    "the declared ClipVarianceFactor; Sheppard step^2/12 quantisation "
    "correction applied per frame"
)


def _stored_variance_below_physical_read(entry: dict, iso: float, variance: float,
                                         *, measured_only: bool = False) -> bool:
    """Check both physical-noise units without squaring a potentially large DN value."""
    def value(key):
        if measured_only:
            return next((y for x, y in entry.get(key, [])
                         if abs(x-math.log2(iso)) < 1e-7), None)
        return curve_value(entry, key, iso)

    read_variance_logs = []
    rn_dn = value("read_noise_dn_log2iso")
    if rn_dn is not None and rn_dn > 0:
        read_variance_logs.append(2 * math.log2(rn_dn))
    rn_e_log = value("read_noise_log2iso_log2e")
    gain_log = value("gain_log2iso_log2epd")
    if rn_e_log is not None and gain_log is not None:
        read_variance_logs.append(2 * (rn_e_log - gain_log))
    # The electron curve may be present without the optional DN-RN curve.
    # Check either independent unit declaration that is available. Retain
    # the existing five-percent DN-unit tolerance at measured intersections
    # and selected ISO values; interpolation is not allowed to evade it.
    return bool(read_variance_logs) and (variance == 0 or
        math.log2(variance) < math.log2(.95) + max(read_variance_logs))


def stored_dark_variance(entry: dict, iso: float | None) -> tuple[float | None, str]:
    """Reference-DN variance, with source semantics distinct from electron RN.

    Only Collect's explicit measurement or its unambiguous legacy converter
    contract authorizes a stored-code variance. This is not a universal 1/12
    adjustment to PTC intercepts, P2P data, or DNG NoiseProfile coefficients.
    Consumers must still establish camera, readout, ISO and DN applicability.
    """
    if entry.get("source_format") != "dngscan-jptc-collect-1":
        return None, "read-variance-semantics-unspecified"
    contract = entry.get("acquisition_contract") or {}
    curve = entry.get("stored_dark_variance_dn2_log2iso") or []
    if curve:
        if (contract.get("stored_dark_variance_measurement") != "paired-frame-pre-sheppard"
                or contract.get("stored_dark_variance_domain") != "linearized-raw-dn"):
            return None, "stored-dark-variance-domain-unverified"
        if contract.get("sigma_clip_correction") != "applied":
            return None, "stored-dark-variance-sigma-clip-unresolved"
        value = curve_value(entry, "stored_dark_variance_dn2_log2iso", iso)
        if value is None:
            return None, "stored-dark-variance-outside-measured-domain"
        if not math.isfinite(value) or value < 0:
            return None, "stored-dark-variance-nonfinite"
        if _stored_variance_below_physical_read(entry, iso, value):
            return None, "stored-dark-variance-below-physical-read-variance"
        measured = any(abs(math.log2(iso)-x) < 1e-7 for x, _ in curve)
        return value, ("measured-stored-dark-variance" if measured else
                       "interpolated-stored-dark-variance")
    # Old bundled/user Collect files did not retain the pre-subtraction
    # curve. Recover it only where the converter explicitly says what it
    # subtracted in an identity-linearized DN domain. Companded or missing
    # acquisition declarations cannot establish this transport.
    if (entry.get("noise_aperture") != _LEGACY_SHEPPARD_APERTURE
            or contract.get("sigma_clip_correction") != "applied"
            or contract.get("linearisation_curve") != "identity"):
        return None, "stored-dark-variance-unverified"
    try:
        step = float(contract["adc_step"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "stored-dark-variance-adc-step-unverified"
    if not math.isfinite(step) or step < 0:
        return None, "stored-dark-variance-adc-step-unverified"
    rn_dn = curve_value(entry, "read_noise_dn_log2iso", iso)
    if rn_dn is None:
        return None, "stored-dark-variance-outside-measured-domain"
    # The legacy curve averaged green-plane RMS values, losing their
    # individual variances. This restores the declared quantisation term
    # to that scalar approximation; only the new curve retains the mean
    # measured total variance without this aggregation loss.
    variance = rn_dn * rn_dn + step * step / 12.
    if not math.isfinite(variance):
        return None, "stored-dark-variance-adc-step-unverified"
    if _stored_variance_below_physical_read(entry, iso, variance):
        return None, "stored-dark-variance-below-physical-read-variance"
    return variance, "legacy-sheppard-restored-scalar-variance-approximation"


def curve_value(entry: dict, key: str, iso: float) -> float | None:
    """Interpolate within measured domains, respecting jumps and failed noise points."""
    curve = entry.get(key) or []
    if not curve or not iso or iso <= 0 or not math.isfinite(float(iso)):
        return None
    if key in ("read_noise_log2iso_log2e", "read_noise_dn_log2iso") and read_noise_issue(entry, iso, key):
        return None
    x = math.log2(iso)
    if x < curve[0][0]-1e-7 or x > curve[-1][0]+1e-7:
        return None
    for px, py in curve:
        if abs(x-px) < 1e-7:
            return float(py)
    for (x0,y0), (x1,y1) in zip(curve, curve[1:]):
        if x0 < x < x1:
            if any(x0 < math.log2(j) <= x1+1e-7 for j in entry.get("gain_jump_isos", [])):
                return None
            return float(y0 + (x-x0)/(x1-x0)*(y1-y0))
    return None


def _json_load(path: Path) -> dict:
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError("calibration file is too large")
    def invalid_constant(value):
        raise ValueError(f"non-finite JSON value: {value}")
    result = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("calibration JSON must contain an object")
    return result


def _atomic_write(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".calibration-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(item, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _digest(item: dict) -> str:
    return hashlib.sha256(json.dumps(item, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _summary(record: dict, path: Path, prior: dict) -> dict:
    gains = prior.get("gain_log2iso_log2epd") or []
    domain = gains or prior.get("read_noise_dn_log2iso") or []
    warnings = []
    q = prior.get("quality") or {}
    if q.get("status") in ("high-residual", "unconverged"):
        warnings.append("quality-" + q["status"])
    if q.get("estimator_spread", 0) > .10:
        warnings.append("estimator-spread")
    if not prior.get("shutter"):
        warnings.append("readout-mode-unavailable")
    if prior.get("unverified_readout_fields"):
        warnings.append("sub-readout-mode-not-verified")
    contract = prior.get("acquisition_contract") or {}
    if contract.get("sigma_clip_correction") == "unresolved":
        warnings.append("sigma-clip-correction-unresolved")
    if contract.get("linearisation_curve") == "companded":
        warnings.append("quantisation-correction-assumption-unverified")
    if (prior.get("source_format") == "dngscan-jptc-collect-1"
            and not prior.get("stored_dark_variance_dn2_log2iso")):
        warnings.append("stored-dark-variance-not-directly-retained")
    if not gains:
        warnings.append("no-absolute-gain")
    if not prior.get("read_noise_log2iso_log2e"):
        warnings.append("no-electron-read-noise")
    if prior.get("read_noise_unresolved_isos"):
        warnings.append("read-noise-unresolved-measurements")
    if gains and not prior.get("reference_dn_range"):
        warnings.append("reference-dn-range-unavailable")
    return {"id": record["id"], "label": prior["id"], "active": bool(record.get("active", True)),
            "make": prior["make_contains"], "models": prior["model_labels"],
            "shutter": prior.get("shutter"), "source_shutter": prior.get("source_shutter"),
            "user_applicability": prior.get("user_applicability", {}), "iso_min": 2**domain[0][0] if domain else None,
            "iso_max": 2**domain[-1][0] if domain else None,
            "has_gain": bool(gains), "has_read_noise": bool(prior.get("read_noise_log2iso_log2e")),
            "has_stored_dark_variance": bool(prior.get("stored_dark_variance_dn2_log2iso")),
            "read_noise_unresolved_isos": prior.get("read_noise_unresolved_isos", []),
            "warnings": warnings, "source_format": prior["source_format"], "path": str(path),
            "imported_at": record.get("imported_at"), "channel_model": prior["noise_model_channels"],
            "mode_scope": prior["mode_scope"],
            "unverified_readout_fields": prior.get("unverified_readout_fields", {})}


def _read_record(path: Path) -> tuple[dict, dict]:
    item = _json_load(path)
    if item.get("format") != _USER_FORMAT or item.get("id") != path.stem:
        raise ValueError("invalid user calibration record")
    if not isinstance(item.get("active"), bool):
        raise ValueError("active must be a boolean")
    payload = item.get("payload")
    if _digest(payload) != item.get("content_sha256"):
        raise ValueError("calibration content checksum mismatch")
    if item["id"] != item["content_sha256"][:24]:
        raise ValueError("calibration identity checksum mismatch")
    prior = _validated_prior(payload)
    return item, prior


def _records() -> list[tuple[Path, dict | None, dict | None, str | None]]:
    out = []
    try:
        root = data_path()
        if root.exists() and not root.is_dir():
            raise NotADirectoryError("calibration store is not a directory")
        paths = sorted(root.glob("*.json"))
    except OSError as exc:
        return [(data_path(), None, None, str(exc))]
    for path in paths:
        try:
            record, prior = _read_record(path)
            out.append((path, record, prior, None))
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            out.append((path, None, None, str(exc)))
    return out


def import_calibration(path: str | Path, *, active: bool = True,
                       shutter_override: str | None = None) -> dict:
    """Validate a JPTC Collect directory or derived JSON and store locally.

    Re-import is idempotent by canonical payload digest. The source payload
    and acquisition evidence are retained verbatim for later inspection.
    """
    if not isinstance(active, bool):
        raise ValueError("active must be a boolean")
    source = Path(path).expanduser()
    if source.is_dir():
        # Match only the files consumed by build; reject foreign format
        # versions instead of accepting similarly named, incompatible CSVs.
        dark = _find(source, "dark-scalars.csv", "*dark*scalars.csv")
        if dark is None:
            raise ValueError("Collect directory requires dark-scalars.csv")
        main_header, _ = _parse_rows(dark)
        for file, formats in [(dark, {"JPTC-DARK/1"}),
                              *[(p, {"JPTC/2"}) for p in source.glob("*ptc-iso*.csv") if "unusable" not in p.name],
                              *[(p, {"JPTC-ISOGAIN/1"}) for p in source.glob("*gain-levels*.csv")],
                              *[(p, {"JPTC-SPECTRUM/1"}) for p in source.glob("*spectrum-*.csv")]]:
            if file.stat().st_size > _MAX_JSON_BYTES:
                raise ValueError(f"{file.name}: measurement file too large")
            header, _ = _parse_rows(file)
            if header.get("Format") not in formats:
                raise ValueError(f"{file.name}: unsupported measurement format")
            for field in ("Camera", "ShutterType", "Compression", "ImageWidth", "ImageHeight"):
                if header.get(field) and main_header.get(field) and _normalise(header[field]) != _normalise(main_header[field]):
                    raise ValueError(f"{file.name}: {field} disagrees with dark measurements")
        try:
            payload = sanitize_json(build(source, None))
        except (SystemExit, KeyError, ZeroDivisionError, OverflowError, TypeError, IndexError) as exc:
            raise ValueError(f"invalid Collect measurements: {exc}") from exc
        payload["source"]["kind"] = "user-imported JPTC Collect measurements"
        # Browser uploads are reconstructed in randomly named temporary
        # directories. Their identity must depend on measurement bytes,
        # never on that transient directory name.
        payload["id"] = f"{payload['camera']} (user JPTC Collect)"
        payload["source"]["set"] = "local-" + _digest(payload["source"]["inputs"])[:12]
        payload["source"].pop("url_base", None)
        payload["source"].pop("url_note", None)
    else:
        payload = _json_load(source)
        if payload.get("format") == _USER_FORMAT:
            payload = payload.get("payload")
    if shutter_override is not None:
        if shutter_override not in ("any", "mechanical", "electronic", "efcs"):
            raise ValueError("unsupported shutter override")
        payload["user_applicability"] = {"shutter_override": shutter_override,
                                          "source_shutter": payload.get("shutter"),
                                          "declaration": "explicit user import choice"}
    prior = _validated_prior(payload)
    digest = _digest(payload)
    identity = digest[:24]
    target = data_path() / (identity + ".json")
    record = {"format": _USER_FORMAT, "id": identity, "content_sha256": digest, "active": active,
              "imported_at": datetime.now(timezone.utc).isoformat(), "payload": payload}
    _atomic_write(target, record)
    return _summary(record, target, prior)


def list_calibrations() -> list[dict]:
    """List imported records, including damaged records with a visible error."""
    return [_summary(record, path, prior) if record is not None else
            {"id": path.stem, "label": path.stem, "active": False, "error": error,
             "warnings": ["invalid-record"], "path": str(path)}
            for path, record, prior, error in _records()]


def _record_path(identity: str) -> Path:
    if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{24}", identity):
        raise ValueError("invalid calibration id")
    return data_path() / (identity + ".json")


def remove_calibration(identity: str) -> dict:
    target = _record_path(identity)
    target.unlink()
    return {"id": identity, "removed": True}


def set_calibration_active(identity: str, active: bool) -> dict:
    if not isinstance(active, bool):
        raise ValueError("active must be a boolean")
    target = _record_path(identity)
    record, prior = _read_record(target)
    record["active"] = active
    _atomic_write(target, record)
    return _summary(record, target, prior)


def calibration_fingerprint() -> str:
    """Content identity for analysis/preview caches; invalid records also count."""
    digest = hashlib.sha256(str(data_path().resolve()).encode())
    try:
        paths = sorted(data_path().glob("*.json"))
    except OSError as exc:
        digest.update(type(exc).__name__.encode())
        return digest.hexdigest()
    for path in paths:
        digest.update(path.name.encode())
        try:
            digest.update(path.read_bytes())
        except OSError as exc:
            digest.update(type(exc).__name__.encode())
    return digest.hexdigest()


def _match_reason(prior: dict, make: str, model: str, shutter: str | None,
                  iso: float | None) -> str:
    if normalise_make(make) != prior["make_equals"] or normalise_model(make, model) not in prior["model_equals"]:
        return "camera-mismatch"
    mode = prior.get("shutter")
    if mode is None:
        return "readout-mode-unavailable"
    if mode != "any":
        if not shutter:
            return "file-readout-mode-unavailable"
        if _shutter(shutter) != mode:
            return "readout-mode-mismatch"
    q = prior.get("quality") or {}
    if q.get("status") in ("high-residual", "unconverged"):
        return "quality-" + q["status"]
    if q.get("estimator_spread", 0) > .10:
        return "estimator-spread"
    if not prior.get("gain_log2iso_log2epd"):
        return "no-absolute-gain"
    if not prior.get("reference_dn_range"):
        return "reference-dn-range-unavailable"
    if iso is not None and curve_value(prior, "gain_log2iso_log2epd", iso) is None:
        return "iso-out-of-domain-or-gain-jump"
    return "usable"


def calibration_diagnostics(make: str | None, model: str | None, shutter: str | None = None,
                            iso: float | None = None) -> list[dict]:
    out = []
    for path, record, prior, error in _records():
        if error is not None:
            out.append({"id": path.stem, "status": "invalid", "reason": error})
            continue
        reason = _match_reason(prior, make or "", model or "", shutter, iso)
        if reason == "camera-mismatch":
            continue
        if not record.get("active"):
            reason = "inactive"
        noise_status = read_noise_status(prior, iso)
        stored_variance, stored_status = stored_dark_variance(prior, iso)
        out.append({"id": record["id"], "label": prior["id"],
                    "status": ("gain-only" if reason == "usable" and noise_status.startswith("read-noise-unresolved")
                               else "usable" if reason == "usable" else "not-applied"),
                    "reason": noise_status if reason == "usable" and noise_status.startswith("read-noise-unresolved") else reason,
                    "gain_status": "usable" if reason == "usable" else "not-applied",
                    "read_noise_status": noise_status,
                    "stored_dark_variance_status": stored_status,
                    "stored_dark_variance_dn2": stored_variance,
                    "has_read_noise_at_iso": iso is not None and curve_value(prior, "read_noise_log2iso_log2e", iso) is not None,
                    "mode_scope": prior["mode_scope"],
                    "unverified_readout_fields": prior.get("unverified_readout_fields", {}),
                    "warnings": _summary(record, path, prior)["warnings"]})
    return out


def matching_prior(make: str, model: str, *, shutter: str | None = None,
                   iso: float | None = None) -> dict | None:
    candidates = []
    for path, record, prior, error in _records():
        if record is None or not record.get("active"):
            continue
        if _match_reason(prior, make, model, shutter, iso) != "usable":
            continue
        entry = dict(prior)
        entry["calibration_id"] = record["id"]
        entry["source"] = f"User JPTC calibration ({record['id']})"
        entry["mode_match"] = "user-explicit-any-mode" if prior.get("shutter") == "any" else "user-exact-shutter"
        entry["model_equals"] = set(entry["model_equals"])
        candidates.append((record.get("imported_at", ""), record["id"], entry))
    return max(candidates, key=lambda c: (c[0], c[1]))[2] if candidates else None
