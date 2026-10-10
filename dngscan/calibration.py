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
    from .calibration_ptc import fit_channel
    fit = fit_channel(parsed["header"], rows, "G1", black_g1, white)
    from .readout import collect_fields

    header = parsed["header"]
    readout_fields = collect_fields(header)
    # JPTC/2 records the same LibRaw RawSize as Collect. Keep its scope
    # through the single-point converter, without inheriting the paired-dark
    # variance or quantisation semantics that only Collect can establish.
    acquisition_contract = readout_fields.pop("acquisition_contract_fields")
    geometry = [header.get("ImageWidth"), header.get("ImageHeight")]
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
        "compression": header.get("Compression"),
        "geometry": geometry,
        "acquisition_contract": acquisition_contract,
        **readout_fields,
        "noise_aperture": ("paired-frame stored temporal variance (difference variance / 2; "
                           "declared sigma clipping undone; quantisation retained)"
                           if fit.get("variance_domain") else
                           "single-frame spatial std (includes PRNU; fit restricted to the shot-noise decades)"),
        "source": {
            "kind": "JPTC/2 first-party measurement",
            "file": path.name,
            "input_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "mode": header.get("Mode"),
            "geometry": geometry,
            "raw_size": header.get("RawSize"),
            "compression": header.get("Compression"),
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
    phase_samples: dict[int, dict] = {}
    description = header.get("CfaPattern", "").strip().upper()
    green_indices = {i for i, c in enumerate(description) if c == "G"} if description else GREEN_INDICES
    greens_present = any(int(r["ColorIndex"]) in green_indices for r in rows)
    for r in rows:
        iso = int(r["ISO"])
        if iso <= 0 or int(r["ColorIndex"]) not in range(4):
            raise ValueError("dark rows require positive ISO and CFA index 0..3")
        for key in ("BlackA", "BlackB", "StdDiffClipped"):
            if not math.isfinite(float(r[key])) or (key == "StdDiffClipped" and float(r[key]) < 0):
                raise ValueError(f"invalid dark statistic: {key}")
        bl = 0.5 * (float(r["BlackA"]) + float(r["BlackB"]))
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
        rn = math.sqrt(var_t) if var_t > 0 else None
        key = r.get("Channel", "").strip() or f'index:{int(r["ColorIndex"])}'
        phases = phase_samples.setdefault(iso, {})
        phase = phases.setdefault(key, {"channel": r.get("Channel", "").strip() or None,
            "color_index": int(r["ColorIndex"]), "bl": [], "rn_dn": [],
            "total_var": [], "quant_frac": []})
        if phase["color_index"] != int(r["ColorIndex"]):
            raise ValueError("dark channel has contradictory colour indices")
        phase["bl"].append(bl)
        phase["rn_dn"].append(rn)
        phase["total_var"].append(var_diff / 2.)
        phase["quant_frac"].append(qfrac)
        # Preserve every measured phase before forming the legacy green
        # summary. Gain levels must subtract their own phase's black offset.
        if greens_present and int(r["ColorIndex"]) not in green_indices:
            continue
        d = per_iso.setdefault(iso, {"bl": [], "rn": [], "row": [], "col": [], "tot": [], "qfrac": []})
        d["bl"].append(bl)
        d["rn"].append(rn)
        d["qfrac"].append(qfrac)
        if r.get("WithinRowVarDiff", "").strip():
            d["row"].append(float(r["WithinRowVarDiff"]) / 2.0)
            d["col"].append(float(r["WithinColVarDiff"]) / 2.0)
        d["tot"].append((float(r["StdDiffClipped"]) ** 2) / 2.0 / factor)
    out = {}
    for iso, d in per_iso.items():
        rns = [v for v in d["rn"] if v is not None]
        out[iso] = {"bl": float(np.mean(d["bl"])),
                    "rn_dn": float(np.sqrt(np.mean(np.square(rns)))) if len(rns) == len(d["rn"]) else None,
                    "quant_frac": float(np.mean(d["qfrac"])),
                    "row_var": float(np.mean(d["row"])) if d["row"] else None,
                    "col_var": float(np.mean(d["col"])) if d["col"] else None,
                    "total_var": float(np.mean(d["tot"]))}
        out[iso]["phases"] = {}
        for key, phase in phase_samples[iso].items():
            item = {"channel": phase["channel"], "color_index": phase["color_index"]}
            if description:
                if phase["color_index"] >= len(description):
                    raise ValueError("dark ColorIndex is outside CfaPattern colour description")
                item.update(color_desc=description, color=description[phase["color_index"]])
            for name in ("bl", "rn_dn", "total_var", "quant_frac"):
                values = phase[name]
                item[name] = None if any(v is None for v in values) else float(
                    np.sqrt(np.mean(np.square(values))) if name == "rn_dn" else np.mean(values))
            item["unresolved"] = item["rn_dn"] is None
            out[iso]["phases"][key] = item
    return header, out

def _phase_black(dark_iso, row):
    """Match a measured phase, never substitute the other green's pedestal."""
    phases = dark_iso.get("phases")
    if not phases:  # Compatibility for callers supplying old scalar dark maps.
        return dark_iso.get("bl")
    channel = row.get("Channel", "").strip()
    cid = int(row["ColorIndex"])
    if channel and channel in phases:
        phase = phases[channel]
        return phase["bl"] if phase["color_index"] == cid else None
    matches = [p for p in phases.values() if p["color_index"] == cid]
    return matches[0]["bl"] if len(matches) == 1 else None


def _solve_gain_edges(nodes, edges):
    """Connected log-gain least squares; edge weights are explicitly relative."""
    neighbours = {iso: set() for iso in nodes}
    for edge in edges:
        neighbours[edge["a"]].add(edge["b"])
        neighbours[edge["b"]].add(edge["a"])
    unseen, components = set(nodes), []
    while unseen:
        todo, connected = [min(unseen)], set()
        while todo:
            iso = todo.pop()
            if iso in connected:
                continue
            connected.add(iso)
            todo.extend(neighbours[iso] - connected)
        unseen -= connected
        isos = sorted(connected)
        local = [e for e in edges if e["a"] in connected]
        columns = {iso: i for i, iso in enumerate(isos[1:])}
        design = np.zeros((len(local), max(0, len(isos)-1)))
        values, weights = [], []
        for i, edge in enumerate(local):
            if edge["a"] in columns:
                design[i, columns[edge["a"]]] = -1.
            if edge["b"] in columns:
                design[i, columns[edge["b"]]] = 1.
            values.append(edge["log_ratio"])
            weights.append(edge["weight"])
        logs = np.zeros(len(isos))
        if local:
            w = np.sqrt(np.asarray(weights) / max(weights))
            logs[1:] = np.linalg.lstsq(design * w[:, None], np.asarray(values) * w, rcond=None)[0]
        residuals = design @ logs[1:] - np.asarray(values)
        facts = []
        for edge, residual in zip(local, residuals):
            fact = dict(edge)
            fact["observed_gain_ratio"] = math.exp(fact.pop("log_ratio"))
            fact["residual_ev"] = float(residual / math.log(2.))
            facts.append(fact)
        components.append({"isos": isos, "relative_gain": {iso: float(math.exp(v)) for iso,v in zip(isos,logs)},
            "edges": facts, "residual_rms_ev": float(np.sqrt(np.mean(residuals**2)) / math.log(2.)) if local else 0.,
            "anchored": False})
    return components


def read_isogain(path: Path, dark: dict, *, return_diagnostics=False, anchor_iso=None) -> dict:
    """Relative e-/DN within a connected ISO graph, with optional evidence.

    Same-shutter edges cancel nominal time. Mixed ladders use only the
    paired component; auto-shutter levels remain separately diagnosed. The
    public scalar return remains compatible with the original importer.
    """
    header, rows = _parse_rows(path)
    samples, rejected = [], []
    description = header.get("CfaPattern", "").strip().upper()
    green_indices = {i for i, c in enumerate(description) if c == "G"} if description else GREEN_INDICES
    greens_present = any(int(r["ColorIndex"]) in green_indices for r in rows)
    for index, r in enumerate(rows):
        if greens_present and int(r["ColorIndex"]) not in green_indices:
            continue
        if (not all(math.isfinite(float(r[k])) for k in ("ClipFrac", "ShutterSec", "Mean"))
                or not 0 <= float(r["ClipFrac"]) <= 1 or float(r["ShutterSec"]) <= 0):
            raise ValueError("invalid gain-ladder statistics")
        if float(r["ClipFrac"]) > CLIP_FRAC_MAX:
            rejected.append({"row": index, "reason": "clipped"})
            continue
        iso = int(r["ISO"])
        if iso <= 0:
            raise ValueError("gain rows require a positive ISO")
        bl = _phase_black(dark.get(iso, {}), r)
        if bl is None:
            rejected.append({"row": index, "reason": "phase-black-unavailable"})
            continue
        dn = float(r["Mean"]) - bl
        if dn < 200.:
            rejected.append({"row": index, "reason": "signal-below-200-DN"})
            continue
        samples.append({"iso": iso, "phase": r.get("Channel", "").strip() or f'index:{int(r["ColorIndex"])}',
                        "time": float(r["ShutterSec"]), "group": r.get("ShutterGroup", "").strip(), "signal": dn})
    groups = []
    for sample in sorted(samples, key=lambda s: (s["time"], s["group"], s["phase"], s["iso"], s["signal"])):
        found = next((g for g in groups if
            (sample["group"] and g["label"] == sample["group"])
            or (not sample["group"] and not g["label"] and
                abs(g["time"] - sample["time"]) <= 1e-4 * max(g["time"], sample["time"]))), None)
        if found is None:
            found = {"label": sample["group"], "time": sample["time"], "samples": []}
            groups.append(found)
        if abs(found["time"] - sample["time"]) > 1e-4 * max(found["time"], sample["time"]):
            raise ValueError("ShutterGroup contains different shutter settings")
        found["samples"].append(sample)
    paired = [g for g in groups if len({s["iso"] for s in g["samples"]}) >= 2]
    inferred = "auto-shutter" if not paired else "paired-shutter" if len(paired) == len(groups) else "mixed"
    protocol = header.get("Ladder", "").strip().lower() or inferred
    if protocol not in ("paired-shutter", "auto-shutter", "mixed"):
        raise ValueError("unsupported ISO gain Ladder protocol")

    def edges_for(group_samples, name, use_time):
        buckets = {}
        for sample in group_samples:
            buckets.setdefault((sample["phase"], sample["iso"]), []).append(sample)
        result = []
        for phase in sorted({p for p, _ in buckets}):
            isos = sorted(iso for p, iso in buckets if p == phase)
            for a, b in zip(isos, isos[1:]):
                first, second = buckets[phase, a], buckets[phase, b]
                if use_time:
                    va = float(np.mean([s["time"] / s["signal"] for s in first]))
                    vb = float(np.mean([s["time"] / s["signal"] for s in second]))
                    ratio = vb / va
                else:
                    ratio = np.mean([s["signal"] for s in first]) / np.mean([s["signal"] for s in second])
                result.append({"a": a, "b": b, "phase": phase, "group": name,
                    "log_ratio": math.log(float(ratio)), "weight": 2. / (1./len(first) + 1./len(second)),
                    "repeat_counts": [len(first), len(second)]})
        return result

    paired_edges = [edge for index,g in enumerate(paired)
                    for edge in edges_for(g["samples"], g["label"] or f"shutter-group-{index}", False)]
    auto_edges = edges_for(samples, "nominal-shutter", True)
    edges = auto_edges if protocol == "auto-shutter" else paired_edges
    nodes = {s["iso"] for s in samples}
    components = _solve_gain_edges(nodes, edges)
    selected = next((c for c in components if (anchor_iso if anchor_iso is not None else min(nodes,default=0)) in c["isos"]), None)
    rel = {} if selected is None or (not selected["edges"] and len(nodes) > 1) else selected["relative_gain"]
    diagnostics = {"protocol": protocol, "inferred_protocol": inferred,
        "policy": "nominal-time-dependent" if protocol == "auto-shutter" else "paired-edges-only",
        "weight_semantics": "harmonic repeat counts; relative weights, not measurement confidence intervals",
        "components": components, "selected_isos": sorted(rel), "rejected_rows": rejected,
        "disconnected_isos": sorted(nodes - set(rel)),
        "auto_shutter_components": _solve_gain_edges(nodes, auto_edges) if protocol != "auto-shutter" else [],
        "black_scope": "per-phase" if any(d.get("phases") for d in dark.values()) else "legacy-scalar"}
    return (rel, diagnostics) if return_diagnostics else rel

def read_whiteness(path: Path, lo=(0.05, 0.20), hi=(0.35, 0.499), *,
                   phase_mapping=None, return_details=False, axis=None, scalar_rows=()) -> dict:
    """Measured per-position ratios, with colour selection only when mapped.

    The scalar compatibility result selects the worst departure from one;
    opposite anomalies must never cancel by averaging two green planes.
    """
    from .noise_spectrum import conservative_ratio, spectrum_curves
    spectrum_header, rows = _parse_rows(path)
    if not rows:
        return {"ratios_log2iso": {}, "summary": {}} if return_details else {}
    freqs = np.asarray([float(r["freq"]) for r in rows])
    if not np.all(np.isfinite(freqs)) or np.any(freqs < 0) or np.any(freqs > .5):
        raise ValueError("spectrum frequencies must be finite and within [0,.5]")
    curves = spectrum_curves(rows, freqs, lo, hi)
    out: dict[int, list[float]] = {}
    mapping = phase_mapping or {}
    if (spectrum_header.get("CfaPattern") and mapping.get("color_description") and
            spectrum_header["CfaPattern"].upper() != mapping["color_description"]):
        raise ValueError("spectrum colour description disagrees with dark CFA mapping")
    if mapping.get("status") in ("bayer", "single-colour"):
        for phase, record in mapping.get("phases", {}).items():
            if record["color"] != "G" and mapping["status"] != "single-colour":
                continue
            for x, ratio in curves.get(phase, ()):
                out.setdefault(round(2 ** x), []).append(ratio)
    summary = {iso: float(conservative_ratio(v)) for iso, v in sorted(out.items())}
    if not return_details:
        return summary
    result = {"ratios_log2iso": curves, "summary": summary}
    if axis is not None:
        from .noise_spectrum import complete_axis_spectrum
        result["complete"] = complete_axis_spectrum(spectrum_header, rows, axis, scalar_rows=scalar_rows)
    return result

def ptc_anchor(set_dir: Path, dark: dict) -> tuple[int, dict] | None:
    from .calibration_ptc import read_anchors
    record = next((r for r in read_anchors(set_dir, dark) if r["status"] == "usable"), None)
    return (record["iso"], record["fit"]) if record else None

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
            "fit_points", "fit_points_excluded", "quality", "variance_domain",
            "difference_variance_divisor", "temporal_variance_source", "temporal_fallback_reason",
            "pair_drift_limit_relative", "pair_drift_rejected", "gain_fit_standard_error",
            "gain_fit_interval_95", "gain_fit_interval_status", "gain_fit_interval_semantics",
            "fit_uncertainty_semantics", "spatial_crosscheck", "spatial_temporal_gain_disagreement_relative")
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

def _collect_raw_geometry_coverage(set_dir: Path, dark_path: Path) -> dict:
    """Keep every supplied measurement's RawSize, including explicit gaps.

    A dark-frame size cannot fill a missing PTC/gain/spectrum declaration.
    This lives in build(), so packaged and interactive imports share the
    acquisition contract rather than validating only the GUI entry point.
    """
    from .readout import collect_fields
    files = sorted({dark_path, *set_dir.glob("*gain-levels*.csv"),
                    *set_dir.glob("*spectrum-*.csv"),
                    *(p for p in set_dir.glob("*ptc-iso*.csv") if "unusable" not in p.name)})
    coverage, declared = {}, None
    for path in files:
        header, _ = _parse_rows(path)
        contract = collect_fields(header).get("readout_contract") or {}
        size = contract.get("libraw_raw_geometry")
        coverage[path.name] = size
        if size is not None:
            if declared is not None and declared != size:
                raise ValueError(f"{path.name}: RawSize disagrees with other measurements")
            declared = size
    result = {"raw_geometry_measurements": coverage}
    if declared is not None:
        result["raw_geometry_complete"] = all(size is not None for size in coverage.values())
    return result

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
    from .readout import collect_fields
    readout_fields = collect_fields(header)
    entry["acquisition_contract"].update(readout_fields.pop("acquisition_contract_fields"))
    entry["acquisition_contract"].update(_collect_raw_geometry_coverage(set_dir, dark_path))
    entry.update(readout_fields)
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
    if gain_path:
        rel, entry["gain_ladder_diagnostics"] = read_isogain(gain_path, dark, return_diagnostics=True)
    else:
        rel = {}
    phase_products = {}
    for iso, measurement in sorted(dark.items()):
        for key, phase in measurement.get("phases", {}).items():
            product = phase_products.setdefault(key, {"channel": phase["channel"],
                "color_index": phase["color_index"], "black_dn_log2iso": [],
                "read_noise_dn_log2iso": [], "stored_dark_variance_dn2_log2iso": [],
                "read_noise_unresolved_isos": []})
            for identity in ("color_desc", "color"):
                if identity in phase:
                    product[identity] = phase[identity]
            if product["color_index"] != phase["color_index"]:
                raise ValueError("dark CFA colour identity changes across ISO")
            x = math.log2(iso)
            product["black_dn_log2iso"].append([x, phase["bl"]])
            product["stored_dark_variance_dn2_log2iso"].append([x, phase["total_var"]])
            if phase["rn_dn"] is None:
                product["read_noise_unresolved_isos"].append(iso)
            else:
                product["read_noise_dn_log2iso"].append([x, phase["rn_dn"]])
    entry["phase_calibration"] = phase_products
    from .calibration_ptc import read_anchors, anchor_gain_graph, gain_jumps
    records = read_anchors(set_dir, dark)
    for record in records:
        input_hashes[record["file"]] = record["sha256"]
    entry["ptc_anchors"] = records
    components = entry.get("gain_ladder_diagnostics", {}).get("components", [])
    primary, gain_curve, intervals, conflicts = anchor_gain_graph(records, components)
    entry["gain_support_intervals"] = [list(i) for i in intervals]
    entry["ptc_anchor_diagnostics"] = {
        "weight_semantics": "equal independent anchor log-scale weights; fit uncertainty is conditional only",
        "conflicts": conflicts,
        "unanchored_components": [c["isos"] for c in components if not c.get("anchored")],
    }
    if gain_path:
        entry["source"]["formats"].append("JPTC-ISOGAIN/1")
    if primary is not None:
        a_iso, fit = primary["iso"], primary["fit"]
        entry["source"]["formats"].append("JPTC/2 (all qualified ptc anchors)")
        entry["ptc_anchor"] = _anchor_evidence(a_iso, fit)
        if conflicts:
            entry["ptc_anchor"]["quality"] = "conflicting-anchors"
        entry["gain_log2iso_log2epd"] = [[math.log2(i), math.log2(g)] for i,g in gain_curve.items()]
        entry["read_noise_log2iso_log2e"] = [
            [math.log2(i), math.log2(dark[i]["rn_dn"] * gain_curve[i])]
            for i in sorted(gain_curve) if i in dark and dark[i]["rn_dn"] is not None
            and dark[i]["rn_dn"] * gain_curve[i] > 0]
        entry["gain_jump_isos"] = sorted({j for c in components for j in gain_jumps(c["relative_gain"])})
        entry["unity_gain_ev"] = math.log2(a_iso * fit["gain_e_per_dn"])
        entry["fwc_e"] = fit["fwc_e"]
        entry["fwc_model_spread_e"] = fit["fwc_model_spread_e"]
        # Per-position gain is measured only where independently fitted PTC
        # columns can be mapped to the actual collector CFA layout. Never
        # clone the scalar green anchor into the other colour phases.
        for key, product in phase_products.items():
            measured = [(r["iso"], f) for r in records if r["status"] == "usable"
                for f in r.get("phase_fits", {}).values()
                if f.get("phase") == key and f.get("quality") == "ok"]
            if not measured:
                continue
            ranges = [f["white_level_used"]-f["black_level_used"] for _,f in measured]
            if max(ranges)/min(ranges)-1. > .05:
                product["gain_unavailable_reason"] = "phase DN ranges differ across anchors"
                continue
            values = {}
            for i,f in measured:
                values.setdefault(i,[]).append(f["gain_e_per_dn"])
            if any(max(v)/min(v)-1. > .05 for v in values.values()):
                product["gain_unavailable_reason"] = "conflicting same-ISO phase PTC anchors"
                continue
            product.update(gain_log2iso_log2epd=[[math.log2(i),float(np.mean(np.log2(v)))]
                    for i,v in sorted(values.items())],
                reference_dn_range=float(np.mean(ranges)),
                gain_provenance="independent-phase-ptc",
                fit_quality="ok", uncertainty="conditional fit error only; see ptc_anchors phase_fits; acquisition uncertainty not quantified",
                gain_support_intervals=[[i,i] for i in sorted(values)])
            error = [(i,f["gain_fit_standard_error"]) for i,f in measured if f.get("gain_fit_standard_error") is not None]
            if len(error) == len(values):
                product["gain_standard_error_e_per_dn_log2iso"] = [[math.log2(i),e] for i,e in sorted(error)]
    from .noise_spectrum import SCHEMA, dark_phase_mapping
    mapping_header, mapping_rows = _parse_rows(dark_path)
    spectrum = {"schema": SCHEMA, "mapping": dark_phase_mapping(mapping_header, mapping_rows), "axes": {}}
    for ax in ("h", "v"):
        sp = _find(set_dir, f"spectrum-{ax}.csv", f"*spectrum-{ax}.csv")
        if sp is not None:
            input_hashes[sp.name] = _sha256(sp)
            detail = read_whiteness(sp, phase_mapping=spectrum["mapping"], return_details=True,
                                   axis=ax, scalar_rows=mapping_rows)
            spectrum["axes"][ax] = {
                "ratios_log2iso": detail["ratios_log2iso"],
                "frequency_unit": "cycles/channel-plane-pixel",
                "ratio_bands": {"mid": [.05, .20], "high": [.35, .499]},
                "source_file": sp.name, "source_sha256": input_hashes[sp.name],
                **detail.get("complete", {}),
            }
            w = detail["summary"]
            if w:
                entry["source"]["formats"].append(f"JPTC-SPECTRUM/1 ({ax})")
                entry[f"noise_whiteness_{ax}_log2iso"] = [
                    [math.log2(i), round(v, 4)] for i, v in w.items()]
    if spectrum["axes"]:
        entry["noise_spectrum"] = spectrum
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
             "mode_scope": "camera-shutter-iso-dn-and-declared-readout",
             "unverified_readout_fields": {k: item[k] for k in ("compression", "geometry")
                                           if item.get(k) not in (None, [], [None, None], ["", ""])}}
    entry.update(stored_dark_variance_fields(item))
    entry.update(phase_calibration_fields(item))
    products = entry.get("phase_calibration") or {}
    mapped = [p for p in products.values() if p.get("channel") and p.get("color_desc")]
    if mapped:
        entry["noise_model_channels"] = ("partial-independent-phase-gain" if any(
            p.get("gain_provenance") == "independent-phase-ptc" for p in mapped)
            else "phase-temporal-variance/shared-gain")
    elif products:
        entry["noise_model_channels"] = "unmapped-phase-measurements/scalar-green"
    if item.get("noise_spectrum") is not None:
        from .noise_spectrum import validate_spectrum
        entry["noise_spectrum"] = validate_spectrum(item["noise_spectrum"])
    from .readout import measurement_fields
    entry.update(measurement_fields(item))
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
        if item.get("gain_support_intervals") is not None:
            entry["gain_support_intervals"] = _support_intervals(item["gain_support_intervals"])
        for provenance in ("ptc_anchors", "ptc_anchor_diagnostics", "gain_ladder_diagnostics"):
            if item.get(provenance) is not None:
                entry[provenance] = item[provenance]
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
            if entry["quality"].get("status") != "conflicting-anchors" and (at_anchor is None or abs(2**at_anchor/gain - 1) > .05):
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


def phase_calibration_fields(item: dict) -> dict:
    """Validate measured phase identity and units before runtime use."""
    products = item.get("phase_calibration")
    if products is None:
        return {}
    if not isinstance(products, dict):
        raise ValueError("phase_calibration must be an object")
    out = {}
    for key, record in products.items():
        if not isinstance(record, dict) or not isinstance(key, str):
            raise ValueError("phase calibration must contain named objects")
        cid = record.get("color_index")
        if type(cid) is not int or cid not in range(4):
            raise ValueError("phase calibration ColorIndex must be an integer 0..3")
        channel = record.get("channel")
        if channel not in (None, "C00", "C01", "C10", "C11") or (channel and key != channel):
            raise ValueError("phase calibration channel must match its Cxx position")
        if channel is None and key != f"index:{cid}":
            raise ValueError("legacy phase calibration key must match its ColorIndex")
        clean = {"channel": channel, "color_index": cid}
        description = record.get("color_desc")
        if description is not None:
            if (not isinstance(description, str) or not 1 <= len(description) <= 4
                    or not description.isalpha() or cid >= len(description)
                    or record.get("color") != description[cid]):
                raise ValueError("phase calibration colour disagrees with color_desc")
            clean.update(color_desc=description, color=record["color"])
        elif record.get("color") is not None:
            raise ValueError("phase calibration colour needs its original color_desc")
        for curve in ("black_dn_log2iso", "read_noise_dn_log2iso", "stored_dark_variance_dn2_log2iso"):
            clean[curve] = _curve(record.get(curve), f"phase {key} {curve}",
                                  logarithmic=False, allow_zero=True)
        clean["read_noise_unresolved_isos"] = _read_noise_unresolved_isos(
            record.get("read_noise_unresolved_isos"), clean["read_noise_dn_log2iso"])
        if record.get("gain_log2iso_log2epd") is not None:
            clean["gain_log2iso_log2epd"] = _curve(record["gain_log2iso_log2epd"], f"phase {key} gain")
            clean["reference_dn_range"] = _number(record.get("reference_dn_range"), f"phase {key} DN range")
            if record.get("gain_provenance") != "independent-phase-ptc":
                raise ValueError("phase gain must have independent-phase-ptc provenance")
            clean["gain_provenance"] = "independent-phase-ptc"
            if record.get("gain_support_intervals") is not None:
                clean["gain_support_intervals"] = _support_intervals(record["gain_support_intervals"])
        for evidence in ("uncertainty", "fit_quality", "gain_unavailable_reason"):
            if record.get(evidence) is not None:
                value = record[evidence]
                if not isinstance(value, str) or not value or len(value) > 2048:
                    raise ValueError(f"phase {evidence} must be bounded nonempty text")
                clean[evidence] = value
        if record.get("gain_standard_error_e_per_dn_log2iso") is not None:
            clean["gain_standard_error_e_per_dn_log2iso"] = _curve(
                record["gain_standard_error_e_per_dn_log2iso"], f"phase {key} gain standard error",
                logarithmic=False, allow_zero=True)
        for x, physical in clean["read_noise_dn_log2iso"]:
            stored = next((v for px,v in clean["stored_dark_variance_dn2_log2iso"] if abs(px-x)<1e-7), None)
            if stored is not None and physical*physical > stored*1.05 + 1e-12:
                raise ValueError("phase stored variance is below physical read-noise variance")
        out[key] = clean
    return {"phase_calibration": out}


def _support_intervals(value):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("gain_support_intervals must be a list")
    result = []
    for interval in value:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("gain support interval must contain two ISO values")
        lo, hi = (_number(v, "gain support ISO") for v in interval)
        if lo > hi or hi > 2**24:
            raise ValueError("invalid gain support ISO domain")
        result.append([lo, hi])
    return sorted(result)


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
    if key in ("gain_log2iso_log2epd", "read_noise_log2iso_log2e") and "gain_support_intervals" in entry:
        if not any(lo-1e-7 <= iso <= hi+1e-7 for lo,hi in entry["gain_support_intervals"]):
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
        declared_raw_size = None
        for file, formats in [(dark, {"JPTC-DARK/1"}),
                              *[(p, {"JPTC/2"}) for p in source.glob("*ptc-iso*.csv") if "unusable" not in p.name],
                              *[(p, {"JPTC-ISOGAIN/1"}) for p in source.glob("*gain-levels*.csv")],
                              *[(p, {"JPTC-SPECTRUM/1"}) for p in source.glob("*spectrum-*.csv")]]:
            if file.stat().st_size > _MAX_JSON_BYTES:
                raise ValueError(f"{file.name}: measurement file too large")
            header, _ = _parse_rows(file)
            if header.get("Format") not in formats:
                raise ValueError(f"{file.name}: unsupported measurement format")
            if header.get("RawSize"):
                from .readout import collect_fields
                raw_size = collect_fields(header)["readout_contract"]["libraw_raw_geometry"]
                if declared_raw_size is not None and raw_size != declared_raw_size:
                    raise ValueError(f"{file.name}: RawSize disagrees with other measurements")
                declared_raw_size = raw_size
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
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, indent=2)
    size = sum(len(piece.encode("utf-8")) for piece in encoder.iterencode(record)) + 1
    if size > _MAX_JSON_BYTES:
        raise ValueError("combined calibration record is too large; split the measurement set before importing")
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
                  iso: float | None, readout: dict | None = None, *, _check_readout: bool = True) -> str:
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
    if _check_readout:
        from .readout import match
        status, reason = match(prior, readout)
        if status in ("unverified", "mismatch"):
            return reason
    return "usable"


def calibration_diagnostics(make: str | None, model: str | None, shutter: str | None = None,
                            iso: float | None = None, readout: dict | None = None) -> list[dict]:
    out = []
    for path, record, prior, error in _records():
        if error is not None:
            out.append({"id": path.stem, "status": "invalid", "reason": error})
            continue
        reason = _match_reason(prior, make or "", model or "", shutter, iso, readout)
        if reason == "camera-mismatch":
            continue
        if not record.get("active"):
            reason = "inactive"
        noise_status = read_noise_status(prior, iso)
        stored_variance, stored_status = stored_dark_variance(prior, iso)
        from .readout import match
        readout_status, readout_reason = match(prior, readout)
        warnings = _summary(record, path, prior)["warnings"]
        if readout_status == "matched":
            warnings = [warning for warning in warnings if warning != "sub-readout-mode-not-verified"]
        out.append({"id": record["id"], "label": prior["id"],
                    "status": ("gain-only" if reason == "usable" and noise_status.startswith("read-noise-unresolved")
                               else "usable" if reason == "usable" else "not-applied"),
                    "reason": noise_status if reason == "usable" and noise_status.startswith("read-noise-unresolved") else reason,
                    "gain_status": "usable" if reason == "usable" else "not-applied",
                    "read_noise_status": noise_status,
                    "stored_dark_variance_status": stored_status,
                    "stored_dark_variance_dn2": stored_variance,
                    "readout_match_status": readout_status, "readout_match_reason": readout_reason,
                    "readout_contract": prior.get("readout_contract"),
                    "capture_readout": readout,
                    "has_read_noise_at_iso": iso is not None and curve_value(prior, "read_noise_log2iso_log2e", iso) is not None,
                    "mode_scope": prior["mode_scope"],
                    "unverified_readout_fields": ({} if readout_status == "matched" else prior.get("unverified_readout_fields", {})),
                    "warnings": warnings})
    return out


def matching_prior(make: str, model: str, *, shutter: str | None = None,
                   iso: float | None = None, readout: dict | None = None) -> dict | None:
    candidates = []
    for path, record, prior, error in _records():
        if record is None or not record.get("active"):
            continue
        if _match_reason(prior, make, model, shutter, iso, _check_readout=False) != "usable":
            continue
        entry = dict(prior)
        entry["calibration_id"] = record["id"]
        entry["source"] = f"User JPTC calibration ({record['id']})"
        entry["mode_match"] = "user-explicit-any-mode" if prior.get("shutter") == "any" else "user-exact-shutter"
        entry["model_equals"] = set(entry["model_equals"])
        # Preserve the selected user calibration on a readout failure. Skipping
        # it would silently borrow an equally unverified curated/bulk model.
        from .priors import with_readout
        entry = with_readout(entry, readout, shutter=shutter)
        priority = {"matched": 2, "not-declared": 1}.get(entry["readout_match_status"], 0)
        candidates.append((priority, record.get("imported_at", ""), record["id"], entry))
    return max(candidates, key=lambda c: c[:3])[3] if candidates else None
