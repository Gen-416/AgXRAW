# SPDX-License-Identifier: GPL-3.0-or-later
"""PTC estimator contracts and independent anchor qualification."""
from __future__ import annotations

import math
import re

import numpy as np


def fit_temporal(means_a, means_b, difference_std, black, white, factor=1., clip_fraction=None):
    """Fit stored-DN temporal variance = Std(A-B)^2 / (2*clip factor).

    No quantisation term is added or subtracted. Mean-pair drift can leave
    PRNU in the difference, so pairs differing by more than 1% of signal
    are excluded. This declared engineering limit is not a proof of stable
    lighting or equal camera processing between the frames.
    """
    arrays = [np.asarray(a, dtype=np.float64) for a in (means_a, means_b, difference_std)]
    a, b, sigma = arrays
    if any(not np.isfinite(v).all() for v in arrays) or np.any(sigma < 0):
        raise ValueError("differential PTC statistics must be finite and nonnegative")
    if not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError("differential PTC ClipVarianceFactor must be in (0,1]")
    signal = (a + b) / 2. - black
    variance = sigma**2 / (2. * factor)
    drift = np.abs(a-b) / np.maximum(signal, 1.)
    usable = (a < .995*white) & (b < .995*white) & (signal > 0.) & (drift <= .01)
    if clip_fraction is not None:
        clipping = np.asarray(clip_fraction, dtype=np.float64)
        if not np.isfinite(clipping).all() or np.any((clipping < 0) | (clipping > 1)):
            raise ValueError("invalid differential PTC clipping fraction")
        usable &= clipping <= .01
    span = white-black
    alternatives, results = {}, {}
    for fraction in (.10, .35):
        selection = usable & (signal < fraction*span)
        x, y = signal[selection], variance[selection]
        if len(x) < 4:
            continue
        keep = np.ones(len(x), dtype=bool)
        for _ in range(4):
            design = np.stack((x[keep], np.ones(keep.sum())), axis=1)
            slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
            residual = y - slope*x - intercept
            rms = float(np.sqrt(np.mean(residual[keep]**2)))
            new = np.abs(residual) <= max(2.5*rms, 1e-10*max(float(np.mean(y)),1.))
            if new.sum() < 4 or np.array_equal(new,keep):
                break
            keep = new
        design = np.stack((x[keep], np.ones(keep.sum())), axis=1)
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        if slope <= 0:
            continue
        residual = y[keep] - design @ np.array([slope,intercept])
        rms = float(np.sqrt(np.mean(residual**2)) / max(float(np.mean(y[keep])),1e-30))
        inverse = np.linalg.pinv(design.T @ design)
        leverage = np.sum((design @ inverse)*design,axis=1)
        scaled = residual/np.maximum(1.-leverage,1e-6)
        covariance = inverse @ ((design*scaled[:,None]).T @ (design*scaled[:,None])) @ inverse
        gain = float(1./slope)
        error = float(math.sqrt(max(0.,covariance[0,0]))/slope**2)
        alternatives[f"temporal-linear-{fraction:.2f}"] = gain
        results[fraction] = (gain,float(intercept),rms,int(keep.sum()),int((~keep).sum()),error)
    if not results:
        raise ValueError("not enough stable unsaturated pairs for a temporal PTC fit")
    fraction = .35 if .35 in results else .10
    gain, intercept, rms, count, removed, error = results[fraction]
    spread = (max(alternatives.values())-min(alternatives.values()))/gain
    read_dn = math.sqrt(max(intercept,0.))
    slope_error = error/gain**2
    slope = 1./gain
    interval = [1./(slope+1.96*slope_error),1./(slope-1.96*slope_error)] if slope > 1.96*slope_error else None
    return {"gain_e_per_dn": gain, "fit_model": "paired-frame temporal linear PTC",
        "fit_model_effective": f"temporal-linear-{fraction:.2f}", "fit_window_frac": fraction,
        "gain_alternatives": alternatives, "gain_estimator_spread_rel": spread,
        "gain_fit_standard_error": error,
        "gain_fit_interval_95": interval,
        "gain_fit_interval_status": "conditional-asymptotic" if interval else "upper-unbounded",
        "gain_fit_interval_semantics": "HC3 normal-approximation slope interval inverted to gain; not total acquisition uncertainty",
        "fit_uncertainty_semantics": "conditional HC3 regression error; excludes lighting, clipping and calibration systematics",
        "prnu_status": "excluded-by-pair-difference", "prnu": None,
        "read_noise_dn": read_dn, "read_noise_e": read_dn*gain,
        "read_noise_status": "stored-temporal-intercept" if intercept > 0 else "below-resolution",
        "fwc_e": span*gain, "fwc_model_spread_e": span*gain*spread,
        "fwc_semantics": "ADC code-saturation capacity (white-black)*gain; physical full well not claimed",
        "last_unsaturated_signal_e": float(signal[usable].max())*gain,
        "fit_relative_rms": rms, "fit_points": count, "fit_points_excluded": removed,
        "sat_plateau_dn": span, "quality": "high-residual" if rms > .05 else "ok",
        "variance_domain": "stored-linearized-raw-dn2", "difference_variance_divisor": 2.*factor,
        "pair_drift_limit_relative": .01, "pair_drift_rejected": int((drift>.01).sum())}


def fit_channel(header, rows, channel, black, white):
    from .calibration import fit_ptc
    means = np.asarray([float(r[f"{channel}_Mean"]) for r in rows])
    std = np.asarray([float(r[f"{channel}_Std"]) for r in rows])
    if not np.isfinite(means).all() or not np.isfinite(std).all() or np.any(std<0):
        raise ValueError("PTC statistics must be finite and standard deviations nonnegative")
    try:
        spatial = fit_ptc(means,std,black,white)
    except ValueError as exc:
        spatial = {"status": "unavailable", "reason": str(exc)}
    columns = rows[0] if rows else {}
    reason = "paired-fields-unavailable"
    if f"{channel}_MeanB" in columns:
        source = None
        factor = 1.
        try:
            candidate = float(header.get("ClipVarianceFactor", "nan"))
        except ValueError:
            candidate = float("nan")
        if f"{channel}_StdDiffClipped" in columns and 0 < candidate <= 1:
            source, factor = f"{channel}_StdDiffClipped", candidate
        elif f"{channel}_StdDiff" in columns:
            source = f"{channel}_StdDiff"
        else:
            reason = "difference-sigma-clip-correction-unresolved"
        if source is not None:
            try:
                temporal = fit_temporal(means,[float(r[f"{channel}_MeanB"]) for r in rows],
                    [float(r[source]) for r in rows],black,white,factor,
                    [float(r[f"{channel}_ClipFrac"]) for r in rows] if f"{channel}_ClipFrac" in columns else None)
            except ValueError as exc:
                reason = f"temporal-fit-unavailable: {exc}"
            else:
                temporal["temporal_variance_source"] = source
                temporal["spatial_crosscheck"] = {key: spatial.get(key) for key in
                    ("gain_e_per_dn","fit_model_effective","quality","fit_relative_rms","prnu")}
                if spatial.get("gain_e_per_dn"):
                    temporal["spatial_temporal_gain_disagreement_relative"] = spatial["gain_e_per_dn"]/temporal["gain_e_per_dn"]-1.
                return temporal
    if not spatial.get("gain_e_per_dn"):
        raise ValueError(f"no usable temporal or spatial PTC: {reason}")
    spatial["temporal_fallback_reason"] = reason
    return spatial


def _channel_phase(dark_iso, channel):
    """PTC G1 shares a sensor row with R; cdesc alone is not a layout."""
    phases = dark_iso.get("phases", {})
    mapped = {key: value for key,value in phases.items()
              if key in ("C00", "C01", "C10", "C11") and value.get("color") in ("R", "G", "B")}
    if channel in ("R", "B"):
        matches = [(key,value) for key,value in mapped.items() if value["color"] == channel]
    else:
        partner = "R" if channel == "G1" else "B"
        row = next((key[1] for key,value in mapped.items() if value["color"]==partner), None)
        matches = [(key,value) for key,value in mapped.items() if value["color"]=="G" and key[1]==row]
    return matches[0] if len(matches)==1 else (None,None)


def read_anchors(set_dir, dark):
    from .calibration import _parse_rows, _sha256, infer_white
    result = []
    for path in sorted(set_dir.glob("*ptc-iso*.csv")):
        record = {"file":path.name,"sha256":_sha256(path),"status":"rejected"}
        try:
            match = re.search(r"ptc-iso(\d+)",path.stem)
            if not match:
                raise ValueError("PTC filename has no ISO")
            iso = int(match[1])
            if iso <= 0 or "unusable" in path.name:
                raise ValueError("PTC marked unusable or invalid ISO")
            record["iso"] = iso
            header, rows = _parse_rows(path)
            if not rows:
                raise ValueError("empty PTC file")
            if header.get("ISO") and float(header["ISO"]) != iso:
                raise ValueError("PTC declared ISO disagrees with filename ISO")
            descriptions = {p["color_desc"] for p in dark.get(iso,{}).get("phases",{}).values()
                            if p.get("color_desc")}
            if header.get("CfaPattern") and descriptions and descriptions != {header["CfaPattern"].strip().upper()}:
                raise ValueError("PTC colour description disagrees with dark CFA mapping")
            black_values = [float(v) for v in header.get("BlackLevel","").split(",") if v.strip()]
            phase_fits = {}
            for channel,index in (("R",0),("G1",1),("G2",3),("B",2)):
                if f"{channel}_Mean" not in rows[0] or f"{channel}_Std" not in rows[0]:
                    continue
                phase_key, phase = _channel_phase(dark.get(iso,{}), channel)
                cid = phase["color_index"] if phase is not None else index
                black = black_values[cid] if len(black_values)>cid else None
                if black is None:
                    candidates = [phase] if phase is not None else [p for p in dark.get(iso,{}).get("phases",{}).values()
                        if p["color_index"]==index and not p.get("color_desc")]
                    if len(candidates)==1:
                        black = candidates[0]["bl"]
                    elif channel=="G1":
                        black = dark.get(iso,{}).get("bl")
                if black is None:
                    continue
                means = np.asarray([float(r[f"{channel}_Mean"]) for r in rows])
                std = np.asarray([float(r[f"{channel}_Std"]) for r in rows])
                white = infer_white(means,std)
                if white is None:
                    continue
                fit = fit_channel(header,rows,channel,black,white)
                fit["black_level_used"],fit["white_level_used"] = black,white
                if phase_key is not None:
                    fit["phase"] = phase_key
                    fit["color_index"] = cid
                phase_fits[channel] = fit
            if "G1" not in phase_fits:
                raise ValueError("G1 anchor needs black level and a measurable white plateau")
            record.update(fit=phase_fits["G1"],phase_fits=phase_fits)
            record["status"] = "usable" if record["fit"]["quality"]=="ok" else "rejected"
            if record["status"]!="usable":
                record["reason"] = "quality-"+record["fit"]["quality"]
        except (ValueError,KeyError,TypeError,OverflowError) as exc:
            record["reason"] = str(exc)
        result.append(record)
    return sorted(result,key=lambda r:(r.get("iso",float("inf")),r["file"]))


def anchor_gain_graph(records, components):
    """Anchor each connected component; disconnected measurements never bridge."""
    from .calibration import _pchip
    usable = sorted((r for r in records if r["status"]=="usable"),key=lambda r:(r["iso"],r["file"]))
    primary = usable[0] if usable else None
    curve, intervals, conflicts = {},[],[]
    # The scalar runtime curve has one reference DN range. Different code
    # scales cannot inherit the first anchor's denominator by coincidence.
    spans = [(r, r['fit']['white_level_used']-r['fit']['black_level_used']) for r in usable
             if 'white_level_used' in r['fit'] and 'black_level_used' in r['fit']]
    if spans and max(v for _,v in spans)/min(v for _,v in spans)-1. > .05:
        conflicts.append({'files':[r['file'] for r,_ in spans],
            'reason':'PTC reference DN ranges disagree by more than existing 5% scale tolerance',
            'reference_dn_ranges':{r['file']:v for r,v in spans}})
    used = set()
    assignments = {id(c): [] for c in components}
    for record in usable:
        exact = [c for c in components if record["iso"] in c["relative_gain"]]
        candidates = exact or [c for c in components if min(c["relative_gain"]) < record["iso"] < max(c["relative_gain"])
            and not crosses_gain_jump(c["relative_gain"],record["iso"])]
        if len(candidates)==1:
            assignments[id(candidates[0])].append(record)
        elif len(candidates)>1:
            record["anchor_attachment"] = "ambiguous-overlapping-components; independent point only"
    for component in components:
        rel = component["relative_gain"]
        isos = sorted(rel)
        attached = assignments[id(component)]
        if not attached:
            component["anchor_status"]="unanchored"
            continue
        scales = []
        for record in attached:
            iso = record["iso"]
            ratio = rel.get(iso)
            if ratio is None:
                segment = next(s for s in support_segments(rel) if s[0] <= iso <= s[-1])
                ratio = 2.**_pchip(np.log2(segment),np.log2([rel[i] for i in segment]),math.log2(iso))
            scales.append(record["fit"]["gain_e_per_dn"]/ratio)
            used.add(record["file"])
        scale = float(np.exp(np.mean(np.log(scales))))
        spread = max(scales)/min(scales)-1.
        component.update(anchored=True,anchor_files=[r["file"] for r in attached],
                         anchor_scale_spread_relative=spread,anchor_status="conflict" if spread>.05 else "consistent")
        if spread>.05:
            conflicts.append({"files":component["anchor_files"],"relative_scale_spread":spread,
                              "reason":"PTC anchors disagree by more than existing 5% scale tolerance"})
        curve.update({iso:ratio*scale for iso,ratio in rel.items()})
        # Include each independent anchor as a measured point as well.
        for iso in sorted({r["iso"] for r in attached}):
            values = [r["fit"]["gain_e_per_dn"] for r in attached if r["iso"]==iso]
            curve[iso] = float(np.exp(np.mean(np.log(values))))
        overlaps = any(other is not component and other.get("anchored") and
            max(min(rel),min(other["relative_gain"])) < min(max(rel),max(other["relative_gain"]))
            for other in components)
        # Two disconnected graphs may have interleaved ISO values. A single
        # flattened runtime curve cannot safely interpolate their overlap.
        overlaps = overlaps or any(other is not component and assignments[id(other)] and
            max(min(rel),min(other["relative_gain"])) < min(max(rel),max(other["relative_gain"]))
            for other in components)
        if overlaps:
            component["interpolation_status"] = "overlapping disconnected domains; measured points only"
            intervals.extend([iso,iso] for iso in sorted(set(isos) | {r["iso"] for r in attached}))
        else:
            intervals.extend([s[0],s[-1]] for s in support_segments(rel))
    for iso in sorted({r["iso"] for r in usable if r["file"] not in used}):
        standalone = [r for r in usable if r["file"] not in used and r["iso"]==iso]
        gains = [r["fit"]["gain_e_per_dn"] for r in standalone]
        gain = float(np.exp(np.mean(np.log(gains))))
        if max(gains)/min(gains)-1 > .05:
            conflicts.append({"iso":iso,"files":[r["file"] for r in standalone],"reason":"independent PTC anchors disagree"})
        if iso in curve:
            continue
        curve[iso] = gain
        intervals.append([iso,iso])
    return primary,dict(sorted(curve.items())),sorted({tuple(i) for i in intervals}),conflicts


def gain_jumps(relative):
    """Existing plateau-to-plateau candidates, without claiming DCG identity."""
    isos = sorted(relative)
    u = [relative[i]*i for i in isos]
    flat = lambda a,b: abs(math.log(b/a)) < math.log(1.08)
    return [isos[k] for k in range(2,len(u)-1)
            if flat(u[k-2],u[k-1]) and flat(u[k],u[k+1]) and u[k]/u[k-1]>1.15]


def crosses_gain_jump(relative, iso):
    isos = sorted(relative)
    return any(a < iso < b and b in gain_jumps(relative) for a,b in zip(isos,isos[1:]))


def support_segments(relative):
    jumps = set(gain_jumps(relative))
    segments = []
    for iso in sorted(relative):
        if not segments or iso in jumps:
            segments.append([])
        segments[-1].append(iso)
    return segments
