#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Import a JPTC collect set (y-g-jiang first-party bench measurements).

A collect set directory (https://y-g-jiang.github.io/data/collect/<id>/)
holds up to four instruments for one camera+mode:

  ptc-iso*.csv    JPTC/2       exposure ramp  -> absolute gain / FWC anchor
  gain-levels.csv JPTC-ISOGAIN/1 per-ISO means -> RELATIVE gain vs ISO
                  (g1/g2 = (t1/t2)*(M2-BL2)/(M1-BL1), the file's own formula)
  dark-scalars.csv JPTC-DARK/1  paired darks   -> temporal read noise per ISO
                  (StdDiff is the std of A-B verbatim; /sqrt(2) for one
                  frame, /sqrt(ClipVarianceFactor) undoes the declared
                  sigma clip), plus row/column banding decomposition
  spectrum-h/v.csv JPTC-SPECTRUM/1 noise power spectra -> whiteness metric
                  (high-band over mid-band mean power of the pair-difference
                  spectrum: ~1 = white/clean, <1 = spatial filtering baked
                  into the RAW, >1 = sharpening)

Derived entry (format dngscan-jptc-collect-1): absolute gain curve
gain(iso) anchored at the PTC fit, read-noise curves in DN and electrons,
dual-conversion-gain switch detection (gain*iso jump >15%), FWC with the
PTC bracket uncertainty, banding and whiteness evidence. Licensing: the
author granted credit-based use 2026-08-25 (NOTICE.md).

Usage:
    python tools/import_jptc_collect.py <set-dir> --out dngscan/data/priors/jptc_collect/<id>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from import_jptc import fit_ptc  # noqa: E402

GREEN_INDICES = {1, 3}          # LibRaw colour indices for G/G2
CLIP_FRAC_MAX = 0.01


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
    try:
        factor = float(header.get("ClipVarianceFactor", 1.0))
    except ValueError:
        factor = 1.0  # old collector wrote 'undefined'; clip left uncorrected
    try:
        adc_step = float(header.get("AdcStep", 0.0))
    except ValueError:
        adc_step = 0.0
    per_iso: dict[int, dict] = {}
    greens_present = any(int(r["ColorIndex"]) in GREEN_INDICES for r in rows)
    for r in rows:
        # monochrome sensors (e.g. M11 Monochrom) use a single colour index;
        # fall back to every plane when no green-indexed rows exist
        if greens_present and int(r["ColorIndex"]) not in GREEN_INDICES:
            continue
        iso = int(r["ISO"])
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
    from import_jptc import infer_white
    white = infer_white(g1_mean, g1_std)
    if white is None:
        return None
    return iso, fit_ptc(g1_mean, g1_std, black, white)


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
        "noise_aperture": "paired-frame temporal std (FPN excluded); sigma clip "
                          "undone via the declared ClipVarianceFactor; Sheppard "
                          "step^2/12 quantisation correction applied per frame",
    }
    entry["source"]["formats"].append("JPTC-DARK/1")
    entry["source"]["inputs"] = input_hashes
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
        entry["ptc_anchor"] = {
            "iso": a_iso, "gain_e_per_dn": fit["gain_e_per_dn"],
            "read_noise_e": fit["read_noise_e"], "fwc_e": fit["fwc_e"],
            "fwc_e_uncertainty": fit["fwc_e_uncertainty"],
            "prnu": fit["prnu"], "fit_relative_rms": fit["fit_relative_rms"],
            "fit_model": fit["fit_model"], "model_sensitivity": fit["model_sensitivity"],
            "quality": fit["quality"],
        }
        entry["unity_gain_ev"] = round(math.log2(a_iso * fit["gain_e_per_dn"]), 4)
        entry["fwc_e"] = fit["fwc_e"]
        entry["fwc_e_uncertainty"] = fit["fwc_e_uncertainty"]
        if a_iso in dark:
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
            from scipy.interpolate import PchipInterpolator
            xs = np.log2(np.asarray(sorted(rel)))
            ys = np.log2(np.asarray([rel[i] for i in sorted(rel)]))
            rel_at_anchor = float(2.0 ** PchipInterpolator(
                xs, ys, extrapolate=False)(math.log2(a_iso)))
        else:
            rel_at_anchor = rel[a_iso]
        scale = fit["gain_e_per_dn"] / rel_at_anchor
        gain_curve = {iso: r * scale for iso, r in rel.items()}
        entry["ptc_anchor"] = {
            "iso": a_iso, "gain_e_per_dn": fit["gain_e_per_dn"],
            "read_noise_e": fit["read_noise_e"], "fwc_e": fit["fwc_e"],
            "fwc_e_uncertainty": fit["fwc_e_uncertainty"],
            "prnu": fit["prnu"], "fit_relative_rms": fit["fit_relative_rms"],
            "fit_model": fit["fit_model"], "model_sensitivity": fit["model_sensitivity"],
            "quality": fit["quality"],
        }
        entry["gain_log2iso_log2epd"] = [
            [math.log2(i), math.log2(g)] for i, g in gain_curve.items()]
        entry["read_noise_log2iso_log2e"] = [
            [math.log2(i), math.log2(dark[i]["rn_dn"] * gain_curve[i])]
            for i in sorted(gain_curve)
            if i in dark and dark[i]["rn_dn"] is not None
            and dark[i]["rn_dn"] * gain_curve[i] > 0]
        # DCG detection: gain*iso is constant under the reciprocal law;
        # an upward jump >15% between adjacent ISOs is a conversion-gain
        # switch (downward jumps are extended-ISO re-scaling, ignored).
        # DCG detection: gain*iso is constant under the reciprocal law, so a
        # conversion-gain switch is a jump BETWEEN two locally flat plateaus.
        # Extended-ISO segments make u rise from the very start (flat gain),
        # so a jump only counts when both neighbours are plateau-like
        # (adjacent ratio < 1.08 on each side).
        # A plateau-to-plateau jump is EITHER a conversion-gain switch OR
        # the extended-to-native-base boundary; the ladder alone cannot tell
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
        entry["fwc_e_uncertainty"] = fit["fwc_e_uncertainty"]
    for ax in ("h", "v"):
        sp = _find(set_dir, f"spectrum-{ax}.csv", f"*spectrum-{ax}.csv")
        if sp is not None:
            input_hashes[sp.name] = _sha256(sp)
            w = read_whiteness(sp)
            if w:
                entry["source"]["formats"].append(f"JPTC-SPECTRUM/1 ({ax})")
                entry[f"noise_whiteness_{ax}_log2iso"] = [
                    [round(math.log2(i), 4), round(v, 4)] for i, v in w.items()]
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets-json", type=Path, default=None,
                    help="sets.json for tester/measuredAt metadata")
    args = ap.parse_args()
    meta = None
    if args.sets_json and args.sets_json.exists():
        for s in json.loads(args.sets_json.read_text()).get("sets", []):
            if s.get("id") == args.set_dir.name:
                meta = s
    entry = build(args.set_dir, meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entry, ensure_ascii=False, indent=1) + "\n")
    haves = ", ".join(entry["source"]["formats"])
    print(f"wrote {args.out.name}: {entry['camera']} [{haves}] "
          f"rn_pts={len(entry.get('read_noise_dn_log2iso', []))} "
          f"gain_pts={len(entry.get('gain_log2iso_log2epd', []))} "
          f"jumps={entry.get('gain_jump_isos')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
