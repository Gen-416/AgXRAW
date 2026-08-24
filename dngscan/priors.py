# SPDX-License-Identifier: GPL-3.0-or-later
"""Static sensor priors from public measurements (PhotonsToPhotos, Bill Claff).

These are *published chart data*, not our own bench measurements: they calibrate the
absolute scale (electrons, PDR) that a single frame cannot provide, while the empirical
per-frame analysis remains the primary signal. Everything degrades gracefully to None
when the camera or ISO is unknown.

Data source: https://www.photonstophotos.net/Charts/PDR.htm and
https://www.photonstophotos.net/Charts/RN_e.htm (series extracted 2026-07-06).
x is log2(ISO); PDR y is EV; read-noise y is log2(input-referred electrons).
Points P2P plots with hollow markers (suspect/NR-affected) are kept but the threshold
is recorded in `suspect_iso_min`.
"""

from __future__ import annotations

import math
from typing import Any

# Sigma fp (full-frame 24MP BSI, 14-bit). unity_gain_ev: ISO at which 1 DN = 1 e-
# is 2**7.29 ~= 156. fwc_e is P2P's saturation at the lowest-gain point.
SIGMA_FP = {
    "id": "Sigma fp",
    "make_contains": "SIGMA",
    "model_equals": {"SIGMA FP", "FP"},
    "unity_gain_ev": 7.29,
    "fwc_e": 74884,
    "pdr_log2iso_ev": [
        (5.00, 11.02), (5.33, 10.98), (5.67, 11.00), (6.00, 11.00), (6.33, 10.70),
        (6.67, 10.41), (7.00, 9.85), (7.33, 9.22), (7.67, 9.38), (8.00, 9.38),
        (8.33, 9.41), (8.67, 9.38), (9.00, 9.07), (9.33, 8.73), (9.67, 8.40),
        (10.00, 8.07), (10.33, 7.75), (10.67, 7.42), (11.00, 7.10), (11.33, 6.78),
        (11.67, 6.46), (12.00, 6.09), (12.33, 5.79), (12.67, 5.46), (13.00, 5.10),
        (13.33, 4.80), (13.67, 4.46), (14.00, 4.11), (14.33, 3.82), (14.67, 3.47),
        (15.00, 3.13),
    ],
    "read_noise_log2iso_log2e": [
        (5.00, 2.76), (5.33, 2.41), (5.67, 2.09), (6.00, 1.78), (6.33, 1.68),
        (6.67, 1.63), (7.00, 1.83), (7.33, 2.04), (7.67, 0.70), (8.00, 0.34),
        (8.33, 0.01), (8.67, -0.35), (9.00, -0.40), (9.33, -0.43), (9.67, -0.46),
        (10.00, -0.49), (10.33, -0.54), (10.67, -0.56), (11.00, -0.58), (11.33, -0.61),
        (11.67, -0.63), (12.00, -0.65), (12.33, -0.68), (12.67, -0.68), (13.00, -0.71),
        (13.33, -0.77), (13.67, -0.72), (14.00, -0.73), (14.33, -0.73), (14.67, -0.77),
        (15.00, -0.72),
    ],
    # P2P marks values from here up with hollow markers. Note: fp's read noise below
    # ~1 e- from ISO ~400 is widely attributed to spatial filtering baked into the DNG;
    # the empirical RAW-health autocorrelation check is the per-frame verdict on that.
    "suspect_iso_min": 10322,
    "dcg_switch_iso": 200,  # read-noise curve drops sharply at log2(ISO)=7.67
    "source": "PhotonsToPhotos PDR.htm / RN_e.htm, retrieved 2026-07-06",
}

def _load_json_priors() -> list[dict[str, Any]]:
    """Chart-extracted entries from sensor_priors.json (same schema, list values).

    Kept as data rather than code: adding a camera is extracting its
    PhotonsToPhotos series and appending an entry — no code change. Curves are
    stored as [[log2iso, y], ...] and normalized to tuples here.
    """
    import json
    from pathlib import Path

    path = Path(__file__).with_name("sensor_priors.json")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = []
    for item in raw.get("priors", []):
        entry = dict(item)
        for key in ("pdr_log2iso_ev", "read_noise_log2iso_log2e"):
            entry[key] = [(float(x), float(y)) for x, y in entry.get(key, [])]
        entry["model_equals"] = {str(m).upper() for m in entry.get("model_equals", [])}
        entries.append(entry)
    return entries


PRIOR_TABLE = [SIGMA_FP] + _load_json_priors()

# Secondary tables, loaded lazily on the first curated-table miss:
#   tier 2 — JPTC/2 first-party bench measurements (data/priors/jptc/*.json),
#            single-ISO PTC fits; gain/FWC are measured, read noise is a
#            one-point curve (constant extrapolation), no PDR curve.
#   tier 3 — P2P bulk table (data/priors/p2p_bulk.json), 135 cameras derived
#            from PhotonsToPhotos chart data; provenance and the licensing
#            decision live in that file's header and in NOTICE.md. Deleting
#            the file removes the tier; everything degrades to None.
_JPTC_CACHE: list[dict[str, Any]] | None = None
_BULK_CACHE: list[dict[str, Any]] | None = None


def _jptc_entries() -> list[dict[str, Any]]:
    global _JPTC_CACHE
    if _JPTC_CACHE is not None:
        return _JPTC_CACHE
    import json
    from pathlib import Path

    entries: list[dict[str, Any]] = []
    for path in sorted((Path(__file__).parent / "data" / "priors" / "jptc").glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if item.get("format") != "dngscan-jptc-prior-1":
            continue
        iso = float(item["iso"])
        gain = float(item["gain_e_per_dn"])
        if iso <= 0 or gain <= 0:
            continue
        x = math.log2(iso)
        # read_noise_e == 0 means "below the single-frame PTC's resolution"
        # (mechanical-shutter floors of ~0.4 DN clamp the intercept at zero),
        # not a measured value — omit the curve so consumers get None.
        rn = float(item.get("read_noise_e") or 0.0)
        rn_curve = [(x, math.log2(rn))] if rn > 0 else []
        shutter = item.get("shutter")
        entries.append({
            "id": item["id"],
            "make_contains": item["brand"],
            "model_equals": {str(item["model"]).upper()},
            "unity_gain_ev": math.log2(iso * gain),
            "fwc_e": float(item["fwc_e"]),
            "fwc_e_uncertainty": float(item.get("fwc_e_uncertainty", 0.0)),
            "read_noise_log2iso_log2e": rn_curve,
            "pdr_log2iso_ev": [],
            "measured_iso": int(iso),
            "shutter": shutter,
            "prnu": item.get("prnu"),
            "source": f"JPTC/2 first-party measurement ({path.name})",
        })
    # One model can have several measurements (shutter modes, ISOs). Order
    # decides which one find_priors returns: lowest ISO first (fwc_e is at
    # the measured ISO, so only the lowest-ISO entry approximates native
    # full well), then entries whose read noise resolved, then mechanical.
    entries.sort(key=lambda e: (
        e["measured_iso"],
        0 if e["read_noise_log2iso_log2e"] else 1,
        0 if e.get("shutter") == "mechanical" else 1,
    ))
    _JPTC_CACHE = entries
    return entries


def _bulk_entries() -> list[dict[str, Any]]:
    global _BULK_CACHE
    if _BULK_CACHE is not None:
        return _BULK_CACHE
    import json
    from pathlib import Path

    path = Path(__file__).parent / "data" / "priors" / "p2p_bulk.json"
    entries: list[dict[str, Any]] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if raw.get("format") == "dngscan-p2p-bulk-priors-1":
        for item in raw.get("entries", []):
            entry = dict(item)
            for key in ("pdr_log2iso_ev", "read_noise_log2iso_log2e"):
                entry[key] = [(float(x), float(y)) for x, y in entry.get(key, [])]
            entry["source"] = "PhotonsToPhotos via p2p_bulk.json (see its provenance header)"
            entries.append(entry)
    _BULK_CACHE = entries
    return entries


def find_priors(make: str | None, model: str | None) -> dict[str, Any] | None:
    if not make or not model:
        return None
    make_u = make.upper().strip()
    model_u = model.upper().strip()
    # Tier 1: curated entries (hand-checked series, DCG annotations).
    # Tier 2: first-party JPTC measurements. Tier 3: P2P bulk table.
    for entry in PRIOR_TABLE + _jptc_entries():
        if str(entry["make_contains"]).upper() in make_u and model_u in entry["model_equals"]:
            return entry
    make_token = make_u.split()[0] if make_u.split() else make_u
    for entry in _bulk_entries():
        name_u = str(entry["make_model"]).upper()
        if make_token in name_u and model_u in name_u:
            return entry
    return None


def _interp(curve: list[tuple[float, float]], x: float) -> float:
    if x <= curve[0][0]:
        return curve[0][1]
    if x >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    return float("nan")


def gain_e_per_dn(priors: dict[str, Any], iso: int) -> float | None:
    if not iso or iso <= 0:
        return None
    return float(2.0 ** priors["unity_gain_ev"] / iso)


def read_noise_e(priors: dict[str, Any], iso: int) -> float | None:
    if not iso or iso <= 0:
        return None
    curve = priors.get("read_noise_log2iso_log2e")
    if not curve:
        return None
    return float(2.0 ** _interp(curve, math.log2(iso)))


def pdr_ev(priors: dict[str, Any], iso: int) -> float | None:
    if not iso or iso <= 0:
        return None
    curve = priors.get("pdr_log2iso_ev")
    if not curve:
        return None
    return float(_interp(curve, math.log2(iso)))
