# SPDX-License-Identifier: GPL-3.0-or-later
"""Static sensor priors from public measurements (PhotonsToPhotos, Bill Claff).

These are *published chart data*, not our own bench measurements: they calibrate the
absolute scale (electrons, PDR) that a single frame cannot provide, while the empirical
per-frame analysis remains the primary signal. Everything degrades gracefully to None
when the camera or ISO is unknown.

Data source: https://www.photonstophotos.net/Charts/PDR.htm and
https://www.photonstophotos.net/Charts/RN_e.htm (series extracted 2026-07-06).
x is log2(EXIF ISO setting); PDR y is EV; read-noise y is log2(input-referred
electrons). NOTE (2026-08-24 axis audit): the P2P chart's own x-axis is
ISO = 3.125 * 2^x — the original extraction read it as ISO = 2^x, which put
every curve (and every chart-anchored unity_gain_ev) log2(3.125) = 1.6439 EV
low. All curated curves have been re-referenced to EXIF ISO; they now start
at each camera's native base ISO. Points P2P plots with hollow markers
(suspect/NR-affected) are kept but the threshold is recorded in
`suspect_iso_min`.
"""

from __future__ import annotations

import math
from typing import Any

# Sigma fp (full-frame 24MP BSI, 14-bit). unity_gain_ev: ISO at which 1 DN = 1 e-
# is 2**8.93 ~= 488. fwc_e is P2P's saturation at the lowest-gain point.
# unity_gain_ev corrected 7.29 -> 8.93 in the 2026-08-24 axis audit (see module
# docstring): 8.93 = log2(100 * 74884/15359) — fwc_e over the 14-bit range at
# native base ISO 100 (DNG black 1024, measured on owner files) — identical to
# the axis fix (7.29 + 1.6439). First-party confirmation: pair-difference PTC
# envelopes on three owner ISO-100 DNG frames measure 4.70-4.77 e-/DN vs the
# 4.88 implied (-3%; the estimator biases low on natural scenes), refuting the
# old 1.56 e-/DN outright. Pixel density 74884/(5.98um)^2 = 2.1 ke-/um^2 is
# physical; the old value implied 0.7 ke-/um^2.
SIGMA_FP = {
    "id": "Sigma fp",
    "make_contains": "SIGMA",
    "model_equals": {"SIGMA FP", "FP"},
    "unity_gain_ev": 8.93,
    "fwc_e": 74884,
    "pdr_log2iso_ev": [
        (6.6439, 11.02), (6.9739, 10.98), (7.3139, 11.0), (7.6439, 11.0),
        (7.9739, 10.7), (8.3139, 10.41), (8.6439, 9.85), (8.9739, 9.22),
        (9.3139, 9.38), (9.6439, 9.38), (9.9739, 9.41), (10.3139, 9.38),
        (10.6439, 9.07), (10.9739, 8.73), (11.3139, 8.4), (11.6439, 8.07),
        (11.9739, 7.75), (12.3139, 7.42), (12.6439, 7.1), (12.9739, 6.78),
        (13.3139, 6.46), (13.6439, 6.09), (13.9739, 5.79), (14.3139, 5.46),
        (14.6439, 5.1), (14.9739, 4.8), (15.3139, 4.46), (15.6439, 4.11),
        (15.9739, 3.82), (16.3139, 3.47), (16.6439, 3.13),
    ],
    "read_noise_log2iso_log2e": [
        (6.6439, 2.76), (6.9739, 2.41), (7.3139, 2.09), (7.6439, 1.78), (7.9739, 1.68),
        (8.3139, 1.63), (8.6439, 1.83), (8.9739, 2.04), (9.3139, 0.7), (9.6439, 0.34),
        (9.9739, 0.01), (10.3139, -0.35), (10.6439, -0.4), (10.9739, -0.43),
        (11.3139, -0.46), (11.6439, -0.49), (11.9739, -0.54), (12.3139, -0.56),
        (12.6439, -0.58), (12.9739, -0.61), (13.3139, -0.63), (13.6439, -0.65),
        (13.9739, -0.68), (14.3139, -0.68), (14.6439, -0.71), (14.9739, -0.77),
        (15.3139, -0.72), (15.6439, -0.73), (15.9739, -0.73), (16.3139, -0.77),
        (16.6439, -0.72),
    ],
    # P2P marks values from here up with hollow markers (decoded: the ISO 32000
    # setting). Note: fp's read noise below ~1 e- from ISO ~1000 is widely
    # attributed to spatial filtering baked into the DNG; the empirical
    # RAW-health autocorrelation check is the per-frame verdict on that.
    "suspect_iso_min": 32256,
    # Read-noise curve drops sharply at the ISO 640 setting (stored x 9.3139) —
    # the known IMX410 (A7 III / Z 6) dual-conversion-gain point. The audit's
    # axis fix moved this from the previously recorded "ISO 200".
    "dcg_switch_iso": 640,
    "source": "PhotonsToPhotos PDR.htm / RN_e.htm, retrieved 2026-07-06; "
              "x axis re-referenced to EXIF ISO and unity_gain_ev re-anchored "
              "2026-08-24 (see module docstring and the comments above)",
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
    # Collect sets (data/priors/jptc_collect/): multi-instrument entries with
    # gain and read-noise CURVES (see tools/import_jptc_collect.py).
    shutter_map = {"机械快门": "mechanical", "电子快门": "electronic"}
    for path in sorted((Path(__file__).parent / "data" / "priors" / "jptc_collect").glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if item.get("format") != "dngscan-jptc-collect-1":
            continue
        entry: dict[str, Any] = {
            "id": item["id"],
            "make_contains": item["make"],
            "model_equals": {str(m).upper() for m in item.get("model_candidates", [])},
            "gain_log2iso_log2epd": [(float(x), float(y)) for x, y in
                                     item.get("gain_log2iso_log2epd", [])],
            "read_noise_log2iso_log2e": [(float(x), float(y)) for x, y in
                                         item.get("read_noise_log2iso_log2e", [])],
            "pdr_log2iso_ev": [],
            "gain_jump_isos": item.get("gain_jump_isos", []),
            "shutter": shutter_map.get(str(item.get("shutter")), item.get("shutter")),
            "measured_iso": int(item["ptc_anchor"]["iso"]) if item.get("ptc_anchor") else 10 ** 9,
            "noise_whiteness_h_log2iso": item.get("noise_whiteness_h_log2iso"),
            "banding_log2iso": item.get("banding_log2iso"),
            "source": f"JPTC collect set ({path.name})",
        }
        for k in ("unity_gain_ev", "fwc_e", "fwc_e_uncertainty"):
            if k in item:
                entry[k] = item[k]
        if item.get("ptc_anchor"):
            entry["prnu"] = item["ptc_anchor"].get("prnu")
        entries.append(entry)
    # One model can have several measurements (shutter modes, ISOs). Order
    # decides which one find_priors returns: entries with a full gain curve
    # first (they encode extended-segment and conversion-gain structure the
    # single-ISO reciprocal law cannot), then lowest anchor ISO (fwc_e is at
    # the measured ISO), then resolved read noise, then mechanical shutter.
    entries.sort(key=lambda e: (
        0 if e.get("gain_log2iso_log2epd") else 1,
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
    # A measured gain curve (JPTC collect tier) wins over the reciprocal
    # law: extended-ISO segments and conversion-gain switches break the
    # unity-gain extrapolation, and the curve encodes both.
    curve = priors.get("gain_log2iso_log2epd")
    if curve:
        return float(2.0 ** _interp(curve, math.log2(iso)))
    if "unity_gain_ev" not in priors:
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
