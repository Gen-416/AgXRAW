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


def find_priors(make: str | None, model: str | None) -> dict[str, Any] | None:
    if not make or not model:
        return None
    make_u = make.upper().strip()
    model_u = model.upper().strip()
    for entry in PRIOR_TABLE:
        if str(entry["make_contains"]).upper() in make_u and model_u in entry["model_equals"]:
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
    return float(2.0 ** _interp(priors["read_noise_log2iso_log2e"], math.log2(iso)))


def pdr_ev(priors: dict[str, Any], iso: int) -> float | None:
    if not iso or iso <= 0:
        return None
    return float(_interp(priors["pdr_log2iso_ev"], math.log2(iso)))
