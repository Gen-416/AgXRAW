# SPDX-License-Identifier: GPL-3.0-or-later
"""Measured CFA spectral evidence, with positions kept distinct from colours."""
from __future__ import annotations

import math
import re

SCHEMA = "dngscan-noise-spectrum-1"
PHASES = ("C00", "C01", "C10", "C11")


def dark_phase_mapping(header, rows):
    """Resolve collector Cxx positions from measured Channel + ColorIndex.

    Upstream's CfaPattern header is LibRaw's colour *description*, not an
    RGGB layout. Missing metadata cannot establish a Bayer phase mapping.
    """
    description = str(header.get("CfaPattern", "")).strip().upper()
    mapping = {}
    if not description or len(description) > 4 or not description.isalpha():
        return {"status": "unavailable", "reason": "colour-description-unavailable", "phases": {}}
    for row in rows:
        phase = str(row.get("Channel", "")).strip()
        if not phase:
            continue
        if phase not in PHASES:
            raise ValueError("dark Channel must name a 2x2 CFA position C00/C01/C10/C11")
        try:
            index = int(row["ColorIndex"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("dark CFA mapping requires a valid ColorIndex") from exc
        if not 0 <= index < len(description):
            raise ValueError("dark ColorIndex is outside the declared colour description")
        current = {"color_index": index, "color": description[index]}
        if phase in mapping and mapping[phase] != current:
            raise ValueError("dark CFA position-to-colour mapping changes across ISO or frames")
        mapping[phase] = current
    if not mapping:
        return {"status": "unavailable", "reason": "channel-position-mapping-unavailable",
                "color_description": description, "phases": {}}
    colours = [mapping[p]["color"] for p in PHASES if p in mapping]
    if len(mapping) != 4:
        status, reason = "partial", "incomplete-cfa-position-mapping"
    elif sorted(colours) == ["B", "G", "G", "R"]:
        status, reason = "bayer", "measured-position-to-colour-mapping"
    elif len(set(colours)) == 1:
        status, reason = "single-colour", "measured-single-colour-position-mapping"
    else:
        status, reason = "unsupported", "not-a-supported-2x2-colour-arrangement"
    return {"status": status, "reason": reason, "color_description": description,
            "phases": mapping}


def validate_spectrum(value):
    """Validate retained per-phase summaries before granting model authority."""
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported noise_spectrum schema")
    mapping = value.get("mapping") or {}
    if not isinstance(mapping, dict) or not isinstance(mapping.get("phases", {}), dict):
        raise ValueError("noise_spectrum mapping must be an object")
    phases = mapping.get("phases", {})
    description = mapping.get("color_description", "")
    for phase, record in phases.items():
        if (phase not in PHASES or not isinstance(record, dict) or
                type(record.get("color_index")) is not int or
                not 0 <= record["color_index"] < len(description) or
                record.get("color") != description[record["color_index"]]):
            raise ValueError("invalid noise_spectrum CFA mapping")
    if mapping.get("status") == "bayer":
        if set(phases) != set(PHASES) or sorted(r["color"] for r in phases.values()) != ["B", "G", "G", "R"]:
            raise ValueError("noise_spectrum Bayer mapping is incomplete")
    axes = value.get("axes", {})
    if not isinstance(axes, dict) or any(axis not in ("h", "v") for axis in axes):
        raise ValueError("noise_spectrum axes must be h/v")
    for axis in axes.values():
        if not isinstance(axis, dict) or not isinstance(axis.get("ratios_log2iso", {}), dict):
            raise ValueError("noise_spectrum phase curves must be objects")
        for phase, curve in axis.get("ratios_log2iso", {}).items():
            if phase not in PHASES or not isinstance(curve, list):
                raise ValueError("invalid noise_spectrum phase curve")
            previous = -math.inf
            for point in curve:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError("invalid noise_spectrum ISO ratio point")
                x, ratio = map(float, point)
                if not math.isfinite(x) or not math.isfinite(ratio) or x <= previous or ratio < 0:
                    raise ValueError("noise_spectrum ISO ratios must be finite and ordered")
                previous = x
    return value


def spectrum_curves(rows, freqs, lo=(.05, .20), hi=(.35, .499)):
    """Keep every measured position; a missing band has no ratio evidence."""
    import numpy as np

    curves = {}
    if not rows:
        return curves
    low = (freqs >= lo[0]) & (freqs <= lo[1])
    high = (freqs >= hi[0]) & (freqs <= hi[1])
    for column in rows[0]:
        match = re.fullmatch(r"iso([1-9][0-9]*)_(C[01][01])_diff", column)
        if not match:
            continue
        iso, phase = int(match[1]), match[2]
        power = np.asarray([float(row[column]) for row in rows])
        if not np.all(np.isfinite(power)) or np.any(power < 0):
            raise ValueError("spectrum power must be finite and nonnegative")
        if not low.any() or not high.any() or float(power[low].mean()) <= 0:
            continue
        ratio = float(power[high].mean() / power[low].mean())
        curves.setdefault(phase, []).append([math.log2(iso), ratio])
    return {phase: sorted(points) for phase, points in curves.items()}


def conservative_ratio(values):
    """A compatibility scalar which cannot average opposing anomalies away."""
    return max(values, key=lambda r: abs(math.log(max(float(r), 1e-300))))
