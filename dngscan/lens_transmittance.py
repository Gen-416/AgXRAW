# SPDX-License-Identifier: GPL-3.0-or-later
"""Access layer for the measured lens/filter transmittance library.

Data: dngscan/data/lens_transmittance.json — first-party spectral
transmittance measurements (see the file's provenance block and NOTICE.md),
uniform 1 nm grid. Everything degrades to an empty library when the data
file is absent (it is a single excisable file).

These are peak-normalized *relative* spectral shapes, not absolute throughput.
They do not determine T-stop, electrons, SNR or a universal camera RGB cast.
Color prediction additionally requires a qualified camera response, illuminant
and reference optical path. This library does not apply pixel corrections.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

_CACHE: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        path = Path(__file__).parent / "data" / "lens_transmittance.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        if raw.get("format") != "dngscan-lens-transmittance-1":
            raw = {"entries": []}
        entries = raw.get("entries", [])
        ids = [e["id"] for e in entries]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate lens measurement id")
        _CACHE = {e["id"]: e for e in entries}
    return _CACHE


def list_entries(category: str | None = None) -> list[dict[str, Any]]:
    """[{id, name, category}] for the whole library or one category."""
    out = []
    for e in _load().values():
        if category is None or e["category"] == category:
            out.append({"id": e["id"], "name": e["name"], "category": e["category"]})
    return out


def get_curve(entry_id: str) -> tuple[list[float], list[float]] | None:
    """(wavelengths_nm, relative shape / 100) or None.

    Compatibility API: the returned numbers have no absolute transmission unit.
    Inspect get_metadata() before optical-stack matching or spectral modeling.
    """
    e = _load().get(entry_id)
    if e is None:
        return None
    start, step = float(e["wavelength_start_nm"]), float(e["step_nm"])
    t = [v / 100.0 for v in e["transmittance_percent"]]
    wl = [start + step * i for i in range(len(t))]
    return wl, t


def mean_transmittance(entry_id: str, lo_nm: float = 420.0, hi_nm: float = 680.0) -> float | None:
    """Unweighted sample mean of relative shape over the selected band.

    This is neither photopic throughput nor absolute lens transmission. It must
    not be used to alter exposure, estimated electrons or a sensor noise model.
    """
    curve = get_curve(entry_id)
    if curve is None:
        return None
    wl, t = curve
    vals = [v for w, v in zip(wl, t) if lo_nm <= w <= hi_nm]
    return sum(vals) / len(vals) if vals else float("nan")


def warmth_ratio(entry_id: str) -> float | None:
    """log2 of red-band (600-680nm) over blue-band (420-500nm) mean T.

    A coarse spectral-shape ratio. It is not a camera white-balance temperature,
    a colorimetric measurement or an RGB correction. Such predictions require
    camera SSF, illuminant and a known reference optical path.
    """
    r = mean_transmittance(entry_id, 600.0, 680.0)
    b = mean_transmittance(entry_id, 420.0, 500.0)
    if r is None or b is None or r <= 0 or b <= 0:
        return None
    return math.log2(r / b)


def get_metadata(entry_id: str) -> dict[str, Any] | None:
    """Independent metadata copy, with conservative semantics for legacy data."""
    entry = _load().get(entry_id)
    if entry is None:
        return None
    out = {k: copy.deepcopy(v) for k, v in entry.items() if k != "transmittance_percent"}
    out.setdefault("measurement_id", entry_id)
    out.setdefault("quantity", "relative_spectral_transmittance")
    out.setdefault("normalization", {"method": "legacy-unspecified", "absolute_scale_known": False})
    out.setdefault("is_bare_lens", None)
    out.setdefault("measurement_kind", "unknown")
    out.setdefault("optical_stack", {"lens": entry.get("name"), "filters": None})
    out["pixel_correction_applied"] = False
    out["absolute_throughput_available"] = False
    return out


def match_exact(name: str, *, category: str = "Lens", is_bare_lens: bool | None = True,
                optical_stack: dict[str, Any] | None = None,
                measurement_id: str | None = None) -> dict[str, Any]:
    """Resolve an exact measurement identity without brand/focal-length guessing.

    A name with repeated measurements remains ambiguous. An explicit ID can
    select a spectrum for research/diagnostics, but never qualifies unknown
    filters as a bare lens or grants a pixel correction capability.
    """
    candidates = [get_metadata(key) for key, entry in _load().items()
                  if entry.get("name") == name and entry.get("category") == category
                  and (measurement_id is None or key == measurement_id)]
    candidates = [entry for entry in candidates if entry is not None]
    base = {"candidates": candidates, "pixel_correction_applied": False,
            "color_correction_available": False}
    if not candidates:
        return {**base, "status": "unmeasured", "reason": "no-exact-measurement"}
    eligible = []
    for entry in candidates:
        if entry.get("measurement_kind") == "synthetic-reference":
            continue
        if category == "Lens" and is_bare_lens is not None and entry.get("is_bare_lens") is not is_bare_lens:
            continue
        if optical_stack is not None and entry.get("optical_stack") != optical_stack:
            continue
        eligible.append(entry)
    if not eligible:
        return {**base, "status": "incompatible", "reason": "optical-stack-unknown-or-mismatched"}
    if len(eligible) > 1:
        return {**base, "status": "ambiguous", "reason": "multiple-exact-measurements"}
    return {**base, "status": "matched", "reason": "exact-measurement-identity",
            "measurement": eligible[0]}
