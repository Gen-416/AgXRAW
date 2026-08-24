# SPDX-License-Identifier: GPL-3.0-or-later
"""Access layer for the measured lens/filter transmittance library.

Data: dngscan/data/lens_transmittance.json — first-party spectral
transmittance measurements (see the file's provenance block and NOTICE.md),
uniform 1 nm grid. Everything degrades to an empty library when the data
file is absent (it is a single excisable file).

The curves are the physical input for lens-aware spectral work: multiplying
T(lambda) into an SSF exposure model predicts a lens's color cast (classic
single-coated or thoriated glass pulls hundreds of kelvin of warm shift),
which the color pipeline can then declare instead of guessing.
"""

from __future__ import annotations

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
        _CACHE = {e["id"]: e for e in raw.get("entries", [])}
    return _CACHE


def list_entries(category: str | None = None) -> list[dict[str, Any]]:
    """[{id, name, category}] for the whole library or one category."""
    out = []
    for e in _load().values():
        if category is None or e["category"] == category:
            out.append({"id": e["id"], "name": e["name"], "category": e["category"]})
    return out


def get_curve(entry_id: str) -> tuple[list[float], list[float]] | None:
    """(wavelengths_nm, transmittance as 0..1 fraction) or None."""
    e = _load().get(entry_id)
    if e is None:
        return None
    start, step = float(e["wavelength_start_nm"]), float(e["step_nm"])
    t = [v / 100.0 for v in e["transmittance_percent"]]
    wl = [start + step * i for i in range(len(t))]
    return wl, t


def mean_transmittance(entry_id: str, lo_nm: float = 420.0, hi_nm: float = 680.0) -> float | None:
    """Average T over a band (default: photopically dominant band)."""
    curve = get_curve(entry_id)
    if curve is None:
        return None
    wl, t = curve
    vals = [v for w, v in zip(wl, t) if lo_nm <= w <= hi_nm]
    return sum(vals) / len(vals) if vals else float("nan")


def warmth_ratio(entry_id: str) -> float | None:
    """log2 of red-band (600-680nm) over blue-band (420-500nm) mean T.

    A coarse, model-free cast indicator: 0 = neutral glass; positive = warm
    (yellowed/single-coated), negative = cool. Full cast prediction belongs
    to the spectral pipeline; this is the honest scalar for tables.
    """
    r = mean_transmittance(entry_id, 600.0, 680.0)
    b = mean_transmittance(entry_id, 420.0, 500.0)
    if r is None or b is None or r <= 0 or b <= 0:
        return None
    return math.log2(r / b)
