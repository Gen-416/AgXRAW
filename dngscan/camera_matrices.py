# SPDX-License-Identifier: GPL-3.0-or-later
"""Fallback camera colour matrices for bodies the installed LibRaw predates.

New camera models decode fine long before the installed LibRaw learns their colour
tables (the mosaics are format-stable; the per-model Adobe coefficients are not).
When ``rgb_xyz_matrix`` comes back empty, this table supplies the same kind of
matrix from LibRaw *master*'s colordata (GPL, same licence family as this project):
Adobe-derived D65 XYZ->camera rows, published as integers scaled by 10000.

Scope and honesty:
- These matrices feed the fixed-Kelvin WB solve (wb.kelvin_camera_multipliers) and
  reporting. They cannot be injected into LibRaw's own postprocess colour
  conversion — an unknown-to-LibRaw body still renders through whatever LibRaw
  falls back to internally, and the export report says so. The real fix for that
  half is upgrading LibRaw itself (tools/build_libraw_master.sh); this table
  covers the newer-than-master window and defends older build environments.
  Entries for bodies the release LibRaw already knows (A7S III, X100VI, Z f)
  never trigger there — deliberate redundancy, not dead code.
- Entries marked ``borrowed_from`` reuse a same-sensor sibling's matrix (declared
  approximation, e.g. X-E5 <- X100VI: both 40MP X-Trans CMOS 5 HR). No entry is
  eyeballed.
- DNG-writing cameras (Ricoh GR IV, iPhone) never need this table: their
  ColorMatrix tags travel in-file and take priority in solve_wb_for_mode.
"""
from __future__ import annotations

from typing import Any

from ._deps import np

# make_contains / model_equals matching mirrors priors.find_priors.
# Matrix rows: XYZ->cam for R, G, B, Adobe coefficients / 10000 (D65 table).
_FALLBACK_MATRICES: list[dict[str, Any]] = [
    {
        "make_contains": "SONY",
        "model_equals": {"ILCE-7M5"},
        "label": "Sony A7 V (ILCE-7M5)",
        "matrix": [[9089, -3577, -787], [-3563, 11326, 2557], [-114, 928, 5904]],
        "source": "LibRaw master colordata.cpp, retrieved 2026-07-30",
    },
    {
        "make_contains": "SONY",
        "model_equals": {"ILCE-7SM3"},
        "label": "Sony A7S III (ILCE-7SM3)",
        "matrix": [[6912, -2127, -469], [-4470, 12175, 2587], [-398, 1478, 6492]],
        "source": "LibRaw master colordata.cpp, retrieved 2026-07-30",
    },
    {
        "make_contains": "NIKON",
        "model_equals": {"Z F"},
        "label": "Nikon Z f",
        "matrix": [[11607, -4491, -977], [-4522, 12460, 2304], [-458, 1519, 7616]],
        "source": "LibRaw master colordata.cpp, retrieved 2026-07-30",
    },
    {
        "make_contains": "FUJIFILM",
        "model_equals": {"X100VI"},
        "label": "Fujifilm X100VI",
        "matrix": [[11809, -5358, -1141], [-4248, 12164, 2343], [-514, 1097, 5848]],
        "source": "LibRaw master colordata.cpp, retrieved 2026-07-30",
    },
    {
        "make_contains": "FUJIFILM",
        "model_equals": {"X-E5"},
        "label": "Fujifilm X-E5",
        "matrix": [[11809, -5358, -1141], [-4248, 12164, 2343], [-514, 1097, 5848]],
        "source": "LibRaw master colordata.cpp, retrieved 2026-07-30",
        "borrowed_from": "X100VI (same 40MP X-Trans CMOS 5 HR sensor)",
    },
    # Sony A7R VI (ILCE-7RM6): no published Adobe/LibRaw coefficients located yet
    # (2026-07-30); the body is real (PhotonsToPhotos measures it) but the colour
    # table has not landed upstream. Deliberately absent rather than guessed —
    # the degradation path covers it with an explicit warning.
]


def fallback_xyz_to_cam(
    make: str | None, model: str | None
) -> tuple[Any, str] | None:
    """(3x3 float matrix, provenance note) for a known new body, else None."""
    if not make or not model:
        return None
    make_u, model_u = make.upper().strip(), model.upper().strip()
    for entry in _FALLBACK_MATRICES:
        if entry["make_contains"] in make_u and model_u in entry["model_equals"]:
            matrix = np.asarray(entry["matrix"], dtype=np.float64) / 10000.0
            note = f"{entry['label']}（{entry['source']}）"
            borrowed = entry.get("borrowed_from")
            if borrowed:
                note += f"，矩阵借自 {borrowed}"
            return matrix, note
    return None
