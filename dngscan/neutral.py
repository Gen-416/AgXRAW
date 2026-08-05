# SPDX-License-Identifier: GPL-3.0-or-later
"""Generic fixed-curve tone compression (Lightroom-style export baseline).

Unlike `lum`, endpoints are not compiled from scene statistics — the same canned
shoulder/toe applies to every frame so A/B against AgX measures creative geometry,
not a different exposure analysis path.
"""
from __future__ import annotations

from typing import Any

from . import lum as lum_engine

# Fixed sigmoid window: generic display-referred compression, not scene-derived C1.
NEUTRAL_BLACK_EV = -6.5
NEUTRAL_WHITE_EV = 3.0
NEUTRAL_CONTRAST = 2.35
NEUTRAL_TOE_POWER = 1.35
NEUTRAL_SHOULDER_POWER = 2.65


def apply_neutral_core(rgb_rec2020: Any, plan: Any) -> Any:
    """Luminance-ratio mapping through the fixed neutral curve (no AgX colour path)."""
    return lum_engine.apply_lum_core(rgb_rec2020, plan)
