# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene-driven purity compensation ("punch") for the AgX render.

AgX Base is deliberately flat: the inset attenuates purity up front, and only deep-toe
content gets it back through per-channel expansion — which is why high-ISO night shots
come out punchy while daylight wide-DR scenes look washed (measured on _SDI0238; the
shoulder was ruled out empirically). Blender ships the same base behind a "Punchy" look
for exactly this reason.

This operator lifts Oklab chroma on the Rec.2020 display-linear AgX output. Strength is
computed per-scene by the tone plan (bright x quality x DR gating; zero for night/high-ISO
scenes, which short-circuits to identity — byte-identical renders). All weights multiply
into the gain INCREMENT, so the gain is >= 1 everywhere (never desaturates), fades to zero
on the neutral axis and in deep shadows (no noise amplification), fades out in highlights
(preserving AgX's desaturate-to-white path), soft-knees on already-vivid chroma, and is
damped inside the skin hue band.
"""

from __future__ import annotations

from typing import Any

from ._deps import np
from .constants import OKLAB_M1, OKLAB_M2, OKLAB_M1_INV, OKLAB_M2_INV, RGB_TO_XYZ, XYZ_TO_RGB
from .color import apply_rgb_matrix3
from .look import _hue_in_arc, _smoothstep

PUNCH_CHROMA_MAX = 1.5  # gain ceiling at strength/weight 1.0; Blender Punchy sat~1.4 territory
PUNCH_SKIN_DAMP = 0.55  # skin hue arc keeps 55% of the lift
SKIN_HUE_LO = 20.0
SKIN_HUE_HI = 60.0


def apply_punch_rec2020(rgb_rec2020: Any, strength: float) -> Any:
    """Chroma-lift the Rec.2020 display-linear buffer; exact identity at strength <= 0."""
    if strength <= 1e-3 or np is None:
        return rgb_rec2020
    s = np.float32(min(1.0, float(strength)))
    rgb = np.nan_to_num(rgb_rec2020.astype(np.float32, copy=False), nan=0.0, posinf=1e6, neginf=0.0)
    xyz = apply_rgb_matrix3(rgb, RGB_TO_XYZ["Rec2020"])
    lab = apply_rgb_matrix3(np.cbrt(np.maximum(apply_rgb_matrix3(xyz, OKLAB_M1), 0.0)), OKLAB_M2)
    lab_l, lab_a, lab_b = lab[:, 0], lab[:, 1], lab[:, 2]

    chroma = np.hypot(lab_a, lab_b)
    hue = np.degrees(np.arctan2(lab_b, lab_a)) % 360.0
    weight = _smoothstep(0.005, 0.03, chroma)  # zero on the neutral axis
    weight = weight * _smoothstep(0.08, 0.22, lab_l)  # deep-shadow fade: don't amplify toe noise
    weight = weight * (1.0 - _smoothstep(0.72, 0.92, lab_l))  # keep AgX highlight fade-to-white
    weight = weight * (1.0 - 0.35 * _smoothstep(0.20, 0.42, chroma))  # gentle knee only on extreme chroma; gamut fit catches overflow
    weight = weight * (1.0 - (1.0 - PUNCH_SKIN_DAMP) * _hue_in_arc(hue, SKIN_HUE_LO, SKIN_HUE_HI))
    gain = 1.0 + np.float32(PUNCH_CHROMA_MAX - 1.0) * s * weight  # increment-weighted: gain >= 1 everywhere

    lab_out = np.stack([lab_l, lab_a * gain, lab_b * gain], axis=1)
    lms_ = apply_rgb_matrix3(lab_out, OKLAB_M2_INV)
    out = apply_rgb_matrix3(apply_rgb_matrix3(lms_ * lms_ * lms_, OKLAB_M1_INV), XYZ_TO_RGB["Rec2020"])
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32, copy=False)
