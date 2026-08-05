# SPDX-License-Identifier: GPL-3.0-or-later
"""Declared lens conversion filters: glass in front of the lens, as published constants.

A Wratten conversion filter is specified by Kodak in exactly two published numbers: its
mired shift and its filter factor. The mired shift is the definition of the filter's
colour action (85B: +131 mired warms 5500K light to tungsten's 3200K; 80A: -131 does the
reverse), so deriving the scene-linear matrix from it is working from the specification,
not from taste. The matrix is a Bradford-adapted von Kries transport between the two
white points, expressed in linear Rec.2020.

The filter's purpose — the first semantics — is to move the *illuminant* onto the
film's calibration point: daylight stock under tungsten light plus an 80A sees light ×
filter ~= its design illuminant, and in the film's own frame the neutral axis does not
move at all. Used as intended, the filter pairs with a mismatched declared WB (WB
3200K + 85B in daylight; WB 5500K + 80A under tungsten), the two cancel to near
neutral, and every downstream calibration window remains valid. Applying the filter
alone on a matched balance — the deliberate colour cast — is the *declared degenerate
case*: real film driven that way runs deep into excitation regions our separation fits
never covered, so the prefeed's chromaticity windows fading toward identity there is a
refusal to extrapolate, not a malfunction.

Deliberate simplifications, stated rather than hidden:
- Spectral transmission refinement (integrating a digitized T(lambda) against camera
  SSFs) would capture the small material-dependent residual beyond the mired action;
  pending machine-readable curves, the published-constant derivation is the honest core.
- The published filter factor (light absorbed by the glass) is recorded as metadata but
  normalized out of the matrix: digital application has no reason to discard photons,
  and Kodak's exposure guidance already assumes the photographer compensates.

There is no strength control. A filter has no half-installed state.

Position in the pipeline: scene-linear Rec.2020, after decode/WB, before the prefeed —
the same order as the physical light path (filter before the film's spectral response).
This module stays separate from scene_transform because the two hold different
invariants: the prefeed keeps the working frame's neutral axis fixed, while the filter
re-expresses the light itself — moving the rendered neutral axis whenever its partner
illuminant mismatch is absent.
"""
from __future__ import annotations

from typing import Any

from ._deps import np
from .constants import RGB_TO_XYZ
from .wb import cct_to_xy

# name -> (mired shift, published filter factor in stops, label, source note)
# Mired shifts and factors from Kodak's published Wratten filter data.
LENS_FILTERS: dict[str, tuple[float, float, str]] = {
    "85b": (+131.0, 2.0 / 3.0, "85B · 日光转钨丝（5500K->3200K）"),
    "85": (+112.0, 2.0 / 3.0, "85 · 日光转 Type A（5500K->3400K）"),
    "80a": (-131.0, 2.0, "80A · 钨丝转日光（3200K->5500K）"),
    "81a": (+18.0, 1.0 / 3.0, "81A · 轻度暖化"),
    "82a": (-21.0, 1.0 / 3.0, "82A · 轻度冷化"),
}
LENS_FILTER_CHOICES = ("none",) + tuple(LENS_FILTERS)

# Mired shifts are (approximately) position-invariant — that is why Kodak catalogues
# filters in mireds — so a von Kries diagonal computed from any in-range CCT pair with
# the right delta represents the filter. The pair is taken symmetrically about the
# working white's mired value (m0 +- delta/2): the diagonal is then measured where the
# rendered cast actually lands, and equal-and-opposite filters (85B/80A) invert each
# other exactly in chromaticity by construction, with no locus-curvature residue. One
# global exposure scalar remains — each matrix Y-normalizes its own neutral, not the
# other's cast — measured M80A . M85B = 0.95649 I (-0.064 EV); the round-trip is
# chromatically exact, not radiometrically free.
_WORKING_WHITE_CCT = 6504.0

_BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float64,
)


def lens_filter_label(name: str) -> str:
    if name == "none":
        return "无"
    entry = LENS_FILTERS.get(name)
    return entry[2] if entry is not None else name


def validate_lens_filter(name: str) -> str:
    if name == "none" or name in LENS_FILTERS:
        return name
    raise ValueError(f"未知镜前滤镜：{name}（可选：{'/'.join(LENS_FILTER_CHOICES)}）")


def shifted_cct(base_cct: float, mired_shift: float) -> float:
    """Apply a mired shift: filters are linear in reciprocal megakelvin, not in Kelvin."""
    mired = 1e6 / float(base_cct) + float(mired_shift)
    if mired <= 0.0:
        raise ValueError(f"mired shift {mired_shift} from {base_cct}K leaves no light")
    return 1e6 / mired


def _xyz_white(cct: float) -> np.ndarray:
    x, y = cct_to_xy(cct)
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


_MATRIX_CACHE: dict[tuple[str, float | None], Any] = {}


def lens_filter_matrix(name: str) -> Any | None:
    """Linear-Rec.2020 matrix for one declared filter; None for 'none'.

    A Bradford von Kries diagonal realising the published mired shift, anchored on a
    fixed in-range CCT pair (see _ANCHOR_CCT) and luminance-normalized on the neutral
    axis. Note the semantics: the shift acts on the *rendered* balance — a neutral in
    the working space picks up the cast of "working white shifted by delta mired" —
    which is exactly how a filter behaves on already-balanced film. Numbers like
    "85B turns 5500K light into 3200K" describe light, not the rendered cast; the
    rendered neutral lands at the working white's CCT shifted by the same mireds.
    """
    if name == "none":
        return None
    entry = LENS_FILTERS.get(name)
    if entry is None:
        raise ValueError(f"未知镜前滤镜：{name}")
    cache_key = (name, None)
    cached = _MATRIX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    m0 = 1e6 / _WORKING_WHITE_CCT
    base = 1e6 / (m0 - entry[0] / 2.0)
    target = shifted_cct(base, entry[0])

    src = _BRADFORD @ _xyz_white(base)
    dst = _BRADFORD @ _xyz_white(target)
    adapt = np.linalg.inv(_BRADFORD) @ np.diag(dst / src) @ _BRADFORD

    xyz_from_rec = np.array(RGB_TO_XYZ["Rec2020"], dtype=np.float64)
    rec_from_xyz = np.linalg.inv(xyz_from_rec)
    matrix = rec_from_xyz @ adapt @ xyz_from_rec

    # Y-normalize on the neutral axis: the glass absorbs light uniformly enough that
    # Kodak specifies it as a single exposure factor; digitally we keep the photons.
    white = matrix @ np.ones(3)
    y = float(np.array(RGB_TO_XYZ["Rec2020"], dtype=np.float64)[1] @ white)
    if y <= 1e-9:
        raise ValueError(f"degenerate filter matrix for {name}")
    result = (matrix / y).astype(np.float32)
    result.setflags(write=False)
    _MATRIX_CACHE[cache_key] = result
    return result


def apply_lens_filter_rec2020(rgb: Any, name: str = "none") -> Any:
    """Apply a declared filter to flat scene-linear Rec.2020 samples ([N,3])."""
    matrix = lens_filter_matrix(name)
    if matrix is None:
        return rgb
    arr = np.asarray(rgb, dtype=np.float32)
    return arr @ matrix.T.astype(np.float32)
