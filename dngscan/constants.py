# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared numeric constants and color primary matrices."""
from __future__ import annotations

import math
from typing import Any

from ._deps import np
from . import agx as agx_engine


EPS = 1e-12


GAMUT_EPS = 1e-3


EV_REPORT_FLOOR = -14.0


GRAY_EV = math.log2(0.18)


MIDGRAY_HEADROOM_STOPS = 3.0


NOISE_DR_EPS = 1e-9


SNR_TILE = 16


SNR_LOW_PERCENTILE = 20.0


SNR_BRIGHT_UNRELIABLE_STOP = -2.5


CEILING_MIN_PILE_PIXELS = 256


CEILING_MIN_PILE_FRACTION = 2e-5


OUTPUT_GAMUT_SPACES = {"srgb": "sRGB", "p3": "P3"}


OUTPUT_GAMUT_LABELS = {"srgb": "sRGB", "p3": "Display P3"}


JPEG_OUTPUT_FORMATS = ("sdr", "ultrahdr", "ultrahdr-heic")
HDR_OUTPUT_FORMATS = ("ultrahdr", "ultrahdr-heic")


# HDR authoring policy. Apple defines headroom as a ratio and does not mandate an absolute
# reference-white luminance. dngscan uses 100 nit as its authoring convention because its
# SDR reference white is normalized to 1.0; 800/4000 nit are project defaults, not AgX or
# ISO 21496-1 constants and not format limits.
HDR_REFERENCE_WHITE_NITS = 100.0
DEFAULT_HDR_HEADROOM_EV = 3.0
DEFAULT_HDR_PEAK_NITS = HDR_REFERENCE_WHITE_NITS * (2.0 ** DEFAULT_HDR_HEADROOM_EV)
MAX_HDR_PEAK_NITS = 4000.0
MAX_HDR_HEADROOM_EV = math.log2(MAX_HDR_PEAK_NITS / HDR_REFERENCE_WHITE_NITS)
# Scene and output mid gray. One value, used as both the scene EV origin and the output
# stop origin; the two coordinates are distinct even though the anchor number is shared.
SCENE_MIDGRAY = 0.18
# Stops from output mid gray up to output reference white (T = 1.0). Named for what it is:
# a property of the output coordinate system. It is emphatically *not* a measurement of
# where diffuse white sits in any particular scene, and must never be read as a claim that
# some scene EV is a white object -- the old DIFFUSE_WHITE_EV name invited exactly that.
OUTPUT_REFERENCE_WHITE_STOPS = math.log2(1.0 / SCENE_MIDGRAY)
# darktable's internal AgX encoding exponent, applied as linear = encoded ** gamma. It is
# not a display transfer function. HDR v2 holds it fixed: the extended white endpoint is
# carried by the shoulder, so gamma no longer has to buy peak at the toe's expense.
DARKTABLE_BASE_GAMMA = 2.2
# AgX's historical -10..+6.5 EV normalization span. `contrast` is quoted against it, so a
# plan's encoded slope is contrast * (W-B) / 16.5 rather than contrast itself.
AGX_REFERENCE_RANGE_EV = 16.5

# ITU-R BT.2020/D65 linear-light Y coefficients. Keep these separate from the inverse
# of dngscan's legacy rounded XYZ matrix: that inverse is intentionally frozen for SDR
# pixel compatibility and its Y row does not sum to exactly one.
REC2020_LUMA = (0.2627, 0.6780, 0.0593)
# HDR display rendering transforms. "agx" is dngscan's native extended-white curve around
# darktable's AgX formation, described by docs/HDR_AGX_V2_IMPLEMENTATION_PLAN.zh-CN.md.
# darktable itself does not define this extended-P3 rendition or the gain-map contract.
HDR_DRT_CHOICES = ("agx",)
DEFAULT_HDR_DRT = "agx"


XYZ_TO_RGB = {
    "sRGB": np.array(  # type: ignore[union-attr]
        [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]],
        dtype=np.float64,
    )
    if np is not None
    else None,
    "P3": np.array(  # type: ignore[union-attr]
        [[2.4934, -0.9314, -0.4027], [-0.8295, 1.7627, 0.0236], [0.0358, -0.0762, 0.9569]],
        dtype=np.float64,
    )
    if np is not None
    else None,
    "Rec2020": np.array(  # type: ignore[union-attr]
        [[1.7167, -0.3557, -0.2534], [-0.6667, 1.6165, 0.0158], [0.0176, -0.0428, 0.9421]],
        dtype=np.float64,
    )
    if np is not None
    else None,
}


RGB_TO_XYZ = {
    name: np.linalg.inv(matrix).astype(np.float64) if np is not None and matrix is not None else None
    for name, matrix in XYZ_TO_RGB.items()
}


REC2020_TO_SRGB = (
    (XYZ_TO_RGB["sRGB"] @ RGB_TO_XYZ["Rec2020"]).astype(np.float64)
    if np is not None
    else None
)


SRGB_TO_REC2020 = (
    (XYZ_TO_RGB["Rec2020"] @ RGB_TO_XYZ["sRGB"]).astype(np.float64)
    if np is not None
    else None
)


AGX_INSET = agx_engine.AGX_INSET_REC2020


AGX_OUTSET = agx_engine.AGX_OUTSET_REC2020


# As-shot, the LibRaw daylight-metadata anchor (kept for compatibility and as the
# prefeed calibration reference), and declared fixed-Kelvin standards. Fixed Kelvin is a
# declaration, not an adjustment: 6500K = D65 display white, 5500K = photographic
# daylight / daylight-balanced film, 3400K/3200K = Type A/B tungsten film, 9300K = the
# traditional Japanese broadcast white point.
WB_CHOICES = ("camera", "daylight", "6500k", "5500k", "3400k", "3200k", "9300k")


DEMOSAIC_CHOICES = ("auto", "dht", "dcb", "ahd", "aahd", "vng", "ppg")


DEMOSAIC_AUTO_PREFERENCE = ("DHT", "DCB", "AHD")


# Scene-linear RGB producers. Evidence (CFA masks, mosaic) always stays on LibRaw.
DECODER_CHOICES = ("libraw", "coreimage")


COREIMAGE_VERSION_CHOICES = ("auto", "9", "8", "7")

# Core Image scale policy. ``aligned`` is the production comparison contract: a single
# per-file scalar puts RAW 9 on the LibRaw decoded-green median without targeting any
# absolute brightness. ``unity`` preserves Apple's native units, while ``measured`` is
# the old fixed Sigma-fp fit kept only for reproducing historical A/B renders.
COREIMAGE_SCALE_CHOICES = ("aligned", "unity", "measured")
COREIMAGE_SCALE_DEFAULT_MODE = "aligned"
# Legacy median ratio retained for explicit --coreimage-scale measured runs.
COREIMAGE_SCALE_MEASURED_RATIO = 1.0293


OKLAB_M1 = (
    np.array(  # XYZ(D65) -> LMS
        [
            [0.8189330101, 0.3618667424, -0.1288597137],
            [0.0329845436, 0.9293118715, 0.0361456387],
            [0.0482003018, 0.2643662691, 0.6338517070],
        ],
        dtype=np.float64,
    )
    if np is not None
    else None
)


OKLAB_M2 = (
    np.array(  # LMS' -> Oklab
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ],
        dtype=np.float64,
    )
    if np is not None
    else None
)


OKLAB_M1_INV = np.linalg.inv(OKLAB_M1).astype(np.float64) if np is not None and OKLAB_M1 is not None else None


OKLAB_M2_INV = np.linalg.inv(OKLAB_M2).astype(np.float64) if np is not None and OKLAB_M2 is not None else None


CHROMA_CHOICES = ("444", "422", "420")
