# SPDX-License-Identifier: GPL-3.0-or-later
"""Palette probe volume and colour-difference metrics
(FILM_APPEARANCE_RECIPE_PLAN §14.1, phase P0).

The appearance plan's whole premise is that "full looks weak" has to become a
statement about WHICH hues, at WHICH exposures, at WHICH purity. That needs a
stimulus whose every sample is addressable and a difference measure that can
be decomposed. Both live here; nothing in the render path imports this module.

Two deliberate choices:

- **The exposure axis is scene EV, not lightness.** A recipe indexed by Oklab
  L would move under a different white point or dynamic range, so the same
  scene position would land in a different row of the table. `log2(Y / 0.18)`
  is stable across SDR, HDR and output gamut.
- **Differences are reported decomposed AND as dE00.** A single dE00 cannot
  tell a hue path from a saturation boost, and telling those apart is the
  stated failure criterion for the whole feature (plan §17 risk 2). So every
  comparison also returns hue rotation in degrees, log2 chroma ratio and
  lightness delta.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    OKLAB_M1,
    OKLAB_M1_INV,
    OKLAB_M2,
    OKLAB_M2_INV,
    RGB_TO_XYZ,
    XYZ_TO_RGB,
)

MID_GREY = 0.18
LUMA_REC2020 = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)

# §14.1 axes.
HUE_COUNT = 24
CHROMA_LEVELS = (0.25, 0.5, 0.75, 1.0)
PROBE_EVS = (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0)

# Named regions the recipes make claims about (§10). Hue is the Oklab hue the
# patch is BUILT at; the probe reports where it actually lands.
NAMED_PATCHES: dict[str, tuple[float, float]] = {
    # name: (oklab hue in degrees, chroma fraction of the gamut ray)
    "skin_light": (48.0, 0.30),
    "skin_deep": (40.0, 0.45),
    "foliage": (140.0, 0.55),
    "sky_cyan": (230.0, 0.45),
    "brick": (30.0, 0.55),
    "neon_magenta": (340.0, 0.90),
}


# --------------------------------------------------------------------------
# colour maths
# --------------------------------------------------------------------------

def _mat(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float64) @ np.asarray(m, dtype=np.float64).T


def rec2020_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """[..., 3] linear Rec.2020 -> Oklab. Negative LMS is cube-rooted with the
    sign preserved rather than clamped: the probe deliberately visits colours
    outside the display gamut, and clamping there would quietly move the very
    samples the gamut-pressure metric is about."""
    xyz = _mat(rgb, RGB_TO_XYZ["Rec2020"])
    lms = _mat(xyz, OKLAB_M1)
    return _mat(np.cbrt(lms), OKLAB_M2)


def oklab_to_rec2020(lab: np.ndarray) -> np.ndarray:
    lms_ = _mat(lab, OKLAB_M2_INV)
    xyz = _mat(lms_ ** 3, OKLAB_M1_INV)
    return _mat(xyz, XYZ_TO_RGB["Rec2020"])


def rec2020_to_lab(rgb: np.ndarray, white: np.ndarray | None = None) -> np.ndarray:
    """CIELAB (D65) from linear Rec.2020, for CIEDE2000.

    Oklab is the recipe's working space because it is perceptually smoother
    for hue paths, but dE00 is defined on CIELAB and the acceptance thresholds
    in §15.2 are quoted in dE00. Converting rather than substituting keeps
    those numbers comparable to everyone else's.
    """
    xyz = _mat(rgb, RGB_TO_XYZ["Rec2020"])
    if white is None:
        white = _mat(np.ones((1, 3)), RGB_TO_XYZ["Rec2020"])[0]
    r = xyz / np.maximum(white, 1e-12)
    delta = 6.0 / 29.0
    f = np.where(
        r > delta ** 3, np.cbrt(np.maximum(r, 0.0)),
        r / (3.0 * delta ** 2) + 4.0 / 29.0,
    )
    return np.stack(
        [116.0 * f[..., 1] - 16.0,
         500.0 * (f[..., 0] - f[..., 1]),
         200.0 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def delta_e00(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 between two CIELAB arrays, elementwise over the leading axes."""
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = 0.5 * (c1 + c2)
    g = 0.5 * (1.0 - np.sqrt(c_bar ** 7 / (c_bar ** 7 + 25.0 ** 7 + 1e-30)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dlp = l2 - l1
    dcp = c2p - c1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, np.where(dhp < -180.0, dhp + 360.0, dhp))
    dhp = np.where(c1p * c2p == 0.0, 0.0, dhp)
    dHp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp) / 2.0)

    lp_bar = 0.5 * (l1 + l2)
    cp_bar = 0.5 * (c1p + c2p)
    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    hp_bar = np.where(
        c1p * c2p == 0.0, hsum,
        np.where(
            hdiff <= 180.0, 0.5 * hsum,
            np.where(hsum < 360.0, 0.5 * (hsum + 360.0), 0.5 * (hsum - 360.0)),
        ),
    )
    t = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )
    d_theta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    rc = 2.0 * np.sqrt(cp_bar ** 7 / (cp_bar ** 7 + 25.0 ** 7 + 1e-30))
    sl = 1.0 + (0.015 * (lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (lp_bar - 50.0) ** 2)
    sc = 1.0 + 0.045 * cp_bar
    sh = 1.0 + 0.015 * cp_bar * t
    rt = -np.sin(np.radians(2.0 * d_theta)) * rc
    return np.sqrt(
        (dlp / sl) ** 2 + (dcp / sc) ** 2 + (dHp / sh) ** 2
        + rt * (dcp / sc) * (dHp / sh)
    )


# --------------------------------------------------------------------------
# the probe volume
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeIndex:
    """Where every sample of the probe volume lives.

    The volume is a flat [N, 3] list, not an image: nothing here is spatial,
    and giving it a picture shape would only invite someone to run a spatial
    operator on it and read the result as colour.
    """

    kind: np.ndarray        # "wheel" | "neutral" | "named"
    hue_deg: np.ndarray     # requested Oklab hue, NaN for neutral
    chroma_frac: np.ndarray
    ev: np.ndarray
    label: tuple[str, ...]


def _ray_chroma(hue_deg: np.ndarray) -> np.ndarray:
    """Chroma of the Rec.2020 gamut boundary along each hue at L=0.7.

    Chroma is declared as a FRACTION of this ray so a "0.5 chroma" patch means
    the same relative purity at every hue. An absolute Oklab chroma would put
    the yellow samples far outside the gamut and the blue ones nowhere near
    it, and the resulting table would say more about Rec.2020 than about any
    film.
    """
    hue = np.radians(np.asarray(hue_deg, dtype=np.float64))
    out = np.empty_like(hue)
    for i, h in enumerate(hue.ravel()):
        lo, hi = 0.0, 0.6
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lab = np.array([[0.7, mid * np.cos(h), mid * np.sin(h)]])
            rgb = oklab_to_rec2020(lab)[0]
            if np.all(rgb >= -1e-9) and np.all(rgb <= 1.0 + 1e-9):
                lo = mid
            else:
                hi = mid
        out.ravel()[i] = lo
    return out


def palette_volume() -> tuple[np.ndarray, ProbeIndex]:
    """The §14.1 stimulus as scene-linear Rec.2020 [N, 3], plus its index.

    Each sample is built in Oklab at a declared hue and chroma fraction, then
    scaled in LINEAR light so its luminance lands on the requested scene EV.
    Building the colour first and setting exposure second is what keeps the
    hue/chroma coordinates comparable down the whole EV axis; scaling in Oklab
    instead would change chroma as a side effect of changing exposure.
    """
    hues = np.arange(HUE_COUNT, dtype=np.float64) * (360.0 / HUE_COUNT)
    rays = _ray_chroma(hues)

    rgb_rows: list[np.ndarray] = []
    kinds: list[str] = []
    hue_col: list[float] = []
    chroma_col: list[float] = []
    ev_col: list[float] = []
    labels: list[str] = []

    def _emit(base_rgb: np.ndarray, kind: str, hue: float, cfrac: float,
              ev: float, label: str) -> None:
        y = float(base_rgb @ LUMA_REC2020)
        scale = (MID_GREY * (2.0 ** ev)) / max(y, 1e-9)
        rgb_rows.append(base_rgb * scale)
        kinds.append(kind)
        hue_col.append(hue)
        chroma_col.append(cfrac)
        ev_col.append(ev)
        labels.append(label)

    for hi, hue in enumerate(hues):
        for cfrac in CHROMA_LEVELS:
            c = cfrac * rays[hi]
            lab = np.array([0.7, c * np.cos(np.radians(hue)), c * np.sin(np.radians(hue))])
            base = np.maximum(oklab_to_rec2020(lab[None, :])[0], 0.0)
            for ev in PROBE_EVS:
                _emit(base, "wheel", float(hue), float(cfrac), float(ev),
                      f"wheel_h{int(round(hue)):03d}_c{cfrac:g}_ev{ev:+g}")

    for ev in np.linspace(-7.0, 7.0, 29):
        _emit(np.ones(3), "neutral", float("nan"), 0.0, float(ev),
              f"neutral_ev{ev:+.1f}")

    for name, (hue, cfrac) in NAMED_PATCHES.items():
        c = cfrac * float(_ray_chroma(np.array([hue]))[0])
        lab = np.array([0.7, c * np.cos(np.radians(hue)), c * np.sin(np.radians(hue))])
        base = np.maximum(oklab_to_rec2020(lab[None, :])[0], 0.0)
        for ev in PROBE_EVS:
            _emit(base, "named", hue, cfrac, float(ev), f"{name}_ev{ev:+g}")

    volume = np.asarray(rgb_rows, dtype=np.float32)
    index = ProbeIndex(
        kind=np.asarray(kinds),
        hue_deg=np.asarray(hue_col, dtype=np.float64),
        chroma_frac=np.asarray(chroma_col, dtype=np.float64),
        ev=np.asarray(ev_col, dtype=np.float64),
        label=tuple(labels),
    )
    return volume, index


# --------------------------------------------------------------------------
# decomposed comparison
# --------------------------------------------------------------------------

def decompose(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Oklab lightness / chroma / hue plus the scene-EV coordinate."""
    lab = rec2020_to_oklab(np.asarray(rgb, dtype=np.float64))
    c = np.hypot(lab[..., 1], lab[..., 2])
    h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0
    y = np.asarray(rgb, dtype=np.float64) @ LUMA_REC2020
    return {
        "L": lab[..., 0],
        "C": c,
        "h_deg": h,
        "ev": np.log2(np.maximum(y, 1e-12) / MID_GREY),
        "lab": lab,
    }


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    """b relative to a, decomposed AND as dE00.

    `d_hue_deg` is wrapped to (-180, 180]; `log2_chroma_ratio` is undefined
    where the reference chroma is ~0, and is returned as NaN there rather than
    as a huge finite number that would poison any percentile taken over it.
    """
    da, db = decompose(a), decompose(b)
    dh = (db["h_deg"] - da["h_deg"] + 180.0) % 360.0 - 180.0
    # Rec.2020 (1,1,1) is not exactly Oklab-neutral through this matrix chain —
    # it lands around C = 4e-4 — so an exact-zero test would call the grey ramp
    # chromatic and hand back hue rotations computed from rounding noise. The
    # threshold sits an order of magnitude above that and two below the least
    # saturated wheel sample (C ~ 0.03).
    neutral = da["C"] < 1e-3
    ratio = np.where(
        neutral, np.nan,
        np.log2(np.maximum(db["C"], 1e-12) / np.maximum(da["C"], 1e-12)),
    )
    return {
        "d_hue_deg": np.where(neutral, np.nan, dh),
        "log2_chroma_ratio": ratio,
        "d_L": db["L"] - da["L"],
        "d_ev": db["ev"] - da["ev"],
        "delta_e00": delta_e00(rec2020_to_lab(a), rec2020_to_lab(b)),
    }


def gamut_pressure(rgb: np.ndarray, gamut: str = "srgb") -> dict[str, float]:
    """Share of samples outside the target gamut before any fit, and by how
    much. Rising pressure is not automatically a fault — a stronger palette
    will push harder — but it must be reported, because a recipe that buys its
    look purely by driving colours out of gamut has moved the work to the
    gamut fitter rather than done it."""
    space = {"srgb": "sRGB", "p3": "P3D65", "rec2020": "Rec2020"}[gamut]
    out = _mat(_mat(np.asarray(rgb, dtype=np.float64), RGB_TO_XYZ["Rec2020"]),
               XYZ_TO_RGB[space])
    below = np.minimum(out, 0.0)
    over = np.maximum(out - 1.0, 0.0)
    outside = np.any((out < -1e-6) | (out > 1.0 + 1e-6), axis=-1)
    return {
        "outside_fraction": float(outside.mean()),
        "max_negative": float(-below.min()),
        "max_over_one": float(over.max()),
        "mean_excess": float((np.abs(below) + over).sum(axis=-1).mean()),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    """Median / p75 / p95 / max of a metric, NaN-safe."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"median": float("nan"), "p75": float("nan"),
                "p95": float("nan"), "max": float("nan")}
    return {
        "median": float(np.median(v)),
        "p75": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }
