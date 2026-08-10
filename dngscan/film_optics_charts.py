# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic scene-linear charts for analog-optics diagnosis (FILM_OPTICS_V2
§10.2, phase P0).

These are the fixed, generated stimuli that make a spatial defect a NUMBER
instead of an impression. Nothing in the render path imports this module; it
exists so tests, `tools/film_optics_report.py` and the future
`tools/calibrate_film_optics.py` all drive the operators with the same inputs.

Two conventions hold everywhere here:

- Output is scene-linear Rec.2020 with 0.18 = mid grey, i.e. the same domain
  the film observer eats. "EV" always means log2(value / 0.18).
- Feature geometry is declared in FILM-PLANE MILLIMETRES and converted to
  pixels through `px_per_mm`. That is what lets the same chart be rendered on
  a 36 mm gate and on a 16 mm gate and compared honestly: the physical kernel
  is unchanged, only its share of the frame moves. Never write pixel sizes
  into a chart definition.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MID_GREY = 0.18

# 135 full frame and Super-16, the two gates §10.2 asks to compare.
GATE_35MM_W_MM = 36.0
GATE_S16_W_MM = 12.35


def ev(value: float) -> float:
    """Scene-linear value for an EV relative to 18% grey."""
    return MID_GREY * float(np.exp2(value))


def px_per_mm(width: int, gate_w_mm: float = GATE_35MM_W_MM) -> float:
    return float(width) / float(gate_w_mm)


@dataclass(frozen=True)
class Emitter:
    """One light source on the emitter chart."""

    diameter_mm: float
    exposure_ev: float
    color: tuple[float, float, float]  # relative layer weights, peak 1.0
    label: str


# Colours are declared as Rec.2020 relative primaries, not as "R/G/B pixels":
# a blue emitter must be able to prove it does NOT produce the same halo as a
# white one (§10.1 test 6), which only works if the source keeps its spectrum
# until the layer exposure.
EMITTER_COLORS: dict[str, tuple[float, float, float]] = {
    "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.04, 0.02),
    "green": (0.06, 1.0, 0.06),
    "blue": (0.05, 0.10, 1.0),
}

# §10.2: diameters covering 1/2/4/8/32 px at a 6000 px wide 36 mm gate, held
# in mm so the same physical source can be rendered at any resolution.
EMITTER_DIAMETERS_MM: tuple[float, ...] = (0.006, 0.012, 0.024, 0.048, 0.192)
EMITTER_EV: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0)
BACKGROUND_EV: tuple[float, ...] = (-6.0, -3.0, 0.0)


def _disc(
    h: int, w: int, cy: float, cx: float, radius_px: float, supersample: int = 0
) -> np.ndarray:
    """Area-covered disc mask in [0, 1].

    Supersampled coverage, not a hard `r <= R` test: a sub-pixel emitter is
    the whole point of the 1 px case, and a binary mask would make its total
    energy jump between 0 and 1 as the centre moves. Coverage keeps the
    declared exposure x area product exact enough to compare radial profiles
    across diameters.

    The sample rate ADAPTS to the radius. A fixed 4x4 grid resolves nothing
    below about half a pixel: two discs of 0.36 px and 0.71 px diameter both
    caught exactly the same four sub-samples and came out with identical
    energy, which silently breaks the area law the small-source cases depend
    on. Pass a positive `supersample` to override.
    """
    if supersample <= 0:
        supersample = int(np.clip(np.ceil(8.0 / max(radius_px, 0.05)), 4, 64))
    pad = int(np.ceil(radius_px)) + 2
    y0 = max(int(np.floor(cy)) - pad, 0)
    y1 = min(int(np.ceil(cy)) + pad, h)
    x0 = max(int(np.floor(cx)) - pad, 0)
    x1 = min(int(np.ceil(cx)) + pad, w)
    out = np.zeros((h, w), dtype=np.float32)
    if y1 <= y0 or x1 <= x0:
        return out
    s = max(int(supersample), 1)
    off = (np.arange(s) + 0.5) / s
    yy = (np.arange(y0, y1)[:, None] + off[None, :]).reshape(-1)
    xx = (np.arange(x0, x1)[:, None] + off[None, :]).reshape(-1)
    dy = yy - cy
    dx = xx - cx
    inside = (dy[:, None] ** 2 + dx[None, :] ** 2) <= radius_px * radius_px
    cov = inside.reshape(y1 - y0, s, x1 - x0, s).mean(axis=(1, 3))
    out[y0:y1, x0:x1] = cov.astype(np.float32)
    return out


def emitter_chart(
    height: int,
    width: int,
    *,
    gate_w_mm: float = GATE_35MM_W_MM,
    background_ev: float = -3.0,
    diameters_mm: tuple[float, ...] = EMITTER_DIAMETERS_MM,
    exposures_ev: tuple[float, ...] = EMITTER_EV,
    color: str = "white",
) -> tuple[np.ndarray, list[tuple[Emitter, tuple[float, float]]]]:
    """Grid of emitters: one row per exposure, one column per diameter.

    Returns the scene-linear image and the placement list, so a radial
    profile can be taken about each source without re-deriving its centre.
    Sources are spaced with generous margins because the whole question under
    test is how far light travels from them.
    """
    img = np.full((height, width, 3), np.float32(ev(background_ev)), dtype=np.float32)
    scale = px_per_mm(width, gate_w_mm)
    rgb = EMITTER_COLORS[color]
    placed: list[tuple[Emitter, tuple[float, float]]] = []
    rows, cols = len(exposures_ev), len(diameters_mm)
    for r, e in enumerate(exposures_ev):
        for c, d_mm in enumerate(diameters_mm):
            cy = height * (r + 0.5) / rows
            cx = width * (c + 0.5) / cols
            radius_px = 0.5 * d_mm * scale
            mask = _disc(height, width, cy, cx, max(radius_px, 0.05))
            for ch in range(3):
                img[..., ch] += np.float32(ev(e) * rgb[ch]) * mask
            placed.append(
                (Emitter(d_mm, e, rgb, f"{color}_{d_mm:g}mm_ev{e:+g}"), (cy, cx))
            )
    return img, placed


def single_emitter(
    height: int,
    width: int,
    *,
    gate_w_mm: float = GATE_35MM_W_MM,
    diameter_mm: float = 0.048,
    exposure_ev: float = 4.0,
    background_ev: float = -6.0,
    color: str = "white",
) -> tuple[np.ndarray, tuple[float, float]]:
    """One centred source on a dark field — the radial-profile stimulus.

    Isolation matters: with several sources in frame, any spread wide enough
    to be the actual problem overlaps its neighbours and the measured profile
    is a sum, not a PSF.
    """
    img = np.full((height, width, 3), np.float32(ev(background_ev)), dtype=np.float32)
    scale = px_per_mm(width, gate_w_mm)
    rgb = EMITTER_COLORS[color]
    cy, cx = height / 2.0, width / 2.0
    mask = _disc(height, width, cy, cx, max(0.5 * diameter_mm * scale, 0.05))
    for ch in range(3):
        img[..., ch] += np.float32(ev(exposure_ev) * rgb[ch]) * mask
    return img, (cy, cx)


def edge_chart(
    height: int,
    width: int,
    *,
    dark_ev: float = -4.0,
    bright_ev: float = 3.0,
    tilt_deg: float = 5.0,
) -> np.ndarray:
    """Slanted dark/bright edge for slanted-edge MTF (ISO 12233 style).

    The tilt is what makes the measurement work: a perfectly vertical edge is
    sampled at one sub-pixel phase per row, so the edge spread function can
    never be reconstructed above the pixel Nyquist. ~5 degrees gives many
    distinct phases without letting the edge leave the window.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    slope = np.tan(np.deg2rad(tilt_deg))
    signed = xx - (width / 2.0 + slope * (yy - height / 2.0))
    # One-pixel-wide linear ramp across the edge: band-limited enough that the
    # measured MTF is the system's, not the stimulus's staircase.
    cov = np.clip(signed + 0.5, 0.0, 1.0)
    lo, hi = ev(dark_ev), ev(bright_ev)
    plane = (lo + (hi - lo) * cov).astype(np.float32)
    return np.repeat(plane[:, :, None], 3, axis=2)


def siemens_star(
    height: int,
    width: int,
    *,
    spokes: int = 36,
    dark_ev: float = -3.0,
    bright_ev: float = 1.0,
    radius_frac: float = 0.45,
) -> np.ndarray:
    """Radial spoke target: resolution loss shows as a grey core whose radius
    is the cutoff. Read together with the MTF, it separates "detail is gone"
    from "detail is buried in grain"."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cy, cx = height / 2.0, width / 2.0
    dy, dx = yy - cy, xx - cx
    theta = np.arctan2(dy, dx)
    r = np.hypot(dy, dx)
    wave = 0.5 * (1.0 + np.cos(spokes * theta))
    lo, hi = ev(dark_ev), ev(bright_ev)
    plane = lo + (hi - lo) * wave
    outside = r > radius_frac * min(height, width)
    plane = np.where(outside, ev(dark_ev), plane).astype(np.float32)
    return np.repeat(plane[:, :, None], 3, axis=2)


def detail_grid(
    height: int,
    width: int,
    *,
    gate_w_mm: float = GATE_35MM_W_MM,
    line_widths_mm: tuple[float, ...] = (0.012, 0.024, 0.048, 0.096),
    dark_ev: float = -3.0,
    bright_ev: float = 1.0,
) -> np.ndarray:
    """Bar groups at declared film-plane line widths — the "fine branches and
    window mullions" case. Grain that erases the finest group while the MTF
    still claims response is the acutance failure §4.4 warns about."""
    img = np.full((height, width, 3), np.float32(ev(dark_ev)), dtype=np.float32)
    scale = px_per_mm(width, gate_w_mm)
    band = height // max(len(line_widths_mm), 1)
    for i, lw_mm in enumerate(line_widths_mm):
        period = max(int(round(2.0 * lw_mm * scale)), 2)
        y0 = i * band
        y1 = min(y0 + band, height)
        cols = ((np.arange(width) // (period // 2)) % 2) == 0
        img[y0:y1, cols, :] = np.float32(ev(bright_ev))
    return img


def density_wedge(
    height: int,
    width: int,
    *,
    steps: int = 21,
    low_ev: float = -7.0,
    high_ev: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """21 uniform patches spanning the film latitude, plus the patch EVs.

    Uniform is the requirement, not decoration: grain RMS, skew and PSD are
    only defined on a flat field, and a patch has to be big enough to hold
    several hundred correlation lengths or the estimate is noise about noise.
    """
    evs = np.linspace(low_ev, high_ev, steps)
    img = np.empty((height, width, 3), dtype=np.float32)
    edges = np.linspace(0, width, steps + 1).astype(int)
    for i, e in enumerate(evs):
        img[:, edges[i]:edges[i + 1], :] = np.float32(ev(float(e)))
    return img, evs


def uniform_patch(height: int, width: int, exposure_ev: float) -> np.ndarray:
    """A single flat field — the grain statistics stimulus."""
    return np.full((height, width, 3), np.float32(ev(exposure_ev)), dtype=np.float32)


CHART_BUILDERS = {
    "emitter_grid": emitter_chart,
    "single_emitter": single_emitter,
    "edge": edge_chart,
    "siemens_star": siemens_star,
    "detail_grid": detail_grid,
    "density_wedge": density_wedge,
    "uniform_patch": uniform_patch,
}
