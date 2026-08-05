# SPDX-License-Identifier: GPL-3.0-or-later
"""Full display LUT filters (log encode -> .cube -> display).

Unlike dngscan.look (chromatic geometry only, L untouched), these are output
transforms: Kodak 2383 FPE (Cineon log in) and RED IPP2 (Log3G10/RWG in).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._deps import np
from .color import apply_rgb_matrix3, bt1886_eotf, rec2020_to_output, rec2020_to_srgb, rec2020_to_xyz, srgb_to_output
from .log_encode import XYZ_TO_RWG, XYZ_TO_SGAMUT3CINE, encode_for_source
from .lut_io import get_cube, sample_cube

_ASSETS = Path(__file__).resolve().parents[1] / "dngscan_assets" / "vendor_luts"


@dataclass(frozen=True)
class DisplayFilter:
    label: str
    cube: Path
    source: str  # cineon | log3g10
    input_space: str  # rec709 | rwg — linear RGB space fed to the encoder
    display_eotf: str = "gamma"  # gamma | bt1886 — how to decode LUT output to linear
    display_gamma: float = 2.4  # used when display_eotf == gamma
    # What the encoder is fed: "display" = the AgX display render (right for print film
    # emulation, which in grading sits on top of a finished look); "scene" = the
    # scene-linear buffer (right for camera OUTPUT transforms like RED IPP2, which carry
    # their own tone mapping — feeding them AgX output would stack two shoulders).
    feed: str = "display"


DISPLAY_FILTERS: dict[str, DisplayFilter] = {
    "kodak_2383_d65": DisplayFilter(
        label="Kodak 2383 D65 (Resolve FPE)",
        cube=_ASSETS / "resolve_film_looks" / "Rec709 Kodak 2383 D65.cube",
        source="cineon",
        input_space="rec709",
        display_gamma=2.4,
        feed="display",
    ),
    "red_ipp2_rec709_medium": DisplayFilter(
        label="RED IPP2 Rec709 Medium",
        cube=_ASSETS
        / "red_ipp2"
        / "REC709"
        / "RWG_Log3G10 to REC709_BT1886 with MEDIUM_CONTRAST and R_2_Medium size_33 v1.13.cube",
        source="log3g10",
        input_space="rwg",
        display_eotf="bt1886",
        display_gamma=2.4,
        feed="scene",
    ),
    # Sony's official "Type A" = Alexa-style 709 rendering, published for exactly the
    # "match ARRI on set" use case. A camera OUTPUT transform like IPP2: feed it scene.
    "sony_lc709a": DisplayFilter(
        label="Sony LC-709TypeA (Alexa-style 709)",
        cube=_ASSETS / "sony_look_profiles" / "2_SGamut3CineSLog3_To_LC-709TypeA.cube",
        source="slog3",
        input_space="sgamut3cine",
        display_eotf="bt1886",
        display_gamma=2.4,
        feed="scene",
    ),
}

FILTER_CHOICES: tuple[str, ...] = ("none",) + tuple(DISPLAY_FILTERS)


def filter_available(name: str) -> bool:
    if name == "none":
        return True
    spec = DISPLAY_FILTERS.get(name)
    return spec is not None and spec.cube.is_file()


def _linear_to_encoder_input(rec2020_linear: np.ndarray, spec: DisplayFilter) -> np.ndarray:
    if spec.input_space == "rec709":
        return rec2020_to_srgb(rec2020_linear)
    if spec.input_space == "rwg":
        xyz = rec2020_to_xyz(rec2020_linear)
        return apply_rgb_matrix3(xyz, XYZ_TO_RWG)
    if spec.input_space == "sgamut3cine":
        xyz = rec2020_to_xyz(rec2020_linear)
        return apply_rgb_matrix3(xyz, XYZ_TO_SGAMUT3CINE)
    raise ValueError(f"unknown input_space: {spec.input_space}")


def _decode_lut_display(lut_out: Any, spec: DisplayFilter) -> Any:
    v = np.clip(lut_out, 0.0, 1.0)
    if spec.display_eotf == "bt1886":
        return bt1886_eotf(v)
    if spec.display_gamma > 1.0 + 1e-6:
        return np.power(v, spec.display_gamma)
    return v


def apply_display_filter_rec2020(
    mapped_rec2020: Any,
    output_gamut: str,
    filter_name: str,
    strength: float = 1.0,
    scene_rec2020: Any | None = None,
) -> Any:
    """Blend the AgX display render with a log-encoded display LUT.

    feed="display" filters (print film emulation) encode the AgX render itself;
    feed="scene" filters (camera output transforms, e.g. RED IPP2) act as parallel
    renderers: they encode the scene-linear buffer and strength blends the two
    complete renderings."""
    if filter_name == "none" or strength <= 0.0:
        flat = mapped_rec2020.reshape(-1, 3)
        return rec2020_to_output(flat, output_gamut).reshape(mapped_rec2020.shape)

    if filter_name not in DISPLAY_FILTERS:
        raise ValueError(f"unknown display filter: {filter_name}")
    spec = DISPLAY_FILTERS[filter_name]
    if not spec.cube.is_file():
        raise FileNotFoundError(f"缺少 display LUT：{spec.cube}")

    lut = get_cube(spec.cube)
    flat = mapped_rec2020.reshape(-1, 3).astype(np.float32, copy=False)
    agx_display = rec2020_to_output(flat, output_gamut)

    if spec.feed == "scene":
        if scene_rec2020 is None:
            raise ValueError(f"display filter {filter_name} needs the scene-linear buffer (feed='scene')")
        encoder_src = scene_rec2020.reshape(-1, 3).astype(np.float32, copy=False)
    else:
        encoder_src = flat
    encoder_in = _linear_to_encoder_input(encoder_src, spec)
    encoded = encode_for_source(encoder_in, spec.source)
    lut_out = sample_cube(lut, encoded)
    display_709 = _decode_lut_display(lut_out, spec)
    filtered = srgb_to_output(display_709, output_gamut)

    s = np.float32(min(1.5, max(0.0, strength)))
    out = agx_display * (np.float32(1.0) - s) + filtered * s
    return out.reshape(mapped_rec2020.shape).astype(np.float32, copy=False)
