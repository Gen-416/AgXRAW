# SPDX-License-Identifier: GPL-3.0-or-later
"""Color space transforms, Oklab, and gamut fitting."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ._deps import np
from .constants import (
    EPS, GAMUT_EPS, OKLAB_M1, OKLAB_M2, OKLAB_M1_INV, OKLAB_M2_INV,
    OUTPUT_GAMUT_LABELS, OUTPUT_GAMUT_SPACES, RGB_TO_XYZ, XYZ_TO_RGB,
)


def output_gamut_space(output_gamut: str) -> str:
    if output_gamut not in OUTPUT_GAMUT_SPACES:
        raise ValueError(f"unknown output gamut: {output_gamut}")
    return OUTPUT_GAMUT_SPACES[output_gamut]


def output_gamut_label(output_gamut: str) -> str:
    return OUTPUT_GAMUT_LABELS.get(output_gamut, output_gamut)


def read_first_existing(paths: list[Path]) -> bytes | None:
    for path in paths:
        try:
            if path.is_file():
                return path.read_bytes()
        except OSError:
            continue
    return None


def output_icc_profile_bytes(output_gamut: str) -> bytes | None:
    if output_gamut == "p3":
        profile = read_first_existing(
            [
                Path("/System/Library/ColorSync/Profiles/Display P3.icc"),
                Path("/Library/ColorSync/Profiles/Display P3.icc"),
            ]
        )
        if profile is None:
            raise RuntimeError("未找到 Display P3 ICC，无法安全导出 P3，请改用 sRGB")
        return profile
    if output_gamut != "srgb":
        raise ValueError(f"unknown output gamut: {output_gamut}")
    system_profile = read_first_existing(
        [
            Path("/System/Library/ColorSync/Profiles/sRGB Profile.icc"),
            Path("/Library/ColorSync/Profiles/sRGB Profile.icc"),
        ]
    )
    if system_profile is not None:
        return system_profile
    try:
        from PIL import ImageCms

        profile = ImageCms.createProfile("sRGB")
        return ImageCms.ImageCmsProfile(profile).tobytes()
    except Exception:
        return None


def srgb_encode(linear: Any) -> Any:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)


def encode_display_linear(linear: Any, output_gamut: str) -> Any:
    """Display-referred OETF for JPEG delivery (explicit per output gamut).

    Display P3 SDR JPEG uses P3 primaries with the same piecewise transfer as sRGB;
    the ICC profile tags the container. Both paths stay explicit for testability.
    """
    _ = output_gamut_space(output_gamut)
    return srgb_encode(linear)


def srgb_decode(encoded: Any) -> Any:
    v = np.clip(encoded, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def apply_rgb_matrix3(rgb: Any, matrix: Any) -> Any:
    out = np.empty((rgb.shape[0], 3), dtype=np.float32)
    out[:, 0] = matrix[0, 0] * rgb[:, 0] + matrix[0, 1] * rgb[:, 1] + matrix[0, 2] * rgb[:, 2]
    out[:, 1] = matrix[1, 0] * rgb[:, 0] + matrix[1, 1] * rgb[:, 1] + matrix[1, 2] * rgb[:, 2]
    out[:, 2] = matrix[2, 0] * rgb[:, 0] + matrix[2, 1] * rgb[:, 1] + matrix[2, 2] * rgb[:, 2]
    return out


def rec2020_to_xyz(rgb: Any) -> Any:
    return apply_rgb_matrix3(rgb, RGB_TO_XYZ["Rec2020"])


def rec2020_to_srgb(rgb: Any) -> Any:
    return apply_rgb_matrix3(rec2020_to_xyz(rgb), XYZ_TO_RGB["sRGB"])


def rec2020_to_output(rgb: Any, output_gamut: str) -> Any:
    return apply_rgb_matrix3(rec2020_to_xyz(rgb), XYZ_TO_RGB[output_gamut_space(output_gamut)])


def srgb_to_output(rgb: Any, output_gamut: str) -> Any:
    if output_gamut == "srgb":
        return rgb
    return apply_rgb_matrix3(apply_rgb_matrix3(rgb, RGB_TO_XYZ["sRGB"]), XYZ_TO_RGB[output_gamut_space(output_gamut)])


def luminance_from_rec2020(rgb: Any) -> Any:
    matrix = RGB_TO_XYZ["Rec2020"]
    y = (matrix[1, 0] * rgb[:, 0] + matrix[1, 1] * rgb[:, 1] + matrix[1, 2] * rgb[:, 2]).astype(
        np.float32, copy=False
    )
    return np.clip(np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=0.0), 0.0, None)


def luminance_from_srgb(rgb: Any) -> Any:
    matrix = RGB_TO_XYZ["sRGB"]
    y = (matrix[1, 0] * rgb[:, 0] + matrix[1, 1] * rgb[:, 1] + matrix[1, 2] * rgb[:, 2]).astype(
        np.float32, copy=False
    )
    return np.clip(np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=0.0), 0.0, None)


def luminance_from_rgb_space(rgb: Any, output_gamut: str) -> Any:
    matrix = RGB_TO_XYZ[output_gamut_space(output_gamut)]
    y = (matrix[1, 0] * rgb[:, 0] + matrix[1, 1] * rgb[:, 1] + matrix[1, 2] * rgb[:, 2]).astype(
        np.float32, copy=False
    )
    return np.clip(np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=0.0), 0.0, None)


def rgb_to_oklab(rgb: Any, output_gamut: str) -> tuple[Any, Any, Any]:
    space = output_gamut_space(output_gamut)
    xyz = apply_rgb_matrix3(rgb, RGB_TO_XYZ[space])
    lms = apply_rgb_matrix3(xyz, OKLAB_M1)
    lab = apply_rgb_matrix3(np.cbrt(lms), OKLAB_M2)
    return lab[:, 0], lab[:, 1], lab[:, 2]


def oklab_to_output_rgb(lab_l: Any, lab_a: Any, lab_b: Any, output_gamut: str) -> Any:
    space = output_gamut_space(output_gamut)
    lab = np.stack([lab_l, lab_a, lab_b], axis=1)
    lms_ = apply_rgb_matrix3(lab, OKLAB_M2_INV)
    xyz = apply_rgb_matrix3(lms_ * lms_ * lms_, OKLAB_M1_INV)
    return apply_rgb_matrix3(xyz, XYZ_TO_RGB[space])


def _rgb_rows_in_unit_gamut(rgb: Any, tolerance: Any) -> Any:
    """Return the per-row gamut mask without allocating min/max reductions."""
    upper = 1.0 + tolerance
    return (
        (rgb[:, 0] >= -tolerance)
        & (rgb[:, 0] <= upper)
        & (rgb[:, 1] >= -tolerance)
        & (rgb[:, 1] <= upper)
        & (rgb[:, 2] >= -tolerance)
        & (rgb[:, 2] <= upper)
    )


def fit_to_output_gamut(rgb: Any, output_gamut: str, alpha: float = 0.05, iters: int = 16) -> Any:
    """Bring out-of-gamut linear RGB into [0,1] with Oklab adaptive-L0 clipping: hold hue,
    trade a little lightness for saturation only at the extremes. In-gamut pixels are left
    untouched. Replaces per-channel clipping, which skews hue on saturated colors."""
    rgb = np.nan_to_num(rgb.astype(np.float32, copy=False), nan=0.0, posinf=1e6, neginf=-1e6)
    tol = np.float32(1e-4)
    oog = ~_rgb_rows_in_unit_gamut(rgb, tol)
    if not np.any(oog):
        return np.clip(rgb, 0.0, 1.0)
    sub = rgb[oog]
    lab_l, lab_a, lab_b = rgb_to_oklab(sub, output_gamut)
    chroma = np.hypot(lab_a, lab_b)
    ld = lab_l - np.float32(0.5)
    abs_ld = np.abs(ld)
    e1 = np.float32(0.5) + abs_ld + np.float32(alpha) * chroma
    l0 = np.float32(0.5) * (1.0 + np.sign(ld) * (e1 - np.sqrt(np.maximum(e1 * e1 - 2.0 * abs_ld, 0.0))))
    lo = np.zeros_like(lab_l)
    hi = np.ones_like(lab_l)
    for _ in range(iters):
        t = 0.5 * (lo + hi)
        rgb_t = oklab_to_output_rgb(l0 * (1.0 - t) + t * lab_l, t * lab_a, t * lab_b, output_gamut)
        inside = _rgb_rows_in_unit_gamut(rgb_t, tol)
        lo = np.where(inside, t, lo)
        hi = np.where(inside, hi, t)
    fit = oklab_to_output_rgb(l0 * (1.0 - lo) + lo * lab_l, lo * lab_a, lo * lab_b, output_gamut)
    out = rgb.copy()
    out[oog] = fit
    return np.clip(out, 0.0, 1.0)


def clamp_float(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def smoothstep(edge0: float, edge1: float, x: Any) -> Any:
    if edge1 <= edge0 + EPS:
        return np.zeros_like(x, dtype=np.float32)
    t = np.clip((x - np.float32(edge0)) / np.float32(edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rec709_inverse_oetf(encoded: Any) -> Any:
    """Decode Rec.709 camera OETF code values to linear light."""
    v = np.clip(encoded, 0.0, 1.0)
    a = np.float32(0.099)
    return np.where(v < np.float32(0.081), v / np.float32(4.5), np.power((v + a) / (np.float32(1.0) + a), 1.0 / 0.45))


def bt1886_eotf(encoded: Any) -> Any:
    """Decode normalized BT.1886 display code to linear light with ideal black."""
    v = np.clip(encoded, 0.0, 1.0)
    return np.power(v, np.float32(2.4))
