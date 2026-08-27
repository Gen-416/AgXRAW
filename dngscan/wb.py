# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed-Kelvin white balance: declared references, never eyeballed neutrality.

The mode list is deliberately a set of named standards rather than a slider. Adaptive
"find neutral by eye" loses to the camera's own metering every time (the eye chromatically
adapts while you look); a declared reference has no eye in the loop — its accuracy is a
property of the camera's colour calibration, not of anyone's judgement.

LibRaw path: the target CCT becomes camera-RGB multipliers through the DNG relation
``CameraNeutral = ColorMatrix @ XYZ(white)``. Files carrying DNG calibration tags use
the dual-illuminant interpolation between the calibration matrices
(``interpolated_color_matrix``, generalized to the DNG 1.6 third illuminant); the
single-matrix solve on LibRaw's one Adobe D65-referenced table is the fallback for
formats without those tags, and no calibration at all is a refusal, not a guess
(``solve_kelvin_wb``). Daylight-family targets (>= 4000 K) sit on the CIE
daylight locus with the revised-c2 temperature correction; tungsten targets are true
blackbodies on the Planckian locus (Kim et al. approximation), because Type A/B film is
balanced for incandescent sources, not for D-series daylight.

RAW 9 path: since the hot-WB migration both decoders decode at the fixed as-shot
neutral and receive the SAME project hot-WB matrix after the linear handoff
(``raw_io.rebalance_raw_bundle``); Apple's own neutralTemperature/neutralTint interface
is not part of the declaration. The two decoders still differ in reconstruction and in
the colour matrix Apple applied, which is measured and reported, not hidden.
"""
from __future__ import annotations

import math
from typing import Any

from ._deps import np

# Mode name -> (CCT, short label). Labels state what each temperature *is*; the GUI and
# CLI surface them so a user picks a standard, not a number they have to research.
KELVIN_WB_MODES: dict[str, tuple[float, str]] = {
    "6500k": (6500.0, "D65 · sRGB/Rec.709 显示标准白点"),
    "5500k": (5500.0, "摄影日光 · 日光卷胶片基准"),
    "3400k": (3400.0, "Type A 钨丝卷（photoflood 摄影灯）"),
    "3200k": (3200.0, "Type B 钨丝卷（3200K 影棚钨丝/卤素灯）"),
    "9300k": (9300.0, "日本广播电视白点（9300K，NTSC-J 传统）"),
}

# CIE recommends evaluating D-series illuminants with the revised radiation constant:
# the historical daylight-locus polynomial expects T multiplied by 1.4388/1.4380.
_D_SERIES_C2_CORRECTION = 1.4388 / 1.4380


def kelvin_mode_cct(wb_mode: str) -> float | None:
    entry = KELVIN_WB_MODES.get(str(wb_mode))
    return entry[0] if entry is not None else None


def cct_to_xy(cct: float) -> tuple[float, float]:
    """Chromaticity of the declared white: daylight locus >= 4000 K, Planckian below.

    Tungsten film is balanced for incandescent (blackbody) light, so 3200/3400 take the
    Planckian locus; D-series targets take the CIE daylight locus with the revised-c2
    temperature correction so 6500 lands on modern D65.

    Known seam: the two loci do not meet — at the 4000 K switchover the daylight locus
    sits ~0.005 Duv above the Planckian, so chromaticity is discontinuous across it.
    That is acceptable *only because* every declared mode keeps well clear of the seam
    (nearest: 3400 K, 600 K away — an invariant pinned by tests). A mode near 4000 K
    must not be added without first bridging the seam (e.g. a declared blend band).
    """
    t = float(cct)
    if not 1667.0 <= t <= 25000.0:
        raise ValueError(f"CCT {t} K outside supported range 1667..25000")
    if t >= 4000.0:
        td = t * _D_SERIES_C2_CORRECTION
        it = 1e3 / td
        it2 = it * it
        it3 = it2 * it
        if td <= 7000.0:
            x = 0.244063 + 0.09911 * it + 2.9678 * it2 - 4.6070 * it3
        else:
            x = 0.237040 + 0.24748 * it + 1.9018 * it2 - 2.0064 * it3
        y = -3.000 * x * x + 2.870 * x - 0.275
        return x, y
    it = 1e3 / t
    it2 = it * it
    it3 = it2 * it
    x = 0.179910 + 0.8776956 * it - 0.2343589 * it2 - 0.2661239 * it3
    if t >= 2222.0:
        y = -0.16748867 + 2.09137015 * x - 1.37418593 * x * x - 0.9549476 * x ** 3
    else:
        # Kim et al. (2002) low-range cubic: -1.1063814, not the 2222-4000 K
        # coefficient (self-review 2026-08-27: the copy-paste put a +0.027 y
        # error at 1700 K and a 0.019 jump at the 2222 K seam; every declared
        # mode is >= 3200 K, so nothing shipped was affected).
        y = -0.20219683 + 2.18555832 * x - 1.34811020 * x * x - 1.1063814 * x ** 3
    return x, y


def _xy_to_xyz(x: float, y: float) -> np.ndarray:
    if y <= 0.0:
        raise ValueError("degenerate chromaticity")
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def kelvin_camera_multipliers(cct: float, xyz_to_cam: Any) -> list[float]:
    """G-normalized [R, G, B, G2] multipliers rendering the declared white as neutral.

    ``xyz_to_cam`` is LibRaw's ``rgb_xyz_matrix`` (the Adobe/DNG ColorMatrix rows for
    R, G, B[, G2]): CameraNeutral = M @ XYZ(white); the multiplier for each channel is
    the reciprocal of its neutral response, normalized so green is 1.
    """
    matrix = np.asarray(xyz_to_cam, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3 or matrix.shape[0] < 3:
        raise ValueError("camera colour matrix unavailable; Kelvin WB needs ColorMatrix")
    m3 = matrix[:3, :]
    if not np.all(np.isfinite(m3)) or float(np.abs(m3).sum()) <= 1e-9:
        raise ValueError(
            "camera colour matrix is empty for this file; "
            "Kelvin WB is unavailable (as-shot still works)"
        )
    x, y = cct_to_xy(float(cct))
    neutral = m3 @ _xy_to_xyz(x, y)
    if np.any(neutral <= 1e-9):
        raise ValueError(f"non-physical camera neutral for {cct} K: {neutral}")
    mult = neutral[1] / neutral
    return [float(mult[0]), 1.0, float(mult[2]), 1.0]


def interpolated_color_matrix(calibration: Any, cct: float) -> Any:
    """Adobe-style dual-illuminant interpolation of the DNG colour matrices.

    Weights are linear in reciprocal CCT between the two calibration illuminants,
    clamped at the ends. Because the target CCT here is *declared* rather than solved
    from an as-shot neutral, the DNG SDK's fixed-point iteration is unnecessary — the
    forward interpolation is exact for this use.
    """
    # R6 item 3: generalized to the DNG 1.6 third illuminant — collect every
    # (cct, matrix) pair the file declares, sort by reciprocal CCT, clamp
    # outside the span, and interpolate linearly between the BRACKETING pair
    # (the spec's dual-illuminant rule applied piecewise).
    pairs = [(float(calibration.cct1), np.asarray(calibration.matrix1, dtype=np.float64))]
    if calibration.matrix2 is not None and calibration.cct2 is not None:
        pairs.append(
            (float(calibration.cct2), np.asarray(calibration.matrix2, dtype=np.float64))
        )
    if getattr(calibration, "matrix3", None) is not None and getattr(
        calibration, "cct3", None
    ) is not None:
        pairs.append(
            (float(calibration.cct3), np.asarray(calibration.matrix3, dtype=np.float64))
        )
    if len(pairs) == 1:
        return pairs[0][1]
    pairs.sort(key=lambda p: 1.0 / p[0])
    inv = 1.0 / float(cct)
    if inv <= 1.0 / pairs[0][0]:
        return pairs[0][1]
    if inv >= 1.0 / pairs[-1][0]:
        return pairs[-1][1]
    for (cct_a, m_a), (cct_b, m_b) in zip(pairs, pairs[1:]):
        inv_a, inv_b = 1.0 / cct_a, 1.0 / cct_b
        if inv_a <= inv <= inv_b:
            if abs(inv_b - inv_a) < 1e-12:
                return m_a
            w = (inv_b - inv) / (inv_b - inv_a)
            return w * m_a + (1.0 - w) * m_b
    return pairs[-1][1]


def asshot_reference_cct(calibration: Any, camera_wb: Any) -> float:
    """CCT of the decode-side (as-shot) white, solved against the file's own tags.

    This is the DNG-SDK-style fixed-point that ``interpolated_color_matrix`` documents
    as unnecessary for *declared* targets: for an as-shot neutral the matrix depends on
    the CCT and the CCT depends on the matrix, so iterate.  The camera multipliers are
    the reciprocal of the camera-neutral response (see ``kelvin_camera_multipliers``);
    inverting the interpolated matrix maps that neutral to XYZ, McCamy's approximation
    maps chromaticity back to CCT, and the loop converges in a few steps.  The result
    anchors the hot-WB decode matrix C0 to the illuminant the fixed reconstruction was
    actually balanced for, instead of an arbitrary fixed reference.
    """
    values = np.asarray(list(camera_wb)[:3] if camera_wb else [], dtype=np.float64)
    if values.size < 3 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"invalid as-shot multipliers for CCT solve: {camera_wb!r}")
    neutral = values[1] / values
    cct = 5000.0
    for _ in range(16):
        matrix = np.asarray(
            interpolated_color_matrix(calibration, cct), dtype=np.float64
        )[:3, :3]
        try:
            xyz = np.linalg.solve(matrix, neutral)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"singular colour calibration during as-shot CCT solve: {exc}"
            ) from exc
        total = float(np.sum(xyz))
        if not np.all(np.isfinite(xyz)) or total <= 0.0 or float(xyz[1]) <= 0.0:
            raise ValueError("non-physical as-shot white during CCT solve")
        x = float(xyz[0]) / total
        y = float(xyz[1]) / total
        denominator = 0.1858 - y
        if abs(denominator) < 1e-9:
            raise ValueError("degenerate chromaticity during as-shot CCT solve")
        n = (x - 0.3320) / denominator
        candidate = 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33
        candidate = min(25000.0, max(1667.0, float(candidate)))
        if abs(candidate - cct) < 1.0:
            return candidate
        cct = candidate
    return cct


def matrix_health(
    dng_calibration: Any | None,
    xyz_to_cam: Any | None,
    cct: float = 6500.0,
) -> dict[str, Any] | None:
    """Diagnostic condition-number check of the active colour calibration.

    Route-E (two-route doctrine, 2026-08-26): a decoder colour defect must not
    be misread as an AgX gamut problem, so the report states which matrix the
    declaration rides on and how well-conditioned it is. Thresholds derive
    from measurement, not convention: the 18 real DNG matrices in the
    evidence-shell corpus span kappa 2.40..4.23 (median 3.01) and the fallback
    fleet 2.56..3.18, so >6 (1.5x the measured max) reads as 偏高 and >10 as
    异常 — either means the tags deserve suspicion (digitization damage,
    wrong matrix for the body), never a render refusal. Diagnostic only.
    """
    if dng_calibration is not None:
        matrix = interpolated_color_matrix(dng_calibration, float(cct))
        source = "dng-dual"
    elif xyz_to_cam is not None:
        matrix = np.asarray(xyz_to_cam, dtype=np.float64)[:3, :3]
        source = "single-matrix"
    else:
        return None
    kappa = float(np.linalg.cond(np.asarray(matrix, dtype=np.float64)))
    status = "正常"
    if not math.isfinite(kappa) or kappa > 10.0:
        status = "异常"
    elif kappa > 6.0:
        status = "偏高"
    return {"source": source, "cct": float(cct), "kappa": kappa, "status": status}


def solve_kelvin_wb(
    cct: float,
    *,
    dng_calibration: Any | None = None,
    xyz_to_cam: Any | None = None,
) -> list[float]:
    """Camera multipliers for a declared CCT, best calibration first.

    DNG dual-illuminant calibration interpolates between the file's own measured
    matrices (the accurate path); LibRaw's single Adobe matrix is the fallback for
    non-DNG formats. No calibration at all is a refusal, not a guess.
    """
    if dng_calibration is not None:
        return kelvin_camera_multipliers(
            cct,
            interpolated_color_matrix(dng_calibration, cct),
        )
    if xyz_to_cam is not None:
        return kelvin_camera_multipliers(cct, xyz_to_cam)
    raise ValueError(
        "no colour calibration available for this file; "
        "Kelvin WB needs DNG ColorMatrix tags or a known camera matrix"
    )
