# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-file decoder support probe: exactly what each decoder can do, in tiers.

"Support" was a vague word spread across two decoders with several partial
degrees each. This module makes it one deterministic report per file:

LibRaw tiers (colour-table ladder, worst first):
    unsupported_format   the file cannot be opened at all (format gap — e.g.
                         Nikon HE/HE* TicoRAW NEFs; the fallback table cannot help)
    no_color_calibration decodes, but no DNG tags, no LibRaw matrix, no fallback
                         entry: usable with unpredictable colour deviation
    fallback_matrix      decodes; Kelvin WB rescued by camera_matrices.py, but
                         LibRaw's internal conversion still lacks the model
    dng_calibrated       decodes; the file carries its own ColorMatrix tags
    full                 decodes; LibRaw knows the model's colour matrix

Core Image tiers:
    unavailable          no Core Image on this system
    unsupported          Core Image does not accept this file
    raw7 / raw8          accepted, but only by an older decode model
    raw9                 the newest decode model is offered
    (+ blocked_by_libraw flag: the evidence policy always sources sensor facts
     from LibRaw, so a format LibRaw cannot open is intentionally unavailable
     through every scene decoder.)

Sensor priors presence is reported alongside (analysis-scale trust, not colour).
Surfaces: `--support` in the CLI, the GUI decoder card, and the format-gap error.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._deps import np


def _libraw_tier(path: Path, make: str | None, model: str | None) -> dict[str, Any]:
    import rawpy

    from . import dng_metadata
    from .camera_matrices import fallback_xyz_to_cam

    try:
        with rawpy.imread(str(path)) as raw:
            matrix = getattr(raw, "rgb_xyz_matrix", None)
            has_matrix = False
            if matrix is not None:
                m = np.asarray(matrix, dtype=np.float64)
                has_matrix = m.size >= 9 and float(np.abs(m[:3, :3]).sum()) > 1e-9
    except FileNotFoundError:
        raise
    except Exception as exc:
        detail = str(exc)
        if path.suffix.lower() == ".nef":
            detail += "（若为 HE/HE* 高效压缩：TicoRAW 编码，LibRaw 因授权无法解码）"
        return {"status": "unsupported_format", "detail": detail}
    if has_matrix:
        return {"status": "full", "detail": "机型颜色矩阵在 LibRaw 表内"}
    if dng_metadata.read_dng_color_calibration(path) is not None:
        return {"status": "dng_calibrated", "detail": "文件自带 DNG 双光源标定"}
    if fallback_xyz_to_cam(make, model) is not None:
        return {
            "status": "fallback_matrix",
            "detail": "颜色矩阵来自内置回退表；LibRaw 内部转换仍无该机型",
        }
    return {
        "status": "no_color_calibration",
        "detail": "无任何颜色标定：输出可能有不可预测的色彩偏差，功能照常",
    }


def _coreimage_tier(path: Path, libraw_blocked: bool) -> dict[str, Any]:
    from . import coreimage_decode

    probe = coreimage_decode.probe_raw9_support(path)
    versions = [str(v) for v in probe.get("versions_offered", ())]
    if not probe.get("coreimage_available"):
        status = "unavailable"
    elif probe.get("raw9_supported"):
        status = "raw9"
    elif probe.get("fallback_version") is not None:
        status = f"raw{probe['fallback_version']}"
    else:
        status = "unsupported"
    return {
        "status": status,
        "versions": versions,
        "blocked_by_libraw": bool(
            libraw_blocked and status not in ("unavailable", "unsupported")
        ),
    }


_LIBRAW_LABELS = {
    "full": "✓ 完整支持（{detail}）",
    "dng_calibrated": "✓ 完整支持（{detail}）",
    "fallback_matrix": "△ 可用（{detail}）",
    "no_color_calibration": "△ 可用但色彩无锚（{detail}）",
    "unsupported_format": "✗ 不可用（格式缺口：{detail}）",
}


def probe_decode_support(path: Path) -> dict[str, Any]:
    """The unified per-file report. Cheap: metadata open + version probe only."""
    from . import dng_metadata
    from .priors import find_priors

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    shot = dng_metadata.read_dng_shot_info(path)
    libraw = _libraw_tier(path, shot.make, shot.model)
    blocked = libraw["status"] == "unsupported_format"
    coreimage = _coreimage_tier(path, blocked)
    priors = find_priors(shot.make, shot.model) is not None
    from .evidence import libraw_runtime_id

    evidence = {
        "provider": "libraw",
        "provider_version": libraw_runtime_id(),
        "status": libraw["status"],
        "detail": libraw["detail"],
    }

    ident = f"{shot.make or '?'} {shot.model or '?'}".strip()
    lines = [f"机型：{ident}"]
    lines.append(
        "Evidence（LibRaw）："
        + _LIBRAW_LABELS[libraw["status"]].format(detail=libraw["detail"])
    )
    lines.append(
        "LibRaw 场景解码："
        + _LIBRAW_LABELS[libraw["status"]].format(detail=libraw["detail"])
    )
    ci = coreimage["status"]
    if ci == "raw9":
        ci_line = "Apple RAW：✓ RAW 9（最新解码模型）"
    elif ci.startswith("raw"):
        ci_line = f"Apple RAW：△ 仅 RAW {ci[3:]}（此文件不支持 RAW 9，可显式降级）"
    elif ci == "unsupported":
        ci_line = "Apple RAW：✗ 不支持此文件"
    else:
        ci_line = "Apple RAW：✗ 此系统无 Core Image 解码器"
    if coreimage["blocked_by_libraw"]:
        ci_line += "；⚠ 但统一 Evidence 策略要求 LibRaw，该文件当前整体不可用"
    lines.append(ci_line)
    lines.append(
        "传感器先验：" + ("✓ 有（PhotonsToPhotos 实测标尺）" if priors
                     else "－ 无（绝对档位/动态范围为单帧估计）")
    )
    return {
        "make": shot.make,
        "model": shot.model,
        "libraw": libraw,
        "evidence": evidence,
        "coreimage": coreimage,
        "priors": priors,
        "lines": lines,
    }
