# SPDX-License-Identifier: GPL-3.0-or-later
"""Preview/export job logic for the local web GUI."""
from __future__ import annotations

import base64
import dataclasses
import io
import json
import math
import multiprocessing as mp
import threading
from queue import Empty
import os
import secrets
import time
from pathlib import Path
from typing import Any, Callable

import dngscan as dg
from dngscan.debug_util import maybe_print_exc
from dngscan.grade import RENDER_MODE, resolve_grade_params

from .constants import (
    PROXY_LONG_EDGE,
    RAW_EXTS,
    REALTIME_PREVIEW_JPEG_QUALITY,
    REALTIME_PREVIEW_JPEG_SUBSAMPLING,
)
from .histogram import display_histogram, hdr_earned_ev, scene_ev_base, scene_ev_histogram
from .preview_cache import PREVIEW_STORE, PreviewEntry
from .preview_scheduler import PREVIEW_COORDINATOR


# RENDER_LOCK retired (scheduler plan S2): concurrency is owned by the
# RenderScheduler's per-class slots — see dngscan/gui/scheduler.py.
from .scheduler import SCHEDULER


class PreviewSuperseded(RuntimeError):
    """The browser has already requested a newer parameter generation."""


def make_preview_b64(
    path: Path,
    width: int | None = PROXY_LONG_EDGE,
    icc_profile: bytes | None = None,
) -> str:
    from PIL import Image

    with Image.open(path) as src:
        if icc_profile is None:
            icc_profile = src.info.get("icc_profile")
        im = src.convert("RGB")
    if width is not None and max(im.size) > width:
        im.thumbnail((width, width), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    save_kwargs = {
        "format": "JPEG",
        "quality": REALTIME_PREVIEW_JPEG_QUALITY,
        "subsampling": REALTIME_PREVIEW_JPEG_SUBSAMPLING,
    }
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    im.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def preview_b64_from_u8(
    rgb_u8: object,
    icc_profile: bytes | None = None,
    width: int | None = None,
) -> str:
    from PIL import Image

    im = Image.fromarray(rgb_u8, "RGB")
    if width is not None and max(im.size) > width:
        im.thumbnail((width, width), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    save_kwargs = {
        "format": "JPEG",
        "quality": REALTIME_PREVIEW_JPEG_QUALITY,
        "subsampling": REALTIME_PREVIEW_JPEG_SUBSAMPLING,
    }
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    im.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def annotate_preview_rgb_u8(rgb_u8: object, lines: list[str]) -> object:
    from PIL import Image, ImageDraw, ImageFont

    np = dg.np
    if np is None or not lines:
        return rgb_u8
    base = np.asarray(rgb_u8, dtype=np.uint8)
    im = Image.fromarray(base, "RGB")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = im.size
    pad = max(10, h // 100)
    font_size = max(16, h // 42)
    font = None
    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    line_gap = max(4, font_size // 6)
    text_heights = []
    text_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_widths.append(bbox[2] - bbox[0])
        text_heights.append(bbox[3] - bbox[1])
    box_w = max(text_widths) + pad * 2
    box_h = sum(text_heights) + line_gap * (len(lines) - 1) + pad * 2
    draw.rectangle((pad, pad, pad + box_w, pad + box_h), fill=(12, 16, 24, 210))
    y_cursor = pad + pad // 2
    for line, th in zip(lines, text_heights):
        draw.text((pad * 2, y_cursor), line, fill=(255, 236, 170, 255), font=font)
        y_cursor += th + line_gap
    composed = Image.alpha_composite(im.convert("RGBA"), overlay)
    return np.asarray(composed.convert("RGB"), dtype=np.uint8)


def auto_ev_payload(result: dg.AutoEvResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "ev": result.ev,
        "ev_boost": result.ev_boost,
        "ev_median_target": result.ev_median_target,
        "highlight_limited": result.highlight_limited,
        "highlight_cap_ev": result.highlight_cap_ev,
        "anchored_median_ev": result.anchored_median_ev,
    }


def preview_metrics_from_u8(rgb_u8: object, gamut: str) -> dict[str, float]:
    np = dg.np
    if np is None:
        return {}
    rgb = np.asarray(rgb_u8, dtype=np.uint8)
    flat_u8 = rgb.reshape(-1, 3)
    encoded = flat_u8.astype(np.float32) / np.float32(255.0)
    linear = dg.srgb_decode(encoded)
    max_channel = np.max(flat_u8, axis=1)
    weights = dg.RGB_TO_XYZ[dg.output_gamut_space(gamut)][1].astype(np.float32)
    y = (
        weights[0] * linear[:, 0].astype(np.float32)
        + weights[1] * linear[:, 1].astype(np.float32)
        + weights[2] * linear[:, 2].astype(np.float32)
    )
    return {
        "luma_p999_pct": float(np.percentile(y, 99.9) * 100.0),
        "near_white_pct": float(np.mean(max_channel >= 250) * 100.0),
        "clipped_channel_pct": float(np.mean(max_channel >= 254) * 100.0),
    }


# Declared sampling for the post-export display metrics (D10): a deterministic
# stride of ~800k pixels replaces the full-frame walk. Measured cost on the
# 24.5MP reference: worst metric deviation +0.0092 percentage points (median
# luma), headroom EV unchanged at 4 decimals, 1.33s -> 52ms. The delivery
# report labels the sample size; metrics_sample_px carries it.
METRICS_SAMPLE_TARGET = 800_000


def output_luminance_metrics_u8(encoded_u8: object, gamut: str, ev: float) -> dict[str, float]:
    np = dg.np
    if np is None:
        return {}
    encoded_u8 = np.asarray(encoded_u8, dtype=np.uint8)
    flat_u8 = encoded_u8.reshape(-1, 3)
    step = max(1, math.ceil(flat_u8.shape[0] / METRICS_SAMPLE_TARGET))
    if step > 1:
        flat_u8 = flat_u8[::step]
    matrix = dg.RGB_TO_XYZ[dg.output_gamut_space(gamut)]
    y = np.empty((flat_u8.shape[0],), dtype=np.float32)
    max_channel = np.empty((flat_u8.shape[0],), dtype=np.float32)
    near_count = 0
    clipped_count = 0
    chunk = 1_000_000
    for start in range(0, flat_u8.shape[0], chunk):
        end = min(start + chunk, flat_u8.shape[0])
        piece_u8 = flat_u8[start:end]
        encoded = piece_u8.astype(np.float32) / np.float32(255.0)
        linear = dg.srgb_decode(encoded)
        y[start:end] = (
            matrix[1, 0] * linear[:, 0]
            + matrix[1, 1] * linear[:, 1]
            + matrix[1, 2] * linear[:, 2]
        )
        max_channel[start:end] = np.max(linear, axis=1)
        max_u8 = np.max(piece_u8, axis=1)
        near_count += int(np.count_nonzero(max_u8 >= 250))
        clipped_count += int(np.count_nonzero(max_u8 >= 254))
    y = np.clip(np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    y_p99, y_p999 = [float(v) for v in np.percentile(y, [99.0, 99.9])]
    max_p999 = float(np.percentile(max_channel, 99.9))
    headroom_luma_ev = math.log2(0.95 / max(y_p999, 1e-9))
    headroom_rgb_ev = math.log2(0.98 / max(max_p999, 1e-9))
    return {
        "median_luma_pct": float(np.median(y) * 100.0),
        "mean_luma_pct": float(np.mean(y) * 100.0),
        "luma_p99_pct": y_p99 * 100.0,
        "luma_p999_pct": y_p999 * 100.0,
        "max_channel_p999_pct": max_p999 * 100.0,
        "near_white_pct": float(near_count / max(flat_u8.shape[0], 1) * 100.0),
        "clipped_channel_pct": float(clipped_count / max(flat_u8.shape[0], 1) * 100.0),
        "headroom_luma_ev": float(headroom_luma_ev),
        "headroom_rgb_ev": float(headroom_rgb_ev),
        "estimated_ev_before_luma_limit": float(ev + headroom_luma_ev),
        "metrics_sample_px": float(flat_u8.shape[0]),
    }


def output_luminance_metrics(path: Path, gamut: str, ev: float) -> dict[str, float]:
    from PIL import Image

    with Image.open(path) as im:
        encoded_u8 = dg.np.asarray(im.convert("RGB"), dtype=dg.np.uint8)
    return output_luminance_metrics_u8(encoded_u8, gamut, ev)


def estimate_ev_headroom(
    bundle: dg.RawBundle,
    analysis: dg.Analysis | None,
    gamut: str,
    current_ev: float,
    max_samples: int = 220_000,
    look: str = "none",
    look_strength: float = 1.0,
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    punch_scale: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    adjustments: dg.RenderAdjustments | None = None,
    endpoint_mode: str = "adaptive",
    film_curve: str = "none",
    film_mode: str = "observe",
    film_crossover: str = "off",
    film_exposure_ev: float = 0.0,
    film_print_timing: str = "fixed",
    film_print_medium: str = "",
    film_print_exposure_ev: float = 0.0,
    film_grain: float = 0.0,
    film_halation: float = 0.0,
    film_bloom: float = 0.0,
    film_interimage: str = "declared",
    film_appearance: str = "technical",
    film_appearance_strength: float = 1.0,
    film_optics_seed: int = 0,
    color_head_y: float = 0.0,
    color_head_m: float = 0.0,
    lens_filter: str | None = None,
) -> dict[str, float | str]:
    if analysis is None:
        return {}
    safe_ev = dg.max_safe_ev(
        bundle,
        analysis,
        gamut,
        from_ev=current_ev,
        max_samples=max_samples,
        look=look,
        look_strength=look_strength,
        display_filter=display_filter,
        filter_strength=filter_strength,
        scene_transform=scene_transform,
        scene_transform_strength=scene_transform_strength,
        punch_scale=punch_scale,
        tone_core=tone_core,
        lum_norm=lum_norm,
        agx_primaries=agx_primaries,
        adjustments=adjustments,
        endpoint_mode=endpoint_mode,
        film_curve=film_curve,
        film_mode=film_mode,
        film_crossover=film_crossover,
        film_exposure_ev=film_exposure_ev,
        film_print_timing=film_print_timing,
        film_print_medium=film_print_medium,
        film_print_exposure_ev=film_print_exposure_ev,
        film_grain=film_grain,
        film_halation=film_halation,
        film_bloom=film_bloom,
        film_interimage=film_interimage,
        film_appearance=film_appearance,
        film_appearance_strength=film_appearance_strength,
        film_optics_seed=film_optics_seed,
        color_head_y=color_head_y,
        color_head_m=color_head_m,
        lens_filter=lens_filter,
    )
    return {
        "safe_ev_remaining": max(0.0, float(safe_ev - current_ev)),
        "estimated_safe_ev": float(safe_ev),
        "headroom_limit": "p99.9高光/通道顶白/近白比例阈值",
    }


def list_dir(raw: str) -> dict:
    p = Path(raw).expanduser() if raw else Path.home()
    if not p.is_dir():
        p = Path.home()
    dirs: list[str] = []
    files: list[str] = []
    try:
        for entry in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            try:
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    dirs.append(entry.name)
                elif entry.suffix.lower() in RAW_EXTS:
                    files.append(entry.name)
            except OSError:
                continue
    except PermissionError:
        pass
    return {"cwd": str(p), "parent": str(p.parent), "dirs": dirs, "files": files}


def raw9_support(params: dict) -> dict:
    """Return a cheap per-file RAW 9 capability probe for the GUI."""
    inp = Path(str(params.get("input", ""))).expanduser()
    if not inp.is_file():
        raise FileNotFoundError(f"文件不存在：{inp}")
    from dngscan import coreimage_decode
    from dngscan.decode_support import probe_decode_support

    try:
        support_lines = probe_decode_support(inp)["lines"]
    except Exception as exc:  # the probe must never block the GUI flow
        support_lines = [f"支持探测失败：{exc}"]
    probe = coreimage_decode.probe_raw9_support(inp)
    offered = [str(value) for value in probe["versions_offered"]]
    fallback = probe["fallback_version"]
    if not probe["coreimage_available"]:
        message = "此系统没有可用的 Apple Core Image RAW 解码器。"
    elif probe["error"]:
        message = f"Apple RAW 无法打开这个文件：{probe['error']}"
    elif probe["raw9_supported"]:
        message = "此文件支持 Apple RAW 9。"
    elif fallback is not None:
        message = f"此文件不支持 Apple RAW 9；系统最高可使用 RAW {fallback}。"
    else:
        detail = "、".join(offered) if offered else "无"
        message = f"此文件不支持 Apple RAW 9，也没有可用的 RAW 8/7 降级路径（报告版本：{detail}）。"
    return {
        "ok": True,
        "support_lines": support_lines,
        "coreimage_available": bool(probe["coreimage_available"]),
        "raw9_supported": bool(probe["raw9_supported"]),
        "versions_offered": offered,
        "fallback_version": fallback,
        "probe_error": probe["error"],
        "message": message,
    }


def parse_job_params(params: dict) -> tuple[Path, str, str, str, float, float, int, bool, Path | None, bool]:
    inp = Path(str(params["input"])).expanduser()
    if not inp.is_file():
        raise FileNotFoundError(f"文件不存在：{inp}")
    highlight = str(params.get("highlight", "clip"))
    if highlight not in ("clip", "blend", "reconstruct"):
        raise ValueError(f"未知高光处理：{highlight}")
    gamut = str(params.get("gamut", "srgb"))
    if gamut not in ("srgb", "p3"):
        raise ValueError(f"未知输出色域：{gamut}")
    output_format = str(params.get("format", "sdr"))
    if output_format not in dg.JPEG_OUTPUT_FORMATS:
        raise ValueError(f"未知输出格式：{output_format}")
    if dg.is_hdr_output_format(output_format):
        gamut = "p3"
    ev = _finite_number(params.get("ev", 0.0), "ev", -20.0, 20.0)
    hdr_headroom = _finite_number(
        params.get("hdrHeadroom", dg.DEFAULT_HDR_HEADROOM_EV), "hdrHeadroom",
        0.0, float(dg.MAX_HDR_HEADROOM_EV) + 1e-9,
    )
    if not 0.0 <= hdr_headroom <= float(dg.MAX_HDR_HEADROOM_EV) + 1e-9:
        raise ValueError(
            f"HDR capacity 必须在 0–{dg.MAX_HDR_HEADROOM_EV:.6f} EV "
            "（对应最多 4000 nit）"
        )
    quality = int(params.get("quality", 100))
    if not 1 <= quality <= 100:
        raise ValueError("质量需在 1-100 之间")
    want_png = bool(params.get("png", False))
    outdir = Path(str(params["outdir"])).expanduser() if params.get("outdir") else None
    ev_auto = bool(params.get("evAuto", False))
    return inp, highlight, gamut, output_format, ev, hdr_headroom, quality, want_png, outdir, ev_auto



def parse_punch(params: dict) -> float:
    try:
        value = float(params.get("punch", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.5, value))


def parse_render_adjustments(params: dict) -> dg.RenderAdjustments:
    # Each field lists its accepted payload keys in priority order. The shoulder
    # control keeps its former shoulderStartOffset / shoulder_start_offset names as
    # read aliases so persisted settings and older callers survive the rename to the
    # shoulder-white semantics; old values (range [-0.5, +3]) all sit inside the new
    # declared range.
    fields = {
        "midtone_brightness": (("midtoneBrightness",), -1.0, 1.0),
        "midtone_contrast": (("midtoneContrast",), -1.0, 1.0),
        "shadow_transition": (("shadowTransition",), -1.0, 1.0),
        "highlight_transition": (("highlightTransition",), -1.0, 1.0),
        "highlight_fade": (("highlightFade",), -1.0, 1.0),
        "toe_end_offset": (("toeEndOffset",), -3.0, 0.5),
        "shoulder_white_offset": (
            ("shoulderWhiteOffset", "shoulderStartOffset", "shoulder_start_offset"),
            -2.0,
            3.0,
        ),
    }
    values: dict[str, float] = {}
    for field, (keys, low, high) in fields.items():
        raw = 0.0
        for key in (*keys, field):
            if key in params:
                raw = params[key]
                break
        key = keys[0]
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 必须是数字") from exc
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} 需在 {low:g} 到 {high:g} 之间")
        values[field] = value
    return dg.RenderAdjustments(**values)


def parse_endpoint_mode(params: dict) -> str:
    mode = str(params.get("endpointMode", params.get("endpoint_mode", "adaptive")))
    if mode not in dg.ENDPOINT_MODE_CHOICES:
        raise ValueError(f"未知端点模式：{mode}")
    return mode


def parse_scene_transform(params: dict) -> tuple[str, float]:
    transform = dg.validate_scene_transform(str(params.get("sceneTransform", "none")))
    # Full mode's input-domain contract at the GUI's parameter source: every
    # consumer (plan, histograms, brightness reference, cache keys, preview
    # and export) reads the transform through this parse (review batch 10).
    from dngscan.scene_transform import effective_scene_transform

    film_curve = str(params.get("filmCurve", params.get("film_curve", "none")))
    film_mode = str(params.get("filmMode", params.get("film_mode", "observe")))
    transform = effective_scene_transform(transform, film_mode, film_curve)
    strength = float(params.get("sceneTransformStrength", params.get("scene_transform_strength", 1.0)))
    if not 0.0 <= strength <= 3.0:
        raise ValueError("scene transform 强度需在 0-3 之间")
    return transform, strength


def parse_tone_core(params: dict) -> tuple[str, str]:
    core = str(params.get("toneCore", params.get("tone_core", "agx")))
    norm = str(params.get("lumNorm", params.get("lum_norm", "y")))
    if core not in dg.TONE_CORE_CHOICES:
        raise ValueError(f"未知 tone 核：{core}")
    if norm not in dg.LUM_NORM_CHOICES:
        raise ValueError(f"未知 lum norm：{norm}")
    return core, norm


def parse_agx_primaries(params: dict) -> str:
    value = str(params.get("agxPrimaries", params.get("agx_primaries", "base")))
    resolved = dg.agx_engine.resolve_agx_primaries(value)
    if resolved not in dg.agx_engine.AGX_PRIMARIES_PRESETS:
        raise ValueError(f"未知 AgX 基调：{value}")
    return resolved


def _cache_float(value: float) -> float:
    return round(float(value), 6)


def _adjustment_key(adjustments: dg.RenderAdjustments | None) -> tuple[float, ...]:
    if adjustments is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return tuple(
        _cache_float(value)
        for value in (
            adjustments.midtone_brightness,
            adjustments.midtone_contrast,
            adjustments.shadow_transition,
            adjustments.highlight_transition,
            adjustments.highlight_fade,
            adjustments.toe_end_offset,
            adjustments.shoulder_white_offset,
        )
    )


def _cached_render_plan(
    cached: PreviewEntry,
    bundle: dg.RawBundle,
    gamut: str,
    scene_transform: str,
    scene_transform_strength: float,
    punch_scale: float,
    tone_core: str,
    lum_norm: str,
    agx_primaries: str,
    film_curve: str,
    adjustments: dg.RenderAdjustments | None,
    endpoint_mode: str = "adaptive",
    color_head_y: float = 0.0,
    color_head_m: float = 0.0,
    film_mode: str = "observe",
    film_crossover: str = "off",
    film_exposure_ev: float = 0.0,
    film_print_timing: str = "fixed",
    film_print_medium: str = "",
    film_print_exposure_ev: float = 0.0,
    film_grain: float = 0.0,
    film_halation: float = 0.0,
    film_bloom: float = 0.0,
    film_interimage: str = "declared",
    film_appearance: str = "technical",
    film_appearance_strength: float = 1.0,
    film_optics_seed: int = 0,
) -> dg.RenderPlan:
    """Compile expensive scene statistics once, then apply cheap UI biases."""
    key = (
        gamut,
        scene_transform,
        _cache_float(scene_transform_strength),
        _cache_float(punch_scale),
        tone_core,
        lum_norm,
        agx_primaries,
        film_curve,
        _cache_float(color_head_y),
        _cache_float(color_head_m),
        film_mode,
        film_crossover,
        _cache_float(film_exposure_ev),
        film_print_timing,
        film_print_medium,
        _cache_float(film_print_exposure_ev),
        _cache_float(film_grain),
        _cache_float(film_halation),
        _cache_float(film_bloom),
        int(film_optics_seed),
        str(getattr(bundle, "lens_filter", "none")),
        endpoint_mode,
    )
    base = cached.get_or_build_plan(
        key,
        lambda: dg.build_render_plan(
            bundle,
            cached.analysis,
            RENDER_MODE,
            gamut,
            scene_transform,
            scene_transform_strength,
            punch_scale,
            tone_core,
            lum_norm,
            agx_primaries=agx_primaries,
            film_curve=film_curve,
            film_mode=film_mode,
            film_crossover=film_crossover,
            film_exposure_ev=film_exposure_ev,
            film_print_timing=film_print_timing,
        film_print_medium=film_print_medium,
        film_print_exposure_ev=film_print_exposure_ev,
            film_grain=film_grain,
            film_halation=film_halation,
            film_bloom=film_bloom,
            film_interimage=film_interimage,
            film_appearance=film_appearance,
            film_appearance_strength=film_appearance_strength,
            film_optics_seed=film_optics_seed,
            adjustments=None,
            endpoint_mode=endpoint_mode,
            color_head_y=color_head_y,
            color_head_m=color_head_m,
        ),
    )
    return dg.apply_render_adjustments(base, adjustments)


def _preview_pixel_key(
    bundle: dg.RawBundle,
    gamut: str,
    ev: float,
    look: str,
    look_strength: float,
    display_filter: str,
    filter_strength: float,
    scene_transform: str,
    scene_transform_strength: float,
    punch_scale: float,
    tone_core: str,
    lum_norm: str,
    agx_primaries: str,
    lens_filter: str,
    film_curve: str,
    adjustments: dg.RenderAdjustments | None,
    endpoint_mode: str = "adaptive",
    color_head_y: float = 0.0,
    color_head_m: float = 0.0,
    film_mode: str = "observe",
    film_crossover: str = "off",
    film_exposure_ev: float = 0.0,
    film_print_timing: str = "fixed",
    film_print_medium: str = "",
    film_print_exposure_ev: float = 0.0,
    film_grain: float = 0.0,
    film_halation: float = 0.0,
    film_bloom: float = 0.0,
    film_interimage: str = "declared",
    film_appearance: str = "technical",
    film_appearance_strength: float = 1.0,
    film_optics_seed: int = 0,
) -> tuple[Any, ...]:
    return (
        gamut,
        _cache_float(ev),
        look,
        _cache_float(look_strength),
        display_filter,
        _cache_float(filter_strength),
        scene_transform,
        _cache_float(scene_transform_strength),
        _cache_float(punch_scale),
        tone_core,
        lum_norm,
        agx_primaries,
        lens_filter,
        film_curve,
        _cache_float(color_head_y),
        _cache_float(color_head_m),
        film_mode,
        film_crossover,
        _cache_float(film_exposure_ev),
        film_print_timing,
        film_print_medium,
        _cache_float(film_print_exposure_ev),
        _cache_float(film_grain),
        _cache_float(film_halation),
        _cache_float(film_bloom),
        int(film_optics_seed),
        endpoint_mode,
        _adjustment_key(adjustments),
        # Exposure is represented by ``ev`` above; the scale contract guards against
        # accidentally sharing frames across decoder/cache versions.
        _cache_float(getattr(bundle, "scene_scale", 1.0)),
        str(getattr(bundle, "scene_decoder_runtime", "") or ""),
    )


def _preview_frame_key(
    pixel_key: tuple[Any, ...], include_metrics: bool
) -> tuple[Any, ...]:
    """Exact browser representation layered on top of reusable rendered pixels."""
    return (*pixel_key, bool(include_metrics))


def parse_decoder(params: dict) -> tuple[str, str]:
    from dngscan.constants import COREIMAGE_VERSION_CHOICES, DECODER_CHOICES
    from dngscan import coreimage_decode

    decoder = str(params.get("decoder", "libraw"))
    version = str(params.get("coreimageVersion", params.get("coreimage_version", "auto")))
    if decoder not in DECODER_CHOICES:
        raise ValueError(f"未知解码器：{decoder}")
    if version not in COREIMAGE_VERSION_CHOICES:
        raise ValueError(f"未知 Core Image 版本：{version}")
    if decoder == "coreimage" and not coreimage_decode.available():
        raise RuntimeError("Core Image 解码器在此系统不可用（需要 macOS + PyObjC Quartz）")
    wb = str(params.get("wb", "camera"))
    if decoder == "coreimage" and wb == "daylight":
        raise ValueError(
            "Core Image 解码器不支持“相机日光标定”模式（LibRaw 元数据乘数无验证映射）；"
            "固定色温声明（如 5500K）与拍摄值均可用"
        )
    return decoder, version


def parse_demosaic(params: dict, decoder: str) -> str:
    from dngscan.constants import DEMOSAIC_CHOICES

    demosaic = str(params.get("demosaic", "auto"))
    if demosaic not in DEMOSAIC_CHOICES:
        raise ValueError(f"未知解拜耳算法：{demosaic}")
    return "auto" if decoder == "coreimage" else demosaic


def export_preview_jpeg(
    inp: Path,
    highlight: str,
    gamut: str,
    ev: float,
    quality: int,
    max_width: int = 1400,
    wb: str = "camera",
    look: str = "none",
    look_strength: float = 1.0,
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    auto_ev: dg.AutoEvResult | None = None,
    punch_scale: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    cached: PreviewEntry | None = None,
    adjustments: dg.RenderAdjustments | None = None,
    decoder: str = "libraw",
    coreimage_version: str = "auto",
    demosaic: str = "auto",
    lens_filter: str = "none",
    film_curve: str = "none",
    film_mode: str = "observe",
    film_crossover: str = "off",
    film_exposure_ev: float = 0.0,
    film_print_timing: str = "fixed",
    film_print_medium: str = "",
    film_print_exposure_ev: float = 0.0,
    film_grain: float = 0.0,
    film_halation: float = 0.0,
    film_bloom: float = 0.0,
    film_interimage: str = "declared",
    film_appearance: str = "technical",
    film_appearance_strength: float = 1.0,
    film_optics_seed: int | None = None,
    endpoint_mode: str = "adaptive",
    color_head_y: float = 0.0,
    color_head_m: float = 0.0,
    include_metrics: bool = True,
    is_current: Callable[[], bool] | None = None,
) -> dict:
    dg.require_dependencies()
    if decoder == "coreimage":
        highlight = "reconstruct"
    if cached is None:
        cached = PREVIEW_STORE.get(
            inp,
            highlight,
            wb,
            tone_core == "gated",
            decoder,
            coreimage_version,
            demosaic,
        )

    def ensure_current() -> None:
        if is_current is not None and not is_current():
            raise PreviewSuperseded()

    ensure_current()

    if film_optics_seed is None:
        # the entry's one realization: preview grain IS export grain
        film_optics_seed = int(getattr(cached, "realization_id", 0) or 0)
    proxy_bundle = dg.with_intent_exposure(
        cached.bundle, user_ev=ev, tone_core=tone_core
    )
    if lens_filter != "none":
        # Shallow copy: cached proxy bundles are shared across requests and must not
        # inherit one request's declared glass.
        import dataclasses as _dc

        proxy_bundle = _dc.replace(proxy_bundle, lens_filter=lens_filter)
    pixel_key = _preview_pixel_key(
        proxy_bundle,
        gamut,
        ev,
        look,
        look_strength,
        display_filter,
        filter_strength,
        scene_transform,
        scene_transform_strength,
        punch_scale,
        tone_core,
        lum_norm,
        agx_primaries,
        lens_filter,
        film_curve,
        adjustments,
        endpoint_mode,
        color_head_y,
        color_head_m,
        film_mode,
        film_crossover,
        film_exposure_ev,
        film_print_timing,
        film_print_medium,
        film_print_exposure_ev,
        film_grain,
        film_halation,
        film_bloom,
        film_optics_seed,
    )
    frame_key = _preview_frame_key(pixel_key, include_metrics)
    if auto_ev is None:
        frame = cached.get_frame(frame_key)
        if frame is not None:
            ensure_current()
            frame["cache_hit"] = True
            return frame
    with SCHEDULER.slot("preview"):
        try:
            ensure_current()
        except PreviewSuperseded:
            # dropped at the slot boundary: a newer generation arrived while
            # this request queued (the observable S2 acceptance signal)
            SCHEDULER.note_dropped()
            raise
        rgb_u8 = cached.get_pixels(pixel_key) if auto_ev is None else None
        pixel_cache_hit = rgb_u8 is not None
        # The compiled plan is consulted even on a pixel-cache hit: the histogram
        # annotations (curve endpoints, reliable tail, earned HDR headroom) must
        # quote the exact plan those pixels consumed. The base compile is
        # LRU-cached, so on interactive frames this is a dictionary hit.
        render_plan = _cached_render_plan(
            cached,
            proxy_bundle,
            gamut,
            scene_transform,
            scene_transform_strength,
            punch_scale,
            tone_core,
            lum_norm,
            agx_primaries,
            film_curve,
            adjustments,
            endpoint_mode,
            color_head_y,
            color_head_m,
            film_mode,
            film_crossover,
            # P5c fix: the preview plan silently dropped every film state
            # after crossover (exposure/timing/medium/print exposure since
            # P2/P3, optics now) — the preview rendered EV0 fixed timing
            # regardless of the dials while the export honoured them.
            film_exposure_ev,
            film_print_timing,
            film_print_medium,
            film_print_exposure_ev,
            film_grain,
            film_halation,
            film_bloom,
            film_optics_seed,
        )
        if rgb_u8 is None:
            ensure_current()
            rgb_u8 = dg.render_output_u8(
                proxy_bundle, cached.analysis, gamut, render_plan,
                look, look_strength, display_filter, filter_strength,
                scene_transform, scene_transform_strength,
                tone_core, lum_norm, agx_primaries,
                dither_noise=cached.get_or_build_dither_noise(),
            )
            ensure_current()
            if auto_ev is None:
                rgb_u8 = cached.put_pixels(pixel_key, rgb_u8)
        icc_profile = dg.output_icc_profile_bytes(gamut)
        # Both histograms ride the same response as the frame they describe, so
        # the page's latest-wins logic keeps image and histograms in lockstep.
        # The scene EV0 population is compiled once per proxy/transform state
        # (user EV is an exact shift, see gui.histogram); the display histogram
        # reads the u8 frame before any auto-EV overlay is painted on it.
        scene_hist_base = cached.get_or_build_plan(
            (
                "scene_ev_hist",
                scene_transform,
                _cache_float(scene_transform_strength),
                lens_filter,
            ),
            lambda: scene_ev_base(
                proxy_bundle, cached.analysis, scene_transform, scene_transform_strength
            ),
        )
        scene_hist = scene_ev_histogram(scene_hist_base, render_plan, ev)
        display_hist = display_histogram(rgb_u8)
        if auto_ev is not None:
            rgb_u8 = annotate_preview_rgb_u8(rgb_u8, dg.auto_ev_overlay_lines(auto_ev))
        metrics = preview_metrics_from_u8(rgb_u8, gamut) if include_metrics else {}
        preview = preview_b64_from_u8(rgb_u8, icc_profile=icc_profile)
        ensure_current()
    payload = {
        "ok": True,
        "preview": preview,
        "metrics": metrics,
        "metrics_kind": "preview" if include_metrics else "deferred",
        "gain": proxy_bundle.exposure_gain,
        "ev": ev,
        "highlight": dg.highlight_mode_cn(highlight),
        "gamut": dg.output_gamut_label(gamut),
        "scene_transform": dg.scene_transform_label(scene_transform),
        "scene_transform_strength": scene_transform_strength,
        "tone_core": tone_core,
        "lum_norm": lum_norm,
        "decoder": str(getattr(proxy_bundle, "scene_decoder", decoder) or decoder),
        "decoder_version": getattr(proxy_bundle, "scene_decoder_version", None),
        "ev_auto": auto_ev_payload(auto_ev),
        "cache_hit": False,
        "pixel_cache_hit": pixel_cache_hit,
        "scene_histogram": scene_hist,
        "display_histogram": display_hist,
        "hdr_earned_ev": hdr_earned_ev(render_plan),
    }
    if auto_ev is None:
        cached.put_frame(frame_key, payload)
    return payload


def parse_grade(params: dict) -> tuple[str, float, str, float]:
    return resolve_grade_params(params)


def _finite_number(raw, name: str, lo: float, hi: float) -> float:
    """Range-checked FINITE float (review batch 17): Python's JSON decoder
    happily accepts NaN/Infinity literals, which would ride into exposure,
    auto-EV and plan compilation as silent poison."""
    import math

    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是数值")
    if not math.isfinite(v):
        raise ValueError(f"{name} 必须是有限数值")
    if not lo <= v <= hi:
        raise ValueError(f"{name} 须在 [{lo:g}, {hi:g}] 内")
    return v


def parse_film_params(params: dict) -> tuple:
    """(lens_filter, film_curve, film_mode, film_crossover, color_head_y,
    color_head_m, film_exposure_ev, film_print_timing)."""
    lens_filter = dg.validate_lens_filter(str(params.get("lensFilter", params.get("lens_filter", "none"))))
    film_curve = str(params.get("filmCurve", params.get("film_curve", "none")))
    if film_curve not in dg.FILM_CURVE_CHOICES:
        raise ValueError(f"未知曲线预设：{film_curve}")
    from dngscan.film_curve import film_process, validate_color_head_cc

    color_head_y = validate_color_head_cc(
        params.get("colorHeadY", params.get("color_head_y", 0.0)), "色头 Y"
    )
    color_head_m = validate_color_head_cc(
        params.get("colorHeadM", params.get("color_head_m", 0.0)), "色头 M"
    )
    if color_head_y > 0.0 or color_head_m > 0.0:
        # The enlarger colour head is a printing decision: it needs a print stage,
        # which only negative presets model (reversal film is its own display
        # medium). Reject rather than ignore — the GUI hides the control in those
        # states, so a nonzero value here is a direct-API contract violation.
        if film_curve == "none":
            raise ValueError("放大机色头需要负片胶片曲线预设（当前未选择胶片）")
        if film_process(film_curve) != "negative":
            raise ValueError(
                f"放大机色头仅对负片预设有效：{film_curve} 是反转片，无印相环节"
            )
    film_mode = str(params.get("filmMode", params.get("film_mode", "observe")))
    if film_mode not in ("observe", "full"):
        raise ValueError(f"未知胶片分工模式：{film_mode}")
    _timing_req = str(params.get("filmPrintTiming", params.get("film_print_timing", "fixed")) or "fixed")
    if film_mode == "full" and _timing_req != "custom" and (
        color_head_y > 0.0 or color_head_m > 0.0
    ):
        raise ValueError(
            "full 模式暂不支持放大机色头：接管核心是完整烘焙的光谱链，"
            "请切回 observe 或把色头归零"
        )
    tone_core_req = str(params.get("toneCore", params.get("tone_core", "agx")))
    if film_mode == "full" and tone_core_req != "agx":
        raise ValueError(
            "full 模式只在 AgX tone core 上运行；请把 tone 核切回 agx 或使用 observe"
        )
    neutral_req = params.get("filmNeutralization", params.get("film_neutralization"))
    crossover_req = params.get("filmCrossover", params.get("film_crossover"))
    if neutral_req is not None and crossover_req is not None:
        raise ValueError(
            "filmCrossover 已弃用为 filmNeutralization 的别名;两者不能同时给出"
        )
    if neutral_req is not None:
        mapping = {
            "technical-neutral": "off", "bounded": "off",
            "print-balanced": "print",
            "native": "datasheet", "datasheet": "datasheet",
        }
        if str(neutral_req) not in mapping:
            raise ValueError(f"未知灰阶中性化：{neutral_req}")
        film_crossover = mapping[str(neutral_req)]
    else:
        film_crossover = str(crossover_req if crossover_req is not None else "off")
        if film_crossover not in ("off", "print", "datasheet"):
            raise ValueError(f"未知层间漂移开关：{film_crossover}")
    film_exposure_ev = _finite_number(
        params.get("filmExposure", params.get("film_exposure_ev", 0.0)) or 0.0,
        "胶片曝光", -8.0, 8.0,
    )
    film_print_timing = str(params.get("filmPrintTiming", params.get("film_print_timing", "fixed")) or "fixed")
    if film_print_timing not in ("fixed", "retimed", "custom"):
        raise ValueError(f"未知印相 timing:{film_print_timing}")
    film_print_medium = str(params.get("filmPrintMedium", params.get("film_print_medium", "")) or "")
    film_print_exposure_ev = _finite_number(
        params.get("filmPrintExposure", params.get("film_print_exposure_ev", 0.0)) or 0.0,
        "印相曝光", -8.0, 8.0,
    )
    if film_mode != "full" and (
        film_exposure_ev != 0.0 or film_print_timing != "fixed"
        or film_print_medium != "" or film_print_exposure_ev != 0.0
    ):
        raise ValueError(
            "胶片曝光/印相 timing/介质/印相曝光属于接管显影(full 模式);GUI 在"
            "其他状态灰显/隐藏这些控件,非零载荷是直接 API 合同违规"
        )
    if film_print_timing == "custom" and film_crossover != "datasheet":
        raise ValueError(
            "custom timing 与有界灰阶中性化互斥:请配 filmNeutralization=datasheet"
        )
    if film_print_timing != "custom" and film_print_exposure_ev != 0.0:
        raise ValueError("手动印相曝光仅在 timing=custom 下有意义")
    # P5c (§11): 模拟光学 tiers. off/light/standard map to declared amount
    # triples; custom reads the three sliders. Non-full modes must carry no
    # optics payload (the GUI hides the control there).
    optics_tier = str(params.get("filmOptics", params.get("film_optics", "off")) or "off")
    tiers = {
        "off": (0.0, 0.0, 0.0),
        "light": (0.25, 0.2, 0.15),
        "standard": (0.5, 0.4, 0.3),
    }
    if optics_tier in tiers:
        film_grain, film_halation, film_bloom = tiers[optics_tier]
    elif optics_tier == "custom":
        film_grain = _finite_number(
            params.get("filmGrain", params.get("film_grain", 0.0)) or 0.0,
            "模拟光学颗粒", 0.0, 1.0,
        )
        film_halation = _finite_number(
            params.get("filmHalation", params.get("film_halation", 0.0)) or 0.0,
            "模拟光学halation", 0.0, 1.0,
        )
        film_bloom = _finite_number(
            params.get("filmBloom", params.get("film_bloom", 0.0)) or 0.0,
            "模拟光学bloom", 0.0, 1.0,
        )
    else:
        raise ValueError(f"未知模拟光学档位:{optics_tier}(可选 off/light/standard/custom)")
    if film_mode != "full" and (film_grain or film_halation or film_bloom):
        raise ValueError(
            "模拟光学属于接管显影(full 模式);GUI 在其他状态隐藏该控件,"
            "非零载荷是直接 API 合同违规"
        )
    # 胶片解释控件组 (appearance P1): interimage 底座开关与外观层模式。
    film_interimage = str(
        params.get("filmInterimage", params.get("film_interimage", "declared"))
        or "declared"
    )
    if film_interimage not in ("declared", "off"):
        raise ValueError(f"未知层间放大档:{film_interimage}(可选 declared/off)")
    film_appearance = str(
        params.get("filmAppearance", params.get("film_appearance", "technical"))
        or "technical"
    )
    if film_appearance not in ("technical", "reference"):
        raise ValueError(
            f"未知胶片解释:{film_appearance}(可选 technical/reference)"
        )
    film_appearance_strength = _finite_number(
        params.get(
            "filmAppearanceStrength",
            params.get("film_appearance_strength", 1.0),
        ) or 0.0,
        "参考印相强度", 0.0, 1.5,
    )
    if film_mode != "full" and (
        film_interimage != "declared" or film_appearance != "technical"
    ):
        raise ValueError("胶片解释控件属于接管显影(full 模式)")
    return (lens_filter, film_curve, film_mode, film_crossover, color_head_y,
            color_head_m, film_exposure_ev, film_print_timing,
            film_print_medium, film_print_exposure_ev,
            film_grain, film_halation, film_bloom,
            film_interimage, film_appearance, film_appearance_strength)


def effective_optics_seed(params: dict, entry) -> int:
    """Seed lifecycle (batch 15): an explicit integer in the payload always
    wins (reproducible); otherwise the loaded entry's one realization_id —
    the same value serves preview, probes, export and both HDR legs, so the
    grain a preview shows IS the grain the export carries."""
    raw = params.get("filmOpticsSeed", params.get("film_optics_seed"))
    if raw is not None and str(raw) != "" and str(raw) != "auto":
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError("filmOpticsSeed 需要整数或留空(auto)")
    return int(getattr(entry, "realization_id", 0) or 0)


def run_preview(params: dict) -> dict:
    inp, highlight, gamut, output_format, ev, _, quality, _, _, ev_auto = parse_job_params(params)
    try:
        generation = int(params.get("generation", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("generation 必须是整数") from exc
    session = str(params.get("previewSession", "") or f"legacy:{inp}")
    if not PREVIEW_COORDINATOR.register(session, generation):
        return {"ok": True, "superseded": True, "generation": generation}

    def is_current() -> bool:
        return PREVIEW_COORDINATOR.is_current(session, generation)

    wb = str(params.get("wb", "camera"))
    if wb not in dg.WB_CHOICES:
        raise ValueError(f"未知白平衡模式：{wb}")
    decoder, coreimage_version = parse_decoder(params)
    demosaic = parse_demosaic(params, decoder)
    if decoder == "coreimage":
        highlight = "reconstruct"
    look, look_strength, display_filter, filter_strength = parse_grade(params)
    scene_transform, scene_transform_strength = parse_scene_transform(params)
    punch_scale = parse_punch(params)
    adjustments = parse_render_adjustments(params)
    tone_core, lum_norm = parse_tone_core(params)
    agx_primaries = parse_agx_primaries(params)
    (lens_filter, film_curve, film_mode, film_crossover, color_head_y,
     color_head_m, film_exposure_ev, film_print_timing,
     film_print_medium, film_print_exposure_ev,
     film_grain, film_halation, film_bloom,
     film_interimage, film_appearance, film_appearance_strength,
     ) = parse_film_params(params)
    film_optics_seed = params.get("filmOpticsSeed", params.get("film_optics_seed"))
    film_optics_seed = (
        int(film_optics_seed)
        if film_optics_seed not in (None, "", "auto") else None
    )
    endpoint_mode = parse_endpoint_mode(params)
    try:
        cached = PREVIEW_STORE.get(
            inp,
            highlight,
            wb,
            tone_core == "gated",
            decoder,
            coreimage_version,
            demosaic,
        )
        if not is_current():
            raise PreviewSuperseded()
        if film_optics_seed is None:
            # resolve ONCE against the loaded entry (review batch 16: the
            # auto-EV branch received None and film plan compilation crashed
            # on int(None) whenever any film preset was active)
            film_optics_seed = int(getattr(cached, "realization_id", 0) or 0)
        auto_ev_result = None
        if ev_auto:
            auto_ev_result = dg.compute_auto_ev(
                cached.bundle,
                cached.analysis,
                gamut,
                look=look,
                look_strength=look_strength,
                display_filter=display_filter,
                filter_strength=filter_strength,
                scene_transform=scene_transform,
                scene_transform_strength=scene_transform_strength,
                punch_scale=punch_scale,
                tone_core=tone_core,
                lum_norm=lum_norm,
                agx_primaries=agx_primaries,
                adjustments=adjustments,
                endpoint_mode=endpoint_mode,
                film_curve=film_curve,
                film_mode=film_mode,
                film_crossover=film_crossover,
        film_exposure_ev=film_exposure_ev,
        film_print_timing=film_print_timing,
        film_print_medium=film_print_medium,
        film_print_exposure_ev=film_print_exposure_ev,
        film_grain=film_grain,
        film_halation=film_halation,
        film_bloom=film_bloom,
        film_interimage=film_interimage,
        film_appearance=film_appearance,
        film_appearance_strength=film_appearance_strength,
        film_optics_seed=film_optics_seed,
                color_head_y=color_head_y,
                color_head_m=color_head_m,
                lens_filter=lens_filter,
            )
            if not is_current():
                raise PreviewSuperseded()
            ev = auto_ev_result.ev
        result = export_preview_jpeg(
            inp,
            highlight,
            gamut,
            ev,
            min(quality, 95),
            wb=wb,
            look=look,
            look_strength=look_strength,
            display_filter=display_filter,
            filter_strength=filter_strength,
            scene_transform=scene_transform,
            scene_transform_strength=scene_transform_strength,
            auto_ev=auto_ev_result,
            punch_scale=punch_scale,
            tone_core=tone_core,
            lum_norm=lum_norm,
            agx_primaries=agx_primaries,
            cached=cached,
            adjustments=adjustments,
            decoder=decoder,
            coreimage_version=coreimage_version,
            demosaic=demosaic,
            lens_filter=lens_filter,
            film_curve=film_curve,
            film_mode=film_mode,
            film_crossover=film_crossover,
        film_exposure_ev=film_exposure_ev,
        film_print_timing=film_print_timing,
        film_print_medium=film_print_medium,
        film_print_exposure_ev=film_print_exposure_ev,
        film_grain=film_grain,
        film_halation=film_halation,
        film_bloom=film_bloom,
        film_interimage=film_interimage,
        film_appearance=film_appearance,
        film_appearance_strength=film_appearance_strength,
        film_optics_seed=film_optics_seed,
            endpoint_mode=endpoint_mode,
            color_head_y=color_head_y,
            color_head_m=color_head_m,
            include_metrics=bool(params.get("includeMetrics", True)),
            is_current=is_current,
        )
        result["generation"] = generation
        return result
    except PreviewSuperseded:
        return {"ok": True, "superseded": True, "generation": generation}


def _finite_or_none(value: object) -> float | None:
    """JSON-safe float: json.dumps emits bare NaN, which JSON.parse rejects."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def detected_scene_params(
    bundle: dg.RawBundle,
    analysis: dg.Analysis,
    plan: dg.RenderPlan | None = None,
) -> dict:
    """Measured scene facts that inform the user's later adjustments.

    Compiled from the same plan machinery the render will use, at the proxy decode's
    resolution. These are the numbers the pipeline itself consults — the reliable tail
    that budgets HDR, the clipping share that withdraws chroma freedom, the compiled
    curve endpoints — surfaced before any slider is touched.
    """
    plan = plan if plan is not None else dg.build_render_plan(bundle, analysis, RENDER_MODE, "p3")
    scene = plan.scene
    tone = plan.tone
    reliable_tail = _finite_or_none(getattr(scene, "reliable_tail_ev_p9999", None))
    earned = None
    if reliable_tail is not None:
        earned = max(0.0, reliable_tail - float(dg.OUTPUT_REFERENCE_WHITE_STOPS))
    # Compiled transition facts: the toe-end near-black crossing and the
    # shoulder-white near-white crossing, after every clamp — the two numbers the
    # offset sliders move, measured from the same params the render consumes. The
    # shoulder-start anchor stays reported as a plain compiled fact.
    toe_end_ev = shoulder_start_ev = shoulder_white_ev = None
    if getattr(tone, "tone_core", "agx") != "neutral":
        try:
            from dngscan.drt import compiled_curve_transitions

            transitions = compiled_curve_transitions(tone)
            toe_end_ev = _finite_or_none(transitions["toe_end_ev"])
            shoulder_start_ev = _finite_or_none(transitions["shoulder_start_ev"])
            shoulder_white_ev = _finite_or_none(transitions["shoulder_white_ev"])
        except Exception:
            pass
    return {
        "data_support": getattr(bundle, "camera_data_support", None),
        "wb_degradation": getattr(bundle, "wb_degradation", None),
        "raw_clip_union_pct": _finite_or_none(analysis.cell_union_pct),
        "reliable_tail_ev": reliable_tail,
        "tail_ev": _finite_or_none(getattr(scene, "tail_ev_p9999", None)),
        "body_median_ev": _finite_or_none(getattr(scene, "body_ev_p50", None)),
        "sparse_emitter": bool(getattr(scene, "sparse_emitter_tail", False)),
        "black_ev": _finite_or_none(tone.black_ev),
        "white_ev": _finite_or_none(tone.white_ev),
        "contrast": _finite_or_none(tone.contrast),
        "toe_end_ev": toe_end_ev,
        "shoulder_start_ev": shoulder_start_ev,
        "shoulder_white_ev": shoulder_white_ev,
        "endpoint_mode": str(getattr(tone, "endpoint_mode", "adaptive")),
        "endpoint_note": getattr(tone, "endpoint_note", None),
        "hdr_earned_ev": earned,
    }


def prepare_preview(params: dict) -> dict:
    """Warm the fixed proxy and current immutable base plan after selection."""
    inp, highlight, gamut, _, _, _, _, _, _, _ = parse_job_params(params)
    wb = str(params.get("wb", "camera"))
    if wb not in dg.WB_CHOICES:
        raise ValueError(f"未知白平衡模式：{wb}")
    decoder, coreimage_version = parse_decoder(params)
    demosaic = parse_demosaic(params, decoder)
    if decoder == "coreimage":
        highlight = "reconstruct"
    tone_core, lum_norm = parse_tone_core(params)
    scene_transform, scene_transform_strength = parse_scene_transform(params)
    punch_scale = parse_punch(params)
    adjustments = parse_render_adjustments(params)
    agx_primaries = parse_agx_primaries(params)
    (lens_filter, film_curve, film_mode, film_crossover, color_head_y,
     color_head_m, film_exposure_ev, film_print_timing,
     film_print_medium, film_print_exposure_ev,
     film_grain, film_halation, film_bloom,
     film_interimage, film_appearance, film_appearance_strength,
     ) = parse_film_params(params)
    film_optics_seed = params.get("filmOpticsSeed", params.get("film_optics_seed"))
    film_optics_seed = (
        int(film_optics_seed)
        if film_optics_seed not in (None, "", "auto") else None
    )
    endpoint_mode = parse_endpoint_mode(params)
    with SCHEDULER.slot("prepare"):
        entry = PREVIEW_STORE.get(
            inp,
            highlight,
            wb,
            tone_core == "gated",
            decoder,
            coreimage_version,
            demosaic,
        )
        proxy_bundle = entry.bundle
        if film_optics_seed is None:
            film_optics_seed = int(getattr(entry, "realization_id", 0) or 0)
        if lens_filter != "none":
            import dataclasses as _dc

            proxy_bundle = _dc.replace(proxy_bundle, lens_filter=lens_filter)
        # Compile the plan the automatic first frame will consume.  UI-only tone
        # adjustments are applied after this immutable base plan and stay sub-ms.
        detected_plan = _cached_render_plan(
            entry,
            proxy_bundle,
            gamut,
            scene_transform,
            scene_transform_strength,
            punch_scale,
            tone_core,
            lum_norm,
            agx_primaries,
            film_curve,
            adjustments,
            endpoint_mode,
            color_head_y,
            color_head_m,
            film_mode,
            film_crossover,
            film_exposure_ev,
            film_print_timing,
            film_print_medium,
            film_print_exposure_ev,
            film_grain,
            film_halation,
            film_bloom,
            film_optics_seed,
        )
    height, width = entry.bundle.scene_rec2020_render.shape[:2]
    try:
        detected = detected_scene_params(entry.bundle, entry.analysis, detected_plan)
    except Exception:
        # Detection is guidance, not a gate: a plan-compile failure here must not
        # block the preview session it decorates.
        detected = None
    return {
        "ok": True,
        "prepared": True,
        "width": int(width),
        "height": int(height),
        "decoder": str(getattr(entry.bundle, "scene_decoder", decoder) or decoder),
        "decoder_version": getattr(entry.bundle, "scene_decoder_version", None),
        "detected": detected,
    }


def export_suffix_parts(
    highlight: str,
    gamut: str,
    output_format: str,
    grade: str = "none",
    grade_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    endpoint_mode: str = "adaptive",
    color_head_y: float = 0.0,
    color_head_m: float = 0.0,
    film_mode: str = "observe",
    film_crossover: str = "off",
    film_exposure_ev: float = 0.0,
    film_print_timing: str = "fixed",
    film_print_medium: str = "",
    film_print_exposure_ev: float = 0.0,
    film_grain: float = 0.0,
    film_halation: float = 0.0,
    film_bloom: float = 0.0,
    film_interimage: str = "declared",
    film_appearance: str = "technical",
    film_appearance_strength: float = 1.0,
) -> str:
    """Build the filename stem suffix for GUI JPEG/PNG exports."""
    parts = [tone_core]
    if tone_core == "lum" and lum_norm != "y":
        parts.append(lum_norm)
    if tone_core == "agx" and agx_primaries != "base":
        parts.append(agx_primaries)
    if endpoint_mode != "adaptive":
        # Evidence endpoints change the compiled curve; the filename must not let an
        # evidence export silently overwrite the adaptive one.
        parts.append(endpoint_mode)
    if film_mode == "full":
        # Same overwrite rule: the film-takeover render (and its declared
        # crossover) must not silently replace an observe-mode export.
        parts.append("filmfull")
        if film_crossover == "datasheet":
            parts.append("xover")
        if float(film_exposure_ev) != 0.0:
            parts.append(
                ("fexp" + f"{float(film_exposure_ev):+.2f}")
                .replace("+", "p").replace("-", "m").replace(".", "_")
            )
        if film_print_timing == "retimed":
            parts.append("retimed")
        if film_print_timing == "custom":
            parts.append("customprint")
        if film_print_medium:
            parts.append(
                film_print_medium.replace("__", "-").replace("_", "")[:24]
            )
        if float(film_grain) or float(film_halation) or float(film_bloom):
            # An optics render must not silently overwrite the clean one.
            parts.append(
                ("optics" + f"-g{float(film_grain):g}h{float(film_halation):g}"
                 f"b{float(film_bloom):g}").replace(".", "_")
            )
    if highlight != "clip":
        parts.append(highlight)
    if gamut != "srgb":
        parts.append(gamut)
    if dg.is_hdr_output_format(output_format):
        parts.append("hdr_heic" if output_format == "ultrahdr-heic" else "hdr")
    if grade != "none":
        parts.append(grade.replace(":", "_"))
        if abs(float(grade_strength) - 1.0) > 1e-6:
            parts.append(f"gs{float(grade_strength):g}")
    if scene_transform != "none":
        parts.append(scene_transform)
        if abs(float(scene_transform_strength) - 1.0) > 1e-6:
            parts.append(f"st{float(scene_transform_strength):g}")
    if float(color_head_y) > 0.0 or float(color_head_m) > 0.0:
        # A filtered print must not silently overwrite the neutral one.
        parts.append(f"ch_y{float(color_head_y):g}m{float(color_head_m):g}")
    return "_".join(parts)


def _optics_budget_mib_for_fingerprint() -> int:
    from dngscan.render import _optics_budget_mib

    return _optics_budget_mib()


def export_plan_fingerprint(**params: object) -> str:
    """A stable short fingerprint of every render-affecting export parameter.

    The readable suffix names the headline choices, but it cannot carry all of
    them (film stock, lens filter, WB, EV, manual tone adjustments…) without
    becoming unusable — and any omission lets two different renders share a
    path and silently overwrite each other. The fingerprint closes that gap:
    identical parameters keep an identical name (re-exporting the same recipe
    intentionally replaces the file), any differing parameter changes it.
    """
    canonical = "\0".join(f"{key}={params[key]!r}" for key in sorted(params))
    import hashlib

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:6]


def _cached_full_analysis(
    inp: Path,
    highlight: str,
    wb: str,
    decoder: str,
    coreimage_version: str,
    demosaic: str,
) -> Any | None:
    """The preview session's persisted full-resolution Analysis, or None.

    The cache digest binds the file signature (path, mtime, size), the LibRaw
    runtime, the decode parameters, the system Core Image decoder build (when
    that decoder owns the pixels) and the cache schema version — recomputing
    it here at export time means a stale or foreign entry can never match. The
    stored analysis was computed with diagnostics off and the default (full)
    gamut set, a superset of any single-gamut export request, by the same
    analyze() on an identically-decoded bundle: reuse is exact by construction.
    """
    from . import preview_cache as pc

    # The WB-independent disk entry stores the fixed camera DecodeContext's full-res
    # analysis.  Interactive BalanceContexts refresh scene metrics on the proxy, but an
    # export must not mistake those for full-resolution percentiles.  Non-camera WB
    # therefore recomputes only at export until the exact fused full-res analysis cache
    # lands; the expensive RAW evidence/demosaic work is already gone from the WB stage.
    if wb != "camera":
        return None

    if decoder == "coreimage":
        # Mirror PreviewCache.get's parameter normalization for this decoder.
        highlight, demosaic = "reconstruct", "auto"
    try:
        _, digest = pc._cache_identity(
            Path(inp), highlight, wb, decoder, coreimage_version, demosaic
        )
        cache_path = pc._cache_dir() / f"{digest}.npz"
        if not cache_path.is_file():
            return None
        with dg.np.load(cache_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
        if int(metadata.get("version", -1)) != pc.PREVIEW_CACHE_VERSION:
            return None
        return pc._analysis_from_json(metadata["analysis"])
    except Exception:
        return None


def run_export(params: dict) -> dict:
    dg.require_dependencies()
    inp, highlight, gamut, output_format, ev, hdr_headroom, quality, want_png, outdir_arg, ev_auto = parse_job_params(
        params
    )
    if dg.is_hdr_output_format(output_format):
        available, reason = dg.apple_gainmap_backend_status()
        if not available:
            raise RuntimeError(reason)
    outdir = outdir_arg if outdir_arg is not None else inp.parent
    outdir.mkdir(parents=True, exist_ok=True)

    demosaic = str(params.get("demosaic", "auto"))
    chroma = str(params.get("chroma", "444"))
    delivery_name = str(params.get("deliveryProfile", params.get("delivery_profile", "archive")))
    try:
        delivery = dg.resolve_delivery_profile(
            delivery_name,
            quality=int(params["quality"]) if params.get("quality") is not None else None,
            chroma=chroma if params.get("chroma") is not None else None,
            container=dg.container_for_output_format(output_format),
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    # Profile owns the encode knobs once resolved (archive forces 100/444).
    quality = int(delivery.quality)
    chroma = str(delivery.chroma)
    if dg.is_hdr_output_format(output_format):
        # Mirror the CLI's honesty contract: the HDR container's primary-image
        # subsampling is emergent from quality inside Core Image, so a request the
        # encoder cannot honour must fail loudly, not write a contradicting file.
        # The page constrains its own controls; this guards direct API clients.
        if chroma == "422":
            raise ValueError(
                "HDR gain-map 容器不提供 4:2:2 主图采样；"
                "Core Image 按 quality 决定采样（q100→4:4:4，share→通常 4:2:0）"
            )
        if chroma == "444" and not delivery.is_archive:
            raise ValueError(
                "HDR 容器的 4:4:4 只在 q100（archive 档）下产生并被门禁验证；"
                "请改用 archive 交付档（或让色度采样跟随交付档）"
            )
    wb = str(params.get("wb", "camera"))
    if wb not in dg.WB_CHOICES:
        raise ValueError(f"未知白平衡模式：{wb}")
    decoder, coreimage_version = parse_decoder(params)
    if decoder == "coreimage":
        highlight = "reconstruct"
        demosaic = "auto"
    look, look_strength, display_filter, filter_strength = parse_grade(params)
    if dg.is_hdr_output_format(output_format) and (look != "none" or display_filter != "none"):
        raise RuntimeError(
            "Ultrahdr 第一版仅支持 look=none 与 display_filter=none；"
            "现有 display look/filter 尚未 HDR 化"
        )
    scene_transform, scene_transform_strength = parse_scene_transform(params)
    punch_scale = parse_punch(params)
    adjustments = parse_render_adjustments(params)
    if dg.is_hdr_output_format(output_format) and abs(float(adjustments.highlight_fade)) > 1e-9:
        raise RuntimeError("HDR 尚未定义显示侧高光褪白；请将该项恢复为自动")
    tone_core, lum_norm = parse_tone_core(params)
    if dg.is_hdr_output_format(output_format) and tone_core != "agx":
        raise RuntimeError("HDR 输出当前只实现 AgX tone core")
    agx_primaries = parse_agx_primaries(params)
    (lens_filter, film_curve, film_mode, film_crossover, color_head_y,
     color_head_m, film_exposure_ev, film_print_timing,
     film_print_medium, film_print_exposure_ev,
     film_grain, film_halation, film_bloom,
     film_interimage, film_appearance, film_appearance_strength,
     ) = parse_film_params(params)
    film_optics_seed = params.get("filmOpticsSeed", params.get("film_optics_seed"))
    film_optics_seed = (
        int(film_optics_seed)
        if film_optics_seed not in (None, "", "auto") else None
    )
    endpoint_mode = parse_endpoint_mode(params)
    if film_optics_seed is None:
        # the same realization the preview showed, when its entry is loaded;
        # a cold export (no preview session) mints its own
        entry = PREVIEW_STORE.peek(
            inp, highlight, wb, tone_core == "gated",
            decoder, coreimage_version, demosaic,
        ) if hasattr(PREVIEW_STORE, "peek") else None
        if entry is not None:
            film_optics_seed = int(getattr(entry, "realization_id", 0) or 0)
        else:
            import secrets

            film_optics_seed = secrets.randbits(32) | 1
    bundle = dg.load_raw(
        inp,
        highlight,
        demosaic=demosaic,
        wb_mode=wb,
        decoder=decoder,
        coreimage_version=coreimage_version,
    )
    bundle.lens_filter = lens_filter

    # /prepare already computed and persisted this exact full-resolution
    # Analysis (same decode identity incl. file signature and LibRaw build,
    # diagnostics off, all gamuts). Reuse is exact by construction; any miss or
    # doubt falls back to computing it here. The diagnostics dashboard needs
    # the y/ev images, so want_png always recomputes.
    analysis = None
    y = ev_img = None
    if not want_png:
        analysis = _cached_full_analysis(
            inp, highlight, wb, decoder, coreimage_version, demosaic
        )
    if analysis is None:
        analysis, y, ev_img = dg.analyze(
            bundle,
            4,
            diagnostics=want_png,
            gamut_names=None if want_png else (dg.output_gamut_space(gamut),),
        )
    auto_ev_result = None
    if ev_auto:
        auto_ev_result = dg.compute_auto_ev(
            bundle,
            analysis,
            gamut,
            look=look,
            look_strength=look_strength,
            display_filter=display_filter,
            filter_strength=filter_strength,
            scene_transform=scene_transform,
            scene_transform_strength=scene_transform_strength,
            punch_scale=punch_scale,
            tone_core=tone_core,
            lum_norm=lum_norm,
            agx_primaries=agx_primaries,
            adjustments=adjustments,
            endpoint_mode=endpoint_mode,
            film_curve=film_curve,
            film_mode=film_mode,
            film_crossover=film_crossover,
        film_exposure_ev=film_exposure_ev,
        film_print_timing=film_print_timing,
        film_print_medium=film_print_medium,
        film_print_exposure_ev=film_print_exposure_ev,
        film_grain=film_grain,
        film_halation=film_halation,
        film_bloom=film_bloom,
        film_interimage=film_interimage,
        film_appearance=film_appearance,
        film_appearance_strength=film_appearance_strength,
        film_optics_seed=film_optics_seed,
            color_head_y=color_head_y,
            color_head_m=color_head_m,
            lens_filter=lens_filter,
        )
        ev = auto_ev_result.ev
    bundle = dg.with_intent_exposure(bundle, user_ev=ev, tone_core=tone_core)
    render_plan = dg.build_render_plan(
        bundle,
        analysis,
        RENDER_MODE,
        gamut,
        scene_transform,
        scene_transform_strength,
        punch_scale,
        tone_core,
        lum_norm,
        agx_primaries=agx_primaries,
        adjustments=adjustments,
        film_curve=film_curve,
        film_mode=film_mode,
        film_crossover=film_crossover,
        film_exposure_ev=film_exposure_ev,
        film_print_timing=film_print_timing,
        film_print_medium=film_print_medium,
        film_print_exposure_ev=film_print_exposure_ev,
        film_grain=film_grain,
        film_halation=film_halation,
        film_bloom=film_bloom,
        film_interimage=film_interimage,
        film_appearance=film_appearance,
        film_appearance_strength=film_appearance_strength,
        film_optics_seed=film_optics_seed,
        endpoint_mode=endpoint_mode,
        color_head_y=color_head_y,
        color_head_m=color_head_m,
    )

    grade_id = str(params.get("grade", "none"))
    grade_strength = float(params.get("gradeStrength", params.get("grade_strength", 1.0)))
    suffix = export_suffix_parts(
        highlight,
        gamut,
        output_format,
        grade_id,
        grade_strength,
        scene_transform,
        scene_transform_strength,
        tone_core,
        lum_norm,
        agx_primaries,
        endpoint_mode,
        color_head_y=color_head_y,
        color_head_m=color_head_m,
        film_mode=film_mode,
        film_crossover=film_crossover,
        film_exposure_ev=film_exposure_ev,
        film_print_timing=film_print_timing,
        film_print_medium=film_print_medium,
        film_print_exposure_ev=film_print_exposure_ev,
        film_grain=film_grain,
        film_halation=film_halation,
        film_bloom=film_bloom,
        film_interimage=film_interimage,
        film_appearance=film_appearance,
        film_appearance_strength=film_appearance_strength,
        # NOT the seed: the readable stem names the headline choices, and a
        # random realization id would make every export a new filename. The
        # seed rides the fingerprint below, which is what prevents two
        # different renders from sharing a path (review batch 18 P0: passing
        # it here raised TypeError on EVERY GUI export).
    )
    fingerprint = export_plan_fingerprint(
        wb=wb,
        ev=float(ev),
        highlight=highlight,
        decoder=decoder,
        coreimage_version=coreimage_version,
        demosaic=demosaic,
        gamut=gamut,
        output_format=output_format,
        grade=grade_id,
        grade_strength=float(grade_strength),
        scene_transform=scene_transform,
        scene_transform_strength=float(scene_transform_strength),
        punch_scale=float(punch_scale),
        tone_core=tone_core,
        lum_norm=lum_norm,
        agx_primaries=agx_primaries,
        endpoint_mode=endpoint_mode,
        lens_filter=lens_filter,
        film_curve=film_curve,
        film_mode=film_mode,
        film_crossover=film_crossover,
        film_exposure_ev=film_exposure_ev,
        film_print_timing=film_print_timing,
        film_print_medium=film_print_medium,
        film_print_exposure_ev=film_print_exposure_ev,
        film_grain=film_grain,
        film_halation=film_halation,
        film_bloom=film_bloom,
        film_interimage=film_interimage,
        film_appearance=film_appearance,
        film_appearance_strength=film_appearance_strength,
        film_optics_seed=film_optics_seed,
        # The optics budget tier picks the spread-grid size since P3, so it
        # changes rendered bytes whenever any spatial amount is engaged. It
        # participates only then: an unused env var must not fork the names
        # of identical non-optics exports (same rule as hdr_headroom below).
        optics_budget_mib=(
            _optics_budget_mib_for_fingerprint()
            if (film_grain or film_halation or film_bloom) else 0
        ),
        color_head_y=float(color_head_y),
        color_head_m=float(color_head_m),
        adjustments=dataclasses.astuple(adjustments),
        # Encode-affecting parameters (review batch 11): the HDR headroom and
        # the delivery/encode knobs change the written bytes, so they must
        # change the name. Headroom only participates when the container is
        # HDR — an unused slider value must not fork identical SDR exports.
        hdr_headroom=(
            float(hdr_headroom) if dg.is_hdr_output_format(output_format) else 0.0
        ),
        delivery=delivery_name,
        quality=int(quality),
        chroma=str(chroma),
    )
    out_ext = ".heic" if output_format == "ultrahdr-heic" else ".jpg"
    out_path = outdir / f"{inp.stem}_{suffix}_p{fingerprint}{out_ext}"
    # Staged ownership (plan S4; GUI in batch 18, unconditional in batch 19):
    # the diagnostic dashboard is the LAST consumer of the analysis buffers,
    # so it runs FIRST and they are released for every path — previously a
    # png=1 export encoded its JPEG/HDR with xyz_render, y and ev_img still
    # resident.
    png_path = None
    png_temp = None
    if want_png:
        png_path = outdir / f"{inp.stem}_{suffix}_p{fingerprint}_scan.png"
        # The dashboard runs FIRST (it is the analysis buffers' last
        # consumer) but must not claim its final name until the main export
        # succeeds: a later ICC/HDR/write failure would otherwise leave a
        # PNG that looks finished — possibly pairing with an older JPEG from
        # a different render (review batch 20). Write a temp beside it and
        # rename atomically once the export is done.
        # Keep the .png suffix: matplotlib picks its writer from the
        # extension, so a ".part1234" tail made savefig raise (caught by a
        # real run — the mocked failure test could not see it).
        png_temp = png_path.with_name(
            f"{png_path.stem}.part{os.getpid()}{png_path.suffix}"
        )
        with SCHEDULER.slot("export"):
            dg.plot_dashboard(
                bundle, analysis, y, ev_img, png_temp, auto_ev=auto_ev_result
            )
    bundle = dg.release_analysis_buffers(bundle)
    y = ev_img = None
    try:
        with SCHEDULER.slot("export"):
            # Intent exposure already applied via with_intent_exposure above; do not
            # mutate a shared bundle in place.
            icc_profile = dg.output_icc_profile_bytes(gamut)
            export_result = dg.export_jpeg(
                path=inp,
                out_path=out_path,
                quality=quality,
                bundle=bundle,
                analysis=analysis,
                tone_plan=render_plan,
                output_gamut=gamut,
                output_format=output_format,
                hdr_headroom=hdr_headroom,
                subsampling=dg.chroma_to_subsampling(chroma),
                look=look,
                look_strength=look_strength,
                display_filter=display_filter,
                filter_strength=filter_strength,
                scene_transform=scene_transform,
                scene_transform_strength=scene_transform_strength,
                tone_core=tone_core,
                lum_norm=lum_norm,
                agx_primaries=agx_primaries,
                punch_scale=punch_scale,
                return_rgb=output_format == "sdr",
                delivery=delivery,
                chroma=chroma,
            )
            hdr_export_info = export_result if isinstance(export_result, dict) else None
            if hdr_export_info is not None and hdr_export_info.get("output_path"):
                # The writer corrects a container/suffix mismatch; report the real file.
                out_path = Path(str(hdr_export_info["output_path"]))
            if hdr_export_info is not None and out_path.is_file():
                hdr_export_info["file_size_bytes"] = out_path.stat().st_size
            rendered_u8 = export_result[1] if isinstance(export_result, tuple) else None
            if rendered_u8 is None and output_format == "ultrahdr-heic":
                from dngscan.gainmap import read_primary_rgb_u8

                rendered_u8 = read_primary_rgb_u8(out_path)
            if rendered_u8 is not None:
                metrics = output_luminance_metrics_u8(rendered_u8, gamut, ev)
            else:
                metrics = output_luminance_metrics(out_path, gamut, ev)
            metrics.update(
                estimate_ev_headroom(
                    bundle,
                    analysis,
                    gamut,
                    ev,
                    # B5: the probe's own bisection quantum is 1/128 EV; measured,
                    # 220k vs 600k samples land within one quantum of each other,
                    # so the function default (220k) is the declared operating point.
                    look=look,
                    look_strength=look_strength,
                    display_filter=display_filter,
                    filter_strength=filter_strength,
                    scene_transform=scene_transform,
                    scene_transform_strength=scene_transform_strength,
                    punch_scale=punch_scale,
                    tone_core=tone_core,
                    lum_norm=lum_norm,
                    agx_primaries=agx_primaries,
                    adjustments=adjustments,
                    endpoint_mode=endpoint_mode,
                    film_curve=film_curve,
                    film_mode=film_mode,
                    film_crossover=film_crossover,
            film_exposure_ev=film_exposure_ev,
            film_print_timing=film_print_timing,
            film_print_medium=film_print_medium,
            film_print_exposure_ev=film_print_exposure_ev,
            film_grain=film_grain,
            film_halation=film_halation,
            film_bloom=film_bloom,
            film_interimage=film_interimage,
            film_appearance=film_appearance,
            film_appearance_strength=film_appearance_strength,
            film_optics_seed=film_optics_seed,
                    color_head_y=color_head_y,
                    color_head_m=color_head_m,
                    lens_filter=lens_filter,
                )
            )
            preview = (
                preview_b64_from_u8(
                    rendered_u8, icc_profile=icc_profile, width=PROXY_LONG_EDGE
                )
                if rendered_u8 is not None
                else make_preview_b64(out_path, icc_profile=icc_profile)
            )
            if auto_ev_result is not None:
                np = dg.np
                if rendered_u8 is None:
                    from PIL import Image

                    with Image.open(out_path) as im:
                        rendered_u8 = np.asarray(im.convert("RGB"), dtype=np.uint8)
                annotated = annotate_preview_rgb_u8(
                    rendered_u8, dg.auto_ev_overlay_lines(auto_ev_result)
                )
                preview = preview_b64_from_u8(
                    annotated, icc_profile=icc_profile, width=PROXY_LONG_EDGE
                )
            saved = [str(out_path)]
            if png_temp is not None:
                # the main output is written: the dashboard may take its name
                os.replace(png_temp, png_path)
                png_temp = None
                saved.append(str(png_path))
    finally:
        # A half-finished dashboard temp must never survive a failed main
        # export (review batch 20); after a successful rename it is already
        # gone, so this is idempotent.
        if png_temp is not None:
            png_temp.unlink(missing_ok=True)

    return {
        "ok": True,
        "saved": saved,
        "preview": preview,
        "metrics": metrics,
        "metrics_kind": "full",
        "gain": bundle.exposure_gain,
        "ev": ev,
        "ev_auto": auto_ev_payload(auto_ev_result),
        "format": (
            "HDR gain-map HEIC"
            if output_format == "ultrahdr-heic"
            else "HDR gain-map JPEG"
            if dg.is_hdr_output_format(output_format)
            else "SDR JPEG"
        ),
        "hdr_headroom": hdr_headroom if dg.is_hdr_output_format(output_format) else 0.0,
        "hdr_diagnostics": (
            hdr_export_info.get("diagnostics") if hdr_export_info is not None else None
        ),
        "hdr_container": hdr_export_info,
        "highlight": dg.highlight_mode_cn(highlight),
        "gamut": dg.output_gamut_label(gamut),
        "scene_transform": dg.scene_transform_label(scene_transform),
        "scene_transform_strength": scene_transform_strength,
        "tone_core": tone_core,
        "lum_norm": lum_norm,
        "decoder": str(getattr(bundle, "scene_decoder", decoder) or decoder),
        "decoder_version": getattr(bundle, "scene_decoder_version", None),
    }


def _export_worker(params: dict, result_queue: Any) -> None:
    """Spawn target: keep full-resolution arrays out of the GUI server process."""
    try:
        result_queue.put(("ok", run_export(params)))
    except Exception as exc:
        maybe_print_exc()
        result_queue.put(("error", str(exc)))


EXPORT_TIMEOUT_SECONDS = float(os.environ.get("DNGSCAN_EXPORT_TIMEOUT", "600"))


def run_export_isolated(params: dict) -> dict:
    """Run one full export in a disposable process and return its small result payload.

    The process is deliberately short-lived: NumPy/libraw allocations then return to the
    OS after every export instead of accumulating in the long-running GUI process.
    """
    # Resolve the grain realization IN THE PARENT (review batch 18): the
    # spawned child gets a FRESH PREVIEW_STORE, so a child-side "auto" would
    # mint a brand-new seed and export grain the preview never showed. The
    # resolved integer travels in the payload; an explicit seed passes
    # through unchanged.
    params = dict(params)
    raw_seed = params.get("filmOpticsSeed", params.get("film_optics_seed"))
    if raw_seed in (None, "", "auto"):
        try:
            inp, highlight, _, _, _, _, _, _, _, _ = parse_job_params(params)
            wb = str(params.get("wb", "camera"))
            decoder, coreimage_version = parse_decoder(params)
            demosaic = parse_demosaic(params, decoder)
            tone_core, _ = parse_tone_core(params)
            entry = PREVIEW_STORE.peek(
                inp, "reconstruct" if decoder == "coreimage" else highlight,
                wb, tone_core == "gated", decoder, coreimage_version, demosaic,
            )
        except Exception:
            entry = None
        params["filmOpticsSeed"] = int(
            getattr(entry, "realization_id", 0) or secrets.randbits(32) | 1
        )
        params.pop("film_optics_seed", None)
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_export_worker,
        args=(dict(params), result_queue),
        name="dngscan-export",
    )
    message: tuple[str, Any] | None = None
    timed_out = False
    # Deadline (review batch 17): a wedged decoder/operator previously hung
    # this request forever WHILE HOLDING RENDER_LOCK, freezing the whole GUI.
    deadline = time.monotonic() + EXPORT_TIMEOUT_SECONDS
    with SCHEDULER.slot("export"):
        process.start()
        while message is None:
            try:
                message = result_queue.get(timeout=0.25)
            except Empty:
                if not process.is_alive():
                    break
                if time.monotonic() > deadline:
                    timed_out = True
                    process.terminate()
                    process.join(timeout=5.0)
                    if process.is_alive():
                        process.kill()
                    break
        process.join()
    try:
        result_queue.close()
        result_queue.join_thread()
    except (OSError, ValueError):
        pass
    if timed_out:
        raise RuntimeError(
            f"导出超时（>{EXPORT_TIMEOUT_SECONDS:.0f}s），工作进程已终止；"
            "若为超大文件可设 DNGSCAN_EXPORT_TIMEOUT 提高上限"
        )
    if message is None:
        raise RuntimeError(f"导出工作进程崩溃（exit code {process.exitcode}）")
    status, payload = message
    if status != "ok":
        raise RuntimeError(str(payload))
    return payload
