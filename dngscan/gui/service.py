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
import time
from pathlib import Path
from typing import Any, Callable

import dngscan as dg
from dngscan.debug_util import maybe_print_exc
from dngscan.grade import RENDER_MODE, resolve_grade_id, resolve_grade_params
from dngscan.calibration import calibration_diagnostics, calibration_fingerprint

from .constants import (
    PROXY_LONG_EDGE,
    RAW_EXTS,
    REALTIME_PREVIEW_JPEG_QUALITY,
    REALTIME_PREVIEW_JPEG_SUBSAMPLING,
)
from .histogram import display_histogram, hdr_earned_ev, scene_ev_base, scene_ev_histogram
from .preview_cache import PREVIEW_STORE, PreviewEntry
from .preview_scheduler import PREVIEW_COORDINATOR, PreviewSuperseded


# RENDER_LOCK retired (scheduler plan S2): concurrency is owned by the
# RenderScheduler's per-class slots — see dngscan/gui/scheduler.py.
from .scheduler import SCHEDULER


def _selection_check(params: dict) -> Callable[[], bool]:
    client = str(params.get("previewClient", "") or "")
    try:
        epoch = int(params.get("selectionEpoch", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("selectionEpoch 必须是整数") from exc
    accepted = PREVIEW_COORDINATOR.register_selection(client, epoch)
    return lambda: accepted and PREVIEW_COORDINATOR.selection_is_current(client, epoch)


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
    chroma_nr: float = 0.0,
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
        chroma_nr=chroma_nr,
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
        message = f"此文件不支持 Apple RAW 9，也没有可用的 RAW 8/7/6 降级路径（报告版本：{detail}）。"
    # GUI review 2026-08-27 item 5: the API can exist while the runtime
    # context cannot be built; surface both contexts so the page can grey
    # the decoder with the reason instead of failing at the first preview.
    runtime_interactive = runtime_export = None
    if probe["coreimage_available"]:
        try:
            runtime_interactive = bool(coreimage_decode.runtime_available(interactive=True))
            runtime_export = bool(coreimage_decode.runtime_available(interactive=False))
        except Exception:
            runtime_interactive = runtime_export = False
        if runtime_interactive is False:
            message = "此系统的 Core Image 运行时上下文不可用（API 存在但无法建立解码上下文）。"
    return {
        "ok": True,
        "support_lines": support_lines,
        "coreimage_available": bool(probe["coreimage_available"]),
        "runtime_interactive": runtime_interactive,
        "runtime_export": runtime_export,
        "raw9_supported": bool(probe["raw9_supported"]),
        "versions_offered": offered,
        "fallback_version": fallback,
        "probe_error": probe["error"],
        "message": message,
    }


CLIP_OVERLAY_THRESHOLD = 0.5
# What the layer IS (review R5 item 2): the render-time SOFT clip mask —
# raw_io builds it as smoothstep(0.95, 0.99) of the per-channel full-well
# fraction, feathered and resized to the proxy — thresholded at 0.5, i.e.
# the region at or above ~97% of full well where the pipeline's own chroma
# retreat / HDR chroma gates engage. It is NOT the hard clip statistic
# (raw >= fullwell - margin); that number comes from the full-resolution
# Analysis.clip_pct and is reported alongside as the authority. Marker
# colours per channel set: the near-full-well channels light up in their
# own colour, all three read white.
_CLIP_OVERLAY_ALPHA = 168


def _hard_clip_pct_by_colour(analysis: Any) -> dict[str, float] | None:
    """Full-resolution hard-clip share per CFA colour (R/G/B) from
    Analysis.clip_pct (raw >= fullwell - margin), averaging duplicate
    greens; plus the 2x2-cell union the detected-params card shows."""
    clip = getattr(analysis, "clip_pct", None)
    labels = getattr(analysis, "labels", None) or {}
    if not isinstance(clip, dict) or not clip:
        return None
    buckets: dict[str, list[float]] = {}
    for cid, pct in clip.items():
        letter = str(labels.get(cid, "")).strip()[:1].upper()
        if letter not in ("R", "G", "B"):
            continue
        buckets.setdefault(letter, []).append(float(pct))
    if not buckets:
        return None
    out = {k.lower(): (sum(v) / len(v)) for k, v in buckets.items()}
    # JSON has no NaN (Starlette encodes with allow_nan=False): a non-finite
    # union is "not measured", spelled None like detected_scene_params does.
    out["union"] = _finite_or_none(getattr(analysis, "cell_union_pct", None))
    return out


def clip_overlay_rgba(clip_masks: Any, threshold: float = CLIP_OVERLAY_THRESHOLD) -> Any:
    """RGBA u8 marker plane from per-channel CFA clip masks [h, w, 3] in [0, 1].

    The proxy masks are bilinear resizes of the half-resolution evidence masks,
    so a cell is "clipped" once its resized weight reaches ``threshold``.
    Returns None when no channel clips anywhere (the page then hides the
    layer without a decode)."""
    np = dg.np
    masks = np.asarray(clip_masks, dtype=np.float32)
    if masks.ndim != 3 or masks.shape[-1] != 3:
        raise ValueError(f"clip masks must be [h, w, 3], got {masks.shape}")
    hit = masks >= np.float32(threshold)
    if not bool(hit.any()):
        return None
    any_hit = hit.any(axis=-1)
    rgba = np.zeros(masks.shape[:2] + (4,), dtype=np.uint8)
    for c in range(3):
        rgba[..., c] = np.where(hit[..., c], 255, 0).astype(np.uint8)
    # a clipped pixel with only one or two channels saturated still gets a
    # visible base so single-channel red/green/blue reads as a marker, not
    # as image colour: dim the unclipped channels rather than zeroing them
    rgba[..., :3] = np.where(any_hit[..., None], np.maximum(rgba[..., :3], 48), 0).astype(np.uint8)
    rgba[..., 3] = np.where(any_hit, _CLIP_OVERLAY_ALPHA, 0).astype(np.uint8)
    return rgba


def clip_overlay(params: dict) -> dict:
    """RAW near-full-well layer for the preview: which proxy pixels sit at or
    above ~97% of full well in which CFA channel, drawn from the SAME soft
    evidence masks the render's clip retreat and the HDR chroma gates
    consume (see CLIP_OVERLAY_THRESHOLD). The hard clip share is a separate,
    full-resolution number (``hard_clip_pct``) and is what "over-exposed"
    means numerically. Evidence is decode-derived and WB-independent, so the
    layer is fetched once per prepared entry and composited client-side over
    every frame. Core Image decodes carry no aligned CFA masks; the page
    greys the toggle on ``has_masks`` false."""
    inp, highlight, _gamut, _fmt, _ev, _, _q, _png, _out, _auto = parse_job_params(params)
    wb = str(params.get("wb", "camera"))
    if wb not in dg.WB_CHOICES:
        raise ValueError(f"未知白平衡模式：{wb}")
    decoder, coreimage_version = parse_decoder(params)
    demosaic = parse_demosaic(params, decoder)
    coreimage_scale, clip_margin = parse_decode_extras(params, decoder)
    if decoder == "coreimage":
        highlight = "reconstruct"
    tone_core, _norm = parse_tone_core(params)
    cached = PREVIEW_STORE.get(
        inp, highlight, wb, tone_core == "gated", decoder, coreimage_version, demosaic,
        coreimage_scale=coreimage_scale, margin=clip_margin,
    )
    masks = getattr(cached.bundle, "clip_masks", None)
    hard = _hard_clip_pct_by_colour(getattr(cached, "analysis", None))
    if masks is None:
        return {"ok": True, "has_masks": False, "overlay": None, "mask_pct": None, "hard_clip_pct": hard}
    np = dg.np
    masks32 = np.asarray(masks, dtype=np.float32)
    hit = masks32 >= np.float32(CLIP_OVERLAY_THRESHOLD)
    total = float(max(1, hit.shape[0] * hit.shape[1]))
    mask_pct = {
        "r": 100.0 * float(hit[..., 0].sum()) / total,
        "g": 100.0 * float(hit[..., 1].sum()) / total,
        "b": 100.0 * float(hit[..., 2].sum()) / total,
        "any": 100.0 * float(hit.any(axis=-1).sum()) / total,
    }
    rgba = clip_overlay_rgba(masks32)
    overlay = None
    if rgba is not None:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
        overlay = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "ok": True,
        "has_masks": True,
        "overlay": overlay,
        "width": int(masks32.shape[1]),
        "height": int(masks32.shape[0]),
        # share of proxy pixels the LAYER marks (>= ~97% full well)
        "mask_pct": mask_pct,
        # the authoritative hard-clip share (full resolution, >= fullwell - margin)
        "hard_clip_pct": hard,
    }


def parse_job_params(params: dict) -> tuple[Path, str, str, str, float, float, int, bool, Path | None, bool]:
    obsolete = [key for key in params if key.startswith(("film", "colorHead", "color_head"))]
    if obsolete:
        raise ValueError("胶片功能已拆分到 AgXFilm；请刷新页面后重试。")
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
    quality = int(params.get("quality", 95))
    if not 1 <= quality <= 100:
        raise ValueError("质量需在 1-100 之间")
    want_png = bool(params.get("png", False))
    outdir = Path(str(params["outdir"])).expanduser() if params.get("outdir") else None
    ev_auto = bool(params.get("evAuto", "ev" not in params))
    return inp, highlight, gamut, output_format, ev, hdr_headroom, quality, want_png, outdir, ev_auto


def parse_punch(params: dict) -> float:
    """Fail closed like every sibling parser and the CLI (--punch 0..1.5):
    a non-numeric or out-of-range punch used to be silently substituted, so a
    payload of 3 rendered at 1.5 under a filename that did not say so
    (self-review 2026-08-27)."""
    raw = params.get("punch", 1.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"punch 需为数字：{raw!r}") from None
    if not (math.isfinite(value) and 0.0 <= value <= 1.5):
        raise ValueError(f"punch 需在 0..1.5 之间：{raw!r}")
    return value


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
        raise ValueError(f"未知黑白点依据：{mode}")
    return mode


def parse_scene_transform(params: dict) -> tuple[str, float]:
    transform = dg.validate_scene_transform(str(params.get("sceneTransform", "none")))

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


def reject_gated_coreimage(tone_core: str, decoder: str) -> None:
    """CLI contract parity (R2 item 2): gated means "per-pixel CFA evidence
    gates the colour path", and the Core Image pipeline has no aligned mask —
    the combination is meaningless rather than merely degraded, so the
    service refuses it exactly like the CLI instead of letting gated decay
    to raw_permission≈0 silently. The GUI hides the pair; a payload carrying
    it is a direct-API contract violation."""
    if str(decoder) == "coreimage" and str(tone_core) == "gated":
        raise ValueError(
            "toneCore=gated 需要逐像素 CFA 证据,而 decoder=coreimage 是独立"
            "管线(Core Image 执行 DNG opcode,几何与 LibRaw 不可对齐);"
            "请改用 agx/lum/neutral 或切回 libraw"
        )


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
    adjustments: dg.RenderAdjustments | None,
    endpoint_mode: str = "adaptive",
    *,
    chroma_nr: float = 0.0,
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
        str(getattr(bundle, "lens_filter", "none")),
        endpoint_mode,
        _cache_float(chroma_nr),
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
            chroma_nr=chroma_nr,
            adjustments=None,
            endpoint_mode=endpoint_mode,
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
    adjustments: dg.RenderAdjustments | None,
    endpoint_mode: str = "adaptive",
    *,
    chroma_nr: float = 0.0,
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
        # review batch 23: the chroma-NR dial changes rendered bytes
        _cache_float(chroma_nr),
        endpoint_mode,
        _adjustment_key(adjustments),
        # Exposure is represented by ``ev`` above; the scale contract guards against
        # accidentally sharing frames across decoder/cache versions.
        _cache_float(getattr(bundle, "scene_scale", 1.0)),
        str(getattr(bundle, "scene_decoder_runtime", "") or ""),
        (_spatial_budget_mib_for_fingerprint() if float(chroma_nr) > 0.0 else 0),
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
    # A11 item 1: no runtime gate HERE. parse_decoder serves preview
    # (interactive context) and export (export context) alike, and gating
    # both on the export probe wrongly refused previews on hosts where
    # only the interactive context works. load_raw prechecks the ACTUAL
    # workload (runtime_available(interactive=scene_half_size)) at the
    # decode entry — the single point that knows which context runs.
    if decoder == "coreimage" and not coreimage_decode.available():
        raise RuntimeError(
            "Core Image 解码器在此系统不可用（需要 macOS + PyObjC Quartz）"
        )
    # A9 item 4: the RAW9 daylight rejection is GONE — raw_io implements
    # the project hot-WB on top of the fixed AsShot decode (the transport
    # matrix composes decode->applied with the daylight frame's inverse),
    # the CLI allows it and tests pin it. One behaviour across GUI, CLI
    # and the Python API.
    return decoder, version


def parse_lens_filter(params: dict) -> str:
    from dngscan.lens_filter import validate_lens_filter

    return validate_lens_filter(str(params.get("lensFilter", params.get("lens_filter", "none"))))


def parse_chroma_nr(params: dict, output_format: str) -> float:
    """Parse calibrated scene-stage chroma smoothing shared by SDR and HDR."""
    del output_format
    raw = params.get("chromaNr", params.get("chroma_nr", 0.0))
    value = _finite_number(raw if raw not in (None, "") else 0.0, "色度降噪", 0.0, 1.0)
    return float(value)


def parse_decode_extras(params: dict, decoder: str) -> tuple[str, int]:
    """(coreimage_scale, clip_margin) — the two CLI decode dials the GUI now
    exposes (owner 2026-08-28). The scale policy only exists on the Core
    Image decoder (LibRaw silently reads "aligned" so the cache identity
    stays the default); the clip margin is the per-channel full-well
    threshold back-off in DN that the CLI calls --margin."""
    from dngscan.constants import COREIMAGE_SCALE_CHOICES, COREIMAGE_SCALE_DEFAULT_MODE

    scale = str(params.get("coreimageScale", params.get("coreimage_scale", COREIMAGE_SCALE_DEFAULT_MODE)) or COREIMAGE_SCALE_DEFAULT_MODE)
    if scale not in COREIMAGE_SCALE_CHOICES:
        raise ValueError(f"未知 Core Image 尺度策略：{scale}（可选 {'/'.join(COREIMAGE_SCALE_CHOICES)}）")
    if decoder != "coreimage":
        scale = COREIMAGE_SCALE_DEFAULT_MODE
    raw = params.get("clipMargin", params.get("margin", 4))
    try:
        as_float = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"clipMargin 需为整数 DN：{raw!r}") from None
    # R7 item 5: the CLI's type=int rejects "4.9"; int() here silently
    # truncated it — same entry, same verdict.
    if not (math.isfinite(as_float) and as_float == int(as_float)):
        raise ValueError(f"clipMargin 需为整数 DN：{raw!r}")
    margin = int(as_float)
    if not 0 <= margin <= 64:
        raise ValueError(f"clipMargin 域为 [0, 64] DN：{raw!r}")
    return scale, margin


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
    endpoint_mode: str = "adaptive",
    include_metrics: bool = True,
    is_current: Callable[[], bool] | None = None,
    coreimage_scale: str = "aligned",
    clip_margin: int = 4,
    chroma_nr: float = 0.0,
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
            coreimage_scale=coreimage_scale,
            margin=clip_margin,
        )

    def ensure_current() -> None:
        if is_current is not None and not is_current():
            raise PreviewSuperseded()

    ensure_current()

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
        adjustments,
        endpoint_mode,
        chroma_nr=chroma_nr,
    )
    frame_key = _preview_frame_key(pixel_key, include_metrics)
    frame_key = (frame_key, tuple(auto_ev_payload(auto_ev).values()) if auto_ev is not None else None)
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
        rgb_u8 = cached.get_pixels(pixel_key)
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
            adjustments,
            endpoint_mode,
            chroma_nr=chroma_nr,
        )
        # A shared plan may have returned this slot while another owner built
        # it; reacquiring admission is another latest-selection boundary.
        ensure_current()
        if rgb_u8 is None:
            rgb_u8 = dg.render_output_u8(
                proxy_bundle, cached.analysis, gamut, render_plan,
                look, look_strength, display_filter, filter_strength,
                scene_transform, scene_transform_strength,
                tone_core, lum_norm, agx_primaries,
                dither_noise=cached.get_or_build_dither_noise(),
            )
            ensure_current()
            rgb_u8 = cached.put_pixels(pixel_key, rgb_u8, _take_ownership=True,
                                       report={"status": getattr(proxy_bundle, "chroma_nr_status", "disabled"),
                                               "reason": getattr(proxy_bundle, "chroma_nr_reason", None)})
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
        ensure_current()
        scene_hist = scene_ev_histogram(scene_hist_base, render_plan, ev)
        display_hist = display_histogram(rgb_u8)
        # R4: metrics read the RENDERED frame — measuring after the auto-EV
        # overlay counted the annotation box's own near-white text as clipped
        # highlights, on exactly the path whose metrics judge headroom.
        metrics = preview_metrics_from_u8(rgb_u8, gamut) if include_metrics else {}
        if auto_ev is not None and include_metrics:
            rgb_u8 = annotate_preview_rgb_u8(rgb_u8, dg.auto_ev_overlay_lines(auto_ev))
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
        "chroma_nr": cached.get_pixel_report(pixel_key),
        "chroma_nr_capability": chroma_nr_capability(
            proxy_bundle, cached.analysis, scene_transform=scene_transform,
            scene_transform_strength=scene_transform_strength,
        ),
    }
    cached.put_frame(frame_key, payload)
    return payload


def parse_hdr_dials(params: dict, output_format: str) -> tuple:
    """HDR latitude dials (owner decision 2026-08-14, taste-to-dial): None =
    the registered policy defaults, byte-identical; explicit values are the
    user's latitude and are refused under SDR (a dial that silently does
    nothing teaches the user it is broken). Evidence gates are not dials."""
    out = []
    for key, alt, name, lo, hi in (
        ("hdrRho", "hdr_rho", "HDR 高光保色基准", 0.0, 1.0),
        ("hdrWhiteMargin", "hdr_white_margin", "HDR 白点留量", 0.0, 2.0),
        ("hdrShoulderStart", "hdr_shoulder_start", "HDR 高光压缩起点", -1.0, 3.0),
    ):
        raw = params.get(key, params.get(alt))
        if raw in (None, "", "auto"):
            out.append(None)
            continue
        val = _finite_number(raw, name, lo, hi)
        if not dg.is_hdr_output_format(output_format):
            raise ValueError(
                f"{key} 属于 HDR 编码(ultrahdr/ultrahdr-heic);SDR 输出没有"
                " HDR 肩部/色度模型"
            )
        out.append(float(val))
    return tuple(out)


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


def run_preview(params: dict) -> dict:
    inp, highlight, gamut, output_format, ev, _, quality, _, _, ev_auto = parse_job_params(params)
    try:
        generation = int(params.get("generation", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("generation 必须是整数") from exc
    selection_current = _selection_check(params)
    session = str(params.get("previewSession", "") or f"legacy:{inp}")
    if not selection_current() or not PREVIEW_COORDINATOR.register(session, generation):
        return {"ok": True, "superseded": True, "generation": generation}

    def is_current() -> bool:
        return selection_current() and PREVIEW_COORDINATOR.is_current(session, generation)

    wb = str(params.get("wb", "camera"))
    if wb not in dg.WB_CHOICES:
        raise ValueError(f"未知白平衡模式：{wb}")
    decoder, coreimage_version = parse_decoder(params)
    demosaic = parse_demosaic(params, decoder)
    coreimage_scale, clip_margin = parse_decode_extras(params, decoder)
    if decoder == "coreimage":
        highlight = "reconstruct"
    look, look_strength, display_filter, filter_strength = parse_grade(params)
    scene_transform, scene_transform_strength = parse_scene_transform(params)
    punch_scale = parse_punch(params)
    adjustments = parse_render_adjustments(params)
    tone_core, lum_norm = parse_tone_core(params)
    reject_gated_coreimage(tone_core, decoder)
    agx_primaries = parse_agx_primaries(params)
    lens_filter = parse_lens_filter(params)
    endpoint_mode = parse_endpoint_mode(params)
    chroma_nr = parse_chroma_nr(params, output_format)
    try:
        cached = PREVIEW_STORE.get(
            inp,
            highlight,
            wb,
            tone_core == "gated",
            decoder,
            coreimage_version,
            demosaic,
            coreimage_scale=coreimage_scale,
            margin=clip_margin,
            _is_current=is_current,
        )
        if not is_current():
            raise PreviewSuperseded()
        auto_ev_result = None
        if ev_auto:
            auto_options = dict(
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
                lens_filter=lens_filter,
                chroma_nr=chroma_nr,
            )
            # Cache actual normalized inputs, not request spelling/UI metadata.
            auto_key = json.dumps(
                {"gamut": gamut, "output_format": output_format, **auto_options},
                sort_keys=True, default=lambda value: dataclasses.asdict(value),
            )
            auto_ev_result = cached.get_or_build_auto_ev(
                auto_key, lambda: dg.compute_auto_ev(
                    cached.bundle, cached.analysis, gamut, **auto_options
                ),
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
            endpoint_mode=endpoint_mode,
            include_metrics=bool(params.get("includeMetrics", True)),
            is_current=is_current,
            coreimage_scale=coreimage_scale,
            clip_margin=clip_margin,
            chroma_nr=chroma_nr,
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


def chroma_nr_capability(
    bundle: dg.RawBundle,
    analysis: dg.Analysis,
    *,
    scene_transform: str = "none",
    scene_transform_strength: float = 0.0,
) -> dict:
    """Current render prerequisites, independent of the last NR request.

    Passing this static check permits an approximate attempt. The eventual
    render report still decides whether numerical/spatial propagation worked.
    No image texture is measured and no noise/correction map is allocated.
    """
    from ..noise_propagation import chroma_nr_skip_reason
    from ..spatial import chroma_nr_grid_shape

    model = getattr(analysis, "noise_model", None) or getattr(bundle, "noise_model", None)
    shape = getattr(getattr(bundle, "scene_rec2020_render", None), "shape", ())
    coarse_shape = chroma_nr_grid_shape(bundle, *shape[:2]) if len(shape) >= 2 else None
    reason = chroma_nr_skip_reason(
        bundle, model, coarse_shape, scene_transform=scene_transform,
        scene_transform_strength=scene_transform_strength,
    )
    return {"available": reason is None,
            "status": "available" if reason is None else "unavailable",
            "reason": reason, "approximate": True}


def detected_scene_params(
    bundle: dg.RawBundle,
    analysis: dg.Analysis,
    plan: dg.RenderPlan | None = None,
    *,
    scene_transform: str = "none",
    scene_transform_strength: float = 0.0,
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
    from ..hdr_agx_plan import scene_headroom_ev
    from ..report import processing_evidence_summary
    earned = scene_headroom_ev(scene)
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
    noise_model = getattr(analysis, "noise_model", None)
    noise = {
        "status": getattr(noise_model, "status", "unavailable"),
        "source": getattr(noise_model, "source", None),
        "reason": getattr(noise_model, "reason", None),
        "domain": getattr(noise_model, "domain", None),
        "approximation": getattr(noise_model, "approximation", None),
        "correlation": getattr(noise_model, "correlation", "unknown"),
        "spectral_ratios": getattr(noise_model, "spectral_ratios", {}),
        "noise_reduction_status": getattr(noise_model, "noise_reduction_status", "absent"),
        "fallback_source": getattr(noise_model, "fallback_source", None),
        "fallback_reason": getattr(noise_model, "fallback_reason", None),
        "evidence_status": getattr(analysis, "noise_evidence_status", None),
    }
    calibration_status = calibration_diagnostics(
        getattr(bundle, "shot_make", None), getattr(bundle, "shot_model", None),
        shutter=getattr(bundle, "shot_shutter", None), iso=getattr(bundle, "shot_iso", None),
        readout=getattr(bundle, "capture_readout", None),
    )
    return {
        "data_support": getattr(bundle, "camera_data_support", None),
        "decoder_actual": f"{bundle.scene_decoder} {bundle.scene_decoder_version or ''}".strip(),
        "decoder_fallback": bundle.scene_decoder_fallback,
        "demosaic_actual": ((getattr(bundle, "noise_decode", None) or {}).get("demosaic_algorithm")
                            if bundle.scene_decoder == "libraw" else None),
        "capture_readout": dict(getattr(bundle, "capture_readout", None) or {}),
        "evidence_provider": bundle.evidence_provider,
        "evidence_error": bundle.evidence_error,
        "reliability_source": scene.reliability_source,
        "reference_error": bundle.scene_reference_error,
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
        "chroma_nr": {"status": getattr(bundle, "chroma_nr_status", "disabled"),
                      "reason": getattr(bundle, "chroma_nr_reason", None)},
        "chroma_nr_capability": chroma_nr_capability(
            bundle, analysis, scene_transform=scene_transform,
            scene_transform_strength=scene_transform_strength,
        ),
        "noise_model": noise,
        "processing_evidence": processing_evidence_summary(bundle, analysis, scene),
        "calibrations": calibration_status,
    }


def prepare_preview(params: dict) -> dict:
    is_current = _selection_check(params)
    try:
        if not is_current():
            raise PreviewSuperseded()
        with SCHEDULER.slot("prepare"):
            if not is_current():
                raise PreviewSuperseded()
            return _prepare_preview_current(params, is_current)
    except PreviewSuperseded:
        SCHEDULER.note_dropped()
        return {"ok": True, "superseded": True, "prepared": False}


def _prepare_preview_current(params: dict, is_current: Callable[[], bool]) -> dict:
    """Warm the fixed proxy and current immutable base plan after selection."""
    inp, highlight, gamut, output_format, _, _, _, _, _, _ = parse_job_params(params)
    wb = str(params.get("wb", "camera"))
    if wb not in dg.WB_CHOICES:
        raise ValueError(f"未知白平衡模式：{wb}")
    decoder, coreimage_version = parse_decoder(params)
    demosaic = parse_demosaic(params, decoder)
    coreimage_scale, clip_margin = parse_decode_extras(params, decoder)
    if decoder == "coreimage":
        highlight = "reconstruct"
    tone_core, lum_norm = parse_tone_core(params)
    reject_gated_coreimage(tone_core, decoder)
    scene_transform, scene_transform_strength = parse_scene_transform(params)
    punch_scale = parse_punch(params)
    adjustments = parse_render_adjustments(params)
    agx_primaries = parse_agx_primaries(params)
    lens_filter = parse_lens_filter(params)
    endpoint_mode = parse_endpoint_mode(params)
    chroma_nr = parse_chroma_nr(params, output_format)
    if not is_current():
        raise PreviewSuperseded()
    entry = PREVIEW_STORE.get(
        inp,
        highlight,
        wb,
        tone_core == "gated",
        decoder,
        coreimage_version,
        demosaic,
        coreimage_scale=coreimage_scale,
        margin=clip_margin,
        _is_current=is_current,
    )
    if not is_current():
        raise PreviewSuperseded()
    proxy_bundle = entry.bundle
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
        adjustments,
        endpoint_mode,
        chroma_nr=chroma_nr,
    )
    if not is_current():
        raise PreviewSuperseded()
    height, width = entry.bundle.scene_rec2020_render.shape[:2]
    try:
        detected = detected_scene_params(
            proxy_bundle, entry.analysis, detected_plan,
            scene_transform=scene_transform,
            scene_transform_strength=scene_transform_strength,
        )
    except Exception:
        # Detection is guidance, not a gate: a plan-compile failure here must not
        # block the preview session it decorates.
        detected = None
    if not is_current():
        raise PreviewSuperseded()
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
    coreimage_scale: str = "aligned",
    clip_margin: int = 4,
    decoder: str = "libraw",
    chroma_nr: float = 0.0,
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
    if str(decoder) == "coreimage" and str(coreimage_scale) != "aligned":
        parts.append(f"ciscale-{coreimage_scale}")
    if int(clip_margin) != 4:
        parts.append(f"margin{int(clip_margin)}")
    if float(chroma_nr) > 0.0:
        # a repaired render must not silently overwrite the untouched one
        parts.append(f"cnr{float(chroma_nr):g}".replace(".", "_"))
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
    return "_".join(parts)


def _spatial_budget_mib_for_fingerprint() -> int:
    from dngscan.spatial import spatial_budget_mib

    return spatial_budget_mib()


def export_plan_fingerprint(**params: object) -> str:
    """A stable short fingerprint of every render-affecting export parameter.

    The readable suffix names the headline choices, but it cannot carry all of
    them (lens filter, WB, EV, manual tone adjustments…) without
    becoming unusable — and any omission lets two different renders share a
    path and silently overwrite each other. The fingerprint closes that gap:
    identical parameters keep an identical name (re-exporting the same recipe
    intentionally replaces the file), any differing parameter changes it.

    Review batch 21: the INPUT is one of those parameters — the output name
    only carries the stem, so two different RAWs sharing a stem (memory-card
    counter resets are routine) exported with the same recipe used to
    overwrite each other; callers pass the resolved path and file size. Also
    widened 6 -> 12 hex: 24 bits reaches birthday-collision territory within
    a few thousand recipes, and a collision here IS a silent overwrite.
    """
    canonical = "\0".join(f"{key}={params[key]!r}" for key in sorted(params))
    import hashlib

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _cached_full_analysis(
    inp: Path,
    highlight: str,
    wb: str,
    decoder: str,
    coreimage_version: str,
    demosaic: str,
    coreimage_scale: str = "aligned",
    margin: int = 4,
    decoded_bundle: Any | None = None,
    envelope: str | None = None,
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
            Path(inp), highlight, wb, decoder, coreimage_version, demosaic,
            coreimage_scale if decoder == "coreimage" else "aligned", int(margin),
        )
        if envelope is not None and decoded_bundle is not None:
            from dngscan.fast_plan import NATIVE_ABI_VERSION
            try:
                data = json.loads(envelope)
                if (data.get("schema") == 1 and data.get("cache_version") == pc.PREVIEW_CACHE_VERSION
                        and data.get("native_abi") == NATIVE_ABI_VERSION
                        and data.get("digest") == digest
                        and data.get("source_bundle") == pc._bundle_metadata(decoded_bundle)):
                    return pc._analysis_from_json(data["analysis"])
            except (TypeError, ValueError, KeyError):
                pass
        cache_path = pc._cache_dir() / f"{digest}.npz"
        if not cache_path.is_file():
            return None
        with dg.np.load(cache_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
        if int(metadata.get("version", -1)) != pc.PREVIEW_CACHE_VERSION:
            return None
        if decoded_bundle is not None:
            # Runtime fallback is per attempt. A cache requested as Apple auto
            # may contain pixels from RAW 8 or LibRaw, or lack sensor evidence.
            current = pc._bundle_metadata(decoded_bundle)
            if metadata.get("source_bundle") != current:
                return None
        return pc._analysis_from_json(metadata["analysis"])
    except Exception:
        return None


def _preview_analysis_envelope(entry: PreviewEntry) -> str | None:
    """Small private snapshot; no proxy pixels cross the process boundary."""
    from . import preview_cache as pc
    from dngscan.fast_plan import NATIVE_ABI_VERSION
    if not isinstance(entry, PreviewEntry) or not entry.cache_digest or entry.source_metadata is None:
        return None
    return json.dumps({"schema": 1, "cache_version": pc.PREVIEW_CACHE_VERSION,
                       "native_abi": NATIVE_ABI_VERSION, "digest": entry.cache_digest,
                       "source_bundle": entry.source_metadata,
                       "analysis": pc._analysis_to_json(entry.analysis)}, allow_nan=True)


def _preview_decode_contract(bundle):
    fields = ("scene_decoder", "scene_decoder_version", "evidence_provider",
              "scene_reliability_source", "scene_scale", "scene_decoder_fallback")
    return {field: getattr(bundle, field) for field in fields}


def _load_export_scene(inp, highlight, wb, decoder, version, demosaic, scale, preview=None,
                       analysis_luminance_only=False):
    """Pin a displayed auto result; export must not silently switch algorithms."""
    chosen_decoder, chosen_version = decoder, version
    pinned = preview is not None and decoder == "coreimage" and version == "auto"
    if pinned:
        chosen_decoder = preview.scene_decoder
        if chosen_decoder == "coreimage":
            chosen_version = str(preview.scene_decoder_version)
    bundle = dg.load_raw(inp, highlight, demosaic=demosaic, wb_mode=wb,
                         decoder=chosen_decoder, coreimage_version=chosen_version,
                         coreimage_scale=scale, _defer_clip_masks=True,
                         _analysis_luminance_only=analysis_luminance_only)
    if pinned:
        fields = ("scene_decoder", "scene_decoder_version", "evidence_provider",
                  "scene_reliability_source", "scene_scale")
        if any(getattr(preview, field) != getattr(bundle, field) for field in fields):
            raise RuntimeError("解码或证据能力与已显示预览不同，请刷新预览后再导出")
        bundle.scene_decoder_fallback = preview.scene_decoder_fallback
    return bundle


def run_export(params: dict) -> dict:
    calibration_generation = calibration_fingerprint()
    inp, highlight, gamut, output_format, ev, hdr_headroom, quality, want_png, outdir_arg, ev_auto = parse_job_params(
        params
    )
    # The dashboard PNG needs matplotlib (an optional extra): gate it HERE,
    # before the full-resolution analysis, instead of failing inside
    # plot_dashboard after the JPEG's work was already done and lost.
    dg.require_dependencies(dashboard=bool(want_png))
    if dg.is_hdr_output_format(output_format):
        available, reason = dg.apple_gainmap_backend_status()
        if not available:
            raise RuntimeError(reason)
    outdir = outdir_arg if outdir_arg is not None else inp.parent
    outdir.mkdir(parents=True, exist_ok=True)

    chroma = str(params.get("chroma", "444"))
    hdr_rho, hdr_white_margin, hdr_shoulder_start = parse_hdr_dials(
        params, output_format
    )
    delivery_name = str(params.get("deliveryProfile", params.get("delivery_profile", "auto")))
    if "deliveryProfile" not in params and "delivery_profile" not in params and (
        params.get("quality") is not None or params.get("chroma") is not None
    ):
        # Keep explicit legacy API encode knobs manual, just like the CLI.
        delivery_name = "archive" if quality == 100 and chroma == "444" else "share"
    try:
        delivery = dg.resolve_delivery_profile(
            delivery_name,
            quality=int(params["quality"]) if params.get("quality") is not None else None,
            chroma=chroma if params.get("chroma") is not None else None,
            container=dg.container_for_output_format(output_format),
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if dg.is_hdr_output_format(output_format):
        delivery = dg.resolve_hdr_chroma(
            delivery, explicit_chroma=params.get("chroma")
        )
    from dataclasses import replace
    delivery = replace(delivery,
        heif_encoder=str(params.get("heifEncoder", "auto")),
        heif_bit_depth=int(params.get("heifBitDepth", 10)),
        heif_preset=str(params.get("heifPreset", "slow")),
        heif_tune=str(params.get("heifTune", "ssim")))
    quality = int(delivery.quality)
    chroma = str(delivery.chroma)
    wb = str(params.get("wb", "camera"))
    if wb not in dg.WB_CHOICES:
        raise ValueError(f"未知白平衡模式：{wb}")
    decoder, coreimage_version = parse_decoder(params)
    coreimage_scale, clip_margin = parse_decode_extras(params, decoder)
    # R4: same validator as the preview path — a bogus demosaic used to fall
    # through to load_raw's silent auto preference while the fingerprint
    # baked the bogus string into the filename (parse_demosaic also resolves
    # the coreimage-forced "auto").
    demosaic = parse_demosaic(params, decoder)
    if decoder == "coreimage":
        highlight = "reconstruct"
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
    reject_gated_coreimage(tone_core, decoder)
    if dg.is_hdr_output_format(output_format) and tone_core != "agx":
        raise RuntimeError("HDR 输出当前只实现 AgX tone core")
    agx_primaries = parse_agx_primaries(params)
    lens_filter = parse_lens_filter(params)
    endpoint_mode = parse_endpoint_mode(params)
    chroma_nr = parse_chroma_nr(params, output_format)
    preview_entry = PREVIEW_STORE.peek(
        inp, highlight, wb, tone_core == "gated", decoder, coreimage_version, demosaic,
        coreimage_scale=coreimage_scale, margin=clip_margin,
    ) if hasattr(PREVIEW_STORE, "peek") else None
    preview_contract = preview_entry.bundle if preview_entry is not None else None
    if preview_contract is None and isinstance(params.get("_previewDecode"), dict):
        from types import SimpleNamespace
        preview_contract = SimpleNamespace(**params["_previewDecode"])
    bundle = _load_export_scene(
        inp, highlight, wb, decoder, coreimage_version, demosaic, coreimage_scale,
        preview=preview_contract,
        analysis_luminance_only=not want_png,
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
            inp, highlight, wb, decoder, coreimage_version, demosaic,
            coreimage_scale, clip_margin,
            decoded_bundle=bundle,
            envelope=params.get("_previewAnalysis"),
        )
    if analysis is None:
        analysis, y, ev_img = dg.analyze(
            bundle,
            clip_margin,
            diagnostics=want_png,
            gamut_names=None if want_png else (dg.output_gamut_space(gamut),),
            _return_planes=bool(want_png),
        )
    else:
        # Analysis also resolves the endpoint used by the spatial masks.
        # Reusing its statistics must replay that state on this fresh decode.
        from ..raw_io import refresh_clip_masks_from_fullwell

        refresh_clip_masks_from_fullwell(bundle, analysis.channel_fullwell)
        bundle = dg.release_analysis_buffers(bundle)
    auto_ev_result = None
    auto_plans = []
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
            lens_filter=lens_filter,
            chroma_nr=chroma_nr,
            _plan_sink=auto_plans,
        )
        ev = auto_ev_result.ev
    bundle = dg.with_intent_exposure(bundle, user_ev=ev, tone_core=tone_core)
    if auto_plans:
        render_plan = auto_plans[-1]
    else:
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
            endpoint_mode=endpoint_mode,
            chroma_nr=chroma_nr,
        )

    # review batch 24: the id the render resolves, not the raw payload key
    grade_id, grade_strength = resolve_grade_id(params)
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
        coreimage_scale=coreimage_scale,
        clip_margin=int(clip_margin),
        decoder=decoder,
        chroma_nr=chroma_nr,
    )
    # Analysis and the compiled plan now own the calibration coefficients.
    # If another process updated them during analysis, retry before writing
    # an output rather than attaching one generation's name to another model.
    if calibration_fingerprint() != calibration_generation:
        raise RuntimeError("实测标定在导出分析期间发生变化，请重新导出。")
    fingerprint = export_plan_fingerprint(
        input_path=str(inp.resolve()),
        input_size=int(inp.stat().st_size),
        # review batch 24: a same-size in-place replacement of the RAW is a
        # different input; the preview cache already keys on mtime
        input_mtime_ns=int(inp.stat().st_mtime_ns),
        calibration=calibration_generation,
        wb=wb,
        ev=float(ev),
        highlight=highlight,
        decoder=decoder,
        coreimage_version=coreimage_version,
        coreimage_scale=coreimage_scale,
        clip_margin=int(clip_margin),
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
        chroma_nr=float(chroma_nr),
        spatial_budget_mib=(_spatial_budget_mib_for_fingerprint() if float(chroma_nr) > 0.0 else 0),
        adjustments=dataclasses.astuple(adjustments),
        # Encode-affecting parameters (review batch 11): the HDR headroom and
        # the delivery/encode knobs change the written bytes, so they must
        # change the name. Headroom only participates when the container is
        # HDR — an unused slider value must not fork identical SDR exports.
        hdr_headroom=(
            float(hdr_headroom) if dg.is_hdr_output_format(output_format) else 0.0
        ),
        # HDR latitude dials: same participation rule as hdr_headroom.
        hdr_rho=hdr_rho if dg.is_hdr_output_format(output_format) else None,
        hdr_white_margin=(
            hdr_white_margin if dg.is_hdr_output_format(output_format) else None
        ),
        hdr_shoulder_start=(
            hdr_shoulder_start if dg.is_hdr_output_format(output_format) else None
        ),
        delivery=delivery_name,
        quality=int(quality),
        chroma=str(chroma),
        heif_settings=((delivery.heif_encoder,delivery.heif_bit_depth,delivery.heif_preset,delivery.heif_tune)
                       if delivery.container=="heic" else None),
    )
    out_ext = ".heic" if dg.container_for_output_format(output_format) == "heic" else ".jpg"
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
                hdr_rho=hdr_rho,
                hdr_white_margin=hdr_white_margin,
                hdr_shoulder_start=hdr_shoulder_start,
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
                return_rgb=True,
                delivery=delivery,
                chroma=chroma,
            )
            hdr_export_info = export_result if isinstance(export_result, dict) else None
            if hdr_export_info is not None and hdr_export_info.get("output_path"):
                # The writer corrects a container/suffix mismatch; report the real file.
                out_path = Path(str(hdr_export_info["output_path"]))
            if hdr_export_info is not None and out_path.is_file():
                hdr_export_info["file_size_bytes"] = out_path.stat().st_size
            rendered_u8 = (export_result.pop("_decoded_rgb", None) if isinstance(export_result, dict)
                           else export_result[1] if isinstance(export_result, tuple) else None)
            if rendered_u8 is None and dg.container_for_output_format(output_format) == "heic":
                from dngscan.gainmap import read_primary_rgb_u8

                rendered_u8 = read_primary_rgb_u8(out_path, gamut)
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
                    lens_filter=lens_filter,
                    chroma_nr=chroma_nr,
                )
            )
            preview_rgb = rendered_u8
            if auto_ev_result is not None and want_png:
                np = dg.np
                if rendered_u8 is None:
                    from PIL import Image

                    with Image.open(out_path) as im:
                        rendered_u8 = np.asarray(im.convert("RGB"), dtype=np.uint8)
                preview_rgb = annotate_preview_rgb_u8(
                    rendered_u8, dg.auto_ev_overlay_lines(auto_ev_result)
                )
            preview = (
                preview_b64_from_u8(
                    preview_rgb, icc_profile=icc_profile, width=PROXY_LONG_EDGE
                )
                if preview_rgb is not None
                else make_preview_b64(out_path, icc_profile=icc_profile)
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
        "chroma_nr": {"status": getattr(bundle, "chroma_nr_status", "disabled"),
                      "reason": getattr(bundle, "chroma_nr_reason", None)},
        "chroma_nr_capability": chroma_nr_capability(
            bundle, analysis, scene_transform=scene_transform,
            scene_transform_strength=scene_transform_strength,
        ),
        "gain": bundle.exposure_gain,
        "ev": ev,
        "ev_auto": auto_ev_payload(auto_ev_result),
        "format": (
            "HDR gain-map HEIC"
            if output_format == "ultrahdr-heic"
            else "HDR gain-map JPEG"
            if dg.is_hdr_output_format(output_format)
            else "SDR HEIC" if output_format == "sdr-heic" else "SDR JPEG"
        ),
        "hdr_headroom": hdr_headroom if dg.is_hdr_output_format(output_format) else 0.0,
        "hdr_diagnostics": (
            hdr_export_info.get("diagnostics") if hdr_export_info is not None else None
        ),
        "hdr_container": hdr_export_info if dg.is_hdr_output_format(output_format) else None,
        "delivery": hdr_export_info,
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
    params = dict(params)
    # The export worker has a fresh PREVIEW_STORE. Carry only small capability
    # facts across the process boundary, so auto cannot reselect another decoder.
    params.pop("_previewDecode", None)
    params.pop("_previewAnalysis", None)
    try:
        inp, highlight, _, _, _, _, _, _, _, _ = parse_job_params(params)
        decoder, version = parse_decoder(params)
        demosaic = parse_demosaic(params, decoder)
        scale, margin = parse_decode_extras(params, decoder)
        core, _ = parse_tone_core(params)
        preview = PREVIEW_STORE.peek(inp, highlight, str(params.get("wb", "camera")),
            core == "gated", decoder, version, demosaic, coreimage_scale=scale, margin=margin)
        if preview is not None:
            params["_previewDecode"] = _preview_decode_contract(preview.bundle)
            if str(params.get("wb", "camera")) == "camera":
                envelope = _preview_analysis_envelope(preview)
                if envelope is not None:
                    params["_previewAnalysis"] = envelope
    except (ValueError, OSError, AttributeError):
        pass  # The worker reports invalid public parameters through the normal path.
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
