# SPDX-License-Identifier: GPL-3.0-or-later
"""Apple-native ISO 21496-1 gain-map packaging and round-trip probes.

This module does not construct HDR pixels. dngscan's HDR extension around darktable-style
AgX formation owns the alternate rendition; Core Image only packages an already-complete
SDR/HDR pair.
"""
from __future__ import annotations

import os
import platform
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from ._deps import np
from .color import srgb_decode
from .delivery import (
    ARCHIVE_TOLERANCES,
    DeliveryProfile,
    DeliveryTolerances,
    FinishedPair,
    profile_from_encode_settings,
    reprofile_for_container,
)

# Backward-compatible aliases for the archive operating point. Prefer DeliveryProfile.
BASE_MEAN_CODE_ERROR_LIMIT = ARCHIVE_TOLERANCES.base_mean_code_error
BASE_CHANNEL_BIAS_CODE_ERROR_LIMIT = ARCHIVE_TOLERANCES.base_channel_bias_code_error
BASE_BLOCK_P99_CODE_ERROR_LIMIT = ARCHIVE_TOLERANCES.base_block_p99_code_error
HDR_BLOCK_MEDIAN_RELATIVE_ERROR_LIMIT = ARCHIVE_TOLERANCES.hdr_block_median_relative_error
# Core Image's ISO gain-map interpolation reaches about 4.12% on the sparse
# 32x64 conformance ramp even though its block median is 0.08% and chroma error
# is 0.12%. Keep archive p95 above that measured encoder floor.
HDR_BLOCK_P95_RELATIVE_ERROR_LIMIT = ARCHIVE_TOLERANCES.hdr_block_p95_relative_error
HDR_BLOCK_P99_RELATIVE_ERROR_LIMIT = ARCHIVE_TOLERANCES.hdr_block_p99_relative_error
HDR_BLOCK_CHROMA_ERROR_LIMIT = ARCHIVE_TOLERANCES.hdr_block_chroma_error


def _apple_gainmap_api_status() -> tuple[bool, str]:
    """Report API availability without claiming RGB round-trip correctness."""
    if platform.system() != "Darwin":
        return False, "HDR gain-map JPEG 当前需要 macOS Core Image"
    try:
        import Quartz  # type: ignore
    except Exception as exc:
        return False, f"缺少 PyObjC Quartz 绑定：{exc}；请安装 pyobjc-framework-Quartz"

    required = (
        "CIContext",
        "CIImage",
        "kCIFormatRGBA8",
        "kCIFormatRGBAh",
        "kCIImageRepresentationHDRImage",
        "kCIImageRepresentationHDRGainMapAsRGB",
        "kCIImageAuxiliaryHDRGainMap",
        "kCGImageDestinationEncodeRequest",
        "kCGImageDestinationEncodeToISOGainmap",
        "kCGImageDestinationEncodeRequestOptions",
        "kCGImageDestinationEncodeBaseIsSDR",
        "kCGColorSpaceDisplayP3",
        "kCGColorSpaceExtendedLinearDisplayP3",
    )
    missing = [name for name in required if not hasattr(Quartz, name)]
    if missing:
        return False, "当前 macOS/PyObjC 不暴露 ISO gain-map 编码 API：" + ", ".join(missing)
    if not hasattr(Quartz.CIContext, "writeJPEGRepresentationOfImage_toURL_colorSpace_options_error_"):
        return False, "当前 Core Image 不支持直接写入 HDR JPEG"
    if not hasattr(
        Quartz.CIContext,
        "writeHEIFRepresentationOfImage_toURL_format_colorSpace_options_error_",
    ):
        return False, "当前 Core Image 不支持直接写入 HDR HEIC"
    context = Quartz.CIContext.contextWithOptions_(
        {Quartz.kCIContextCacheIntermediates: _nsnumber_bool(False)}
    )
    if context is None:
        return False, "Core Image 无法创建可用的 CIContext"
    return True, "Apple Core Image ISO 21496-1 gain-map APIs available"


def _read_expanded_hdr_rgba_half(path: Path) -> Any:
    """Decode the composite HDR rendition, not merely the auxiliary container."""
    import Quartz  # type: ignore
    from Foundation import NSURL  # type: ignore

    linear_p3 = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceExtendedLinearDisplayP3)
    url = NSURL.fileURLWithPath_(str(path))
    base = Quartz.CIImage.imageWithContentsOfURL_(url)
    gainmap = Quartz.CIImage.imageWithContentsOfURL_options_(
        url,
        {Quartz.kCIImageAuxiliaryHDRGainMap: _nsnumber_bool(True)},
    )
    if base is None or gainmap is None or linear_p3 is None:
        raise RuntimeError("Core Image 无法回读扩展 HDR rendition")
    if not hasattr(base, "imageByApplyingGainMap_"):
        raise RuntimeError("当前 Core Image 不支持应用 HDR gain map")
    image = base.imageByApplyingGainMap_(gainmap)
    if image is None:
        raise RuntimeError("Core Image 无法组合 SDR 底图与 HDR gain map")
    extent = image.extent()
    width = int(round(float(extent.size.width)))
    height = int(round(float(extent.size.height)))
    if width <= 0 or height <= 0:
        raise RuntimeError("Core Image 回读 HDR rendition 得到空图像")
    row_bytes = width * 8
    buf = bytearray(height * row_bytes)
    context = Quartz.CIContext.contextWithOptions_(
        {Quartz.kCIContextCacheIntermediates: _nsnumber_bool(False)}
    )
    if context is None:
        raise RuntimeError("Core Image 无法创建 HDR 回读 CIContext")
    context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
        image,
        buf,
        row_bytes,
        extent,
        Quartz.kCIFormatRGBAh,
        linear_p3,
    )
    return np.frombuffer(memoryview(buf), dtype=np.float16).reshape(height, width, 4).copy()


@lru_cache(maxsize=1)
def _apple_rgb_gainmap_roundtrip_status() -> tuple[bool, str]:
    """Prove that independent RGB HDR geometry survives encode and expansion."""
    ok, reason = _apple_gainmap_api_status()
    if not ok:
        return ok, reason

    h, patch_w = 24, 24
    gains = np.array(
        [[3.0, 3.0, 3.0], [1.25, 2.0, 3.0], [3.0, 1.25, 2.0], [2.0, 3.0, 1.25]],
        dtype=np.float32,
    )
    base = np.full((h, patch_w * len(gains), 3), 180, dtype=np.uint8)
    base_linear = srgb_decode(base.astype(np.float32) / np.float32(255.0))
    hdr = np.empty(base.shape[:2] + (4,), dtype=np.float16)
    for index, gain in enumerate(gains):
        x0, x1 = index * patch_w, (index + 1) * patch_w
        hdr[:, x0:x1, :3] = (base_linear[:, x0:x1] * gain).astype(np.float16)
    hdr[..., 3] = np.float16(1.0)

    try:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rgb_gainmap_probe.jpg"
            write_apple_gainmap_jpeg(
                base,
                hdr,
                path,
                100,
                3.0,
                _verify_roundtrip_capability=False,
            )
            expanded = _read_expanded_hdr_rgba_half(path)[..., :3].astype(np.float32)
        expected_means = []
        actual_means = []
        inset = 4
        for index in range(len(gains)):
            x0, x1 = index * patch_w + inset, (index + 1) * patch_w - inset
            expected_means.append(np.mean(hdr[inset:-inset, x0:x1, :3], axis=(0, 1)))
            actual_means.append(np.mean(expanded[inset:-inset, x0:x1, :3], axis=(0, 1)))
        expected = np.asarray(expected_means, dtype=np.float32)
        actual = np.asarray(actual_means, dtype=np.float32)
        expected_chroma = expected / np.maximum(np.sum(expected, axis=1, keepdims=True), 1e-6)
        actual_chroma = actual / np.maximum(np.sum(actual, axis=1, keepdims=True), 1e-6)
        chroma_error = float(np.max(np.abs(actual_chroma - expected_chroma)))
        relative_error = float(
            np.mean(np.abs(actual - expected) / np.maximum(np.abs(expected), 0.05))
        )
        if chroma_error > 0.06 or relative_error > 0.25:
            return False, (
                "当前 Core Image 虽能写 RGB gain-map 容器，但扩展回读不能还原独立 RGB "
                f"HDR rendition（chroma error={chroma_error:.3f}, relative error={relative_error:.3f}）；"
                "已停用 HDR 导出，避免生成语义错误的文件"
            )
    except Exception as exc:
        return False, f"Apple RGB gain-map encode/decode round-trip 探针失败：{exc}"
    return True, "Apple Core Image ISO gain-map RGB round-trip verified"


_ROUNDTRIP_BAND_ROWS = 512  # multiple of 8 so block means never straddle bands


def _exact_upper_percentile(top_values: Any, total_count: int, q: float) -> float:
    """np.percentile('linear') for an upper quantile from the retained top-K order stats.

    Position (n-1)*q/100 interpolates two order statistics; for q near 100 both live in
    the global top ~1%, which per-band partial selection retains exactly. This is exact
    selection, not an approximation — only the values that can never be touched by the
    interpolation are discarded.
    """
    n = int(total_count)
    if n <= 0:
        return 0.0
    pos = (n - 1) * (float(q) / 100.0)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    frac = pos - lo
    # Ascending rank r maps to descending rank n-1-r; top_sorted[j] is the j-th largest.
    top_sorted = np.sort(np.asarray(top_values, dtype=np.float32))[::-1]
    need = n - lo  # how many largest values must have been retained
    if need > top_sorted.size:
        raise ValueError(
            f"top-K selection kept {top_sorted.size} values but rank needs {need}"
        )
    v_lo = top_sorted[n - 1 - lo]
    v_hi = top_sorted[n - 1 - hi]
    # Replicate np.percentile's _lerp exactly (difference taken in the input dtype,
    # reversed form for frac >= 0.5), so the result is bit-identical to the historical
    # whole-array computation, not merely close.
    diff = v_hi - v_lo  # float32 subtraction, as numpy does it
    if frac >= 0.5:
        return float(v_hi - diff * (1.0 - frac))
    return float(v_lo + diff * frac)


def _roundtrip_error(path: Path, intended_hdr_half: Any) -> dict[str, float]:
    """How far the file's expanded HDR rendition sits from the one that was written.

    The whole rendition is measured. RGB gain maps can carry chromatic corrections below
    reference white when the completed SDR and HDR gamut projectors differ; excluding that
    region would let a file pass without proving that its shadows and midtones survive.

    Processing is banded over rows with both sources kept float16: the statistics are
    identical to the historical whole-frame float32 computation (elementwise math per
    band, exact percentiles), but the function no longer materializes two full-frame
    float32 copies plus a full-frame chroma-difference array. The per-pixel chroma p99
    uses exact partial selection (_exact_upper_percentile) instead of storing every
    masked chroma difference.
    """
    expanded = _read_expanded_hdr_rgba_half(path)[..., :3]
    intended = np.asarray(intended_hdr_half)[..., :3]
    if expanded.shape != intended.shape:
        return {
            "chroma_error": float("inf"),
            "relative_error": float("inf"),
            "median_relative_error": float("inf"),
            "p95_relative_error": float("inf"),
            "p99_relative_error": float("inf"),
            "p999_relative_error": float("inf"),
            "block_median_relative_error": float("inf"),
            "block_p95_relative_error": float("inf"),
            "block_p99_relative_error": float("inf"),
            "block_chroma_error": float("inf"),
        }

    height, width = expanded.shape[:2]
    total_px = height * width
    relative = np.empty(total_px, dtype=np.float32)

    # Exact top-K retention for the masked per-pixel chroma p99: both order statistics
    # interpolated by p99 sit inside the top 1% (+2 slack) even if every sample is masked.
    top_k = int(np.ceil(0.01 * total_px * 3)) + 8
    chroma_top = np.empty(0, dtype=np.float32)
    chroma_count = 0

    h8 = height - height % 8
    w8 = width - width % 8
    blocks_a = (
        np.empty((h8 // 8, w8 // 8, 3), dtype=np.float32) if h8 > 0 and w8 > 0 else None
    )
    blocks_e = (
        np.empty((h8 // 8, w8 // 8, 3), dtype=np.float32) if h8 > 0 and w8 > 0 else None
    )

    for row0 in range(0, height, _ROUNDTRIP_BAND_ROWS):
        row1 = min(row0 + _ROUNDTRIP_BAND_ROWS, height)
        a = expanded[row0:row1].astype(np.float32).reshape(-1, 3)
        e = intended[row0:row1].astype(np.float32).reshape(-1, 3)
        # Normalize one RGB-vector error by that pixel's strongest intended component. A
        # tiny secondary channel must not turn a sub-code JPEG error into a huge percent.
        e_peak = np.max(np.abs(e), axis=1)
        relative[row0 * width : row1 * width] = np.max(np.abs(a - e), axis=1) / np.maximum(
            e_peak, 0.05
        )
        chroma_mask = e_peak > 0.05
        if bool(np.any(chroma_mask)):
            ac = a[chroma_mask]
            ec = e[chroma_mask]
            chroma = np.abs(
                ac / np.maximum(ac.sum(axis=1, keepdims=True), 1e-6)
                - ec / np.maximum(ec.sum(axis=1, keepdims=True), 1e-6)
            ).reshape(-1)
            chroma_count += chroma.size
            merged = np.concatenate((chroma_top, chroma))
            if merged.size > top_k:
                merged = np.partition(merged, merged.size - top_k)[-top_k:]
            chroma_top = merged
        if blocks_a is not None and row0 < h8:
            band_h8 = min(row1, h8) - row0
            band_h8 -= band_h8 % 8
            if band_h8 > 0:
                ab = expanded[row0 : row0 + band_h8, :w8].astype(np.float32)
                eb = intended[row0 : row0 + band_h8, :w8].astype(np.float32)
                b0 = row0 // 8
                blocks_a[b0 : b0 + band_h8 // 8] = ab.reshape(
                    band_h8 // 8, 8, w8 // 8, 8, 3
                ).mean(axis=(1, 3))
                blocks_e[b0 : b0 + band_h8 // 8] = eb.reshape(
                    band_h8 // 8, 8, w8 // 8, 8, 3
                ).mean(axis=(1, 3))

    chroma_p99 = (
        _exact_upper_percentile(chroma_top, chroma_count, 99.0)
        if chroma_count > 0
        else 0.0
    )
    median_relative = float(np.median(relative))
    p95_relative = float(np.percentile(relative, 95.0))
    p99_relative = float(np.percentile(relative, 99.0))
    p999_relative = float(np.percentile(relative, 99.9))

    if blocks_a is not None:
        block_relative = np.max(np.abs(blocks_a - blocks_e), axis=2) / np.maximum(
            np.max(np.abs(blocks_e), axis=2), 0.05
        )
        block_median = float(np.median(block_relative))
        block_p95 = float(np.percentile(block_relative, 95.0))
        block_p99 = float(np.percentile(block_relative, 99.0))
        block_mask = np.max(np.abs(blocks_e), axis=2) > 0.05
        if bool(np.any(block_mask)):
            abc = blocks_a[block_mask]
            ebc = blocks_e[block_mask]
            block_chroma = np.abs(
                abc / np.maximum(abc.sum(axis=1, keepdims=True), 1e-6)
                - ebc / np.maximum(ebc.sum(axis=1, keepdims=True), 1e-6)
            )
            block_chroma_p99 = float(np.percentile(block_chroma, 99.0))
        else:
            block_chroma_p99 = 0.0
    else:
        block_median = block_p95 = block_p99 = block_chroma_p99 = float("inf")
    return {
        "chroma_error": chroma_p99,
        "relative_error": p99_relative,
        "median_relative_error": median_relative,
        "p95_relative_error": p95_relative,
        "p99_relative_error": p99_relative,
        "p999_relative_error": p999_relative,
        "block_median_relative_error": block_median,
        "block_p95_relative_error": block_p95,
        "block_p99_relative_error": block_p99,
        "block_chroma_error": block_chroma_p99,
    }


def _hdr_roundtrip_is_acceptable(
    metrics: dict[str, float],
    tolerances: DeliveryTolerances = ARCHIVE_TOLERANCES,
) -> bool:
    """Whether the expanded HDR preserves tone and colour geometry.

    Block gates carry the tone/low-frequency contract. The pixel-scale chroma gate is
    kept alongside them because 8x8 block means average over the same grid 4:2:0
    subsamples chroma on, making them nearly blind to exactly that loss; per-profile
    limits state how much pixel-scale chroma damage each delivery contract accepts.
    """
    return bool(
        metrics["block_median_relative_error"]
        <= tolerances.hdr_block_median_relative_error
        and metrics["block_p95_relative_error"]
        <= tolerances.hdr_block_p95_relative_error
        and metrics["block_p99_relative_error"]
        <= tolerances.hdr_block_p99_relative_error
        and metrics["block_chroma_error"] <= tolerances.hdr_block_chroma_error
        and metrics["chroma_error"] <= tolerances.hdr_pixel_chroma_error
    )


def read_primary_rgb_u8(path: Path) -> Any:
    """Decode the primary SDR image without requiring Pillow HEIC support.

    JPEG still goes through Pillow when available (matches historical base gates). HEIC
    always uses ImageIO via Core Image so macOS can verify the gain-map unaware view.
    """
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        try:
            from PIL import Image

            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception:
            pass

    import Quartz  # type: ignore
    from Foundation import NSURL  # type: ignore

    url = NSURL.fileURLWithPath_(str(path))
    image = Quartz.CIImage.imageWithContentsOfURL_(url)
    if image is None:
        raise RuntimeError(f"无法解码主图：{path}")
    extent = image.extent()
    width = int(round(float(extent.size.width)))
    height = int(round(float(extent.size.height)))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"主图尺寸无效：{path}")
    p3 = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceDisplayP3)
    context = Quartz.CIContext.contextWithOptions_(
        {Quartz.kCIContextCacheIntermediates: _nsnumber_bool(False)}
    )
    if context is None or p3 is None:
        raise RuntimeError("Core Image 无法创建主图解码上下文")
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
        image,
        rgba,
        int(rgba.strides[0]),
        extent,
        Quartz.kCIFormatRGBA8,
        p3,
    )
    return np.ascontiguousarray(rgba[:, :, :3])


def _base_roundtrip_error(path: Path, intended_rgb_u8: Any) -> dict[str, float]:
    """Encoded-domain error of the SDR rendition seen by gain-map-unaware readers.

    Raw per-pixel error is content-dependent because lossy codecs remain lossy even at
    quality 100; high-ISO noise is especially expensive. Signed channel bias and 8x8
    block means expose an actual colour/tone transform while discounting zero-mean
    high-frequency texture loss.
    """
    decoded = np.asarray(read_primary_rgb_u8(path), dtype=np.uint8)
    intended = np.asarray(intended_rgb_u8, dtype=np.uint8)
    if decoded.shape != intended.shape:
        return {
            "base_mean_code_error": float("inf"),
            "base_p99_code_error": float("inf"),
            "base_max_code_error": float("inf"),
            "base_channel_bias_code_error": float("inf"),
            "base_block_p99_code_error": float("inf"),
        }
    # Banded over rows like _roundtrip_error: per-band elementwise math is
    # identical to the historical whole-frame computation, the upper percentile
    # uses exact top-K selection, and the means use float64 accumulators (more
    # accurate than the historical float32 pairwise mean; the difference class
    # is far below any delivery tolerance). Nothing full-frame float32 is
    # materialized.
    height, width = decoded.shape[:2]
    total_px = height * width
    h8 = height - height % 8
    w8 = width - width % 8
    top_k = int(np.ceil(0.01 * total_px)) + 8
    pixel_top = np.empty(0, dtype=np.float32)
    abs_sum = 0.0
    signed_sum = np.zeros(3, dtype=np.float64)
    max_err = 0.0
    block_means = (
        np.empty((h8 // 8, w8 // 8, 3), dtype=np.float32) if h8 > 0 and w8 > 0 else None
    )
    for row0 in range(0, height, _ROUNDTRIP_BAND_ROWS):
        row1 = min(row0 + _ROUNDTRIP_BAND_ROWS, height)
        signed = decoded[row0:row1].astype(np.float32) - intended[row0:row1].astype(
            np.float32
        )
        channel_error = np.abs(signed)
        pixel_error = np.max(channel_error, axis=2).reshape(-1)
        abs_sum += float(channel_error.sum(dtype=np.float64))
        signed_sum += signed.sum(axis=(0, 1), dtype=np.float64)
        if pixel_error.size:
            max_err = max(max_err, float(pixel_error.max()))
        merged = np.concatenate((pixel_top, pixel_error))
        if merged.size > top_k:
            merged = np.partition(merged, merged.size - top_k)[-top_k:]
        pixel_top = merged
        if block_means is not None and row0 < h8:
            band_h8 = min(row1, h8) - row0
            band_h8 -= band_h8 % 8
            if band_h8 > 0:
                b0 = row0 // 8
                block_means[b0 : b0 + band_h8 // 8] = signed[:band_h8, :w8].reshape(
                    band_h8 // 8, 8, w8 // 8, 8, 3
                ).mean(axis=(1, 3))
    if block_means is not None:
        block_error = np.max(np.abs(block_means), axis=2)
        block_p99 = float(np.percentile(block_error, 99.0))
    else:
        block_p99 = float("inf")
    return {
        "base_mean_code_error": float(abs_sum / (total_px * 3)) if total_px else 0.0,
        "base_p99_code_error": _exact_upper_percentile(pixel_top, total_px, 99.0),
        "base_max_code_error": max_err,
        "base_channel_bias_code_error": float(np.max(np.abs(signed_sum / total_px)))
        if total_px
        else 0.0,
        "base_block_p99_code_error": block_p99,
    }


def _base_roundtrip_is_acceptable(
    metrics: dict[str, float],
    tolerances: DeliveryTolerances = ARCHIVE_TOLERANCES,
) -> bool:
    """Whether JPEG loss preserved the base rendition's low-frequency appearance."""
    return bool(
        metrics["base_mean_code_error"] <= tolerances.base_mean_code_error
        and metrics["base_channel_bias_code_error"]
        <= tolerances.base_channel_bias_code_error
        and metrics["base_block_p99_code_error"] <= tolerances.base_block_p99_code_error
    )


def apple_gainmap_backend_status() -> tuple[bool, str]:
    """Whether an ISO gain-map JPEG can actually be produced on this system.

    Reports API availability only. Correctness is no longer decided here, because a
    synthetic probe can only answer for its own test pattern: the RGB round-trip probe
    uses gain ratios up to 2.4x between channels and fails at chroma error 0.144, while
    the renditions this pipeline actually produces round-trip at 0.0015 on the same
    machine. Gating production on that would refuse files that are demonstrably correct.

    Every write is instead verified against its own pixels after the fact, which is the
    guarantee that actually matters and is strictly stronger. The probe stays available
    as a capability description; see _apple_rgb_gainmap_roundtrip_status.
    """
    return _apple_gainmap_api_status()


def _nsnumber_bool(value: bool) -> Any:
    from Foundation import NSNumber  # type: ignore

    return NSNumber.numberWithBool_(bool(value))


def _nsdata_no_copy(array: Any) -> Any:
    from Foundation import NSData  # type: ignore

    if not array.flags.c_contiguous:
        raise ValueError("Core Image 输入必须是 C-contiguous array")
    return NSData.dataWithBytesNoCopy_length_freeWhenDone_(array, int(array.nbytes), False)


def _ciimage_from_rgba(array: Any, pixel_format: int, color_space: Any) -> tuple[Any, Any]:
    import Quartz  # type: ignore

    h, w = array.shape[:2]
    data = _nsdata_no_copy(array)
    image = Quartz.CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
        data,
        int(array.strides[0]),
        (float(w), float(h)),
        pixel_format,
        color_space,
    )
    if image is None:
        raise RuntimeError("Core Image 无法创建 HDR 编码输入")
    return image, data


def inspect_gainmap_file(path: Path) -> dict[str, Any]:
    """Read the properties needed to prove an Apple-written ISO gain-map file."""
    import Quartz  # type: ignore
    from Foundation import NSURL  # type: ignore

    source = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(path)), None)
    if source is None:
        raise RuntimeError(f"ImageIO 无法读取输出文件：{path}")
    primary = Quartz.CGImageSourceCopyPropertiesAtIndex(source, 0, None) or {}
    file_props = Quartz.CGImageSourceCopyProperties(source, None) or {}
    contents = file_props.get(Quartz.kCGImagePropertyFileContentsDictionary, {})
    images = contents.get("Images", ()) if contents else ()
    first = images[0] if images else {}
    auxiliary = first.get(Quartz.kCGImagePropertyAuxiliaryData, ()) if first else ()
    iso_type = str(getattr(Quartz, "kCGImageAuxiliaryDataTypeISOGainMap", ""))
    gainmap = next(
        (item for item in auxiliary if str(item.get("AuxiliaryDataType", "")) == iso_type),
        None,
    )
    pixel_format = int(gainmap.get("PixelFormat", 0)) if gainmap is not None else 0
    pixel_format_name = (
        pixel_format.to_bytes(4, "big").decode("ascii", errors="replace") if pixel_format else ""
    )
    return {
        "has_iso_gainmap": gainmap is not None,
        "headroom": float(primary.get("Headroom", 1.0)),
        "profile": str(primary.get(Quartz.kCGImagePropertyProfileName, "")),
        "width": int(primary.get(Quartz.kCGImagePropertyPixelWidth, 0)),
        "height": int(primary.get(Quartz.kCGImagePropertyPixelHeight, 0)),
        "chroma_subsampling": str(first.get("ChromaSubsampling", "")),
        "gainmap_width": int(gainmap.get("Width", 0)) if gainmap is not None else 0,
        "gainmap_height": int(gainmap.get("Height", 0)) if gainmap is not None else 0,
        "gainmap_pixel_format": pixel_format_name,
    }


def inspect_gainmap_jpeg(path: Path) -> dict[str, Any]:
    """Backward-compatible alias for JPEG/HEIC ISO gain-map inspection."""
    return inspect_gainmap_file(path)


def write_apple_gainmap_jpeg(
    base_rgb_u8: Any,
    hdr_rgba_half: Any,
    out_path: Path,
    quality: int,
    hdr_headroom_ev: float,
    *,
    delivery: DeliveryProfile | None = None,
    chroma: str = "444",
    _verify_roundtrip_capability: bool = True,
) -> dict[str, Any]:
    """Write and validate a Display P3 JPEG carrying an ISO 21496-1 gain map."""
    profile = delivery or profile_from_encode_settings(
        int(quality), str(chroma), container="jpeg"
    )
    profile = reprofile_for_container(profile, "jpeg")
    return write_apple_gainmap_file(
        base_rgb_u8,
        hdr_rgba_half,
        out_path,
        hdr_headroom_ev,
        delivery=profile,
        _verify_roundtrip_capability=_verify_roundtrip_capability,
    )


def write_apple_gainmap_heic(
    base_rgb_u8: Any,
    hdr_rgba_half: Any,
    out_path: Path,
    quality: int,
    hdr_headroom_ev: float,
    *,
    delivery: DeliveryProfile | None = None,
    chroma: str = "444",
    _verify_roundtrip_capability: bool = True,
) -> dict[str, Any]:
    """Write and validate a Display P3 HEIC carrying an ISO 21496-1 gain map."""
    profile = delivery or profile_from_encode_settings(
        int(quality), str(chroma), container="heic"
    )
    profile = reprofile_for_container(profile, "heic")
    return write_apple_gainmap_file(
        base_rgb_u8,
        hdr_rgba_half,
        out_path,
        hdr_headroom_ev,
        delivery=profile,
        _verify_roundtrip_capability=_verify_roundtrip_capability,
    )


def write_apple_gainmap_file(
    base_rgb_u8: Any,
    hdr_rgba_half: Any,
    out_path: Path,
    hdr_headroom_ev: float,
    *,
    delivery: DeliveryProfile,
    _verify_roundtrip_capability: bool = True,
) -> dict[str, Any]:
    """Write JPEG or HEIC ISO gain-map packaging from finished formation masters."""
    profile = delivery
    quality = int(profile.quality)
    tolerances = profile.tolerances
    container = str(profile.container)
    if container not in ("jpeg", "heic"):
        raise ValueError(f"不支持的 gain-map 容器：{container}")
    if not 1 <= quality <= 100:
        raise ValueError("编码 quality 必须在 1-100 之间")

    base = np.asarray(base_rgb_u8)
    hdr = np.asarray(hdr_rgba_half)
    if base.dtype != np.uint8 or base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("HDR gain-map 底图必须是 HxWx3 uint8")
    if hdr.dtype != np.float16 or hdr.shape != base.shape[:2] + (4,):
        raise ValueError("HDR alternate 必须是与底图同尺寸的 HxWx4 float16")
    if not bool(np.all(np.isfinite(hdr))):
        raise ValueError("HDR alternate 含 NaN/Inf，无法声明可靠的 content headroom")

    ok, reason = (
        apple_gainmap_backend_status()
        if _verify_roundtrip_capability
        else _apple_gainmap_api_status()
    )
    if not ok:
        raise RuntimeError(reason)

    import Quartz  # type: ignore
    from Foundation import NSNumber, NSURL  # type: ignore

    if not base.flags.c_contiguous:
        base = np.ascontiguousarray(base)
    if not hdr.flags.c_contiguous:
        hdr = np.ascontiguousarray(hdr)

    p3 = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceDisplayP3)
    linear_p3 = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceExtendedLinearDisplayP3)
    if p3 is None or linear_p3 is None:
        raise RuntimeError("系统未提供 Display P3 / Extended Linear Display P3 色彩空间")

    base_rgba = np.empty(base.shape[:2] + (4,), dtype=np.uint8)
    base_rgba[:, :, :3] = base
    base_rgba[:, :, 3] = np.uint8(255)
    base_image, base_data = _ciimage_from_rgba(base_rgba, Quartz.kCIFormatRGBA8, p3)
    hdr_image, hdr_data = _ciimage_from_rgba(hdr, Quartz.kCIFormatRGBAh, linear_p3)
    base_image = base_image.imageBySettingContentHeadroom_(1.0)
    requested_headroom = float(2.0 ** float(hdr_headroom_ev))
    actual_headroom = float(np.max(hdr[:, :, :3]))
    if actual_headroom > requested_headroom * 1.001:
        raise RuntimeError(
            f"HDR rendition 超过所选余量：{actual_headroom:.3f}x > {requested_headroom:.3f}x"
        )
    if actual_headroom <= 1.0 + 1e-3:
        raise RuntimeError("该场景没有高于 reference white 的有效 HDR 内容")
    hdr_image = hdr_image.imageBySettingContentHeadroom_(actual_headroom)

    context = Quartz.CIContext.contextWithOptions_(
        {Quartz.kCIContextCacheIntermediates: _nsnumber_bool(False)}
    )
    if context is None:
        raise RuntimeError("Core Image 无法创建 HDR 编码 CIContext")
    encode_request_options: dict[str, Any] = {
        Quartz.kCGImageDestinationEncodeBaseIsSDR: _nsnumber_bool(True),
    }
    if tolerances.gainmap_subsample_factor is not None:
        subsample_key = getattr(
            Quartz, "kCGImageDestinationEncodeGainMapSubsampleFactor", None
        )
        if subsample_key is None:
            raise RuntimeError(
                "此 macOS 的 ImageIO 不支持 gain-map 下采样 "
                "(kCGImageDestinationEncodeGainMapSubsampleFactor)；"
                "请升级系统或改用 --delivery-profile archive"
            )
        encode_request_options[subsample_key] = NSNumber.numberWithInt_(
            int(tolerances.gainmap_subsample_factor)
        )
    options = {
        Quartz.kCGImageDestinationLossyCompressionQuality: float(quality) / 100.0,
        Quartz.kCIImageRepresentationHDRImage: hdr_image,
        Quartz.kCIImageRepresentationHDRGainMapAsRGB: _nsnumber_bool(True),
        Quartz.kCGImageDestinationEncodeRequest: Quartz.kCGImageDestinationEncodeToISOGainmap,
        Quartz.kCGImageDestinationEncodeRequestOptions: encode_request_options,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".heic" if container == "heic" else ".jpg"
    temp_path = out_path.with_name(f".{out_path.name}.{uuid.uuid4().hex}.tmp{suffix}")
    label = "HEIC" if container == "heic" else "JPEG"
    try:
        url = NSURL.fileURLWithPath_(str(temp_path))
        if container == "heic":
            result = context.writeHEIFRepresentationOfImage_toURL_format_colorSpace_options_error_(
                base_image,
                url,
                Quartz.kCIFormatRGBA8,
                p3,
                options,
                None,
            )
        else:
            result = context.writeJPEGRepresentationOfImage_toURL_colorSpace_options_error_(
                base_image,
                url,
                p3,
                options,
                None,
            )
        success, error = result if isinstance(result, tuple) else (bool(result), None)
        if not success:
            raise RuntimeError(f"Core Image 写入 ISO gain-map {label} 失败：{error}")

        _ = (base_data, hdr_data, base_rgba, hdr)
        info = inspect_gainmap_file(temp_path)
        if not info["has_iso_gainmap"]:
            raise RuntimeError(f"Core Image 输出不含 ISO 21496-1 gain map")
        if info["profile"] != "Display P3":
            raise RuntimeError(f"HDR {label} 底图色彩配置错误：{info['profile'] or '无 ICC'}")
        if tolerances.require_chroma_444 and info["chroma_subsampling"] != "4:4:4":
            raise RuntimeError(
                f"HDR {label} 主图未保持 4:4:4：{info['chroma_subsampling'] or '未知'}"
            )
        fmt = str(info["gainmap_pixel_format"] or "")
        if fmt in ("", "L008"):
            raise RuntimeError(
                f"HDR {label} gain map 不是 RGB 辅助图（got {fmt or '未知'}）；"
                "独立 HDR color geometry 需要 RGB gain map"
            )
        if info["headroom"] <= 1.0:
            raise RuntimeError(f"HDR {label} 未声明有效的扩展动态范围")
        headroom_error_ev = abs(
            float(np.log2(max(float(info["headroom"]), 1e-9) / actual_headroom))
        )
        info["headroom_error_ev"] = headroom_error_ev
        if headroom_error_ev > 0.05:
            raise RuntimeError(
                f"HDR {label} 声明余量与 alternate 峰值不一致："
                f"误差={headroom_error_ev:.4f} EV；已丢弃该文件"
            )
        base_roundtrip = _base_roundtrip_error(temp_path, base)
        info.update(base_roundtrip)
        if not _base_roundtrip_is_acceptable(base_roundtrip, tolerances):
            raise RuntimeError(
                "写出的 SDR 底图无法保持输入 rendition："
                f"平均码值误差={base_roundtrip['base_mean_code_error']:.3f}，"
                f"p99={base_roundtrip['base_p99_code_error']:.1f}，"
                f"max={base_roundtrip['base_max_code_error']:.1f}，"
                f"通道偏差={base_roundtrip['base_channel_bias_code_error']:.3f}，"
                f"8x8块p99={base_roundtrip['base_block_p99_code_error']:.3f}；已丢弃该文件"
            )
        roundtrip = _roundtrip_error(temp_path, hdr)
        info.update(roundtrip)
        if not _hdr_roundtrip_is_acceptable(roundtrip, tolerances):
            raise RuntimeError(
                "写出的 HDR rendition 无法从文件还原："
                f"中位相对误差={roundtrip['median_relative_error']:.4f}，"
                f"p95相对误差={roundtrip['p95_relative_error']:.4f}，"
                f"p99相对误差={roundtrip['p99_relative_error']:.4f}，"
                f"p99色品误差={roundtrip['chroma_error']:.4f}，"
                f"8x8中位/p95/p99={roundtrip['block_median_relative_error']:.4f}/"
                f"{roundtrip['block_p95_relative_error']:.4f}/"
                f"{roundtrip['block_p99_relative_error']:.4f}，"
                f"8x8色品p99={roundtrip['block_chroma_error']:.4f}；已丢弃该文件"
            )
        os.replace(temp_path, out_path)
        info["gainmap_as_rgb"] = True
        info["delivery_profile"] = profile.name
        info["delivery_quality"] = quality
        info["delivery_chroma_requested"] = profile.chroma
        info["delivery_container"] = container
        return info
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def encode_finished_pair(
    pair: FinishedPair,
    out_path: Path,
    delivery: DeliveryProfile,
) -> dict[str, Any]:
    """Encode formation masters with a delivery profile. No re-formation."""
    return write_apple_gainmap_file(
        pair.sdr_rgb_u8,
        pair.hdr_rgba_f16,
        out_path,
        pair.display_headroom_ev,
        delivery=delivery,
    )


def encode_finished_pair_jpeg(
    pair: FinishedPair,
    out_path: Path,
    delivery: DeliveryProfile,
) -> dict[str, Any]:
    """JPEG-only alias kept for callers that predate HEIC delivery."""
    jpeg_delivery = delivery
    if delivery.container != "jpeg":
        jpeg_delivery = DeliveryProfile(
            name=delivery.name,
            quality=delivery.quality,
            chroma=delivery.chroma,
            container="jpeg",
            tolerances=delivery.tolerances,
        )
    return encode_finished_pair(pair, out_path, jpeg_delivery)
