# SPDX-License-Identifier: GPL-3.0-or-later
"""Verified SDR HEIF delivery, independent of HDR evidence and gain-map support."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import struct
import tempfile

from .delivery import DeliveryProfile


def save_sdr_heif(rgb, out_path: Path, delivery: DeliveryProfile, output_gamut="srgb",
                  *, source_raw: Path | None = None, return_rgb=False):
    from . import heif_encoder
    from ._deps import np
    from .auto_encode import select_heif_encoding
    from .color import output_icc_profile_bytes
    from .gainmap import (inspect_gainmap_file, read_primary_rgb_u8,
                          _base_and_coding_metrics_arrays, _base_roundtrip_is_acceptable)
    from .heif_gainmap import _parse

    if delivery.container != "heic":
        raise ValueError("SDR HEIF requires a HEIC delivery profile")
    if delivery.heif_bit_depth not in (8, 10):
        raise ValueError("SDR HEIF requires 8-bit or 10-bit output")
    rgb = heif_encoder._validate_rgb(rgb)
    precise = delivery.heif_bit_depth == 10
    if not precise and rgb.dtype != np.uint8:
        raise ValueError("8-bit SDR HEIF requires uint8 RGB")
    if precise:
        from .gainmap import read_primary_rgb_float, _base_and_coding_metrics_float
        intended = (rgb.astype(np.float32) / np.float32(255.)
                    if rgb.dtype == np.uint8 else rgb)
    else:
        intended = rgb
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_x265 = delivery.heif_encoder == "x265" or (
        delivery.heif_encoder == "auto" and heif_encoder.available())
    if not use_x265 and (delivery.name == "auto" or delivery.heif_bit_depth != 8
                         or delivery.chroma != "420"):
        raise ValueError("Apple SDR HEIF 当前仅能可靠写入 8-bit/4:2:0；"
                         "自动压缩、10-bit 或 4:2:2/4:4:4 请使用 libheif/x265，"
                         "或手动选择 share、8-bit、4:2:0")
    # Keep one exact ICC reference across candidates and metadata-final
    # verification; generated fallback profiles carry a creation timestamp.
    expected_icc = output_icc_profile_bytes(output_gamut)
    if not expected_icc:
        raise RuntimeError("SDR HEIF 导出缺少输出 ICC")

    def verify(path, profile):
        info = inspect_gainmap_file(path)
        if (info["width"], info["height"]) != (rgb.shape[1], rgb.shape[0]):
            raise RuntimeError("SDR HEIF 回读尺寸与输入不符")
        if info["has_iso_gainmap"] or info["headroom"] > 1.001:
            raise RuntimeError("SDR HEIF 不应含 HDR gain map 或扩展余量")
        if info["bit_depth"] != profile.heif_bit_depth:
            raise RuntimeError("SDR HEIF 回读位深与请求不符")
        if info["chroma_subsampling"] != ":".join(profile.chroma):
            raise RuntimeError("SDR HEIF 回读采样与请求不符")
        _, _, primary, _, _, props, assocs, _ = _parse(path.read_bytes())
        profiles = [props[i-1][1][4:] for _, i in assocs.get(primary, [])
                    if props[i-1][0] == b"colr" and props[i-1][1][:4] in (b"prof", b"rICC")]
        if not profiles or any(profile != expected_icc for profile in profiles):
            raise RuntimeError("SDR HEIF 回读 ICC 与请求不符")
        nclx = [props[i-1][1] for _, i in assocs.get(primary, [])
                if props[i-1][0] == b"colr" and props[i-1][1][:4] == b"nclx"]
        expected_nclx = b"nclx" + struct.pack(">HHHB", 12 if output_gamut == "p3" else 1,
                                               13, 1, 128)
        if any(profile != expected_nclx for profile in nclx):
            raise RuntimeError("SDR HEIF 回读 NCLX 与请求不符")
        info["nclx_verified"] = bool(nclx)
        if precise:
            decoded = read_primary_rgb_float(path, output_gamut, _borrow_rgb=True)
            metrics = _base_and_coding_metrics_float(decoded, intended)
        else:
            decoded = read_primary_rgb_u8(path, output_gamut, _borrow_rgb=True)
            metrics = _base_and_coding_metrics_arrays(decoded, intended)
        if not _base_roundtrip_is_acceptable(metrics, profile.tolerances):
            raise RuntimeError("SDR HEIF 回读误差超出交付门限")
        info.update(metrics)
        return info, decoded

    def encode(q, chroma, auxiliary, candidate):
        profile = replace(delivery, name="share" if delivery.name == "auto" else delivery.name,
                          quality=q, chroma=chroma)
        encoder = heif_encoder.encode if use_x265 else heif_encoder.encode_apple
        options = dict(bit_depth=profile.heif_bit_depth, preset=profile.heif_preset,
                       tune=profile.heif_tune, output_gamut=output_gamut,
                       icc_profile=expected_icc)
        if use_x265 and precise and rgb.dtype != np.uint8:
            options["dither_quantization"] = True
        encoder_info = encoder(rgb, candidate, q, chroma, **options)
        info, _ = verify(candidate, profile)
        return {**info, **encoder_info, "delivery_profile": profile.name,
                "icc_embedded": True,
                "readback_precision": "float32" if precise else "uint8"}

    with tempfile.TemporaryDirectory(prefix=".agxraw-sdr-heif-", dir=out_path.parent) as td:
        candidate = Path(td) / "verified.heic"
        if delivery.name == "auto":
            info = select_heif_encoding(candidate, encode)
        else:
            info = encode(delivery.quality, delivery.chroma, None, candidate)
        from .export import carry_capture_metadata
        info["exif_carried"] = bool(source_raw is not None and
                                    carry_capture_metadata(source_raw, candidate, "heic"))
        # Metadata rewriting must preserve the coded image AND the requested geometry/profile.
        final_profile = replace(delivery, quality=int(info["delivery_quality"]),
                                chroma=str(info["delivery_chroma_requested"]))
        _, decoded = verify(candidate, final_profile)
        if return_rgb:
            # GUI preview remains uint8. All quality decisions above consume
            # the unquantized float readback, including metadata-final checks.
            if precise:
                preview = np.empty(decoded.shape, dtype=np.uint8)
                for y in range(0, decoded.shape[0], 128):
                    preview[y:y+128] = np.rint(np.clip(decoded[y:y+128], 0, 1)
                                              * np.float32(255.)).astype(np.uint8)
                info["_decoded_rgb"] = preview
            else:
                info["_decoded_rgb"] = np.ascontiguousarray(decoded)
        del decoded
        info["file_size_bytes"] = candidate.stat().st_size
        os.replace(candidate, out_path)
    info["output_path"] = str(out_path)
    return info
