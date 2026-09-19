# SPDX-License-Identifier: GPL-3.0-or-later
"""Verified SDR HEIF delivery, independent of HDR evidence and gain-map support."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile

from .delivery import DeliveryProfile


def save_sdr_heif(rgb, out_path: Path, delivery: DeliveryProfile, output_gamut="srgb",
                  *, source_raw: Path | None = None, return_rgb=False):
    from . import heif_encoder
    from .auto_encode import coding_metrics, select_heif_encoding
    from .color import output_icc_profile_bytes
    from .gainmap import (inspect_gainmap_file, read_primary_rgb_u8,
                          _base_roundtrip_error_arrays, _base_roundtrip_is_acceptable)
    from .heif_gainmap import _parse

    if delivery.container != "heic":
        raise ValueError("SDR HEIF requires a HEIC delivery profile")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_x265 = delivery.heif_encoder == "x265" or (
        delivery.heif_encoder == "auto" and heif_encoder.available())
    if not use_x265 and (delivery.name == "auto" or delivery.heif_bit_depth != 8
                         or delivery.chroma != "420"):
        raise ValueError("Apple SDR HEIF 当前仅能可靠写入 8-bit/4:2:0；"
                         "自动压缩、10-bit 或 4:2:2/4:4:4 请使用 libheif/x265，"
                         "或手动选择 share、8-bit、4:2:0")

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
        if output_icc_profile_bytes(output_gamut) not in profiles:
            raise RuntimeError("SDR HEIF 回读 ICC 与请求不符")
        decoded = read_primary_rgb_u8(path, output_gamut)
        metrics = _base_roundtrip_error_arrays(decoded, rgb)
        if not _base_roundtrip_is_acceptable(metrics, profile.tolerances):
            raise RuntimeError("SDR HEIF 回读误差超出交付门限")
        info.update(metrics)
        info.update(coding_metrics(decoded, rgb))
        return info

    def encode(q, chroma, auxiliary, candidate):
        profile = replace(delivery, name="share" if delivery.name == "auto" else delivery.name,
                          quality=q, chroma=chroma)
        encoder = heif_encoder.encode if use_x265 else heif_encoder.encode_apple
        encoder_info = encoder(rgb, candidate, q, chroma, bit_depth=profile.heif_bit_depth,
                               preset=profile.heif_preset, tune=profile.heif_tune,
                               output_gamut=output_gamut)
        info = verify(candidate, profile)
        return {**info, **encoder_info, "delivery_profile": profile.name,
                "icc_embedded": True}

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
        verify(candidate, final_profile)
        if return_rgb:
            info["_decoded_rgb"] = read_primary_rgb_u8(candidate, output_gamut)
        info["file_size_bytes"] = candidate.stat().st_size
        os.replace(candidate, out_path)
    info["output_path"] = str(out_path)
    return info
