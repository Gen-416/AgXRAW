# SPDX-License-Identifier: GPL-3.0-or-later
"""SDR and Apple ISO gain-map HDR JPEG export."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ._deps import np
from .color import output_gamut_label, output_icc_profile_bytes
from .constants import (
    DEFAULT_HDR_DRT, DEFAULT_HDR_HEADROOM_EV, HDR_DRT_CHOICES,
)
from .delivery import (
    DeliveryProfile,
    FinishedPair,
    container_for_output_format,
    is_hdr_output_format,
    profile_from_encode_settings,
    reprofile_for_container,
    resolve_delivery_profile,
)
from .gainmap import apple_gainmap_backend_status, encode_finished_pair
from .models import Analysis, RawBundle, RenderPlan, ToneCompressionPlan
from .render import render_output_u8


def chroma_to_subsampling(name: str) -> int:
    # PIL subsampling: 0 = 4:4:4 (full chroma), 1 = 4:2:2, 2 = 4:2:0 (smallest).
    return {"444": 0, "422": 1, "420": 2}.get(name, 0)


def save_jpeg_array(
    rgb_u8: Any, out_path: Path, quality: int, output_gamut: str = "srgb", subsampling: int = 0
) -> bool:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("JPEG 导出需要 Pillow，请先安装 pillow 再重试") from exc
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rgb_u8.dtype != np.uint8:
        rgb_u8 = np.clip(rgb_u8, 0, 255).astype(np.uint8)
    # Written by PIL directly: mpimg.imsave was a thin pass-through to this same
    # encoder call, at the price of importing matplotlib in every export worker.
    pil_kwargs: dict[str, Any] = {"quality": int(quality), "subsampling": int(subsampling), "optimize": True}
    icc_profile = output_icc_profile_bytes(output_gamut)
    if icc_profile is not None:
        pil_kwargs["icc_profile"] = icc_profile
    # R4: temp-write + atomic replace, matching the HDR writer — a direct
    # write destroyed a pre-existing good file the moment an interrupted run
    # left a truncated JPEG behind.
    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    try:
        Image.fromarray(rgb_u8, mode="RGB").save(
            str(tmp_path), format="JPEG", **pil_kwargs
        )
        os.replace(tmp_path, out_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return icc_profile is not None


def _scrubbed_capture_metadata(src_raw: Path, out_path: Path):
    """CGImageMetadata from the capture, scrubbed for a rendered delivery.

    Orientation forced to 1 (the render writes upright pixels), mosaic and
    capture-container descriptors removed, Exif pixel dimensions rewritten
    to the DELIVERED file's own geometry. Returns None when the metadata
    cannot be produced safely — the caller must then skip the carry rather
    than write unscrubbed tags (R4 F6a).
    """
    import Quartz  # type: ignore
    from Foundation import NSURL  # type: ignore

    src = Quartz.CGImageSourceCreateWithURL(
        NSURL.fileURLWithPath_(str(src_raw)), None
    )
    if src is None:
        return None
    meta = Quartz.CGImageSourceCopyMetadataAtIndex(src, 0, None)
    if meta is None:
        return None
    mutable = Quartz.CGImageMetadataCreateMutableCopy(meta)
    if mutable is None:
        return None
    Quartz.CGImageMetadataSetValueMatchingImageProperty(
        mutable,
        Quartz.kCGImagePropertyTIFFDictionary,
        Quartz.kCGImagePropertyTIFFOrientation,
        1,
    )
    # The capture is a mosaic container; its FORMAT descriptors must not
    # follow the metadata onto a rendered file (PhotometricInterpretation=
    # 32803 means CFA; PixelX/YDimension are mandatory fields that must
    # equal the RENDERED geometry; compression/components/gamma/colour-space
    # describe the RAW container, not this delivery, whose gamut is declared
    # by its embedded ICC).
    for tag_path in (
        "tiff:PhotometricInterpretation",
        "exif:CFAPattern",
        "exif:SensingMethod",
        "exif:PixelXDimension",
        "exif:PixelYDimension",
        "exif:CompressedBitsPerPixel",
        "exif:ComponentsConfiguration",
        "exif:Gamma",
        "exif:ColorSpace",
    ):
        try:
            Quartz.CGImageMetadataRemoveTagWithPath(mutable, None, tag_path)
        except Exception:
            pass
    out_src = Quartz.CGImageSourceCreateWithURL(
        NSURL.fileURLWithPath_(str(out_path)), None
    )
    if out_src is None:
        return None
    props = Quartz.CGImageSourceCopyPropertiesAtIndex(out_src, 0, None)
    if props:
        width = props.get(Quartz.kCGImagePropertyPixelWidth)
        height = props.get(Quartz.kCGImagePropertyPixelHeight)
        if width and height:
            Quartz.CGImageMetadataSetValueMatchingImageProperty(
                mutable,
                Quartz.kCGImagePropertyExifDictionary,
                Quartz.kCGImagePropertyExifPixelXDimension,
                int(width),
            )
            Quartz.CGImageMetadataSetValueMatchingImageProperty(
                mutable,
                Quartz.kCGImagePropertyExifDictionary,
                Quartz.kCGImagePropertyExifPixelYDimension,
                int(height),
            )
    return mutable


def _rewrite_with_metadata(
    out_path: Path, tmp_path: Path, meta, uti: str, merge: bool = True
) -> bool:
    """Lossless metadata rewrite out_path -> tmp_path via ImageIO.

    CGImageDestinationCopyImageSource forwards the encoded bitstream(s)
    without re-encoding; True only when the destination finalized and the
    tmp exists. The caller owns tmp cleanup and the final replace.

    `merge` is container-specific, measured behaviour: JPEG maps merged
    metadata back into APP1 EXIF correctly, but the HEIC writer DROPS the
    exif/tiff namespaces under merge (only explicitly-set tags survive) —
    replacing the metadata wholesale (merge=False) is what lands them, and
    the caller's post-rewrite gate re-inspection is what makes that safe
    (the ISO gain map, declared headroom, profile and chroma live in
    container structures the metadata replacement does not touch —
    verified, and re-verified per file before the tmp may replace)."""
    import Quartz  # type: ignore
    from Foundation import NSURL  # type: ignore

    src = Quartz.CGImageSourceCreateWithURL(
        NSURL.fileURLWithPath_(str(out_path)), None
    )
    if src is None:
        return False
    dest = Quartz.CGImageDestinationCreateWithURL(
        NSURL.fileURLWithPath_(str(tmp_path)), uti, 1, None
    )
    if dest is None:
        return False
    options = {
        Quartz.kCGImageDestinationMetadata: meta,
        Quartz.kCGImageDestinationMergeMetadata: bool(merge),
    }
    result = Quartz.CGImageDestinationCopyImageSource(dest, src, options, None)
    ok = bool(result[0] if isinstance(result, tuple) else result)
    return ok and tmp_path.is_file()


def carry_capture_metadata(src_raw: Path, out_jpeg: Path) -> bool:
    """Losslessly copy the capture's EXIF/TIFF/GPS/XMP metadata into a written JPEG.

    R3 item 6: a final-photo converter that drops shot time, body, lens,
    exposure and copyright is a delivery gap even when every pixel is right.
    Pixels and the embedded ICC stay byte-identical (no re-encode). Best-
    effort by design — returns False and leaves the JPEG untouched when
    ImageIO or the source metadata is unavailable (non-macOS hosts keep the
    previous pixels-only behaviour).
    """
    tmp_path = out_jpeg.with_name(f"{out_jpeg.name}.meta.{os.getpid()}.tmp")
    try:
        meta = _scrubbed_capture_metadata(src_raw, out_jpeg)
        if meta is None:
            return False
        if not _rewrite_with_metadata(out_jpeg, tmp_path, meta, "public.jpeg"):
            return False
        tmp_path.replace(out_jpeg)
        return True
    except Exception:
        return False
    finally:
        # R4 F6b: the tmp must not survive ANY failure path.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def carry_capture_metadata_hdr(
    src_raw: Path, out_path: Path, container: str
) -> bool:
    """EXIF/XMP carry for the ISO gain-map deliveries (JPEG MPF / HEIC).

    The HDR writer's own verification gates already passed on out_path, and
    metadata must NEVER cost them: the rewrite lands in a tmp which is
    RE-INSPECTED (gain map present, base profile, chroma, gain-map pixel
    format, declared headroom unchanged) before it may replace the verified
    file. Any mismatch discards the tmp and keeps the un-carried delivery —
    metadata is best-effort, the gates are not.
    """
    from .gainmap import inspect_gainmap_file

    uti = "public.heic" if str(container) == "heic" else "public.jpeg"
    tmp_path = out_path.with_name(f"{out_path.name}.meta.{os.getpid()}.tmp")
    try:
        pre = inspect_gainmap_file(out_path)
        if not pre.get("has_iso_gainmap"):
            return False
        meta = _scrubbed_capture_metadata(src_raw, out_path)
        if meta is None:
            return False
        if not _rewrite_with_metadata(
            out_path, tmp_path, meta, uti, merge=(uti == "public.jpeg")
        ):
            return False
        post = inspect_gainmap_file(tmp_path)
        import math as _math

        headroom_ok = (
            float(post.get("headroom", 0.0)) > 1.0
            and abs(
                _math.log2(
                    max(float(post.get("headroom", 0.0)), 1e-9)
                    / max(float(pre.get("headroom", 0.0)), 1e-9)
                )
            )
            < 1e-3
        )
        if not (
            post.get("has_iso_gainmap")
            and post.get("profile") == pre.get("profile")
            and post.get("chroma_subsampling") == pre.get("chroma_subsampling")
            and post.get("gainmap_pixel_format") == pre.get("gainmap_pixel_format")
            and headroom_ok
        ):
            return False
        tmp_path.replace(out_path)
        return True
    except Exception:
        return False
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def export_ultrahdr_jpeg(
    path: Path,
    out_path: Path,
    quality: int,
    bundle: RawBundle,
    analysis: Analysis,
    tone_plan: ToneCompressionPlan | RenderPlan | None = None,
    hdr_headroom: float = DEFAULT_HDR_HEADROOM_EV,
    hdr_rho: float | None = None,
    hdr_white_margin: float | None = None,
    hdr_shoulder_start: float | None = None,
    look: str = "none",
    look_strength: float = 1.0,
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    punch_scale: float = 1.0,
    hdr_drt: str = DEFAULT_HDR_DRT,
    delivery: DeliveryProfile | None = None,
    chroma: str = "444",
) -> dict[str, Any]:
    """Write Display P3 Ultrahdr (JPEG or HEIC) carrying an ISO 21496-1 gain map."""
    output_gamut = "p3"
    profile = delivery or profile_from_encode_settings(int(quality), str(chroma))
    container_label = "HEIC" if profile.container == "heic" else "JPEG"
    # A .jpg that actually holds HEIC bytes (or the reverse) misleads every downstream
    # consumer; the container decides the suffix, and the rewrite is reported in info.
    suffix = out_path.suffix.lower()
    if profile.container == "heic" and suffix not in (".heic", ".heif"):
        out_path = out_path.with_suffix(".heic")
    elif profile.container == "jpeg" and suffix in (".heic", ".heif"):
        out_path = out_path.with_suffix(".jpg")
    if str(hdr_drt) not in HDR_DRT_CHOICES:
        raise RuntimeError(f"未知 HDR DRT：{hdr_drt}（可选：{'/'.join(HDR_DRT_CHOICES)}）")
    if look != "none" or display_filter != "none":
        raise RuntimeError(
            "Ultrahdr 第一版仅支持 look=none 与 display_filter=none；"
            "现有 display look/filter 尚未 HDR 化，不能静默忽略"
        )
    supported, reason = apple_gainmap_backend_status()
    if not supported:
        raise RuntimeError(f"Cannot export Apple ISO gain-map HDR {container_label}: {reason}")

    try:
        from .grade import RENDER_MODE
        from .hdr_agx import achieved_headroom, to_gainmap_alternate
        from .hdr_agx_plan import compile_hdr_agx_plan, describe_hdr_plan
        from .models import HdrDisplayTarget, RenderPlan as _RenderPlan
        from .tone import build_render_plan

        plan = tone_plan if tone_plan is not None else build_render_plan(
            bundle, analysis, RENDER_MODE, output_gamut, scene_transform,
            scene_transform_strength, punch_scale, tone_core, lum_norm,
            agx_primaries=agx_primaries,
        )
        if not isinstance(plan, _RenderPlan):
            raise RuntimeError("Ultrahdr 需要完整 RenderPlan")
        if abs(float(plan.color.display_highlight_chroma_retreat)) > 1e-9:
            raise RuntimeError(
                "HDR 尚未定义 SDR 显示侧的高光褪白算子；"
                "请将 highlight fade 设为 0"
            )

        target = HdrDisplayTarget(peak_nits=100.0 * float(2.0 ** float(hdr_headroom)))
        hdr_plan = compile_hdr_agx_plan(
            plan,
            target,
            analysis=analysis,
            scene_decoder=str(bundle.scene_decoder),
            rho_base=hdr_rho,
            white_margin_ev=hdr_white_margin,
            shoulder_start_ev=hdr_shoulder_start,
        )
        if hdr_plan.tone.rendered_headroom_ev <= 0.0:
            raise RuntimeError(
                "该场景的可靠高光尾部不支持任何 HDR 余量："
                f"{describe_hdr_plan(hdr_plan)}。请改用 --output-format sdr"
            )

        film_full = (
            str(getattr(plan.tone, "film_mode", "observe")) == "full"
            and str(getattr(plan.tone, "curve_preset", "none")) != "none"
        )
        if film_full:
            # P6 (§10) "胶片印相 + scene HDR 扩展": the film print is the SDR
            # base; the HDR leg is a C1 scene-highlight gain above the
            # print's reference white — never a second development.
            from .hdr_agx import render_ultrahdr_film_pair

            base_u8, hdr_linear = render_ultrahdr_film_pair(
                bundle, analysis, plan, hdr_plan, output_gamut
            )
        else:
            from .hdr_agx import render_ultrahdr_agx_pair

            base_u8, hdr_linear = render_ultrahdr_agx_pair(
                bundle,
                analysis,
                plan,
                hdr_plan,
                output_gamut,
                scene_transform,
                scene_transform_strength,
            )
        actual = achieved_headroom(hdr_linear)
        # Encode-boundary guard at the scene-authorized content peak, not the display
        # capacity: the design contract (§9) keeps the alternate's ceiling at 2^H_content,
        # and the renderer already fitted its volume to exactly this endpoint.
        peak = float(hdr_plan.tone.peak_linear)
        pair = FinishedPair(
            sdr_rgb_u8=base_u8,
            hdr_rgba_f16=to_gainmap_alternate(hdr_linear, peak),
            display_headroom_ev=float(hdr_plan.tone.display_headroom_ev),
            output_gamut=output_gamut,
        )
        # The float32 rendition (a full-frame buffer) has served its purpose; the pair
        # carries the float16 alternate from here on.
        del hdr_linear
        info = encode_finished_pair(pair, out_path, profile)
        # EXIF/XMP carry AFTER the writer's own verification gates: the
        # rewrite is re-inspected and may only replace the verified file
        # when every gate quantity is unchanged (metadata never costs the
        # delivery contract).
        info["exif_carried"] = carry_capture_metadata_hdr(
            path, out_path, profile.container
        )
        if info["exif_carried"] and out_path.is_file():
            info["file_size_bytes"] = out_path.stat().st_size
        info["output_path"] = str(out_path)
        info["hdr_plan"] = (
            "胶片印相+scene HDR 扩展：" + describe_hdr_plan(hdr_plan)
            if film_full else describe_hdr_plan(hdr_plan)
        )
        info["display_headroom_ev"] = float(hdr_plan.tone.display_headroom_ev)
        info["requested_headroom_ev"] = float(hdr_plan.tone.requested_headroom_ev)
        info["rendered_headroom_ev"] = float(hdr_plan.tone.rendered_headroom_ev)
        # R2 item 17: two honest numbers, not one ambiguous "actual" — this is
        # the render's p99.99 usage figure (one specular pixel is not usage);
        # the container's declared ContentHeadroom (= alternate peak) arrives
        # from the writer as file_headroom_ev and the two may legitimately
        # differ by the sparse-highlight tail between p99.99 and max.
        info["actual_headroom_ev"] = float(actual)
        info["reliable_tail_ev"] = float(hdr_plan.tone.reliable_tail_ev)
        info["shoulder_start_ev"] = float(hdr_plan.tone.shoulder_start_ev)
        info["hdr_white_ev"] = float(hdr_plan.tone.white_ev)
        info["shoulder_alpha"] = float(hdr_plan.tone.shoulder_alpha)
        info["shoulder_segments"] = len(hdr_plan.tone.shoulder_segments)
        info["channel_separation"] = float(hdr_plan.color.channel_separation)
        # Latitude dials (taste-to-dial): report which of the three the user
        # moved off auto — the compiled values above already reflect them,
        # but the reader must be able to tell a dialed render from policy.
        _dials = {
            k: float(v)
            for k, v in (
                ("hdr_rho", hdr_rho),
                ("hdr_white_margin", hdr_white_margin),
                ("hdr_shoulder_start", hdr_shoulder_start),
            )
            if v is not None
        }
        if _dials:
            info["hdr_latitude_dials"] = _dials
        return info
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Cannot export Apple ISO gain-map HDR {container_label}: {exc}"
        ) from exc


def export_srgb_jpeg(
    path: Path,
    out_path: Path,
    quality: int,
    bundle: RawBundle,
    analysis: Analysis,
    tone_plan: ToneCompressionPlan | RenderPlan | None = None,
    output_gamut: str = "srgb",
    subsampling: int = 0,
    look: str = "none",
    look_strength: float = 1.0,
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    return_rgb: bool = False,
) -> Any:
    try:
        rgb = render_output_u8(
            bundle, analysis, output_gamut, tone_plan,
            look, look_strength, display_filter, filter_strength,
            scene_transform, scene_transform_strength,
            tone_core, lum_norm, agx_primaries,
        )
        embedded = save_jpeg_array(rgb, out_path, quality, output_gamut, subsampling)
        carry_capture_metadata(path, out_path)
        return (embedded, rgb) if return_rgb else embedded
    except Exception as exc:
        raise RuntimeError(f"Cannot export 8-bit {output_gamut_label(output_gamut)} JPEG: {exc}") from exc


def export_jpeg(
    path: Path,
    out_path: Path,
    quality: int,
    bundle: RawBundle,
    analysis: Analysis,
    tone_plan: ToneCompressionPlan | RenderPlan | None = None,
    output_gamut: str = "srgb",
    output_format: str = "sdr",
    hdr_headroom: float = DEFAULT_HDR_HEADROOM_EV,
    hdr_rho: float | None = None,
    hdr_white_margin: float | None = None,
    hdr_shoulder_start: float | None = None,
    subsampling: int = 0,
    look: str = "none",
    look_strength: float = 1.0,
    display_filter: str = "none",
    filter_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
    tone_core: str = "agx",
    lum_norm: str = "y",
    agx_primaries: str = "base",
    punch_scale: float = 1.0,
    return_rgb: bool = False,
    hdr_drt: str = DEFAULT_HDR_DRT,
    delivery: DeliveryProfile | None = None,
    chroma: str = "444",
) -> Any:
    if is_hdr_output_format(output_format):
        cont = container_for_output_format(output_format)
        profile = delivery
        if profile is None:
            profile = profile_from_encode_settings(
                int(quality), str(chroma), container=cont
            )
        elif profile.container != cont:
            profile = reprofile_for_container(profile, cont)
        # Keyword forwarding (A13 lesson): the positional form silently
        # shifted every argument after a signature grew — the HDR dials
        # would have swallowed `look`.
        return export_ultrahdr_jpeg(
            path,
            out_path,
            quality,
            bundle,
            analysis,
            tone_plan=tone_plan,
            hdr_headroom=hdr_headroom,
            hdr_rho=hdr_rho,
            hdr_white_margin=hdr_white_margin,
            hdr_shoulder_start=hdr_shoulder_start,
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
            hdr_drt=hdr_drt,
            delivery=profile,
            chroma=chroma,
        )
    if output_format != "sdr":
        raise ValueError(f"unknown output format: {output_format}")
    return export_srgb_jpeg(
        path, out_path, quality, bundle, analysis, tone_plan, output_gamut, subsampling,
        look, look_strength, display_filter, filter_strength,
        scene_transform, scene_transform_strength,
        tone_core, lum_norm, agx_primaries, return_rgb,
    )
