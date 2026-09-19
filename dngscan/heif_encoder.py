# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional libheif HEVC encoder with independent quality/chroma/bit-depth controls.

Only the stable public C API is used. No shell encoder or intermediate PNG is
needed. Callers must independently read back colour, bit depth and sampling.
"""
from __future__ import annotations

import ctypes as C
import ctypes.util
from functools import lru_cache
from pathlib import Path

from ._deps import np


class _Error(C.Structure):
    _fields_ = [('code', C.c_int), ('subcode', C.c_int), ('message', C.c_char_p)]


class _Nclx(C.Structure):
    _fields_ = [('version', C.c_uint8), ('primaries', C.c_int), ('transfer', C.c_int),
                ('matrix', C.c_int), ('full_range', C.c_uint8), ('coordinates', C.c_float*8)]


@lru_cache(maxsize=1)
def _library():
    path = ctypes.util.find_library('heif')
    if not path:
        raise RuntimeError('可调 HEIF 编码需要 libheif（macOS: brew install libheif）')
    lib = C.CDLL(path)
    def bind(name, result, *args):
        f = getattr(lib, name); f.restype = result; f.argtypes = list(args)
    P, I = C.c_void_p, C.c_int
    bind('heif_context_alloc', P)
    bind('heif_context_free', None, P)
    bind('heif_context_get_encoder_for_format', _Error, P, I, C.POINTER(P))
    bind('heif_encoder_release', None, P)
    bind('heif_encoder_get_name', C.c_char_p, P)
    bind('heif_encoder_set_parameter', _Error, P, C.c_char_p, C.c_char_p)
    bind('heif_encoder_set_lossy_quality', _Error, P, I)
    bind('heif_image_create', _Error, I, I, I, I, C.POINTER(P))
    bind('heif_image_add_plane', _Error, P, I, I, I, I)
    bind('heif_image_get_plane', P, P, I, C.POINTER(I))
    bind('heif_image_release', None, P)
    bind('heif_image_set_nclx_color_profile', _Error, P, C.POINTER(_Nclx))
    bind('heif_image_set_raw_color_profile', _Error, P, C.c_char_p, P, C.c_size_t)
    bind('heif_context_encode_image', _Error, P, P, P, P, C.POINTER(P))
    bind('heif_image_handle_release', None, P)
    bind('heif_context_write_to_file', _Error, P, C.c_char_p)
    bind('heif_context_read_from_file', _Error, P, C.c_char_p, P)
    bind('heif_context_get_image_handle', _Error, P, C.c_uint32, C.POINTER(P))
    bind('heif_image_handle_get_width', I, P)
    bind('heif_image_handle_get_height', I, P)
    bind('heif_decode_image', _Error, P, C.POINTER(P), I, I, P)
    bind('heif_image_get_plane_readonly', P, P, I, C.POINTER(I))
    return lib


@lru_cache(maxsize=1)
def available() -> bool:
    try:
        lib = _library()
        ctx, enc = lib.heif_context_alloc(), C.c_void_p()
        if not ctx:
            return False
        try:
            error = lib.heif_context_get_encoder_for_format(ctx, 1, C.byref(enc))
            return not error.code and bool(enc) and b'x265' in lib.heif_encoder_get_name(enc).lower()
        finally:
            if enc: lib.heif_encoder_release(enc)
            lib.heif_context_free(ctx)
    except (OSError, AttributeError, RuntimeError):
        return False


def encode(rgb, path: Path, quality: int, chroma: str = '420', *,
           bit_depth: int = 10, preset: str = 'slow', tune: str = 'ssim',
           output_gamut: str = 'p3', auxiliary: bool = False) -> dict:
    """Encode finished nonlinear RGB, uint8 or float [0,1], as HEVC.

    Float input preserves master precision; increasing bit depth on uint8 input
    only improves the coding/conversion step, never recovers discarded detail.
    """
    from .color import output_icc_profile_bytes
    if not 1 <= quality <= 100 or chroma not in ('420','422','444'):
        raise ValueError('invalid HEIF quality/chroma')
    if bit_depth not in (8,10) or preset not in ('fast','medium','slow','slower'):
        raise ValueError('invalid HEIF bit depth/preset')
    if tune not in ('ssim','psnr','grain') or output_gamut not in ('srgb','p3'):
        raise ValueError('invalid HEIF tune/gamut')
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError('HEIF requires finite HxWx3 RGB')
    lib = _library()
    def check(error):
        if error.code:
            raise RuntimeError('libheif: '+error.message.decode('utf-8', errors='replace'))
    ctx = lib.heif_context_alloc()
    enc, img, handle = C.c_void_p(), C.c_void_p(), C.c_void_p()
    if not ctx:
        raise MemoryError('heif_context_alloc')
    try:
        check(lib.heif_context_get_encoder_for_format(ctx, 1, C.byref(enc)))
        name = lib.heif_encoder_get_name(enc).decode()
        if 'x265' not in name.lower():
            raise RuntimeError('可调 HEIF 路径需要 libheif 的 x265 编码器插件')
        check(lib.heif_encoder_set_lossy_quality(enc, quality))
        for key,value in (('chroma',chroma),('preset',preset),('tune',tune)):
            check(lib.heif_encoder_set_parameter(enc,key.encode(),value.encode()))
        # Avoid one process occupying every core while GUI previews run.
        if 'x265' in name.lower():
            check(lib.heif_encoder_set_parameter(enc,b'x265:pools',b'4'))
        h,w = rgb.shape[:2]
        check(lib.heif_image_create(w,h,1,10 if bit_depth==8 else 14,C.byref(img)))
        check(lib.heif_image_add_plane(img,10,w,h,bit_depth))
        stride = C.c_int()
        ptr = lib.heif_image_get_plane(img,10,C.byref(stride))
        if not ptr or stride.value < w*3*(1 if bit_depth==8 else 2):
            raise RuntimeError('invalid libheif interleaved plane')
        for y in range(0,h,128):
            band = rgb[y:y+128].astype(np.float32)
            if rgb.dtype == np.uint8:
                band /= 255.0
            band = np.rint(np.clip(band,0,1)*((1<<bit_depth)-1)).astype(np.uint8 if bit_depth==8 else '<u2')
            for i,row in enumerate(band):
                C.memmove(ptr+(y+i)*stride.value,row.ctypes.data,row.nbytes)
        # Gain-map RGB is numerical data, not display RGB. Preserve the
        # unspecified transfer/primaries and BT.601 YCbCr matrix of ImageIO's
        # RGB auxiliary; attaching the primary's ICC would transform gains.
        nclx = _Nclx(1,2 if auxiliary else 12 if output_gamut=='p3' else 1,
                     2 if auxiliary else 13,6 if auxiliary else 1,1)
        check(lib.heif_image_set_nclx_color_profile(img,C.byref(nclx)))
        if not auxiliary:
            icc = output_icc_profile_bytes(output_gamut)
            if not icc:
                raise RuntimeError('missing HEIF output ICC')
            check(lib.heif_image_set_raw_color_profile(img,b'prof',icc,len(icc)))
        check(lib.heif_context_encode_image(ctx,img,enc,None,C.byref(handle)))
        check(lib.heif_context_write_to_file(ctx,str(path).encode()))
        return {'encoder':name,'bit_depth':bit_depth,'preset':preset,'tune':tune,
                'delivery_quality':quality,'delivery_chroma_requested':chroma,
                'delivery_container':'heic'}
    finally:
        if handle: lib.heif_image_handle_release(handle)
        if img: lib.heif_image_release(img)
        if enc: lib.heif_encoder_release(enc)
        lib.heif_context_free(ctx)


def read_rgb_item(path: Path, item: int):
    """Decode an 8-bit data image without display colour management or HDR expansion."""
    lib = _library()
    ctx = lib.heif_context_alloc()
    handle, image = C.c_void_p(), C.c_void_p()
    if not ctx:
        raise MemoryError('heif_context_alloc')
    def check(error):
        if error.code:
            raise RuntimeError('libheif decode: ' + error.message.decode(errors='replace'))
    try:
        check(lib.heif_context_read_from_file(ctx, str(path).encode(), None))
        check(lib.heif_context_get_image_handle(ctx, int(item), C.byref(handle)))
        width, height = lib.heif_image_handle_get_width(handle), lib.heif_image_handle_get_height(handle)
        if min(width, height) <= 0:
            raise ValueError('invalid HEIF data-image dimensions')
        check(lib.heif_decode_image(handle, C.byref(image), 1, 10, None))
        stride = C.c_int()
        ptr = lib.heif_image_get_plane_readonly(image, 10, C.byref(stride))
        if not ptr or stride.value < width * 3:
            raise ValueError('invalid HEIF RGB plane')
        out = np.empty((height, width, 3), np.uint8)
        for row in range(height):
            C.memmove(out[row].ctypes.data, ptr + row * stride.value, width * 3)
        return out
    finally:
        if image: lib.heif_image_release(image)
        if handle: lib.heif_image_handle_release(handle)
        lib.heif_context_free(ctx)


def encode_apple(rgb, path: Path, quality: int, chroma: str = "420", *,
                 bit_depth: int = 10, preset="slow", tune="ssim", output_gamut="p3"):
    """System SDR writer. The caller verifies every requested property after writing.

    Apple does not expose x265 preset/tune; those controls apply only to x265.
    Unsupported depth/sampling requests must fail the caller's readback gate.
    """
    import Quartz
    from Foundation import NSData, NSNumber, NSURL
    from .gainmap import _ciimage_from_rgba
    from .color import output_icc_profile_bytes
    if not 1 <= quality <= 100 or chroma not in ("420", "422", "444") or bit_depth not in (8, 10):
        raise ValueError("invalid Apple HEIF controls")
    if output_gamut not in ("srgb", "p3"):
        raise ValueError("invalid Apple HEIF gamut")
    icc = output_icc_profile_bytes(output_gamut)
    if not icc:
        raise RuntimeError("missing HEIF output ICC")
    color = Quartz.CGColorSpaceCreateWithICCData(NSData.dataWithBytes_length_(icc, len(icc)))
    source = np.asarray(rgb)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3 or not source.size:
        raise ValueError("Apple SDR HEIF requires HxWx3 uint8 RGB")
    rgba = np.ones((*source.shape[:2], 4), dtype=np.float16 if bit_depth == 10 else np.uint8)
    rgba[..., :3] = source.astype(np.float32) / 255 if bit_depth == 10 else source
    if bit_depth == 8:
        rgba[..., 3] = 255
    pixel_format = Quartz.kCIFormatRGBAh if bit_depth == 10 else Quartz.kCIFormatRGBA8
    image, owner = _ciimage_from_rgba(rgba, pixel_format, color)
    context = Quartz.CIContext.contextWithOptions_({Quartz.kCIContextCacheIntermediates: False})
    if context is None or color is None:
        raise RuntimeError("Core Image 无法创建 SDR HEIF 编码上下文")
    request = {}
    key = getattr(Quartz, "kCGImageDestinationEncodeBasePixelFormatRequest", None)
    if key is not None:
        request[key] = NSNumber.numberWithUnsignedInt_(int.from_bytes((chroma + "f").encode(), "big"))
    options = {Quartz.kCGImageDestinationLossyCompressionQuality: quality / 100.,
               Quartz.kCGImageDestinationEncodeRequestOptions: request}
    result = context.writeHEIFRepresentationOfImage_toURL_format_colorSpace_options_error_(
        image, NSURL.fileURLWithPath_(str(path)),
        Quartz.kCIFormatRGB10 if bit_depth == 10 else Quartz.kCIFormatRGBA8,
        color, options, None)
    success, error = result if isinstance(result, tuple) else (bool(result), None)
    if not success:
        raise RuntimeError(f"Apple SDR HEIF 编码失败：{error}")
    from .heif_gainmap import embed_primary_icc
    embed_primary_icc(path, icc)
    return {"encoder": "Apple ImageIO", "delivery_quality": quality,
            "delivery_chroma_requested": chroma, "delivery_container": "heic"}
