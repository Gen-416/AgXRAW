# SPDX-License-Identifier: GPL-3.0-or-later
"""RAW decode via rawpy and scene-linear render buffers."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ._deps import np, rawpy
from . import metadata as dng_metadata
from .wb import kelvin_mode_cct, solve_kelvin_wb
from .constants import (
    RGB_TO_XYZ as _RGB_TO_XYZ,
    XYZ_TO_RGB as _XYZ_TO_RGB,
    COREIMAGE_SCALE_DEFAULT_MODE,
    DECODER_CHOICES,
    DEMOSAIC_AUTO_PREFERENCE,
    DEMOSAIC_CHOICES,
    WB_CHOICES,
)

# D65 white in XYZ (Rec.2020's own white) and Rec.2020's XYZ->RGB, used by the
# hot-WB ladder: the former for LibRaw's row normalization, the latter as the
# "identity decode" conjugation basis for rungs that LibRaw rendered raw_color.
D65_XYZ = tuple(float(sum(row)) for row in _RGB_TO_XYZ["Rec2020"])
XYZ_TO_RGB_REC2020 = tuple(tuple(float(v) for v in row) for row in _XYZ_TO_RGB["Rec2020"])
from .models import RawBundle
from .evidence import EvidenceAcquisitionError, acquire_raw_evidence

def rawpy_highlight_mode(name: str) -> Any:
    modes = getattr(rawpy, "HighlightMode", object)
    mapping = {
        "clip": getattr(modes, "Clip", 0),
        "blend": getattr(modes, "Blend", getattr(modes, "Clip", 0)),
        "reconstruct": getattr(modes, "ReconstructDefault", getattr(modes, "Clip", 0)),
    }
    if name not in mapping:
        raise ValueError(f"unknown highlight mode: {name}")
    return mapping[name]


def highlight_mode_cn(name: str) -> str:
    return {
        "clip": "完全过曝",
        "blend": "高光混合",
        "reconstruct": "高光重建",
    }.get(name, name)


def _fixed_asshot_wb_kwargs(camera_wb: Any) -> dict[str, Any]:
    """The production LibRaw WB contract, shared with the alignment reference.

    Demosaic/highlight always sees one immutable per-capture preconditioner;
    supplying the as-shot values explicitly avoids a decoder-specific
    camera-WB fallback changing this fixed boundary.
    """
    if camera_wb and len(camera_wb) >= 3 and all(
        np.isfinite(value) and value > 0.0 for value in camera_wb[:3]
    ):
        fixed_wb = [float(value) for value in camera_wb[:4]]
        while len(fixed_wb) < 4:
            fixed_wb.append(fixed_wb[1])
        return {"use_camera_wb": False, "user_wb": fixed_wb}
    return {"use_camera_wb": True}


def _apply_gain_maps_mosaic(
    raw: Any, maps: list, black_levels: list[float], white_level: int,
    camera_white_levels: list[float] | None = None,
    loss_mask: Any | None = None,
) -> None:
    """Apply pre-demosaic GainMap opcodes to the live rawpy mosaic in place.

    Runs AFTER the evidence copies (bundle.raw_image, clip masks) are taken: clip
    evidence is sensor truth and must stay pre-correction. Values gain toward the
    corners (fp measures up to x1.384); results clip at the target sensel's own
    channel white level — a corner pixel pushed past white saturates exactly as
    the DNG rendering path intends.

    R3 item 2 — plane semantics follow the DNG SDK (dng_opcode_GainMap /
    dng_gain_map::Interpolate): the opcode applies to image planes
    [plane, plane+planes), and the gain for image plane p reads map plane
    min(p, map_planes - 1). The mosaic stage has exactly ONE image
    plane (index 0), so an opcode with plane > 0 does not apply here at all,
    and an applicable opcode reads map plane 0 — never an average across
    map planes, which is only coincidentally right for map_planes == 1.
    """
    img = raw.raw_image_visible
    if img.ndim == 3:
        # Linear DNG contains image planes, unlike a one-plane CFA mosaic.
        # Reuse the in-place strided kernel per plane, without copying RGB.
        from types import SimpleNamespace
        for m in maps:
            for c in range(m.plane, min(m.plane + m.planes, min(3, img.shape[2]))):
                mp = min(c, m.map_planes - 1)
                single = replace(m, plane=0, planes=1, map_planes=1,
                                 gains=np.asarray(m.gains)[..., mp:mp+1])
                view = SimpleNamespace(raw_image_visible=img[..., c],
                    raw_colors_visible=np.broadcast_to(np.uint8(0), img.shape[:2]))
                _apply_gain_maps_mosaic(view, [single],
                    [channel_black_level(black_levels,c)], white_level,
                    [channel_fullwell(white_level,camera_white_levels or [],c)],
                    None if loss_mask is None else loss_mask[..., c])
        return
    colors = raw.raw_colors_visible
    h, w = img.shape
    blacks = np.asarray(black_levels or [0.0], dtype=np.float32)
    whites = np.asarray(
        [
            float(v) if float(v) > 0 else float(white_level)
            for v in (camera_white_levels or [])
        ] or [float(white_level)],
        dtype=np.float32,
    )
    from . import _fast

    native = _fast.kernel("apply_gain_map_mosaic")
    for m in maps:
        from copy import copy
        m=copy(m)
        # DNG AreaSpec's empty rectangle means the complete image. Resolve it
        # before dispatch so both native and NumPy kernels see identical bounds.
        if m.bottom <= m.top or m.right <= m.left:
            if m.row_pitch != 1 or m.col_pitch != 1:
                raise ValueError("empty DNG GainMap area requires unit pitches")
            m.top,m.left,m.bottom,m.right=0,0,h,w
        else:
            # AreaSpec coordinates are signed. Intersect with the image while
            # retaining the declared pitch's phase, as dng_area_spec::Overlap.
            top=m.top+max(0,(-m.top+m.row_pitch-1)//m.row_pitch)*m.row_pitch
            left=m.left+max(0,(-m.left+m.col_pitch-1)//m.col_pitch)*m.col_pitch
            m.top,m.left,m.bottom,m.right=top,left,min(m.bottom,h),min(m.right,w)
        if m.bottom <= m.top or m.right <= m.left:
            # An empty intersection is a no-op, not the authored empty AreaSpec.
            # Do not pass negative clipped bounds into the unsigned Rust ABI.
            continue
        if int(getattr(m, "plane", 0)) > 0 or int(getattr(m, "planes", 1)) < 1:
            # Targets image planes the 1-plane mosaic does not have (or none).
            continue
        if (
            native is not None
            and img.dtype == np.uint16
            and colors.dtype == np.uint8
            and img.flags["WRITEABLE"]
        ):
            # Stage 1 (2026-09-15): the Rust kernel replicates the bilinear
            # expression below (float64 grid arithmetic, float32 difference,
            # clip to the sensel's own white) element for element.
            native(
                img, colors, m,
                [float(v) for v in blacks], [float(v) for v in whites],
                loss_mask,
            )
            continue
        rows = np.arange(m.top, min(m.bottom, h), m.row_pitch)
        cols = np.arange(m.left, min(m.right, w), m.col_pitch)
        if rows.size == 0 or cols.size == 0:
            continue
        gains_grid = np.asarray(m.gains, dtype=np.float64)[:, :, 0]
        iv = np.clip(((rows + 0.5) / h - m.origin_v) / max(m.spacing_v, 1e-9), 0, m.points_v - 1)
        ih = np.clip(((cols + 0.5) / w - m.origin_h) / max(m.spacing_h, 1e-9), 0, m.points_h - 1)
        v0 = np.clip(np.floor(iv).astype(int), 0, m.points_v - 2) if m.points_v > 1 else np.zeros(rows.size, int)
        h0 = np.clip(np.floor(ih).astype(int), 0, m.points_h - 2) if m.points_h > 1 else np.zeros(cols.size, int)
        fv = (iv - v0)[:, None] if m.points_v > 1 else np.zeros((rows.size, 1))
        fh = (ih - h0)[None, :] if m.points_h > 1 else np.zeros((1, cols.size))
        # rows/cols are arithmetic sequences, so the sampled sites form a strided
        # view of the mosaic; the corner gathers hoist the two row selections and
        # the arithmetic keeps the original expression, dtypes and operation
        # order — every element is bit-identical to the historical fancy-indexed
        # version, without the np.ix_ gather/scatter copies.
        h1 = np.minimum(h0 + 1, m.points_h - 1)
        v1 = np.minimum(v0 + 1, m.points_v - 1)
        img_view = img[m.top : min(m.bottom, h) : m.row_pitch,
                       m.left : min(m.right, w) : m.col_pitch]
        cidx = colors[m.top : min(m.bottom, h) : m.row_pitch,
                      m.left : min(m.right, w) : m.col_pitch]
        # ROW BANDS (scheduler plan S4): a 1x1-pitch GainMap samples every
        # sensel, so the bilinear expression below held about ten FULL-FRAME
        # float64 temporaries at once — 1.77 GB of the measured 3.3 GB decode
        # peak for a 24 MP file. Every operation here is elementwise, so
        # banding changes no result bit; it only bounds the transients.
        band = max(1, 8_000_000 // max(int(cols.size), 1))
        for r0 in range(0, int(rows.size), band):
            r1 = min(r0 + band, int(rows.size))
            rows_lo = gains_grid[v0[r0:r1]]
            rows_hi = gains_grid[v1[r0:r1]]
            g00 = rows_lo[:, h0]
            g01 = rows_lo[:, h1]
            g10 = rows_hi[:, h0]
            g11 = rows_hi[:, h1]
            fv_band = fv[r0:r1]
            gains = (g00 * (1 - fv_band) * (1 - fh) + g01 * (1 - fv_band) * fh
                     + g10 * fv_band * (1 - fh) + g11 * fv_band * fh)
            sub = img_view[r0:r1].astype(np.float32)
            b = (
                blacks[np.clip(cidx[r0:r1], 0, blacks.size - 1)]
                if blacks.size > 1 else np.float32(blacks[0])
            )
            wl = (
                whites[np.clip(cidx[r0:r1], 0, whites.size - 1)]
                if whites.size > 1 else np.float32(whites[0])
            )
            corrected = b + (sub - b) * gains
            if loss_mask is not None:
                loss_view = loss_mask[m.top:min(m.bottom, h):m.row_pitch,
                                      m.left:min(m.right, w):m.col_pitch]
                loss_view[r0:r1] |= ((sub < wl) & (corrected >= wl)).astype(np.uint8)
            corrected = np.clip(corrected, 0.0, wl)
            img_view[r0:r1] = corrected.astype(img.dtype)


def _apply_vignette_render(render: Any, vignette: Any, orientation_flip: int = 0,
                           loss_mask: Any | None = None, channel_limits: Any | None = None) -> Any:
    """Apply a post-demosaic FixVignetteRadial to the scene render, in row bands.

    g(r) = 1 + sum k_i (r/m)^(2(i+1)) with the optical centre at (cx_hat, cy_hat) and
    m the max centre-to-corner distance (DNG 1.4). A pure per-pixel scalar gain: it
    commutes with WB and matrices, so applying it to the finished linear render is
    exact for the gain. When operating on white-balanced camera planes, the
    caller conjugates the DNG [0,1] clipping bounds by that same WB scaling.
    """
    # Opcode coordinates belong to the un-oriented image. Undo LibRaw's
    # orientation as a view before evaluating the radial field; this also
    # preserves the old arithmetic exactly for every reflected/rotated pixel.
    inverse_flip = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 5, 7: 7}
    if orientation_flip not in inverse_flip:
        raise ValueError(f"unknown LibRaw flip code {orientation_flip}")
    working = _orient_like_libraw(render, inverse_flip[orientation_flip])
    loss = None if loss_mask is None else _orient_like_libraw(loss_mask, inverse_flip[orientation_flip])
    h, w = working.shape[:2]
    cx, cy = float(vignette.cx_hat) * w, float(vignette.cy_hat) * h
    m2 = max((cx) ** 2 + (cy) ** 2, (w - cx) ** 2 + (cy) ** 2,
             (cx) ** 2 + (h - cy) ** 2, (w - cx) ** 2 + (h - cy) ** 2)
    limit = float(np.iinfo(render.dtype).max) if np.issubdtype(render.dtype, np.integer) else None
    if channel_limits is not None:
        limit = np.asarray(channel_limits, dtype=np.float32)
    xs = (np.arange(w, dtype=np.float64) + 0.5 - cx) ** 2
    k = [float(v) for v in vignette.k]
    out = working
    for y0 in range(0, h, 512):
        y1 = min(y0 + 512, h)
        ys = (np.arange(y0, y1, dtype=np.float64) + 0.5 - cy) ** 2
        r2 = (ys[:, None] + xs[None, :]) / m2
        g = 1.0 + r2 * (k[0] + r2 * (k[1] + r2 * (k[2] + r2 * (k[3] + r2 * k[4]))))
        band = out[y0:y1].astype(np.float32) * g[:, :, None].astype(np.float32)
        if limit is not None:
            if loss is not None:
                np.maximum(loss[y0:y1], band >= limit, out=loss[y0:y1])
            band = np.clip(band, 0.0, limit)
        out[y0:y1] = band.astype(render.dtype)
    return render

def solve_wb_for_mode(
    wb_mode: str,
    path: Path,
    xyz_to_cam: Any | None,
    make: str | None = None,
    model: str | None = None,
) -> tuple[list[float] | None, str | None]:
    """(Fixed-Kelvin multipliers or None, degradation/provenance note or None).

    Calibration ladder, most trusted first: the file's own DNG dual-illuminant
    tags -> LibRaw's per-model Adobe matrix -> this project's fallback matrix
    table for bodies the installed LibRaw predates (camera_matrices.py; the note
    records the borrowed provenance). When every rung is missing the request
    DEGRADES instead of refusing: the caller renders with the camera's as-shot
    balance and must surface the returned note — a declared degradation is
    usable, a silent one would be a hidden white balance.
    """
    cct = kelvin_mode_cct(wb_mode)
    if cct is None:
        return None, None
    calibration = dng_metadata.read_dng_color_calibration(path)
    matrix = None
    if xyz_to_cam is not None:
        candidate = np.asarray(xyz_to_cam, dtype=np.float64)
        if candidate.size >= 9 and float(np.abs(candidate[:3, :3]).sum()) > 1e-9:
            matrix = candidate[:3, :3]
    note: str | None = None
    if calibration is None and matrix is None:
        from .camera_matrices import fallback_xyz_to_cam

        fallback = fallback_xyz_to_cam(make, model)
        if fallback is not None:
            matrix, source_note = fallback
            note = f"颜色标定来自回退矩阵表：{source_note}"
    try:
        return solve_kelvin_wb(cct, dng_calibration=calibration, xyz_to_cam=matrix), note
    except ValueError as exc:
        return None, (
            f"声明 {wb_mode} 白平衡不可用（{exc}）；已退化为相机 AsShot。"
            "该机型缺少颜色标定数据：结果可用，但白平衡声明与色彩精度可能有偏差"
        )


def camera_data_support_note(
    has_dng_calibration: bool,
    has_libraw_matrix: bool,
    fallback_available: bool,
    has_priors: bool,
    make: str | None,
    model: str | None,
) -> str | None:
    """One consolidated per-file marker: does this body have enough data to render
    accurately? None means fully supported. The render always proceeds — the marker
    is a truthful label, never a gate.

    Colour calibration is the rendering-accuracy data: without any matrix the
    decoder's colour conversion for this model is unanchored and the deviation is
    unpredictable (not merely "slightly off"). Missing sensor priors only degrade
    the *analysis* numbers (absolute stops/DR), so alone they do not raise this
    marker — the priors line already reports that honestly.
    """
    if has_dng_calibration or has_libraw_matrix:
        return None
    ident = f"{make or '?'} {model or '?'}".strip()
    if fallback_available:
        return (
            f"机型 {ident} 的颜色标定不在解码器数据表内：白平衡求解已由内置回退"
            "矩阵代偿，但解码器内部色彩转换仍无该机型矩阵，色彩精度可能有偏差。"
            "功能照常执行"
        )
    note = (
        f"机型 {ident} 暂无足够数据支撑准确运算（DNG 标签 / LibRaw 表 / 回退矩阵"
        "均无颜色标定）：输出图片结果可能有无法预测的偏差。功能照常执行"
    )
    if not has_priors:
        note += "；该机型亦无传感器先验，绝对档位/动态范围为单帧估计"
    return note


def libraw_wb_headroom_gain(wb_values: list[float] | None) -> float:
    """Container headroom LibRaw reserves for non-clipping highlight modes.

    With blend/reconstruct, LibRaw divides the whole post-WB image by the largest
    normalized WB multiplier so the boosted channel can be reconstructed above nominal
    sensor white without overflowing uint16. That is storage scaling, not exposure.
    """
    if not wb_values:
        return 1.0
    values = np.asarray(wb_values[:4], dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0
    normalized = values / max(float(np.min(values)), 1e-12)
    return float(max(1.0, np.max(normalized)))


def baseline_exposure_gain(baseline_exposure: float | None) -> float:
    """Linear gain for a DNG BaselineExposure, as a scale divisor rather than a multiply.

    BaselineExposure is file-authored baseline rendering compensation. It is not a
    measurement of capture exposure and does not target image content to middle gray.
    LibRaw ignores it outright (verified on iPhone files whose tags differ by 2 EV), while
    Core Image applies it unless overridden; honouring it here keeps both decoders faithful
    to the same DNG recipe before the user's explicit EV adjustment.

    It is applied by dividing scene_scale, never by scaling the buffer: the gain reaches
    5.65x on an iPhone low-light frame, which would clip everything above 0.18 in a uint16
    buffer normalised to sensor saturation. Dividing the scale leaves the codes untouched
    and re-interprets them, so no precision is lost and no highlight is destroyed.
    """
    if baseline_exposure is None:
        return 1.0
    value = float(baseline_exposure)
    if not np.isfinite(value):
        return 1.0
    # Guard against a corrupt tag rewriting the exposure by an absurd amount.
    return float(2.0 ** max(-8.0, min(8.0, value)))


def scene_green_median(scene_rgb: Any) -> float:
    """Robust green-channel level of one decoded scene-linear frame.

    The Core Image alignment ratio uses this statistic from both decoders. An earlier
    implementation divided both values by the same RAW-green median and called the
    results sensor gains; that common term cancels exactly. The operation is therefore a
    per-file decoded-level comparison, not absolute sensor calibration. It applies one
    scalar to the whole frame, so it preserves within-image light ratios and does not
    force a night scene or any other content toward 18% gray.
    """
    green = np.asarray(scene_rgb, dtype=np.float32)[:, :, 1].ravel()
    green = green[np.isfinite(green) & (green > 1e-4)]
    if green.size == 0:
        return float("nan")
    return float(np.median(green))


# Bounds on the Core Image alignment factor. Measured factors run 0.57..0.94 across iPhone
# 16 Pro and Sigma fp; anything far outside that means a statistic failed rather than a
# decoder disagreeing, and a render must not be destroyed by a bad measurement.
COREIMAGE_ALIGN_MIN = 0.25
COREIMAGE_ALIGN_MAX = 4.0


def coreimage_uses_file_alignment(mode: str) -> bool:
    """Whether a Core Image scale policy requests the LibRaw A/B reference render."""
    return mode == "aligned"


def coreimage_alignment_factor(reference_level: float, coreimage_level: float) -> float:
    """Scalar that matches RAW 9's decoded green median to LibRaw's for the same file.

    This makes decoder A/B comparisons share a practical exposure ruler. It is neither a
    sensor-absolute calibration nor auto exposure: there is no external brightness target,
    and all pixels receive the same factor. Because decoder color matrices, reconstruction,
    lens opcodes, and framing can affect the medians, ``unity`` remains available when the
    native Core Image scale itself is what should be inspected.

    Invalid or implausible measurements return identity. The caller records the failure so
    a silent fallback cannot be mistaken for a successful alignment.
    """
    if not (np.isfinite(reference_level) and np.isfinite(coreimage_level)):
        return 1.0
    if reference_level <= 0.0 or coreimage_level <= 0.0:
        return 1.0
    factor = float(reference_level) / float(coreimage_level)
    if factor < COREIMAGE_ALIGN_MIN or factor > COREIMAGE_ALIGN_MAX:
        return 1.0
    return factor


def libraw_scene_scale(
    encoded_max: float,
    highlight_mode_name: str,
    wb_values: list[float] | None,
    baseline_exposure: float | None = None,
) -> float:
    """Decode uint16 code values into one exposure unit independent of highlight mode."""
    scale = float(encoded_max)
    if highlight_mode_name != "clip":
        scale /= libraw_wb_headroom_gain(wb_values)
    return scale / baseline_exposure_gain(baseline_exposure)


def scene_rec2020_to_xyz_render(scene_rec2020: Any, scene_scale: float) -> Any:
    """Derive XYZ render buffer from a single Rec.2020 demosaic (same geometry as scene)."""
    from .color import rec2020_to_xyz

    scene = np.asarray(scene_rec2020)
    if np.issubdtype(scene.dtype, np.integer):
        flat = scene.reshape(-1, 3)
        out = np.empty((flat.shape[0], 3), dtype=np.uint16)
        chunk = 1_000_000
        for start in range(0, flat.shape[0], chunk):
            end = min(start + chunk, flat.shape[0])
            # Keep float64 here for byte-for-byte compatibility with the original
            # analysis buffer, but never materialize a full-frame float64 RGB copy.
            linear = flat[start:end].astype(np.float64) / float(scene_scale)
            xyz = rec2020_to_xyz(linear)
            max_linear = float(np.iinfo(out.dtype).max) / float(scene_scale)
            out[start:end] = (
                np.clip(xyz, 0.0, max_linear) * float(scene_scale)
            ).astype(np.uint16)
        return out.reshape(scene.shape)
    xyz = rec2020_to_xyz(scene.reshape(-1, 3)).reshape(scene.shape)
    return xyz.astype(scene.dtype, copy=False)


def normalized_camera_wb(wb_values: list[float] | None) -> Any:
    """Return finite RGB camera gains with green fixed to one.

    LibRaw accepts four CFA multipliers (two greens on Bayer sensors), while its normal
    three-channel reconstruction has already merged G2 into green.  The public WB modes
    solve/declare both greens from G1, so G1 remains the explicit normalization anchor;
    a zero/missing metadata G2 therefore cannot perturb the hot transform.
    """
    if not wb_values or len(wb_values) < 3:
        raise ValueError("white-balance multipliers are unavailable")
    values = np.asarray(wb_values[:4], dtype=np.float64)
    if not np.all(np.isfinite(values[:3])) or np.any(values[:3] <= 0.0):
        raise ValueError(f"invalid white-balance multipliers: {wb_values!r}")
    green = float(values[1])
    return np.asarray(
        [float(values[0]) / green, float(values[1]) / green, float(values[2]) / green],
        dtype=np.float64,
    )


def d65_row_normalize(xyz_to_cam: Any) -> Any:
    """Scale each XYZ->camera row so the D65 white maps to camera (1, 1, 1).

    This is LibRaw's own convention (cam_rgb rows normalized to sum 1, i.e.
    cam_xyz @ XYZ(sRGB white) == 1): a matrix in this convention maps the
    balanced camera neutral to the Rec.2020/sRGB (D65) white, which is what
    lets two DIFFERENT matrices sit on the two sides of a hot-WB conjugation.
    """
    m = np.asarray(xyz_to_cam, dtype=np.float64)[:3, :3].copy()
    neutral = m @ np.asarray(D65_XYZ, dtype=np.float64)
    if not np.all(np.isfinite(neutral)) or np.any(np.abs(neutral) <= 1e-12):
        raise ValueError("colour matrix maps the D65 white to a zero channel")
    return m / neutral[:, None]


def _libraw_applied_xyz_to_cam(bundle: Any, evidence: Any) -> Any:
    """The XYZ->camera matrix the fixed LibRaw decode actually applied, in
    LibRaw's normalization: rgb_cam recovered exactly when LibRaw adopted the
    embedded cmatrix, else the evidence cam_xyz row-normalized the way
    LibRaw normalizes it before inverting."""
    raw_cmatrix = getattr(bundle, "wb_color_matrix", None)
    if raw_cmatrix is not None:
        cmatrix = np.asarray(raw_cmatrix, dtype=np.float64)
        adopted = (
            cmatrix.ndim == 2
            and cmatrix.shape[0] >= 1
            and cmatrix.shape[1] >= 1
            and np.isfinite(cmatrix[0, 0])
            and float(cmatrix[0, 0]) > 0.125
            and dng_metadata.is_dng_container(bundle.path)
        )
        if adopted:
            derived = color_matrix_xyz_to_cam(raw_cmatrix)
            if derived is not None:
                return derived
    return d65_row_normalize(evidence)


def color_matrix_xyz_to_cam(color_matrix: Any | None) -> Any | None:
    """Equivalent XYZ->camera matrix from an adopted LibRaw ``cmatrix``.

    When LibRaw adopts ``color_matrix`` it becomes the camera -> linear-sRGB(D65)
    ``rgb_cam`` matrix that output conversion goes through (Rec.2020 output is
    ``(sRGB->Rec2020) @ rgb_cam``).  Its underlying camera->sRGB rows are normalized so
    the post-WB camera neutral maps to sRGB white; on DNGs LibRaw builds it from
    ColorMatrix2 (measured on Sigma fp: it equals the D65-row-normalized ColorMatrix2
    to tag precision, while ``rgb_xyz_matrix`` is all-zero).  The 3x4 fourth column is
    the second green; after LibRaw's three-channel reconstruction G2 is merged into G,
    so the column folds into green (it is zero on three-colour sensors).  Inverting
    through sRGB->XYZ yields an ``xyz_to_cam`` whose pseudo-inverse reproduces LibRaw's
    true camera->Rec.2020 exactly — no Adobe-table approximation involved.  The
    per-channel row normalization is harmless to the hot transform as long as decode
    and target use this same matrix, because diagonal gains commute; that is why the
    caller never mixes this convention with an unnormalized DNG target matrix.
    Returns None when the matrix is absent, non-finite, empty, or singular.
    """
    if color_matrix is None:
        return None
    matrix = np.asarray(color_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 3:
        return None
    cam_to_srgb = matrix[:3, :3].copy()
    if matrix.shape[1] >= 4:
        cam_to_srgb[:, 1] += matrix[:3, 3]
    if not np.all(np.isfinite(cam_to_srgb)) or float(np.abs(cam_to_srgb).sum()) <= 1e-9:
        return None
    from .constants import RGB_TO_XYZ

    cam_to_xyz = np.asarray(RGB_TO_XYZ["sRGB"], dtype=np.float64) @ cam_to_srgb
    try:
        xyz_to_cam = np.linalg.inv(cam_to_xyz)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(xyz_to_cam)):
        return None
    return xyz_to_cam


def resolve_hot_wb_c0(
    bundle: RawBundle, target_cct: float | None = None
) -> tuple[Any, Any, str]:
    """(decode C0, target matrix, source) for the hot-WB stage, best rung first.

    Calibration ladder, mirroring ``solve_wb_for_mode``'s philosophy — every rung must
    correspond to what the fixed decode actually did:

    1. ``wb_xyz_to_cam`` (evidence ``rgb_xyz_matrix``): when the file carries DNG
       colour calibration tags and a fixed-Kelvin target is requested, decode C0 is
       the matrix LibRaw ACTUALLY applied (``_libraw_applied_xyz_to_cam``) and the
       target is the file's dual-illuminant interpolation at the declared CCT,
       both D65 row-normalised (``d65_row_normalize``) so the two sides share one
       neutral convention (source ``"evidence+cct"``; self-review 2026-08-27 P1
       replaced the earlier "both sides interpolated, unnormalised" formulation,
       which paired a matrix the decoder never applied with an unnormalised
       target).  The evidence matrix itself is LibRaw's ``cam_xyz`` — on DNGs sourced from
       ColorMatrix2 and therefore pinned to its calibration illuminant (~D65); using
       it directly as C0 against a target interpolated at the declared CCT would put
       the two sides on different illuminant anchors (the former "seam A", a ~1500 K
       anchor gap measured on synthetic calibrations).  Without calibration tags (or
       for non-Kelvin targets) the evidence matrix serves both sides unchanged
       (source ``"evidence"``): a single matrix on both sides is self-consistent
       because diagonal gains commute through it, and mixing an interpolated side
       with an evidence side is exactly the hidden white-balance shift rung 2 warns
       about.
    2. LibRaw ``color_matrix`` (``rgb_cam``): the matrix the decoder truly applied,
       converted by ``color_matrix_xyz_to_cam``.  Guarded by LibRaw's own embedded
       cmatrix adoption test (pinned ``identify.cpp``): rawpy's ``color_matrix``
       surfaces ``rawdata.color.cmatrix`` from *before* that gate, and LibRaw only
       memcpys it into ``rgb_cam`` for DNG containers (``dng_version`` non-zero, i.e.
       a DNGVersion tag in IFD0) whose ``cmatrix[0][0] > 0.125``; on every other file
       the decode ran identity colour (``raw_color=1``), so building C0 from the
       rejected cmatrix would describe a transform the decoder never applied — the
       rung must fall through instead.  The target stays the *same* matrix:
       its rows carry a D65-neutral normalization, and pairing it with an unnormalized
       interpolated DNG matrix would insert those per-channel neutral scales into the
       transform — a hidden white-balance shift.  The pure diagonal rebalance on this
       rung reproduces exactly what LibRaw would have rendered with the target
       multipliers, up to the declared demosaic-order difference.
    3. The file's own DNG dual-illuminant tags: decode C0 is interpolated at the
       as-shot CCT (``wb.asshot_reference_cct``, the DNG-SDK-style fixed point on the
       decode-side multipliers), the target at the declared CCT — both in the same
       unnormalized convention.
    4. The project fallback matrix table (``camera_matrices``) — bodies newer than the
       pinned LibRaw shooting non-DNG containers, where rungs 1-3 all miss.  The same
       single-illuminant matrix serves both sides (no per-CCT interpolation exists on
       this rung; diagonal gains commute through the shared matrix, so the convention
       stays consistent), exactly the rung ``solve_wb_for_mode`` already uses for the
       target multipliers on these bodies.  Salvaged from the parallel session's
       ladder draft — its one increment over the merged fix.
    Missing everything raises ValueError; the caller degrades explicitly to camera.
    """
    candidate = bundle.wb_xyz_to_cam
    if candidate is not None:
        matrix = np.asarray(candidate, dtype=np.float64)
        if (
            matrix.ndim == 2
            and matrix.shape[0] >= 3
            and matrix.shape[1] == 3
            and np.all(np.isfinite(matrix[:3, :3]))
            and float(np.abs(matrix[:3, :3]).sum()) > 1e-9
        ):
            if target_cct is not None:
                calibration = dng_metadata.read_dng_color_calibration(bundle.path)
                if calibration is not None:
                    from .wb import interpolated_color_matrix

                    # Self-review 2026-08-27 (P1): the two sides of a hot-WB
                    # conjugation may differ ONLY in the same D65-row-normalized
                    # convention LibRaw itself renders in. LibRaw does not
                    # interpolate: it selects the daylight ColorMatrix and
                    # row-normalizes it so the balanced camera neutral maps to
                    # D65 (identify.cpp). The decode side must therefore be the
                    # matrix the decoder APPLIED — rgb_cam when LibRaw adopted
                    # the embedded cmatrix, else the evidence matrix in the same
                    # normalization — and the interpolated target at the
                    # declared CCT must be normalized the same way. The former
                    # "interpolate both sides at as-shot/declared CCT" rendered
                    # the declared white at R/G 0.40, B/G 1.41 (5500 K).
                    decode = _libraw_applied_xyz_to_cam(bundle, matrix)
                    target = d65_row_normalize(
                        interpolated_color_matrix(calibration, target_cct)
                    )
                    return decode, target, "evidence+cct"
            return matrix, matrix, "evidence"
    raw_cmatrix = getattr(bundle, "wb_color_matrix", None)
    if raw_cmatrix is not None:
        # LibRaw adoption gate (seam B): only a DNG container whose embedded
        # cmatrix[0][0] > 0.125 ever had this matrix copied into rgb_cam; anything
        # else decoded through identity colour and must fall to the next rung.
        cmatrix = np.asarray(raw_cmatrix, dtype=np.float64)
        adopted = (
            cmatrix.ndim == 2
            and cmatrix.shape[0] >= 1
            and cmatrix.shape[1] >= 1
            and np.isfinite(cmatrix[0, 0])
            and float(cmatrix[0, 0]) > 0.125
            and dng_metadata.is_dng_container(bundle.path)
        )
        if adopted:
            derived = color_matrix_xyz_to_cam(raw_cmatrix)
            if derived is not None:
                return derived, derived, "color_matrix"
    # Rungs 3 and 4 are reached only when LibRaw had neither a table cam_xyz
    # nor an adopted embedded cmatrix — i.e. it decoded with identity colour
    # (raw_color). The decoded "Rec.2020" buffer is then camera RGB, and the
    # only transform the decoder's own output supports is the pure diagonal
    # rebalance G_t * G0^-1 in that space (self-review 2026-08-27, P3):
    # conjugating through a matrix the decoder never applied would mix
    # channels it never mixed. Both sides carry Rec.2020's own XYZ->RGB so
    # hot_wb_matrix_rec2020's C0 collapses to the identity. The declared
    # Kelvin MULTIPLIERS still come from the DNG tags / fallback table via
    # solve_wb_for_mode; only the conjugation is identity.
    identity = np.asarray(XYZ_TO_RGB_REC2020, dtype=np.float64)
    calibration = dng_metadata.read_dng_color_calibration(bundle.path)
    if calibration is not None:
        return identity, identity, "dng_calibration"
    from .camera_matrices import fallback_xyz_to_cam

    hit = fallback_xyz_to_cam(bundle.shot_make, bundle.shot_model)
    if hit is not None:
        return identity, identity, "fallback_table"
    raise ValueError("camera ColorMatrix is unavailable for hot white balance")


def hot_wb_matrix_rec2020(
    xyz_to_cam: Any,
    decode_wb: list[float],
    target_wb: list[float],
    target_xyz_to_cam: Any | None = None,
) -> Any:
    """Rec.2020 matrix changing only user WB after one fixed reconstruction.

    ``xyz_to_cam`` is the fixed decoder ColorMatrix (XYZ -> camera channels).  Let C be
    camera -> Rec.2020 and G the diagonal WB gains.  A decoder scene reconstructed with
    immutable ``C0/G0`` is rebalanced by ``Ctarget Gtarget (C0 G0)^-1``; DNG fixed-Kelvin
    modes may therefore use their white-point-interpolated target matrix.  This is the
    algebraic camera-linear cache boundary: adaptive demosaic/highlight decisions remain
    fixed, while the user-authored balance is a cheap linear hot-stage shared by preview
    and export.
    """
    matrix = np.asarray(xyz_to_cam, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] != 3:
        raise ValueError("camera ColorMatrix is unavailable for hot white balance")
    xyz_to_camera = matrix[:3, :]
    if not np.all(np.isfinite(xyz_to_camera)) or float(np.abs(xyz_to_camera).sum()) <= 1e-9:
        raise ValueError("camera ColorMatrix is empty for hot white balance")
    target_matrix = np.asarray(
        xyz_to_cam if target_xyz_to_cam is None else target_xyz_to_cam,
        dtype=np.float64,
    )
    if target_matrix.ndim != 2 or target_matrix.shape[0] < 3 or target_matrix.shape[1] != 3:
        raise ValueError("target camera ColorMatrix is unavailable for hot white balance")
    target_xyz_to_camera = target_matrix[:3, :]
    if not np.all(np.isfinite(target_xyz_to_camera)) or float(np.abs(target_xyz_to_camera).sum()) <= 1e-9:
        raise ValueError("target camera ColorMatrix is empty for hot white balance")
    camera_to_xyz = np.linalg.pinv(xyz_to_camera)
    target_camera_to_xyz = np.linalg.pinv(target_xyz_to_camera)
    from .constants import XYZ_TO_RGB

    xyz_to_rec2020 = np.asarray(XYZ_TO_RGB["Rec2020"], dtype=np.float64)
    decode_camera_to_rec2020 = xyz_to_rec2020 @ camera_to_xyz
    target_camera_to_rec2020 = xyz_to_rec2020 @ target_camera_to_xyz
    decode_stage = decode_camera_to_rec2020 @ np.diag(normalized_camera_wb(decode_wb))
    target_stage = target_camera_to_rec2020 @ np.diag(normalized_camera_wb(target_wb))
    condition = float(np.linalg.cond(decode_stage))
    if not np.isfinite(condition) or condition > 1e6:
        raise ValueError(f"camera ColorMatrix is ill-conditioned ({condition:.3g})")
    transform = target_stage @ np.linalg.inv(decode_stage)
    result = transform.astype(np.float32)
    result.setflags(write=False)
    return result


def apply_hot_wb_rec2020(scene_rec2020: Any, matrix: Any) -> Any:
    """Apply one hot-WB matrix in bounded chunks, preserving signed headroom."""
    source = np.asarray(scene_rec2020)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("scene Rec.2020 buffer must be HxWx3")
    flat = source[:, :, :3].reshape(-1, 3)
    out = np.empty((flat.shape[0], 3), dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    chunk = 1_000_000
    for start in range(0, flat.shape[0], chunk):
        end = min(start + chunk, flat.shape[0])
        values = flat[start:end].astype(np.float32, copy=False)
        out[start:end, 0] = m[0, 0] * values[:, 0] + m[0, 1] * values[:, 1] + m[0, 2] * values[:, 2]
        out[start:end, 1] = m[1, 0] * values[:, 0] + m[1, 1] * values[:, 1] + m[1, 2] * values[:, 2]
        out[start:end, 2] = m[2, 0] * values[:, 0] + m[2, 1] * values[:, 1] + m[2, 2] * values[:, 2]
    return out.reshape(source.shape[0], source.shape[1], 3)


def wb_window_transport_matrix_rec2020(bundle: RawBundle) -> Any | None:
    """Full Rec.2020 transport moving daylight-calibrated chroma windows into the
    bundle's applied balance frame; None means identity or "fall back to ratios".

    The pixels receive the complete hot-WB matrix Ctarget Gtarget (C0 G0)^-1 — a
    3x3 with channel mixing — so the prefeed windows must move by the same map,
    not by a two-channel von Kries approximation (measured on _SDI0150 + Portra
    400, the diagonal transport left material windows at weights near 1e-16 where
    the true center should weigh 1). The window transport is the current frame's
    matrix composed with the inverse of the daylight (calibration) frame's:
    M = T(decode->applied) . T(decode->daylight)^-1.
    """
    from .wb import kelvin_mode_cct

    try:
        wb_mode = str(getattr(bundle, "wb_mode", "camera") or "camera")
        if wb_mode == "daylight":
            return None
        decode_wb = list(
            getattr(bundle, "decode_wb", None) or getattr(bundle, "camera_wb", None) or []
        )
        daylight_wb = list(getattr(bundle, "daylight_wb", None) or [])
        if len(decode_wb) < 3 or len(daylight_wb) < 3:
            return None
        d_c0, d_ct, _ = resolve_hot_wb_c0(bundle, kelvin_mode_cct("daylight"))
        t_daylight = np.asarray(
            hot_wb_matrix_rec2020(d_c0, decode_wb, daylight_wb, d_ct), dtype=np.float64
        )
        applied_wb = list(getattr(bundle, "applied_wb", None) or [])
        if wb_mode == "camera" or len(applied_wb) < 3 or applied_wb == decode_wb:
            t_current = np.eye(3, dtype=np.float64)
        else:
            c0, ct, _ = resolve_hot_wb_c0(bundle, kelvin_mode_cct(wb_mode))
            t_current = np.asarray(
                hot_wb_matrix_rec2020(c0, decode_wb, applied_wb, ct), dtype=np.float64
            )
        matrix = t_current @ np.linalg.inv(t_daylight)
        if not np.all(np.isfinite(matrix)):
            return None
        if np.allclose(matrix, np.eye(3), atol=1e-6):
            return None
        return matrix
    except (ValueError, AttributeError, TypeError, np.linalg.LinAlgError):
        return None


def rebalance_raw_bundle(bundle: RawBundle, wb_mode: str) -> RawBundle:
    """Derive one user balance without re-reading or re-demosaicing the RAW.

    The input is expected to be the immutable camera/as-shot DecodeContext.  Missing
    calibration degrades visibly to that base instead of silently using an unrelated
    chromatic adaptation.
    """
    if wb_mode not in WB_CHOICES:
        raise ValueError(f"unknown wb mode: {wb_mode}")
    if wb_mode == "camera":
        # Preserve the fixed decoder codes exactly.  Compact disk entries intentionally
        # omit XYZ, which is fine because the camera BalanceContext also keeps its
        # persisted full-resolution Analysis and never needs a scene-only reanalysis.
        # An explicit camera request also invalidates any stale degradation note from a
        # previous non-camera balance: the user chose AsShot, nothing degraded.  (The
        # degraded fallbacks below are different: they return camera pixels WITH their
        # note, because there the camera result is a truthfully-declared downgrade.)
        return replace(
            bundle,
            wb_mode="camera",
            applied_wb=list(bundle.camera_wb),
            wb_degradation=None,
        )
    decode_wb = list(bundle.decode_wb or bundle.camera_wb)
    if wb_mode == "daylight":
        target_wb = list(bundle.daylight_wb or [])
        note = None if target_wb else "LibRaw daylight multipliers unavailable; degraded to camera AsShot"
    else:
        target_wb, note = solve_wb_for_mode(
            wb_mode,
            bundle.path,
            bundle.wb_xyz_to_cam,
            make=bundle.shot_make,
            model=bundle.shot_model,
        )
        target_wb = list(target_wb or [])
    if not target_wb:
        return replace(
            bundle,
            wb_mode="camera",
            applied_wb=list(bundle.camera_wb),
            wb_degradation=note,
        )
    try:
        decode_xyz_to_cam, target_xyz_to_cam, _c0_source = resolve_hot_wb_c0(
            bundle, kelvin_mode_cct(wb_mode)
        )
        transform = hot_wb_matrix_rec2020(
            decode_xyz_to_cam,
            decode_wb,
            target_wb,
            target_xyz_to_cam,
        )
    except ValueError as exc:
        degradation = f"声明 {wb_mode} 白平衡不可用（{exc}）；已退化为相机 AsShot"
        return replace(
            bundle,
            wb_mode="camera",
            applied_wb=list(bundle.camera_wb),
            wb_degradation=degradation,
        )

    scene = apply_hot_wb_rec2020(bundle.scene_rec2020_render, transform)
    xyz = scene_rec2020_to_xyz_render(scene, bundle.scene_scale)
    # R2 item 20: the stored full-resolution tone-plan sample is scene pixels
    # in the same storage domain, so the SAME hot-WB transform applies — a
    # replace() copy alone would compile balanced previews from the as-shot
    # statistics.
    tone_sample = getattr(bundle, "_tone_plan_sample", None)
    if tone_sample is not None:
        tone_sample = apply_hot_wb_rec2020(
            np.asarray(tone_sample)[None, :, :], transform
        )[0]
    reference = bundle.scene_reliable_reference_rec2020
    if reference is not None and np.asarray(reference).size:
        reference = apply_hot_wb_rec2020(
            np.asarray(reference)[None, :, :], transform
        )[0]
    return replace(
        bundle,
        scene_rec2020_render=scene,
        xyz_render=xyz,
        render_scale=bundle.scene_scale,
        wb_mode=wb_mode,
        applied_wb=[float(value) for value in target_wb],
        wb_degradation=note,
        _tone_plan_sample=tone_sample,
        scene_reliable_reference_rec2020=reference,
        _clip_masks_cache_shape=None,
        _clip_masks_resized=None,
        _raw_guidance_cache_shape=None,
        _raw_guidance_resized=None,
    )


def resolve_demosaic_algorithm(raw: Any, requested: str) -> Any:
    """Pick a DemosaicAlgorithm for the full-res export, or None (libraw default).

    Non-Bayer sensors (e.g. X-Trans) keep libraw's native path. 'auto' takes the best
    available Bayer detail algorithm (DHT preferred); an explicit request is honored when
    the build supports it, else it falls back to auto."""
    if rawpy is None:
        return None
    pattern = getattr(raw, "raw_pattern", None)
    is_bayer = pattern is not None and getattr(pattern, "shape", None) == (2, 2)
    if not is_bayer:
        return None

    def supported(name: str) -> Any:
        alg = getattr(rawpy.DemosaicAlgorithm, name.upper(), None)
        if alg is not None and getattr(alg, "isSupported", False):
            return alg
        return None

    if requested and requested != "auto":
        chosen = supported(requested)
        if chosen is not None:
            return chosen
    for name in DEMOSAIC_AUTO_PREFERENCE:
        chosen = supported(name)
        if chosen is not None:
            return chosen
    return None


def render_to_scene_rec2020(
    raw: Any,
    highlight_mode_name: str = "clip",
    half_size: bool = False,
    demosaic: Any = None,
    wb_kwargs: dict[str, Any] | None = None,
    *, camera_rgb: bool = False, calibration_kwargs: dict[str, Any] | None = None,
) -> Any:
    if not hasattr(rawpy.ColorSpace, "Rec2020"):
        raise RuntimeError("rawpy.ColorSpace.Rec2020 is not available; cannot make scene-linear export buffer")
    return raw.postprocess(
        output_color=rawpy.ColorSpace.raw if camera_rgb else rawpy.ColorSpace.Rec2020,
        gamma=(1, 1),
        half_size=half_size,
        demosaic_algorithm=(None if half_size else demosaic),
        no_auto_bright=True,
        adjust_maximum_thr=0.0,
        highlight_mode=rawpy_highlight_mode(highlight_mode_name),
        output_bps=16,
        user_flip=0 if camera_rgb else None,
        **(calibration_kwargs or {}),
        **(wb_kwargs or {"use_camera_wb": True}),
    )


def channel_label(color_desc: str, cid: int) -> str:
    if 0 <= int(cid) < len(color_desc):
        return color_desc[int(cid)].upper()
    return str(cid)


def _mosaic_loss_rgb(loss: Any, colors: Any, color_desc: str) -> Any:
    """Conservative 2x2 reduction of a one-byte per-sensel processing log."""
    if loss.ndim == 3:
        return loss[..., :3].astype(np.float16)
    h, w = loss.shape
    out = np.zeros(((h + 1)//2, (w + 1)//2, 3), dtype=np.float16)
    for r in range(2):
        for c in range(2):
            plane = loss[r::2, c::2]
            ids = colors[r::2, c::2]
            for cid in np.unique(ids):
                label = channel_label(color_desc, int(cid))[:1]
                if label in "RGB":
                    dest = out[:plane.shape[0], :plane.shape[1], "RGB".index(label)]
                    np.maximum(dest, (plane != 0) & (ids == cid), out=dest)
    return out


def _resize_loss_to_shape(mask: Any, shape: tuple[int, int]) -> Any:
    """Nearest expansion keeps a recorded clipping event at full strength."""
    if mask.shape[:2] == shape:
        return mask
    from PIL import Image
    out = np.empty((*shape, 3), dtype=np.float16)
    for c in range(3):
        plane = Image.fromarray(np.asarray(mask[..., c], dtype=np.float32))
        out[..., c] = np.asarray(plane.resize((shape[1], shape[0]), Image.Resampling.NEAREST))
    return out


def _decode_corrected_libraw(raw: Any, path: Path, evidence: Any, highlight: str,
                             half_size: bool, demosaic: Any, *, track_loss: bool = True):
    """One production recipe, also used by the Core Image scale reference."""
    from . import dng_opcodes as ops
    recipe = ops.read_plan(path)
    from . import embedded_lens
    lens = embedded_lens.read(path)
    if lens is not None:
        recipe.post.extend((lens.vignette,lens.warp))
        recipe.names.extend((lens.source+" shading",lens.source+" distortion/TCA"))
    loss = np.zeros(evidence.raw_image.shape, dtype=np.uint8) if track_loss and (recipe.stage1 or recipe.stage2 or evidence.spatial_black is not None) else None
    from . import dng_point_ops
    # Area/pitch/table coordinates are sensor pixels. Reduced demosaic would
    # discard them before a stage-3 point transform, so process that uncommon
    # recipe at full resolution and reduce only the finished scene.
    reduce_after_ops = half_size and any(isinstance(op,(dng_point_ops.PointOp,ops.TrimBounds)) for op in recipe.post)
    if recipe.stage1:
        from .spatial_black import sensor_tags
        tags = sensor_tags(path,{50712})
        table=np.asarray(tags.get(50712,()),dtype=np.uint16)
        image=raw.raw_image
        if table.size:
            if np.any(np.diff(table.astype(np.int64))<=0):
                raise ValueError("stage-1 opcodes need original codes; LibRaw's non-invertible LinearizationTable has discarded them")
            for y in range(0,image.shape[0],128):
                band=image[y:y+128]
                index=np.searchsorted(table,band)
                if np.any(index>=table.size) or not np.array_equal(table[index],band):
                    raise ValueError("cannot recover DNG stage-1 codes from LinearizationTable")
                band[:]=index
        full_loss=np.zeros(image.shape,dtype=np.uint8) if track_loss else None
        for op in recipe.stage1:
            if isinstance(op,dng_point_ops.BadPixels):
                dng_point_ops.repair_bad_pixels(raw,op,loss)
            else:
                dng_point_ops.apply(image,op,loss=full_loss)
        if full_loss is not None:
            y,x=raw.sizes.top_margin,raw.sizes.left_margin
            visible_loss=full_loss[y:y+loss.shape[0],x:x+loss.shape[1]]
            if visible_loss.ndim==3:visible_loss=visible_loss[...,:3]
            np.maximum(loss,visible_loss,out=loss)
        if table.size:
            for y in range(0,image.shape[0],128):
                image[y:y+128]=table[np.minimum(image[y:y+128],table.size-1)]
        if loss is not None:
            levels=np.asarray(list(recipe.white_levels) or evidence.camera_white_levels
                              or [evidence.white_level],dtype=np.float32)
            work=raw.raw_image_visible
            if work.ndim==3:work=work[...,:3]
            for y in range(0,work.shape[0],128):
                cid=np.minimum(evidence.raw_colors[y:y+128],levels.size-1)
                wl=levels[cid]
                loss[y:y+128] |= ((evidence.raw_image[y:y+128]<wl)&(work[y:y+128]>=wl)).astype(np.uint8)
    working_black = evidence.black_levels
    working_white = list(recipe.white_levels) or evidence.camera_white_levels or [evidence.white_level]
    calibration_kwargs = None
    if evidence.spatial_black is not None:
        from .spatial_black import apply_to_working
        calibration_kwargs = apply_to_working(raw, evidence.spatial_black,
            evidence.black_levels, list(recipe.white_levels) or [evidence.white_level], loss)
        working_black, working_white = [0.] * 4, [65535.] * 4
        recipe.names.insert(0, "SpatialBlackLevel")
    for op in recipe.stage2:
        if isinstance(op,dng_point_ops.PointOp):
            img=raw.raw_image_visible
            levels=working_white
            levels=(levels*4)[:4]
            dng_point_ops.apply(img,op,black=working_black,white=levels,
                               colors=np.asarray(raw.raw_colors_visible) if img.ndim==2 else None,loss=loss)
        else:
            _apply_gain_maps_mosaic(raw, [op], working_black,
                                   65535 if calibration_kwargs else evidence.white_level,
                                   working_white, loss)
    processing = _mosaic_loss_rgb(loss, evidence.raw_colors, evidence.color_desc) if loss is not None else None
    del loss
    # Colour mixing must follow camera-plane opcodes. The as-shot WB is diagonal,
    # so it commutes with the per-plane warp; keep LibRaw's demosaic/reconstruction.
    camera_rgb = True
    scene = render_to_scene_rec2020(raw, highlight, half_size and not reduce_after_ops, demosaic,
                                   _fixed_asshot_wb_kwargs(evidence.camera_wb), camera_rgb=camera_rgb,
                                   calibration_kwargs=calibration_kwargs)
    if scene.ndim != 3 or scene.shape[2] != 3:
        raise ValueError("DNG corrections require three camera colour planes")
    if processing is not None:
        # DefaultScale and half-size affect the *input* geometry of every
        # subsequent warp, so transport the raster before resampling it.
        processing = _resize_loss_to_shape(processing, scene.shape[:2])
    # LibRaw has already applied DefaultScale (pixel_aspect) before returning
    # this buffer. The remaining pixel aspect is one; applying the DNG aspect
    # a second time distorts both the image and its evidence footprints.
    recipe.post = [replace(op, aspect=1.0) if isinstance(op, ops.Warp) else op
                   for op in recipe.post]
    flip = evidence.orientation_flip
    inverse = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 5, 7: 7}
    if not camera_rgb:
        scene = _orient_like_libraw(scene, inverse[flip])
    # DNG stage-3 clips normalized camera planes at one. LibRaw's fixed WB
    # commutes with a per-plane spatial operator, but its clipping bounds
    # must be scaled too (scale_colors uses min(WB) for clip, max otherwise).
    wb = np.asarray(evidence.camera_wb[:4], dtype=np.float64)
    positive_wb = wb[np.isfinite(wb) & (wb > 0)]
    if camera_rgb and (wb.size < 3 or not np.all(np.isfinite(wb[:3]) & (wb[:3] > 0))):
        raise ValueError("DNG camera-plane corrections require explicit positive as-shot WB")
    limits = (np.minimum(65535., 65535. * wb[:3] /
                        (positive_wb.min() if highlight == "clip" else positive_wb.max()))
              if camera_rgb else None)
    shading = []
    if recipe.gain_maps:
        shading.append("gainmap")
    terminal_trims=[]
    for op in recipe.post:
        if terminal_trims and not isinstance(op,ops.TrimBounds):
            raise ValueError("DNG operations after TrimBounds require retained image-origin coordinates; use Apple RAW")
        if isinstance(op,ops.TrimBounds):
            if scene.shape[:2] != evidence.raw_image.shape[:2]:
                raise ValueError("stage-3 TrimBounds coordinates require a full-resolution square-pixel decode")
            terminal_trims.append(op)
        elif isinstance(op,dng_point_ops.PointOp):
            if scene.shape[:2] != evidence.raw_image.shape[:2]:
                raise ValueError("stage-3 point-op coordinates require a full-resolution square-pixel decode")
            if processing is None and track_loss:processing=np.zeros(scene.shape,dtype=np.float16)
            dng_point_ops.apply(scene,op,black=0.,white=limits,loss=processing)
        elif isinstance(op, ops.Warp):
            scene = ops.warp_image(scene, op)
            if track_loss:
                # A zero raster still records out-of-frame extrapolation as loss.
                if processing is None:
                    processing = np.zeros((*scene.shape[:2], 3), dtype=np.float16)
                processing = ops.warp_image(processing, op, loss=True)
                processing = _resize_loss_to_shape(processing, scene.shape[:2])
            for y in range(0, scene.shape[0], 128):
                band = scene[y:y+128]
                if processing is not None:
                    np.maximum(processing[y:y+128], band >= limits, out=processing[y:y+128])
                band[:] = np.minimum(band, limits).astype(np.uint16)
        else:
            if track_loss:
                processing = (np.zeros(scene.shape, dtype=np.float16) if processing is None
                              else _resize_loss_to_shape(processing, scene.shape[:2]))
            if isinstance(op, embedded_lens.RadialVignette):
                embedded_lens.apply_vignette(scene, op, processing, limits)
            else:
                _apply_vignette_render(scene, op, loss_mask=processing, channel_limits=limits)
            shading.append("vignette")
    if terminal_trims:
        recipe.crop=ops.terminal_trim_crop(terminal_trims,evidence.raw_image.shape,recipe.crop)
    if camera_rgb:
        scene = ops.camera_to_rec2020(scene, ops.libraw_camera_matrix(
            raw.color_matrix, raw.rgb_xyz_matrix,
            is_dng=dng_metadata.is_dng_container(path)))
    scene = ops.crop_image(scene, recipe.crop, evidence.raw_image.shape)
    # Match rawpy's contiguous handoff. A cropped/transposed view would make
    # every downstream reshape(-1, 3) silently copy the complete frame again.
    scene = np.ascontiguousarray(_orient_like_libraw(scene, flip))
    if processing is not None:
        processing = ops.crop_image(processing, recipe.crop, evidence.raw_image.shape)
        processing = _orient_like_libraw(processing, flip)
    if reduce_after_ops:
        # Box reduction is bounded to row bands. Permission takes maximum.
        h,w=scene.shape[:2];hh,ww=h//2,w//2
        reduced=np.empty((hh,ww,3),dtype=np.float32)
        for y in range(0,hh,64):
            end=min(y+64,hh)
            reduced[y:end]=scene[2*y:2*end,:2*ww].reshape(end-y,2,ww,2,3).mean(axis=(1,3))
        scene=reduced
        if processing is not None:
            processing=_bin_2x2_max(processing)
        if terminal_trims and recipe.crop is not None:
            # Box reduction drops an unmatched final row/column. Record the
            # actual retained sensor window so independently rebuilt masks and
            # Apple reference samples cannot stretch it over the discarded edge.
            cy,cx,_,_=recipe.crop
            recipe.crop=(float(round(cy)),float(round(cx)),float(2*hh),float(2*ww))
    return scene, processing, recipe, "+".join(shading) or None


def _merge_processing_loss(masks: Any, processing: Any | None) -> Any:
    if processing is not None:
        # Only the final feathered half mask is mutable here. Preserve the old
        # path for aliases, unsupported layouts and dtypes without an implicit
        # full-frame copy to make them eligible for the native kernel.
        if (isinstance(masks, np.ndarray) and isinstance(processing, np.ndarray)
                and masks.ndim == 3 and masks.shape[2] == 3
                and processing.ndim == 3 and processing.shape[2] == 3
                and masks.dtype == np.float16
                and processing.dtype in (np.dtype(np.float16), np.dtype(np.float32))
                # NumPy's mixed f32/half SIMD maximum is faster when no resize
                # is needed; only the half/half or fused-resize cases dispatch.
                and (processing.dtype == np.float16 or masks.shape[:2] != processing.shape[:2])
                and masks.flags.c_contiguous and masks.flags.writeable
                and masks.flags.aligned and processing.flags.aligned
                and min(*masks.shape[:2], *processing.shape[:2]) > 0
                and max(*processing.shape[:2]) <= np.iinfo(np.int32).max
                and not np.may_share_memory(masks, processing)):
            from . import _fast
            native = _fast.kernel("merge_processing_loss_f16_inplace")
            if native is not None:
                try:
                    if masks.shape[:2] == processing.shape[:2]:
                        # Same-size half/half maximum keeps the original bits.
                        result = native(masks, processing)
                    else:
                        # Pillow chooses the nearest source coordinate. Mode-I
                        # one-dimensional strips reproduce those exact choices
                        # using O(H+W) scratch, rather than resizing three planes.
                        # The kernel rounds sampled processing to half BEFORE
                        # max, matching _resize_loss_to_shape's half output.
                        from PIL import Image
                        sh, sw = processing.shape[:2]
                        h, w = masks.shape[:2]
                        yi = Image.fromarray(np.arange(sh, dtype=np.int32)[:, None])
                        xi = Image.fromarray(np.arange(sw, dtype=np.int32)[None, :])
                        y_indices = np.asarray(yi.resize((1, h), Image.Resampling.NEAREST),
                                               dtype=np.intp).reshape(-1)
                        x_indices = np.asarray(xi.resize((w, 1), Image.Resampling.NEAREST),
                                               dtype=np.intp).reshape(-1)
                        result = native(masks, processing, y_indices, x_indices)
                    # The native preflight may decline special numeric cases
                    # without writing anything. That is a supported fallback in
                    # strict mode too; it is distinct from a kernel exception.
                    if result is not None:
                        return masks
                except Exception as exc:
                    _fast.handle_kernel_error("merge_processing_loss_f16_inplace", exc)
        # One float16 raster at most; never manufacture a full float32 RGB copy.
        np.maximum(masks, _resize_loss_to_shape(processing, masks.shape[:2]), out=masks)
    return masks


def channel_black_level(black_levels: list[float], cid: int) -> float:
    if black_levels:
        return float(black_levels[int(cid) % len(black_levels)])
    return 0.0


def channel_fullwell(white_level: int, camera_white_levels: list[float], cid: int) -> float:
    if camera_white_levels and int(cid) < len(camera_white_levels) and camera_white_levels[int(cid)] > 0:
        return float(camera_white_levels[int(cid)])
    return float(white_level)


def _smoothstep(edge0: float, edge1: float, x: Any) -> Any:
    t = np.clip((x - np.float32(edge0)) / np.float32(max(edge1 - edge0, 1e-9)), 0.0, 1.0)
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


def _bin_2x2_max(mask: Any) -> Any:
    h, w = mask.shape[:2]
    h2 = max(1, h // 2)
    w2 = max(1, w // 2)
    cropped = mask[: h2 * 2, : w2 * 2]
    return cropped.reshape(h2, 2, w2, 2, mask.shape[2]).max(axis=(1, 3))


def _orient_like_libraw(arr: Any, flip: int) -> Any:
    """Apply LibRaw's ``flip`` code the way ``postprocess`` does.

    dcraw/LibRaw flip bits: 1 = mirror horizontally, 2 = mirror vertically,
    4 = transpose, and the common combinations 3 (180 deg), 5 (90 deg CCW
    here, i.e. rot90 k=1 on the row-major array), 6 (90 deg CW), 7 (transpose
    + 180). Self-review 2026-08-27: codes 1/2/4 were previously mapped with
    EXIF orientation semantics (1 = unchanged, 2 = mirror-H, 4 = mirror-V),
    which disagrees with rawpy.postprocess(user_flip=k) — verified on
    _SDI0150 for every code. No shipped camera writes 1/2/4, so the corpus
    was unaffected. The non-LibRaw code 8 is no longer accepted.
    """
    flip = int(flip or 0)
    if flip == 0:
        return arr
    if flip == 1:
        return np.fliplr(arr)
    if flip == 2:
        return np.flipud(arr)
    if flip == 3:
        return np.rot90(arr, 2)
    if flip == 4:
        return np.transpose(arr, (1, 0, 2)) if arr.ndim == 3 else np.transpose(arr)
    if flip == 5:
        return np.rot90(arr, 1)
    if flip == 6:
        return np.rot90(arr, 3)
    if flip == 7:
        return np.fliplr(np.rot90(arr, 1))
    raise ValueError(f"unknown LibRaw flip code {flip}")


def _resize_mask_to_shape(mask: Any, shape: tuple[int, int]) -> Any:
    target_h, target_w = shape
    if mask.shape[:2] == (target_h, target_w):
        return mask
    from PIL import Image

    out = np.empty((target_h, target_w, mask.shape[2]), dtype=np.float32)
    for idx in range(mask.shape[2]):
        im = Image.fromarray(mask[:, :, idx].astype(np.float32, copy=False), mode="F")
        im = im.resize((target_w, target_h), Image.Resampling.BILINEAR)
        out[:, :, idx] = np.asarray(im, dtype=np.float32)
    return out


def _feather_masks_f16(mask: Any) -> Any:
    """Row-banded feather that writes the float16 result directly.

    Stage 1 (2026-09-15): the Rust kernel `feather_masks_f16` reproduces this
    function bit for bit (tests/test_rust_stage1.py) and is used under the
    shared native policy; this NumPy body is the reference implementation.

    Scheduler plan S4: the whole-frame version held the aligned float32
    mask, a float32 output, a clipped copy and four per-channel temporaries
    at full resolution — 1.37 GB of the measured decode peak at 24 MP. The
    filter is separable and local (radius 2), so a band that gathers its own
    two-row halo (edge-clamped at the true frame boundary, exactly as the
    whole-frame pad did) reproduces every element bit for bit.
    """
    from . import _fast

    native = _fast.kernel("feather_masks_f16")
    if native is not None:
        return native(np.ascontiguousarray(mask, dtype=np.float32))
    kernel = np.asarray([1, 4, 6, 4, 1], dtype=np.float32) / np.float32(16.0)
    radius = len(kernel) // 2
    h, w, channels = mask.shape
    out = np.empty((h, w, channels), dtype=np.float16)
    band = max(1, 8_000_000 // max(w, 1))
    for y0 in range(0, h, band):
        y1 = min(y0 + band, h)
        rows = np.clip(np.arange(y0 - radius, y1 + radius), 0, h - 1)
        for channel in range(channels):
            plane = mask[rows, :, channel].astype(np.float32, copy=False)
            acc = np.zeros((y1 - y0, w), dtype=np.float32)
            scratch = np.empty_like(acc)
            for i, weight in enumerate(kernel):
                np.multiply(plane[i:i + (y1 - y0)], np.float32(weight), out=scratch)
                np.add(acc, scratch, out=acc)
            padded = np.pad(acc, ((0, 0), (radius, radius)), mode="edge")
            band_out = np.zeros_like(acc)
            for i, weight in enumerate(kernel):
                np.multiply(padded[:, i:i + w], np.float32(weight), out=scratch)
                np.add(band_out, scratch, out=band_out)
            out[y0:y1, :, channel] = np.clip(band_out, 0.0, 1.0).astype(np.float16)
    return out


def _feather_masks(mask: Any) -> Any:
    # Small separable Gaussian-like kernel, enough to hide demosaic/half-size seams.
    kernel = np.asarray([1, 4, 6, 4, 1], dtype=np.float32) / np.float32(16.0)
    radius = len(kernel) // 2
    source = mask.astype(np.float32, copy=False)
    out = np.empty_like(source, dtype=np.float32)
    for channel in range(source.shape[2]):
        plane = source[:, :, channel]
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (radius, radius)
            padded = np.pad(plane, pad, mode="edge")
            acc = np.zeros_like(plane, dtype=np.float32)
            scratch = np.empty_like(plane, dtype=np.float32)
            for i, weight in enumerate(kernel):
                sl = [slice(None), slice(None)]
                sl[axis] = slice(i, i + plane.shape[axis])
                np.multiply(padded[tuple(sl)], np.float32(weight), out=scratch)
                np.add(acc, scratch, out=acc)
            plane = acc
        out[:, :, channel] = plane
    return np.clip(out, 0.0, 1.0)


def _build_bayer_clip_mask_planes(
    raw_image: Any,
    raw_pattern: Any,
    color_desc: str,
    white_level: int,
    black_levels: list[float],
    camera_white_levels: list[float],
) -> Any:
    """Build the 2x2-binned mask directly from Bayer planes.

    This is equivalent to constructing a full-resolution RGB mask and taking a
    2x2 maximum, but avoids the much larger intermediate arrays.
    """
    pattern = np.asarray(raw_pattern)
    if pattern.shape != (2, 2):
        return None
    h2 = raw_image.shape[0] // 2
    w2 = raw_image.shape[1] // 2
    if h2 == 0 or w2 == 0:
        return None
    binned = np.zeros((h2, w2, 3), dtype=np.float32)
    for row in range(2):
        for col in range(2):
            cid = int(pattern[row, col])
            label = channel_label(color_desc, cid)
            if label.startswith("R"):
                out_idx = 0
            elif label.startswith("G"):
                out_idx = 1
            elif label.startswith("B"):
                out_idx = 2
            else:
                continue
            black = channel_black_level(black_levels, cid)
            fullwell = channel_fullwell(white_level, camera_white_levels, cid)
            denom = max(fullwell - black, 1.0)
            plane = raw_image[
                row : row + h2 * 2 : 2,
                col : col + w2 * 2 : 2,
            ].astype(np.float32, copy=False)
            raw_norm = (plane - np.float32(black)) / np.float32(denom)
            channel_soft = _smoothstep(0.95, 0.99, raw_norm)
            np.maximum(binned[:, :, out_idx], channel_soft, out=binned[:, :, out_idx])
    return binned


def build_clip_masks(
    raw_image: Any,
    raw_colors: Any,
    color_desc: str,
    white_level: int,
    black_levels: list[float],
    camera_white_levels: list[float],
    orientation_flip: int,
    scene_shape: tuple[int, int],
    raw_pattern: Any | None = None,
    geometry_ops: tuple = (),
    crop_sensor: tuple | None = None,
    spatial_black: Any | None = None,
) -> Any:
    """Build half-resolution RGB soft clip masks from pre-WB raw DN values."""
    if spatial_black is not None:
        from .spatial_black import clip_mask
        levels=[channel_fullwell(white_level,camera_white_levels,int(c)) for c in range(4)]
        binned=clip_mask(raw_image,raw_colors,color_desc,black_levels,levels,spatial_black)
    elif raw_image.ndim == 3:
        soft = np.empty(raw_image.shape, dtype=np.float32)
        for c in range(3):
            black = channel_black_level(black_levels, c)
            white = channel_fullwell(white_level, camera_white_levels, c)
            soft[..., c] = _smoothstep(.95, .99,
                (raw_image[..., c].astype(np.float32) - black) / max(white - black, 1.0))
        binned = _bin_2x2_max(soft)
    else:
        binned = _build_bayer_clip_mask_planes(
        raw_image,
        raw_pattern,
        color_desc,
        white_level,
        black_levels,
        camera_white_levels,
    )
    if binned is None:
        h, w = raw_image.shape[:2]
        soft = np.zeros((h, w, 3), dtype=np.float32)
        for cid in np.unique(raw_colors):
            cid_int = int(cid)
            label = channel_label(color_desc, cid_int)
            if label.startswith("R"):
                out_idx = 0
            elif label.startswith("G"):
                out_idx = 1
            elif label.startswith("B"):
                out_idx = 2
            else:
                continue
            black = channel_black_level(black_levels, cid_int)
            fullwell = channel_fullwell(white_level, camera_white_levels, cid_int)
            denom = max(fullwell - black, 1.0)
            raw_norm = (raw_image.astype(np.float32, copy=False) - np.float32(black)) / np.float32(denom)
            channel_soft = _smoothstep(0.95, 0.99, raw_norm)
            soft[:, :, out_idx] = np.maximum(
                soft[:, :, out_idx], np.where(raw_colors == cid_int, channel_soft, 0.0)
            )
        binned = _bin_2x2_max(soft)
    from .dng_opcodes import align_sensor_loss
    oriented = align_sensor_loss(binned, raw_image.shape, scene_shape,
                                 orientation_flip, geometry_ops, crop_sensor)
    aligned = _resize_mask_to_shape(oriented, scene_shape)
    return _feather_masks_f16(aligned)


def release_analysis_buffers(bundle: RawBundle) -> RawBundle:
    """Drop the analysis-stage buffer a render never reads (plan S4).

    Ownership boundary: ``xyz_render`` exists for ``analyze`` /
    ``reanalyze_balanced_scene`` / the diagnostic dashboard, and nothing in
    the render or export path reads it. Releasing it after those have run
    returns its pages to the allocator for the export stage to reuse
    (140 MB at 12 MP, ~290 MB at 24 MP). MEASURED HONESTLY: the process
    RSS high-water mark does NOT fall, because it is set later by the HDR
    encode stage — this bounds the live heap, it does not lower the peak.

    The CFA mosaic is deliberately NOT released here: ``RawEvidence``
    retains the same arrays, so clearing the bundle's reference was
    measured to free exactly nothing (and the gated core reads them at
    render time anyway).
    """
    from dataclasses import replace

    return replace(bundle, xyz_render=None)


def refresh_clip_masks_from_fullwell(
    bundle: RawBundle, channel_fullwell: dict[int, int]
) -> bool:
    """Rebuild LibRaw's soft headroom mask when analysis found a real saturation pile.

    load_raw needs an initial mask before analysis exists, so it starts from metadata
    per-channel white levels. Once analysis has a trustworthy observed full well, the
    render permission map must use that same per-channel endpoint; otherwise hard clip
    statistics and near-clip color retreat can disagree on cameras whose metadata white
    is inaccurate. Returns whether a rebuild was needed.
    """
    if getattr(bundle, "scene_decoder", "libraw") != "libraw":
        return False
    if getattr(bundle, "clip_masks", None) is None or not channel_fullwell:
        return False
    channel_ids = [int(x) for x in sorted(np.unique(bundle.raw_colors).tolist())]
    metadata_levels = {
        cid: int(
            bundle.camera_white_levels[cid]
            if cid < len(bundle.camera_white_levels)
            and bundle.camera_white_levels[cid] > 0
            else bundle.white_level
        )
        for cid in channel_ids
    }
    resolved = {
        cid: int(channel_fullwell.get(cid, metadata_levels[cid])) for cid in channel_ids
    }
    current = getattr(bundle, "_clip_mask_fullwell", None) or metadata_levels
    if resolved == current:
        return False
    resolved_levels = [0.0] * (max(channel_ids) + 1 if channel_ids else 0)
    for cid in channel_ids:
        resolved_levels[cid] = float(channel_fullwell.get(cid, metadata_levels[cid]))
    bundle.clip_masks = build_clip_masks(
        bundle.raw_image,
        bundle.raw_colors,
        bundle.color_desc,
        bundle.white_level,
        bundle.black_levels,
        resolved_levels,
        bundle.orientation_flip,
        bundle.scene_rec2020_render.shape[:2],
        bundle.raw_pattern,
        getattr(bundle, "scene_geometry_ops", ()),
        getattr(bundle, "scene_crop_sensor", None),
        getattr(getattr(bundle,"evidence",None),"spatial_black",None),
    )
    _merge_processing_loss(bundle.clip_masks, getattr(bundle, "processing_clip_masks", None))
    bundle._clip_masks_cache_shape = None
    bundle._clip_masks_resized = None
    bundle._clip_mask_fullwell = resolved
    bundle.raw_guidance = None
    bundle._raw_guidance_cache_shape = None
    bundle._raw_guidance_resized = None
    bundle._raw_guidance_has_sensor_snr = False
    bundle._raw_guidance_has_resolved_fullwell = False
    return True


def _unsupported_format_guidance(path: Path, shot: Any, exc: Exception) -> str:
    """A targeted refusal for files LibRaw cannot open — name the cause, give outs.

    New-body decode failures split into two classes. Colour-table gaps degrade
    gracefully elsewhere (SENSOR_SUPPORT ladder); FORMAT gaps stop the decoder
    cold, and the honest response is a precise diagnosis instead of a generic
    "unsupported". The canonical case: Nikon High Efficiency (HE/HE*) NEFs use
    intoPIX TicoRAW, which LibRaw (and darktable's rawspeed) cannot licence —
    even LibRaw master fails on them, so the fallback matrix table cannot help.
    """
    ident = f"{shot.make or '?'} {shot.model or '?'}".strip()
    lines = [
        f"LibRaw 无法打开此文件（机型 {ident}，{path.suffix or '无后缀'}）：{exc}",
    ]
    if path.suffix.lower() == ".nef":
        lines += [
            "若这是较新的尼康机身（Z9/Z8/Z6III/Z50II 世代）且拍摄时选择了"
            "高效压缩（HE/HE*），则该格式使用 intoPIX TicoRAW 编码，LibRaw "
            "因授权无法解码——升级 LibRaw 也无济于事。可用的出路：",
            "  1. 用 Adobe DNG Converter（免费，支持 HE）把 NEF 转成 DNG，"
            "转换后本工具全功能可用；",
            "  2. 相机内改用『无损压缩』RAW（同世代机身的无损 NEF 可正常解码）；",
            "  3. 可尝试 --decoder coreimage --coreimage-version auto；若 Apple 支持该文件，"
            "可导出场景图像，但无 LibRaw 传感器证据时不会声明 RAW 剪切/SNR 测量。",
        ]
    else:
        lines += [
            "若这是较新的机型，可尝试 tools/build_libraw_master.sh 升级到 "
            "LibRaw master 快照；仍失败请反馈样张（机型支持策略见 "
            "docs/SENSOR_SUPPORT.zh-CN.md）。",
        ]
    lines.append("用 `--support` 可查看此文件在两条解码线上的逐档支持报告。")
    return "\n".join(lines)


def load_raw(
    path: Path,
    scene_highlight_mode: str = "clip",
    scene_half_size: bool = False,
    demosaic: str = "auto",
    wb_mode: str = "camera",
    decoder: str = "libraw",
    coreimage_version: str = "auto",
    coreimage_scale: str = COREIMAGE_SCALE_DEFAULT_MODE,
) -> RawBundle:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {path}")
    if decoder not in DECODER_CHOICES:
        raise ValueError(f"unknown decoder: {decoder}; expected one of {DECODER_CHOICES}")
    if wb_mode not in WB_CHOICES:
        raise ValueError(f"unknown wb mode: {wb_mode}")
    requested_wb_mode = wb_mode
    rawpy_highlight_mode(scene_highlight_mode)
    # CIRAWFilter exposes one calibrated reconstruction path rather than LibRaw's
    # clip/blend/reconstruct switch. Its comparison reference must always use
    # reconstruction too, including direct Python API calls that kept the old "clip"
    # default; otherwise the reported mode and the scale calculation describe different
    # pipelines.
    effective_highlight_mode = (
        "reconstruct" if decoder == "coreimage" else scene_highlight_mode
    )
    shot = dng_metadata.read_dng_shot_info(path)

    scene_decoder = "libraw"
    scene_decoder_version: str | None = None
    scene_decoder_runtime: str | None = None
    scene_scale_mode: str | None = None
    scene_align_factor: float = 1.0
    scene_align_error: str | None = None
    scene_opcode_names: tuple[str, ...] = ()
    evidence_shape: tuple[int, int] | None = None
    scene_geometry_crop: tuple[float, float, float, float] | None = None
    scene_geometry_corr: float | None = None
    scene_rec2020_render: Any | None = None
    xyz_render: Any | None = None
    scene_scale = 1.0
    render_scale = 1.0
    clip_masks: Any | None = None
    lens_shading: str | None = None
    processing_clip_masks = None
    scene_geometry_ops = ()
    scene_crop_sensor = None
    scene_correction_note = None
    scene_processing_loss_pct = 0.0 if decoder == "libraw" else None
    effective_baseline_exposure = shot.baseline_exposure
    baseline_exposure_baked_in = False
    evidence_error = None
    scene_reference_error = None
    scene_decoder_fallback = None
    reliable_reference = None
    reliable_reference_pct = None
    reliability_source = "sensor-spatial"

    # Evidence is acquired before scene decoding through a decoder-independent API.
    # This call is intentionally identical for LibRaw and Apple RAW: scene selection
    # cannot change the source, values, provenance, or failure domain of analysis data.
    try:
        evidence = acquire_raw_evidence(path)
        evidence_stage1_note = (
            "Linear DNG：颜色平面剪切可测；不声明 CFA 独立噪声或电子域校准"
            if evidence.sample_kind == "linear-camera-rgb" else None)
    except EvidenceAcquisitionError as exc:
        if exc.unsupported_format or "unsupported file format" in str(exc).lower():
            message = _unsupported_format_guidance(path, shot, exc)
        else:
            message = f"Cannot acquire RAW evidence with rawpy/libraw: {exc}"
        if decoder == "libraw":
            raise RuntimeError(message) from exc
        evidence = None
        evidence_stage1_note = "传感器证据不可用；仅使用解码图像统计，不声明 RAW 剪切或传感器 SNR"
        evidence_error = message

    raw_image = getattr(evidence, "raw_image", None)
    raw_colors = getattr(evidence, "raw_colors", None)
    white_level = getattr(evidence, "white_level", None)
    daylight_wb = getattr(evidence, "daylight_wb", None)
    raw_pattern = getattr(evidence, "raw_pattern", [])
    black_levels = getattr(evidence, "black_levels", [])
    camera_wb = getattr(evidence, "camera_wb", [])
    camera_white_levels = getattr(evidence, "camera_white_levels", [])
    orientation_flip = getattr(evidence, "orientation_flip", 0)
    color_desc = getattr(evidence, "color_desc", "")

    # Fixed-Kelvin WB is a scene recipe derived from evidence calibration. It is not
    # evidence itself, and therefore remains free to degrade per scene decoder.
    kelvin_wb, wb_note = solve_wb_for_mode(
        requested_wb_mode,
        path,
        getattr(evidence, "xyz_to_cam", None),
        make=shot.make,
        model=shot.model,
    )
    # Scene colour-support reporting remains decoder-specific: Core Image owns its
    # colour tables. This does not alter the shared RawEvidence contract.
    camera_data_support: str | None = None
    if decoder == "libraw":
        from .camera_matrices import fallback_xyz_to_cam
        from .priors import find_priors

        matrix_attr = evidence.xyz_to_cam
        has_matrix = False
        if matrix_attr is not None:
            matrix = np.asarray(matrix_attr, dtype=np.float64)
            has_matrix = (
                matrix.size >= 9
                and float(np.abs(matrix[:3, :3]).sum()) > 1e-9
            )
        camera_data_support = camera_data_support_note(
            dng_metadata.read_dng_color_calibration(path) is not None,
            has_matrix,
            fallback_xyz_to_cam(shot.make, shot.model) is not None,
            find_priors(shot.make, shot.model) is not None,
            shot.make,
            shot.model,
        )
    wb_degradation: str | None = None
    kelvin_requested = kelvin_mode_cct(requested_wb_mode) is not None
    effective_wb_mode = requested_wb_mode
    if kelvin_requested and kelvin_wb is None:
        wb_degradation = wb_note
        # Project-owned hot WB needs the same explicit camera calibration on both scene
        # decoders.  If it is absent, neither path may claim the requested declaration.
        effective_wb_mode = "camera"
    elif wb_note:
        wb_degradation = wb_note

    if decoder == "libraw":
        # A fresh handle is a deliberate scene-decoder boundary. Evidence has already
        # been copied and cannot be mutated by GainMap application or postprocess.
        try:
            with rawpy.imread(str(path)) as raw:
                # Demosaic/highlight always sees one immutable per-capture
                # preconditioner.  User WB is deliberately not passed into LibRaw.
                # Supplying the as-shot values explicitly also avoids a decoder-specific
                # camera-WB fallback changing this fixed boundary.
                demosaic_alg = resolve_demosaic_algorithm(raw, demosaic)
                scene_rec2020_render, processing_clip_masks, recipe, lens_shading = _decode_corrected_libraw(
                    raw, path, evidence, effective_highlight_mode, scene_half_size, demosaic_alg,
                )
                from .dng_opcodes import Warp
                scene_geometry_ops = tuple(op for op in recipe.post if isinstance(op, Warp))
                scene_crop_sensor = recipe.crop
                scene_opcode_names = tuple(recipe.names)
                if recipe.skipped:
                    scene_correction_note = "跳过不支持的可选 DNG 校正: " + ", ".join(recipe.skipped)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot decode RAW scene with rawpy/libraw: {exc}"
            ) from exc
        if scene_rec2020_render.ndim != 3 or scene_rec2020_render.shape[2] < 3:
            raise RuntimeError("scene Rec.2020 render did not produce a 3-channel image")

        # Camera -> Rec.2020 now retains signed/over-range float values in
        # the same 16-bit code units as LibRaw's camera-plane buffer.
        applied_wb = camera_wb
        scene_scale = libraw_scene_scale(
            65535.0, effective_highlight_mode, applied_wb,
            baseline_exposure=shot.baseline_exposure,
        )
        xyz_render = scene_rec2020_to_xyz_render(scene_rec2020_render, scene_scale)
        render_scale = scene_scale
        clip_masks = build_clip_masks(
            raw_image,
            raw_colors,
            color_desc,
            white_level,
            [float(x) for x in black_levels],
            [float(x) for x in camera_white_levels],
            orientation_flip,
            scene_rec2020_render.shape[:2],
            raw_pattern,
            scene_geometry_ops,
            scene_crop_sensor,
            evidence.spatial_black,
        )
        _merge_processing_loss(clip_masks, processing_clip_masks)
        if processing_clip_masks is not None:
            scene_processing_loss_pct = float(np.mean(np.max(processing_clip_masks, axis=2) > 0) * 100)
        evidence_shape = (
            int(scene_rec2020_render.shape[0]),
            int(scene_rec2020_render.shape[1]),
        )

    if decoder == "coreimage":
        # Keep Apple's reconstruction fixed as well.  neutralTemperature belongs to the
        # old decoder-coupled path; the project hot-WB matrix runs after this call.
        #
        # DECLARATION (review R5 item 1): a fixed-Kelvin mode on this decoder is
        # the PROJECT hot-WB — the same Rec.2020 matrix the LibRaw path applies
        # for the same file and target — composed onto Apple's fixed AsShot
        # decode. It is not Apple's neutralTemperature and does not claim to
        # reproduce it: Apple's colour transform is opaque, so the conjugate
        # matrix is exact for a linear decode and approximate here. Measured
        # against the same RAW 9 decoder's neutralTemperature (fp _SDI0150 /
        # _SDI0199, 3200 K / 5500 K): median xy difference 0.015-0.044, median
        # RGB direction angle 2.5-8.0 degrees. What IS pinned, at pixel level,
        # is the declared property — one hot-WB matrix across decoders
        # (tests/test_coreimage_decode.py) and a neutral render of the
        # declared white on the LibRaw path (tests/test_wb.py).
        neutral_cct = None
        from . import coreimage_decode
        scale_compensation = coreimage_decode.scale_compensation_for_mode(coreimage_scale)

        # A10 item 2: the decode below renders with interactive=half_size,
        # so the precheck probes THAT workload — preview and export
        # contexts carry different options and can fail independently.
        try:
            if not coreimage_decode.runtime_available(interactive=bool(scene_half_size)):
                raise RuntimeError("Core Image runtime unavailable for this workload")
            ci_float, info = coreimage_decode.decode_scene_rec2020(
                path, half_size=scene_half_size, version=coreimage_version,
                scale_compensation=scale_compensation,
                neutral_cct=neutral_cct,
            )
        except Exception as exc:
            # Auto is a capability ladder; an explicitly selected Apple version
            # stays strict so failed experiments cannot silently use another decoder.
            if coreimage_version != "auto" or evidence is None:
                raise RuntimeError(f"Cannot decode Apple RAW scene: {exc}; evidence: {evidence_error or 'available'}") from exc
            fallback = load_raw(
                path, scene_highlight_mode=effective_highlight_mode,
                scene_half_size=scene_half_size, demosaic=demosaic,
                wb_mode=requested_wb_mode, decoder="libraw",
            )
            return replace(fallback, scene_decoder_fallback=f"Apple RAW auto → LibRaw: {exc}")
        failures = info.get("fallback_errors") or []
        scene_decoder_fallback = "; ".join(failures) if failures else None
        reliability_source = "decoded-image-estimate"
        scene_rec2020_render, scene_scale = coreimage_decode.scene_float_to_half(ci_float)
        ci_authored_baseline = info.get("baseline_exposure_authored")
        if ci_authored_baseline is not None:
            try:
                candidate = float(ci_authored_baseline)
            except (TypeError, ValueError, OverflowError):
                candidate = float("nan")
            if np.isfinite(candidate):
                # This is the exact decoder-version-specific value that was cleared, so
                # it is the authoritative value to restore on the Core Image path. The
                # metadata parser remains the fallback when the getter is unavailable.
                effective_baseline_exposure = candidate

        # Apple's direct scene-linear recipe clears BaselineExposure inside CIRAWFilter.
        # Restore the recorded file intent as a scale divisor, exactly as LibRaw does. If
        # an older API could not clear the property, leave the already baked gain alone.
        if bool(info.get("baseline_exposure_cleared")):
            scene_scale = float(scene_scale) / baseline_exposure_gain(
                effective_baseline_exposure
            )
        else:
            applied = info.get("baseline_exposure_applied")
            try:
                baseline_exposure_baked_in = abs(float(applied)) > 1e-6
            except (TypeError, ValueError, OverflowError):
                baseline_exposure_baked_in = True
        scene_opcode_names = tuple(coreimage_decode.read_dng_opcodes(path)["names"])
        needs_correction_evidence = bool(scene_opcode_names)
        if needs_correction_evidence:
            scene_processing_loss_pct = None
        if evidence is not None:
            # Align one decoded statistic per file. This additional LibRaw *scene
            # comparison never replaces RawEvidence. It supplies an optional
            # scale reference plus an aggregate correction-loss estimate.
            reference_level = float("nan")
            coreimage_level = float("nan")
            try:
                # A fresh handle: reading the mosaic above leaves this LibRaw handle
                # unable to postprocess (LibRawOutOfOrderCallError).
                # Both decoders now compare the same fixed as-shot DecodeContext.  User
                # WB happens only after this scalar alignment and therefore cannot make
                # the A/B scale measurement cross illuminants.
                with rawpy.imread(str(path)) as reference_raw:
                    # The reference must be the PRODUCTION LibRaw decode, not a
                    # bare one: the DNG's dark-field corrections (pre-demosaic
                    # GainMap, post-render FixVignetteRadial) are part of what
                    # both decoders render, and skipping them here read lens
                    # shading as a decoder exposure difference — measured up to
                    # +0.88 EV on iPhone Standard RAW, pulling aligned-mode
                    # RAW 9 toward the uncorrected dark reference.
                    reference_scene, reference_loss, reference_recipe, _ = _decode_corrected_libraw(
                        reference_raw, path, evidence, effective_highlight_mode,
                        True, None,
                    )
                    scene_processing_loss_pct = (float(np.mean(np.max(reference_loss, axis=2) > 0) * 100)
                                                 if reference_loss is not None else 0.0)
                # Decode the reference with the same storage-scale contract as the main
                # LibRaw path. Normalising reconstruct by 65535 would lose its reserved
                # WB headroom and can shift this statistic by more than one EV.
                # Self-review 2026-08-27 (P1): the reference render above was
                # decoded with camera_wb, so the storage-scale contract must
                # use camera_wb's headroom too. The previous helper returned
                # the requested Kelvin multipliers whenever any were present,
                # which re-exposed every Core Image + Kelvin export by up to
                # ~0.4 EV as a function of the WB choice alone (measured
                # +0.053 EV at 3200K, -0.234 EV at 5500K on _SDI0150).
                reference_scale = libraw_scene_scale(
                    65535.0, effective_highlight_mode, camera_wb,
                    baseline_exposure=effective_baseline_exposure,
                )
                reference_level = scene_green_median(
                    np.asarray(reference_scene, dtype=np.float32) / reference_scale
                )
                coreimage_level = scene_green_median(
                    np.asarray(scene_rec2020_render, dtype=np.float32) / float(scene_scale)
                )
                raw_factor = reference_level / coreimage_level
                if not np.isfinite(raw_factor) or not (COREIMAGE_ALIGN_MIN <= raw_factor <= COREIMAGE_ALIGN_MAX):
                    raise ValueError(f"implausible decoded-green alignment factor {raw_factor!r}")
                from .scene_reference import reliable_reference_samples
                reliable_reference, reliable_reference_pct = reliable_reference_samples(
                    evidence, reference_scene, reference_scale, reference_loss, reference_recipe,
                )
                if coreimage_uses_file_alignment(coreimage_scale):
                    scene_align_factor = float(raw_factor)
                    scene_scale = float(scene_scale) / scene_align_factor
                # Even unity mode needs a common exposure unit for the reference.
                # This scalar does not imply spatial correspondence or identical color.
                reliable_reference *= np.float32(scene_align_factor / raw_factor)
                reliability_source = "sensor-reference"
            except Exception as exc:  # noqa: BLE001 - a render must not fail over a metric
                error = f"{type(exc).__name__}: {exc}"
                scene_reference_error = error
                if coreimage_uses_file_alignment(coreimage_scale):
                    scene_align_error = error
                if needs_correction_evidence and scene_processing_loss_pct is None:
                    scene_correction_note = f"DNG 校正损失参考不可用；HDR 使用有上限的解码图像估计: {error}"
            finally:
                # Do not carry the half-size reference into the full-size XYZ stage.
                reference_scene = None
                reference_loss = None
        else:
            scene_processing_loss_pct = None
            scene_reference_error = evidence_error
            if coreimage_uses_file_alignment(coreimage_scale):
                scene_align_error = "LibRaw evidence unavailable; using Apple-native scale"
        xyz_render = scene_rec2020_to_xyz_render(scene_rec2020_render, scene_scale)
        render_scale = scene_scale
        scene_decoder = "coreimage"
        scene_decoder_version = str(info.get("version") or coreimage_version)
        scene_decoder_runtime = str(info.get("decoder_runtime_id") or "") or None
        scene_scale_mode = coreimage_scale
        scene_opcode_names = tuple(coreimage_decode.read_dng_opcodes(path)["names"])
        # Strict Core Image pipeline: this is a SEPARATE path, not a LibRaw back end.
        # Core Image executes the file's DNG opcodes (measured on Sigma fp: per-plane
        # WarpRectilinear plus a lens-shading GainMap), so its frame is a nonlinear warp
        # of LibRaw's — corners move by tens of pixels. Per-pixel CFA evidence therefore
        # cannot be carried across, and pretending otherwise would put clip retreat on
        # the wrong pixels. Masks are dropped rather than re-mapped; the aggregate RAW
        # facts (levels, clip %, SNR, noise floor, WB testimony) stay valid because they
        # are distributions, not pixel positions, and continue to come from LibRaw.
        clip_masks = None
        evidence_shape = None
        scene_geometry_crop = None

    if scene_rec2020_render is None or xyz_render is None:
        raise RuntimeError("scene decoder did not produce a render buffer")

    base_bundle = RawBundle(
        path=path,
        raw_image=raw_image,
        raw_colors=raw_colors,
        xyz_render=xyz_render,
        render_scale=render_scale,
        scene_rec2020_render=scene_rec2020_render,
        scene_scale=scene_scale,
        white_level=white_level,
        black_levels=[float(x) for x in black_levels],
        camera_wb=[float(x) for x in camera_wb],
        color_desc=color_desc,
        raw_pattern=raw_pattern,
        camera_white_levels=[float(x) for x in camera_white_levels],
        # RAW 9 has one calibrated reconstruction path. LibRaw's clip/blend/gated
        # selector does not map onto CIRAWFilter and must not be reported as if it did.
        scene_highlight_mode=effective_highlight_mode,
        orientation_flip=orientation_flip,
        wb_mode="camera",
        wb_degradation=wb_degradation,
        evidence_stage1_note=evidence_stage1_note,
        camera_data_support=camera_data_support,
        daylight_wb=daylight_wb,
        shot_make=shot.make,
        shot_model=shot.model,
        shot_iso=shot.iso,
        shot_shutter=getattr(evidence, "shot_shutter", None),
        baseline_exposure=effective_baseline_exposure,
        baseline_exposure_baked_in=baseline_exposure_baked_in,
        applied_wb=[float(x) for x in camera_wb],
        lens_shading=lens_shading,
        processing_clip_masks=processing_clip_masks,
        scene_geometry_ops=scene_geometry_ops,
        scene_crop_sensor=scene_crop_sensor,
        scene_correction_note=scene_correction_note,
        scene_processing_loss_pct=scene_processing_loss_pct,
        scene_reliable_reference_rec2020=reliable_reference,
        scene_reliable_reference_pct=reliable_reference_pct,
        scene_reliability_source=reliability_source,
        scene_reference_error=scene_reference_error,
        scene_decoder_fallback=scene_decoder_fallback,
        clip_masks=clip_masks,
        scene_decoder=scene_decoder,
        scene_decoder_version=scene_decoder_version,
        scene_decoder_runtime=scene_decoder_runtime,
        scene_scale_mode=scene_scale_mode,
        scene_align_factor=scene_align_factor,
        scene_align_error=scene_align_error,
        scene_opcode_names=scene_opcode_names,
        evidence_shape=evidence_shape,
        scene_geometry_crop=scene_geometry_crop,
        scene_geometry_corr=scene_geometry_corr,
        evidence=evidence,
        evidence_provider=getattr(evidence, "provider", "unavailable"),
        evidence_provider_version=getattr(evidence, "provider_version", None),
        evidence_error=evidence_error,
        wb_xyz_to_cam=(
            None
            if getattr(evidence, "xyz_to_cam", None) is None
            else np.asarray(evidence.xyz_to_cam, dtype=np.float64).copy()
        ),
        decode_wb=[float(x) for x in camera_wb],
        wb_color_matrix=(
            None
            if getattr(evidence, "color_matrix", None) is None
            else np.asarray(evidence.color_matrix, dtype=np.float64).copy()
        ),
    )
    if effective_wb_mode == "camera":
        return base_bundle
    balanced = rebalance_raw_bundle(base_bundle, effective_wb_mode)
    if wb_degradation and not balanced.wb_degradation:
        balanced.wb_degradation = wb_degradation
    return balanced
