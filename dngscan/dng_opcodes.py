# SPDX-License-Identifier: GPL-3.0-or-later
"""DNG opcode execution at the mosaic / camera RGB boundaries.

Coordinates and polynomial ordering follow DNG 1.4 and Adobe's
``dng_filter_warp::GetSrcPixelPosition``. Camera planes are warped before
their colour matrix. Permission rasters use the same mapping and the maximum
loss over the interpolation footprint, never interpolated confidence.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._deps import np
from . import metadata as md


@dataclass(frozen=True)
class Warp:
    coefficients: tuple[tuple[float, ...], ...]
    cx: float
    cy: float
    fisheye: bool = False
    aspect: float = 1.0


@dataclass
class OpcodePlan:
    gain_maps: list = field(default_factory=list)
    post: list = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    crop: tuple[float, float, float, float] | None = None  # y,x,h,w, active-area pixels
    white_levels: tuple[float, ...] = ()


NAMES = {1: "WarpRectilinear", 2: "WarpFisheye", 3: "FixVignetteRadial", 9: "GainMap"}


def read_plan(path: Path) -> OpcodePlan:
    """Read only the main RAW IFD; refuse unsupported required operations.

    Preview IFD opcodes must not be concatenated with the RAW recipe. Optional
    unsupported operations are explicitly reported. A malformed required list
    is a decode error, rather than an apparently successful uncorrected image.
    """
    plan = OpcodePlan()
    with open(path, "rb") as fh:
        head = fh.read(8)
        if len(head) != 8 or head[:2] not in (b"II", b"MM"):
            return plan
        endian = "<" if head[:2] == b"II" else ">"
        if struct.unpack(endian + "H", head[2:4])[0] != 42:
            return plan
        root = md._read_ifd_entries(fh, struct.unpack(endian + "L", head[4:])[0], endian)
        if not any(t == md.TAG_DNG_VERSION for t, *_ in root):
            return plan
        wanted = {254, 256, 257, 262, 330, 50717, 50718, 50719, 50720, 51008, 51009, 51022}

        def values(entries):
            result = {}
            for tag, typ, count, data in entries:
                if tag in wanted:
                    if count * md._TYPE_SIZES.get(typ, 1) > 16_000_000:
                        raise ValueError("DNG correction metadata exceeds supported size")
                    result[tag] = md._entry_values(fh, typ, count, data, endian)
            return result

        first = values(root)
        ifds = [first]
        for off in first.get(330, [])[:64]:
            ifds.append(values(md._read_ifd_entries(fh, int(off), endian)))
        candidates = [v for v in ifds if v.get(262, [0])[0] in (32803, 34892)
                      and not (int(v.get(254, [0])[0]) & 1)]
        if not candidates:
            return plan
        tags = max(candidates, key=lambda v: v.get(256, [0])[0] * v.get(257, [0])[0])
        plan.white_levels = tuple(float(v) for v in tags.get(50717, ()))
        if any(not math.isfinite(v) or v <= 0 for v in plan.white_levels):
            raise ValueError("invalid DNG WhiteLevel")
        scale = tags.get(50718, [1.0, 1.0])
        if len(scale) != 2 or not all(math.isfinite(v) and v > 0 for v in scale):
            raise ValueError("invalid DNG DefaultScale")
        aspect = float(scale[0] / scale[1])
        origin, size = tags.get(50719, [0., 0.]), tags.get(50720)
        if origin is not None and size is not None:
            if len(origin) != 2 or len(size) != 2 or not all(
                math.isfinite(v) and v >= 0 for v in origin + size
            ) or min(size) <= 0:
                raise ValueError("invalid DNG DefaultCrop")
            plan.crop = (float(origin[1]), float(origin[0]), float(size[1]), float(size[0]))
        for stage, tag in enumerate((51008, 51009, 51022), 1):
            if tag not in tags:
                continue
            raw = tags[tag]
            if len(raw) != 1 or not isinstance(raw[0], bytes):
                raise ValueError(f"invalid DNG OpcodeList{stage}")
            blob = raw[0]
            if len(blob) < 4:
                raise ValueError(f"truncated DNG OpcodeList{stage}")
            count = struct.unpack_from(">L", blob)[0]
            if count > 1024:
                raise ValueError("too many DNG opcodes")
            pos = 4
            for _ in range(count):
                if pos + 16 > len(blob):
                    raise ValueError(f"truncated DNG OpcodeList{stage}")
                oid, version, flags, length = struct.unpack_from(">4L", blob, pos)
                pos += 16
                if pos + length > len(blob):
                    raise ValueError(f"truncated DNG opcode {oid}")
                payload = blob[pos:pos + length]
                pos += length
                name = NAMES.get(oid, f"Opcode{oid}")
                supported = version <= 0x01040000 and (
                    (stage == 2 and oid == 9) or (stage == 3 and oid in (1, 2, 3))
                )
                if not supported:
                    label = f"OpcodeList{stage}/{name} (version {version:#x})"
                    if flags & 1:
                        plan.skipped.append(label)
                        continue
                    raise ValueError(f"LibRaw pipeline does not support required DNG {label}; use Apple RAW")
                if oid == 9:
                    op = md._parse_gain_map_payload(payload)
                    if op is None or not np.all(np.isfinite(op.gains)) or np.any(op.gains < 0):
                        raise ValueError("invalid DNG GainMap")
                    if not all(math.isfinite(v) for v in (op.spacing_v, op.spacing_h, op.origin_v, op.origin_h)):
                        raise ValueError("invalid DNG GainMap coordinates")
                    plan.gain_maps.append(op)
                elif oid == 3:
                    if length != 56:
                        raise ValueError("invalid DNG FixVignetteRadial size")
                    vals = struct.unpack(">7d", payload)
                    if not all(math.isfinite(v) for v in vals) or not all(0 <= v <= 1 for v in vals[5:]):
                        raise ValueError("invalid DNG FixVignetteRadial parameters")
                    plan.post.append(md.DngVignetteRadial(tuple(vals[:5]), vals[5], vals[6]))
                else:
                    planes = struct.unpack_from(">L", payload)[0] if length >= 4 else 0
                    stride = 4 if oid == 2 else 6
                    if planes not in (1, 3) or length != 4 + planes * stride * 8 + 16:
                        raise ValueError(f"invalid DNG {name} planes or size")
                    vals = struct.unpack_from(f">{planes * stride + 2}d", payload, 4)
                    if not all(math.isfinite(v) for v in vals) or not all(0 <= v <= 1 for v in vals[-2:]):
                        raise ValueError(f"invalid DNG {name} parameters")
                    coeff = tuple(tuple(vals[i * stride:(i + 1) * stride]) + ((0., 0.) if oid == 2 else ()) for i in range(planes))
                    plan.post.append(Warp(coeff, vals[-2], vals[-1], oid == 2, aspect))
                plan.names.append(name)
            if pos != len(blob):
                raise ValueError(f"trailing bytes in DNG OpcodeList{stage}")
    return plan


def _coordinates(op: Warp, h: int, w: int, y0: int, y1: int, channel: int):
    """Destination-to-source positions, in un-oriented image coordinates."""
    cx, cy = op.cx * w, op.cy * h
    radius = math.hypot(max(cx, w - cx), max(cy, h - cy) / op.aspect)
    x = (np.arange(w, dtype=np.float64)[None, :] - cx) / radius
    y = (np.arange(y0, y1, dtype=np.float64)[:, None] - cy) / (radius * op.aspect)
    rr = np.minimum(x * x + y * y, 1.0)
    k0, k1, k2, k3, t0, t1 = op.coefficients[min(channel, len(op.coefficients) - 1)]
    if op.fisheye:
        r = np.sqrt(rr)
        t = np.arctan(r)
        t2 = t * t
        ratio = np.divide(t * (k0 + t2 * (k1 + t2 * (k2 + t2 * k3))), r,
                          out=np.ones_like(r), where=rr >= 1e-12)
    else:
        ratio = k0 + rr * (k1 + rr * (k2 + rr * k3))
    sx = cx + radius * (x * ratio + t1 * (rr + 2 * x * x) + 2 * t0 * x * y)
    sy = cy + radius * op.aspect * (y * ratio + t0 * (rr + 2 * y * y) + 2 * t1 * x * y)
    return sy, sx


def warp_image(image: Any, op: Warp, *, loss: bool = False) -> Any:
    """Cubic camera RGB resampling, or conservative footprint-maximum loss.

    Integer image output keeps the existing LibRaw storage contract. The Rust
    implementation processes destination rows directly without coordinate maps.
    """
    from . import _fast
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("DNG warp requires a nonempty 3-channel image")
    if not op.fisheye and all(tuple(k) == (1., 0., 0., 0., 0., 0.) for k in op.coefficients):
        return image
    native = _fast.kernel("warp_dng")
    if native is not None:
        return native(image, op.coefficients, op.cx, op.cy, op.aspect, op.fisheye, loss)
    h, w = image.shape[:2]
    out = np.empty_like(image, dtype=np.float16 if loss else np.uint16)
    for y0 in range(0, h, 64):
        y1 = min(y0 + 64, h)
        for c in range(3):
            sy, sx = _coordinates(op, h, w, y0, y1, c)
            outside = (sy < 0) | (sy > h - 1) | (sx < 0) | (sx > w - 1)
            sy, sx = np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)
            iy, ix = np.floor(sy).astype(np.intp), np.floor(sx).astype(np.intp)
            fy, fx = sy - iy, sx - ix
            def weights(t):
                return (-.5*t + t*t - .5*t*t*t, 1 - 2.5*t*t + 1.5*t*t*t,
                        .5*t + 2*t*t - 1.5*t*t*t, -.5*t*t + .5*t*t*t)
            wy, wx = weights(fy), weights(fx)
            acc = np.zeros(sy.shape, dtype=np.float64)
            for j in range(4):
                for i in range(4):
                    src = image[np.clip(iy + j - 1, 0, h - 1), np.clip(ix + i - 1, 0, w - 1), c]
                    if loss:
                        np.maximum(acc, src, out=acc)
                    else:
                        acc += src * wy[j] * wx[i]
            out[y0:y1, :, c] = np.maximum(acc, outside) if loss else np.clip(acc, 0, 65535)
    return out


def crop_image(image: Any, crop: tuple | None, sensor_shape: tuple[int, int]) -> Any:
    if crop is None:
        return image
    y, x, h, w = crop
    sy, sx = image.shape[0] / sensor_shape[0], image.shape[1] / sensor_shape[1]
    y0, x0 = max(0, round(y * sy)), max(0, round(x * sx))
    y1, x1 = min(image.shape[0], round((y+h)*sy)), min(image.shape[1], round((x+w)*sx))
    if y1 <= y0 or x1 <= x0:
        raise ValueError("DNG DefaultCrop is outside the decoded image")
    return image[y0:y1, x0:x1]


def align_loss(bundle: Any, values: Any) -> Any:
    """Warp an un-oriented RAW-space loss raster into the scene geometry."""
    from .raw_io import _orient_like_libraw
    for op in getattr(bundle, "scene_geometry_ops", ()):
        values = warp_image(values, op, loss=True)
    values = crop_image(values, getattr(bundle, "scene_crop_sensor", None), bundle.raw_image.shape)
    return _orient_like_libraw(values, bundle.orientation_flip)


def camera_to_rec2020(image: Any, rgb_cam: Any) -> Any:
    """LibRaw's actual camera matrix and Rec.2020 coefficients, in row bands.

    Read rgb_cam AFTER postprocess: LibRaw can adopt an embedded DNG matrix at
    processing time. Do not invert a clipped Rec.2020 image to recover planes.
    """
    matrix = np.asarray(rgb_cam, dtype=np.float32)[:3, :3]
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-9:
        raise ValueError("DNG camera-plane correction requires a valid LibRaw colour matrix")
    rec = np.asarray(((.627452, .329249, .043299), (.069109, .919531, .011360),
                      (.016398, .088030, .895572)), dtype=np.float64)
    out_matrix = np.zeros((3, 3), dtype=np.float32)
    for k in range(3):
        out_matrix += (rec[:, k, None] * matrix[None, k, :]).astype(np.float32)
    out = np.empty_like(image, dtype=np.uint16)
    for y in range(0, image.shape[0], 128):
        src = image[y:y+128].astype(np.float32)
        for c in range(3):
            v = src[..., 0]*out_matrix[c, 0] + src[..., 1]*out_matrix[c, 1] + src[..., 2]*out_matrix[c, 2]
            out[y:y+128, :, c] = np.clip(v, 0, 65535).astype(np.uint16)
    return out
