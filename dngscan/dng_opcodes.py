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
    knots: tuple[float, ...] = ()
    scales: tuple[tuple[float, ...], ...] = ()
    scale: float = 1.0


@dataclass(frozen=True)
class TrimBounds:
    """An ordered stage-3 crop in the current un-oriented camera image."""
    bounds: tuple[int, int, int, int]  # top, left, bottom, right


@dataclass
class OpcodePlan:
    stage1: list = field(default_factory=list)
    stage2: list = field(default_factory=list)
    gain_maps: list = field(default_factory=list)
    post: list = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    crop: tuple[float, float, float, float] | None = None  # y,x,h,w, active-area pixels
    white_levels: tuple[float, ...] = ()


NAMES = {1:"WarpRectilinear",2:"WarpFisheye",3:"FixVignetteRadial",4:"FixBadPixelsConstant",
         5:"FixBadPixelsList",6:"TrimBounds",7:"MapTable",8:"MapPolynomial",9:"GainMap",
         10:"DeltaPerRow",11:"DeltaPerColumn",12:"ScalePerRow",13:"ScalePerColumn",14:"WarpRectilinear2"}


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
            previous_optional_warp2 = False
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
                if previous_optional_warp2 and oid in (1,2):
                    continue  # DNG 1.6 fallback warp skip rule.
                previous_optional_warp2 = False
                supported = version <= 0x01060000 and (
                    (stage == 1 and oid in (4,5,7,8,10,11,12,13)) or
                    (stage == 2 and oid in (7,8,9,10,11,12,13)) or
                    (stage == 3 and oid in (1,2,3,6,7,8,10,11,12,13,14))
                )
                if not supported:
                    label = f"OpcodeList{stage}/{name} (version {version:#x})"
                    if flags & 1:
                        plan.skipped.append(label)
                        continue
                    raise ValueError(f"LibRaw pipeline does not support required DNG {label}; use Apple RAW")
                if stage == 3 and oid != 6 and any(isinstance(op,TrimBounds) for op in plan.post):
                    raise ValueError("DNG operations after TrimBounds require retained image-origin coordinates; use Apple RAW")
                if oid == 6:
                    if length != 16:
                        raise ValueError("invalid DNG TrimBounds size")
                    bounds=struct.unpack('>4l',payload)
                    t,l,b,r=bounds
                    if min(t,l)<0 or b<=t or r<=l:
                        raise ValueError("invalid DNG TrimBounds rectangle")
                    plan.post.append(TrimBounds(bounds))
                elif oid in (4,5,7,8,10,11,12,13):
                    from .dng_point_ops import parse
                    op = parse(oid,payload,stage)
                    (plan.stage1 if stage==1 else plan.stage2 if stage==2 else plan.post).append(op)
                elif oid == 14:
                    planes=struct.unpack_from(">L",payload)[0] if length>=4 else 0
                    if planes not in (1,3) or length != 4+planes*19*8+20:
                        raise ValueError("invalid WarpRectilinear2 planes or size")
                    vals=struct.unpack_from(f">{planes*19+2}d",payload,4)
                    reciprocal=struct.unpack_from(">L",payload,length-4)[0]
                    if reciprocal not in (0,1) or not np.isfinite(vals).all() or not all(0<=v<=1 for v in vals[-2:]):
                        raise ValueError("invalid WarpRectilinear2 parameters")
                    coeff=tuple(tuple(vals[i*19:(i+1)*19])+(float(reciprocal),) for i in range(planes))
                    if any(not 0<=c[17]<c[18]<=1 for c in coeff):
                        raise ValueError("invalid WarpRectilinear2 radius range")
                    # Division poles and non-invertible radial fields cannot
                    # be sent to a sampler as NaN/Inf coordinates.
                    for c in coeff:
                        r=np.linspace(0.,1.,8193)
                        f=np.polynomial.polynomial.polyval(np.clip(r,c[17],c[18]),c[:15])
                        mapped=r/f if reciprocal else r*f
                        if np.any(f<=0) or not np.isfinite(mapped).all() or np.any(np.diff(mapped)<=0):
                            raise ValueError("non-invertible WarpRectilinear2 radial field")
                    plan.post.append(Warp(coeff,vals[-2],vals[-1],aspect=aspect))
                    previous_optional_warp2=bool(flags&1)
                elif oid == 9:
                    op = md._parse_gain_map_payload(payload)
                    if op is None or not np.all(np.isfinite(op.gains)) or np.any(op.gains < 0):
                        raise ValueError("invalid DNG GainMap")
                    if not all(math.isfinite(v) for v in (op.spacing_v, op.spacing_h, op.origin_v, op.origin_h)):
                        raise ValueError("invalid DNG GainMap coordinates")
                    plan.gain_maps.append(op)
                    plan.stage2.append(op)
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
    if op.knots:
        x,y = x/op.scale,y/op.scale
        factor = np.interp(np.sqrt(x*x+y*y),op.knots,op.scales[channel])
        return cy+radius*op.aspect*y*factor, cx+radius*x*factor
    rr = np.minimum(x * x + y * y, 1.0)
    coeff = op.coefficients[min(channel,len(op.coefficients)-1)]
    if len(coeff)==20:
        t0,t1=coeff[15:17]
        radius=np.sqrt(np.clip(rr,coeff[17]**2,coeff[18]**2))
        ratio=np.polynomial.polynomial.polyval(radius,coeff[:15])
        if coeff[19]:ratio=1.0/ratio
        # Preserve the image-space normalization radius below.
        radius=math.hypot(max(cx,w-cx),max(cy,h-cy)/op.aspect)
    elif op.fisheye:
        k0,k1,k2,k3,t0,t1=coeff
        r = np.sqrt(rr)
        t = np.arctan(r)
        t2 = t * t
        ratio = np.divide(t * (k0 + t2 * (k1 + t2 * (k2 + t2 * k3))), r,
                          out=np.ones_like(r), where=rr >= 1e-12)
    else:
        k0,k1,k2,k3,t0,t1=coeff
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
    if not op.knots and not op.fisheye and all(tuple(k) == (1., 0., 0., 0., 0., 0.) for k in op.coefficients):
        return image
    native = _fast.kernel("warp_dng")
    if native is not None:
        return native(image, op.coefficients, op.cx, op.cy, op.aspect, op.fisheye, loss,
                      op.knots, op.scales, op.scale)
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


def terminal_trim_crop(trims, shape, crop=None):
    """Resolve terminal trims without losing their original image coordinates.

    A terminal stage-3 trim commutes with only the colour matrix and final
    DefaultCrop. Earlier trims cannot be folded here: they change the coordinate
    contract seen by later spatial operations or demosaic.
    """
    t,l,b,r=0,0,int(shape[0]),int(shape[1])
    for op in trims:
        nt,nl,nb,nr=op.bounds
        if nt<t or nl<l or nb>b or nr>r:
            raise ValueError("DNG TrimBounds is outside the current image bounds")
        t,l,b,r=nt,nl,nb,nr
    if crop is not None:
        y,x,h,w=crop
        t,l,b,r=max(t,y),max(l,x),min(b,y+h),min(r,x+w)
    if b<=t or r<=l:
        raise ValueError("DNG DefaultCrop does not overlap TrimBounds")
    return float(t),float(l),float(b-t),float(r-l)


def _fractional_crop_loss(values, crop, sensor_shape, output_shape):
    """Resample a crop not aligned to the evidence grid, taking footprint max.

    Rounding a one-pixel crop on a two-pixel evidence grid moves the masks.
    Work directly in continuous grid coordinates instead; bounded row bands
    avoid expanding the entire sensor raster just to slice out the crop.
    """
    y,x,h,w=crop
    sy,sx=values.shape[0]/sensor_shape[0],values.shape[1]/sensor_shape[1]
    t,l,b,r=max(0.,y*sy),max(0.,x*sx),min(values.shape[0],(y+h)*sy),min(values.shape[1],(x+w)*sx)
    if all(abs(v-round(v))<1e-9 for v in (t,l,b,r)):
        return None
    if b<=t or r<=l:
        raise ValueError("DNG crop is outside the sensor evidence")
    oh,ow=output_shape
    ys=np.linspace(t,b,oh+1);xs=np.linspace(l,r,ow+1)
    ylo=np.floor(ys[:-1]+1e-9).astype(np.intp);yhi=np.ceil(ys[1:]-1e-9).astype(np.intp)
    xlo=np.floor(xs[:-1]+1e-9).astype(np.intp);xhi=np.ceil(xs[1:]-1e-9).astype(np.intp)
    out=np.zeros((oh,ow,3),dtype=values.dtype)
    for start in range(0,oh,128):
        stop=min(start+128,oh)
        for dy in range(int(np.max(yhi[start:stop]-ylo[start:stop]))):
            yy=ylo[start:stop]+dy
            for dx in range(int(np.max(xhi-xlo))):
                xx=xlo+dx
                valid=(yy[:,None]<yhi[start:stop,None])&(xx[None,:]<xhi[None,:])
                src=values[np.minimum(yy,values.shape[0]-1)[:,None],
                           np.minimum(xx,values.shape[1]-1)[None,:]]
                np.maximum(out[start:stop],np.where(valid[...,None],src,0),out=out[start:stop])
    return out


def align_sensor_loss(values, sensor_shape, scene_shape, flip, geometry=(), crop=None):
    """Apply DefaultScale, camera warps, crop and orientation in scene order."""
    from .raw_io import _orient_like_libraw, _resize_loss_to_shape
    if geometry:
        sh, sw = scene_shape
        if flip & 4:
            sh, sw = sw, sh
        if crop is not None:
            _, _, ch, cw = crop
            sh, sw = sh * sensor_shape[0] / ch, sw * sensor_shape[1] / cw
        target = (values.shape[0], max(1, round(values.shape[0] * sw / sh)))
        values = _resize_loss_to_shape(values, target)
    for op in geometry:
        values = warp_image(values, op, loss=True)
    if crop is not None:
        output_shape=scene_shape[::-1] if flip & 4 else scene_shape
        cropped=_fractional_crop_loss(values,crop,sensor_shape,output_shape)
        if cropped is not None:
            return _orient_like_libraw(cropped,flip)
    values = crop_image(values, crop, sensor_shape)
    return _orient_like_libraw(values, flip)


def align_loss(bundle: Any, values: Any) -> Any:
    """Warp an un-oriented RAW-space loss raster into the scene geometry."""
    return align_sensor_loss(values, bundle.raw_image.shape,
        bundle.scene_rec2020_render.shape[:2], bundle.orientation_flip,
        getattr(bundle,"scene_geometry_ops",()), getattr(bundle,"scene_crop_sensor",None))


def libraw_camera_matrix(cmatrix: Any, cam_xyz: Any = None, *, is_dng=True) -> Any:
    """Recover the three-colour rgb_cam used by LibRaw's default matrix path.

    rawpy.color_matrix exposes rawdata.color.cmatrix, NOT rgb_cam. DNG adopts
    that matrix when cmatrix[0][0] > .125. Other RGB cameras derive rgb_cam
    from cam_xyz by cam_xyz_coeff's row normalization and inversion.
    """
    embedded = np.asarray(cmatrix, dtype=np.float64)
    if (is_dng and embedded.ndim == 2 and embedded.shape[0] >= 3
            and embedded.shape[1] >= 3 and embedded[0, 0] > .125):
        matrix = embedded[:3, :3].copy()
        if embedded.shape[1] >= 4:
            matrix[:, 1] += embedded[:3, 3]
    else:
        xyz = np.asarray(cam_xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[0] < 3 or xyz.shape[1] != 3:
            raise ValueError("camera RGB conversion needs a calibrated colour matrix")
        if xyz.shape[0] > 3 and np.any(xyz[3]):
            raise ValueError("four-colour camera matrices require a four-plane decode")
        # These are LibRaw's exact xyz_rgb constants, not rounded display
        # matrices; using a different white changes the channel normalization.
        srgb_xyz = np.asarray(((.4124564,.3575761,.1804375),
                               (.2126729,.7151522,.0721750),
                               (.0193339,.1191920,.9503041)))
        cam_rgb = xyz[:3] @ srgb_xyz
        neutral = cam_rgb.sum(axis=1)
        if not np.isfinite(cam_rgb).all() or np.any(neutral <= .00001):
            raise ValueError("camera RGB conversion needs a calibrated colour matrix")
        try:
            matrix = np.linalg.inv(cam_rgb / neutral[:, None])
        except np.linalg.LinAlgError as exc:
            raise ValueError("singular camera colour matrix") from exc
    matrix = matrix.astype(np.float32)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-9:
        raise ValueError("camera RGB conversion needs a valid LibRaw colour matrix")
    return matrix


def camera_to_rec2020(image: Any, rgb_cam: Any) -> Any:
    """Convert corrected camera planes without clipping negative/over-range RGB."""
    matrix = np.asarray(rgb_cam, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera conversion requires a finite 3x3 matrix")
    rec = np.asarray(((.627452, .329249, .043299), (.069109, .919531, .011360),
                      (.016398, .088030, .895572)), dtype=np.float64)
    out_matrix = np.zeros((3, 3), dtype=np.float32)
    for k in range(3):
        out_matrix += (rec[:, k, None] * matrix[None, k, :]).astype(np.float32)
    out = np.empty_like(image, dtype=np.float32)
    for y in range(0, image.shape[0], 128):
        src = image[y:y+128].astype(np.float32)
        for c in range(3):
            v = src[..., 0]*out_matrix[c, 0] + src[..., 1]*out_matrix[c, 1] + src[..., 2]*out_matrix[c, 2]
            out[y:y+128, :, c] = v
    return out
