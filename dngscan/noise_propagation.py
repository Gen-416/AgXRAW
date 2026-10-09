# SPDX-License-Identifier: GPL-3.0-or-later
"""Noise propagation for declared linear operators.

These functions do not estimate noise from an image. Covariances are in the
same linear units as their signal; spatial independence is an explicit input
contract, not an assumption inferred from a photograph's texture.
"""
from __future__ import annotations

from ._deps import np


def _warp_points(op, y, x, h, w, channel):
    """Vector form of dng_opcodes._coordinates for quadrature points.

    Only the ordinary six-coefficient rectilinear warp is admitted by the
    low-frequency approximation; the renderer remains the source of truth.
    """
    if op.fisheye or op.knots:
        raise ValueError("only smooth rectilinear noise warps are supported")
    coeff = op.coefficients[min(channel, len(op.coefficients) - 1)]
    if len(coeff) != 6:
        raise ValueError("extended warp noise propagation unavailable")
    cx, cy = op.cx * w, op.cy * h
    radius = np.hypot(max(cx, w - cx), max(cy, h - cy) / op.aspect)
    nx = (x - cx) / radius
    ny = (y - cy) / (radius * op.aspect)
    rr = np.minimum(nx * nx + ny * ny, 1.)
    k0, k1, k2, k3, t0, t1 = coeff
    ratio = k0 + rr * (k1 + rr * (k2 + rr * k3))
    sx = cx + radius * (nx * ratio + t1 * (rr + 2 * nx * nx) + 2 * t0 * nx * ny)
    sy = cy + radius * op.aspect * (ny * ratio + t0 * (rr + 2 * ny * ny) + 2 * t1 * nx * ny)
    return sy, sx


def _sample_gain(op, y, x, shape, phase):
    """Sample one mosaic GainMap, respecting its image-plane/CFA pitch."""
    if op.get("plane", 0) > 0 or op.get("planes", 1) < 1:
        return np.ones(np.broadcast_shapes(y.shape, x.shape), dtype=np.float64)
    row_pitch, col_pitch = int(op["row_pitch"]), int(op["col_pitch"])
    if row_pitch not in (1, 2) or col_pitch not in (1, 2):
        raise ValueError("GainMap noise propagation requires unit or Bayer-phase pitches")
    h, w = shape
    top, left, bottom, right = (int(op[key]) for key in ("top", "left", "bottom", "right"))
    if bottom <= top or right <= left:
        if row_pitch != 1 or col_pitch != 1:
            raise ValueError("empty GainMap area requires unit pitches")
        top, left, bottom, right = 0, 0, h, w
    if ((phase[0] - top) % row_pitch or (phase[1] - left) % col_pitch):
        return np.ones(np.broadcast_shapes(y.shape, x.shape), dtype=np.float64)
    gains = np.asarray(op["gains"], dtype=np.float64)[..., 0]
    if gains.ndim != 2 or not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError("invalid noise GainMap")
    nv, nh = gains.shape
    iv = np.clip(((y + .5) / h - float(op["origin_v"])) / max(float(op["spacing_v"]), 1e-9), 0, nv - 1)
    ih = np.clip(((x + .5) / w - float(op["origin_h"])) / max(float(op["spacing_h"]), 1e-9), 0, nh - 1)
    v0 = np.minimum(np.floor(iv).astype(int), max(nv - 2, 0))
    h0 = np.minimum(np.floor(ih).astype(int), max(nh - 2, 0))
    fv, fh = iv - v0, ih - h0
    v1, h1 = np.minimum(v0 + 1, nv - 1), np.minimum(h0 + 1, nh - 1)
    gain = ((gains[v0, h0] * (1 - fh) + gains[v0, h1] * fh) * (1 - fv)
            + (gains[v1, h0] * (1 - fh) + gains[v1, h1] * fh) * fv)
    return np.where((y >= top) & (y < bottom) & (x >= left) & (x < right), gain, 1.)


def coarse_spatial_moments(descriptor, output_shape, colours):
    """Gain moments and source-area ratios for the coarse noise model.

    Four-point Gaussian quadrature approximates slowly varying lens gains.
    Gains from multiple opcodes are combined before squaring. The second
    moment includes ``1/min(det(J),1)`` at each quadrature node. A single
    rectilinear camera-plane warp is handled using its source coordinates
    and local Jacobian area. This does not reproduce cubic-resampling or
    demosaicing cross-pixel covariance; it is a low-frequency density model.
    Folding, strong area changes and extrapolated boundary cells have no
    authority and are excluded from denoising.
    """
    dh, dw = output_shape
    maps, warps = descriptor.get("gain_maps", ()), descriptor.get("warp_ops", ())
    if not maps and not warps:
        ones = np.ones((dh, dw, 3), np.float64)
        return ones, ones, ones, np.ones((dh, dw), bool)
    from .dng_opcodes import Warp
    from .raw_io import _orient_like_libraw

    if len(warps) > 1:
        raise ValueError("multiple warp noise propagation unavailable")
    op = Warp(**warps[0]) if warps else None
    flip = int(descriptor.get("orientation_flip", 0))
    inverse = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 5, 7: 7}
    shape_probe = _orient_like_libraw(np.empty((dh, dw), np.uint8), inverse[flip])
    uh, uw = shape_probe.shape
    h, w = map(float, descriptor["full_sensor_shape"])
    ph, pw = map(float, descriptor.get("pre_crop_shape", (h, w)))
    crop = descriptor.get("decoded_crop")
    if crop is None:
        cy, cx, ch, cw = map(float, descriptor.get("effective_sensor_crop", descriptor["sensor_crop"]))
        cy, ch, cx, cw = cy * ph / h, ch * ph / h, cx * pw / w, cw * pw / w
    else:
        cy, cx, ch, cw = map(float, crop)
    means = np.zeros((uh, uw, 3), np.float64)
    seconds = np.zeros_like(means)
    areas = np.ones_like(means)
    valid = np.ones((uh, uw), bool)
    nodes = (.5 - 1 / np.sqrt(12), .5 + 1 / np.sqrt(12))
    for phase_index, label in enumerate(colours):
        channel = "RGB".index(label)
        weight = 1 / (4 * colours.count(label))
        if op is not None:
            # Internal quadrature points alone do not cover the cell's
            # cubic footprint: require all four cell corners to be valid.
            edge_y = cy + np.arange(uh + 1)[:, None] * ch / uh - .5
            edge_x = cx + np.arange(uw + 1)[None, :] * cw / uw - .5
            corner_y, corner_x = _warp_points(op, edge_y, edge_x, ph, pw, channel)
            corners = ((corner_y >= 2) & (corner_y <= ph - 3)
                       & (corner_x >= 2) & (corner_x <= pw - 3))
            valid &= corners[:-1, :-1] & corners[1:, :-1] & corners[:-1, 1:] & corners[1:, 1:]
        for dy in nodes:
            y = cy + (np.arange(uh)[:, None] + dy) * ch / uh - .5
            for dx in nodes:
                x = cx + (np.arange(uw)[None, :] + dx) * cw / uw - .5
                sy, sx = y, x
                area = 1.
                if op is not None:
                    sy, sx = _warp_points(op, y, x, ph, pw, channel)
                    eps = .25
                    yyp, xyp = _warp_points(op, y + eps, x, ph, pw, channel)
                    yym, xym = _warp_points(op, y - eps, x, ph, pw, channel)
                    yxp, xxp = _warp_points(op, y, x + eps, ph, pw, channel)
                    yxm, xxm = _warp_points(op, y, x - eps, ph, pw, channel)
                    area = ((yyp - yym) * (xxp - xxm) - (yxp - yxm) * (xyp - xym)) / (4 * eps ** 2)
                    valid &= (np.isfinite(area) & (area >= .5) & (area <= 2)
                              & (sy >= 2) & (sy <= ph - 3) & (sx >= 2) & (sx <= pw - 3))
                # GainMap lives on native sensor centres, whereas the warp
                # sees the actual half-size/DefaultScale decoded canvas.
                sy = (sy + .5) * h / ph - .5
                sx = (sx + .5) * w / pw - .5
                gain = np.ones((uh, uw), np.float64)
                for gain_map in maps:
                    gain *= _sample_gain(gain_map, sy, sx, (h, w), divmod(phase_index, 2))
                means[..., channel] += weight * gain
                seconds[..., channel] += weight * np.square(gain) / np.clip(area, 1e-6, 1.)
                # Cubic sampling does not antialias expansion in source
                # coordinates. Do not invent additional independent samples
                # when det(J)>1; retain the conservative destination count.
                # Use the smallest sampled area in the cell, rather than
                # E[g²]/E[J], which can underestimate varying gain/J noise.
                np.minimum(areas[..., channel], np.clip(area, 1e-6, 1.), out=areas[..., channel])
    return tuple(_orient_like_libraw(value, flip) for value in (means, seconds, areas, valid))


def calibrated_chroma_variance(bundle, model, scene_dec, *, return_validity=False):
    """Approximate low-frequency covariance from a calibrated CFA model.

    The coarse-cell means use the number of independent sensor samples of
    each colour, then the decoder's recorded linear colour/WB transform.
    This is a low-frequency approximation, not a per-demosaiced-pixel noise
    model. Unknown spatial gains, warps and unrecorded decoder transforms
    fail closed. Returns ``(projected_variance, reason)``.
    """
    if (getattr(bundle, "scene_loss_support_untrusted", False)
            or (getattr(bundle, "noise_decode", None) or {}).get("loss_support_untrusted", False)):
        return None, "decoder loss propagation support is uncertified"
    if model is None or getattr(model, "status", None) != "valid":
        return None, "independent noise calibration unavailable"
    if getattr(model, "domain", None) != "normalized-raw":
        return None, "noise calibration is not in normalized RAW units"
    if getattr(model, "correlation", None) == "measured-spectral-imbalance":
        return None, "measured spectral imbalance requires correlated-noise propagation"
    descriptor = getattr(bundle, "noise_decode", None) or {}
    if not descriptor.get("supported", False):
        return None, descriptor.get("reason") or "decoder has no calibrated linear noise handoff"
    if str(getattr(bundle, "scene_decoder", "libraw")) != "libraw":
        return None, "Apple decoder covariance is not calibrated"
    if getattr(bundle, "scene_geometry_ops", ()) and not descriptor.get("warp_ops"):
        return None, "spatial warp covariance is not propagated"
    if getattr(bundle, "lens_shading", None) and not descriptor.get("gain_maps"):
        return None, "lens shading covariance is not propagated"
    if str(getattr(bundle, "wb_mode", "camera")) != str(descriptor.get("wb_mode", "camera")):
        return None, "hot white-balance covariance is not recorded"
    raw = getattr(bundle, "raw_image", None)
    raw_shape = descriptor.get("sensor_window_shape")
    if raw_shape is None and raw is not None and np.ndim(raw) == 2:
        raw_shape = np.shape(raw)
    if raw_shape is None or len(raw_shape) != 2 or min(raw_shape) <= 0:
        return None, "noise calibration requires CFA sensor evidence"
    pattern = np.asarray(getattr(bundle, "raw_pattern", ()))
    if pattern.shape != (2, 2):
        return None, "only Bayer CFA coarse noise propagation is calibrated"
    labels = str(getattr(bundle, "color_desc", ""))
    try:
        colours = [labels[int(cid)].upper() for cid in pattern.flat]
    except (IndexError, ValueError):
        return None, "CFA colour identity unavailable"
    if sorted(colours) != ["B", "G", "G", "R"]:
        return None, "only Bayer RGB coarse noise propagation is calibrated"
    dec = np.asarray(scene_dec, dtype=np.float64)
    cells = raw_shape[0] * raw_shape[1] / (dec.shape[0] * dec.shape[1])
    if cells < 4:
        return None, "coarse noise approximation requires at least four sensels per cell"
    try:
        matrix = np.asarray(descriptor["normalized_raw_to_scene"], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("invalid matrix")
        matrix = matrix * float(getattr(bundle, "exposure_gain", 1.0))
        if str(getattr(bundle, "lens_filter", "none")) != "none":
            from .lens_filter import lens_filter_matrix

            matrix = np.asarray(lens_filter_matrix(bundle.lens_filter)) @ matrix
        if np.linalg.cond(matrix) > 1e6:
            raise ValueError("ill-conditioned matrix")
        # Local average signal is only used in the known a*x+b shot-noise
        # relation. It never measures the amplitude of image detail.
        camera_mean = np.clip(dec @ np.linalg.inv(matrix).T, 0, 1)
        gain_mean, gain_second, _area_ratio, valid = coarse_spatial_moments(descriptor, dec.shape[:2], colours)
        camera_var = np.empty_like(camera_mean)
        for i, label in enumerate("RGB"):
            coefficients = model.coefficients(label)
            if coefficients is None and label == "G":
                greens = [model.coefficients(name) for name in ("G1", "G2")]
                if all(value is not None for value in greens):
                    coefficients = tuple(np.mean(greens, axis=0))
            if coefficients is None:
                return None, f"{label} noise coefficients unavailable"
            a, b = coefficients
            count = cells * colours.count(label) / 4
            camera_var[..., i] = (float(a) * camera_mean[..., i] * gain_second[..., i] / gain_mean[..., i]
                                   + float(b) * gain_second[..., i]) / count
        projection = np.eye(3) - np.ones((3, 1)) * np.asarray([.2627, .6780, .0593])[None, :]
        chroma_matrix = projection @ matrix
        chroma_var = camera_var @ np.square(chroma_matrix).T
    except (KeyError, ValueError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
        return None, f"decoder noise transform unavailable: {exc}"
    if not np.isfinite(chroma_var).all() or np.any(chroma_var < 0):
        return None, "propagated noise variance is invalid"
    result = chroma_var.astype(np.float32), "approximate CFA coarse-cell shot/read covariance; lens quadrature and warp-area model"
    return (*result, valid) if return_validity else result


def transform_covariance(covariance, matrix):
    """Propagate full channel covariance through ``out = matrix @ input``."""
    cov = np.asarray(covariance, dtype=np.float64)
    mat = np.asarray(matrix, dtype=np.float64)
    if cov.shape[-2:] != (3, 3) or mat.shape != (3, 3):
        raise ValueError("noise covariance and transform must end in 3x3")
    if not np.isfinite(cov).all() or not np.isfinite(mat).all():
        raise ValueError("noise covariance and transform must be finite")
    if not np.allclose(cov, np.swapaxes(cov, -1, -2), atol=1e-14):
        raise ValueError("noise covariance must be symmetric")
    if np.any(np.linalg.eigvalsh(cov) < -1e-14):
        raise ValueError("noise covariance must be positive semidefinite")
    return np.einsum("ij,...jk,lk->...il", mat, cov, mat)


def area_variance(independent_variance, out_h: int, out_w: int):
    """Exact variance of an area mean of spatially independent samples.

    The weights are squared, including fractional edge cells. Do not call
    this on demosaiced/warped pixels unless their cross-pixel covariance has
    already been accounted for: plain variance divided by area is not valid
    for those correlated inputs.
    """
    src = np.asarray(independent_variance, dtype=np.float64)
    if src.ndim < 2 or min(out_h, out_w) < 1:
        raise ValueError("variance needs image axes and a positive output shape")
    if not np.isfinite(src).all() or np.any(src < 0):
        raise ValueError("variance must be finite and nonnegative")
    h, w = src.shape[:2]
    if out_h > h or out_w > w:
        raise ValueError("noise area propagation only supports decimation")
    result = src
    for axis, n, out_n in ((1, w, out_w), (0, h, out_h)):
        shape = list(result.shape)
        shape[axis] = out_n
        reduced = np.empty(shape, dtype=np.float64)
        edges = np.arange(out_n + 1, dtype=np.float64) * n / out_n
        for i in range(out_n):
            lo, hi = edges[i:i + 2]
            start, stop = int(np.floor(lo)), min(int(np.ceil(hi)), n)
            positions = np.arange(start, stop)
            weights = np.maximum(np.minimum(positions + 1, hi) - np.maximum(positions, lo), 0) / (hi - lo)
            sl = [slice(None)] * result.ndim
            sl[axis] = slice(start, stop)
            weight_shape = [1] * result.ndim
            weight_shape[axis] = weights.size
            dest = [slice(None)] * result.ndim
            dest[axis] = i
            reduced[tuple(dest)] = np.sum(result[tuple(sl)] * np.square(weights).reshape(weight_shape), axis=axis)
        result = reduced
    return result


def _convolve_axis_zero(plane, kernel, axis):
    """Zero-padded convolution; callers exclude boundary support."""
    arr = np.asarray(plane, dtype=np.float64)
    taps = np.asarray(kernel, dtype=np.float64)
    radius = len(taps) // 2
    if len(taps) <= 17:
        pad = [(0, 0)] * arr.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(arr, pad, mode="constant")
        out = np.zeros_like(arr)
        for i, tap in enumerate(taps):
            if tap:
                sl = [slice(None)] * arr.ndim
                sl[axis] = slice(i, i + arr.shape[axis])
                out += tap * padded[tuple(sl)]
        return out
    n_fft = 1 << (arr.shape[axis] + len(taps) - 2).bit_length()
    spectrum = np.fft.rfft(arr, n=n_fft, axis=axis)
    shape = [1] * arr.ndim
    shape[axis] = n_fft // 2 + 1
    spectrum *= np.fft.rfft(taps, n=n_fft).reshape(shape)
    out = np.fft.irfft(spectrum, n=n_fft, axis=axis)
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(radius, radius + arr.shape[axis])
    return out[tuple(sl)]


def _separable_variance_filter(variance, kernel):
    return _convolve_axis_zero(_convolve_axis_zero(variance, kernel, 1), kernel, 0)


def atrous_detail_variance(independent_variance, level: int):
    """Variance of a B3 à-trous detail from independent input samples.

    Previous scales share samples. We square the *accumulated* detail
    filter, rather than add the variance of two correlated smooth images.
    Reflection reuses samples at the boundary. Those pixels are assigned
    zero authority here and therefore left untouched by the denoiser.
    """
    var = np.asarray(independent_variance, dtype=np.float64)
    if var.ndim != 2 or not np.isfinite(var).all() or np.any(var < 0):
        raise ValueError("independent noise variance must be a finite nonnegative plane")
    if not isinstance(level, int) or level < 0 or level > 12:
        raise ValueError("unsupported wavelet level")
    previous = np.asarray([1.0], dtype=np.float64)
    b3 = np.asarray([1., 4., 6., 4., 1.]) / 16.
    for k in range(level + 1):
        step = 1 << k
        holes = np.zeros(4 * step + 1, dtype=np.float64)
        holes[::step] = b3
        current = np.convolve(previous, holes)
        if k == level:
            break
        previous = current
    padding = (len(current) - len(previous)) // 2
    previous = np.pad(previous, (padding, padding))
    radius = len(current) // 2
    if 2 * radius >= min(var.shape):
        return np.zeros_like(var, dtype=np.float32)
    if np.all(var == var.flat[0]):
        energy = (np.square(previous).sum() ** 2 + np.square(current).sum() ** 2
                  - 2 * (previous * current).sum() ** 2)
        out = np.full_like(var, var.flat[0] * energy)
    else:
        out = _separable_variance_filter(var, np.square(previous))
        out += _separable_variance_filter(var, np.square(current))
        out -= 2 * _separable_variance_filter(var, previous * current)
    out[:radius] = 0
    out[-radius:] = 0
    out[:, :radius] = 0
    out[:, -radius:] = 0
    return np.maximum(out, 0).astype(np.float32)
