# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatially supported, multiscale delivery-detail backstop.

Finished master and decoded file luminance are compared at the same positions.
At distances 1/2/4/8, an 8x8 tile averages its largest 16 clipped coefficient
losses. Directions are combined by position before spatial counting. A single
bad pixel cannot reach the catastrophic score of 1. Partial tiles keep the full
support divisor. The SDR 8-code floor and HDR linear approximation discount small
quantization/dither errors. HDR also retains the .01 absolute and 8% endpoint
mean magnitude floors. Coefficients below that floor
also receive proportional reference-activity weight; flat areas cannot turn
codec ringing into an allegation that existing detail was lost.

This backstop does not estimate sensor noise or certify weak texture. Equal
variance with unrelated noise does not restore signed coefficients. Its floors
and .90 absolute limit are engineering budgets, not perceptual equivalence
claims. Auto additionally bounds the worst scalar against its high-quality
reference; it does not compare position grids. Local chroma-texture preservation
is NOT guaranteed: fine chroma is intentionally lossy in 4:2:0 and remains
constrained by the existing pixel-chroma and coding-error budgets.
Only bounded row bands/tiles are allocated, never a complete float32 pyramid.
"""
from __future__ import annotations

import math

from ._deps import np
from .hdr_color import output_luma_weights

_DETAIL_ROWS = 32  # multiple of 8; global tile origins stay fixed
_DETAIL_SCALES = (1, 2, 4, 8)
LOCAL_DETAIL_LOSS_LIMIT = .90
_SDR_LUMA = np.array([.299, .587, .114], np.float32)
_HDR_LUMA = output_luma_weights('p3').astype(np.float32)
# Conservative derivative-based linear budget for 8 nonlinear sRGB codes. The exact
# inverse-transfer derivative is this factor times Y**(7/12); sqrt(Y) bounds it
# above for 0 < Y < 1, and the existing .08*Y floor dominates at Y >= 1. The
# .01 floor also covers the sRGB toe. This is a delivery-error budget, not an
# estimate of sensor noise or a visibility threshold.
_HDR_EIGHT_CODE_LINEAR = np.float32((8. / 255.) * (2.4 / 1.055))


def _tile_loss(score, tile_size=8):
    h, w = score.shape
    hp, wp = (h + tile_size - 1) // tile_size * tile_size, (w + tile_size - 1) // tile_size * tile_size
    if (h, w) != (hp, wp):
        padded = np.zeros((hp, wp), np.float32)
        padded[:h, :w] = score
        score = padded
    tiles = score.reshape(hp // tile_size, tile_size, wp // tile_size, tile_size).transpose(0, 2, 1, 3).reshape(-1, tile_size * tile_size)
    count = tile_size * tile_size // 4
    top = np.partition(tiles, tiles.shape[1] - count, axis=1)[:, -count:]
    return float(np.max(np.mean(top, axis=1, dtype=np.float32)))


def _measure_detail(decoded, intended, *, linear_hdr, sdr_code_scale=1.):
    height, width = intended.shape[:2]
    weights = _HDR_LUMA if linear_hdr else _SDR_LUMA
    worst = 0.0
    floor = .01 if linear_hdr else 8.0
    for row in range(0, height, _DETAIL_ROWS):
        rows = min(_DETAIL_ROWS, height - row)
        stop = min(height, row + rows + _DETAIL_SCALES[-1])
        a = decoded[row:stop, :, :3].astype(np.float32)
        e = intended[row:stop, :, :3].astype(np.float32)
        if sdr_code_scale != 1.:
            # Normalized nonlinear SDR remains floating point. This only
            # expresses the established eight-code floor in the same units.
            a *= sdr_code_scale
            e *= sdr_code_scale
        if not (np.isfinite(a).all() and np.isfinite(e).all()):
            return float('inf')
        a = ((a[..., 0] * weights[0] + a[..., 1] * weights[1]) + a[..., 2] * weights[2])[..., None]
        e = ((e[..., 0] * weights[0] + e[..., 1] * weights[1]) + e[..., 2] * weights[2])[..., None]
        if not (np.isfinite(a).all() and np.isfinite(e).all()):
            return float('inf')
        for distance in _DETAIL_SCALES:
            score = np.zeros((rows, width), np.float32)
            for vertical in (False, True):
                count = min(rows, height - row - distance) if vertical else width - distance
                if count <= 0:
                    continue
                if vertical:
                    aa, ab = a[:count], a[distance:distance + count]
                    ea, eb = e[:count], e[distance:distance + count]
                else:
                    aa, ab = a[:rows, :count], a[:rows, distance:distance + count]
                    ea, eb = e[:rows, :count], e[:rows, distance:distance + count]
                reference = eb - ea
                error = ab - aa
                error -= reference
                np.abs(error, out=error)
                np.abs(reference, out=reference)
                if linear_hdr:
                    magnitude = np.abs(ea) + np.abs(eb)
                    quantization = magnitude * .5
                    np.sqrt(quantization, out=quantization)
                    quantization *= _HDR_EIGHT_CODE_LINEAR
                    magnitude *= .04
                    np.maximum(magnitude, floor, out=magnitude)
                    np.maximum(magnitude, quantization, out=magnitude)
                else:
                    magnitude = floor
                denominator = np.maximum(reference, magnitude)
                reference /= magnitude
                np.minimum(reference, 1., out=reference)
                error /= denominator
                np.minimum(error, 1., out=error)
                error *= reference
                scalar = np.max(error, axis=2)
                target = score[:count] if vertical else score[:, :count]
                np.maximum(target, scalar, out=target)
            worst = max(worst, _tile_loss(score))
    return worst


def _valid_detail_inputs(decoded, intended, linear_hdr):
    if not (isinstance(decoded, np.ndarray) and isinstance(intended, np.ndarray)):
        return False
    if (decoded.shape != intended.shape or decoded.ndim != 3
            or decoded.shape[2] not in (3, 4) or not decoded.size):
        return False
    allowed = (np.dtype(np.uint8), np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)) if linear_hdr else (np.dtype(np.uint8),)
    return all(a.dtype in allowed and a.dtype.isnative and a.flags.aligned for a in (decoded, intended))


def _local_detail_loss_numpy(decoded, intended, *, linear_hdr=False):
    """Independent banded NumPy reference, including invalid-input rejection."""
    if not _valid_detail_inputs(decoded, intended, linear_hdr):
        return float('inf')
    try:
        # Finite input can still overflow Y, differences or normalized errors.
        # Never let max(0,nan) turn an invalid computation into a clean result.
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            return _measure_detail(decoded, intended, linear_hdr=linear_hdr)
    except FloatingPointError:
        return float('inf')


def local_detail_loss(decoded, intended, *, linear_hdr=False):
    """Worst supported luma-detail distortion, or inf for invalid RGB.

    RGB/RGBA strided sources remain untouched; alpha is ignored. HDR accepts
    u8/f16/f32/f64 as f32 samples, with no implicit normalization or full-frame
    forcecast; SDR accepts u8. Non-native byte order/unaligned storage is rejected.
    """
    if not _valid_detail_inputs(decoded, intended, linear_hdr):
        return float('inf')
    from . import _fast
    native = _fast.kernel('local_detail_loss')
    if native is not None:
        try:
            weights = _HDR_LUMA if linear_hdr else _SDR_LUMA
            result = float(native(decoded, intended, linear_hdr, weights.tolist()))
            return result if math.isfinite(result) and 0. <= result <= 1. else float('inf')
        except ValueError:
            # Invalid source or derived arithmetic is a failed measurement,
            # not an invitation to accept a different fallback interpretation.
            return float('inf')
        except Exception as exc:
            _fast.handle_kernel_error('local_detail_loss', exc)
    return _local_detail_loss_numpy(decoded, intended, linear_hdr=linear_hdr)


def _valid_sdr_float_inputs(decoded, intended):
    """Normalized nonlinear master and finite decoded SDR; never forcecast.

    Decoded YCbCr reconstruction may overshoot the normalized master domain.
    Keep those errors measurable rather than clipping them away. Alpha is
    ignored, matching the existing detail contract.
    """
    if not (isinstance(decoded, np.ndarray) and isinstance(intended, np.ndarray)):
        return False
    if (decoded.shape != intended.shape or decoded.ndim != 3
            or decoded.shape[2] not in (3, 4) or not decoded.size):
        return False
    allowed = (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64))
    if not all(a.dtype in allowed and a.dtype.isnative and a.flags.aligned
               for a in (decoded, intended)):
        return False
    for row in range(0, decoded.shape[0], 128):
        a, e = decoded[row:row + 128, :, :3], intended[row:row + 128, :, :3]
        if not (np.isfinite(a).all() and np.isfinite(e).all()
                and np.all(e >= 0.) and np.all(e <= 1.)):
            return False
    return True


def local_detail_loss_sdr_float(decoded, intended):
    """Detail backstop for nonlinear SDR floats in normalized RGB units.

    The independent NumPy path measures fractional codes directly; it never
    enters the uint8 native kernel or quantizes the input. Multiplication by
    255 preserves the existing eight-code activity floor and delivery budget.
    """
    if not _valid_sdr_float_inputs(decoded, intended):
        return float('inf')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            return _measure_detail(decoded, intended, linear_hdr=False, sdr_code_scale=255.)
    except FloatingPointError:
        return float('inf')


def detail_is_acceptable(metrics, key, limit=LOCAL_DETAIL_LOSS_LIMIT):
    """Missing/non-finite measurements cannot silently bypass a delivery gate."""
    try:
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(value) and value <= float(limit)
