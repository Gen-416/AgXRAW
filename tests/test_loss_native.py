# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent pre-native loss-mask oracles and ABI 15 dispatch contracts.

The reference bodies deliberately retain NumPy's operand order, 128-row bands,
invalid-footprint zeros, and Pillow's float32 -> nearest -> float16 sequence.
Comparisons use bytes: allclose/array_equal would hide signed zero or NaN payloads.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from dngscan import _fast, dng_opcodes, raw_io


CROP_KERNEL = "crop_loss_footprint"
MERGE_KERNEL = "merge_processing_loss_f16_inplace"


def _crop_indices(values, crop, sensor_shape, output_shape):
    y, x, h, w = crop
    sy, sx = values.shape[0] / sensor_shape[0], values.shape[1] / sensor_shape[1]
    t, l = max(0., y * sy), max(0., x * sx)
    b, r = min(values.shape[0], (y + h) * sy), min(values.shape[1], (x + w) * sx)
    if all(abs(v - round(v)) < 1e-9 for v in (t, l, b, r)):
        return None
    if b <= t or r <= l:
        raise ValueError("DNG crop is outside the sensor evidence")
    oh, ow = output_shape
    ys, xs = np.linspace(t, b, oh + 1), np.linspace(l, r, ow + 1)
    return (np.floor(ys[:-1] + 1e-9).astype(np.intp),
            np.ceil(ys[1:] - 1e-9).astype(np.intp),
            np.floor(xs[:-1] + 1e-9).astype(np.intp),
            np.ceil(xs[1:] - 1e-9).astype(np.intp))


def _legacy_footprint(values, ylo, yhi, xlo, xhi):
    oh, ow = len(ylo), len(xlo)
    out = np.zeros((oh, ow, 3), dtype=values.dtype)
    for start in range(0, oh, 128):
        stop = min(start + 128, oh)
        for dy in range(int(np.max(yhi[start:stop] - ylo[start:stop]))):
            yy = ylo[start:stop] + dy
            for dx in range(int(np.max(xhi - xlo))):
                xx = xlo + dx
                valid = ((yy[:, None] < yhi[start:stop, None]) &
                         (xx[None, :] < xhi[None, :]))
                src = values[np.minimum(yy, values.shape[0] - 1)[:, None],
                             np.minimum(xx, values.shape[1] - 1)[None, :]]
                np.maximum(out[start:stop], np.where(valid[..., None], src, 0),
                           out=out[start:stop])
    return out


def _legacy_crop(values, crop, sensor_shape, output_shape):
    indices = _crop_indices(values, crop, sensor_shape, output_shape)
    return None if indices is None else _legacy_footprint(values, *indices)


def _legacy_resize(processing, shape):
    if processing.shape[:2] == shape:
        return processing
    out = np.empty((*shape, 3), dtype=np.float16)
    for c in range(3):
        plane = Image.fromarray(np.asarray(processing[..., c], dtype=np.float32))
        out[..., c] = np.asarray(plane.resize((shape[1], shape[0]), Image.Resampling.NEAREST))
    return out


def _legacy_merge(masks, processing):
    if processing is not None:
        np.maximum(masks, _legacy_resize(processing, masks.shape[:2]), out=masks)
    return masks


def _nearest_indices(source, target):
    # Use Pillow itself, not an assumed floor((x+.5)*scale) approximation.
    plane = Image.fromarray(np.arange(source, dtype=np.int32)[None, :])
    return np.asarray(plane.resize((target, 1), Image.Resampling.NEAREST))[0].astype(np.intp)


def _orient(values, flip):
    # LibRaw mirrors in source coordinates, then transposes.
    out = values[:, ::-1] if flip & 1 else values
    out = out[::-1] if flip & 2 else out
    return out.transpose(1, 0, 2) if flip & 4 else out


def _special_values(dtype, shape):
    if dtype == np.float16:
        bits = np.array([0, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00,
                         0x7e01, 0xfe13, 0x7c01, 0xfc23], dtype=np.uint16)
    else:
        bits = np.array([0, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000,
                         0xff800000, 0x7fc00001, 0xffc00123, 0x7f800001,
                         0xff800123], dtype=np.uint32)
    return np.resize(bits.view(dtype), shape).copy()


def _native_extension():
    ext = _fast._load_extension()
    return ext if ext is not None and all(hasattr(ext, name) for name in (CROP_KERNEL, MERGE_KERNEL)) else None


class _BitsTest(unittest.TestCase):
    def assertBitsEqual(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.tobytes(order="C"), expected.tobytes(order="C"))


class LossOracleTests(_BitsTest):
    def test_odd_crops_epsilon_boundaries_and_single_axes_match_legacy(self):
        rng = np.random.default_rng(731)
        cases = (
            ((7, 11), (14, 22), (1., 3., 11., 17.), (13, 19)),
            ((7, 11), (14, 22), (-1., -.25, 11., 17.), (129, 1)),
            ((1, 11), (1, 22), (0., 1., 1., 19.), (1, 17)),
            ((11, 1), (22, 1), (1., 0., 19., 1.), (17, 1)),
            ((7, 11), (7, 11), (2. + 0.5e-9, 1., 3., 7.), (3, 7)),
            ((7, 11), (7, 11), (2. + 1.5e-9, 1., 3., 7.), (3, 7)),
            ((7, 11), (7, 11), (2. - 1.5e-9, 1., 3., 7.), (3, 7)),
        )
        for dtype in (np.float16, np.float32, np.float64):
            for shape, sensor, crop, output in cases:
                with self.subTest(dtype=dtype, shape=shape, crop=crop):
                    values = rng.uniform(-1., 1., (*shape, 3)).astype(dtype)
                    expected = _legacy_crop(values, crop, sensor, output)
                    with mock.patch.object(_fast, "kernel", return_value=None):
                        actual = dng_opcodes._fractional_crop_loss(values, crop, sensor, output)
                    if expected is None:
                        self.assertIsNone(actual)
                    else:
                        self.assertBitsEqual(actual, expected)

    def test_crop_orientation_uses_fractional_footprint_before_all_eight_flips(self):
        values = np.arange(7 * 11 * 3, dtype=np.float32).reshape(7, 11, 3) / 231
        sensor, crop, unrotated = (14, 22), (1., 3., 11., 17.), (9, 13)
        expected = _legacy_crop(values, crop, sensor, unrotated)
        for flip in range(8):
            scene = unrotated[::-1] if flip & 4 else unrotated
            with self.subTest(flip=flip):
                actual = dng_opcodes.align_sensor_loss(values, sensor, scene, flip, crop=crop)
                self.assertBitsEqual(actual, _orient(expected, flip))

    def test_merge_pillow_order_and_special_bits_for_small_odd_shapes(self):
        for dtype in (np.float16, np.float32, np.float64):
            for source, target in (((3, 5), (3, 5)), ((3, 5), (7, 9)),
                                   ((17, 29), (7, 3)), ((1, 7), (9, 1)), ((7, 1), (1, 9))):
                with self.subTest(dtype=dtype, source=source, target=target):
                    with np.errstate(invalid="ignore", over="ignore"):
                        values = _special_values(np.float16 if dtype == np.float16 else np.float32,
                                                 (*source, 3)).astype(dtype)
                        masks = _special_values(np.float16, (*target, 3))
                        expected = _legacy_merge(masks.copy(), values)
                        with mock.patch.object(_fast, "kernel", return_value=None):
                            actual = raw_io._merge_processing_loss(masks.copy(), values)
                    self.assertBitsEqual(actual, expected)


class LossDispatchTests(_BitsTest):
    @contextmanager
    def fake_extension(self, *, crop_result="oracle", merge_result="oracle"):
        def crop(values, ylo, yhi, xlo, xhi):
            return _legacy_footprint(values, ylo, yhi, xlo, xhi) if crop_result == "oracle" else crop_result

        def merge(masks, processing, y_indices=None, x_indices=None):
            if merge_result != "oracle":
                return merge_result
            return _legacy_merge(masks, processing)

        crop_call, merge_call = mock.Mock(side_effect=crop), mock.Mock(side_effect=merge)
        ext = SimpleNamespace(**{CROP_KERNEL: crop_call, MERGE_KERNEL: merge_call})
        with mock.patch.object(_fast, "_load_extension", return_value=ext), \
             mock.patch.dict(os.environ, DNGSCAN_FAST="1", DNGSCAN_FAST_SKIP=""):
            yield crop_call, merge_call

    def test_crop_dispatch_borrows_strided_readonly_dtype_without_whole_copy(self):
        for dtype in (np.float16, np.float32):
            source = np.arange(7 * 11 * 3, dtype=dtype).reshape(7, 11, 3)
            for values in (source[::-1, ::-1, ::-1], source.transpose(1, 0, 2),
                           np.broadcast_to(source[:1, :1], source.shape)):
                values.flags.writeable = False
                sensor = (values.shape[0] * 2, values.shape[1] * 2)
                crop = (1., 1., sensor[0] - 2., sensor[1] - 2.)
                expected = _legacy_crop(values, crop, sensor, (13, 17))
                before = values.tobytes()
                with self.fake_extension() as (call, _):
                    actual = dng_opcodes._fractional_crop_loss(values, crop, sensor, (13, 17))
                call.assert_called_once()
                self.assertIs(call.call_args.args[0], values)
                self.assertEqual(call.call_args.args[0].dtype, dtype)
                self.assertEqual(before, values.tobytes())
                self.assertBitsEqual(actual, expected)

    def test_merge_dispatch_borrows_dtype_and_uses_exact_pillow_nearest_maps(self):
        for dtype in (np.float16, np.float32):
            source = np.linspace(-.2, 1.2, 17 * 29 * 3, dtype=dtype).reshape(17, 29, 3)
            for processing in (source[::-1, ::-1, ::-1], np.broadcast_to(source[:1, :1], source.shape)):
                processing.flags.writeable = False
                for shape in ((17, 29), (7, 13), (19, 1), (1, 37)):
                    masks = np.full((*shape, 3), .2, np.float16)
                    expected = _legacy_merge(masks.copy(), processing)
                    before, address = processing.tobytes(), masks.ctypes.data
                    with self.fake_extension() as (_, call):
                        actual = raw_io._merge_processing_loss(masks, processing)
                    self.assertIs(actual, masks)
                    self.assertEqual(masks.ctypes.data, address)
                    self.assertEqual(processing.tobytes(), before)
                    self.assertBitsEqual(actual, expected)
                    if dtype == np.float32 and shape == processing.shape[:2]:
                        call.assert_not_called()
                        continue
                    call.assert_called_once()
                    self.assertIs(call.call_args.args[0], masks)
                    self.assertIs(call.call_args.args[1], processing)
                    args, kwargs = call.call_args.args, call.call_args.kwargs
                    yi = args[2] if len(args) > 2 else kwargs.get("y_indices")
                    xi = args[3] if len(args) > 3 else kwargs.get("x_indices")
                    if shape == processing.shape[:2]:
                        self.assertIsNone(yi)
                        self.assertIsNone(xi)
                    else:
                        np.testing.assert_array_equal(yi, _nearest_indices(17, shape[0]))
                        np.testing.assert_array_equal(xi, _nearest_indices(29, shape[1]))
                        self.assertEqual(yi.dtype, np.dtype(np.intp))
                        self.assertEqual(xi.dtype, np.dtype(np.intp))

    def test_same_shape_float32_keeps_numpy_simd_without_native_lookup(self):
        source = np.linspace(-.2, 1.2, 7 * 11 * 3, dtype=np.float32).reshape(7, 11, 3)
        special = _special_values(np.float32, source.shape)
        for mode in ("auto", "1"):
            for processing in (source, source[::-1, ::-1, ::-1], special):
                with self.subTest(mode=mode, strides=processing.strides), np.errstate(invalid="ignore"):
                    processing.flags.writeable = False
                    masks = np.full(processing.shape, .2, np.float16)
                    before, address = processing.tobytes(), masks.ctypes.data
                    expected = _legacy_merge(masks.copy(), processing)
                    with mock.patch.dict(os.environ, DNGSCAN_FAST=mode), \
                         mock.patch.object(_fast, "kernel", side_effect=AssertionError("mixed SIMD must not dispatch")) as lookup:
                        result = raw_io._merge_processing_loss(masks, processing)
                    lookup.assert_not_called()
                    self.assertIs(result, masks)
                    self.assertEqual(masks.ctypes.data, address)
                    self.assertEqual(processing.tobytes(), before)
                    self.assertBitsEqual(result, expected)

    def test_skip_keys_bypass_each_native_call_under_strict_mode(self):
        values = np.ones((3, 5, 3), np.float32)
        crop, sensor, shape = (1., 1., 3., 7.), (6, 10), (7, 9)
        with self.fake_extension() as (crop_call, merge_call):
            with mock.patch.dict(os.environ, DNGSCAN_FAST_SKIP=CROP_KERNEL):
                actual = dng_opcodes._fractional_crop_loss(values, crop, sensor, shape)
                self.assertBitsEqual(actual, _legacy_crop(values, crop, sensor, shape))
            with mock.patch.dict(os.environ, DNGSCAN_FAST_SKIP=MERGE_KERNEL):
                masks = np.zeros((*shape, 3), np.float16)
                actual = raw_io._merge_processing_loss(masks, values)
                self.assertBitsEqual(actual, _legacy_merge(np.zeros_like(masks), values))
        crop_call.assert_not_called()
        merge_call.assert_not_called()

    def test_explicit_unsupported_returns_to_legacy_even_in_strict_mode(self):
        values = _special_values(np.float32, (3, 5, 3))
        crop, sensor, shape = (1., 1., 3., 7.), (6, 10), (7, 9)
        masks = _special_values(np.float16, (*shape, 3))
        with np.errstate(invalid="ignore", over="ignore"):
            expected_crop = _legacy_crop(values, crop, sensor, shape)
            expected_merge = _legacy_merge(masks.copy(), values)
            with self.fake_extension(crop_result=None, merge_result=None) as (crop_call, merge_call):
                actual_crop = dng_opcodes._fractional_crop_loss(values, crop, sensor, shape)
                actual_merge = raw_io._merge_processing_loss(masks, values)
        crop_call.assert_called_once()
        merge_call.assert_called_once()
        self.assertBitsEqual(actual_crop, expected_crop)
        self.assertBitsEqual(actual_merge, expected_merge)

    def test_unsupported_dtype_layout_and_alias_keep_legacy_without_casting(self):
        values = np.arange(3 * 5 * 3, dtype=np.float64).reshape(3, 5, 3)
        unaligned = np.ndarray((3, 5, 3), dtype=np.float32,
                               buffer=bytearray(3 * 5 * 3 * 4 + 1), offset=1)
        unaligned[...] = 1.
        unaligned_masks = np.ndarray((3, 5, 3), dtype=np.float16,
                                     buffer=bytearray(3 * 5 * 3 * 2 + 1), offset=1)
        with self.fake_extension() as (crop_call, merge_call):
            for source in (values, unaligned):
                actual = dng_opcodes._fractional_crop_loss(source, (1., 1., 3., 7.), (6, 10), (7, 9))
                self.assertBitsEqual(actual, _legacy_crop(source, (1., 1., 3., 7.), (6, 10), (7, 9)))
            for masks, processing in (
                (np.zeros((3, 5, 3), np.float16), values),
                (np.zeros((3, 5, 3), np.float32), values.astype(np.float32)),
                (np.zeros((3, 5, 3), np.float16)[:, ::-1], values.astype(np.float16)),
                (np.zeros((3, 5, 3), np.float16), unaligned),
                (unaligned_masks, values.astype(np.float16)),
            ):
                self.assertBitsEqual(raw_io._merge_processing_loss(masks, processing),
                                     _legacy_merge(np.zeros_like(masks), processing))
            for view in (lambda a: a, lambda a: a[::-1], lambda a: a[::2, ::2]):
                masks = np.arange(5 * 7 * 3, dtype=np.float16).reshape(5, 7, 3)
                expected = masks.copy()
                _legacy_merge(expected, view(expected))
                self.assertIs(raw_io._merge_processing_loss(masks, view(masks)), masks)
                self.assertBitsEqual(masks, expected)
            readonly = np.zeros((3, 5, 3), np.float16)
            readonly.flags.writeable = False
            before = readonly.tobytes()
            with self.assertRaises(ValueError):
                raw_io._merge_processing_loss(readonly, values.astype(np.float16))
            self.assertEqual(readonly.tobytes(), before)
        crop_call.assert_not_called()
        merge_call.assert_not_called()

    def test_kernel_exceptions_fallback_in_auto_but_propagate_in_strict(self):
        values = np.ones((3, 5, 3), np.float32)
        crop, sensor, shape = (1., 1., 3., 7.), (6, 10), (7, 9)
        with self.fake_extension() as (crop_call, merge_call):
            crop_call.side_effect = ValueError("crop preflight failure")
            merge_call.side_effect = ValueError("merge preflight failure")
            with self.assertRaises(_fast.NativeKernelError):
                dng_opcodes._fractional_crop_loss(values, crop, sensor, shape)
            masks = np.zeros((*shape, 3), np.float16)
            before = masks.copy()
            with self.assertRaises(_fast.NativeKernelError):
                raw_io._merge_processing_loss(masks, values)
            self.assertBitsEqual(masks, before)
            with mock.patch.dict(os.environ, DNGSCAN_FAST="auto"), self.assertLogs("dngscan._fast", level="WARNING"):
                actual_crop = dng_opcodes._fractional_crop_loss(values, crop, sensor, shape)
                actual_merge = raw_io._merge_processing_loss(masks, values)
            self.assertBitsEqual(actual_crop, _legacy_crop(values, crop, sensor, shape))
            self.assertBitsEqual(actual_merge, _legacy_merge(before, values))

    def test_swapped_float_byteorder_uses_legacy_without_reinterpreting_bits(self):
        crop, sensor, shape = (1., 1., 3., 7.), (6, 10), (7, 9)
        with self.fake_extension() as (crop_call, merge_call), np.errstate(invalid="ignore"):
            for dtype in (np.dtype(np.float16), np.dtype(np.float32)):
                values = _special_values(dtype, (3, 5, 3)).astype(dtype.newbyteorder("S"))
                self.assertFalse(values.dtype.isnative)
                before = values.tobytes()
                self.assertBitsEqual(dng_opcodes._fractional_crop_loss(values, crop, sensor, shape),
                                     _legacy_crop(values, crop, sensor, shape))
                for target_shape in ((3, 5), shape):
                    masks = np.full((*target_shape, 3), .2, np.float16)
                    expected = _legacy_merge(masks.copy(), values)
                    self.assertBitsEqual(raw_io._merge_processing_loss(masks, values), expected)
                self.assertEqual(values.tobytes(), before)
            masks = np.full((3, 5, 3), .2, dtype=np.dtype(np.float16).newbyteorder("S"))
            processing = np.full((3, 5, 3), .5, np.float16)
            expected = _legacy_merge(masks.copy(), processing)
            self.assertBitsEqual(raw_io._merge_processing_loss(masks, processing), expected)
        crop_call.assert_not_called()
        merge_call.assert_not_called()

    def test_empty_output_crop_retains_legacy_result_or_error_without_native(self):
        values = np.ones((3, 5, 3), np.float32)
        crop, sensor = (1., 1., 3., 7.), (6, 10)
        with self.fake_extension() as (crop_call, _):
            for shape in ((0, 5), (0, 0)):
                self.assertBitsEqual(dng_opcodes._fractional_crop_loss(values, crop, sensor, shape),
                                     _legacy_crop(values, crop, sensor, shape))
            # The original positive-height/zero-width loop reduces an empty
            # footprint array and raises. Do not invent new empty semantics.
            with self.assertRaises(ValueError):
                _legacy_crop(values, crop, sensor, (5, 0))
            with self.assertRaises(ValueError):
                dng_opcodes._fractional_crop_loss(values, crop, sensor, (5, 0))
        crop_call.assert_not_called()


class LossNativeTests(_BitsTest):
    @classmethod
    def setUpClass(cls):
        cls.ext = _native_extension()
        if cls.ext is None:
            raise unittest.SkipTest("ABI 15 loss kernels not built")

    def test_crop_finite_strides_and_invalid_footprint_zero_order_are_bit_exact(self):
        rng = np.random.default_rng(829)
        for dtype in (np.float16, np.float32):
            source = rng.uniform(-.5, 1.2, (7, 11, 3)).astype(dtype)
            for values in (source, source[::-1, ::-1, ::-1], source.transpose(1, 0, 2),
                           np.broadcast_to(source[:1, :1], source.shape)):
                values.flags.writeable = False
                sensor = (values.shape[0] * 2, values.shape[1] * 2)
                crop = (1., 1., sensor[0] - 2., sensor[1] - 2.)
                for shape in ((1, 17), (131, 1), (129, 7)):
                    indices = _crop_indices(values, crop, sensor, shape)
                    before = values.tobytes()
                    actual = self.ext.crop_loss_footprint(values, *indices)
                    self.assertIsNotNone(actual)
                    self.assertBitsEqual(actual, _legacy_footprint(values, *indices))
                    self.assertEqual(values.tobytes(), before)
                    self.assertFalse(np.shares_memory(actual, values))
        # Zero-width footprints mixed with wider ones still execute invalid->+0.
        values = np.full((3, 3, 3), -0., np.float16)
        indices = tuple(np.array(x, dtype=np.intp) for x in ([0, 1, 2], [2, 1, 3], [0, 1], [1, 3]))
        self.assertBitsEqual(self.ext.crop_loss_footprint(values, *indices),
                             _legacy_footprint(values, *indices))
        edge_indices = tuple(np.array(x, dtype=np.intp) for x in ([3, 0], [3, 3], [3, 0], [3, 3]))
        self.assertBitsEqual(self.ext.crop_loss_footprint(values, *edge_indices),
                             _legacy_footprint(values, *edge_indices))

    def test_merge_direct_and_nearest_are_bit_exact_and_inplace(self):
        rng = np.random.default_rng(923)
        for dtype in (np.float16, np.float32):
            source = rng.uniform(-.4, 1.4, (17, 29, 3)).astype(dtype)
            # Values just either side of a half rounding boundary must follow
            # direct maximum->half or resized half->maximum, as applicable.
            source.flat[:3] = [.5002441, .5002442, -.00000002]
            for processing in (source, source[::-1, ::-1, ::-1],
                               np.broadcast_to(source[:1, :1], source.shape)):
                processing.flags.writeable = False
                for shape in ((17, 29), (7, 13), (19, 1), (1, 37)):
                    masks = rng.uniform(-.3, 1.1, (*shape, 3)).astype(np.float16)
                    expected = _legacy_merge(masks.copy(), processing)
                    before, address = processing.tobytes(), masks.ctypes.data
                    maps = (None, None) if shape == processing.shape[:2] else (
                        _nearest_indices(17, shape[0]), _nearest_indices(29, shape[1]))
                    result = self.ext.merge_processing_loss_f16_inplace(masks, processing, *maps)
                    self.assertIs(result, masks)
                    self.assertBitsEqual(masks, expected)
                    self.assertEqual(masks.ctypes.data, address)
                    self.assertEqual(processing.tobytes(), before)

    def test_special_values_match_legacy_or_explicitly_decline_before_write(self):
        for dtype in (np.float16, np.float32):
            processing = _special_values(dtype, (3, 5, 3))
            before = processing.tobytes()
            indices = _crop_indices(processing, (1., 1., 3., 7.), (6, 10), (7, 9))
            with np.errstate(invalid="ignore", over="ignore"):
                actual = self.ext.crop_loss_footprint(processing, *indices)
                if actual is not None:
                    self.assertBitsEqual(actual, _legacy_footprint(processing, *indices))
                elif dtype == np.float16:
                    self.fail("float16 crop unexpectedly declined supported bitwise maximum")
                self.assertBitsEqual(
                    dng_opcodes._fractional_crop_loss(processing, (1., 1., 3., 7.), (6, 10), (7, 9)),
                    _legacy_footprint(processing, *indices))
                for shape in ((3, 5), (7, 9)):
                    masks = _special_values(np.float16, (*shape, 3))
                    prior = masks.copy()
                    maps = (None, None) if shape == (3, 5) else (
                        _nearest_indices(3, shape[0]), _nearest_indices(5, shape[1]))
                    result = self.ext.merge_processing_loss_f16_inplace(masks, processing, *maps)
                    if result is None:
                        self.assertBitsEqual(masks, prior)
                    else:
                        self.assertIs(result, masks)
                        self.assertBitsEqual(masks, _legacy_merge(prior.copy(), processing))
                    public = raw_io._merge_processing_loss(prior.copy(), processing)
                    self.assertBitsEqual(public, _legacy_merge(prior.copy(), processing))
            self.assertEqual(processing.tobytes(), before)

    def test_half_conversion_boundaries_and_operation_order_match_oracle(self):
        # Exact binary midpoints with adjacent float32 neighbours: zero/subnormal,
        # subnormal/normal, even/odd normal mantissas, and finite/overflow.
        boundaries = (
            ("zero/subnormal", 2.**-25, (0x0000, 0x0000, 0x0001)),
            ("odd subnormal", 3. * 2.**-25, (0x0001, 0x0002, 0x0002)),
            ("subnormal/normal", 2.**-14 - 2.**-25, (0x03ff, 0x0400, 0x0400)),
            ("even normal", 1. + 2.**-11, (0x3c00, 0x3c00, 0x3c01)),
            ("odd normal", 1. + 3. * 2.**-11, (0x3c01, 0x3c02, 0x3c02)),
            ("finite/overflow", 65520., (0x7bff, 0x7c00, 0x7c00)),
        )
        cases = []
        for name, midpoint, expected_bits in boundaries:
            center = np.float32(midpoint)
            neighbours = (np.nextafter(center, np.float32(-np.inf)), center,
                          np.nextafter(center, np.float32(np.inf)))
            for position, (value, bits) in enumerate(zip(neighbours, expected_bits)):
                cases.append((f"{name}/{position}/positive", value, bits))
                cases.append((f"{name}/{position}/negative", -value, bits | 0x8000))
        cases.extend((
            ("smallest float32", np.nextafter(np.float32(0), np.float32(1)), 0x0000),
            ("negative smallest float32", -np.nextafter(np.float32(0), np.float32(1)), 0x8000),
            ("largest float32", np.finfo(np.float32).max, 0x7c00),
            ("negative largest float32", -np.finfo(np.float32).max, 0xfc00),
            ("positive infinity", np.float32(np.inf), 0x7c00),
            ("negative infinity", np.float32(-np.inf), 0xfc00),
        ))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for name, value, bits in cases:
                with self.subTest(boundary=name):
                    # The -inf channel exposes the conversion result; +inf and
                    # 0.5 exercise a dominating mask and ordinary max selection.
                    processing = np.full((1, 1, 3), value, np.float32)
                    processing.flags.writeable = False
                    before = processing.tobytes()
                    for shape in ((1, 1), (2, 3)):
                        masks = np.broadcast_to(np.array([-np.inf, np.inf, .5], np.float16),
                                                (*shape, 3)).copy()
                        expected = _legacy_merge(masks.copy(), processing)
                        self.assertEqual(int(expected.view(np.uint16)[0, 0, 0]), bits)
                        maps = ((None, None) if shape == (1, 1) else
                                (_nearest_indices(1, 2), _nearest_indices(1, 3)))
                        result = self.ext.merge_processing_loss_f16_inplace(masks, processing, *maps)
                        self.assertIs(result, masks)
                        self.assertBitsEqual(masks, expected)
                    self.assertEqual(processing.tobytes(), before)

            # This tie makes operation order observable: direct float32 maximum
            # selects the positive sample before half-rounding, while the mapped
            # half maximum retains its left -0 after the sample rounded to +0.
            processing = np.full((1, 1, 3), 2.**-26, np.float32)
            direct = np.full(processing.shape, -0., np.float16)
            prior = direct.copy()
            self.assertIsNone(self.ext.merge_processing_loss_f16_inplace(direct, processing))
            self.assertBitsEqual(direct, prior)
            self.assertBitsEqual(raw_io._merge_processing_loss(direct, processing),
                                 _legacy_merge(prior, processing))
            self.assertTrue(np.all(direct.view(np.uint16) == 0x0000))
            mapped = np.full((2, 3, 3), -0., np.float16)
            expected = _legacy_merge(mapped.copy(), processing)
            self.assertIs(self.ext.merge_processing_loss_f16_inplace(
                mapped, processing, _nearest_indices(1, 2), _nearest_indices(1, 3)), mapped)
            self.assertBitsEqual(mapped, expected)
            self.assertTrue(np.all(mapped.view(np.uint16) == 0x8000))

    def test_invalid_crop_arguments_raise_without_mutating_input(self):
        values = np.ones((3, 5, 3), np.float16)
        valid = tuple(np.array(x, dtype=np.intp) for x in ([0, 1], [2, 3], [0, 2], [2, 5]))
        cases = [(values.astype(np.float64), valid), (values[..., :2], valid),
                 (values[:0], valid),
                 (values, tuple(np.empty(0, np.intp) for _ in range(4)))]
        for position, replacement in (
            (0, [-1, 1]), (1, [2, 4]), (0, [3, 1]), (3, [2, 6]),
            (1, [2]), (2, []), (2, np.array([0, 2], dtype=np.int32)),
            (2, np.array([[0, 2]], dtype=np.intp)),
        ):
            indices = list(valid)
            indices[position] = (replacement if isinstance(replacement, np.ndarray)
                                 else np.asarray(replacement, dtype=np.intp))
            cases.append((values, tuple(indices)))
        for source, indices in cases:
            with self.subTest(shape=source.shape, indices=indices):
                before = source.tobytes()
                with self.assertRaises((TypeError, ValueError)):
                    self.ext.crop_loss_footprint(source, *indices)
                self.assertEqual(source.tobytes(), before)

    def test_invalid_merge_arguments_fail_before_any_write(self):
        processing = np.ones((3, 5, 3), np.float32)
        masks = np.zeros((3, 5, 3), np.float16)
        readonly = masks.copy()
        readonly.flags.writeable = False
        # A foreign buffer object can share addresses while having a different
        # Python base chain. Native alias checks must use memory, not ownership.
        foreign_owner = (ctypes.c_uint16 * masks.size).from_address(masks.ctypes.data)
        foreign_alias = np.ctypeslib.as_array(foreign_owner).view(np.float16).reshape(masks.shape)
        cases = [
            (masks.astype(np.float32), processing, None, None),
            (masks[:, ::-1], processing, None, None),
            (readonly, processing, None, None),
            (masks, processing.astype(np.float64), None, None),
            (masks, processing[:0], None, None),
            (masks, processing[..., :2], None, None),
            (masks, processing[:2], None, None),
            (masks[:0], processing[:0], None, None),
            (masks, processing, np.arange(3, dtype=np.intp), None),
            (masks, processing, np.array([0, 1, 3], np.intp), np.arange(5, dtype=np.intp)),
            (masks, processing, np.arange(3, dtype=np.intp), np.array([0, 1, 2, 3, -1], np.intp)),
            (masks, processing, np.arange(2, dtype=np.intp), np.arange(5, dtype=np.intp)),
            (masks, processing, np.arange(3, dtype=np.int32), np.arange(5, dtype=np.intp)),
            (masks, masks, None, None),
            (masks, masks[::-1], None, None),
            (masks, foreign_alias, None, None),
        ]
        for target, source, yi, xi in cases:
            with self.subTest(target=target.shape, source=source.shape, yi=yi, xi=xi):
                before = target.tobytes()
                with self.assertRaises((TypeError, ValueError)):
                    self.ext.merge_processing_loss_f16_inplace(target, source, yi, xi)
                self.assertEqual(target.tobytes(), before)

    def test_late_unsupported_sample_declines_before_writing_earlier_pixels(self):
        for dtype, mapped, special in (
            (np.float32, False, np.nan), (np.float32, False, -0.),
            (np.float32, True, np.nan), (np.float16, True, np.nan),
        ):
            processing = np.full((17, 29, 3), .9, dtype)
            processing[-1, -1, -1] = special
            masks = np.full((7, 13, 3) if mapped else processing.shape, .1, np.float16)
            before = masks.copy()
            maps = ((_nearest_indices(17, 7), _nearest_indices(29, 13))
                    if mapped else (None, None))
            result = self.ext.merge_processing_loss_f16_inplace(masks, processing, *maps)
            self.assertIsNone(result)
            self.assertBitsEqual(masks, before)

    def test_swapped_dtype_is_rejected_before_native_reads_or_writes(self):
        indices = tuple(np.array(x, np.intp) for x in ([0, 1], [2, 3], [0, 2], [2, 5]))
        for dtype in (np.dtype(np.float16), np.dtype(np.float32)):
            source = np.full((3, 5, 3), .5, dtype=dtype.newbyteorder("S"))
            source_before = source.tobytes()
            with self.assertRaises((ValueError, TypeError)):
                self.ext.crop_loss_footprint(source, *indices)
            masks = np.full(source.shape, .2, np.float16)
            before = masks.tobytes()
            with self.assertRaises((ValueError, TypeError)):
                self.ext.merge_processing_loss_f16_inplace(masks, source)
            self.assertEqual(masks.tobytes(), before)
            self.assertEqual(source.tobytes(), source_before)
        source = np.ones((3, 5, 3), np.float16)
        masks = np.full(source.shape, .2, dtype=np.dtype(np.float16).newbyteorder("S"))
        before = masks.tobytes()
        with self.assertRaises((ValueError, TypeError)):
            self.ext.merge_processing_loss_f16_inplace(masks, source)
        self.assertEqual(masks.tobytes(), before)
        swapped_indices = tuple(index.astype(np.dtype(np.intp).newbyteorder("S")) for index in indices)
        with self.assertRaises((ValueError, TypeError)):
            self.ext.crop_loss_footprint(source, *swapped_indices)

    def test_infinities_alone_do_not_require_unsupported_fallback(self):
        for dtype in (np.float16, np.float32):
            processing = np.resize(np.array([np.inf, -np.inf, .3], dtype=dtype), (3, 5, 3))
            indices = _crop_indices(processing, (1., 1., 3., 7.), (6, 10), (7, 9))
            result = self.ext.crop_loss_footprint(processing, *indices)
            self.assertIsNotNone(result)
            self.assertBitsEqual(result, _legacy_footprint(processing, *indices))
            masks = np.full(processing.shape, .1, np.float16)
            expected = _legacy_merge(masks.copy(), processing)
            self.assertIs(self.ext.merge_processing_loss_f16_inplace(masks, processing), masks)
            self.assertBitsEqual(masks, expected)

    def test_thread_budgets_do_not_change_pixels(self):
        # Barely cross the existing parallel threshold; this is a correctness
        # fixture (~1.6 MB), not an image-size benchmark.
        processing = np.linspace(0., 1., 257 * 513 * 3, dtype=np.float32).reshape(257, 513, 3)
        yi, xi = np.arange(257, dtype=np.intp), np.arange(513, dtype=np.intp)
        expected_crop = processing.copy()
        expected_masks = np.maximum(np.float16(.3), processing).astype(np.float16)
        try:
            for budget in (1, 2, 3, 8):
                self.ext.set_thread_budget(budget)
                self.assertBitsEqual(self.ext.crop_loss_footprint(processing, yi, yi + 1, xi, xi + 1),
                                     expected_crop)
                masks = np.full(processing.shape, .3, np.float16)
                self.ext.merge_processing_loss_f16_inplace(masks, processing)
                self.assertBitsEqual(masks, expected_masks)
        finally:
            self.ext.set_thread_budget(0)


if __name__ == "__main__":
    unittest.main()
