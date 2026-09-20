# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen NumPy RGB-group oracle and native integer-count/dispatch contracts.

The oracle keeps the pre-native threshold-map, complete-CFA-period and
mean-times-100 operations. A separate LUT oracle checks the native counts.
No fixture RAW files, image decoder or large benchmark is needed here.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import struct
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np

from dngscan import _fast, analysis


KERNEL = "sensor_rgb_clip_counts_u16"
LABELS = {0: "R", 1: "G1", 2: "B", 3: "G2"}
PATTERN = [[0, 1], [3, 2]]


def _legacy_threshold_map(colors, thresholds):
    default = int(min(thresholds.values())) if thresholds else 0
    max_color = int(np.max(colors)) if colors.size else -1
    levels = np.full(max_color + 1, default, dtype=np.int32)
    for cid, value in thresholds.items():
        cid_i = int(cid)
        if 0 <= cid_i <= max_color:
            levels[cid_i] = value
    return levels[colors]


def _legacy_metrics(raw_image, raw_colors, thresholds, labels, raw_pattern):
    # Frozen compute_color_clip_metrics body from eaa42a0, before native dispatch.
    groups = {cid: label[:1].upper() for cid, label in labels.items()}
    if set(groups.values()) != {"R", "G", "B"}:
        return {}
    raw = np.asarray(raw_image)
    colors = np.asarray(raw_colors)
    if raw.ndim == 3:
        if colors.shape != raw.shape:
            return {}
        clipped = raw >= _legacy_threshold_map(colors, thresholds)
        per_group = [np.any(clipped & np.isin(colors, [c for c, g in groups.items() if g == group]),
                            axis=2) for group in ("R", "G", "B")]
    else:
        pattern = np.asarray(raw_pattern)
        if pattern.ndim != 2 or not pattern.size:
            return {}
        ph, pw = pattern.shape
        h, w = raw.shape
        h, w = h // ph * ph, w // pw * pw
        if not h or not w:
            return {}
        colors = colors[:h, :w]
        clipped = raw[:h, :w] >= _legacy_threshold_map(colors, thresholds)
        per_group = []
        for group in ("R", "G", "B"):
            ids = [cid for cid, name in groups.items() if name == group]
            group_clipped = clipped & np.isin(colors, ids)
            per_group.append(group_clipped.reshape(h // ph, ph, w // pw, pw).any(axis=(1, 3)))
    counts = np.sum(per_group, axis=0)
    return {k: float(np.mean(counts == k) * 100.0) for k in (1, 2, 3)}


def _lut_counts(raw, colors, thresholds, groups, ph, pw):
    """Independent NumPy integer-count reference for the low-level LUT API."""
    levels = np.asarray(thresholds, np.int32)[colors]
    bits = np.asarray(groups, np.uint8)[colors]
    clipped = raw >= levels
    if raw.ndim == 3:
        present = [np.any(clipped & ((bits & bit) != 0), axis=2) for bit in (1, 2, 4)]
    else:
        h, w = raw.shape
        h, w = h // ph * ph, w // pw * pw
        present = [(clipped[:h, :w] & ((bits[:h, :w] & bit) != 0))
                   .reshape(h // ph, ph, w // pw, pw).any(axis=(1, 3)) for bit in (1, 2, 4)]
    count = np.sum(present, axis=0)
    return [int(np.count_nonzero(count == k)) for k in range(4)]


def _luts(thresholds, labels):
    default = int(min(thresholds.values())) if thresholds else 0
    levels = np.full(256, default, np.int32)
    for cid, value in thresholds.items():
        if 0 <= int(cid) < 256:
            levels[int(cid)] = value
    groups = np.zeros(256, np.uint8)
    for cid, label in labels.items():
        if 0 <= cid < 256:
            groups[cid] = {"R": 1, "G": 2, "B": 4}[label[:1].upper()]
    return levels.tolist(), groups.tolist()


def _bayer(h=9, w=13):
    colors = np.tile(np.asarray(PATTERN, np.uint8), ((h + 1) // 2, (w + 1) // 2))[:h, :w]
    raw = np.random.default_rng(1642).integers(0, 65536, colors.shape, dtype=np.uint16)
    return raw, colors, {0: 20000, 1: 30000, 2: 50000, 3: 40000}, LABELS, PATTERN


def _linear(h=9, w=13):
    raw = np.random.default_rng(1604).integers(0, 65536, (h, w, 4), dtype=np.uint16)
    colors = np.broadcast_to(np.asarray([7, 17, 129, 250], np.uint8), raw.shape)
    return raw, colors, {7: 20000, 17: 30000, 129: 40000, 250: 50000}, \
        {7: "R", 17: "G1", 129: "G2", 250: "B"}, []


class _MetricsTest(unittest.TestCase):
    def assertMetricsBits(self, actual, expected):
        self.assertEqual(list(actual), list(expected))
        for key, value in expected.items():
            self.assertIs(type(actual[key]), float)
            self.assertEqual(struct.pack("=d", actual[key]), struct.pack("=d", value), key)

    def assertLegacy(self, args):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = _legacy_metrics(*args)
            actual = analysis.compute_color_clip_metrics(*args)
        self.assertMetricsBits(actual, expected)
        return actual


class SensorRgbDispatchTests(_MetricsTest):
    @contextmanager
    def fake_extension(self):
        call = mock.Mock(side_effect=_lut_counts)
        extension = SimpleNamespace(**{KERNEL: call})
        with mock.patch.object(_fast, "_load_extension", return_value=extension), \
             mock.patch.dict(os.environ, DNGSCAN_FAST="1", DNGSCAN_FAST_SKIP=""):
            yield call

    def test_bayer_and_xtrans_complete_period_percentages_are_bit_exact(self):
        pattern = [[1, 2, 1, 1, 0, 1], [0, 1, 0, 2, 1, 2], [1, 2, 1, 1, 0, 1],
                   [1, 0, 1, 1, 2, 1], [2, 1, 2, 0, 1, 0], [1, 0, 1, 1, 2, 1]]
        colors = np.tile(np.asarray(pattern, np.uint8), (3, 4))[:13, :19]
        raw = np.random.default_rng(602).integers(0, 65536, colors.shape, dtype=np.uint16)
        cases = [_bayer(), (raw, colors, {0: 20000, 1: 30000, 2: 40000},
                            {0: "R", 1: "G", 2: "B"}, pattern)]
        with self.fake_extension() as call:
            for args in cases:
                with self.subTest(pattern=np.asarray(args[-1]).shape):
                    self.assertLegacy(args)
            self.assertEqual(call.call_count, 2)

    def test_two_green_sites_are_one_colour_and_zero_clip_cells_remain_population(self):
        raw = np.zeros((4, 6), np.uint16)
        colors = np.tile(np.asarray(PATTERN, np.uint8), (2, 3))
        raw[:2, :2][np.isin(colors[:2, :2], [1, 3])] = 1000
        raw[:2, 2:4] = 1000
        with self.fake_extension() as call:
            actual = self.assertLegacy((raw, colors, {c: 1000 for c in range(4)}, LABELS, PATTERN))
        self.assertEqual(actual, {1: 1 / 6 * 100., 2: 0., 3: 1 / 6 * 100.})
        call.assert_called_once()

    def test_odd_borders_are_excluded_from_cfa_population(self):
        raw, colors, _, labels, pattern = _bayer(5, 7)
        raw.fill(0)
        raw[-1] = 65535
        raw[:, -1] = 65535
        raw[:2, :2][colors[:2, :2] == 0] = 65535
        with self.fake_extension() as call:
            actual = self.assertLegacy((raw, colors, {i: 65535 for i in range(4)}, labels, pattern))
        self.assertEqual(actual, {1: 1 / 6 * 100., 2: 0., 3: 0.})
        call.assert_called_once()

    def test_pattern_only_sets_period_actual_colors_determine_groups(self):
        raw, colors, thresholds, labels, _ = _bayer()
        colors = np.roll(colors, 1, axis=0)
        with self.fake_extension() as call:
            self.assertLegacy((raw, colors, thresholds, labels, [[9, 9], [9, 9]]))
        call.assert_called_once()

    def test_linear_four_channels_sparse_ids_and_broadcast_are_borrowed(self):
        args = _linear()
        before = [arr.tobytes() for arr in args[:2]]
        with self.fake_extension() as call:
            self.assertLegacy(args)
        call.assert_called_once()
        self.assertIs(call.call_args.args[0], args[0])
        self.assertIs(call.call_args.args[1], args[1])
        self.assertEqual(call.call_args.args[1].strides[:2], (0, 0))
        self.assertEqual(before, [arr.tobytes() for arr in args[:2]])

    def test_strides_transpose_broadcast_and_readonly_do_not_copy_or_mutate(self):
        raw, colors, thresholds, labels, pattern = _bayer()
        cases = [(raw[::-1, ::-1], colors[::-1, ::-1]),
                 (raw.T, colors.T), (raw[::2, ::2], colors[::2, ::2]),
                 (np.broadcast_to(raw[:1], raw.shape), np.broadcast_to(colors[:1], colors.shape))]
        for image, cmap in cases:
            with self.subTest(strides=image.strides, color_strides=cmap.strides):
                image.flags.writeable = False
                cmap.flags.writeable = False
                before = image.tobytes(), cmap.tobytes()
                with self.fake_extension() as call:
                    self.assertLegacy((image, cmap, thresholds, labels, pattern))
                call.assert_called_once()
                self.assertIs(call.call_args.args[0], image)
                self.assertIs(call.call_args.args[1], cmap)
                self.assertEqual(before, (image.tobytes(), cmap.tobytes()))

    def test_missing_threshold_uses_minimum_of_whole_map_including_unseen_channel(self):
        raw, colors, _, labels, pattern = _bayer()
        thresholds = {0: 60000, 1: 50000, 2: 55000, 200: -10}
        with self.fake_extension() as call:
            self.assertLegacy((raw, colors, thresholds, labels, pattern))
        self.assertEqual(call.call_args.args[2][3], -10)
        self.assertEqual(call.call_args.args[2][0], 60000)

    def test_empty_thresholds_clip_every_mapped_group(self):
        raw, colors, _, labels, pattern = _bayer()
        with self.fake_extension() as call:
            actual = self.assertLegacy((raw, colors, {}, labels, pattern))
        self.assertEqual(actual, {1: 0., 2: 0., 3: 100.})
        call.assert_called_once()

    def test_numpy_integer_metadata_and_negative_unseen_keys_preserve_policy(self):
        raw, colors, _, _, pattern = _bayer()
        labels = {np.int64(k): v.lower() for k, v in LABELS.items()}
        thresholds = {np.int64(0): np.int32(65535), np.int64(-3): np.int32(-1)}
        with self.fake_extension() as call:
            self.assertLegacy((raw, colors, thresholds, labels, pattern))
        call.assert_called_once()
        self.assertEqual(call.call_args.args[2][1], -1)

    def test_unlabelled_actual_channel_is_ignored_in_rgb_groups(self):
        raw, colors, thresholds, labels, pattern = _bayer()
        colors = colors.copy()
        colors[::2, ::2] = 255
        raw[::2, ::2] = 65535
        with self.fake_extension() as call:
            actual = self.assertLegacy((raw, colors, thresholds, labels, pattern))
        self.assertEqual(actual[3], 0.)
        self.assertEqual(call.call_args.args[3][255], 0)

    def test_unknown_or_missing_label_groups_keep_empty_result(self):
        raw, colors, thresholds, _, pattern = _bayer()
        with self.fake_extension() as call:
            for labels in ({}, {0: "R", 1: "G"}, {**LABELS, 200: "X"},
                           {**LABELS, 200: ""}):
                self.assertEqual(self.assertLegacy((raw, colors, thresholds, labels, pattern)), {})
        call.assert_not_called()

    def test_float_signed_non_native_and_unaligned_inputs_keep_numpy(self):
        raw, colors, thresholds, labels, pattern = _bayer()
        unaligned = np.ndarray(raw.shape, np.uint16, buffer=bytearray(raw.nbytes + 1), offset=1)
        unaligned[...] = raw
        float_special = raw.astype(np.float64)
        float_special.flat[:3] = np.nan, np.inf, -np.inf
        with self.fake_extension() as call:
            for image, cmap in ((raw.astype(np.float32), colors), (float_special, colors),
                                (raw.astype(np.int32), colors),
                                (raw.astype(raw.dtype.newbyteorder("S")), colors),
                                (unaligned, colors), (raw, colors.astype(np.int16))):
                with self.subTest(dtype=image.dtype, aligned=image.flags.aligned, colors=cmap.dtype):
                    self.assertLegacy((image, cmap, thresholds, labels, pattern))
        call.assert_not_called()

    def test_unused_singleton_odd_and_minimum_strides_keep_numpy_in_strict_mode(self):
        # NumPy permits unused singleton strides that ndarray's borrowed-view
        # representation rejects. ALIGNED alone does not establish eligibility.
        odd_raw = np.ndarray((1, 2), np.uint16, buffer=bytearray(4), strides=(1, 2))
        odd_raw[...] = [[10000, 60000]]
        raw = np.asarray([[10000, 60000]], np.uint16)
        colors = np.asarray([[0, 1]], np.uint8)
        min_stride_colors = np.ndarray((1, 2), np.uint8, buffer=bytearray((0, 1)),
                                        strides=(np.iinfo(np.intp).min, 1))
        thresholds = {0: 20000, 1: 50000, 2: 50000}
        with self.fake_extension() as call:
            for image, cmap in ((odd_raw, colors), (raw, min_stride_colors)):
                with self.subTest(raw_strides=image.strides, color_strides=cmap.strides):
                    self.assertTrue(image.flags.aligned)
                    self.assertTrue(cmap.flags.aligned)
                    self.assertLegacy((image, cmap, thresholds, LABELS, [[0]]))
        call.assert_not_called()

    def test_unsupported_metadata_retains_reference_results_and_errors(self):
        raw, colors, thresholds, labels, pattern = _bayer()
        valid_fallbacks = [(raw, colors, {str(k): v for k, v in thresholds.items()}, labels, pattern),
                           (raw, colors, {k: v + .75 for k, v in thresholds.items()}, labels, pattern),
                           (raw, colors, thresholds, {str(k): v for k, v in labels.items()}, pattern)]
        with self.fake_extension() as call:
            for args in valid_fallbacks:
                self.assertLegacy(args)
            for changed in ({0: 2 ** 70}, {0: float("nan")}, {0: float("inf")}):
                args = raw, colors, changed, labels, pattern
                try:
                    _legacy_metrics(*args)
                except Exception as error:
                    with self.assertRaises(type(error)):
                        analysis.compute_color_clip_metrics(*args)
                else:
                    self.fail("invalid reference metadata unexpectedly accepted")
        call.assert_not_called()

    def test_empty_invalid_period_and_shape_fallbacks_preserve_reference(self):
        raw, colors, thresholds, labels, pattern = _bayer()
        cases = [(raw, colors, thresholds, labels, []),
                 (raw, colors, thresholds, labels, [0, 1, 2]),
                 (raw[:1], colors[:1], thresholds, labels, pattern),
                 (raw[:0], colors[:0], thresholds, labels, pattern),
                 (raw, colors[:, :1], thresholds, labels, pattern),
                 (np.empty((0, 4, 3), np.uint16), np.empty((0, 4, 3), np.uint8),
                  thresholds, labels, []),
                 (np.empty((2, 4, 3), np.uint16), np.empty((2, 4, 1), np.uint8),
                  thresholds, labels, [])]
        with self.fake_extension() as call:
            for args in cases:
                with self.subTest(shape=args[0].shape, colors=args[1].shape, pattern=args[-1]):
                    self.assertLegacy(args)
        call.assert_not_called()

    def test_invalid_raw_dimensions_keep_reference_exception(self):
        with self.fake_extension() as call:
            for shape in ((5,), (2, 3, 4, 5)):
                args = (np.zeros(shape, np.uint16), np.zeros(shape, np.uint8), {}, LABELS, PATTERN)
                with self.assertRaises(ValueError):
                    _legacy_metrics(*args)
                with self.assertRaises(ValueError):
                    analysis.compute_color_clip_metrics(*args)
        call.assert_not_called()

    def test_off_and_skip_bypass_native_even_in_strict_mode(self):
        args = _bayer()
        with self.fake_extension() as call:
            with mock.patch.dict(os.environ, DNGSCAN_FAST="0"):
                self.assertLegacy(args)
            with mock.patch.dict(os.environ, DNGSCAN_FAST_SKIP=f"unused, {KERNEL}"):
                self.assertLegacy(args)
        call.assert_not_called()

    def test_kernel_error_falls_back_in_auto_and_raises_in_strict(self):
        args = _bayer()
        before = [arr.tobytes() for arr in args[:2]]
        with self.fake_extension() as call:
            call.side_effect = ValueError("intentional preflight failure")
            with self.assertRaises(_fast.NativeKernelError):
                analysis.compute_color_clip_metrics(*args)
            with mock.patch.dict(os.environ, DNGSCAN_FAST="auto"), \
                 self.assertLogs("dngscan._fast", level="WARNING"):
                self.assertLegacy(args)
        self.assertEqual(before, [arr.tobytes() for arr in args[:2]])

    def test_missing_extension_falls_back_in_auto_and_raises_in_strict(self):
        args = _bayer()
        with mock.patch.object(_fast, "_load_extension", return_value=None), \
             mock.patch.dict(os.environ, DNGSCAN_FAST="auto", DNGSCAN_FAST_SKIP=""):
            self.assertLegacy(args)
            with mock.patch.dict(os.environ, DNGSCAN_FAST="1"):
                with self.assertRaises(_fast.NativeKernelError):
                    analysis.compute_color_clip_metrics(*args)

    def test_missing_entry_falls_back_in_auto_and_raises_in_strict(self):
        args = _bayer()
        with mock.patch.object(_fast, "_load_extension", return_value=SimpleNamespace()), \
             mock.patch.dict(os.environ, DNGSCAN_FAST="auto", DNGSCAN_FAST_SKIP=""):
            self.assertLegacy(args)
            with mock.patch.dict(os.environ, DNGSCAN_FAST="1"):
                with self.assertRaises(_fast.NativeKernelError):
                    analysis.compute_color_clip_metrics(*args)

    def test_invalid_native_populations_obey_error_policy(self):
        args = _bayer()
        total = (args[0].shape[0] // 2) * (args[0].shape[1] // 2)
        with self.fake_extension() as call:
            call.side_effect = None
            for counts in (None, [total], [total, 0, 0, 1], [total + 1, -1, 0, 0]):
                with self.subTest(counts=counts):
                    call.return_value = counts
                    with self.assertRaises(_fast.NativeKernelError):
                        analysis.compute_color_clip_metrics(*args)
                    with mock.patch.dict(os.environ, DNGSCAN_FAST="auto"), \
                         self.assertLogs("dngscan._fast", level="WARNING"):
                        self.assertLegacy(args)


class SensorRgbNativeTests(_MetricsTest):
    @classmethod
    def setUpClass(cls):
        cls.ext = _fast._load_extension()
        if cls.ext is None or not hasattr(cls.ext, KERNEL):
            raise unittest.SkipTest("ABI 16 sensor RGB kernel not built")

    def check_native(self, args):
        raw, colors, thresholds, labels, pattern = args
        levels, groups = _luts(thresholds, labels)
        ph, pw = np.asarray(pattern).shape if raw.ndim == 2 else (1, 1)
        before = raw.tobytes(), colors.tobytes()
        got = getattr(self.ext, KERNEL)(raw, colors, levels, groups, ph, pw)
        self.assertEqual(list(got), _lut_counts(raw, colors, levels, groups, ph, pw))
        self.assertEqual(len(got), 4)
        self.assertTrue(all(type(v) is int and v >= 0 for v in got))
        total = raw.shape[0] * raw.shape[1] if raw.ndim == 3 else \
            (raw.shape[0] // ph) * (raw.shape[1] // pw)
        self.assertEqual(sum(got), total)
        self.assertEqual(before, (raw.tobytes(), colors.tobytes()))
        with mock.patch.dict(os.environ, DNGSCAN_FAST="1", DNGSCAN_FAST_SKIP=""):
            self.assertLegacy(args)

    def test_native_counts_and_full_percentages_match_cfa_and_linear_oracles(self):
        for args in (_bayer(), _linear()):
            with self.subTest(shape=args[0].shape):
                self.check_native(args)
        raw, colors, thresholds, labels, _ = _bayer(19, 25)
        self.check_native((raw, colors, thresholds, labels, [[9] * 6 for _ in range(6)]))

    def test_native_transpose_negative_zero_strides_and_readonly(self):
        for args in (_bayer(), _linear()):
            raw, colors, thresholds, labels, pattern = args
            if raw.ndim == 2:
                views = [(raw.T, colors.T), (raw[::-1, ::-1], colors[::-1, ::-1]),
                         (np.broadcast_to(raw[:1], raw.shape), np.broadcast_to(colors[:1], colors.shape))]
            else:
                views = [(raw.transpose(1, 0, 2), colors.transpose(1, 0, 2)),
                         (raw[::-1, ::-1, ::-1], colors[::-1, ::-1, ::-1]),
                         (np.broadcast_to(raw[:1, :1], raw.shape), colors)]
            for image, cmap in views:
                with self.subTest(shape=image.shape, strides=image.strides):
                    image.flags.writeable = False
                    cmap.flags.writeable = False
                    self.check_native((image, cmap, thresholds, labels, pattern))

    def test_native_integer_boundaries_and_unlabelled_channels(self):
        raw, colors, _, labels, pattern = _bayer()
        raw.flat[:4] = [0, 1, 65534, 65535]
        colors = colors.copy()
        colors.flat[:4] = [0, 1, 2, 255]
        for thresholds in ({}, {0: -2147483648, 1: 0, 2: 65535, 3: 2147483647},
                           {0: 65536, 1: -1, 200: -100}):
            with self.subTest(thresholds=thresholds):
                self.check_native((raw, colors, thresholds, labels, pattern))

    def test_native_empty_views_and_zero_channel_linear_keep_population(self):
        for shape, pattern in (((0, 7), PATTERN), ((1, 1), PATTERN),
                               ((0, 7, 3), []), ((3, 7, 0), [])):
            raw, colors = np.zeros(shape, np.uint16)[::-1], np.zeros(shape, np.uint8)[::-1]
            with self.subTest(shape=shape):
                self.check_native((raw, colors, {}, LABELS, pattern))

    def test_native_rejects_non_native_dtype_unaligned_and_bad_linear_period(self):
        raw, colors, thresholds, labels, _ = _bayer()
        levels, groups = _luts(thresholds, labels)
        unaligned = np.ndarray(raw.shape, np.uint16, buffer=bytearray(raw.nbytes + 1), offset=1)
        unaligned[...] = raw
        for image, cmap in ((raw.astype(np.float32), colors), (raw.astype(np.int16), colors),
                            (raw.astype(raw.dtype.newbyteorder("S")), colors),
                            (unaligned, colors), (raw, colors.astype(np.int16))):
            with self.subTest(dtype=image.dtype, aligned=image.flags.aligned, colors=cmap.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    getattr(self.ext, KERNEL)(image, cmap, levels, groups, 2, 2)
        linear, cmap, _, _, _ = _linear()
        with self.assertRaises(ValueError):
            getattr(self.ext, KERNEL)(linear, cmap, levels, groups, 2, 1)

    def test_native_pathological_singleton_stride_is_rejected_without_reading(self):
        raw = np.lib.stride_tricks.as_strided(np.zeros((1, 1), np.uint16),
                                             shape=(1, 1), strides=(np.iinfo(np.intp).min, 2))
        colors = np.zeros((1, 1), np.uint8)
        levels, groups = _luts({}, LABELS)
        with self.assertRaises(ValueError):
            getattr(self.ext, KERNEL)(raw, colors, levels, groups, 1, 1)

    def test_native_invalid_shape_lut_and_period_are_rejected(self):
        raw, colors, thresholds, labels, _ = _bayer()
        levels, groups = _luts(thresholds, labels)
        valid = [raw, colors, levels, groups, 2, 2]
        bad_args = []
        for index, value in ((0, raw.ravel()), (1, colors[:, :-1]),
                             (2, levels[:-1]), (2, levels + [0]),
                             (3, groups[:-1]), (3, groups + [0]),
                             (3, [3] + groups[1:]), (3, [255] + groups[1:]),
                             (2, [2 ** 40] + levels[1:]),
                             (4, 0), (5, 0), (4, -1), (5, -1)):
            changed = valid.copy()
            changed[index] = value
            bad_args.append(changed)
        before = raw.tobytes(), colors.tobytes()
        for index, args in enumerate(bad_args):
            with self.subTest(case=index):
                with self.assertRaises((ValueError, TypeError, OverflowError)):
                    getattr(self.ext, KERNEL)(*args)
        self.assertEqual(before, (raw.tobytes(), colors.tobytes()))


if __name__ == "__main__":
    unittest.main()
