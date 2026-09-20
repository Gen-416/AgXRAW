# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen per-channel NumPy oracles and exact unsigned-sensor native contracts."""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
import struct
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from dngscan import _fast, analysis


CEILING = "sensor_ceiling_counts_u16"
CLIP = "sensor_channel_clip_counts_u16"


def _legacy_ceilings(raw_image, raw_colors, channel_ids, sat=None):
    # Pre-native detect_ceilings, including missing-channel/metadata ordering.
    ceilings, exact_counts, near_counts, spike_ok = {}, {}, {}, {}
    for cid in channel_ids:
        vals = raw_image[raw_colors == cid]
        if vals.size == 0:
            raise RuntimeError(f"no visible pixels for raw color channel {cid}")
        ceil = int(np.max(vals))
        exact = int(np.count_nonzero(vals == ceil))
        level = int((sat or {}).get(cid, 0)) or int(np.iinfo(raw_image.dtype).max
                                                    if np.issubdtype(raw_image.dtype, np.integer)
                                                    else 65535)
        window = max(2, int(round(level / analysis.CEILING_NEAR_WINDOW_SCALE)))
        near = int(np.count_nonzero(vals >= max(ceil - window, 0)))
        min_pile = max(analysis.CEILING_MIN_PILE_PIXELS,
                       int(math.ceil(vals.size * analysis.CEILING_MIN_PILE_FRACTION)))
        ceilings[cid], exact_counts[cid], near_counts[cid] = ceil, exact, near
        spike_ok[cid] = exact >= min_pile or near >= min_pile
    return ceilings, exact_counts, near_counts, spike_ok


def _legacy_clip(raw_image, raw_colors, channel_ids, thresholds):
    # Pre-native compute_clip_pct_by_thresholds: int conversion precedes empty.
    out = {}
    for cid in channel_ids:
        vals = raw_image[raw_colors == cid]
        threshold = int(thresholds.get(cid, 0))
        out[cid] = float(np.mean(vals >= threshold) * 100.0) if vals.size else 0.0
    return out


def _ceiling_counts(raw, colors, windows):
    maxima, totals, exact, near = ([0] * 256 for _ in range(4))
    for cid in np.unique(colors):
        index = int(cid)
        vals = raw[colors == cid]
        maxima[index], totals[index] = int(vals.max()), int(vals.size)
        exact[index] = int(np.count_nonzero(vals == maxima[index]))
        near[index] = int(np.count_nonzero(vals >= max(maxima[index] - windows[index], 0)))
    return maxima, totals, exact, near


def _clip_counts(raw, colors, thresholds):
    totals, hits = [0] * 256, [0] * 256
    for cid in np.unique(colors):
        index = int(cid)
        vals = raw[colors == cid]
        totals[index], hits[index] = int(vals.size), int(np.count_nonzero(vals >= thresholds[index]))
    return totals, hits


def _mosaic(h=9, w=13):
    pattern = np.asarray([[2, 17], [129, 250]], np.uint8)
    colors = np.tile(pattern, ((h + 1) // 2, (w + 1) // 2))[:h, :w]
    raw = np.random.default_rng(1731).integers(0, 65536, colors.shape, dtype=np.uint16)
    return raw, colors, [250, 17, 2, 129], {2: 16383, 17: 24576, 129: 65535, 250: 0}


def _linear(h=5, w=7):
    raw = np.random.default_rng(1732).integers(0, 65536, (h, w, 4), dtype=np.uint16)
    colors = np.broadcast_to(np.asarray([2, 17, 129, 250], np.uint8), raw.shape)
    return raw, colors, [250, 17, 2, 129], {2: 16383, 17: 24576, 129: 65535, 250: 0}


def _xtrans():
    pattern = np.asarray([[1, 2, 1, 1, 0, 1], [0, 1, 0, 2, 1, 2], [1, 2, 1, 1, 0, 1],
                          [1, 0, 1, 1, 2, 1], [2, 1, 2, 0, 1, 0], [1, 0, 1, 1, 2, 1]], np.uint8)
    colors = np.tile(pattern, (3, 4))[:13, :19]
    raw = np.random.default_rng(1733).integers(0, 65536, colors.shape, dtype=np.uint16)
    return raw, colors, [2, 0, 1], {0: 16383, 1: 65535, 2: 0}


class _OracleTest(unittest.TestCase):
    def assertCeilings(self, actual, expected):
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 4)
        for index, (got, want) in enumerate(zip(actual, expected)):
            self.assertEqual(list(got), list(want))
            for value in got.values():
                self.assertIs(type(value), bool if index == 3 else int)

    def assertClipBits(self, actual, expected):
        self.assertEqual(list(actual), list(expected))
        for cid, value in expected.items():
            self.assertIs(type(actual[cid]), float)
            self.assertEqual(struct.pack("=d", actual[cid]), struct.pack("=d", value), cid)

    def check_ceilings(self, args):
        expected = _legacy_ceilings(*args)
        actual = analysis.detect_ceilings(*args)
        self.assertCeilings(actual, expected)
        return actual

    def check_clip(self, args):
        expected = _legacy_clip(*args)
        actual = analysis.compute_clip_pct_by_thresholds(*args)
        self.assertClipBits(actual, expected)
        return actual


class SensorChannelDispatchTests(_OracleTest):
    @contextmanager
    def fake_extension(self):
        ceiling_call = mock.Mock(side_effect=_ceiling_counts)
        clip_call = mock.Mock(side_effect=_clip_counts)
        ext = SimpleNamespace(**{CEILING: ceiling_call, CLIP: clip_call})
        with mock.patch.object(_fast, "_load_extension", return_value=ext), \
             mock.patch.dict(os.environ, DNGSCAN_FAST="1", DNGSCAN_FAST_SKIP=""):
            yield ceiling_call, clip_call

    def test_sparse_mosaic_and_linear_counts_match_frozen_results(self):
        with self.fake_extension() as (ceiling, clip):
            for args in (_mosaic(), _xtrans(), _linear()):
                with self.subTest(shape=args[0].shape):
                    self.check_ceilings(args)
                    self.check_clip(args)
        self.assertEqual(ceiling.call_count, 3)
        self.assertEqual(clip.call_count, 3)

    def test_requested_order_duplicates_and_numpy_integer_keys_are_preserved(self):
        raw, colors, _, levels = _mosaic()
        first = np.int64(250)
        ids = [first, np.uint8(2), np.int32(129), first, 2, 17]
        with self.fake_extension() as (ceiling, clip):
            got = self.check_ceilings((raw, colors, ids, levels))
            pct = self.check_clip((raw, colors, tuple(ids), levels))
        self.assertIs(next(iter(got[0])), first)
        self.assertIs(next(iter(pct)), first)
        self.assertEqual(list(pct), [250, 2, 129, 17])
        ceiling.assert_called_once()
        clip.assert_called_once()

    def test_all_sensels_including_odd_edges_contribute_to_each_population(self):
        raw, colors, ids, _ = _mosaic(5, 7)
        raw.fill(0)
        raw[-1] = 65535
        raw[:, -1] = 65535
        levels = {cid: 65535 for cid in ids}
        with self.fake_extension() as (ceiling, clip):
            self.check_ceilings((raw, colors, ids, levels))
            actual = self.check_clip((raw, colors, ids, levels))
        self.assertGreater(actual[2], 0.)
        self.assertEqual(sum(_clip_counts(*clip.call_args.args)[0]), 35)
        ceiling.assert_called_once()
        clip.assert_called_once()

    def test_windows_use_python_round_zero_fallback_negative_floor_and_inclusive_edge(self):
        raw = np.asarray([[100, 98, 97, 96, 92, 91]], np.uint16)
        colors = np.zeros(raw.shape, np.uint8)
        cases = [(None, 8), ({}, 8), ({0: 0}, 8), ({0: -1}, 2),
                 ({0: 20480}, 2), ({0: 24576}, 3), ({0: 28672}, 4),
                 ({0: 65535}, 8), ({0: 2147483647}, 65535)]
        with self.fake_extension() as (ceiling, _):
            for sat, window in cases:
                with self.subTest(sat=sat):
                    self.check_ceilings((raw, colors, [0], sat))
                    self.assertEqual(ceiling.call_args.args[2][0], window)

    def test_exact_near_pile_boundaries_and_fraction_rounding_remain_python_policy(self):
        colors = np.zeros((1, 7), np.uint8)
        with self.fake_extension() as (ceiling, _), \
             mock.patch.object(analysis, "CEILING_MIN_PILE_PIXELS", 2), \
             mock.patch.object(analysis, "CEILING_MIN_PILE_FRACTION", .3):
            for values, expected in (([100, 100, 97, 1, 2, 3, 4], False),
                                     ([100, 99, 98, 1, 2, 3, 4], True),
                                     ([100, 100, 100, 1, 2, 3, 4], True)):
                raw = np.asarray([values], np.uint16)
                actual = self.check_ceilings((raw, colors, [0], {0: 16383}))
                self.assertEqual(actual[3], {0: expected})
        self.assertEqual(ceiling.call_count, 3)

    def test_clip_threshold_zero_default_is_independent_of_other_channels(self):
        raw, colors, ids, _ = _mosaic()
        thresholds = {2: 65536, 250: 65535, 200: 60000}
        with self.fake_extension() as (_, clip):
            actual = self.check_clip((raw, colors, ids, thresholds))
        self.assertEqual(actual[17], 100.)
        self.assertEqual(actual[129], 100.)
        self.assertEqual(clip.call_args.args[2][17], 0)
        self.assertEqual(clip.call_args.args[2][129], 0)

    def test_empty_raw_and_absent_channels_preserve_error_or_positive_zero(self):
        cases = [(np.empty(shape, np.uint16), np.empty(shape, np.uint8))
                 for shape in ((0, 7), (3, 0), (0, 0, 3), (3, 7, 0))]
        cases.append((np.zeros((3, 5), np.uint16), np.zeros((3, 5), np.uint8)))
        with self.fake_extension() as (ceiling, clip):
            for raw, colors in cases:
                with self.subTest(shape=raw.shape):
                    with self.assertRaisesRegex(RuntimeError, "^no visible pixels for raw color channel 17$"):
                        analysis.detect_ceilings(raw, colors, [17], {})
                    self.assertClipBits(analysis.compute_clip_pct_by_thresholds(raw, colors, [17], {}),
                                        {17: 0.0})
        self.assertEqual(ceiling.call_count, len(cases))
        self.assertEqual(clip.call_count, len(cases))

    def test_empty_requested_channels_never_validate_input_or_metadata(self):
        with self.fake_extension() as (ceiling, clip):
            for ids in ([], ()):
                self.assertEqual(analysis.detect_ceilings(object(), object(), ids, object()), ({}, {}, {}, {}))
                self.assertEqual(analysis.compute_clip_pct_by_thresholds(object(), object(), ids, object()), {})
        ceiling.assert_not_called()
        clip.assert_not_called()

    def test_missing_ceiling_channel_precedes_bad_metadata_in_requested_order(self):
        raw = np.zeros((2, 3), np.uint16)
        colors = np.zeros(raw.shape, np.uint8)
        with self.fake_extension() as (ceiling, _):
            for sat in ({17: float("nan")}, {0: float("nan")}, {0: 10 ** 1000}):
                with self.subTest(sat_type=type(next(iter(sat.values())))):
                    with self.assertRaisesRegex(RuntimeError, "^no visible pixels for raw color channel 17$"):
                        analysis.detect_ceilings(raw, colors, [17, 0], sat)
        ceiling.assert_not_called()

    def test_absent_clip_channel_still_converts_its_bad_threshold(self):
        raw = np.zeros((2, 3), np.uint16)
        colors = np.zeros(raw.shape, np.uint8)
        with self.fake_extension() as (_, clip):
            for value, error in ((float("nan"), ValueError), (float("inf"), OverflowError),
                                 ("not a threshold", ValueError)):
                with self.subTest(value=value), self.assertRaises(error):
                    analysis.compute_clip_pct_by_thresholds(raw, colors, [17], {17: value})
        clip.assert_not_called()

    def test_policy_errors_keep_original_types_and_empty_channel_precedence(self):
        raw = np.zeros((2, 3), np.uint16)
        colors = np.zeros(raw.shape, np.uint8)
        with self.fake_extension(), mock.patch.object(analysis, "CEILING_NEAR_WINDOW_SCALE", 0):
            with self.assertRaises(ZeroDivisionError):
                analysis.detect_ceilings(raw, colors, [0], {})
            with self.assertRaisesRegex(RuntimeError, "no visible pixels for raw color channel 17"):
                analysis.detect_ceilings(raw, colors, [17, 0], {})
        with self.fake_extension(), mock.patch.object(analysis, "CEILING_MIN_PILE_FRACTION", float("nan")):
            with self.assertRaises(ValueError):
                analysis.detect_ceilings(raw, colors, [0], {})
            with self.assertRaisesRegex(RuntimeError, "no visible pixels for raw color channel 17"):
                analysis.detect_ceilings(raw, colors, [17, 0], {})

    def test_strided_transposed_broadcast_readonly_inputs_are_borrowed_unchanged(self):
        for raw, colors, ids, levels in (_mosaic(), _linear()):
            views = [(raw[::-1, ::-1], colors[::-1, ::-1]),
                     (np.swapaxes(raw, 0, 1), np.swapaxes(colors, 0, 1)),
                     (np.broadcast_to(raw[:1], raw.shape), np.broadcast_to(colors[:1], colors.shape))]
            for image, cmap in views:
                image.flags.writeable = False
                cmap.flags.writeable = False
                existing = [cid for cid in ids if np.any(cmap == cid)]
                before = image.tobytes(), cmap.tobytes()
                with self.subTest(shape=image.shape, strides=image.strides), \
                     self.fake_extension() as (ceiling, clip):
                    self.check_ceilings((image, cmap, existing, levels))
                    self.check_clip((image, cmap, ids, levels))
                    for call in (ceiling, clip):
                        call.assert_called_once()
                        self.assertIs(call.call_args.args[0], image)
                        self.assertIs(call.call_args.args[1], cmap)
                    self.assertEqual(before, (image.tobytes(), cmap.tobytes()))

    def test_singleton_odd_and_minimum_strides_use_reference_in_strict_mode(self):
        odd = np.ndarray((1, 2), np.uint16, buffer=bytearray(4), strides=(1, 2))
        odd[...] = [[0, 65535]]
        raw = np.asarray([[0, 65535]], np.uint16)
        colors = np.asarray([[0, 1]], np.uint8)
        minimum = np.ndarray((1, 2), np.uint8, buffer=bytearray((0, 1)),
                             strides=(np.iinfo(np.intp).min, 1))
        with self.fake_extension() as (ceiling, clip):
            for image, cmap in ((odd, colors), (raw, minimum)):
                self.assertTrue(image.flags.aligned)
                self.assertTrue(cmap.flags.aligned)
                self.check_ceilings((image, cmap, [0, 1], {}))
                self.check_clip((image, cmap, [0, 1], {}))
        ceiling.assert_not_called()
        clip.assert_not_called()

    def test_other_dtypes_endian_unaligned_and_subclasses_keep_reference(self):
        class ArraySubclass(np.ndarray):
            pass
        raw, colors, ids, levels = _mosaic()
        unaligned = np.ndarray(raw.shape, np.uint16, buffer=bytearray(raw.nbytes + 1), offset=1)
        unaligned[...] = raw
        cases = [(raw.astype(np.float32) + .75, colors), (raw.astype(np.int32), colors),
                 (raw.astype(np.uint8), colors), (raw.astype(raw.dtype.newbyteorder("S")), colors),
                 (unaligned, colors), (raw, colors.astype(np.int16)),
                 (raw.view(ArraySubclass), colors), (raw, colors.view(ArraySubclass))]
        with self.fake_extension() as (ceiling, clip):
            for image, cmap in cases:
                with self.subTest(dtype=image.dtype, image_type=type(image), colors_type=type(cmap)):
                    self.check_ceilings((image, cmap, ids, levels))
                    self.check_clip((image, cmap, ids, levels))
        ceiling.assert_not_called()
        clip.assert_not_called()

    def test_boolean_prefix_indexing_and_unusual_dimensions_remain_reference(self):
        raw, colors, ids, levels = _linear()
        prefix_colors = np.resize(np.asarray(ids, np.uint8), raw.shape[:2])
        with self.fake_extension() as (ceiling, clip):
            for image, cmap in ((raw, prefix_colors), (raw.ravel(), colors.ravel()),
                                (raw.reshape(1, *raw.shape), colors.reshape(1, *colors.shape))):
                self.check_ceilings((image, cmap, ids, levels))
                self.check_clip((image, cmap, ids, levels))
        ceiling.assert_not_called()
        clip.assert_not_called()

    def test_atypical_channel_ids_and_metadata_are_not_coerced_for_native(self):
        class Mapping(dict):
            pass
        raw, colors, ids, levels = _mosaic()
        with self.fake_extension() as (ceiling, clip):
            for requested in ([float(cid) for cid in ids], np.asarray(ids), Mapping.fromkeys(ids)):
                self.check_ceilings((raw, colors, requested, levels))
                self.check_clip((raw, colors, requested, levels))
            for metadata in (Mapping(levels), {cid: str(value) for cid, value in levels.items()},
                             {cid: value + .75 for cid, value in levels.items()}):
                self.check_ceilings((raw, colors, ids, metadata))
                self.check_clip((raw, colors, ids, metadata))
            for cid in (-1, 256):
                with self.assertRaisesRegex(RuntimeError, f"no visible pixels for raw color channel {cid}"):
                    analysis.detect_ceilings(raw, colors, [cid], {})
                self.check_clip((raw, colors, [cid], {}))
            self.check_clip((raw, colors, ids, {cid: 10 ** 1000 for cid in ids}))
            for ids_factory in (lambda: iter(ids), lambda: (cid for cid in ids)):
                self.assertCeilings(analysis.detect_ceilings(raw, colors, ids_factory(), levels),
                                    _legacy_ceilings(raw, colors, ids_factory(), levels))
                self.assertClipBits(analysis.compute_clip_pct_by_thresholds(raw, colors, ids_factory(), levels),
                                    _legacy_clip(raw, colors, ids_factory(), levels))
        ceiling.assert_not_called()
        clip.assert_not_called()

    def test_unrequested_bad_metadata_is_ignored_and_bool_ids_use_reference(self):
        raw = np.asarray([[1, 2, 65535]], np.uint16)
        colors = np.asarray([[0, 1, 1]], np.uint8)
        with self.fake_extension() as (ceiling, clip):
            self.check_ceilings((raw, colors, [0, 1], {200: float("nan")}))
            self.check_clip((raw, colors, [0, 1], {200: float("nan")}))
            ceiling.assert_called_once()
            clip.assert_called_once()
            ceiling.reset_mock()
            clip.reset_mock()
            self.check_ceilings((raw, colors, [False, True], {}))
            self.check_clip((raw, colors, [False, True], {}))
            ceiling.assert_not_called()
            clip.assert_not_called()

    def test_float_nan_and_inf_keep_original_results_or_conversion_errors(self):
        colors = np.zeros((1, 3), np.uint8)
        with self.fake_extension() as (ceiling, clip):
            for value, error in ((float("nan"), ValueError), (float("inf"), OverflowError)):
                raw = np.asarray([[0., 2., value]], np.float32)
                with self.assertRaises(error):
                    analysis.detect_ceilings(raw, colors, [0], {})
                self.check_clip((raw, colors, [0], {}))
        ceiling.assert_not_called()
        clip.assert_not_called()

    def test_off_and_each_skip_key_disable_only_the_requested_kernel(self):
        args = _mosaic()
        with self.fake_extension() as (ceiling, clip):
            with mock.patch.dict(os.environ, DNGSCAN_FAST="0"):
                self.check_ceilings(args)
                self.check_clip(args)
            ceiling.assert_not_called()
            clip.assert_not_called()
            with mock.patch.dict(os.environ, DNGSCAN_FAST_SKIP=CEILING):
                self.check_ceilings(args)
                self.check_clip(args)
            ceiling.assert_not_called()
            clip.assert_called_once()
            clip.reset_mock()
            with mock.patch.dict(os.environ, DNGSCAN_FAST_SKIP=CLIP):
                self.check_ceilings(args)
                self.check_clip(args)
            ceiling.assert_called_once()
            clip.assert_not_called()

    def test_native_errors_fallback_in_auto_and_raise_in_strict_without_mutation(self):
        args = _mosaic()
        before = args[0].tobytes(), args[1].tobytes()
        with self.fake_extension() as (ceiling, clip):
            ceiling.side_effect = ValueError("intentional ceiling preflight")
            clip.side_effect = ValueError("intentional clip preflight")
            for fn in (analysis.detect_ceilings, analysis.compute_clip_pct_by_thresholds):
                with self.assertRaises(_fast.NativeKernelError):
                    fn(*args)
            with mock.patch.dict(os.environ, DNGSCAN_FAST="auto"), \
                 self.assertLogs("dngscan._fast", level="WARNING"):
                self.check_ceilings(args)
                self.check_clip(args)
        self.assertEqual(before, (args[0].tobytes(), args[1].tobytes()))

    def test_missing_extension_or_entry_obey_auto_and_strict_policy(self):
        args = _mosaic()
        for extension in (None, SimpleNamespace()):
            with mock.patch.object(_fast, "_load_extension", return_value=extension), \
                 mock.patch.dict(os.environ, DNGSCAN_FAST="auto", DNGSCAN_FAST_SKIP=""):
                self.check_ceilings(args)
                self.check_clip(args)
                with mock.patch.dict(os.environ, DNGSCAN_FAST="1"):
                    for fn in (analysis.detect_ceilings, analysis.compute_clip_pct_by_thresholds):
                        with self.assertRaises(_fast.NativeKernelError):
                            fn(*args)

    def test_invalid_native_counts_obey_error_policy_without_fake_sensor_facts(self):
        raw, colors, ids, levels = args = _mosaic()
        good_ceilings = _ceiling_counts(raw, colors, [8] * 256)
        good_clip = _clip_counts(raw, colors, [0] * 256)
        malformed_ceilings = [None, ([0],) * 4]
        bad = [values.copy() for values in good_ceilings]
        bad[0][ids[0]] = 65536
        malformed_ceilings.append(bad)
        bad = [values.copy() for values in good_ceilings]
        bad[2][ids[0]] = bad[3][ids[0]] + 1
        malformed_ceilings.append(bad)
        bad = [values.copy() for values in good_ceilings]
        bad[1][ids[0]] += 1
        malformed_ceilings.append(bad)
        malformed_clip = [None, ([0], [0])]
        bad = [values.copy() for values in good_clip]
        bad[1][ids[0]] = bad[0][ids[0]] + 1
        malformed_clip.append(bad)
        bad = [values.copy() for values in good_clip]
        bad[0][ids[0]] += 1
        malformed_clip.append(bad)
        with self.fake_extension() as (ceiling, clip):
            for fn, call, cases, check in (
                    (analysis.detect_ceilings, ceiling, malformed_ceilings, self.check_ceilings),
                    (analysis.compute_clip_pct_by_thresholds, clip, malformed_clip, self.check_clip)):
                call.side_effect = None
                for index, result in enumerate(cases):
                    with self.subTest(kernel=fn.__name__, case=index):
                        call.return_value = result
                        with self.assertRaises(_fast.NativeKernelError):
                            fn(*args)
                        with mock.patch.dict(os.environ, DNGSCAN_FAST="auto"), \
                             self.assertLogs("dngscan._fast", level="WARNING"):
                            check(args)


class SensorChannelNativeTests(_OracleTest):
    @classmethod
    def setUpClass(cls):
        cls.ext = _fast._load_extension()
        if cls.ext is None or not all(hasattr(cls.ext, name) for name in (CEILING, CLIP)):
            raise unittest.SkipTest("ABI 17 sensor channel kernels not built")

    def check_direct(self, raw, colors, windows, thresholds):
        before = raw.tobytes(), colors.tobytes()
        maxima, totals, exact, near = getattr(self.ext, CEILING)(raw, colors, windows)
        clip_totals, hits = getattr(self.ext, CLIP)(raw, colors, thresholds)
        self.assertEqual((list(maxima), list(totals), list(exact), list(near)),
                         _ceiling_counts(raw, colors, windows))
        self.assertEqual((list(clip_totals), list(hits)), _clip_counts(raw, colors, thresholds))
        self.assertEqual(list(totals), list(clip_totals))
        self.assertEqual(sum(totals), raw.size)
        for values in (maxima, totals, exact, near, hits):
            self.assertEqual(len(values), 256)
            self.assertTrue(all(type(value) is int and value >= 0 for value in values))
        self.assertEqual(before, (raw.tobytes(), colors.tobytes()))

    def test_native_counts_and_public_results_match_bayer_xtrans_and_linear(self):
        cases = [_mosaic(), _xtrans(), _linear()]
        with mock.patch.dict(os.environ, DNGSCAN_FAST="1", DNGSCAN_FAST_SKIP=""):
            for args in cases:
                with self.subTest(shape=args[0].shape):
                    self.check_direct(*args[:2], [8] * 256, [30000] * 256)
                    self.check_ceilings(args)
                    self.check_clip(args)

    def test_native_negative_transposed_zero_strides_and_readonly_views(self):
        for raw, colors, _, _ in (_mosaic(), _linear()):
            views = [(raw[::-1, ::-1], colors[::-1, ::-1]),
                     (np.swapaxes(raw, 0, 1), np.swapaxes(colors, 0, 1)),
                     (np.broadcast_to(raw[:1], raw.shape), np.broadcast_to(colors[:1], colors.shape))]
            if raw.ndim == 3:
                views.append((raw[..., ::-1], colors[..., ::-1]))
            for image, cmap in views:
                with self.subTest(shape=image.shape, strides=image.strides):
                    image.flags.writeable = False
                    cmap.flags.writeable = False
                    self.check_direct(image, cmap, [3] * 256, [32768] * 256)

    def test_native_exact_maxima_window_zero_full_range_and_signed_thresholds(self):
        raw = np.asarray([[0, 1, 2, 65534, 65535, 65535]], np.uint16)
        colors = np.asarray([[0, 0, 2, 2, 255, 255]], np.uint8)
        for window in (0, 1, 2, 65535):
            for threshold in (-2147483648, -1, 0, 1, 65535, 65536, 2147483647):
                with self.subTest(window=window, threshold=threshold):
                    self.check_direct(raw, colors, [window] * 256, [threshold] * 256)
        self.check_direct(np.zeros((3, 7), np.uint16), np.full((3, 7), 255, np.uint8),
                          [0] * 256, [0] * 256)

    def test_native_empty_shapes_have_all_zero_statistics(self):
        for shape in ((0, 7), (3, 0), (0, 7, 4), (3, 7, 0)):
            raw = np.empty(shape, np.uint16)[::-1]
            colors = np.empty(shape, np.uint8)[::-1]
            with self.subTest(shape=shape):
                self.check_direct(raw, colors, [8] * 256, [0] * 256)

    def test_native_invalid_luts_and_shapes_are_rejected(self):
        raw, colors, _, _ = _mosaic()
        for name, lut in ((CEILING, [8] * 256), (CLIP, [0] * 256)):
            invalid = [(raw.ravel(), colors.ravel(), lut), (raw, colors[:, :-1], lut),
                       (raw, colors, lut[:-1]), (raw, colors, lut + [0])]
            invalid.extend((raw, colors, [value] + lut[1:]) for value in
                           ((-1, 65536, 2 ** 40) if name == CEILING else (-2 ** 40, 2 ** 40)))
            for index, args in enumerate(invalid):
                with self.subTest(kernel=name, case=index):
                    with self.assertRaises((TypeError, ValueError, OverflowError)):
                        getattr(self.ext, name)(*args)

    def test_native_wrong_dtype_endian_unaligned_and_unused_bad_strides_are_rejected(self):
        raw, colors, _, _ = _mosaic()
        unaligned = np.ndarray(raw.shape, np.uint16, buffer=bytearray(raw.nbytes + 1), offset=1)
        unaligned[...] = raw
        odd = np.ndarray((1, 2), np.uint16, buffer=bytearray(4), strides=(1, 2))
        min_colors = np.ndarray((1, 2), np.uint8, buffer=bytearray(2),
                                strides=(np.iinfo(np.intp).min, 1))
        cases = [(raw.astype(np.float32), colors), (raw.astype(np.int32), colors),
                 (raw.astype(raw.dtype.newbyteorder("S")), colors), (unaligned, colors),
                 (raw, colors.astype(np.int16)), (odd, np.zeros((1, 2), np.uint8)),
                 (np.zeros((1, 2), np.uint16), min_colors)]
        for name, lut in ((CEILING, [8] * 256), (CLIP, [0] * 256)):
            for index, (image, cmap) in enumerate(cases):
                with self.subTest(kernel=name, case=index):
                    with self.assertRaises((TypeError, ValueError)):
                        getattr(self.ext, name)(image, cmap, lut)


if __name__ == "__main__":
    unittest.main()
