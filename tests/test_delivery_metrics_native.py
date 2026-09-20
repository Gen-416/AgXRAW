"""Exact shared delivery scans and export-local capacity, independent of codecs."""
import concurrent.futures
import gc
import os
import unittest
import weakref
from unittest import mock

import numpy as np

from dngscan import _fast
from dngscan.auto_encode import coding_metrics

EXT = _fast._load_extension()
READY = EXT is not None and hasattr(EXT, 'HdrMetricsWorkspace')


@unittest.skipUnless(READY, 'new ABI18 delivery metrics build required')
class FusedBaseCodingTests(unittest.TestCase):
    def test_exact_old_fields_for_complete_partial_and_singleton_blocks(self):
        rng = np.random.default_rng(61)
        old_buffer = np.getbufsize()
        try:
            for buffer in (32, 8192):
                np.setbufsize(buffer)
                for h, w in ((1, 1), (1, 17), (7, 3), (8, 8), (15, 9), (17, 16),
                              (131, 17), (129, 67), (513, 31)):
                    decoded = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                    intended = rng.integers(0, 256, decoded.shape, dtype=np.uint8)
                    expected = {**EXT.base_roundtrip_metrics(decoded, intended),
                                **coding_metrics(decoded, intended)}
                    self.assertEqual(EXT.base_and_coding_metrics_u8(decoded, intended, buffer),
                                     expected, (h, w, buffer))
        finally:
            np.setbufsize(old_buffer)

    def test_rgba_strides_broadcast_and_readonly_have_no_input_copy_or_mutation(self):
        rng = np.random.default_rng(62)
        a = rng.integers(0, 256, (17, 23, 4), dtype=np.uint8)
        b = rng.integers(0, 256, (17, 23, 4), dtype=np.uint8)
        for decoded, intended in ((a, b), (a[::-1, ::-1], b[::-1, ::-1]),
                                  (a.transpose(1, 0, 2), b.transpose(1, 0, 2)),
                                  (np.broadcast_to(a[:1, :1], a.shape), b)):
            decoded.flags.writeable = False
            intended.flags.writeable = False
            before = (decoded.tobytes(), intended.tobytes())
            ar, br = decoded[..., :3], intended[..., :3]
            expected = {**EXT.base_roundtrip_metrics(ar, br), **coding_metrics(ar, br)}
            self.assertEqual(EXT.base_and_coding_metrics_u8(decoded, intended), expected)
            self.assertEqual((decoded.tobytes(), intended.tobytes()), before)

    def test_invalid_inputs_are_rejected_without_forcecast(self):
        good = np.zeros((3, 5, 3), np.uint8)
        for bad in (good.astype(np.uint16), good[..., 0], good[:2], good[:0], good[..., :2]):
            with self.assertRaises((TypeError, ValueError)):
                EXT.base_and_coding_metrics_u8(bad, good)
        with self.assertRaises(ValueError):
            EXT.base_and_coding_metrics_u8(good, good, 0)
        for array in (good, good.transpose(1, 0, 2)):
            with self.assertRaises(ValueError):
                EXT.base_and_coding_metrics_u8(array, array, 32 if np.getbufsize() != 32 else 8192)


@unittest.skipUnless(READY, 'new ABI18 delivery metrics build required')
class HdrMetricsWorkspaceTests(unittest.TestCase):
    WEIGHTS = [.22897, .69174, .07929]

    def test_all_fields_match_old_ranks_with_resize_and_invalid_recovery(self):
        self.addCleanup(EXT.set_thread_budget, 0)
        rng = np.random.default_rng(63)
        workspace = EXT.HdrMetricsWorkspace()
        for shape in ((1, 1, 4), (7, 13, 4), (17, 23, 3), (129, 67, 4), (9, 8, 3)):
            intended = rng.uniform(-.1, 8, shape).astype(np.float16)
            actual = (intended.astype(np.float32) * .97).astype(np.float16)
            if shape[2] == 4:
                intended[..., 3] = np.nan
                actual[..., 3] = np.inf
            for workers in (1, 2, 8):
                EXT.set_thread_budget(workers)
                for a, e in ((actual, intended), (actual[::-1, ::-1], intended[::-1, ::-1])):
                    expected = EXT.hdr_roundtrip_metrics(a, e, self.WEIGHTS)
                    self.assertEqual(workspace.measure(a, e, self.WEIGHTS), expected)
            bad = actual.copy()
            bad[-1, -1, 2] = np.nan
            self.assertEqual(workspace.measure(bad, intended, self.WEIGHTS),
                             EXT.hdr_roundtrip_metrics(bad, intended, self.WEIGHTS))
            self.assertEqual(workspace.measure(actual, intended, self.WEIGHTS),
                             EXT.hdr_roundtrip_metrics(actual, intended, self.WEIGHTS))

    def test_capacity_is_bounded_and_never_keeps_source_owners(self):
        workspace = EXT.HdrMetricsWorkspace(0)
        source = np.ones((17, 31, 4), np.float16)
        ref = weakref.ref(source)
        workspace.measure(source, source, self.WEIGHTS)
        self.assertEqual(workspace.retained_bytes(), 0)
        del source
        gc.collect()
        self.assertIsNone(ref())
        with self.assertRaises(ValueError):
            EXT.HdrMetricsWorkspace(536870913)

    def test_concurrent_calls_are_isolated_and_validation_keeps_empty_semantics(self):
        workspace = EXT.HdrMetricsWorkspace()
        a = np.ones((13, 19, 4), np.float16)
        inputs = [a, a * np.float16(.9), a * np.float16(1.1)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda value: workspace.measure(value, a, self.WEIGHTS), inputs))
        for value, result in zip(inputs, results):
            self.assertEqual(result, EXT.hdr_roundtrip_metrics(value, a, self.WEIGHTS))
        empty = np.empty((0, 19, 4), np.float16)
        self.assertEqual(workspace.measure(empty, empty, self.WEIGHTS),
                         EXT.hdr_roundtrip_metrics(empty, empty, self.WEIGHTS))
        unaligned = np.ndarray(a.shape, np.float16, bytearray(a.nbytes + 1), offset=1)
        unaligned[...] = a
        self.assertEqual(EXT.hdr_roundtrip_metrics(unaligned, a, self.WEIGHTS),
                         EXT.hdr_roundtrip_metrics(a, a, self.WEIGHTS))
        for bad in (a.astype(np.float32), a[:, :, :2], a[:12], unaligned):
            with self.assertRaises((TypeError, ValueError)):
                workspace.measure(bad, a, self.WEIGHTS)


@unittest.skipUnless(READY, 'new ABI18 delivery metrics build required')
class MetricsDispatchTests(unittest.TestCase):
    def test_dispatch_and_explicit_ablation_preserve_every_field(self):
        from dngscan import gainmap
        rng = np.random.default_rng(64)
        a = rng.integers(0, 256, (17, 31, 4), np.uint8)[..., :3]
        b = rng.integers(0, 256, a.shape, np.uint8)
        with mock.patch.dict(os.environ, DNGSCAN_FAST='1', DNGSCAN_FAST_SKIP=''):
            actual = gainmap._base_and_coding_metrics_arrays(a, b)
        with mock.patch.dict(os.environ, DNGSCAN_FAST='1', DNGSCAN_FAST_SKIP='base_and_coding_metrics_u8'):
            expected = gainmap._base_and_coding_metrics_arrays(a, b)
        self.assertEqual(actual, expected)
        a = rng.random((17, 31, 4)).astype(np.float16)[..., :3]
        b = (a * np.float16(.98)).copy()
        with mock.patch.dict(os.environ, DNGSCAN_FAST='1', DNGSCAN_FAST_SKIP=''):
            workspace = gainmap._new_hdr_metrics_workspace()
            self.assertEqual(gainmap._roundtrip_error_arrays(a, b, _workspace=workspace),
                             gainmap._roundtrip_error_arrays(a, b))
        for mode, skip in (('0', ''), ('1', 'hdr_roundtrip_metrics'), ('1', 'HdrMetricsWorkspace')):
            with mock.patch.dict(os.environ, DNGSCAN_FAST=mode, DNGSCAN_FAST_SKIP=skip):
                self.assertIsNone(gainmap._new_hdr_metrics_workspace())

    def test_rgba_public_metrics_keep_native_alpha_ignorance(self):
        from dngscan import gainmap
        rng = np.random.default_rng(66)
        a = rng.random((17, 31, 4)).astype(np.float16)
        b = (a * np.float16(.99)).copy()
        a[..., 3], b[..., 3] = np.nan, np.inf
        expected = EXT.hdr_roundtrip_metrics(a, b, gainmap._HDR_LUMA_WEIGHTS.tolist())
        with mock.patch.dict(os.environ, DNGSCAN_FAST='1', DNGSCAN_FAST_SKIP=''):
            self.assertEqual(gainmap._roundtrip_error_arrays(a, b), expected)
            self.assertEqual(gainmap._roundtrip_error_arrays(a, b,
                _workspace=gainmap._new_hdr_metrics_workspace()), expected)

    def test_forcecast_adapters_copy_unaligned_storage_only(self):
        def unaligned(values):
            result = np.ndarray(values.shape, values.dtype, bytearray(values.nbytes + 1), offset=1)
            result[...] = values
            self.assertFalse(result.flags.aligned)
            return result

        rng = np.random.default_rng(65)
        plane = rng.random((7, 9, 3), dtype=np.float32)
        self.assertEqual(EXT.feather_masks_f16(unaligned(plane)).tobytes(),
                         EXT.feather_masks_f16(plane).tobytes())
        rgb = plane.reshape(-1, 3).astype(np.float64)
        args = (.7, 1.3, 1.1, .2)
        self.assertEqual(EXT.film_compression_ev(unaligned(rgb), *args).tobytes(),
                         EXT.film_compression_ev(rgb, *args).tobytes())
        raw = rng.integers(0, 65535, plane.shape, np.uint16)
        args = ([[1., .03, 0., 0., 0., 0.]], .5, .5, 1., False, False)
        self.assertEqual(EXT.warp_dng(unaligned(raw), *args).tobytes(),
                         EXT.warp_dng(raw, *args).tobytes())

    def test_unaligned_half_uses_numpy_without_calling_workspace(self):
        from dngscan import gainmap
        a = np.ndarray((7, 9, 3), np.float16, bytearray(7 * 9 * 3 * 2 + 1), offset=1)
        a.fill(1.)
        workspace = mock.Mock()
        with mock.patch.dict(os.environ, DNGSCAN_FAST='1', DNGSCAN_FAST_SKIP=''):
            actual = gainmap._roundtrip_error_arrays(a, a, _workspace=workspace)
        with mock.patch.dict(os.environ, DNGSCAN_FAST='0'):
            expected = gainmap._roundtrip_error_arrays(a, a)
        self.assertEqual(actual, expected)
        workspace.measure.assert_not_called()


if __name__ == '__main__':
    unittest.main()
