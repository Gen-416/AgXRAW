# SPDX-License-Identifier: GPL-3.0-or-later
"""The already-half handoff preserves all original codes and buffer ownership."""
import gc
import unittest
import weakref

import numpy as np

from dngscan.coreimage_decode import scene_float_to_half


def previous_handoff(rgb):
    linear = np.asarray(rgb, dtype=np.float32)
    limit = float(np.finfo(np.float16).max)
    finite = np.nan_to_num(linear, nan=0.0, posinf=limit, neginf=-limit)
    return np.clip(finite, -limit, limit).astype(np.float16)


class CoreImageHalfStorageTests(unittest.TestCase):
    def test_all_half_bit_patterns_match_previous_handoff(self):
        source = np.arange(65536, dtype=np.uint16).view(np.float16).reshape(256, 256)
        before = source.view(np.uint16).copy()
        with np.errstate(all="ignore"):
            expected = previous_handoff(source)
            actual, scale = scene_float_to_half(source)
        self.assertEqual(scale, 1.0)
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
        np.testing.assert_array_equal(source.view(np.uint16), before)
        self.assertFalse(np.shares_memory(actual, source))

    def test_strides_broadcast_and_read_only_inputs_keep_independent_writable_output(self):
        source = np.array([0., -0., np.nan, np.inf, -np.inf, 65504., -65504., 1e-7,
                           .25, .5, 1., 2.], np.float16).reshape(2, 2, 3)
        readonly = source.copy()
        readonly.flags.writeable = False
        for view in (source, source[::-1, ::-1], source.transpose(1, 0, 2),
                     np.broadcast_to(source[:1], (5, 2, 3)), readonly):
            with self.subTest(strides=view.strides, writable=view.flags.writeable):
                expected = previous_handoff(view)
                result, _ = scene_float_to_half(view)
                np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))
                self.assertFalse(np.shares_memory(result, view))
                self.assertTrue(result.flags.writeable)

    def test_output_does_not_retain_input_owner(self):
        source = np.ones((3, 4, 3), dtype=np.float16)
        reference = weakref.ref(source)
        output, _ = scene_float_to_half(source)
        del source
        gc.collect()
        self.assertIsNone(reference())
        output[0, 0, 0] = 2
        self.assertEqual(float(output[0, 0, 0]), 2.)

    def test_generic_dtypes_and_empty_inputs_keep_original_conversion(self):
        for dtype in (np.float32, np.float64, np.uint16, np.dtype(">f2")):
            values = [0., 1., 65504., 65535.] if np.issubdtype(dtype, np.integer) else [
                0., -0., np.nan, np.inf, -np.inf, 70000., -70000., .3333]
            with np.errstate(all="ignore"):
                source = np.array(values, dtype=dtype)
                result, scale = scene_float_to_half(source)
                expected = previous_handoff(source)
            np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))
            self.assertEqual(scale, 1.)
        result, _ = scene_float_to_half(np.empty((0, 2, 3), np.float16))
        self.assertEqual(result.shape, (0, 2, 3))


if __name__ == "__main__":
    unittest.main()
