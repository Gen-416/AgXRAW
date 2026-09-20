# SPDX-License-Identifier: GPL-3.0-or-later
"""Acquired sensor codes must outlive LibRaw without a writable base alias."""
from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from dngscan import evidence as evidence_module
from dngscan.evidence import EvidenceAcquisitionError, acquire_raw_evidence


class _RawContext:
    def __init__(self, image, colors=None, *, overwrite_on_close=False):
        self.raw_image_visible = image
        self._colors = colors
        self.color_reads = 0
        self.overwrite_on_close = overwrite_on_close
        self.closed = False
        self.white_level = 4095
        self.black_level_per_channel = [64, 65, 66, 67]
        self.camera_whitebalance = [2.0, 1.0, 1.5, 1.0]
        self.daylight_whitebalance = [2.2, 1.0, 1.4, 1.0]
        self.camera_white_level_per_channel = [4095, 4094, 4093, 4092]
        self.color_desc = b"RGBG\x00"
        self.raw_pattern = np.asarray([[0, 1], [3, 2]], dtype=np.uint8)
        self.rgb_xyz_matrix = np.eye(4, 3, dtype=np.float32)
        self.color_matrix = np.eye(3, 4, dtype=np.float32)
        self.sizes = SimpleNamespace(flip=5)

    @property
    def raw_colors_visible(self):
        self.color_reads += 1
        if self._colors is None:
            raise RuntimeError("LinearRGB must not request a CFA color plane")
        return self._colors

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True
        if self.overwrite_on_close:
            self.raw_image_visible[...] = 0
            if self._colors is not None:
                self._colors[...] = 0
        return False


def _acquire(source):
    with (patch("dngscan.evidence.rawpy.imread", return_value=source),
          patch("dngscan.spatial_black.read", return_value=None),
          patch("dngscan.embedded_lens.read", return_value=None)):
        return acquire_raw_evidence(Path("ownership-test.dng"))


class EvidenceOwnershipTests(unittest.TestCase):
    def assert_immutable_bytes_chain(self, array):
        """A readonly owning ndarray or a readonly view of writable data is insufficient."""
        self.assertFalse(array.flags.writeable)
        with self.assertRaises(ValueError):
            array[...] = 0
        current = array
        seen = set()
        while isinstance(current, np.ndarray):
            self.assertNotIn(id(current), seen)
            seen.add(id(current))
            self.assertFalse(current.flags.owndata)
            self.assertFalse(current.flags.writeable)
            with self.assertRaises(ValueError):
                current.setflags(write=True)
            current = current.base
        self.assertIsInstance(current, bytes)
        return current

    def test_cfa_preserves_values_and_detaches_noncontiguous_views(self):
        for layout in ("negative", "transposed", "stepped"):
            with self.subTest(layout=layout):
                image = np.arange(96, dtype=np.uint16).reshape(8, 12)
                colors = (np.arange(96, dtype=np.uint8) % 4).reshape(8, 12)
                if layout == "negative":
                    image, colors = image[::-1, ::-2], colors[::-1, ::-2]
                elif layout == "transposed":
                    image, colors = image.T, colors.T
                else:
                    image, colors = image[::2, 1::3], colors[::2, 1::3]
                expected_image, expected_colors = image.copy(), colors.copy()
                source = _RawContext(image, colors, overwrite_on_close=True)
                acquired = _acquire(source)
                self.assertTrue(source.closed)
                self.assertEqual(source.color_reads, 1)
                self.assertEqual(acquired.sample_kind, "cfa")
                for actual, expected in ((acquired.raw_image, expected_image),
                                         (acquired.raw_colors, expected_colors)):
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.strides, expected.strides)
                    self.assertTrue(actual.flags.c_contiguous)
                    self.assertEqual(actual.tobytes(), expected.tobytes())
                    self.assert_immutable_bytes_chain(actual)
                source.raw_image_visible[...] = 123
                source._colors[...] = 3
                np.testing.assert_array_equal(acquired.raw_image, expected_image)
                np.testing.assert_array_equal(acquired.raw_colors, expected_colors)

    def test_linear_rgb_keeps_only_first_three_channels_with_immutable_broadcast_ids(self):
        for channels in (3, 4):
            with self.subTest(channels=channels):
                image = np.arange(4 * 5 * channels, dtype=np.uint16).reshape(4, 5, channels)
                if channels == 4:
                    image[..., 3] = 65535
                image = image[::-1, ::2, :]
                expected = image[..., :3].copy()
                source = _RawContext(image, overwrite_on_close=True)
                acquired = _acquire(source)
                self.assertEqual(source.color_reads, 0)
                self.assertTrue(source.closed)
                self.assertEqual(acquired.sample_kind, "linear-camera-rgb")
                self.assertEqual(acquired.raw_pattern, [])
                self.assertEqual(acquired.raw_image.dtype, expected.dtype)
                self.assertEqual(acquired.raw_image.shape, expected.shape)
                self.assertEqual(acquired.raw_image.strides, expected.strides)
                self.assertEqual(acquired.raw_image.tobytes(), expected.tobytes())
                self.assert_immutable_bytes_chain(acquired.raw_image)
                colors = acquired.raw_colors
                self.assertEqual(colors.shape, expected.shape)
                self.assertEqual(colors.dtype, np.dtype(np.uint8))
                self.assertEqual(colors.strides, (0, 0, 1))
                np.testing.assert_array_equal(colors, np.broadcast_to([0, 1, 2], expected.shape))
                owner = self.assert_immutable_bytes_chain(colors)
                self.assertEqual(owner, bytes((0, 1, 2)))
                self.assertEqual(len(owner), 3)

    def test_byte_order_is_preserved_without_reinterpreting_codes(self):
        for dtype in ("<u2", ">u2"):
            with self.subTest(dtype=dtype):
                image = np.asarray([[1, 256, 4095], [1025, 7, 65535]], dtype=dtype)
                colors = np.asarray([[0, 1, 2], [3, 0, 1]], dtype=np.uint8)
                expected_bytes = image.tobytes()
                acquired = _acquire(_RawContext(image, colors, overwrite_on_close=True))
                self.assertEqual(acquired.raw_image.dtype.str, np.dtype(dtype).str)
                self.assertEqual(acquired.raw_image.tobytes(), expected_bytes)
                np.testing.assert_array_equal(acquired.raw_image, [[1, 256, 4095], [1025, 7, 65535]])
                self.assert_immutable_bytes_chain(acquired.raw_image)

    def test_acquisition_retains_no_rawpy_handle_or_writable_source_allocation(self):
        for channels in (None, 4):
            with self.subTest(channels=channels):
                shape = (4, 6) if channels is None else (4, 6, channels)
                image_owner = np.arange(np.prod(shape), dtype=np.uint16)
                image = image_owner.reshape(shape)[::-1, ::2]
                color_owner = (np.arange(24, dtype=np.uint8) % 4) if channels is None else None
                colors = color_owner.reshape(4, 6)[::-1, ::2] if color_owner is not None else None
                source = _RawContext(image, colors)
                refs = [weakref.ref(value) for value in (source, image, image_owner, colors, color_owner)
                        if value is not None]
                acquired = _acquire(source)
                del source, image, image_owner, colors, color_owner
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs))
                self.assert_immutable_bytes_chain(acquired.raw_image)
                self.assert_immutable_bytes_chain(acquired.raw_colors)
                self.assertGreater(int(acquired.raw_image.max()), 0)

    def test_metadata_list_contract_and_independent_copies_are_preserved(self):
        source = _RawContext(np.arange(16, dtype=np.uint16).reshape(4, 4),
                             np.tile([[0, 1], [3, 2]], (2, 2)).astype(np.uint8))
        acquired = _acquire(source)
        for target_name, source_name in (
            ("black_levels", "black_level_per_channel"),
            ("camera_wb", "camera_whitebalance"),
            ("daylight_wb", "daylight_whitebalance"),
            ("camera_white_levels", "camera_white_level_per_channel"),
        ):
            with self.subTest(field=target_name):
                actual = getattr(acquired, target_name)
                original = getattr(source, source_name)
                self.assertIsInstance(actual, list)
                self.assertIsNot(actual, original)
                self.assertTrue(all(isinstance(value, float) for value in actual))
                expected = list(actual)
                original[0] = -999
                self.assertEqual(actual, expected)
        self.assertIsInstance(acquired.raw_pattern, list)
        self.assertTrue(all(isinstance(row, list) for row in acquired.raw_pattern))
        source.raw_pattern[...] = 9
        self.assertEqual(acquired.raw_pattern, [[0, 1], [3, 2]])
        source.rgb_xyz_matrix[...] = 0
        source.color_matrix[...] = 0
        np.testing.assert_array_equal(acquired.xyz_to_cam, np.eye(4, 3, dtype=np.float32))
        np.testing.assert_array_equal(acquired.color_matrix, np.eye(3, 4, dtype=np.float32))

    def test_missing_white_level_uses_detached_codes(self):
        source = _RawContext(np.asarray([[3, 7], [5, 2]], dtype=np.uint16),
                             np.asarray([[0, 1], [3, 2]], dtype=np.uint8), overwrite_on_close=True)
        source.white_level = None
        source.daylight_whitebalance = None
        acquired = _acquire(source)
        self.assertEqual(acquired.white_level, 7)
        self.assertIsNone(acquired.daylight_wb)

    def test_empty_and_mismatched_planes_keep_acquisition_error_semantics(self):
        cases = (
            ((0, 3), (0, 3), "no visible sensor pixels"),
            ((2, 3), (0, 3), "no visible sensor pixels"),
            ((2, 3), (3, 2), "shapes differ"),
        )
        for image_shape, color_shape, message in cases:
            with self.subTest(image=image_shape, colors=color_shape):
                source = _RawContext(np.zeros(image_shape, dtype=np.uint16),
                                     np.zeros(color_shape, dtype=np.uint8))
                with self.assertRaisesRegex(EvidenceAcquisitionError, message) as caught:
                    _acquire(source)
                self.assertFalse(caught.exception.unsupported_format)
                self.assertIsInstance(caught.exception.__cause__, RuntimeError)
                self.assertTrue(source.closed)

    def test_decode_exception_classification_is_unchanged(self):
        unsupported = evidence_module.rawpy.LibRawFileUnsupportedError("synthetic unsupported RAW")
        for error in (unsupported, RuntimeError("synthetic decode failure")):
            with self.subTest(kind=type(error).__name__):
                with patch("dngscan.evidence.rawpy.imread", side_effect=error):
                    with self.assertRaises(EvidenceAcquisitionError) as caught:
                        acquire_raw_evidence(Path("ownership-test.dng"))
                self.assertEqual(caught.exception.unsupported_format, error is unsupported)
                self.assertIs(caught.exception.__cause__, error)
        missing = FileNotFoundError("synthetic missing file")
        with patch("dngscan.evidence.rawpy.imread", side_effect=missing):
            with self.assertRaises(FileNotFoundError) as caught:
                acquire_raw_evidence(Path("ownership-test.dng"))
        self.assertIs(caught.exception, missing)


if __name__ == "__main__":
    unittest.main()
