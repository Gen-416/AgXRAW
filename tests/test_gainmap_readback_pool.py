# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-read native cleanup, with Python bitmap ownership across pool drains."""
from contextlib import contextmanager
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import numpy as np

from dngscan import gainmap


class _NativeObject:
    def __init__(self, harness):
        harness.assert_active()
        self.harness = harness
        self.alive = True
        harness.resources.append(self)
        harness.native_refs.append(weakref.ref(self))

    def release(self):
        self.alive = False

    def assert_live(self):
        self.harness.assert_active()
        if not self.alive:
            raise AssertionError("native object used after pool drain")


class _Image(_NativeObject):
    def extent(self):
        self.assert_live()
        return SimpleNamespace(size=SimpleNamespace(
            width=0 if self.harness.failure == "dimensions" else 3, height=2))

    def imageByApplyingGainMap_(self, gainmap_image):
        self.assert_live()
        gainmap_image.assert_live()
        if self.harness.failure == "composite":
            return None
        return _Image(self.harness)


class _Context(_NativeObject):
    def render_toBitmap_rowBytes_bounds_format_colorSpace_(self, image, bitmap, row_bytes,
                                                          extent, pixel_format, color):
        self.assert_live()
        image.assert_live()
        color.assert_live()
        if self.harness.failure == "render":
            raise RuntimeError("injected bitmap render failure")
        self.harness.renders += 1
        dtype = np.uint8 if pixel_format == "RGBA8" else np.float16
        expected = (np.arange(24).reshape(2, 3, 4) + self.harness.renders * 16).astype(dtype)
        if isinstance(bitmap, np.ndarray):
            bitmap[...] = expected
            self.harness.bitmap_refs.append(weakref.ref(bitmap))
        else:
            bitmap[:] = expected.tobytes()
        self.harness.expected.append(expected)


class _ReadbackHarness:
    def __init__(self, failure=None):
        self.failure = failure
        self.active = False
        self.resources = []
        self.native_refs = []
        self.events = []
        self.renders = 0
        self.expected = []
        self.bitmap_refs = []

    def assert_active(self):
        if not self.active:
            raise AssertionError("native read escaped its autorelease pool")

    @contextmanager
    def pool(self):
        if self.active:
            raise AssertionError("unexpected nested readback pool")
        self.active = True
        self.events.append("enter")
        try:
            yield
        finally:
            for obj in self.resources:
                obj.release()
            self.resources.clear()
            self.active = False
            self.events.append("drain")

    def primary(self, _url):
        self.assert_active()
        return None if self.failure == "image" else _Image(self)

    def auxiliary(self, _url, _options):
        self.assert_active()
        return None if self.failure == "gainmap" else _Image(self)

    def context(self, _options):
        self.assert_active()
        return None if self.failure == "context" else _Context(self)

    @contextmanager
    def modules(self):
        quartz = SimpleNamespace(
            CGColorSpaceCreateWithName=lambda _: _NativeObject(self),
            kCGColorSpaceDisplayP3="p3", kCGColorSpaceSRGB="srgb",
            kCGColorSpaceExtendedLinearDisplayP3="linear-p3",
            kCIImageAuxiliaryHDRGainMap="gainmap", kCIContextCacheIntermediates="cache",
            kCIFormatRGBAh="RGBAh", kCIFormatRGBA8="RGBA8",
            CIImage=SimpleNamespace(imageWithContentsOfURL_=self.primary,
                                    imageWithContentsOfURL_options_=self.auxiliary),
            CIContext=SimpleNamespace(contextWithOptions_=self.context),
        )
        foundation = SimpleNamespace(NSURL=SimpleNamespace(fileURLWithPath_=lambda value: value))
        with mock.patch.dict(sys.modules, Quartz=quartz, Foundation=foundation,
                             objc=SimpleNamespace(autorelease_pool=self.pool)), \
             mock.patch.object(gainmap, "_nsnumber_bool", side_effect=bool):
            yield


class ReadbackPoolTests(unittest.TestCase):
    def test_drained_reads_keep_independent_python_bitmaps_and_existing_array_contract(self):
        for reader, borrowed in (("hdr", False), ("sdr", True), ("sdr", False)):
            with self.subTest(reader=reader, borrowed=borrowed):
                harness = _ReadbackHarness()
                with harness.modules():
                    if reader == "hdr":
                        first = gainmap._read_expanded_hdr_rgba_half(Path("one.heic"))
                        second = gainmap._read_expanded_hdr_rgba_half(Path("two.heic"))
                    else:
                        first = gainmap.read_primary_rgb_u8(Path("one.heic"), _borrow_rgb=borrowed)
                        second = gainmap.read_primary_rgb_u8(Path("two.heic"), _borrow_rgb=borrowed)
                self.assertEqual(harness.events, ["enter", "drain", "enter", "drain"])
                self.assertFalse(harness.active)
                self.assertFalse(np.shares_memory(first, second))
                gc.collect()
                self.assertTrue(all(ref() is None for ref in harness.native_refs))
                expected = harness.expected
                if reader == "sdr":
                    expected = [values[..., :3] for values in expected]
                np.testing.assert_array_equal(first, expected[0])
                np.testing.assert_array_equal(second, expected[1])
                if reader == "hdr" or borrowed:
                    self.assertFalse(first.flags.writeable)
                    with self.assertRaises(ValueError):
                        first.flags.writeable = True
                else:
                    self.assertTrue(first.flags.writeable)
                self.assertEqual(first.flags.c_contiguous, reader == "hdr" or not borrowed)
                if borrowed:
                    self.assertEqual(first.strides, (12, 4, 1))
                    self.assertTrue(np.shares_memory(first, harness.bitmap_refs[0]()))
                    self.assertTrue(np.shares_memory(second, harness.bitmap_refs[1]()))
                elif reader == "sdr":
                    self.assertTrue(all(ref() is None for ref in harness.bitmap_refs))

    def test_read_exceptions_drain_native_temporaries_without_publishing_a_bitmap(self):
        for reader, failures in (("hdr", ("image", "gainmap", "composite", "dimensions", "context", "render")),
                                 ("sdr", ("image", "dimensions", "context", "render"))):
            for failure in failures:
                with self.subTest(reader=reader, failure=failure):
                    harness = _ReadbackHarness(failure)
                    with harness.modules(), self.assertRaises(RuntimeError):
                        if reader == "hdr":
                            gainmap._read_expanded_hdr_rgba_half(Path("bad.heic"))
                        else:
                            gainmap.read_primary_rgb_u8(Path("bad.heic"), _borrow_rgb=True)
                    self.assertEqual(harness.events, ["enter", "drain"])
                    self.assertFalse(harness.active)
                    gc.collect()
                    self.assertTrue(all(ref() is None for ref in harness.native_refs))

    def test_invalid_primary_gamut_also_drains_the_read_pool(self):
        harness = _ReadbackHarness()
        with harness.modules(), self.assertRaisesRegex(ValueError, "readback gamut"):
            gainmap.read_primary_rgb_u8(Path("bad.heic"), "unknown")
        self.assertEqual(harness.events, ["enter", "drain"])

    def test_pillow_jpeg_path_does_not_need_objc_or_a_native_pool(self):
        pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        image = mock.MagicMock()
        image.__enter__.return_value = image
        image.convert.return_value = pixels
        pillow = SimpleNamespace(Image=SimpleNamespace(open=mock.Mock(return_value=image)))
        with mock.patch.dict(sys.modules, PIL=pillow, objc=None, Quartz=None, Foundation=None):
            actual = gainmap.read_primary_rgb_u8(Path("photo.JPG"), _borrow_rgb=True)
        np.testing.assert_array_equal(actual, pixels)
        image.convert.assert_called_once_with("RGB")

    def test_failed_pillow_decode_falls_back_to_a_single_native_pool(self):
        harness = _ReadbackHarness()
        pillow = SimpleNamespace(Image=SimpleNamespace(open=mock.Mock(side_effect=OSError("unsupported"))))
        with harness.modules(), mock.patch.dict(sys.modules, PIL=pillow):
            actual = gainmap.read_primary_rgb_u8(Path("photo.jpeg"))
        self.assertEqual(harness.events, ["enter", "drain"])
        np.testing.assert_array_equal(actual, harness.expected[0][..., :3])


if __name__ == "__main__":
    unittest.main()
