# SPDX-License-Identifier: GPL-3.0-or-later
"""Float SDR readback retains code levels and drains native temporary owners."""
from contextlib import contextmanager
import gc
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import numpy as np

from dngscan import gainmap
from tests.test_gainmap_readback_pool import _NativeObject, _ReadbackHarness


class _NativeOptions(dict):
    pass


class _FloatImage(_NativeObject):
    def extent(self):
        self.assert_live()
        return SimpleNamespace(size=SimpleNamespace(
            width=0 if self.harness.failure == "dimensions" else 512, height=2))


class _FloatContext(_NativeObject):
    def render_toBitmap_rowBytes_bounds_format_colorSpace_(self, image, bitmap, row_bytes,
                                                          extent, pixel_format, color):
        self.assert_live()
        image.assert_live()
        color.assert_live()
        if self.harness.failure == "render":
            raise RuntimeError("injected float bitmap render failure")
        if pixel_format != "RGBAf" or bitmap.dtype != np.float32 or row_bytes != 512 * 16:
            raise AssertionError("float SDR readback lost precision or row stride")
        self.harness.renders += 1
        ramp = np.linspace(0., 1., 1024, dtype=np.float32).reshape(2, 512)
        expected = np.ones((2, 512, 4), np.float32)
        expected[..., :3] = ramp[..., None] + np.float32(self.harness.renders * 1e-6)
        bitmap[...] = expected
        self.harness.bitmap_refs.append(weakref.ref(bitmap))
        self.harness.expected.append(expected)


class _FloatHarness(_ReadbackHarness):
    def __init__(self, failure=None):
        super().__init__(failure)
        self.options = []
        self.context_options = []
        self.gamuts = []

    def primary(self, _url):
        self.assert_active()
        return None if self.failure == "image" else _FloatImage(self)

    def primary_options(self, url, options):
        if not isinstance(options, _NativeOptions):
            raise AssertionError("URL loader requires a native NSDictionary")
        self.options.append(dict(options))
        return self.primary(url)

    def context(self, options):
        self.assert_active()
        self.context_options.append(dict(options))
        return None if self.failure == "context" else _FloatContext(self)

    def color(self, name):
        self.gamuts.append(name)
        return _NativeObject(self)

    @contextmanager
    def modules(self, *, expand_option=True):
        quartz = SimpleNamespace(
            CGColorSpaceCreateWithName=self.color,
            kCGColorSpaceDisplayP3="nonlinear-p3", kCGColorSpaceSRGB="nonlinear-srgb",
            kCIContextCacheIntermediates="cache", kCIContextWorkingFormat="working-format",
            kCIFormatRGBAf="RGBAf",
            CIImage=SimpleNamespace(imageWithContentsOfURL_=self.primary,
                                    imageWithContentsOfURL_options_=self.primary_options),
            CIContext=SimpleNamespace(contextWithOptions_=self.context))
        if expand_option:
            quartz.kCIImageExpandToHDR = "expand-to-HDR"
        foundation = SimpleNamespace(NSURL=SimpleNamespace(fileURLWithPath_=lambda value: value),
                                     NSDictionary=SimpleNamespace(dictionaryWithDictionary_=_NativeOptions))
        with mock.patch.dict(sys.modules, Quartz=quartz, Foundation=foundation,
                             objc=SimpleNamespace(autorelease_pool=self.pool)), \
             mock.patch.object(gainmap, "_nsnumber_bool", side_effect=bool):
            yield


class SdrFloatReadbackTests(unittest.TestCase):
    def test_float_levels_target_transfer_and_ownership_survive_pool_drain(self):
        for borrowed in (False, True):
            for gamut in ("srgb", "p3"):
                with self.subTest(borrowed=borrowed, gamut=gamut):
                    harness = _FloatHarness()
                    with harness.modules():
                        first = gainmap.read_primary_rgb_float(Path("one.heic"), gamut, _borrow_rgb=borrowed)
                        second = gainmap.read_primary_rgb_float(Path("two.heic"), gamut, _borrow_rgb=borrowed)
                    self.assertEqual(harness.events, ["enter", "drain", "enter", "drain"])
                    self.assertEqual(harness.options, [{"expand-to-HDR": False}] * 2)
                    self.assertEqual(harness.context_options, [{"cache": False, "working-format": "RGBAf"}] * 2)
                    self.assertEqual(harness.gamuts, ["nonlinear-" + gamut] * 2)
                    self.assertEqual(first.dtype, np.float32)
                    self.assertEqual(np.unique(first[..., 0]).size, 1024)
                    self.assertFalse(np.shares_memory(first, second))
                    self.assertEqual(first.flags.c_contiguous, not borrowed)
                    self.assertEqual(first.flags.writeable, not borrowed)
                    np.testing.assert_array_equal(first, harness.expected[0][..., :3])
                    np.testing.assert_array_equal(second, harness.expected[1][..., :3])
                    if borrowed:
                        self.assertEqual(first.strides, (512 * 16, 16, 4))
                        with self.assertRaises(ValueError):
                            first.flags.writeable = True
                        self.assertTrue(np.shares_memory(first, harness.bitmap_refs[0]()))
                    gc.collect()
                    self.assertTrue(all(ref() is None for ref in harness.native_refs))

    def test_old_api_retains_default_sdr_loading_without_expansion(self):
        harness = _FloatHarness()
        with harness.modules(expand_option=False):
            actual = gainmap.read_primary_rgb_float(Path("old-api.heic"))
        self.assertFalse(harness.options)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(harness.events, ["enter", "drain"])

    def test_errors_drain_native_temporaries_and_never_publish_partial_pixels(self):
        for failure in ("image", "dimensions", "context", "render"):
            with self.subTest(failure=failure):
                harness = _FloatHarness(failure)
                with harness.modules(), self.assertRaises(RuntimeError):
                    gainmap.read_primary_rgb_float(Path("bad.heic"), _borrow_rgb=True)
                self.assertEqual(harness.events, ["enter", "drain"])
                gc.collect()
                self.assertTrue(all(ref() is None for ref in harness.native_refs))
        harness = _FloatHarness()
        with harness.modules(), self.assertRaisesRegex(ValueError, "readback gamut"):
            gainmap.read_primary_rgb_float(Path("bad.heic"), "unknown")
        self.assertEqual(harness.events, ["enter", "drain"])


@unittest.skipUnless(platform.system() == "Darwin", "real Core Image readback requires macOS")
class SdrFloatActualReadbackTests(unittest.TestCase):
    def test_real_png16_preserves_levels_nonlinear_transfer_and_target_gamut(self):
        try:
            import objc
            import Quartz
            from Foundation import NSData, NSURL
        except ImportError as exc:
            self.skipTest(f"Core Image bindings unavailable: {exc}")
        from dngscan.color import srgb_decode, srgb_encode, srgb_to_output

        with TemporaryDirectory() as directory, objc.autorelease_pool():
            rgba = np.ones((16, 1024, 4), np.float32)
            rgba[..., 0] = np.linspace(0., 1., 1024, dtype=np.float32)
            rgba[..., 1], rgba[..., 2] = .25, .5
            color = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB)
            data = NSData.dataWithBytes_length_(rgba.tobytes(), rgba.nbytes)
            image = Quartz.CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
                data, int(rgba.strides[0]), (1024, 16), Quartz.kCIFormatRGBAf, color)
            context = Quartz.CIContext.contextWithOptions_({
                Quartz.kCIContextCacheIntermediates: False,
                Quartz.kCIContextWorkingFormat: Quartz.kCIFormatRGBAf})
            if context is None:
                self.skipTest("Core Image context unavailable for this workload")
            path = Path(directory) / "nonlinear-srgb.png"
            ok, error = context.writePNGRepresentationOfImage_toURL_format_colorSpace_options_error_(
                image, NSURL.fileURLWithPath_(str(path)), Quartz.kCIFormatRGBA16, color, {}, None)
            self.assertTrue(ok, error)
            srgb = gainmap.read_primary_rgb_float(path, "srgb", _borrow_rgb=True)
            self.assertGreater(np.unique(srgb[..., 0]).size, 900)
            np.testing.assert_allclose(srgb, rgba[..., :3], atol=3e-5, rtol=0.)
            p3 = gainmap.read_primary_rgb_float(path, "p3", _borrow_rgb=True)
            expected = srgb_encode(srgb_to_output(
                srgb_decode(rgba[..., :3]).reshape(-1, 3), "p3")).reshape(p3.shape)
            # ColorSync uses profile transforms; project matrices are rounded.
            np.testing.assert_allclose(p3, expected, atol=.001, rtol=0.)

if __name__ == '__main__':
    unittest.main()
