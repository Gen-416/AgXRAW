# SPDX-License-Identifier: GPL-3.0-or-later
"""Actual gain-map HEIF preserves floating SDR levels through packaging."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dngscan.color import srgb_decode
from dngscan.delivery import resolve_delivery_profile


class HdrHeifFloatLiveTests(unittest.TestCase):
    def test_apple_only_10bit_never_silently_writes_an_8bit_base(self):
        from dngscan.gainmap import (apple_gainmap_backend_status,
                                    read_primary_rgb_float,
                                    write_apple_gainmap_file)
        if not apple_gainmap_backend_status()[0]:
            self.skipTest("Apple ISO gain-map APIs required")
        try:
            import Quartz
            has_writer = hasattr(Quartz.CIContext,
                "writeHEIF10RepresentationOfImage_toURL_colorSpace_options_error_")
        except ImportError:
            has_writer = False
        if not has_writer:
            self.skipTest("dedicated Apple HEIF10 writer required")
        base = np.broadcast_to(
            np.linspace(.02, .98, 1024, dtype=np.float32)[None, :, None],
            (64, 1024, 3)).copy()
        hdr = np.ones(base.shape[:2] + (4,), dtype=np.float16)
        hdr[..., :3] = (srgb_decode(base) * np.float32(4.)).astype(np.float16)
        profile = replace(resolve_delivery_profile("archive", container="heic"),
                          heif_encoder="apple", heif_bit_depth=10)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "apple.heic"
            info = write_apple_gainmap_file(base, hdr, path, 2., delivery=profile)
            decoded = read_primary_rgb_float(path, "p3")
            self.assertEqual(info["bit_depth"], 10)
            self.assertEqual(info["readback_precision"], "float32")
            self.assertGreater(np.unique(decoded[..., 0]).size, 512)
            self.assertTrue(info["has_iso_gainmap"])

    def test_float_base_survives_manual_and_auto_gainmap_packaging(self):
        from dngscan import heif_encoder
        from dngscan.gainmap import (apple_gainmap_backend_status,
                                    read_primary_rgb_float,
                                    write_apple_gainmap_file)

        if not heif_encoder.available() or not apple_gainmap_backend_status()[0]:
            self.skipTest("libheif/x265 and Apple ISO gain-map APIs required")
        # This finished nonlinear P3 ramp has 1024 distinct source levels.
        # A u8 intermediate, even followed by a 10-bit codec, cannot retain it.
        base = np.broadcast_to(
            np.linspace(.02, .98, 1024, dtype=np.float32)[None, :, None],
            (64, 1024, 3)).copy()
        hdr = np.ones(base.shape[:2] + (4,), dtype=np.float16)
        hdr[..., :3] = (srgb_decode(base) * np.float32(4.)).astype(np.float16)
        before_base, before_hdr = base.copy(), hdr.copy()
        with tempfile.TemporaryDirectory() as td:
            for name, encoder in (("archive", "x265"), ("auto", "x265"), ("auto", "apple")):
                with self.subTest(profile=name, encoder=encoder):
                    profile = replace(resolve_delivery_profile(name, container="heic"),
                                      heif_encoder=encoder, heif_bit_depth=10)
                    path = Path(td) / f"{name}-{encoder}.heic"
                    info = write_apple_gainmap_file(
                        base, hdr, path, 2., delivery=profile)
                    decoded = read_primary_rgb_float(path, "p3")
                    self.assertEqual(info["bit_depth"], 10)
                    self.assertTrue(info["has_iso_gainmap"])
                    self.assertEqual(info["readback_precision"], "float32")
                    if encoder == "x265":
                        self.assertEqual(info["quantization_dither"], "TPDF-10bit-seed0")
                    self.assertGreater(np.unique(decoded[..., 0]).size, 512)
                    codes = decoded * np.float32(255.)
                    self.assertGreater(float(np.max(np.abs(codes - np.rint(codes)))), .1)
                    self.assertGreater(info["headroom"], 1.)
                    self.assertLessEqual(info["headroom_error_ev"], .05)
                    self.assertLess(info["median_relative_error"], .05)
        np.testing.assert_array_equal(base, before_base)
        np.testing.assert_array_equal(hdr, before_hdr)


if __name__ == "__main__":
    unittest.main()
