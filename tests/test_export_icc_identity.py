# SPDX-License-Identifier: GPL-3.0-or-later
"""Generated ICC timestamps cannot invalidate an otherwise correct export."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageCms

from dngscan import export, heif_encoder
from dngscan.analysis import analyze
from dngscan.delivery import resolve_delivery_profile
from dngscan.heif_delivery import save_sdr_heif
from dngscan.raw_io import load_raw
from tests.test_general_raw_fallback import write_uncalibrated_dng
from tests import test_sdr_heif, test_sdr_heif_precision


def timestamp_profiles():
    """Real LCMS sRGB profiles differing only by a valid creation timestamp."""
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    first = profile[:34] + struct.pack(">H", 1) + profile[36:]
    second = profile[:34] + struct.pack(">H", 2) + profile[36:]
    return first, second


class ExportIccIdentityTests(unittest.TestCase):
    def test_real_dng_manual_and_auto_jpeg_resolve_one_exact_profile(self):
        first, second = timestamp_profiles()
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(Path(directory) / "profiles")):
            source = Path(directory) / "sensor.dng"
            write_uncalibrated_dng(source)
            bundle = load_raw(source, scene_half_size=True)
            analysis, _, _ = analyze(bundle, 4, diagnostics=False)
            for name in ("share", "auto"):
                out = Path(directory) / (name + ".jpg")
                profile = resolve_delivery_profile(name)
                with self.subTest(profile=name), \
                     patch.object(export, "output_icc_profile_bytes",
                                  side_effect=[first, second]) as lookup, \
                     patch.object(export, "carry_capture_metadata", return_value=False):
                    export.export_srgb_jpeg(source, out, 95, bundle, analysis, delivery=profile)
                    lookup.assert_called_once_with("srgb")
                with Image.open(out) as image:
                    image.load()
                    self.assertEqual(image.info["icc_profile"], first)
                    self.assertEqual(image.size, bundle.scene_rec2020_render.shape[1::-1])

    def test_strict_profile_identity_rejects_a_changed_metadata_profile(self):
        first, second = timestamp_profiles()
        rgb = np.full((64, 64, 3), 128, np.uint8)

        def corrupt_profile(_source, candidate):
            payload = candidate.read_bytes()
            self.assertIn(first, payload)
            candidate.write_bytes(payload.replace(first, second))
            return True

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "existing.jpg"
            out.write_bytes(b"previous verified image")
            with patch.object(export, "render_output_u8", return_value=rgb), \
                 patch.object(export, "output_icc_profile_bytes", return_value=first), \
                 patch.object(export, "carry_capture_metadata", side_effect=corrupt_profile):
                with self.assertRaisesRegex(RuntimeError, "ICC"):
                    export.export_srgb_jpeg(Path("source.dng"), out, 95, None, None)
            self.assertEqual(out.read_bytes(), b"previous verified image")
            self.assertEqual(list(Path(directory).iterdir()), [out])

    def test_sdr_heif_search_and_final_verification_share_one_profile(self):
        # Container I/O is stubbed, while the delivery and strict ICC gates run.
        # Encoding must receive the same reference used after metadata rewriting.
        for name in ("share", "auto"):
            with self.subTest(profile=name), tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "existing.heic"
                out.write_bytes(b"previous")
                rgb = np.full((8, 8, 3), .4, np.float32)
                fixture = test_sdr_heif_precision.SdrHeifFloatDeliveryTests()
                profile = replace(resolve_delivery_profile(name, container="heic"),
                                  heif_encoder="x265")
                with fixture.fixture(out, rgb) as (calls, reads), \
                     patch("dngscan.color.output_icc_profile_bytes",
                           side_effect=[b"precision-icc", b"new-timestamp-icc"]) as lookup:
                    save_sdr_heif(rgb, out, profile, source_raw=Path("source.dng"))
                    lookup.assert_called_once_with("srgb")
                    self.assertTrue(calls)
                    self.assertTrue(all(options["icc_profile"] == b"precision-icc"
                                        for _, options in calls))
                    self.assertTrue(reads[-1])

    def test_eight_bit_heif_uses_the_same_export_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "existing.heic"
            out.write_bytes(b"previous")
            fixture = test_sdr_heif.SdrHeifFinalReadbackTests()
            profile = replace(resolve_delivery_profile("share", container="heic"),
                              heif_encoder="x265", heif_bit_depth=8)
            with fixture.codec_fixture(out, bit_depth=8) as (rgb, _, _), \
                 patch("dngscan.color.output_icc_profile_bytes",
                       side_effect=[b"test-icc", b"new-timestamp-icc"]) as lookup:
                save_sdr_heif(rgb, out, profile, source_raw=Path("source.dng"))
                lookup.assert_called_once_with("srgb")

    @unittest.skipUnless(heif_encoder.available(), "libheif/x265 unavailable")
    def test_real_heif_encoder_embeds_explicit_profile_without_regeneration(self):
        from dngscan.heif_gainmap import _parse

        first, _ = timestamp_profiles()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "encoded.heic"
            with patch("dngscan.color.output_icc_profile_bytes",
                       side_effect=AssertionError("must use export reference")):
                heif_encoder.encode(np.full((32, 32, 3), 128, np.uint8), out, 95,
                                    output_gamut="srgb", preset="fast", icc_profile=first)
            _, _, primary, _, _, props, assocs, _ = _parse(out.read_bytes())
            profiles = [props[i - 1][1][4:] for _, i in assocs.get(primary, [])
                        if props[i - 1][0] == b"colr"
                        and props[i - 1][1][:4] in (b"prof", b"rICC")]
            self.assertEqual(profiles, [first])


if __name__ == "__main__":
    unittest.main()
