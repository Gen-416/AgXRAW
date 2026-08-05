# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dngscan._deps import np
from dngscan.color import srgb_decode
from dngscan.gainmap import (
    apple_gainmap_backend_status,
    inspect_gainmap_jpeg,
    write_apple_gainmap_jpeg,
)
from dngscan.cli import parse_args


class GainMapInterfaceTests(unittest.TestCase):
    def test_cli_ultrahdr_rejects_display_look(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "photo.dng",
                    "--output-format",
                    "ultrahdr",
                    "--grade",
                    "look:optic_warm_cyan",
                ]
            )

    def test_cli_hdr_keeps_fixed_delivery_defaults(self) -> None:
        args = parse_args(
            [
                "photo.dng",
                "--output-format",
                "ultrahdr",
            ]
        )
        self.assertEqual(args.grade, "none")
        self.assertEqual(args.jpeg_quality, 100)
        self.assertEqual(args.chroma, "444")
        self.assertEqual(args.delivery_profile, "archive")
        self.assertEqual(args.hdr_drt, "agx")

    def test_cli_hdr_rejects_non_agx_tone_core(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["photo.dng", "--output-format", "ultrahdr", "--tone-core", "lum"])

    def test_cli_hdr_rejects_sdr_highlight_fade(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                ["photo.dng", "--output-format", "ultrahdr", "--highlight-fade", "0.1"]
            )

class AppleGainMapWriterTests(unittest.TestCase):
    def test_writer_rejects_non_finite_hdr_before_core_image(self) -> None:
        from dngscan.gainmap import write_apple_gainmap_jpeg

        base = np.zeros((2, 2, 3), dtype=np.uint8)
        hdr = np.ones((2, 2, 4), dtype=np.float16)
        hdr[0, 0, 0] = np.float16(np.nan)
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            write_apple_gainmap_jpeg(
                base,
                hdr,
                Path("unused.jpg"),
                100,
                3.0,
                _verify_roundtrip_capability=False,
            )

    def test_backend_status_reports_api_availability_only(self) -> None:
        """Replaces a gate that asserted HDR stays paused, which the AgX core now lifts.

        Correctness deliberately moved out of this function. A capability probe can only
        answer for its own test pattern -- the synthetic RGB probe uses 2.4x gain ratios
        between channels and fails, while real renditions round-trip two orders of
        magnitude better -- so every write is verified against its own pixels instead.
        """
        available, reason = apple_gainmap_backend_status()
        self.assertIsInstance(available, bool)
        self.assertTrue(reason)
        if available:
            # On a capable system this must be the API check, not the synthetic probe.
            from dngscan.gainmap import _apple_gainmap_api_status

            self.assertEqual((available, reason), _apple_gainmap_api_status())

    def test_public_writer_rejects_unverified_backend(self) -> None:
        base = np.full((4, 4, 3), 128, dtype=np.uint8)
        hdr = np.ones((4, 4, 4), dtype=np.float16)
        with mock.patch(
            "dngscan.gainmap.apple_gainmap_backend_status",
            return_value=(False, "round-trip not verified"),
        ):
            with self.assertRaisesRegex(RuntimeError, "round-trip not verified"):
                write_apple_gainmap_jpeg(base, hdr, Path("unused.jpg"), 100, 3.0)

    def test_declared_headroom_tracks_actual_rendition_peak(self) -> None:
        available, reason = apple_gainmap_backend_status()
        if not available:
            self.skipTest(reason)

        h, w = 16, 32
        ramp = np.linspace(64, 255, w, dtype=np.uint8)
        base = np.repeat(ramp[None, :, None], h, axis=0)
        base = np.repeat(base, 3, axis=2)
        linear = srgb_decode(base.astype(np.float32) / 255.0)
        gain = np.linspace(1.0, 2.0, w, dtype=np.float32)[None, :, None]
        hdr = np.empty((h, w, 4), dtype=np.float16)
        hdr[:, :, :3] = (linear * gain).astype(np.float16)
        hdr[:, :, 3] = np.float16(1.0)

        with tempfile.TemporaryDirectory() as td:
            info = write_apple_gainmap_jpeg(
                base, hdr, Path(td) / "scene_headroom.jpg", 100, 3.0
            )
            self.assertAlmostEqual(info["headroom"], 2.0, delta=0.02)

    def test_writes_iso_gainmap_display_p3_jpeg(self) -> None:
        available, reason = apple_gainmap_backend_status()
        if not available:
            self.skipTest(reason)

        h, w = 32, 64
        ramp = np.linspace(0, 255, w, dtype=np.uint8)
        base = np.empty((h, w, 3), dtype=np.uint8)
        base[:, :, 0] = ramp[None, :]
        base[:, :, 1] = np.minimum(ramp[None, :], 220)
        base[:, :, 2] = np.minimum(ramp[None, :], 180)
        linear = srgb_decode(base.astype(np.float32) / 255.0)
        gain = np.exp2(np.linspace(0.0, 3.0, w, dtype=np.float32))[None, :, None]
        hdr = np.empty((h, w, 4), dtype=np.float16)
        hdr[:, :, :3] = np.clip(linear * gain, 0.0, 8.0).astype(np.float16)
        hdr[:, :, 3] = np.float16(1.0)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "native_iso_gainmap.jpg"
            written = write_apple_gainmap_jpeg(base, hdr, path, 100, 3.0)
            inspected = inspect_gainmap_jpeg(path)
            from PIL import Image

            with Image.open(path) as image:
                decoded_primary = np.asarray(image.convert("RGB"), dtype=np.uint8)
            self.assertTrue(path.is_file())
            self.assertTrue(written["has_iso_gainmap"])
            self.assertTrue(inspected["has_iso_gainmap"])
            self.assertEqual(inspected["profile"], "Display P3")
            self.assertEqual(inspected["chroma_subsampling"], "4:4:4")
            self.assertNotEqual(inspected["gainmap_pixel_format"], "L008")
            self.assertTrue(inspected["gainmap_pixel_format"])
            self.assertEqual(
                (inspected["gainmap_width"], inspected["gainmap_height"]), (w, h)
            )
            self.assertGreater(inspected["headroom"], 1.0)
            self.assertEqual((inspected["width"], inspected["height"]), (w, h))
            primary_error = np.abs(decoded_primary.astype(np.int16) - base.astype(np.int16))
            self.assertLess(float(np.mean(primary_error)), 1.0)


if __name__ == "__main__":
    unittest.main()
