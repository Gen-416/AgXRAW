# SPDX-License-Identifier: GPL-3.0-or-later
"""Delivery profile resolution: encode knobs stay out of formation."""
from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from dngscan.cli import parse_args
from dngscan.color import srgb_decode
from dngscan.delivery import (
    ARCHIVE_JPEG_QUALITY,
    SHARE_HEIC_TOLERANCES,
    SHARE_JPEG_QUALITY,
    SHARE_TOLERANCES,
    DeliveryProfile,
    reprofile_for_container,
    resolve_delivery_profile,
)
from dngscan.gainmap import apple_gainmap_backend_status, write_apple_gainmap_jpeg
from dngscan._deps import np


class DeliveryProfileTests(unittest.TestCase):
    def test_archive_defaults(self) -> None:
        profile = resolve_delivery_profile("archive")
        self.assertEqual(profile.quality, ARCHIVE_JPEG_QUALITY)
        self.assertEqual(profile.chroma, "444")
        self.assertTrue(profile.tolerances.require_chroma_444)

    def test_share_defaults(self) -> None:
        profile = resolve_delivery_profile("share")
        self.assertEqual(profile.quality, SHARE_JPEG_QUALITY)
        self.assertEqual(profile.chroma, "420")
        self.assertFalse(profile.tolerances.require_chroma_444)

    def test_archive_rejects_soft_quality(self) -> None:
        with self.assertRaisesRegex(ValueError, "share"):
            resolve_delivery_profile("archive", quality=90)

    def test_cli_ultrahdr_archive_defaults(self) -> None:
        args = parse_args(["photo.dng", "--output-format", "ultrahdr"])
        self.assertEqual(args.delivery_profile, "archive")
        self.assertEqual(args.jpeg_quality, 100)
        self.assertEqual(args.chroma, "444")
        self.assertEqual(args.delivery.name, "archive")

    def test_cli_ultrahdr_share_defaults(self) -> None:
        args = parse_args(
            ["photo.dng", "--output-format", "ultrahdr", "--delivery-profile", "share"]
        )
        self.assertEqual(args.jpeg_quality, 90)
        self.assertEqual(args.chroma, "420")
        self.assertEqual(args.delivery.name, "share")

    def test_cli_archive_rejects_soft_quality_flag(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "photo.dng",
                    "--output-format",
                    "ultrahdr",
                    "--delivery-profile",
                    "archive",
                    "--jpeg-quality",
                    "90",
                ]
            )


    def test_cli_ultrahdr_heic_sets_heic_container(self) -> None:
        args = parse_args(
            ["photo.dng", "--output-format", "ultrahdr-heic", "--jpeg", "out.jpg"]
        )
        self.assertEqual(args.delivery.container, "heic")
        self.assertEqual(args.jpeg_quality, 100)

    def test_cli_explicit_knobs_without_profile_are_honoured(self) -> None:
        """Pre-profile invocations keep working; gates are inferred, not vetoed."""
        args = parse_args(["photo.dng", "--jpeg", "out.jpg", "--jpeg-quality", "85"])
        self.assertEqual(args.jpeg_quality, 85)
        self.assertEqual(args.delivery.name, "share")
        args = parse_args(
            ["photo.dng", "--jpeg", "out.jpg",
             "--jpeg-quality", "90", "--chroma", "420"]
        )
        self.assertEqual((args.jpeg_quality, args.chroma), (90, "420"))
        self.assertEqual(args.delivery.name, "share")
        args = parse_args(["photo.dng", "--jpeg", "out.jpg"])
        self.assertEqual(args.delivery.name, "archive")
        self.assertEqual(args.jpeg_quality, 100)

    def test_cli_hdr_rejects_unhonourable_chroma(self) -> None:
        for extra in (
            ["--chroma", "422"],
            ["--delivery-profile", "share", "--chroma", "444"],
        ):
            with self.subTest(extra=extra):
                with self.assertRaises(SystemExit):
                    parse_args(
                        ["photo.dng", "--output-format", "ultrahdr", *extra]
                    )

    def test_share_heic_gets_its_own_calibration(self) -> None:
        jpeg = resolve_delivery_profile("share", container="jpeg")
        heic = resolve_delivery_profile("share", container="heic")
        self.assertIs(jpeg.tolerances, SHARE_TOLERANCES)
        self.assertIs(heic.tolerances, SHARE_HEIC_TOLERANCES)
        self.assertGreater(
            heic.tolerances.hdr_block_p95_relative_error,
            jpeg.tolerances.hdr_block_p95_relative_error,
        )

    def test_reprofile_for_container_rederives_standard_gates(self) -> None:
        moved = reprofile_for_container(
            resolve_delivery_profile("share", container="jpeg"), "heic"
        )
        self.assertIs(moved.tolerances, SHARE_HEIC_TOLERANCES)
        custom = DeliveryProfile(
            name="share", quality=90, chroma="420", container="jpeg",
            tolerances=dataclasses.replace(
                SHARE_TOLERANCES, hdr_block_median_relative_error=0.5
            ),
        )
        moved_custom = reprofile_for_container(custom, "heic")
        self.assertEqual(
            moved_custom.tolerances.hdr_block_median_relative_error, 0.5
        )

    def test_pixel_chroma_gate_is_wired_per_profile(self) -> None:
        """Block means average over the 4:2:0 grid; only the pixel gate sees that loss."""
        for name, container in (
            ("archive", "jpeg"), ("share", "jpeg"), ("share", "heic"),
        ):
            with self.subTest(profile=name, container=container):
                tol = resolve_delivery_profile(name, container=container).tolerances
                self.assertGreater(tol.hdr_pixel_chroma_error, 0.0)
                self.assertTrue(
                    tol.hdr_pixel_chroma_error < float("inf")
                )


class ShareEncodeLiveTests(unittest.TestCase):
    def test_share_profile_writes_iso_gainmap(self) -> None:
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
        profile = resolve_delivery_profile("share")

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "share_iso_gainmap.jpg"
            written = write_apple_gainmap_jpeg(
                base,
                hdr,
                path,
                profile.quality,
                3.0,
                delivery=profile,
                chroma=profile.chroma,
            )
            self.assertTrue(path.is_file())
            self.assertTrue(written["has_iso_gainmap"])
            self.assertEqual(written["delivery_profile"], "share")
            self.assertEqual(written["chroma_subsampling"], "4:2:0")
            self.assertNotEqual(written["gainmap_pixel_format"], "L008")
            archive = write_apple_gainmap_jpeg(
                base,
                hdr,
                Path(td) / "archive.jpg",
                100,
                3.0,
                delivery=resolve_delivery_profile("archive"),
            )
            self.assertLess(path.stat().st_size, Path(td, "archive.jpg").stat().st_size)
            self.assertEqual(archive["chroma_subsampling"], "4:4:4")

    def test_heic_archive_writes_iso_gainmap(self) -> None:
        from dngscan.gainmap import write_apple_gainmap_heic

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
        profile = resolve_delivery_profile("archive", container="heic")

        with tempfile.TemporaryDirectory() as td:
            heic_path = Path(td) / "archive.heic"
            jpeg_path = Path(td) / "archive.jpg"
            heic = write_apple_gainmap_heic(
                base, hdr, heic_path, 100, 3.0, delivery=profile
            )
            jpeg = write_apple_gainmap_jpeg(
                base,
                hdr,
                jpeg_path,
                100,
                3.0,
                delivery=resolve_delivery_profile("archive", container="jpeg"),
            )
            self.assertTrue(heic["has_iso_gainmap"])
            self.assertEqual(heic["delivery_container"], "heic")
            self.assertEqual(heic["chroma_subsampling"], "4:4:4")
            self.assertNotEqual(heic["gainmap_pixel_format"], "L008")
            self.assertLess(heic_path.stat().st_size, jpeg_path.stat().st_size)
            self.assertEqual(jpeg["delivery_container"], "jpeg")


if __name__ == "__main__":
    unittest.main()
