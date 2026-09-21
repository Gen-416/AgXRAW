# SPDX-License-Identifier: GPL-3.0-or-later
"""The fixed sharing preset preserves pixels and reports final delivery size."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stderr
from dataclasses import replace
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from PIL import Image, JpegImagePlugin

from dngscan import export
from dngscan._deps import np
from dngscan.cli import parse_args
from dngscan.color import output_icc_profile_bytes
from dngscan.delivery import (
    SHARE_TOLERANCES,
    delivery_size_report,
    reprofile_for_container,
    resolve_delivery_profile,
    resolve_hdr_chroma,
)


class ShareHqProfileTests(unittest.TestCase):
    def test_cli_accepts_fixed_preset_for_sdr_and_hdr_jpeg(self):
        for output_format in ("sdr", "ultrahdr"):
            for explicit in ([], ["--jpeg-quality", "97", "--chroma", "420"]):
                with self.subTest(output_format=output_format, explicit=explicit):
                    args = parse_args([
                        "capture.dng", "--output-format", output_format,
                        "--delivery-profile", "share-hq", *explicit,
                    ])
                    self.assertEqual(args.delivery_profile, "share-hq")
                    self.assertEqual((args.jpeg_quality, args.chroma), (97, "420"))
                    self.assertEqual(args.delivery.container, "jpeg")
                    self.assertEqual(args.delivery.tolerances, SHARE_TOLERANCES)

    def test_conflicting_overrides_are_rejected_instead_of_silently_relabelled(self):
        for knobs, flags in (
            ({"quality": 95}, ["--jpeg-quality", "95"]),
            ({"quality": 100}, ["--jpeg-quality", "100"]),
            ({"chroma": "422"}, ["--chroma", "422"]),
            ({"chroma": "444"}, ["--chroma", "444"]),
        ):
            with self.subTest(knobs=knobs):
                with self.assertRaisesRegex(ValueError, "share-hq.*97"):
                    resolve_delivery_profile("share-hq", **knobs)
                errors = StringIO()
                with redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
                    parse_args(["capture.dng", "--delivery-profile", "share-hq", *flags])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("share-hq", errors.getvalue())

    def test_heif_cannot_reinterpret_the_jpeg_quality_scale(self):
        profile = resolve_delivery_profile("share-hq")
        for operation in (
            lambda: resolve_delivery_profile("share-hq", container="heic"),
            lambda: reprofile_for_container(profile, "heic"),
            lambda: replace(profile, container="heic"),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(ValueError, "share-hq.*JPEG"):
                    operation()
        for output_format in ("sdr-heic", "ultrahdr-heic"):
            with self.subTest(output_format=output_format):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
                    parse_args(["capture.dng", "--output-format", output_format,
                                "--delivery-profile", "share-hq"])
                self.assertEqual(caught.exception.code, 2)

    def test_profile_replacement_and_hdr_sampling_cannot_break_fixed_contract(self):
        profile = resolve_delivery_profile("share-hq")
        for operation in (
            lambda: replace(profile, quality=99),
            lambda: replace(profile, chroma="444"),
            lambda: resolve_hdr_chroma(profile, explicit_chroma="422"),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(ValueError, "share-hq.*97"):
                    operation()
        self.assertEqual(resolve_hdr_chroma(profile, explicit_chroma="420"), profile)

    def test_existing_default_and_manual_profiles_keep_their_behavior(self):
        for output_format in ("sdr", "ultrahdr", "sdr-heic", "ultrahdr-heic"):
            with self.subTest(output_format=output_format):
                self.assertEqual(parse_args([
                    "capture.dng", "--output-format", output_format,
                ]).delivery_profile, "auto")
        manual = parse_args(["capture.dng", "--jpeg-quality", "97", "--chroma", "420"])
        self.assertEqual(manual.delivery_profile, "share")
        self.assertEqual(resolve_delivery_profile("share").quality, 95)

    def test_twenty_decimal_megabytes_is_an_advisory_boundary(self):
        profile = resolve_delivery_profile("share-hq")
        for size, exceeds in ((19_999_999, False), (20_000_000, False), (20_000_001, True)):
            with self.subTest(size=size):
                report = delivery_size_report(profile, size)
                self.assertEqual(report["file_size_bytes"], size)
                self.assertEqual(report["share_size_limit_bytes"], 20_000_000)
                self.assertIs(report["share_size_exceeded"], exceeds)
                self.assertEqual("size_warning" in report, exceeds)
                if exceeds:
                    self.assertIn("20 MB", report["size_warning"])
                    self.assertIn("原尺寸", report["size_warning"])
        for name in ("auto", "archive", "share"):
            with self.subTest(name=name):
                report = delivery_size_report(resolve_delivery_profile(name), 21_000_000)
                self.assertEqual(report, {"file_size_bytes": 21_000_000})


class ShareHqJpegExportTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.random.default_rng(97).integers(0, 256, (37, 59, 3), dtype=np.uint8)
        self.profile = resolve_delivery_profile("share-hq")

    def test_real_jpeg_is_q97_420_with_original_dimensions_even_with_legacy_arguments(self):
        reference = BytesIO()
        Image.fromarray(self.rgb).save(reference, format="JPEG", quality=97, subsampling=2)
        reference.seek(0)
        with Image.open(reference) as image:
            expected_quantization = image.quantization

        for quality, subsampling in ((97, 2), (100, 0)):
            for return_rgb in (False, True):
                with self.subTest(quality=quality, subsampling=subsampling, return_rgb=return_rgb):
                    with tempfile.TemporaryDirectory() as directory:
                        target = Path(directory) / "photo.jpg"
                        with mock.patch.object(export, "render_output_u8", return_value=self.rgb), \
                             mock.patch.object(export, "carry_capture_metadata", return_value=False):
                            info = export.export_srgb_jpeg(
                                Path("capture.dng"), target, quality, None, None,
                                subsampling=subsampling, delivery=self.profile, return_rgb=return_rgb,
                            )
                        self.assertIsInstance(info, dict)
                        self.assertEqual(info["delivery_profile"], "share-hq")
                        self.assertEqual(info["delivery_quality"], 97)
                        self.assertEqual(info["delivery_chroma_requested"], "420")
                        self.assertEqual(info["chroma_subsampling"], "4:2:0")
                        self.assertEqual(info["file_size_bytes"], target.stat().st_size)
                        self.assertEqual(info["output_path"], str(target))
                        self.assertFalse(info["exif_carried"])
                        self.assertFalse(info["share_size_exceeded"])
                        with Image.open(target) as image:
                            image.load()
                            self.assertEqual(image.size, (59, 37))
                            self.assertEqual(JpegImagePlugin.get_sampling(image), 2)
                            self.assertEqual(image.quantization, expected_quantization)
                            self.assertEqual(image.info["icc_profile"], output_icc_profile_bytes("srgb"))
                            if return_rgb:
                                np.testing.assert_array_equal(info["_decoded_rgb"], np.asarray(image))
                            else:
                                self.assertNotIn("_decoded_rgb", info)

    def test_metadata_can_cross_size_boundary_without_reencoding_or_resizing(self):
        before_carry = []

        def carry(_source, candidate):
            before_carry.append(candidate.stat().st_size)
            # A sparse trailer simulates metadata size growth without allocating 20 MB.
            # JPEG readers stop at EOI, so the real primary remains decodable.
            with candidate.open("r+b") as file:
                file.truncate(20_000_001)
            return True

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "photo.jpg"
            with mock.patch.object(export, "render_output_u8", return_value=self.rgb), \
                 mock.patch.object(export, "carry_capture_metadata", side_effect=carry), \
                 mock.patch.object(export, "save_jpeg_array", wraps=export.save_jpeg_array) as encode:
                info = export.export_srgb_jpeg(
                    Path("capture.dng"), target, 97, None, None,
                    subsampling=2, delivery=self.profile, return_rgb=True,
                )
            encode.assert_called_once()
            self.assertEqual(len(before_carry), 1)
            self.assertLess(before_carry[0], 20_000_000)
            self.assertEqual(target.stat().st_size, 20_000_001)
            self.assertEqual(info["file_size_bytes"], target.stat().st_size)
            self.assertTrue(info["exif_carried"])
            self.assertTrue(info["share_size_exceeded"])
            self.assertIn("size_warning", info)
            self.assertEqual(info["_decoded_rgb"].shape, self.rgb.shape)
            self.assertEqual(info["delivery_quality"], 97)

    def test_existing_share_export_preserves_boolean_and_tuple_return_types(self):
        for return_rgb in (False, True):
            with self.subTest(return_rgb=return_rgb), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "photo.jpg"
                with mock.patch.object(export, "render_output_u8", return_value=self.rgb), \
                     mock.patch.object(export, "carry_capture_metadata", return_value=False):
                    result = export.export_srgb_jpeg(
                        Path("capture.dng"), target, 95, None, None, subsampling=2,
                        delivery=resolve_delivery_profile("share"), return_rgb=return_rgb,
                    )
                if return_rgb:
                    self.assertIsInstance(result, tuple)
                    self.assertIs(result[0], True)
                    with Image.open(target) as image:
                        np.testing.assert_array_equal(result[1], np.asarray(image))
                else:
                    self.assertIs(result, True)

    def test_hdr_size_notice_counts_the_completed_container_after_metadata(self):
        from tests.test_hdr_native import _scene_plan

        tone = SimpleNamespace(
            rendered_headroom_ev=2., peak_linear=4., display_headroom_ev=2.,
            requested_headroom_ev=2., reliable_tail_ev=3., shoulder_start_ev=1.,
            white_ev=2., shoulder_alpha=.5, shoulder_segments=(),
        )
        hdr_plan = SimpleNamespace(tone=tone, color=SimpleNamespace(channel_separation=.5))
        hdr = np.ones((*self.rgb.shape[:2], 4), dtype=np.float16)

        def encode(pair, candidate, profile):
            self.assertEqual((profile.name, profile.quality, profile.chroma), ("share-hq", 97, "420"))
            self.assertIs(pair.sdr_rgb_u8, self.rgb)
            candidate.write_bytes(b"base plus auxiliary and container metadata")
            return {"delivery_profile": profile.name, "delivery_quality": profile.quality,
                    "file_size_bytes": candidate.stat().st_size}

        def carry(_source, candidate, container):
            self.assertEqual(container, "jpeg")
            with candidate.open("r+b") as file:
                file.truncate(20_000_001)
            return True

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            target = Path(directory) / "hdr.jpg"
            stack.enter_context(mock.patch.object(export, "apple_gainmap_backend_status", return_value=(True, "")))
            stack.enter_context(mock.patch("dngscan.hdr_agx_plan.compile_hdr_agx_plan", return_value=hdr_plan))
            stack.enter_context(mock.patch("dngscan.hdr_agx_plan.describe_hdr_plan", return_value="test plan"))
            stack.enter_context(mock.patch("dngscan.hdr_agx.render_ultrahdr_agx_pair_packed",
                                           return_value=(self.rgb, hdr, 1.)))
            encode_mock = stack.enter_context(mock.patch.object(export, "encode_finished_pair", side_effect=encode))
            stack.enter_context(mock.patch.object(export, "carry_capture_metadata_hdr", side_effect=carry))
            info = export.export_ultrahdr_jpeg(
                Path("capture.dng"), target, 100, SimpleNamespace(scene_decoder="test"), None,
                tone_plan=_scene_plan(), delivery=self.profile,
            )
            encode_mock.assert_called_once()
            self.assertEqual(info["file_size_bytes"], target.stat().st_size)
            self.assertEqual(info["file_size_bytes"], 20_000_001)
            self.assertTrue(info["share_size_exceeded"])
            self.assertIn("size_warning", info)
            self.assertTrue(info["exif_carried"])
            self.assertEqual(info["output_path"], str(target))


if __name__ == "__main__":
    unittest.main()
