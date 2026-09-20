# SPDX-License-Identifier: GPL-3.0-or-later
"""SDR HEIF routing and bounded codec search, including failure atomicity."""
from dataclasses import replace
from contextlib import ExitStack, contextmanager
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from dngscan.auto_encode import select_heif_encoding
from dngscan.cli import parse_args
from dngscan.delivery import container_for_output_format, is_hdr_output_format, resolve_delivery_profile
from dngscan.export import export_jpeg


class SdrHeifRoutingTests(unittest.TestCase):
    def test_gui_keeps_sdr_heif_gamut_and_does_not_require_hdr_capacity(self):
        from dngscan.gui.service import parse_job_params
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "photo.dng"
            path.touch()
            parsed = parse_job_params({"input": str(path), "format": "sdr-heic", "gamut": "srgb",
                                       "hdrHeadroom": 0})
        self.assertEqual(parsed[2:4], ("srgb", "sdr-heic"))
        self.assertEqual(parsed[5], 0.)

    def test_sdr_heif_is_independent_of_hdr_controls_and_capacity(self):
        args = parse_args(["photo.dng", "--output-format", "sdr-heic", "--tone-core", "lum",
                           "--delivery-profile", "share", "--jpeg-quality", "83", "--chroma", "422"])
        self.assertFalse(is_hdr_output_format(args.output_format))
        self.assertEqual(container_for_output_format(args.output_format), "heic")
        with patch("dngscan.export.export_ultrahdr_jpeg", side_effect=AssertionError("HDR path")), \
             patch("dngscan.export.export_srgb_jpeg", return_value={}) as sdr:
            export_jpeg(Path("photo.dng"), Path("out.jpg"), 83, None, None,
                        output_format="sdr-heic", hdr_headroom=0., delivery=args.delivery)
        self.assertEqual(sdr.call_args.args[1], Path("out.heic"))
        profile = sdr.call_args.args[-1]
        self.assertEqual((profile.quality, profile.chroma, profile.container), (83, "422", "heic"))

    def test_requested_sdr_jpeg_cannot_inherit_a_heif_container(self):
        with patch("dngscan.export.export_srgb_jpeg", return_value={}) as sdr:
            export_jpeg(Path("photo.dng"), Path("out.jpg"), 95, None, None,
                        delivery=resolve_delivery_profile("share", container="heic"))
        self.assertEqual(sdr.call_args.args[-1].container, "jpeg")

    def test_apple_does_not_silently_ignore_requested_depth_or_sampling(self):
        from dngscan.heif_delivery import save_sdr_heif
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "photo.heic"
            out.write_bytes(b"previous")
            profile = replace(resolve_delivery_profile("share", container="heic"), heif_encoder="apple")
            with self.assertRaisesRegex(ValueError, "8-bit/4:2:0"):
                save_sdr_heif(np.zeros((8, 8, 3), np.uint8), out, profile)
            self.assertEqual(out.read_bytes(), b"previous")


class HeifSearchTests(unittest.TestCase):
    def test_searches_heif_quality_sampling_and_intermediate_auxiliary_precision(self):
        calls = []
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "photo.heic"
            def encode(q, chroma, aux, path):
                calls.append((q, chroma, aux))
                if aux == 95:
                    raise RuntimeError("gain map lost a highlight")
                size = q * 10 + (1000 if aux == 100 else 700 if aux == 85 else 800)
                path.write_bytes(bytes([q]) * size)
                return {"delivery_quality": q, "delivery_chroma_requested": chroma,
                        "gainmap_encoding_quality": aux, "coding_luma_rmse": .5,
                        "coding_chroma_rmse": .5 if chroma == "444" else 5.,
                        "highlight_max_luma_error": .05}
            info = select_heif_encoding(out, encode, gainmap=True)
            self.assertEqual((info["delivery_quality"], info["delivery_chroma_requested"],
                              info["gainmap_encoding_quality"]), (70, "444", 85))
            self.assertEqual(calls[0], (95, "444", 100))
            self.assertEqual({c for _, c, _ in calls}, {"444", "422", "420"})
            self.assertEqual({a for _, _, a in calls}, {95, 90, 85, 80, 100})
            self.assertLessEqual(len(calls), 14)
            self.assertEqual(list(Path(td).iterdir()), [out])

    def test_reference_failure_keeps_destination_and_does_not_weaken_reference(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "photo.heic"
            out.write_bytes(b"previous")
            calls = []
            def encode(q, chroma, aux, path):
                calls.append(q)
                path.write_bytes(b"invalid")
                raise RuntimeError("HDR reconstruction failed")
            with self.assertRaisesRegex(RuntimeError, "参考"):
                select_heif_encoding(out, encode, gainmap=True)
            self.assertEqual(calls, [95])
            self.assertEqual(out.read_bytes(), b"previous")
            self.assertEqual(list(Path(td).iterdir()), [out])

    def test_hdr_and_chroma_error_budgets_remain_active(self):
        with tempfile.TemporaryDirectory() as td:
            def encode(q, chroma, aux, path):
                reference = (q, chroma, aux) == (95, "444", 100)
                path.write_bytes(b"x" * (1000 if reference else 100))
                return {"delivery_quality": q, "delivery_chroma_requested": chroma,
                        "gainmap_encoding_quality": aux,
                        "highlight_max_luma_error": .05 if reference else .5}
            result = select_heif_encoding(Path(td) / "photo.heic", encode, gainmap=True)
            self.assertEqual(result["delivery_quality"], 95)
            self.assertEqual(result["gainmap_encoding_quality"], 100)


class SdrHeifFinalReadbackTests(unittest.TestCase):
    @contextmanager
    def codec_fixture(self, out, *, final_value=101):
        """Stub only container I/O; use the real pixel metrics and delivery gates."""
        rgb = np.full((8, 8, 3), 100, np.uint8)
        icc = b"test-icc"
        reads, buffers = [], []

        def encode(source, candidate, quality, chroma, **kwargs):
            candidate.write_bytes(f"{quality}:{chroma}".encode())
            return {"delivery_quality": quality, "delivery_chroma_requested": chroma,
                    "delivery_container": "heic"}

        def inspect(candidate):
            chroma = candidate.read_bytes().split(b"|")[0].split(b":")[1].decode()
            return {"width": 8, "height": 8, "bit_depth": 10, "headroom": 1.,
                    "has_iso_gainmap": False, "chroma_subsampling": ":".join(chroma)}

        def carry(source, candidate, container):
            self.assertEqual(container, "heic")
            candidate.write_bytes(candidate.read_bytes() + b"|metadata")
            return True

        def read(candidate, gamut):
            after_metadata = candidate.read_bytes().endswith(b"|metadata")
            reads.append(after_metadata)
            # Neither candidate validation nor final validation may commit early.
            self.assertEqual(out.read_bytes(), b"previous")
            decoded = np.full_like(rgb, final_value if after_metadata else 100)
            buffers.append(weakref.ref(decoded))
            return decoded

        with ExitStack() as stack:
            stack.enter_context(patch("dngscan.heif_encoder.encode", side_effect=encode))
            stack.enter_context(patch("dngscan.gainmap.inspect_gainmap_file", side_effect=inspect))
            stack.enter_context(patch("dngscan.gainmap.read_primary_rgb_u8", side_effect=read))
            stack.enter_context(patch("dngscan.color.output_icc_profile_bytes", return_value=icc))
            stack.enter_context(patch("dngscan.heif_gainmap._parse", return_value=(
                None, None, 1, None, None, [(b"colr", b"prof" + icc)],
                {1: [(True, 1)]}, None)))
            stack.enter_context(patch("dngscan.export.carry_capture_metadata", side_effect=carry))
            yield rgb, reads, buffers

    def test_manual_reuses_pixels_from_complete_final_metadata_verification(self):
        from dngscan.heif_delivery import save_sdr_heif
        profile = replace(resolve_delivery_profile("share", container="heic"),
                          heif_encoder="x265")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "photo.heic"
            out.write_bytes(b"previous")
            with self.codec_fixture(out) as (rgb, reads, buffers):
                info = save_sdr_heif(rgb, out, profile, source_raw=Path(td) / "source.dng",
                                     return_rgb=True)
            self.assertEqual(reads, [False, True])
            self.assertIs(info["_decoded_rgb"], buffers[-1]())
            np.testing.assert_array_equal(info["_decoded_rgb"], np.full_like(rgb, 101))
            self.assertTrue(info["exif_carried"])
            self.assertTrue(out.read_bytes().endswith(b"|metadata"))
            self.assertEqual(info["file_size_bytes"], out.stat().st_size)
            # Keep the existing reported candidate metrics, independent of pixel reuse.
            self.assertEqual(info["base_mean_code_error"], 0.)

    def test_auto_keeps_one_read_per_candidate_and_one_final_read(self):
        from dngscan.heif_delivery import save_sdr_heif
        profile = replace(resolve_delivery_profile("auto", container="heic"),
                          heif_encoder="x265")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "photo.heic"
            out.write_bytes(b"previous")
            with self.codec_fixture(out) as (rgb, reads, buffers):
                info = save_sdr_heif(rgb, out, profile, source_raw=Path(td) / "source.dng",
                                     return_rgb=True)
            self.assertGreater(len(info["auto_attempts"]), 1)
            self.assertEqual(reads, [False] * len(info["auto_attempts"]) + [True])
            self.assertIs(info["_decoded_rgb"], buffers[-1]())
            np.testing.assert_array_equal(info["_decoded_rgb"], np.full_like(rgb, 101))
            self.assertTrue(all(ref() is None for ref in buffers[:-1]))
            json.dumps(info["auto_attempts"])

    def test_final_pixel_validation_failure_keeps_previous_output(self):
        from dngscan.heif_delivery import save_sdr_heif
        profile = replace(resolve_delivery_profile("share", container="heic"),
                          heif_encoder="x265")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "photo.heic"
            out.write_bytes(b"previous")
            with self.codec_fixture(out, final_value=255) as (rgb, reads, _):
                with self.assertRaisesRegex(RuntimeError, "回读误差"):
                    save_sdr_heif(rgb, out, profile, source_raw=Path(td) / "source.dng",
                                  return_rgb=True)
            self.assertEqual(reads, [False, True])
            self.assertEqual(out.read_bytes(), b"previous")
            self.assertEqual(list(Path(td).iterdir()), [out])

    def test_cli_result_contains_no_decoded_pixels_and_releases_buffers(self):
        from dngscan.heif_delivery import save_sdr_heif
        profile = replace(resolve_delivery_profile("share", container="heic"),
                          heif_encoder="x265")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "photo.heic"
            out.write_bytes(b"previous")
            with self.codec_fixture(out) as (rgb, reads, buffers):
                info = save_sdr_heif(rgb, out, profile, source_raw=Path(td) / "source.dng",
                                     return_rgb=False)
            self.assertEqual(reads, [False, True])
            self.assertNotIn("_decoded_rgb", info)
            json.dumps(info)
            gc.collect()
            self.assertTrue(all(ref() is None for ref in buffers))


class SdrHeifLiveTests(unittest.TestCase):
    def test_depth_sampling_gamut_and_metadata_survive_readback(self):
        from dngscan.heif_encoder import available
        from dngscan.heif_delivery import save_sdr_heif
        try:
            import Quartz
            context = Quartz.CIContext.contextWithOptions_({})
        except ImportError:
            context = None
        if not available() or context is None:
            self.skipTest("libheif/x265 and a working Core Image context required")
        x = np.linspace(0, 255, 64, dtype=np.uint8)
        rgb = np.stack(np.broadcast_arrays(x[None, :], x[:, None], np.full((64, 64), 100, np.uint8)), axis=-1)
        with tempfile.TemporaryDirectory() as td:
            for gamut, depth, chroma in (("srgb", 8, "420"), ("p3", 10, "422"), ("srgb", 10, "444")):
                profile = replace(resolve_delivery_profile("share", quality=95, chroma=chroma, container="heic"),
                                  heif_encoder="x265", heif_bit_depth=depth)
                from dngscan.gainmap import read_primary_rgb_u8
                with patch("dngscan.gainmap.read_primary_rgb_u8", wraps=read_primary_rgb_u8) as read:
                    result = save_sdr_heif(rgb, Path(td) / f"{gamut}-{depth}-{chroma}.heic", profile,
                                           gamut, return_rgb=True)
                self.assertEqual(read.call_count, 2)
                self.assertEqual(result["bit_depth"], depth)
                self.assertEqual(result["chroma_subsampling"], ":".join(chroma))
                self.assertFalse(result["has_iso_gainmap"])
                self.assertTrue(result["icc_embedded"])
                self.assertEqual(result["_decoded_rgb"].shape, rgb.shape)


if __name__ == "__main__":
    unittest.main()
