# SPDX-License-Identifier: GPL-3.0-or-later
"""SDR HEIF routing and bounded codec search, including failure atomicity."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
                result = save_sdr_heif(rgb, Path(td) / f"{gamut}-{depth}-{chroma}.heic", profile,
                                       gamut, return_rgb=True)
                self.assertEqual(result["bit_depth"], depth)
                self.assertEqual(result["chroma_subsampling"], ":".join(chroma))
                self.assertFalse(result["has_iso_gainmap"])
                self.assertTrue(result["icc_embedded"])
                self.assertEqual(result["_decoded_rgb"].shape, rgb.shape)


if __name__ == "__main__":
    unittest.main()
