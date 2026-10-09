# SPDX-License-Identifier: GPL-3.0-or-later
"""10-bit SDR keeps its floating master through encoding and final verification."""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import heif_encoder
from dngscan.delivery import resolve_delivery_profile
from dngscan.heif_delivery import save_sdr_heif


class HeifFloatQuantizationTests(unittest.TestCase):
    def test_float_gradient_uses_1024_levels_and_legacy_u8_only_256(self):
        gradient = np.broadcast_to(np.linspace(0, 1, 1024, dtype=np.float32)[None, :, None],
                                   (3, 1024, 3))
        direct = heif_encoder._quantized_band(gradient, 10)
        legacy = heif_encoder._quantized_band(np.rint(gradient * 255).astype(np.uint8), 10)
        self.assertEqual(np.unique(direct).size, 1024)
        self.assertEqual(np.unique(legacy).size, 256)
        for source in (gradient[::-1, ::-1], gradient.transpose(1, 0, 2)):
            result = heif_encoder._quantized_band(source, 10)
            self.assertTrue(result.flags.c_contiguous)
            np.testing.assert_array_equal(result, np.rint(source * 1023).astype(np.uint16))

    def test_tpdf_is_deterministic_in_10bit_units_and_keeps_source_untouched(self):
        source = np.linspace(0, 1, 257 * 33 * 3, dtype=np.float32).reshape(257, 33, 3)
        before = source.copy()
        results = []
        for _ in range(2):
            rng = np.random.default_rng(0)
            results.append(np.concatenate([heif_encoder._quantized_band(source[y:y+128], 10, rng=rng)
                                           for y in range(0, source.shape[0], 128)]))
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(source, before)
        self.assertGreater(np.unique(results[0]).size, 1000)
        self.assertLessEqual(np.max(np.abs(results[0].astype(np.float32) - source * 1023)), 1.5)
        self.assertTrue(np.all(results[0] >= 0))
        self.assertTrue(np.all(results[0] <= 1023))
        self.assertFalse(np.array_equal(results[0], heif_encoder._quantized_band(source, 10)))

    def test_bad_domain_or_integer_units_are_rejected_before_loading_encoder(self):
        invalid = [np.zeros((8, 8, 3), np.uint16), np.ones((8, 8, 3), np.int32),
                   np.full((8, 8, 3), np.nan, np.float32),
                   np.full((8, 8, 3), np.inf, np.float32),
                   np.full((8, 8, 3), -.001, np.float32),
                   np.full((8, 8, 3), 1.001, np.float32),
                   np.zeros((0, 8, 3), np.float32)]
        with patch("dngscan.heif_encoder._library", side_effect=AssertionError("encoder must not load")):
            for source in invalid:
                with self.subTest(dtype=source.dtype, shape=source.shape):
                    with self.assertRaises(ValueError):
                        heif_encoder.encode(source, Path("unused.heic"), 95)
            for source, options in ((np.zeros((8, 8, 3), np.uint8), {}),
                                    (np.zeros((8, 8, 3), np.float32), {"bit_depth": 8}),
                                    (np.zeros((8, 8, 3), np.float32), {"auxiliary": True})):
                with self.assertRaisesRegex(ValueError, "TPDF"):
                    heif_encoder.encode(source, Path("unused.heic"), 95,
                                        dither_quantization=True, **options)


class SdrHeifFloatDeliveryTests(unittest.TestCase):
    @contextmanager
    def fixture(self, out, intended, *, read_delta=0., final_delta=0., invalid_nclx=False):
        calls, reads = [], []
        icc = b"precision-icc"
        def encode(source, candidate, quality, chroma, **options):
            calls.append((source, options))
            candidate.write_bytes(f"{quality}|{chroma}|".encode()
                + b"x" * (quality * 5 + (200 if chroma == "444" else 0)))
            return {"delivery_quality": quality, "delivery_chroma_requested": chroma,
                    "delivery_container": "heic"}
        def inspect(candidate):
            chroma = candidate.read_bytes().split(b"|")[1].decode()
            return {"width": intended.shape[1], "height": intended.shape[0], "bit_depth": 10,
                    "has_iso_gainmap": False, "headroom": 1., "chroma_subsampling": ":".join(chroma)}
        def read(candidate, gamut, **options):
            final = candidate.read_bytes().endswith(b"metadata")
            reads.append(final)
            self.assertEqual(out.read_bytes(), b"previous")
            return intended.copy() + np.float32(final_delta if final else read_delta)
        def carry(source, candidate, container):
            candidate.write_bytes(candidate.read_bytes() + b"metadata")
            return True
        nclx = b"nclx" + struct.pack(">HHHB", 1, 1 if invalid_nclx else 13, 1, 128)
        with ExitStack() as stack:
            stack.enter_context(patch("dngscan.heif_encoder.encode", side_effect=encode))
            stack.enter_context(patch("dngscan.gainmap.inspect_gainmap_file", side_effect=inspect))
            stack.enter_context(patch("dngscan.gainmap.read_primary_rgb_float", side_effect=read))
            stack.enter_context(patch("dngscan.gainmap.read_primary_rgb_u8",
                                     side_effect=AssertionError("10-bit quality gate must stay floating")))
            stack.enter_context(patch("dngscan.color.output_icc_profile_bytes", return_value=icc))
            stack.enter_context(patch("dngscan.heif_gainmap._parse", return_value=(
                None, None, 1, None, None, [(b"colr", b"prof" + icc), (b"colr", nclx)],
                {1: [(True, 1), (True, 2)]}, None)))
            stack.enter_context(patch("dngscan.export.carry_capture_metadata", side_effect=carry))
            yield calls, reads

    def test_manual_and_auto_candidates_preserve_float_and_verify_final_metadata(self):
        intended = np.broadcast_to(np.linspace(.2, .8, 1024, dtype=np.float32)[None, :, None],
                                   (8, 1024, 3)).copy()
        for name in ("share", "auto"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                out = Path(td) / "photo.heic"
                out.write_bytes(b"previous")
                profile = replace(resolve_delivery_profile(name, container="heic"), heif_encoder="x265")
                with self.fixture(out, intended) as (calls, reads):
                    info = save_sdr_heif(intended, out, profile, source_raw=Path("source.dng"), return_rgb=True)
                self.assertTrue(all(source is intended for source, _ in calls))
                self.assertTrue(all(options["dither_quantization"] for _, options in calls))
                self.assertTrue(all(options["bit_depth"] == 10 for _, options in calls))
                self.assertEqual(reads, [False] * len(calls) + [True])
                self.assertEqual(info["readback_precision"], "float32")
                self.assertTrue(info["nclx_verified"])
                self.assertEqual(info["_decoded_rgb"].dtype, np.uint8)
                np.testing.assert_array_equal(info["_decoded_rgb"], np.rint(intended * 255).astype(np.uint8))
                self.assertEqual(info["base_mean_code_error"], 0.)
                self.assertTrue(out.read_bytes().endswith(b"metadata"))

    def test_sub_8bit_error_is_measured_without_quantizing_readback(self):
        intended = np.full((8, 8, 3), .5, np.float32)
        profile = replace(resolve_delivery_profile("share", container="heic"), heif_encoder="x265")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "photo.heic"
            out.write_bytes(b"previous")
            delta = .125 / 1023.
            with self.fixture(out, intended, read_delta=delta, final_delta=delta):
                info = save_sdr_heif(intended, out, profile, source_raw=Path("source.dng"))
            self.assertGreater(info["base_mean_code_error"], .02)
            self.assertLess(info["base_mean_code_error"], .04)

    def test_final_float_or_nclx_failure_preserves_previous_destination(self):
        intended = np.full((8, 8, 3), .5, np.float32)
        profile = replace(resolve_delivery_profile("share", container="heic"), heif_encoder="x265")
        for options, reason in (({"final_delta": .1}, "回读误差"),
                                ({"invalid_nclx": True}, "NCLX")):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as td:
                out = Path(td) / "photo.heic"
                out.write_bytes(b"previous")
                with self.fixture(out, intended, **options):
                    with self.assertRaisesRegex(RuntimeError, reason):
                        save_sdr_heif(intended, out, profile, source_raw=Path("source.dng"))
                self.assertEqual(out.read_bytes(), b"previous")
                self.assertEqual(list(Path(td).iterdir()), [out])

    def test_8bit_delivery_keeps_u8_reader_and_return_buffer(self):
        from tests.test_sdr_heif import SdrHeifFinalReadbackTests
        profile = replace(resolve_delivery_profile("share", container="heic"),
                          heif_encoder="x265", heif_bit_depth=8)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "photo.heic"
            out.write_bytes(b"previous")
            fixture = SdrHeifFinalReadbackTests()
            with patch("dngscan.gainmap.read_primary_rgb_float",
                       side_effect=AssertionError("8-bit reader contract changed")), \
                 fixture.codec_fixture(out, bit_depth=8) as (rgb, reads, buffers):
                info = save_sdr_heif(rgb, out, profile, source_raw=Path("source.dng"), return_rgb=True)
            self.assertIs(info["_decoded_rgb"], buffers[-1]())
            self.assertEqual(reads, [False, True])
            self.assertEqual(info["readback_precision"], "uint8")


class SdrHeifFloatLiveTests(unittest.TestCase):
    def test_real_float_ramp_keeps_sub_8bit_levels_in_manual_and_auto_delivery(self):
        try:
            import Quartz
            context = Quartz.CIContext.contextWithOptions_({})
        except ImportError:
            context = None
        if not heif_encoder.available() or context is None:
            self.skipTest("libheif/x265 and a working Core Image context required")
        from dngscan.gainmap import read_primary_rgb_float

        source = np.broadcast_to(
            np.linspace(.02, .98, 1024, dtype=np.float32)[None, :, None],
            (64, 1024, 3)).copy()
        before = source.copy()
        with tempfile.TemporaryDirectory() as td:
            for name, gamut in (("share", "srgb"), ("share", "p3"), ("auto", "p3")):
                with self.subTest(profile=name, gamut=gamut):
                    controls = {} if name == "auto" else {"quality": 100, "chroma": "444"}
                    profile = replace(resolve_delivery_profile(name, container="heic", **controls),
                                      heif_encoder="x265", heif_bit_depth=10)
                    path = Path(td) / f"{name}-{gamut}.heic"
                    info = save_sdr_heif(source, path, profile, gamut, return_rgb=True)
                    decoded = read_primary_rgb_float(path, gamut)
                    self.assertEqual(info["bit_depth"], 10)
                    self.assertEqual(info["readback_precision"], "float32")
                    self.assertEqual(info["quantization_dither"], "TPDF-10bit-seed0")
                    self.assertEqual(info["_decoded_rgb"].dtype, np.uint8)
                    self.assertTrue(info["icc_embedded"])
                    self.assertFalse(info["has_iso_gainmap"])
                    self.assertGreater(np.unique(decoded[..., 0]).size, 512)
                    # A float API alone cannot prove precise decoding. The
                    # actual readback must contain values beyond the u8 grid.
                    codes = decoded * np.float32(255.)
                    self.assertGreater(float(np.max(np.abs(codes - np.rint(codes)))), .1)
                    np.testing.assert_array_equal(source, before)


if __name__ == "__main__":
    unittest.main()
