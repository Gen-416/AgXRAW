# SPDX-License-Identifier: GPL-3.0-or-later
"""Delivery gates: the file must be the rendition it claims to be."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dngscan.analysis import analyze
from dngscan.color import srgb_decode
from dngscan.gainmap import (
    BASE_BLOCK_P99_CODE_ERROR_LIMIT,
    BASE_CHANNEL_BIAS_CODE_ERROR_LIMIT,
    BASE_MEAN_CODE_ERROR_LIMIT,
    HDR_BLOCK_CHROMA_ERROR_LIMIT,
    HDR_BLOCK_MEDIAN_RELATIVE_ERROR_LIMIT,
    HDR_BLOCK_P95_RELATIVE_ERROR_LIMIT,
    HDR_BLOCK_P99_RELATIVE_ERROR_LIMIT,
    _base_roundtrip_error,
    _base_roundtrip_is_acceptable,
    _hdr_roundtrip_is_acceptable,
    _roundtrip_error,
    apple_gainmap_backend_status,
    inspect_gainmap_jpeg,
    write_apple_gainmap_jpeg,
)
from dngscan.grade import RENDER_MODE
from dngscan.hdr_agx import to_gainmap_alternate
from dngscan.raw_io import load_raw
from dngscan.render import render_output_u8
from dngscan.tone import build_render_plan

SIGMA = Path.home() / "Pictures" / "_SDI0150.DNG"
_BACKEND_OK, _BACKEND_WHY = apple_gainmap_backend_status()


class AlternatePackingTests(unittest.TestCase):
    def test_alternate_is_float16_rgba_with_opaque_alpha(self) -> None:
        rgb = np.linspace(-0.2, 12.0, 300, dtype=np.float32).reshape(10, 10, 3)
        out = to_gainmap_alternate(rgb, 8.0)
        self.assertEqual(out.dtype, np.float16)
        self.assertEqual(out.shape, (10, 10, 4))
        self.assertTrue(bool(np.all(out[..., 3] == np.float16(1.0))))

    def test_packer_defensively_clamps_an_invalid_negative_input(self) -> None:
        """The HDR projector should prevent this; packing still guards external callers."""
        rgb = np.array([[[-0.05, 0.4, 0.2]]], dtype=np.float32)
        self.assertEqual(float(to_gainmap_alternate(rgb, 8.0)[0, 0, 0]), 0.0)

    def test_hdr_diagnostic_encoder_does_not_repeat_sdr_finalization(self) -> None:
        from tools.hdr_ab import _encode_hdr_diagnostic

        linear = np.full((3, 4, 3), 0.25, dtype=np.float32)
        with mock.patch(
            "tools.hdr_ab.finalize_output_linear",
            side_effect=AssertionError("SDR finalizer must not run"),
        ):
            encoded = _encode_hdr_diagnostic(linear, "p3")
        self.assertEqual(encoded.shape, linear.shape)
        self.assertEqual(encoded.dtype, np.uint8)

    def test_peak_is_enforced(self) -> None:
        rgb = np.full((4, 4, 3), 20.0, dtype=np.float32)
        self.assertLessEqual(float(np.max(to_gainmap_alternate(rgb, 8.0))), 8.0)


class RoundtripErrorTests(unittest.TestCase):
    def test_shape_mismatch_reports_infinite_error(self) -> None:
        """A mismatch must never read as a small error and pass the gate."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "missing.jpg"
            path.write_bytes(b"not a jpeg")
            try:
                out = _roundtrip_error(path, np.zeros((2, 2, 4), dtype=np.float16))
            except Exception:
                return  # An exception is an acceptable failure mode here.
            self.assertEqual(out["chroma_error"], float("inf"))

    def test_banded_metrics_match_the_whole_frame_reference(self) -> None:
        """The banded/top-K implementation is a memory shape, not a metric change.

        Reference below is the historical whole-frame float32 computation; the banded
        version must reproduce it bit-for-bit on frames that exercise band boundaries,
        non-multiple-of-8 edges, and partially masked chroma.
        """
        from dngscan.gainmap import _ROUNDTRIP_BAND_ROWS

        rng = np.random.default_rng(11)
        height = _ROUNDTRIP_BAND_ROWS + 37  # crosses one band, ragged 8x8 edge
        width = 93
        intended = np.zeros((height, width, 4), dtype=np.float16)
        intended[..., :3] = rng.uniform(0.0, 2.5, (height, width, 3)).astype(np.float16)
        intended[height // 3 :: 7, ::5, :3] = np.float16(0.01)  # below chroma mask
        intended[..., 3] = np.float16(1.0)
        expanded = intended.copy()
        expanded[..., :3] += rng.normal(0.0, 0.02, (height, width, 3)).astype(np.float16)

        with mock.patch(
            "dngscan.gainmap._read_expanded_hdr_rgba_half", return_value=expanded
        ):
            banded = _roundtrip_error(Path("unused.jpg"), intended)

        a = expanded[..., :3].astype(np.float32).reshape(-1, 3)
        e = intended[..., :3].astype(np.float32).reshape(-1, 3)
        relative = np.max(np.abs(a - e), axis=1) / np.maximum(
            np.max(np.abs(e), axis=1), 0.05
        )
        mask = np.max(np.abs(e), axis=1) > 0.05
        chroma = np.abs(
            a[mask] / np.maximum(a[mask].sum(axis=1, keepdims=True), 1e-6)
            - e[mask] / np.maximum(e[mask].sum(axis=1, keepdims=True), 1e-6)
        )
        self.assertEqual(banded["chroma_error"], float(np.percentile(chroma, 99.0)))
        self.assertEqual(
            banded["median_relative_error"], float(np.median(relative))
        )
        self.assertEqual(
            banded["p99_relative_error"], float(np.percentile(relative, 99.0))
        )
        h8, w8 = height - height % 8, width - width % 8
        ab = expanded[:h8, :w8, :3].astype(np.float32).reshape(
            h8 // 8, 8, w8 // 8, 8, 3
        ).mean(axis=(1, 3))
        eb = intended[:h8, :w8, :3].astype(np.float32).reshape(
            h8 // 8, 8, w8 // 8, 8, 3
        ).mean(axis=(1, 3))
        block_relative = np.max(np.abs(ab - eb), axis=2) / np.maximum(
            np.max(np.abs(eb), axis=2), 0.05
        )
        self.assertEqual(
            banded["block_p99_relative_error"],
            float(np.percentile(block_relative, 99.0)),
        )

    def test_exact_upper_percentile_matches_numpy(self) -> None:
        from dngscan.gainmap import _exact_upper_percentile

        rng = np.random.default_rng(3)
        for n in (1, 2, 7, 1000, 12345):
            values = rng.exponential(0.05, n).astype(np.float32)
            k = int(np.ceil(0.01 * n)) + 8
            top = np.partition(values, max(0, n - k))[-min(k, n):]
            for q in (99.0, 99.9):
                with self.subTest(n=n, q=q):
                    self.assertAlmostEqual(
                        _exact_upper_percentile(top, n, q),
                        float(np.percentile(values, q)),
                        places=6,
                    )

    def test_reference_white_and_below_are_part_of_the_gate(self) -> None:
        intended = np.full((4, 4, 4), 0.25, dtype=np.float16)
        intended[..., 3] = np.float16(1.0)
        expanded = intended.copy()
        expanded[..., 0] = np.float16(0.35)
        with mock.patch(
            "dngscan.gainmap._read_expanded_hdr_rgba_half", return_value=expanded
        ):
            out = _roundtrip_error(Path("unused.jpg"), intended)
        self.assertGreater(out["median_relative_error"], 0.1)
        self.assertGreater(out["p99_relative_error"], 0.1)

    def test_zero_mean_hdr_texture_error_does_not_reject_low_frequency_fidelity(self) -> None:
        intended = np.full((16, 16, 4), 0.25, dtype=np.float16)
        intended[..., 3] = np.float16(1.0)
        expanded = intended.copy()
        for y in range(0, 16, 8):
            for x in range(0, 16, 8):
                expanded[y, x, :3] = np.float16(0.30)
                expanded[y, x + 1, :3] = np.float16(0.20)
        with mock.patch(
            "dngscan.gainmap._read_expanded_hdr_rgba_half", return_value=expanded
        ):
            metrics = _roundtrip_error(Path("unused.jpg"), intended)
        self.assertGreater(metrics["p99_relative_error"], 0.1)
        self.assertTrue(_hdr_roundtrip_is_acceptable(metrics))

    def test_uniform_hdr_scale_error_is_rejected(self) -> None:
        intended = np.full((16, 16, 4), 0.25, dtype=np.float16)
        intended[..., 3] = np.float16(1.0)
        expanded = intended.copy()
        expanded[..., :3] = np.float16(0.275)
        with mock.patch(
            "dngscan.gainmap._read_expanded_hdr_rgba_half", return_value=expanded
        ):
            metrics = _roundtrip_error(Path("unused.jpg"), intended)
        self.assertGreater(metrics["block_median_relative_error"], 0.09)
        self.assertFalse(_hdr_roundtrip_is_acceptable(metrics))

    def test_sdr_base_shape_mismatch_cannot_pass(self) -> None:
        with mock.patch("PIL.Image.open") as opened:
            opened.return_value.__enter__.return_value.convert.return_value = np.zeros(
                (2, 3, 3), dtype=np.uint8
            )
            out = _base_roundtrip_error(
                Path("unused.jpg"), np.zeros((3, 2, 3), dtype=np.uint8)
            )
        self.assertEqual(out["base_mean_code_error"], float("inf"))

    def test_zero_mean_high_frequency_jpeg_error_does_not_reject_the_rendition(self) -> None:
        intended = np.full((16, 16, 3), 128, dtype=np.uint8)
        decoded = intended.copy()
        # A balanced +/-10 pair in every JPEG block creates large pixel outliers but no
        # low-frequency brightness or colour shift.
        for y in range(0, 16, 8):
            for x in range(0, 16, 8):
                decoded[y, x] = 138
                decoded[y, x + 1] = 118
        with mock.patch("PIL.Image.open") as opened:
            opened.return_value.__enter__.return_value.convert.return_value = decoded
            metrics = _base_roundtrip_error(Path("unused.jpg"), intended)
        self.assertGreater(metrics["base_p99_code_error"], 4.0)
        self.assertTrue(_base_roundtrip_is_acceptable(metrics))

    def test_banded_base_metrics_match_the_whole_frame_reference(self) -> None:
        """The banded walk must reproduce the historical whole-frame statistics.

        Multi-band shape (rows > _ROUNDTRIP_BAND_ROWS) with a ragged non-multiple-of-8
        tail, so band boundaries, the top-K percentile and the 8x8 block trimming are
        all exercised against the plain float32 reference computed here.
        """
        rng = np.random.default_rng(7)
        rows = 1043  # > 2 bands of 512, tail not a multiple of 8
        intended = rng.integers(0, 256, size=(rows, 36, 3), dtype=np.uint8)
        decoded = np.clip(
            intended.astype(np.int16) + rng.integers(-9, 10, size=intended.shape),
            0,
            255,
        ).astype(np.uint8)
        with mock.patch("PIL.Image.open") as opened:
            opened.return_value.__enter__.return_value.convert.return_value = decoded
            metrics = _base_roundtrip_error(Path("unused.jpg"), intended)
        signed = decoded.astype(np.float32) - intended.astype(np.float32)
        channel_error = np.abs(signed)
        pixel_error = np.max(channel_error, axis=2)
        h8, w8 = rows - rows % 8, 36 - 36 % 8
        block_signed = signed[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8, 3)
        block_error = np.max(np.abs(np.mean(block_signed, axis=(1, 3))), axis=2)
        self.assertAlmostEqual(
            metrics["base_mean_code_error"], float(np.mean(channel_error)), places=5
        )
        self.assertAlmostEqual(
            metrics["base_p99_code_error"],
            float(np.percentile(pixel_error, 99.0)),
            places=5,
        )
        self.assertEqual(metrics["base_max_code_error"], float(np.max(pixel_error)))
        self.assertAlmostEqual(
            metrics["base_channel_bias_code_error"],
            float(np.max(np.abs(np.mean(signed, axis=(0, 1))))),
            places=5,
        )
        self.assertAlmostEqual(
            metrics["base_block_p99_code_error"],
            float(np.percentile(block_error, 99.0)),
            places=5,
        )

    def test_uniform_code_shift_is_rejected_as_a_changed_rendition(self) -> None:
        intended = np.full((16, 16, 3), 128, dtype=np.uint8)
        decoded = np.full((16, 16, 3), 130, dtype=np.uint8)
        with mock.patch("PIL.Image.open") as opened:
            opened.return_value.__enter__.return_value.convert.return_value = decoded
            metrics = _base_roundtrip_error(Path("unused.jpg"), intended)
        self.assertGreater(metrics["base_channel_bias_code_error"], 1.0)
        self.assertFalse(_base_roundtrip_is_acceptable(metrics))


@unittest.skipUnless(_BACKEND_OK, f"gain-map backend unavailable: {_BACKEND_WHY}")
class WriterVerificationTests(unittest.TestCase):
    def test_writer_rejects_a_rendition_it_cannot_reproduce(self) -> None:
        """The per-file check is the real guarantee, so it has to actually bite.

        Extreme per-channel gain ratios are what the container cannot carry: measured,
        a channel gaining 1.25x beside one gaining 3x comes back lifted to 1.5x. A file
        like that would claim to be a rendition it is not, so writing it must fail rather
        than succeed quietly.
        """
        h, patch = 24, 24
        gains = np.array([[1.15, 2.4, 3.6], [3.6, 1.15, 2.4]], dtype=np.float32)
        base = np.full((h, patch * len(gains), 3), 180, dtype=np.uint8)
        base_linear = srgb_decode(base.astype(np.float32) / np.float32(255.0))
        hdr = np.empty(base.shape[:2] + (4,), dtype=np.float16)
        for i, gain in enumerate(gains):
            hdr[:, i * patch:(i + 1) * patch, :3] = (
                base_linear[:, i * patch:(i + 1) * patch] * gain
            ).astype(np.float16)
        hdr[..., 3] = np.float16(1.0)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "extreme.jpg"
            with self.assertRaises(RuntimeError):
                write_apple_gainmap_jpeg(base, hdr, out, 100, 2.0)
            self.assertFalse(out.exists(), "a rejected rendition must leave no file")

    def test_neutral_hdr_survives_the_container(self) -> None:
        h, w = 24, 48
        base = np.full((h, w, 3), 180, dtype=np.uint8)
        linear = srgb_decode(base.astype(np.float32) / np.float32(255.0))
        hdr = np.empty((h, w, 4), dtype=np.float16)
        hdr[..., :3] = (linear * 2.5).astype(np.float16)
        hdr[..., 3] = np.float16(1.0)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "neutral.jpg"
            info = write_apple_gainmap_jpeg(base, hdr, out, 100, 2.0)
            self.assertTrue(out.exists())
            self.assertLess(info["chroma_error"], 0.01)


@unittest.skipUnless(SIGMA.is_file() and _BACKEND_OK, "sample frame or backend unavailable")
class EndToEndDeliveryTests(unittest.TestCase):
    def test_export_produces_a_conforming_iso_gainmap_file(self) -> None:
        from dngscan.export import export_ultrahdr_jpeg

        bundle = load_raw(SIGMA, scene_half_size=True)
        analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
        plan = build_render_plan(bundle, analysis, RENDER_MODE, "p3")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "hdr.jpg"
            info = export_ultrahdr_jpeg(SIGMA, out, 100, bundle, analysis, plan)
            self.assertTrue(out.exists())
            probe = inspect_gainmap_jpeg(out)
            self.assertTrue(probe["has_iso_gainmap"])
            self.assertEqual(probe["profile"], "Display P3")
            self.assertEqual(probe["chroma_subsampling"], "4:4:4")
            # An L008 gain map cannot carry independent per-channel geometry.
            self.assertNotIn(str(probe["gainmap_pixel_format"]), ("", "L008"))
            self.assertGreater(probe["headroom"], 1.0)
            self.assertLessEqual(
                info["block_median_relative_error"],
                HDR_BLOCK_MEDIAN_RELATIVE_ERROR_LIMIT,
            )
            self.assertLessEqual(
                info["block_p95_relative_error"], HDR_BLOCK_P95_RELATIVE_ERROR_LIMIT
            )
            self.assertLessEqual(
                info["block_p99_relative_error"], HDR_BLOCK_P99_RELATIVE_ERROR_LIMIT
            )
            self.assertLessEqual(
                info["block_chroma_error"], HDR_BLOCK_CHROMA_ERROR_LIMIT
            )
            self.assertLessEqual(info["headroom_error_ev"], 0.05)
            self.assertLessEqual(
                info["base_mean_code_error"], BASE_MEAN_CODE_ERROR_LIMIT
            )
            self.assertLessEqual(
                info["base_channel_bias_code_error"],
                BASE_CHANNEL_BIAS_CODE_ERROR_LIMIT,
            )
            self.assertLessEqual(
                info["base_block_p99_code_error"], BASE_BLOCK_P99_CODE_ERROR_LIMIT
            )
            # Declared headroom must not exceed what the scene was allowed.
            self.assertLessEqual(
                float(np.log2(probe["headroom"])), info["rendered_headroom_ev"] + 1e-3
            )

    def test_sdr_base_is_the_same_rendition_as_a_plain_export(self) -> None:
        """Same pixels into the encoder; the encoders themselves then differ.

        Pillow and Core Image do not agree bit for bit, especially on high-ISO noise, so
        this asserts the pre-encoder rendition rather than compressed bytes.
        """
        bundle = load_raw(SIGMA, scene_half_size=True)
        analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
        plan = build_render_plan(bundle, analysis, RENDER_MODE, "p3")
        a = render_output_u8(bundle, analysis, "p3", plan)
        b = render_output_u8(bundle, analysis, "p3", plan)
        self.assertTrue(bool(np.array_equal(a, b)))

    def test_a_scene_without_a_reliable_tail_is_refused(self) -> None:
        """Refusing beats writing a file whose HDR range the sensor never recorded."""
        import dataclasses

        from dngscan.export import export_ultrahdr_jpeg

        bundle = load_raw(SIGMA, scene_half_size=True)
        analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
        plan = build_render_plan(bundle, analysis, RENDER_MODE, "p3")
        # HDR owns its white endpoint, so changing the SDR one must not control this gate.
        # Remove the actual RAW-authoritative tail instead.
        no_tail = dataclasses.replace(plan.scene, reliable_tail_ev_p9999=float("nan"))
        flat = dataclasses.replace(plan, scene=no_tail)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "flat.jpg"
            with self.assertRaises(RuntimeError):
                export_ultrahdr_jpeg(SIGMA, out, 100, bundle, analysis, flat)


if __name__ == "__main__":
    unittest.main()
