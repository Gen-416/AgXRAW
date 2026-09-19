# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from dngscan.analysis import estimate_raw_noise_floor
from dngscan.auto_encode import coding_metrics, select_encoding
from dngscan.auto_ev import compute_auto_ev
from dngscan.cli import parse_args
from dngscan.delivery import resolve_delivery_profile
from dngscan.gui.preview_cache import build_proxy_entry
from dngscan.sampling import sample_indices
from dngscan.tone import scene_tone_metrics
from tests.test_preview_cache import _analysis, _bundle


class AutomaticAnalysisTests(unittest.TestCase):
    def test_noise_is_invariant_to_channel_offsets_and_linear_gradients(self):
        h, w = 512, 768
        b = _bundle()
        b.raw_colors = np.tile(np.array(b.raw_pattern, np.uint8), (h//2, w//2))
        b.white_level = 16383
        rng = np.random.default_rng(12)
        noise = rng.normal(0, 3, (h, w))
        levels = {k: 16383 for k in range(4)}
        results = []
        for offsets, gradient in (([1000]*4, 0), ([2000, 1000, 500, 1000], 0),
                                  ([2000, 1000, 500, 1000], 2)):
            b.raw_image = np.round(np.asarray(offsets)[b.raw_colors] + noise
                                   + gradient*np.arange(w)[None, :]).astype(np.uint16)
            results.append(estimate_raw_noise_floor(b, levels)*16383)
        for value in results:
            self.assertAlmostEqual(value, 3., delta=.4)
        self.assertAlmostEqual(results[0], results[1], places=6)
        self.assertAlmostEqual(results[1], results[2], delta=.15)

    def test_periodic_highlights_survive_translation(self):
        b = _bundle()
        b.scene_scale = b.exposure_gain = 1.
        b.clip_masks = None
        b.scene_rec2020_render = np.full((1200, 1600, 3), .01, np.float32)
        b.scene_rec2020_render.reshape(-1, 3)[3*np.arange(10000, 12000)+1] = 32.
        tails = []
        for offset in (0, 1, 2):
            moved = replace(b, scene_rec2020_render=np.roll(b.scene_rec2020_render, offset, axis=1))
            tails.append(scene_tone_metrics(moved, _analysis()).reliable_tail_ev_p9999)
        np.testing.assert_allclose(tails, np.log2(32/.18), atol=3e-5)

    def test_sampling_is_bounded_ordered_and_repeatable(self):
        for n in (0, 7, 3000000):
            idx = sample_indices(n, 8192)
            np.testing.assert_array_equal(idx, sample_indices(n, 8192))
            self.assertEqual(idx.size, min(n, 8192))
            self.assertTrue(np.all(np.diff(idx) > 0))
            self.assertTrue(np.all((idx >= 0) & (idx < n)))

    def test_auto_ev_uses_identical_full_resolution_evidence_on_proxy(self):
        b = _bundle()
        b.scene_scale = 1.
        b.clip_masks = None
        b.scene_rec2020_render = np.full((1600, 2400, 3), .02, np.float32)
        b.scene_rec2020_render.reshape(-1, 3)[::457] = .75
        a = _analysis()
        proxy = build_proxy_entry(b, a).bundle
        self.assertNotEqual(proxy.scene_rec2020_render.shape, b.scene_rec2020_render.shape)
        for core in ("agx", "neutral"):
            full = compute_auto_ev(b, a, tone_core=core)
            preview = compute_auto_ev(proxy, a, tone_core=core)
            self.assertEqual(full, preview)


class AutomaticEncodingTests(unittest.TestCase):
    def test_hdr_local_error_floor_keeps_chroma_and_highlight_guards(self):
        from dngscan.auto_encode import additional_error_acceptable
        reference={"block_p95_luma_error":.019,"highlight_max_luma_error":.63,
                   "chroma_error":.097,"coding_luma_rmse":.62,"coding_chroma_rmse":4.54}
        candidate={**reference,"block_p95_luma_error":.025,"highlight_max_luma_error":.64,
                   "chroma_error":.099,"coding_luma_rmse":.91,"coding_chroma_rmse":4.67}
        self.assertTrue(additional_error_acceptable(candidate,reference))
        self.assertFalse(additional_error_acceptable({**candidate,"chroma_error":.15},reference))
        self.assertFalse(additional_error_acceptable({**candidate,"highlight_max_luma_error":.8},reference))

    def test_defaults_and_explicit_manual_settings(self):
        args = parse_args(["photo.dng"])
        self.assertEqual((args.delivery_profile, args.jpeg_quality, args.chroma), ("auto", 99, "422"))
        self.assertEqual(args.ev, "auto")
        self.assertEqual(float(parse_args(["photo.dng", "--ev", "-1"]).ev), -1.)
        self.assertEqual(resolve_delivery_profile("share").quality, 95)
        self.assertEqual(resolve_delivery_profile("archive").quality, 100)

    def test_selects_smallest_acceptable_actual_file_and_cleans_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/"photo.jpg"
            out.write_bytes(b"old")
            def encode(q, path):
                sizes = {99: 1000, 98: 900, 97: 800, 96: 700, 95: 600}
                path.write_bytes(bytes([q])*sizes[q])
                return {"delivery_quality": q, "coding_luma_rmse": {99: .6, 98: .8, 97: .9, 96: 1.3, 95: 1.5}[q]}
            result = select_encoding(out, encode)
            self.assertEqual(result["delivery_quality"], 97)
            self.assertEqual(out.read_bytes(), bytes([97])*800)
            self.assertEqual(list(Path(td).iterdir()), [out])

    def test_failed_reference_preserves_destination(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/"photo.jpg"; out.write_bytes(b"old")
            def encode(q, path):
                path.write_bytes(b"bad")
                raise RuntimeError("bad ICC")
            with self.assertRaises(RuntimeError):
                select_encoding(out, encode)
            self.assertEqual(out.read_bytes(), b"old")
            self.assertEqual(list(Path(td).iterdir()), [out])

    def test_hdr_error_or_failed_smaller_candidate_keeps_reference(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/"photo.jpg"
            def encode(q, path):
                path.write_bytes(bytes([q])*(1000 if q == 99 else 500))
                if q == 98: raise RuntimeError("HDR roundtrip rejected")
                return {"delivery_quality": q, "coding_luma_rmse": 1.,
                        "highlight_max_luma_error": .1 if q == 99 else .2}
            result = select_encoding(out, encode)
            self.assertEqual(result["delivery_quality"], 99)
            self.assertEqual(result["auto_saved_pct"], 0.)

    def test_opposite_sign_errors_do_not_cancel(self):
        source = np.full((33, 47, 3), 128, np.uint8)
        changed = source.copy(); changed[::2] += 10; changed[1::2] -= 10
        self.assertAlmostEqual(coding_metrics(changed, source)["coding_luma_rmse"], 10., places=5)
        self.assertAlmostEqual(coding_metrics(changed, source)["coding_local_luma_p99"], 10., places=5)

    def test_420_requires_chroma_fidelity_even_when_luma_is_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/"photo.jpg"
            qualities = []
            def encode(q, path, chroma="422"):
                qualities.append(q)
                path.write_bytes(bytes([q])*(q*10 if chroma == "422" else 400))
                return {"delivery_quality": q, "delivery_chroma_requested": chroma,
                        "coding_luma_rmse": .8, "coding_chroma_rmse": 1. if chroma == "422" else 2.}
            result = select_encoding(out, encode, encode_420=lambda q, p: encode(q, p, "420"))
            self.assertEqual(result["delivery_chroma_requested"], "422")
            self.assertGreaterEqual(min(qualities), 95)
            self.assertLessEqual(max(qualities), 99)

    def test_failed_highest_candidate_uses_verified_reference(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/"photo.jpg"
            def encode(q, path):
                if q == 99: raise RuntimeError("HDR reconstruction gate")
                path.write_bytes(bytes([q])*1000)
                return {"delivery_quality": q, "coding_luma_rmse": .8}
            result = select_encoding(out, encode)
            self.assertEqual(result["auto_reference_quality"], 98)
            self.assertEqual(result["delivery_quality"], 98)

    def test_sdr_writer_renders_once_and_returns_actual_decoded_pixels(self):
        from PIL import Image, JpegImagePlugin
        from dngscan.export import export_srgb_jpeg
        rgb = np.random.default_rng(17).integers(0, 256, (129, 177, 3), np.uint8)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/"photo.jpg"
            with mock.patch("dngscan.export.render_output_u8", return_value=rgb) as render, \
                 mock.patch("dngscan.export.carry_capture_metadata", return_value=False):
                info = export_srgb_jpeg(Path("source.dng"), out, 99, _bundle(), _analysis(),
                                        delivery=resolve_delivery_profile("auto"), return_rgb=True)
            render.assert_called_once()
            with Image.open(out) as im:
                np.testing.assert_array_equal(info["_decoded_rgb"], np.asarray(im.convert("RGB")))
                self.assertEqual(JpegImagePlugin.get_sampling(im), 1)
                self.assertTrue(im.info["icc_profile"])
            self.assertGreaterEqual(info["delivery_quality"], 95)

    def test_missing_icc_cannot_publish_an_untagged_automatic_delivery(self):
        from dngscan.export import export_srgb_jpeg
        rgb = np.full((32, 48, 3), 128, np.uint8)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/"photo.jpg"
            out.write_bytes(b"previous delivery")
            with mock.patch("dngscan.export.render_output_u8", return_value=rgb), \
                 mock.patch("dngscan.export.output_icc_profile_bytes", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "ICC"):
                    export_srgb_jpeg(Path("source.dng"), out, 99, _bundle(), _analysis(),
                                     delivery=resolve_delivery_profile("auto"))
            self.assertEqual(out.read_bytes(), b"previous delivery")


if __name__ == "__main__":
    unittest.main()
