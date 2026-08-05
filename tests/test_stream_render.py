# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import unittest
from unittest import mock

from dngscan._deps import np
from dngscan.models import RawBundle, ToneCompressionPlan
from dngscan.render import (
    deterministic_dither_planes,
    quantize_final_output_linear_to_u8,
    render_output_linear,
    render_output_u8,
)


class StreamRenderTest(unittest.TestCase):
    def test_fused_u8_path_matches_legacy_linear_path(self) -> None:
        rng = np.random.default_rng(12)
        scene = rng.integers(0, 65536, size=(41, 53, 3), dtype=np.uint16)
        bundle = RawBundle(
            path=Path("synthetic.dng"),
            raw_image=np.zeros((2, 2), dtype=np.uint16),
            raw_colors=np.asarray([[0, 1], [3, 2]], dtype=np.uint8),
            xyz_render=np.zeros_like(scene),
            render_scale=65535.0,
            scene_rec2020_render=scene,
            scene_scale=65535.0,
            white_level=65535,
            black_levels=[0.0, 0.0, 0.0, 0.0],
            camera_wb=[1.0, 1.0, 1.0, 1.0],
            color_desc="RGBG",
            raw_pattern=[[0, 1], [3, 2]],
            camera_white_levels=[65535.0] * 4,
        )
        plan = ToneCompressionPlan(
            target_gamut="Rec2020",
            luma_p1=0.01,
            luma_p50=0.18,
            luma_p99=1.0,
            luma_p999=2.0,
            black_ev=-8.0,
            white_ev=5.0,
            dynamic_range_ev=13.0,
            contrast=3.0,
            toe_power=1.5,
            shoulder_power=2.9,
            chroma_p95=0.5,
            negative_rgb_pct=0.0,
            over_rgb_pct=0.0,
        )
        with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "0"}):
            legacy = quantize_final_output_linear_to_u8(
                render_output_linear(bundle, object(), "srgb", plan), "srgb"
            )
            fused = render_output_u8(bundle, object(), "srgb", plan)
        np.testing.assert_array_equal(fused, legacy)

    def test_native_output_path_matches_reference_within_one_code_value(self) -> None:
        import dngscan._fast as fast_backend

        if not fast_backend.available():
            self.skipTest("native extension not built")
        bundle, plan = self._bundle_and_plan(73, 97, seed=31)
        for gamut in ("srgb", "p3"):
            for core in ("agx", "neutral", "lum"):
                core_plan = replace(plan, tone_core=core)
                with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "0"}):
                    reference = render_output_u8(
                        bundle, object(), gamut, core_plan
                    )
                with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "1"}):
                    native = render_output_u8(bundle, object(), gamut, core_plan)
                delta = np.abs(native.astype(np.int16) - reference.astype(np.int16))
                self.assertLessEqual(int(delta.max()), 1, (gamut, core))
                self.assertEqual(float(np.percentile(delta, 99)), 0.0, (gamut, core))
                self.assertLessEqual(float(np.mean(delta != 0)), 0.01, (gamut, core))

    def test_precomputed_dither_planes_are_byte_identical(self) -> None:
        bundle, plan = self._bundle_and_plan(73, 97, seed=43)
        with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "0"}):
            streamed = render_output_u8(bundle, object(), "srgb", plan)
            cached = render_output_u8(
                bundle,
                object(),
                "srgb",
                plan,
                dither_noise=deterministic_dither_planes(
                    bundle.scene_rec2020_render.shape
                ),
            )
        np.testing.assert_array_equal(cached, streamed)

    def test_two_plane_quantizer_keeps_authoritative_operation_order(self) -> None:
        from dngscan.render import dither_quantize_u8_with_noise

        rng = np.random.default_rng(90210)
        encoded = rng.random((200_000, 3), dtype=np.float32)
        noise_a = rng.random(encoded.shape, dtype=np.float32)
        noise_b = rng.random(encoded.shape, dtype=np.float32)
        actual = dither_quantize_u8_with_noise(encoded, noise_a, noise_b)
        sequential = encoded * np.float32(255.0) + np.float32(0.5)
        sequential = sequential + noise_a
        sequential = sequential - noise_b
        expected = np.clip(np.floor(sequential), 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(actual, expected)

        # Prove why the cache may not collapse the two source planes.
        combined = np.clip(
            np.floor(
                encoded * np.float32(255.0)
                + np.float32(0.5)
                + (noise_a - noise_b)
            ),
            0,
            255,
        ).astype(np.uint8)
        self.assertGreater(int(np.count_nonzero(combined != expected)), 0)

    def test_native_failure_reuses_noise_for_exact_auto_fallback(self) -> None:
        import dngscan._fast as fast_backend

        bundle, plan = self._bundle_and_plan(43, 59, seed=37)
        with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "0"}):
            reference = render_output_u8(bundle, object(), "srgb", plan)
        with (
            mock.patch.dict(os.environ, {"DNGSCAN_FAST": "auto"}),
            mock.patch.object(
                fast_backend, "supports_output_finalizer", return_value=True
            ),
            mock.patch.object(
                fast_backend,
                "finalize_rec2020_u8_f32",
                side_effect=RuntimeError("synthetic native failure"),
            ),
        ):
            fallback = render_output_u8(bundle, object(), "srgb", plan)
        np.testing.assert_array_equal(fallback, reference)

    def _bundle_and_plan(self, h: int, w: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        scene = rng.integers(0, 65536, size=(h, w, 3), dtype=np.uint16)
        bundle = RawBundle(
            path=Path("synthetic.dng"),
            raw_image=np.zeros((2, 2), dtype=np.uint16),
            raw_colors=np.asarray([[0, 1], [3, 2]], dtype=np.uint8),
            xyz_render=np.zeros_like(scene),
            render_scale=65535.0,
            scene_rec2020_render=scene,
            scene_scale=65535.0,
            white_level=65535,
            black_levels=[0.0, 0.0, 0.0, 0.0],
            camera_wb=[1.0, 1.0, 1.0, 1.0],
            color_desc="RGBG",
            raw_pattern=[[0, 1], [3, 2]],
            camera_white_levels=[65535.0] * 4,
        )
        plan = ToneCompressionPlan(
            target_gamut="Rec2020",
            luma_p1=0.01, luma_p50=0.18, luma_p99=1.0, luma_p999=2.0,
            black_ev=-8.0, white_ev=5.0, dynamic_range_ev=13.0,
            contrast=3.0, toe_power=1.5, shoulder_power=2.9,
            chroma_p95=0.5, negative_rgb_pct=0.0, over_rgb_pct=0.0,
        )
        return bundle, plan

    def test_threaded_stream_path_matches_unthreaded(self) -> None:
        # The threaded two-worker pipeline is only taken above STREAM_THREAD_MIN_PIXELS;
        # shrink the thresholds so this test actually exercises that branch (multiple
        # render chunks per quantize group, plus a partial tail group). Dither noise is
        # consumed per quantize group, so with IDENTICAL constants the threaded and
        # unthreaded paths must be byte-identical — this isolates threading, ordering,
        # and the grouped flush (a dropped tail group would show up as zeros).
        import dngscan.render as render_mod

        bundle, plan = self._bundle_and_plan(60, 70)  # 4200 px
        saved = (
            render_mod.STREAM_THREAD_MIN_PIXELS,
            render_mod.STREAM_RENDER_CHUNK,
            render_mod.STREAM_QUANTIZE_CHUNK,
        )
        try:
            render_mod.STREAM_RENDER_CHUNK = 512
            render_mod.STREAM_QUANTIZE_CHUNK = 1_024
            render_mod.STREAM_THREAD_MIN_PIXELS = 10**9  # force single-pass
            unthreaded = render_output_u8(bundle, object(), "srgb", plan)
            render_mod.STREAM_THREAD_MIN_PIXELS = 1_000  # force threaded
            threaded = render_output_u8(bundle, object(), "srgb", plan)
        finally:
            (
                render_mod.STREAM_THREAD_MIN_PIXELS,
                render_mod.STREAM_RENDER_CHUNK,
                render_mod.STREAM_QUANTIZE_CHUNK,
            ) = saved
        np.testing.assert_array_equal(threaded, unthreaded)
        self.assertGreater(int(threaded.max()), 0)

    def test_complex_color_chunking_is_byte_identical(self) -> None:
        import dngscan.render as render_mod

        bundle, plan = self._bundle_and_plan(60, 70, seed=47)
        saved = (
            render_mod.STREAM_THREAD_MIN_PIXELS,
            render_mod.STREAM_RENDER_CHUNK,
            render_mod.STREAM_COMPLEX_COLOR_CHUNK,
            render_mod.STREAM_QUANTIZE_CHUNK,
        )
        try:
            render_mod.STREAM_RENDER_CHUNK = 512
            render_mod.STREAM_COMPLEX_COLOR_CHUNK = 256
            render_mod.STREAM_QUANTIZE_CHUNK = 1_024
            render_mod.STREAM_THREAD_MIN_PIXELS = 10**9
            serial = render_output_u8(
                bundle,
                object(),
                "srgb",
                plan,
                look="optic_warm_cyan",
            )
            render_mod.STREAM_THREAD_MIN_PIXELS = 1_000
            threaded = render_output_u8(
                bundle,
                object(),
                "srgb",
                plan,
                look="optic_warm_cyan",
            )
        finally:
            (
                render_mod.STREAM_THREAD_MIN_PIXELS,
                render_mod.STREAM_RENDER_CHUNK,
                render_mod.STREAM_COMPLEX_COLOR_CHUNK,
                render_mod.STREAM_QUANTIZE_CHUNK,
            ) = saved
        np.testing.assert_array_equal(threaded, serial)

    def test_misaligned_stream_chunking_raises(self) -> None:
        import dngscan.render as render_mod

        bundle, plan = self._bundle_and_plan(50, 40)  # 2000 px
        saved = (
            render_mod.STREAM_THREAD_MIN_PIXELS,
            render_mod.STREAM_RENDER_CHUNK,
            render_mod.STREAM_QUANTIZE_CHUNK,
        )
        try:
            render_mod.STREAM_THREAD_MIN_PIXELS = 1_000
            render_mod.STREAM_RENDER_CHUNK = 700  # 1024 % 700 != 0
            render_mod.STREAM_QUANTIZE_CHUNK = 1_024
            with self.assertRaises(ValueError):
                render_output_u8(bundle, object(), "srgb", plan)
        finally:
            (
                render_mod.STREAM_THREAD_MIN_PIXELS,
                render_mod.STREAM_RENDER_CHUNK,
                render_mod.STREAM_QUANTIZE_CHUNK,
            ) = saved


class SubsetGamutReportTest(unittest.TestCase):
    def test_plain_export_report_tolerates_gamut_subset(self) -> None:
        # A plain export (no --scan/--csv) analyzes only the output gamut. The report
        # path must not KeyError on the missing gamut names — this crashed every plain
        # CLI export after the JPEG was already written (exit 1, "error: 'sRGB'").
        from dataclasses import replace as dc_replace

        from dngscan.report import csv_row, print_report
        import io
        from contextlib import redirect_stdout

        bundle, plan = StreamRenderTest()._bundle_and_plan(20, 20)
        import dngscan.core as dg

        analysis, _, _ = dg.analyze(bundle, 4, diagnostics=False, gamut_names=("P3",))
        self.assertNotIn("sRGB", analysis.gamut_out_pct)
        row = csv_row(
            bundle, analysis, None, None, None, "", False, 0.0, plan, "p3",
            None, "none", 1.0, "none", 1.0,
        )
        self.assertEqual(row["gamut_out_srgb_bright_pct"], "")
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report(
                bundle, analysis, None, None, None, None, "", False, 0.0, plan,
                "p3", None, "none", 1.0, "none", 1.0,
            )
        self.assertIn("高亮色域越界", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
