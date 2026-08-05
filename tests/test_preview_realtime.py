# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import dngscan as dg
from dngscan.gui.constants import REALTIME_PREVIEW_LONG_EDGE
from dngscan.gui.preview_cache import PreviewEntry
from dngscan.gui.preview_scheduler import PREVIEW_COORDINATOR, PreviewCoordinator
from dngscan.gui.service import (
    _cached_render_plan,
    export_preview_jpeg,
    preview_b64_from_u8,
    run_preview,
)


class PreviewCoordinatorTests(unittest.TestCase):
    def test_newest_generation_supersedes_older_requests(self) -> None:
        coordinator = PreviewCoordinator()
        self.assertTrue(coordinator.register("session", 1))
        self.assertTrue(coordinator.is_current("session", 1))
        self.assertTrue(coordinator.register("session", 3))
        self.assertFalse(coordinator.is_current("session", 1))
        self.assertFalse(coordinator.register("session", 2))
        self.assertTrue(coordinator.is_current("session", 3))

    def test_legacy_generation_zero_does_not_change_session_state(self) -> None:
        coordinator = PreviewCoordinator()
        self.assertTrue(coordinator.register("session", 0))
        self.assertTrue(coordinator.is_current("session", 0))
        self.assertTrue(coordinator.register("session", 4))
        self.assertTrue(coordinator.is_current("session", 0))
        self.assertTrue(coordinator.is_current("session", 4))

    def test_run_preview_rejects_stale_generation_before_decode(self) -> None:
        PREVIEW_COORDINATOR.clear()
        with tempfile.NamedTemporaryFile(suffix=".dng") as source:
            PREVIEW_COORDINATOR.register("browser-session", 2)
            result = run_preview(
                {
                    "input": source.name,
                    "previewSession": "browser-session",
                    "generation": 1,
                }
            )
        self.assertEqual(
            result, {"ok": True, "superseded": True, "generation": 1}
        )

    def test_hdr_delivery_choice_does_not_restrict_sdr_preview_tone_core(self) -> None:
        PREVIEW_COORDINATOR.clear()
        entry = MagicMock()
        with tempfile.NamedTemporaryFile(suffix=".dng") as source, patch(
            "dngscan.gui.service.PREVIEW_STORE.get", return_value=entry
        ), patch(
            "dngscan.gui.service.export_preview_jpeg", return_value={"ok": True}
        ) as export:
            result = run_preview(
                {
                    "input": source.name,
                    "format": "ultrahdr-heic",
                    "toneCore": "neutral",
                    "previewSession": "hdr-neutral-preview",
                    "generation": 1,
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["generation"], 1)
        export.assert_called_once()
        self.assertEqual(export.call_args.kwargs["tone_core"], "neutral")
        self.assertEqual(export.call_args.args[2], "p3")


class PreviewPlanCacheTests(unittest.TestCase):
    def test_base_plan_is_reused_across_interactive_adjustments(self) -> None:
        bundle = MagicMock()
        bundle.lens_filter = "none"
        entry = PreviewEntry(bundle=bundle, analysis=MagicMock())
        base_plan = object()
        adjusted_plan = object()

        def compile_plan(adjustments: dg.RenderAdjustments | None) -> object:
            return _cached_render_plan(
                entry,
                bundle,
                "srgb",
                "none",
                1.0,
                1.0,
                "agx",
                "y",
                "base",
                "none",
                adjustments,
            )

        with patch(
            "dngscan.gui.service.dg.build_render_plan", return_value=base_plan
        ) as build, patch(
            "dngscan.gui.service.dg.apply_render_adjustments",
            return_value=adjusted_plan,
        ) as apply:
            self.assertIs(compile_plan(None), adjusted_plan)
            self.assertIs(
                compile_plan(dg.RenderAdjustments(midtone_brightness=0.5)),
                adjusted_plan,
            )

        build.assert_called_once()
        self.assertEqual(apply.call_count, 2)

    def test_fixed_resolution_constant_is_1920(self) -> None:
        self.assertEqual(REALTIME_PREVIEW_LONG_EDGE, 1920)

    def test_preview_jpeg_uses_high_quality_444_and_long_edge_fit(self) -> None:
        from PIL import Image, JpegImagePlugin

        portrait = dg.np.zeros((2400, 1600, 3), dtype=dg.np.uint8)
        encoded = base64.b64decode(preview_b64_from_u8(portrait, width=1920))
        with Image.open(io.BytesIO(encoded)) as image:
            self.assertEqual(image.size, (1280, 1920))
            self.assertEqual(JpegImagePlugin.get_sampling(image), 0)

    def test_metrics_variant_reuses_rendered_pixels(self) -> None:
        bundle = MagicMock()
        bundle.exposure_gain = 1.0
        bundle.scene_scale = 1.0
        bundle.scene_decoder = "libraw"
        bundle.scene_decoder_runtime = "test"
        bundle.scene_rec2020_render = dg.np.zeros((12, 18, 3), dtype=dg.np.uint16)
        # The realtime histograms run the real reliable-sample selection on this
        # bundle; give it honest scalar decode facts instead of MagicMock attrs.
        bundle.lens_filter = "none"
        bundle.wb_mode = "camera"
        bundle.camera_wb = None
        bundle.applied_wb = None
        bundle.daylight_wb = None
        bundle.clip_masks = None
        entry = PreviewEntry(bundle=bundle, analysis=MagicMock())
        pixels = dg.np.zeros((12, 18, 3), dtype=dg.np.uint8)
        with patch(
            "dngscan.gui.service.dg.with_intent_exposure", return_value=bundle
        ), patch(
            "dngscan.gui.service._cached_render_plan", return_value=MagicMock()
        ), patch(
            "dngscan.gui.service.dg.render_output_u8", return_value=pixels
        ) as render, patch(
            "dngscan.gui.service.dg.output_icc_profile_bytes", return_value=None
        ), patch(
            "dngscan.gui.service.preview_metrics_from_u8", return_value={"v": 1.0}
        ):
            deferred = export_preview_jpeg(
                Path("synthetic.dng"), "clip", "srgb", 0.0, 95,
                cached=entry, include_metrics=False,
            )
            measured = export_preview_jpeg(
                Path("synthetic.dng"), "clip", "srgb", 0.0, 95,
                cached=entry, include_metrics=True,
            )
        self.assertEqual(render.call_count, 1)
        self.assertFalse(deferred["pixel_cache_hit"])
        self.assertTrue(measured["pixel_cache_hit"])
        self.assertEqual(measured["metrics"], {"v": 1.0})


if __name__ == "__main__":
    unittest.main()
