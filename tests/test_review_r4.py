# SPDX-License-Identifier: GPL-3.0-or-later
"""Gates for the R4 full-project self-review remediation.

Each test pins one of the review's verified findings so the defect class
cannot silently return. Source-level pins are used where the behaviour
lives in the GUI page's embedded JS.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PAGE = Path(__file__).resolve().parents[1] / "dngscan" / "gui" / "page.py"


def _film_plan(**kw) -> SimpleNamespace:
    base = dict(
        curve_preset="portra400", film_mode="full", film_crossover="datasheet",
        film_exposure_ev=0.0, film_print_timing="fixed",
        film_print_medium="", film_print_exposure_ev=0.0,
        color_head_y=0.0, color_head_m=0.0,
        film_development="measured_default",
        film_dev_contrast=0.0, film_dev_fog=0.0, film_dev_density=0.0,
        film_compression=0.0, film_compression_knee=2.0,
        film_highlight_density=0.0,
        film_grain=0.0, film_halation=0.0, film_bloom=0.0,
        film_optics_seed=0, film_media_scatter="declared",
        film_interimage="declared", film_appearance="technical",
        film_appearance_strength=1.0, film_appearance_variant="reference",
        film_richness=0.0, film_color_density=0.0, film_neutral_bias=1.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class SnrCurveSerializationTests(unittest.TestCase):
    """F1: analyses always carry ndarray SNR curves now; the preview disk
    cache must round-trip them instead of dying in json.dumps."""

    def test_write_disk_entry_serializes_a_real_analysis(self) -> None:
        """End-to-end: a real analyze() product must pass json.dumps via
        _analysis_to_json (the exact call that crashed cold loads)."""
        from dngscan.gui import preview_cache as pc
        from tests.golden_support import build_daylight_wide_dr

        scene = build_daylight_wide_dr()
        payload = pc._analysis_to_json(scene.analysis)
        text = json.dumps(payload, allow_nan=True)
        back = pc._analysis_from_json(json.loads(text))
        for group, curve in scene.analysis.snr_curves.items():
            np.testing.assert_allclose(
                back.snr_curves[group]["snr_db"],
                np.asarray(curve["snr_db"], dtype=np.float32),
                rtol=0, atol=1e-6, equal_nan=True,
            )


class GuidanceProxyBundleTests(unittest.TestCase):
    """F2/F5: the resolved-fullwell upgrade must not demand a mosaic the
    cache-proxy bundle no longer has."""

    def test_proxy_bundle_keeps_cached_guidance(self) -> None:
        from dngscan.guidance import ensure_raw_guidance

        cached_maps = object()
        bundle = SimpleNamespace(
            raw_image=None,
            raw_colors=None,
            clip_masks=np.zeros((4, 4, 3), dtype=np.float16),
            raw_guidance=cached_maps,
            _raw_guidance_has_sensor_snr=False,
            _raw_guidance_has_resolved_fullwell=False,
        )
        analysis = SimpleNamespace(
            channel_fullwell={0: 16000, 1: 16000, 2: 16000},
            gain_e_per_dn=None, prior_read_noise_e=None,
        )
        got = ensure_raw_guidance(bundle, analysis)
        self.assertIs(got, cached_maps)

    def test_build_maps_without_mosaic_returns_existing(self) -> None:
        from dngscan.guidance import build_raw_guidance_maps

        cached_maps = object()
        bundle = SimpleNamespace(
            raw_image=None,
            clip_masks=np.zeros((4, 4, 3), dtype=np.float16),
            raw_guidance=cached_maps,
        )
        self.assertIs(build_raw_guidance_maps(bundle, None), cached_maps)

    def test_upgrade_flag_is_a_declared_bundle_field(self) -> None:
        import dataclasses

        from dngscan.models import RawBundle

        names = {f.name for f in dataclasses.fields(RawBundle)}
        self.assertIn("_raw_guidance_has_resolved_fullwell", names)


class ScatterHaloSupportTests(unittest.TestCase):
    """P3: the halo bound must cover the small-sigma kernel's hard +-2 tap
    support, not just ceil(3 sigma)."""

    def test_small_sigma_stage_declares_two_rows(self) -> None:
        from dngscan.film_optics import _scatter_components, scatter_halo_px
        from dngscan.film_optics_assets import (
            DEFAULT_PRINT_OPTICS, DEFAULT_STOCK_OPTICS,
            load_print_optics, load_stock_optics,
        )

        stock = load_stock_optics(DEFAULT_STOCK_OPTICS)
        medium = load_print_optics(DEFAULT_PRINT_OPTICS)
        kernels = (stock.emulsion_scatter, medium.formation_scatter)
        # 43 um/px: the pitch the review measured a 4.0e-4 seam at — every
        # live component is sub-pixel, so each active stage must account the
        # 5-tap kernel's support of 2, never ceil(3*sigma)=1.
        mm_per_px = 0.043
        total = 0
        for kernel in kernels:
            if kernel is None:
                continue
            live = [
                scale
                for ch in range(3)
                for scale, _w in _scatter_components(kernel, ch, mm_per_px)
            ]
            if not live:
                continue
            self.assertTrue(all(s < 1.0 for s in live))
            total += 2
        self.assertGreater(total, 0)
        self.assertEqual(scatter_halo_px(kernels, mm_per_px), total)


class AutoEvProbeSamplingTests(unittest.TestCase):
    """F4: the decimated probe (which dilutes point speculars ~cell-area x)
    is reserved for real look amounts; the scatter-only default keeps the
    strided real-pixel probe."""

    def test_scatter_only_plan_keeps_strided_probe(self) -> None:
        import inspect

        from dngscan import auto_ev

        src = inspect.getsource(auto_ev.max_safe_ev)
        self.assertIn("_probe_needs_spatial", src)
        self.assertNotIn("_film_spatial_engaged(probe_tone)", src)


class ServiceContractTests(unittest.TestCase):
    def test_film_full_without_curve_is_rejected(self) -> None:
        from dngscan.gui.service import parse_film_params

        with self.assertRaises(ValueError):
            parse_film_params({"filmMode": "full"})
        # observe without curve stays legal
        out = parse_film_params({"filmMode": "observe"})
        self.assertEqual(out[2], "observe")

    def test_export_demosaic_is_validated(self) -> None:
        import inspect

        from dngscan.gui import service

        src = inspect.getsource(service.run_export)
        self.assertIn("parse_demosaic(params, decoder)", src)
        self.assertNotIn('demosaic = str(params.get("demosaic', src)


class ReportHonestyTests(unittest.TestCase):
    def test_policy_line_reports_actual_chroma(self) -> None:
        from dngscan.report import jpeg_policy_cn

        self.assertIn("4:2:0", jpeg_policy_cn("agx", "p3", chroma="420"))
        self.assertIn("4:4:4", jpeg_policy_cn("agx", "p3", chroma="444"))

    def test_film_line_discloses_scatter_only_default(self) -> None:
        from dngscan.report import jpeg_tone_plan_cn

        plan = _film_plan()
        line = jpeg_tone_plan_cn(None, None, "agx", plan, "p3")
        self.assertIn("模拟光学", line)
        self.assertIn("介质散射=declared", line)
        self.assertIn("无观感量", line)
        off = jpeg_tone_plan_cn(
            None, None, "agx", _film_plan(film_media_scatter="off"), "p3"
        )
        self.assertNotIn("模拟光学", off)


class CliGamutContractTests(unittest.TestCase):
    def test_explicit_srgb_with_hdr_is_refused(self) -> None:
        from dngscan.cli import parse_args

        with self.assertRaises(SystemExit):
            parse_args([
                "x.dng", "--jpeg", "o.jpg", "--output-format", "ultrahdr",
                "--output-gamut", "srgb",
            ])

    def test_defaults_resolve_per_format(self) -> None:
        from dngscan.cli import parse_args

        sdr = parse_args(["x.dng", "--jpeg", "o.jpg"])
        self.assertEqual(sdr.output_gamut, "srgb")
        hdr = parse_args(["x.dng", "--jpeg", "o.jpg", "--output-format", "ultrahdr"])
        self.assertEqual(hdr.output_gamut, "p3")
        explicit = parse_args([
            "x.dng", "--jpeg", "o.jpg", "--output-format", "ultrahdr",
            "--output-gamut", "p3",
        ])
        self.assertEqual(explicit.output_gamut, "p3")


class GuiSourcePins(unittest.TestCase):
    """The page is one embedded JS string; these pins keep the R4 GUI fixes
    from silently regressing (same technique as test_gui_guards)."""

    def setUp(self) -> None:
        self.src = PAGE.read_text(encoding="utf-8")

    def test_tone_core_change_clears_decoder_stash(self) -> None:
        self.assertIn('delete $("#toneCore").dataset.librawValue;', self.src)

    def test_save_settings_persists_stashed_tone_core(self) -> None:
        self.assertIn(
            'toneCore:$("#toneCore").dataset.librawValue||$("#toneCore").value',
            self.src,
        )

    def test_film_deselect_resets_mode(self) -> None:
        self.assertIn('$("#filmMode").value="observe";', self.src)

    def test_restore_settings_accepts_custom_timing(self) -> None:
        self.assertIn('["fixed","retimed","custom"].includes(s.filmPrintTiming)', self.src)

    def test_libraw_fallback_rewrites_body_from_dom(self) -> None:
        self.assertIn('body.highlight=$("#highlight").value;', self.src)
        self.assertIn('body.demosaic=$("#demosaic").value;', self.src)

    def test_preview_metrics_run_before_annotation(self) -> None:
        from pathlib import Path as _P

        service_src = (
            _P(__file__).resolve().parents[1] / "dngscan" / "gui" / "service.py"
        ).read_text(encoding="utf-8")
        metrics_at = service_src.index(
            "metrics = preview_metrics_from_u8(rgb_u8, gamut) if include_metrics"
        )
        annotate_at = service_src.index(
            "rgb_u8 = annotate_preview_rgb_u8(rgb_u8, dg.auto_ev_overlay_lines(auto_ev))"
        )
        self.assertLess(metrics_at, annotate_at)


if __name__ == "__main__":
    unittest.main()
