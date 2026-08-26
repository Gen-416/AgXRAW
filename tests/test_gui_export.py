# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for GUI export naming."""

from __future__ import annotations

import unittest

from dngscan._deps import np
from dngscan.gui.page import render_page
from dngscan.gui.preview_cache import downsample_mean
from dngscan.gui.service import export_plan_fingerprint, export_suffix_parts


class ExportSuffixTests(unittest.TestCase):
    def test_proxy_downsample_reaches_requested_long_edge(self) -> None:
        source = np.zeros((303, 202, 3), dtype=np.uint16)
        proxy = downsample_mean(source, 128)
        self.assertEqual(proxy.shape, (128, 85, 3))

    def test_public_gui_is_concise_and_has_no_vendor_luts(self) -> None:
        html = render_page("/tmp").decode("utf-8")
        self.assertNotIn("更新预览", html)
        self.assertIn('<button class="go" id="go">导出</button>', html)
        self.assertNotIn(">导出 JPEG</button>", html)
        self.assertIn("前馈校正", html)
        self.assertIn("中间调亮度", html)
        self.assertIn("中间调对比", html)
        self.assertIn("暗部过渡", html)
        self.assertIn("高光过渡", html)
        self.assertIn("高光褪白", html)
        self.assertIn("中频纯度", html)
        self.assertIn("HDR gain-map · JPEG", html)
        self.assertIn("HDR gain-map · HEIC", html)
        self.assertIn("只恢复漫反射白以上的真实亮度档数", html)
        self.assertIn("/raw9-support", html)
        self.assertIn("此文件不支持 RAW 9", html)
        self.assertIn('type="file" id="filePicker"', html)
        self.assertIn('accept=".3fr,.arw,.cr2,.cr3,.dcr,.dng', html)
        self.assertIn('apiFetch("/upload?name="', html)
        self.assertNotIn("文件只传给本机", html)
        self.assertNotIn("filePickerHint", html)
        self.assertNotIn('id="browseBtn"', html)
        self.assertNotIn('id="browser"', html)
        self.assertNotIn('optgroup label="本地 LUT"', html)
        # Vendor display LUTs must never leak into the public GUI. Named film
        # observation presets ("Kodak Portra 400", "Fujifilm Superia X-TRA 400") are
        # NOT vendor LUTs — they are dngscan's own calibrated declarations fitted from
        # published datasheet data — so the guard targets LUT product names, not the
        # manufacturers whose stocks the film feature legitimately names.
        # "2383" left this list in P3: the print-MEDIUM selector legitimately
        # names Kodak 2383/2393 as dngscan's own calibrated print declarations
        # (the same carve-out as the film stock names); the guard keeps
        # targeting LUT product names and .cube payloads.
        for vendor_lut in ("ARRI Classic", "ARRI Reveal", "RED IPP2", "LC-709", ".cube"):
            self.assertNotIn(vendor_lut, html)
        self.assertIn("Kodak Portra 400", html)
        self.assertIn("Fujifilm Superia X-TRA 400", html)

    def test_default_agx_only(self) -> None:
        self.assertEqual(export_suffix_parts("clip", "srgb", "sdr"), "agx")

    def test_plan_fingerprint_separates_renders_the_suffix_cannot(self) -> None:
        base = dict(
            wb="camera", ev=0.0, highlight="clip", gamut="p3", output_format="sdr",
            film_curve="portra400", film_mode="observe", lens_filter="none",
            adjustments=(0.0,) * 7,
        )
        same = export_plan_fingerprint(**base)
        self.assertEqual(same, export_plan_fingerprint(**base))
        self.assertEqual(len(same), 6)
        for key, value in (
            ("film_curve", "velvia100"),   # 不同曲线预设的 observe 输出
            ("wb", "5500k"),
            ("ev", 0.7),
            ("lens_filter", "85b"),
            ("adjustments", (0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ):
            changed = dict(base); changed[key] = value
            self.assertNotEqual(
                same, export_plan_fingerprint(**changed),
                f"fingerprint must change with {key}",
            )

    def test_export_call_site_fingerprints_every_encode_parameter(self) -> None:
        """Batch-11: the fingerprint FUNCTION hashes whatever it is given —
        the overwrite hole was the CALL SITE omitting parameters that change
        the written bytes (hdr_headroom, delivery, quality, chroma). Pin the
        actual keyword set with the AST so an omitted key fails here instead
        of silently colliding filenames."""
        import ast, inspect
        import dngscan.gui.service as service

        tree = ast.parse(inspect.getsource(service))
        keyword_sets = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "export_plan_fingerprint"
            ):
                keyword_sets.append({k.arg for k in node.keywords if k.arg})
        self.assertTrue(keyword_sets, "fingerprint call site not found")
        required = {
            "wb", "ev", "highlight", "decoder", "coreimage_version", "demosaic",
            "gamut", "output_format", "grade", "grade_strength",
            "scene_transform", "scene_transform_strength", "punch_scale",
            "tone_core", "lum_norm", "agx_primaries", "endpoint_mode",
            "lens_filter", "film_curve", "film_mode", "film_crossover",
            "color_head_y", "color_head_m", "adjustments",
            "hdr_headroom", "delivery", "quality", "chroma",
            "film_exposure_ev", "film_print_timing",
            "film_print_medium", "film_print_exposure_ev",
        }
        for kws in keyword_sets:
            self.assertTrue(
                required <= kws,
                f"fingerprint call missing keys: {sorted(required - kws)}",
            )

    def test_nondefault_primaries_path_is_named(self) -> None:
        self.assertEqual(
            export_suffix_parts("clip", "srgb", "sdr", agx_primaries="smooth"),
            "agx_smooth",
        )
        self.assertEqual(
            export_suffix_parts("clip", "srgb", "sdr", tone_core="gated", agx_primaries="base"),
            "gated",
        )

    def test_includes_grade(self) -> None:
        self.assertEqual(
            export_suffix_parts("clip", "srgb", "sdr", "look:optic_warm_cyan", 1.0),
            "agx_look_optic_warm_cyan",
        )
        self.assertEqual(
            export_suffix_parts("clip", "srgb", "sdr", "filter:kodak_2383_d65", 1.0),
            "agx_filter_kodak_2383_d65",
        )

    def test_includes_grade_strength_when_not_one(self) -> None:
        self.assertEqual(
            export_suffix_parts("clip", "srgb", "sdr", "look:optic_warm_cyan", 0.8),
            "agx_look_optic_warm_cyan_gs0.8",
        )

    def test_includes_scene_transform(self) -> None:
        self.assertEqual(
            export_suffix_parts("clip", "p3", "sdr", "none", 1.0, "arri_skin_d55", 0.75),
            "agx_p3_arri_skin_d55_st0.75",
        )

    def test_neutral_export_suffix(self) -> None:
        self.assertEqual(
            export_suffix_parts("clip", "srgb", "sdr", tone_core="neutral"),
            "neutral",
        )
        self.assertEqual(
            export_suffix_parts("clip", "srgb", "sdr", tone_core="lum"),
            "lum",
        )
        self.assertEqual(
            export_suffix_parts("clip", "srgb", "sdr", tone_core="lum", lum_norm="power"),
            "lum_power",
        )


class MetricsSamplingTests(unittest.TestCase):
    """D10: the post-export display metrics run on a declared ~800k stride sample."""

    def test_small_frames_stay_exact_and_report_their_count(self) -> None:
        import numpy as np
        from dngscan.gui.service import METRICS_SAMPLE_TARGET, output_luminance_metrics_u8

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        metrics = output_luminance_metrics_u8(img, "srgb", 0.0)
        self.assertEqual(metrics["metrics_sample_px"], 64 * 64)
        self.assertLess(64 * 64, METRICS_SAMPLE_TARGET)

    def test_large_frames_sample_within_the_declared_precision(self) -> None:
        import numpy as np
        from dngscan.gui.service import METRICS_SAMPLE_TARGET, output_luminance_metrics_u8

        rng = np.random.default_rng(11)
        # 3.2M px with realistic structure: smooth ramp + noise + a bright patch.
        h, w = 1600, 2000
        ramp = np.linspace(20, 200, w, dtype=np.float32)[None, :, None]
        img = np.clip(
            ramp + rng.normal(0, 12, size=(h, w, 3)), 0, 255
        ).astype(np.uint8)
        img[:64, :64] = 254
        sampled = output_luminance_metrics_u8(img, "srgb", 0.0)
        self.assertLessEqual(sampled["metrics_sample_px"], 2 * METRICS_SAMPLE_TARGET)
        self.assertLess(sampled["metrics_sample_px"], h * w)
        # Full-frame reference computed inline by defeating the stride via a
        # reshape into a single already-small-enough axis is not possible, so
        # compare against numpy directly on the exact same definition.
        flat = img.reshape(-1, 3).astype(np.float32) / np.float32(255.0)
        from dngscan.color import srgb_decode
        import dngscan as dg

        linear = srgb_decode(flat)
        matrix = dg.RGB_TO_XYZ[dg.output_gamut_space("srgb")]
        y = np.clip(linear @ np.asarray(matrix[1], dtype=np.float32), 0.0, 1.0)
        self.assertLess(abs(sampled["median_luma_pct"] - float(np.median(y)) * 100.0), 0.05)
        self.assertLess(abs(sampled["mean_luma_pct"] - float(np.mean(y)) * 100.0), 0.05)
        self.assertLess(
            abs(sampled["luma_p999_pct"] - float(np.percentile(y, 99.9)) * 100.0), 0.25
        )


if __name__ == "__main__":
    unittest.main()
