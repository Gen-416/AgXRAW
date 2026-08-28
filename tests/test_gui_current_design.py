# SPDX-License-Identifier: GPL-3.0-or-later
"""GUI review 2026-08-27: the page keeps up with the current design.

Wires pinned here (substring assertions against the served HTML, same crude
contract style as test_gui_page): the RAW over-exposure layer, the hidden
film-mode reset on the standalone curve select, the HDR latitude clamp, the
matplotlib gate on the dashboard checkbox, and the film copy that no longer
calls shipped features experimental. Plus unit tests for the overlay
builder, the service entry, and the export-side dashboard gate."""
from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dngscan._deps import np
from dngscan.gui.page import PAGE, render_page

ROOT = Path(__file__).resolve().parents[1]


class ClipOverlayPageWires(unittest.TestCase):
    def test_overlay_layer_and_toggle_exist_and_are_fetched_after_prepare(self) -> None:
        self.assertIn('<img id="clipOverlay"', PAGE)
        self.assertIn('id="clipOverlayToggle"', PAGE)
        self.assertIn('postJob("/clip-overlay",body)', PAGE)
        prep = PAGE[PAGE.index("async function preparePreview"):]
        prep = prep[: prep.index("function beginBusy")]
        self.assertIn("loadClipOverlay(body);", prep)
        # missing masks grey the toggle with the reason on screen
        self.assertIn('t.disabled=true;lab.classList.add("dim")', PAGE)
        self.assertIn("Core Image 解码没有逐像素 CFA 证据", PAGE)
        self.assertIn('clipOverlay:$("#clipOverlayToggle").checked', PAGE)


class FilmModeResetWire(unittest.TestCase):
    def test_curve_none_resets_the_hidden_takeover_declaration(self) -> None:
        body = PAGE[PAGE.index("function updateFilmModeUi"):]
        body = body[: body.index("filmWasFull=full;")]
        self.assertIn('if(!hasCurve&&$("#filmMode").value==="full")$("#filmMode").value="observe";', body)


class HdrLatitudeClampWire(unittest.TestCase):
    def test_number_inputs_clamp_to_their_domain_before_the_payload(self) -> None:
        block = PAGE[PAGE.index('for(const id of ["hdrRho","hdrWhiteMargin","hdrShoulderStart"]){$("#"+id).addEventListener("change"'):]
        block = block[: block.index("saveSettings();});}") + 20]
        self.assertIn("Math.min(hi,Math.max(lo,v))", block)
        self.assertIn("已钳到", block)


class MatplotlibGateWire(unittest.TestCase):
    def test_dashboard_checkbox_is_greyed_without_matplotlib(self) -> None:
        self.assertIn("MATPLOTLIB_AVAILABLE_FLAG", PAGE)
        self.assertIn("if(!MATPLOTLIB_AVAILABLE){", PAGE)
        self.assertIn("png.disabled=true", PAGE)
        with mock.patch("dngscan.gui.page._dashboard_import_errors", return_value=["matplotlib: nope"]):
            html = render_page("/tmp").decode()
        self.assertIn("const MATPLOTLIB_AVAILABLE=false;", html)
        with mock.patch("dngscan.gui.page._dashboard_import_errors", return_value=[]):
            html = render_page("/tmp").decode()
        self.assertIn("const MATPLOTLIB_AVAILABLE=true;", html)


class FilmCopyMatchesShippedDesign(unittest.TestCase):
    def test_shipped_film_features_are_not_labelled_experimental(self) -> None:
        film = PAGE[PAGE.index('id="mobileImagingCard"'):PAGE.index('id="colorPanel"')]
        self.assertNotIn("实验", film)
        self.assertNotIn("试点", film)
        self.assertNotIn("q(0)", film)
        self.assertIn("接管 · 胶片显影链", film)
        self.assertIn("固定 · 默认", film)

    def test_film_tooltips_are_basic(self) -> None:
        film = PAGE[PAGE.index('id="mobileImagingCard"'):PAGE.index('id="colorPanel"')]
        titles = re.findall(r'title="([^"]*)"', film)
        self.assertTrue(titles)
        longest = max(titles, key=len)
        self.assertLessEqual(len(longest), 120, f"tooltip too long for the GUI (belongs in docs): {longest[:80]}…")

    def test_retimed_presets_are_those_with_a_print_asset_on_disk(self) -> None:
        html = render_page("/tmp").decode()
        m = re.search(r"const FILM_RETIMED=(\[.*?\]);", html)
        self.assertIsNotNone(m)
        import json
        listed = set(json.loads(m.group(1)))
        v2 = ROOT / "dngscan" / "data" / "film_v2"
        expected = set()
        for npz in v2.glob("*.npz"):
            if npz.name.startswith(("print__", "b2__")):
                continue
            with np.load(npz, allow_pickle=False) as z:
                if str(np.asarray(z.get("kind", ""))) != "stock" or bool(z["reversal"]):
                    continue
            if any(v2.glob(f"print__{npz.stem}__*.npz")):
                expected.add(npz.stem)
        self.assertEqual(listed, expected)
        self.assertGreater(len(expected), 0)


class ClipOverlayBuilderTests(unittest.TestCase):
    def test_marker_colours_follow_the_clipped_channel_set(self) -> None:
        from dngscan.gui.service import clip_overlay_rgba

        masks = np.zeros((2, 3, 3), dtype=np.float32)
        masks[0, 0, 0] = 1.0            # R only
        masks[0, 1, :] = 1.0            # all three
        masks[1, 2, 1] = 0.49           # below threshold: not clipped
        rgba = clip_overlay_rgba(masks)
        self.assertEqual(rgba.shape, (2, 3, 4))
        self.assertEqual(int(rgba[0, 0, 0]), 255)
        self.assertGreater(int(rgba[0, 0, 3]), 0)
        self.assertLess(int(rgba[0, 0, 1]), 255)
        self.assertEqual(tuple(int(v) for v in rgba[0, 1, :3]), (255, 255, 255))
        self.assertEqual(int(rgba[1, 2, 3]), 0)
        self.assertEqual(int(rgba[1, 0, 3]), 0)

    def test_no_clipping_returns_none(self) -> None:
        from dngscan.gui.service import clip_overlay_rgba

        self.assertIsNone(clip_overlay_rgba(np.zeros((4, 4, 3), dtype=np.float32)))


class ClipOverlayServiceTests(unittest.TestCase):
    def _params(self):
        return {"input": str(ROOT / "README.md"), "wb": "camera", "decoder": "libraw"}

    def test_masks_become_a_png_layer_with_percentages(self) -> None:
        from dngscan.gui import service
        from PIL import Image

        masks = np.zeros((6, 8, 3), dtype=np.float16)
        masks[:3, :4, 0] = 1.0
        entry = SimpleNamespace(bundle=SimpleNamespace(clip_masks=masks))
        with mock.patch.object(service.PREVIEW_STORE, "get", return_value=entry), mock.patch.object(
            service, "parse_decoder", return_value=("libraw", "auto")
        ):
            out = service.clip_overlay(self._params())
        self.assertTrue(out["has_masks"])
        self.assertEqual((out["width"], out["height"]), (8, 6))
        self.assertAlmostEqual(out["clip_pct"]["r"], 25.0)
        self.assertAlmostEqual(out["clip_pct"]["any"], 25.0)
        self.assertEqual(out["clip_pct"]["g"], 0.0)
        im = Image.open(io.BytesIO(base64.b64decode(out["overlay"])))
        self.assertEqual(im.mode, "RGBA")
        self.assertEqual(im.size, (8, 6))

    def test_missing_masks_report_has_masks_false(self) -> None:
        from dngscan.gui import service

        entry = SimpleNamespace(bundle=SimpleNamespace(clip_masks=None))
        with mock.patch.object(service.PREVIEW_STORE, "get", return_value=entry), mock.patch.object(
            service, "parse_decoder", return_value=("coreimage", "auto")
        ):
            out = service.clip_overlay(self._params())
        self.assertFalse(out["has_masks"])
        self.assertIsNone(out["overlay"])


class ExportDashboardGateTests(unittest.TestCase):
    def test_export_gates_matplotlib_before_the_full_analysis(self) -> None:
        from dngscan.gui import service

        seen = {}

        def fake_require(*, dashboard=False):
            seen["dashboard"] = dashboard
            raise RuntimeError("stop here")

        params = {"input": str(ROOT / "README.md"), "png": True, "wb": "camera"}
        with mock.patch.object(service.dg, "require_dependencies", side_effect=fake_require):
            with self.assertRaisesRegex(RuntimeError, "stop here"):
                service.run_export(params)
        self.assertTrue(seen.get("dashboard"))


class CliReportOnDemandTests(unittest.TestCase):
    def test_report_flag_defaults_off(self) -> None:
        from dngscan.cli import parse_args

        self.assertFalse(parse_args(["photo.dng", "--jpeg", "out.jpg"]).report)
        self.assertTrue(parse_args(["photo.dng", "--jpeg", "out.jpg", "--report"]).report)

    def test_plain_conversion_prints_only_the_written_file(self) -> None:
        samples = Path.home() / "Pictures" / "AgXRAW样张"
        raws = sorted(samples.glob("*.DNG")) + sorted(samples.glob("*.dng"))
        if not raws:
            self.skipTest("no sample RAW available")
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plain.jpg"
            env = dict(os.environ, DNGSCAN_FAST="0")
            res = subprocess.run(
                [sys.executable, "-m", "dngscan", str(raws[0]), "--jpeg", str(out), "--jpeg-quality", "70"],
                capture_output=True, text=True, env=env, cwd=ROOT, timeout=600,
            )
            self.assertEqual(res.returncode, 0, res.stderr[-800:])
            lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
            self.assertTrue(any(ln.startswith("JPEG 图像:") for ln in lines), res.stdout)
            self.assertNotIn("色彩矩阵:", res.stdout)
            res2 = subprocess.run(
                [sys.executable, "-m", "dngscan", str(raws[0]), "--jpeg", str(out), "--jpeg-quality", "70", "--report"],
                capture_output=True, text=True, env=env, cwd=ROOT, timeout=600,
            )
            self.assertEqual(res2.returncode, 0, res2.stderr[-800:])
            self.assertIn("色彩矩阵:", res2.stdout)


if __name__ == "__main__":
    unittest.main()
