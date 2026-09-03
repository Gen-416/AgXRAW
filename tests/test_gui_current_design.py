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
        self.assertIn('postJob("/clip-overlay",body,controller.signal)', PAGE)
        prep = PAGE[PAGE.index("async function preparePreview"):]
        prep = prep[: prep.index("function beginBusy")]
        self.assertIn("loadClipOverlay(body);", prep)
        # missing masks grey the toggle with the reason on screen
        self.assertIn('t.disabled=true;lab.classList.add("dim")', PAGE)
        self.assertIn("Core Image 解码没有逐像素 CFA 证据", PAGE)
        self.assertIn('clipOverlay:$("#clipOverlayToggle").checked', PAGE)

    def test_overlay_is_labelled_as_near_full_well_with_hard_clip_authority(self) -> None:
        # R5 item 2: the layer is the soft retreat mask (>= ~97% full well),
        # not the hard clip statistic; the label says so and the hard number
        # is shown next to it from the full-resolution analysis.
        self.assertIn(">RAW 满阱</label>", PAGE)
        self.assertNotIn(">RAW 过曝</label>", PAGE)
        self.assertIn("硬剪切 R ", PAGE)
        self.assertIn("≥97% 满阱", PAGE)

    def test_new_session_resets_the_layer_before_the_new_frame_lands(self) -> None:
        # R6 item 3: the old file's marks must not sit on the new frame while
        # the new layer is still in flight — the session start aborts, clears
        # and hides the layer.
        body = PAGE[PAGE.index("function beginPreviewSession"):]
        body = body[: body.index("function scheduleLivePreview")]
        self.assertIn("resetClipOverlay();", body)
        reset = PAGE[PAGE.index("function resetClipOverlay"):PAGE.index("function beginPreviewSession")]
        self.assertIn("clipOverlayAbort.abort()", reset)
        self.assertIn('img.style.display="none"', reset)
        self.assertIn("CLIP_OVERLAY={has:false,b64:null,pct:null};", reset)

    def test_overlay_fetch_is_latest_wins(self) -> None:
        # R5 item 3: bound to the issuing preview session and aborting its
        # predecessor, so a slow response for the previous file cannot land.
        body = PAGE[PAGE.index("async function loadClipOverlay"):]
        body = body[: body.index('$("#clipOverlayToggle").addEventListener')]
        self.assertIn("const session=PREVIEW_SESSION_ID;", body)
        self.assertIn("if(clipOverlayAbort)clipOverlayAbort.abort();", body)
        self.assertIn('postJob("/clip-overlay",body,controller.signal)', body)
        self.assertIn("session!==PREVIEW_SESSION_ID)return;", body)


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
        analysis = SimpleNamespace(
            clip_pct={0: 1.0, 1: 0.5, 2: 0.0, 3: 0.7},
            labels={0: "R", 1: "G1", 2: "B", 3: "G2"},
            cell_union_pct=1.25,
        )
        entry = SimpleNamespace(bundle=SimpleNamespace(clip_masks=masks), analysis=analysis)
        with mock.patch.object(service.PREVIEW_STORE, "get", return_value=entry), mock.patch.object(
            service, "parse_decoder", return_value=("libraw", "auto")
        ):
            out = service.clip_overlay(self._params())
        self.assertTrue(out["has_masks"])
        self.assertEqual((out["width"], out["height"]), (8, 6))
        self.assertAlmostEqual(out["mask_pct"]["r"], 25.0)
        self.assertAlmostEqual(out["mask_pct"]["any"], 25.0)
        self.assertEqual(out["mask_pct"]["g"], 0.0)
        # the authoritative hard-clip share rides alongside, greens averaged
        self.assertAlmostEqual(out["hard_clip_pct"]["r"], 1.0)
        self.assertAlmostEqual(out["hard_clip_pct"]["g"], 0.6)
        self.assertAlmostEqual(out["hard_clip_pct"]["b"], 0.0)
        self.assertAlmostEqual(out["hard_clip_pct"]["union"], 1.25)
        im = Image.open(io.BytesIO(base64.b64decode(out["overlay"])))
        self.assertEqual(im.mode, "RGBA")
        self.assertEqual(im.size, (8, 6))

    def test_missing_masks_report_has_masks_false(self) -> None:
        from dngscan.gui import service

        entry = SimpleNamespace(bundle=SimpleNamespace(clip_masks=None), analysis=None)
        with mock.patch.object(service.PREVIEW_STORE, "get", return_value=entry), mock.patch.object(
            service, "parse_decoder", return_value=("coreimage", "auto")
        ):
            out = service.clip_overlay(self._params())
        self.assertFalse(out["has_masks"])
        self.assertIsNone(out["overlay"])
        self.assertIsNone(out["hard_clip_pct"])


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


class AchievedHeadroomSemanticsTests(unittest.TestCase):
    def test_max_channel_headroom_is_not_diluted_by_other_channels(self) -> None:
        # R5 item 5: a single-channel highlight covering 0.016% of the pixels
        # (8 of 50,000) sits above p99.99 per pixel (max channel -> 3 EV) but
        # below it once H*W*3 is flattened (8 of 150,000 samples -> 0 EV).
        from dngscan.hdr_agx import achieved_headroom

        img = np.ones((200, 250, 3), dtype=np.float32) * 0.5
        img[0, 0:8, 0] = 8.0
        self.assertAlmostEqual(achieved_headroom(img), 3.0, places=6)
        flat = float(np.percentile(img, 99.99))
        self.assertLessEqual(flat, 1.0)  # the old semantics really said 0 EV
        self.assertEqual(achieved_headroom(np.ones((4, 4, 3), dtype=np.float32)), 0.0)


class PtcWindowLabelTests(unittest.TestCase):
    def test_sparse_ramp_labels_carry_the_window_actually_fitted(self) -> None:
        # R5 item 4: with <4 points below 0.10*S_sat the fit widens to 0.35
        # and every label/alternative key must say 0.35, not 0.10.
        import importlib.util

        spec = importlib.util.spec_from_file_location("import_jptc", ROOT / "tools" / "import_jptc.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rng = np.random.default_rng(3)
        black, white = 512.0, 16383.0
        gain = 4.0  # e-/DN
        cap = white - black
        # 2 points below 0.10*cap, the rest spread 0.12..0.90 of capacity
        fracs = np.concatenate([[0.03, 0.07], np.linspace(0.12, 0.90, 24)])
        signal = fracs * cap
        var = signal / gain + 2.0 ** 2  # shot noise + read noise (no PRNU)
        means = black + signal
        stds = np.sqrt(var)
        fit = mod.fit_ptc(means, stds, black, white)
        self.assertEqual(fit["fit_window_frac"], 0.35)
        self.assertIn("linear-0.35", fit["gain_alternatives"])
        self.assertNotIn("linear-0.10", fit["gain_alternatives"])
        self.assertIn("linear-0.35", fit["fit_relative_rms_alternatives"])
        self.assertIn("0.35", fit["fit_model"])
        self.assertNotIn("linear-0.10", fit["fit_model_effective"])


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


class EveryCliDialHasAGuiControl(unittest.TestCase):
    """Owner 2026-08-28: every adjustable CLI parameter is a GUI control.
    The page's payload must carry each render-affecting CLI dial (report,
    diagnostics and debug outputs excluded — the GUI produces images only)."""

    PAYLOAD_KEY_FOR_CLI = {
        "--coreimage-scale": "coreimageScale",
        "--margin": "clipMargin",
        "--chroma-nr": "chromaNr",
        "--film-development": "filmDevelopment",
        "--film-dev-contrast": "filmDevContrast",
        "--film-dev-fog": "filmDevFog",
        "--film-dev-density": "filmDevDensity",
        "--film-compression": "filmCompression",
        "--film-compression-knee": "filmCompressionKnee",
        "--film-highlight-density": "filmHighlightDensity",
        "--film-media-scatter": "filmMediaScatter",
        "--film-optics-seed": "filmOpticsSeed",
        "--film-interimage-beta": "filmInterimageBeta",
        "--film-appearance-variant": "filmAppearanceVariant",
        "--hdr-rho": "hdrRho",
        "--lens-filter": "lensFilter",
        "--endpoint-mode": "endpointMode",
    }

    def test_payload_carries_every_cli_dial(self) -> None:
        body = PAGE[PAGE.index("function payload("):]
        body = body[: body.index("return p;")]
        for flag, key in self.PAYLOAD_KEY_FOR_CLI.items():
            with self.subTest(flag=flag):
                self.assertIn(f"{key}:", body)

    def test_new_controls_exist_and_reset_outside_full(self) -> None:
        for cid in ("filmDevelopment", "filmDevContrast", "filmDevFog", "filmDevDensity", "filmCompression",
                    "filmCompressionKnee", "filmHighlightDensity", "filmMediaScatter", "filmOpticsSeed",
                    "coreimageScale", "clipMargin", "chromaNr"):
            self.assertIn(f'id="{cid}"', PAGE)
        reset = PAGE[PAGE.index("function updateFilmModeUi"):]
        reset = reset[: reset.index("const preset=$(\"#filmCurve\").value;")]
        self.assertIn('$("#filmDevelopment").value="measured_default";', reset)
        self.assertIn('$("#filmCompression").value=0;', reset)
        self.assertIn('$("#filmMediaScatter").value="declared";', reset)
        # editorial_custom couples with neutralization=native and disables retimed
        full = PAGE[PAGE.index("const devCustom=$(\"#filmDevelopment\").value===\"editorial_custom\";"):]
        full = full[: full.index("// 模拟光学")]
        self.assertIn("const canRetime=FILM_RETIMED.includes(preset)&&!devCustom;", full)
        self.assertIn('const forceNative=timing.value==="custom"||devCustom;', full)
        # the Core Image scale block follows the version block's visibility
        self.assertIn('scl.style.display=raw9?"":"none"', PAGE)


class ServiceDialParsingTests(unittest.TestCase):
    def _base(self):
        return {"filmCurve": "portra400", "filmMode": "full", "filmNeutralization": "native"}

    def test_development_and_compression_rules_mirror_the_cli(self) -> None:
        from dngscan.gui.service import parse_film_params

        t = parse_film_params(dict(self._base(), filmDevelopment="editorial_custom", filmDevContrast=0.2,
                                   filmCompression=0.5, filmCompressionKnee=3, filmHighlightDensity=1.0))
        self.assertEqual(t[22:29], ("editorial_custom", 0.2, 0.0, 0.0, 0.5, 3.0, 1.0))
        self.assertEqual(parse_film_params(self._base())[22:29], ("measured_default", 0.0, 0.0, 0.0, 0.0, 2.0, 0.0))
        for bad, why in (
            (dict(self._base(), filmDevContrast=0.1), "measured_default locks the deltas"),
            (dict(self._base(), filmDevelopment="editorial_custom", filmNeutralization="technical-neutral"), "needs native"),
            (dict(self._base(), filmDevelopment="editorial_custom", filmPrintTiming="retimed"), "no retimed"),
            (dict(self._base(), filmHighlightDensity=0.5), "needs compression"),
            (dict(filmCurve="portra400", filmMode="observe", filmCompression=0.3), "full only"),
            (dict(self._base(), filmCompressionKnee=7), "knee domain"),
        ):
            with self.subTest(why=why), self.assertRaises(ValueError):
                parse_film_params(bad)

    def test_decode_extras(self) -> None:
        from dngscan.gui.service import parse_decode_extras

        self.assertEqual(parse_decode_extras({"coreimageScale": "unity", "clipMargin": "8"}, "coreimage"), ("unity", 8))
        self.assertEqual(parse_decode_extras({"coreimageScale": "unity"}, "libraw"), ("aligned", 4))
        with self.assertRaises(ValueError):
            parse_decode_extras({"clipMargin": 99}, "libraw")
        with self.assertRaises(ValueError):
            parse_decode_extras({"coreimageScale": "bogus"}, "coreimage")

    def test_export_suffix_names_every_new_dial(self) -> None:
        from dngscan.gui.service import export_suffix_parts

        s = export_suffix_parts("clip", "p3", "sdr", film_mode="full", film_development="editorial_custom",
                                film_dev_contrast=-0.2, film_dev_fog=0.1, film_dev_density=0.3,
                                film_compression=0.5, film_compression_knee=2.0, film_highlight_density=1.0,
                                film_media_scatter="off", explicit_optics_seed=7, film_grain=0.5,
                                coreimage_scale="unity", clip_margin=8, decoder="coreimage")
        for tok in ("dev-c", "comp0_5k2hd1", "scatteroff", "seed7", "ciscale-unity", "margin8"):
            self.assertIn(tok, s)
        # R7 item 6: without grain the seed changes no pixel — no token
        no_grain = export_suffix_parts("clip", "p3", "sdr", film_mode="full", explicit_optics_seed=7)
        self.assertNotIn("seed", no_grain)
        plain = export_suffix_parts("clip", "p3", "sdr", film_mode="full")
        for tok in ("dev-", "comp", "scatteroff", "seed", "ciscale", "margin"):
            self.assertNotIn(tok, plain)

    def test_cache_identity_keeps_default_digests(self) -> None:
        from unittest import mock
        from dngscan.gui import preview_cache as pc

        with mock.patch.object(pc, "_evidence_cache_identity", return_value=("sig",)), \
             mock.patch.object(pc, "_scene_decoder_runtime_id", return_value="rt"):
            base = pc._cache_identity(ROOT / "README.md", "clip", "camera", "libraw", "auto", "auto")
            same = pc._cache_identity(ROOT / "README.md", "clip", "camera", "libraw", "auto", "auto", "aligned", 4)
            self.assertEqual(base, same)
            margin = pc._cache_identity(ROOT / "README.md", "clip", "camera", "libraw", "auto", "auto", "aligned", 8)
            self.assertNotEqual(base[1], margin[1])
            scale_libraw = pc._cache_identity(ROOT / "README.md", "clip", "camera", "libraw", "auto", "auto", "unity", 4)
            self.assertEqual(base, scale_libraw)  # the scale policy does not exist on LibRaw
            scale_ci = pc._cache_identity(ROOT / "README.md", "reconstruct", "camera", "coreimage", "auto", "auto", "unity", 4)
            scale_ci_default = pc._cache_identity(ROOT / "README.md", "reconstruct", "camera", "coreimage", "auto", "auto", "aligned", 4)
            self.assertNotEqual(scale_ci[1], scale_ci_default[1])


class GreyingGapWires(unittest.TestCase):
    def test_runtime_context_and_per_file_versions_grey_the_decoder(self) -> None:
        body = PAGE[PAGE.index("async function ensureRaw9Support"):]
        body = body[: body.index("if(j.raw9_supported)return true;")]
        self.assertIn("j.runtime_interactive===false", body)
        self.assertIn('o.disabled=!ok;o.title=ok?"":"此文件不提供 RAW "+o.value;', body)

    def test_core_deps_and_optics_assets_flags_are_baked_and_acted_on(self) -> None:
        self.assertIn("CORE_DEPS_MISSING_JSON", PAGE)
        self.assertIn("FILM_OPTICS_OK_FLAG", PAGE)
        self.assertIn('$("#go").disabled=true', PAGE)
        self.assertIn("胶片光学资产缺失或校验失败，接管模式不可用", PAGE)
        with mock.patch("dngscan.gui.page._core_import_errors", return_value=["rawpy: nope"]), \
             mock.patch("dngscan.gui.page._film_optics_assets_ok", return_value=False):
            html = render_page("/tmp").decode()
        self.assertIn('const CORE_DEPS_MISSING=["rawpy: nope"];', html)
        self.assertIn("const FILM_OPTICS_OK=false;", html)

    def test_raw9_probe_reports_runtime_contexts(self) -> None:
        from dngscan.gui import service

        probe = {"coreimage_available": True, "error": None, "raw9_supported": True,
                 "versions_offered": ["9", "8"], "fallback_version": None}
        with mock.patch("dngscan.coreimage_decode.probe_raw9_support", return_value=probe), \
             mock.patch("dngscan.coreimage_decode.runtime_available", return_value=False), \
             mock.patch("dngscan.decode_support.probe_decode_support", return_value={"lines": []}):
            out = service.raw9_support({"input": str(ROOT / "README.md")})
        self.assertFalse(out["runtime_interactive"])
        self.assertIn("运行时上下文不可用", out["message"])

    def test_save_settings_has_no_duplicate_keys(self) -> None:
        i = PAGE.index("function saveSettings("); j = PAGE.index("}));}catch(e){}", i)
        keys = re.findall(r"(?:^|[{,\n]\s*)([a-zA-Z]+):", PAGE[i:j])
        self.assertEqual(sorted(k for k in set(keys) if keys.count(k) > 1), [])


class ReviewR7Tests(unittest.TestCase):
    def test_headroom_estimate_forwards_every_film_dial(self) -> None:
        # R7 item 1: the exported "EV still safe" figure must come from the
        # SAME film chain as the export, dials included.
        from dngscan.gui import service

        seen = {}

        def fake_max_safe_ev(*args, **kwargs):
            seen.update(kwargs)
            return 1.0

        import inspect

        sig = inspect.signature(service.estimate_ev_headroom)
        defaults = {
            "bundle": SimpleNamespace(scene_rec2020_render=np.full((4, 4, 3), 0.18, dtype=np.float32), lens_filter="none"),
            "analysis": SimpleNamespace(),  # non-None: the function returns {} without an analysis
            "gamut": "p3", "current_ev": 0.0,
        }
        required = {
            name: defaults.get(name, 0.0 if p.annotation in (float, "float") else "none")
            for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        with mock.patch.object(service.dg, "max_safe_ev", side_effect=fake_max_safe_ev):
            service.estimate_ev_headroom(
                **required,
                film_development="editorial_custom", film_dev_contrast=0.2, film_dev_fog=0.1,
                film_dev_density=-0.3, film_compression=0.5, film_compression_knee=3.0,
                film_highlight_density=1.0,
            )
        for k, v in (("film_development", "editorial_custom"), ("film_dev_contrast", 0.2), ("film_dev_fog", 0.1),
                     ("film_dev_density", -0.3), ("film_compression", 0.5), ("film_compression_knee", 3.0),
                     ("film_highlight_density", 1.0)):
            self.assertEqual(seen.get(k), v, k)

    def test_peek_uses_the_same_identity_as_get(self) -> None:
        # R7 item 2: an export must find the preview's entry (and its grain
        # realization) under non-default decode dials too.
        from dngscan.gui import preview_cache as pc

        calls = []

        def fake_identity(*args):
            calls.append(args)
            return (("k",) + tuple(str(a) for a in args[1:]), "digest")

        store = pc.PreviewCache.__new__(pc.PreviewCache)
        store.entries = {}
        import threading
        store.lock = threading.Lock()
        with mock.patch.object(pc, "_cache_identity", side_effect=fake_identity):
            store.peek(ROOT / "README.md", "clip", "camera", False, "coreimage", "auto", "auto",
                       coreimage_scale="unity", margin=8)
        self.assertEqual(calls[-1][6:], ("unity", 8))
        with mock.patch.object(pc, "_cache_identity", side_effect=fake_identity):
            store.peek(ROOT / "README.md", "clip", "camera", False, "libraw", "auto", "auto",
                       coreimage_scale="unity", margin=8)
        self.assertEqual(calls[-1][6:], ("aligned", 8))

    def test_clip_margin_rejects_non_integers_like_the_cli(self) -> None:
        from dngscan.gui.service import parse_decode_extras

        with self.assertRaises(ValueError):
            parse_decode_extras({"clipMargin": 4.9}, "libraw")
        with self.assertRaises(ValueError):
            parse_decode_extras({"clipMargin": "4.9"}, "libraw")
        self.assertEqual(parse_decode_extras({"clipMargin": "6"}, "libraw"), ("aligned", 6))
        self.assertEqual(parse_decode_extras({"clipMargin": 6.0}, "libraw"), ("aligned", 6))

    def test_export_checks_the_export_runtime_context(self) -> None:
        body = PAGE[PAGE.index('$("#exportConfirm").onclick'):]
        body = body[: body.index('postJob("/export"')]
        self.assertIn("pj.runtime_export===false", body)
        self.assertIn("导出上下文", body)


class ColourHeadRangeTests(unittest.TestCase):
    def test_sliders_cover_the_working_band_with_an_opt_in_full_travel(self) -> None:
        # Owner 2026-08-28: 0-40 CC by default (the 2-10 CC fine band with
        # room), the 200 CC hardware travel behind a checkbox; a restored
        # value above 40 widens the range rather than being clamped.
        for cid in ("colorHeadY", "colorHeadM"):
            self.assertRegex(PAGE, rf'id="{cid}" min="0" max="40" step="5"')
        self.assertIn('id="colorHeadWide"', PAGE)
        body = PAGE[PAGE.index("function applyColorHeadRange"):]
        body = body[: body.index('$("#colorHeadWide").addEventListener')]
        self.assertIn("const max=wide?200:40;", body)
        self.assertIn("if(Number(el.value)>max){el.value=String(max);changed=true;}", body)
        self.assertIn('if(Math.max(Number($("#colorHeadY").value),Number($("#colorHeadM").value))>40)$("#colorHeadWide").checked=true;', PAGE)
        self.assertIn('colorHeadWide:$("#colorHeadWide").checked', PAGE)


class ResetDefaultsButtonTests(unittest.TestCase):
    def test_button_restores_served_defaults_and_reprepares(self) -> None:
        # Owner 2026-08-28: one button, page-initial values (snapshot taken
        # before localStorage replays), file path and folder kept, then a
        # fresh prepare because decode-level settings may have moved.
        self.assertIn('id="resetDefaults"', PAGE)
        i = PAGE.index("const PAGE_DEFAULTS=new Map();")
        self.assertLess(i, PAGE.index("restoreSettings();\n", i))
        self.assertIn('const RESET_KEEP=new Set(["input","outdir"]);', PAGE)
        body = PAGE[PAGE.index("function resetToDefaults"):PAGE.index('$("#resetDefaults").addEventListener')]
        self.assertIn("saveSettings();", body)
        self.assertIn('if($("#input").value.trim())preparePreview();', body)
        self.assertIn('"updateFilmModeUi"', body)
        self.assertIn('"updateDecoderUi"', body)
