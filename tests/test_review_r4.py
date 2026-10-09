# SPDX-License-Identifier: GPL-3.0-or-later
"""Gates for the R4 full-project self-review remediation.

Each test pins one of the review's verified findings so the defect class
cannot silently return. Source-level pins are used where the behaviour
lives in the GUI page's embedded JS.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PAGE = Path(__file__).resolve().parents[1] / "dngscan" / "gui" / "page.py"




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






class ServiceContractTests(unittest.TestCase):

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
        from dngscan.gui.page import PAGE as html

        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is unavailable")
        settings = "const SETTINGS_IDS=" + html.split("const SETTINGS_IDS=", 1)[1].split(
            "function restoreSettings(){", 1)[0]
        script = r'''
const STORE_KEY="settings";
const toneCore={type:"select-one",value:"agx",dataset:{librawValue:"gated"}};
const $=selector=>selector==="#toneCore"?toneCore:null;
let saved=null;
const localStorage={setItem:(key,value)=>{if(key!==STORE_KEY)throw Error("wrong key");saved=JSON.parse(value);}};
''' + settings + r'''
saveSettings();
if(saved.toneCore!=="gated")throw Error("decoder stash was not persisted");
delete toneCore.dataset.librawValue;
saveSettings();
if(saved.toneCore!=="agx")throw Error("visible tone core was not persisted");
'''
        result = subprocess.run([node, "-e", script], capture_output=True,
                                text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)



    def test_auto_fallback_keeps_the_request_for_server_capability_selection(self) -> None:
        body = self.src.split('async function ensureRaw9Support(body){', 1)[1].split('let DETECTED_READY', 1)[0]
        self.assertIn('if(body.coreimageVersion==="auto")return true;', body)
        self.assertNotIn('body.decoder="libraw"', body)
        self.assertNotIn('window.confirm', body)

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
