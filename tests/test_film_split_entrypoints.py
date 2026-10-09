# SPDX-License-Identifier: GPL-3.0-or-later
"""The active application has no dependencies on the archived film feature."""
from __future__ import annotations

import contextlib
import io
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from dngscan.cli import parse_args
from dngscan.gui import service
from dngscan.gui.page import PAGE, render_page


class FilmSplitEntrypointsTests(unittest.TestCase):
    def test_cli_rejects_removed_film_options_and_keeps_raw_defaults(self):
        defaults = parse_args(["photo.dng"])
        self.assertEqual((defaults.tone_core, defaults.wb, defaults.scene_transform),
                         ("agx", "camera", "none"))
        for flags in (("--film", "portra400"), ("--film-mode", "full"),
                      ("--color-head-y", "5")):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as error:
                parse_args(["photo.dng", *flags])
            self.assertEqual(error.exception.code, 2)

    def test_stale_gui_film_requests_fail_with_migration_guidance(self):
        for key in ("filmCurve", "film_mode", "colorHeadY", "color_head_m"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "AgXFilm"):
                service.parse_job_params({"input": "photo.dng", key: "none"})

    def test_removed_controls_have_no_remaining_dom_references(self):
        html = render_page("/tmp", "test-token").decode()
        ids = set(re.findall(r'\bid="([^"]+)"', html))
        references = set(re.findall(r'\$\("#([\w-]+)"\)', html))
        self.assertFalse(references - ids, f"missing controls: {references - ids}")
        self.assertFalse(any(name.startswith(("film", "colorHead")) for name in ids))
        for required in ("toneCore", "lensFilter", "chromaNr", "calibrationDialog",
                         "decoder", "heifBitDepth", "hdrHeadroom"):
            self.assertIn(required, ids)
        payload = PAGE.split("function payload(", 1)[1].split("return p;", 1)[0]
        self.assertNotRegex(payload, r"\b(?:film\w*|colorHead\w*):")

    def test_served_javascript_parses_after_panel_removal(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is unavailable")
        html = render_page("/tmp", "test-token").decode()
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gui.js"
            path.write_text(script)
            result = subprocess.run([node, "--check", str(path)], capture_output=True,
                                    text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_v10_settings_keep_explicit_output_choices_without_film_controls(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is unavailable")
        ids = PAGE.split("const SETTINGS_IDS=[", 1)[1].split("];", 1)[0]
        restore = "function restoreSettings(){" + PAGE.split(
            "function restoreSettings(){", 1)[1].split('["quality","outdir","png"]', 1)[0]
        script = "const SETTINGS_IDS=[" + ids + "];\n" + r'''
const STORE_KEY="v11",V10_STORE_KEY="v10",V9_STORE_KEY="v9",V8_STORE_KEY="v8";
const V7_STORE_KEY="v7",V6_STORE_KEY="v6",V5_STORE_KEY="v5",LEGACY_STORE_KEY="v4";
const MATPLOTLIB_AVAILABLE=true;
const elements=new Map(SETTINGS_IDS.map(id=>[id,{value:"",type:"text",tagName:"INPUT",dataset:{}}]));
for(const [id,values] of [["agxPrimaries",["base","smooth"]],["deliveryProfile",["auto","archive"]]]){
  const el=elements.get(id);el.tagName="SELECT";el.options=values.map(value=>({value}));
}
const $=selector=>{const el=elements.get(selector.slice(1));if(!el)throw Error(selector);return el;};
const localStorage={getItem:key=>key==="v10"?JSON.stringify({agxPrimaries:"smooth",deliveryProfile:"archive",filmCurve:"portra400",filmMode:"full",chromaNr:"0.5"}):null};
const saveSettings=()=>{};
const setEvLabel=()=>{},setHdrLabel=()=>{},setGradeStrengthLabel=()=>{},setSceneTransformStrengthLabel=()=>{},setPunchLabel=()=>{};
const setAdjustmentLabels=()=>{},setChromaNrLabel=()=>{},updateGradeUi=()=>{},updateSceneTransformUi=()=>{},updateToneCoreUi=()=>{};
const updateFormatUi=()=>{},updateDecoderUi=()=>{},updateHdrOptionGate=()=>{};
''' + restore + r'''
restoreSettings();
if($("#agxPrimaries").value!=="smooth"||$("#deliveryProfile").value!=="archive"||$("#chromaNr").value!=="0.5")throw Error("lost explicit v10 choices");
'''
        result = subprocess.run([node, "-e", script], capture_output=True,
                                text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
