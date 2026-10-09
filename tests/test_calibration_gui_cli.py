# SPDX-License-Identifier: GPL-3.0-or-later
"""User calibration reaches CLI, GUI transport and analysis cache identity."""
from __future__ import annotations

import contextlib
from dataclasses import MISSING, fields
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from dngscan import calibration, cli
from dngscan._deps import np
from dngscan.gui import calibration_service
from dngscan.gui.page import PAGE
from dngscan.gui.preview_cache import (
    _analysis_from_json, _analysis_to_json, _cache_identity,
    _read_disk_entry, _write_disk_entry, build_proxy_entry, PreviewEntry, MAX_PIXEL_CACHE_ITEMS,
)
from dngscan.models import Analysis, RawBundle
from dngscan.noise_model import NoiseModel


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "dngscan/data/priors/jptc_collect/a7rm6-elec-20260818.json"


class CalibrationEntryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, {"DNGSCAN_CALIBRATION_DIR": str(self.root / "profiles")})
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary.cleanup()

    @staticmethod
    def _cli(*arguments: str) -> tuple[int, object]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(cli, "require_dependencies", side_effect=AssertionError("RAW decode not needed")):
            code = cli.main(["calibration", *arguments])
        return code, json.loads(output.getvalue())

    def test_cli_import_list_disable_enable_remove_without_a_raw(self) -> None:
        code, installed = self._cli("import", str(PROFILE), "--shutter-mode", "any")
        self.assertEqual(code, 0)
        self.assertEqual(installed["shutter"], "any")
        self.assertEqual(installed["source_shutter"], "电子快门")
        identity = installed["id"]
        self.assertEqual(self._cli("list")[1][0]["id"], identity)
        self._cli("disable", identity)
        self.assertFalse(self._cli("list")[1][0]["active"])
        self._cli("enable", identity)
        self.assertTrue(self._cli("list")[1][0]["active"])
        self._cli("remove", identity)
        self.assertEqual(self._cli("list"), (0, []))

    def test_picker_import_persists_selected_json_and_invalidates_live_preview(self) -> None:
        text = PROFILE.read_text(encoding="utf-8")
        with patch.object(calibration_service.PREVIEW_STORE, "clear_memory") as clear, patch.object(
            calibration_service.PREVIEW_COORDINATOR, "clear"
        ) as generations:
            result = calibration_service.import_calibration({"files": [{"name": "测量/profile.json", "text": text}], "shutterMode": "any"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["calibration"]["shutter"], "any")
        self.assertEqual(len(calibration.list_calibrations()), 1)
        clear.assert_called_once()
        generations.assert_called_once()

    def test_picker_directory_preserves_basenames_without_selected_root(self) -> None:
        seen = {}

        def consume(path: Path, **options):
            seen["dark"] = (path / "dark-scalars.csv").read_text()
            seen["ptc"] = (path / "ptc-iso100.csv").read_text()
            seen["options"] = options
            return {"id": "test"}

        with patch.object(calibration, "import_calibration", side_effect=consume), patch.object(
            calibration_service, "_invalidate_preview"
        ):
            calibration_service.import_calibration({"files": [
                {"name": "my-measurement/dark-scalars.csv", "text": "paired dark"},
                {"name": "my-measurement/ptc-iso100.csv", "text": "PTC"},
            ]})
        self.assertEqual(seen, {"dark": "paired dark", "ptc": "PTC", "options": {"shutter_override": None}})

    def test_picker_rejects_arbitrary_server_paths_traversal_and_oversized_text(self) -> None:
        for name in ("/tmp/profile.json", "../profile.json", "safe/../../profile.json", "safe\\profile.json", "profile.dng", "a//profile.json"):
            with self.subTest(name=name), self.assertRaises(ValueError), patch.object(calibration, "import_calibration") as install:
                calibration_service.import_calibration({"files": [{"name": name, "text": "{}"}]})
                install.assert_not_called()
        with patch.object(calibration_service, "MAX_CALIBRATION_TEXT_BYTES", 4), self.assertRaisesRegex(ValueError, "8 MiB"):
            calibration_service.import_calibration({"files": [{"name": "profile.json", "text": "12345"}]})
        self.assertEqual(calibration.list_calibrations(), [])

    def test_preview_and_export_analysis_identity_changes_with_active_measurements(self) -> None:
        source = self.root / "capture.dng"
        source.write_bytes(b"synthetic raw file identity")

        def identity():
            return _cache_identity(source, "clip", "camera")

        initial = identity()
        installed = calibration.import_calibration(PROFILE)
        active = identity()
        calibration.set_calibration_active(installed["id"], False)
        inactive = identity()
        calibration.remove_calibration(installed["id"])
        removed = identity()
        self.assertNotEqual(initial, active)
        self.assertNotEqual(active, inactive)
        self.assertEqual(initial, removed)

    def test_noise_model_and_curve_provenance_survive_persisted_analysis(self) -> None:
        required = {field.name: None for field in fields(Analysis)
                    if field.default is MISSING and field.default_factory is MISSING}
        model = NoiseModel(status="valid", source="User JPTC", reason="matched",
                           channel_variance={"G": (0.0001, 0.000002)},
                           correlation="measured-spectral-imbalance", spectral_ratios={"h": 0.1, "v": 0.8})
        required["snr_curves"] = {"G": {"stops": np.asarray([-3., 0.]), "snr_db": np.asarray([22., 31.]),
                                               "count": np.asarray([1, 1]), "ids": [1, 3], "kind": "model", "source": model.source}}
        analysis = Analysis(**required, noise_model=model)
        restored = _analysis_from_json(json.loads(json.dumps(_analysis_to_json(analysis))))
        self.assertIsInstance(restored.noise_model, NoiseModel)
        self.assertEqual(restored.noise_model.coefficients("G"), model.coefficients("G"))
        self.assertEqual(restored.noise_model.correlation, model.correlation)
        self.assertEqual(restored.noise_model.spectral_ratios, model.spectral_ratios)
        self.assertEqual(restored.snr_curves["G"]["kind"], "model")
        self.assertEqual(restored.snr_curves["G"]["source"], model.source)

    def test_compact_and_disk_preview_keep_noise_transfer_without_raw_buffers(self) -> None:
        model = NoiseModel(status="valid", source="synthetic calibration", reason="matched",
                           channel_variance={label: (1e-4, 2e-6) for label in "RGB"})
        descriptor = {"supported": True, "sensor_window_shape": [64, 64], "wb_mode": "camera",
                      "normalized_raw_to_scene": np.eye(3).tolist()}
        required = {field.name: None for field in fields(Analysis)
                    if field.default is MISSING and field.default_factory is MISSING}
        analysis = Analysis(**required, noise_model=model)
        source = RawBundle(path=self.root / "synthetic.dng", raw_image=np.ones((64, 64), np.uint16),
                           raw_colors=np.zeros((64, 64), np.uint8), xyz_render=np.zeros((64, 64, 3)),
                           render_scale=1., scene_rec2020_render=np.full((64, 64, 3), .2, np.float32),
                           scene_scale=1., white_level=16383, black_levels=[512.] * 4,
                           camera_wb=[1.] * 4, color_desc="RGBG", raw_pattern=[[0, 1], [3, 2]],
                           camera_white_levels=[16383.] * 4, noise_decode=descriptor, noise_model=model)
        with patch("dngscan.gui.preview_cache.PROXY_LONG_EDGE", 32):
            entry = build_proxy_entry(source, analysis)
        self.assertIsNone(entry.bundle.raw_image)
        self.assertIsNone(entry.bundle.raw_colors)
        self.assertEqual(entry.bundle.noise_decode, descriptor)
        self.assertIsNot(entry.bundle.noise_decode, descriptor)
        self.assertEqual(entry.bundle.noise_model.coefficients("R"), model.coefficients("R"))
        destination = self.root / "preview.npz"
        _write_disk_entry(destination, entry)
        restored = _read_disk_entry(destination, source.path, False)
        self.assertIsNotNone(restored)
        self.assertIsNone(restored.bundle.raw_image)
        self.assertEqual(restored.bundle.noise_decode, descriptor)
        self.assertEqual(restored.bundle.noise_model.coefficients("B"), model.coefficients("B"))

    def test_cached_pixel_report_survives_representation_changes_and_is_bounded(self) -> None:
        entry = PreviewEntry(bundle=None, analysis=None)
        pixels = np.zeros((2, 2, 3), np.uint8)
        report = {"status": "skipped", "reason": "Apple decoder covariance is not calibrated"}
        entry.put_pixels("source", pixels, report=report)
        report["status"] = "disabled"
        self.assertEqual(entry.get_pixel_report("source")["status"], "skipped")
        retrieved = entry.get_pixel_report("source")
        retrieved["status"] = "disabled"
        self.assertEqual(entry.get_pixel_report("source")["status"], "skipped")
        for index in range(MAX_PIXEL_CACHE_ITEMS):
            entry.put_pixels(index, pixels, report={"status": "active-approximate"})
        self.assertIsNone(entry.get_pixels("source"))
        self.assertIsNone(entry.get_pixel_report("source"))
        self.assertEqual(len(entry._pixel_reports), MAX_PIXEL_CACHE_ITEMS)

    def test_profile_change_during_export_analysis_stops_before_writing_image(self) -> None:
        from dngscan.gui import service
        from tests.test_preview_cache import _analysis, _bundle

        source = self.root / "source.dng"
        source.write_bytes(b"fixture")
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(service, "calibration_fingerprint", side_effect=["before", "after"]))
            stack.enter_context(patch.object(service.dg, "require_dependencies"))
            stack.enter_context(patch.object(service.PREVIEW_STORE, "peek", return_value=None))
            stack.enter_context(patch.object(service, "_load_export_scene", return_value=_bundle()))
            stack.enter_context(patch.object(service, "_cached_full_analysis", return_value=None))
            stack.enter_context(patch.object(service.dg, "analyze", return_value=(_analysis(), None, None)))
            stack.enter_context(patch.object(service.dg, "build_render_plan", return_value=object()))
            write = stack.enter_context(patch.object(service.dg, "export_jpeg"))
            with self.assertRaisesRegex(RuntimeError, "标定在导出分析期间发生变化"):
                service.run_export({"input": str(source), "outdir": str(self.root),
                                    "evAuto": False, "filmOpticsSeed": 1})
            write.assert_not_called()
        self.assertEqual(list(self.root.glob("*.jpg")), [])


try:
    from fastapi.testclient import TestClient
    from dngscan.gui.fastapi_app import create_app
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "GUI extra unavailable")
class CalibrationHttpTests(unittest.TestCase):
    def test_management_routes_require_local_session_before_dispatch(self) -> None:
        app = create_app(service_module=object(), session_token="calibration-test", expected_port=48765)
        with TestClient(app, base_url="http://127.0.0.1:48765") as client:
            for path in ("list", "import", "remove", "active"):
                response = client.post("/calibration/" + path, json={})
                self.assertEqual(response.status_code, 403)
            with patch.object(calibration_service, "import_calibration", return_value={"ok": True}) as install:
                body = {"files": [{"name": "profile.json", "text": "{}"}]}
                response = client.post("/calibration/import", json=body, headers={"X-DngScan-Token": "calibration-test"})
                self.assertEqual(response.json(), {"ok": True})
                install.assert_called_once_with(body)

    def test_import_body_is_bounded_before_json_parsing_and_import(self) -> None:
        app = create_app(service_module=object(), session_token="calibration-test", expected_port=48765)
        with TestClient(app, base_url="http://127.0.0.1:48765") as client, patch.object(
            calibration_service, "MAX_CALIBRATION_REQUEST_BYTES", 64
        ), patch.object(calibration_service, "import_calibration") as install:
            response = client.post("/calibration/import", content=b"x" * 65,
                                   headers={"X-DngScan-Token": "calibration-test"})
        self.assertFalse(response.json()["ok"])
        self.assertIn("16 MiB", response.json()["error"])
        install.assert_not_called()


NODE = shutil.which("node")


@unittest.skipUnless(NODE, "Node.js is required for GUI callback execution")
class CalibrationPageRuntimeTests(unittest.TestCase):
    def test_picker_reads_selected_text_and_carries_explicit_mode(self) -> None:
        start = PAGE.index("async function importCalibrationSelection(picker)")
        source = PAGE[start:PAGE.index('$("#calibrationOpen").onclick=', start)]
        harness = r"""
const assert=require('node:assert/strict');
const vm=require('node:vm');
const source=require('node:fs').readFileSync(0,'utf8');
const calls=[],statuses=[];
const context=vm.createContext({
  $:id=>{assert.equal(id,'#calibrationShutterMode');return {value:'any'};},
  setCalibrationStatus:(text,error)=>statuses.push({text,error}),
  changeCalibration:async(route,body)=>calls.push({route,body}),
});
vm.runInContext(source,context);
const picker={files:[{name:'profile.json',webkitRelativePath:'measurements/profile.json',size:2,text:async()=>'{}'},
                    {name:'capture.DNG',size:100000000,text:async()=>{throw Error('RAW must not be read');}}],value:'selected',disabled:false};
context.picker=picker;
(async()=>{
  await vm.runInContext('importCalibrationSelection(picker)',context);
  assert.equal(calls.length,1);assert.equal(calls[0].route,'/calibration/import');
  assert.deepEqual(JSON.parse(JSON.stringify(calls[0].body)),{files:[{name:'measurements/profile.json',text:'{}'}],shutterMode:'any'});
  assert.equal(picker.disabled,false);assert.equal(picker.value,'');assert.equal(statuses.length,0);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run([NODE, "-e", harness], input=source, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_successful_change_discards_stale_preview_before_preparing(self) -> None:
        start = PAGE.index("async function changeCalibration(route,body)")
        source = PAGE[start:PAGE.index("async function importCalibrationSelection", start)]
        harness = r"""
const assert=require('node:assert/strict');const vm=require('node:vm');
const source=require('node:fs').readFileSync(0,'utf8');const calls=[],statuses=[];
const elements={'#input':{value:'/photos/raw.dng'},'#revealBtn':{style:{display:'block'}}};
const context=vm.createContext({$:id=>elements[id],lastSavedPath:'old.jpg',RAW9_PROBES:new Map([['old',{}]]),RAW9_PROBE_REQUESTS:new Map([['old',{}]]),
  postJob:async()=>{calls.push('mutate');return {ok:true,calibrations:[]};},renderCalibrations:()=>calls.push('list'),
  beginPreviewSession:()=>calls.push('invalidate'),setCalibrationStatus:(text,error)=>statuses.push({text,error}),
  fetchDecodeSupport:()=>calls.push('probe'),preparePreview:async()=>calls.push('prepare')});
vm.runInContext(source,context);
(async()=>{
  await vm.runInContext('changeCalibration("/calibration/active",{id:"x",active:false})',context);
  assert.deepEqual(calls,['mutate','list','invalidate','probe','prepare']);assert.equal(context.lastSavedPath,'');
  assert.equal(context.RAW9_PROBES.size+context.RAW9_PROBE_REQUESTS.size,0);
  assert.equal(elements['#revealBtn'].style.display,'none');assert.equal(statuses.some(status=>status.error),false);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run([NODE, "-e", harness], input=source, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
