# SPDX-License-Identifier: GPL-3.0-or-later
"""Execute selection callbacks: syntax checks cannot catch missing globals."""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest

from dngscan.gui.page import PAGE


NODE = shutil.which("node")


def _page_section(start: str, end: str) -> str:
    offset = PAGE.index(start)
    return PAGE[offset:PAGE.index(end, offset)]


CAPABILITY_SOURCE = "\n".join((
    _page_section("let CHROMA_NR_CAPABILITY=", "function setGradeStrengthLabel()"),
    _page_section("function setChromaNrLabel()", "function fmtBias("),
    _page_section("function calibrationReason(", "function renderCalibrations("),
    _page_section("let DETECTED_READY=false;", "// Each probe line lands"),
))

# This is intentionally a small DOM rather than a replacement for any of the
# capability or label functions. Running those actual functions catches stale
# globals, rejected-model warnings, and settings callbacks that syntax checks
# cannot exercise.
CAPABILITY_HARNESS = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const payload = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const elements = new Map();
const $ = id => {
  if (!elements.has(id)) {
    const classes = new Set();
    elements.set(id, {value:'', checked:false, disabled:false, textContent:'',
      innerHTML:'', title:'', type:'text', tagName:'INPUT', dataset:{}, style:{}, options:[],
      classList:{toggle:(name,on)=>on?classes.add(name):classes.delete(name),
        contains:name=>classes.has(name), remove:name=>classes.delete(name)},
      removeAttribute:()=>{}});
  }
  return elements.get(id);
};
$('#chromaNr').value='0.40';
const calls=[];
const context=vm.createContext({$, console, ...payload.globals,
  saveSettings:()=>calls.push('save'), scheduleLivePreview:()=>calls.push('preview')});
vm.runInContext(payload.source,context);
const detected = (overrides={}) => ({
  decoder_actual:'LibRaw', demosaic_actual:'AHD', evidence_provider:'libraw',
  raw_clip_union_pct:0, body_median_ev:null, black_ev:null, white_ev:null,
  contrast:null, reliable_tail_ev:2, reliability_source:'raw-cfa', hdr_earned_ev:1,
  processing_evidence:{label:'通用成像（噪声未标定）',detail:'可以正常完成成像与导出'},
  capture_readout:{raw_geometry:[4224,3024],sample_bits:16,storage_lossless:true},
  noise_model:{status:'unavailable',source:'none',reason:'no-matched-calibration'},
  chroma_nr_capability:{available:false,reason:'independent noise calibration unavailable'},
  ...overrides});
eval(payload.test);
"""
HARNESS = r"""
const assert = require("node:assert/strict");
const vm = require("node:vm");
const payload = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
const calls = [], statuses = [], listeners = new Map();
const file = {name: "sample RAW.DNG"};
const elements = new Map([
  ["#filePicker", {files: [file], value: "selected", disabled: false}],
  ["#input", {value: "/old/source.dng"}],
  ["#outdir", {value: ""}],
  ["#revealBtn", {style: {display: "block"}}],
  ["#coreimageVersion", {value: "8"}],
]);
for (const [id, element] of elements) {
  element.addEventListener = (event, callback) => {
    assert.equal(event, "change");
    listeners.set(id, callback);
  };
}
const context = vm.createContext({
  $: id => {assert.ok(elements.has(id), `unexpected DOM lookup ${id}`); return elements.get(id);},
  INIT_DIR: "/exports",
  lastSavedPath: "/old/export.jpg",
  beginPreviewSession: () => calls.push("begin"),
  saveSettings: () => calls.push("save"),
  fetchDecodeSupport: path => {assert.equal(path, "/uploads/sample.dng"); calls.push("support");},
  preparePreview: async () => {calls.push("prepare");},
  setStatus: (text, kind) => statuses.push({text, kind}),
  apiFetch: async (url, options) => {
    assert.equal(url, "/upload?name=sample%20RAW.DNG");
    assert.equal(options.method, "POST");
    assert.equal(options.headers["Content-Type"], "application/octet-stream");
    assert.equal(options.body, file);
    assert.equal(elements.get("#filePicker").disabled, true);
    assert.equal(elements.get("#input").value, "");
    assert.deepEqual(calls, ["begin"]);
    calls.push("upload");
    return {json: async () => ({ok: true, path: "/uploads/sample.dng"})};
  },
});
// Use the page's actual declarations, so a deleted global is not supplied by
// the harness and silently hidden. Seed stale entries before dispatch.
vm.runInContext(payload.declarations, context);
vm.runInContext('RAW9_PROBES.set("old", {}); RAW9_PROBE_REQUESTS.set("old", {});', context);
vm.runInContext(payload.listener, context);
(async () => {
  await listeners.get(payload.selector)();
  if (payload.selector === "#filePicker") {
    assert.equal(statuses.some(status => status.kind === "err"), false,
      JSON.stringify(statuses));
    assert.deepEqual(calls, ["begin", "upload", "save", "support", "prepare"]);
    assert.equal(vm.runInContext("RAW9_PROBES.size + RAW9_PROBE_REQUESTS.size", context), 0);
    assert.equal(elements.get("#input").value, "/uploads/sample.dng");
    assert.equal(elements.get("#outdir").value, "/exports");
    assert.equal(elements.get("#filePicker").disabled, false);
    assert.equal(elements.get("#revealBtn").style.display, "none");
    assert.equal(context.lastSavedPath, "");
    assert.equal(statuses.at(-1).kind, "ok");
  } else {
    assert.deepEqual(calls, ["save", "prepare"]);
    assert.equal(elements.get("#coreimageVersion").value, "8");
    assert.equal(vm.runInContext("RAW9_PROBES.size + RAW9_PROBE_REQUESTS.size", context), 2);
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
"""


@unittest.skipUnless(NODE, "Node.js is required for GUI callback execution")
class PageSelectionRuntimeTests(unittest.TestCase):
    def _run_listener(self, selector: str) -> None:
        start = PAGE.index(f'$("{selector}").addEventListener("change",')
        if selector == "#filePicker":
            end = PAGE.index("\nasync function listOutDir", start)
        else:
            end = PAGE.index("\n", start)
        declarations_start = PAGE.index("const RAW9_PROBES=")
        declarations_end = PAGE.index("function raw9Probe(", declarations_start)
        result = subprocess.run(
            [NODE, "-e", HARNESS],
            input=json.dumps({
                "selector": selector,
                "listener": PAGE[start:end],
                "declarations": PAGE[declarations_start:declarations_end],
            }),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_uploaded_file_reaches_support_probe_and_preview(self) -> None:
        self._run_listener("#filePicker")

    def test_explicit_apple_version_change_reprepares_preview(self) -> None:
        self._run_listener("#coreimageVersion")


@unittest.skipUnless(NODE, "Node.js is required for GUI callback execution")
class PageCapabilityRuntimeTests(unittest.TestCase):
    def _run(self, test: str, extra_source: str = "", **globals: object) -> None:
        result = subprocess.run(
            [NODE, "-e", CAPABILITY_HARNESS],
            input=json.dumps({"source": CAPABILITY_SOURCE + "\n" + extra_source,
                              "test": test, "globals": globals}),
            text=True, capture_output=True, timeout=10, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_calibration_is_a_normal_general_imaging_state(self) -> None:
        self._run(r"""
context.renderDetectedParams(detected());
assert.match($('#noiseModelFact').textContent,/通用成像/);
assert.equal($('#noiseModelFact').classList.contains('warn'),false);
assert.equal($('#processingSupportFact').classList.contains('warn'),false);
assert.equal($('#chromaNrFact').classList.contains('warn'),false);
assert.match($('#chromaNrFact').textContent,/仍可正常成像/);
assert.ok($('#chromaNr').disabled);
assert.equal($('#chromaNr').value,'0.40');
assert.match($('#readoutFact').textContent,/4224×3024.*16-bit.*无损/);
""")

    def test_valid_file_noise_model_does_not_imply_usable_chroma_nr(self) -> None:
        self._run(r"""
for (const [reason,explanation] of [
  ['opaque-decoder-noise-transfer','Apple 系统解码'],
  ['non-Bayer-noise-transfer-unavailable','非 Bayer'],
  ['noise-transfer-for-spatial-or-point-corrections-unavailable','镜头或像素校正'],
]) {
  context.renderDetectedParams(detected({
    noise_model:{status:'valid',source:'DNG NoiseProfile',reason:'file-declared-model'},
    processing_evidence:{label:'文件噪声声明辅助成像',detail:'有模型但传播条件另行检查'},
    chroma_nr_capability:{available:false,reason},
  }));
  assert.match($('#noiseModelFact').textContent,/可用.*DNG 文件声明/);
  assert.equal($('#noiseModelFact').classList.contains('warn'),false);
  assert.ok($('#chromaNr').disabled);
  assert.ok($('#chromaNrFact').textContent.includes(explanation));
  assert.ok(!$('#chromaNrFact').textContent.includes(reason));
  // An older successful render must not override this file's capability.
  context.renderChromaNrStatus({status:'active-approximate'});
  assert.ok(!$('#chromaNrFact').textContent.startsWith('已应用 ·'));
  assert.equal($('#chromaNr').value,'0.40');
}
""")

    def test_explicit_rejection_and_spectral_constraints_remain_visible(self) -> None:
        self._run(r"""
for (const status of ['unresolved','rejected']) {
  context.renderDetectedParams(detected({noise_model:{status,reason:'read-noise-unresolved'}}));
  assert.equal($('#noiseModelFact').classList.contains('warn'),true);
  assert.ok(!$('#noiseModelFact').textContent.includes('未提供'));
}
context.renderDetectedParams(detected({
  noise_model:{status:'valid',source:'DNG NoiseProfile',reason:'file-declared-model',
    correlation:'measured-spectral-imbalance',spectral_ratios:{h:0.1}},
  chroma_nr_capability:{available:false,
    reason:'measured spectral imbalance requires correlated-noise propagation'},
}));
assert.match($('#noiseModelFact').textContent,/可用.*频谱不均衡.*限制 HDR 尾部.*0.100/);
assert.equal($('#noiseModelFact').classList.contains('warn'),true);
assert.match($('#chromaNrFact').textContent,/频谱不均衡/);
assert.equal($('#chromaNrFact').classList.contains('warn'),true);
assert.ok($('#chromaNr').disabled);
""")

    def test_hdr_facts_distinguish_sensor_reference_and_image_estimate(self) -> None:
        self._run(r"""
context.renderDetectedParams(detected({hdr_earned_ev:2.3}));
assert.match($('#hdrSceneFact').textContent,/实测可靠尾部.*可用余量 \+2.30 EV/);
assert.ok(!$('#hdrSceneFact').textContent.includes('上限1EV'));
context.renderDetectedParams(detected({reliability_source:'sensor-reference',hdr_earned_ev:1.9}));
assert.match($('#hdrSceneFact').textContent,/独立传感器参考尾部.*\+1.90 EV/);
assert.ok(!$('#hdrSceneFact').textContent.includes('上限1EV'));
context.renderDetectedParams(detected({reliability_source:'decoded-image-estimate',hdr_earned_ev:0.8}));
assert.match($('#hdrSceneFact').textContent,/图像估计.*上限1EV.*非传感器实测/);
context.renderDetectedParams(detected({hdr_earned_ev:0}));
assert.match($('#hdrSceneFact').textContent,/建议 SDR.*全尺寸分析/);
assert.equal($('#hdrSceneFact').classList.contains('warn'),false);
context.renderDetectedParams(detected({reliable_tail_ev:null}));
assert.equal($('#hdrSceneFact').classList.contains('warn'),true);
""")

    def test_capability_switch_preserves_strength_and_clears_previous_application(self) -> None:
        callback = _page_section('$("#chromaNr").oninput=', '\n[')
        self._run(r"""
context.renderChromaNrCapability({available:true,reason:''});
context.renderChromaNrStatus({status:'active-approximate'});
assert.match($('#chromaNrFact').textContent,/已应用/);
context.renderChromaNrCapability({available:false,reason:'opaque-decoder-noise-transfer'});
assert.ok($('#chromaNr').disabled);
assert.equal($('#chromaNr').value,'0.40');
context.renderChromaNrCapability({available:true,reason:''});
assert.equal($('#chromaNr').disabled,false);
assert.match($('#chromaNrFact').textContent,/等待当前预览验证/);
assert.ok(!$('#chromaNrFact').textContent.includes('已应用'));
context.renderChromaNrStatus({status:'active-approximate'});
$('#chromaNr').value='0.65';
$('#chromaNr').oninput();
assert.equal($('#chromaNrVal').textContent,'0.65');
assert.match($('#chromaNrFact').textContent,/等待当前预览验证/);
assert.equal(vm.runInContext('CHROMA_NR_REPORT',context),null);
assert.deepEqual(calls,['save','preview']);
""", callback)

    def test_preview_results_show_actual_application_and_localized_skip_reason(self) -> None:
        handler = _page_section("function handleJobResult(", '$("#evReferenceBtn").onclick=')
        self._run(r"""
Object.assign(context, {
  applyJobEv:()=>{}, renderDeliveryReport:()=>{}, renderSceneHistogram:()=>{},
  renderDisplayHistogram:()=>{}, setStatus:()=>{}, setPreviewImage:()=>{},
  fmtEv:()=>'', fullFrameReferenceText:()=>'',
});
assert.equal(context.handleJobResult({ok:true,preview:'image',
  chroma_nr_capability:{available:true,reason:''},
  chroma_nr:{status:'active-approximate'}},'预览'),true);
assert.match($('#chromaNrFact').textContent,/已应用.*模型近似/);
assert.equal(context.handleJobResult({ok:false,
  chroma_nr_capability:{available:false,reason:'opaque-decoder-noise-transfer'}},'预览'),false);
assert.match($('#chromaNrFact').textContent,/已应用/);
context.handleJobResult({ok:true,preview:'image',
  chroma_nr:{status:'skipped',reason:'spatial warp covariance is not propagated'}},'预览');
assert.match($('#chromaNrFact').textContent,/已跳过.*畸变校正/);
assert.equal($('#chromaNrFact').classList.contains('warn'),true);
""", handler)

    def test_live_hdr_headroom_refresh_keeps_evidence_source_and_resets_with_file(self) -> None:
        handler = _page_section("function handleJobResult(", '$("#evReferenceBtn").onclick=')
        self._run(r"""
Object.assign(context, {
  applyJobEv:()=>{}, renderDeliveryReport:()=>{}, renderSceneHistogram:()=>{},
  renderDisplayHistogram:()=>{}, setStatus:()=>{}, setPreviewImage:()=>{},
  fmtEv:()=>'', fullFrameReferenceText:()=>'',
});
context.renderDetectedParams(detected({
  reliable_tail_ev:3,reliability_source:'sensor-reference',hdr_earned_ev:1.8,
}));
context.handleJobResult({ok:true,preview:'image',hdr_earned_ev:0.6},'预览');
assert.match($('#hdrSceneFact').textContent,/独立传感器参考尾部 \+3.00 EV.*可用余量 \+0.60 EV/);
context.handleJobResult({ok:true,preview:'image'},'预览');
assert.match($('#hdrSceneFact').textContent,/\+0.60 EV/);
context.handleJobResult({ok:false,hdr_earned_ev:4},'预览');
assert.match($('#hdrSceneFact').textContent,/\+0.60 EV/);
context.handleJobResult({ok:true,preview:'image',hdr_earned_ev:0},'预览');
assert.match($('#hdrSceneFact').textContent,/建议 SDR/);
context.renderDetectedParams(detected({
  reliable_tail_ev:0.9,reliability_source:'decoded-image-estimate',hdr_earned_ev:0.5,
}));
context.handleJobResult({ok:true,preview:'image',hdr_earned_ev:0.2},'预览');
assert.match($('#hdrSceneFact').textContent,/图像估计.*上限1EV.*\+0.90 EV.*可用余量 \+0.20 EV/);
context.renderDetectedParams(null);
assert.equal(vm.runInContext('HDR_SCENE_FACTS',context),null);
context.handleJobResult({ok:true,preview:'image',hdr_earned_ev:4},'预览');
assert.equal($('#hdrSceneFact').textContent,'');
""", handler)

    def test_new_preview_session_removes_all_previous_evidence_and_capability(self) -> None:
        session_source = "\n".join((
            _page_section("const PREVIEW_CLIENT_ID=", "function setPreviewBadge("),
            _page_section("let clipOverlayAbort=", "function scheduleLivePreview()"),
        ))
        self._run(r"""
context.renderDetectedParams(detected({
  chroma_nr_capability:{available:true,reason:''},
  noise_model:{status:'rejected',reason:'unmatched-dn-scale'},
}));
context.renderChromaNrStatus({status:'active-approximate'});
const id=context.beginPreviewSession();
assert.ok(id.endsWith(':1'));
for (const selector of ['#decoderFact','#readoutFact','#processingSupportFact',
  '#wbFact','#clipFact','#evFact','#toneFact','#hdrSceneFact',
  '#noiseModelFact','#calibrationMatchFact']) {
  assert.equal($(selector).textContent,'',selector);
  assert.equal($(selector).classList.contains('warn'),false,selector);
}
for (const selector of ['#noiseModelFact','#processingSupportFact','#readoutFact'])
  assert.equal($(selector).title,'',selector);
assert.equal(vm.runInContext('CHROMA_NR_CAPABILITY',context),null);
assert.equal(vm.runInContext('CHROMA_NR_REPORT',context),null);
assert.equal(vm.runInContext('HDR_SCENE_FACTS',context),null);
assert.ok($('#chromaNr').disabled);
assert.match($('#chromaNrFact').textContent,/选择 RAW/);
assert.equal($('#chromaNr').value,'0.40');
""", session_source)

    def test_saved_nr_strength_restores_through_real_label_and_settings_functions(self) -> None:
        settings_source = "\n".join((
            _page_section('const STORE_KEY=', 'const COREIMAGE_AVAILABLE='),
            _page_section('const SETTINGS_IDS=', '["quality","outdir","png"]'),
        ))
        self._run(r"""
const saved=new Map([['dngscan.settings.v11',JSON.stringify({chromaNr:'0.70'})]]);
context.localStorage={getItem:key=>saved.get(key)||null,setItem:(key,value)=>saved.set(key,value)};
for (const name of ['setEvLabel','setHdrLabel','setGradeStrengthLabel',
  'setSceneTransformStrengthLabel','setPunchLabel','setAdjustmentLabels',
  'updateGradeUi','updateSceneTransformUi','updateToneCoreUi','updateFormatUi',
  'updateDecoderUi','updateHdrOptionGate']) context[name]=()=>{};
context.restoreSettings();
assert.equal($('#chromaNr').value,'0.70');
assert.equal($('#chromaNrVal').textContent,'0.70');
context.renderChromaNrCapability({available:false,reason:'non-Bayer-noise-transfer-unavailable'});
context.saveSettings();
assert.equal(JSON.parse(saved.get('dngscan.settings.v11')).chromaNr,'0.70');
""", settings_source, MATPLOTLIB_AVAILABLE=True)

    def test_calibration_readout_reasons_distinguish_unknown_and_mismatch(self) -> None:
        self._run(r"""
assert.match(context.calibrationReason('file-readout-mode-unavailable'),/RAW 未记录快门模式/);
assert.match(context.calibrationReason('sub-readout-not-declared'),/未声明.*未验证/);
assert.match(context.calibrationReason('declared-readout-constraints-matched'),/逐文件读出条件匹配/);
assert.match(context.calibrationReason('file-raw-geometry-unavailable'),/尺寸未记录/);
assert.match(context.calibrationReason('file-raw-geometry-mismatch'),/尺寸与标定不匹配/);
assert.match(context.calibrationReason('file-storage-lossless-unverified'),/无法确认.*无损/);
assert.match(context.calibrationReason('lossy-raw-storage-not-supported-by-external-prior'),/有损 RAW.*不适用/);
""")


@unittest.skipUnless(NODE, "Node.js is required for GUI callback execution")
class PageDeliveryRuntimeTests(unittest.TestCase):
    def test_heif_precision_controls_and_report_follow_actual_output_metadata(self) -> None:
        source = "\n".join((
            _page_section("function heifConfigurationProblem()", "let HDR_BACKEND_OK="),
            _page_section("function fmtMB(", "// Realtime histograms:"),
        ))
        harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const elements=new Map();
const $=id=>{
  if(!elements.has(id))elements.set(id,{value:'',disabled:false,textContent:'',innerHTML:'',style:{},options:[]});
  return elements.get(id);
};
$('#format').value='sdr-heic';
$('#deliveryProfile').value='auto';
$('#deliveryProfile').options=['auto','share-hq','share','archive'].map(value=>({value}));
$('#heifEncoder').value='x265';$('#heifBitDepth').value='10';$('#toneCore').value='agx';
const context=vm.createContext({$,saveSettings:()=>{}});
vm.runInContext(require('node:fs').readFileSync(0,'utf8'),context);
context.applyDeliveryConstraints();
assert.match($('#heifPrecisionHint').textContent,/SDR 主图.*浮点.*10-bit.*浮点回读/);
assert.equal($('#heifBitDepth').disabled,false);
assert.equal($('#exportConfirm').disabled,false);
context.renderDeliveryReport({delivery:{delivery_profile:'auto',delivery_container:'heic',
  delivery_quality:96,encoder:'x265',bit_depth:10,chroma_subsampling:'4:4:4',
  sdr_master_precision:'float32',readback_precision:'float32'}});
assert.match($('#deliveryReportBody').innerHTML,/SDR 母版精度.*一次 10-bit 量化/);
assert.match($('#deliveryReportBody').innerHTML,/回读验证精度.*浮点（未转为 8-bit）/);
$('#format').value='ultrahdr-heic';$('#heifEncoder').value='apple';$('#heifBitDepth').value='8';
context.applyDeliveryConstraints();
assert.match($('#heifPrecisionHint').textContent,/8-bit SDR 底图/);
assert.equal($('#heifBitDepth').disabled,false);
assert.equal($('#exportConfirm').disabled,false);
$('#heifBitDepth').value='10';context.applyDeliveryConstraints();
assert.match($('#heifPrecisionHint').textContent,/SDR 底图.*10-bit/);
assert.equal($('#heifBitDepth').disabled,false);
context.renderDeliveryReport({hdr_container:{delivery_profile:'auto',delivery_container:'heic',
  delivery_quality:96,encoder:'apple',bit_depth:10,chroma_subsampling:'4:4:4',
  has_iso_gainmap:true,quantization_dither:'TPDF-10bit',readback_precision:'float32'}});
assert.match($('#deliveryReportBody').innerHTML,/SDR 底图精度.*一次 10-bit 量化/);
$('#format').value='sdr-heic';$('#heifBitDepth').value='8';
$('#deliveryProfile').value='share';context.applyDeliveryDefaults();
context.applyDeliveryConstraints();
assert.match($('#heifPrecisionHint').textContent,/8-bit SDR 输出/);
assert.equal($('#exportConfirm').disabled,false);
context.renderDeliveryReport({delivery:{delivery_profile:'share',delivery_container:'heic',
  delivery_quality:95,encoder:'apple',bit_depth:8,chroma_subsampling:'4:2:0',
  readback_precision:'uint8'}});
assert.ok(!$('#deliveryReportBody').innerHTML.includes('一次 10-bit 量化'));
assert.match($('#deliveryReportBody').innerHTML,/回读验证精度.*8-bit/);
context.renderDeliveryReport({});
assert.equal($('#deliveryReport').style.display,'none');
assert.equal($('#deliveryReportBody').innerHTML,'');
"""
        result = subprocess.run([NODE, "-e", harness], input=source, text=True,
                                capture_output=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_apple_sdr_configuration_gate_and_hdr_tone_gate_do_not_coerce_settings(self) -> None:
        source = _page_section("function heifConfigurationProblem()", "let HDR_BACKEND_OK=")
        harness = r"""
const assert=require('node:assert/strict');
const vm=require('node:vm');
const elements=new Map();
const $=id=>{
  if(!elements.has(id))elements.set(id,{value:'',disabled:false,textContent:'',title:'',style:{},options:[],
    classList:{toggle:()=>{}}});
  return elements.get(id);
};
$('#format').value='sdr-heic';$('#deliveryProfile').value='auto';
$('#deliveryProfile').options=['auto','share-hq','share','archive'].map(value=>({value}));
$('#heifEncoder').value='apple';$('#heifBitDepth').value='10';$('#toneCore').value='agx';
const context=vm.createContext({$,saveSettings:()=>{}});
vm.runInContext(require('node:fs').readFileSync(0,'utf8'),context);
context.applyDeliveryConstraints();
assert.ok(context.heifConfigurationProblem());
assert.equal($('#exportConfirm').disabled,true);
assert.notEqual($('#toneCoreExportHint').style.display,'none');
assert.ok($('#toneCoreExportHint').textContent.includes(context.heifConfigurationProblem()));
assert.ok(!$('#heifPrecisionHint').textContent.includes('保留浮点精度'));
assert.equal($('#heifEncoder').value,'apple');
assert.equal($('#heifBitDepth').value,'10');
assert.equal($('#deliveryProfile').value,'auto');
$('#heifEncoder').value='x265';context.applyDeliveryConstraints();
assert.equal(context.heifConfigurationProblem(),'');
assert.equal($('#exportConfirm').disabled,false);
assert.equal($('#toneCoreExportHint').style.display,'none');
assert.match($('#heifPrecisionHint').textContent,/浮点.*10-bit/);
$('#heifEncoder').value='apple';$('#heifBitDepth').value='8';$('#deliveryProfile').value='share';
context.applyDeliveryDefaults();
assert.equal($('#chroma').value,'420');
assert.equal(context.heifConfigurationProblem(),'');
assert.equal($('#exportConfirm').disabled,false);
$('#chroma').value='422';context.applyDeliveryConstraints();
assert.ok(context.heifConfigurationProblem());
assert.equal($('#exportConfirm').disabled,true);
assert.equal($('#chroma').value,'422');
$('#format').value='ultrahdr-heic';$('#toneCore').value='lum';
context.applyDeliveryConstraints();
assert.equal(context.heifConfigurationProblem(),'');
assert.equal($('#exportConfirm').disabled,true);
assert.match($('#toneCoreExportHint').textContent,/HDR.*只支持 AgX/);
$('#toneCore').value='agx';context.updateToneCoreExportUi();
assert.equal($('#exportConfirm').disabled,false);
assert.equal($('#toneCoreExportHint').style.display,'none');
"""
        result = subprocess.run([NODE, "-e", harness], input=source, text=True,
                                capture_output=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_share_hq_controls_payload_heif_switch_and_visible_size_notice(self) -> None:
        def section(start: str, end: str) -> str:
            return PAGE[PAGE.index(start):PAGE.index(end, PAGE.index(start))]

        source = "\n".join((
            section("function heifConfigurationProblem()", "let HDR_BACKEND_OK="),
            section("function payload()", "async function postJob("),
            section('$("#exportConfirm").onclick=', '$("#revealBtn").onclick='),
            section("function fmtMB(", "// Realtime histograms:"),
        ))
        harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = require('node:fs').readFileSync(0, 'utf8');
const elements = new Map();
const $ = id => {
  if(!elements.has(id))elements.set(id, {value:'', checked:false, disabled:false, style:{}, options:[]});
  return elements.get(id);
};
$('#input').value='/sample.dng';
$('#format').value='sdr';
$('#decoder').value='libraw';
$('#toneCore').value='agx';
$('#deliveryProfile').options=['auto','share-hq','share','archive'].map(value=>({value}));
$('#deliveryProfile').value='share-hq';
$('#quality').value='100'; $('#chroma').value='444';
const statuses=[];
let result;
const context=vm.createContext({
  $, lastSavedPath:'', ensureRaw9Support:async()=>true,
  closeOutputDialog:()=>{}, beginBusy:()=>{}, endBusy:()=>{},
  applyJobEv:()=>{}, setPreviewImage:()=>{},
  setStatus:(text,kind)=>statuses.push({text,kind}),
  formatText:x=>x, fmtEv:()=>'+0.00', highlightText:()=>'', gamutText:()=>'',
  decoderText:()=>'', toneCoreText:()=>'', sceneTransformText:()=>'',
  fullFrameReferenceText:()=>'', metricText:()=>'',
  postJob:async(path,body)=>{assert.equal(path,'/export'); assert.equal(body.quality,97);
    assert.equal(body.chroma,'420'); return result;},
});
vm.runInContext(source,context);
context.applyDeliveryDefaults();
assert.equal($('#quality').value,'97'); assert.equal($('#chroma').value,'420');
assert.ok($('#quality').disabled && $('#chroma').disabled);
assert.equal($('#shareSizeHint').style.display,'block');
const body=context.payload();
assert.equal(body.deliveryProfile,'share-hq'); assert.equal(body.quality,97); assert.equal(body.chroma,'420');
// Format switches and restored incompatible settings use the same constraints.
for(const format of ['sdr-heic','ultrahdr-heic']){
  $('#format').value=format; $('#deliveryProfile').value='share-hq';
  context.applyDeliveryConstraints();
  assert.equal($('#deliveryProfile').value,'auto');
  assert.ok($('#deliveryProfile').options.find(o=>o.value==='share-hq').disabled);
  assert.ok(!('quality' in context.payload()) && !('chroma' in context.payload()));
}
$('#format').value='ultrahdr'; $('#deliveryProfile').value='share-hq';
context.applyDeliveryConstraints();
assert.ok(!$('#deliveryProfile').options.find(o=>o.value==='share-hq').disabled);
assert.equal(context.payload().quality,97);
assert.equal(context.fmtMB(20000000),'20.00 MB');
result={ok:true,saved:['/out.jpg'],ev:0,gain:1,format:'SDR JPEG',preview:'',
  delivery:{delivery_profile:'share-hq',delivery_container:'jpeg',delivery_quality:97,
    chroma_subsampling:'4:2:0',file_size_bytes:21000000,share_size_limit_bytes:20000000,
    share_size_exceeded:true,size_warning:'文件 21.00 MB，超过 20 MB 分享参考线；已保留质量 97 / 4:2:0 和原尺寸。'}};
(async()=>{
  await $('#exportConfirm').onclick();
  assert.equal(statuses.at(-1).kind,'warn');
  assert.match(statuses.at(-1).text,/已保存.*\/out.jpg/);
  assert.ok(statuses.at(-1).text.includes(result.delivery.size_warning));
  assert.ok($('#deliveryReportBody').innerHTML.includes(result.delivery.size_warning));
  assert.equal(context.lastSavedPath,'/out.jpg');
  assert.equal($('#go').disabled,false);
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
        result = subprocess.run([NODE, "-e", harness], input=source, text=True,
                                capture_output=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
