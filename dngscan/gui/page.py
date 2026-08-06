# SPDX-License-Identifier: GPL-3.0-or-later
"""Single-page HTML shell for the local dngscan web GUI."""
from __future__ import annotations

import json

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>dngscan</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden}
body{margin:0;font:14px/1.5 -apple-system,"PingFang SC",system-ui,sans-serif;background:#15171c;color:#e7e9ee}
.wrap{width:100%;height:100dvh;max-width:1900px;margin:0 auto;padding:10px 14px;display:grid;grid-template-rows:auto minmax(0,1fr);overflow:hidden}
h1{font-size:17px;font-weight:600;margin:0;white-space:nowrap}
.brandDetail{display:inline}
.topBar{display:grid;grid-template-columns:max-content minmax(280px,1fr);grid-template-rows:auto auto;column-gap:14px;row-gap:4px;align-items:center;min-width:0;margin:0 0 8px}
.topBar h1{grid-row:1 / span 2}
.topBar input[type=file]{width:100%;min-width:260px}
.topBar .ctlFact{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.card{background:#1d2028;border:1px solid #2b2f3a;border-radius:8px;padding:12px;margin:0;min-width:0}
.secTitle{font-size:12px;font-weight:600;color:#8fa0c4;text-transform:uppercase;letter-spacing:.06em;margin:0 0 10px}
.workspace{display:grid;grid-template-columns:minmax(520px,36%) minmax(0,1fr);gap:10px;min-width:0;min-height:0;height:100%;overflow:hidden}
.controlPanel{display:grid;grid-template-rows:auto minmax(0,1fr);gap:8px;min-width:0;min-height:0;overflow:hidden}
.dashboardTabs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px;padding:4px;background:#1d2028;border:1px solid #2b2f3a;border-radius:9px}
.dashTab{border:1px solid transparent;border-radius:7px;background:transparent;color:#939baa;padding:7px 10px;font:inherit;font-weight:600;cursor:pointer;white-space:nowrap}
.dashTab:hover{color:#dce3f3;background:#242936}
.dashTab.active{color:#fff;background:#2b3953;border-color:#48628d;box-shadow:0 1px 8px rgba(0,0,0,.18)}
.dashboardPanel{display:none;grid-template-columns:minmax(0,1fr);gap:8px;align-content:start;min-height:0;overflow:hidden}
.dashboardPanel.active{display:grid}
.previewCard{height:100%;min-height:0;display:flex;flex-direction:column;overflow:hidden}
.mobileNav{display:none}
.actions{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
label{display:block;font-size:12px;color:#9aa1b0;margin:0 0 6px}
input[type=text],input[type=number],select{width:100%;background:#12141a;border:1px solid #2b2f3a;border-radius:8px;color:#e7e9ee;padding:8px 10px;font:inherit}
input[type=file]{width:100%;background:#12141a;border:1px solid #2b2f3a;border-radius:8px;color:#cdd2dd;padding:5px;font:inherit;cursor:pointer}
input[type=file]::file-selector-button{background:#2c3444;border:1px solid #46536b;border-radius:6px;color:#eef2ff;padding:7px 12px;margin-right:10px;font:inherit;font-weight:600;cursor:pointer}
input[type=file]:disabled{opacity:.5;cursor:default}
.row{display:flex;gap:12px;flex-wrap:wrap}
.row>div{flex:1;min-width:150px}
.row>.evMain{flex:1 1 100%;min-width:0}
.modes{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.modes button{flex:1;min-width:56px;overflow:hidden;background:#12141a;border:1px solid #2b2f3a;border-radius:8px;color:#cdd2dd;padding:8px 4px;cursor:pointer;font:inherit;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1px;line-height:1.25;min-height:48px}
.modes button .m{font-weight:600;color:#e7e9ee;font-size:13px;font-variant-numeric:tabular-nums;white-space:nowrap}
.modes button .d{font-size:10px;color:#828a99;white-space:nowrap;max-width:100%;overflow:hidden;text-overflow:ellipsis}
.modes button.sel{border-color:#5b8cff;background:#1a2233}
.modes button#evReferenceBtn{flex:1.8;min-width:100px}
.sliderField{flex:1;min-width:170px}
.labelRow{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:6px}
.labelRow label{margin:0;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.labelRow .val{font-size:13px;color:#e7e9ee;font-variant-numeric:tabular-nums;white-space:nowrap}
input[type=range]{display:block;width:100%;margin:6px 0 2px;accent-color:#5b8cff;height:18px}
button.go{background:#5b8cff;border:0;border-radius:9px;color:#fff;padding:11px 18px;font:inherit;font-weight:600;cursor:pointer}
button.go:disabled{opacity:.5;cursor:default}
button.ghost{background:#12141a;border:1px solid #2b2f3a;border-radius:8px;color:#cdd2dd;padding:8px 12px;cursor:pointer;font:inherit;white-space:nowrap}
.previewLive{margin-left:auto;border:1px solid #33415c;border-radius:999px;padding:4px 9px;color:#91b4ff;background:#151b27;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
.previewLive.busy{color:#ffc46b;border-color:#614d2e}
.previewLive.err{color:#ff8a8a;border-color:#663939}
.muted{color:#828a99;font-size:12px}
.coreFacts{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.coreFacts span{background:#151922;border:1px solid #303746;border-radius:6px;padding:5px 8px;color:#9aa1b0;font-size:12px}
.coreFacts span.control{border-color:#4a5568;background:#171b24}
.coreFacts b{color:#e7e9ee;font-weight:500}
#controlHint{margin-top:10px;color:#9aa7c0;font-size:12px;line-height:1.55;min-height:0}
#controlHint:empty{display:none}
#status{position:absolute;z-index:3;top:8px;left:8px;right:8px;max-height:3em;margin:0;padding:5px 8px;overflow:hidden;border:1px solid rgba(82,91,110,.72);border-radius:6px;background:rgba(17,20,26,.9);white-space:pre-line;pointer-events:none}
#status:empty{display:none}
.err{color:#ff8a8a}.ok{color:#8ae08a}.warn{color:#ffc46b}
.browserList{display:none;margin-top:10px;border:1px solid #2b2f3a;border-radius:8px;max-height:260px;overflow:auto;background:#12141a}
.browserList div{padding:6px 10px;cursor:pointer;border-bottom:1px solid #20242e;font-size:13px}
.browserList div:hover{background:#1a2233}
.browserList div.pick{color:#8ae08a;font-weight:600;position:sticky;top:0;background:#12141a}
#previewWrap{position:relative;margin-top:10px;min-height:260px;flex:1;overflow:hidden;background:#11141a;border:1px solid #2b2f3a;border-radius:8px}
#preview{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:none;transition:opacity .15s ease}
#previewWrap.loading #preview{opacity:.4}
#spinner{display:none;position:absolute;left:50%;top:50%;width:34px;height:34px;margin:-17px 0 0 -17px;border:3px solid rgba(255,255,255,.22);border-top-color:#eef2ff;border-radius:50%;animation:spin .8s linear infinite}
#previewWrap.loading #spinner{display:block}
@keyframes spin{to{transform:rotate(360deg)}}
.dim{opacity:.45;pointer-events:none}
#deliveryReport{margin-top:10px;border:1px solid #2b2f3a;border-radius:8px;padding:8px 12px;background:#11141a}
#deliveryReport summary{cursor:pointer;font-size:13px;color:#9aa3b2;user-select:none}
#deliveryReport[open] summary{margin-bottom:8px}
.reportGrid{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;font-size:12.5px}
.reportGrid dt{color:#9aa3b2;white-space:nowrap}
.reportGrid dd{margin:0;color:#e6e9f0;font-variant-numeric:tabular-nums}
.reportGrid dd.warn{color:#f0b35e}
.histCanvas{display:none;width:100%;height:82px;margin-top:8px;background:#11141a;border:1px solid #2b2f3a;border-radius:8px}
.ctlFact{margin-top:6px;color:#8fa0c4;font-size:11.5px;line-height:1.5;font-variant-numeric:tabular-nums;white-space:pre-line}
.ctlFact:empty{display:none}
.ctlFact.warn{color:#f0b35e}
.chk{display:flex;align-items:center;gap:8px}.chk input{width:auto}
.outdirRow{display:flex;gap:8px;align-items:stretch}
.outdirRow input{flex:1}
dialog.outputDialog{width:min(680px,calc(100vw - 32px));max-height:90vh;padding:0;border:1px solid #353b48;border-radius:12px;background:#1d2028;color:#e7e9ee;box-shadow:0 24px 80px rgba(0,0,0,.55);overflow:hidden}
dialog.outputDialog::backdrop{background:rgba(7,9,13,.72);backdrop-filter:blur(3px)}
.dialogPanel{max-height:90vh;padding:18px;overflow:auto}
.dialogHeader{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px}
.dialogTitle{margin:0;font-size:17px;font-weight:600}
.dialogActions{display:flex;justify-content:flex-end;gap:10px;margin-top:18px;padding-top:14px;border-top:1px solid #2b2f3a}
@media (max-width:1240px){
  .workspace{grid-template-columns:minmax(470px,46%) minmax(0,1fr)}
  .modes button .d{display:none}
}
@media (max-height:900px){
  body{font-size:13px}
  .wrap{padding:8px 10px}
  .topBar{margin-bottom:6px;row-gap:2px}
  .card{padding:9px}
  .secTitle{margin-bottom:7px}
  .dashboardPanel,.controlPanel{gap:6px}
  input[type=text],input[type=number],select{padding:6px 8px}
  .modes{margin-top:5px}
  .modes button{min-height:40px;padding:5px 3px}
  .dashboardPanel .row[style*="margin-top:12px"]{margin-top:8px!important}
  .coreFacts{margin-top:8px}
  #controlHint{margin-top:6px;line-height:1.35}
  .histCanvas{height:68px;margin-top:6px}
}
@media (max-width:767px), (max-width:900px) and (max-height:500px){
  body{font-size:12.5px}
  .wrap{padding:max(6px,env(safe-area-inset-top,0px)) max(8px,env(safe-area-inset-right,0px)) max(6px,env(safe-area-inset-bottom,0px)) max(8px,env(safe-area-inset-left,0px));grid-template-rows:auto minmax(0,1fr)}
  h1{font-size:15px}
  .brandDetail{display:none}
  .topBar{grid-template-columns:max-content minmax(0,1fr);grid-template-rows:auto auto;column-gap:8px;row-gap:2px;margin-bottom:6px}
  .topBar h1{grid-row:1}
  .topBar input[type=file]{min-width:0;min-height:44px;padding:3px}
  .topBar input[type=file]::file-selector-button{min-height:36px;padding:5px 8px;margin-right:6px}
  .topBar .ctlFact{grid-column:1 / -1;max-height:1.5em}
  .workspace{grid-template-columns:minmax(0,1fr);grid-template-rows:minmax(160px,32%) minmax(0,1fr) auto;gap:6px}
  .previewCard{grid-column:1;grid-row:1;padding:7px}
  .controlPanel{grid-column:1;grid-row:2;display:block;overflow:hidden}
  .dashboardTabs{display:none}
  .dashboardPanel,.dashboardPanel.active,.dashboardPanel[hidden]{display:contents!important}
  .dashboardPanel>.card{display:none!important}
  body[data-mobile-card="decode"] [data-mobile-card="decode"],
  body[data-mobile-card="exposure"] [data-mobile-card="exposure"],
  body[data-mobile-card="tone"] [data-mobile-card="tone"],
  body[data-mobile-card="imaging"] [data-mobile-card="imaging"],
  body[data-mobile-card="color"] [data-mobile-card="color"]{display:block!important}
  .mobileNav{grid-column:1;grid-row:3;display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:3px;padding:3px;background:#1d2028;border:1px solid #2b2f3a;border-radius:9px}
  .mobileNav button{min-width:0;min-height:48px;border:1px solid transparent;border-radius:7px;background:transparent;color:#939baa;padding:4px 2px;font:inherit;font-weight:600;cursor:pointer;white-space:nowrap}
  .mobileNav button.active{color:#fff;background:#2b3953;border-color:#48628d}
  .card{padding:8px}
  .secTitle{margin-bottom:5px}
  .dashboardPanel .row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px 8px}
  .dashboardPanel .row>div{min-width:0!important}
  .dashboardPanel .row>.evMain,.dashboardPanel .ctlFact{grid-column:1 / -1}
  .dashboardPanel .row[style*="margin-top:12px"]{margin-top:6px!important}
  label{margin-bottom:3px;line-height:1.3}
  input[type=text],input[type=number],select{min-height:44px;padding:5px 7px}
  input[type=range]{height:44px;margin:0}
  .labelRow{margin-bottom:0}
  .modes{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:3px;margin-top:3px}
  .modes button,.modes button#evReferenceBtn{min-width:0;min-height:44px;padding:3px 1px}
  .modes button .m{font-size:11.5px}
  .coreFacts{margin-top:5px;gap:4px}
  .coreFacts span{padding:3px 5px;font-size:11px}
  #controlHint{margin-top:4px;line-height:1.3;max-height:2.6em;overflow:hidden}
  .ctlFact{max-height:3em;overflow:hidden}
  [data-mobile-card="decode"] .ctlFact{max-height:1.5em;white-space:nowrap;text-overflow:ellipsis}
  #toneFact{max-height:1.5em;white-space:nowrap;text-overflow:ellipsis}
  #previewWrap{min-height:0;margin-top:5px}
  .histCanvas{height:56px;margin-top:4px}
  .actions{gap:5px}
  button.go,button.ghost{min-height:44px;padding:6px 11px}
  .previewLive{font-size:11px;padding:3px 7px}
  #status{top:5px;left:5px;right:5px;max-height:3em}
  #deliveryReport{margin-top:5px;padding:5px 8px}
  dialog.outputDialog{width:calc(100vw - 16px);max-height:calc(100dvh - 16px)}
  .dialogPanel{max-height:calc(100dvh - 16px);padding:12px;scrollbar-width:none}
  .dialogPanel::-webkit-scrollbar{display:none}
}
@media (max-width:900px) and (max-height:500px){
  .workspace{grid-template-columns:minmax(250px,42%) minmax(0,1fr) 54px;grid-template-rows:minmax(0,1fr)}
  .previewCard{grid-column:1;grid-row:1}
  .controlPanel{grid-column:2;grid-row:1}
  .mobileNav{grid-column:3;grid-row:1;grid-template-columns:1fr;grid-template-rows:repeat(5,minmax(0,1fr))}
  .mobileNav button{min-height:44px}
  .ctlFact{max-height:1.5em;white-space:nowrap;text-overflow:ellipsis}
}
</style></head>
<body><div class="wrap">
<div class="topBar">
  <h1><span class="brand">dngscan</span><span class="brandDetail"> · RAW 分析与转换</span></h1>
  <input type="file" id="filePicker" accept="RAW_ACCEPT" title="RAW 文件">
  <div class="ctlFact" id="fileFact" style="margin-top:0"></div>
  <input type="hidden" id="input">
</div>

<div class="workspace">
<div class="controlPanel">

<div class="dashboardTabs" role="tablist" aria-label="调整分区">
  <button type="button" class="dashTab active" id="captureTab" role="tab" aria-selected="true" aria-controls="capturePanel" data-panel="capturePanel">基础</button>
  <button type="button" class="dashTab" id="toneTab" role="tab" aria-selected="false" aria-controls="tonePanel" data-panel="tonePanel">色调</button>
  <button type="button" class="dashTab" id="colorTab" role="tab" aria-selected="false" aria-controls="colorPanel" data-panel="colorPanel">色彩 / 风格</button>
</div>

<section class="dashboardPanel active" id="capturePanel" role="tabpanel" aria-labelledby="captureTab">

<div class="card" id="mobileDecodeCard" data-mobile-card="decode">
  <div class="secTitle">RAW 解码</div>
  <div class="row">
    <div style="flex:1;min-width:170px" id="decoderBlock">
      <label>解码器</label>
      <select id="decoder" title="scene-linear RGB 来源；CFA 统计始终由 LibRaw 读取。">
        <option value="libraw">LibRaw · 默认</option>
        <option value="coreimage">Apple RAW · 9 优先</option>
      </select>
    </div>
    <div style="flex:1;min-width:140px;display:none" id="coreimageVersionBlock">
      <label>CI 版本</label>
      <select id="coreimageVersion" title="auto 选择文件支持的最高版本；显式版本在不支持时会报错。">
        <option value="auto">自动</option>
        <option value="9">9</option>
        <option value="8">8</option>
        <option value="7">7</option>
      </select>
    </div>
    <div style="flex:1;min-width:170px">
      <label>解拜耳</label>
      <select id="demosaic" title="仅 LibRaw；RAW 9 使用 Apple 的 CoreML 解拜耳与降噪模型。">
        <option value="auto">自动 · DHT</option>
        <option value="dht">DHT</option>
        <option value="dcb">DCB</option>
        <option value="ahd">AHD</option>
        <option value="aahd">AAHD</option>
        <option value="vng">VNG</option>
        <option value="ppg">PPG</option>
      </select>
    </div>
    <div class="ctlFact" id="decodeTierFact" style="flex-basis:100%;margin-top:0"></div>
    <div class="ctlFact" id="decoderFact" style="flex-basis:100%;margin-top:0"></div>
    <div style="flex:1;min-width:170px">
      <label>白平衡</label>
      <select id="wb" title="拍摄值信相机测光；固定色温是声明的标准参考（经文件自身颜色标定求解），用于胶片模拟等需要整卷一致配平的场景，不是肉眼调整。">
        <option value="camera">拍摄值 · As Shot</option>
        <option value="6500k">6500K · D65 显示标准</option>
        <option value="5500k">5500K · 摄影日光/日光卷</option>
        <option value="3400k">3400K · Type A 钨丝卷</option>
        <option value="3200k">3200K · Type B 钨丝卷</option>
        <option value="9300k">9300K · 日本广播白点</option>
        <option value="daylight">相机日光标定 · 旧</option>
      </select>
    </div>
    <div style="flex:1;min-width:160px">
      <label>高光</label>
      <select id="highlight" title="LibRaw 的高光恢复方式；RAW 9 固定使用 Apple 重建。">
        <option value="clip">保持剪切 · 原始</option>
        <option value="blend">通道混合 · 温和</option>
        <option value="reconstruct">邻域重建 · 完整</option>
      </select>
      <div class="ctlFact" id="clipFact"></div>
    </div>
    <div style="flex:1;min-width:170px">
      <label>镜前滤镜</label>
      <select id="lensFilter" title="Wratten 转换滤镜，按柯达出版的 mired 位移推导；作用于前馈之前，可靠尾部与 HDR 预算都透过滤镜测量。">
        <option value="none">无</option>
        <option value="85b">85B · 日光转钨丝</option>
        <option value="85">85 · 日光转 Type A</option>
        <option value="80a">80A · 钨丝转日光</option>
        <option value="81a">81A · 轻度暖化</option>
        <option value="82a">82A · 轻度冷化</option>
      </select>
    </div>
    <div class="ctlFact" id="wbFact" style="flex-basis:100%;margin-top:0"></div>
  </div>
</div>

<div class="card" id="mobileExposureCard" data-mobile-card="exposure">
  <div class="secTitle">曝光</div>
  <div class="row">
    <div class="evMain">
      <div class="labelRow"><label title="0 EV 保留拍摄时的亮度关系。">曝光 EV</label><span class="val" id="evval">+0.00</span></div>
      <input type="range" id="ev" min="-3" max="3" step="0.05" value="0">
      <div class="modes">
        <button type="button" data-ev="-0.50"><span class="m">-0.50</span></button>
        <button type="button" data-ev="0"><span class="m">0.00</span></button>
        <button type="button" data-ev="0.50"><span class="m">+0.50</span></button>
        <button type="button" data-ev="1.00"><span class="m">+1.00</span></button>
        <button type="button" id="evReferenceBtn" title="将可靠主体中位对齐 18% 灰，并限制高光溢出。"><span class="m">亮度参考</span></button>
      </div>
      <div class="ctlFact" id="evFact"></div>
      <canvas id="sceneHist" class="histCanvas" title="可靠场景亮度直方图：EV 相对 18% 灰，与 tone 规划同一采样口径（剔除 RAW 剪切与地板钳制样本）。注记线为编译曲线端点、0 EV 与可靠尾部 p99.99。"></canvas>
    </div>
  </div>
</div>

</section>

<section class="dashboardPanel" id="tonePanel" role="tabpanel" aria-labelledby="toneTab" hidden>

<div class="card" id="toneAdjustCard" data-mobile-card="tone">
  <div class="secTitle">明暗</div>
  <div class="row">
    <div class="sliderField">
      <div class="labelRow"><label title="只改变曲线内部亮度，不移动曝光、黑点或白点。">中间调亮度</label><span class="val" id="midtoneBrightnessVal">自动</span></div>
      <input type="range" id="midtoneBrightness" min="-1" max="1" step="0.05" value="0" title="向左压低主体，向右提亮主体。">
    </div>
    <div class="sliderField">
      <div class="labelRow"><label title="围绕固定校准 pivot 改变中间调斜率，不移动 pivot 位置。">中间调对比</label><span class="val" id="midtoneContrastVal">自动</span></div>
      <input type="range" id="midtoneContrast" min="-1" max="1" step="0.05" value="0" title="向左柔和，向右增强。">
    </div>
  </div>
  <div class="row" style="margin-top:12px">
    <div class="sliderField">
      <div class="labelRow"><label title="微调 toe 的形状，不移动黑点。">暗部过渡</label><span class="val" id="shadowTransitionVal">自动</span></div>
      <input type="range" id="shadowTransition" min="-1" max="1" step="0.05" value="0" title="向左更深，向右更开放。">
    </div>
    <div class="sliderField">
      <div class="labelRow"><label title="微调 shoulder 的形状，不移动白点。">高光过渡</label><span class="val" id="highlightTransitionVal">自动</span></div>
      <input type="range" id="highlightTransition" min="-1" max="1" step="0.05" value="0" title="向左更直接，向右更柔和。">
    </div>
  </div>
  <div class="row" style="margin-top:12px">
    <div style="flex:1;min-width:170px">
      <label title="adaptive=端点追随场景百分位（默认）；evidence=端点钉在证据界：黑端点=实测噪声底 EV（有传感器先验用先验读出噪声），白端点只信可靠 RAW 尾部。pivot 锚定不变（0EV→18%）。">端点模式</label>
      <select id="endpointMode">
        <option value="adaptive">场景自适应 · 默认</option>
        <option value="evidence">证据界 · 噪声底/可靠尾部</option>
      </select>
    </div>
  </div>
  <div class="row" style="margin-top:12px">
    <div class="sliderField">
      <div class="labelRow"><label title="把曲线落到近黑的 EV 位置下移：让更深的阴影保持可读、更晚坠向黑点；重解趾部形状实现，不移动黑点、白点与曝光锚。">趾部收黑</label><span class="val" id="toeEndOffsetVal">自动</span></div>
      <input type="range" id="toeEndOffset" min="-3" max="0.5" step="0.05" value="0" title="向左更深的阴影仍可读（更晚收黑），向右更早收黑、暗部更紧。">
    </div>
    <div class="sliderField">
      <div class="labelRow"><label title="移动曲线升到近白参考（黑地板到白点跨度 90%）的场景 EV：重解肩部曲率实现，不移动黑点、白点与曝光锚。">肩部收白</label><span class="val" id="shoulderWhiteOffsetVal">自动</span></div>
      <input type="range" id="shoulderWhiteOffset" min="-2" max="3" step="0.05" value="0" title="向右更晚收白（高光层次更晚合并、滚降更柔），向左更早收白、肩部更硬。">
    </div>
  </div>
  <div class="ctlFact" id="toneFact"></div>
</div>

<div class="card" id="mobileImagingCard" data-mobile-card="imaging">
  <div class="secTitle">成像</div>
  <div class="row">
    <div style="flex:1;min-width:190px">
      <label>胶片观察位置</label>
      <select id="film" title="一次设置多层独立声明：白平衡（日光卷 5500K / 钨丝电影卷 3200K）+ 光谱前馈 + 曲线预设 + 风格配对（前馈强度与 AgX 原色几何，编辑初稿可改）。胶片决定观察者看见了什么，AgX 决定怎么显影。选中后相关控件同步更新，随时可单独调整——没有任何一层被烘焙。">
        <option value="none">无 · 场景自适应</option>
FILM_OPTIONS
      </select>
    </div>
    <div style="flex:1;min-width:190px">
      <label>曲线预设</label>
      <select id="filmCurve" title="AgX 参数空间里的具名胶片坐标（数据手册特性曲线最小二乘解）；选中后整卷一致、场景自适应关闭。">
        <option value="none">场景自适应 · 默认</option>
FILM_CURVE_OPTIONS
      </select>
    </div>
  </div>
  <div class="row" id="colorHeadBlock" style="margin-top:12px">
    <div class="sliderField">
      <div class="labelRow"><label title="放大机色头黄（Y）分色滤镜，真实暗房单位：CC 密度档位，30CC=0.30 光学密度≈相纸蓝敏层 1 档印相曝光衰减。暗房口诀：成片偏什么色，加什么色的滤镜——偏黄加 Y、去黄。方向已渲染级验证：portra400 人像样张 +30CC Y 使成片中位 b*（黄蓝轴）从 +9.4 移到 -36.8（100% 像素向去黄方向移动）。响应由该预设的拟合光谱印相模型推导（构建期把滤镜放进放大机光路重曝相纸并按暗房惯例重解曝光时间，中灰亮度不变），运行时按 CC 与曝光插值——不是后置 RGB 增益。仅负片预设显示：反转片无印相环节，物理上没有色头。">色头 Y（黄）</label><span class="val" id="colorHeadYVal">0</span></div>
      <input type="range" id="colorHeadY" min="0" max="200" step="5" value="0" title="向右加黄滤镜档位：成片去黄（偏蓝）。0=预设的中性印相决定。">
    </div>
    <div class="sliderField">
      <div class="labelRow"><label title="放大机色头品（M）分色滤镜：CC 密度档位，30CC≈相纸绿敏层 1 档印相曝光衰减。暗房口诀：成片偏品加 M、去品。方向已渲染级验证：portra400 人像样张 +30CC M 使成片中位 a*（品绿轴）从 +0.8 移到 -45.2（100% 像素向去品方向移动）。同 Y：响应来自拟合光谱印相模型，改档位后自动重解曝光时间，中灰亮度不变。仅负片预设显示：反转片无印相环节。">色头 M（品）</label><span class="val" id="colorHeadMVal">0</span></div>
      <input type="range" id="colorHeadM" min="0" max="200" step="5" value="0" title="向右加品滤镜档位：成片去品（偏绿）。0=预设的中性印相决定。">
    </div>
    <div class="ctlFact" id="colorHeadHint" style="flex-basis:100%"></div>
  </div>
  <div class="row" id="filmModeRow" style="margin-top:12px;display:none">
    <div style="flex:1;min-width:190px">
      <label>显影分工</label>
      <select id="filmMode" title="observe=胶片声明观察者看见了什么，颜色由 AgX 显影（默认，已验证路径）；full=胶片显影模型整体接管（离线烘焙光谱链 65³ 查找表：观察者逆矩阵→三层乳剂→特性曲线→印相链；实验，仅 SDR 输出，且不支持放大机色头——完整光谱链后不能追加中性轴近似）。">
        <option value="observe">观察 · AgX 显影 · 默认</option>
        <option value="full">接管 · 胶片显影模型（实验 · 仅 SDR）</option>
      </select>
    </div>
    <div id="filmCrossoverBlock" style="flex:1;min-width:190px;display:none">
      <label>层间漂移</label>
      <select id="filmCrossover" title="crossover 声明开关（仅接管模式）。关=数字中性化变体：接管 LUT 输出按像素亮度曝光除以随包的有界中性染色曲线，介质灰阶偏中性两档以内严格中性；数据手册=光谱链原样：中灰由印相求解锚定，暗部/亮部按层间数据漂移（如 Velvia 阴影温和偏冷）——量级未经外部裁决。">
        <option value="off">关 · 中性轴恒等 · 默认</option>
        <option value="datasheet">数据手册 · 中灰锚定（实验）</option>
      </select>
    </div>
  </div>
  <div class="row" style="margin-top:12px">
    <div style="flex:2;min-width:210px">
      <label>压缩核心</label>
      <select id="toneCore" title="选择亮度压缩与高光色彩路径。">
        <optgroup label="成片">
          <option value="agx" selected>AgX · 默认</option>
          <option value="gated">RAW 门控 · 保真</option>
        </optgroup>
        <optgroup label="非 AgX 对照">
          <option value="neutral">固定亮度曲线 · 诊断</option>
          <option value="lum">场景 C1 · 仅亮度</option>
        </optgroup>
      </select>
    </div>
    <div id="lumNormBlock" style="flex:1;min-width:140px;display:none">
      <label>亮度度量</label>
      <select id="lumNorm">
        <option value="y">Y · 场景亮度</option>
        <option value="power">折中 · 亮度与峰值</option>
        <option value="max">最大通道</option>
      </select>
    </div>
    <div id="agxPrimariesBlock" style="flex:1;min-width:150px">
      <label>AgX 色彩路径</label>
      <select id="agxPrimaries" title="控制饱和高光如何向白色收敛。">
        <option value="base" selected>darktable · 默认</option>
        <option value="smooth">平滑收色</option>
        <option value="punchy">纯度增强</option>
        <option value="muted">纯度柔和</option>
      </select>
    </div>
  </div>
  <div class="coreFacts" id="coreFacts" aria-live="polite"></div>
  <div id="controlHint"></div>
</div>

</section>

<section class="dashboardPanel" id="colorPanel" role="tabpanel" aria-labelledby="colorTab" hidden>

<div class="card" data-mobile-card="color">
  <div class="secTitle">颜色</div>
  <div class="row">
    <div id="punchBlock" class="sliderField">
      <div class="labelRow"><label>中频纯度</label><span class="val" id="punchVal">1.00</span></div>
      <input type="range" id="punch" min="0" max="1.5" step="0.05" value="1" title="1 使用场景分析值；0 关闭。">
    </div>
    <div id="highlightFadeBlock" class="sliderField">
      <div class="labelRow"><label title="只调整接近显示白的色度，不改变亮度 shoulder。">高光褪白</label><span class="val" id="highlightFadeVal">自动</span></div>
      <input type="range" id="highlightFade" min="-1" max="1" step="0.05" value="0" title="向左保留更多颜色，向右更早褪向白色。">
    </div>
  </div>
</div>

<div class="card" data-mobile-card="color">
  <div class="secTitle">前馈校正 · 实验</div>
  <div class="row">
    <div style="flex:2;min-width:210px">
      <label>前馈校正</label>
      <select id="sceneTransform" title="在 AgX 前校正相机的 scene-linear 色彩响应。">
SCENE_TRANSFORM_OPTIONS
      </select>
    </div>
    <div id="sceneTransformStrengthBlock" class="sliderField" style="display:none">
      <div class="labelRow"><label>前馈强度</label><span class="val" id="sceneTransformStrengthVal">1.00</span></div>
      <input type="range" id="sceneTransformStrength" min="0" max="3" step="0.05" value="1" title="1 为校准强度；更高数值用于比较。">
    </div>
  </div>
</div>

<div class="card" data-mobile-card="color">
  <div class="secTitle">风格 · LUT</div>
  <div class="row">
    <div style="flex:2;min-width:210px">
      <label>色彩风格</label>
      <select id="grade" title="曲线后的可选色彩处理。">
GRADE_OPTIONS
      </select>
    </div>
    <div class="ctlFact" id="gradeModeHint" style="display:none"></div>
    <div id="gradeStrengthBlock" class="sliderField" style="display:none">
      <div class="labelRow"><label>风格强度</label><span class="val" id="gradeStrengthVal">1.00</span></div>
      <input type="range" id="gradeStrength" min="0" max="1.5" step="0.05" value="1">
    </div>
  </div>
</div>

</section>

</div>

<div class="card previewCard">
  <div class="actions">
    <button class="go" id="go">导出</button>
    <button class="ghost" id="revealBtn" style="display:none">在 Finder 显示</button>
    <span class="previewLive" id="previewLiveBadge">实时 · PREVIEW_LONG_EDGEpx</span>
  </div>
  <details id="deliveryReport" style="display:none">
    <summary>投递报告 · 本次导出的实测真值</summary>
    <dl class="reportGrid" id="deliveryReportBody"></dl>
  </details>
  <div id="previewWrap"><img id="preview"><div id="spinner"></div><div id="status" role="status" aria-live="polite"></div></div>
  <canvas id="displayHist" class="histCanvas" title="显示码值直方图：与预览同帧的 1920px 已渲染帧，RGB 三通道与 luma，0–255。HDR 格式下为 SDR 底图直方图。"></canvas>
</div>

<nav class="mobileNav" role="tablist" aria-label="手机调整分区">
  <button type="button" class="active" id="mobileDecodeTab" role="tab" aria-selected="true" aria-controls="mobileDecodeCard" data-mobile-target="decode">解码</button>
  <button type="button" id="mobileExposureTab" role="tab" aria-selected="false" aria-controls="mobileExposureCard" data-mobile-target="exposure">曝光</button>
  <button type="button" id="mobileToneTab" role="tab" aria-selected="false" aria-controls="toneAdjustCard" data-mobile-target="tone">明暗</button>
  <button type="button" id="mobileImagingTab" role="tab" aria-selected="false" aria-controls="mobileImagingCard" data-mobile-target="imaging">成像</button>
  <button type="button" id="mobileColorTab" role="tab" aria-selected="false" aria-controls="colorPanel" data-mobile-target="color">色彩</button>
</nav>
</div>

<dialog class="outputDialog" id="outputDialog" aria-labelledby="outputDialogTitle">
  <div class="dialogPanel">
    <div class="dialogHeader">
      <h2 class="dialogTitle" id="outputDialogTitle">输出参数</h2>
    </div>
    <div class="row">
      <div style="flex:1;min-width:170px">
        <label>格式</label>
        <select id="format">
          <option value="sdr">SDR JPEG</option>
          <option value="ultrahdr">HDR gain-map · JPEG</option>
          <option value="ultrahdr-heic">HDR gain-map · HEIC</option>
        </select>
        <div class="ctlFact" id="formatModeHint" style="display:none"></div>
      </div>
      <div style="flex:1;min-width:160px">
        <label>交付档</label>
        <select id="deliveryProfile" title="只影响最后编码，不重算 AgX/HDR。archive=q100/4:4:4 验证级保真（全尺寸约 60MB）；share=q90/4:2:0 流媒体发布档（约 11–27MB，微信原图 25MB 限制内，HDR gain map 完整保留）。">
          <option value="archive">Archive · 保真</option>
          <option value="share">Share · 流媒体</option>
        </select>
      </div>
      <div style="flex:1;min-width:140px">
        <label>色域</label>
        <select id="gamut">
          <option value="srgb">sRGB · 兼容优先</option>
          <option value="p3">Display P3 · 宽色域</option>
        </select>
      </div>
      <div style="flex:0;min-width:110px">
        <label>质量</label>
        <input type="number" id="quality" min="1" max="100" value="100">
      </div>
      <div style="flex:1;min-width:140px">
        <label>色度采样</label>
        <select id="chroma" title="4:4:4 保留完整色度，4:2:0 文件更小。Ultrahdr 主图采样主要由 quality 决定。">
          <option value="444">4:4:4 · 完整</option>
          <option value="422">4:2:2</option>
          <option value="420">4:2:0 · 最小</option>
        </select>
      </div>
    </div>
    <div class="row" id="hdrBlock" style="margin-top:12px">
      <div style="min-width:220px">
        <div class="labelRow"><label>HDR 余量上限</label><span class="val" id="hdrHeadroomVal">+3.00 EV</span></div>
        <input type="range" id="hdrHeadroom" min="1" max="MAX_HDR_HEADROOM_ATTR" step="0.02" value="3">
        <div class="ctlFact" id="hdrSceneFact"></div>
        <div class="muted" id="hdrHint">实际余量由场景决定；只恢复漫反射白以上的真实亮度档数。</div>
      </div>
    </div>
    <div style="margin-top:12px">
      <label>文件夹</label>
      <div class="outdirRow">
        <input type="text" id="outdir" placeholder="默认保存到照片文件夹">
        <button class="ghost" id="outdirBtn" type="button">选择</button>
      </div>
      <div id="outdirBrowser" class="browserList"></div>
    </div>
    <div class="chk" style="margin-top:12px">
      <input type="checkbox" id="png"><label for="png" style="margin:0">附带分析图</label>
    </div>
    <div class="ctlFact" id="priorsFact"></div>
    <div class="muted" id="toneCoreExportHint" style="display:none;margin-top:10px"></div>
    <div class="dialogActions">
      <button class="ghost" id="outputCancel" type="button">取消</button>
      <button class="go" id="exportConfirm" type="button">导出</button>
    </div>
  </div>
</dialog>

<script>
const $=s=>document.querySelector(s);
const MOBILE_CARD_KEY="dngscan.mobile.card.v1";
const MOBILE_CARD_IDS=["decode","exposure","tone","imaging","color"];
function setMobileCard(cardId,focusTab=false){
  if(!MOBILE_CARD_IDS.includes(cardId))cardId="decode";
  document.body.dataset.mobileCard=cardId;
  document.querySelectorAll(".mobileNav button").forEach(tab=>{
    const active=tab.dataset.mobileTarget===cardId;
    tab.classList.toggle("active",active);
    tab.setAttribute("aria-selected",active?"true":"false");
    tab.tabIndex=active?0:-1;
    if(active&&focusTab)tab.focus();
  });
  try{localStorage.setItem(MOBILE_CARD_KEY,cardId);}catch(error){}
}
const mobileNavTabs=[...document.querySelectorAll(".mobileNav button")];
mobileNavTabs.forEach((tab,index)=>{
  tab.addEventListener("click",()=>setMobileCard(tab.dataset.mobileTarget));
  tab.addEventListener("keydown",event=>{
    if(!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Home","End"].includes(event.key))return;
    event.preventDefault();
    let next=index;
    if(["ArrowLeft","ArrowUp"].includes(event.key))next=(index-1+mobileNavTabs.length)%mobileNavTabs.length;
    if(["ArrowRight","ArrowDown"].includes(event.key))next=(index+1)%mobileNavTabs.length;
    if(event.key==="Home")next=0;
    if(event.key==="End")next=mobileNavTabs.length-1;
    setMobileCard(mobileNavTabs[next].dataset.mobileTarget,true);
  });
});
let initialMobileCard="decode";
try{initialMobileCard=localStorage.getItem(MOBILE_CARD_KEY)||initialMobileCard;}catch(error){}
setMobileCard(initialMobileCard);
const DASHBOARD_PANEL_KEY="dngscan.dashboard.panel.v1";
function setDashboardPanel(panelId,focusTab=false){
  const panel=document.getElementById(panelId);
  if(!panel)return;
  document.querySelectorAll(".dashboardPanel").forEach(candidate=>{
    const active=candidate===panel;
    candidate.classList.toggle("active",active);
    candidate.hidden=!active;
  });
  document.querySelectorAll(".dashTab").forEach(tab=>{
    const active=tab.dataset.panel===panelId;
    tab.classList.toggle("active",active);
    tab.setAttribute("aria-selected",active?"true":"false");
    tab.tabIndex=active?0:-1;
    if(active&&focusTab)tab.focus();
  });
  try{localStorage.setItem(DASHBOARD_PANEL_KEY,panelId);}catch(error){}
}
const dashboardTabs=[...document.querySelectorAll(".dashTab")];
dashboardTabs.forEach((tab,index)=>{
  tab.addEventListener("click",()=>setDashboardPanel(tab.dataset.panel));
  tab.addEventListener("keydown",event=>{
    if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;
    event.preventDefault();
    let next=index;
    if(event.key==="ArrowLeft")next=(index-1+dashboardTabs.length)%dashboardTabs.length;
    if(event.key==="ArrowRight")next=(index+1)%dashboardTabs.length;
    if(event.key==="Home")next=0;
    if(event.key==="End")next=dashboardTabs.length-1;
    setDashboardPanel(dashboardTabs[next].dataset.panel,true);
  });
});
let initialDashboardPanel="capturePanel";
try{initialDashboardPanel=localStorage.getItem(DASHBOARD_PANEL_KEY)||initialDashboardPanel;}catch(error){}
setDashboardPanel(initialDashboardPanel);
const STORE_KEY="dngscan.settings.v9";
const V8_STORE_KEY="dngscan.settings.v8";
const V7_STORE_KEY="dngscan.settings.v7";
const V6_STORE_KEY="dngscan.settings.v6";
const V5_STORE_KEY="dngscan.settings.v5";
const LEGACY_STORE_KEY="dngscan.settings.v4";
const COREIMAGE_AVAILABLE=COREIMAGE_AVAILABLE_FLAG;
function setGradeStrengthLabel(){const v=+$("#gradeStrength").value;$("#gradeStrengthVal").textContent=v.toFixed(2);}
function updateGradeUi(){$("#gradeStrengthBlock").style.display=$("#grade").value!=="none"?"block":"none";}
function setPunchLabel(){const v=+$("#punch").value;$("#punchVal").textContent=v.toFixed(2);}
function fmtBias(v){return Math.abs(v)<0.001?"自动":(v>0?"+":"")+v.toFixed(2);}
function fmtEvBias(v){return Math.abs(v)<0.001?"自动":(v>0?"+":"")+v.toFixed(2)+" EV";}
function setAdjustmentLabels(){
  ["midtoneBrightness","midtoneContrast","shadowTransition","highlightTransition","highlightFade"].forEach(id=>{$("#"+id+"Val").textContent=fmtBias(+$("#"+id).value);});
  ["toeEndOffset","shoulderWhiteOffset"].forEach(id=>{$("#"+id+"Val").textContent=fmtEvBias(+$("#"+id).value);});
}
function setSceneTransformStrengthLabel(){const v=+$("#sceneTransformStrength").value;$("#sceneTransformStrengthVal").textContent=v.toFixed(2);}
function updateSceneTransformUi(){$("#sceneTransformStrengthBlock").style.display=$("#sceneTransform").value!=="none"?"block":"none";}
const CORE_FACTS={
  gated:["亮度 <b>darktable C1</b>","色彩 <b>RAW 门控</b>"],
  agx:["亮度 <b>darktable C1</b>","色彩 <b>全图 AgX</b>"],
  lum:["曲线 <b>场景 C1</b>","色彩 <b>保持比例</b>"],
  neutral:["曲线 <b>固定</b>","色彩 <b>保持比例</b>"]
};
const CONTROL_HINTS={
  gated:"RAW 证据决定 AgX 色彩路径的混合量。",
  agx:"默认成片；全图使用 AgX 色彩路径。",
  neutral:"固定 Y 比例曲线，用来检查色调核与高饱和边界。",
  lum:"共用场景 C1，仅压缩亮度。"
};
function updateToneCoreUi(){
  const core=$("#toneCore").value;const lum=core==="lum";const neutral=core==="neutral";const control=neutral||lum;
  $("#lumNormBlock").style.display=lum?"block":"none";
  $("#agxPrimariesBlock").style.display=core==="agx"?"block":"none";
  $("#punchBlock").style.display=control?"none":"block";
  $("#toneAdjustCard").style.display=neutral?"none":"block";
  $("#highlightFadeBlock").style.display=neutral?"none":"block";
  const facts=(CORE_FACTS[core]||[]).map(v=>"<span"+(control?' class="control"':"")+">"+v+"</span>").join("");
  $("#coreFacts").innerHTML=facts;
  $("#controlHint").textContent=CONTROL_HINTS[core]||"";
}
function applyDeliveryConstraints(){
  // Archive pins q100/4:4:4 by contract; share leaves the knobs to the user. Restoring
  // saved settings must not clobber them, so this only enforces, never fills defaults.
  // HDR containers additionally pin chroma to what Core Image actually emits at the
  // profile's quality (q100→4:4:4, share→4:2:0): the control would otherwise promise a
  // subsampling the encoder cannot honour.
  const archive=$("#deliveryProfile").value==="archive";
  const hdr=["ultrahdr","ultrahdr-heic"].includes($("#format").value);
  if(archive){$("#quality").value="100";$("#chroma").value="444";}
  else if(hdr){$("#chroma").value="420";}
  $("#quality").disabled=archive;
  $("#chroma").disabled=archive||hdr;
}
function applyDeliveryDefaults(){
  // Only on an explicit profile switch: seed share's calibrated defaults.
  if($("#deliveryProfile").value==="share"){$("#quality").value="90";$("#chroma").value="420";}
  applyDeliveryConstraints();
}
function updateToneCoreExportUi(){
  const hdr=["ultrahdr","ultrahdr-heic"].includes($("#format").value);
  const incompatible=hdr&&$("#toneCore").value!=="agx";
  const hint=$("#toneCoreExportHint");
  hint.textContent=incompatible?"当前压缩核心可继续用于实时预览；HDR 容器导出目前只支持 AgX。请切换到 AgX，或改用 SDR JPEG。":"";
  hint.style.display=incompatible?"block":"none";
  $("#exportConfirm").disabled=incompatible;
  $("#exportConfirm").title=incompatible?"HDR 容器导出目前只支持 AgX":"";
}
let HDR_BACKEND_OK=true;
function updateHdrOptionGate(){
  // Mode gating (declared convention): an option a mode cannot use is greyed
  // with the reason on screen, and a selection the backend would reject never
  // rides the payload — it snaps back with a visible notice.
  const fullFilm=$("#filmCurve").value!=="none"&&$("#filmMode").value==="full";
  const hint=$("#formatModeHint");
  for(const v of ["ultrahdr","ultrahdr-heic"]){
    const option=[...$("#format").options].find(o=>o.value===v);
    if(option)option.disabled=!HDR_BACKEND_OK||fullFilm;
  }
  hint.textContent=fullFilm?"接管模式（full）暂仅支持 SDR：胶片接管显影没有 HDR 对应物。":"";
  hint.style.display=fullFilm?"":"none";
  if(fullFilm&&["ultrahdr","ultrahdr-heic"].includes($("#format").value)){
    $("#format").value="sdr";updateFormatUi();saveSettings();
    setStatus("full 模式暂仅支持 SDR，导出格式已切回 SDR JPEG。","warn");
  }
}
function updateGradeModeUi(){
  const hdr=["ultrahdr","ultrahdr-heic"].includes($("#format").value);
  const grade=$("#grade");
  const hint=$("#gradeModeHint");
  if(hdr&&grade.value!=="none"){
    grade.value="none";updateGradeUi();saveSettings();
    setStatus("HDR 容器暂不支持 display look，已重置为无。","warn");
  }
  grade.disabled=hdr;
  hint.textContent=hdr?"HDR 容器暂不支持 display look/filter——请用 SDR 导出。":"";
  hint.style.display=hdr?"":"none";
}
function updateFormatUi(){
  const hdr=["ultrahdr","ultrahdr-heic"].includes($("#format").value);
  $("#hdrBlock").style.display=hdr?"flex":"none";
  if(hdr)$("#gamut").value="p3";
  $("#gamut").disabled=hdr;
  $("#highlightFade").disabled=hdr;
  $("#highlightFadeBlock").title=hdr?"HDR 色彩几何独立处理高光，不使用 SDR 显示侧褪白。":"";
  applyDeliveryConstraints();
  updateToneCoreExportUi();
  updateToneCoreUi();
  updateGradeModeUi();
}
async function checkHdrBackend(){
  const option=[...$("#format").options].find(o=>o.value==="ultrahdr");
  const optionHeic=[...$("#format").options].find(o=>o.value==="ultrahdr-heic");
  try{
    const response=await fetch("/hdr-status");const status=await response.json();
    if(status.available){HDR_BACKEND_OK=true;updateHdrOptionGate();return;}
    HDR_BACKEND_OK=false;
    option.disabled=true;option.textContent="HDR JPEG · 当前不可用";
    optionHeic.disabled=true;optionHeic.textContent="HDR HEIC · 当前不可用";
    $("#hdrHint").textContent=status.reason||"HDR 后端未通过回读验证。";
    if(["ultrahdr","ultrahdr-heic"].includes($("#format").value)){
      $("#format").value="sdr";updateFormatUi();saveSettings();
      setStatus(status.reason||"HDR 后端未通过回读验证，已切回 SDR。","warn");
    }
  }catch(error){
    option.disabled=true;option.textContent="HDR JPEG · 探测失败";
    optionHeic.disabled=true;optionHeic.textContent="HDR HEIC · 探测失败";
  }
}
function setEvLabel(){const v=+$("#ev").value;$("#evval").textContent=(v>=0?"+":"")+v.toFixed(2);}
function setHdrLabel(){const v=+$("#hdrHeadroom").value;$("#hdrHeadroomVal").textContent="+"+v.toFixed(2)+" EV";}
function fmtPct(v){if(v===undefined||!isFinite(v))return "";if(v<=0)return "0%";if(v<0.005)return "<0.01%";if(v<1)return "~"+v.toFixed(2)+"%";return v.toFixed(1)+"%";}
function fmtEv(v){return (v>=0?"+":"")+v.toFixed(2);}
function metricText(j){
  if(!j.metrics)return "";
  const m=j.metrics;
  if(m.luma_p999_pct===undefined)return "";
  const room=m.safe_ev_remaining!==undefined?m.safe_ev_remaining:m.headroom_luma_ev;
  const sampleWan=m.metrics_sample_px?Math.round(m.metrics_sample_px/1e4):0;
  const label=j.metrics_kind==="full"?(sampleWan?" · 全图抽样 "+sampleWan+"万px 实测":" · 全分辨率真值"):" · 预览估计";
  const roomText=j.metrics_kind==="full"&&room!==undefined?" · 可再加约 "+fmtEv(room)+"EV":"";
  return label+
    " · p99.9亮度 "+fmtPct(m.luma_p999_pct)+
    " · 近白 "+fmtPct(m.near_white_pct)+
    " · 顶白 "+fmtPct(m.clipped_channel_pct)+
    roomText;
}
function fullFrameReferenceText(j){
  if(!j.ev_auto)return "";
  const a=j.ev_auto;
  let t=" · 全图亮度参考 "+fmtEv(a.ev_boost)+" EV";
  if(a.highlight_limited)t+="（高光限制，参考目标 "+fmtEv(a.ev_median_target)+"）";
  return t;
}
function sceneTransformText(j){
  if(!j.scene_transform||j.scene_transform==="无")return "";
  const s=j.scene_transform_strength!==undefined?" "+(+j.scene_transform_strength).toFixed(2):"";
  return "，相机校正 "+j.scene_transform+s;
}
function toneCoreText(j){
  if(!j.tone_core)return "";
  const labels={gated:"实验·RAW 门控",agx:"成片·AgX 全图",lum:"对照·场景 C1 仅亮度",neutral:"诊断·固定 Y 比例曲线"};
  const norms={y:"Y",power:"折中",max:"最大通道"};
  const norm=j.tone_core==="lum"&&j.lum_norm?"（"+(norms[j.lum_norm]||j.lum_norm)+"）":"";
  return "，策略 "+(labels[j.tone_core]||j.tone_core)+norm;
}
function highlightText(v){return ({clip:"保持剪切",blend:"高光混合",reconstruct:"高光重建"})[v]||v;}
function gamutText(v){return ({srgb:"sRGB",p3:"Display P3"})[v]||v;}
function formatText(v){return ({sdr:"SDR JPEG",ultrahdr:"HDR gain-map JPEG","ultrahdr-heic":"HDR gain-map HEIC"})[v]||v;}
function decoderText(j){
  if(j.decoder!=="coreimage")return "";
  const version=(j.decoder_version||"").replace(/\\.dng$/i,"");
  return "，解码 Apple RAW"+(version?" "+version:"");
}
function applyJobEv(j){
  if(j.ev!==undefined){$("#ev").value=j.ev;setEvLabel();saveSettings();}
}
function updateDecoderUi(){
  const block=$("#decoderBlock");
  const ver=$("#coreimageVersionBlock");
  const highlight=$("#highlight");
  const demosaic=$("#demosaic");
  if(!COREIMAGE_AVAILABLE){
    $("#decoder").value="libraw";
    block.classList.add("dim");
    $("#decoder").disabled=true;
    ver.style.display="none";
    highlight.disabled=false;
    demosaic.disabled=false;
    return;
  }
  block.classList.remove("dim");
  $("#decoder").disabled=false;
  const raw9=$("#decoder").value==="coreimage";
  ver.style.display=raw9?"block":"none";
  if(raw9){
    if(!highlight.disabled)highlight.dataset.librawValue=highlight.value;
    if(!demosaic.disabled)demosaic.dataset.librawValue=demosaic.value;
    highlight.value="reconstruct";
    demosaic.value="auto";
    highlight.disabled=true;
    demosaic.disabled=true;
  }else{
    highlight.disabled=false;
    demosaic.disabled=false;
    if(highlight.dataset.librawValue){highlight.value=highlight.dataset.librawValue;delete highlight.dataset.librawValue;}
    if(demosaic.dataset.librawValue){demosaic.value=demosaic.dataset.librawValue;delete demosaic.dataset.librawValue;}
  }
  // RAW 9 accepts fixed-Kelvin declarations natively; only the LibRaw-metadata
  // "daylight" anchor has no validated mapping there.
  if(raw9 && $("#wb").value==="daylight"){
    $("#wb").value="camera";
  }
}
function saveSettings(){
  try{localStorage.setItem(STORE_KEY,JSON.stringify({
    ev:$("#ev").value,quality:$("#quality").value,
    lensFilter:$("#lensFilter").value,filmCurve:$("#filmCurve").value,film:$("#film").value,
    filmMode:$("#filmMode").value,filmCrossover:$("#filmCrossover").value,
    highlight:$("#highlight").dataset.librawValue||$("#highlight").value,gamut:$("#gamut").value,wb:$("#wb").value,demosaic:$("#demosaic").dataset.librawValue||$("#demosaic").value,
    decoder:$("#decoder").value,coreimageVersion:$("#coreimageVersion").value,
    lensFilter:$("#lensFilter").value,filmCurve:$("#filmCurve").value,
    colorHeadY:$("#colorHeadY").value,colorHeadM:$("#colorHeadM").value,
    chroma:$("#chroma").value,format:$("#format").value,
    deliveryProfile:$("#deliveryProfile").value,
    toneCore:$("#toneCore").value,lumNorm:$("#lumNorm").value,agxPrimaries:$("#agxPrimaries").value,
    grade:$("#grade").value,gradeStrength:$("#gradeStrength").value,
    sceneTransform:$("#sceneTransform").value,sceneTransformStrength:$("#sceneTransformStrength").value,punch:$("#punch").value,
    midtoneBrightness:$("#midtoneBrightness").value,midtoneContrast:$("#midtoneContrast").value,
    shadowTransition:$("#shadowTransition").value,highlightTransition:$("#highlightTransition").value,highlightFade:$("#highlightFade").value,
    endpointMode:$("#endpointMode").value,
    toeEndOffset:$("#toeEndOffset").value,shoulderWhiteOffset:$("#shoulderWhiteOffset").value,
    hdrHeadroom:$("#hdrHeadroom").value,outdir:$("#outdir").value,png:$("#png").checked
  }));}catch(e){}
}
function restoreSettings(){
  let s={};let migrated=false;
  try{
    const current=localStorage.getItem(STORE_KEY);
    const v8=localStorage.getItem(V8_STORE_KEY);
    const v7=localStorage.getItem(V7_STORE_KEY);
    const v6=localStorage.getItem(V6_STORE_KEY);
    const v5=localStorage.getItem(V5_STORE_KEY);
    s=JSON.parse(current||v8||v7||v6||v5||localStorage.getItem(LEGACY_STORE_KEY)||"{}")||{};
    // v7 and earlier labelled smooth as the default. The pinned darktable scene
    // default is base, so move stored old defaults to the corrected baseline.
    if(!current&&s.agxPrimaries==="smooth"){
      s.agxPrimaries="base";migrated=true;
    }
    if(!current&&!v7&&!v6&&!v5&&s.toneCore==="gated"&&s.agxPrimaries==="base"){
      s.toneCore="agx";migrated=true;
    }
  }catch(e){}
  if(s.ev!==undefined)$("#ev").value=s.ev;
  if(s.quality)$("#quality").value=s.quality;
  if(s.highlight)$("#highlight").value=s.highlight;
  if(s.gamut)$("#gamut").value=s.gamut;
  if(s.wb)$("#wb").value=s.wb;
  if(s.demosaic)$("#demosaic").value=s.demosaic;
  if(s.decoder&&[...$("#decoder").options].some(o=>o.value===s.decoder))$("#decoder").value=s.decoder;
  if(s.coreimageVersion&&[...$("#coreimageVersion").options].some(o=>o.value===s.coreimageVersion))$("#coreimageVersion").value=s.coreimageVersion;
  if(s.chroma)$("#chroma").value=s.chroma;
  if(s.lensFilter&&[...$("#lensFilter").options].some(o=>o.value===s.lensFilter))$("#lensFilter").value=s.lensFilter;
  if(s.filmCurve&&[...$("#filmCurve").options].some(o=>o.value===s.filmCurve))$("#filmCurve").value=s.filmCurve;
  if(s.colorHeadY!==undefined)$("#colorHeadY").value=s.colorHeadY;
  if(s.colorHeadM!==undefined)$("#colorHeadM").value=s.colorHeadM;
  if(s.film&&[...$("#film").options].some(o=>o.value===s.film))$("#film").value=s.film;
  if(s.filmMode&&[...$("#filmMode").options].some(o=>o.value===s.filmMode))$("#filmMode").value=s.filmMode;
  if(s.filmCrossover&&[...$("#filmCrossover").options].some(o=>o.value===s.filmCrossover))$("#filmCrossover").value=s.filmCrossover;
  if(s.deliveryProfile&&[...$("#deliveryProfile").options].some(o=>o.value===s.deliveryProfile))$("#deliveryProfile").value=s.deliveryProfile;
  if(s.toneCore&&[...$("#toneCore").options].some(o=>o.value===s.toneCore))$("#toneCore").value=s.toneCore;
  if(s.lumNorm&&[...$("#lumNorm").options].some(o=>o.value===s.lumNorm))$("#lumNorm").value=s.lumNorm;
  if(s.agxPrimaries&&[...$("#agxPrimaries").options].some(o=>o.value===s.agxPrimaries))$("#agxPrimaries").value=s.agxPrimaries;
  if(s.grade&&[...$("#grade").options].some(o=>o.value===s.grade))$("#grade").value=s.grade;
  else if(s.filter&&s.filter!=="none"){
    const fid="filter:"+s.filter;
    if([...$("#grade").options].some(o=>o.value===fid))$("#grade").value=fid;
    else if([...$("#grade").options].some(o=>o.value===s.filter))$("#grade").value=s.filter;
  }
  else if(s.look&&s.look!=="none"){
    const lid="look:"+s.look;
    if([...$("#grade").options].some(o=>o.value===lid))$("#grade").value=lid;
    else if([...$("#grade").options].some(o=>o.value===s.look))$("#grade").value=s.look;
  }
  if(s.gradeStrength!==undefined)$("#gradeStrength").value=s.gradeStrength;
  else if(s.filterStrength!==undefined&&s.filter&&s.filter!=="none")$("#gradeStrength").value=s.filterStrength;
  else if(s.lookStrength!==undefined)$("#gradeStrength").value=s.lookStrength;
  if(s.sceneTransform&&[...$("#sceneTransform").options].some(o=>o.value===s.sceneTransform))$("#sceneTransform").value=s.sceneTransform;
  if(s.sceneTransformStrength!==undefined)$("#sceneTransformStrength").value=s.sceneTransformStrength;
  if(s.punch!==undefined)$("#punch").value=s.punch;
  ["midtoneBrightness","midtoneContrast","shadowTransition","highlightTransition","highlightFade","toeEndOffset","shoulderWhiteOffset"].forEach(id=>{if(s[id]!==undefined)$("#"+id).value=s[id];});
  if(s.endpointMode&&[...$("#endpointMode").options].some(o=>o.value===s.endpointMode))$("#endpointMode").value=s.endpointMode;
  if(s.format)$("#format").value=s.format;
  if(s.hdrHeadroom!==undefined)$("#hdrHeadroom").value=s.hdrHeadroom;
  if(s.outdir)$("#outdir").value=s.outdir;
  if(s.png!==undefined)$("#png").checked=!!s.png;
  setEvLabel();setHdrLabel();setGradeStrengthLabel();setSceneTransformStrengthLabel();setPunchLabel();setAdjustmentLabels();updateGradeUi();updateSceneTransformUi();updateToneCoreUi();updateFormatUi();updateDecoderUi();updateFilmModeUi();updateColorHeadUi();updateHdrOptionGate();
  if(migrated)saveSettings();
}
["quality","outdir","png"].forEach(id=>$("#"+id).addEventListener("change",saveSettings));
$("#gamut").addEventListener("change",()=>{saveSettings();scheduleLivePreview();});
$("#highlight").addEventListener("change",()=>{saveSettings();preparePreview();});
$("#demosaic").addEventListener("change",()=>{saveSettings();preparePreview();});
$("#chroma").addEventListener("change",saveSettings);
$("#grade").addEventListener("change",()=>{updateGradeUi();saveSettings();scheduleLivePreview();});
$("#deliveryProfile").addEventListener("change",()=>{applyDeliveryDefaults();saveSettings();});
$("#decoder").addEventListener("change",()=>{updateDecoderUi();saveSettings();preparePreview();});
$("#coreimageVersion").addEventListener("change",()=>{RAW9_APPROVALS.delete($("#input").value.trim());saveSettings();preparePreview();});
$("#wb").addEventListener("change",()=>{updateDecoderUi();updateGradeUi();saveSettings();preparePreview();});
const FILM_COMBOS=FILM_COMBOS_JSON;
$("#film").addEventListener("change",()=>{
  const combo=FILM_COMBOS[$("#film").value];
  if(combo){
    $("#wb").value=combo.wb;
    if([...$("#sceneTransform").options].some(o=>o.value===combo.st))$("#sceneTransform").value=combo.st;
    $("#filmCurve").value=combo.fc;
    if(combo.sts!==undefined){$("#sceneTransformStrength").value=combo.sts;setSceneTransformStrengthLabel();}
    if(combo.pr&&[...$("#agxPrimaries").options].some(o=>o.value===combo.pr))$("#agxPrimaries").value=combo.pr;
  }else{
    $("#wb").value="camera";$("#sceneTransform").value="none";$("#filmCurve").value="none";
    $("#sceneTransformStrength").value=1;setSceneTransformStrengthLabel();
    $("#agxPrimaries").value="base";
  }
  updateFilmModeUi();updateColorHeadUi();updateHdrOptionGate();updateDecoderUi();updateSceneTransformUi();saveSettings();preparePreview();
});
// Film-takeover controls (EXPERIMENTAL): the mode row appears only with an active
// curve preset, and the crossover declaration only in full (takeover) mode.
// Controls the takeover LUT ignores by construction: the baked spectral
// chain replaces the AgX formation wholesale, so tone shaping, primaries and
// punch are inert in full mode and the backend REJECTS full + non-agx cores.
// Disable them (with the reason in the tooltip) rather than let dead sliders
// pretend to edit.
const FILM_FULL_INERT_IDS=["toneCore","midtoneBrightness","midtoneContrast",
  "shadowTransition","highlightTransition","highlightFade","punch",
  "agxPrimaries","lumNorm","sceneTransform","sceneTransformStrength",
  "toeEndOffset","shoulderWhiteOffset","endpointMode"];
// Controls whose STALE VALUE would still alter a full-mode export if left in
// the payload (the backend also forces them off; the reset keeps the UI and
// the payload telling the same story).
const FILM_FULL_RESET_ZERO_IDS=["highlightFade"];
function updateFilmModeUi(){
  const hasCurve=$("#filmCurve").value!=="none";
  $("#filmModeRow").style.display=hasCurve?"":"none";
  const full=hasCurve&&$("#filmMode").value==="full";
  $("#filmCrossoverBlock").style.display=full?"":"none";
  // Scene-adaptive auto punch is an editorial compensation, not film
  // calibration data — film presets compile punch to 0, so the multiplier
  // slider is genuinely dead there and says so instead of pretending.
  const punchEl=$("#punch");
  if(punchEl){
    punchEl.disabled=hasCurve;
    const v=$("#punchVal");
    if(v){v.textContent=hasCurve?"胶片预设下关闭":Number(punchEl.value).toFixed(2);}
  }
  if(full&&$("#toneCore").value!=="agx"){
    $("#toneCore").value="agx";
    updateToneCoreUi();
  }
  for(const id of FILM_FULL_INERT_IDS){
    const el=$("#"+id);
    if(!el) continue;
    el.disabled=full;
    if(full){el.dataset.fullInert="1";}
    else{delete el.dataset.fullInert;}
  }
  if(full){
    for(const id of FILM_FULL_RESET_ZERO_IDS){
      const el=$("#"+id);
      if(el){el.value=0;}
    }
    if(typeof setAdjustmentLabels==="function"){setAdjustmentLabels();}
  }
}
$("#filmMode").addEventListener("change",()=>{updateFilmModeUi();updateColorHeadUi();updateHdrOptionGate();saveSettings();scheduleLivePreview();});
$("#filmCrossover").addEventListener("change",()=>{saveSettings();scheduleLivePreview();});
$("#lensFilter").addEventListener("change",()=>{saveSettings();scheduleLivePreview();});

// Enlarger colour head: shown only while the selected curve preset is a NEGATIVE
// (a print stage physically exists); reversal presets and "none" hide the block
// and reset the dials — the payload must never carry filtration the server would
// reject as physically meaningless.
const FILM_COLOR_HEADS=FILM_COLOR_HEADS_JSON;
function setColorHeadLabels(){
  $("#colorHeadYVal").textContent=$("#colorHeadY").value+" CC";
  $("#colorHeadMVal").textContent=$("#colorHeadM").value+" CC";
}
function updateColorHeadUi(){
  // Discoverability contract (refined): once ANY film curve is selected the
  // colour-head block stays visible — unavailable states disable with the
  // reason on screen (reversal / full mode). With NO film selected at all the
  // whole block hides: film adjustments below an empty selector are noise.
  // Disabled states also RESET to 0 — a stale non-zero Y/M must never ride a
  // payload the backend would reject (full mode) or silently ignore.
  const preset=$("#filmCurve").value;
  const block=$("#colorHeadBlock");
  if(preset==="none"){
    block.style.display="none";
    for(const id of ["colorHeadY","colorHeadM"]){const el=$("#"+id);el.disabled=true;el.value=0;}
    setColorHeadLabels();
    return;
  }
  block.style.display="";
  const isNegative=!!FILM_COLOR_HEADS[preset];
  const isFull=$("#filmMode").value==="full";
  let reason="";
  if(!isNegative){reason="反转片没有印相色头：幻灯片自身就是显示介质，物理上不存在放大机环节";}
  else if(isFull){reason="接管 LUT 固定烘焙于 0CC——切换“显影分工”回“观察”模式即可启用色头";}
  const enabled=!reason;
  for(const id of ["colorHeadY","colorHeadM"]){
    const el=$("#"+id);
    el.disabled=!enabled;
    if(!enabled){el.value=0;}
  }
  const hint=$("#colorHeadHint");
  hint.textContent=reason;
  hint.style.display=enabled?"none":"";
  setColorHeadLabels();
}
["colorHeadY","colorHeadM"].forEach(id=>$("#"+id).oninput=()=>{setColorHeadLabels();saveSettings();scheduleLivePreview();});
$("#filmCurve").addEventListener("change",()=>{updateFilmModeUi();updateColorHeadUi();updateHdrOptionGate();saveSettings();scheduleLivePreview();});
$("#toneCore").addEventListener("change",()=>{updateToneCoreUi();updateToneCoreExportUi();saveSettings();preparePreview();});
$("#lumNorm").addEventListener("change",()=>{saveSettings();scheduleLivePreview();});
$("#agxPrimaries").addEventListener("change",()=>{saveSettings();scheduleLivePreview();});
$("#sceneTransform").addEventListener("change",()=>{updateSceneTransformUi();saveSettings();scheduleLivePreview();});
$("#format").addEventListener("change",()=>{updateFormatUi();saveSettings();scheduleLivePreview();});
$("#ev").oninput=()=>{setEvLabel();saveSettings();scheduleLivePreview();};
$("#hdrHeadroom").oninput=()=>{setHdrLabel();saveSettings();};
$("#gradeStrength").oninput=()=>{setGradeStrengthLabel();saveSettings();scheduleLivePreview();};
$("#punch").oninput=()=>{setPunchLabel();saveSettings();scheduleLivePreview();};
[
  "midtoneBrightness","midtoneContrast","shadowTransition","highlightTransition","highlightFade"
].forEach(id=>$("#"+id).oninput=()=>{setAdjustmentLabels();saveSettings();scheduleLivePreview();});
[
  "toeEndOffset","shoulderWhiteOffset"
].forEach(id=>{
  $("#"+id).oninput=()=>{setAdjustmentLabels();saveSettings();scheduleLivePreview();};
  // The compiled toe-end / shoulder-start facts in #toneFact come from /prepare; a
  // released slider refreshes them so the printed EV matches the curve on screen.
  $("#"+id).addEventListener("change",()=>{preparePreview();});
});
$("#endpointMode").addEventListener("change",()=>{saveSettings();preparePreview();});
$("#sceneTransformStrength").oninput=()=>{setSceneTransformStrengthLabel();saveSettings();scheduleLivePreview();};
restoreSettings();
checkHdrBackend();
document.querySelectorAll("button[data-ev]").forEach(b=>b.onclick=()=>{$("#ev").value=b.dataset.ev;setEvLabel();saveSettings();scheduleLivePreview();});
let lastSavedPath="";

let curDir=INIT_DIR;
$("#filePicker").addEventListener("change",async()=>{
  const picker=$("#filePicker");const file=picker.files&&picker.files[0];
  if(!file)return;
  // A newly selected source invalidates every in-flight response immediately,
  // including one that might finish while the upload is still in progress.
  beginPreviewSession();
  picker.disabled=true;$("#input").value="";lastSavedPath="";$("#revealBtn").style.display="none";
  setStatus("正在读取 "+file.name+"…","");
  try{
    const response=await fetch("/upload?name="+encodeURIComponent(file.name),{
      method:"POST",headers:{"Content-Type":"application/octet-stream"},body:file
    });
    const result=await response.json();
    if(!result.ok){picker.value="";setStatus("文件选择失败："+result.error,"err");return;}
    $("#input").value=result.path;
    RAW9_PROBES.clear();RAW9_PROBE_REQUESTS.clear();RAW9_APPROVALS.clear();
    if(!$("#outdir").value.trim())$("#outdir").value=INIT_DIR;
    saveSettings();
    setStatus("已选择："+file.name,"ok");
    fetchDecodeSupport(result.path);
    await preparePreview();
  }catch(error){
    picker.value="";
    setStatus("文件选择失败："+error,"err");
  }finally{
    picker.disabled=false;
  }
});

async function listOutDir(d){
  const r=await fetch("/list?dir="+encodeURIComponent(d));const j=await r.json();
  const b=$("#outdirBrowser");b.innerHTML="";
  const mk=(t,fn,cls)=>{const e=document.createElement("div");e.textContent=t;e.onclick=fn;if(cls)e.className=cls;b.appendChild(e);};
  mk("✓ 就用这里："+j.cwd,()=>{$("#outdir").value=j.cwd;b.style.display="none";saveSettings();},"pick");
  mk("↺ 使用默认目录："+INIT_DIR,()=>{$("#outdir").value=INIT_DIR;b.style.display="none";saveSettings();});
  mk("⬆︎ "+j.parent,()=>listOutDir(j.parent));
  j.dirs.forEach(d2=>mk("📁 "+d2,()=>listOutDir(j.cwd+"/"+d2)));
}
$("#outdirBtn").onclick=()=>{
  const b=$("#outdirBrowser");
  if(b.style.display==="block"){b.style.display="none";return;}
  b.style.display="block";
  const seed=$("#outdir").value.trim()
    ||curDir;
  listOutDir(seed);
};

function payload(){
  const input=$("#input").value.trim();
  if(!input){setStatus("请先选择一个 DNG/RAW 文件","err");return null;}
  return {
    input,highlight:$("#highlight").value,gamut:$("#gamut").value,wb:$("#wb").value,demosaic:$("#demosaic").value,
    decoder:$("#decoder").value,coreimageVersion:$("#coreimageVersion").value,
    lensFilter:$("#lensFilter").value,filmCurve:$("#filmCurve").value,
    colorHeadY:+$("#colorHeadY").value,colorHeadM:+$("#colorHeadM").value,
    filmMode:$("#filmMode").value,filmCrossover:$("#filmCrossover").value,
    chroma:$("#chroma").value,format:$("#format").value,
    deliveryProfile:$("#deliveryProfile").value,
    toneCore:$("#toneCore").value,lumNorm:$("#lumNorm").value,agxPrimaries:$("#agxPrimaries").value,
    grade:$("#grade").value,gradeStrength:+$("#gradeStrength").value,
    sceneTransform:$("#sceneTransform").value,sceneTransformStrength:+$("#sceneTransformStrength").value,
    punch:+$("#punch").value,
    midtoneBrightness:+$("#midtoneBrightness").value,midtoneContrast:+$("#midtoneContrast").value,
    shadowTransition:+$("#shadowTransition").value,highlightTransition:+$("#highlightTransition").value,
    highlightFade:["ultrahdr","ultrahdr-heic"].includes($("#format").value)?0:+$("#highlightFade").value,
    endpointMode:$("#endpointMode").value,
    toeEndOffset:+$("#toeEndOffset").value,shoulderWhiteOffset:+$("#shoulderWhiteOffset").value,
    hdrHeadroom:+$("#hdrHeadroom").value,ev:+$("#ev").value,quality:+$("#quality").value,
    outdir:$("#outdir").value.trim(),png:$("#png").checked
  };
}

async function postJob(path, body, signal){
  const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body),signal});
  return await r.json();
}
const RAW9_PROBES=new Map();
const RAW9_PROBE_REQUESTS=new Map();
const RAW9_APPROVALS=new Map();
function raw9Probe(input){
  const cached=RAW9_PROBES.get(input);
  if(cached)return Promise.resolve(cached);
  let pending=RAW9_PROBE_REQUESTS.get(input);
  if(!pending){
    pending=postJob("/raw9-support",{input}).then(j=>{
      if(j.ok)RAW9_PROBES.set(input,j);
      return j;
    }).finally(()=>{
      if(RAW9_PROBE_REQUESTS.get(input)===pending)RAW9_PROBE_REQUESTS.delete(input);
    });
    RAW9_PROBE_REQUESTS.set(input,pending);
  }
  return pending;
}
async function ensureRaw9Support(body){
  if(body.decoder!=="coreimage")return true;
  const key=body.input;
  const j=await raw9Probe(key);
  if(!j.ok){setStatus("RAW 9 探测失败："+(j.error||"未知错误"),"err");return false;}
  const switchToLibRaw=(message)=>{
    window.alert(message+"\\n\\n将改用 LibRaw。");
    $("#decoder").value="libraw";updateDecoderUi();saveSettings();
    body.decoder="libraw";body.coreimageVersion="auto";
    setStatus(message+" 已改用 LibRaw。","warn");
  };
  if(!j.coreimage_available||j.probe_error){switchToLibRaw(j.message);return true;}
  const offered=(j.versions_offered||[]).map(v=>String(v).replace(/\\.dng$/i,""));
  if(body.coreimageVersion!=="auto"){
    if(!offered.includes(String(body.coreimageVersion))){
      setStatus(j.message+" 当前指定的 RAW "+body.coreimageVersion+" 也不可用。","err");
      return false;
    }
    if(body.coreimageVersion!=="9"){
      setStatus(j.message+" 当前明确使用 RAW "+body.coreimageVersion+"。","warn");
    }
    return true;
  }
  if(j.raw9_supported)return true;
  if(!j.fallback_version){switchToLibRaw(j.message);return true;}
  const approved=RAW9_APPROVALS.get(key);
  if(approved===j.fallback_version){body.coreimageVersion=approved;return true;}
  const useFallback=window.confirm(
    j.message+"\\n\\n确定：继续使用 Apple RAW "+j.fallback_version+"\\n取消：改用 LibRaw"
  );
  if(useFallback){
    RAW9_APPROVALS.set(key,j.fallback_version);
    body.coreimageVersion=j.fallback_version;
    setStatus("此文件将使用 Apple RAW "+j.fallback_version+"。","warn");
  }else{
    $("#decoder").value="libraw";updateDecoderUi();saveSettings();
    body.decoder="libraw";body.coreimageVersion="auto";
    setStatus("此文件不支持 RAW 9，已改用 LibRaw。","warn");
  }
  return true;
}
let DETECTED_READY=false;
function setFact(sel,text,isWarn){const el=$(sel);el.textContent=text||"";el.classList.toggle("warn",!!isWarn);}
function renderDetectedParams(d){
  // Measured scene facts land NEXT TO the control they inform, so the number
  // is in view while the hand is on the slider — never a scroll away.
  DETECTED_READY=!!(d&&typeof d==="object");
  if(!DETECTED_READY){
    ["#decoderFact","#wbFact","#clipFact","#evFact","#toneFact","#hdrSceneFact"].forEach(s=>setFact(s,""));
    return;
  }
  const ev=v=>(v>=0?"+":"")+(+v).toFixed(2)+" EV";
  setFact("#decoderFact",d.data_support?"⚠ 机型数据："+d.data_support:"",true);
  setFact("#wbFact",d.wb_degradation?"⚠ 白平衡："+d.wb_degradation:"",true);
  setFact("#clipFact",d.raw_clip_union_pct!==null?"实测 RAW 剪切 "+(+d.raw_clip_union_pct).toFixed(2)+"%（≥1 通道）":"");
  const evBits=[];
  if(d.body_median_ev!==null)evBits.push("实测主体中位 "+ev(d.body_median_ev));
  if(d.sparse_emitter)evBits.push("稀疏光源 · 夜景/舞台策略");
  setFact("#evFact",evBits.join(" · "));
  const toneBits=[];
  if(d.black_ev!==null&&d.white_ev!==null)toneBits.push("编译曲线 "+ev(d.black_ev)+" .. "+ev(d.white_ev));
  if(d.contrast!==null)toneBits.push("对比 "+(+d.contrast).toFixed(2));
  if(d.toe_end_ev!==null&&d.toe_end_ev!==undefined)toneBits.push("趾部收黑 "+ev(d.toe_end_ev));
  if(d.shoulder_white_ev!==null&&d.shoulder_white_ev!==undefined)toneBits.push("肩部收白 "+ev(d.shoulder_white_ev));
  if(d.endpoint_mode==="evidence")toneBits.push("端点 证据界"+(d.endpoint_note?"（"+d.endpoint_note+"）":""));
  setFact("#toneFact",toneBits.join(" · "));
  if(d.reliable_tail_ev!==null){
    const bits=["实测可靠尾部 "+ev(d.reliable_tail_ev)+"（p99.99）"];
    if(d.hdr_earned_ev!==null)bits.push("场景可挣余量 +"+(+d.hdr_earned_ev).toFixed(2)+" EV");
    setFact("#hdrSceneFact",bits.join(" · "));
  }else{
    setFact("#hdrSceneFact","⚠ 可靠尾部不可用 · HDR 余量将为 0",true);
  }
}
// Each probe line lands next to the control it informs: file identity and the
// Evidence tier by the picker, the two decoder tiers by the decoder select,
// the sensor-priors state by the analysis-plates toggle. Unrecognized lines
// (e.g. probe failures) fall through to the decoder slot as warnings.
const SUPPORT_ROUTE=[["机型：","#fileFact"],["Evidence（LibRaw）：","#fileFact"],
  ["LibRaw 场景解码：","#decodeTierFact"],["Apple RAW：","#decodeTierFact"],
  ["传感器先验：","#priorsFact"]];
function renderDecodeSupport(lines){
  const buckets=new Map();
  for(const line of lines||[]){
    const hit=SUPPORT_ROUTE.find(([p])=>line.startsWith(p));
    const sel=hit?hit[1]:"#decodeTierFact";
    if(!buckets.has(sel))buckets.set(sel,[]);
    buckets.get(sel).push(line);
  }
  for(const [prefix,sel] of SUPPORT_ROUTE)if(!buckets.has(sel))buckets.set(sel,[]);
  for(const [sel,rows] of buckets){
    const joined=sel==="#fileFact"?rows.join(" · "):rows.join("\\n");
    setFact(sel,joined,/[✗⚠]|失败/.test(joined));
  }
}
async function fetchDecodeSupport(input){
  // File capability is stable for the selected source. Render it once when the
  // file changes; preview reconfiguration must not replay this success notice.
  try{
    const j=await raw9Probe(input);
    if($("#input").value.trim()===input&&j&&j.support_lines)renderDecodeSupport(j.support_lines);
  }catch(_){/* probe display is best-effort */}
}
const PREVIEW_CLIENT_ID=(globalThis.crypto&&crypto.randomUUID)?crypto.randomUUID():(Date.now()+"-"+Math.random());
let PREVIEW_SESSION_SERIAL=0;
let PREVIEW_SESSION_ID=PREVIEW_CLIENT_ID+":0";
let PREVIEW_GENERATION=0;
let PREVIEW_READY=false;
let previewRaf=0;
let previewAbort=null;
let prepareAbort=null;
function setPreviewBadge(text,state){
  const badge=$("#previewLiveBadge");
  badge.textContent=text;
  badge.className="previewLive"+(state?" "+state:"");
}
function beginPreviewSession(){
  PREVIEW_SESSION_SERIAL+=1;
  PREVIEW_SESSION_ID=PREVIEW_CLIENT_ID+":"+PREVIEW_SESSION_SERIAL;
  PREVIEW_GENERATION=0;PREVIEW_READY=false;
  if(previewRaf){cancelAnimationFrame(previewRaf);previewRaf=0;}
  if(previewAbort){previewAbort.abort();previewAbort=null;}
  if(prepareAbort){prepareAbort.abort();prepareAbort=null;}
  return PREVIEW_SESSION_ID;
}
function scheduleLivePreview(){
  if(!PREVIEW_READY||!$("#input").value.trim())return;
  if(previewRaf)return;
  previewRaf=requestAnimationFrame(()=>{previewRaf=0;requestPreview();});
}
async function requestPreview({includeMetrics=false,busy=false,evAuto=false,prefix="预览"}={}){
  if(!PREVIEW_READY)return false;
  const body=payload();if(!body)return false;
  const generation=++PREVIEW_GENERATION;
  body.previewSession=PREVIEW_SESSION_ID;
  body.generation=generation;
  body.includeMetrics=!!includeMetrics;
  if(evAuto)body.evAuto=true;
  if(previewAbort)previewAbort.abort();
  const controller=new AbortController();previewAbort=controller;
  if(busy)beginBusy();
  setPreviewBadge(evAuto?"亮度参考计算中 · PREVIEW_LONG_EDGEpx":"处理中 · PREVIEW_LONG_EDGEpx","busy");
  try{
    const j=await postJob("/preview",body,controller.signal);
    if(controller.signal.aborted||generation!==PREVIEW_GENERATION||j.superseded)return false;
    if(!handleJobResult(j,prefix)){
      setStatus("错误："+(j.error||"预览失败"),"err");setPreviewBadge("实时预览错误 · PREVIEW_LONG_EDGEpx","err");return false;
    }
    setPreviewBadge("实时 · PREVIEW_LONG_EDGEpx","");
    return true;
  }catch(e){
    if(e&&e.name==="AbortError")return false;
    if(generation===PREVIEW_GENERATION){setStatus("请求失败："+e,"err");setPreviewBadge("实时预览错误 · PREVIEW_LONG_EDGEpx","err");}
    return false;
  }finally{
    if(generation===PREVIEW_GENERATION){
      if(previewAbort===controller)previewAbort=null;
      if(busy)endBusy();
    }
  }
}
async function preparePreview(){
  const body=payload();if(!body)return;
  const session=beginPreviewSession();
  body.previewSession=session;
  setPreviewBadge("准备实时预览 · PREVIEW_LONG_EDGEpx","busy");
  try{if(!await ensureRaw9Support(body)){setPreviewBadge("实时预览未就绪 · PREVIEW_LONG_EDGEpx","err");return;}}catch(e){setStatus("RAW 9 探测失败："+e,"err");setPreviewBadge("实时预览错误 · PREVIEW_LONG_EDGEpx","err");return;}
  const controller=new AbortController();prepareAbort=controller;
  try{
    const j=await postJob("/prepare",body,controller.signal);
    if(controller.signal.aborted||session!==PREVIEW_SESSION_ID)return;
    if(j&&j.ok){
      renderDetectedParams(j.detected);PREVIEW_READY=true;setPreviewBadge("实时 · PREVIEW_LONG_EDGEpx","");
      await requestPreview();
    }else if(j&&j.error){setStatus(j.error,"err");renderDetectedParams(null);setPreviewBadge("实时预览错误 · PREVIEW_LONG_EDGEpx","err");}
  }catch(e){
    if(!(e&&e.name==="AbortError")&&session===PREVIEW_SESSION_ID){setStatus("预览准备失败："+e,"err");setPreviewBadge("实时预览错误 · PREVIEW_LONG_EDGEpx","err");}
  }finally{
    if(prepareAbort===controller)prepareAbort=null;
  }
}
function beginBusy(){const w=$("#previewWrap");w.classList.add("loading");}
function endBusy(){const w=$("#previewWrap");w.classList.remove("loading");}
function setPreviewImage(b64, ondone){
  const img=$("#preview");
  img.onload=()=>{img.style.display="block";endBusy();if(ondone)ondone();};
  img.onerror=()=>{endBusy();};
  img.src="data:image/jpeg;base64,"+b64;
}

function fmtMB(bytes){return bytes>=1048576?(bytes/1048576).toFixed(2)+" MB":Math.round(bytes/1024)+" KB";}
function renderDeliveryReport(j){
  const box=$("#deliveryReport");const body=$("#deliveryReportBody");
  const c=j.hdr_container;
  if(!c||typeof c!=="object"){box.style.display="none";body.innerHTML="";return;}
  const rows=[];
  const add=(k,v,warn)=>{if(v!==undefined&&v!==null&&v!=="")rows.push("<dt>"+k+"</dt><dd"+(warn?' class="warn"':"")+">"+v+"</dd>");};
  add("交付",(c.delivery_profile||"")+" · "+(c.delivery_container==="heic"?"HEIC":"JPEG")+" · q"+c.delivery_quality);
  if(c.file_size_bytes!==undefined)add("文件大小",fmtMB(c.file_size_bytes));
  add("主图采样",c.chroma_subsampling,c.chroma_subsampling!=="4:4:4"&&c.delivery_profile==="archive");
  if(c.rendered_headroom_ev!==undefined){
    add("HDR 余量","场景挣得 +"+(+c.rendered_headroom_ev).toFixed(2)+" EV · 实际使用 +"+(+c.actual_headroom_ev).toFixed(2)+" EV · 容量 +"+(+c.display_headroom_ev).toFixed(2)+" EV");
  }
  if(c.shoulder_segments!==undefined){
    const shape=c.shoulder_segments<=1?"单段":"细分（"+c.shoulder_segments+" 段）";
    add("HDR shoulder",shape+(c.shoulder_alpha!==undefined?" · alpha "+(+c.shoulder_alpha).toFixed(3):""));
  }
  if(c.block_p99_relative_error!==undefined){
    add("HDR 回读误差","块级 p99 "+(+c.block_p99_relative_error*100).toFixed(2)+"% · 块级色品 "+(+c.block_chroma_error*100).toFixed(2)+"% · 像素色品 "+(+c.chroma_error*100).toFixed(2)+"%");
  }
  if(c.base_mean_code_error!==undefined){
    add("SDR 底图误差","平均 "+(+c.base_mean_code_error).toFixed(2)+" 码值 · 8×8 p99 "+(+c.base_block_p99_code_error).toFixed(2)+" 码值");
  }
  if(c.headroom_error_ev!==undefined)add("声明余量误差",(+c.headroom_error_ev).toFixed(4)+" EV");
  if(c.channel_separation!==undefined)add("色度自由度 rho",(+c.channel_separation).toFixed(3));
  if(c.hdr_plan)add("HDR plan",c.hdr_plan);
  body.innerHTML=rows.join("");
  box.style.display=rows.length?"block":"none";
}
// Realtime histograms: pure canvas, display-only by design (no hover, no
// selection, no listeners). Both are drawn from the same /preview response as
// the frame they describe, so latest-wins keeps image and histograms in step.
const HIST_TOP=15,HIST_PAD=4,HIST_FONT="11px -apple-system,system-ui,sans-serif";
function histSetup(canvas){
  const w=canvas.clientWidth||600,h=canvas.clientHeight||92;
  if(canvas.width!==w)canvas.width=w;
  if(canvas.height!==h)canvas.height=h;
  const ctx=canvas.getContext("2d");
  ctx.clearRect(0,0,w,h);
  return ctx;
}
function histLogs(counts){return counts.map(v=>Math.log1p(v));}
function histMarkers(ctx,markers,W,H){
  // Left-to-right label layout with a bottom fallback row, so close markers
  // (e.g. 0EV next to p99.99) never paint over each other.
  markers.sort((a,b)=>a.x-b.x);
  let endTop=-1e9,endBottom=-1e9;
  for(const m of markers){
    if(!(m.x>=0&&m.x<=W))continue;
    ctx.save();
    if(m.dash)ctx.setLineDash([3,3]);
    ctx.strokeStyle=m.color;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(m.x+.5,HIST_TOP-3);ctx.lineTo(m.x+.5,H-HIST_PAD);ctx.stroke();
    ctx.setLineDash([]);ctx.fillStyle=m.color;ctx.font=HIST_FONT;
    const tw=ctx.measureText(m.label).width;
    let tx=m.x+3;if(tx+tw>W-2)tx=m.x-tw-3;
    if(tx>=endTop+4){ctx.fillText(m.label,tx,10);endTop=tx+tw;}
    else if(tx>=endBottom+4){ctx.fillText(m.label,tx,H-HIST_PAD-2);endBottom=tx+tw;}
    ctx.restore();
  }
}
function histSeries(ctx,logs,max,W,H,stroke,fill){
  const n=logs.length,base=H-HIST_PAD,span=H-HIST_TOP-HIST_PAD;
  ctx.beginPath();ctx.moveTo(0,base);
  for(let i=0;i<n;i++)ctx.lineTo((i+0.5)/n*W,base-Math.max(0,logs[i]/max)*span);
  ctx.lineTo(W,base);
  if(fill){ctx.fillStyle=fill;ctx.fill();}
  if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}
}
function renderSceneHistogram(h){
  const c=$("#sceneHist");
  if(!h||!h.counts){c.style.display="none";return;}
  c.style.display="block";
  const ctx=histSetup(c),W=c.width,H=c.height;
  const logs=histLogs(h.counts),max=Math.max(1e-6,...logs);
  const n=logs.length,bw=W/n,base=H-HIST_PAD,span=H-HIST_TOP-HIST_PAD;
  ctx.fillStyle="rgba(91,140,255,.6)";
  for(let i=0;i<n;i++){
    const v=logs[i]/max;if(v<=0)continue;
    ctx.fillRect(i*bw,base-v*span,Math.max(bw-0.5,0.75),v*span);
  }
  const evx=ev=>(ev-h.ev_min)/(h.ev_max-h.ev_min)*W;
  const markers=[{x:evx(h.pivot_ev||0),color:"#7d8698",label:"0EV",dash:true}];
  if(h.black_ev!=null)markers.push({x:evx(h.black_ev),color:"#8fa0c4",label:"黑 "+(+h.black_ev).toFixed(1)});
  if(h.white_ev!=null)markers.push({x:evx(h.white_ev),color:"#e7e9ee",label:"白 "+(+h.white_ev).toFixed(1)});
  if(h.reliable_tail_ev!=null)markers.push({x:evx(h.reliable_tail_ev),color:"#ffc46b",label:"p99.99"});
  histMarkers(ctx,markers,W,H);
}
function renderDisplayHistogram(h,earnedEv){
  const c=$("#displayHist");
  if(!h||!h.luma){c.style.display="none";return;}
  c.style.display="block";
  const ctx=histSetup(c),W=c.width,H=c.height;
  const series=[["r","rgba(255,122,122,.9)"],["g","rgba(126,217,135,.9)"],["b","rgba(122,162,255,.9)"]];
  const lumaLogs=histLogs(h.luma);
  let max=Math.max(...lumaLogs);
  const chLogs={};
  for(const [name] of series){chLogs[name]=histLogs(h[name]);max=Math.max(max,...chLogs[name]);}
  if(!(max>0))max=1;
  histSeries(ctx,lumaLogs,max,W,H,null,"rgba(231,233,238,.28)");
  for(const [name,stroke] of series)histSeries(ctx,chLogs[name],max,W,H,stroke,null);
  const hdr=["ultrahdr","ultrahdr-heic"].includes($("#format").value);
  if(hdr&&earnedEv!=null){
    ctx.fillStyle="#ffc46b";ctx.font=HIST_FONT;
    const note="SDR 底图 · HDR 已挣余量 +"+(+earnedEv).toFixed(1)+" EV";
    ctx.fillText(note,W-ctx.measureText(note).width-6,10);
  }
}
function handleJobResult(j, prefix){
  if(!j.ok)return false;
  applyJobEv(j);
  renderDeliveryReport(j);
  // Full exports do not recompute the realtime histograms; the last live pair
  // stays valid for the same parameters, so absent fields leave them untouched.
  if(j.scene_histogram)renderSceneHistogram(j.scene_histogram);
  if(j.display_histogram)renderDisplayHistogram(j.display_histogram,j.hdr_earned_ev);
  // Scene facts live in the detection card and export truth in the delivery report;
  // a successful preview needs no small print. Auto-EV keeps its one-line feedback.
  setStatus(j.ev_auto?prefix+"：EV "+fmtEv(j.ev)+fullFrameReferenceText(j):"","ok");
  setPreviewImage(j.preview);
  return true;
}

$("#evReferenceBtn").onclick=async()=>{
  if(!payload())return;
  if(!PREVIEW_READY)await preparePreview();
  if(!PREVIEW_READY)return;
  $("#evReferenceBtn").disabled=true;$("#revealBtn").style.display="none";setStatus("正在计算亮度参考…","");
  try{await requestPreview({includeMetrics:true,busy:true,evAuto:true,prefix:"全图亮度参考预览"});}
  finally{$("#evReferenceBtn").disabled=false;}
};

function openOutputDialog(){
  const dialog=$("#outputDialog");
  if(typeof dialog.showModal==="function")dialog.showModal();
  else dialog.setAttribute("open","");
}
function closeOutputDialog(){
  const dialog=$("#outputDialog");
  if(typeof dialog.close==="function")dialog.close();
  else dialog.removeAttribute("open");
}
$("#go").onclick=openOutputDialog;
$("#outputCancel").onclick=closeOutputDialog;
$("#exportConfirm").onclick=async()=>{
  const body=payload();if(!body){closeOutputDialog();return;}
  try{if(!await ensureRaw9Support(body))return;}catch(e){setStatus("RAW 9 探测失败："+e,"err");return;}
  closeOutputDialog();
  $("#go").disabled=true;$("#exportConfirm").disabled=true;$("#revealBtn").style.display="none";beginBusy();setStatus("正在全尺寸导出…","");
  try{
    const j=await postJob("/export",body);
    if(!j.ok){endBusy();setStatus("错误："+j.error,"err");}
    else{applyJobEv(j);setStatus("已保存："+j.saved.join(" · ")+"（"+formatText(j.format)+"，EV "+fmtEv(j.ev)+"，曝光增益 "+j.gain.toFixed(3)+"，高光 "+highlightText(j.highlight)+"，色域 "+gamutText(j.gamut)+decoderText(j)+toneCoreText(j)+sceneTransformText(j)+fullFrameReferenceText(j)+metricText(j)+"）","ok");
      renderDeliveryReport(j);
      lastSavedPath=j.saved[0]||"";$("#revealBtn").style.display=lastSavedPath?"inline-block":"none";setPreviewImage(j.preview);}
  }catch(e){endBusy();setStatus("请求失败："+e,"err");}
  $("#go").disabled=false;$("#exportConfirm").disabled=false;updateToneCoreExportUi();
};
$("#revealBtn").onclick=async()=>{
  if(!lastSavedPath)return;
  $("#revealBtn").disabled=true;
  try{
    const j=await postJob("/reveal",{path:lastSavedPath});
    if(!j.ok)setStatus("Finder 打开失败："+j.error,"err");
  }catch(e){setStatus("Finder 请求失败："+e,"err");}
  $("#revealBtn").disabled=false;
};
function setStatus(t,c){const s=$("#status");s.textContent=t;s.className=c||"";s.title=t||"";}
</script>
</div></body></html>
"""


_LOOK_LABELS = {
    "optic_warm_cyan": "暖肤冷调",
}


def _grade_options_html() -> str:
    from ..display_filter import DISPLAY_FILTERS, FILTER_CHOICES, filter_available
    from ..grade import grade_id_for_filter, grade_id_for_look
    from ..look import LOOK_CHOICES

    lines = ['        <option value="none">无</option>']
    lines.append('        <optgroup label="内置风格">')
    for name in LOOK_CHOICES:
        if name == "none":
            continue
        label = _LOOK_LABELS.get(name, name.replace("fuji_", "Fujifilm ").replace("_", " "))
        gid = grade_id_for_look(name)
        lines.append(f'          <option value="{gid}">{label}</option>')
    lines.append("        </optgroup>")
    available_filters = [name for name in FILTER_CHOICES if name != "none" and filter_available(name)]
    if available_filters:
        lines.append('        <optgroup label="本地 LUT">')
        for name in available_filters:
            gid = grade_id_for_filter(name)
            lines.append(f'          <option value="{gid}">{DISPLAY_FILTERS[name].label}</option>')
        lines.append("        </optgroup>")
    return "\n".join(lines)


def _scene_transform_options_html() -> str:
    from ..scene_transform import SCENE_TRANSFORM_CHOICES, scene_transform_label

    lines = []
    for name in SCENE_TRANSFORM_CHOICES:
        lines.append(f'        <option value="{name}">{scene_transform_label(name)}</option>')
    return "\n".join(lines)


def _film_options_html() -> tuple[str, str, str, str]:
    from ..film_curve import (
        FILM_CURVE_PRESETS,
        color_head_supported,
        film_style_pairing,
    )
    from ..scene_transform import SCENE_TRANSFORMS

    film_opts, curve_opts, combos, heads = [], [], {}, {}
    for key, preset in FILM_CURVE_PRESETS.items():
        label = str(preset.get("label", key))
        fit = preset.get("fit", {})
        rms = fit.get("rms_stop")
        note = f"（拟合残差 {rms:.3f} stop）" if isinstance(rms, (int, float)) else ""
        film_opts.append(f'        <option value="{key}">{label}</option>')
        curve_opts.append(f'        <option value="{key}" title="{note}">{label}</option>')
        combo = preset.get("combo", {})
        st = str(combo.get("scene_transform", "none"))
        strength, primaries = film_style_pairing(key)
        combos[key] = {
            "wb": str(combo.get("wb", "5500k")),
            "st": st if st in SCENE_TRANSFORMS else "none",
            "fc": key,
            # Editorial style pairing (observe mode's declared look layer): the
            # combo sets these controls visibly, same as the other layers —
            # nothing baked, everything overridable.
            "sts": strength,
            "pr": primaries,
        }
        # Negative presets carry a spectrally derived colour-head field; reversal
        # presets honestly have no such control (no printing stage).
        if color_head_supported(key):
            heads[key] = True
    return (
        "\n".join(film_opts),
        "\n".join(curve_opts),
        json.dumps(combos, ensure_ascii=False),
        json.dumps(heads, ensure_ascii=False),
    )


def render_page(init_dir: str) -> bytes:
    from dngscan import coreimage_decode
    from dngscan.constants import MAX_HDR_HEADROOM_EV
    from dngscan.gui.constants import RAW_EXTS, REALTIME_PREVIEW_LONG_EDGE

    film_opts, curve_opts, combos_json, color_heads_json = _film_options_html()

    html = (
        PAGE.replace("INIT_DIR", json.dumps(init_dir))
        .replace("RAW_ACCEPT", ",".join(sorted(RAW_EXTS)))
        .replace("PREVIEW_LONG_EDGE", str(REALTIME_PREVIEW_LONG_EDGE))
        .replace("GRADE_OPTIONS", _grade_options_html())
        .replace("SCENE_TRANSFORM_OPTIONS", _scene_transform_options_html())
        .replace("COREIMAGE_AVAILABLE_FLAG", "true" if coreimage_decode.available() else "false")
        # Keep the slider ceiling on the same source of truth as the CLI's
        # --hdr-headroom bound (log2(4000/100) = 5.32); step 0.02 lands on it exactly.
        .replace("MAX_HDR_HEADROOM_ATTR", f"{MAX_HDR_HEADROOM_EV:.2f}")
        .replace("FILM_OPTIONS", film_opts)
        .replace("FILM_CURVE_OPTIONS", curve_opts)
        .replace("FILM_COMBOS_JSON", combos_json)
        .replace("FILM_COLOR_HEADS_JSON", color_heads_json)
    )
    return html.encode("utf-8")
