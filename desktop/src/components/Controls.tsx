import type { Bootstrap, JobParams } from '../api/generated';

type Props = { intent: JobParams; bootstrap: Bootstrap; update: (patch: Partial<JobParams>) => void };
function Choice({ label, value, options, onChange, disabled = false }: {
  label: string; value: string | undefined; options: [string, string][];
  onChange: (value: string) => void; disabled?: boolean;
}) {
  return <label className="field">{label}<select value={value} disabled={disabled} onChange={e => onChange(e.target.value)}>
    {options.map(([key, title]) => <option key={key} value={key}>{title}</option>)}
  </select></label>;
}
export function Controls({ intent, bootstrap, update }: Props) {
  return <>
    <section><h2>曝光与影调</h2>
      <label className="check"><input type="checkbox" checked={intent.evAuto} onChange={e => update({ evAuto: e.target.checked })} />受约束自动规划</label>
      <label className="field">手动曝光 EV<input type="number" min="-20" max="20" step="0.1" value={intent.ev ?? 0}
        onChange={e => { if (e.target.value !== '') update({ ev: Number(e.target.value), evAuto: false }); }} /></label>
      <p className="hint">自动联动白端点与肩部；手动曝光使用固定端点。默认额外亮度为零。</p>
      {([['midtoneBrightness', '中间调亮度'], ['midtoneContrast', '中间调对比'],
        ['shadowTransition', '阴影过渡'], ['highlightTransition', '高光过渡']] as const).map(([key, label]) =>
        <label className="field" key={key}>{label}<span className="value">{Number(intent[key] ?? 0).toFixed(2)}</span>
          <input type="range" min="-1" max="1" step="0.05" value={intent[key] ?? 0} onChange={e => update({ [key]: Number(e.target.value) })} />
        </label>)}
    </section>
    <details><summary>解码与颜色</summary><div className="detailsBody">
      <Choice label="解码器" value={intent.decoder} options={bootstrap.capabilities.coreimage
        ? [['libraw', 'LibRaw'], ['coreimage', 'Apple RAW']] : [['libraw', 'LibRaw']]}
        onChange={decoder => update({ decoder })} />
      {intent.decoder === 'coreimage' && <Choice label="Apple RAW 版本" value={intent.coreimageVersion}
        options={['auto', '9', '8', '7', '6'].map(v => [v, v === 'auto' ? '自动选择' : v])}
        onChange={coreimageVersion => update({ coreimageVersion })} />}
      <Choice label="白平衡" value={intent.wb} options={[
        ['camera', '拍摄值'], ['6500k', '6500K'], ['5500k', '5500K'], ['3400k', '3400K'], ['3200k', '3200K'], ['9300k', '9300K']]}
        onChange={wb => update({ wb })} />
      <Choice label="高光处理" value={intent.decoder === 'coreimage' ? 'reconstruct' : intent.highlight} disabled={intent.decoder === 'coreimage'}
        options={ [['clip', '保持原始过曝'], ['blend', '温和混合'], ['reconstruct', '重建']] }
        onChange={highlight => update({ highlight })} />
      <Choice label="AgX 色彩浓淡" value={intent.agxPrimaries}
        options={ [['base', '标准'], ['smooth', '平滑'], ['punchy', '浓郁'], ['muted', '柔和']] }
        onChange={agxPrimaries => update({ agxPrimaries })} />
      <p className="hint">色度清理保持默认关闭。完整标定管理与研究控制仍可在原有工作台使用。</p>
    </div></details>
  </>;
}

export function DeliveryControls({ intent, bootstrap, update }: Props) {
  return <section><h2>保存成片</h2>
    <Choice label="格式" value={intent.format} options={[
      ['sdr', 'JPEG · SDR'], ['sdr-heic', 'HEIF · SDR'],
      ...(bootstrap.capabilities.hdr ? [['ultrahdr', 'JPEG · HDR'], ['ultrahdr-heic', 'HEIF · HDR']] as [string, string][] : []),
    ]} onChange={format => update({ format,
      ...(format.endsWith('heic') && intent.deliveryProfile === 'share-hq' ? { deliveryProfile: 'auto' } : {}) })} />
    <Choice label="编码策略" value={intent.deliveryProfile} options={[
      ['auto', '自动 · 质量验证'], ['share', '分享'], ['archive', '存档'],
      ...(!intent.format?.endsWith('heic') ? [['share-hq', '97 / 4:2:0 · 原尺寸']] as [string, string][] : []),
    ]} onChange={deliveryProfile => update({ deliveryProfile })} />
    <Choice label="SDR 输出色域" value={intent.gamut} disabled={intent.format?.startsWith('ultrahdr')}
      options={ [['srgb', 'sRGB'], ['p3', 'Display P3']] } onChange={gamut => update({ gamut })} />
    <label className="field">输出文件夹<input value={intent.outdir ?? ''} onChange={e => update({ outdir: e.target.value })} /></label>
  </section>;
}
