import type { WorkspaceState } from '../state/workspace';

export function Preview({ state }: { state: WorkspaceState }) {
  const auto = state.preview?.ev_auto;
  const info = state.detected?.detected;
  return <section className="previewPanel">
    <div className="previewHeader"><h2>成片预览</h2><span className="badge">SDR · 1920 px</span></div>
    <div className={`imageStage ${state.busy ? 'pending' : ''}`} aria-busy={state.busy}>
      {state.preview?.preview ? <img alt="AgX 成片预览" src={`data:image/jpeg;base64,${state.preview.preview}`} />
        : <div className="empty"><span className="emptyMark">AgX</span><p>打开照片，让光线自然落入画面</p><small>使用现有成像管线与默认规则</small></div>}
    </div>
    {state.preview && <div className="previewFacts">
      <span>{state.preview.decoder}{state.preview.decoder_version ? ` ${state.preview.decoder_version}` : ''}</span>
      <span>曝光 {Number(state.preview.ev).toFixed(2)} EV</span>
      {auto && <span>{auto.planning_status === 'selected' ? '白端点与肩部联动' : auto.planning_status === 'fallback' ? '保留基础方案' : auto.planning_status === 'identity' ? '保持基础曝光' : '自动曝光'}</span>}
    </div>}
    <div className="taskStatus" role="status">{state.status}</div>
    {state.error && <div className="error" role="alert">{state.error}</div>}
    {state.delivery?.saved && <div className="saved"><strong>已保存</strong>{state.delivery.saved.map(path => <div key={path}>{path}</div>)}</div>}
    {info && <details className="diagnostics"><summary>分析与实际处理信息</summary><pre>{JSON.stringify(info, null, 2)}</pre></details>}
  </section>;
}
