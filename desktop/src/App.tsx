import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { EngineClient, isDesktop } from './api/client';
import { Workspace } from './state/workspace';
import { Controls, DeliveryControls } from './components/Controls';
import { Preview } from './components/Preview';

export function App({ client }: { client: EngineClient }) {
  const [workspace] = useState(() => new Workspace(client));
  const state = useSyncExternalStore(workspace.subscribe, workspace.snapshot);
  const picker = useRef<HTMLInputElement>(null);
  const [path, setPath] = useState('');
  const [desktopError, setDesktopError] = useState('');
  useEffect(() => { void workspace.start(); return () => workspace.dispose(); }, [workspace]);
  useEffect(() => { if (state.intent.input) setPath(state.intent.input); }, [state.intent.input]);
  const pick = async () => {
    try {
      if (!isDesktop()) { picker.current?.click(); return; }
      const { open } = await import('@tauri-apps/plugin-dialog');
      const file = await open({ multiple: false, directory: false, filters: [
        { name: 'RAW 照片', extensions: state.bootstrap!.raw_extensions.map(v => v.slice(1)) },
      ] });
      if (typeof file === 'string') { setDesktopError(''); setPath(file); await workspace.open(file); }
    } catch (e) { setDesktopError(String(e)); }
  };
  const legacy = async () => {
    if (isDesktop()) {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke('open_legacy_workspace');
    } else window.open('/legacy', '_blank', 'noopener,noreferrer');
  };
  const outputFolder = async () => {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const folder = await open({ directory: true, multiple: false });
      if (typeof folder === 'string') workspace.update({ outdir: folder });
    } catch (e) { setDesktopError(String(e)); }
  };
  return <div className="app">
    <header><div className="brand">AgXRAW<span>RAW 显影与分享</span></div>
      <nav><button className="quiet" onClick={() => { void legacy().catch(e => setDesktopError(String(e))); }}>原有完整工作台</button>
        <button className="primary" disabled={!state.bootstrap} onClick={() => { void pick(); }}>打开 RAW</button></nav>
    </header>
    <div className="sourceBar"><input aria-label="RAW 文件路径" placeholder="也可以粘贴本机 RAW 路径" value={path} onChange={e => setPath(e.target.value)}
      onKeyDown={e => { if (e.key === 'Enter') void workspace.open(path); }} />
      <button disabled={!state.bootstrap || !path.trim()} onClick={() => { void workspace.open(path); }}>读取</button></div>
    <input className="filePicker" type="file" aria-label="选择 RAW 文件" ref={picker} accept={state.bootstrap?.raw_extensions.join(',')}
      onChange={e => { const file = e.target.files?.[0]; if (file) { setPath(file.name); void workspace.open(client.upload(file)); } e.target.value = ''; }} />
    {desktopError && <div role="alert" className="error">{desktopError}</div>}
    <main><aside>
      {!state.bootstrap && <button onClick={() => { void workspace.start(); }}>重新连接处理服务</button>}
      {state.bootstrap && <><Controls intent={state.intent} bootstrap={state.bootstrap} update={p => workspace.update(p)} />
        <DeliveryControls intent={state.intent} bootstrap={state.bootstrap} update={p => workspace.update(p)} />
        {isDesktop() && <button onClick={() => { void outputFolder(); }}>选择输出文件夹</button>}
        <div className="actions"><button className="primary" disabled={!state.intent.input || state.busy || state.exporting}
          onClick={() => { void workspace.export(); }}>{state.exporting ? '正在导出…' : '全尺寸导出'}</button>
          <button onClick={() => workspace.reset()}>恢复默认</button></div>
        <button disabled={!state.intent.input || state.busy} onClick={() => { void workspace.render(); }}>刷新预览</button>
        {state.bootstrap.capabilities.dependency_errors.length > 0 && <div className="error">{state.bootstrap.capabilities.dependency_errors.join('\n')}</div>}
      </>}
    </aside><Preview state={state} /></main>
    <footer>默认 AgX · 保留原始证据 · 编码后实际回读验证</footer>
  </div>;
}
