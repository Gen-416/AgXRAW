import { API_VERSION, type Bootstrap, type Command, type JobParams, type Json, type Reply, type Result } from './generated';

export interface Transport {
  bootstrap(): Promise<unknown>;
  request(payload: unknown): Promise<unknown>;
  upload?(file: File): Promise<string>;
}
const object = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

export class EngineClient {
  constructor(private transport: Transport) {}
  upload(file: File): Promise<string> {
    if (!this.transport.upload) return Promise.reject(new Error('请使用桌面原生文件选择'));
    return this.transport.upload(file);
  }
  async bootstrap(): Promise<Bootstrap> {
    const value = await this.transport.bootstrap();
    if (!object(value) || value.version !== API_VERSION || !object(value.defaults)
      || typeof value.default_directory !== 'string' || !Array.isArray(value.raw_extensions)
      || !value.raw_extensions.every(v => typeof v === 'string') || !object(value.capabilities)
      || typeof value.capabilities.coreimage !== 'boolean' || typeof value.capabilities.hdr !== 'boolean'
      || typeof value.session_token !== 'string' || !value.session_token
      || (value.initial_source !== undefined && typeof value.initial_source !== 'string')
      || !Array.isArray(value.capabilities.dependency_errors)
      || !value.capabilities.dependency_errors.every(v => typeof v === 'string')) throw new Error('应用接口版本或启动信息不兼容');
    return value as unknown as Bootstrap;
  }
  async call(command: Command, params: JobParams | Record<string, Json>): Promise<Result> {
    const id = crypto.randomUUID();
    const raw = await this.transport.request({ version: API_VERSION, id, command, params });
    if (!object(raw) || raw.version !== API_VERSION || raw.id !== id || typeof raw.ok !== 'boolean') {
      throw new Error('后端响应与当前请求不匹配');
    }
    if (!raw.ok) {
      if (!object(raw.error) || typeof raw.error.message !== 'string') throw new Error('后端错误格式不兼容');
      throw new Error(raw.error.message);
    }
    if (!object(raw.result) || raw.result.ok !== true) throw new Error('后端返回了无效的操作结果');
    const result = raw.result;
    if (result.superseded !== undefined && typeof result.superseded !== 'boolean') throw new Error('预览状态无效');
    for (const key of ['decoder', 'decoder_version']) {
      if (result[key] != null && typeof result[key] !== 'string') throw new Error('解码器信息无效');
    }
    if (result.detected != null && !object(result.detected)) throw new Error('分析信息无效');
    if (result.ev_auto != null && (!object(result.ev_auto)
      || (result.ev_auto.planning_status != null && typeof result.ev_auto.planning_status !== 'string'))) {
      throw new Error('自动规划信息无效');
    }
    if (command === 'preview.render' && !result.superseded
      && (typeof result.preview !== 'string' || typeof result.ev !== 'number' || !Number.isFinite(result.ev))) {
      throw new Error('预览像素或曝光信息缺失');
    }
    if (command === 'delivery.export' && (!Array.isArray(result.saved)
      || !result.saved.every(v => typeof v === 'string'))) throw new Error('导出文件信息缺失');
    return (raw as unknown as Extract<Reply, { ok: true }>).result;
  }
}

export function httpTransport(): Transport {
  let token: string | null = null;
  return {
    async bootstrap() {
      const response = await fetch('/api/v1/bootstrap');
      if (!response.ok) throw new Error('无法连接本机处理服务');
      const result = await response.json();
      token = result.session_token;
      return result;
    },
    async request(payload) {
      if (!token) throw new Error('处理服务尚未就绪');
      const response = await fetch('/api/v1/commands', { method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-DngScan-Token': token }, body: JSON.stringify(payload) });
      if (!response.ok) throw new Error(`本机请求失败 (${response.status})`);
      return response.json();
    },
    async upload(file) {
      if (!token) throw new Error('处理服务尚未就绪');
      const response = await fetch(`/upload?name=${encodeURIComponent(file.name)}`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-DngScan-Token': token }, body: file,
      });
      const result = await response.json();
      if (!response.ok || result.ok !== true || typeof result.path !== 'string') throw new Error(result.error || '读取文件失败');
      return result.path;
    },
  };
}

export const isDesktop = () => '__TAURI_INTERNALS__' in window;
export async function createClient(): Promise<EngineClient> {
  if (!isDesktop()) return new EngineClient(httpTransport());
  const { invoke } = await import('@tauri-apps/api/core');
  return new EngineClient({ bootstrap: () => invoke('engine_bootstrap'),
    request: payload => invoke('engine_request', { payload }) });
}
