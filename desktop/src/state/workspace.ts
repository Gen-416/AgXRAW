import type { Bootstrap, JobParams, Result } from '../api/generated';
import type { EngineClient } from '../api/client';

export interface WorkspaceState {
  bootstrap: Bootstrap | null;
  intent: JobParams;
  preview: Result | null;
  detected: Result | null;
  delivery: Result | null;
  busy: boolean;
  exporting: boolean;
  status: string;
  error: string | null;
}

/** User intent, request identity and displayed result have separate owners.
 * Stale replies cannot publish pixels, errors or clear a newer request's busy flag.
 */
export class Workspace {
  private listeners = new Set<() => void>();
  private epoch = 0;
  private generation = 0;
  private clientId = crypto.randomUUID();
  private session = crypto.randomUUID();
  private revision = 0;
  private intentVersion = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private state: WorkspaceState = { bootstrap: null, intent: {}, preview: null,
    detected: null, delivery: null, busy: false, exporting: false, status: '正在连接处理服务…', error: null };
  constructor(private client: EngineClient) {}
  snapshot = () => this.state;
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; };
  private publish(next: Partial<WorkspaceState>) {
    this.state = { ...this.state, ...next };
    this.listeners.forEach(f => f());
  }
  async start() {
    try {
      const bootstrap = await this.client.bootstrap();
      this.publish({ bootstrap, intent: { ...bootstrap.defaults, outdir: bootstrap.default_directory },
        status: '选择 RAW，使用默认 AgX 自动规划', error: null });
      if (bootstrap.initial_source) await this.open(bootstrap.initial_source);
    } catch (e) { this.publish({ error: String(e), status: '处理服务连接失败' }); }
  }
  private request(): JobParams {
    return structuredClone({ ...this.state.intent, previewClient: this.clientId, selectionEpoch: this.epoch,
      previewSession: this.session, generation: this.generation, includeMetrics: true });
  }
  async open(path: string | Promise<string>) {
    if (typeof path === 'string' && !path.trim()) return;
    this.epoch++; this.generation++; this.revision++; this.session = crypto.randomUUID();
    if (this.timer) clearTimeout(this.timer);
    const revision = this.revision;
    this.publish({ intent: { ...this.state.intent, input: typeof path === 'string' ? path : '' }, preview: null, detected: null,
      delivery: null, busy: true, status: '正在读取并分析 RAW…', error: null });
    try {
      const resolved = await path;
      if (revision !== this.revision) return;
      this.publish({ intent: { ...this.state.intent, input: resolved } });
      const request = this.request();
      const prepared = await this.client.call('preview.prepare', request);
      if (revision !== this.revision) return;
      if (prepared.superseded) {
        this.publish({ busy: false, status: '准备请求已失效，请重新读取' });
        return;
      }
      this.publish({ detected: prepared });
      await this.render();
    } catch (e) {
      if (revision === this.revision) this.publish({ error: String(e), busy: false, status: '读取失败' });
    }
  }
  update(patch: Partial<JobParams>) {
    this.setIntent({ ...this.state.intent, ...patch });
  }
  private setIntent(intent: JobParams) {
    this.revision++; this.generation++; this.intentVersion++;
    this.publish({ intent, error: null });
    if (this.timer) clearTimeout(this.timer);
    if (this.state.intent.input) this.timer = setTimeout(() => { void this.render(); }, 140);
  }
  async render() {
    if (!this.state.intent.input) return;
    const revision = this.revision, request = this.request();
    this.publish({ busy: true, status: '正在更新预览…', error: null });
    try {
      const preview = await this.client.call('preview.render', request);
      if (revision !== this.revision) return;
      if (preview.superseded) {
        this.publish({ busy: false, status: '预览请求已失效，请刷新预览' });
        return;
      }
      // The returned automatic EV is display information, never written back
      // into user intent. An explicit manual edit alone exits automatic mode.
      this.publish({ preview, busy: false, status: '预览已更新' });
    } catch (e) {
      if (revision === this.revision) this.publish({ error: String(e), busy: false, status: '预览失败' });
    }
  }
  async export() {
    if (!this.state.intent.input || this.state.exporting) return;
    const epoch = this.epoch, version = this.intentVersion;
    const request = this.request(); // Freeze the entire declaration for this export.
    this.publish({ exporting: true, error: null, status: '正在全尺寸导出与回读验证…' });
    try {
      const delivery = await this.client.call('delivery.export', request);
      if (epoch === this.epoch) this.publish({ delivery: { ...delivery, exported_intent: request },
        status: version === this.intentVersion ? '导出完成' : '导出完成 · 使用开始导出时的参数' });
    } catch (e) {
      if (epoch === this.epoch) this.publish({ error: String(e), status: '导出失败' });
    } finally { this.publish({ exporting: false }); }
  }
  reset() {
    if (!this.state.bootstrap) return;
    const { input, outdir } = this.state.intent;
    this.setIntent({ ...this.state.bootstrap.defaults, input, outdir });
  }
  dispose() { if (this.timer) clearTimeout(this.timer); this.revision++; this.listeners.clear(); }
}
