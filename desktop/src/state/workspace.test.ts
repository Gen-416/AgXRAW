import { describe, it, expect, vi, afterEach } from 'vitest';
import { Workspace } from './workspace';
import type { EngineClient } from '../api/client';
import type { JobParams } from '../api/generated';

const bootstrap = { version: 1, defaults: { evAuto: true, ev: 0, deliveryProfile: 'auto', format: 'sdr' },
  default_directory: '/output', raw_extensions: ['.dng'], commands: [], session_token: 'test',
  capabilities: { coreimage: false, hdr: false, hdr_reason: '', dependency_errors: [] } };
const deferred = <T,>() => { let resolve!: (v: T) => void; let reject!: (e: Error) => void;
  const promise = new Promise<T>((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };

afterEach(() => vi.useRealTimers());
describe('workspace ownership', () => {
  it('late prepare and preview replies cannot replace a newer source', async () => {
    const first = deferred<unknown>();
    const call = vi.fn((command, params) => command === 'preview.prepare' && params.input === '/old.dng'
      ? first.promise : Promise.resolve(command === 'preview.prepare' ? { ok: true }
        : { ok: true, preview: params.input, ev: .1 }));
    const ws = new Workspace({ bootstrap: async () => bootstrap, call } as unknown as EngineClient);
    await ws.start();
    const old = ws.open('/old.dng'); await Promise.resolve();
    await ws.open('/new.dng');
    first.resolve({ ok: true }); await old;
    expect(ws.snapshot().preview?.preview).toBe('/new.dng');
    expect(call.mock.calls.filter(([command]) => command === 'preview.render')).toHaveLength(1);
    ws.dispose();
  });
  it('new file selection invalidates an in-flight result before upload finishes', async () => {
    const frame = deferred<unknown>(), upload = deferred<string>();
    const call = vi.fn((command) => command === 'preview.prepare' ? Promise.resolve({ ok: true }) : frame.promise);
    const ws = new Workspace({ bootstrap: async () => bootstrap, call } as unknown as EngineClient);
    await ws.start(); const old = ws.open('/old.dng'); await Promise.resolve(); await Promise.resolve();
    const newer = ws.open(upload.promise);
    frame.resolve({ ok: true, preview: 'old', ev: 2 }); await old;
    expect(ws.snapshot().preview).toBeNull(); expect(ws.snapshot().busy).toBe(true);
    upload.reject(new Error('upload failed')); await newer;
    expect(ws.snapshot().error).toContain('upload failed'); expect(ws.snapshot().busy).toBe(false);
    ws.dispose();
  });
  it('late failures do not clear a newer request busy flag', async () => {
    const old = deferred<unknown>(), next = deferred<unknown>();
    const ws = new Workspace({ bootstrap: async () => bootstrap,
      call: (_command: string, params: { input: string }) => params.input === '/old.dng' ? old.promise : next.promise,
    } as unknown as EngineClient);
    await ws.start(); const a = ws.open('/old.dng'); await Promise.resolve();
    const b = ws.open('/new.dng'); await Promise.resolve();
    old.reject(new Error('late failure')); await a;
    expect(ws.snapshot().busy).toBe(true); expect(ws.snapshot().error).toBeNull();
    next.resolve({ ok: true, superseded: true }); await b;
    expect(ws.snapshot().busy).toBe(false);
    ws.dispose();
  });
  it('automatic EV stays result-only and export freezes the complete intent', async () => {
    vi.useFakeTimers();
    const exporting = deferred<unknown>();
    const call = vi.fn((command: string, _params: JobParams) => command === 'delivery.export' ? exporting.promise
      : Promise.resolve(command === 'preview.prepare' ? { ok: true } : { ok: true, preview: 'image', ev: .25 }));
    const ws = new Workspace({ bootstrap: async () => bootstrap, call } as unknown as EngineClient);
    await ws.start(); await ws.open('/source.dng');
    expect(ws.snapshot().intent.ev).toBe(0); expect(ws.snapshot().intent.evAuto).toBe(true);
    const declaration = { shutter: 'mechanical' };
    ws.update({ captureDeclaration: declaration });
    const job = ws.export();
    const sent = call.mock.calls.find(([command]) => command === 'delivery.export')![1];
    ws.update({ ev: .5, evAuto: false });
    declaration.shutter = 'electronic';
    expect(sent.ev).toBe(0); expect(sent.evAuto).toBe(true);
    expect(sent.captureDeclaration?.shutter).toBe('mechanical');
    expect(sent).not.toHaveProperty('quality'); expect(sent).not.toHaveProperty('chroma');
    exporting.resolve({ ok: true, saved: ['/output/image.jpg'] }); await job;
    expect(ws.snapshot().exporting).toBe(false);
    ws.update({ quality: 97, chroma: '420' });
    ws.reset(); expect(ws.snapshot().intent.evAuto).toBe(true); expect(ws.snapshot().intent.input).toBe('/source.dng');
    expect(ws.snapshot().intent).not.toHaveProperty('quality');
    expect(ws.snapshot().intent).not.toHaveProperty('chroma');
    ws.dispose();
  });
});
