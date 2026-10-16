import { describe, it, expect } from 'vitest';
import { EngineClient } from './client';

describe('versioned engine boundary', () => {
  it('rejects wrong versions and mismatched request identities', async () => {
    for (const reply of [{ version: 99, id: 'wrong', ok: true }, { version: 1, id: 'wrong', ok: true }]) {
      const client = new EngineClient({ bootstrap: async () => ({}), request: async () => reply });
      await expect(client.call('preview.render', { input: '/source.dng' })).rejects.toThrow('不匹配');
    }
  });
  it('surfaces structured engine errors and rejects invalid successful pixels', async () => {
    const client = new EngineClient({ bootstrap: async () => ({}), request: async payload => {
      const p = payload as { id: string; command: string };
      return p.command === 'delivery.export'
        ? { version: 1, id: p.id, ok: false, error: { code: 'engine-error', message: 'HDR 不可用' } }
        : { version: 1, id: p.id, ok: true, result: { ok: true } };
    } });
    await expect(client.call('delivery.export', { input: '/source.dng' })).rejects.toThrow('HDR 不可用');
    await expect(client.call('preview.render', { input: '/source.dng' })).rejects.toThrow('预览像素');
  });
});
