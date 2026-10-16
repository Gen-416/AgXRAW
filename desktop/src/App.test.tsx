import { afterEach, expect, it, vi } from 'vitest';
import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react';
import { App } from './App';
import type { EngineClient } from './api/client';

afterEach(cleanup);
it('opens, previews and exports through shared commands without inventing encode defaults', async () => {
  const call = vi.fn(async (command: string, _params: unknown) => command === 'preview.prepare' ? { ok: true, width: 128, height: 128 }
    : command === 'preview.render' ? { ok: true, preview: 'aW1hZ2U=', ev: .2, decoder: 'libraw', ev_auto: { planning_status: 'selected' } }
    : { ok: true, saved: ['/output/source.jpg'] });
  const client = { call, bootstrap: async () => ({ version: 1,
    defaults: { evAuto: true, ev: 0, format: 'sdr', deliveryProfile: 'auto' },
    default_directory: '/output', raw_extensions: ['.dng'],
    capabilities: { coreimage: false, hdr: false, dependency_errors: [] },
  }) } as unknown as EngineClient;
  render(<App client={client} />);
  await waitFor(() => expect(screen.getByText('读取').hasAttribute('disabled')).toBe(true));
  fireEvent.change(screen.getByLabelText('RAW 文件路径'), { target: { value: '/source.dng' } });
  fireEvent.click(screen.getByText('读取'));
  await screen.findByAltText('AgX 成片预览');
  expect(screen.getByText('白端点与肩部联动')).toBeTruthy();
  fireEvent.click(screen.getByText('全尺寸导出'));
  await screen.findByText('/output/source.jpg');
  const exportParams = call.mock.calls.find(([command]) => command === 'delivery.export')![1];
  expect(exportParams).not.toHaveProperty('quality');
  expect(exportParams).not.toHaveProperty('chroma');
});
