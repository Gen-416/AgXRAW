import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';
import { readFileSync, mkdtempSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

test('real RAW → default preview → full-size verified JPEG', async ({ page }) => {
  const source = process.env.AGXRAW_TEST_RAW;
  test.skip(!source, 'Set AGXRAW_TEST_RAW to a private local sample; RAWs are not committed.');
  const hash = () => createHash('sha256').update(readFileSync(source!)).digest('hex');
  const before = hash(), output = mkdtempSync(join(tmpdir(), 'agxraw-desktop-e2e-'));
  const errors: string[] = [], requests: Record<string, unknown>[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {
    if (request.url().endsWith('/api/v1/commands')) requests.push(request.postDataJSON());
  });
  await page.goto('/');
  await expect(page.getByText('选择 RAW，使用默认 AgX 自动规划')).toBeVisible();
  await page.getByLabel('RAW 文件路径').fill(source!);
  await page.getByText('读取', { exact: true }).click();
  await expect(page.getByAltText('AgX 成片预览')).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText('预览已更新', { exact: true })).toBeVisible({ timeout: 120_000 });
  await page.getByLabel('输出文件夹').fill(output);
  await expect(page.getByRole('button', { name: '全尺寸导出' })).toBeEnabled({ timeout: 120_000 });
  await page.getByRole('button', { name: '全尺寸导出' }).click();
  await expect(page.getByText('导出完成', { exact: true })).toBeVisible({ timeout: 120_000 });
  const paths = await page.locator('.saved div').allTextContents();
  expect(paths.length).toBeGreaterThan(0);
  paths.forEach(path => expect(existsSync(path)).toBe(true));
  expect(hash()).toBe(before);
  const exportRequest = requests.find(value => value.command === 'delivery.export')!;
  const params = exportRequest.params as Record<string, unknown>;
  expect(params.evAuto).toBe(true);
  expect(params).not.toHaveProperty('quality');
  expect(params).not.toHaveProperty('chroma');
  expect(errors).toEqual([]);
  await page.screenshot({ path: join(output, 'workspace.png'), fullPage: true });
  console.log(JSON.stringify({ source_sha256: before, outputs: paths, screenshot: join(output, 'workspace.png'), errors }));
});
