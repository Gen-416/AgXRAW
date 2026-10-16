import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './e2e', timeout: 180_000, workers: 1,
  use: { baseURL: 'http://127.0.0.1:1420', channel: 'chrome', headless: true,
    viewport: { width: 1440, height: 1050 } },
});
