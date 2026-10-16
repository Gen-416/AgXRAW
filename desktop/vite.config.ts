import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: './',
  clearScreen: false,
  build: { target: 'safari16' },
  test: { environment: 'jsdom', include: ['src/**/*.test.{ts,tsx}'] },
});
