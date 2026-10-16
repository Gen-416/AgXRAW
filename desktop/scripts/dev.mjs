// Browser development adapter. The engine stays loopback-only; the Vite
// proxy preserves that server's existing Host/Origin/token checks.
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { createServer, loadEnv } from 'vite';

const cwd = fileURLToPath(new URL('..', import.meta.url));
const root = resolve(cwd, '..');
const config = loadEnv('development', cwd, 'AGXRAW_');
const python = process.env.AGXRAW_PYTHON || config.AGXRAW_PYTHON
  || (existsSync(resolve(root, '.venv/bin/python')) ? resolve(root, '.venv/bin/python') : 'python3');
const backend = spawn(python, ['-u', '-m', 'dngscan.application', '--serve', '--handshake'],
  { cwd: root, stdio: ['pipe', 'pipe', 'inherit'] });
let server;
const stop = async () => {
  await server?.close();
  backend.stdin.end();
  const force = setTimeout(() => backend.kill('SIGKILL'), 4000);
  force.unref();
};
process.once('SIGINT', () => { void stop(); });
process.once('SIGTERM', () => { void stop(); });
try {
  const ready = await new Promise((accept, reject) => {
    const timeout = setTimeout(() => reject(new Error('处理服务启动超时')), 30000);
    const lines = createInterface({ input: backend.stdout });
    lines.once('line', line => {
      clearTimeout(timeout);
      lines.close();
      try { accept(JSON.parse(line)); } catch { reject(new Error('处理服务启动信息无效')); }
    });
    backend.once('error', reject);
    backend.once('exit', code => { clearTimeout(timeout); reject(new Error(`处理服务退出 (${code})`)); });
  });
  const target = ready.url;
  const proxy = () => ({ target, changeOrigin: true, headers: { Origin: target },
    configure(p) { p.on('proxyReq', request => { request.removeHeader('referer'); }); } });
  server = await createServer({ root: cwd, server: { host: '127.0.0.1', port: 1420, strictPort: true,
    proxy: { '/api': proxy(), '/upload': proxy(), '/legacy': { ...proxy(), rewrite: () => '/' } } } });
  await server.listen();
  server.printUrls();
  backend.once('exit', () => { void server.close(); });
} catch (e) { console.error(String(e)); await stop(); process.exitCode = 1; }
