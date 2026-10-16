import { spawn, execFileSync } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadEnv } from 'vite';

const cwd = fileURLToPath(new URL('..', import.meta.url));
const env = { ...loadEnv('development', cwd, 'AGXRAW_'), ...process.env };
try {
  const cargo = execFileSync('rustup', ['which', 'cargo'], { encoding: 'utf8' }).trim();
  env.PATH = `${dirname(cargo)}:${env.PATH}`;
} catch { /* A normal cargo installation on PATH needs no rustup adapter. */ }
const child = spawn(resolve(cwd, 'node_modules/.bin/tauri'), process.argv.slice(2), { cwd, env, stdio: 'inherit' });
child.on('exit', code => { process.exitCode = code ?? 1; });
child.on('error', error => { console.error(error.message); process.exitCode = 1; });
