#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Launch actual Tauri, wait for its frontend's RAW preview, verify ownership.

This opens a temporary native window and closes only the process it creates.
Private RAWs are read-only. Reports contain hashes, not source paths or pixels.
Requires a built desktop binary and an engine interpreter with GUI dependencies.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import tempfile
from urllib.request import build_opener, ProxyHandler, Request
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dngscan.application.contracts import default_job
from dngscan.protected_output import raw_protection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--binary', type=Path, default=ROOT/'desktop/src-tauri/target/debug/agxraw_desktop')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    protection = raw_protection(source)
    protection.check(args.output)
    def sha():
        with source.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    before = sha()
    env = {**os.environ, 'AGXRAW_PYTHON': sys.executable, 'AGXRAW_TRACE_COMMANDS': '1'}
    process = subprocess.Popen([str(args.binary), '--open', str(source)], cwd=ROOT, env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, text=True)
    lines = queue.Queue()
    def read():
        for line in process.stderr:
            lines.put(line)
    threading.Thread(target=read, daemon=True).start()
    url = None
    traces = []
    opener = build_opener(ProxyHandler({}))
    started = time.monotonic()
    try:
        deadline = started + 45
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=.25)
            except queue.Empty:
                if process.poll() is not None:
                    raise RuntimeError('Tauri exited before completing its frontend preview')
                continue
            found = re.search(r'dngscan GUI: (http://127\.0\.0\.1:\d+/)', line)
            if found:
                url = found.group(1)
            try:
                trace = json.loads(line)
            except ValueError:
                continue
            if isinstance(trace, dict) and 'agxraw_command' in trace:
                traces.append(trace)
                if trace['state'] == 'failed':
                    raise RuntimeError(f"Native frontend command failed: {trace['agxraw_command']}")
                if trace['agxraw_command'] == 'preview.render' and trace['state'] == 'succeeded':
                    break
        else:
            raise RuntimeError('Timed out waiting for the native frontend preview')
        if url is None:
            raise RuntimeError('No app-owned backend readiness event')
        with opener.open(url+'api/v1/bootstrap', timeout=10) as response:
            bootstrap = json.load(response)
        payload = {'version': 1, 'id': 'native-cache-verification', 'command': 'preview.render',
                   'params': {**default_job(), 'input': str(source), 'includeMetrics': True}}
        request = Request(url+'api/v1/commands', data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json', 'X-DngScan-Token': bootstrap['session_token']})
        with opener.open(request, timeout=15) as response:
            reply = json.load(response)
        assert reply['ok'], reply
        result = reply['result']
        assert result['cache_hit'], 'Native frontend did not populate the matching shared preview frame'
        assert before == sha(), 'Source changed'
        report = {'schema': 'agxraw-native-entry/1', 'source_sha256': before,
                  'source_unchanged': True, 'actual_frontend_commands': traces,
                  'matching_preview_cache_hit': True, 'ev': result['ev'], 'auto_ev': result['ev_auto'],
                  'preview_JPEG_sha256': hashlib.sha256(base64.b64decode(result['preview'])).hexdigest(),
                  'seconds_to_preview': time.monotonic()-started}
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=3)
        process.stderr.close()
    # The desktop process owns the engine even on termination. No orphan server.
    closed = False
    if url:
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            try:
                opener.open(url+'api/v1/bootstrap', timeout=.5).close()
            except (URLError, TimeoutError, OSError):
                closed = True
                break
            time.sleep(.1)
    assert closed, 'App-owned backend remained alive after desktop exit'
    report['backend_closed_after_desktop_exit'] = closed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='.agxraw-desktop-',
                                         suffix='.part', dir=args.output.parent, delete=False) as stream:
            candidate = Path(stream.name)
            stream.write(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        protection.replace(candidate, args.output)
    finally:
        if candidate is not None:
            candidate.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
