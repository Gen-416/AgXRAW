#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fresh-process real RAW -> final CLI delivery with exact decision/payload checks.

Select each checkout with --repo; use the same source and format on both sides.
The timed region is the real CLI, including metadata and final publication.
Input/output hashing and additional verification readback are outside timing.
RSS includes these checks. --compare requires identical decisions, compressed
rendition identity and primary decoded bytes; whole-file equality is also shown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from unittest.mock import patch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--decoder', choices=('libraw', 'coreimage'), default='libraw')
    p.add_argument('--format', choices=('sdr', 'sdr-heic', 'ultrahdr', 'ultrahdr-heic'), required=True)
    p.add_argument('--quality', type=int)
    p.add_argument('--chroma', choices=('444', '422', '420'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--compare', type=Path)
    p.add_argument('--require-file-match', action='store_true',
                   help='also fail if complete file bytes differ (requires --compare)')
    a = p.parse_args()
    if a.require_file_match and a.compare is None:
        p.error('--require-file-match requires --compare')
    suffix = '.heic' if a.format.endswith('heic') else '.jpg'
    if a.out.suffix.lower() != suffix:
        p.error(f'format requires {suffix} output')
    if not a.source.is_file() or not (a.repo / 'dngscan/cli.py').is_file():
        p.error('source and repository must exist')
    if any(x.exists() or x.is_symlink() for x in (a.out, a.report)):
        p.error('output and report must be new')
    if a.out.resolve() == a.report.resolve():
        p.error('output and report must be different paths')
    previous = None
    if a.compare is not None:
        try:
            previous = json.loads(a.compare.read_text())
            if not isinstance(previous.get('identity'), dict) or not isinstance(previous.get('file_sha256'), str):
                raise ValueError('comparison report requires identity and file_sha256')
        except (OSError, ValueError, AttributeError) as exc:
            p.error(str(exc))
    source_sha = sha256(a.source)
    os.environ['DNGSCAN_FAST'] = '1'
    os.environ['DNGSCAN_FAST_SKIP'] = ''
    sys.path.insert(0, str(a.repo.resolve()))
    import numpy as np
    from dngscan import cli, _fast
    from dngscan.delivery_integrity import encoded_content_signature
    from dngscan.gainmap import read_primary_rgb_u8
    from tools.benchmark_gainmap_search import json_safe
    ext = _fast._load_extension()
    if ext is None:
        raise RuntimeError('strict native extension required')
    delivery = []
    export = cli.export_jpeg

    def capture(*args, **kwargs):
        value = export(*args, **kwargs)
        delivery.append(value)
        return value

    argv = [str(a.source), '--jpeg', str(a.out), '--decoder', a.decoder,
            '--output-format', a.format, '--ev', 'auto', '--output-gamut', 'p3']
    if a.quality is not None:
        argv.extend(('--jpeg-quality', str(a.quality)))
    if a.chroma is not None:
        argv.extend(('--chroma', a.chroma))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    wall, cpu = time.perf_counter(), time.process_time()
    with patch.object(cli, 'export_jpeg', capture):
        status = cli.main(argv)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    if status or len(delivery) != 1 or not a.out.is_file():
        raise RuntimeError(f'CLI delivery did not finish: status={status}')
    info = delivery[0]
    if isinstance(info, dict):
        info = {k: v for k, v in info.items() if k not in ('output_path', '_decoded_rgb')}
    primary = read_primary_rgb_u8(a.out, 'p3')
    container = 'heic' if a.format.endswith('heic') else 'jpeg'
    identity = json_safe({
        'source_sha256': source_sha, 'format': a.format, 'decoder': a.decoder,
        'quality': a.quality, 'chroma': a.chroma, 'delivery': info,
        'payload': encoded_content_signature(a.out, container),
        'primary': {'shape': list(primary.shape), 'dtype': str(primary.dtype),
                    'sha256': hashlib.sha256(primary.tobytes()).hexdigest()},
    })
    if sha256(a.source) != source_sha:
        raise RuntimeError('source changed during benchmark')
    report = {'identity': identity, 'wall_s': wall, 'cpu_s': cpu,
              'file_sha256': sha256(a.out), 'bytes': a.out.stat().st_size,
              'repo': str(a.repo), 'commit': subprocess.check_output(
                  ['git', '-C', str(a.repo), 'rev-parse', 'HEAD'], text=True).strip(),
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'platform': platform.platform(), 'native_abi': ext.native_abi_version()},
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                              (2**20 if sys.platform == 'darwin' else 1024)}
    if previous is not None:
        report['exact'] = identity == previous['identity']
        report['file_exact'] = report['file_sha256'] == previous['file_sha256']
    a.report.parent.mkdir(parents=True, exist_ok=True)
    with a.report.open('x') as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('wall_s', 'cpu_s', 'peak_rss_mib', 'bytes')} |
                     {'exact': report.get('exact'), 'file_exact': report.get('file_exact')}), flush=True)
    matched = report.get('exact', True) and (not a.require_file_match or report.get('file_exact', False))
    return 0 if matched else 2


if __name__ == '__main__':
    raise SystemExit(main())
