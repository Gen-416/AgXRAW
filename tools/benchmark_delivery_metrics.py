#!/usr/bin/env python3
"""Exact standalone delivery scans; source generation/hash excluded from timing."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', choices=('base', 'hdr'), required=True)
    parser.add_argument('--size', nargs=2, type=int, default=(6000, 4000))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--retention-mib', type=int, default=256)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if min(*args.size, args.workers, args.repeats) < 1 or not 0 <= args.retention_mib <= 512 \
            or args.out.exists() or args.out.is_symlink():
        parser.error('positive sizes/workers/repeats, bounded retention and a new output are required')
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    if baseline is not None and (not isinstance(baseline, dict)
            or baseline.get('operation') != args.operation or baseline.get('size') != list(args.size)
            or not isinstance(baseline.get('identity'), dict)):
        parser.error('comparison must describe the same operation and dimensions')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ['DNGSCAN_FAST'] = '1'
    import numpy as np
    from dngscan import _fast, cpu_budget
    from dngscan.auto_encode import coding_metrics
    from tools.benchmark_loss_pipeline import array_record
    ext = _fast._require_extension()
    if not hasattr(ext, 'HdrMetricsWorkspace'):
        raise RuntimeError('new ABI18 delivery metrics build required')
    w, h = args.size
    rng = np.random.default_rng(64)
    workspace = None
    if args.operation == 'base':
        intended = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        actual = np.empty((h, w, 4), np.uint8)
        actual[..., :3] = intended
        actual[..., 3] = 255
        actual[::3, ::5, :3] ^= np.uint8(7)
        def measure():
            if args.reference:
                compact = np.ascontiguousarray(actual[..., :3])
                return {**ext.base_roundtrip_metrics(compact, intended), **coding_metrics(compact, intended)}
            return ext.base_and_coding_metrics_u8(actual, intended, np.getbufsize())
    else:
        intended = np.empty((h, w, 4), np.float16)
        for row in range(0, h, 128):
            band = rng.random((min(128, h - row), w, 3), dtype=np.float32)
            band *= np.float32(8.1)
            band -= np.float32(.1)
            intended[row:row + 128, :, :3] = band
        intended[..., 3] = 1
        actual = intended.copy()
        actual[..., :3] *= np.float16(.97)
        weights = [.22897, .69174, .07929]
        workspace = ext.HdrMetricsWorkspace(args.retention_mib * 2 ** 20)
        def measure():
            if args.reference:
                return ext.hdr_roundtrip_metrics(actual, intended, weights)
            return workspace.measure(actual, intended, weights)
    actual.flags.writeable = intended.flags.writeable = False
    source_identity = {'actual': array_record(actual), 'intended': array_record(intended)}
    with cpu_budget.claim(args.workers):
        start = time.perf_counter()
        result = measure()
        cold = time.perf_counter() - start
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = measure()
            times.append(time.perf_counter() - start)
    identity = {**source_identity, 'metrics': {name: float(value).hex() for name, value in result.items()}}
    exact = baseline is None or baseline['identity'] == identity
    report = {'operation': args.operation, 'size': list(args.size), 'workers': args.workers,
              'reference': args.reference, 'native_abi': ext.native_abi_version(),
              'numpy': np.__version__, 'python': platform.python_version(), 'platform': platform.platform(),
              'cold_s': cold, 'wall_s': times, 'median_s': statistics.median(times),
              'retained_bytes': workspace.retained_bytes() if workspace else 0,
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                              (2 ** 20 if sys.platform == 'darwin' else 1024),
              'identity': identity, 'comparison': {'exact': exact}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({key: report[key] for key in ('operation', 'workers', 'reference', 'cold_s', 'median_s',
                                                 'retained_bytes', 'peak_rss_mib', 'comparison')}))
    return 0 if exact else 2


if __name__ == '__main__':
    raise SystemExit(main())
