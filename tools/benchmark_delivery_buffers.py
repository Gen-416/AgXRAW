#!/usr/bin/env python3
"""Fresh-process exact ablations of HDR packing, readback ownership and HEIF planes."""
import argparse
import json
from pathlib import Path
import platform
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', required=True, choices=('pack', 'readback', 'plane8', 'plane10'))
    parser.add_argument('--size', nargs=2, type=int, default=(6000, 4000), metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if min(args.size) < 1 or args.out.exists() or args.out.is_symlink():
        parser.error('positive dimensions and new output path required')
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    if baseline is not None and (
        not isinstance(baseline, dict)
        or baseline.get('operation') != args.operation
        or baseline.get('size') != list(args.size)
        or not isinstance(baseline.get('identity'), dict)
    ):
        parser.error('comparison must describe the same operation and dimensions')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from dngscan.hdr_agx import achieved_headroom, to_gainmap_alternate
    from dngscan.heif_encoder import _quantized_band
    from tools.benchmark_loss_pipeline import array_record

    width, height = args.size
    stats = {}
    if args.operation == 'pack':
        source = np.random.default_rng(93).random((height, width, 3), dtype=np.float32)
        source *= np.float32(10.1)
        source -= np.float32(.1)
        start = time.perf_counter()
        if args.reference:
            top = float(np.percentile(np.max(source[..., :3], axis=-1), 99.99))
            usage = float(np.log2(top)) if top > 1. else 0.
        else:
            usage = achieved_headroom(source)
        stats['headroom_s'] = time.perf_counter() - start
        start = time.perf_counter()
        if args.reference:
            clipped = np.clip(np.asarray(source, dtype=np.float32), 0., 8.)
            result = np.empty((height, width, 4), np.float16)
            result[..., :3] = clipped.astype(np.float16, copy=False)
            result[..., 3] = np.float16(1)
        else:
            result = to_gainmap_alternate(source, 8.)
        stats['packing_s'] = time.perf_counter() - start
        identity = {'array': array_record(result), 'headroom': usage}
    elif args.operation == 'readback':
        owner = bytearray(height * width * 8)
        start = time.perf_counter()
        if args.reference:
            result = np.frombuffer(memoryview(owner), np.float16).reshape(height, width, 4).copy()
        else:
            result = np.frombuffer(memoryview(owner).toreadonly(), np.float16).reshape(height, width, 4)
        stats['ownership_s'] = time.perf_counter() - start
        identity = {'array': array_record(result)}
    else:
        source = np.random.default_rng(93).integers(0, 256, (height, width, 3), dtype=np.uint8)
        bits = 8 if args.operation == 'plane8' else 10
        result = np.empty(source.shape, np.uint8 if bits == 8 else '<u2')
        start = time.perf_counter()
        for row in range(0, height, 128):
            if args.reference:
                band = source[row:row + 128].astype(np.float32)
                band /= 255.0
                band = np.rint(np.clip(band, 0, 1) * ((1 << bits) - 1)).astype(result.dtype)
            else:
                band = _quantized_band(source[row:row + 128], bits)
            result[row:row + 128] = band
        stats['quantization_s'] = time.perf_counter() - start
        identity = {'array': array_record(result)}
    report = {'operation': args.operation, 'size': args.size, 'reference': args.reference,
              'timings': stats, 'identity': identity, 'platform': platform.platform(),
              'python': platform.python_version(), 'numpy': np.__version__,
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                              (2 ** 20 if sys.platform == 'darwin' else 1024)}
    exact = baseline is None or baseline['identity'] == identity
    if baseline is not None:
        report['comparison'] = {'exact': exact}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({key: report[key] for key in ('operation', 'reference', 'timings', 'peak_rss_mib')}
                     | {'exact': exact}))
    return 0 if exact else 2


if __name__ == '__main__':
    raise SystemExit(main())
