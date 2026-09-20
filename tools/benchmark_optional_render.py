#!/usr/bin/env python3
"""Profile optional non-film stages and bound native plan-call overhead.

Deterministic synthetic fixtures isolate gated formation, ChromaNR and RAW
guidance from decoding. Empty native calls give an UPPER bound on plan parsing
plus Python/FFI/allocation overhead; they do not measure parsing alone. Native
gated routing is experimental only and its exactness is reported, never enabled.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import cProfile
import dataclasses
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time
from unittest import mock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pixels', type=int, default=500000)
    parser.add_argument('--nr-width', type=int, default=1408)
    parser.add_argument('--raw-width', type=int, default=3000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--reference', action='store_true', help='only B3 and RAW evidence use their previous NumPy bodies')
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink() or min(
            args.pixels, args.nr_width, args.raw_width, args.repeats) < 1:
        parser.error('positive sizes/repeats and a new output path are required')
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    if baseline is not None and (not isinstance(baseline, dict)
            or any(baseline.get(name) != getattr(args, name)
                   for name in ('pixels', 'nr_width', 'raw_width'))
            or not isinstance(baseline.get('measurements'), dict)):
        parser.error('comparison must describe the same synthetic dimensions')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.environ['DNGSCAN_FAST'] = '1'
    os.environ['DNGSCAN_FAST_SKIP'] = ''
    import numpy as np
    from dngscan import _fast, agx, chroma_nr, cpu_budget, gated_drt, guidance, hdr_agx, render
    from tests.test_hdr_native import _formation_setup, _scene_plan
    from tests.golden_support import _bundle_from_scene, build_staggered_clip
    from tools.benchmark_loss_pipeline import array_record

    ext = _fast._load_extension()
    if ext is None:
        raise RuntimeError('matching native extension required')
    report = {'python': platform.python_version(), 'numpy': np.__version__,
              'platform': platform.platform(), 'native_abi': ext.native_abi_version(),
              'pixels': args.pixels, 'nr_width': args.nr_width, 'raw_width': args.raw_width,
              'repeats': args.repeats, 'reference': args.reference, 'seed': 41, 'measurements': {}}

    def measure(name, fn, *, repeats=None, profile=False):
        elapsed, result = [], None
        for _ in range(repeats or args.repeats):
            start = time.perf_counter()
            result = fn()
            elapsed.append(time.perf_counter() - start)
        record = {'wall_s': elapsed, 'median_s': statistics.median(elapsed)}
        if profile:
            profiler = cProfile.Profile()
            profiler.runcall(fn)
            stats = pstats.Stats(profiler)
            rows = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:25]
            record['profile'] = [{'function': f'{Path(key[0]).name}:{key[1]}:{key[2]}',
                                  'calls': value[1], 'self_s': value[2], 'cumulative_s': value[3]}
                                 for key, value in rows]
        report['measurements'][name] = record
        return result

    rng = np.random.default_rng(41)
    rgb = (np.exp2(rng.uniform(-14, 7, (args.pixels, 3))) * .18).astype(np.float32)
    rgb[::17] *= -1
    masks = rng.uniform(0, 1, rgb.shape).astype(np.float32)
    pure = _scene_plan()
    agx_spec = _fast.compile_agx_plan(pure.tone)
    output_spec = _fast.compile_output_plan('p3', .05)
    hdr_plan, tone, inset, outset, formation_y, tables, peak = _formation_setup()
    hdr_spec = hdr_agx._compile_native_hdr_plan(
        hdr_plan, tone, inset, outset, formation_y, tables, peak, 'p3')
    empty = np.empty((0, 3), np.float32)
    noise_a, noise_b = render.generate_dither_noise(rng, rgb.shape)
    with ExitStack() as stack, cpu_budget.claim(1):
        if args.reference:
            def old_bin(arr, ph, pw):
                h, w = arr.shape[:2]
                h2, w2 = max(1, h // ph), max(1, w // pw)
                return arr[:h2 * ph, :w2 * pw].reshape(h2, ph, w2, pw, arr.shape[2]).min(axis=(1, 3))
            stack.enter_context(mock.patch.object(chroma_nr, '_atrous_smooth', side_effect=chroma_nr._atrous_smooth_reference))
            stack.enter_context(mock.patch.object(guidance, '_binned_raw_evidence', return_value=None))
            stack.enter_context(mock.patch.object(guidance, '_bin_period_min', side_effect=old_bin))
        for name, empty_fn, full_fn in (
            ('agx', lambda: ext.apply_agx_core_f32(empty, agx_spec),
             lambda: ext.apply_agx_core_f32(rgb, agx_spec)),
            ('hdr', lambda: ext.apply_hdr_formation_f32(empty, None, hdr_spec),
             lambda: ext.apply_hdr_formation_f32(rgb, masks, hdr_spec)),
            ('output', lambda: ext.finalize_rec2020_u8_f32(empty, empty, empty, output_spec),
             lambda: ext.finalize_rec2020_u8_f32(rgb, noise_a, noise_b, output_spec)),
        ):
            empty_fn()
            def repeated():
                for _ in range(1000):
                    empty_fn()
            measure(name + '_1000_empty_calls', repeated)
            measure(name + '_one_chunk', full_fn)
        measure('intent_and_retreat_one_chunk', lambda: __import__(
            'dngscan.retreat', fromlist=['apply_clip_retreat_rec2020']).apply_clip_retreat_rec2020(
                np.nan_to_num((rgb / np.float32(1.3)) * np.float32(1.1)), masks, .4), profile=True)

        gated_plan = dataclasses.replace(pure.tone, tone_core='gated')
        old = measure('gated_reference', lambda: gated_drt._apply_gated_core_reference(
            rgb, gated_plan, pure.color, masks), profile=True)
        # The parent gated branch applies its own punch AFTER pure AgX formation.
        gated_agx = _fast.compile_agx_plan(dataclasses.replace(pure.tone, punch_strength=0.))
        with mock.patch.object(agx, 'apply_core', side_effect=lambda values, *_args:
                               ext.apply_agx_core_f32(values, gated_agx)):
            candidate = measure('gated_experimental_native_branch', lambda:
                gated_drt._apply_gated_core_reference(rgb, gated_plan, pure.color, masks))
        old_u8 = ext.finalize_rec2020_u8_f32(old, noise_a, noise_b, output_spec)
        candidate_u8 = ext.finalize_rec2020_u8_f32(candidate, noise_a, noise_b, output_spec)
        report['gated_parity'] = {'changed_f32': int(np.count_nonzero(old != candidate)),
                                 'max_abs_f32': float(np.max(np.abs(old - candidate))),
                                 'changed_u8': int(np.count_nonzero(old_u8 != candidate_u8)),
                                 'reference': array_record(old), 'candidate': array_record(candidate)}

        width = args.nr_width
        decimated = rng.uniform(-.01, .8, (max(8, width * 2 // 3), width, 3)).astype(np.float32)
        for factor in (1., 6000. / width):
            correction = measure(f'chroma_nr_factor_{factor:.4f}', lambda:
                chroma_nr.chroma_correction_map(decimated, .6, factor), profile=True)
            report['measurements'][f'chroma_nr_factor_{factor:.4f}']['identity'] = array_record(correction)

        width = args.raw_width
        height = max(2, width * 2 // 3)
        raw = rng.integers(0, 65536, (height, width), dtype=np.uint16)
        colors = np.tile(np.array([[0, 1], [3, 2]], np.uint8), ((height + 1) // 2, (width + 1) // 2))[:height, :width]
        scene = np.zeros((height // 2, width // 2, 3), np.uint16)
        bundle = _bundle_from_scene(scene, raw_image=raw, raw_colors=colors,
                                    clip_masks=np.zeros(scene.shape, np.float16))
        analysis = dataclasses.replace(build_staggered_clip().analysis,
            gain_e_per_dn=1., prior_read_noise_e=2., prior_quality_status='ok', prior_model_spread=0.)
        for name, fn in (
            ('raw_guidance_headroom', lambda: guidance._raw_headroom_rgb(bundle, scene.shape[:2], analysis)),
            ('raw_guidance_snr', lambda: guidance._raw_snr_confidence(bundle, analysis, scene.shape[:2])),
        ):
            values = measure(name, fn, profile=True)
            report['measurements'][name]['identity'] = array_record(values)
    exact = baseline is None or all(
        baseline['measurements'].get(name, {}).get('identity') == values['identity']
        for name, values in report['measurements'].items() if 'identity' in values)
    if baseline is not None:
        report['comparison'] = {'exact': exact}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({'timings': {k: v['median_s'] for k, v in report['measurements'].items()},
                      'gated_parity': report['gated_parity'], 'exact': exact}, indent=2))
    return 0 if exact else 2


if __name__ == '__main__':
    raise SystemExit(main())
