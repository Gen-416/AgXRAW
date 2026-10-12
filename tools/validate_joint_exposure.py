#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare production joint planning to an existing frozen RAW comparison.

Private RAWs and comparison assets remain outside Git. The comparison must
supply reproduce/inputs.json and per-case reports. Full renders optionally
reproduce its JPEG97/420/common-ICC convention, not formal export acceptance.
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from PIL import Image
from dngscan.analysis import analyze
from dngscan.auto_ev import compute_auto_ev
from dngscan.gui.preview_cache import build_proxy_entry
from dngscan.raw_io import load_raw, release_analysis_buffers
from dngscan.render import render_output_encoded_float
from dngscan.scene_scale import with_intent_exposure


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def source_identity():
    """Include newly added sources, which a tracked diff alone cannot identify."""
    paths = subprocess.check_output(
        ['git', 'ls-files', '-co', '--exclude-standard', '--', 'dngscan'],
        cwd=REPO, text=True,
    ).splitlines()
    paths.append(str(Path(__file__).resolve().relative_to(REPO)))
    return {name: sha(REPO/name) for name in sorted(set(paths))
            if (REPO/name).is_file()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case', action='append', default=[])
    p.add_argument('--full-case', action='append', default=[])
    args = p.parse_args()
    cases = json.loads((args.comparison_dir/'reproduce/inputs.json').read_text())
    result = {'schema': 'agxraw-joint-promotion/1', 'repo_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(), 'working_tree_diff_sha256': hashlib.sha256(
        subprocess.check_output(['git', 'diff', 'HEAD'], cwd=REPO)).hexdigest(),
        'source_identity_sha256': source_identity(),
        'calculation_backend': os.environ.get('DNGSCAN_FAST', 'auto'), 'cases': []}
    for c in cases:
        if args.case and c['id'] not in args.case:
            continue
        start = time.monotonic()
        source = Path(c['source'])
        assert sha(source) == c['sha256']
        expected = json.loads((args.comparison_dir/c['id']/'report.json').read_text())
        b = load_raw(source, scene_half_size=False, decoder='libraw', wb_mode='camera', scene_highlight_mode='clip')
        a, _, _ = analyze(b, 4, diagnostics=False, gamut_names=('sRGB',))
        sink = []
        auto = compute_auto_ev(b, a, 'srgb', _plan_sink=sink)
        v = expected['variants']['bounded-joint']
        assert auto.ev == v['ev'], (c['id'], auto.ev, v['ev'])
        assert asdict(auto.render_plan.tone) == v['tone'], c['id']
        assert sink[-1] is auto.render_plan
        state = 'fallback' if expected['selection']['fallback'] else 'selected' if expected['selection'].get('selected') else 'identity'
        assert auto.planning['status'] == state
        # The preview carries full-source rows, never its averaged highlights.
        proxy = build_proxy_entry(b, a)
        preview = compute_auto_ev(proxy.bundle, a, 'srgb')
        assert preview.ev == auto.ev and preview.render_plan == auto.render_plan, c['id']
        row = {'id': c['id'], 'source_sha256': c['sha256'], 'shape': expected['shape'],
               'actual_decoder': b.scene_decoder, 'ev': auto.ev, 'planning': auto.planning,
               'frozen_EV_and_tone_equal': True, 'full_proxy_plan_equal': True}
        del proxy, preview
        b = release_analysis_buffers(b)
        if c['id'] in args.full_case:
            encoded = render_output_encoded_float(with_intent_exposure(b, user_ev=auto.ev), a, 'srgb', tone_plan=auto.render_plan)
            assert np.isfinite(encoded).all()
            u8 = np.rint(np.clip(encoded, 0., 1.) * 255).astype(np.uint8)
            with Image.open(args.comparison_dir/c['id']/'bounded-joint.jpg') as old:
                icc = old.info['icc_profile']
            output = io.BytesIO()
            Image.fromarray(u8).save(output, format='JPEG', quality=97, subsampling=2, icc_profile=icc)
            digest = hashlib.sha256(output.getvalue()).hexdigest()
            assert digest == v['JPEG_sha256'], (c['id'], digest, v['JPEG_sha256'])
            row.update(full_SDR_JPEG_byte_equal=True, JPEG_sha256=digest)
            del encoded, u8, output
        assert sha(source) == c['sha256']
        row.update(source_unchanged=True, seconds=time.monotonic()-start)
        result['cases'].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(c['id'], state, f'EV={auto.ev:.4f}', f'{row["seconds"]:.1f}s', flush=True)
        del b, a, auto, sink
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
