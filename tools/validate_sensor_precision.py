#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reproducible C01/C02/C07/C08/C09 experiments; contains no private RAW.

Run from a source checkout with its tests (the optional real LibRaw probe
reuses the repository's generated DNG fixture, not a mocked decoder).
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import tracemalloc
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from dngscan.sensor_research import (
    LUMA, PHASES, PersonalBlackCalibration, signed_black_correct,
    signed_bilinear_bayer, dark_stack_products, joint_chroma_shrink,
    quantized_gain_chain,
)
from dngscan.spatial_black import SpatialBlack, apply_to_working


def statistics(values):
    return {'mean': float(np.mean(values)), 'variance': float(np.var(values)),
            'minimum': float(np.min(values)), 'negative_fraction': float(np.mean(values < 0)),
            'distinct_codes': int(len(np.unique(values)))}


def measured_call(function):
    tracemalloc.start()
    start = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {'seconds': elapsed, 'tracemalloc_peak_bytes': peak,
                    'memory_scope': 'tracked Python/NumPy allocations in this call; not total process RSS'}


def black_experiments(size):
    rng = np.random.default_rng(20261009)
    sigma = np.asarray([3., 4., 5., 6.]).reshape(2, 2)
    phase_black = np.asarray([512., 513., 514., 515.]).reshape(2, 2)
    rows = []
    for fraction in (0., .25, .5, .75):
        pattern = phase_black + fraction
        horizontal = np.linspace(-.2, .2, size)
        field = np.tile(pattern, (size // 2, size // 2)) + horizontal[None, :]
        sd = np.tile(sigma, (size // 2, size // 2))
        for signal_sigma in (0., .5, 1., 2., 4.):
            raw = np.rint(field + signal_sigma * sd + rng.normal(size=(size, size)) * sd).astype(np.uint16)
            residual = raw.astype(float) - field
            working = SimpleNamespace(raw_image_visible=raw.copy())
            maximum = float(pattern.max() + horizontal.max())
            spatial = SpatialBlack(horizontal, np.zeros(size), pattern[..., None], (0, 0), (maximum,))
            loss = np.zeros(raw.shape, np.uint8)
            apply_to_working(working, spatial, [512.] * 4, 16383., loss)
            clipped_dn = working.raw_image_visible.astype(float) * (16383. - maximum) / 65535.
            rows.append({'black_fraction_dn': fraction, 'signal_in_phase_sigmas': signal_sigma,
                'signed_reference': statistics(residual), 'production_spatial_black_dn_equivalent': statistics(clipped_dn),
                'phase_mean_bias_dn': [float((clipped_dn - residual)[i::2, j::2].mean()) for i,j in ((0,0),(0,1),(1,0),(1,1))],
                'upper_loss_fraction': float(loss.mean()),
                'signed_low_tail_is_not_a_bad_pixel_mask': True})
    image = rng.normal(0, 4, (size, size))
    _, perf = measured_call(lambda: signed_bilinear_bayer(image, 'RGGB', phase_gains=[2., 1., 1.1, 1.5]))
    calibration = PersonalBlackCalibration('Research', 'Bayer', 'body-1', 'linear-1', 100, (.01, .02),
        dict(zip(PHASES, phase_black.ravel() + .25)), dict.fromkeys(PHASES, .05), 'synthetic-dark-hash')
    capture = {'make':'Research', 'model':'Bayer', 'body_serial':'body-1', 'readout_identity':'linear-1',
               'iso':100, 'exposure_seconds':.015}
    raw = np.tile(phase_black + .25, (size // 2, size // 2)) + .1
    corrected, provenance = signed_black_correct(raw, 512., calibration, capture,
                                               metadata_phase_black=dict.fromkeys(PHASES, 512.))
    return {'matrix': rows, 'shape': [size, size], 'signed_bilinear_performance': perf,
            'float64_rgb_output_bytes': size * size * 3 * 8,
            'personal_black_prototype': {'corrected_weak_gray': statistics(corrected), 'provenance': provenance},
            'default_decision': 'retain metadata/LibRaw compatibility path; signed reference remains opt-in research'}


def fixed_pattern_experiment(size):
    rng = np.random.default_rng(8473)
    shape = (min(size, 128), min(size, 128))
    y, x = np.indices(shape)
    fixed = 3 * np.sin(2*np.pi*y/16) + .8*np.cos(2*np.pi*x/32)
    fixed[13, 21] += 40
    train = 512 + fixed + rng.normal(0, 2., (64, *shape))
    products, perf = measured_call(lambda: dark_stack_products(train, calibration_identity='synthetic-iso-time-temp'))
    weak = .2 * np.cos(2*np.pi*x/11)
    heldout = 512 + fixed + weak + rng.normal(0, 2., (32, *shape))
    corrected = heldout - products['mean_bias_map_dn']
    return {'train_frames':64, 'heldout_frames':32, 'shape': list(shape),
        'before_mean_error_dn2':float(np.mean((heldout.mean(0)-512-weak)**2)),
        'after_mean_error_dn2':float(np.mean((corrected.mean(0)-weak)**2)),
        'temporal_variance_before_dn2':float(heldout.var(0).mean()),
        'temporal_variance_after_dn2':float(corrected.var(0).mean()),
        'mean_calibration_variance_dn2':float(np.mean(products['mean_uncertainty_dn']**2)),
        'hot_candidate_detected':bool(products['hot_pixel_candidates'][13,21]), 'performance':perf,
        'default_decision':'no automatic correction; needs matched real stack plus heldout verification'}


def chroma_experiments(size):
    from dngscan.chroma_nr import _atrous_smooth, chroma_correction_map
    from dngscan.noise_propagation import atrous_detail_variance
    rng = np.random.default_rng(883)
    n = min(size, 128)
    y, x = np.indices((n, n))
    projection = np.eye(3) - np.ones((3,1)) * LUMA[None,:]
    covariance = np.asarray([[1., .3, .05], [.3, .6, -.1], [.05, -.1, 1.7]]) * .0001
    factor = float(atrous_detail_variance(np.ones((n, n)), 0)[n//2,n//2])
    rows = []
    perf = None
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        direction = np.array([np.cos(angle), 0., np.sin(angle)]) @ projection.T
        direction /= np.linalg.norm(direction)
        for label, pattern in (('fine-line', (x % 16 == 0).astype(float)),
                               ('woven', np.sin(2*np.pi*x/16)*np.cos(2*np.pi*y/16)),
                               ('colour-edge', (x >= n//2).astype(float)),
                               ('luminance-edge', np.zeros_like(x))):
            for amplitude in (.003, .015, .06):
                clean = .2 + pattern[...,None] * direction * amplitude
                if label == 'luminance-edge':
                    clean += (x >= n//2)[...,None] * .1
                noisy = clean + rng.multivariate_normal(np.zeros(3), covariance, (n,n))
                smooth = _atrous_smooth(noisy, 0)
                joint, timing = measured_call(lambda: joint_chroma_shrink(noisy-smooth, covariance*factor, strength=.5))
                perf = timing
                result = smooth + joint[0]
                # Existing production kernel is also measured, but it uses
                # several bands and structure protection: this is not a fair
                # same-algorithm speed/quality ranking against one joint band.
                production = noisy + chroma_correction_map(noisy.astype(np.float32), .5, 8,
                                                          noise_covariance=covariance)
                target = clean - (clean @ LUMA)[...,None]
                support = np.linalg.norm(target,axis=-1) > 1e-10
                row = {'pattern':label, 'angle_radians':float(angle), 'amplitude':amplitude,
                       'joint_one_band_chroma_mse':float(np.mean(((result-clean) @ projection.T)**2)),
                       'production_multiband_chroma_mse':float(np.mean(((production-clean) @ projection.T)**2)),
                       'joint_luma_change_max':float(np.max(np.abs((result-noisy) @ LUMA)))}
                for name, value in (('joint',result),('production',production)):
                    recovered = value - (value @ LUMA)[...,None]
                    denom = float(np.sum(target*target))
                    row[name+'_chroma_amplitude_gain'] = float(np.sum(recovered*target)/denom) if denom > 1e-15 else None
                    if support.any():
                        a, b = recovered[support], target[support]
                        cosine = np.sum(a*b,axis=-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-15)
                        row[name+'_mean_chroma_angle_degrees'] = float(np.degrees(np.arccos(np.clip(cosine,-1,1))).mean())
                rows.append(row)
    return {'matrix':rows, 'shape':[n,n], 'one_band_performance_last_call':perf,
            'comparison_scope':'joint one-band prototype versus separately measured production multiband; not an equivalence or superiority claim',
            'default_decision':'keep calibrated structure-protected production kernel; no automatic joint-shrink activation'}


def quantization_experiments():
    rows=[]
    for label, raw in (('near-black',np.linspace(0,8,4096)), ('gradient',np.linspace(1,4000,32768)),
                       ('near-white',np.linspace(4000,4095,4096))):
        for gains in ([1.003],[1.003,.997,1.007,.993]):
            reference = raw.copy()
            for gain in gains:
                reference = np.clip(reference*gain,0,4095)
            for policy in ('truncate','nearest','deferred'):
                value=quantized_gain_chain(raw,gains,white=4095,policy=policy)
                error=value-reference
                rows.append({'signal':label,'operations':len(gains),'policy':policy,
                             'mean_error_original_dn':float(error.mean()),'rms_error_original_dn':float(np.sqrt(np.mean(error**2))),
                             'max_error_original_dn':float(np.max(np.abs(error)))})
    return {'matrix':rows, 'domain':'original 12-bit RAW DN, not normalized uint16',
            'default_decision':'do not change compatibility rounding without matched real-calibration validation'}


def pipeline_experiments():
    """Actual decoder and SDR/HDR formation; fixed plan isolates image changes."""
    import struct
    import rawpy
    from tests.test_pipeline_corrections import write_sensor_dng
    from dngscan.raw_io import load_raw
    from dngscan.analysis import analyze
    from dngscan.tone import build_render_plan
    from dngscan.render import render_output_encoded_float
    from dngscan.hdr_agx import scene_render_to_hdr_display_linear
    from dngscan.hdr_agx_plan import compile_hdr_agx_plan
    from dngscan import dng_opcodes as ops
    rng=np.random.default_rng(291)
    rows=[]
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'near-black.dng'
        for demosaic in ('bilinear','dht'):
            for neutral in ((1.,1.,1.),(.5,1.,.75)):
                for signal in (0.,2.,8.):
                    write_sensor_dng(path,black_pattern=[[512,513],[514,515]],signal=0,neutral=neutral)
                    data=bytearray(path.read_bytes()); count=struct.unpack_from('<H',data,8)[0]
                    for i in range(count):
                        off=10+12*i
                        if struct.unpack_from('<H',data,off)[0]==273:
                            ptr=struct.unpack_from('<L',data,off+8)[0]
                            black=np.tile(np.array([[512.,513.],[514.,515.]]),(64,64))
                            pixels=np.rint(black+signal+rng.normal(0,4,(128,128))).astype('<u2')
                            data[ptr:ptr+pixels.nbytes]=pixels.tobytes()
                    path.write_bytes(data)
                    bundle=load_raw(path,demosaic=demosaic)
                    analysis,_,_=analyze(bundle,4)
                    plan=build_render_plan(bundle,analysis,'agx','srgb')
                    hdr=compile_hdr_agx_plan(plan,analysis=analysis)
                    baseline=bundle.scene_rec2020_render.astype(float)/bundle.scene_scale
                    # Prototype is a camera-domain precision reference. No
                    # external colour matrix is fitted to hide decode errors.
                    pattern=np.asarray(bundle.raw_pattern)
                    colours=[bundle.color_desc[int(cid)] for cid in pattern.flat]
                    wb=np.asarray(bundle.camera_wb)
                    multipliers=[(wb[int(cid)] if wb[int(cid)]>0 else wb[1])/wb[wb>0].min() for cid in pattern.flat]
                    signed=signed_bilinear_bayer(pixels.astype(float)-black,colours,phase_gains=multipliers)
                    # Use the exact file-derived matrix and fixed normalization,
                    # never a fitted gain/offset/reference matching transform.
                    matrix=ops.libraw_camera_matrix(bundle.evidence.color_matrix,bundle.evidence.xyz_to_cam,is_dng=True)
                    norm=65535./(4095.-max(bundle.evidence.spatial_black.max_black))
                    signed_scene=ops.camera_to_rec2020(signed*norm,matrix)
                    signed_bundle=replace(bundle,scene_rec2020_render=signed_scene)
                    row={'demosaic':demosaic,'as_shot_neutral':list(neutral),'signal_dn':signal,
                         'production_scene_rgb_mean':baseline.mean((0,1)).tolist(),
                         'signed_bilinear_camera_dn_mean':signed.mean((0,1)).tolist(),
                         'signed_bilinear_scene_rgb_mean':(signed_scene/bundle.scene_scale).mean((0,1)).tolist(),
                         'comparison_scope':'same file matrix and declared normalization, fixed production plan; prototype linear interpolation differs from adaptive decoder'}
                    for ev in (-2.,0.,2.):
                        exposed=replace(bundle,exposure_gain=2.**ev)
                        sdr=render_output_encoded_float(exposed,analysis,tone_plan=plan)
                        formed=scene_render_to_hdr_display_linear(exposed,plan,hdr,analysis=analysis)
                        experimental=replace(signed_bundle,exposure_gain=2.**ev)
                        signed_sdr=render_output_encoded_float(experimental,analysis,tone_plan=plan)
                        signed_hdr=scene_render_to_hdr_display_linear(experimental,plan,hdr,analysis=analysis)
                        row[f'exposure_{ev:g}']={'sdr_encoded_mean':sdr.mean((0,1)).tolist(),
                                                'hdr_display_linear_mean':formed.mean((0,1)).tolist(),
                                                'signed_reference_sdr_encoded_mean':signed_sdr.mean((0,1)).tolist(),
                                                'signed_reference_hdr_display_linear_mean':signed_hdr.mean((0,1)).tolist(),
                                                'sdr_finite':bool(np.isfinite(sdr).all()),'hdr_finite':bool(np.isfinite(formed).all())}
                    rows.append(row)
    return {'matrix':rows,'rawpy':rawpy.__version__,'libraw':list(rawpy.libraw_version),
            'scope':'synthetic real-decode shadow probe; not sensor-calibrated signed-path quality acceptance'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--include-pipeline',action='store_true')
    args=parser.parse_args()
    if args.size < 32 or args.size % 2:
        parser.error('size must be even and at least 32')
    result={'schema':'agxraw-sensor-precision-research-1',
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'working_changes':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines(),
            'research_source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                for name in ('dngscan/sensor_research.py', 'tools/validate_sensor_precision.py',
                             'dngscan/spatial_black.py', 'dngscan/raw_io.py')},
            'platform':platform.platform(),'python':sys.version.split()[0],'numpy':np.__version__,
            'command':' '.join(sys.argv), 'evidence_class':'synthetic and real-decoder synthetic DNG; no real personal calibration',
            'C01_C02':black_experiments(args.size),'C07':fixed_pattern_experiment(args.size),
            'C08':chroma_experiments(args.size),'C09':quantization_experiments()}
    if args.include_pipeline:
        result['C02_decoder_outputs']=pipeline_experiments()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'near_black_cases':len(result['C01_C02']['matrix']),
                      'chroma_cases':len(result['C08']['matrix']),
                      'pipeline_cases':len(result.get('C02_decoder_outputs',{}).get('matrix',[]))}))


if __name__=='__main__':
    main()
