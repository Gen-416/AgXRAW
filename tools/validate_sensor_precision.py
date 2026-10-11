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
from dngscan.protected_output import ProtectedSources
from dngscan.delivery_transaction import DeliveryTransaction


PRECISION_BUDGET_VERSION = 'agxraw-production-boundaries-1'
PRECISION_BUDGET = {
    'integer_nearest_unclipped_max_codes': .5,
    'integer_truncate_unclipped_max_codes': 1.,
    'half_normal_relative_rounding_bound': 2.**-11,
    'half_subnormal_absolute_rounding_bound': 2.**-25,
    'warp_single_storage_bound_codes': 1. + 2.**-9,
    'scope': 'same algorithm and domain; clipping, decoder changes and resampling policy are separate',
}


def precision_error(reference, actual):
    """Whole-array bias/RMS/max, bounded deterministic quantile sample; no fitting."""
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape or not reference.size:
        raise ValueError('precision comparison requires equal nonempty shapes')
    count, total, squared, maximum, changed, invalid = 0, 0., 0., 0., 0, 0
    a, b = reference.reshape(-1), actual.reshape(-1)
    for start in range(0, a.size, 262144):
        x, y = a[start:start+262144].astype(np.float64), b[start:start+262144].astype(np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        invalid += int((~finite).sum())
        error = y[finite]-x[finite]
        count += error.size
        total += float(error.sum()); squared += float(error @ error)
        maximum = max(maximum, float(np.abs(error).max(initial=0.)))
        changed += int(np.count_nonzero(error))
    step = max(1, (a.size+65535)//65536)
    sample = np.abs(b[::step].astype(np.float64)-a[::step].astype(np.float64))
    sample = sample[np.isfinite(sample)]
    return {'shape': list(reference.shape), 'reference_dtype': str(reference.dtype),
            'actual_dtype': str(actual.dtype), 'finite_pairs': count, 'nonfinite_pairs': invalid,
            'bias': total/count if count else None, 'rms': (squared/count)**.5 if count else None,
            'maximum_absolute_error': maximum if count else None, 'changed_fraction': changed/count if count else None,
            'sample_absolute_quantiles_p50_p95_p99_p999': np.quantile(sample,[.5,.95,.99,.999]).tolist() if sample.size else [],
            'quantile_scope': 'deterministic at-most-65536 sample; other metrics use every pair'}


def half_rounding_audit(reference, actual=None):
    """IEEE conversion bound only; classify overflow separately from rounding."""
    value = np.asarray(reference)
    with np.errstate(over='ignore', invalid='ignore'):
        converted = value.astype(np.float16)
    actual = converted if actual is None else np.asarray(actual)
    if value.shape != actual.shape:
        raise ValueError('half rounding comparison shape differs')
    checked, violations, overflow, differing = 0, 0, 0, 0
    x, y, rounded = value.reshape(-1), actual.reshape(-1), converted.reshape(-1)
    for start in range(0, x.size, 262144):
        ref = x[start:start+262144].astype(np.float64)
        observed = y[start:start+262144].astype(np.float64)
        expected = rounded[start:start+262144].astype(np.float64)
        valid = np.isfinite(ref) & (np.abs(ref)<=65504.)
        bound = np.abs(ref)*2.**-11 + 2.**-25
        checked += int(valid.sum())
        violations += int(np.count_nonzero(valid & (~np.isfinite(observed) | (np.abs(observed-ref)>bound))))
        overflow += int(np.count_nonzero(np.isfinite(ref)&~valid))
        differing += int(np.count_nonzero(valid & (observed != expected)))
    return {'rounding_budget': PRECISION_BUDGET, 'checked_values': checked,
            'out_of_finite_half_range': overflow, 'bound_violations': violations,
            'differs_from_float_reference_cast_to_half': differing,
            'within_rounding_bound': violations == 0,
            'interpretation': 'bitmap format comparison can include renderer effects; pure cast is a numerical reference, not sensor truth'}


def production_boundary_experiments(size):
    """Same spatial-black formula and same warp; never switch demosaicer."""
    from dngscan.dng_opcodes import Warp, warp_image
    n = min(size, 256)
    rng = np.random.default_rng(20261011)
    pattern = np.array([[512.25, 513.5], [514.75, 515.125]], dtype=np.float64)[..., None]
    horizontal, vertical = np.linspace(-.2,.2,n), np.linspace(-.1,.1,n)
    model = SpatialBlack(horizontal, vertical, pattern, (0,0), (float(pattern.max()+.3),))
    field = model.band(0,n,n)
    raw = np.rint(field+rng.normal(0,4,(n,n))+np.linspace(0,4095,n)[None,:]).astype(np.uint16)
    signed = (raw.astype(np.float64)-field)*65535./(4095.-model.max_black[0])
    clipped = np.clip(signed,0,65535.)
    working = SimpleNamespace(raw_image_visible=raw.copy())
    apply_to_working(working,model,[512.]*4,4095.)
    actual = working.raw_image_visible
    numerical = precision_error(clipped,actual)
    numerical['budget_passed'] = numerical['maximum_absolute_error'] <= .5
    y,x = np.indices((n,n))
    rgb = np.stack((x/(n-1),y/(n-1),(x+y)/(2*n-2)),axis=-1)*65535.
    rgb[:n//3] *= .0001
    rgb[n//2:n//2+2] = 65535
    rgb = np.rint(rgb).astype(np.uint16)
    warp = Warp(((1.,.017,-.003,0.,.0004,-.0003),)*3,.5,.5)
    floating = warp_image(rgb.astype(np.float32),warp)
    integer = warp_image(rgb,warp)
    warp_numerical = precision_error(np.clip(floating,0,65535.),integer)
    warp_numerical['budget_passed'] = warp_numerical['maximum_absolute_error'] <= PRECISION_BUDGET['warp_single_storage_bound_codes']
    twice_float, twice_int = warp_image(floating,warp), warp_image(integer,warp)
    return {'schema': PRECISION_BUDGET_VERSION, 'predeclared_budgets': PRECISION_BUDGET,
            'spatial_black': {'formula': '(raw - exact spatial black) * 65535 / (white - max black)',
                'same_formula_float64_to_uint16': numerical,
                'clipping_policy_error_separate': precision_error(signed,clipped),
                'lower_clipped_fraction': float((signed<0).mean()), 'upper_clipped_fraction': float((signed>65535).mean()),
                'signed_negative_fraction': float((signed<0).mean()), 'stored_negative_fraction': 0.,
                'domain': 'normalized uint16 working codes; not original sensor DN'},
            'late_warp': {'same_kernel_single_storage': warp_numerical,
                'two_pass_cumulative': precision_error(np.clip(twice_float,0,65535.),twice_int),
                'domain': 'camera RGB working codes', 'algorithm': 'same Warp coefficients and bicubic footprint',
                'clipping_policy': 'float warp retains extended values; uint16 clamps/truncates; range policy separate',
                'float_negative_fraction': float((floating<0).mean()), 'float_above_white_fraction': float((floating>65535).mean()),
                'cumulative_interpretation': 'includes first clipping and storage propagated through second resampling; neither pure quantization nor a single-rounding bound'},
            'half_cast': half_rounding_audit(np.r_[np.linspace(-2.,8.,131072),np.linspace(-2.**-14,2.**-14,8192)]),
            'missing_boundaries': ['actual stage-2 opcode combinations and private codec unpack references',
                'LibRaw internal integer stages cannot be isolated by changing demosaicer',
                'real held-out weak-texture and calibrated output noise comparisons'],
            'production_change': False}


def coreimage_precision_experiment(path, version, *, full=False):
    """Render the SAME CIImage/context/extent/colorspace twice; no auto fallback."""
    from dngscan import coreimage_decode as ci
    from PIL import Image
    path = Path(path)
    source = ci._recipe_source_identity(path)
    if source.get('status') != 'hashed':
        raise ValueError('precision experiment requires a readable, hashed RAW source')
    report = {'source_sha256': source.get('sha256'), 'requested_version': version,
              'scope': 'same configured CIImage, context, bounds and extended-linear Rec.2020; bitmap format only',
              'size': 'full' if full else 'native-preview', 'production_change': False}
    if not ci.available():
        return {**report,'status':'unavailable','reason':'coreimage-runtime-unavailable'}
    offered = ci.supported_versions(path)
    report['offered_tokens'] = list(offered)
    try:
        token = ci.resolve_decoder_version(version,offered)
    except Exception as exc:
        return {**report,'status':'unavailable','reason':str(exc)}
    filt = ci._open_filter(path)
    scale = 1. if full else ci.preview_scale_factor(filt)
    cfg = ci.configure_linear_filter(filt,version=token,scale_factor=scale)
    if ci._scene_input_qualification(cfg)['status'] != 'qualified':
        raise ci.SceneInputContractError(cfg)
    image = filt.outputImage()
    if image is None:
        raise RuntimeError('CIRAWFilter has no output CIImage')
    extent = image.extent()
    xywh = [float(extent.origin.x),float(extent.origin.y),float(extent.size.width),float(extent.size.height)]
    width,height = int(round(xywh[2])),int(round(xywh[3]))
    if min(width,height)<1: raise RuntimeError('empty precision experiment extent')
    q = ci._require_quartz()
    ctx = ci._render_context(q,interactive=not full)
    color = q.CGColorSpaceCreateWithName(q.kCGColorSpaceExtendedLinearITUR_2020)
    if color is None: raise RuntimeError('extended-linear Rec.2020 unavailable')
    bounds = q.CGRectMake(xywh[0],xywh[1],width,height)
    def bitmap(dtype,fmt):
        stride = width*4*np.dtype(dtype).itemsize
        data = bytearray(height*stride)
        start = time.perf_counter()
        ctx.render_toBitmap_rowBytes_bounds_format_colorSpace_(image,data,stride,bounds,fmt,color)
        array = np.frombuffer(data,dtype=dtype).reshape(height,width,4)[...,:3].copy()
        return array,time.perf_counter()-start
    floating,t_f = bitmap(np.float32,q.kCIFormatRGBAf)
    half,t_h = bitmap(np.float16,q.kCIFormatRGBAh)
    repeated,t_repeat = bitmap(np.float32,q.kCIFormatRGBAf)
    repeatability = precision_error(floating,repeated)
    del repeated
    if not np.isfinite(floating).all() or not np.isfinite(half).all():
        raise RuntimeError('precision render produced nonfinite values')
    boundaries = []
    # Legacy fixed compensation is conditional, unlike current per-file
    # aligned sceneScale (which changes the scalar, not the stored half pixels).
    legacy = float(ci.COREIMAGE_SCALE_COMPENSATION)
    for label,gain in (('unity',1.),('conditional-legacy-fixed-compensation',legacy)):
        pre_cast = half.astype(np.float32)*np.float32(gain)
        after = pre_cast.astype(np.float16)
        target = (max(1,height//2),max(1,width//2))
        reference = np.empty((*target,3),np.float32)
        for channel in range(3):
            plane = Image.fromarray(after[...,channel].astype(np.float32))
            reference[...,channel] = np.asarray(plane.resize((target[1],target[0]),Image.Resampling.BILINEAR))
        production = ci._resize_rgb(after,target)
        boundaries.append({'case':label,'gain':gain,'gain_rounding':half_rounding_audit(pre_cast,after),
            'resize_storage_rounding':half_rounding_audit(reference,production),
            'resize_numerical':precision_error(reference,production),'same_resize_kernel':'PIL bilinear, identical source and target',
            'target_shape':list(target),'not_a_claim_this_resize_or_legacy_gain_ran_in_default_full_export':True})
    ci._assert_recipe_source_current(path,source,phase='precision-experiment-commit')
    report.update(status='measured',actual_token=cfg['actual_decoder_token'],scale_factor=cfg['scale_factor'],
        native_extent_xywh=xywh,shape=list(floating.shape),controls=cfg['controls'],
        runtime=ci.decoder_runtime_id(),hardware=ci.decoder_hardware_id(),
        context=ci._CONTEXT_RECIPES.get(not full,{}),
        same_image_float_repeatability=repeatability,bitmap_precision=precision_error(floating,half),
        bitmap_rounding=half_rounding_audit(floating,half),later_half_boundaries=boundaries,
        negative_fraction_float=float((floating<0).mean()),above_one_fraction_float=float((floating>1).mean()),
        seconds={'RGBAf':t_f,'RGBAh':t_h,'RGBAf_repeat':t_repeat},source_unchanged=True,
        interpretation='format difference is not independent sensor truth, decoder accuracy or a measured noise/texture loss; no automatic precision policy change')
    return report


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
    parser.add_argument('--production-boundaries',action='store_true',
                        help='isolate same-formula spatial black, late warp and half storage')
    parser.add_argument('--apple-raw',type=Path,action='append',default=[],
                        help='repeat for private RAW inputs; fixed Apple token, no fallback')
    parser.add_argument('--apple-version',choices=('9','8','7','6'),default='9')
    parser.add_argument('--apple-full',action='store_true',help='full CIImage; default is native preview')
    args=parser.parse_args()
    if args.size < 32 or args.size % 2:
        parser.error('size must be even and at least 32')
    protection = ProtectedSources(args.apple_raw).check(args.output)
    result={'schema':'agxraw-sensor-precision-research-2',
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'working_changes':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines(),
            'research_source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                for name in ('dngscan/sensor_research.py', 'tools/validate_sensor_precision.py',
                             'dngscan/spatial_black.py', 'dngscan/raw_io.py',
                             'dngscan/coreimage_decode.py')},
            'platform':platform.platform(),'python':sys.version.split()[0],'numpy':np.__version__,
            'command':' '.join(sys.argv), 'evidence_class':'synthetic and real-decoder synthetic DNG; no real personal calibration',
            'C01_C02':black_experiments(args.size),'C07':fixed_pattern_experiment(args.size),
            'C08':chroma_experiments(args.size),'C09':quantization_experiments()}
    if args.include_pipeline:
        result['C02_decoder_outputs']=pipeline_experiments()
    if args.production_boundaries:
        result['production_boundaries']=production_boundary_experiments(args.size)
    if args.apple_raw:
        result['apple_bitmap_precision']=[]
        for path in args.apple_raw:
            try:
                case=coreimage_precision_experiment(path,args.apple_version,full=args.apple_full)
            except Exception as exc:
                case={'status':'failed','reason':type(exc).__name__,'error':str(exc)}
            result['apple_bitmap_precision'].append(case)
    with DeliveryTransaction(args.output,protected_sources=protection) as transaction:
        transaction.path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        transaction.commit()
    print(json.dumps({'output':str(args.output),'near_black_cases':len(result['C01_C02']['matrix']),
                      'chroma_cases':len(result['C08']['matrix']),
                      'pipeline_cases':len(result.get('C02_decoder_outputs',{}).get('matrix',[])),
                      'apple_cases':[row['status'] for row in result.get('apple_bitmap_precision',[])]}))
    return int(any(row['status']=='failed' for row in result.get('apple_bitmap_precision',[])))


if __name__=='__main__':
    raise SystemExit(main())
