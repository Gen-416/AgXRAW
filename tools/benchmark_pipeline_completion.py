#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fresh-process exact identity and timings for one full-resolution SDR or HDR job.

Run this script with --repo pointing at an unchanged baseline or the optimized
checkout. --optimized enables the private analysis/AutoEV handoffs. Each process
renders exactly one requested master; encoding has its own fixed-master benchmark.
Wall/CPU times exclude hashing. RSS includes verification, not just the job.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--decoder',choices=('libraw','coreimage'),default='libraw')
    p.add_argument('--mode',choices=('sdr','hdr','hdr-packed'),required=True)
    p.add_argument('--optimized',action='store_true')
    p.add_argument('--core',choices=('agx','gated','neutral','lum'),default='agx')
    p.add_argument('--chroma-nr',type=float,default=0.)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--compare',type=Path)
    a=p.parse_args()
    if a.out.exists() or a.out.is_symlink() or not a.source.is_file() or not (a.repo/'dngscan/raw_io.py').is_file():
        p.error('source/repo must exist and output must be new')
    if a.mode!='sdr' and a.core!='agx':
        p.error('this HDR oracle covers AgX only')
    source_stat = a.source.stat()
    source_identity = (source_stat.st_dev, source_stat.st_ino, source_stat.st_size,
                       source_stat.st_mtime_ns, source_stat.st_ctime_ns)
    previous = None
    if a.compare is not None:
        try:
            previous = json.loads(a.compare.read_text())
            if not isinstance(previous.get('identity'), dict):
                raise ValueError('comparison report has no identity')
        except (OSError, ValueError, AttributeError) as exc:
            p.error(str(exc))
    os.environ['DNGSCAN_FAST']='1'
    os.environ['DNGSCAN_FAST_SKIP']=''
    sys.path.insert(0,str(a.repo.resolve()))
    import numpy as np
    from dngscan import _fast,raw_io
    from dngscan.analysis import analyze
    from dngscan.auto_ev import compute_auto_ev
    from dngscan.scene_scale import with_intent_exposure
    from dngscan.tone import build_render_plan
    from dngscan.render import render_output_u8
    from dngscan.hdr_agx import render_ultrahdr_agx_pair
    from dngscan.hdr_agx_plan import compile_hdr_agx_plan
    from dngscan.models import HdrDisplayTarget
    from tools.benchmark_loss_pipeline import array_record,json_value
    ext=_fast._load_extension()
    if ext is None: raise RuntimeError('native extension required on both sides')
    stages={}
    def stage(name,fn):
        w,c=time.perf_counter(),time.process_time()
        result=fn()
        stages[name]={'wall_s':time.perf_counter()-w,'cpu_s':time.process_time()-c}
        return result
    load_kwargs={'_defer_clip_masks':True}
    if a.optimized:load_kwargs['_analysis_luminance_only']=True
    bundle=stage('load',lambda:raw_io.load_raw(a.source,decoder=a.decoder,**load_kwargs))
    inputs={'raw':array_record(bundle.raw_image),'colors':array_record(bundle.raw_colors),
            'scene':array_record(bundle.scene_rec2020_render),
            'processing':array_record(bundle.processing_clip_masks),
            'reference':array_record(bundle.scene_reliable_reference_rec2020)}
    kwargs={'_return_planes':False} if a.optimized else {}
    # The baseline CLI retains both named return planes through export. The
    # optimized summary-only path returns None for each; mirror those actual
    # lifetimes instead of accidentally retaining just the last `_` result.
    analysis,analysis_y,analysis_ev=stage('analyze',lambda:analyze(bundle,4,diagnostics=False,gamut_names=('P3',),**kwargs))
    masks=array_record(bundle.clip_masks)
    sink=[]
    kwargs={'_plan_sink':sink} if a.optimized else {}
    ev=stage('auto_ev',lambda:compute_auto_ev(bundle,analysis,gamut='p3',tone_core=a.core,chroma_nr=a.chroma_nr,**kwargs))
    bundle=with_intent_exposure(bundle,user_ev=ev.ev,tone_core=a.core)
    plan=stage('plan',lambda:sink[0] if sink else build_render_plan(bundle,analysis,'agx','p3',tone_core=a.core,chroma_nr=a.chroma_nr))
    identity={'inputs':inputs,'analysis':json_value(analysis),'masks':masks,'ev':json_value(ev),'plan':json_value(plan),
              'decoder':bundle.scene_decoder,'version':bundle.scene_decoder_version,'runtime':bundle.scene_decoder_runtime,
              'fallback':bundle.scene_decoder_fallback,'scene_scale':bundle.scene_scale,'align_factor':bundle.scene_align_factor,
              'reliability':bundle.scene_reliability_source,'processing_loss_pct':bundle.scene_processing_loss_pct}
    bundle=raw_io.release_analysis_buffers(bundle)
    if a.mode=='sdr':
        sdr=stage('render',lambda:render_output_u8(bundle,analysis,'p3',plan,tone_core=a.core))
        identity['sdr']=array_record(sdr)
    else:
        hdr_plan=stage('hdr_plan',lambda:compile_hdr_agx_plan(plan,HdrDisplayTarget(peak_nits=800),analysis=analysis,scene_decoder=bundle.scene_decoder))
        if a.mode == 'hdr-packed':
            from dngscan.hdr_agx import to_gainmap_alternate, achieved_headroom
            if a.optimized:
                from dngscan.hdr_agx import render_ultrahdr_agx_pair_packed
                sdr,hdr,headroom=stage('render_pack',lambda:render_ultrahdr_agx_pair_packed(bundle,analysis,plan,hdr_plan,'p3'))
            else:
                sdr,linear=stage('render',lambda:render_ultrahdr_agx_pair(bundle,analysis,plan,hdr_plan,'p3'))
                headroom=stage('headroom',lambda:achieved_headroom(linear))
                hdr=stage('pack',lambda:to_gainmap_alternate(linear,float(hdr_plan.tone.peak_linear)))
                del linear
            identity['achieved_headroom']=headroom
        else:
            sdr,hdr=stage('render',lambda:render_ultrahdr_agx_pair(bundle,analysis,plan,hdr_plan,'p3'))
        identity.update(sdr=array_record(sdr),hdr=array_record(hdr),hdr_plan=json_value(hdr_plan))
    result={'repo':str(a.repo),'commit':subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip(),
            'source':str(a.source),'source_size':a.source.stat().st_size,'optimized':a.optimized,'mode':a.mode,
            'core':a.core,'chroma_nr':a.chroma_nr,'decoder':a.decoder,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'abi':ext.native_abi_version(),'cpus':os.cpu_count()},
            'stages':stages,'total_wall_s':sum(v['wall_s'] for v in stages.values()),'total_cpu_s':sum(v['cpu_s'] for v in stages.values()),
            'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform=='darwin' else 1024),'identity':identity}
    after_stat = a.source.stat()
    after_identity = (after_stat.st_dev, after_stat.st_ino, after_stat.st_size,
                      after_stat.st_mtime_ns, after_stat.st_ctime_ns)
    if source_identity != after_identity:
        raise RuntimeError('source changed during measurement')
    result['source_identity'] = source_identity
    if previous is not None:
        result['exact']=identity==previous['identity']
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as output:
        output.write(json.dumps(json_value(result),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('total_wall_s','total_cpu_s','peak_rss_mib') }|{'exact':result.get('exact')},ensure_ascii=False),flush=True)
    return 0 if result.get('exact',True) else 2

if __name__=='__main__':raise SystemExit(main())
