# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dngscan import _fast, dng_opcodes as ops, raw_io
from dngscan.metadata import DngVignetteRadial, _parse_gain_map_payload
from tests.test_lens_shading import _synthetic_gain_map_payload
from tests.test_preview_cache import _bundle, _analysis


def write_dng_tags(path, opcodes):
    """Tiny metadata-only DNG; opcode header bytes are deliberately literal."""
    tags = [(50706, 1, bytes([1,4,0,0]), 4), (262, 3, struct.pack('<H',32803),1),
            (256,4,struct.pack('<L',64),1), (257,4,struct.pack('<L',48),1)]
    for tag, values in opcodes.items():
        blob = struct.pack('>L',len(values)) + b''.join(
            struct.pack('>4L',oid,version,flags,len(payload))+payload for oid,version,flags,payload in values)
        tags.append((tag,7,blob,len(blob)))
    tags.sort()
    start = 8 + 2 + 12*len(tags)+4
    data, entries = bytearray(), bytearray()
    for tag,typ,payload,count in tags:
        value = payload.ljust(4,b'\0') if len(payload)<=4 else struct.pack('<L',start+len(data))
        entries += struct.pack('<HHL',tag,typ,count)+value
        if len(payload)>4: data += payload
    path.write_bytes(b'II'+struct.pack('<HLH',42,8,len(tags))+entries+bytes(4)+data)


def write_sensor_dng(path, limit=1., lut=False, spatial_black=False):
    pix=np.full((128,128),int(.88*4095),dtype='<u2')
    entries=[]
    def add(tag,typ,values):
        if typ==2:
            data=values.encode()+b'\0';n=len(data)
        elif typ in (5,10):
            data=b''.join(struct.pack('<'+('l' if typ==10 else 'L')*2,int(round(v*1000000)),1000000) for v in values);n=len(values)
        else:
            data=struct.pack('<'+{1:'B',3:'H',4:'L'}[typ]*len(values),*values);n=len(values)
        entries.append((tag,typ,n,data))
    for tag,typ,v in [(254,4,[0]),(256,4,[128]),(257,4,[128]),(258,3,[16]),(259,3,[1]),(262,3,[32803]),(271,2,'Review'),(272,2,'Synthetic'),(273,4,[0]),(277,3,[1]),(278,4,[128]),(279,4,[pix.nbytes]),(284,3,[1]),(33421,3,[2,2]),(33422,1,[0,1,1,2]),(50706,1,[1,4,0,0]),(50707,1,[1,1,0,0]),(50708,2,'Review Synthetic'),(50710,1,[0,1,2]),(50711,3,[1]),(50713,3,[1,1]),(50714,5,[0]),(50717,4,[8190 if lut else 4095]),(50719,4,[0,0]),(50720,4,[128,128]),(50721,10,[1,0,0,0,1,0,0,0,1]),(50728,5,[1,1,1]),(50730,10,[1]),(50734,5,[limit]),(50778,3,[21])]: add(tag,typ,v)
    if lut:add(50712,3,list(range(0,8192,2)))
    if spatial_black:add(50715,10,np.linspace(-64,64,128))
    entries.sort();off=8+2+12*len(entries)+4;body=bytearray();heads=[]
    for tag,typ,n,data in entries:
        if tag==273:
            heads.append((tag,typ,n,None));continue
        if len(data)<=4:heads.append((tag,typ,n,data.ljust(4,b'\0')))
        else:
            heads.append((tag,typ,n,struct.pack('<L',off+len(body))))
            body+=data
            if len(body)%2:body+=b'\0'
    payload=b''.join(struct.pack('<HHL',tag,typ,n)+(data if data is not None else struct.pack('<L',off+len(body))) for tag,typ,n,data in heads)
    path.write_bytes(b'II'+struct.pack('<HLH',42,8,len(entries))+payload+struct.pack('<L',0)+body+pix.tobytes())
    return int(pix[0,0])


class OpcodeTests(unittest.TestCase):
    def test_order_and_required_flags(self):
        warp = struct.pack('>L8d',1,1.,.1,0.,0.,0.,0.,.5,.5)
        vignette = struct.pack('>7d',1.,0.,0.,0.,0.,.5,.5)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'sample.dng'
            write_dng_tags(p,{51022:[(3,0x01030000,0,vignette),(1,0x01030000,0,warp),
                                     (3,0x01030000,0,vignette)]})
            plan=ops.read_plan(p)
            self.assertEqual(plan.names,['FixVignetteRadial','WarpRectilinear','FixVignetteRadial'])
            self.assertEqual(len(plan.post),3)
            for tag in (51008,51009,51022):
                write_dng_tags(p,{tag:[(999,0x01030000,0,b'')]})
                with self.assertRaisesRegex(ValueError,'required DNG'):
                    ops.read_plan(p)
                write_dng_tags(p,{tag:[(999,0x01030000,1,b'')]})
                self.assertEqual(len(ops.read_plan(p).skipped),1)
            write_dng_tags(p,{51022:[(1,0x01030000,0,warp[:-1])]})
            with self.assertRaisesRegex(ValueError,'size'):
                ops.read_plan(p)

    def test_warp_polynomial_and_channel_identity(self):
        # A linear coordinate ramp makes the analytic source position observable.
        h,w=48,64
        image=np.broadcast_to(np.arange(w,dtype=np.uint16)[None,:,None]*100,(h,w,3)).copy()
        op=ops.Warp(((.8,0,0,0,0,0),(1.,0,0,0,0,0),(1.1,0,0,0,0,0)),.5,.5)
        out=ops.warp_image(image,op)
        for c,scale in enumerate((.8,1.,1.1)):
            expected=(32+(np.arange(8,56)-32)*scale)*100
            np.testing.assert_allclose(out[20,8:56,c],expected,atol=1)
        self.assertIs(ops.warp_image(image,ops.Warp(((1.,0,0,0,0,0),),.5,.5)),image)

    def test_native_reference_warp_and_conservative_loss(self):
        rng=np.random.default_rng(118)
        scene=rng.integers(0,65536,(45,61,3),dtype=np.uint16)[::-1]
        loss=(scene>64000).astype(np.float16)
        for fisheye in (False,True):
            op=ops.Warp(((1.05,-.2,.08,0,.003,-.001),),.4,.55,fisheye,1.1)
            with patch.dict(os.environ,DNGSCAN_FAST='0'):
                ref=ops.warp_image(scene,op)
                mask=ops.warp_image(loss,op,loss=True)
            if _fast.available():
                np.testing.assert_allclose(ops.warp_image(scene,op),ref,atol=1)
                np.testing.assert_array_equal(ops.warp_image(loss,op,loss=True),mask)
            # Interpolation never dilutes a recorded clipping event.
            self.assertEqual(set(np.unique(mask)),{0.,1.})

    def test_tangential_xy_order(self):
        op=ops.Warp(((1.,0,0,0,.02,.03),),.5,.5)
        sy,sx=ops._coordinates(op,60,80,30,31,0)
        # Radius is 50; point (60,30) has normalized (x,y)=(.4,0).
        self.assertAlmostEqual(sy[0,60],30+50*.02*.16)
        self.assertAlmostEqual(sx[0,60],60+50*.03*.48)


class Stage1ProviderTests(unittest.TestCase):
    def test_linearization_and_response_limit_are_already_applied(self):
        from dngscan.evidence import acquire_raw_evidence
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'sensor.dng'
            stored=write_sensor_dng(p)
            original=acquire_raw_evidence(p)
            write_sensor_dng(p,lut=True)
            linearized=raw_io.load_raw(p)
            np.testing.assert_array_equal(linearized.raw_image,2*stored)
            self.assertIsNone(linearized.evidence_stage1_note)
            write_sensor_dng(p,limit=.8)
            limited=raw_io.load_raw(p)
            self.assertEqual(limited.white_level,original.white_level)
            self.assertLess(limited.camera_white_levels[0],original.camera_white_levels[0])
            self.assertEqual(ops.read_plan(p).white_levels,(4095.,))
            self.assertIsNone(limited.evidence_stage1_note)
            self.assertTrue(np.all(limited.clip_masks==1))
            write_sensor_dng(p,spatial_black=True)
            spatial=raw_io.load_raw(p)
            self.assertIn('BlackLevelDeltaH',spatial.evidence_stage1_note)
            np.testing.assert_array_equal(spatial.raw_image,stored)


class ProcessingLossTests(unittest.TestCase):
    def test_cached_export_replays_analysis_mask_state(self):
        from dngscan.gui import service
        analysis=_analysis()
        analysis.channel_fullwell={i:800 for i in range(4)}
        fresh=_bundle()
        fresh.raw_image[:]=790
        expected=_bundle()
        expected.raw_image[:]=790
        raw_io.refresh_clip_masks_from_fullwell(expected,analysis.channel_fullwell)
        class ReachedPlan(Exception): pass
        def inspect_bundle(bundle,*args,**kwargs):
            np.testing.assert_array_equal(bundle.clip_masks,expected.clip_masks)
            self.assertEqual(bundle._clip_mask_fullwell,analysis.channel_fullwell)
            raise ReachedPlan()
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'test.dng';source.write_bytes(b'fixture')
            fresh.path=source
            with patch.object(service.dg,'require_dependencies'), \
                 patch.object(service.dg,'load_raw',return_value=fresh), \
                 patch.object(service,'_cached_full_analysis',return_value=analysis), \
                 patch.object(service.dg,'analyze',side_effect=AssertionError('cache should be reused')), \
                 patch.object(service.dg,'build_render_plan',side_effect=inspect_bundle):
                with self.assertRaises(ReachedPlan):
                    service.run_export({'input':str(source),'format':'sdr','wb':'camera','ev':0,
                                        'filmOpticsSeed':1,'outdir':td})

    def test_gainmap_logs_intermediate_clipping_without_touching_evidence(self):
        for native in ('0','1') if _fast.available() else ('0',):
            with self.subTest(native=native), patch.dict(os.environ,DNGSCAN_FAST=native):
                sensor=np.full((8,8),3000,dtype=np.uint16)
                corrected=sensor.copy()
                raw=SimpleNamespace(raw_image_visible=corrected,
                                    raw_colors_visible=np.zeros((8,8),dtype=np.uint8))
                loss=np.zeros((8,8),dtype=np.uint8)
                maps=[_parse_gain_map_payload(_synthetic_gain_map_payload(g)) for g in (2.,.5)]
                raw_io._apply_gain_maps_mosaic(raw,maps,[0.],4000,loss_mask=loss)
                np.testing.assert_array_equal(sensor,3000)
                np.testing.assert_array_equal(corrected,2000)
                np.testing.assert_array_equal(loss,1)

    def test_radial_clipping_is_separate_and_oriented(self):
        scene=np.full((32,48,3),50000,dtype=np.uint16)
        op=DngVignetteRadial((1.,0.,0.,0.,0.),.25,.6)
        reference=scene.copy(); loss=np.zeros(scene.shape,dtype=np.float16)
        raw_io._apply_vignette_render(reference,op,loss_mask=loss)
        self.assertTrue(np.any(loss))
        for flip in range(8):
            oriented=raw_io._orient_like_libraw(scene.copy(),flip)
            tracked=np.zeros(oriented.shape,dtype=np.float16)
            raw_io._apply_vignette_render(oriented,op,flip,tracked)
            np.testing.assert_array_equal(tracked,raw_io._orient_like_libraw(loss,flip))

    def test_radial_white_bound_follows_camera_wb_not_container(self):
        # The weaker WB plane reaches DNG logical white before uint16 maximum.
        scene=np.full((32,48,3),20000,dtype=np.uint16)
        loss=np.zeros(scene.shape,dtype=np.float16)
        limits=np.array([65535.,24000.,48000.])
        raw_io._apply_vignette_render(scene,DngVignetteRadial((1.,0.,0.,0.,0.),.5,.5),
                                     loss_mask=loss,channel_limits=limits)
        self.assertEqual(int(scene[0,0,1]),24000)
        self.assertEqual(float(loss[0,0,1]),1.)
        self.assertEqual(float(loss[0,0,0]),0.)

    def test_fullwell_refresh_preserves_processing_loss(self):
        bundle=_bundle()
        bundle.processing_clip_masks=np.zeros((4,4,3),dtype=np.float16)
        bundle.processing_clip_masks[1,1,2]=1
        before=bundle.raw_image.copy()
        self.assertTrue(raw_io.refresh_clip_masks_from_fullwell(bundle,{0:800,1:800,2:800,3:800}))
        self.assertEqual(float(bundle.clip_masks[2,2,2]),1.)
        np.testing.assert_array_equal(bundle.raw_image,before)

    def test_correction_diagnostics_survive_disk_metadata(self):
        from dngscan.gui.preview_cache import _bundle_metadata,_bundle_from_cache
        bundle=_bundle();bundle.lens_shading='gainmap+vignette'
        bundle.scene_correction_note='optional opcode skipped'
        bundle.scene_processing_loss_pct=12.5
        bundle.evidence_stage1_note='spatial black approximation'
        restored=_bundle_from_cache(bundle.path,_bundle_metadata(bundle),bundle.scene_rec2020_render,bundle.clip_masks,None)
        self.assertEqual(restored.lens_shading,bundle.lens_shading)
        self.assertEqual(restored.scene_correction_note,bundle.scene_correction_note)
        self.assertEqual(restored.scene_processing_loss_pct,12.5)
        self.assertEqual(restored.evidence_stage1_note,bundle.evidence_stage1_note)


if __name__=='__main__':
    unittest.main()
