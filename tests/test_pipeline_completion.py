# SPDX-License-Identifier: GPL-3.0-or-later
"""Regressions at camera-plane, opcode and sensor-evidence boundaries."""
import os, struct, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from dngscan import dng_opcodes as ops, dng_point_ops as point, raw_io, priors, analysis, _fast
from dngscan.evidence import acquire_raw_evidence
from dngscan.spatial_black import SpatialBlack, apply_to_working
from tests.test_pipeline_corrections import write_sensor_dng, write_dng_tags
from tests.test_lens_shading import _synthetic_gain_map_payload
from dngscan.metadata import _parse_gain_map_payload
from dataclasses import replace


class CameraPlaneTests(unittest.TestCase):
    def test_cam_xyz_fallback_is_neutral_and_float_does_not_clip(self):
        cm=np.zeros((3,4),np.float32)
        xyz=np.array([[.9089,-.3577,-.0787],[-.3563,1.1326,.2557],[-.0114,.0928,.5904],[0,0,0]])
        matrix=ops.libraw_camera_matrix(cm,xyz,is_dng=False)
        np.testing.assert_allclose(matrix@np.ones(3),1,atol=2e-7)
        v=np.array([[[62000,500,200]]],np.uint16)
        out=ops.camera_to_rec2020(v,np.array([[2,-1,0],[-.5,1.5,0],[0,-.2,1.2]],np.float32))
        self.assertEqual(out.dtype,np.float32)
        self.assertGreater(out.max(),65535)
        self.assertLess(out.min(),0)
        with self.assertRaises(ValueError):ops.libraw_camera_matrix(cm,cm.T,is_dng=False)

    def test_embedded_fourth_green_is_folded(self):
        m=np.eye(3,4,dtype=np.float32);m[:,3]=[.1,.2,.3]
        actual=ops.libraw_camera_matrix(m)
        np.testing.assert_allclose(actual[:,1],[.1,1.2,.3])

    def test_linear_dng_evidence_and_analysis(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'linear.dng';write_sensor_dng(p,linear=True)
            b=raw_io.load_raw(p);a,*_=analysis.analyze(b,4,diagnostics=False,gamut_names=('P3',))
            self.assertEqual(b.raw_image.shape,(128,128,3))
            self.assertEqual(b.evidence.sample_kind,'linear-camera-rgb')
            self.assertEqual(a.noise_evidence_status,'linear-camera-rgb')
            self.assertIsNone(a.gain_e_per_dn)
            self.assertTrue(np.isnan(a.noise_floor))

    def test_linear_gainmap_plane_selection_and_loss_native_reference(self):
        m=_parse_gain_map_payload(_synthetic_gain_map_payload(1.))
        m=replace(m,plane=1,planes=2,map_planes=2,gains=np.full((m.points_v,m.points_h,2),[2.,.5]))
        for fast in ('0','1') if _fast.available() else ('0',):
            with self.subTest(fast=fast),patch.dict(os.environ,DNGSCAN_FAST=fast):
                image=np.full((8,8,4),3000,np.uint16);loss=np.zeros((8,8,3),np.uint8)
                raw_io._apply_gain_maps_mosaic(SimpleNamespace(raw_image_visible=image),[m],[0]*4,4000,loss_mask=loss)
                np.testing.assert_array_equal(image[...,0],3000)
                np.testing.assert_array_equal(image[...,1],4000)
                np.testing.assert_array_equal(image[...,2],1500)
                np.testing.assert_array_equal(image[...,3],3000)
                np.testing.assert_array_equal(loss[...,1],1)

    def test_spatial_black_matches_sdk_range_normalization(self):
        model=SpatialBlack(np.array([-20.,0.,20.,40.]),np.array([0.,10.]),np.full((1,1,1),100.),(0,0),(150.,))
        original=np.full((2,4),1000,np.uint16);working=original.copy()
        raw=SimpleNamespace(raw_image_visible=working,raw_colors_visible=np.zeros((2,4),np.uint8))
        apply_to_working(raw,model,[100.],4095)
        expected=np.rint((original-model.band(0,2,4))*(4095-100)/(4095-150)+100)
        np.testing.assert_array_equal(working,expected)
        np.testing.assert_array_equal(original,1000)

    def test_scaled_geometry_is_shared_by_all_loss_maps(self):
        values=np.zeros((32,32,3),np.float16);values[7:9,15:18]=1
        op=ops.Warp(((1,.15,0,0,0,0),),.5,.5)
        bundle=SimpleNamespace(raw_image=np.zeros((64,64)),scene_rec2020_render=np.zeros((64,128,3)),orientation_flip=0,scene_geometry_ops=(op,),scene_crop_sensor=None)
        expected=ops.warp_image(raw_io._resize_loss_to_shape(values,(32,64)),op,loss=True)
        np.testing.assert_array_equal(ops.align_loss(bundle,values),expected)


class PointOpcodeTests(unittest.TestCase):
    def test_point_domains_plane_pitch_and_row_offsets(self):
        area=(1,2,5,6,1,1,2,2)
        image=np.full((6,8,3),110,np.uint16)
        op=point.PointOp(10,area,(.1,.2),2)
        point.apply(image,op,black=10,white=1010)
        self.assertEqual(image[1,2,1],210);self.assertEqual(image[3,4,1],310)
        self.assertEqual(image[2,2,1],110);self.assertEqual(image[1,2,0],110)
        image=np.array([[2000]],np.uint16)
        point.apply(image,point.PointOp(8,(0,0,1,1,0,1,1,1),(20.,2.),1))
        self.assertEqual(image[0,0],4020)

    def test_stage3_point_preview_runs_in_full_sensor_coordinates(self):
        payload=struct.pack('>4l5L2d',0,0,128,128,0,3,1,1,1,0.,.5)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'point.dng';write_sensor_dng(p,opcodes={51022:[(8,payload)]})
            full=raw_io.load_raw(p);half=raw_io.load_raw(p,scene_half_size=True)
            expected=full.scene_rec2020_render.reshape(64,2,64,2,3).mean(axis=(1,3))
            np.testing.assert_array_equal(half.scene_rec2020_render,expected)

    def test_bad_pixel_phase_and_original_neighbours(self):
        colors=np.tile([[0,1],[3,2]],(5,5)).astype(np.uint8)
        image=np.full((10,10),1200,np.uint16);image[4,4]=0
        raw=SimpleNamespace(raw_image=image,raw_colors=colors,color_desc=b'RGBG',sizes=SimpleNamespace(top_margin=0,left_margin=0))
        loss=np.zeros(image.shape,np.uint8)
        point.repair_bad_pixels(raw,point.BadPixels(1,0),loss)
        self.assertEqual(image[4,4],1200);self.assertEqual(loss[4,4],1)
        with self.assertRaisesRegex(ValueError,'phase'):point.repair_bad_pixels(raw,point.BadPixels(0,0))

    def test_warp2_skip_fallback_and_native_reference(self):
        coeff=(1.,.05)+tuple(0. for _ in range(13))+(0.002,-.001,0.,1.)
        payload=struct.pack('>L21dL',1,*coeff,.5,.5,0)
        old=struct.pack('>L8d',1,1.,.1,0.,0.,0.,0.,.5,.5)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'metadata.dng';write_dng_tags(p,{51022:[(14,0x01060000,1,payload),(1,0x01030000,0,old)]})
            plan=ops.read_plan(p);self.assertEqual(plan.names,['WarpRectilinear2'])
        image=np.random.default_rng(5).integers(0,65535,(35,41,3),dtype=np.uint16)
        with patch.dict(os.environ,DNGSCAN_FAST='0'):expected=ops.warp_image(image,plan.post[0])
        if _fast.available():np.testing.assert_allclose(ops.warp_image(image,plan.post[0]),expected,atol=1)
        spline=ops.Warp(((1.,0,0,0,0,0),),.5,.5,knots=(0.,.5,1.),scales=((1.,1.1,1.2),)*3,scale=1.2)
        with patch.dict(os.environ,DNGSCAN_FAST='0'):expected=ops.warp_image(image,spline)
        if _fast.available():np.testing.assert_allclose(ops.warp_image(image,spline),expected,atol=1)


class PriorEvidenceTests(unittest.TestCase):
    def test_dn_scale_and_unsafe_priors_are_consistent(self):
        prior={'id':'test','measured_iso':100,'fwc_e':16000.,'gain_e_per_dn_at_measured_iso':1.,'read_noise_log2iso_log2e':[[6.643856,1.]],'pdr_log2iso_ev':[[6.643856,10.]],'suspect_iso_min':1600}
        with patch.object(priors,'gain_e_per_dn',return_value=1.):
            self.assertEqual(priors.gain_for_file(prior,100,64000.),.25)
            self.assertIsNone(priors.gain_for_file(prior,100,23000.))
        for iso,status in ((1600,'independent'),(100,'spatially-correlated'),(100,'linear-camera-rgb')):
            with patch.object(priors,'find_priors',return_value=prior):
                result=analysis.sensor_prior_evidence('test','test',iso,nf=.001,fullwell=16000,mean_black=0,coding_range=16000,noise_status=status)
                self.assertTrue(all(x is None for x in result[4:]))


if __name__=='__main__':unittest.main()
