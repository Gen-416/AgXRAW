# SPDX-License-Identifier: GPL-3.0-or-later
"""File-backed regressions against DNG calibration semantics, not kernel parity."""
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dngscan import _fast, dng_opcodes, raw_io
from dngscan._deps import rawpy
from tests.test_pipeline_corrections import write_sensor_dng, write_dng_tags


def gain_payload(gains, *, empty=False, first=0, planes=1):
    return (struct.pack('>10L',0,0,0 if empty else 128,0 if empty else 128,
                        first,planes,1,1,1,1)
            + struct.pack('>4dL',1,1,0,0,len(gains))
            + struct.pack('>'+str(len(gains))+'f',*gains))


class DngCalibrationRegressionTests(unittest.TestCase):
    def test_repeat_black_without_deltas_is_subtracted_once_before_gain(self):
        # Identical 1000-DN signals on scalar vs striped 4x4 black pedestals.
        # The stripe's period deliberately exceeds the CFA's 2x2 period.
        pattern=np.tile(np.array([[0,0,1000,1000]],np.uint16),(4,1))
        original=1000+np.tile(pattern,(32,32))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'black.dng'
            for fast in ('0','1') if _fast.available() else ('0',):
                with self.subTest(fast=fast),patch.dict(os.environ,DNGSCAN_FAST=fast):
                    for gain in (1.,2.):
                        opcodes={} if gain==1 else {51009:[(9,gain_payload([gain]))]}
                        for half in (False,True):
                            write_sensor_dng(p,black_pattern=[[1000]],opcodes=opcodes)
                            reference=raw_io.load_raw(p,scene_half_size=half).scene_rec2020_render
                            for delta in (None,np.zeros(128)):
                                write_sensor_dng(p,black_pattern=pattern,black_deltas=delta,opcodes=opcodes)
                                actual=raw_io.load_raw(p,scene_half_size=half)
                                self.assertIsNotNone(actual.evidence.spatial_black)
                                np.testing.assert_array_equal(actual.raw_image,original)
                                # Compare actual LibRaw output: stale cblack subtraction
                                # used to leave severe stripes despite unit tests passing.
                                np.testing.assert_allclose(actual.scene_rec2020_render[10:-10,10:-10],
                                    reference[10:-10,10:-10],atol=3,rtol=0)

    def test_point_transform_uses_normalized_domain_after_spatial_black(self):
        polynomial=struct.pack('>4l5L2d',0,0,128,128,0,1,1,1,1,0.,.5)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'black.dng'
            pattern=np.array([[0,500],[750,1000]])
            write_sensor_dng(p,black_pattern=pattern,opcodes={51009:[(8,polynomial)]})
            actual=raw_io.load_raw(p)
            write_sensor_dng(p,black_pattern=[[1000]],signal=500)
            reference=raw_io.load_raw(p)
            np.testing.assert_allclose(actual.scene_rec2020_render[10:-10,10:-10],
                reference.scene_rec2020_render[10:-10,10:-10],atol=3,rtol=0)

    def test_empty_area_and_absolute_map_planes_through_real_decoder(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'gain.dng'
            for fast in ('0','1') if _fast.available() else ('0',):
                with self.subTest(fast=fast),patch.dict(os.environ,DNGSCAN_FAST=fast):
                    write_sensor_dng(p,signal=3200)
                    reference=raw_io.load_raw(p)
                    write_sensor_dng(p,signal=3200,opcodes={51009:[(9,gain_payload([.5],empty=True))]})
                    actual=raw_io.load_raw(p)
                    np.testing.assert_allclose(actual.scene_rec2020_render,
                        reference.scene_rec2020_render*.5,atol=2,rtol=0)
                    np.testing.assert_array_equal(actual.raw_image,reference.raw_image)
                    write_sensor_dng(p,linear=True)
                    reference=raw_io.load_raw(p)
                    with rawpy.imread(str(p)) as raw:
                        matrix=dng_opcodes.camera_to_rec2020(np.eye(3,dtype=np.uint16).reshape(1,3,3),
                            dng_opcodes.libraw_camera_matrix(raw.color_matrix,raw.rgb_xyz_matrix))[0].T
                    write_sensor_dng(p,linear=True,opcodes={51009:[(9,
                        gain_payload([.25,.5,.75],first=1,planes=2))]})
                    actual=raw_io.load_raw(p)
                    measured=(np.linalg.solve(matrix,actual.scene_rec2020_render[64,64])/
                              np.linalg.solve(matrix,reference.scene_rec2020_render[64,64]))
                    np.testing.assert_allclose(measured,[1.,.5,.75],atol=4e-4,rtol=0)

    def test_terminal_stage3_trim_preserves_scene_and_evidence_coordinates(self):
        y,x=np.indices((128,128))
        pixels=400+y*8+x*4
        pixels[40:46,60:66]=4095
        trims=[(6,struct.pack('>4l',8,10,124,120)),
               (6,struct.pack('>4l',10,12,120,116))]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'trim.dng'
            write_sensor_dng(p,signal=pixels)
            original=raw_io.load_raw(p)
            write_sensor_dng(p,signal=pixels,opcodes={51022:trims})
            full=raw_io.load_raw(p)
            half=raw_io.load_raw(p,scene_half_size=True)
            np.testing.assert_array_equal(full.raw_image,original.raw_image)
            np.testing.assert_array_equal(full.scene_rec2020_render,
                original.scene_rec2020_render[10:120,12:116])
            np.testing.assert_array_equal(full.clip_masks,original.clip_masks[10:120,12:116])
            self.assertEqual(full.scene_crop_sensor,(10.,12.,110.,104.))
            np.testing.assert_array_equal(half.scene_rec2020_render,
                full.scene_rec2020_render.reshape(55,2,52,2,3).mean(axis=(1,3)))

    def test_trim_rejects_unsupported_stage_and_coordinate_order(self):
        trim=struct.pack('>4l',8,10,124,120)
        vignette=struct.pack('>7d',0.,0.,0.,0.,0.,.5,.5)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'trim.dng'
            for stage in (51008,51009):
                write_dng_tags(p,{stage:[(6,0x01030000,0,trim)]})
                with self.assertRaisesRegex(ValueError,'required DNG'):
                    dng_opcodes.read_plan(p)
                write_dng_tags(p,{stage:[(6,0x01030000,1,trim)]})
                optional=dng_opcodes.read_plan(p)
                self.assertEqual(optional.names,[])
                self.assertEqual(len(optional.skipped),1)
                self.assertIn('TrimBounds',optional.skipped[0])
            write_dng_tags(p,{51022:[(6,0x01030000,0,trim),(3,0x01030000,0,vignette)]})
            with self.assertRaisesRegex(ValueError,'after TrimBounds'):
                dng_opcodes.read_plan(p)
            write_sensor_dng(p,scale=[2.,1.],opcodes={51022:[(6,trim)]})
            with self.assertRaisesRegex(RuntimeError,'square-pixel'):
                raw_io.load_raw(p)
            write_sensor_dng(p,opcodes={51022:[(6,struct.pack('>4l',0,0,129,128))]})
            with self.assertRaisesRegex(RuntimeError,'outside the current image'):
                raw_io.load_raw(p)

    def test_gainmap_parser_rejects_invalid_area_and_spacing(self):
        from dngscan.metadata import _parse_gain_map_payload
        valid=gain_payload([1.])
        self.assertIsNone(_parse_gain_map_payload(valid+b'\0'))
        for offset in (20,24,28):  # plane count, row pitch, column pitch
            malformed=bytearray(valid)
            struct.pack_into('>L',malformed,offset,0)
            self.assertIsNone(_parse_gain_map_payload(malformed))
        malformed=bytearray(valid)
        struct.pack_into('>d',malformed,40,0.)
        self.assertIsNone(_parse_gain_map_payload(malformed))
        malformed=bytearray(gain_payload([1.],empty=True))
        struct.pack_into('>L',malformed,24,2)
        self.assertIsNone(_parse_gain_map_payload(malformed))

    def test_signed_gain_area_clips_without_shifting_pitch_phase(self):
        from dngscan.metadata import _parse_gain_map_payload
        payload=bytearray(gain_payload([2.]))
        struct.pack_into('>4l',payload,0,-3,-2,6,6)
        struct.pack_into('>2L',payload,24,2,3)
        op=_parse_gain_map_payload(payload)
        for fast in ('0','1') if _fast.available() else ('0',):
            with self.subTest(fast=fast),patch.dict(os.environ,DNGSCAN_FAST=fast):
                pixels=np.full((8,8),1000,np.uint16)
                raw=SimpleNamespace(raw_image_visible=pixels,
                    raw_colors_visible=np.zeros((8,8),np.uint8))
                raw_io._apply_gain_maps_mosaic(raw,[op],[100.],4000)
                expected=np.full((8,8),1000,np.uint16)
                expected[1:6:2,1:6:3]=1900
                np.testing.assert_array_equal(pixels,expected)

    def test_disjoint_gain_area_leaves_image_unchanged(self):
        from dngscan.metadata import _parse_gain_map_payload
        for bounds in ((-5,-5,-1,-1),(-5,2,-1,6),(2,-5,6,-1),
                       (10,10,14,14),(2,10,6,14),(10,2,14,6)):
            payload=bytearray(gain_payload([2.]))
            struct.pack_into('>4l',payload,0,*bounds)
            struct.pack_into('>2L',payload,24,2,3)
            op=_parse_gain_map_payload(payload)
            self.assertIsNotNone(op)
            for fast in ('0','1') if _fast.available() else ('0',):
                with self.subTest(bounds=bounds,fast=fast),patch.dict(os.environ,DNGSCAN_FAST=fast):
                    pixels=np.full((8,8),1000,np.uint16)
                    raw=SimpleNamespace(raw_image_visible=pixels,
                        raw_colors_visible=np.zeros((8,8),np.uint8))
                    raw_io._apply_gain_maps_mosaic(raw,[op],[100.],4000)
                    np.testing.assert_array_equal(pixels,np.full((8,8),1000,np.uint16))
                    self.assertEqual((op.top,op.left,op.bottom,op.right),bounds)

    def test_fractional_evidence_crop_uses_exact_sensor_footprints(self):
        values=np.arange(4*4*3,dtype=np.float16).reshape(4,4,3)
        full=np.repeat(np.repeat(values,2,axis=0),2,axis=1)[1:5,1:5]
        np.testing.assert_array_equal(dng_opcodes.align_sensor_loss(
            values,(8,8),(4,4),0,crop=(1,1,4,4)),full)
        half=full.reshape(2,2,2,2,3).max(axis=(1,3))
        np.testing.assert_array_equal(dng_opcodes.align_sensor_loss(
            values,(8,8),(2,2),0,crop=(1,1,4,4)),half)
        np.testing.assert_array_equal(dng_opcodes.align_sensor_loss(
            values,(8,8),(4,4),5,crop=(1,1,4,4)),np.rot90(full))

    def test_reference_samples_share_trim_and_warp_geometry_with_main_decode(self):
        from dngscan.scene_reference import reliable_reference_samples
        y,x=np.indices((128,128))
        pixels=400+y*8+x*4
        pixels[40:46,60:66]=4095
        warp=struct.pack('>L8d',1,1.,.12,0.,0.,0.,0.,.5,.5)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'trim-reference.dng'
            for top,left in ((10,12),(11,13)):
                for geometry in ([],[(1,warp)]):
                    with self.subTest(origin=(top,left),warp=bool(geometry)):
                        ops=geometry+[(6,struct.pack('>4l',top,left,120,116))]
                        write_sensor_dng(p,signal=pixels,opcodes={51022:ops})
                        full=raw_io.load_raw(p)
                        bundle=raw_io.load_raw(p,scene_half_size=True)
                        h,w=bundle.scene_rec2020_render.shape[:2]
                        np.testing.assert_array_equal(bundle.scene_rec2020_render,
                            full.scene_rec2020_render[:2*h,:2*w].reshape(h,2,w,2,3).mean(axis=(1,3)))
                        with rawpy.imread(str(p)) as raw:
                            scene,loss,recipe,_=raw_io._decode_corrected_libraw(
                                raw,p,bundle.evidence,'clip',True,None)
                        samples,pct=reliable_reference_samples(bundle.evidence,scene,
                            bundle.scene_scale,loss,recipe)
                        reliable=np.max(bundle.clip_masks.reshape(-1,3),axis=1)<.1
                        expected=(scene.reshape(-1,3)/np.float32(bundle.scene_scale))[reliable]
                        np.testing.assert_array_equal(samples,expected)
                        self.assertAlmostEqual(pct,np.mean(reliable)*100)
                        self.assertEqual(recipe.crop,(float(top),float(left),float(2*h),float(2*w)))


if __name__=='__main__':unittest.main()
