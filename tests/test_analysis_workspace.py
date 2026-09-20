# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact analysis storage/reduction and job-local AutoEV handoff contracts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from dngscan import analysis as an, raw_io, auto_ev
from dngscan.models import RenderAdjustments
from dngscan.scene_scale import with_intent_exposure
from dngscan.spatial_black import SpatialBlack
from dngscan.tone import build_render_plan
from tests.test_preview_cache import _bundle, _analysis
from tools.benchmark_loss_pipeline import json_value


class LuminanceStorageTests(unittest.TestCase):
    def test_exact_stored_y_all_storage_types_and_strides(self):
        rng = np.random.default_rng(943)
        for dtype in (np.float16, np.float32, np.float64, np.uint16):
            scene = rng.uniform(-2, 100, (19, 23, 3)).astype(dtype)
            if np.issubdtype(dtype, np.floating):
                scene.reshape(-1)[:5] = [0., -0., np.nan, np.inf, -np.inf]
            for data in (scene, scene[::-1, ::2], np.broadcast_to(scene[:1], (7, 23, 3))):
                for scale in (1., 7.125, 65535.):
                    with self.subTest(dtype=dtype, strides=data.strides, scale=scale), np.errstate(all='ignore'):
                        xyz = raw_io.scene_rec2020_to_xyz_render(data, scale)
                        y = raw_io._scene_rec2020_to_y_render(data, scale)
                        self.assertEqual(y.dtype, xyz.dtype)
                        self.assertEqual(y.tobytes(), xyz[..., 1].tobytes())
                        b = replace(_bundle(), xyz_render=xyz, render_scale=scale)
                        narrow = replace(b, xyz_render=None, _analysis_y_render=y)
                        self.assertEqual(an._bundle_luminance(b).tobytes(), an._bundle_luminance(narrow).tobytes())

    def test_green_conversion_selection_and_median_rounding(self):
        rng = np.random.default_rng(304)
        for dtype in (np.float16, np.float32, np.uint16):
            scene = rng.uniform(0, 200, (23, 19, 3)).astype(dtype)
            if dtype != np.uint16:
                scene[0, :6, 1] = [np.nan, np.inf, -np.inf, 0., -0., 1e-4]
            for scale in (1., 7.1, 65535.):
                old = raw_io.scene_green_median(np.asarray(scene, dtype=np.float32) / scale)
                new = raw_io._stored_scene_green_median(scene, scale)
                self.assertEqual(old, new)

    def test_summary_retains_all_analysis_fields_without_output_planes(self):
        b = _bundle()
        b.scene_rec2020_render = np.random.default_rng(33).uniform(0, 400, b.scene_rec2020_render.shape).astype(np.float32)
        b.xyz_render = raw_io.scene_rec2020_to_xyz_render(b.scene_rec2020_render, b.scene_scale)
        expected, y, ev = an.analyze(b, 4, diagnostics=False)
        b = replace(b, xyz_render=None,
                    _analysis_y_render=raw_io._scene_rec2020_to_y_render(b.scene_rec2020_render, b.scene_scale))
        actual, actual_y, actual_ev = an.analyze(b, 4, diagnostics=False, _return_planes=False)
        self.assertEqual(json_value(actual), json_value(expected))
        self.assertIsNone(actual_y)
        self.assertIsNone(actual_ev)
        self.assertIsNotNone(y)
        self.assertIsNotNone(ev)
        self.assertIsNone(raw_io.release_analysis_buffers(b)._analysis_y_render)
        # Missing evidence still uses exact decoded-image metrics.
        b = replace(b, raw_image=None, raw_colors=None)
        self.assertEqual(json_value(an.analyze(b, 4)[0]), json_value(an.analyze(b, 4, _return_planes=False)[0]))


class GamutMedianReuseTests(unittest.TestCase):
    def test_native_known_median_matches_legacy_selection(self):
        from dngscan import _fast
        from dngscan.color import RGB_TO_XYZ, XYZ_TO_RGB
        native = _fast.kernel('gamut_counts')
        if native is None:
            self.skipTest('native disabled')
        rng = np.random.default_rng(277)
        for count in (1, 2, 3, 256, 257, 2001):
            scene = rng.normal(0.5, 1., (count, 3)).astype(np.float32)
            for y in (rng.uniform(0, 1, count).astype(np.float32), np.ones(count, np.float32)):
                args = (scene, y, 1., list(RGB_TO_XYZ['Rec2020'].reshape(-1)),
                        [list(XYZ_TO_RGB['P3'].reshape(-1))], 1e-8, 1e-6)
                self.assertEqual(native(*args), native(*args, median=float(np.median(y))))


class PhaseReductionTests(unittest.TestCase):
    def test_shared_phase_reductions_match_independent_oracle(self):
        from dngscan import spatial_black
        rng = np.random.default_rng(877)
        for pattern in (np.array([[0, 1], [3, 2]], np.uint8),
                        np.array([[1,0,1,1,2,1],[2,1,2,0,1,0],[1,0,1,1,2,1],
                                  [1,2,1,1,0,1],[0,1,0,2,1,2],[1,2,1,1,0,1]],np.uint8)):
            for shape in ((129, 143), (15, 17), (3, 7)):
                for spatial in (False, True):
                    b = _bundle()
                    ph,pw=pattern.shape
                    b.raw_pattern=pattern.tolist()
                    b.raw_colors=np.tile(pattern, ((shape[0]+ph-1)//ph,(shape[1]+pw-1)//pw))[:shape[0],:shape[1]]
                    b.raw_image=rng.integers(100, 1200, shape, np.uint16)
                    b.black_levels=[63.,65.,61.,64.]
                    if spatial:
                        model=SpatialBlack(np.arange(shape[1],dtype=np.float32)*.01,
                            np.arange(shape[0],dtype=np.float32)*.02,
                            np.full((1,1,1),63.,np.float32),(0,0),(70.,))
                        b.evidence=SimpleNamespace(spatial_black=model)
                    else:
                        b.evidence=None
                    ids=list(map(int,np.unique(b.raw_colors)))
                    labels=an.channel_labels('RGBG',ids)
                    fullwell={cid:16383 for cid in ids}
                    expected_nf=an.estimate_raw_noise_floor(b,fullwell)
                    expected_snr=an.compute_snr_curves(b,ids,labels,fullwell)
                    cache={}
                    with patch.object(spatial_black,'corrected_plane',wraps=spatial_black.corrected_plane) as corrected:
                        nf=an.estimate_raw_noise_floor(b,fullwell,_phase_stats=cache)
                        calls=corrected.call_count
                        snr=an.compute_snr_curves(b,ids,labels,fullwell,_phase_stats=cache)
                        self.assertEqual(corrected.call_count,calls)
                    self.assertEqual(json_value(nf),json_value(expected_nf))
                    self.assertEqual(json_value(snr),json_value(expected_snr))
                    self.assertTrue(all(a.ndim == 1 for arrays in cache.values() for a in arrays))


class AutoEvWorkTests(unittest.TestCase):
    def test_baseline_stats_and_last_high_probe_only_computed_once(self):
        b=_bundle()
        sample=np.full((1000,3),.1,np.float32)
        b.scene_rec2020_render=sample.reshape(20,50,3)
        calls=[]
        def render(bundle,analysis,gamut,ev,*args,**kwargs):
            calls.append(ev)
            return sample * np.float32(2**ev)
        with patch.object(auto_ev,'render_sample_linear_output',side_effect=render), \
             patch.object(auto_ev,'output_highlight_stats',wraps=auto_ev.output_highlight_stats) as stats:
            cap=auto_ev.max_safe_ev(b,None,'p3',search_hi=1.,tone_plan=SimpleNamespace())
        self.assertEqual(calls,[0.,.5,1.])
        self.assertEqual(stats.call_count,len(calls))
        self.assertEqual(cap,1.)

    def test_planning_and_probe_do_not_expand_same_geometry_half_mask(self):
        b, a = _bundle(), _analysis()
        build_render_plan(b, a, 'agx', 'p3')
        auto_ev.compute_auto_ev(b, a, gamut='p3')
        self.assertIsNone(b._clip_masks_resized)
        self.assertEqual(b.clip_masks.dtype, np.float16)

    def test_job_local_plan_matches_final_compile_with_all_relevant_declarations(self):
        b,a=_bundle(),_analysis()
        for kwargs in ({}, {'tone_core':'neutral','endpoint_mode':'fixed'},
                       {'adjustments':RenderAdjustments(midtone_contrast=.1),'chroma_nr':.2,'agx_primaries':'base'}):
            sink=[]
            result=auto_ev.compute_auto_ev(b,a,gamut='p3',_plan_sink=sink,**kwargs)
            exposed=with_intent_exposure(b,user_ev=result.ev,tone_core=kwargs.get('tone_core','agx'))
            final=build_render_plan(exposed,a,'agx','p3',**kwargs)
            self.assertEqual(len(sink),1)
            self.assertEqual(json_value(sink[0]),json_value(final))
            self.assertFalse(hasattr(result,'plan'))


if __name__ == '__main__':
    unittest.main()
