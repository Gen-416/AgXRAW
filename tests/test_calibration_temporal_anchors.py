# SPDX-License-Identifier: GPL-3.0-or-later
"""Paired PTC units, anchor provenance and disconnected gain domains."""
import copy
import csv
import io
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dngscan import calibration
from dngscan.calibration_ptc import fit_temporal, fit_channel, read_anchors, anchor_gain_graph


def write_csv(path, columns, rows, **header):
    stream = io.StringIO()
    for key,value in header.items():
        stream.write(f"#{key}: {value}\n")
    writer = csv.writer(stream)
    writer.writerow(columns)
    writer.writerows(rows)
    path.write_text(stream.getvalue())


def ptc_rows(gain=2., black=100., white=4095., factor=.8, drift=0.):
    signals = np.r_[np.geomspace(10.,1300.,20), [white-black,white-black]]
    means = signals+black
    spatial_var = signals/gain + 4. + .0009*signals**2
    sigma = np.sqrt(spatial_var)
    sigma[-2:] = 0.
    diff = np.sqrt((signals/gain+4.)*2.)
    return [dict(G1_Mean=str(m), G1_Std=str(s), G1_MeanB=str(m+d),
                 G1_StdDiff=str(v), G1_StdDiffClipped=str(v*math.sqrt(factor)), G1_ClipFrac='0')
            for m,s,v,d in zip(means,sigma,diff,drift*signals)]


class TemporalPTCTests(unittest.TestCase):
    def test_differential_factor_two_and_clip_correction_are_applied_once(self):
        rows = ptc_rows()
        result = fit_channel({'ClipVarianceFactor': '.8'},rows,'G1',100.,4095.)
        self.assertAlmostEqual(result['gain_e_per_dn'],2.,places=12)
        self.assertAlmostEqual(result['read_noise_dn'],2.,places=12)
        self.assertAlmostEqual(result['difference_variance_divisor'],1.6)
        self.assertEqual(result['variance_domain'],'stored-linearized-raw-dn2')
        self.assertEqual(result['prnu_status'],'excluded-by-pair-difference')
        self.assertIn('spatial_crosscheck',result)
        self.assertGreater(abs(result['spatial_temporal_gain_disagreement_relative']),.01)
        self.assertIn('conditional', result['fit_uncertainty_semantics'])
        self.assertLessEqual(result['gain_fit_interval_95'][0],2.)
        self.assertGreaterEqual(result['gain_fit_interval_95'][1],2.)

    def test_raw_difference_without_clip_factor_keeps_the_same_gain(self):
        rows = ptc_rows()
        result = fit_channel({},rows,'G1',100.,4095.)
        self.assertAlmostEqual(result['gain_e_per_dn'],2.,places=12)
        self.assertEqual(result['difference_variance_divisor'],2.)
        self.assertEqual(result['temporal_variance_source'],'G1_StdDiff')

    def test_missing_difference_data_keeps_explicit_spatial_fallback(self):
        rows = [{k:v for k,v in r.items() if 'Diff' not in k} for r in ptc_rows()]
        result = fit_channel({},rows,'G1',100.,4095.)
        self.assertNotIn('variance_domain',result)
        self.assertIn('unresolved',result['temporal_fallback_reason'])

    def test_unstable_mean_pairs_are_excluded_and_do_not_claim_temporal_evidence(self):
        rows = ptc_rows(drift=.03)
        result = fit_channel({'ClipVarianceFactor':'.8'},rows,'G1',100.,4095.)
        self.assertNotIn('variance_domain',result)
        self.assertIn('not enough stable',result['temporal_fallback_reason'])

    def test_one_unstable_pair_is_reported_without_biasing_other_pairs(self):
        rows = ptc_rows()
        rows[5]['G1_MeanB'] = str(float(rows[5]['G1_Mean'])+100)
        result = fit_channel({'ClipVarianceFactor':'.8'},rows,'G1',100.,4095.)
        self.assertEqual(result['pair_drift_rejected'],1)
        self.assertAlmostEqual(result['gain_e_per_dn'],2.,places=12)

    def test_differential_estimator_does_not_require_a_successful_spatial_fit(self):
        rows = ptc_rows()
        for row in rows:
            row['G1_Std'] = '0'
        result = fit_channel({'ClipVarianceFactor':'.8'},rows,'G1',100.,4095.)
        self.assertAlmostEqual(result['gain_e_per_dn'],2.,places=12)

    def test_paired_synthetic_flat_frames_remove_common_prnu(self):
        rng = np.random.default_rng(406)
        fixed = rng.normal(0.,.03,65536)
        rows = []
        bright_ratio = None
        for signal in np.geomspace(25.,1300.,18):
            mean_e = signal*2.*(1.+fixed)
            a = 100.+rng.poisson(mean_e)/2.+rng.normal(0.,2.,fixed.size)
            b = 100.+rng.poisson(mean_e)/2.+rng.normal(0.,2.,fixed.size)
            rows.append({'G1_Mean':str(a.mean()),'G1_MeanB':str(b.mean()),
                         'G1_Std':str(a.std()),'G1_StdDiff':str((a-b).std())})
            bright_ratio = a.var()/((a-b).var()/2.)
        result = fit_channel({},rows,'G1',100.,4095.)
        self.assertLess(abs(result['gain_e_per_dn']/2.-1.),.03)
        self.assertGreater(bright_ratio,2.)
        self.assertEqual(result['prnu_status'],'excluded-by-pair-difference')


class AllAnchorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def ptc(self,iso,gain=2.,suffix=''):
        rows = ptc_rows(gain=gain)
        path = self.root/f'ptc-iso{iso}{suffix}.csv'
        write_csv(path,list(rows[0]),[list(r.values()) for r in rows],
                  BlackLevel='100,100,100,100',ClipVarianceFactor=.8)
        return path

    def dark(self):
        columns = ['ISO','Channel','ColorIndex','BlackA','BlackB','StdDiffClipped']
        rows = [[iso,key,cid,100,100,math.sqrt(8.)] for iso in (100,200,400,800)
                for key,cid in [('C00',2),('C01',0),('C10',3),('C11',1)]]
        write_csv(self.root/'dark-scalars.csv',columns,rows,Camera='SONY Test',
                  CfaPattern='GBRG',AdcStep=0,ClipVarianceFactor=1,RawSize='128x128',ShutterType='mechanical')

    def test_unusable_first_ptc_does_not_hide_later_valid_anchor(self):
        self.ptc(100).write_text('broken')
        good = self.ptc(200)
        records = read_anchors(self.root,{})
        self.assertEqual([r['status'] for r in records],['rejected','usable'])
        self.assertEqual(calibration.ptc_anchor(self.root,{})[0],200)
        self.assertEqual(records[1]['sha256'],calibration._sha256(good))

    def test_all_anchors_and_hashes_survive_collect_and_runtime_validation(self):
        self.dark()
        first,second = self.ptc(100,2.),self.ptc(400,.5)
        columns = ['ISO','Channel','ColorIndex','ClipFrac','Mean','ShutterSec','ShutterGroup']
        rows = [[iso,'C01',0,0,mean,1,'A'] for iso,mean in [(100,600),(200,1100),(400,2100)]]
        write_csv(self.root/'gain-levels.csv',columns,rows,Ladder='paired-shutter',CfaPattern='GBRG')
        item = calibration.build(self.root,None)
        self.assertEqual(len(item['ptc_anchors']),2)
        self.assertTrue(all(p.name in item['source']['inputs'] for p in (first,second)))
        self.assertFalse(item['ptc_anchor_diagnostics']['conflicts'])
        runtime = calibration._validated_prior(item)
        self.assertEqual(len(runtime['ptc_anchors']),2)
        self.assertEqual(runtime['gain_support_intervals'],[[100.,400.]])
        self.assertAlmostEqual(2**calibration.curve_value(runtime,'gain_log2iso_log2epd',200),1.)
        # Actual mapped G1 is C01 (green id 0), not fixed ColorIndex 1.
        self.assertEqual(item['phase_calibration']['C01']['color'],'G')
        self.assertEqual(item['phase_calibration']['C01']['gain_provenance'],'independent-phase-ptc')
        self.assertNotIn('gain_provenance',item['phase_calibration']['C10'])

    def test_disconnected_components_require_independent_anchor_and_keep_a_gap(self):
        records = [{'iso':100,'file':'a','status':'usable','fit':{'gain_e_per_dn':2.}},
                   {'iso':400,'file':'b','status':'usable','fit':{'gain_e_per_dn':.4}}]
        components = [{'isos':[100,200],'relative_gain':{100:1.,200:.5}},
                      {'isos':[400,800],'relative_gain':{400:1.,800:.5}},
                      {'isos':[1600,3200],'relative_gain':{1600:1.,3200:.5}}]
        _,curve,intervals,conflicts = anchor_gain_graph(records,components)
        self.assertEqual(sorted(curve),[100,200,400,800])
        self.assertEqual(intervals,[(100,200),(400,800)])
        self.assertEqual(components[2]['anchor_status'],'unanchored')
        runtime = {'gain_log2iso_log2epd':[[math.log2(i),math.log2(g)] for i,g in curve.items()],
                   'gain_support_intervals':intervals}
        self.assertIsNone(calibration.curve_value(runtime,'gain_log2iso_log2epd',300))
        self.assertIsNotNone(calibration.curve_value(runtime,'gain_log2iso_log2epd',150))
        self.assertFalse(conflicts)

    def test_duplicate_and_conflicting_anchors_are_order_independent_and_reported(self):
        records = [{'iso':100,'file':'a','status':'usable','fit':{'gain_e_per_dn':2.}},
                   {'iso':100,'file':'b','status':'usable','fit':{'gain_e_per_dn':3.}}]
        first = anchor_gain_graph(copy.deepcopy(records),[])
        second = anchor_gain_graph(copy.deepcopy(records[::-1]),[])
        self.assertEqual(first[1],second[1])
        self.assertAlmostEqual(first[1][100],math.sqrt(6.))
        self.assertTrue(first[3])

    def test_anchor_outside_ladder_remains_singleton_without_extrapolation(self):
        records = [{'iso':800,'file':'a','status':'usable','fit':{'gain_e_per_dn':.25}}]
        _,curve,intervals,_ = anchor_gain_graph(records,[{'isos':[100,200],'relative_gain':{100:1.,200:.5}}])
        self.assertEqual(curve,{800:.25})
        self.assertEqual(intervals,[(800,800)])

    def test_known_gain_jump_is_not_used_as_an_interpolation_interval(self):
        rel = {50:2.,100:1.,200:.5,400:.5,800:.25}
        records = [{'iso':100,'file':'a','status':'usable','fit':{'gain_e_per_dn':2.}},
                   {'iso':400,'file':'b','status':'usable','fit':{'gain_e_per_dn':1.}}]
        _,curve,intervals,conflicts = anchor_gain_graph(records,[{'isos':sorted(rel),'relative_gain':rel}])
        self.assertEqual(intervals,[(50,200),(400,800)])
        entry = {'gain_log2iso_log2epd':[[math.log2(i),math.log2(v)] for i,v in curve.items()],
                 'gain_support_intervals':intervals,'gain_jump_isos':[400]}
        self.assertIsNone(calibration.curve_value(entry,'gain_log2iso_log2epd',300))
        self.assertAlmostEqual(2**calibration.curve_value(entry,'gain_log2iso_log2epd',100),2.)
        self.assertFalse(conflicts)

    def test_interleaved_disconnected_domains_cannot_mix_runtime_interpolation(self):
        records = [{'iso':100,'file':'a','status':'usable','fit':{'gain_e_per_dn':2.}},
                   {'iso':200,'file':'b','status':'usable','fit':{'gain_e_per_dn':3.}}]
        components = [{'isos':[100,400],'relative_gain':{100:1.,400:.25}},
                      {'isos':[200,800],'relative_gain':{200:1.,800:.25}}]
        _,curve,intervals,_ = anchor_gain_graph(records,components)
        self.assertEqual(intervals,[(100,100),(200,200),(400,400),(800,800)])
        entry = {'gain_log2iso_log2epd':[[math.log2(i),math.log2(v)] for i,v in curve.items()],
                 'gain_support_intervals':intervals}
        self.assertIsNone(calibration.curve_value(entry,'gain_log2iso_log2epd',150))


class PhaseCalibrationContractTests(unittest.TestCase):
    def test_variance_pooling_and_arbitrary_colour_description(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'dark.csv'
            write_csv(path,['ISO','Channel','ColorIndex','BlackA','BlackB','StdDiffClipped'],
                [[100,'C01',0,10,10,math.sqrt(2.)],[100,'C01',0,10,10,math.sqrt(18.)],
                 [100,'C10',3,20,20,math.sqrt(50.)],[100,'C00',2,30,30,100]],
                CfaPattern='GBRG',ClipVarianceFactor=1,AdcStep=0)
            _,dark = calibration.read_dark(path)
            self.assertAlmostEqual(dark[100]['rn_dn'],math.sqrt((1+9+25)/3))
            self.assertAlmostEqual(dark[100]['phases']['C01']['rn_dn'],math.sqrt(5.))
            self.assertEqual(dark[100]['phases']['C01']['color'],'G')

    def test_phase_validator_retains_failure_barriers_and_rejects_identity_or_units(self):
        product = {'channel':'C01','color_index':0,'color_desc':'GBRG','color':'G',
            'read_noise_dn_log2iso':[[math.log2(100),2.],[math.log2(400),1.]],
            'stored_dark_variance_dn2_log2iso':[[math.log2(100),4.],[math.log2(400),1.]],
            'read_noise_unresolved_isos':[200]}
        item = {'phase_calibration':{'C01':product}}
        valid = calibration.phase_calibration_fields(item)['phase_calibration']['C01']
        self.assertEqual(valid['read_noise_unresolved_isos'],[200.])
        self.assertIsNone(calibration.curve_value(valid,'read_noise_dn_log2iso',200))
        bad = copy.deepcopy(item); bad['phase_calibration']['C01']['color']='R'
        with self.assertRaisesRegex(ValueError,'colour'):
            calibration.phase_calibration_fields(bad)
        bad = copy.deepcopy(item); bad['phase_calibration']['C01']['read_noise_dn_log2iso'][0][1]=3.
        with self.assertRaisesRegex(ValueError,'below physical'):
            calibration.phase_calibration_fields(bad)


if __name__ == '__main__':
    unittest.main()
