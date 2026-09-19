# SPDX-License-Identifier: GPL-3.0-or-later
"""Sensor topology and unavailable-evidence contracts at analysis consumers."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from dngscan.analysis import analyze, compute_cell_metrics, compute_color_clip_metrics
from dngscan.color import rec2020_to_xyz
from dngscan.hdr_agx_plan import compile_channel_separation
from dngscan.models import RawBundle


class ColorGroupClipTests(unittest.TestCase):
    def test_green_pair_is_one_lost_colour_like_red_or_blue(self):
        colors = np.tile(np.asarray([[0, 1], [3, 2]], dtype=np.uint8), (10, 10))
        thresholds = {k: 1000 for k in range(4)}
        labels = {0: 'R', 1: 'G1', 2: 'B', 3: 'G2'}
        for ids in ((0,), (2,), (1, 3)):
            with self.subTest(ids=ids):
                raw = np.zeros(colors.shape, np.uint16)
                raw[:2][np.isin(colors[:2], ids)] = 1000
                old = compute_cell_metrics(raw, colors, thresholds)[3]
                groups = compute_color_clip_metrics(raw, colors, thresholds, labels, [[0, 1], [3, 2]])
                self.assertEqual(groups, {1: 10.0, 2: 0.0, 3: 0.0})
                self.assertEqual(old[2 if len(ids) == 2 else 1], 10.0)
                analysis = SimpleNamespace(cell_k_of_all_pct=old,
                    color_clip_k_of_all_pct=groups, gamut_out_pct={})
                self.assertEqual(compile_channel_separation(analysis), 0.5)

    def test_two_actual_lost_colours_withdraw_channel_separation(self):
        colors = np.tile(np.asarray([[0, 1], [3, 2]], dtype=np.uint8), (10, 10))
        raw = np.zeros(colors.shape, np.uint16)
        raw[:2][np.isin(colors[:2], (0, 2))] = 1000
        groups = compute_color_clip_metrics(raw, colors, {k: 1000 for k in range(4)},
                    {0: 'R', 1: 'G1', 2: 'B', 3: 'G2'}, [[0, 1], [3, 2]])
        self.assertEqual(groups, {1: 0.0, 2: 10.0, 3: 0.0})
        self.assertEqual(compile_channel_separation(SimpleNamespace(
            color_clip_k_of_all_pct=groups, gamut_out_pct={})), 0.0)

    def test_linear_rgb_and_larger_cfa_period_count_rgb_groups(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint16)
        rgb[0, :, :2] = 1000
        colors = np.broadcast_to(np.arange(3, dtype=np.uint8), rgb.shape)
        actual = compute_color_clip_metrics(rgb, colors, {k: 1000 for k in range(3)},
                                           {0: 'R', 1: 'G', 2: 'B'}, [])
        self.assertEqual(actual, {1: 0.0, 2: 10.0, 3: 0.0})
        pattern = np.tile(np.asarray([[0, 1], [1, 2]], dtype=np.uint8), (3, 3))
        colors = np.tile(pattern, (2, 2))
        raw = np.zeros(colors.shape, dtype=np.uint16)
        raw[:6, :6][colors[:6, :6] == 1] = 1000
        actual = compute_color_clip_metrics(raw, colors, {k: 1000 for k in range(3)},
                                           {0: 'R', 1: 'G', 2: 'B'}, pattern.tolist())
        self.assertEqual(actual, {1: 25.0, 2: 0.0, 3: 0.0})

    def test_old_sensel_counts_never_stand_in_for_missing_colour_evidence(self):
        analysis = SimpleNamespace(cell_k_of_all_pct={1: 0., 2: 0., 3: 0., 4: 0.},
                                   gamut_out_pct={})
        self.assertEqual(compile_channel_separation(analysis), 0.0)


class EvidenceReportingTests(unittest.TestCase):
    def test_green_pair_is_not_reported_as_multiple_lost_colours(self):
        from dngscan.report import darktable_guidance_lines
        analysis = SimpleNamespace(noise_evidence_status='independent', clip_pct={1: 10., 3: 10.},
            ev_p999=0., cfa_cell_supported=True, cell_union_pct=10., cell_ge2_of_clipped_pct=100.,
            color_clip_k_of_all_pct={1: 10., 2: 0., 3: 0.}, snr1_dr={}, gamut_out_pct={},
            channel_ids=[1, 3], labels={1: 'G1', 3: 'G2'})
        report = '\n'.join(darktable_guidance_lines(SimpleNamespace(camera_wb=[1.] * 4), analysis))
        self.assertIn('单个 RGB 颜色组', report)
        self.assertNotIn('细节修复有限', report)

    def test_apple_only_matrix_report_does_not_claim_fallback_matrix_was_applied(self):
        from unittest.mock import patch
        from dngscan.report import matrix_health_line_cn
        bundle = SimpleNamespace(scene_decoder='coreimage', evidence_provider='unavailable',
                                 wb_mode='camera')
        with patch('dngscan.raw_io.resolve_hot_wb_c0', side_effect=AssertionError('not a measured Apple matrix')):
            report = matrix_health_line_cn(bundle)
        self.assertIn('Apple 内部标定', report)
        self.assertNotIn('κ=', report)


class AppleOnlyAnalysisTests(unittest.TestCase):
    def test_absent_sensor_measurements_are_explicit_but_scene_metrics_work(self):
        scene = np.linspace(.01, 2., 100 * 100 * 3, dtype=np.float32).reshape(100, 100, 3)
        bundle = RawBundle(path=Path('apple-only.raw'), raw_image=None, raw_colors=None,
            xyz_render=rec2020_to_xyz(scene.reshape(-1, 3)).reshape(scene.shape), render_scale=1., scene_rec2020_render=scene,
            scene_scale=1., white_level=0, black_levels=[], camera_wb=[], color_desc='',
            raw_pattern=[], camera_white_levels=[], evidence_provider='unavailable',
            scene_decoder='coreimage')
        analysis, y, ev = analyze(bundle, margin=4, diagnostics=False)
        self.assertTrue(math.isfinite(analysis.ev_median))
        self.assertEqual(y.shape, scene.shape[:2])
        self.assertTrue(np.isfinite(ev).all())
        self.assertEqual(analysis.noise_evidence_status, 'unavailable')
        self.assertEqual(analysis.channel_ids, [])
        self.assertEqual(analysis.color_clip_k_of_all_pct, {})
        self.assertEqual(analysis.snr_curves, {})
        self.assertIsNone(analysis.container_bits_est)
        for name in ('cell_union_pct', 'fullwell', 'threshold', 'noise_floor', 'usable_dr_ev'):
            self.assertTrue(math.isnan(getattr(analysis, name)), name)
        self.assertEqual(compile_channel_separation(analysis, 'coreimage'), 0.)
        from dngscan.auto_ev import compute_auto_ev
        from dngscan.tone import build_render_plan
        result = compute_auto_ev(bundle, analysis)
        self.assertTrue(math.isfinite(result.ev))
        self.assertTrue(math.isfinite(result.highlight_cap_ev))
        plan = build_render_plan(bundle, analysis, 'agx', 'p3', endpoint_mode='evidence')
        self.assertIn('非传感器实测', plan.tone.endpoint_note)
        self.assertTrue(math.isnan(analysis.noise_floor))


class NonBayerFullyClippedTests(unittest.TestCase):
    def test_full_clipping_vetoes_image_estimate_on_xtrans_and_linear_rgb(self):
        from dngscan.tone import scene_tone_metrics
        for kind in ('xtrans', 'linear-rgb'):
            with self.subTest(kind=kind):
                if kind == 'xtrans':
                    pattern = np.tile(np.asarray([[0, 1], [1, 2]], dtype=np.uint8), (3, 3))
                    colors = np.tile(pattern, (20, 20))
                    raw = np.full(colors.shape, 1000, dtype=np.uint16)
                else:
                    pattern = np.asarray([], dtype=np.uint8)
                    raw = np.full((120, 120, 3), 1000, dtype=np.uint16)
                    colors = np.broadcast_to(np.arange(3, dtype=np.uint8), raw.shape)
                scene = np.full((60, 60, 3), 4., dtype=np.float32)
                bundle = RawBundle(path=Path('all-clipped.raw'), raw_image=raw, raw_colors=colors,
                    xyz_render=rec2020_to_xyz(scene.reshape(-1, 3)).reshape(scene.shape),
                    render_scale=1., scene_rec2020_render=scene, scene_scale=1.,
                    white_level=1000, black_levels=[0.] * 3, camera_wb=[1.] * 3,
                    color_desc='RGB', raw_pattern=pattern.tolist(), camera_white_levels=[1000.] * 3,
                    scene_decoder='coreimage', scene_reliable_reference_rec2020=None)
                analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
                self.assertEqual(analysis.cell_union_pct, 100.)
                self.assertEqual(analysis.color_clip_k_of_all_pct[3], 100.)
                self.assertTrue(math.isnan(scene_tone_metrics(bundle, analysis).reliable_tail_ev_p9999))
                # Old/capability-limited analyses may lack the union metric,
                # but a complete measured RGB-group count still proves loss.
                analysis.cell_union_pct = float('nan')
                self.assertTrue(math.isnan(scene_tone_metrics(bundle, analysis).reliable_tail_ev_p9999))


if __name__ == '__main__':
    unittest.main()
