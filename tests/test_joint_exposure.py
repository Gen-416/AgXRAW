# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded response constraints, complete plan handoff and manual intent."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from dngscan.auto_ev import compute_auto_ev, resolve_export_ev
from dngscan.joint_exposure import select_joint_plan
from dngscan.models import RenderAdjustments
from dngscan.tone import build_render_plan
from tests.golden_support import build_night_sparse_lamps, build_daylight_wide_dr


class JointSelectionTests(unittest.TestCase):
    def setUp(self):
        self.scene = build_night_sparse_lamps()
        self.base = build_render_plan(self.scene.bundle, self.scene.analysis, 'agx', 'srgb')

    def test_missing_populations_remain_unmeasured_and_do_not_invent_semantics(self):
        def probe(plan, ev, indices):
            return np.full((512 if indices is None else len(indices), 3), .1)
        plan, ev, report = select_joint_plan(self.base, .1, 2., 'srgb', 512, probe)
        self.assertAlmostEqual(ev, .29)
        self.assertEqual(report['status'], 'selected')
        self.assertEqual(plan.tone.black_ev, self.base.tone.black_ev)
        self.assertEqual(plan.tone.toe_power, self.base.tone.toe_power)
        self.assertEqual(plan.tone.shoulder_power, self.base.tone.shoulder_power)
        self.assertEqual(plan.tone.white_ev, self.base.tone.white_ev + ev)
        for name in ['shadow', 'dark', 'bright']:
            self.assertIsNone(report['selected']['canonical_relations'][name+'_delta_EV_p05_p50_p95'])

    def test_highlight_failure_retains_neutral_plan_without_pass_claim(self):
        def probe(plan, ev, indices):
            return np.full((512 if indices is None else len(indices), 3), .95)
        plan, ev, report = select_joint_plan(self.base, .1, 2., 'srgb', 512, probe)
        self.assertIs(plan, self.base)
        self.assertEqual(ev, .1)
        self.assertEqual(report['status'], 'fallback')
        self.assertIsNone(report['selected'])
        self.assertTrue(all(not c['pass'] for c in report['candidates']))

    def test_canonical_recheck_can_reject_a_passing_proposal(self):
        def probe(plan, ev, indices):
            count = 220_001 if indices is None else len(indices)
            reference = plan is self.base
            return np.full((count, 3), .1 if reference or indices is not None else .95)
        plan, ev, report = select_joint_plan(self.base, .1, 2., 'srgb', 220_001, probe)
        self.assertEqual(report['status'], 'fallback')
        checked = [r for r in report['candidates'] if 'canonical_margin' in r]
        self.assertTrue(checked)
        self.assertTrue(all(r['proposal_margin'] >= 0 and r['canonical_margin'] < 0 for r in checked))
        self.assertIs(plan, self.base)

    def test_excessive_shadow_lift_is_rejected_even_without_highlight_risk(self):
        def probe(plan, ev, indices):
            count = 512 if indices is None else len(indices)
            return np.full((count, 3), .01 if plan is self.base else .02)
        plan, _, report = select_joint_plan(self.base, .1, 2., 'srgb', 512, probe)
        self.assertEqual(report['status'], 'fallback')
        self.assertIs(plan, self.base)
        self.assertTrue(any(r.get('proposal_margin', -1) >= 0 and not r['proposal_relations']['pass']
                            for r in report['candidates']))

    def test_zero_target_is_identity_without_claiming_gates_pass(self):
        with patch('dngscan.auto_ev.output_highlight_stats') as stats:
            plan, ev, report = select_joint_plan(self.base, 0., 0., 'srgb', 512, None)
        stats.assert_not_called()
        self.assertIs(plan, self.base)
        self.assertEqual(ev, 0.)
        self.assertEqual(report['status'], 'identity')
        self.assertIsNone(report['selected'])


class JointPipelineTests(unittest.TestCase):
    def test_preview_consumes_selected_curve_and_separates_same_ev_manual_pixels(self):
        from dngscan.gui import service
        from dngscan.gui.preview_cache import build_proxy_entry
        scene = build_night_sparse_lamps()
        entry = build_proxy_entry(scene.bundle, scene.analysis)
        auto = compute_auto_ev(entry.bundle, scene.analysis, 'srgb')
        self.assertEqual(auto.planning['status'], 'selected')
        arguments = dict(inp=scene.bundle.path, highlight='clip', gamut='srgb',
                         ev=auto.ev, quality=95, cached=entry, include_metrics=False)
        with patch.object(service, '_cached_render_plan', wraps=service._cached_render_plan) as compile_plan, \
             patch.object(service.dg, 'render_output_u8', wraps=service.dg.render_output_u8) as render:
            automatic = service.export_preview_jpeg(**arguments, auto_ev=auto)
            compile_plan.assert_not_called()
            self.assertIs(render.call_args.args[3], auto.render_plan)
            manual = service.export_preview_jpeg(**arguments)
            self.assertEqual(compile_plan.call_count, 1)
            self.assertEqual(render.call_count, 2)
            again = service.export_preview_jpeg(**arguments, auto_ev=auto)
            self.assertEqual(render.call_count, 2)
            self.assertTrue(again['cache_hit'])
        self.assertNotEqual(automatic['preview'], manual['preview'])
        self.assertEqual(automatic['ev_auto']['planning_status'], 'selected')

    def test_remaining_headroom_uses_the_actual_selected_curve(self):
        from dngscan.gui import service
        scene = build_night_sparse_lamps()
        auto = compute_auto_ev(scene.bundle, scene.analysis, 'srgb')
        with patch.object(service.dg, 'max_safe_ev', return_value=auto.ev) as safe:
            service.estimate_ev_headroom(scene.bundle, scene.analysis, 'srgb', auto.ev,
                                         tone_plan=auto.render_plan)
        self.assertIs(safe.call_args.kwargs['tone_plan'], auto.render_plan)

    def test_plan_sink_and_result_carry_the_exact_selected_curve(self):
        scene = build_night_sparse_lamps()
        sink = []
        original = scene.bundle.scene_rec2020_render.copy()
        auto = compute_auto_ev(scene.bundle, scene.analysis, 'srgb', _plan_sink=sink)
        self.assertIs(sink[0], auto.render_plan)
        self.assertEqual(auto.render_plan.tone.view_brightness, 1.)
        self.assertEqual(auto.planning['policy'], 'bounded-agx-2026-10-12-v1')
        self.assertGreaterEqual(auto.ev, auto.planning['anchor_ev'])
        self.assertLessEqual(auto.ev, auto.planning['anchor_ev']+.25)
        np.testing.assert_array_equal(scene.bundle.scene_rec2020_render, original)

    def test_proxy_reuses_full_source_plan_and_auto_result(self):
        from dngscan.gui.preview_cache import build_proxy_entry
        scene = build_daylight_wide_dr()
        full = compute_auto_ev(scene.bundle, scene.analysis, 'srgb')
        proxy = build_proxy_entry(scene.bundle, scene.analysis)
        small = compute_auto_ev(proxy.bundle, scene.analysis, 'srgb')
        self.assertEqual(full.ev, small.ev)
        self.assertEqual(full.render_plan, small.render_plan)
        self.assertEqual(full.planning, small.planning)

    def test_user_brightness_is_honoured_in_auto_and_manual_agx(self):
        scene = build_night_sparse_lamps()
        default = build_render_plan(scene.bundle, scene.analysis, 'agx', 'srgb')
        self.assertEqual(default.tone.view_brightness, 1.)
        adjustments = RenderAdjustments(midtone_brightness=.5)
        manual_plan = build_render_plan(scene.bundle, scene.analysis, 'agx', 'srgb', adjustments=adjustments)
        auto = compute_auto_ev(scene.bundle, scene.analysis, 'srgb', adjustments=adjustments)
        self.assertGreater(auto.render_plan.tone.view_brightness, 1.)
        self.assertEqual(auto.render_plan.tone.view_brightness, manual_plan.tone.view_brightness)
        self.assertEqual(resolve_export_ev(.75, scene.bundle, scene.analysis, 'srgb'), (.75, None))

    def test_other_cores_keep_fixed_endpoint_reference(self):
        scene = build_daylight_wide_dr()
        for core in ['lum', 'neutral', 'gated']:
            with self.subTest(core=core):
                base = build_render_plan(scene.bundle, scene.analysis, 'agx', 'srgb', tone_core=core)
                auto = compute_auto_ev(scene.bundle, scene.analysis, 'srgb', tone_core=core)
                self.assertIsNone(auto.planning)
                self.assertEqual(auto.render_plan, base)


if __name__ == '__main__':
    unittest.main()
