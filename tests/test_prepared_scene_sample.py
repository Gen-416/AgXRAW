# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical source rows retain nested sampling, evidence and EV operation order."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dngscan import auto_ev, tone, retreat
from dngscan.prepared_sample import PreparedSceneSample
from dngscan.sampling import sample_indices
from tests.test_preview_cache import _bundle, _analysis
from tools.benchmark_loss_pipeline import json_value


class PreparedSceneSampleTests(unittest.TestCase):
    def test_auto_ev_prepares_only_full_source_nonfilm_nongated(self):
        for options in ({}, {'tone_core': 'gated'}, {'film_curve': 'velvia100'},
                        {'film_mode': 'full'}, {'proxy': True}):
            b, a = _bundle(), _analysis()
            kwargs = dict(options)
            if kwargs.pop('proxy', False):
                b._tone_plan_sample = b.scene_rec2020_render.reshape(-1, 3)
            plan = SimpleNamespace(scene=SimpleNamespace(body_ev_p50=0.))
            with patch.object(PreparedSceneSample, 'from_bundle', wraps=PreparedSceneSample.from_bundle) as prepare, \
                 patch.object(auto_ev, 'build_render_plan', return_value=plan), \
                 patch.object(auto_ev, 'max_safe_ev', return_value=1.):
                auto_ev.compute_auto_ev(b, a, 'p3', **kwargs)
            self.assertEqual(prepare.call_count, 1 if not options else 0)

    def test_nested_population_masks_and_input_ownership(self):
        # Cross the canonical cap: the 220k probe must sample *within* the
        # 800k population, never choose 220k different full-image locations.
        b = _bundle()
        shape = (801, 1001, 3)
        b.scene_rec2020_render = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        b.clip_masks = (b.scene_rec2020_render % 7 / 6).astype(np.float16)
        prepared = PreparedSceneSample.from_bundle(b)
        indices = sample_indices(shape[0] * shape[1])
        selected = sample_indices(len(indices), 220_000)
        np.testing.assert_array_equal(prepared.source_indices, indices)
        self.assertFalse(np.array_equal(indices[selected], sample_indices(shape[0] * shape[1], 220_000)))
        np.testing.assert_array_equal(prepared.rgb[selected], b.scene_rec2020_render.reshape(-1, 3)[indices[selected]])
        np.testing.assert_array_equal(prepared.masks[selected], b.clip_masks.reshape(-1, 3)[indices[selected]].astype(np.float32))
        self.assertFalse(prepared.rgb.flags.writeable)
        self.assertFalse(prepared.masks.flags.writeable)
        self.assertTrue(b.scene_rec2020_render.flags.writeable)
        self.assertIsNone(b._tone_plan_sample)
        self.assertIsNone(b._clip_masks_resized)

    def test_proxy_reuses_full_source_rows_and_absent_masks(self):
        b = _bundle()
        b._tone_plan_sample = np.ones((71, 3), np.float16)
        b._tone_plan_sample_masks = None
        sample = PreparedSceneSample.from_bundle(b)
        self.assertIs(sample.rgb, b._tone_plan_sample)
        self.assertIsNone(sample.masks)
        self.assertIsNone(sample.source_indices)
        self.assertIs(sample.bind(b)._tone_plan_sample, sample.rgb)

    def test_resized_masks_and_reliability_threshold_keep_old_geometry(self):
        from dngscan.tone import reliable_scene_ev_selection
        for dtype in (np.float16, np.float32):
            for shape in ((3, 5, 3), (8, 8, 3)):
                b = _bundle()
                values = np.array([0., .0999, .1, .1001, 1., np.nan], dtype=dtype)
                b.clip_masks = np.resize(values, shape)
                indices = sample_indices(np.prod(b.scene_rec2020_render.shape[:2]))
                expected = retreat.clip_masks_for_render(b, b.scene_rec2020_render.shape[:2])[indices]
                prepared = PreparedSceneSample.from_bundle(b)
                self.assertEqual(prepared.masks.tobytes(), expected.tobytes())
                old = reliable_scene_ev_selection(b, _analysis())
                new = reliable_scene_ev_selection(prepared.bind(b), _analysis())
                for actual, reference in zip(new[:3], old[:3]):
                    self.assertEqual(actual.tobytes(), reference.tobytes())
                self.assertEqual(new[3], old[3])
        b.clip_masks = None
        self.assertIsNone(PreparedSceneSample.from_bundle(b).masks)

    def test_plan_and_probe_match_at_each_ev_and_storage_type(self):
        a = _analysis()
        rng = np.random.default_rng(346)
        for dtype in (np.float16, np.float32, np.uint16):
            b = _bundle()
            b.scene_rec2020_render = rng.uniform(0, 200, b.scene_rec2020_render.shape).astype(dtype)
            p = PreparedSceneSample.from_bundle(b).bind(b)
            for transform in ('none', 'chroma'):
                # Use an actual supported transform from the declared choices.
                if transform == 'chroma':
                    from dngscan.scene_transform import SCENE_TRANSFORM_CHOICES
                    choices = [x for x in SCENE_TRANSFORM_CHOICES if x != 'none']
                    if not choices:
                        continue
                    transform = choices[0]
                with self.subTest(dtype=dtype, transform=transform):
                    kwargs = dict(scene_transform=transform)
                    plan = tone.build_render_plan(b, a, 'agx', 'p3', **kwargs)
                    actual_plan = tone.build_render_plan(p, a, 'agx', 'p3', **kwargs)
                    self.assertEqual(json_value(plan), json_value(actual_plan))
                    captured = []
                    original = auto_ev.render_sample_linear_output
                    def observed(*args, **kw):
                        out = original(*args, **kw)
                        captured.append(out.tobytes())
                        return out
                    with patch.object(auto_ev, 'render_sample_linear_output', side_effect=observed):
                        expected = auto_ev.max_safe_ev(b, a, 'p3', tone_plan=plan, **kwargs)
                        old_outputs = list(captured)
                        captured.clear()
                        actual = auto_ev.max_safe_ev(p, a, 'p3', tone_plan=actual_plan, **kwargs)
                    self.assertEqual(actual, expected)
                    self.assertEqual(captured, old_outputs)


if __name__ == '__main__':
    unittest.main()
