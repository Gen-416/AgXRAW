# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise reference capability failures through the real load/analyze/planner seams."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dngscan.analysis import analyze
from dngscan.hdr_agx_plan import compile_hdr_agx_plan
from dngscan.raw_io import load_raw
from dngscan.tone import build_render_plan

SAMPLE = Path.home() / 'Pictures' / 'AgXRAW样张' / '_SDI0150.DNG'


def _decoded(value):
    return (np.full((100, 100, 3), value, dtype=np.float32),
            {'version': '9', 'baseline_exposure_authored': 0.,
             'baseline_exposure_cleared': True})


@unittest.skipUnless(SAMPLE.is_file(), 'RAW evidence fixture unavailable')
class ReferenceLoadCapabilityTests(unittest.TestCase):
    def test_optional_reference_failure_retains_sensor_analysis_and_bounded_hdr(self):
        with patch('dngscan.coreimage_decode.runtime_available', return_value=True), \
             patch('dngscan.coreimage_decode.decode_scene_rec2020', return_value=_decoded(4.)), \
             patch('dngscan.coreimage_decode.read_dng_opcodes', return_value={'names': ()}), \
             patch('dngscan.raw_io._decode_corrected_libraw', side_effect=RuntimeError('injected unsupported correction')):
            bundle = load_raw(SAMPLE, decoder='coreimage', scene_half_size=True)
        self.assertIsNotNone(bundle.evidence)
        self.assertIsNotNone(bundle.raw_image)
        self.assertIsNone(bundle.scene_reliable_reference_rec2020)
        self.assertIsNone(bundle.scene_processing_loss_pct)
        self.assertIn('injected unsupported correction', bundle.scene_reference_error)
        self.assertEqual(bundle.scene_reliability_source, 'decoded-image-estimate')
        analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
        self.assertTrue(analysis.channel_ids)
        self.assertTrue(math.isfinite(analysis.cell_union_pct))
        hdr = compile_hdr_agx_plan(build_render_plan(bundle, analysis, 'agx', 'p3'),
                                   analysis=analysis, scene_decoder='coreimage')
        self.assertGreater(hdr.tone.rendered_headroom_ev, 0.)
        self.assertLessEqual(hdr.tone.rendered_headroom_ev, 1.)
        self.assertEqual(hdr.color.channel_separation, 0.)

    def test_successfully_empty_reference_does_not_silently_become_estimate(self):
        reference = np.full((100, 100, 3), 4. * 65535., dtype=np.float32)
        with patch('dngscan.coreimage_decode.runtime_available', return_value=True), \
             patch('dngscan.coreimage_decode.decode_scene_rec2020', return_value=_decoded(4.)), \
             patch('dngscan.raw_io.libraw_scene_scale', return_value=65535.), \
             patch('dngscan.raw_io._decode_corrected_libraw', return_value=(reference, None, SimpleNamespace(post=(), crop=None), None)), \
             patch('dngscan.scene_reference.reliable_reference_samples', return_value=(np.empty((0, 3), dtype=np.float32), 0.)):
            bundle = load_raw(SAMPLE, decoder='coreimage', scene_half_size=True)
        self.assertEqual(bundle.scene_reliability_source, 'sensor-reference')
        self.assertEqual(bundle.scene_reliable_reference_rec2020.shape, (0, 3))
        analysis, _, _ = analyze(bundle, margin=4, diagnostics=False)
        hdr = compile_hdr_agx_plan(build_render_plan(bundle, analysis, 'agx', 'p3'),
                                   analysis=analysis, scene_decoder='coreimage')
        self.assertTrue(math.isnan(hdr.tone.reliable_tail_ev))
        self.assertEqual(hdr.tone.rendered_headroom_ev, 0.)

    def test_reference_units_track_both_aligned_and_unity_scene_scale(self):
        for mode in ('aligned', 'unity'):
            with self.subTest(mode=mode):
                reference = np.full((100, 100, 3), .2 * 65535., dtype=np.float32)
                samples = np.full((1000, 3), .2, dtype=np.float32)
                with patch('dngscan.coreimage_decode.runtime_available', return_value=True), \
                     patch('dngscan.coreimage_decode.decode_scene_rec2020', return_value=_decoded(.4)), \
                     patch('dngscan.raw_io.libraw_scene_scale', return_value=65535.), \
                     patch('dngscan.raw_io._decode_corrected_libraw', return_value=(reference, None, SimpleNamespace(post=(), crop=None), None)), \
                     patch('dngscan.scene_reference.reliable_reference_samples', return_value=(samples, 100.)):
                    bundle = load_raw(SAMPLE, decoder='coreimage', scene_half_size=True, coreimage_scale=mode)
                self.assertEqual(bundle.scene_reliability_source, 'sensor-reference')
                scene_y = float(bundle.scene_rec2020_render[0, 0, 1]) / bundle.scene_scale
                self.assertAlmostEqual(float(bundle.scene_reliable_reference_rec2020[0, 1]), scene_y, places=6)


if __name__ == '__main__':
    unittest.main()
