# SPDX-License-Identifier: GPL-3.0-or-later
"""HDR HEIF keeps its nonlinear SDR master until the 10-bit boundary."""
from contextlib import ExitStack
from dataclasses import replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import hdr_agx
from dngscan.delivery import FinishedPair, resolve_delivery_profile
from dngscan.export import export_ultrahdr_jpeg
from dngscan.hdr_agx_plan import compile_hdr_agx_plan
from dngscan.render import render_output_encoded_float
from dngscan.tone import build_render_plan


class HdrFloatBaseFormationTests(unittest.TestCase):
    def test_float_base_matches_standalone_master_and_hdr_packing_is_unchanged(self):
        from dngscan import _fast
        from tests.golden_support import build_staggered_clip

        scene = build_staggered_clip()
        bundle = replace(scene.bundle,
                         scene_rec2020_render=scene.bundle.scene_rec2020_render[:7, :19],
                         clip_masks=None)
        before = bundle.scene_rec2020_render.copy()
        plan = build_render_plan(bundle, scene.analysis, 'agx', 'p3')
        hdr_plan = compile_hdr_agx_plan(plan, analysis=scene.analysis)
        for fast in ('0', '1'):
            if fast == '1' and not _fast.available():
                continue
            for chunk in (20, 40):
                with self.subTest(fast=fast, chunk=chunk), \
                        patch.dict(os.environ, DNGSCAN_FAST=fast), \
                        patch.multiple(hdr_agx, STREAM_RENDER_CHUNK=chunk,
                                       STREAM_QUANTIZE_CHUNK=40, STREAM_THREAD_MIN_PIXELS=80):
                    legacy, legacy_hdr, legacy_usage = hdr_agx.render_ultrahdr_agx_pair_packed(
                        bundle, scene.analysis, plan, hdr_plan)
                    reference = render_output_encoded_float(bundle, scene.analysis, 'p3', plan)
                    with patch.object(hdr_agx, 'generate_dither_noise',
                                      side_effect=AssertionError('float master must not be quantized')), \
                            patch('dngscan.render._prepare_chroma_nr_map', return_value=None) as chroma, \
                            patch.object(hdr_agx, 'scene_intent_rec2020',
                                         wraps=hdr_agx.scene_intent_rec2020) as intent:
                        floating, alternate, usage = hdr_agx.render_ultrahdr_agx_pair_packed(
                            bundle, scene.analysis, plan, hdr_plan, sdr_float=True)
                    self.assertEqual(floating.dtype, np.float32)
                    self.assertEqual(legacy.dtype, np.uint8)
                    np.testing.assert_allclose(floating, reference, atol=2e-7, rtol=0)
                    np.testing.assert_array_equal(alternate, legacy_hdr)
                    self.assertEqual(usage, legacy_usage)
                    self.assertEqual(chroma.call_count, 1)
                    self.assertEqual(intent.call_count, (133 + chunk - 1) // chunk)
                    self.assertTrue(np.isfinite(floating).all())
                    self.assertGreaterEqual(float(floating.min()), 0.)
                    self.assertLessEqual(float(floating.max()), 1.)
                    np.testing.assert_array_equal(bundle.scene_rec2020_render, before)

    def test_scene_gradient_reaches_more_than_256_base_codes(self):
        from tests.golden_support import build_daylight_wide_dr

        scene = build_daylight_wide_dr()
        pixels = np.broadcast_to(np.linspace(0, 65535, 4096, dtype=np.uint16)[None, :, None],
                                 (8, 4096, 3)).copy()
        bundle = replace(scene.bundle, scene_rec2020_render=pixels, clip_masks=None)
        plan = build_render_plan(bundle, scene.analysis, 'agx', 'p3')
        hdr_plan = compile_hdr_agx_plan(plan, analysis=scene.analysis)
        master, alternate, _ = hdr_agx.render_ultrahdr_agx_pair_packed(
            bundle, scene.analysis, plan, hdr_plan, sdr_float=True)
        self.assertGreater(np.unique(np.rint(master[..., 1] * 1023)).size, 256)
        self.assertEqual(alternate.dtype, np.float16)
        self.assertEqual(alternate.shape, (8, 4096, 4))


class FinishedPairPrecisionTests(unittest.TestCase):
    def test_legacy_constructor_and_float_master_are_unambiguous(self):
        byte = np.zeros((8, 8, 3), np.uint8)
        floating = np.full((8, 8, 3), .5, np.float32)
        hdr = np.ones((8, 8, 4), np.float16)
        legacy = FinishedPair(byte, hdr, 2.)
        precise = FinishedPair(None, hdr, 2., sdr_rgb_float=floating)
        self.assertIs(legacy.sdr_rgb, byte)
        self.assertIs(legacy.sdr_rgb_u8, byte)
        self.assertEqual(legacy.sdr_precision, 'uint8')
        self.assertIs(precise.sdr_rgb, floating)
        self.assertIsNone(precise.sdr_rgb_u8)
        self.assertEqual(precise.sdr_precision, 'float32')
        for byte_master, float_master in ((None, None), (byte, floating),
                                         (floating, None), (None, byte)):
            with self.subTest(byte=byte_master is not None, floating=float_master is not None), \
                    self.assertRaises(ValueError):
                FinishedPair(byte_master, hdr, 2., sdr_rgb_float=float_master)

    def test_export_routes_10bit_heif_float_and_keeps_jpeg_and_8bit_bytes(self):
        from tests.test_hdr_native import _scene_plan

        tone = SimpleNamespace(rendered_headroom_ev=2., peak_linear=4., display_headroom_ev=2.,
                               requested_headroom_ev=2., reliable_tail_ev=3., shoulder_start_ev=1.,
                               white_ev=2., shoulder_alpha=.5, shoulder_segments=())
        hdr_plan = SimpleNamespace(tone=tone, color=SimpleNamespace(channel_separation=.5))
        byte = np.zeros((8, 8, 3), np.uint8)
        floating = np.full((8, 8, 3), .5, np.float32)
        alternate = np.ones((8, 8, 4), np.float16)
        with TemporaryDirectory() as td:
            for name in ('auto', 'share', 'archive'):
                for container, depth in (('jpeg', 10), ('heic', 8), ('heic', 10)):
                    precise = container == 'heic' and depth == 10
                    profile = replace(resolve_delivery_profile(name, container=container), heif_bit_depth=depth)
                    with self.subTest(profile=name, container=container, depth=depth), ExitStack() as stack:
                        stack.enter_context(patch('dngscan.export.apple_gainmap_backend_status', return_value=(True, '')))
                        stack.enter_context(patch('dngscan.hdr_agx_plan.compile_hdr_agx_plan', return_value=hdr_plan))
                        stack.enter_context(patch('dngscan.hdr_agx_plan.describe_hdr_plan', return_value='plan'))
                        render = stack.enter_context(patch('dngscan.hdr_agx.render_ultrahdr_agx_pair_packed',
                                                           return_value=(floating if precise else byte, alternate, 1.)))
                        def encode(pair, path, delivered):
                            self.assertEqual(pair.sdr_precision, 'float32' if precise else 'uint8')
                            self.assertIs(pair.sdr_rgb, floating if precise else byte)
                            self.assertIs(pair.hdr_rgba_f16, alternate)
                            path.write_bytes(b'verified')
                            return {}
                        stack.enter_context(patch('dngscan.export.encode_finished_pair', side_effect=encode))
                        stack.enter_context(patch('dngscan.export.carry_capture_metadata_hdr', return_value=False))
                        export_ultrahdr_jpeg(Path('source.dng'), Path(td) / 'photo.heic', profile.quality,
                                            SimpleNamespace(scene_decoder='libraw'), None,
                                            tone_plan=_scene_plan(), delivery=profile)
                        self.assertEqual(render.call_count, 1)
                        self.assertEqual(render.call_args.kwargs['sdr_float'], precise)


if __name__ == '__main__':
    unittest.main()
