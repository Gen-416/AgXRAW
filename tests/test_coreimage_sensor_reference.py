# SPDX-License-Identifier: GPL-3.0-or-later
"""Opaque Apple geometry gets trustworthy unreconstructed sensor evidence."""
from contextlib import ExitStack
import gc
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from dngscan import raw_io
from dngscan.analysis import analyze
from dngscan.hdr_agx_plan import compile_hdr_agx_plan
from dngscan.tone import build_render_plan
from tests.test_pipeline_corrections import write_sensor_dng


def _write_reference_dng(path, baseline=0., *, clipped=True):
    pixels = np.full((128, 128), 700, np.uint16)
    pixels[60:68, 60:68] = 3000 if clipped else 700
    gain = struct.pack('>10L4dL4f', 0, 0, 128, 128, 0, 1, 1, 1, 2, 2,
                       1., 1., 0., 0., 1, 2., 2., 2., 2.)
    write_sensor_dng(path, signal=pixels, neutral=(.5, 1., .75),
                     opcodes={51009: [(9, gain)]})
    data = bytearray(path.read_bytes())
    for i in range(struct.unpack_from('<H', data, 8)[0]):
        entry = 10 + 12 * i
        if struct.unpack_from('<H', data, entry)[0] == 50730:
            offset = struct.unpack_from('<I', data, entry + 8)[0]
            struct.pack_into('<ll', data, offset, int(round(baseline * 1000)), 1000)
            path.write_bytes(data)
            return
    raise AssertionError('synthetic DNG has no BaselineExposure')


class CoreImageSensorReferenceTests(unittest.TestCase):
    def _apple_decode(self, stack, baseline):
        # Only the opaque Apple scene API is stubbed. All RAW acquisition,
        # GainMap application, scaling, LibRaw decoding and mask transport run.
        stack.enter_context(patch('dngscan.coreimage_decode.runtime_available', return_value=True))
        stack.enter_context(patch('dngscan.coreimage_decode.decode_scene_rec2020', return_value=(
            np.full((64, 64, 3), .4, np.float32),
            {'version': '9', 'baseline_exposure_authored': baseline,
             'baseline_exposure_cleared': True})))
        stack.enter_context(patch('dngscan.coreimage_decode.read_dng_opcodes',
                                  return_value={'names': ('GainMap',)}))

    def test_real_clipped_dng_separates_alignment_from_sensor_evidence_and_releases_master(self):
        original = raw_io._decode_corrected_libraw
        with TemporaryDirectory() as td:
            path = Path(td) / 'reference.dng'
            for half in (False, True):
                for baseline in (0., 1.):
                    with self.subTest(half=half, baseline=baseline), ExitStack() as stack:
                        _write_reference_dng(path, baseline)
                        self._apple_decode(stack, baseline)
                        calls, raster_refs = [], []
                        def decode(*args, **kwargs):
                            calls.append((args[3], kwargs))
                            if args[3] == 'clip':
                                gc.collect()
                                self.assertIsNone(raster_refs[0]())
                            result = original(*args, **kwargs)
                            raster_refs.append(weakref.ref(result[0]))
                            return result
                        stack.enter_context(patch.object(raw_io, '_decode_corrected_libraw', side_effect=decode))
                        bundle = raw_io.load_raw(path, decoder='coreimage', scene_half_size=half)
                        self.assertEqual([mode for mode, _ in calls], ['reconstruct', 'clip'])
                        self.assertTrue(calls[-1][1]['allow_loss_fallback'])
                        self.assertEqual(bundle.scene_highlight_mode, 'reconstruct')
                        self.assertEqual(bundle.scene_reliability_source, 'sensor-reference')
                        self.assertGreater(len(bundle.scene_reliable_reference_rec2020), 256)
                        self.assertGreater(bundle.scene_reliable_reference_pct, 90.)
                        self.assertGreater(bundle.scene_processing_loss_pct, 0.)
                        self.assertLess(bundle.scene_processing_loss_pct, 10.)
                        self.assertIsNone(bundle.scene_reference_error)
                        self.assertIsNone(bundle.scene_align_error)
                        self.assertEqual(bundle.noise_decode['alignment_reference_highlight_mode'], 'reconstruct')
                        self.assertEqual(bundle.noise_decode['evidence_reference_highlight_mode'], 'clip')
                        self.assertEqual(bundle.noise_decode['evidence_reference_status'], 'measured')
                        self.assertIn('bayer-half', bundle.noise_decode['evidence_reference_loss_support'])
                        # Kelvin WB is a later project transform. It must not
                        # change either decoder's fixed as-shot scale reference.
                        balanced = raw_io.rebalance_raw_bundle(bundle, '5500k')
                        loaded_wb = raw_io.load_raw(path, decoder='coreimage', scene_half_size=half, wb_mode='5500k')
                        self.assertEqual(loaded_wb.scene_align_factor, bundle.scene_align_factor)
                        np.testing.assert_array_equal(loaded_wb.scene_reliable_reference_rec2020,
                                                      balanced.scene_reliable_reference_rec2020)
                        self.assertTrue(all(ref() is None for ref in raster_refs))

    def test_clip_evidence_failure_keeps_valid_alignment_and_explicitly_closes_hdr(self):
        original = raw_io._decode_corrected_libraw
        with TemporaryDirectory() as td, ExitStack() as stack:
            path = Path(td) / 'reference.dng'
            _write_reference_dng(path)
            self._apple_decode(stack, 0.)
            accepted = raw_io.load_raw(path, decoder='coreimage', scene_half_size=True)
            def decode(*args, **kwargs):
                if args[3] == 'clip':
                    raise RuntimeError('injected clip evidence failure')
                return original(*args, **kwargs)
            stack.enter_context(patch.object(raw_io, '_decode_corrected_libraw', side_effect=decode))
            rejected = raw_io.load_raw(path, decoder='coreimage', scene_half_size=True)
            self.assertEqual(rejected.scene_align_factor, accepted.scene_align_factor)
            self.assertEqual(rejected.scene_scale, accepted.scene_scale)
            np.testing.assert_array_equal(rejected.scene_rec2020_render, accepted.scene_rec2020_render)
            self.assertIsNone(rejected.scene_align_error)
            self.assertIn('injected clip evidence failure', rejected.scene_reference_error)
            self.assertEqual(rejected.scene_reliability_source, 'sensor-reference')
            self.assertEqual(rejected.scene_reliable_reference_rec2020.shape, (0, 3))
            self.assertEqual(rejected.noise_decode['evidence_reference_status'], 'unavailable')
            analysis, _, _ = analyze(rejected, margin=4, diagnostics=False)
            hdr = compile_hdr_agx_plan(build_render_plan(rejected, analysis, 'agx', 'p3'),
                                       analysis=analysis, scene_decoder='coreimage')
            self.assertEqual(hdr.tone.rendered_headroom_ev, 0.)

    def test_no_pre_loss_reuses_the_single_reconstruct_reference(self):
        with TemporaryDirectory() as td, ExitStack() as stack:
            path = Path(td) / 'reference.dng'
            _write_reference_dng(path, clipped=False)
            self._apple_decode(stack, 0.)
            decode = stack.enter_context(patch.object(raw_io, '_decode_corrected_libraw',
                                                      wraps=raw_io._decode_corrected_libraw))
            bundle = raw_io.load_raw(path, decoder='coreimage', scene_half_size=True)
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(bundle.noise_decode['evidence_reference_highlight_mode'], 'reconstruct')
            self.assertEqual(bundle.noise_decode['evidence_reference_status'], 'measured')


if __name__ == '__main__':
    unittest.main()
