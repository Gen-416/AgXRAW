# SPDX-License-Identifier: GPL-3.0-or-later
"""A decoder code ceiling is distinct from original RAW saturation."""
from pathlib import Path
import os
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import _fast, raw_io, dng_opcodes as ops
from dngscan.analysis import analyze
from dngscan.tone import build_render_plan
from tests.test_pipeline_corrections import write_sensor_dng
from tests.test_reconstructed_lens_range import camera_decode, NEUTRAL, IDENTITY_WARP


def unsaturated_colour_ramp():
    y, x = np.indices((128, 128))
    red = np.broadcast_to(np.linspace(1000, 3500, 128), y.shape)
    return np.where((y % 2 == 0) & (x % 2 == 0), red, 1000).astype(np.uint16)


def write_file(path, pixels, *, opcode=None, orientation=1):
    write_sensor_dng(path, signal=pixels, neutral=NEUTRAL,
                     opcodes={51022: [opcode]} if opcode is not None else None)
    if orientation != 1:
        content = bytearray(path.read_bytes())
        count, = struct.unpack_from('<H', content, 8)
        for i in range(count):
            offset = 10 + 12 * i
            tag, = struct.unpack_from('<H', content, offset)
            if tag == 50730:
                content[offset:offset+12] = struct.pack('<HHIHH', 274, 3, 1, orientation, 0)
                break
        else:
            raise AssertionError('fixture has no optional BaselineExposure')
        path.write_bytes(content)


class DecoderCeilingTests(unittest.TestCase):
    def test_no_boundary_has_no_loss_allocation_and_existing_loss_is_retained(self):
        camera = np.full((257, 31, 3), 20000, np.uint16)
        self.assertIsNone(raw_io._merge_decoder_ceiling_loss(camera, None))
        existing = np.zeros(camera.shape, np.float16)
        existing[100, 10, 1] = 1
        self.assertIs(raw_io._merge_decoder_ceiling_loss(camera, existing), existing)
        self.assertEqual(np.count_nonzero(existing), 1)

    def test_boundary_mask_merges_in_place_without_mutating_camera(self):
        camera = np.full((257, 31, 3), 20000, np.uint16)
        camera[128, 15, 0] = 65535
        original = camera.copy()
        existing = np.zeros(camera.shape, np.float16)
        existing[100, 10, 1] = 1
        result = raw_io._merge_decoder_ceiling_loss(camera, existing)
        self.assertIs(result, existing)
        self.assertEqual(result[128, 15, 0], 1)
        self.assertEqual(np.count_nonzero(result), 2)
        np.testing.assert_array_equal(camera, original)
        extended = camera.astype(np.float32)
        extended[128, 15] = [-100, 65535, 200000]
        self.assertIsNone(raw_io._merge_decoder_ceiling_loss(extended, None))

    def test_actual_wb_ceiling_changes_permission_and_not_sensor_clip_statistics(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'wb-ceiling.dng'
            pixels = unsaturated_colour_ramp()
            write_file(path, pixels)
            for half in (False, True):
                with self.subTest(half=half):
                    with patch('dngscan.raw_io._merge_decoder_ceiling_loss', side_effect=lambda camera, loss: loss), \
                         patch('dngscan.decoder_loss.record_wb_ceiling_loss', side_effect=lambda *args, **kwargs: args[-1] if len(args) > 5 else kwargs.get('loss')):
                        old = raw_io.load_raw(path, scene_half_size=half, demosaic='dht')
                    new = raw_io.load_raw(path, scene_half_size=half, demosaic='dht')
                    np.testing.assert_array_equal(old.raw_image, new.raw_image)
                    np.testing.assert_array_equal(old.scene_rec2020_render, new.scene_rec2020_render)
                    self.assertLess(int(pixels.max()), new.white_level)
                    camera, _ = camera_decode(path, 'clip', half)
                    self.assertTrue(np.any(camera[..., 0] == 65535))
                    self.assertTrue(np.all(new.processing_clip_masks[camera == 65535] == 1))
                    self.assertEqual(old.scene_processing_loss_pct, 0.)
                    self.assertGreater(new.scene_processing_loss_pct, 25.)
                    a_old, _, _ = analyze(old, 4)
                    a_new, _, _ = analyze(new, 4)
                    self.assertEqual(a_old.clip_pct, a_new.clip_pct)
                    self.assertEqual(a_old.color_clip_k_of_all_pct, a_new.color_clip_k_of_all_pct)
                    self.assertTrue(all(value == 0 for value in a_new.clip_pct.values()))
                    p_old = build_render_plan(old, a_old, 'agx')
                    p_new = build_render_plan(new, a_new, 'agx')
                    self.assertEqual(p_old.scene.reliable_sample_pct, 100.)
                    self.assertLess(p_new.scene.reliable_sample_pct, 75.)

    def test_all_highlight_modes_record_actual_integer_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'sensor-ceiling.dng'
            write_file(path, np.full((128, 128), 4095, np.uint16))
            for mode in ('clip', 'blend', 'reconstruct'):
                with self.subTest(mode=mode):
                    bundle = raw_io.load_raw(path, scene_highlight_mode=mode)
                    camera, _ = camera_decode(path, mode, False)
                    if np.any(camera == 65535):
                        self.assertTrue(np.all(bundle.processing_clip_masks[camera == 65535] == 1))
                        self.assertGreater(bundle.scene_processing_loss_pct, 0.)
                    else:
                        # A source ceiling can affect neighbours even when
                        # recovery makes every final camera code interior.
                        if bundle.processing_clip_masks is None:
                            self.assertEqual(bundle.scene_processing_loss_pct, 0.)
                        else:
                            self.assertGreater(bundle.scene_processing_loss_pct, 0.)
                    # RAW saturation may overlap a decoder boundary but is
                    # not equivalent: LibRaw often returns 65534 or recovers
                    # into its interior range, especially in blend mode.
                    self.assertEqual(int(bundle.raw_image.max()), 4095)
                    self.assertEqual(float(bundle.clip_masks.max()), 1.)
                    # Explicit adapter output verifies registration even in
                    # modes that did not hit the boundary on this fixture.
                    camera[64, 64, 1] = 65535
                    with patch('dngscan.raw_io.render_to_scene_rec2020', return_value=camera):
                        marked = raw_io.load_raw(path, scene_highlight_mode=mode)
                    self.assertTrue(np.all(marked.processing_clip_masks[camera == 65535] == 1))

    def test_boundary_loss_follows_warp_half_size_and_orientation(self):
        payload = struct.pack('>L8d', 1, .9, 0., 0., 0., 0., 0., .5, .5)
        warp = ops.Warp(((.9, 0., 0., 0., 0., 0.),), .5, .5)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'warped-ceiling.dng'
            for fast in (('0', '1') if _fast.available() else ('0',)):
                for half in (False, True):
                    for orientation, flip in ((1, 0), (6, 6)):
                        with self.subTest(fast=fast, half=half, flip=flip), patch.dict(os.environ, DNGSCAN_FAST=fast):
                            write_file(path, unsaturated_colour_ramp())
                            unwarped = raw_io.load_raw(path, scene_half_size=half)
                            write_file(path, unsaturated_colour_ramp(), opcode=(1, payload), orientation=orientation)
                            with patch.dict(os.environ, DNGSCAN_FAST='0'):
                                expected = ops.warp_image(unwarped.processing_clip_masks, warp, loss=True)
                            bundle = raw_io.load_raw(path, scene_half_size=half)
                            np.testing.assert_array_equal(bundle.processing_clip_masks,
                                raw_io._orient_like_libraw(expected, flip))
                            self.assertGreater(bundle.scene_processing_loss_pct, 0.)

    def test_identity_lens_never_adds_decoder_boundary_evidence_twice(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'identity-ceiling.dng'
            write_file(path, unsaturated_colour_ramp())
            before = raw_io.load_raw(path)
            write_file(path, unsaturated_colour_ramp(), opcode=(1, IDENTITY_WARP))
            after = raw_io.load_raw(path)
            np.testing.assert_array_equal(after.scene_rec2020_render, before.scene_rec2020_render)
            np.testing.assert_array_equal(after.processing_clip_masks, before.processing_clip_masks)
            self.assertEqual(after.scene_processing_loss_pct, before.scene_processing_loss_pct)


if __name__ == '__main__':
    unittest.main()
