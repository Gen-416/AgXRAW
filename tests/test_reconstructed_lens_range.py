# SPDX-License-Identifier: GPL-3.0-or-later
"""Real LibRaw regressions for late lens corrections on reconstructed camera RGB."""
from __future__ import annotations

import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import _fast, dng_opcodes as ops, embedded_lens, raw_io
from dngscan._deps import rawpy
from dngscan.metadata import DngVignetteRadial
from tests.test_pipeline_corrections import write_sensor_dng


NEUTRAL = (.5, 1., 1. / 3.)
IDENTITY_WARP = struct.pack('>L8d', 1, 1., 0., 0., 0., 0., 0., .5, .5)
UNITY_VIGNETTE = struct.pack('>7d', 0., 0., 0., 0., 0., .5, .5)


def saturated_green_scene():
    """One colour saturates first, with the other two retaining highlight signal."""
    y, x = np.indices((128, 128))
    light = np.broadcast_to(np.linspace(.15, 2.3, 128)[None, :], y.shape)
    response = np.where((y % 2 == 0) & (x % 2 == 0), .5,
                        np.where((y % 2 == 1) & (x % 2 == 1), 1. / 3., 1.))
    return np.minimum(4095, np.rint(light * response * 4095)).astype(np.uint16)


def camera_decode(path, mode, half):
    """Actual LibRaw camera planes before project-owned late lens corrections."""
    with rawpy.imread(str(path)) as raw:
        camera = raw_io.render_to_scene_rec2020(raw, mode, half,
            raw_io.resolve_demosaic_algorithm(raw, 'auto'),
            raw_io._fixed_asshot_wb_kwargs(raw.camera_whitebalance), camera_rgb=True)
        matrix = ops.libraw_camera_matrix(raw.color_matrix, raw.rgb_xyz_matrix, is_dng=True)
    return camera, matrix


def backends():
    return ('0', '1') if _fast.available() else ('0',)


class ReconstructedLensFileTests(unittest.TestCase):
    def test_identity_lens_preserves_actual_reconstruction_and_evidence(self):
        pixels = saturated_green_scene()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'range.dng'
            for fast in backends():
                for half in (False, True):
                    for mode in ('clip', 'blend', 'reconstruct'):
                        with self.subTest(fast=fast, half=half, mode=mode), patch.dict(os.environ, DNGSCAN_FAST=fast):
                            write_sensor_dng(path, signal=pixels, neutral=NEUTRAL)
                            reference = raw_io.load_raw(path, scene_half_size=half, scene_highlight_mode=mode)
                            if mode != 'clip':
                                # This fixture really exercises a reconstructed plane
                                # above its nominal WB-scaled sensor-white boundary.
                                camera, _ = camera_decode(path, mode, half)
                                wb = np.asarray(reference.camera_wb[:3])
                                nominal = 65535. * wb / wb.max()
                                self.assertTrue(np.any(camera > nominal))
                            for opcode in ((1, IDENTITY_WARP), (3, UNITY_VIGNETTE)):
                                write_sensor_dng(path, signal=pixels, neutral=NEUTRAL, opcodes={51022: [opcode]})
                                result = raw_io.load_raw(path, scene_half_size=half, scene_highlight_mode=mode)
                                np.testing.assert_array_equal(result.scene_rec2020_render, reference.scene_rec2020_render)
                                np.testing.assert_array_equal(result.raw_image, pixels)
                                np.testing.assert_array_equal(result.clip_masks, reference.clip_masks)
                                self.assertEqual(result.scene_processing_loss_pct, reference.scene_processing_loss_pct)
                                if reference.processing_clip_masks is None:
                                    self.assertFalse(np.any(result.processing_clip_masks))
                                else:
                                    np.testing.assert_array_equal(result.processing_clip_masks, reference.processing_clip_masks)

    def test_nonidentity_vignette_preserves_late_linear_gain_without_wrap(self):
        pixels = saturated_green_scene()
        vignette = struct.pack('>7d', 1.5, 0., 0., 0., 0., .5, .5)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'gain.dng'
            for fast in backends():
                for half in (False, True):
                    for mode in ('clip', 'blend', 'reconstruct'):
                        with self.subTest(fast=fast, half=half, mode=mode), patch.dict(os.environ, DNGSCAN_FAST=fast):
                            write_sensor_dng(path, signal=pixels, neutral=NEUTRAL)
                            camera, matrix = camera_decode(path, mode, half)
                            h, w = camera.shape[:2]
                            y, x = np.indices((h, w), dtype=np.float64)
                            radius2 = ((x + .5 - w / 2) ** 2 + (y + .5 - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
                            gained = camera.astype(np.float32) * (1. + 1.5 * radius2).astype(np.float32)[..., None]
                            expected_camera = np.clip(gained, 0, 65535).astype(np.uint16) if mode == 'clip' else gained
                            expected = ops.camera_to_rec2020(expected_camera, matrix)
                            write_sensor_dng(path, signal=pixels, neutral=NEUTRAL, opcodes={51022: [(3, vignette)]})
                            result = raw_io.load_raw(path, scene_half_size=half, scene_highlight_mode=mode)
                            np.testing.assert_array_equal(result.scene_rec2020_render, expected)
                            np.testing.assert_array_equal(result.raw_image, pixels)
                            if mode != 'clip':
                                self.assertGreater(float(gained.max()), 65535.)
                                self.assertEqual(result.scene_processing_loss_pct, 0.)
                            else:
                                self.assertGreater(result.scene_processing_loss_pct, 0.)

    def test_nonidentity_warp_transports_actual_highlight_reconstruction(self):
        pixels = saturated_green_scene()
        warp_payload = struct.pack('>L8d', 1, .9, 0., 0., 0., 0., 0., .5, .5)
        warp = ops.Warp(((.9, 0., 0., 0., 0., 0.),), .5, .5)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'warp.dng'
            for fast in backends():
                for half in (False, True):
                    for mode in ('clip', 'blend', 'reconstruct'):
                        with self.subTest(fast=fast, half=half, mode=mode), patch.dict(os.environ, DNGSCAN_FAST=fast):
                            write_sensor_dng(path, signal=pixels, neutral=NEUTRAL)
                            camera, matrix = camera_decode(path, mode, half)
                            with patch.dict(os.environ, DNGSCAN_FAST='0'):
                                expected_camera = ops.warp_image(camera if mode == 'clip' else camera.astype(np.float32), warp)
                            expected = ops.camera_to_rec2020(expected_camera, matrix)
                            write_sensor_dng(path, signal=pixels, neutral=NEUTRAL, opcodes={51022: [(1, warp_payload)]})
                            result = raw_io.load_raw(path, scene_half_size=half, scene_highlight_mode=mode)
                            np.testing.assert_allclose(result.scene_rec2020_render, expected, rtol=2e-7, atol=.02 if mode != 'clip' else 2.)
                            np.testing.assert_array_equal(result.raw_image, pixels)
                            if mode != 'clip':
                                self.assertEqual(result.scene_processing_loss_pct, 0.)

    def test_embedded_lens_uses_same_extended_camera_domain(self):
        pixels = saturated_green_scene()
        vignette = embedded_lens.RadialVignette((0., 1.), (.5, .5))
        warp = ops.Warp(((.9, 0., 0., 0., 0., 0.),), .5, .5)
        profile = embedded_lens.LensProfile(warp, vignette, 'Synthetic embedded lens')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'embedded.dng'
            write_sensor_dng(path, signal=pixels, neutral=NEUTRAL)
            for fast in backends():
                for half in (False, True):
                    for mode in ('clip', 'blend', 'reconstruct'):
                        with self.subTest(fast=fast, half=half, mode=mode), patch.dict(os.environ, DNGSCAN_FAST=fast):
                            camera, matrix = camera_decode(path, mode, half)
                            gained = camera.astype(np.float32) * 2.
                            gained = np.clip(gained, 0, 65535).astype(np.uint16) if mode == 'clip' else gained
                            with patch.dict(os.environ, DNGSCAN_FAST='0'):
                                expected = ops.camera_to_rec2020(ops.warp_image(gained, warp), matrix)
                            with patch.object(embedded_lens, 'read', return_value=profile):
                                result = raw_io.load_raw(path, scene_half_size=half, scene_highlight_mode=mode)
                            np.testing.assert_allclose(result.scene_rec2020_render, expected, rtol=2e-7, atol=.02 if mode != 'clip' else 2.)
                            np.testing.assert_array_equal(result.raw_image, pixels)
                            if mode != 'clip':
                                self.assertEqual(result.scene_processing_loss_pct, 0.)

    def test_stage3_point_transforms_refuse_undefined_reconstructed_domain(self):
        polynomial = struct.pack('>4l5L2d', 0, 0, 128, 128, 0, 3, 1, 1, 1, 0., 1.)
        table = struct.pack('>4l5L', 0, 0, 128, 128, 0, 3, 1, 1, 65536) + np.arange(65536, dtype='>u2').tobytes()
        scale_rows = struct.pack('>4l5L', 0, 0, 128, 128, 0, 3, 1, 1, 128) + struct.pack('>128f', *([1.] * 128))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'point.dng'
            write_sensor_dng(path, signal=saturated_green_scene(), neutral=NEUTRAL)
            reference_loss = raw_io.load_raw(path, scene_highlight_mode='clip').scene_processing_loss_pct
            for opcode in ((8, polynomial), (7, table), (12, scale_rows)):
                write_sensor_dng(path, signal=saturated_green_scene(), neutral=NEUTRAL, opcodes={51022: [opcode]})
                for mode in ('blend', 'reconstruct'):
                    for half in (False, True):
                        with self.subTest(kind=opcode[0], mode=mode, half=half):
                            with self.assertRaisesRegex(RuntimeError, 'stage-3 point transforms.*choose clip mode or Apple RAW'):
                                raw_io.load_raw(path, scene_half_size=half, scene_highlight_mode=mode)
                accepted = raw_io.load_raw(path, scene_highlight_mode='clip')
                self.assertEqual(accepted.scene_highlight_mode, 'clip')
                self.assertEqual(accepted.scene_processing_loss_pct, reference_loss)


class ExtendedLensKernelTests(unittest.TestCase):
    def test_float_warp_preserves_negative_ringing_and_overrange(self):
        source = np.zeros((48, 64, 3), np.float32)
        source[:, 32:] = 100000.
        source = source[::-1]  # Borrowed, strided input remains supported.
        original = source.copy()
        warp = ops.Warp(((.9, 0., 0., 0., 0., 0.),), .5, .5)
        with patch.dict(os.environ, DNGSCAN_FAST='0'):
            reference = ops.warp_image(source, warp)
        self.assertLess(float(reference.min()), 0.)
        self.assertGreater(float(reference.max()), 100000.)
        for fast in backends():
            with self.subTest(fast=fast), patch.dict(os.environ, DNGSCAN_FAST=fast):
                prior = np.zeros(source.shape, np.float16)
                prior[20, 20, 1] = 1
                result, transported = ops.warp_image(source, warp, processing_loss=prior)
                expected_loss = ops.warp_image(prior, warp, loss=True)
                self.assertEqual(result.dtype, np.float32)
                np.testing.assert_allclose(result, reference, rtol=1e-7, atol=.01)
                np.testing.assert_array_equal(transported, expected_loss)
                np.testing.assert_array_equal(source, original)

    def test_integer_warp_tracks_only_new_clipping_and_existing_footprints(self):
        source = np.zeros((48, 64, 3), np.uint16)
        source[:, 32:] = 65000
        warp = ops.Warp(((.9, 0., 0., 0., 0., 0.),), .5, .5)
        with patch.dict(os.environ, DNGSCAN_FAST='0'):
            unlimited = ops.warp_image(source.astype(np.float32), warp)
        expected = ((unlimited < 0) | (unlimited >= 65535)).astype(np.float16)
        self.assertTrue(np.any(expected))
        for fast in backends():
            with self.subTest(fast=fast), patch.dict(os.environ, DNGSCAN_FAST=fast):
                result, mask = ops.warp_image(source, warp, processing_loss=np.zeros(source.shape, np.float16))
                np.testing.assert_allclose(result, np.clip(unlimited, 0, 65535).astype(np.uint16), atol=1)
                np.testing.assert_array_equal(mask, expected)
                saturated = np.full(source.shape, 65535, np.uint16)
                _, mask = ops.warp_image(saturated, warp, processing_loss=np.zeros(source.shape, np.float16))
                self.assertFalse(np.any(mask))

    def test_shading_identity_does_not_recount_existing_saturation(self):
        for dtype in (np.uint16, np.float32):
            source = np.full((24, 32, 3), 65535, dtype)
            for embedded in (False, True):
                with self.subTest(dtype=dtype, embedded=embedded):
                    scene = source.copy()
                    loss = np.zeros(source.shape, np.float16)
                    if embedded:
                        embedded_lens.apply_vignette(scene, embedded_lens.RadialVignette((0., 1.), (1., 1.)), loss,
                                                     np.full(3, 65535.) if dtype == np.uint16 else None)
                    else:
                        raw_io._apply_vignette_render(scene, DngVignetteRadial((0.,) * 5, .5, .5), loss_mask=loss)
                    np.testing.assert_array_equal(scene, source)
                    self.assertFalse(np.any(loss))


if __name__ == '__main__':
    unittest.main()
