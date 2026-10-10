# SPDX-License-Identifier: GPL-3.0-or-later
"""Real LinearRAW files at the fixed decoder-WB / DNG opcode boundary."""
from __future__ import annotations

import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import _fast, dng_opcodes as ops, dng_point_ops as point, raw_io
from tests.test_pipeline_corrections import write_sensor_dng


def write_linear_point_dng(path, *, signal=16384, neutral=(.5, 1., 1.), stage=None,
                           opcode=None, after=(), white=(65535,), black=(0,), limit=1.):
    """Keep the real 16-bit LinearRAW coding range explicit in a tiny DNG."""
    tags = {51009 if stage == 2 else 51022: [opcode]} if opcode else {}
    if after:
        tags.setdefault(51022, []).extend(after)
    write_sensor_dng(path, linear=True, signal=signal, neutral=neutral, opcodes=tags, limit=limit)
    data = bytearray(path.read_bytes())
    count = struct.unpack_from('<H', data, 8)[0]
    replacements = {50717: (4, struct.pack(f'<{len(white)}L', *white), len(white)),
                    50714: (5, b''.join(struct.pack('<LL', v, 1) for v in black), len(black))}
    for i in range(count):
        offset = 10 + 12 * i
        tag = struct.unpack_from('<H', data, offset)[0]
        if tag in replacements:
            typ, payload, n = replacements.pop(tag)
            struct.pack_into('<HL', data, offset + 2, typ, n)
            if len(payload) <= 4:
                data[offset + 8:offset + 12] = payload.ljust(4, b'\0')
            else:
                struct.pack_into('<L', data, offset + 8, len(data))
                data.extend(payload)
    if replacements:
        raise AssertionError('fixture is missing a replaced tag')
    path.write_bytes(data)


def polynomial(values, *, area=(0, 0, 128, 128, 0, 3, 1, 1)):
    return (8, struct.pack('>4l5L', *area, len(values) - 1)
            + struct.pack(f'>{len(values)}d', *values))


def load_camera(path, **kwargs):
    """Observe project-owned camera planes while executing the real transform."""
    transform = ops.camera_to_rec2020
    camera = []

    def observe(image, matrix):
        camera.append(image.copy())
        return transform(image, matrix)

    with patch.object(ops, 'camera_to_rec2020', side_effect=observe):
        bundle = raw_io.load_raw(path, **kwargs)
    return bundle, camera[-1]


class Stage3PointDomainTests(unittest.TestCase):
    def test_real_linear_raw_square_undoes_wb_before_opcode(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'square.dng'
            for fast in ('0', '1') if _fast.available() else ('0',):
                for neutral in ((.5, 1., 1.), (1., 1., 1.), (2., 1., 1.)):
                    with self.subTest(fast=fast, neutral=neutral), patch.dict(os.environ, DNGSCAN_FAST=fast):
                        write_linear_point_dng(path, signal=4096, neutral=neutral)
                        reference, ref_camera = load_camera(path)
                        write_linear_point_dng(path, neutral=neutral, stage=2,
                                               opcode=polynomial((0., 0., 1.)))
                        stage2, stage2_camera = load_camera(path)
                        np.testing.assert_array_equal(stage2.scene_rec2020_render,
                                                      reference.scene_rec2020_render)
                        write_linear_point_dng(path, neutral=neutral, stage=3,
                                               opcode=polynomial((0., 0., 1.)))
                        stage3, camera = load_camera(path)
                        wb = np.asarray(stage3.camera_wb[:3])
                        gain = wb / wb.min()
                        expected = (16384. ** 2 / 65535.) * gain
                        np.testing.assert_allclose(camera, np.broadcast_to(expected, camera.shape),
                                                   rtol=0., atol=.002)
                        # Stage 2 has one extra integer handoff before LibRaw;
                        # stage 3 deliberately retains the fractional code.
                        np.testing.assert_allclose(camera, stage2_camera, rtol=0., atol=.13)
                        self.assertEqual(camera.dtype, np.float32)
                        self.assertFalse(np.any(stage3.processing_clip_masks))
                        np.testing.assert_array_equal(stage3.raw_image, 16384)

    def test_linear_identity_remains_exact_with_nonunit_wb_and_half_size(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'identity.dng'
            for half in (False, True):
                write_linear_point_dng(path, signal=16384)
                reference = raw_io.load_raw(path)
                expected = reference.scene_rec2020_render
                if half:
                    # LibRaw leaves LinearRAW full sized; point-op previews
                    # instead box-reduce after executing native coordinates.
                    expected = expected.reshape(64, 2, 64, 2, 3).mean(axis=(1, 3))
                write_linear_point_dng(path, stage=3, opcode=polynomial((0., 1.)))
                result = raw_io.load_raw(path, scene_half_size=half)
                np.testing.assert_array_equal(result.scene_rec2020_render,
                                              expected)

    def test_decoder_ceiling_evidence_survives_a_darkening_point_operation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'ceiling.dng'
            write_linear_point_dng(path, signal=40000)
            original, before = load_camera(path)
            self.assertEqual(before[64, 64, 0], 65535)
            self.assertEqual(original.processing_clip_masks[64, 64, 0], 1.)
            write_linear_point_dng(path, signal=40000, stage=3,
                                   opcode=polynomial((0., 0., 1.)))
            result, camera = load_camera(path)
            self.assertLess(camera[64, 64, 0], 65535.)
            self.assertEqual(result.processing_clip_masks[64, 64, 0], 1.)
            np.testing.assert_array_equal(result.raw_image, original.raw_image)

    def test_real_linear_raw_nonlinear_table_matches_stage2(self):
        table = np.rint((np.arange(65536, dtype=np.float64) / 65535.) ** 2 * 65535.).astype('>u2')
        opcode = (7, struct.pack('>4l5L', 0, 0, 128, 128, 0, 3, 1, 1, table.size) + table.tobytes())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'table.dng'
            write_linear_point_dng(path, stage=2, opcode=opcode)
            reference = raw_io.load_raw(path)
            write_linear_point_dng(path, stage=3, opcode=opcode)
            result, camera = load_camera(path)
            np.testing.assert_array_equal(camera[64, 64], [8192., 4096., 4096.])
            np.testing.assert_array_equal(result.scene_rec2020_render,
                                          reference.scene_rec2020_render)

    def test_per_plane_coding_spans_and_spatial_black_overrides_match_stage2(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'plane-ranges.dng'
            for fast in ('0', '1') if _fast.available() else ('0',):
                for white, black in (((65535, 60000, 50000), (0,)),
                                     ((65535, 60000, 50000), (100,)),
                                     ((65535, 60000, 50000), (100, 100, 100)),
                                     ((65535, 60000, 50000), (100, 200, 300)),
                                     ((65535,), (100, 200, 300))):
                    for limit in (1., .8):
                        with self.subTest(fast=fast, white=white, black=black, limit=limit), \
                                patch.dict(os.environ, DNGSCAN_FAST=fast):
                            kwargs = dict(signal=10000, white=white, black=black, limit=limit,
                                          opcode=polynomial((0., 0., 1.)))
                            write_linear_point_dng(path, stage=2, **kwargs)
                            _, reference = load_camera(path)
                            write_linear_point_dng(path, stage=3, **kwargs)
                            result, camera = load_camera(path)
                            # One pre-decoder uint16 quantization in List2,
                            # followed by LibRaw truncation, is absent in List3.
                            np.testing.assert_allclose(camera, reference, rtol=0., atol=1.5)
                            self.assertFalse(np.any(result.processing_clip_masks))
                            self.assertEqual(result.evidence.spatial_black is not None,
                                             len(set(black)) > 1)

    def test_nonidentity_lens_after_point_uses_float_wb_logical_white(self):
        vignette = (3, struct.pack('>7d', 1., 0., 0., 0., 0., .5, .5))
        warp = (1, struct.pack('>L8d', 1, .9, 0., 0., 0., 0., 0., .5, .5))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'point-lens.dng'
            write_linear_point_dng(path, stage=3, opcode=polynomial((.5, 1.)))
            _, camera = load_camera(path)
            y, x = np.indices(camera.shape[:2], dtype=np.float64)
            radius2 = ((x + .5 - 64.) ** 2 + (y + .5 - 64.) ** 2) / (2. * 64. ** 2)
            gained = np.minimum(camera * (1. + radius2[..., None]), [131070., 65535., 65535.]).astype(np.float32)
            for fast in ('0', '1') if _fast.available() else ('0',):
                with self.subTest(fast=fast), patch.dict(os.environ, DNGSCAN_FAST=fast):
                    expected = ops.warp_image(gained, ops.Warp(((.9, 0., 0., 0., 0., 0.),), .5, .5))
                    write_linear_point_dng(path, stage=3, opcode=polynomial((.5, 1.)), after=(vignette, warp))
                    result, observed = load_camera(path)
                    np.testing.assert_allclose(observed, expected, rtol=1e-7, atol=.02)
                    self.assertGreater(float(observed[..., 0].min()), 65535.)
                    self.assertTrue(np.any(result.processing_clip_masks))

    def test_restored_wb_can_exceed_uint16_without_wrap_or_late_identity_clipping(self):
        identity_vignette = (3, struct.pack('>7d', 0., 0., 0., 0., 0., .5, .5))
        identity_warp = (1, struct.pack('>L8d', 1, 1., 0., 0., 0., 0., 0., .5, .5))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'extended-wb.dng'
            write_linear_point_dng(path, stage=3, opcode=polynomial((.5, 1.)))
            reference, camera = load_camera(path)
            np.testing.assert_allclose(camera[64, 64], [98303., 49151.5, 49151.5], atol=.01)
            self.assertFalse(np.any(reference.processing_clip_masks))
            for after in ((identity_vignette,), (identity_warp,),
                          (identity_vignette, identity_warp)):
                with self.subTest(after=[op[0] for op in after]):
                    write_linear_point_dng(path, stage=3, opcode=polynomial((.5, 1.)), after=after)
                    result, observed = load_camera(path)
                    np.testing.assert_array_equal(observed, camera)
                    np.testing.assert_array_equal(result.scene_rec2020_render,
                                                  reference.scene_rec2020_render)
                    self.assertFalse(np.any(result.processing_clip_masks))

    def test_float_point_apply_keeps_fractional_values_and_plane_pitch(self):
        image = np.full((6, 8, 3), 16384., np.float32)
        op = point.PointOp(8, (1, 2, 5, 6, 1, 1, 2, 2), (0., 0., 1.), 3)
        point.apply(image, op, white=[131070., 65535., 65535.])
        self.assertAlmostEqual(float(image[1, 2, 1]), 16384. ** 2 / 65535., places=3)
        self.assertEqual(image[2, 2, 1], 16384.)
        self.assertEqual(image[1, 2, 0], 16384.)

    def test_integer_point_output_never_wraps_with_uncapped_logical_white(self):
        image = np.full((1, 1, 3), 40000, np.uint16)
        point.apply(image, point.PointOp(8, (0, 0, 1, 1, 0, 3, 1, 1), (1.,), 3),
                    white=[131070., 65535., 65535.])
        np.testing.assert_array_equal(image, 65535)


if __name__ == '__main__':
    unittest.main()
