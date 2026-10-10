# SPDX-License-Identifier: GPL-3.0-or-later
"""ActiveArea opcodes compared with equivalent stored samples through LibRaw."""
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dngscan import _fast, dng_opcodes, metadata, raw_io
from dngscan.noise_propagation import coarse_spatial_moments
from tests.test_pipeline_corrections import write_sensor_dng


def write_active_dng(path, area, pixels, *, opcodes=None, linear=False, extra_tags=None):
    """Extend the shared uncompressed fixture without requiring a TIFF library."""
    write_sensor_dng(path, signal=1000, linear=linear)
    original = path.read_bytes()
    count = struct.unpack_from('<H', original, 8)[0]
    tags = {}
    for i in range(count):
        tag, typ, n = struct.unpack_from('<HHL', original, 10 + 12*i)
        value = original[18 + 12*i:22 + 12*i]
        size = metadata._TYPE_SIZES[typ]*n
        pos = struct.unpack('<L', value)[0] if size > 4 else 0
        tags[tag] = (typ, n, original[pos:pos+size] if size > 4 else value[:size])
    tags[50829] = (4, 4, struct.pack('<4L', *area))
    for tag, value in ((256,pixels.shape[1]),(257,pixels.shape[0]),
                       (278,pixels.shape[0]),(279,pixels.nbytes)):
        tags[tag] = (4, 1, struct.pack('<L', value))
    tags[50720] = (4, 2, struct.pack('<2L', area[3]-area[1], area[2]-area[0]))
    tags[274] = (3, 1, struct.pack('<H', 1))
    for tag in (50706, 50707):
        tags[tag] = (1, 4, bytes([1, 6, 0, 0]))
    tags.update(extra_tags or {})
    for tag, operations in (opcodes or {}).items():
        blob = struct.pack('>L', len(operations)) + b''.join(
            struct.pack('>4L', oid, 0x01060000, flags, len(payload))+payload
            for oid, flags, payload in operations)
        tags[tag] = (7, len(blob), blob)
    body = bytearray()
    entries = []
    start = 8 + 2 + 12*len(tags) + 4
    for tag, (typ, n, data) in sorted(tags.items()):
        if tag == 273:
            entries.append((tag, typ, n, None))
        elif len(data) <= 4:
            entries.append((tag, typ, n, data.ljust(4, b'\0')))
        else:
            entries.append((tag, typ, n, struct.pack('<L', start+len(body))))
            body += data
            if len(body) % 2:
                body += b'\0'
    headers = b''.join(struct.pack('<HHL', tag, typ, n) +
        (value if value is not None else struct.pack('<L', start+len(body)))
        for tag, typ, n, value in entries)
    path.write_bytes(b'II'+struct.pack('<HLH', 42, 8, len(tags))+headers+bytes(4)+
                     body+np.asarray(pixels, dtype='<u2').tobytes())


def gain_payload(shape, gains, pitch=1):
    grid = np.asarray(gains, dtype=np.float32)
    return (struct.pack('>10L', 0, 0, *shape, 0, 1, pitch, pitch, *grid.shape)
            + struct.pack('>4dL', 1., 1., 0., 0., 1)
            + grid.astype('>f4').tobytes())


class ActiveAreaGainMapTests(unittest.TestCase):
    def test_phase_and_gradient_match_equivalent_stored_dng(self):
        with tempfile.TemporaryDirectory() as td:
            actual_path, ref_path = Path(td)/'map.dng', Path(td)/'stored.dng'
            for fast in ('0', '1') if _fast.available() else ('0',):
                for origin in ((0, 0), (2, 2), (0, 1), (1, 0), (1, 1)):
                    for gradient in (False, True):
                        height = width = 64 if gradient else 126
                        top, left = origin
                        area = (top, left, top+height, left+width)
                        source = np.full((128, 128), 1024 if gradient else 1000, np.uint16)
                        stored = source.copy()
                        if gradient:
                            stored[top:top+height, left:left+width] = 1032 + 16*np.arange(width)
                            payload = gain_payload((height, width), [[1., 2.]], 1)
                        else:
                            stored[top:top+height:2, left:left+width:2] = 2000
                            payload = gain_payload((height, width), [[2.]], 2)
                        write_active_dng(actual_path, area, source, opcodes={51009:[(9, 0, payload)]})
                        write_active_dng(ref_path, area, stored)
                        for half in (False, True):
                            with self.subTest(fast=fast, origin=origin, gradient=gradient, half=half), \
                                 patch.dict(os.environ, DNGSCAN_FAST=fast):
                                actual = raw_io.load_raw(actual_path, scene_half_size=half)
                                expected = raw_io.load_raw(ref_path, scene_half_size=half)
                                np.testing.assert_array_equal(actual.scene_rec2020_render,
                                                              expected.scene_rec2020_render)
                                np.testing.assert_array_equal(actual.raw_image, 1024 if gradient else 1000)

    def test_gain_clipping_and_noise_moments_keep_the_same_phase(self):
        op = metadata._parse_gain_map_payload(gain_payload((8, 8), [[2.]], 2))
        for fast in ('0', '1') if _fast.available() else ('0',):
            image = np.full((7, 7), 3000, np.uint16)
            loss = np.zeros_like(image, dtype=np.uint8)
            raw = SimpleNamespace(raw_image_visible=image, raw_colors_visible=np.zeros_like(loss))
            with patch.dict(os.environ, DNGSCAN_FAST=fast):
                raw_io._apply_gain_maps_mosaic(raw, [op], [0.], 4095, loss_mask=loss,
                                              image_origin=(1, 1), image_shape=(8, 8))
            expected = np.zeros((7, 7), np.uint8)
            expected[1::2, 1::2] = 1
            np.testing.assert_array_equal(loss, expected)
            np.testing.assert_array_equal(image, np.where(expected, 4095, 3000))
        from dataclasses import asdict
        descriptor = dict(full_sensor_shape=[7, 7], sensor_crop=[0, 0, 7, 7],
                          gain_maps=[asdict(op)], gain_map_origin=[1, 1], gain_map_shape=[8, 8])
        # Visible phase (1,1) is the authored ActiveArea phase (0,0).
        mean, second, _, _ = coarse_spatial_moments(descriptor, (2, 2), 'BGGR')
        np.testing.assert_array_equal(mean[..., 0], 2.)
        np.testing.assert_array_equal(second[..., 0], 4.)
        np.testing.assert_array_equal(mean[..., 1:], 1.)

    def test_optional_warp2_consumes_only_one_compatibility_warp(self):
        warp2 = struct.pack('>L21dL', 1, 1., *([0.]*16), 0., 1., .5, .5, 0)
        old = lambda scale: struct.pack('>L8d', 1, scale, 0., 0., 0., 0., 0., .5, .5)
        pixels = np.broadcast_to(np.arange(128, dtype=np.uint16)[None, :, None]*20+500,
                                 (128, 128, 3)).copy()
        with tempfile.TemporaryDirectory() as td:
            p, ref = Path(td)/'compat.dng', Path(td)/'reference.dng'
            write_active_dng(p, (0, 0, 128, 128), pixels, linear=True,
                opcodes={51022:[(14, 1, warp2), (1, 0, old(1.)), (1, 0, old(.8))]})
            write_active_dng(ref, (0, 0, 128, 128), pixels, linear=True,
                opcodes={51022:[(1, 0, old(.8))]})
            self.assertEqual(dng_opcodes.read_plan(p).names, ['WarpRectilinear2', 'WarpRectilinear'])
            for fast in ('0', '1') if _fast.available() else ('0',):
                with patch.dict(os.environ, DNGSCAN_FAST=fast):
                    np.testing.assert_array_equal(raw_io.load_raw(p).scene_rec2020_render,
                                                  raw_io.load_raw(ref).scene_rec2020_render)

    def test_active_area_trim_and_point_rows_keep_authored_coordinates(self):
        from dngscan import dng_point_ops
        image = np.ones((5, 5), np.float32)
        op = dng_point_ops.PointOp(12, (0,0,6,6,0,1,2,2), (2.,3.,4.), 2)
        dng_point_ops.apply(image, op, white=10., image_origin=(1,1))
        expected = np.ones((5,5), np.float32)
        expected[1,1::2] = 3.
        expected[3,1::2] = 4.
        np.testing.assert_array_equal(image, expected)
        with tempfile.TemporaryDirectory() as td:
            p, ref = Path(td)/'trim.dng', Path(td)/'reference.dng'
            pixels = np.full((128,128), 1000, np.uint16)
            write_active_dng(p, (1,1,127,127), pixels,
                opcodes={51022:[(6,0,struct.pack('>4l',0,0,126,126))]})
            write_active_dng(ref, (1,1,127,127), pixels)
            np.testing.assert_array_equal(raw_io.load_raw(p).scene_rec2020_render,
                                          raw_io.load_raw(ref).scene_rec2020_render)

    def test_nonzero_default_crop_and_noise_descriptor_survive_cache(self):
        from dngscan.analysis import analyze
        from dngscan.gui.preview_cache import build_proxy_entry, _read_disk_entry, _write_disk_entry
        source = np.full((128,128), 1000, np.uint16)
        stored = source.copy()
        stored[1:127:2,1:127:2] = 2000
        extras = {50719:(4,2,struct.pack('<2L',7,9)),
                  50720:(4,2,struct.pack('<2L',90,80)),
                  51041:(12,2,struct.pack('<2d',1e-4,4e-6))}
        with tempfile.TemporaryDirectory() as td:
            p, ref = Path(td)/'map.dng', Path(td)/'reference.dng'
            write_active_dng(p, (1,1,127,127), source, extra_tags=extras,
                opcodes={51009:[(9,0,gain_payload((126,126),[[2.]],2))]})
            write_active_dng(ref, (1,1,127,127), stored, extra_tags=extras)
            actual, expected = raw_io.load_raw(p), raw_io.load_raw(ref)
            self.assertEqual(actual.scene_rec2020_render.shape[:2], (80,90))
            np.testing.assert_array_equal(actual.scene_rec2020_render, expected.scene_rec2020_render)
            np.testing.assert_array_equal(actual.processing_clip_masks, expected.processing_clip_masks)
            self.assertEqual(actual.noise_decode['gain_map_origin'], [1,1])
            self.assertEqual(actual.noise_decode['effective_sensor_crop'], [8.,6.,80.,90.])
            analysis, _, _ = analyze(actual, 4, diagnostics=False)
            entry = build_proxy_entry(actual, analysis, 64)
            cache_path = Path(td)/'cached.npz'
            _write_disk_entry(cache_path, entry)
            restored = _read_disk_entry(cache_path, p, require_guidance=True)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.bundle.noise_decode, entry.bundle.noise_decode)
