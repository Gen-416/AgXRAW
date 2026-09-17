# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression gates for native streaming, writable views and boundary inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from dngscan import _fast

ROOT = Path(__file__).resolve().parents[1]
NATIVE = _fast._load_extension() is not None


def _opcode(h, w, **changes):
    fields = dict(
        top=0, left=0, bottom=h, right=w, row_pitch=1, col_pitch=1,
        origin_v=0.0, origin_h=0.0, spacing_v=1.0, spacing_h=1.0,
        points_v=2, points_h=2, gains=np.full((2, 2, 1), 1.1),
        plane=0, planes=1,
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


class HdrValidation(unittest.TestCase):
    def test_every_nonfinite_rgb_component_is_rejected(self):
        from dngscan import gainmap

        for source in (0, 1):
            for channel in range(3):
                for value in (np.nan, np.inf, -np.inf):
                    # < 1% invalid: an upper percentile alone cannot catch it.
                    images = [np.full((521, 17, 3), 0.5, np.float16) for _ in range(2)]
                    images[source][-1, -1, channel] = value
                    with self.subTest(source=source, channel=channel, value=value):
                        with mock.patch.object(_fast, "kernel", return_value=None):
                            ref = gainmap._roundtrip_error_arrays(*images)
                        self.assertFalse(gainmap._hdr_roundtrip_is_acceptable(ref))
                        if NATIVE:
                            native = _fast._load_extension().hdr_roundtrip_metrics(*images)
                            self.assertEqual(native, ref)

    def test_empty_rendition_is_rejected(self):
        from dngscan import gainmap

        for shape in ((0, 8, 3), (8, 0, 3)):
            a = np.empty(shape, np.float16)
            with mock.patch.object(_fast, "kernel", return_value=None):
                ref = gainmap._roundtrip_error_arrays(a, a)
            self.assertFalse(gainmap._hdr_roundtrip_is_acceptable(ref))
            if NATIVE:
                self.assertEqual(_fast._load_extension().hdr_roundtrip_metrics(a, a), ref)


@unittest.skipUnless(NATIVE, "native extension not built")
class NativePipeline(unittest.TestCase):
    def tearDown(self):
        _fast.set_thread_budget(0)

    def test_overlapping_opcodes_on_transposed_and_reversed_views(self):
        from dngscan.raw_io import _apply_gain_maps_mosaic

        rng = np.random.default_rng(17)
        original = rng.integers(0, 16000, (45, 67), dtype=np.uint16)
        colors = rng.integers(0, 4, original.shape, dtype=np.uint8)
        for select in (lambda a: a[2:-3:2, 5:-4:3], lambda a: a.T[::-1, 2:-3]):
            a, b = original.copy(), original.copy()
            view_a, view_b, c = select(a), select(b), select(colors)
            h, w = view_a.shape
            maps = [_opcode(h, w), _opcode(h, w, top=1, col_pitch=2, gains=np.full((2, 2, 1), 0.91))]
            args = (maps, [400., 500., 600., 450.], 15000, [15000., 14500., 15500., 14900.])
            with mock.patch.object(_fast, "kernel", return_value=None):
                _apply_gain_maps_mosaic(SimpleNamespace(raw_image_visible=view_a, raw_colors_visible=c), *args)
            _apply_gain_maps_mosaic(SimpleNamespace(raw_image_visible=view_b, raw_colors_visible=c), *args)
            np.testing.assert_array_equal(a, b)
            self.assertFalse(np.array_equal(b, original))

    def test_invalid_gainmap_cannot_partially_modify_mosaic(self):
        ext = _fast._load_extension()
        for changes in (dict(top=-1), dict(col_pitch=0), dict(points_v=0, gains=np.empty((0, 2, 1))),
                        dict(gains=np.empty((2, 2, 0))), dict(points_h=3)):
            img = np.full((7, 9), 1000, np.uint16)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ext.apply_gain_map_mosaic(img, np.zeros_like(img, dtype=np.uint8), _opcode(7, 9, **changes), [512.], [16383.])
            np.testing.assert_array_equal(img, 1000)

    def test_feather_budgets_and_singleton_axes(self):
        from dngscan.raw_io import _feather_masks_f16

        rng = np.random.default_rng(18)
        for shape in ((1, 1, 3), (1, 13, 4), (11, 1, 3), (513, 263, 3)):
            mask = rng.uniform(-0.3, 1.3, shape).astype(np.float32)
            with mock.patch.object(_fast, "kernel", return_value=None):
                ref = _feather_masks_f16(mask)
            for workers in (1, 3, 8):
                _fast.set_thread_budget(workers)
                np.testing.assert_array_equal(_feather_masks_f16(mask), ref)

    def test_streamed_metrics_across_worker_batches_and_strides(self):
        from dngscan import gainmap

        rng = np.random.default_rng(19)
        rgba = rng.uniform(0, 3, (1609, 41, 4)).astype(np.float16)
        # Alpha is deliberately invalid and must not enter RGB validation.
        rgba[..., 3] = np.nan
        rgb = (rgba[..., :3].astype(np.float32) * rng.uniform(0.8, 1.2, (1609, 41, 3))).astype(np.float16)
        decoded = rng.integers(0, 256, (1609, 41, 3), dtype=np.uint8)
        intended = rng.integers(0, 256, decoded.shape, dtype=np.uint8)
        for select in (lambda a: a, lambda a: a[::-1, ::-2], lambda a: a.transpose(1, 0, 2)):
            a, e = select(rgba)[..., :3], select(rgb)
            with mock.patch.object(_fast, "kernel", return_value=None):
                ref_hdr = gainmap._roundtrip_error_arrays(a, e)
                ref_base = gainmap._base_roundtrip_error_arrays(decoded, intended)
            for workers in (1, 3):
                _fast.set_thread_budget(workers)
                self.assertEqual(_fast._load_extension().hdr_roundtrip_metrics(a, e), ref_hdr)
                self.assertEqual(_fast._load_extension().hdr_roundtrip_metrics(select(rgba), e), ref_hdr)
                self.assertEqual(_fast._load_extension().base_roundtrip_metrics(decoded, intended), ref_base)

    def test_base_histogram_percentile_interpolates_rank_boundary(self):
        from dngscan import gainmap

        for n in (99, 100, 101, 102, 257):
            decoded = np.zeros((1, n, 3), np.uint8)
            decoded[0, -1] = 255
            decoded[0, -2] = 17
            intended = np.zeros_like(decoded)
            with mock.patch.object(_fast, "kernel", return_value=None):
                ref = gainmap._base_roundtrip_error_arrays(decoded, intended)
            self.assertEqual(_fast._load_extension().base_roundtrip_metrics(decoded, intended), ref)

    def test_singleton_blurs_finish_and_match_numpy(self):
        # A subprocess timeout makes the former reflect_index loop a failure
        # rather than hanging the whole test runner indefinitely.
        code = '''
import numpy as np
from unittest import mock
from dngscan import _fast
from dngscan.film_optics import _blur_small_sigma, _gaussian_blur_slabbed
rng = np.random.default_rng(20)
for h, w in ((1, 1), (1, 9), (9, 1), (2, 3)):
    a = rng.uniform(0, 5, (h, w, 3)).astype(np.float32)
    for sigma in (0.3, 1.7, 7.0):
        for periodic in (False, True):
            with mock.patch.object(_fast, "kernel", return_value=None):
                ref = _gaussian_blur_slabbed(a.copy(), sigma, periodic=periodic)
            np.testing.assert_array_equal(_gaussian_blur_slabbed(a.copy(), sigma, periodic=periodic), ref)
    for sigma in (0.3, 0.9):
        with mock.patch.object(_fast, "kernel", return_value=None):
            ref = _blur_small_sigma(a[..., 0], sigma)
        np.testing.assert_array_equal(_blur_small_sigma(a[..., 0], sigma), ref)
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                env={**os.environ, "DNGSCAN_FAST": "1"},
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(NATIVE and sys.platform in ("darwin", "linux"), "native RSS gate needs macOS/Linux")
class NativeMemory(unittest.TestCase):
    def test_transient_memory_is_bounded(self):
        # Fresh processes exclude previous allocations/allocator high-water
        # marks. Limits include output buffers, with headroom for runtime noise.
        for kernel, limit in (("gain", 32), ("feather", 80), ("hdr", 110), ("base", 32)):
            with self.subTest(kernel=kernel):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "tools/benchmark_native_memory.py"),
                     "--kernel", kernel, "--height", "4096", "--width", "2048", "--threads", "1"],
                    cwd=ROOT, env={**os.environ, "DNGSCAN_FAST": "1"},
                    capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads(result.stdout)
                self.assertLess(record["extra_peak_mib"], limit, record)


if __name__ == "__main__":
    unittest.main()
