# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact spatial kernels with borrowed inputs, row scratch and one thread pool."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest import mock

import numpy as np

from dngscan import _fast, film_optics

ROOT = Path(__file__).resolve().parents[1]
NATIVE = _fast._load_extension() is not None


@unittest.skipUnless(NATIVE, "native extension not built")
class SpatialStreaming(unittest.TestCase):
    def tearDown(self):
        _fast.set_thread_budget(0)

    def test_gaussian_borrowed_views_are_unchanged_and_exact(self):
        rng = np.random.default_rng(22)
        base = rng.uniform(-2, 5, (39, 53, 4)).astype(np.float32)
        views = (base[..., :3], base[::-1, ::2, ::-1], base.transpose(1, 0, 2),
                 base[:1, :, :1], base[:, :1, :3], base[:1, :1, :3],
                 np.broadcast_to(base[0:1, 0:1, :3], (9, 13, 3)))
        ext = _fast._load_extension()
        for view in views:
            view.flags.writeable = False
            original = view.copy()
            for sigma in (0.3, 1.7, 20.0):
                for periodic in (False, True):
                    with self.subTest(shape=view.shape, strides=view.strides, sigma=sigma, periodic=periodic):
                        with mock.patch.object(_fast, "kernel", return_value=None):
                            ref = film_optics._gaussian_blur_slabbed(view.copy(), sigma, periodic=periodic)
                        for workers in (1, 3):
                            _fast.set_thread_budget(workers)
                            out = ext.gaussian_blur_slabbed(view, sigma, periodic)
                            np.testing.assert_array_equal(out, ref)
                            self.assertFalse(np.shares_memory(out, view))
                        np.testing.assert_array_equal(view, original)

    def test_blur_bounded_owns_output_even_for_noop(self):
        a = np.arange(60, dtype=np.float32).reshape(4, 5, 3)
        a.flags.writeable = False
        for sigma in (0.0, -1.0, 1.7):
            with mock.patch.object(_fast, "kernel", return_value=None):
                ref = film_optics._blur_bounded(a, sigma)
            out = film_optics._blur_bounded(a, sigma)
            self.assertFalse(np.shares_memory(out, a))
            np.testing.assert_array_equal(out, ref)
        np.testing.assert_array_equal(a.reshape(-1), np.arange(60, dtype=np.float32))

    def test_small_sigma_strided_planes_and_nonfinite_propagation(self):
        rng = np.random.default_rng(23)
        a = rng.uniform(-2, 5, (67, 45, 3)).astype(np.float32)
        a[20, 31, 1] = np.nan
        a[9, 14, 1] = np.inf
        a[45, 1, 1] = -np.inf
        ext = _fast._load_extension()
        for plane in (a[..., 1], a[::-1, ::2, 1], a[..., 1].T):
            original = plane.copy()
            for sigma in (0.0, 0.3, 0.7, 0.95):
                with np.errstate(invalid="ignore"):
                    with mock.patch.object(_fast, "kernel", return_value=None):
                        ref = film_optics._blur_small_sigma(plane, sigma)
                    for workers in (1, 4):
                        _fast.set_thread_budget(workers)
                        np.testing.assert_array_equal(ext.blur_small_sigma(plane, sigma), ref)
                np.testing.assert_array_equal(plane, original)

    def test_empty_spatial_inputs(self):
        ext = _fast._load_extension()
        for shape in ((0, 5, 3), (5, 0, 3), (5, 7, 0)):
            a = np.empty(shape, np.float32)
            self.assertEqual(ext.gaussian_blur_slabbed(a, 1.7, False).shape, shape)
        for shape in ((0, 5), (5, 0)):
            self.assertEqual(ext.blur_small_sigma(np.empty(shape, np.float32), 0.7).shape, shape)
        for shape in ((0, 5, 3), (5, 0, 3)):
            self.assertEqual(ext.apply_scatter_mix(np.empty(shape, np.float32), [(0.3, [(0.7, 1.)])] * 3).shape, shape)

    def test_area_promotes_samples_without_rounding_or_mutation(self):
        rng = np.random.default_rng(24)
        ext = _fast._load_extension()
        for dtype in (np.float16, np.float32, np.float64, np.int16, np.dtype(">f4")):
            base = rng.uniform(-2, 5, (83, 97, 3)).astype(dtype)
            for rows in (base, base[::-1, ::-2], base.transpose(1, 0, 2)):
                original = rows.copy()
                h, w, _ = rows.shape
                rows.flags.writeable = False
                for acc_dtype in (np.float32, np.float64):
                    for band in (h, 1, 17):
                        with self.subTest(dtype=dtype, strides=rows.strides, acc_dtype=acc_dtype, band=band):
                            ref = np.full((19, 23, 3), 0.125, acc_dtype)
                            got = ref.copy()
                            for y0 in range(0, h, band):
                                part = rows[y0:y0 + band]
                                with mock.patch.object(_fast, "kernel", return_value=None):
                                    film_optics.area_decimate_rows(part, y0, h, w, 19, 23, ref)
                                ext.area_decimate_rows(part, y0, h, w, 19, 23, got)
                            np.testing.assert_array_equal(got, ref)
                np.testing.assert_array_equal(rows, original)
            with mock.patch.object(_fast, "kernel", return_value=None):
                ref = film_optics.area_decimate(base, 19, 23)
            np.testing.assert_array_equal(film_optics.area_decimate(base, 19, 23), ref)

    def test_invalid_area_geometry_leaves_accumulator_unchanged(self):
        ext = _fast._load_extension()
        rows = np.ones((5, 7, 3), np.float32)
        for y0, h, oh, ow in ((-1, 5, 2, 3), (1, 5, 2, 3), (0, 5, 0, 3), (0, 5, 2, 0)):
            acc = np.full((2, 3, 3), 0.125, np.float32)
            with self.assertRaises((ValueError, OverflowError)):
                ext.area_decimate_rows(rows, y0, h, 7, oh, ow, acc)
            np.testing.assert_array_equal(acc, 0.125)

    def test_scatter_fused_rows_match_ordered_component_reference(self):
        rng = np.random.default_rng(25)
        base = rng.uniform(-0.5, 5, (73, 97, 3)).astype(np.float32)
        configurations = (
            [(0.3, [(0.3, 0.4), (1.7, 0.6)]), (0.4, [(0.9, 1.)]), (0.2, [])],
            [(0.7, [(4., 0.2), (1.7, 0.3), (0.1, 0.5)])] * 3,
        )
        for a in (base, base[::-1, ::2], base.transpose(1, 0, 2), base[:1], base[:, :1]):
            original = a.copy()
            for configs in configurations:
                ref = np.empty_like(a)
                with mock.patch.object(_fast, "kernel", return_value=None):
                    for ch, (strength, comps) in enumerate(configs):
                        # Copy the plane so the in-place NumPy Gaussian
                        # cannot mutate subsequent components' source.
                        acc = a[..., ch] * np.float32(1. - strength * sum(w for _, w in comps))
                        for sigma, weight in comps:
                            if sigma < 1:
                                blur = film_optics._blur_small_sigma(a[..., ch], sigma)
                            else:
                                blur = film_optics._gaussian_blur_slabbed(a[..., ch:ch + 1].copy(), sigma)[..., 0]
                            acc += np.float32(strength * weight) * blur
                        ref[..., ch] = acc
                for workers in (1, 2, 8):
                    _fast.set_thread_budget(workers)
                    np.testing.assert_array_equal(_fast._load_extension().apply_scatter_mix(a, configs), ref)
                np.testing.assert_array_equal(a, original)


@unittest.skipUnless(NATIVE and sys.platform == "darwin", "native libproc sampling needs macOS")
class SpatialThreadBudget(unittest.TestCase):
    def test_scatter_has_one_budgeted_pool(self):
        from tests.test_scheduler_s3 import _thread_count

        code = '''
import sys
import numpy as np
from dngscan import _fast
_fast.set_thread_budget(int(sys.argv[1]))
img = np.full((401, 601, 3), 0.5, np.float32)
chans = [(0.3, [(1.7, 1.)])] * 3
ext = _fast._load_extension()
print("ready", flush=True)
sys.stdin.readline()
for _ in range(30):
    ext.apply_scatter_mix(img, chans)
'''
        for workers in (1, 2, 4):
            with subprocess.Popen([sys.executable, "-c", code, str(workers)], cwd=ROOT,
                                  env={**os.environ, "DNGSCAN_FAST": "1"}, text=True,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
                try:
                    self.assertEqual(proc.stdout.readline().strip(), "ready")
                    baseline = peak = _thread_count(proc.pid)
                    self.assertGreater(baseline, 0)
                    proc.stdin.write("go\n")
                    proc.stdin.flush()
                    deadline = time.monotonic() + 30
                    while proc.poll() is None and time.monotonic() < deadline:
                        peak = max(peak, _thread_count(proc.pid))
                        time.sleep(0.001)
                    if proc.poll() is None:
                        self.fail("scatter kernel exceeded 30-second timeout")
                    _, err = proc.communicate(timeout=5)
                    self.assertEqual(proc.returncode, 0, err)
                    allowance = 0 if workers == 1 else workers
                    self.assertLessEqual(peak, baseline + allowance + 1,
                                         (workers, baseline, peak))
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()


if __name__ == "__main__":
    unittest.main()
