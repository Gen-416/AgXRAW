# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 2 of the Rust migration (2026-09-15): the film spatial operators of
dngscan/film_optics.py. Every kernel is pinned bit-identical against its
NumPy body on random inputs (the NumPy body stays the reference).

Platform semantics the kernels rely on, pinned here as well (NumPy 2.5 on
the arm64 reference platform with Accelerate):
  * (n,3)@(3,3) float32/float64 matmul = FMA chain in k order (0,1,2);
  * (n,3)@(3,) matvec = sequential (a*w0 + b*w1) + c*w2, no FMA;
  * einsum("cj,...j->...c") = the same sequential product-sum;
  * np.interp = fma(slope, x - xp[j], fp[j]);
  * np.sum over float64 = NumPy's pairwise order (blocks of 8, split >128).
"""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dngscan import _fast


def _native() -> bool:
    ext = _fast._load_extension()
    return ext is not None and hasattr(ext, "area_decimate_rows")


class _NoNative:
    def __enter__(self):
        self._p = mock.patch.object(_fast, "kernel", lambda name: None)
        self._p.start()
        return self

    def __exit__(self, *exc):
        self._p.stop()
        return False


def _same(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


class PlatformSemantics(unittest.TestCase):
    def test_matmul_is_an_fma_chain(self) -> None:
        rng = np.random.default_rng(1)
        u = rng.uniform(-2, 2, (20_000, 3)); t = rng.uniform(-1, 1, (3, 3))
        acc = np.zeros(u.shape[0])
        out = np.empty((u.shape[0], 3))
        for c in range(3):
            acc[:] = 0.0
            for k in range(3):
                acc = np.fromiter((math.fma(float(x), float(t[c, k]), float(a)) for x, a in zip(u[:, k], acc)), np.float64, u.shape[0])
            out[:, c] = acc
        self.assertTrue(np.array_equal(u @ t.T, out))

    def test_matvec_is_sequential_and_interp_is_fma(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.uniform(0, 4, (20_000, 3)).astype(np.float32); w = np.asarray([0.2627, 0.6780, 0.0593], np.float32)
        self.assertTrue(np.array_equal(a @ w, (a[:, 0] * w[0] + a[:, 1] * w[1]) + a[:, 2] * w[2]))
        xp = np.sort(rng.uniform(-3, 3, 50)); fp = rng.uniform(0, 2, 50); x = rng.uniform(-2.9, 2.9, 5000)
        j = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, xp.size - 2)
        slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
        ref = np.fromiter((math.fma(float(s), float(xv - xp[jj]), float(fp[jj])) for s, xv, jj in zip(slope, x, j)), np.float64, x.size)
        self.assertTrue(np.array_equal(np.interp(x, xp, fp), ref))


@unittest.skipUnless(_native(), "native extension not built")
class SpatialOperators(unittest.TestCase):
    def setUp(self) -> None:
        from dngscan.film_optics_assets import compile_film_optics_plan

        plan = SimpleNamespace(
            curve_preset="portra400", film_mode="full", film_crossover="datasheet",
            film_exposure_ev=0.0, film_print_timing="fixed", film_print_medium="",
            film_print_exposure_ev=0.0, color_head_y=0.0, color_head_m=0.0,
            film_development="measured_default", film_dev_contrast=0.0, film_dev_fog=0.0,
            film_dev_density=0.0, film_compression=0.0, film_compression_knee=2.0,
            film_highlight_density=0.0, film_grain=0.5, film_halation=0.4, film_bloom=0.3,
            film_optics_seed=0, film_media_scatter="declared",
        )
        self.optics = compile_film_optics_plan(plan)
        self.stock = self.optics.stock
        self.bloom = self.optics.capture_bloom
        self.formation = self.optics.print_medium.formation_scatter
        self.rng = np.random.default_rng(3)

    def _both(self, fn, *args, **kw):
        with _NoNative():
            ref = fn(*args, **kw)
        got = fn(*args, **kw)
        return ref, got

    def test_area_decimate_rows(self) -> None:
        from dngscan.film_optics import area_decimate_rows

        for (h, w, oh, ow, c) in ((40, 60, 7, 11, 3), (97, 131, 23, 41, 3), (16, 16, 16, 16, 3), (9, 7, 4, 3, 1)):
            src = np.exp2(self.rng.uniform(-6, 6, (h, w, c))).astype(np.float32)
            for dtype in (np.float64, np.float32):
                acc_ref = np.zeros((oh, ow, c), dtype); acc_got = np.zeros((oh, ow, c), dtype)
                band = 13
                for y0 in range(0, h, band):
                    y1 = min(y0 + band, h)
                    with _NoNative():
                        area_decimate_rows(src[y0:y1], y0, h, w, oh, ow, acc_ref)
                    area_decimate_rows(src[y0:y1], y0, h, w, oh, ow, acc_got)
                self.assertTrue(_same(acc_got, acc_ref), (h, w, oh, ow, c, dtype))

    def test_upsample_rows(self) -> None:
        from dngscan.film_optics import upsample_rows

        for (dh, dw, h, w, c) in ((7, 11, 40, 60, 3), (23, 41, 97, 131, 3), (16, 16, 16, 16, 1)):
            m = self.rng.uniform(-1, 3, (dh, dw, c)).astype(np.float32)
            for (y0, y1) in ((0, h), (5, 17), (h - 3, h)):
                ref, got = self._both(upsample_rows, m, y0, y1, h, w)
                self.assertTrue(_same(got, ref), (dh, dw, h, w, y0, y1))

    def test_gaussian_blur_slabbed_and_small_sigma(self) -> None:
        from dngscan.film_optics import _blur_small_sigma, _gaussian_blur_slabbed

        for (h, w, c, sigma, periodic) in ((33, 47, 3, 1.7, False), (64, 40, 1, 12.3, False), (50, 50, 3, 3.0, True), (300, 9, 3, 0.9, False)):
            img = self.rng.uniform(0, 5, (h, w, c)).astype(np.float32)
            with _NoNative():
                ref = _gaussian_blur_slabbed(img.copy(), sigma, periodic=periodic)
            got = _gaussian_blur_slabbed(img.copy(), sigma, periodic=periodic)
            self.assertTrue(_same(got, ref), (h, w, c, sigma, periodic))
        for (h, w, sigma) in ((33, 47, 0.3), (5, 5, 0.7), (64, 40, 0.95)):
            chan = self.rng.uniform(0, 5, (h, w)).astype(np.float32)
            ref, got = self._both(_blur_small_sigma, chan, sigma)
            self.assertTrue(_same(got, ref), (h, w, sigma))

    def test_halation_gate_return_source(self) -> None:
        from dngscan.film_optics import (
            halation_component_source,
            halation_layer_gate,
            halation_pointwise_return,
        )

        hal = self.stock.halation
        ref3 = np.ones(3, np.float32)
        for shape in ((40, 60, 3), (1, 7, 3), (1, 1, 3)):
            e = (np.exp2(self.rng.uniform(-4, 9, shape)) * 0.18).astype(np.float32)
            e[..., 1] *= self.rng.uniform(0.2, 3, shape[:-1]).astype(np.float32)
            comp = hal.components[0]
            for fn, args in ((halation_layer_gate, (e, ref3, comp.gate_ev)),
                             (halation_pointwise_return, (e, ref3, hal)),
                             (halation_component_source, (e, ref3, comp))):
                ref, got = self._both(fn, *args)
                self.assertTrue(_same(got, ref), (fn.__name__, shape))

    def test_capture_bloom(self) -> None:
        from dngscan.film_optics import (
            capture_bloom_apply_rows,
            capture_bloom_gate,
            capture_bloom_source_rows,
        )

        bloom = self.bloom
        y = np.exp2(self.rng.uniform(-6, 8, (40, 60))).astype(np.float32)
        ref, got = self._both(capture_bloom_gate, y, *bloom.scales[0].gate_ev)
        self.assertTrue(_same(got, ref))
        rows = (np.exp2(self.rng.uniform(-6, 9, (12, 60, 3))) * 0.18).astype(np.float32)
        ref, got = self._both(capture_bloom_source_rows, rows, bloom)
        self.assertTrue(_same(got, ref))
        glow = np.exp2(self.rng.uniform(-8, 2, (7, 11, 3))).astype(np.float32)
        flat = rows.reshape(-1, 3)
        for amount in (0.3, 1.0):
            ref, got = self._both(capture_bloom_apply_rows, flat, glow, 5, 17, 40, 60, bloom, amount)
            self.assertTrue(_same(got, ref), amount)

    def test_scatter_mix(self) -> None:
        from dngscan.film_optics import apply_scatter_mix

        for kern in (self.stock.emulsion_scatter, self.formation):
            if kern is None:
                continue
            for (h, w, mm_per_px) in ((24, 40, 36.0 / 6016.0), (30, 30, 36.0 / 1600.0), (9, 70, 36.0 / 20000.0)):
                img = (np.exp2(self.rng.uniform(-6, 7, (h, w, 3))) * 0.18).astype(np.float32)
                ref, got = self._both(apply_scatter_mix, img, mm_per_px, kern)
                self.assertTrue(_same(got, ref), (h, w, mm_per_px))

    def test_sample_field_and_density_grain(self) -> None:
        from dngscan.film_optics import (
            FilmGeometry,
            _grain_ii_for,
            apply_density_grain,
            realization_phases,
            sample_field,
        )

        grain = self.stock.grain
        master = _grain_ii_for(grain, 0)
        gh, gw = master.shape[0] - 1, master.shape[1] - 1
        for (h, w) in ((40, 60), (61, 37), (128, 192)):
            geo = FilmGeometry.fit(h, w)
            for seed in (0, 7, 0x50524E54 ^ 12345):
                phase = realization_phases(seed, gh, gw)
                for band in ((0, h), (11, 29)):
                    sub = geo.rows(*band)
                    ref, got = self._both(sample_field, master, sub, phase)
                    self.assertTrue(_same(got, ref), (h, w, seed, band))
            amounts = self.rng.uniform(-0.1, 2.5, (h * w, 3))
            for seed in (0, 99):
                ref, got = self._both(apply_density_grain, amounts, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0], geo, grain, 0.7, seed)
                self.assertTrue(_same(got, ref), (h, w, seed))

    def test_halation_reinject_rows(self) -> None:
        from dngscan.film_optics import halation_reinject_rows

        hal = self.stock.halation
        h, w = 40, 60
        log_e = self.rng.uniform(-3, 2, (h * w, 3))
        spread = np.exp2(self.rng.uniform(-8, 1, (7, 11, 3))).astype(np.float32)
        ref3 = np.ones(3, np.float32)
        give = (np.exp2(self.rng.uniform(-3, 6, (h * w, 3)))).astype(np.float32)
        for (y0, y1) in ((0, h), (5, 17)):
            n = (y1 - y0) * w
            for gl in (None, give[:n]):
                ref, got = self._both(halation_reinject_rows, log_e[:n], spread, ref3, y0, y1, h, w, hal, 0.6, give_lin=gl)
                self.assertTrue(_same(got, ref), (y0, y1, gl is None))


if __name__ == "__main__":
    unittest.main()
