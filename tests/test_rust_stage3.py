# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 3 of the Rust migration (2026-09-15): the film v2 core per-pixel
chain (film_v2_math / film_develop). The NumPy bodies stay the reference;
every kernel is pinned bit-identical against them on random inputs and on
the shipped stock assets, and the whole apply_film_core path (negative and
reversal branches, with compression and inter-image amplification) is
pinned array-equal native vs NumPy.

Platform semantics replicated (NumPy 2.5 / Accelerate, macOS arm64): the
(n,3)@(3,3) float64 matmul is an FMA chain in k order; (3,3)@(3,) and
rgb@luma are sequential (a*w0+b*w1)+c*w2; np.interp is fma(slope, x-xp[j],
fp[j]); np.mean over 3 is ((a+b)+c)/3; _tetrahedral's (g - i0) promotes
to float64 before the float32 cast; float32 log2 and float64 log10/exp2
are libm.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dngscan import _fast


def _native_available() -> bool:
    ext = _fast._load_extension()
    return ext is not None and hasattr(ext, "tetrahedral")


class _NoNative:
    def __enter__(self):
        self._p = mock.patch.object(_fast, "kernel", lambda name: None)
        self._p.start()
        return self

    def __exit__(self, *exc):
        self._p.stop()
        return False


def _plan(preset: str, **kw):
    base = dict(
        curve_preset=preset, film_mode="full", film_crossover="datasheet",
        film_exposure_ev=0.0, film_print_timing="fixed",
        film_print_medium="", film_print_exposure_ev=0.0,
        color_head_y=0.0, color_head_m=0.0,
        film_development="measured_default",
        film_dev_contrast=0.0, film_dev_fog=0.0, film_dev_density=0.0,
        film_compression=0.0, film_compression_knee=2.0,
        film_highlight_density=0.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _stocks():
    from tests.test_film_v2_assets import _stock_files

    return _stock_files()


def _scene(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ev = rng.uniform(-8.0, 6.0, size=(n, 3))
    rgb = (0.18 * np.exp2(ev)).astype(np.float32)
    # Signed / zero / tiny components are legal inputs (F4).
    rgb[: n // 8] *= rng.choice([-1.0, 1.0], size=(n // 8, 3)).astype(np.float32)
    rgb[n // 8: n // 4] = 0.0
    return np.ascontiguousarray(rgb)


@unittest.skipUnless(_native_available(), "native stage-3 kernels unavailable")
class Stage3KernelParity(unittest.TestCase):
    def setUp(self) -> None:
        from dngscan.film_develop import _load_v2

        self.stock_name = next(
            s for s in _stocks() if s.startswith(("portra", "pro400h", "c200", "gold"))
        )
        self.stock, self.media = _load_v2(self.stock_name)
        self.chroma_stock = next(
            (_load_v2(s)[0] for s in _stocks() if _load_v2(s)[0].get("chroma_delta_lut") is not None),
            None,
        )

    def _both(self, fn, *a, **k):
        with _NoNative():
            ref = fn(*a, **k)
        return ref, fn(*a, **k)

    def test_layer_log_exposure(self) -> None:
        from dngscan.film_v2_math import layer_log_exposure

        rgb = _scene(20011, 1)
        ref, got = self._both(layer_log_exposure, rgb, self.stock["observer"])
        self.assertEqual(got.dtype, np.float64)
        np.testing.assert_array_equal(got, ref)

    def test_chroma_field_log_exposure(self) -> None:
        if self.chroma_stock is None:
            self.skipTest("no stock ships a chroma field")
        from dngscan.film_v2_math import chroma_field_log_exposure

        s = self.chroma_stock
        rgb = _scene(20011, 2)
        ref, got = self._both(
            chroma_field_log_exposure, rgb, s["chroma_delta_lut"], s["chroma_domain"],
            s["chroma_xyz_from_rec2020"], s["observer"],
        )
        np.testing.assert_array_equal(got, ref)

    def test_characteristic_amounts(self) -> None:
        from dngscan.film_v2_math import characteristic_amounts, layer_log_exposure

        with _NoNative():
            log_e = layer_log_exposure(_scene(20011, 3), self.stock["observer"])
        for off in (0.0, 0.7, -1.5):
            ref, got = self._both(
                characteristic_amounts, log_e, self.stock["char_le"], self.stock["char_amounts"], off
            )
            np.testing.assert_array_equal(got, ref)

    def test_film_compression_ev(self) -> None:
        from dngscan.film_v2_math import film_compression_ev

        rgb = _scene(20011, 4).astype(np.float64)
        for kw in (
            dict(impact=1.0, knee_ev=2.0),
            dict(impact=0.35, knee_ev=1.0, width_ev=3.0, highlight_color_density=0.6),
        ):
            ref, got = self._both(film_compression_ev, rgb, **kw)
            np.testing.assert_array_equal(got, ref)

    def test_tetrahedral_random_and_asset(self) -> None:
        from dngscan.film_develop import _tetrahedral

        rng = np.random.default_rng(5)
        n = 9
        lut = rng.standard_normal((n, n, n, 3)).astype(np.float32)
        u = rng.uniform(-0.1, 1.1, size=(30007, 3)).astype(np.float32)
        u[:64] = rng.integers(0, n, size=(64, 3)) / (n - 1)  # exact lattice hits
        ref, got = self._both(_tetrahedral, lut, u, n)
        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_array_equal(got, ref)
        # The shipped assets: every B2 volume and every print-state B1 LUT.
        for ps, b2 in self.media.values():
            ref, got = self._both(_tetrahedral, b2["volume"], u, b2["n"])
            np.testing.assert_array_equal(got, ref)
            if ps is not None:
                ref, got = self._both(_tetrahedral, ps["b1"], u, ps["n"])
                np.testing.assert_array_equal(got, ref)

    def test_interimage_amplify(self) -> None:
        from dngscan.film_v2_math import characteristic_amounts, layer_log_exposure

        ext = _fast._load_extension()
        s = self.stock
        with _NoNative():
            log_e = layer_log_exposure(_scene(20011, 6), s["observer"])
            amounts = characteristic_amounts(log_e, s["char_le"], s["char_amounts"])
            le_mean = np.mean(log_e, axis=1, keepdims=True)
            neutral = characteristic_amounts(np.repeat(le_mean, 3, axis=1), s["char_le"], s["char_amounts"])
        table = np.asarray(s["char_amounts"], dtype=np.float64)
        rail_lo = np.maximum(table.min(axis=0)[None, :], np.asarray(s["lo"], dtype=np.float64)[None, :])
        rail_hi = np.minimum(table.max(axis=0)[None, :], np.asarray(s["hi"], dtype=np.float64)[None, :])
        beta = 0.45
        d = amounts - neutral
        head = np.maximum(np.where(d >= 0.0, rail_hi - neutral, neutral - rail_lo), 1e-9)
        t = np.minimum(np.abs(d) / head, 1.0)
        t = (1.0 + beta) * t / (1.0 + beta * t)
        ref = neutral + np.sign(d) * head * t
        got = ext.interimage_amplify(
            amounts, log_e, [float(v) for v in s["char_le"]], table,
            [float(v) for v in rail_lo.reshape(-1)], [float(v) for v in rail_hi.reshape(-1)], beta,
        )
        np.testing.assert_array_equal(got, ref)

    def test_cast_divide(self) -> None:
        from dngscan.color import EPS
        from dngscan.film_develop import REC2020_LUMA, _divide_by_cast

        rng = np.random.default_rng(7)
        rgb = _scene(20011, 7)
        developed = rng.uniform(0.0, 1.2, size=rgb.shape).astype(np.float32)
        cast_ev = np.linspace(-6.0, 5.0, 12)
        cast = (1.0 + 0.2 * rng.standard_normal((12, 3))).astype(np.float64)
        for off in (0.0, -1.5):
            # The NumPy body divides in place: give each side its own copy.
            with _NoNative():
                ref = _divide_by_cast(developed.copy(), rgb, off, cast_ev, cast)
            got = _divide_by_cast(developed.copy(), rgb, off, cast_ev, cast)
            np.testing.assert_array_equal(got, ref)
        # The luma product is the sequential (a*w0+b*w1)+c*w2 float32 form.
        ev = np.log2(np.maximum(rgb @ REC2020_LUMA, EPS) / np.float32(0.18))
        seq = (rgb[:, 0] * REC2020_LUMA[0] + rgb[:, 1] * REC2020_LUMA[1]) + rgb[:, 2] * REC2020_LUMA[2]
        np.testing.assert_array_equal(ev, np.log2(np.maximum(seq, EPS) / np.float32(0.18)))


@unittest.skipUnless(_native_available(), "native stage-3 kernels unavailable")
class Stage3PipelineParity(unittest.TestCase):
    def _run(self, stock: str, **kw) -> None:
        from dngscan.film_develop import apply_film_core

        rgb = _scene(12007, 11)
        with _NoNative():
            ref = apply_film_core(rgb.copy(), _plan(stock, **kw))
        got = apply_film_core(rgb.copy(), _plan(stock, **kw))
        self.assertEqual(got.dtype, ref.dtype)
        np.testing.assert_array_equal(got, ref)

    def test_negative_default(self) -> None:
        stock = next(s for s in _stocks() if s.startswith(("portra", "pro400h", "c200", "gold")))
        self._run(stock)
        self._run(stock, film_compression=0.6, film_highlight_density=0.4, film_exposure_ev=-1.5)
        self._run(stock, film_interimage="custom", film_interimage_beta=0.8, film_crossover="off")
        self._run(stock, film_print_timing="retimed", film_exposure_ev=1.0, film_crossover="off")
        self._run(stock, film_crossover="print", film_exposure_ev=-0.5)

    def test_reversal(self) -> None:
        rev = [s for s in _stocks() if s.startswith(("velvia", "provia", "ektachrome", "e100"))]
        if not rev:
            self.skipTest("no reversal stock")
        self._run(rev[0])
        self._run(rev[0], film_exposure_ev=-1.0, film_crossover="off")
        self._run(rev[0], film_exposure_ev=0.7, film_crossover="print")

    def test_every_stock_default(self) -> None:
        for stock in _stocks():
            with self.subTest(stock=stock):
                self._run(stock)


if __name__ == "__main__":
    unittest.main()
