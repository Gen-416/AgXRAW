# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 4 (2026-09-17): the last two NumPy residues the profiles still showed.

1. color.apply_rgb_matrix3 — float64 products, the left-associative (a + b) + c
   sum, one round to float32 — as a native kernel (float32 and float64 scenes).
2. FilmSpatialContext._layer_exposure_f32 keeps float32 rows when no film
   compression is engaged, so the halation-prep slab path reaches the native
   Stage A kernels. A float32 scene promoted to float64 is the same numbers, and
   float32-valued float64 products are exact — so the result is identical to the
   float64 NumPy path on every platform (a genuinely float64 scene, i.e. with
   compression, still stays on NumPy; see film_v2_math).
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dngscan import _fast


def _native_available() -> bool:
    ext = _fast._load_extension()
    return ext is not None and hasattr(ext, "apply_rgb_matrix3")


class _NoNative:
    def __enter__(self):
        self._p = mock.patch.object(_fast, "kernel", lambda name: None)
        self._p.start()
        return self

    def __exit__(self, *exc):
        self._p.stop()
        return False


@unittest.skipUnless(_native_available(), "native stage-4 kernels unavailable")
class Matrix3Parity(unittest.TestCase):
    def test_matches_numpy_bit_for_bit(self) -> None:
        from dngscan.color import RGB_TO_XYZ, XYZ_TO_RGB, apply_rgb_matrix3

        rng = np.random.default_rng(3)
        base = (0.18 * np.exp2(rng.uniform(-10, 8, (40_003, 3)))).astype(np.float32)
        base[:500] *= -1.0
        base[500:520] = [np.nan, 1.0, np.inf]
        base[520:540] = 3.0e38
        for scene in (base, base.astype(np.float64) * 1.0000001):
            for m in (RGB_TO_XYZ["Rec2020"], XYZ_TO_RGB["sRGB"], rng.uniform(-2, 2, (3, 3))):
                with _NoNative(), np.errstate(all="ignore"):
                    ref = apply_rgb_matrix3(scene, m)
                got = apply_rgb_matrix3(scene, m)
                self.assertEqual(got.dtype, np.float32)
                np.testing.assert_array_equal(got, ref)

    def test_float32_matrix_stays_on_numpy(self) -> None:
        from dngscan.color import apply_rgb_matrix3

        ext = _fast._load_extension()
        rgb = np.ones((8, 3), dtype=np.float32)
        with mock.patch.object(ext, "apply_rgb_matrix3", side_effect=AssertionError("must not dispatch")):
            apply_rgb_matrix3(rgb, np.eye(3, dtype=np.float32))


@unittest.skipUnless(_native_available(), "native stage-4 kernels unavailable")
class LayerExposureSlabParity(unittest.TestCase):
    def _prep(self, compression: float):
        from dngscan.film_develop import FilmSpatialContext
        from tests.test_film_v2_assets import _stock_files

        stock = next(s for s in _stock_files() if s.startswith("portra"))
        plan = SimpleNamespace(
            film_exposure_ev=0.3, film_compression=compression,
            film_compression_knee=2.0, film_highlight_density=0.2,
        )
        return FilmSpatialContext, FilmSpatialContext._hal_prep_from(plan, stock)

    def test_float32_rows_match_the_float64_reference(self) -> None:
        rng = np.random.default_rng(9)
        rgb = (0.18 * np.exp2(rng.uniform(-9, 6, (37, 53, 3)))).astype(np.float32)
        for compression in (0.0, 0.6):
            ctx, prep = self._prep(compression)
            with _NoNative():
                ref = ctx._layer_exposure_f32(rgb, prep)
            got = ctx._layer_exposure_f32(rgb, prep)
            self.assertEqual(got.dtype, np.float32)
            np.testing.assert_array_equal(got, ref)


if __name__ == "__main__":
    unittest.main()
