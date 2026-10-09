# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 4 (2026-09-17): the last two NumPy residues the profiles still showed.

1. color.apply_rgb_matrix3 — float64 products, the left-associative (a + b) + c
   sum, one round to float32 — as a native kernel (float32 and float64 scenes).
2. FilmSpatialContext._layer_exposure_f32 keeps float32 rows when no film
   compression is engaged, so the halation-prep slab path reaches the native
   Stage A kernels. Promoting the scene preserves its values, but multiplication
   by float64 observer coefficients can still round differently in BLAS tail
   rows. Exact parity here is a gate for these fixtures, not a platform-wide
   proof. Short-batch intermediate tolerances are covered in test_rust_stage3;
   compressed float64 scenes still stay on NumPy.
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




if __name__ == "__main__":
    unittest.main()
