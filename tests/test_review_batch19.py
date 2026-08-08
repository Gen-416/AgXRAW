# SPDX-License-Identifier: GPL-3.0-or-later
"""Review batch 19 regression gates.

1. Integral images are declared, never sniffed: conservation must hold at
   frame sizes where a localized source leaves a zero border in its spread
   map (the sniffing heuristic silently ate the bloom energy there).
4. A bloom-only render must not walk the scene for a halation map nobody
   asked for.
"""
from __future__ import annotations

import inspect
import unittest

import numpy as np

from tests.test_review_batch16 import _negative_stock, _plan


class IntegralContractTests(unittest.TestCase):
    def test_no_content_sniffing_helper_survives(self) -> None:
        from dngscan import film_optics

        self.assertFalse(
            hasattr(film_optics, "_as_integral"),
            "the content-sniffing helper must be gone — a plain spread map "
            "with a zero border was taken for an integral image",
        )
        src = inspect.getsource(film_optics.integral_from_field)
        self.assertIn("np.cumsum", src)

    def test_conservation_holds_at_larger_frames(self) -> None:
        """The sniffing bug passed at 64x96 and lost 0.087 per channel at
        640x960, where the localized source's spread map has a zero border."""
        from dngscan.film_develop import apply_film_core

        stock = _negative_stock()
        luma = np.array([0.2627, 0.6780, 0.0593])
        for h, w in ((64, 96), (320, 480), (640, 960)):
            img = np.full((h, w, 3), 0.01, dtype=np.float32)
            img[h // 2, w // 2] = 20.0
            flat = img.reshape(-1, 3)
            base = apply_film_core(flat, _plan(stock))
            out = apply_film_core(
                flat, _plan(stock, film_bloom=1.0), spatial_shape=(h, w)
            )
            total = float(base.sum(dtype=np.float64))
            drift = abs(float(out.sum(dtype=np.float64)) - total) / total
            self.assertLess(
                drift, 1e-5,
                f"{h}x{w}: scatter lost {drift * 100:.4f}% of the frame energy",
            )
            b3 = base.reshape(h, w, 3)
            o3 = out.reshape(h, w, 3)
            self.assertLess(
                float(o3[h // 2, w // 2] @ luma), float(b3[h // 2, w // 2] @ luma),
                f"{h}x{w}: the core must shed energy",
            )
            self.assertGreater(
                float(o3[h // 2 + 3, w // 2] @ luma),
                float(b3[h // 2 + 3, w // 2] @ luma),
                f"{h}x{w}: the neighbourhood must RECEIVE it",
            )


class PassOrderTests(unittest.TestCase):
    def test_bloom_only_render_skips_the_halation_decimation(self) -> None:
        """Pass A exists for halation alone (review batch 19): a bloom-only
        render used to decimate the whole scene and throw the result away."""
        from unittest import mock

        from dngscan import render as render_mod
        from dngscan.render import render_output_u8
        from dngscan.tone import build_render_plan
        from tests.golden_support import build_daylight_wide_dr

        scene = build_daylight_wide_dr()
        stock = _negative_stock()

        def run(**kw):
            plan = build_render_plan(
                scene.bundle, scene.analysis, "agx", "srgb",
                film_curve=stock, film_mode="full",
                film_crossover="datasheet", **kw,
            )
            from dngscan import film_develop

            calls = []
            real = film_develop.FilmSpatialContext.finish_maps

            def spy(self, *a, **k):
                calls.append(1)
                return real(self, *a, **k)

            # finish_maps IS pass A: it allocates the decimated accumulator
            # and walks the whole scene. area_decimate_rows is not a valid
            # probe any more — pass B legitimately uses it to decimate the
            # bloom SOURCE.
            with mock.patch.object(
                film_develop.FilmSpatialContext, "finish_maps", spy
            ):
                render_output_u8(scene.bundle, scene.analysis, "srgb", plan)
            return len(calls)

        self.assertEqual(
            run(film_bloom=0.6), 0,
            "a bloom-only render must not run the halation decimation pass",
        )
        self.assertGreater(
            run(film_halation=0.6), 0,
            "halation still needs its decimated scene",
        )

    def test_bloom_accumulator_does_not_depend_on_pass_a(self) -> None:
        from dngscan.film_develop import prepare_film_spatial

        ctx = prepare_film_spatial(
            _plan(_negative_stock(), film_bloom=0.5), 64, 96
        )
        self.assertIsNotNone(ctx)
        ctx.begin_bloom_source()   # no finish_maps() first
        self.assertIsNotNone(ctx.bloom_source)
        self.assertEqual(ctx.bloom_source.shape[:2], ctx.spread_shape)


class BudgetTierTests(unittest.TestCase):
    def test_only_honourable_tiers_are_advertised(self) -> None:
        from dngscan.render import OPTICS_BUDGET_TIERS_MIB, _optics_budget_mib

        self.assertNotIn(
            256, OPTICS_BUDGET_TIERS_MIB,
            "the 256 MiB tier cannot be honoured: the fixed context alone "
            "exceeds it once bloom is engaged (measured 466 MiB extra)",
        )
        self.assertIn(512, OPTICS_BUDGET_TIERS_MIB)

    def test_removed_tier_falls_back_to_the_default(self) -> None:
        import os
        from unittest import mock

        from dngscan.render import _optics_budget_mib

        with mock.patch.dict(os.environ, {"DNGSCAN_OPTICS_BUDGET_MIB": "256"}):
            self.assertEqual(_optics_budget_mib(), 512)


if __name__ == "__main__":
    unittest.main()
