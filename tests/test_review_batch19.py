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
        640x960, where the localized source's spread map has a zero border.

        Driven on the OPERATOR now, not through `film_bloom`: P3 pointed that
        amount at the additive editorial capture bloom, so a conservation
        assertion made through it would be asserting the wrong physics. The
        conservative print scatter this test is about is still the operator
        under test — it is just no longer reachable from a user slider.
        """
        from dngscan.film_optics import (
            bloom_apply_rows,
            bloom_delta_map,
            integral_from_field,
            scatter_source,
            scatter_spread,
        )
        from dngscan.film_optics_assets import (
            DEFAULT_PRINT_OPTICS,
            load_print_optics,
        )

        scatter = load_print_optics(DEFAULT_PRINT_OPTICS).print_scatter
        for h, w in ((64, 96), (320, 480), (640, 960)):
            img = np.full((h, w, 3), 0.01, dtype=np.float32)
            img[h // 2, w // 2] = 20.0
            spread = scatter_spread(scatter_source(img, scatter), scatter)
            spread_ii = integral_from_field(spread).astype(np.float32)
            out = bloom_apply_rows(
                img.reshape(-1, 3), spread_ii, 0, h, h, w, scatter, 1.0
            ).reshape(h, w, 3)
            total = float(img.sum(dtype=np.float64))
            drift = abs(float(out.sum(dtype=np.float64)) - total) / total
            self.assertLess(
                drift, 1e-5,
                f"{h}x{w}: scatter lost {drift * 100:.4f}% of the frame energy",
            )
            self.assertLess(
                float(out[h // 2, w // 2].sum()), float(img[h // 2, w // 2].sum()),
                f"{h}x{w}: the core must shed energy",
            )
            self.assertGreater(
                float(out[h // 2 + 3, w // 2].sum()),
                float(img[h // 2 + 3, w // 2].sum()),
                f"{h}x{w}: the neighbourhood must receive it",
            )
            self.assertLessEqual(
                float(np.abs(bloom_delta_map(img, scatter).sum(axis=(0, 1))).max()),
                1e-3,
            )


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
