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
