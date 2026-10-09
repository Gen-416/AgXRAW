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





class BudgetTierTests(unittest.TestCase):
    def test_only_honourable_tiers_are_advertised(self) -> None:
        import os
        from unittest import mock
        from dngscan.spatial import spatial_budget_mib

        for requested, expected in (("256", 512), ("512", 512), ("1024", 1024)):
            with mock.patch.dict(os.environ, {"DNGSCAN_SPATIAL_BUDGET_MIB": requested}):
                self.assertEqual(spatial_budget_mib(), expected)

    def test_removed_tier_falls_back_to_the_default(self) -> None:
        import os
        from unittest import mock
        from dngscan.spatial import spatial_budget_mib

        with mock.patch.dict(os.environ, {"DNGSCAN_SPATIAL_BUDGET_MIB": "256"}):
            self.assertEqual(spatial_budget_mib(), 512)


if __name__ == "__main__":
    unittest.main()
