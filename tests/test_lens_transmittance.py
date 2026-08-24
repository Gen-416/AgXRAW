# SPDX-License-Identifier: GPL-3.0-or-later
"""Gates for the measured lens/filter transmittance library.

Data provenance: first-party bench measurements (see NOTICE.md); the library
is one excisable file, so these tests also document what the pipeline may
assume about it: uniform 1 nm grid, fractions in (0, 1.5], and physically
sensible cast indicators (an orange B&W filter MUST read strongly warm, a
center/GND filter MUST read near-neutral — if those invert, the data or the
loader is corrupted).
"""
from __future__ import annotations

import math
import unittest

from dngscan import lens_transmittance as LT


class LensTransmittanceTests(unittest.TestCase):
    def test_library_loads(self):
        entries = LT.list_entries()
        self.assertGreaterEqual(len(entries), 100)
        cats = {e["category"] for e in entries}
        self.assertEqual(cats, {"Lens", "Filter"})
        self.assertGreaterEqual(len(LT.list_entries("Lens")), 80)
        self.assertGreaterEqual(len(LT.list_entries("Filter")), 30)

    def test_curves_uniform_grid_and_range(self):
        for e in LT.list_entries():
            wl, t = LT.get_curve(e["id"])
            self.assertEqual(len(wl), len(t))
            self.assertEqual(wl[0], 380.0)
            self.assertEqual(wl[-1], 755.0)
            steps = {round(b - a, 6) for a, b in zip(wl, wl[1:])}
            self.assertEqual(steps, {1.0}, e["id"])
            self.assertTrue(all(math.isfinite(v) and 0.0 < v <= 1.5 for v in t), e["id"])

    def test_cast_indicators_physical(self):
        by_name = {e["name"]: e["id"] for e in LT.list_entries()}
        orange = by_name["B+W 43 040 Orange"]
        self.assertGreater(LT.warmth_ratio(orange), 4.0)
        center = by_name["Docter M67 Center Filter"]
        self.assertLess(abs(LT.warmth_ratio(center)), 0.15)
        self.assertGreater(LT.mean_transmittance(center), 0.9)

    def test_unknown_id_none(self):
        self.assertIsNone(LT.get_curve("no-such-lens"))
        self.assertIsNone(LT.warmth_ratio("no-such-lens"))


if __name__ == "__main__":
    unittest.main()
