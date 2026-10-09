# SPDX-License-Identifier: GPL-3.0-or-later
"""Review A8 regression gates: full-well plausibility, deep recipe
immutability, unclipped gamut metrics, plan finiteness, report topology,
and the Core Image runtime probe contract."""
from __future__ import annotations

import copy
import dataclasses
import pickle
import unittest

import numpy as np



class FullwellPlausibilityTests(unittest.TestCase):
    def test_an_ordinary_plateau_does_not_become_the_full_well(self) -> None:
        """A8 item 1 repro: 300 px stuck at 8000 under WhiteLevel 16383
        must NOT override the metadata — that deflated clip %, DR, the CFA
        clip mask and HDR headroom in one stroke."""
        from dngscan.analysis import detect_ceilings, resolve_fullwell

        raw = np.full((1000, 1000), 3000, np.uint16)
        raw.flat[:300] = 8000
        colors = np.zeros_like(raw, np.int32)
        sat = {0: 16383}
        ce, _, _, ok = detect_ceilings(raw, colors, [0], sat)
        fw, _, note, _ = resolve_fullwell([0], ce, ok, sat)
        self.assertEqual(fw, 16383)
        self.assertIn("metadata", note)

    def test_plateaus_at_three_quarters_and_nine_tenths_stay_rejected(self) -> None:
        """A9 item 2: A8's 0.75 gate still let a 13000/16383 plateau
        through. Metadata is authoritative; only a pile within the narrow
        0.95 tolerance may override. Regression at 0.76/0.80/0.90."""
        from dngscan.analysis import detect_ceilings, resolve_fullwell

        colors = np.zeros((1000, 1000), np.int32)
        sat = {0: 16383}
        for frac in (0.76, 0.80, 0.90):
            raw = np.full((1000, 1000), 3000, np.uint16)
            raw.flat[:300] = int(16383 * frac)
            ce, _, _, ok = detect_ceilings(raw, colors, [0], sat)
            fw, _, _, _ = resolve_fullwell([0], ce, ok, sat)
            with self.subTest(frac=frac):
                self.assertEqual(fw, 16383)

    def test_a_genuine_near_white_pile_still_overrides(self) -> None:
        from dngscan.analysis import detect_ceilings, resolve_fullwell

        raw = np.full((1000, 1000), 3000, np.uint16)
        raw.flat[:300] = 16200
        colors = np.zeros_like(raw, np.int32)
        sat = {0: 16383}
        ce, _, _, ok = detect_ceilings(raw, colors, [0], sat)
        fw, _, _, _ = resolve_fullwell([0], ce, ok, sat)
        self.assertEqual(fw, 16200)




class GamutMetricsTests(unittest.TestCase):
    def test_saturated_rec2020_blue_is_counted(self) -> None:
        """A8 item 3: the packed uint16 XYZ clipped Z (~1.061 for pure
        blue) at the representable ceiling, under-reporting exactly the
        saturated blues this metric exists to count. The scene Rec.2020
        buffer is now the unclipped source."""
        from dngscan.analysis import compute_gamut_metrics

        scene = np.zeros((8, 1, 3), np.uint16)
        scene[..., 2] = 65535
        y = np.full((8, 1), 0.06, np.float32)
        pct, _ = compute_gamut_metrics(scene, 65535.0, y, ("sRGB",))
        self.assertEqual(pct["sRGB"], 100.0)






class RuntimeProbeTests(unittest.TestCase):
    def test_runtime_probe_never_raises_and_is_cached(self) -> None:
        from dngscan import coreimage_decode as ci

        first = ci.runtime_available()
        self.assertIsInstance(first, bool)
        self.assertEqual(ci.runtime_available(), first)
        if not ci.available():
            self.assertFalse(first)

    def test_the_probe_renders_the_real_workload_parameters(self) -> None:
        """A11 item 2: pin the probe's render call — RGBAh, rowBytes 8,
        extended linear Rec.2020 — with a fake Quartz, so a host (or CI
        runner) without Core Image still guards the parameter set."""
        import sys
        import unittest.mock as mock

        from dngscan import coreimage_decode as ci

        calls = {}

        class _Ctx:
            def render_toBitmap_rowBytes_bounds_format_colorSpace_(
                self, img, buf, row_bytes, bounds, fmt, cs
            ):
                calls.update(row_bytes=row_bytes, fmt=fmt, cs=cs,
                             buflen=len(buf))

        fake = mock.MagicMock()
        fake.kCIFormatRGBAh = "RGBAh"
        fake.kCGColorSpaceExtendedLinearITUR_2020 = "ext2020-name"
        fake.CGColorSpaceCreateWithName = lambda name: f"cs:{name}"
        fake.CIContext.contextWithOptions_ = lambda opts: _Ctx()
        img = mock.MagicMock()
        img.imageByCroppingToRect_ = lambda rect: img
        fake.CIImage.imageWithColor_ = lambda color: img

        with mock.patch.dict(sys.modules, {"Quartz": fake,
                                           "Foundation": mock.MagicMock()}),              mock.patch.object(ci, "available", return_value=True),              mock.patch.dict(ci._RUNTIME_AVAILABLE, {}, clear=True),              mock.patch.dict(ci._CONTEXTS, {}, clear=True):
            self.assertTrue(ci.runtime_available())
        self.assertEqual(calls["fmt"], "RGBAh")
        self.assertEqual(calls["row_bytes"], 8)
        self.assertEqual(calls["buflen"], 8)
        self.assertEqual(calls["cs"], "cs:ext2020-name")


if __name__ == "__main__":
    unittest.main()
