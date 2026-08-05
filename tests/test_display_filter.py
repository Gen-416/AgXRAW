# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for display LUT filters (log encode -> .cube)."""

from __future__ import annotations

import unittest

import numpy as np

from dngscan.color import bt1886_eotf, rec2020_to_output, rec709_inverse_oetf
from dngscan.display_filter import (
    DISPLAY_FILTERS,
    apply_display_filter_rec2020,
    filter_available,
)
from dngscan.log_encode import cineon_encode, log3g10_encode


class DisplayFilterTests(unittest.TestCase):
    def test_cineon_anchor(self) -> None:
        # Canonical Cineon: (685 + 300*log10(0.18)) / 1023
        got = float(cineon_encode(np.array([0.18]))[0])
        self.assertAlmostEqual(got, 0.4512, places=3)

    def test_cineon_preserves_highlight_headroom(self) -> None:
        got = float(cineon_encode(np.array([1.0]))[0])
        self.assertLess(got, 0.7)
        self.assertGreater(got, 0.6)

    def test_rec709_inverse_oetf_is_not_bt1886(self) -> None:
        v = np.array([0.5], dtype=np.float32)
        self.assertAlmostEqual(float(rec709_inverse_oetf(v)[0]), 0.2597, places=3)
        self.assertAlmostEqual(float(bt1886_eotf(v)[0]), float(0.5**2.4), places=6)

    def test_kodak_filter_no_channel_crush_on_green(self) -> None:
        if not filter_available("kodak_2383_d65"):
            self.skipTest("Kodak cube missing")
        rec = np.array([[[0.18, 0.25, 0.14]]], dtype=np.float32)
        out = apply_display_filter_rec2020(rec, "srgb", "kodak_2383_d65", 1.0)[0, 0]
        self.assertGreater(float(out[0]), 0.05)
        self.assertGreater(float(out[1]), 0.05)
        self.assertGreater(float(out[2]), 0.05)

    def test_log3g10_midgray(self) -> None:
        from dngscan.log_encode import LOG3G10_MIDGRAY

        got = float(log3g10_encode(np.array([0.18]))[0])
        self.assertAlmostEqual(got, LOG3G10_MIDGRAY, places=3)

    def test_filter_none_matches_agx_display(self) -> None:
        rec = np.array([[[0.4, 0.35, 0.3]]], dtype=np.float32)
        flat = rec.reshape(-1, 3)
        expected = rec2020_to_output(flat, "srgb").reshape(rec.shape)
        out = apply_display_filter_rec2020(rec, "srgb", "none", 1.0)
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)

    def test_filter_strength_zero_matches_agx_display(self) -> None:
        rec = np.array([[[0.4, 0.35, 0.3]]], dtype=np.float32)
        flat = rec.reshape(-1, 3)
        expected = rec2020_to_output(flat, "srgb").reshape(rec.shape)
        out = apply_display_filter_rec2020(rec, "srgb", "kodak_2383_d65", 0.0)
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)

    def test_vendor_cubes_are_not_bundled(self) -> None:
        for name in DISPLAY_FILTERS:
            self.assertFalse(filter_available(name), msg=name)

    def test_kodak_filter_preserves_color(self) -> None:
        if not filter_available("kodak_2383_d65"):
            self.skipTest("Kodak cube missing")
        # Mid-gray with strong green bias in Rec.2020 linear
        rec = np.array([[[0.18, 0.22, 0.14]]], dtype=np.float32)
        out = apply_display_filter_rec2020(rec, "srgb", "kodak_2383_d65", 1.0)
        spread = float(np.max(out) - np.min(out))
        self.assertGreater(spread, 0.02, "filter output should not be near grayscale")

    def test_red_filter_preserves_color(self) -> None:
        if not filter_available("red_ipp2_rec709_medium"):
            self.skipTest("RED IPP2 cube missing")
        # IPP2 is feed="scene": it renders the scene-linear buffer in parallel to AgX.
        scene = np.array([[[0.18, 0.22, 0.14]]], dtype=np.float32)
        mapped = np.array([[[0.30, 0.36, 0.24]]], dtype=np.float32)  # stand-in AgX display
        out = apply_display_filter_rec2020(
            mapped, "srgb", "red_ipp2_rec709_medium", 1.0, scene_rec2020=scene
        )
        spread = float(np.max(out) - np.min(out))
        self.assertGreater(spread, 0.05, "filter output should not be near grayscale")

    def test_scene_feed_requires_scene_buffer(self) -> None:
        if not filter_available("red_ipp2_rec709_medium"):
            self.skipTest("RED IPP2 cube missing")
        mapped = np.array([[[0.3, 0.3, 0.3]]], dtype=np.float32)
        with self.assertRaises(ValueError):
            apply_display_filter_rec2020(mapped, "srgb", "red_ipp2_rec709_medium", 1.0)

    def test_log3g10_monotonic(self) -> None:
        x = np.linspace(0.0, 1.0, 32, dtype=np.float64)
        y = log3g10_encode(x)
        self.assertTrue(np.all(np.diff(y) >= -1e-6))


if __name__ == "__main__":
    unittest.main()

    def test_slog3_anchors(self) -> None:
        import numpy as np

        from dngscan.log_encode import SLOG3_MIDGRAY, slog3_encode

        self.assertAlmostEqual(float(slog3_encode(np.array([0.18]))[0]), SLOG3_MIDGRAY, places=4)
        # official 90% reflectance point
        self.assertAlmostEqual(float(slog3_encode(np.array([0.9]))[0]), 0.5845, places=3)
        # linear toe continuity at the join
        lo = float(slog3_encode(np.array([0.011249]))[0])
        hi = float(slog3_encode(np.array([0.011251]))[0])
        self.assertLess(abs(hi - lo), 1e-3)

    def test_sony_lc709a_registered(self) -> None:
        from dngscan.display_filter import DISPLAY_FILTERS, filter_available

        spec = DISPLAY_FILTERS["sony_lc709a"]
        self.assertEqual(spec.feed, "scene")
        self.assertEqual(spec.source, "slog3")
        if filter_available("sony_lc709a"):
            self.assertTrue(spec.cube.is_file())
