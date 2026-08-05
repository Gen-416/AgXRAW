# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the scene-driven purity compensation (punch)."""

from __future__ import annotations

import unittest

import numpy as np

from dngscan.punch import apply_punch_rec2020
from dngscan.tone import _smoothstep_f


def _chroma_oklab(rgb: np.ndarray) -> np.ndarray:
    from dngscan.color import apply_rgb_matrix3
    from dngscan.constants import OKLAB_M1, OKLAB_M2, RGB_TO_XYZ

    xyz = apply_rgb_matrix3(rgb.astype(np.float32), RGB_TO_XYZ["Rec2020"])
    lab = apply_rgb_matrix3(np.cbrt(np.maximum(apply_rgb_matrix3(xyz, OKLAB_M1), 0.0)), OKLAB_M2)
    return np.hypot(lab[:, 1], lab[:, 2])


class PunchOperatorTests(unittest.TestCase):
    def test_zero_strength_is_exact_identity(self) -> None:
        rgb = np.asarray([[0.4, 0.3, 0.2], [0.05, 0.06, 0.07]], dtype=np.float32)
        out = apply_punch_rec2020(rgb, 0.0)
        self.assertIs(out, rgb)  # short-circuit: same object, byte-identical renders

    def test_neutral_axis_untouched(self) -> None:
        rgb = np.asarray([[0.18, 0.18, 0.18], [0.5, 0.5, 0.5]], dtype=np.float32)
        out = apply_punch_rec2020(rgb, 1.0)
        self.assertTrue(np.allclose(out, rgb, atol=2e-4))

    def test_gain_never_desaturates(self) -> None:
        rng = np.random.default_rng(7)
        rgb = rng.uniform(0.01, 1.0, size=(4096, 3)).astype(np.float32)
        c_in = _chroma_oklab(rgb)
        c_out = _chroma_oklab(apply_punch_rec2020(rgb, 1.0))
        self.assertTrue(np.all(c_out >= c_in - 1e-4))

    def test_midtone_color_gets_lift_highlight_does_not(self) -> None:
        mid = np.asarray([[0.28, 0.16, 0.10]], dtype=np.float32)  # colorful midtone
        hi = np.asarray([[0.95, 0.88, 0.82]], dtype=np.float32)  # near-white highlight
        mid_ratio = float(_chroma_oklab(apply_punch_rec2020(mid, 1.0))[0] / _chroma_oklab(mid)[0])
        hi_ratio = float(_chroma_oklab(apply_punch_rec2020(hi, 1.0))[0] / max(_chroma_oklab(hi)[0], 1e-6))
        self.assertGreater(mid_ratio, 1.05)
        self.assertLess(hi_ratio, 1.02)

    def test_skin_band_damped_vs_nonskin(self) -> None:
        # same L/C, hue inside vs outside the skin arc
        from dngscan.color import apply_rgb_matrix3
        from dngscan.constants import OKLAB_M1_INV, OKLAB_M2_INV, XYZ_TO_RGB

        def rgb_from_lab(l_, a_, b_):
            lab = np.asarray([[l_, a_, b_]], dtype=np.float32)
            lms_ = apply_rgb_matrix3(lab, OKLAB_M2_INV)
            return apply_rgb_matrix3(apply_rgb_matrix3(lms_**3, OKLAB_M1_INV), XYZ_TO_RGB["Rec2020"])

        c = 0.09
        skin = rgb_from_lab(0.55, c * np.cos(np.radians(40)), c * np.sin(np.radians(40)))
        nonskin = rgb_from_lab(0.55, c * np.cos(np.radians(250)), c * np.sin(np.radians(250)))
        skin_ratio = float(_chroma_oklab(apply_punch_rec2020(skin, 1.0))[0] / _chroma_oklab(skin)[0])
        nonskin_ratio = float(_chroma_oklab(apply_punch_rec2020(nonskin, 1.0))[0] / _chroma_oklab(nonskin)[0])
        self.assertLess(skin_ratio, nonskin_ratio)
        self.assertGreater(skin_ratio, 1.0)


class PunchStrengthGateTests(unittest.TestCase):
    def _strength(self, ev_p50: float, plan_dr: float, dr: float, scale: float = 1.0) -> float:
        w_bright = _smoothstep_f(-3.0, -1.2, ev_p50)
        w_quality = _smoothstep_f(7.5, 9.5, plan_dr)
        w_dr = _smoothstep_f(6.5, 8.0, dr)
        return min(1.0, max(0.0, w_bright * w_quality * (0.55 + 0.45 * w_dr) * scale))

    def test_night_high_iso_gates_to_zero(self) -> None:
        # ISO 25600-ish: prior-clamped usable DR ~5, deep median
        self.assertEqual(self._strength(ev_p50=-4.1, plan_dr=5.0, dr=9.6), 0.0)

    def test_daylight_low_iso_engages(self) -> None:
        # _SDI0238-like: median near gray, ISO100 usable DR ~11
        self.assertGreater(self._strength(ev_p50=-1.3, plan_dr=11.0, dr=8.4), 0.6)

    def test_scale_zero_disables(self) -> None:
        self.assertEqual(self._strength(ev_p50=-1.0, plan_dr=11.0, dr=9.0, scale=0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
