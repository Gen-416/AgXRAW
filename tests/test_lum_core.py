# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import unittest

from dngscan._deps import np
from dngscan.color import luminance_from_rec2020
from dngscan.lum import apply_lum_core
from dngscan.models import ToneCompressionPlan
from dngscan.retreat import apply_clip_retreat_rec2020, retreat_strength_from_masks


def _plan() -> ToneCompressionPlan:
    return ToneCompressionPlan(
        target_gamut="Rec2020",
        luma_p1=0.01,
        luma_p50=0.18,
        luma_p99=1.0,
        luma_p999=2.0,
        black_ev=-10.0,
        white_ev=6.5,
        dynamic_range_ev=16.5,
        contrast=3.0,
        toe_power=1.5,
        shoulder_power=3.3,
        chroma_p95=0.0,
        negative_rgb_pct=0.0,
        over_rgb_pct=0.0,
        tone_core="lum",
        lum_norm="y",
    )


def test_retreat_strength_classing():
    masks = np.asarray([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    strength = retreat_strength_from_masks(masks)
    assert abs(float(strength[0]) - 0.35) < 1e-6
    assert abs(float(strength[1]) - 0.8375) < 1e-6


def test_retreat_preserves_luma_and_reduces_chroma():
    rgb = np.asarray([[1.0, 0.2, 0.1]], dtype=np.float32)
    mask = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
    out = apply_clip_retreat_rec2020(rgb, mask)
    y_in = float(luminance_from_rec2020(rgb)[0])
    y_out = float(luminance_from_rec2020(out)[0])
    assert abs(y_in - y_out) < 1e-6
    assert float(np.max(np.abs(out - y_out))) < float(np.max(np.abs(rgb - y_in)))


def test_lum_core_midgray_anchor():
    out = apply_lum_core(np.asarray([[0.18, 0.18, 0.18]], dtype=np.float32), _plan())
    assert np.allclose(out, 0.18, atol=1e-5)


def test_lum_core_preserves_rgb_ratios():
    rgb = np.asarray([[0.4, 0.2, 0.1]], dtype=np.float32)
    out = apply_lum_core(rgb, _plan())
    ratio = out[0] / rgb[0]
    assert np.max(ratio) - np.min(ratio) < 1e-6


class LumCoreTest(unittest.TestCase):
    test_retreat_strength_classing = staticmethod(test_retreat_strength_classing)
    test_retreat_preserves_luma_and_reduces_chroma = staticmethod(test_retreat_preserves_luma_and_reduces_chroma)
    test_lum_core_midgray_anchor = staticmethod(test_lum_core_midgray_anchor)
    test_lum_core_preserves_rgb_ratios = staticmethod(test_lum_core_preserves_rgb_ratios)


if __name__ == "__main__":
    unittest.main()
