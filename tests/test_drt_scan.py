# SPDX-License-Identifier: GPL-3.0-or-later
"""DRT geometry point checks (audit R11: the EV × hue × chroma
scan this module was named for was removed historically; the unused
sampler scaffolding is now gone and the docstring says what remains."""
from __future__ import annotations

import unittest

import numpy as np

from dngscan.color import luminance_from_rec2020, rgb_to_oklab
from dngscan.gated_drt import apply_gated_core
from dngscan.models import ColorGeometryPlan, ToneCompressionPlan
from dngscan.render import apply_agx_core, apply_tone_core


def _base_plan(**kwargs) -> ToneCompressionPlan:
    defaults = dict(
        target_gamut="Rec2020",
        luma_p1=0.01,
        luma_p50=0.18,
        luma_p99=1.0,
        luma_p999=2.0,
        black_ev=-7.0,
        white_ev=4.5,
        dynamic_range_ev=11.5,
        contrast=3.0,
        toe_power=1.5,
        shoulder_power=3.3,
        chroma_p95=0.0,
        negative_rgb_pct=0.0,
        over_rgb_pct=0.0,
        use_c1_endpoints=True,
    )
    defaults.update(kwargs)
    return ToneCompressionPlan(**defaults)


class DrtScanTest(unittest.TestCase):
    def test_gated_midtone_path_differs_from_darktable_agx(self) -> None:
        color = ColorGeometryPlan("srgb", 0.0, 0.0)
        masks = np.zeros((2, 3), dtype=np.float32)
        rgb = np.asarray([[0.30, 0.10, 0.22], [0.22, 0.14, 0.35]], dtype=np.float32)
        gated = apply_gated_core(
            rgb,
            _base_plan(tone_core="gated", agx_primaries="smooth"),
            color,
            masks,
        )
        agx = apply_agx_core(rgb, _base_plan(tone_core="agx", agx_primaries="smooth"))

        def mean_chroma(v):
            lab_l, lab_a, lab_b = rgb_to_oklab(v, "srgb")
            return float(np.mean(np.hypot(lab_a, lab_b)))

        self.assertGreater(abs(mean_chroma(gated) - mean_chroma(agx)), 1e-4)

    def test_lum_core_matches_tone_core_lum(self) -> None:
        """Audit R11: the old body compared apply_tone_core against a second
        identical call — a tautology that left the lum DISPATCH route with
        zero coverage anywhere in the suite. This is the claim the name
        makes: routing tone_core="lum" through the dispatcher must be the
        same computation as calling the lum engine directly."""
        from dngscan.lum import apply_lum_core

        rgb = np.asarray([[0.25, 0.12, 0.08], [0.6, 0.5, 0.1]], dtype=np.float32)
        plan = _base_plan(tone_core="lum")
        a = apply_tone_core(rgb, plan)
        b = apply_lum_core(rgb, plan)
        self.assertTrue(np.allclose(a, b, atol=1e-5),
                        f"dispatcher diverged from the lum engine: {a} vs {b}")


if __name__ == "__main__":
    unittest.main()
