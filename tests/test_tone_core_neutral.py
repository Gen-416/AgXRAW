# SPDX-License-Identifier: GPL-3.0-or-later
"""Neutral tone core: fixed generic export curve (Lightroom-style control)."""
from __future__ import annotations

import unittest

import numpy as np

from dngscan.color import luminance_from_rec2020
from dngscan.models import RawBundle
from dngscan.render import apply_tone_core, render_output_linear
from dngscan.tone import (
    build_color_geometry_plan, compute_exposure_gain, exposure_mode_for_tone_core,
    neutral_tone_plan, plan_for_mode,
)


def _minimal_analysis() -> "Analysis":
    from dngscan.models import Analysis

    return Analysis(
        channel_ids=[0, 1, 2],
        labels={0: "R", 1: "G", 2: "B"},
        ceilings={0: 1000, 1: 1000, 2: 1000},
        ceil_spike_counts={0: 0, 1: 0, 2: 0},
        ceil_near_counts={0: 0, 1: 0, 2: 0},
        ceil_spike_ok={0: False, 1: False, 2: False},
        fullwell_channel_ids=[0, 1, 2],
        fullwell_note="test",
        saturation_levels={0: 1000, 1: 1000, 2: 1000},
        channel_fullwell={0: 1000, 1: 1000, 2: 1000},
        channel_thresholds={0: 996, 1: 996, 2: 996},
        fullwell=1000,
        threshold=996,
        clip_pct={0: 0.0, 1: 0.0, 2: 0.0},
        cfa_cell_supported=True,
        cell_union_pct=0.0,
        cell_ge2_of_clipped_pct=0.0,
        cell_k_of_clipped_pct={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        cell_k_of_all_pct={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        ev_p1=-6.0,
        ev_raw_p1=-6.0,
        ev_median=-1.0,
        ev_p99=0.5,
        ev_p999=1.0,
        ev_dr_p1_p999=7.0,
        ev_floor_hit_pct=0.0,
        median_vs_gray_ev=-1.0,
        median_y=0.09,
        noise_floor=0.002,
        usable_dr_ev=9.0,
        snr_curves={},
        snr1_dr={},
        snr1_stop={},
        gamut_out_pct={"sRGB": 0.0, "Display P3": 0.0, "Rec2020": 0.0},
        bright_pixel_pct=0.0,
        survivor_channel="R",
        container_bits_est=14,
        usable_dr_eff_ev=9.0,
    )


class NeutralToneCoreTests(unittest.TestCase):
    def test_exposure_mode_mapping(self) -> None:
        self.assertEqual(exposure_mode_for_tone_core("neutral"), "agx")
        self.assertEqual(exposure_mode_for_tone_core("agx"), "agx")

    def test_neutral_shares_agx_exposure_anchor(self) -> None:
        self.assertAlmostEqual(
            compute_exposure_gain(exposure_mode_for_tone_core("neutral"), 0.0),
            compute_exposure_gain(exposure_mode_for_tone_core("agx"), 0.0),
        )

    def test_neutral_color_plan_keeps_clip_retreat(self) -> None:
        analysis = _minimal_analysis()
        color = build_color_geometry_plan(analysis, "srgb", tone_core="neutral")
        self.assertAlmostEqual(float(color.raw_clip_retreat_strength), 1.0)
        self.assertAlmostEqual(float(color.display_highlight_chroma_retreat), 0.0)

    def test_neutral_compresses_highlights(self) -> None:
        plan = neutral_tone_plan("Rec2020")
        rgb = np.asarray([[2.5, 2.0, 1.5]], dtype=np.float32)
        out = apply_tone_core(rgb, plan)
        self.assertLess(float(luminance_from_rec2020(out)[0]), float(luminance_from_rec2020(rgb)[0]))
        self.assertGreater(float(out[0, 0]), float(rgb[0, 0]) * 0.05)

    def test_neutral_preserves_rgb_ratios(self) -> None:
        plan = neutral_tone_plan("Rec2020")
        rgb = np.asarray([[1.2, 0.6, 0.3]], dtype=np.float32)
        out = apply_tone_core(rgb, plan)
        ratio_in = rgb[0] / rgb[0].sum()
        ratio_out = out[0] / out[0].sum()
        np.testing.assert_allclose(ratio_out, ratio_in, rtol=0, atol=1e-5)

    def test_neutral_render_applies_gain(self) -> None:
        from pathlib import Path

        analysis = _minimal_analysis()
        scene = np.array([[[0.25, 0.20, 0.15]]], dtype=np.float32)
        bundle = RawBundle(
            path=Path("x.dng"),
            raw_image=np.zeros((2, 2), dtype=np.uint16),
            raw_colors=np.zeros((2, 2), dtype=np.uint8),
            xyz_render=np.zeros((1, 1, 3), dtype=np.float32),
            render_scale=65535.0,
            scene_rec2020_render=scene,
            scene_scale=1.0,
            white_level=16383,
            black_levels=[1000.0, 1000.0, 1000.0],
            camera_wb=[1.0, 1.0, 1.0, 0.0],
            color_desc="RGB",
            raw_pattern=[[0, 1], [1, 2]],
            camera_white_levels=[16383, 16383, 16383],
            exposure_gain=2.0,
        )
        plan = plan_for_mode(bundle, analysis, "agx", "srgb", tone_core="neutral")
        self.assertEqual(plan.tone_core, "neutral")
        out = render_output_linear(
            bundle, analysis, "srgb", tone_plan=plan, tone_core="neutral"
        )
        self.assertGreater(float(out[0, 0, 0]), 0.4)


if __name__ == "__main__":
    unittest.main()
