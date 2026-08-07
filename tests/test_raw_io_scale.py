# SPDX-License-Identifier: GPL-3.0-or-later
"""Exposure-unit contracts at the LibRaw uint16 handoff."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from dngscan.raw_io import (
    baseline_exposure_gain,
    coreimage_alignment_factor,
    coreimage_uses_file_alignment,
    scene_green_median,
    libraw_scene_scale,
    libraw_wb_headroom_gain,
    scene_rec2020_to_xyz_render,
    load_raw,
)
from dngscan.tone import scene_rec2020_to_float


class LibRawSceneScaleTests(unittest.TestCase):
    def test_clip_uses_full_uint16_range(self) -> None:
        self.assertEqual(libraw_scene_scale(65535.0, "clip", [1.48, 1.0, 2.33, 0.0]), 65535.0)

    def test_reconstruction_scale_removes_wb_storage_attenuation(self) -> None:
        wb = [1.48, 1.0, 2.33, 0.0]
        self.assertAlmostEqual(libraw_wb_headroom_gain(wb), 2.33)
        self.assertAlmostEqual(libraw_scene_scale(65535.0, "reconstruct", wb), 65535.0 / 2.33)
        self.assertAlmostEqual(libraw_scene_scale(65535.0, "blend", wb), 65535.0 / 2.33)

    def test_wb_gain_is_invariant_to_coefficient_normalization(self) -> None:
        self.assertAlmostEqual(libraw_wb_headroom_gain([2.0, 1.0, 4.0, 1.0]), 4.0)
        self.assertAlmostEqual(libraw_wb_headroom_gain([1.0, 0.5, 2.0, 0.5]), 4.0)

    def test_baseline_exposure_divides_the_scale_rather_than_scaling_the_buffer(self) -> None:
        """The gain reaches 5.65x on an iPhone low-light frame. Applied to a uint16 buffer
        normalised to sensor saturation it would clip everything above 0.18, so it has to
        arrive as a change of scale, leaving the codes untouched."""
        self.assertAlmostEqual(baseline_exposure_gain(1.0), 2.0)
        self.assertAlmostEqual(baseline_exposure_gain(2.4973), 5.6465, places=3)
        self.assertAlmostEqual(
            libraw_scene_scale(65535.0, "clip", None, baseline_exposure=1.0), 65535.0 / 2.0
        )
        # Composes with the highlight-mode storage scaling rather than replacing it.
        wb = [1.48, 1.0, 2.33, 0.0]
        self.assertAlmostEqual(
            libraw_scene_scale(65535.0, "reconstruct", wb, baseline_exposure=1.0),
            65535.0 / 2.33 / 2.0,
        )

    def test_absent_baseline_exposure_is_not_zero(self) -> None:
        """A file without the tag must render unchanged; 0.0 is a real value meaning 1x."""
        self.assertEqual(baseline_exposure_gain(None), 1.0)
        self.assertEqual(baseline_exposure_gain(0.0), 1.0)
        self.assertEqual(baseline_exposure_gain(float("nan")), 1.0)
        self.assertEqual(
            libraw_scene_scale(65535.0, "clip", None, baseline_exposure=None), 65535.0
        )
        # A corrupt tag must not rewrite the exposure by an absurd amount.
        self.assertEqual(baseline_exposure_gain(99.0), 2.0 ** 8)

    def test_alignment_factor_is_a_ratio_of_decoded_green_levels(self) -> None:
        """The per-file A/B ruler is explicit about being a decoded-level comparison.

        It is not a sensor gain calibration, and invalid or implausible measurements
        degrade to identity rather than to a clipped guess.
        """
        self.assertAlmostEqual(coreimage_alignment_factor(1.02, 2.17), 1.02 / 2.17)
        for bad in ((float("nan"), 2.0), (1.0, float("nan")), (0.0, 2.0), (1.0, 0.0)):
            self.assertEqual(coreimage_alignment_factor(*bad), 1.0)
        # A failed statistic must not be able to destroy a render.
        self.assertEqual(coreimage_alignment_factor(1000.0, 1.0), 1.0)
        self.assertEqual(coreimage_alignment_factor(1.0, 1000.0), 1.0)

    def test_only_aligned_mode_requests_a_reference_render(self) -> None:
        self.assertTrue(coreimage_uses_file_alignment("aligned"))
        self.assertFalse(coreimage_uses_file_alignment("unity"))
        self.assertFalse(coreimage_uses_file_alignment("measured"))

    def test_scene_green_median_reports_the_decoded_level(self) -> None:
        scene = np.full((4, 4, 3), 0.2, dtype=np.float32)
        self.assertAlmostEqual(scene_green_median(scene), 0.2, places=6)
        self.assertAlmostEqual(scene_green_median(scene * 0.5), 0.1, places=6)
        self.assertTrue(np.isnan(scene_green_median(np.zeros((4, 4, 3), np.float32))))

    def test_float_scene_scale_below_one_is_honoured(self) -> None:
        scene = np.full((1, 1, 3), 0.2, dtype=np.float16)
        decoded = scene_rec2020_to_float(scene, 0.5)
        self.assertAlmostEqual(float(decoded[0, 0, 1]), 0.4, places=3)

    def test_alignment_reference_must_use_the_libraw_scale_function(self) -> None:
        """The reference has to be decoded the way the LibRaw path decodes, not by the
        container maximum. The Core Image path forces highlight mode "reconstruct", where
        LibRaw divides the buffer by the largest WB multiplier to make room for
        reconstruction; normalising by 65535 instead drops exactly that factor. Measured
        on an iPhone frame with a 2.981 multiplier it made the reference gain 2.98x too
        small and pushed the alignment into its guard rail, rendering 0.8 EV dark."""
        wb = [1.5354, 1.0, 2.981, 0.0]
        naive = 65535.0
        correct = libraw_scene_scale(65535.0, "reconstruct", wb, baseline_exposure=None)
        self.assertAlmostEqual(naive / correct, 2.981, places=3)
        # And with clip, where no headroom is reserved, the two agree.
        self.assertEqual(libraw_scene_scale(65535.0, "clip", wb, baseline_exposure=None), naive)

    def test_xyz_analysis_buffer_preserves_reconstruction_headroom(self) -> None:
        scale = 65535.0 / 2.0
        scene = np.full((1, 1, 3), 32768, dtype=np.uint16)
        xyz = scene_rec2020_to_xyz_render(scene, scale)
        decoded = xyz.astype(np.float64) / scale
        self.assertGreater(float(decoded[0, 0, 2]), 1.05)

    def test_sigma_clip_and_reconstruct_keep_the_same_body_exposure(self) -> None:
        source = Path.home() / "Pictures" / "AgXRAW样张" / "_SDI0150.DNG"
        if not source.is_file():
            raise unittest.SkipTest(f"missing {source}")
        clip = load_raw(source, "clip", scene_half_size=True)
        reconstruct = load_raw(source, "reconstruct", scene_half_size=True)

        def body_median(bundle) -> float:
            rgb = np.asarray(bundle.scene_rec2020_render, dtype=np.float32) / float(bundle.scene_scale)
            y = 0.2627 * rgb[:, :, 0] + 0.6780 * rgb[:, :, 1] + 0.0593 * rgb[:, :, 2]
            return float(np.median(y[(y > 0.002) & (y < 0.5)]))

        delta_ev = float(np.log2(body_median(reconstruct) / body_median(clip)))
        self.assertAlmostEqual(delta_ev, 0.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()
