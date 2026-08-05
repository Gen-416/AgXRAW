# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from dngscan.tone import rank_trim_reconstructed_highlights, scene_tone_metrics


class Raw9ReliableTailTests(unittest.TestCase):
    def test_raw_clip_fraction_removes_brightest_reconstructed_samples(self) -> None:
        ev = np.arange(1000, dtype=np.float32)
        valid = np.ones(1000, dtype=bool)
        reliable = rank_trim_reconstructed_highlights(ev, valid, 10.0)

        self.assertEqual(int(np.count_nonzero(reliable)), 900)
        self.assertTrue(bool(np.all(reliable[:900])))
        self.assertFalse(bool(np.any(reliable[900:])))

    def test_rank_trim_respects_existing_invalid_samples(self) -> None:
        ev = np.arange(1000, dtype=np.float32)
        valid = np.ones(1000, dtype=bool)
        valid[:100] = False
        reliable = rank_trim_reconstructed_highlights(ev, valid, 10.0)

        self.assertEqual(int(np.count_nonzero(reliable)), 810)
        self.assertFalse(bool(np.any(reliable[:100])))
        self.assertFalse(bool(np.any(reliable[910:])))

    def test_zero_clip_is_identity(self) -> None:
        valid = np.asarray([False] + [True] * 999, dtype=bool)
        reliable = rank_trim_reconstructed_highlights(
            np.arange(1000, dtype=np.float32), valid, 0.0
        )
        np.testing.assert_array_equal(reliable, valid)


class ReliableTailAuthorityTests(unittest.TestCase):
    def test_sdr_fallback_does_not_become_hdr_evidence(self) -> None:
        scene = np.ones((100, 100, 3), dtype=np.float32)
        bundle = SimpleNamespace(
            scene_rec2020_render=scene,
            scene_scale=1.0,
            exposure_gain=1.0,
            wb_mode="camera",
            camera_wb=None,
            applied_wb=None,
            daylight_wb=None,
            clip_masks=np.ones_like(scene),
            scene_decoder="libraw",
        )
        analysis = SimpleNamespace(cell_union_pct=100.0)
        metrics = scene_tone_metrics(bundle, analysis)
        self.assertEqual(metrics.reliable_sample_pct, 0.0)
        self.assertTrue(math.isnan(metrics.reliable_tail_ev_p9999))
        # SDR still receives finite body statistics from its defensive fallback.
        self.assertTrue(math.isfinite(metrics.body_ev_p50))


if __name__ == "__main__":
    unittest.main()
