# SPDX-License-Identifier: GPL-3.0-or-later
"""R2 item 20: preview plans compile from the exporter's own statistics.

The cache-proxy bundle carries the FULL-resolution tone-plan sample rows
(and identically strided clip-mask rows) taken at entry build, so the tone
endpoints a preview compiles are exactly the export's — not a downsampled
approximation. Gates: exact plan equality against the full bundle, disk
round-trip preservation, and the hot-WB rebalance transforming the stored
sample alongside the scene."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dngscan.analysis import analyze
from dngscan.grade import RENDER_MODE
from dngscan.raw_io import load_raw, rebalance_raw_bundle
from dngscan.tone import build_render_plan

FRAME = Path.home() / "Pictures" / "AgXRAW样张" / "_SDI0150.DNG"

TONE_FIELDS = (
    "black_ev", "white_ev", "dynamic_range_ev", "contrast", "toe_power",
    "shoulder_power", "view_brightness", "punch_strength",
    "luma_p1", "luma_p50", "luma_p99", "luma_p999",
)
SCENE_FIELDS = (
    "body_ev_p1", "body_ev_p50", "body_ev_p99", "body_ev_p999",
    "reliable_tail_ev_p9999", "tail_ev_p9999", "sparse_emitter_tail",
)


@unittest.skipUnless(FRAME.is_file(), "sample frame unavailable")
class PreviewPlanFullResTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from dngscan.gui.preview_cache import build_proxy_entry

        cls.bundle = load_raw(FRAME, scene_half_size=True)
        cls.analysis, _, _ = analyze(cls.bundle, margin=4, diagnostics=False)
        cls.entry = build_proxy_entry(cls.bundle, cls.analysis)

    def _assert_plans_equal(self, full_plan, preview_plan) -> None:
        for f in TONE_FIELDS:
            self.assertEqual(
                getattr(full_plan.tone, f), getattr(preview_plan.tone, f),
                f"tone.{f} diverged",
            )
        for f in SCENE_FIELDS:
            self.assertEqual(
                getattr(full_plan.scene, f), getattr(preview_plan.scene, f),
                f"scene.{f} diverged",
            )

    def test_proxy_plan_equals_export_plan_exactly(self) -> None:
        self.assertIsNotNone(self.entry.bundle._tone_plan_sample)
        full_plan = build_render_plan(self.bundle, self.analysis, RENDER_MODE, "p3")
        preview_plan = build_render_plan(
            self.entry.bundle, self.analysis, RENDER_MODE, "p3"
        )
        self._assert_plans_equal(full_plan, preview_plan)

    def test_without_the_sample_the_proxy_diverges(self) -> None:
        """The gate must actually be testing something: the proxy's own
        pixels compile measurably different endpoints on a real frame."""
        import dataclasses

        stripped = dataclasses.replace(
            self.entry.bundle, _tone_plan_sample=None, _tone_plan_sample_masks=None
        )
        full_plan = build_render_plan(self.bundle, self.analysis, RENDER_MODE, "p3")
        proxy_plan = build_render_plan(stripped, self.analysis, RENDER_MODE, "p3")
        diffs = [
            f for f in TONE_FIELDS
            if getattr(full_plan.tone, f) != getattr(proxy_plan.tone, f)
        ]
        self.assertTrue(diffs, "proxy statistics happened to match exactly?")

    def test_disk_round_trip_preserves_the_sample_and_the_plan(self) -> None:
        from dngscan.gui import preview_cache as pc

        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "entry.npz"
            pc._write_disk_entry(cache_path, self.entry)
            loaded = pc._read_disk_entry(cache_path, FRAME, False)
            self.assertIsNotNone(loaded)
            sample = loaded.bundle._tone_plan_sample
            self.assertIsNotNone(sample)
            np.testing.assert_array_equal(
                np.asarray(sample), np.asarray(self.entry.bundle._tone_plan_sample)
            )
            full_plan = build_render_plan(
                self.bundle, self.analysis, RENDER_MODE, "p3"
            )
            loaded_plan = build_render_plan(
                loaded.bundle, self.analysis, RENDER_MODE, "p3"
            )
            self._assert_plans_equal(full_plan, loaded_plan)

    def test_rebalance_transforms_the_stored_sample(self) -> None:
        full_bal = rebalance_raw_bundle(self.bundle, "5500k")
        prev_bal = rebalance_raw_bundle(self.entry.bundle, "5500k")
        self.assertIsNotNone(prev_bal._tone_plan_sample)
        full_plan = build_render_plan(full_bal, self.analysis, RENDER_MODE, "p3")
        preview_plan = build_render_plan(prev_bal, self.analysis, RENDER_MODE, "p3")
        self._assert_plans_equal(full_plan, preview_plan)


if __name__ == "__main__":
    unittest.main()
