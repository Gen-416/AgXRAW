# SPDX-License-Identifier: GPL-3.0-or-later
"""NR memory tiers respect the independent CFA sample-count contract."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dngscan.analysis import analyze
from dngscan.gui.service import chroma_nr_capability
from dngscan.noise_propagation import chroma_nr_skip_reason
from dngscan.raw_io import load_raw
from dngscan.render import render_output_encoded_float
from dngscan.spatial import chroma_nr_grid_shape, spread_grid_shape
from dngscan.tone import build_render_plan
from tests.test_general_raw_fallback import write_uncalibrated_dng


class ChromaGridPlanningTests(unittest.TestCase):
    def test_review_twelve_megapixel_full_and_half_grids(self):
        bundle = SimpleNamespace(noise_decode={"sensor_window_shape": [3000, 4000]})
        for tier, expected in (("512", (1056, 1408)), ("1024", (1500, 2000))):
            with self.subTest(tier=tier), patch.dict(os.environ, DNGSCAN_SPATIAL_BUDGET_MIB=tier):
                self.assertEqual(chroma_nr_grid_shape(bundle, 3000, 4000), expected)
                self.assertEqual(chroma_nr_grid_shape(bundle, 1500, 2000), expected)
                dh, dw = expected
                self.assertGreaterEqual(3000 * 4000 / (dh * dw), 4.)
        # Generic spatial sampling is unchanged; only calibrated NR needs this bound.
        with patch.dict(os.environ, DNGSCAN_SPATIAL_BUDGET_MIB="1024"):
            self.assertEqual(spread_grid_shape(3000, 4000), (1536, 2048))

    def test_native_geometry_rounding_orientation_and_released_raw(self):
        for native in ((2999, 4001), (31, 4001), (3, 4001), (1, 4001)):
            for flip in (0, 4):
                bundle = SimpleNamespace(scene_sensor_window_shape=native,
                                         orientation_flip=flip, raw_image=None)
                height, width = native[::-1] if flip else native
                for tier in ("512", "1024"):
                    with self.subTest(native=native, flip=flip, tier=tier), \
                         patch.dict(os.environ, DNGSCAN_SPATIAL_BUDGET_MIB=tier):
                        dh, dw = chroma_nr_grid_shape(bundle, height, width)
                        self.assertLessEqual(dh, height)
                        self.assertLessEqual(dw, width)
                        self.assertGreaterEqual(native[0] * native[1] / (dh * dw), 4.)

    def test_real_full_decode_remains_nr_capable_at_both_memory_tiers(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(Path(directory) / "profiles")):
            source = Path(directory) / "sensor.dng"
            write_uncalibrated_dng(source, profile=True)
            for half in (False, True):
                bundle = load_raw(source, scene_half_size=half)
                analysis, _, _ = analyze(bundle, 4, diagnostics=False)
                self.assertEqual(analysis.noise_model.status, "valid")
                self.assertTrue(bundle.noise_decode["supported"])
                plan = build_render_plan(bundle, analysis, "agx", "srgb", chroma_nr=1.)
                disabled = replace(plan, tone=replace(plan.tone, chroma_nr=0.))
                reference = render_output_encoded_float(bundle, analysis, "srgb", disabled)
                outputs = []
                for tier in ("512", "1024"):
                    with self.subTest(half=half, tier=tier), \
                         patch.dict(os.environ, DNGSCAN_SPATIAL_BUDGET_MIB=tier):
                        self.assertTrue(chroma_nr_capability(bundle, analysis)["available"])
                        output = render_output_encoded_float(bundle, analysis, "srgb", plan)
                        self.assertEqual(bundle.chroma_nr_status, "active-approximate")
                        self.assertGreater(float(np.max(np.abs(output - reference))), 0.)
                        outputs.append(output)
                np.testing.assert_array_equal(outputs[0], outputs[1])
                # The downstream check still rejects unsupported callers that
                # bypass planning and request one cell per native sensel.
                reason = chroma_nr_skip_reason(bundle, analysis.noise_model, (128, 128))
                self.assertEqual(reason, "coarse noise approximation requires at least four sensels per cell")


if __name__ == "__main__":
    unittest.main()
