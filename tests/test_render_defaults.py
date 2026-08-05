# SPDX-License-Identifier: GPL-3.0-or-later
"""Default-geometry contract for full-frame AgX and RAW-gated rendering."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from dngscan.tone import build_render_plan


class RenderDefaultTests(unittest.TestCase):
    @patch("dngscan.guidance.ensure_raw_guidance")
    @patch("dngscan.tone.build_color_geometry_plan", return_value=MagicMock())
    @patch("dngscan.tone.build_tone_compression_plan", return_value=MagicMock())
    @patch("dngscan.tone.scene_tone_metrics", return_value=MagicMock())
    @patch("dngscan.tone.tone_plan_sample_scene_rec2020", return_value=MagicMock())
    def test_gated_core_forces_pinned_darktable_base_geometry(
        self, sample: MagicMock, scene_metrics: MagicMock, tone_plan: MagicMock,
        color_plan: MagicMock, guidance: MagicMock
    ) -> None:
        bundle = SimpleNamespace(exposure_gain=1.0)
        build_render_plan(bundle, MagicMock(), "agx", tone_core="gated", agx_primaries="base")
        self.assertEqual(tone_plan.call_args.kwargs["agx_primaries"], "base")
        guidance.assert_called_once()

    @patch("dngscan.tone.build_color_geometry_plan", return_value=MagicMock())
    @patch("dngscan.tone.build_tone_compression_plan", return_value=MagicMock())
    @patch("dngscan.tone.scene_tone_metrics", return_value=MagicMock())
    @patch("dngscan.tone.tone_plan_sample_scene_rec2020", return_value=MagicMock())
    def test_full_frame_agx_keeps_explicit_geometry(
        self, sample: MagicMock, scene_metrics: MagicMock, tone_plan: MagicMock,
        color_plan: MagicMock
    ) -> None:
        bundle = SimpleNamespace(exposure_gain=1.0)
        build_render_plan(bundle, MagicMock(), "agx", tone_core="agx", agx_primaries="base")
        self.assertEqual(tone_plan.call_args.kwargs["agx_primaries"], "base")


if __name__ == "__main__":
    unittest.main()
