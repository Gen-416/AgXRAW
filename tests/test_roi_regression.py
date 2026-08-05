# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic ROI regression harness for gated DRT (Sigma fp DNG ROIs can be added later)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from dngscan.color import rgb_to_oklab
from dngscan.gated_drt import apply_gated_core
from dngscan.models import ColorGeometryPlan, ToneCompressionPlan
from dngscan.render import apply_agx_core

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "roi_regression.json"


def _plan(tone_core: str = "gated", agx_primaries: str = "base") -> ToneCompressionPlan:
    return ToneCompressionPlan(
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
        tone_core=tone_core,
        agx_primaries=agx_primaries,
        use_c1_endpoints=True,
    )


def _mean_chroma(rgb: np.ndarray) -> float:
    _, a, b = rgb_to_oklab(rgb.reshape(-1, 3), "srgb")
    return float(np.mean(np.hypot(a, b)))


class RoiRegressionTest(unittest.TestCase):
  def test_synthetic_portrait_midtone_differs_from_full_agx(self) -> None:
      rgb = np.asarray([[0.28, 0.11, 0.20], [0.24, 0.13, 0.32]], dtype=np.float32)
      masks = np.zeros_like(rgb)
      color = ColorGeometryPlan("srgb", 0.0, 0.0)
      gated = apply_gated_core(rgb, _plan(), color, masks)
      agx = apply_agx_core(rgb, _plan(tone_core="agx", agx_primaries="smooth"))
      self.assertGreater(abs(_mean_chroma(gated) - _mean_chroma(agx)), 1e-4)

  def test_fixture_expectations_if_present(self) -> None:
      if not _FIXTURES.exists():
          return
      cases = json.loads(_FIXTURES.read_text(encoding="utf-8"))
      color = ColorGeometryPlan("srgb", 0.0, 0.0)
      for case in cases:
          rgb = np.asarray(case["rgb"], dtype=np.float32).reshape(-1, 3)
          masks = np.asarray(case.get("masks", [[0, 0, 0]] * len(rgb)), dtype=np.float32)
          gated = apply_gated_core(rgb, _plan(), color, masks)
          metric = _mean_chroma(gated)
          if "min_chroma" in case:
              self.assertGreaterEqual(metric, float(case["min_chroma"]), case.get("name", ""))
          if "max_chroma" in case:
              self.assertLessEqual(metric, float(case["max_chroma"]), case.get("name", ""))


if __name__ == "__main__":
    unittest.main()
