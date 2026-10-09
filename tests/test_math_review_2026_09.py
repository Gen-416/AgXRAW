# SPDX-License-Identifier: GPL-3.0-or-later
"""Math review 2026-09-03 (four-way read-only review of the maths landed
since the 2026-08-27 audit: halation P5f, HDR batch 21, chroma NR, batches
22-25), pins for each item that changed code.

1. P1 halation give/take: the give term gated the POST-emulsion-scatter
   layer exposure while the take (spread) was accumulated from the
   unscattered scene. At preview pitches the scatter mix is the identity,
   which is where the balance figure was measured; at export pitch a 1-px
   source keeps 29-53% of its peak after the mix, the give gate opened less
   and take exceeded give by up to ~3.5x — the residual form created
   energy. Give now gates the pre-scatter exposure (halation_reinject_rows
   ``give_lin``).
2. P2 fingerprint: the chroma-NR map lives on the spread grid whose size
   the optics tier selects, so a chroma_nr-only export's bytes depend on
   the tier — the fingerprint carried the tier only when film optics were
   engaged.
3. P2 HDR batch 21: the shoulder anchor's chain rule was pinned only by a
   test that re-implemented it; the compiler itself is now checked
   (T_K^p and p·M_K relations between the vb=1 and vb≠1 compiles).
4. P3 grade id: the unified path named the raw strength while the render
   clamps it to [0, 1.5]; "none" carried a strength.
5. Documentation corrections (garrote fractions, luminance stage, band
   truncation, float32 accumulation determinism, one-sided bloom bracket,
   give cap non-conservation, §6.3 composed anchor, kernel residuals).
"""
from __future__ import annotations

import inspect
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]




class FingerprintCarriesTierForChromaNr(unittest.TestCase):
    def test_condition_includes_the_dial(self) -> None:
        from dngscan.gui import service

        src = inspect.getsource(service.run_export)
        block = src[src.find("spatial_budget_mib=("):]
        block = block[: block.find("else 0")]
        self.assertIn("float(chroma_nr) > 0.0", block)


class ShoulderAnchorChainRuleInTheCompiler(unittest.TestCase):
    def test_vb_compile_relates_to_the_unit_compile(self) -> None:
        from dngscan.hdr_agx_plan import compile_hdr_agx_plan
        from dngscan.hdr_curve import body_brightness_power
        from tests.test_hdr_native import _scene_plan

        base = compile_hdr_agx_plan(_scene_plan(view_brightness=1.0), analysis=SimpleNamespace())
        lifted_plan = _scene_plan(view_brightness=1.3)
        lifted = compile_hdr_agx_plan(lifted_plan, analysis=SimpleNamespace())
        self.assertTrue(base.tone.shoulder_segments and lifted.tone.shoulder_segments)
        seg0, seg1 = base.tone.shoulder_segments[0], lifted.tone.shoulder_segments[0]
        p = body_brightness_power(lifted_plan.tone)
        self.assertNotEqual(p, 1.0)
        # T_K^p in stops: z = log2(T/0.18) -> z' = p·(z + log2 0.18) − log2 0.18
        expected_z0 = p * (seg0.z0 + math.log2(0.18)) - math.log2(0.18)
        self.assertAlmostEqual(seg1.z0, expected_z0, places=6)
        # d/de (T^p) in stops is exactly p·M_K
        self.assertAlmostEqual(seg1.m0, p * seg0.m0, places=6)


class GradeIdNamesWhatRenders(unittest.TestCase):
    def test_strength_is_clamped_and_none_has_none(self) -> None:
        from dngscan.grade import resolve_grade_id

        self.assertEqual(resolve_grade_id({"grade": "look:x", "gradeStrength": 2.0}), ("look:x", 1.5))
        self.assertEqual(resolve_grade_id({"grade": "look:x", "gradeStrength": -1.0}), ("look:x", 0.0))
        self.assertEqual(resolve_grade_id({"grade": "none", "gradeStrength": 0.3}), ("none", 1.0))


class DocumentsStateTheReviewedBoundaries(unittest.TestCase):
    def test_wording(self) -> None:
        chroma = (ROOT / "docs" / "CHROMA_NR.zh-CN.md").read_text(encoding="utf-8")
        self.assertIn("BayesShrink", chroma)
        hdr_rs = (ROOT / "rust" / "src" / "hdr.rs").read_text(encoding="utf-8")
        self.assertIn("ABI v11: every matrix stage is an exact float64 stage", hdr_rs)
        plan = (ROOT / "docs" / "HDR_AGX_V2_IMPLEMENTATION_PLAN.zh-CN.md").read_text(encoding="utf-8")
        self.assertIn("T_K = T_body(K)^p", plan)


if __name__ == "__main__":
    unittest.main()
