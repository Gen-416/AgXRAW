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


class HalationGiveTakeGateTheSameExposure(unittest.TestCase):
    def _setup(self):
        from dngscan.film_optics import (
            apply_scatter_mix,
            area_decimate_rows,
            halation_component_source,
            halation_spread_map_from_sources,
        )
        from dngscan.film_optics_assets import DEFAULT_STOCK_OPTICS, load_stock_optics

        stock = load_stock_optics(DEFAULT_STOCK_OPTICS)
        if stock.emulsion_scatter is None:
            self.skipTest("default stock declares no emulsion scatter")
        h, w = 96, 128
        mm_per_px = 36.0 / 6016.0  # export pitch: the scatter halo resolves
        geometry_w_mm = w * mm_per_px
        e = np.full((h, w, 3), 0.2, dtype=np.float32)
        e[h // 2, w // 2, :] = np.float32(2.0 ** 6.5)  # 1-px source
        e_s = apply_scatter_mix(e, mm_per_px, stock.emulsion_scatter)
        ref = np.ones(3, dtype=np.float32)
        dh, dw = h // 7, w // 7  # decimating grid, like a 61 MP export
        sources = []
        for comp in stock.halation.components:
            acc = np.zeros((dh, dw, 3), dtype=np.float64)
            area_decimate_rows(
                halation_component_source(e, ref, comp), 0, h, w, dh, dw, acc
            )
            sources.append(acc.astype(np.float32))
        spread = halation_spread_map_from_sources(
            iter(sources), geometry_w_mm, stock.halation, (dh, dw)
        )
        return stock.halation, e, e_s, ref, spread, h, w

    def test_pre_scatter_give_balances_the_unscattered_take(self) -> None:
        from dngscan.film_optics import halation_reinject_rows, upsample_rows

        hal, e, e_s, ref, spread, h, w = self._setup()
        if hal.dc_mode != "residual":
            self.skipTest("stock is not on the residual reinject")
        log_e = np.log10(np.maximum(e_s, 1e-12)).reshape(-1, 3).astype(np.float64)
        # the spread map holds per-cell MEANS; its full-resolution mass is
        # what the reinject adds (upsample_rows), so normalize by that
        take = upsample_rows(spread, 0, h, h, w)
        take_total = take.sum(axis=(0, 1))
        self.assertGreater(float(take_total.max()), 0.0)

        fixed = 10.0 ** halation_reinject_rows(
            log_e, spread, ref, 0, h, h, w, hal, 1.0, give_lin=e
        ).reshape(h, w, 3)
        old = 10.0 ** halation_reinject_rows(
            log_e, spread, ref, 0, h, h, w, hal, 1.0
        ).reshape(h, w, 3)
        # frame-wide layer-exposure change relative to the transferred mass
        # on the layers that receive halation: the residual form must
        # conserve (|Δ| small); gating the scattered exposure created energy
        # (Δ >> 0). Layers without take are excluded — there the only change
        # is the reinject's floor on the scatter mix's negative lobe (a
        # separate finding, pinned below).
        active = take_total > 1e-6 * take_total.max()
        d_fixed = (fixed - e_s).sum(axis=(0, 1))[active] / take_total[active]
        d_old = (old - e_s).sum(axis=(0, 1))[active] / take_total[active]
        self.assertLess(float(np.abs(d_fixed).max()), 0.05, d_fixed)
        self.assertGreater(float(d_old.max()), 0.3, d_old)

    def test_production_path_passes_the_pre_scatter_exposure(self) -> None:
        from dngscan import film_develop

        src = inspect.getsource(film_develop)
        self.assertIn("pre_scatter_lin = e_lin.copy()", src)
        self.assertIn("give_lin=pre_scatter_lin,", src)


class FingerprintCarriesTierForChromaNr(unittest.TestCase):
    def test_condition_includes_the_dial(self) -> None:
        from dngscan.gui import service

        src = inspect.getsource(service.run_export)
        block = src[src.find("optics_budget_mib=("):]
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
        chroma = (ROOT / "dngscan" / "chroma_nr.py").read_text(encoding="utf-8")
        self.assertIn("1/(1 + (d/T)²)", chroma)
        self.assertIn("LUMINANCE, at the scene stage", chroma)
        develop = (ROOT / "dngscan" / "film_develop.py").read_text(encoding="utf-8")
        self.assertIn("not byte-identical", develop)
        self.assertIn("ONE-SIDED approximation", develop)
        optics = (ROOT / "dngscan" / "film_optics.py").read_text(encoding="utf-8")
        self.assertIn("stated non-conservation", optics)
        self.assertIn("log floor converts", optics)
        hdr_h = (ROOT / "cpp" / "include" / "dngscan_fast" / "hdr_core.h").read_text(encoding="utf-8")
        self.assertIn("ABI v11: exact float64 stages", hdr_h)
        plan = (ROOT / "docs" / "HDR_AGX_V2_IMPLEMENTATION_PLAN.zh-CN.md").read_text(encoding="utf-8")
        self.assertIn("T_K = T_body(K)^p", plan)
        optics_doc = (ROOT / "docs" / "FILM_OPTICS_V2_PLAN.zh-CN.md").read_text(encoding="utf-8")
        self.assertIn("2026-09-03 数学审查（P1）", optics_doc)


if __name__ == "__main__":
    unittest.main()
