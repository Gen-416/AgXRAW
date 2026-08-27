# SPDX-License-Identifier: GPL-3.0-or-later
"""The report names the Stage A model that ACTUALLY ran and its held-out
residual (external review 2026-08-27, P0.3), off numbers baked into the
schema-8 asset from the route-C and route-D records — not the 3x3
baseline quoted for every stock. Also pins that no illuminant tier keys
exist in the ABI (route D: measured, withdrawn)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

from dngscan.film_develop import _V2_DIR, _load_v2

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fit_chroma_field as fcf  # noqa: E402

sys.path.pop(0)


def _line(preset: str) -> str:
    from tests.golden_support import build_daylight_wide_dr
    from dngscan.report import jpeg_tone_plan_cn
    from dngscan.tone import build_render_plan

    scene = build_daylight_wide_dr()
    plan = build_render_plan(
        scene.bundle, scene.analysis, "agx", "srgb",
        film_curve=preset, film_mode="full", film_crossover="datasheet",
    )
    return jpeg_tone_plan_cn(None, None, "agx", plan.tone, "srgb")


class StageAAssetNumbersTests(unittest.TestCase):
    def test_numbers_come_from_the_records_and_no_tier_keys_exist(self) -> None:
        chroma = json.loads((ROOT / "docs" / "chroma_field_cv.json").read_text())["stocks"]
        illum = json.loads((ROOT / "docs" / "illuminant_tier_cv.json").read_text())["stocks"]
        for name in ("portra400", "pro400h", "vision3250d"):
            with self.subTest(stock=name):
                z = np.load(_V2_DIR / f"{name}.npz", allow_pickle=False)
                self.assertFalse([k for k in z.files if k.startswith("tier_")])
                stock = _load_v2(name)[0]
                # The asset's model is the record's frequency decision (F3).
                self.assertEqual(
                    stock["stage_a_model"], "field" if fcf.adopts(chroma[name]) else "3x3"
                )
                key = "field" if stock["stage_a_model"] == "field" else "3x3"
                self.assertAlmostEqual(stock["stage_a_p99_stop"], chroma[name]["runtime"][key]["p99_stop"])
                self.assertAlmostEqual(stock["stage_a_3x3_p99_stop"], chroma[name]["runtime"]["3x3"]["p99_stop"])
                self.assertAlmostEqual(
                    stock["stage_a_p99_under_a"], illum[name]["A"]["shipped_assumed"]["p99_stop"]
                )
                self.assertAlmostEqual(
                    stock["stage_a_p99_under_led"], illum[name]["LED-B3"]["shipped_assumed"]["p99_stop"]
                )
        self.assertEqual(_load_v2("portra400")[0]["stage_a_model"], "field")

    def test_provenance_covers_every_numerical_input(self) -> None:
        """F8 (review 2026-08-27): the field depends on more than the
        negative profile — the reflectance library, both CV records, the
        fit/bake parameters and the CMF/SPD library version. Their hashes
        must match the files in this tree."""
        import hashlib

        z = np.load(_V2_DIR / "portra400.npz", allow_pickle=False)
        names = [str(n) for n in z["stage_a_input_names"]]
        shas = [str(h) for h in z["stage_a_input_sha256"]]
        self.assertEqual(len(names), 3)
        for n, h in zip(names, shas):
            self.assertEqual(hashlib.sha256((ROOT / n).read_bytes()).hexdigest(), h, n)
        params = json.loads(str(np.asarray(z["stage_a_params"])))
        for key in ("ridge", "seeds", "lut_n", "blend_sigma_cells", "dilate_cells",
                    "colour_science", "numpy", "scipy", "adopt_frequency"):
            self.assertIn(key, params)
        self.assertEqual(params["lut_n"], fcf.LUT_N)
        self.assertEqual(params["seeds"], len(fcf.SEEDS))


class StageAReportLineTests(unittest.TestCase):
    def test_field_stock_names_the_field_and_the_baseline(self) -> None:
        stock = _load_v2("portra400")[0]
        line = _line("portra400")
        self.assertIn("StageA=D55色度场", line)
        self.assertIn(f"held-out p99 {stock['stage_a_p99_stop']:.2f}stop", line)
        self.assertIn(f"3×3基线{stock['stage_a_3x3_p99_stop']:.2f}", line)
        self.assertIn(
            f"钨丝光下{stock['stage_a_p99_under_a']:.2f}、高显色LED下{stock['stage_a_p99_under_led']:.2f}",
            line,
        )
        self.assertIn("光源假设=D55（实测无需分档）", line)
        self.assertNotIn("观察者拟合残差", line)

    def test_3x3_stock_says_so_without_a_baseline_clause(self) -> None:
        """Every stock currently adopts the field, so the 3x3 branch of the
        report line is exercised on a loaded stock with its field withdrawn
        (the exact dict shape the loader builds for a retained stock) —
        injected into the loader cache for the duration of the call, never
        skipped."""
        from dngscan import film_develop as fd

        name = "portra400"
        stock, media = fd._load_v2(name)
        retained = dict(stock)
        retained["chroma_lut"] = None
        retained["chroma_domain"] = None
        retained["chroma_xyz_from_rec2020"] = None
        retained["stage_a_model"] = "3x3"
        retained["stage_a_p99_stop"] = retained["stage_a_3x3_p99_stop"]
        saved = fd._V2_CACHE.get(name)
        fd._V2_CACHE[name] = (retained, media)
        try:
            line = _line(name)
        finally:
            if saved is not None:
                fd._V2_CACHE[name] = saved
            else:
                fd._V2_CACHE.pop(name, None)
        self.assertIn("StageA=D553×3", line)
        self.assertIn(f"held-out p99 {retained['stage_a_p99_stop']:.2f}stop", line)
        self.assertNotIn("3×3基线", line)

if __name__ == "__main__":
    unittest.main()
