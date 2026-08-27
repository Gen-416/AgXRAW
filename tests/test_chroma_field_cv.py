# SPDX-License-Identifier: GPL-3.0-or-later
"""Gates on the Stage-A chromaticity-field CV record (route C, runtime-faithful).

The record's decision numbers must describe the operator that ships
(external review 2026-08-27, F1-F3): per fold the field is baked to its LUT
with the SAME pure function the asset builder uses and evaluated through the
SAME runtime dispatcher; the white anchor holds by parametrization; the
adoption is a frequency over repeated fold draws. These gates keep the
record honest against the current spectral assets and pin those structural
promises.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fit_chroma_field as fc  # noqa: E402
import fit_film_curve as ff  # noqa: E402

sys.path.pop(0)

RECORD = ROOT / "docs" / "chroma_field_cv.json"


class ChromaFieldRecordTests(unittest.TestCase):
    STOCK = "portra400"

    @classmethod
    def setUpClass(cls) -> None:
        cls.xyz, cls.rgb, cls.exposures, cls.m = fc.stimulus_and_exposures(
            ff.STOCKS[cls.STOCK]
        )
        cls.record = json.loads(RECORD.read_text())
        cls.entry = cls.record["stocks"][cls.STOCK]

    def test_record_declares_the_runtime_faithful_discipline(self) -> None:
        r = self.record
        self.assertEqual(r["lut"]["n"], fc.LUT_N)
        self.assertEqual(r["lut"]["blend_sigma_cells"], fc.BLEND_SIGMA_CELLS)
        self.assertEqual(r["lut"]["dilate_cells"], fc.DILATE_CELLS)
        self.assertEqual(r["lut"]["edge_taper_cells"], fc.EDGE_TAPER_CELLS)
        self.assertEqual(r["lut"]["edge_zero_cells"], fc.EDGE_ZERO_CELLS)
        self.assertEqual(tuple(r["seeds"]), fc.SEEDS)
        self.assertIn("residual_caveat", r)
        self.assertNotIn("cannot be removed", json.dumps(r["residual_caveat"]))
        # third review F5/F6: family statistics over unique responses; ridge
        # sensitivity measured on the deployed operator
        self.assertLess(r["unique_responses"], len(r["stocks"]))
        self.assertIn("response_id", self.entry)
        self.assertEqual(
            self.record["stocks"]["portra800"]["response_id"],
            self.record["stocks"]["portra800push1"]["response_id"],
        )
        folds0 = fc.cv_folds(self.exposures.shape[0], fc.SEEDS[0])
        runtime_1e2 = fc.summarize(fc.heldout_errors_runtime(
            self.xyz, self.rgb, self.exposures, self.m, folds0, ridge=1e-2, use_field=True
        ))["p99_stop"]
        self.assertAlmostEqual(self.entry["ridge_sensitivity_p99"]["0.01"], runtime_1e2, delta=2e-3)

    def test_stored_runtime_numbers_reproduce_on_the_first_seeds(self) -> None:
        """A 3-seed re-run must land inside the recorded 30-seed IQR of the
        field p99 and reproduce the 3x3 median within the seed spread —
        the record must come from the code that is here."""
        folds_by_seed = [fc.cv_folds(self.exposures.shape[0], s) for s in fc.SEEDS[:3]]
        field = [
            fc.summarize(fc.heldout_errors_runtime(
                self.xyz, self.rgb, self.exposures, self.m, f, use_field=True
            ))["p99_stop"]
            for f in folds_by_seed
        ]
        lo, hi = self.entry["runtime"]["field_p99_iqr"]
        self.assertLessEqual(float(np.median(field)), hi + 0.03)
        self.assertGreaterEqual(float(np.median(field)), lo - 0.03)
        obs = fc.summarize(fc.heldout_errors_runtime(
            self.xyz, self.rgb, self.exposures, self.m, folds_by_seed[0], use_field=False
        ))
        self.assertAlmostEqual(
            obs["p99_stop"], self.entry["runtime"]["3x3"]["p99_stop"], delta=0.05
        )

    def test_adoption_is_a_frequency_over_repeated_folds(self) -> None:
        votes, total = self.entry["runtime"]["adopt_votes"].split("/")
        self.assertEqual(int(total), len(fc.SEEDS))
        self.assertAlmostEqual(
            self.entry["runtime"]["adopt_frequency"], int(votes) / int(total), places=4
        )
        self.assertEqual(
            fc.adopts(self.entry),
            self.entry["runtime"]["adopt_frequency"] >= fc.ADOPT_FREQUENCY,
        )

    def test_white_anchor_holds_by_parametrization(self) -> None:
        """No intercept patch: the fitted model reproduces the white board's
        log2(E/Y) EXACTLY at the white chromaticity for any ridge."""
        train = np.setdiff1d(np.arange(1, self.exposures.shape[0]), fc.cv_folds(self.exposures.shape[0])[0])
        x, y, lum = fc.chromaticity(self.xyz)
        for ridge in (1e-6, 1e-2, 1.0):
            model = fc.fit_field(self.xyz, self.exposures, train, 3, ridge)
            at_white = fc.eval_field(model, x[:1], y[:1])[0]
            np.testing.assert_allclose(
                at_white, np.log2(self.exposures[0] / lum[0]), rtol=1e-12, atol=1e-12
            )
        self.assertGreater(model["cond"], 1.0)
        self.assertLess(model["cond"], 1e6, "standardized design conditioning regressed")

    def test_bake_is_pure_and_anchor_survives_the_lut(self) -> None:
        from dngscan.film_v2_math import layer_log_exposure, stage_a_log_exposure

        train = np.arange(1, self.exposures.shape[0])
        observer = fc.fit_3x3(self.rgb, self.exposures, train)
        model = fc.fit_field(self.xyz, self.exposures, train, 3)
        a = fc.bake_lut(model, self.xyz, self.m, observer)
        b = fc.bake_lut(model, self.xyz, self.m, observer)
        np.testing.assert_array_equal(a[0], b[0])
        stock = fc._runtime_stock(*a, observer)
        g = 0.18 * np.exp2(np.linspace(-6.0, 4.0, 21))
        grey = np.repeat(g[:, None], 3, axis=1)
        np.testing.assert_allclose(
            stage_a_log_exposure(grey, stock), layer_log_exposure(grey, observer), atol=5e-7
        )

    def test_deployed_operator_tracks_the_polynomial(self) -> None:
        """The reason F1 existed: on the training rows the baked operator
        and the continuous polynomial used to differ by 0.30 stop median at
        p99 (the blend band ate the hull boundary). With the dilated band
        they must agree within the LUT's own interpolation class."""
        folds_list = fc.cv_folds(self.exposures.shape[0])
        runtime = fc.summarize(fc.heldout_errors_runtime(
            self.xyz, self.rgb, self.exposures, self.m, folds_list, use_field=True
        ))
        cont = fc.summarize(fc.heldout_errors_continuous(self.xyz, self.exposures, folds_list))
        self.assertLess(abs(runtime["p99_stop"] - cont["p99_stop"]), 0.08)
        self.assertLess(abs(runtime["p95_stop"] - cont["p95_stop"]), 0.05)

    def test_cubic_field_beats_the_3x3_on_heldout_p95(self) -> None:
        r = self.entry["runtime"]
        self.assertLess(r["field"]["p95_stop"], r["3x3"]["p95_stop"] * 0.85)


if __name__ == "__main__":
    unittest.main()
