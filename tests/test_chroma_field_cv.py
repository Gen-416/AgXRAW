# SPDX-License-Identifier: GPL-3.0-or-later
"""Gates on the Stage-A chromaticity-field CV record (route C phase 1).

The measurement lives in docs/chroma_field_cv.json and changed no runtime
behaviour; these gates keep the RECORD honest: the stored numbers must
reproduce from the current spectral assets and fold seed, the candidate's
structural guarantees (white anchor, exposure homogeneity) must hold, and
the headline claim — the cubic field beats the deployed 3x3 on held-out
p95 for the high-metamerism stocks — must stay true if anyone re-fits.
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
        cls.folds = fc.cv_folds(cls.exposures.shape[0])

    def test_stored_record_reproduces_for_the_probe_stock(self) -> None:
        stored = json.loads(RECORD.read_text())["stocks"][self.STOCK]
        fresh_3x3 = fc.summarize(
            fc.heldout_errors_3x3(self.rgb, self.exposures, self.folds)
        )
        fresh_field = fc.summarize(
            fc.heldout_errors_field(self.xyz, self.exposures, self.folds, 3)
        )
        for key in ("p95_stop", "p99_stop", "max_stop"):
            self.assertAlmostEqual(
                stored["3x3"][key], fresh_3x3[key], delta=2e-3,
                msg=f"stored 3x3 {key} drifted from a fresh refit",
            )
            self.assertAlmostEqual(
                stored["poly3"][key], fresh_field[key], delta=2e-3,
                msg=f"stored poly3 {key} drifted from a fresh refit",
            )

    def test_white_anchor_is_exact(self) -> None:
        train = np.setdiff1d(np.arange(1, self.exposures.shape[0]), self.folds[0])
        coefs = fc.fit_field(self.xyz, self.exposures, train, 3)
        x, y, lum = fc.chromaticity(self.xyz)
        feats = fc.poly_features(x, y, 3)
        pred_white = lum[0] * 2.0 ** (feats[0] @ coefs.T)
        np.testing.assert_allclose(
            pred_white, self.exposures[0], rtol=1e-10,
            err_msg="the perfect reflector's exposure must be reproduced exactly",
        )

    def test_exposure_homogeneity_by_construction(self) -> None:
        train = np.setdiff1d(np.arange(1, self.exposures.shape[0]), self.folds[0])
        coefs = fc.fit_field(self.xyz, self.exposures, train, 3)
        x, y, lum = fc.chromaticity(self.xyz)
        feats = fc.poly_features(x, y, 3)
        base = lum[:, None] * 2.0 ** (feats @ coefs.T)
        for k in (0.25, 2.0, 7.5):
            scaled = fc.chromaticity(self.xyz * k)
            feats_k = fc.poly_features(scaled[0], scaled[1], 3)
            pred_k = scaled[2][:, None] * 2.0 ** (feats_k @ coefs.T)
            np.testing.assert_allclose(pred_k, k * base, rtol=1e-9)

    def test_cubic_field_beats_the_3x3_on_heldout_p95(self) -> None:
        err_3x3 = fc.heldout_errors_3x3(self.rgb, self.exposures, self.folds)
        err_field = fc.heldout_errors_field(self.xyz, self.exposures, self.folds, 3)
        self.assertLess(
            float(np.percentile(err_field, 95.0)),
            float(np.percentile(err_3x3, 95.0)) * 0.85,
            "the measured >=15% held-out p95 margin no longer holds",
        )


if __name__ == "__main__":
    unittest.main()
