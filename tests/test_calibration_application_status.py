# SPDX-License-Identifier: GPL-3.0-or-later
"""Import, candidate matching, analysis evidence and pixel use are distinct."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dngscan import calibration, priors
from dngscan.analysis import analyze
from dngscan.gui.page import PAGE
from dngscan.gui.preview_cache import _analysis_from_json, _analysis_to_json
from dngscan.gui.service import detected_scene_params
from dngscan.noise_model import NoiseModel
from dngscan.raw_io import load_raw
from dngscan.render import render_output_encoded_float
from dngscan.tone import build_render_plan
from tests.test_spectral_fallback_pipeline import write_noise_dng
from tests.test_user_calibration import collect_profile, unresolved_collect_profile


class CalibrationApplicationStatusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "profiles"))
        environment.start()
        self.addCleanup(environment.stop)

    def install(self, profile):
        path = self.root / "profile.json"
        path.write_text(json.dumps(profile))
        return calibration.import_calibration(path, shutter_override="any")

    def pipeline(self, profile, *, file_profile=False, amount=0.):
        installed = self.install(profile)
        path = self.root / "capture.dng"
        write_noise_dng(path, file_profile=file_profile)
        bundle = load_raw(path, scene_half_size=True)
        analysis, _, _ = analyze(bundle, 4)
        plan = build_render_plan(bundle, analysis, "agx", "p3", chroma_nr=amount)
        render_output_encoded_float(bundle, analysis, "p3", plan)
        detected = detected_scene_params(bundle, analysis, plan)
        return installed, bundle, analysis, detected["calibrations"][0]

    def test_matching_statistical_model_is_not_personal_pixel_correction(self):
        installed, bundle, analysis, row = self.pipeline(collect_profile())
        self.assertEqual(row["import_status"], "imported")
        self.assertEqual(row["match_status"], "matched")
        self.assertTrue(row["selected_for_analysis"])
        self.assertEqual(row["status"], "usable")
        self.assertEqual(row["analysis_status"], "used")
        self.assertEqual(row["analysis_capabilities"]["variance_coefficients"], "used")
        self.assertEqual(row["capability_sources"]["shot_variance"], analysis.noise_model.source)
        self.assertIn(installed["id"], row["actual_model"]["source"])
        self.assertEqual(row["apply_status"], "not-applied")
        self.assertEqual(row["chroma_application"]["status"], "not-applied")
        self.assertEqual(row["chroma_application"]["pipeline_status"], "disabled")

    def test_actual_chroma_application_does_not_claim_black_or_flat_correction(self):
        _, bundle, _, row = self.pipeline(collect_profile(), amount=.15)
        self.assertEqual(bundle.chroma_nr_status, "active-approximate")
        self.assertEqual(row["chroma_application"]["status"], "applied")
        self.assertEqual(row["apply_status"], "not-applied")

    def test_dn_rejection_is_visible_even_though_candidate_conditions_match(self):
        profile = collect_profile()
        profile["fwc_e"] = 10000 * 4
        for fallback in (False, True):
            with self.subTest(dng_fallback=fallback):
                _, _, analysis, row = self.pipeline(profile, file_profile=fallback, amount=.15)
                self.assertEqual(row["match_status"], "matched")
                self.assertEqual(row["analysis_status"], "rejected")
                self.assertEqual(row["analysis_reason"], "unmatched-dn-scale")
                self.assertEqual(row["status"], "not-applied")
                self.assertEqual(row["gain_status"], "not-applied")
                self.assertEqual(row["reason"], "unmatched-dn-scale")
                self.assertEqual(row["analysis_capabilities"]["variance_coefficients"], "not-used")
                self.assertEqual(row["actual_model"]["source"], analysis.noise_model.source)
                if fallback:
                    self.assertEqual(row["actual_model"]["source"], "DNG NoiseProfile")
                    self.assertEqual(row["capability_sources"]["shot_variance"], "DNG NoiseProfile")
                    # DNG-based chroma can run without crediting the rejected profile.
                    self.assertEqual(row["chroma_application"]["pipeline_status"], "active-approximate")
                    self.assertEqual(row["chroma_application"]["status"], "not-applied")

    def test_dng_coefficients_and_inherited_user_spectrum_keep_separate_sources(self):
        profile = unresolved_collect_profile()
        profile["noise_whiteness_h_log2iso"] = [[math.log2(200), .1]]
        installed, bundle, analysis, row = self.pipeline(profile, file_profile=True, amount=.15)
        user_source = f"User JPTC calibration ({installed['id']})"
        self.assertEqual(analysis.noise_model.source, "DNG NoiseProfile")
        self.assertEqual(row["analysis_status"], "partial")
        self.assertEqual(row["analysis_capabilities"], {
            "gain_evidence": "used", "variance_coefficients": "not-used", "spectral_constraints": "used"})
        self.assertEqual(row["capability_sources"]["shot_variance"], "DNG NoiseProfile")
        self.assertEqual(row["capability_sources"]["gain_evidence"], user_source)
        self.assertEqual(row["capability_sources"]["spectral_constraints"], {"h": user_source})
        self.assertEqual(row["actual_model"]["fallback_reason"], "read-noise-unresolved")
        self.assertEqual(bundle.chroma_nr_status, "skipped")
        self.assertIn("correlated-noise", row["chroma_application"]["reason"])
        self.assertEqual(row["apply_status"], "not-applied")

    def test_match_without_analysis_never_claims_actual_application(self):
        self.install(collect_profile())
        row, = calibration.calibration_diagnostics("SIGMA", "fp", iso=200)
        self.assertEqual(row["status"], "usable")  # Legacy candidate status.
        self.assertEqual(row["analysis_status"], "not-evaluated")
        self.assertFalse(row["selected_for_analysis"])
        self.assertEqual(row["apply_status"], "not-applied")
        self.assertEqual(row["chroma_application"]["status"], "not-applied")

    def test_inactive_or_unselected_profile_cannot_take_credit_for_another_model(self):
        installed = self.install(collect_profile())
        calibration.set_calibration_active(installed["id"], False)
        analysis = SimpleNamespace(prior_id="another prior", gain_e_per_dn=2.,
            noise_model=NoiseModel(status="valid", source="DNG NoiseProfile"))
        row, = calibration.calibration_diagnostics("SIGMA", "fp", iso=200, analysis=analysis,
            chroma_nr={"status": "active-approximate"})
        self.assertEqual(row["reason"], "inactive")
        self.assertEqual(row["analysis_status"], "not-selected")
        self.assertFalse(row["selected_for_analysis"])
        self.assertEqual(row["analysis_capabilities"]["variance_coefficients"], "not-used")
        self.assertEqual(row["chroma_application"]["status"], "not-applied")

    def test_composed_coefficients_and_indexed_spectrum_are_attributed_per_profile(self):
        first = collect_profile()
        first["id"] = "gain profile"
        first_install = self.install(first)
        second = collect_profile()
        second["id"] = "dark profile"
        second_install = self.install(second)
        first_source = f"User JPTC calibration ({first_install['id']})"
        second_source = f"User JPTC calibration ({second_install['id']})"
        selected = {"id": "gain profile", "source": "Compatible capability composition",
            "calibration_id": first_install["id"],
            "capability_sources": {"gain": first_source, "read_noise": second_source,
                                   "stored_dark_variance": second_source},
            "independent_constraint_priors": [{"source": second_source}]}
        model = NoiseModel(status="valid", source=selected["source"],
                           prior_id=selected["id"], prior_source=selected["source"],
                           prior_calibration_id=first_install["id"],
                           capability_sources=selected["capability_sources"],
                           spectral_sources={"source0:h": second_source},
                           spectral_ratios={"source0:h": .1}, correlation="measured-spectral-imbalance")
        analysis = SimpleNamespace(prior_id=selected["id"], noise_model=model, gain_e_per_dn=2.)
        rows = calibration.calibration_diagnostics("SIGMA", "fp", iso=200,
            analysis=analysis, chroma_nr={"status": "skipped", "reason": "spectral"})
        indexed = {row["id"]: row for row in rows}
        self.assertEqual(indexed[first_install["id"]]["analysis_capabilities"], {
            "gain_evidence": "used", "variance_coefficients": "used", "spectral_constraints": "not-used"})
        dark = indexed[second_install["id"]]
        self.assertEqual(dark["analysis_capabilities"], {
            "gain_evidence": "not-used", "variance_coefficients": "used", "spectral_constraints": "used"})
        self.assertEqual(dark["capability_sources"]["spectral_constraints"], {"source0:h": second_source})
        self.assertEqual(dark["capability_sources"]["constant_variance"], second_source)
        self.assertEqual(dark["apply_status"], "not-applied")

    def test_changed_catalogue_does_not_relabel_cached_analysis_sources(self):
        installed, _, analysis, _ = self.pipeline(collect_profile())
        different = collect_profile()
        different["id"] = "newly imported measurement"
        self.install(different)
        # Reporting may read the installed records, but must not select a new
        # prior and attribute its capabilities to the completed old analysis.
        with patch.object(priors, "find_priors", side_effect=AssertionError("must use final model provenance")):
            rows = calibration.calibration_diagnostics("SIGMA", "fp", iso=200,
                analysis=analysis, chroma_nr={"status": "active-approximate"})
        row = next(row for row in rows if row["id"] == installed["id"])
        newer = next(row for row in rows if row["id"] != installed["id"])
        self.assertTrue(row["selected_for_analysis"])
        self.assertEqual(row["analysis_status"], "used")
        self.assertEqual(newer["analysis_status"], "not-selected")
        self.assertEqual(newer["chroma_application"]["status"], "not-applied")

    def test_cached_analysis_preserves_fallback_and_spectral_contributors(self):
        profile = unresolved_collect_profile()
        profile["noise_whiteness_h_log2iso"] = [[math.log2(200), .1]]
        installed, _, analysis, before = self.pipeline(profile, file_profile=True, amount=.15)
        restored = _analysis_from_json(json.loads(json.dumps(_analysis_to_json(analysis))))
        self.assertEqual(restored.noise_model.capability_sources, analysis.noise_model.capability_sources)
        self.assertEqual(restored.noise_model.spectral_sources, analysis.noise_model.spectral_sources)
        self.assertEqual(restored.noise_model.prior_calibration_id, installed["id"])
        row, = calibration.calibration_diagnostics("SIGMA", "fp", iso=200,
            analysis=restored, chroma_nr={"status": "skipped", "reason": "correlated-noise"})
        self.assertEqual(row["capability_sources"], before["capability_sources"])
        self.assertEqual(row["actual_model"], before["actual_model"])
        self.assertEqual(row["analysis_capabilities"], before["analysis_capabilities"])

    def test_gui_uses_analysis_and_pixel_labels_instead_of_unqualified_usable(self):
        self.assertIn("候选条件匹配", PAGE)
        self.assertIn("分析未采用：", PAGE)
        self.assertIn("噪声系数来自 DNG 声明", PAGE)
        self.assertIn("个体黑场／平场像素校正未应用", PAGE)


if __name__ == "__main__":
    unittest.main()
