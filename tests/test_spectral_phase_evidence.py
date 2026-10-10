# SPDX-License-Identifier: GPL-3.0-or-later
"""CFA positions cannot stand in for colours or average away noise constraints."""
from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dngscan import calibration
from dngscan.hdr_agx_plan import compile_tail_snr_gate
from dngscan.noise_spectrum import SCHEMA, dark_phase_mapping, validate_spectrum
from tests import test_spectral_fallback_pipeline as fallback_tests
from tests.test_user_calibration import collect_profile, unresolved_collect_profile


PATTERNS = {"RGGB": (0, 1, 3, 2), "BGGR": (2, 3, 1, 0),
            "GRBG": (1, 0, 2, 3), "GBRG": (1, 2, 0, 3)}
PHASES = ("C00", "C01", "C10", "C11")


def mapping(pattern="RGGB"):
    rows = [{"ISO": str(iso), "Channel": phase, "ColorIndex": str(cid)}
            for iso in (100, 200) for phase, cid in zip(PHASES, PATTERNS[pattern])]
    return dark_phase_mapping({"CfaPattern": "RGBG"}, rows)


def evidence(ratios, *, pattern="RGGB"):
    return {"schema": SCHEMA, "mapping": mapping(pattern),
            "axes": {"h": {"ratios_log2iso": {phase: [[math.log2(200), ratio]]
                                               for phase, ratio in zip(PHASES, ratios)}}}}


class SpectralPhaseImportTests(unittest.TestCase):
    def test_four_bayer_layouts_select_measured_green_positions(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "spectrum-h.csv"
            for name in PATTERNS:
                measured = mapping(name)
                ratios = [.1 if measured["phases"][p]["color"] == "G" else 1. for p in PHASES]
                path.write_text("freq," + ",".join(f"iso200_{p}_diff" for p in PHASES)
                                + "\n0.1,1,1,1,1\n0.4," + ",".join(map(str, ratios)) + "\n")
                with self.subTest(pattern=name):
                    self.assertEqual(calibration.read_whiteness(path, phase_mapping=measured), {200: .1})

    def test_opposite_green_anomalies_are_never_averaged(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "spectrum.csv"
            for ratios in ((.1, 1.9), (1.9, .1)):
                path.write_text("freq,iso200_C01_diff,iso200_C10_diff\n0.1,1,1\n"
                                + f"0.4,{ratios[0]},{ratios[1]}\n")
                result = calibration.read_whiteness(path, phase_mapping=mapping(), return_details=True)
                self.assertEqual(result["summary"], {200: .1})
                self.assertEqual(result["ratios_log2iso"]["C01"][0][1], ratios[0])
                self.assertEqual(result["ratios_log2iso"]["C10"][0][1], ratios[1])

    def test_missing_or_partial_mapping_cannot_guess_green_phases(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "spectrum.csv"
            path.write_text("freq,iso200_C01_diff,iso200_C10_diff\n0.1,1,1\n0.4,.1,.1\n")
            self.assertEqual(calibration.read_whiteness(path), {})
            partial = dark_phase_mapping({"CfaPattern": "RGBG"},
                                         [{"Channel": "C01", "ColorIndex": "1"}])
            self.assertEqual(partial["status"], "partial")
            self.assertEqual(calibration.read_whiteness(path, phase_mapping=partial), {})

    def test_mapping_changes_and_colour_description_conflicts_are_rejected(self):
        rows = [{"Channel": "C00", "ColorIndex": "0"},
                {"Channel": "C00", "ColorIndex": "1"}]
        with self.assertRaisesRegex(ValueError, "changes"):
            dark_phase_mapping({"CfaPattern": "RGBG"}, rows)
        with self.assertRaisesRegex(ValueError, "outside"):
            dark_phase_mapping({"CfaPattern": "RGB"}, [{"Channel": "C11", "ColorIndex": "3"}])

    def test_colour_description_indices_need_not_use_rgbg_order(self):
        rows = [{"Channel": phase, "ColorIndex": str(cid)}
                for phase, cid in zip(PHASES, (2, 0, 3, 1))]
        result = dark_phase_mapping({"CfaPattern": "GBRG"}, rows)
        self.assertEqual(result["status"], "bayer")
        self.assertEqual([result["phases"][p]["color"] for p in PHASES], list("RGGB"))

    def test_single_colour_and_non_bayer_mapping_are_explicit(self):
        rows = [{"Channel": phase, "ColorIndex": "0"} for phase in PHASES]
        result = dark_phase_mapping({"CfaPattern": "G"}, rows)
        self.assertEqual(result["status"], "single-colour")
        rows[-1]["ColorIndex"] = "1"
        result = dark_phase_mapping({"CfaPattern": "GR"}, rows)
        self.assertEqual(result["status"], "unsupported")

    def test_serialized_mapping_cannot_claim_bayer_with_missing_phase(self):
        value = evidence((1., .1, 1.9, 1.))
        value["mapping"]["phases"].pop("C00")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_spectrum(value)

    def test_collect_build_retains_all_phase_evidence_and_source_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "dark-scalars.csv").write_text(
                "#Format: JPTC-DARK/1\n#Camera: Review Sensor\n#CfaPattern: RGBG\n"
                "#ClipVarianceFactor: 1\n#AdcStep: 0\n"
                "ISO,Channel,ColorIndex,BlackA,BlackB,StdDiffClipped\n"
                + "".join(f"200,{phase},{cid},100,100,2\n"
                          for phase, cid in zip(PHASES, PATTERNS["GBRG"])))
            (root / "spectrum-h.csv").write_text(
                "#Format: JPTC-SPECTRUM/1\n#CfaPattern: RGBG\n"
                "freq,iso200_C00_diff,iso200_C01_diff,iso200_C10_diff,iso200_C11_diff\n"
                "0.1,1,1,1,1\n0.4,0.1,1,1,1.9\n")
            built = calibration.build(root, None)
            spectrum = built["noise_spectrum"]
            self.assertEqual(spectrum["mapping"]["phases"]["C00"]["color"], "G")
            self.assertEqual(spectrum["axes"]["h"]["ratios_log2iso"]["C11"][0][1], 1.9)
            self.assertEqual(built["noise_whiteness_h_log2iso"][0][1], .1)
            self.assertEqual(spectrum["axes"]["h"]["source_sha256"],
                             built["source"]["inputs"]["spectrum-h.csv"])

    def test_cross_file_colour_descriptions_cannot_borrow_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "spectrum.csv"
            path.write_text("#CfaPattern: BGRG\nfreq,iso200_C01_diff\n0.1,1\n0.4,.1\n")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                calibration.read_whiteness(path, phase_mapping=mapping())


class SpectralPhaseProductionTests(unittest.TestCase):
    run_pipeline = fallback_tests.SpectralFallbackPipelineTests.run_pipeline

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        env = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "store"))
        env.start()
        self.addCleanup(env.stop)

    def test_per_green_anomaly_survives_real_import_decode_analysis_and_gates(self):
        for fallback in (False, True):
            profile = unresolved_collect_profile() if fallback else collect_profile()
            profile["noise_spectrum"] = evidence((1., .1, 1.9, 1.))
            with self.subTest(fallback=fallback):
                bundle, result, correction = self.run_pipeline(profile)
                self.assertEqual(result.noise_model.correlation, "measured-spectral-imbalance")
                self.assertEqual(result.noise_model.spectral_ratios["h:C01"], .1)
                self.assertEqual(result.noise_model.spectral_ratios["h:C10"], 1.9)
                self.assertEqual(compile_tail_snr_gate(result), 0.)
                self.assertIsNone(correction)
                self.assertEqual(bundle.chroma_nr_status, "skipped")

    def test_applicable_red_or_blue_anomaly_also_constrains_rgb_noise_approximation(self):
        profile = collect_profile()
        profile["noise_spectrum"] = evidence((.1, 1., 1., 1.))
        bundle, result, correction = self.run_pipeline(profile)
        self.assertEqual(result.noise_model.spectral_ratios["h:C00"], .1)
        self.assertEqual(compile_tail_snr_gate(result), 0.)
        self.assertIsNone(correction)


if __name__ == "__main__":
    unittest.main()
