# SPDX-License-Identifier: GPL-3.0-or-later
"""Measured user profiles must actually resolve, without blind extrapolation."""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dngscan import calibration, priors


def single_profile():
    return {"format": "dngscan-jptc-prior-1", "id": "Owner fp ISO 100",
            "brand": "SIGMA", "model": "SIGMA fp", "iso": 100, "shutter": "electronic",
            "gain_e_per_dn": 4.0, "read_noise_e": 3.0, "fwc_e": 15359.0*4,
            "white_level_used": 16383, "black_level_g1": 1024,
            "fit_relative_rms": .01, "gain_estimator_spread_rel": .01, "quality": "ok",
            "channel": "G1", "noise_aperture": "single-frame spatial PTC"}


def collect_profile():
    return {"format": "dngscan-jptc-collect-1", "id": "Owner fp ladder",
            "make": "SIGMA", "model_candidates": ["fp", "SIGMA fp"], "shutter": "electronic",
            "ptc_anchor": {"iso": 100, "gain_e_per_dn": 4, "quality": "ok"},
            "fwc_e": 15359*4, "gain_log2iso_log2epd": [[math.log2(i), math.log2(g)]
              for i,g in [(100,4),(200,2),(400,1),(800,.25)]],
            "read_noise_log2iso_log2e": [[math.log2(i), math.log2(r)]
              for i,r in [(100,3),(200,2),(400,1.5),(800,1)]], "gain_jump_isos": [800]}


class TestUserCalibration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict("os.environ", {"DNGSCAN_CALIBRATION_DIR": str(self.root / "user")})
        self.env.start()
        self.addCleanup(self.env.stop)

    def write_profile(self, item=None):
        path = self.root / "input.json"
        path.write_text(json.dumps(single_profile() if item is None else item), encoding="utf-8")
        return path

    def test_import_supersedes_curated_only_when_mode_and_iso_match(self):
        summary = calibration.import_calibration(self.write_profile())
        e = priors.find_priors("SIGMA", "SIGMA FP", shutter="electronic", iso=100)
        self.assertEqual(e["calibration_id"], summary["id"])
        self.assertTrue(priors.prior_usability(e)[0])
        self.assertEqual(priors.gain_for_file(e, 100, 15359), 4)
        self.assertEqual(priors.read_noise_e(e, 100), 3)
        self.assertEqual(e["noise_model_channels"], "scalar-green")
        for query in ({"shutter": None, "iso": 100}, {"shutter": "mechanical", "iso": 100},
                      {"shutter": "electronic", "iso": 200}):
            self.assertEqual(priors.find_priors("SIGMA", "FP", **query)["id"], "Sigma fp")
        diagnostics = calibration.calibration_diagnostics("SIGMA", "FP", "mechanical", 100)
        self.assertEqual(diagnostics[0]["reason"], "readout-mode-mismatch")

    def test_single_iso_never_extrapolates_gain_or_read_noise(self):
        calibration.import_calibration(self.write_profile())
        e = priors.find_priors("SIGMA", "fp", shutter="electronic", iso=100)
        for iso in (50, 200, 6400):
            self.assertIsNone(priors.gain_e_per_dn(e, iso))
            self.assertIsNone(priors.read_noise_e(e, iso))
        self.assertIsNone(priors.gain_for_file(e, 200, 15359))

    def test_exact_model_and_make(self):
        calibration.import_calibration(self.write_profile())
        self.assertIsNone(calibration.matching_prior("SIGMA", "fp L", shutter="electronic", iso=100))
        self.assertIsNone(calibration.matching_prior("OTHER", "fp", shutter="electronic", iso=100))
        self.assertIsNotNone(calibration.matching_prior("sigma corporation", " sigma   fp ", shutter="electronic", iso=100))

    def test_explicit_mode_override_preserves_source(self):
        original = single_profile()
        path = self.write_profile(original)
        strict = calibration.import_calibration(path)
        broad = calibration.import_calibration(path, shutter_override="any")
        self.assertNotEqual(strict["id"], broad["id"])
        self.assertEqual(broad["source_shutter"], "electronic")
        e = priors.find_priors("SIGMA", "FP", iso=100)
        self.assertEqual(e["calibration_id"], broad["id"])
        stored = json.loads(Path(broad["path"]).read_text())
        self.assertEqual(stored["payload"]["shutter"], original["shutter"])
        self.assertEqual(stored["payload"]["user_applicability"]["shutter_override"], "any")
        self.assertEqual(json.loads(path.read_text()), original)

    def test_undefined_mode_is_visible_and_not_silently_applied(self):
        item = single_profile()
        item["shutter"] = None
        result = calibration.import_calibration(self.write_profile(item))
        self.assertIn("readout-mode-unavailable", result["warnings"])
        self.assertIsNone(calibration.matching_prior("SIGMA", "FP", shutter="electronic", iso=100))
        self.assertEqual(calibration.calibration_diagnostics("SIGMA", "FP", "electronic", 100)[0]["reason"], "readout-mode-unavailable")

    def test_ladder_interpolation_stays_inside_segments(self):
        calibration.import_calibration(self.write_profile(collect_profile()))
        e = priors.find_priors("SIGMA", "FP", shutter="electronic", iso=300)
        self.assertAlmostEqual(priors.gain_e_per_dn(e, 300), 4/3)
        self.assertIsNone(priors.gain_e_per_dn(e, 600))
        self.assertIsNone(priors.read_noise_e(e, 600))
        self.assertEqual(priors.gain_e_per_dn(e, 800), .25)
        self.assertIsNone(calibration.matching_prior("SIGMA", "FP", shutter="electronic", iso=900))

    def test_activation_removal_and_content_cache_identity(self):
        baseline = calibration.calibration_fingerprint()
        s = calibration.import_calibration(self.write_profile())
        after = calibration.calibration_fingerprint()
        self.assertNotEqual(after, baseline)
        self.assertEqual(len(calibration.list_calibrations()), 1)
        self.assertEqual(calibration.import_calibration(self.write_profile())["id"], s["id"])
        calibration.set_calibration_active(s["id"], False)
        self.assertNotEqual(calibration.calibration_fingerprint(), after)
        self.assertEqual(priors.find_priors("SIGMA", "FP", shutter="electronic", iso=100)["id"], "Sigma fp")
        calibration.set_calibration_active(s["id"], True)
        self.assertTrue(priors.find_priors("SIGMA", "FP", shutter="electronic", iso=100)["user_calibration"])
        self.assertTrue(calibration.remove_calibration(s["id"])["removed"])
        self.assertEqual(calibration.calibration_fingerprint(), baseline)

    def test_bad_numeric_and_format_inputs_do_not_install(self):
        for field, value in [("gain_e_per_dn", -1), ("read_noise_e", -1), ("iso", 0),
                             ("gain_estimator_spread_rel", float("nan")), ("format", "unknown"),
                             ("fwc_e", 1), ("brand", ""), ("model", "")]:
            item = single_profile()
            item[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                calibration.import_calibration(self.write_profile(item))
        self.assertEqual(calibration.list_calibrations(), [])
        for identity in ("../input", "a/b", "not-a-digest"):
            with self.assertRaises(ValueError):
                calibration.remove_calibration(identity)

    def test_corrupt_record_is_diagnosed_not_consumed(self):
        s = calibration.import_calibration(self.write_profile())
        path = Path(s["path"])
        record = json.loads(path.read_text())
        record["payload"]["gain_e_per_dn"] = 900
        path.write_text(json.dumps(record))
        listed = calibration.list_calibrations()
        self.assertIn("checksum", listed[0]["error"])
        self.assertEqual(priors.find_priors("SIGMA", "FP", shutter="electronic", iso=100)["id"], "Sigma fp")

    def test_unreadable_store_falls_back_and_reports_error(self):
        with patch.object(Path, "glob", side_effect=PermissionError("store unavailable")):
            self.assertIn("unavailable", calibration.list_calibrations()[0]["error"])
            self.assertIsInstance(calibration.calibration_fingerprint(), str)
            self.assertEqual(priors.find_priors("SIGMA", "FP", shutter="electronic", iso=100)["id"], "Sigma fp")

    def test_failed_atomic_replace_preserves_previous_record(self):
        s = calibration.import_calibration(self.write_profile())
        path = Path(s["path"])
        before = path.read_bytes()
        with patch.object(calibration.os, "replace", side_effect=OSError("simulated failure")):
            with self.assertRaises(OSError):
                calibration.set_calibration_active(s["id"], False)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_quality_failure_retained_for_diagnosis(self):
        item = single_profile()
        item["quality"] = "high-residual"
        s = calibration.import_calibration(self.write_profile(item))
        self.assertIn("quality-high-residual", s["warnings"])
        self.assertIsNone(calibration.matching_prior("SIGMA", "FP", shutter="electronic", iso=100))

    def test_collect_csv_synthetic_sensor_uses_runtime_converter(self):
        directory = self.root / "collect"
        directory.mkdir()
        black, white, gain, rn = 1024, 16383, 4, 3
        (directory / "dark-scalars.csv").write_text(
            "#Format: JPTC-DARK/1\n#Camera: SIGMA fp\n#ShutterType: 电子快门\n"
            "#AdcStep: 0\n#ClipVarianceFactor: 1\n"
            "ISO,ColorIndex,BlackA,BlackB,StdDiffClipped\n" +
            "".join(f"{iso},{ch},{black},{black},{math.sqrt(2)*rn/gain}\n" for iso in (100,200) for ch in (1,3)))
        values = np.geomspace(2, (white-black)*1.15, 80)
        rows = []
        for signal in values:
            mean = min(black+signal, white)
            std = math.sqrt(signal/gain + (rn/gain)**2) if mean < white else 0
            rows.append(f"{mean},{std}")
        (directory / "ptc-iso100.csv").write_text(
            "#Format: JPTC/2\n#BlackLevel: 1024,1024,1024,1024\nG1_Mean,G1_Std\n" + "\n".join(rows))
        (directory / "gain-levels.csv").write_text(
            "#Format: JPTC-ISOGAIN/1\nISO,ColorIndex,ClipFrac,Mean,ShutterSec\n"
            "100,1,0,2024,1\n200,1,0,3024,1\n")
        s = calibration.import_calibration(directory)
        copied = self.root / "different-upload-directory"
        shutil.copytree(directory, copied)
        self.assertEqual(calibration.import_calibration(copied)["id"], s["id"])
        e = priors.find_priors("SIGMA", "fp", shutter="electronic", iso=100)
        self.assertEqual(e["calibration_id"], s["id"])
        self.assertAlmostEqual(priors.gain_e_per_dn(e,100), gain, delta=.02)
        self.assertAlmostEqual(priors.read_noise_e(e,100), rn, delta=.02)
        self.assertAlmostEqual(priors.gain_e_per_dn(e,200), 2, delta=.02)
        self.assertNotIn("url_base", json.loads(Path(s["path"]).read_text())["payload"]["source"])

    def test_collect_boundary_validation_and_partial_profile_diagnostics(self):
        for jumps in ([100], [900], [float("inf")], [-1]):
            item = collect_profile()
            item["gain_jump_isos"] = jumps
            with self.subTest(jumps=jumps), self.assertRaises(ValueError):
                calibration.import_calibration(self.write_profile(item))
        for key, value in (("fwc_e", float("inf")), ("fwc_e", -1)):
            item = collect_profile()
            item[key] = value
            with self.assertRaises(ValueError):
                calibration.import_calibration(self.write_profile(item))
        item = collect_profile()
        item["geometry"], item["compression"] = [6000,4000], "lossless"
        summary = calibration.import_calibration(self.write_profile(item))
        self.assertIn("sub-readout-mode-not-verified", summary["warnings"])
        diagnostic = calibration.calibration_diagnostics("SIGMA", "FP", "electronic",100)[0]
        self.assertEqual(diagnostic["mode_scope"], "shutter-and-dn-scale")
        self.assertIn("compression", diagnostic["unverified_readout_fields"])
        dark_only = {"format": "dngscan-jptc-collect-1", "id": "dark only",
                     "make": "SIGMA", "model_candidates": ["fp"], "shutter": "electronic",
                     "read_noise_dn_log2iso": [[math.log2(100), .8]]}
        summary = calibration.import_calibration(self.write_profile(dark_only))
        self.assertFalse(summary["has_gain"])
        self.assertIn("no-absolute-gain", summary["warnings"])
        rows = calibration.calibration_diagnostics("SIGMA", "FP", "electronic",100)
        self.assertTrue(any(r["reason"] == "no-absolute-gain" for r in rows))

    def test_collect_dn_electron_noise_must_agree_at_measured_intersections(self):
        item = collect_profile()
        item["read_noise_dn_log2iso"] = [[math.log2(100), 300.]]
        with self.assertRaisesRegex(ValueError, "DN/electron units disagree"):
            calibration.import_calibration(self.write_profile(item))
        item["read_noise_dn_log2iso"] = [[math.log2(100), .75*1.04]]
        calibration.import_calibration(self.write_profile(item))
        # Neither an unmeasured intermediate ISO nor a point outside the
        # gain/read-noise domain is validated by invented extrapolation.
        item["read_noise_dn_log2iso"] = [[math.log2(50), 300.],
                                         [math.log2(600), 300.]]
        calibration.import_calibration(self.write_profile(item))
        # Packaged measurements are the actual importer numerical contract.
        directory = Path(__file__).parents[1] / "dngscan/data/priors/jptc_collect"
        for path in sorted(directory.glob("*.json")):
            with self.subTest(profile=path.name):
                calibration.import_calibration(path, active=False)

    def test_pchip_matches_historical_scipy_interpolation(self):
        from scipy.interpolate import PchipInterpolator
        xs = np.array([1.,2.,3.,5.,8.])
        for ys in (np.array([2.,1.,3.,4.,2.]), np.array([0.,0.,0.,1.,2.])):
            reference = PchipInterpolator(xs,ys)
            for x in np.linspace(xs[0],xs[-1],101):
                self.assertAlmostEqual(calibration._pchip(xs,ys,x), float(reference(x)), places=12)


if __name__ == "__main__":
    unittest.main()
