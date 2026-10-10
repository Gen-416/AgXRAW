# SPDX-License-Identifier: GPL-3.0-or-later
"""Stored RAW variance and physical electron read noise are different evidence."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import calibration, priors
from dngscan.analysis import analyze
from dngscan.hdr_agx_plan import compile_tail_snr_gate
from dngscan.noise_model import model_from_prior, resolve_noise_model
from tests.test_noise_model import frame
from tests.test_user_calibration import single_profile, collect_profile, unresolved_collect_profile


VARIANCE_KEY = "stored_dark_variance_dn2_log2iso"
CONTRACT = {"stored_dark_variance_measurement": "paired-frame-pre-sheppard",
            "stored_dark_variance_domain": "linearized-raw-dn",
            "sigma_clip_correction": "applied"}


class StoredDarkVarianceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        env = patch.dict(os.environ, {"DNGSCAN_CALIBRATION_DIR": str(self.root / "store")})
        env.start()
        self.addCleanup(env.stop)

    def install(self, item):
        path = self.root / "measurement.json"
        path.write_text(json.dumps(item), encoding="utf-8")
        return calibration.import_calibration(path)

    def bundle(self, *, iso=100, span_scale=1.):
        bundle = frame()
        return replace(bundle, shot_iso=iso, shot_shutter="electronic",
                       capture_readout={"shutter": "electronic"},
                       white_level=int(16383 * span_scale),
                       black_levels=[1024. * span_scale] * 4,
                       camera_white_levels=[16383. * span_scale] * 4)

    def measured_profile(self):
        item = collect_profile()
        item["acquisition_contract"] = dict(CONTRACT)
        item[VARIANCE_KEY] = [[math.log2(i), v] for i, v in ((100, 1.), (400, 3.))]
        return item

    def csv_directory(self, *, step=1., analog_dn=.5):
        directory = self.root / "collect"
        directory.mkdir()
        black, white, gain = 1024., 16383., 4.
        total = analog_dn * analog_dn + step * step / 12.
        (directory / "dark-scalars.csv").write_text(
            "#Format: JPTC-DARK/1\n#Camera: SIGMA fp\n#ShutterType: 电子快门\n"
            f"#AdcStep: {step}\n#LinearisationCurve: identity\n#ClipVarianceFactor: 1\n"
            "ISO,ColorIndex,BlackA,BlackB,StdDiffClipped\n" +
            "".join(f"100,{ch},{black},{black},{math.sqrt(2 * total):.17g}\n" for ch in (1, 3)))
        rows = []
        for signal in np.geomspace(2, (white-black)*1.15, 80):
            mean = min(black+signal, white)
            std = math.sqrt(signal/gain + total) if mean < white else 0.
            rows.append(f"{mean},{std}")
        (directory / "ptc-iso100.csv").write_text(
            "#Format: JPTC/2\n#BlackLevel: 1024,1024,1024,1024\nG1_Mean,G1_Std\n" + "\n".join(rows))
        return directory, total

    def test_read_dark_preserves_pre_sheppard_variance(self):
        directory, total = self.csv_directory()
        _, dark = calibration.read_dark(directory / "dark-scalars.csv")
        self.assertAlmostEqual(dark[100]["total_var"], total)
        self.assertAlmostEqual(dark[100]["rn_dn"], .5)
        converted = calibration.build(directory, None)
        self.assertEqual(converted["acquisition_contract"]["stored_dark_variance_domain"],
                         "linearized-raw-dn")
        self.assertAlmostEqual(converted[VARIANCE_KEY][0][1], total)

    def test_real_collect_import_reaches_analyze_with_stored_variance_and_physical_rn(self):
        directory, total = self.csv_directory()
        summary = calibration.import_calibration(directory)
        self.assertTrue(summary["has_stored_dark_variance"])
        prior = priors.find_priors("SIGMA", "fp", shutter="electronic", iso=100)
        self.assertEqual(prior["calibration_id"], summary["id"])
        self.assertAlmostEqual(prior[VARIANCE_KEY][0][1], total)
        result, _, _ = analyze(self.bundle(), 4)
        gain = priors.gain_e_per_dn(prior, 100)
        self.assertAlmostEqual(result.prior_read_noise_e, .5 * gain)
        a, b = result.noise_model.coefficients("G1")
        self.assertAlmostEqual(a, 1. / (gain * 15359.), places=15)
        self.assertAlmostEqual(b, total / 15359.**2, places=17)
        self.assertAlmostEqual(result.noise_floor, math.sqrt(total) / 15359., places=15)
        self.assertIn("measured-stored-dark-variance", result.noise_model.approximation)
        diagnostic = calibration.calibration_diagnostics("SIGMA", "fp", "electronic", 100)[0]
        self.assertEqual(diagnostic["stored_dark_variance_status"], "measured-stored-dark-variance")
        self.assertAlmostEqual(diagnostic["stored_dark_variance_dn2"], total)

    def test_stored_variance_transports_only_through_matched_dn_scale(self):
        self.install(self.measured_profile())
        prior = priors.find_priors("SIGMA", "fp", shutter="electronic", iso=100)
        for factor in (1., 4.):
            with self.subTest(scale=factor):
                model = model_from_prior(self.bundle(span_scale=factor), {i: 16383. for i in range(4)}, prior)
                self.assertEqual(model.status, "valid")
                self.assertAlmostEqual(model.coefficients("G1")[1], 1. / 15359.**2, places=17)
        bad = self.bundle(span_scale=1.3)
        self.assertEqual(model_from_prior(bad, {i: 16383. for i in range(4)}, prior).reason,
                         "unmatched-dn-scale")

    def test_variance_curve_does_not_extrapolate_or_cross_gain_jumps(self):
        item = self.measured_profile()
        item[VARIANCE_KEY].append([math.log2(800), 18.])
        prior = calibration._validated_prior(item)
        value, status = calibration.stored_dark_variance(prior, 200)
        self.assertAlmostEqual(value, 2.)
        self.assertEqual(status, "interpolated-stored-dark-variance")
        for iso in (50, 600, 1600):
            with self.subTest(iso=iso):
                self.assertIsNone(calibration.stored_dark_variance(prior, iso)[0])

    def test_measurement_is_retained_without_promoting_unresolved_physical_rn(self):
        item = unresolved_collect_profile()
        item["acquisition_contract"] = dict(CONTRACT)
        item[VARIANCE_KEY] = [[math.log2(200), 7.]]
        self.install(item)
        prior = priors.find_priors("SIGMA", "fp", shutter="electronic", iso=200)
        self.assertEqual(calibration.stored_dark_variance(prior, 200)[0], 7.)
        result, _, _ = analyze(self.bundle(iso=200), 4)
        self.assertEqual(result.noise_model.status, "unresolved")
        self.assertIsNone(result.prior_read_noise_e)
        self.assertTrue(math.isnan(result.noise_floor))
        self.assertEqual(compile_tail_snr_gate(result), 0.)

    def test_positive_stored_variance_survives_a_failed_sheppard_fit(self):
        directory, total = self.csv_directory(analog_dn=0.)
        summary = calibration.import_calibration(directory)
        self.assertTrue(summary["has_stored_dark_variance"])
        prior = priors.find_priors("SIGMA", "fp", shutter="electronic", iso=100)
        self.assertIn(100, prior["read_noise_unresolved_isos"])
        self.assertAlmostEqual(calibration.stored_dark_variance(prior, 100)[0], total)
        result, _, _ = analyze(self.bundle(), 4)
        self.assertIsNotNone(result.gain_e_per_dn)
        self.assertIsNone(result.prior_read_noise_e)
        self.assertEqual(result.noise_model.status, "unresolved")

    def test_unresolved_dng_fallback_retains_independent_negative_spectrum(self):
        item = unresolved_collect_profile()
        item["acquisition_contract"] = dict(CONTRACT)
        item[VARIANCE_KEY] = [[math.log2(200), 7.]]
        item["noise_whiteness_h_log2iso"] = [[math.log2(200), .1]]
        self.install(item)
        with patch("dngscan.spatial_black.sensor_tags", return_value={262: [32803], 51041: [1e-4, 1e-8]}):
            result, _, _ = analyze(self.bundle(iso=200), 4)
        model = result.noise_model
        self.assertEqual(model.source, "DNG NoiseProfile")
        self.assertEqual(model.coefficients("G1"), (1e-4, 1e-8))
        self.assertEqual(model.fallback_reason, "read-noise-unresolved")
        self.assertEqual(model.correlation, "measured-spectral-imbalance")
        self.assertEqual(model.spectral_ratios, {"h": .1})
        self.assertEqual(compile_tail_snr_gate(result), 0.)

    def test_explicit_curve_validates_units_contract_and_variance(self):
        for change in ({"acquisition_contract": {}},
                       {"acquisition_contract": {**CONTRACT, "stored_dark_variance_domain": "encoded-log"}},
                       {VARIANCE_KEY: [[math.log2(100), float("nan")]]},
                       {VARIANCE_KEY: [[math.log2(100), -.1]]},
                       {VARIANCE_KEY: [[math.log2(100), 0.]]},
                       {VARIANCE_KEY: [[math.log2(100), .1]]},
                       {VARIANCE_KEY: [[math.log2(100), .1]], "read_noise_dn_log2iso": [[math.log2(100), .75]]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                calibration._validated_prior({**self.measured_profile(), **change})
        item = single_profile()
        item.update({VARIANCE_KEY: [[math.log2(100), 1.]], "acquisition_contract": dict(CONTRACT)})
        with self.assertRaisesRegex(ValueError, "Collect"):
            calibration._validated_prior(item)

    def test_interpolated_total_below_selected_physical_read_variance_is_unresolved(self):
        item = self.measured_profile()
        item["gain_log2iso_log2epd"] = [[math.log2(i), math.log2(g)]
                                        for i, g in ((100, 4), (200, 2), (400, 1))]
        item["gain_jump_isos"] = []
        item["read_noise_log2iso_log2e"] = [[math.log2(i), math.log2(r)]
                                           for i, r in ((100, 3), (200, 8), (400, 1))]
        item[VARIANCE_KEY] = [[math.log2(i), 1.] for i in (100, 400)]
        reason = "stored-dark-variance-below-physical-read-variance"
        for dn_curve in ([], [[math.log2(i), r] for i, r in ((100, .75), (200, 4), (400, 1))]):
            with self.subTest(has_dn_curve=bool(dn_curve)):
                item["read_noise_dn_log2iso"] = dn_curve
                prior = calibration._validated_prior(item)
                self.assertEqual(calibration.stored_dark_variance(prior, 200), (None, reason))
                model = model_from_prior(self.bundle(iso=200), {i: 16383. for i in range(4)}, prior)
                self.assertEqual(model.status, "unresolved")
                self.assertEqual(model.reason, reason)
                self.assertEqual(model.channel_variance, {})
        # A separate file profile can replace coefficients without erasing
        # measured negative spectral evidence from the applicable Collect.
        item["noise_whiteness_h_log2iso"] = [[math.log2(200), .1]]
        self.install(item)
        with patch("dngscan.spatial_black.sensor_tags", return_value={262: [32803], 51041: [1e-4, 1e-8]}):
            result, _, _ = analyze(self.bundle(iso=200), 4)
        model = result.noise_model
        self.assertEqual(model.source, "DNG NoiseProfile")
        self.assertEqual(model.coefficients("G1"), (1e-4, 1e-8))
        self.assertEqual(model.fallback_reason, reason)
        self.assertEqual(model.correlation, "measured-spectral-imbalance")
        self.assertEqual(model.spectral_ratios, {"h": .1})
        self.assertEqual(compile_tail_snr_gate(result), 0.)
        diagnostic = calibration.calibration_diagnostics("SIGMA", "fp", "electronic", 200)[0]
        self.assertEqual(diagnostic["stored_dark_variance_status"], reason)

    def test_stored_variance_coefficients_are_finite_without_intermediate_electron_overflow(self):
        item = self.measured_profile()
        item[VARIANCE_KEY] = [[math.log2(100), 1e308]]
        prior = calibration._validated_prior(item)
        model = model_from_prior(self.bundle(), {i: 16383. for i in range(4)}, prior)
        self.assertEqual(model.status, "valid")
        self.assertTrue(math.isfinite(model.coefficients("G1")[1]))
        self.assertAlmostEqual(model.coefficients("G1")[1] / 1e308, 1. / 15359.**2)
        # All declared input numbers are finite and DN matching succeeds,
        # but this deliberately extreme normalized variance is not finite.
        item["fwc_e"] = 4e-200
        prior = calibration._validated_prior(item)
        tiny = replace(self.bundle(), white_level=1e-200, black_levels=[0.] * 4,
                       camera_white_levels=[1e-200] * 4)
        model = model_from_prior(tiny, {i: 1e-200 for i in range(4)}, prior)
        self.assertEqual(model.status, "unresolved")
        self.assertEqual(model.reason, "nonfinite-noise-model-coefficients")
        self.assertEqual(model.channel_variance, {})

    def test_legacy_collect_restoration_requires_an_unambiguous_identity_contract(self):
        item = collect_profile()
        item["noise_aperture"] = calibration._LEGACY_SHEPPARD_APERTURE
        item["read_noise_dn_log2iso"] = [[math.log2(100), .75]]
        item["acquisition_contract"] = {"adc_step": "1", "linearisation_curve": "identity",
                                        "sigma_clip_correction": "applied"}
        prior = calibration._validated_prior(item)
        variance, status = calibration.stored_dark_variance(prior, 100)
        self.assertEqual(variance, .75**2 + 1./12.)
        self.assertEqual(status, "legacy-sheppard-restored-scalar-variance-approximation")
        model = model_from_prior(self.bundle(), {i: 16383. for i in range(4)}, prior)
        self.assertAlmostEqual(model.coefficients("G1")[1], variance / 15359.**2, places=17)
        for change in ({"adc_step": "bad"}, {"adc_step": -1}, {"adc_step": float("inf")},
                       {"adc_step": 1e308},
                       {"linearisation_curve": "companded"}, {"sigma_clip_correction": "unresolved"}):
            with self.subTest(change=change):
                other = calibration._validated_prior({**item, "acquisition_contract": {**item["acquisition_contract"], **change}})
                self.assertIsNone(calibration.stored_dark_variance(other, 100)[0])
                fallback = model_from_prior(self.bundle(), {i: 16383. for i in range(4)}, other)
                self.assertAlmostEqual(fallback.coefficients("G1")[1], (.75 / 15359.)**2, places=17)
                self.assertIn("unverified", fallback.approximation)
        for removed in ("noise_aperture", "acquisition_contract", "read_noise_dn_log2iso"):
            other = dict(item)
            other.pop(removed)
            self.assertIsNone(calibration.stored_dark_variance(calibration._validated_prior(other), 100)[0])

    def test_packaged_collect_loader_preserves_measurement_and_contract(self):
        directory, total = self.csv_directory()
        item = calibration.build(directory, None)
        real_read = Path.read_text
        def read(path, *args, **kwargs):
            if path.parent.name == "jptc_collect":
                return json.dumps(item)
            return real_read(path, *args, **kwargs)
        with patch.object(priors, "_JPTC_CACHE", None), patch.object(Path, "read_text", read):
            packaged = next(e for e in priors._jptc_entries() if "collect" in e.get("source", ""))
        self.assertFalse(packaged.get("user_calibration", False))
        self.assertEqual(packaged["source_format"], "dngscan-jptc-collect-1")
        self.assertAlmostEqual(calibration.stored_dark_variance(packaged, 100)[0], total)
        model = model_from_prior(self.bundle(), {i: 16383. for i in range(4)}, packaged)
        self.assertAlmostEqual(model.coefficients("G1")[1], total / 15359.**2, places=17)

    def test_ptc_p2p_and_file_coefficients_receive_no_blanket_quantization_term(self):
        # A single-frame PTC intercept was never Sheppard-subtracted.
        ptc = calibration._validated_prior(single_profile())
        model = model_from_prior(self.bundle(), {i: 16383. for i in range(4)}, ptc)
        self.assertEqual(model.coefficients("G1")[1], (3. / (4. * 15359.))**2)
        # Published fp/P2P data lack Collect's subtraction contract.
        with patch("dngscan.calibration.matching_prior", return_value=None):
            for make, name in (("SIGMA", "fp"), ("Canon", "EOS 5D Mark II")):
                prior = priors.find_priors(make, name, iso=100)
                bundle = self.bundle()
                if make == "Canon":
                    bundle = replace(bundle, white_level=16383, black_levels=[512.] * 4)
                model = model_from_prior(bundle, {i: 16383. for i in range(4)}, prior)
                gain = priors.gain_for_file(prior, 100, bundle.white_level-bundle.black_levels[0])
                rn = priors.read_noise_e(prior, 100)
                self.assertEqual(model.coefficients("G1")[1], (rn / (gain * (bundle.white_level-bundle.black_levels[0])))**2)
        unknown = replace(self.bundle(), shot_make="unmeasured", shot_model="unmeasured")
        with patch("dngscan.spatial_black.sensor_tags", return_value={262: [32803], 51041: [1e-4, 1e-8]}):
            model = resolve_noise_model(unknown, {i: 16383. for i in range(4)})
        self.assertEqual(model.coefficients("G1"), (1e-4, 1e-8))


if __name__ == "__main__":
    unittest.main()
