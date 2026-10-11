# SPDX-License-Identifier: GPL-3.0-or-later
"""Product permissions follow arithmetic dependencies, not a CSV union."""
import copy
from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from dngscan import calibration, priors
from dngscan.analysis import analyze
from dngscan.calibration_capabilities import capability_prior, stages
from dngscan.collect_manifest import prepare_sources
from dngscan.hdr_agx_plan import compile_tail_snr_gate
from dngscan.noise_model import resolve_noise_model
from dngscan.raw_io import load_raw
from tests.test_readout_contract import edit_ifd
from tests.test_spectral_fallback_pipeline import write_noise_dng

FIXTURE = Path(__file__).parent / "fixtures/collect-writer/official.zip"


def without(content, field):
    return b"\n".join(line for line in content.splitlines()
                       if not line.startswith(("#" + field + ":").encode())) + b"\n"


class CapabilityQualificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "store"))
        environment.start(); self.addCleanup(environment.stop)
        with zipfile.ZipFile(FIXTURE) as archive:
            self.entries = [(name, archive.read(name)) for name in archive.namelist()]

    def package(self, name, mutate=lambda content: content):
        folder = self.root / name; folder.mkdir()
        for index, (_, content) in enumerate(self.entries):
            content = without(content, "Firmware").replace(b"6000x4000", b"128x128")
            (folder / f"source-{index}.csv").write_bytes(mutate(content))
        manifest, payloads = prepare_sources(folder)
        self.assertEqual(len(payloads), 1)
        group = manifest["groups"][0]
        return folder, group, payloads[group["id"]]

    def pipeline(self, folder, *, file_profile=False, iso=100):
        calibration.import_calibrations(folder)
        path = self.root / (folder.name + ".dng")
        write_noise_dng(path, file_profile=file_profile)
        edit_ifd(path, {271: (2, 6, b"NIKON\0"), 272: (2, 11, b"TestCamera\0"),
                       34855: (3, 1, struct.pack("<H", iso)),
                       50714: (5, 1, struct.pack("<LL", 100, 1)),
                       50717: (4, 1, struct.pack("<L", 4095))})
        original = hashlib.sha256(path.read_bytes()).hexdigest()
        bundle = load_raw(path, scene_half_size=True,
                          capture_declaration={"version": 1, "shutter": "mechanical"})
        selected = priors.find_priors("NIKON", "TestCamera", shutter="mechanical",
                                     iso=iso, readout=bundle.capture_readout)
        result, _, _ = analyze(bundle, 4, diagnostics=False)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original)
        return bundle, selected, result

    def test_missing_optional_psd_adc_keeps_complete_variance_pipeline(self):
        folder, group, payload = self.package("missing-psd-adc", lambda c:
            without(c, "AdcStep") if b"#Format: JPTC-SPECTRUM/1" in c else c)
        self.assertEqual(group["status"], "ready")
        q = payload["measurement_acquisition"]["capability_qualification"]
        for name in ("ptc_gain", "relative_gain", "absolute_gain", "read_noise_dn", "read_noise_e", "stored_dark_variance"):
            self.assertEqual(q[name]["status"], "qualified")
        for name in ("spectrum_h", "spectrum_v"):
            self.assertEqual(q[name]["status"], "unverified")
            self.assertIn("measurement-adc-step-incomplete", q[name]["reasons"])
        _, selected, result = self.pipeline(folder)
        self.assertEqual(result.noise_model.status, "valid")
        self.assertAlmostEqual(priors.gain_e_per_dn(selected, 100), 2.)
        self.assertAlmostEqual(result.noise_model.coefficients("G1")[1], 4. / 3995.**2)
        self.assertEqual(result.noise_model.correlation, "unknown")
        self.assertFalse(result.noise_model.spectral_ratios)
        state = stages(selected, iso=100)
        self.assertTrue(state["absolute_gain"]["applicable"])
        self.assertFalse(state["spectrum_h"]["applicable"])
        self.assertFalse(any(value["applied"] for value in state.values()))
        self.assertTrue(state["stored_dark_variance"]["valid"])
        self.assertFalse(stages(selected, iso=800)["stored_dark_variance"]["valid"])

    def test_bad_dark_only_disables_products_actually_consuming_dark(self):
        def mutate(content):
            if b"#Format: JPTC-DARK/1" in content:
                content = without(content, "AdcStep")
            if b"#Format: JPTC/2" in content:
                content += b"\n#BlackLevel: 100,100,100,100\n"
            return content
        folder, _, payload = self.package("bad-dark", mutate)
        q = payload["measurement_acquisition"]["capability_qualification"]
        self.assertEqual(q["ptc_gain"]["status"], "qualified")
        for name in ("absolute_gain", "relative_gain", "read_noise_dn", "read_noise_e", "stored_dark_variance", "spectrum_h", "spectrum_v"):
            self.assertEqual(q[name]["status"], "unverified")
        _, selected, result = self.pipeline(folder, file_profile=True)
        self.assertAlmostEqual(priors.gain_e_per_dn(selected, 100), 2.)
        self.assertIsNone(priors.gain_e_per_dn(selected, 150))
        self.assertIsNone(priors.read_noise_e(selected, 100))
        self.assertIsNone(calibration.stored_dark_variance(selected, 100)[0])
        self.assertEqual(result.noise_model.source, "DNG NoiseProfile")
        self.assertFalse(result.noise_model.spectral_ratios)

    def test_bad_relative_ladder_keeps_measured_anchor_and_read_noise(self):
        folder, _, payload = self.package("bad-ladder", lambda c:
            without(c, "ShutterType") if b"#Format: JPTC-ISOGAIN/1" in c else c)
        q = payload["measurement_acquisition"]["capability_qualification"]
        self.assertEqual(q["absolute_gain"]["status"], "unverified")
        self.assertEqual(q["relative_gain"]["status"], "unverified")
        self.assertEqual(q["ptc_gain"]["status"], "qualified")
        self.assertEqual(q["read_noise_e"]["status"], "qualified")
        _, selected, result = self.pipeline(folder)
        self.assertEqual(result.noise_model.status, "valid")
        self.assertAlmostEqual(priors.read_noise_e(selected, 100), (4. - 1./12.)**.5 * 2.)
        self.assertIsNone(priors.gain_e_per_dn(selected, 150))

    def test_axes_do_not_borrow_adc_or_optional_body_scope(self):
        folder, _, payload = self.package("one-axis", lambda c:
            without(c, "AdcStep") + b"\n#BodySerialNumber: synthetic-body\n"
            if b"#Axis: horizontal" in c else c)
        q = payload["measurement_acquisition"]["capability_qualification"]
        self.assertEqual(q["spectrum_h"]["status"], "unverified")
        self.assertEqual(q["spectrum_v"]["status"], "qualified")
        self.assertEqual(q["absolute_gain"]["status"], "qualified")
        _, selected, result = self.pipeline(folder)
        self.assertEqual(result.noise_model.status, "valid")
        self.assertTrue(result.noise_model.spectral_ratios)
        self.assertTrue(all(key.startswith("v:") for key in result.noise_model.spectral_ratios))
        self.assertNotIn("body_serial", capability_prior(selected, "absolute_gain")["readout_contract"])

    def test_explicit_profile_exposure_and_lens_restrictions_survive_views(self):
        _, _, payload = self.package("restrictions")
        payload["readout_contract"].update(exposure_seconds=[.001, .01], lens_model="Measured Lens")
        prior = calibration._validated_prior(payload)
        view = capability_prior(prior, "absolute_gain")
        self.assertEqual(view["readout_contract"]["exposure_seconds"], [.001, .01])
        self.assertEqual(view["readout_contract"]["lens_model"], "Measured Lens")

    def test_valid_independent_spectral_constraint_survives_dng_fallback(self):
        folder, _, payload = self.package("fallback")
        # Preserve real imported per-table permissions; change only a measured
        # diagnostic ratio and explicit read-noise resolution state in JSON.
        payload["read_noise_unresolved_isos"] = [100]
        payload["read_noise_log2iso_log2e"] = [p for p in payload["read_noise_log2iso_log2e"] if p[0] > 7.]
        payload["read_noise_dn_log2iso"] = [p for p in payload["read_noise_dn_log2iso"] if p[0] > 7.]
        for phase in payload["phase_calibration"].values():
            phase["read_noise_unresolved_isos"] = [100]
            phase["read_noise_dn_log2iso"] = [p for p in phase["read_noise_dn_log2iso"] if p[0] > 7.]
        for curve in payload["noise_spectrum"]["axes"]["h"]["ratios_log2iso"].values():
            curve[0][1] = .1
        calibration._install_payload(payload, active=True, shutter_override=None)
        # Prevent the unmodified CSV package from replacing the JSON probe.
        path = self.root / "probe.dng"; write_noise_dng(path, file_profile=True)
        edit_ifd(path, {271: (2, 6, b"NIKON\0"), 272: (2, 11, b"TestCamera\0"),
                       34855: (3, 1, struct.pack("<H", 100)),
                       50714: (5, 1, struct.pack("<LL", 100, 1)),
                       50717: (4, 1, struct.pack("<L", 4095))})
        bundle = load_raw(path, scene_half_size=True,
                          capture_declaration={"version": 1, "shutter": "mechanical"})
        result, _, _ = analyze(bundle, 4, diagnostics=False)
        self.assertEqual(result.noise_model.source, "DNG NoiseProfile")
        self.assertEqual(result.noise_model.correlation, "measured-spectral-imbalance")
        self.assertIn(.1, result.noise_model.spectral_ratios.values())
        self.assertEqual(compile_tail_snr_gate(result), 0.)

    def test_composed_spectral_sources_are_rebound_to_the_actual_new_capture(self):
        from dngscan.calibration_registry import compose_capabilities
        folder, _, _ = self.package("rebind")
        bundle, selected, _ = self.pipeline(folder, file_profile=True)
        base = copy.deepcopy(selected)
        base["read_noise_log2iso_log2e"] = []; base["read_noise_dn_log2iso"] = []
        base["stored_dark_variance_dn2_log2iso"] = []; base["phase_calibration"] = {}
        dark = copy.deepcopy(selected); dark["calibration_id"] = "independent-dark-probe"
        for curve in dark["noise_spectrum"]["axes"]["h"]["ratios_log2iso"].values():
            curve[0][1] = .1
        joined = compose_capabilities(base, [dark], 100)
        self.assertTrue(joined.get("independent_constraint_priors"))
        changed = replace(bundle, capture_readout={**bundle.capture_readout, "shutter": "electronic"})
        model = resolve_noise_model(changed, {i: 4095. for i in range(4)}, joined)
        self.assertEqual(model.source, "DNG NoiseProfile")
        self.assertFalse(model.spectral_ratios)
        self.assertEqual(model.correlation, "unknown")

    def test_scalar_unit_edits_cannot_be_hidden_by_capability_views(self):
        from dngscan.calibration_registry import compose_capabilities
        folder, _, _ = self.package("unit-edit")
        _, selected, _ = self.pipeline(folder)
        base = copy.deepcopy(selected)
        base["read_noise_log2iso_log2e"] = []; base["read_noise_dn_log2iso"] = []
        base["stored_dark_variance_dn2_log2iso"] = []; base["phase_calibration"] = {}
        dark = copy.deepcopy(selected); dark["calibration_id"] = "independent-dark-probe"
        dark["acquisition_contract"]["variance_domain"] = "electron-squared"
        self.assertIn("measurement-variance-domain-unsupported",
                      capability_prior(dark, "read_noise_dn")["readout_declaration_issues"])
        self.assertFalse("capability_sources" in compose_capabilities(base, [dark], 100))


class CapabilityStageTests(unittest.TestCase):
    """Diagnostic permission agrees with production accessors, not curve shape."""

    def entry(self):
        products = ("ptc_gain", "read_noise_dn", "read_noise_e", "spectrum_h")
        return {
            "measurement_acquisition": {"capability_qualification": {
                name: {"status": "qualified", "source_ids": ["synthetic"], "reasons": []}
                for name in products}},
            "capability_matches": {name: {"status": "matched", "reason": "matched"}
                                   for name in products},
            "ptc_gain_log2iso_log2epd": [[math.log2(100), 1.], [math.log2(200), 0.]],
            "read_noise_dn_log2iso": [[math.log2(100), 1.], [math.log2(400), 0.]],
            "read_noise_log2iso_log2e": [[math.log2(100), 2.], [math.log2(400), 0.]],
            "read_noise_unresolved_isos": [200],
            "noise_whiteness_h_log2iso": [[round(math.log2(100), 4), .1]],
        }

    def test_independent_ptc_anchors_do_not_report_interpolated_permission(self):
        entry = self.entry()
        self.assertIsNone(priors.gain_e_per_dn(entry, 150))
        state = stages(entry, iso=150)["ptc_gain"]
        self.assertTrue(state["fit_exists"])
        self.assertFalse(state["valid"])
        self.assertFalse(state["applicable"])
        self.assertTrue(stages(entry, iso=100)["ptc_gain"]["valid"])

    def test_failed_and_uncovered_read_noise_is_not_currently_valid(self):
        entry = self.entry()
        self.assertTrue(stages(entry, iso=100)["read_noise_e"]["valid"])
        for iso in (150, 200, 800):
            with self.subTest(iso=iso):
                self.assertIsNone(priors.read_noise_e(entry, iso))
                for product in ("read_noise_dn", "read_noise_e"):
                    state = stages(entry, iso=iso)[product]
                    self.assertTrue(state["fit_exists"])
                    self.assertFalse(state["valid"])
                    self.assertFalse(state["applicable"])
        entry["read_noise_composition_iso"] = 200
        for product in ("read_noise_dn", "read_noise_e"):
            self.assertFalse(stages(entry, iso=100)[product]["valid"])

    def test_matching_and_iso_evaluation_are_explicit(self):
        entry = self.entry()
        state = stages(entry)["ptc_gain"]
        self.assertTrue(state["fit_exists"])
        self.assertIsNone(state["valid"])
        self.assertIsNone(state["applicable"])
        entry["capability_matches"]["ptc_gain"] = {"status": "mismatch", "reason": "file-shutter-mismatch"}
        state = stages(entry, iso=100, consumers={"ptc_gain": ["sensor-gain-evidence"]})["ptc_gain"]
        self.assertTrue(state["qualified"])
        self.assertFalse(state["matched"])
        self.assertFalse(state["valid"])
        self.assertFalse(state["applicable"])
        self.assertFalse(state["consumers"])
        self.assertIn("profile-level", state["consumer_semantics"])

    def test_spectrum_uses_production_rounded_nominal_iso_support(self):
        entry = self.entry()
        self.assertTrue(stages(entry, iso=100)["spectrum_h"]["valid"])
        state = stages(entry, iso=800)["spectrum_h"]
        self.assertTrue(state["fit_exists"])
        self.assertFalse(state["valid"])
        self.assertFalse(state["applicable"])


if __name__ == "__main__":
    unittest.main()
