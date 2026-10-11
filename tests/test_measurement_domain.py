# SPDX-License-Identifier: GPL-3.0-or-later
"""Real writer/import/unpack/model checks for numeric-domain permission."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from dngscan import calibration, priors, readout
from dngscan.analysis import analyze
from dngscan.collect_manifest import prepare_sources
from dngscan.measurement_domain import DN, DN2, SCHEMA, capture_domain
from dngscan.raw_io import load_raw
from tests.test_readout_contract import edit_ifd
from tests.test_spectral_fallback_pipeline import write_noise_dng
from tests.test_user_calibration import collect_profile

FIXTURE = Path(__file__).parent / "fixtures/collect-writer/official.zip"


def without(content, field):
    return b"\n".join(line for line in content.splitlines() if not line.startswith(("#" + field + ":").encode())) + b"\n"


def sample_domain_fixture():
    """Explicit synthetic descriptor for pure matching tests, not camera evidence."""
    return {"schema": SCHEMA, "provider": "libraw", "provider_version": "synthetic LibRaw 0.22.0",
            "stage": "unpack-before-project-corrections", "signal_unit": DN, "variance_unit": DN2,
            "sample_kind": "cfa", "dtype": "uint16", "channel_description": "RGBG",
            "phase_pattern": [[0, 1], [3, 2]], "linearisation": "identity",
            "coding_spans_dn": [3995.], "sampled_code_difference_gcd": 1,
            "sampled_phase_code_difference_gcd": {"C00": 1, "C01": 1, "C10": 1, "C11": 1},
            "adc_evidence": "synthetic fixture"}


class MeasurementDomainPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "store"))
        self.environment.start(); self.addCleanup(self.environment.stop)
        with zipfile.ZipFile(FIXTURE) as archive:
            self.entries = [(item.filename, archive.read(item)) for item in archive.infolist()]

    def sources(self, variant="base"):
        folder = self.root / (variant + "-csv"); folder.mkdir(exist_ok=True)
        for index, (_, content) in enumerate(self.entries):
            content = without(content, "Firmware").replace(b"6000x4000", b"128x128")
            if variant in ("electrons", "all"):
                content += b"\n#DNDomain: electrons\n"
            if variant in ("electron-squared", "all"):
                content += b"\n#VarianceDomain: electron-squared\n"
            if variant in ("foreign", "all"):
                content = without(content, "Decoder") + b"#Decoder: foreign-decoder-output\n"
            if variant == "aliases":
                content += b"\n#DNDomain: DN\n#VarianceDomain: DN^2\n"
                content = content.replace(b"#Decoder: LibRaw 0.22.2", b"#Decoder: LibRaw 0.23.0-Release")
            if variant == "adc-four":
                content = content.replace(b"#AdcStep: 1\n", b"#AdcStep: 4\n")
            if variant == "adc-fractional":
                content = content.replace(b"#AdcStep: 1\n", b"#AdcStep: 0.5\n")
            if variant == "channel-description":
                content = content.replace(b"#CfaPattern: RGBG", b"#CfaPattern: RGGB")
            (folder / f"measurement-{index}.csv").write_bytes(content)
        return folder

    def raw(self, name="input", *, file_profile=False, extra=None):
        path = self.root / (name + ".dng")
        write_noise_dng(path, file_profile=file_profile)
        edit_ifd(path, {271: (2, 6, b"NIKON\0"), 272: (2, 11, b"TestCamera\0"),
                       34855: (3, 1, struct.pack("<H", 100)),
                       50714: (5, 1, struct.pack("<LL", 100, 1)),
                       50717: (4, 1, struct.pack("<L", 4095)), **(extra or {})})
        return path

    def pipeline(self, variant="base", *, file_profile=False, extra=None):
        sources = self.sources(variant)
        report, payloads = prepare_sources(sources)
        calibration.import_calibrations(sources)
        path = self.raw(variant, file_profile=file_profile, extra=extra)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        bundle = load_raw(path, scene_half_size=True, capture_declaration={"version": 1, "shutter": "mechanical"})
        selected = priors.find_priors("NIKON", "TestCamera", shutter="mechanical", iso=100, readout=bundle.capture_readout)
        analysis, _, _ = analyze(bundle, 4, diagnostics=False)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        return report["groups"][0], payloads, bundle, selected, analysis

    def test_official_writer_defaults_bind_actual_unpack_without_exact_version_requirement(self):
        group, payloads, bundle, selected, result = self.pipeline()
        self.assertEqual(group["status"], "ready")
        self.assertEqual(selected["readout_match_status"], "matched")
        self.assertEqual(selected["measurement_domain_match"]["status"], "matched")
        self.assertNotEqual(selected["acquisition_contract"]["decoder_identity"], bundle.evidence_provider_version)
        self.assertEqual(result.noise_model.status, "valid")
        self.assertAlmostEqual(result.gain_e_per_dn, 2.)
        self.assertEqual(bundle.capture_readout["sample_domain"]["linearisation"], "identity")
        # CfaPattern is cdesc/channel slots, not the spatial RGGB string.
        self.assertEqual(bundle.capture_readout["sample_domain"]["channel_description"], "RGBG")
        self.assertEqual(bundle.capture_readout["sample_domain"]["phase_pattern"], [[0, 1], [3, 2]])
        payload = payloads[group["id"]]
        self.assertEqual(payload["acquisition_contract"]["dn_domain"], DN)
        domains = payload["measurement_acquisition"]["numeric_domains"]
        self.assertTrue(all(row["signal_unit"] == DN and row["variance_unit"] == DN2 for row in domains.values()))
        self.assertTrue(all(row["unit_source"]["dn_domain"] == "supported-format" for row in domains.values()))
        self.assertAlmostEqual(result.noise_model.coefficients("G1")[1], 4. / 3995.**2)

    def test_each_explicit_unsupported_domain_disables_external_model_and_keeps_pixels(self):
        expected = {"electrons": "measurement-dn-domain-unsupported",
                    "electron-squared": "measurement-variance-domain-unsupported",
                    "foreign": "measurement-decoder-domain-unverified",
                    "all": "measurement-decoder-domain-unverified"}
        path = self.raw("reference")
        reference = load_raw(path, scene_half_size=True).scene_rec2020_render.copy()
        for variant, reason in expected.items():
            with self.subTest(variant=variant), patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / (variant + "-store"))):
                group, _, bundle, selected, result = self.pipeline(variant)
                self.assertEqual(group["status"], "partial")
                self.assertFalse(any(group["capabilities"].values()))
                self.assertEqual(selected["readout_match_status"], "unverified")
                self.assertEqual(result.noise_model.status, "rejected")
                self.assertEqual(result.noise_model.reason, reason)
                self.assertIsNone(result.gain_e_per_dn)
                self.assertIsNone(result.noise_model.coefficients("G1"))
                np.testing.assert_array_equal(bundle.scene_rec2020_render, reference)

    def test_invalid_external_domain_preserves_file_profile_fallback(self):
        for variant in ("electrons", "electron-squared", "foreign", "all"):
            with self.subTest(variant=variant), patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / (variant + "-store"))):
                _, _, _, _, result = self.pipeline(variant, file_profile=True)
                self.assertEqual(result.noise_model.status, "valid")
                self.assertEqual(result.noise_model.source, "DNG NoiseProfile")
                self.assertTrue(result.noise_model.fallback_reason.startswith("measurement-"))
                self.assertEqual(result.noise_model.coefficients("G1"), (1e-4, 1e-8))
                self.assertFalse(result.noise_model.spectral_ratios)
                self.assertIsNone(result.gain_e_per_dn)

    def test_supported_unit_aliases_and_other_libraw_version_keep_same_numeric_model(self):
        _, _, _, selected, result = self.pipeline("aliases")
        self.assertEqual(selected["readout_match_status"], "matched")
        self.assertEqual(result.noise_model.status, "valid")
        self.assertEqual(selected["acquisition_contract"]["dn_domain"], DN)
        self.assertEqual(selected["acquisition_contract"]["variance_domain"], DN2)

    def test_actual_nonidentity_unpack_lut_rejects_identity_measurement(self):
        table = np.arange(4096, dtype=np.uint16) * 2
        _, _, bundle, selected, result = self.pipeline(extra={50712: (3, len(table), table.astype("<u2").tobytes())}, file_profile=True)
        self.assertEqual(bundle.capture_readout["sample_domain"]["linearisation"], "companded")
        self.assertEqual(selected["readout_match_status"], "unverified")
        self.assertEqual(result.noise_model.fallback_reason, "file-linearisation-domain-companded")
        self.assertEqual(result.noise_model.source, "DNG NoiseProfile")

    def test_actual_phase_mapping_and_channel_description_cannot_be_substituted(self):
        for variant, extra, reason in (
            ("channel-description", None, "file-channel-description-mismatch"),
            ("base", {33422: (1, 4, bytes([2, 1, 1, 0]))}, "file-measurement-cfa-phase-mismatch"),
        ):
            with self.subTest(variant=variant), patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / (variant + "-store"))):
                _, _, _, selected, result = self.pipeline(variant, extra=extra)
                self.assertEqual(selected["readout_match_status"], "mismatch")
                self.assertEqual(result.noise_model.reason, reason)

    def test_adc_declaration_must_be_representable_and_consistent_with_actual_code_grid(self):
        for variant, reason in (("adc-four", "file-measurement-adc-grid-mismatch"),
                                ("adc-fractional", "measurement-adc-step-noninteger")):
            with self.subTest(variant=variant), patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / (variant + "-store"))):
                _, _, bundle, _, result = self.pipeline(variant)
                self.assertEqual(bundle.capture_readout["sample_domain"]["sampled_code_difference_gcd"], 1)
                self.assertEqual(result.noise_model.status, "rejected")
                self.assertEqual(result.noise_model.reason, reason)

    def test_explicit_json_domain_has_no_undeclared_readout_bypass_or_input_mutation(self):
        legacy = collect_profile()
        self.assertEqual(readout.match(calibration._validated_prior(legacy), {})[0], "not-declared")
        for contract in ({"dn_domain": "electrons"}, {"variance_domain": "electron-squared"},
                         {"decoder_identity": "foreign-decoder-output"}, {"dn_domain": "DN"},
                         {"decoder_identity": "LibRaw 0.22.2"}):
            with self.subTest(contract=contract):
                item = {**legacy, "acquisition_contract": contract}
                before = copy.deepcopy(item)
                prior = calibration._validated_prior(item)
                self.assertEqual(readout.match(prior, {})[0], "unverified")
                self.assertFalse(priors.prior_usability(priors.with_readout(prior, {}))[0])
                self.assertEqual(item, before)

    def test_actual_evidence_domain_is_required_even_for_qualified_import(self):
        _, _, bundle, selected, _ = self.pipeline()
        incomplete = dict(bundle.capture_readout); incomplete.pop("sample_domain")
        bound = priors.with_readout(selected, incomplete)
        self.assertEqual(bound["readout_match_reason"], "file-sample-domain-unavailable")
        self.assertFalse(priors.prior_usability(bound)[0])
        self.assertEqual(selected["readout_match_status"], "matched")
        # An opaque scene formation never becomes the sensor evidence decoder.
        wrong = copy.deepcopy(bundle.capture_readout)
        wrong["sample_domain"]["provider"] = "coreimage"
        self.assertEqual(priors.with_readout(selected, wrong)["readout_match_reason"], "file-unpack-domain-unsupported")

    def test_phase_offsets_do_not_masquerade_as_adc_steps(self):
        from types import SimpleNamespace
        _, _, bundle, selected, _ = self.pipeline()
        # Each phase has a step of 4; different constant phase offsets make
        # the combined CFA GCD 2. Only within-phase differences constrain q.
        image = np.arange(64*64, dtype=np.uint16).reshape(64, 64) * 4
        image[::2, 1::2] += 2
        native = SimpleNamespace(tone_curve=np.arange(65536, dtype=np.uint16))
        descriptor = capture_domain(native, image, [[0, 1], [3, 2]], "RGBG", [4095.], [100.], "test LibRaw")
        self.assertEqual(descriptor["sampled_code_difference_gcd"], 2)
        self.assertTrue(all(step % 4 == 0 for step in descriptor["sampled_phase_code_difference_gcd"].values()))
        capture = copy.deepcopy(bundle.capture_readout); capture["sample_domain"] = descriptor
        profile = copy.deepcopy(selected); profile["acquisition_contract"]["adc_step"] = 4.
        self.assertEqual(priors.with_readout(profile, capture)["measurement_domain_match"]["status"], "matched")

    def test_existing_power_two_storage_transport_scales_the_adc_grid(self):
        _, _, bundle, selected, _ = self.pipeline()
        capture = copy.deepcopy(bundle.capture_readout)
        capture["sample_domain"]["coding_spans_dn"] = [3995.*2]
        capture["sample_domain"]["sampled_phase_code_difference_gcd"] = {"C00": 2, "C01": 2, "C10": 2, "C11": 2}
        bound = priors.with_readout(selected, capture)
        self.assertEqual(bound["measurement_domain_match"]["status"], "matched")
        self.assertEqual(bound["measurement_domain_match"]["coding_scale_log2"], [1])
        capture["sample_domain"]["sampled_phase_code_difference_gcd"]["C00"] = 1
        self.assertEqual(priors.with_readout(selected, capture)["readout_match_reason"], "file-measurement-adc-grid-mismatch")

    def test_supported_versioned_json_still_needs_complete_actual_domain(self):
        _, _, bundle, _, _ = self.pipeline()
        item = collect_profile()
        item.pop("readout_contract", None)
        item["fwc_e"] = 3995.*4
        item["reference_dn_range"] = 3995.
        item["acquisition_contract"] = {"dn_domain": DN, "variance_domain": DN2,
            "decoder_identity": "LibRaw 0.22.2", "linearisation_curve": "identity", "adc_step": 1,
            "cfa_pattern": "RGBG"}
        prior = calibration._validated_prior(item)
        bound = priors.with_readout(prior, bundle.capture_readout, shutter="electronic")
        self.assertEqual(bound["measurement_domain_match"]["status"], "matched")
        self.assertEqual(bound["readout_match_status"], "not-declared")
        capture = copy.deepcopy(bundle.capture_readout); capture["sample_domain"].pop("linearisation")
        self.assertEqual(priors.with_readout(prior, capture)["readout_match_status"], "unverified")

    def test_measured_anchor_and_channel_black_levels_use_the_same_dn_span(self):
        from types import SimpleNamespace
        _, _, bundle, selected, _ = self.pipeline()
        selected.pop("reference_dn_range")
        self.assertAlmostEqual(priors.reference_dn_range(selected), 3995.)
        self.assertEqual(priors.with_readout(selected, bundle.capture_readout)["measurement_domain_match"]["status"], "matched")
        native = SimpleNamespace(tone_curve=np.arange(65536, dtype=np.uint16))
        actual = capture_domain(native, np.arange(64, dtype=np.uint16).reshape(8, 8),
                                [[0, 1], [3, 2]], "RGBG", [4095.], [100., 101., 102., 103.], "LibRaw test")
        self.assertEqual(actual["coding_spans_dn"], [3995., 3994., 3993., 3992.])

    def test_independent_dark_span_qualifies_without_granting_physical_gain(self):
        _, _, bundle, selected, _ = self.pipeline()
        selected["gain_log2iso_log2epd"] = []
        selected["fwc_e"] = None
        selected.pop("unity_gain_ev", None)
        selected["reference_dn_range"] = 3995.
        bound = priors.with_readout(selected, bundle.capture_readout)
        self.assertEqual(bound["measurement_domain_match"]["status"], "matched")
        self.assertIsNone(priors.gain_for_file(bound, 100, 3995.))
        self.assertAlmostEqual(calibration.stored_dark_variance(bound, 100)[0], 4.)

    def test_composed_sources_compare_supported_domains_without_decoder_version_equality(self):
        from dngscan.calibration_registry import compose_capabilities
        _, _, _, selected, _ = self.pipeline()
        base = copy.deepcopy(selected); dark = copy.deepcopy(selected)
        base["read_noise_log2iso_log2e"] = []; base["read_noise_dn_log2iso"] = []
        base["stored_dark_variance_dn2_log2iso"] = []
        base["phase_calibration"] = {}; base["source"] = "gain source"
        dark["source"] = "dark source"
        dark["acquisition_contract"]["decoder_identity"] = "LibRaw 0.23.0-Release"
        joined = compose_capabilities(base, [dark], 100)
        self.assertEqual(joined["capability_sources"]["read_noise"], "dark source")
        self.assertAlmostEqual(calibration.stored_dark_variance(joined, 100)[0], 4.)
        bad = copy.deepcopy(dark); bad["acquisition_contract"]["variance_domain"] = "electron-squared"
        # Even a caller's stale pre-bound status cannot authorize bad units.
        self.assertNotIn("capability_sources", compose_capabilities(base, [bad], 100))


if __name__ == "__main__":
    unittest.main()
