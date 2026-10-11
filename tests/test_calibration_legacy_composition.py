# SPDX-License-Identifier: GPL-3.0-or-later
"""Legacy contracts remain distinct from measured per-table qualifications."""
import copy
from dataclasses import replace
import hashlib
import json
import math
import struct
import unittest

from dngscan import calibration, calibration_registry as registry, priors
from dngscan.analysis import analyze
from dngscan.calibration_capabilities import qualification, usable
from dngscan.noise_model import model_from_prior, resolve_noise_model
from dngscan.raw_io import load_raw
from tests import test_calibration_capability_qualification as measured
from tests.test_readout_contract import edit_ifd
from tests.test_spectral_fallback_pipeline import write_noise_dng


class LegacyCompositionTests(unittest.TestCase):
    def setUp(self):
        self.collect = measured.CapabilityQualificationTests()
        self.collect.setUp()
        self.addCleanup(self.collect.doCleanups)
        self.root = self.collect.root
        entries = list(self.collect.entries)
        self.collect.entries = [(name, value) for name, value in entries if b'#Format: JPTC/2' in value]
        self.gain_folder, _, self.gain_payload = self.collect.package(
            'gain', lambda value: value + b'\n#BlackLevel: 100,100,100,100\n')
        self.collect.entries = [(name, value) for name, value in entries if b'#Format: JPTC-DARK/1' in value]
        _, _, self.dark_payload = self.collect.package('dark')
        self.dark_payload['reference_dn_range'] = 3995.
        self.path = self.root / 'input.dng'
        self.bundle = self.read_bundle(self.path, file_profile=False)
        self.source_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def read_bundle(self, path, *, file_profile):
        write_noise_dng(path, file_profile=file_profile)
        edit_ifd(path, {271: (2, 6, b'NIKON\0'), 272: (2, 11, b'TestCamera\0'),
                            34855: (3, 1, struct.pack('<H', 100)),
                            50714: (5, 1, struct.pack('<LL', 100, 1)),
                            50717: (4, 1, struct.pack('<L', 4095))})
        return load_raw(path, scene_half_size=True,
                        capture_declaration={'version': 1, 'shutter': 'mechanical'})

    def prior(self, role, *, legacy=False):
        payload = copy.deepcopy(self.gain_payload if role == 'gain' else self.dark_payload)
        if legacy:
            # Hand-authored legacy records declare their domain. They do not
            # pretend to be newly qualified Collect CSV measurements.
            normalized = calibration._validated_prior(payload)
            payload = {'format': 'dngscan-jptc-collect-1', 'id': 'legacy ' + role,
                       'make': payload['make'], 'model_candidates': payload['model_candidates'],
                       'shutter': payload['shutter'], 'reference_dn_range': 3995.,
                       'readout_contract': copy.deepcopy(normalized['readout_contract']),
                       'acquisition_contract': copy.deepcopy(normalized['acquisition_contract'])}
            for key in ('gain_log2iso_log2epd', 'read_noise_dn_log2iso',
                        'stored_dark_variance_dn2_log2iso', 'fwc_e', 'ptc_anchor',
                        'gain_support_intervals', 'gain_jump_isos'):
                if key not in normalized:
                    continue
                payload[key] = copy.deepcopy(normalized.get(key, []))
        entry = calibration._validated_prior(payload)
        entry['source'] = ('legacy ' if legacy else 'measured ') + role
        entry['reference_dn_range'] = 3995.
        return priors.with_readout(entry, self.bundle.capture_readout, shutter='mechanical')

    def test_four_combinations_keep_units_supplier_qualification_and_rebinding(self):
        for legacy_gain in (False, True):
            for legacy_dark in (False, True):
                with self.subTest(legacy_gain=legacy_gain, legacy_dark=legacy_dark):
                    gain = self.prior('gain', legacy=legacy_gain)
                    dark = self.prior('dark', legacy=legacy_dark)
                    joined = registry.compose_capabilities(gain, [dark], 100)
                    expected_read = calibration.curve_value(dark, 'read_noise_dn_log2iso', 100)
                    expected_gain = priors.gain_e_per_dn(gain, 100)
                    self.assertAlmostEqual(priors.read_noise_e(joined, 100), expected_read * expected_gain)
                    self.assertIsNone(priors.read_noise_e(joined, 200))
                    self.assertEqual(joined['capability_sources']['read_noise'], dark['source'])
                    self.assertEqual(joined['capability_measurements']['read_noise'], dark.get('measurement_acquisition'))
                    self.assertEqual(qualification(joined, 'read_noise_dn'), qualification(dark, 'read_noise_dn'))
                    if legacy_gain or legacy_dark:
                        self.assertIsNone(qualification(joined, 'read_noise_e'))
                        self.assertEqual(joined['composed_capabilities']['read_noise_e']['qualification_semantics'],
                                         'legacy-explicit-contract')
                    else:
                        self.assertEqual(qualification(joined, 'read_noise_e')['status'], 'qualified')
                    self.assertEqual(len(joined['composed_capabilities']['read_noise_e']['additional_dependencies']), 1)
                    rebound = priors.with_readout(joined, self.bundle.capture_readout, shutter='mechanical')
                    self.assertAlmostEqual(priors.read_noise_e(rebound, 100), expected_read * expected_gain)
                    changed = dict(self.bundle.capture_readout, shutter='electronic')
                    rejected = priors.with_readout(joined, changed, shutter='electronic')
                    self.assertFalse(usable(rejected, 'read_noise_e'))
                    self.assertIsNone(priors.read_noise_e(rejected, 100))

    def test_installing_legacy_dark_after_gain_only_analysis_completes_with_real_dng(self):
        calibration.import_calibrations(self.gain_folder)
        before, _, _ = analyze(self.bundle, 4, diagnostics=False)
        self.assertEqual(before.noise_model.status, 'unavailable')
        self.assertEqual(before.noise_model.reason, 'read-noise-unavailable')
        legacy = self.prior('dark', legacy=True)
        payload = {'format': 'dngscan-jptc-collect-1', 'id': 'independent legacy dark',
                   'make': self.dark_payload['make'], 'model_candidates': self.dark_payload['model_candidates'],
                   'shutter': 'mechanical', 'reference_dn_range': 3995.,
                   'read_noise_dn_log2iso': [[math.log2(100), 2.]],
                   'readout_contract': legacy['readout_contract'],
                   'acquisition_contract': legacy['acquisition_contract']}
        path = self.root / 'legacy-dark.json'
        path.write_text(json.dumps(payload))
        imported = calibration.import_calibrations(path)
        self.assertTrue(imported['profiles'][0]['capabilities']['read_noise_dn'])
        after, _, _ = analyze(self.bundle, 4, diagnostics=False)
        self.assertEqual(after.noise_model.status, 'valid')
        self.assertAlmostEqual(after.gain_e_per_dn, 2.)
        selected = priors.find_priors('NIKON', 'TestCamera', shutter='mechanical', iso=100,
                                     readout=self.bundle.capture_readout)
        self.assertAlmostEqual(priors.read_noise_e(selected, 100), 4.)
        self.assertIsNone(qualification(selected, 'read_noise_e'))
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.source_hash)

    def test_incompatible_or_unresolved_supplier_cannot_be_combined(self):
        for legacy in (False, True):
            gain = self.prior('gain')
            dark = self.prior('dark', legacy=legacy)
            mutations = ({'make_equals': 'OTHER'}, {'model_equals': ['OTHER']},
                         {'shutter': 'electronic'}, {'reference_dn_range': 3996.},
                         {'readout_contract': {**dark['readout_contract'], 'storage_lossless': False}},
                         {'readout_contract': {**dark['readout_contract'], 'libraw_raw_geometry': [256, 128]}},
                         {'acquisition_contract': {**dark['acquisition_contract'], 'dn_domain': 'electrons'}},
                         {'read_noise_unresolved_isos': [100], 'read_noise_dn_log2iso': []})
            for change in mutations:
                with self.subTest(legacy=legacy, fields=list(change)):
                    joined = registry.compose_capabilities(gain, [dict(dark, **change)], 100)
                    self.assertFalse('capability_sources' in joined)
                    self.assertIsNone(priors.read_noise_e(joined, 100))
            self.assertFalse('capability_sources' in registry.compose_capabilities(gain, [dark], 800))

    def test_stored_variance_keeps_its_own_supplier_and_invalid_evidence_falls_back(self):
        profile_bundle = self.read_bundle(self.root / 'file-profile.dng', file_profile=True)
        for legacy_gain in (False, True):
            for legacy_dark in (False, True):
                with self.subTest(legacy_gain=legacy_gain, legacy_dark=legacy_dark):
                    gain, dark = self.prior('gain', legacy=legacy_gain), self.prior('dark', legacy=legacy_dark)
                    joined = registry.compose_capabilities(gain, [dark], 100)
                    self.assertEqual(calibration.stored_dark_variance(joined, 100), calibration.stored_dark_variance(dark, 100))
                    self.assertEqual(qualification(joined, 'stored_dark_variance'), qualification(dark, 'stored_dark_variance'))
                    self.assertEqual(joined['capability_sources']['stored_dark_variance'], dark['source'])
                    actual = model_from_prior(self.bundle, {i: 4095. for i in range(4)}, joined)
                    self.assertEqual(actual.status, 'valid')
                    self.assertAlmostEqual(actual.coefficients('G1')[1], 4. / 3995.**2)
                    broken = dict(dark, stored_dark_variance_dn2_log2iso=[[math.log2(100), .1]])
                    rejected = registry.compose_capabilities(gain, [broken], 100)
                    self.assertFalse('capability_sources' in rejected)
                    fallback = resolve_noise_model(profile_bundle, {i: 4095. for i in range(4)}, rejected)
                    self.assertEqual(fallback.source, 'DNG NoiseProfile')
                    self.assertEqual(fallback.status, 'valid')

    def test_unresolved_legacy_read_noise_keeps_only_applicable_independent_spectrum(self):
        bundle = self.read_bundle(self.root / 'spectral-file-profile.dng', file_profile=True)
        gain, dark = self.prior('gain'), self.prior('dark', legacy=True)
        dark.update(read_noise_dn_log2iso=[], read_noise_unresolved_isos=[100],
                    stored_dark_variance_dn2_log2iso=[],
                    noise_whiteness_h_log2iso=[[math.log2(100), .1]])
        joined = registry.compose_capabilities(gain, [dark], 100)
        self.assertFalse('capability_sources' in joined)
        model = resolve_noise_model(bundle, {i: 4095. for i in range(4)}, joined)
        self.assertEqual(model.source, 'DNG NoiseProfile')
        self.assertEqual(model.status, 'valid')
        self.assertEqual(model.correlation, 'measured-spectral-imbalance')
        self.assertIn(.1, model.spectral_ratios.values())
        changed = replace(bundle, shot_shutter='electronic',
                          capture_readout={**bundle.capture_readout, 'shutter': 'electronic'})
        rebound = resolve_noise_model(changed, {i: 4095. for i in range(4)}, joined)
        self.assertEqual(rebound.source, 'DNG NoiseProfile')
        self.assertFalse(rebound.spectral_ratios)


if __name__ == '__main__':
    unittest.main()
