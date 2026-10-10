# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase calibration units, runtime propagation, and cache round trips."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from dngscan.calibration import _validated_prior
from dngscan.noise_model import NoiseModel, model_from_prior
from dngscan.noise_propagation import calibrated_chroma_variance
from tests.test_noise_model import frame

LUMA = np.asarray([.2627, .6780, .0593], dtype=np.float64)


def profile():
    span = 15359.
    value = {'format': 'dngscan-jptc-collect-1', 'id': 'phase ruler',
             'make': 'SIGMA', 'model_candidates': ['SIGMA fp'], 'shutter': 'any',
             'ptc_anchor': {'iso': 640, 'gain_e_per_dn': 2., 'quality': 'ok'},
             'fwc_e': 2 * span, 'gain_log2iso_log2epd': [[math.log2(640), 1.]],
             'read_noise_log2iso_log2e': [[math.log2(640), 1.]], 'phase_calibration': {}}
    for i, cid in enumerate((0, 1, 3, 2)):
        value['phase_calibration'][f'C{i // 2}{i % 2}'] = {
            'channel': f'C{i // 2}{i % 2}', 'color_index': cid, 'color_desc': 'RGBG',
            'color': 'RGBG'[cid], 'black_dn_log2iso': [[math.log2(640), 1024.]],
            'stored_dark_variance_dn2_log2iso': [[math.log2(640), float((i + 1) ** 2)]],
            'read_noise_dn_log2iso': [[math.log2(640), float(i + 1)]],
            'read_noise_unresolved_isos': []}
    return value


class PhaseNoiseModelTests(unittest.TestCase):
    def test_import_preserves_phase_stored_variance_and_shared_gain_declaration(self):
        prior = _validated_prior(profile())
        bundle = frame()
        model = model_from_prior(bundle, {i: 16383 for i in range(4)}, prior)
        self.assertEqual(model.status, 'valid')
        self.assertEqual(len(model.phase_variance), 4)
        for i, phase in enumerate(('C00', 'C01', 'C10', 'C11')):
            np.testing.assert_allclose(model.phase_variance[phase], (1 / 30718., (i + 1) ** 2 / 15359. ** 2))
            self.assertEqual(model.phase_metadata[phase]['gain_source'], 'shared-scalar-approximation')
            self.assertEqual(model.phase_metadata[phase]['uncertainty'], 'not-quantified')
        self.assertNotEqual(model.coefficients('G1'), model.coefficients('G2'))

    def test_missing_or_wrong_phase_does_not_become_a_measured_plane(self):
        item = profile()
        item['phase_calibration'].pop('C10')
        item['phase_calibration']['C00']['color_index'] = 2
        item['phase_calibration']['C00']['color'] = 'B'
        model = model_from_prior(frame(), {i: 16383 for i in range(4)}, _validated_prior(item))
        self.assertEqual(set(model.phase_variance), {'C01', 'C11'})
        self.assertEqual(model.phase_metadata['C00']['reason'], 'phase-identity-mismatch')
        self.assertEqual(model.phase_metadata['C10']['reason'], 'phase-measurement-unavailable')

    def test_dn_storage_rescale_preserves_normalized_phase_variance(self):
        prior = _validated_prior(profile())
        original = frame()
        normal = model_from_prior(original, {i: 16383 for i in range(4)}, prior)
        scaled = replace(original, white_level=32766, camera_white_levels=[32766.] * 4,
                         black_levels=[2048.] * 4)
        doubled = model_from_prior(scaled, {i: 32766 for i in range(4)}, prior)
        self.assertEqual(normal.phase_variance, doubled.phase_variance)

    def test_independent_phase_gain_requires_physical_provenance_and_dn_reference(self):
        item = profile()
        phase = item['phase_calibration']['C01']
        phase.update(gain_log2iso_log2epd=[[math.log2(640), 2.]], reference_dn_range=15359.,
                     gain_provenance='independent-phase-ptc')
        model = model_from_prior(frame(), {i: 16383 for i in range(4)}, _validated_prior(item))
        self.assertAlmostEqual(model.phase_variance['C01'][0], 1 / (4 * 15359.))
        self.assertEqual(model.phase_metadata['C01']['gain_source'], 'independent-phase-fit')
        phase.pop('gain_provenance')
        with self.assertRaises(ValueError):
            _validated_prior(item)

    def test_cache_roundtrip_retains_phase_tuples_and_metadata(self):
        from dngscan.gui.preview_cache import _analysis_from_json, _analysis_to_json
        from tests.test_preview_cache import _analysis
        model = model_from_prior(frame(), {i: 16383 for i in range(4)}, _validated_prior(profile()))
        analysis = replace(_analysis(), noise_model=model)
        restored = _analysis_from_json(json.loads(json.dumps(_analysis_to_json(analysis))))
        self.assertEqual(restored.noise_model, model)

    def test_phase_failed_read_noise_blocks_interpolation_and_explicit_bad_fit(self):
        item = profile()
        phase = item['phase_calibration']['C01']
        phase['read_noise_dn_log2iso'] = [[math.log2(100),1.],[math.log2(800),1.]]
        phase['stored_dark_variance_dn2_log2iso'] = [[math.log2(100),1.],[math.log2(800),1.]]
        phase['read_noise_unresolved_isos'] = [200]
        model = model_from_prior(frame(),{i:16383 for i in range(4)},_validated_prior(item))
        self.assertNotIn('C01',model.phase_variance)
        self.assertEqual(model.phase_metadata['C01']['reason'],'read-noise-unresolved-interval')
        # Stored total variance can exist even when the physical component
        # was unresolved; that failed point cannot silently become an endpoint.
        phase['read_noise_dn_log2iso'] = []
        phase['stored_dark_variance_dn2_log2iso'] = [[math.log2(200),1.],[math.log2(800),1.]]
        model = model_from_prior(frame(),{i:16383 for i in range(4)},_validated_prior(item))
        self.assertNotIn('C01',model.phase_variance)
        self.assertEqual(model.phase_metadata['C01']['reason'],'read-noise-unresolved-interval')
        phase['read_noise_unresolved_isos'] = []
        phase['fit_quality'] = 'high-residual'
        model = model_from_prior(frame(),{i:16383 for i in range(4)},_validated_prior(item))
        self.assertNotIn('C01',model.phase_variance)
        self.assertEqual(model.phase_metadata['C01']['reason'],'phase-fit-quality-high-residual')

    def test_unequal_green_dn_normalization_retains_raw_model_but_blocks_three_input_transfer(self):
        from dngscan.noise_propagation import chroma_nr_skip_reason
        bundle = frame()
        bundle = replace(bundle,camera_white_levels=[16383.,16383.,16383.,16000.])
        model = model_from_prior(bundle,{i:bundle.camera_white_levels[i] for i in range(4)},_validated_prior(profile()))
        self.assertEqual(model.status,'valid')
        self.assertEqual(model.phase_metadata['C10']['normalized_raw_span_dn'],14976.)
        propagation = SimpleNamespace(raw_image=np.zeros((512,512),np.uint16),
            raw_pattern=bundle.raw_pattern,color_desc=bundle.color_desc,
            scene_decoder='libraw',wb_mode='camera',noise_decode={'supported':True},
            scene_geometry_ops=(),lens_shading=None)
        self.assertEqual(chroma_nr_skip_reason(propagation,model),
                         'unequal-green-normalization-noise-transfer-unavailable')


class PhasePropagationTests(unittest.TestCase):
    def bundle(self):
        return SimpleNamespace(raw_image=np.zeros((512, 512), np.uint16),
            raw_pattern=[[0, 1], [3, 2]], color_desc='RGBG', scene_decoder='libraw',
            wb_mode='camera', exposure_gain=1., lens_filter='none', lens_shading=None,
            scene_geometry_ops=(), noise_decode={'supported': True,
            'normalized_raw_to_scene': np.asarray([[2., .1, 0], [0, .8, .2], [.1, 0, 1.2]])})

    def test_unequal_green_variances_use_squared_merge_weights_monte_carlo(self):
        bundle = self.bundle()
        variances = [.004, .001, .025, .003]
        phases = {f'C{i // 2}{i % 2}': (0., b) for i, b in enumerate(variances)}
        model = NoiseModel(status='valid', phase_variance=phases)
        actual, _ = calibrated_chroma_variance(bundle, model, np.full((64, 64, 3), .2))
        noise = np.random.default_rng(433).normal(size=(160000, 4)) * np.sqrt(np.asarray(variances) / 16)
        rgb = np.stack([noise[:, 0], .5 * (noise[:, 1] + noise[:, 2]), noise[:, 3]], axis=-1)
        scene = rgb @ bundle.noise_decode['normalized_raw_to_scene'].T
        chroma = scene - (scene @ LUMA)[:, None]
        measured = chroma.var(axis=0)
        np.testing.assert_allclose(actual[32, 32], measured, rtol=.012)

    def test_phase_gainmap_squares_before_combining_green_variance(self):
        bundle = self.bundle()
        bundle.noise_decode.update(full_sensor_shape=[512, 512], sensor_crop=[0, 0, 512, 512],
                                  orientation_flip=0)
        phase_map = {'top': 0, 'left': 1, 'bottom': 512, 'right': 512,
            'plane': 0, 'planes': 1, 'row_pitch': 2, 'col_pitch': 2,
            'spacing_v': 1., 'spacing_h': 1., 'origin_v': 0., 'origin_h': 0.,
            'gains': [[[3.], [3.]], [[3.], [3.]]]}
        bundle.noise_decode['gain_maps'] = [phase_map]
        model = NoiseModel(status='valid', phase_variance={
            'C00': (0., .004), 'C01': (0., .001), 'C10': (0., .025), 'C11': (0., .003)})
        actual, _ = calibrated_chroma_variance(bundle, model, np.full((64, 64, 3), .2))
        matrix = bundle.noise_decode['normalized_raw_to_scene']
        p = np.eye(3) - np.ones((3, 1)) * LUMA[None, :]
        camera = np.diag([.004 / 16, (.001 * 9 + .025) / 64, .003 / 16])
        expected = np.diag(p @ matrix @ camera @ matrix.T @ p.T)
        np.testing.assert_allclose(actual[32, 32], expected, rtol=2e-7)

    def test_unequal_green_white_balance_does_not_gain_unvalidated_decoder_authority(self):
        from dngscan import dng_opcodes as ops
        from dngscan.raw_io import _libraw_noise_decode, load_raw
        from tests.test_pipeline_corrections import write_sensor_dng
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'phase-wb.dng'
            write_sensor_dng(path, signal=1000)
            bundle = load_raw(path)
            evidence = replace(bundle.evidence, camera_wb=[2., 1., 1., 1.2])
            raw = SimpleNamespace(color_matrix=evidence.color_matrix, rgb_xyz_matrix=evidence.xyz_to_cam)
            descriptor = _libraw_noise_decode(raw, evidence, ops.read_plan(path), 'clip', bundle.scene_scale, False)
            self.assertFalse(descriptor['supported'])
            self.assertIn('unequal-green-white-balance', descriptor['reason'])


if __name__ == '__main__':
    unittest.main()
