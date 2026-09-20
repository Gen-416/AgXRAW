# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded sensor tile passes must reproduce the complete original reductions."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from dngscan import analysis as an, phase_statistics as phases
from dngscan.spatial_black import SpatialBlack
from tests.test_preview_cache import _bundle
from tools.benchmark_loss_pipeline import json_value


class PhaseStatisticsTests(unittest.TestCase):
    def make_bundle(self, shape, pattern, spatial=False, constant=False):
        bundle = _bundle()
        ph, pw = pattern.shape
        bundle.raw_pattern = pattern.tolist()
        bundle.raw_colors = np.tile(pattern, ((shape[0]+ph-1)//ph, (shape[1]+pw-1)//pw))[:shape[0], :shape[1]].copy()
        bundle.raw_image = np.random.default_rng(17).integers(0, 65536, shape, dtype=np.uint16)
        if constant:
            bundle.raw_image[:] = 65535
        bundle.black_levels = [63., 65., 61., 64.]
        if spatial:
            model = SpatialBlack(np.arange(shape[1]+4, dtype=np.float32)*.01,
                np.arange(shape[0]+5, dtype=np.float32)*.02,
                np.arange(12, dtype=np.float32).reshape(3, 4, 1)+61., (2, 3), (99.,))
            bundle.evidence = SimpleNamespace(spatial_black=model)
        else:
            bundle.evidence = None
        return bundle

    def test_exact_original_noise_snr_health_with_bands_ties_and_spatial_black(self):
        patterns = (np.array([[0, 1], [3, 2]], np.uint8),
                    np.array([[1,0,1,1,2,1], [2,1,2,0,1,0], [1,0,1,1,2,1],
                              [1,2,1,1,0,1], [0,1,0,2,1,2], [1,2,1,1,0,1]], np.uint8))
        for pattern in patterns:
            for shape in ((1, 1), (9, 11), (31, 35), (259, 291), (787, 805)):
                for spatial, constant in ((False, False), (True, False), (False, True)):
                    b = self.make_bundle(shape, pattern, spatial, constant)
                    ids = list(map(int, np.unique(b.raw_colors)))
                    labels = an.channel_labels('RGBG', ids)
                    fullwell = {cid: 65535 for cid in ids}
                    with self.subTest(period=pattern.shape, shape=shape, spatial=spatial, constant=constant):
                        expected_noise = an.estimate_raw_noise_floor(b, fullwell)
                        expected_snr = an.compute_snr_curves(b, ids, labels, fullwell)
                        expected_health = an.raw_health_metrics(b, ids, labels)
                        original = b.raw_image.tobytes()
                        with patch.object(phases, '_corrected_band', wraps=phases._corrected_band) as bands:
                            prepared = phases.build_phase_statistics(b, ids, labels)
                        self.assertTrue(all(call.args[-1] - call.args[-2] <= 128 for call in bands.call_args_list))
                        self.assertEqual(json_value(prepared.noise_floor(b, fullwell)), json_value(expected_noise))
                        self.assertEqual(json_value(an.compute_snr_curves(b, ids, labels, fullwell, _phase_stats=prepared.snr)), json_value(expected_snr))
                        self.assertEqual(json_value(prepared.health), json_value(expected_health))
                        self.assertEqual(b.raw_image.tobytes(), original)
                        self.assertTrue(all(a.ndim == 1 for values in prepared.snr.values() for a in values))

    def test_histogram_percentiles_preserve_numpy_int64_interpolation(self):
        rng = np.random.default_rng(388)
        for size in (1, 2, 3, 19, 20, 21, 33, 127, 65537):
            for values in (rng.integers(0, 65536, size, dtype=np.uint16), np.full(size, 65535, np.uint16)):
                hist = np.bincount(values, minlength=65536)
                for q in (5., 60.):
                    self.assertEqual(phases._histogram_percentile(hist, q), float(np.percentile(values.astype(np.int64), q)))
                p05, p60 = np.percentile(values.astype(np.int64), [5., 60.])
                lo, hi = int(p05), int(max(p60, p05+32))
                old = np.bincount(np.clip(values.astype(np.int64), lo, hi)-lo, minlength=hi-lo+1)
                self.assertEqual(phases._histogram_health(hist), float(np.mean(old[1:-1] == 0)*100.))

    def test_unsupported_layouts_keep_original_oracle(self):
        b = self.make_bundle((35, 39), np.array([[0,1], [3,2]], np.uint8))
        for raw in (b.raw_image.astype(np.float32), b.raw_image.T, b.raw_image[::-1], np.zeros((0, 0), np.uint16)):
            self.assertIsNone(phases.build_phase_statistics(replace(b, raw_image=raw), [0,1,2,3], {0:'R',1:'G',2:'B',3:'G2'}))

    def test_long_narrow_adaptive_noise_tiles_keep_original_oracle(self):
        for width in (10, 12, 14, 18, 20, 22, 24, 26, 28, 30):
            b = self.make_bundle((300, width), np.array([[0,1], [3,2]], np.uint8))
            self.assertIsNone(phases.build_phase_statistics(b, [0,1,2,3], {0:'R',1:'G',2:'B',3:'G2'}))


if __name__ == '__main__':
    unittest.main()
