"""Exact optional ChromaNR and bounded RAW-domain guidance."""
import dataclasses
import os
import unittest
from unittest import mock

import numpy as np

from dngscan import _fast, chroma_nr, guidance
from tests.golden_support import _bundle_from_scene, build_staggered_clip


def old_bin(arr, ph, pw):
    h, w = arr.shape[:2]
    h2, w2 = max(1, h // ph), max(1, w // pw)
    return arr[:h2 * ph, :w2 * pw].reshape(h2, ph, w2, pw, arr.shape[2]).min(axis=(1, 3))


class BoundedGuidanceTests(unittest.TestCase):
    def test_cell_order_and_exceptional_minima(self):
        rng = np.random.default_rng(29)
        for period in ((2, 2), (6, 6), (3, 2)):
            for kind in ('finite', 'zeros', 'nan'):
                data = rng.uniform(0, 1, (139, 25, 3)).astype(np.float32)
                if kind == 'zeros':
                    data = rng.choice(np.array([-0., 0.], np.float32), data.shape)
                if kind == 'nan':
                    data.reshape(-1)[::17] = np.nan
                for view in (data, data[::-1, ::-1], data.transpose(1, 0, 2)):
                    self.assertEqual(guidance._bin_period_min(view, *period).tobytes(),
                                     old_bin(view, *period).tobytes())

    def test_bands_preserve_fullwell_green_alias_snr_geometry_and_half_permission(self):
        rng = np.random.default_rng(30)
        analysis = dataclasses.replace(build_staggered_clip().analysis, gain_e_per_dn=1.37,
            prior_read_noise_e=2.31, prior_quality_status='ok', prior_model_spread=0.,
            channel_fullwell={0: 13001., 1: 12731., 2: 13123., 3: 12717.})
        for period in ((2, 2), (6, 6)):
            h, w = 139, 31
            raw = rng.integers(0, 16384, (h, w), dtype=np.uint16)
            pattern = rng.integers(0, 5, period, dtype=np.uint8)
            colors = np.tile(pattern, ((h + period[0] - 1) // period[0],
                                      (w + period[1] - 1) // period[1]))[:h, :w]
            for flip in (0, 3, 5, 6):
                for reverse in (False, True):
                    view = raw[::-1] if reverse else raw
                    labels = colors[::-1] if reverse else colors
                    scene = np.zeros((23, 17, 3), np.uint16)
                    bundle = _bundle_from_scene(scene, raw_image=view, raw_colors=labels,
                                                clip_masks=np.zeros(scene.shape, np.float16))
                    bundle = dataclasses.replace(bundle, raw_pattern=pattern, color_desc='RGBGX',
                                                 black_levels=[1024., 1025.25, 1026.5, 1027.75, 1024.],
                                                 orientation_flip=flip)
                    actual = guidance.build_raw_guidance_maps(bundle, analysis)
                    with mock.patch.object(guidance, '_binned_raw_evidence', return_value=None), \
                         mock.patch.object(guidance, '_bin_period_min', side_effect=old_bin):
                        expected = guidance.build_raw_guidance_maps(bundle, analysis)
                    for name in ('headroom', 'clip_class', 'snr_confidence', 'raw_permission'):
                        self.assertEqual(getattr(actual, name).tobytes(), getattr(expected, name).tobytes(),
                                         (period, flip, reverse, name))

    def test_invalid_prior_preserves_scene_fallback(self):
        fixture = build_staggered_clip()
        with mock.patch.object(guidance, '_binned_raw_evidence') as binned:
            self.assertIsNone(guidance._raw_snr_confidence(fixture.bundle, None, (5, 5)))
        binned.assert_not_called()


@unittest.skipUnless(_fast._load_extension() is not None, 'matching native extension required')
class NativeB3Tests(unittest.TestCase):
    def test_reflection_strides_singletons_empty_and_large_holes(self):
        ext = _fast._load_extension()
        self.addCleanup(ext.set_thread_budget, 0)
        rng = np.random.default_rng(31)
        base = rng.uniform(-10, 10, (19, 23)).astype(np.float32)
        for plane in (base, base[::-1, ::-2], base.T, np.broadcast_to(base[:1, :1], (7, 9)),
                      base[:1], base[:, :1], base[:0], base[:, :0]):
            for level in (0, 1, 3, 6):
                expected = (chroma_nr._atrous_smooth_reference(plane, level)
                            if plane.size else np.empty(plane.shape, np.float32))
                for workers in (1, 2):
                    ext.set_thread_budget(workers)
                    actual = ext.atrous_smooth_f32(plane, level)
                    self.assertEqual(expected.tobytes(), actual.tobytes())

    def test_invalid_direct_inputs_reject_without_cast_or_write(self):
        ext = _fast._load_extension()
        for values, level in ((np.zeros((4, 5), np.float64), 0),
                              (np.zeros((4, 5, 3), np.float32), 0),
                              (np.zeros((4, 5), np.float32), -1),
                              (np.zeros((4, 5), np.float32), 1000),
                              (np.zeros((4, 5), '>f4'), 0),
                              (np.ndarray((4, 5), np.float32, bytearray(81), offset=1), 0)):
            with self.assertRaises((ValueError, TypeError, OverflowError)):
                ext.atrous_smooth_f32(values, level)

    def test_singleton_isize_min_stride_falls_back_even_in_strict_mode(self):
        limit = int(np.iinfo(np.intp).max)
        source = np.array([[.371]], np.float32)
        # Only one element is addressed; NumPy ALIGNED accepts this legitimate
        # singleton stride, but ndarray's borrowed-view contract rejects it.
        unusual = np.lib.stride_tricks.as_strided(source, shape=(1, 1),
            strides=(-limit - 1, 4))
        expected = chroma_nr._atrous_smooth_reference(unusual, 0)
        with mock.patch.dict(os.environ, DNGSCAN_FAST='1', DNGSCAN_FAST_SKIP=''), \
             mock.patch.object(_fast, '_load_extension') as native:
            actual = chroma_nr._atrous_smooth(unusual, 0)
        native.assert_not_called()
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_nonfinite_and_signed_zero_operation_order(self):
        ext = _fast._load_extension()
        plane = np.random.default_rng(33).random((17, 19), dtype=np.float32)
        plane.reshape(-1)[:8] = (-0., 0., np.nan, np.inf, -np.inf,
                                 np.finfo('f4').max, -np.finfo('f4').max, np.finfo('f4').tiny)
        for level in (0, 1, 5):
            with np.errstate(invalid='ignore', over='ignore'):
                expected = chroma_nr._atrous_smooth_reference(plane, level)
                actual = ext.atrous_smooth_f32(plane, level)
            self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_complete_nr_keeps_mad_shrinkage_and_projection_exact(self):
        scene = np.random.default_rng(32).uniform(-.02, 1, (193, 211, 3)).astype(np.float32)
        for factor in (1., 4.25):
            with mock.patch.dict(os.environ, {'DNGSCAN_FAST': '1', 'DNGSCAN_FAST_SKIP': ''}):
                actual = chroma_nr.chroma_correction_map(scene, .67, factor)
            with mock.patch.object(chroma_nr, '_atrous_smooth', side_effect=chroma_nr._atrous_smooth_reference):
                expected = chroma_nr.chroma_correction_map(scene, .67, factor)
            self.assertEqual(actual.tobytes(), expected.tobytes())


if __name__ == '__main__':
    unittest.main()
