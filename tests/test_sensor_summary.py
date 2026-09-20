# SPDX-License-Identifier: GPL-3.0-or-later
"""Capture-scoped scalar reuse must preserve the pre-cache sensor contract."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import copy
from dataclasses import FrozenInstanceError, fields, replace
import gc
import math
from pathlib import Path
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import weakref

import numpy as np

from dngscan import analysis as an
from dngscan.evidence import _immutable_sensor_copy
from dngscan.models import RawBundle, RawEvidence
from dngscan.scene_reference import reliable_reference_samples
from dngscan.sensor_summary import summarize_sensor


def _bayer_arrays():
    colors = np.tile(np.array([[0, 1], [3, 2]], dtype=np.uint8), (32, 32))
    raw = np.empty(colors.shape, dtype=np.uint16)
    for cid, value in enumerate((990, 980, 500, 995)):
        raw[colors == cid] = value
    return raw, colors


def _evidence(raw=None, colors=None, *, immutable=True, white=1000, levels=None):
    if raw is None:
        raw, colors = _bayer_arrays()
    if immutable:
        raw, colors = _immutable_sensor_copy(raw), _immutable_sensor_copy(colors)
    return RawEvidence(
        path=Path('summary.dng'), raw_image=raw, raw_colors=colors,
        white_level=white, black_levels=[0.] * 4, camera_wb=[1.] * 4,
        daylight_wb=[1.] * 4, color_desc='RGBG', raw_pattern=[[0, 1], [3, 2]],
        camera_white_levels=list([1000.] * 4 if levels is None else levels),
        orientation_flip=0, xyz_to_cam=np.eye(3), provider_version='test-runtime',
    )


def _bundle(evidence):
    scene = np.full((32, 32, 3), .2, dtype=np.float32)
    return RawBundle(
        path=evidence.path, raw_image=evidence.raw_image, raw_colors=evidence.raw_colors,
        xyz_render=scene.copy(), render_scale=1., scene_rec2020_render=scene,
        scene_scale=1., white_level=evidence.white_level,
        black_levels=list(evidence.black_levels), camera_wb=list(evidence.camera_wb),
        color_desc=evidence.color_desc, raw_pattern=evidence.raw_pattern,
        camera_white_levels=list(evidence.camera_white_levels), evidence=evidence,
    )


def _summary(evidence, **overrides):
    args = dict(raw_image=evidence.raw_image, raw_colors=evidence.raw_colors,
                white_level=evidence.white_level,
                camera_white_levels=evidence.camera_white_levels, evidence=evidence)
    args.update(overrides)
    return summarize_sensor(**args)


def _legacy_oracle(raw, colors, white, levels):
    """The original public-helper composition, independent of the cache/helper."""
    ids = [int(x) for x in sorted(np.unique(colors).tolist())]
    sat = an.channel_saturation_levels(ids, levels, white)
    ceilings, exact, near, spike = an.detect_ceilings(raw, colors, ids, sat)
    fullwell, fullwell_ids, note, channel_fullwell = an.resolve_fullwell(ids, ceilings, spike, sat)
    return dict(channel_ids=tuple(ids), ceilings=tuple(ceilings.items()),
                exact_counts=tuple(exact.items()), near_counts=tuple(near.items()),
                spike_ok=tuple(spike.items()), saturation_levels=tuple(sat.items()),
                fullwell=fullwell, fullwell_channel_ids=tuple(fullwell_ids),
                fullwell_note=note, channel_fullwell=tuple(channel_fullwell.items()))


@contextmanager
def _small_analysis():
    """Keep clipping/threshold/mask-refresh real; isolate unrelated camera models."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(an, 'estimate_raw_noise_floor', return_value=.001))
        stack.enter_context(patch.object(an, 'compute_snr_curves', return_value=({}, {}, {})))
        stack.enter_context(patch.object(an, 'raw_health_metrics', return_value=(0., 0.)))
        stack.enter_context(patch.object(an, 'sensor_prior_evidence',
                                        return_value=(None,) * 8))
        yield


class SensorSummaryOracleTests(unittest.TestCase):
    def assert_summary(self, actual, expected):
        self.assertEqual({field.name: getattr(actual, field.name) for field in fields(actual)}, expected)
        for name in ('channel_ids', 'fullwell_channel_ids'):
            self.assertIsInstance(getattr(actual, name), tuple)
            self.assertTrue(all(type(value) is int for value in getattr(actual, name)))
        for name in ('ceilings', 'exact_counts', 'near_counts', 'spike_ok',
                     'saturation_levels', 'channel_fullwell'):
            self.assertIsInstance(getattr(actual, name), tuple)
            for key, value in getattr(actual, name):
                self.assertIs(type(key), int)
                self.assertIs(type(value), bool if name == 'spike_ok' else int)

    def test_bayer_fullwell_combinations_preserve_separate_green_channels(self):
        for values in ((500, 500, 500, 500), (990, 980, 500, 995), (990, 980, 975, 995)):
            with self.subTest(values=values):
                raw, colors = _bayer_arrays()
                for cid, value in enumerate(values):
                    raw[colors == cid] = value
                evidence = _evidence(raw, colors)
                actual = _summary(evidence)
                self.assert_summary(actual, _legacy_oracle(raw, colors, 1000, [1000.] * 4))
                self.assertEqual(actual.channel_ids, (0, 1, 2, 3))
                self.assertEqual(dict(actual.ceilings)[1], values[1])
                self.assertEqual(dict(actual.ceilings)[3], values[3])
                with self.assertRaises(FrozenInstanceError):
                    actual.fullwell = 1

    def test_no_pile_and_metadata_int_fallback_match_old_helpers(self):
        raw, colors = _bayer_arrays()
        for cid in range(4):
            raw[colors == cid] = np.arange(np.count_nonzero(colors == cid), dtype=np.uint16)
        evidence = _evidence(raw, colors, levels=[0.9, 0., -4., 1100.9])
        actual = _summary(evidence)
        self.assert_summary(actual, _legacy_oracle(raw, colors, 1000, evidence.camera_white_levels))
        self.assertEqual(dict(actual.saturation_levels), {0: 1000, 1: 1000, 2: 1000, 3: 1100})
        self.assertFalse(any(dict(actual.spike_ok).values()))
        short = _evidence(raw, colors, levels=[0.9])
        self.assert_summary(_summary(short), _legacy_oracle(raw, colors, 1000, [0.9]))

    def test_xtrans_linear_rgb_sparse_ids_and_strided_arrays(self):
        xtrans = np.array([[1, 0, 1, 1, 2, 1], [2, 1, 2, 0, 1, 0],
                           [1, 0, 1, 1, 2, 1], [1, 2, 1, 1, 0, 1],
                           [0, 1, 0, 2, 1, 2], [1, 2, 1, 1, 0, 1]], dtype=np.uint8)
        xc = np.tile(xtrans, (8, 8))
        lc = np.broadcast_to(np.frombuffer(bytes((0, 1, 2)), dtype=np.uint8), (24, 24, 3))
        sc = np.tile(np.array([[5, 0], [2, 5]], dtype=np.uint8), (24, 24))
        for name, colors in (('xtrans', xc), ('linear', lc), ('sparse', sc), ('strided', xc[::-1, ::2])):
            for dtype in (np.uint16, np.uint32, np.float64):
                with self.subTest(layout=name, dtype=dtype):
                    raw = (950 + colors.astype(np.uint16) * 3).astype(dtype)
                    levels = [1000.] * 6
                    self.assert_summary(summarize_sensor(raw, colors, 1000, levels),
                                        _legacy_oracle(raw, colors, 1000, levels))
        # Integer DN codes above f32's exact range must not be narrowed.
        raw = np.full((20, 20), 2**24 + 1, dtype=np.uint32)
        colors = np.zeros(raw.shape, dtype=np.uint8)
        self.assert_summary(summarize_sensor(raw, colors, 2**24 + 3, []),
                            _legacy_oracle(raw, colors, 2**24 + 3, []))

    def test_near_pile_and_plausibility_boundaries(self):
        colors = np.zeros((32, 32), dtype=np.uint8)
        for count in (255, 256):
            for ceiling in (949, 950):
                with self.subTest(count=count, ceiling=ceiling):
                    raw = np.full(colors.shape, 100, dtype=np.uint16)
                    raw.flat[:count] = ceiling - 2
                    raw.flat[0] = ceiling
                    actual = summarize_sensor(raw, colors, 1000, [])
                    self.assert_summary(actual, _legacy_oracle(raw, colors, 1000, []))
                    self.assertEqual(dict(actual.near_counts)[0], count)
                    self.assertEqual(actual.fullwell, 950 if count == 256 and ceiling == 950 else 1000)
        raw = np.full(colors.shape, 950.75, dtype=np.float64)
        actual = summarize_sensor(raw, colors, 1000, [])
        self.assert_summary(actual, _legacy_oracle(raw, colors, 1000, []))
        self.assertEqual(dict(actual.ceilings)[0], 950)
        self.assertEqual(dict(actual.exact_counts)[0], 0)

    def test_nan_inf_and_empty_inputs_keep_legacy_errors(self):
        cases = []
        for value in (float('nan'), float('inf'), float('-inf')):
            cases.append((np.full((2, 2), value), np.zeros((2, 2), dtype=np.uint8), []))
            cases.append((np.ones((2, 2), dtype=np.uint16), np.zeros((2, 2), dtype=np.uint8), [value]))
        cases.append((np.empty((0, 2), dtype=np.uint16), np.empty((0, 2), dtype=np.uint8), []))
        for raw, colors, levels in cases:
            with self.subTest(raw=raw.tolist(), levels=levels):
                try:
                    _legacy_oracle(raw, colors, 1000, levels)
                except Exception as expected:
                    with self.assertRaises(type(expected)) as caught:
                        summarize_sensor(raw, colors, 1000, levels)
                    self.assertEqual(str(caught.exception), str(expected))
                else:
                    self.fail('Invalid fixture unexpectedly has a legacy summary')


class SensorSummaryMemoTests(unittest.TestCase):
    def test_linear_broadcast_colors_are_eligible_without_materializing(self):
        raw = _immutable_sensor_copy(np.full((24, 24, 3), 500, dtype=np.uint16))
        colors = np.broadcast_to(np.frombuffer(bytes((0, 1, 2)), dtype=np.uint8), raw.shape)
        evidence = replace(_evidence(raw, colors, immutable=False), sample_kind='linear-camera-rgb')
        with patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect:
            first = _summary(evidence)
            self.assertIs(_summary(evidence), first)
        self.assertEqual(detect.call_count, 1)
        self.assertEqual(first.channel_ids, (0, 1, 2))
        self.assertEqual(evidence.raw_colors.strides, (0, 0, 1))

    def test_reference_keeps_sparse_channel_level_indices(self):
        colors = np.tile(np.array([[5, 0], [2, 5]], dtype=np.uint8), (24, 24))
        raw = np.full(colors.shape, 500, dtype=np.uint16)
        evidence = _evidence(raw, colors, levels=[1000., 0., 1200., 0., 0., 1500.])
        scene = np.full((24, 24, 3), .2, dtype=np.float32)
        with patch('dngscan.raw_io.build_clip_masks', return_value=np.zeros(scene.shape, dtype=np.float16)) as masks:
            reference, pct = reliable_reference_samples(evidence, scene, 1., None,
                                                        SimpleNamespace(post=(), crop=None))
        self.assertEqual(masks.call_args.args[5], [1000, 1000, 1200, 1000, 1000, 1500])
        self.assertEqual(reference.shape, (576, 3))
        self.assertEqual(pct, 100.)

    def test_reference_then_analysis_reuses_scalars_but_replays_refresh_and_margin(self):
        from dngscan import raw_io
        evidence = _evidence()
        bundle = _bundle(evidence)
        with patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect, \
             patch.object(raw_io, 'refresh_clip_masks_from_fullwell',
                          wraps=raw_io.refresh_clip_masks_from_fullwell) as refresh, _small_analysis():
            reliable_reference_samples(evidence, bundle.scene_rec2020_render, 1., None,
                                       SimpleNamespace(post=(), crop=None))
            first, _, _ = an.analyze(bundle, 4, diagnostics=False, gamut_names=('srgb',))
            second, _, _ = an.analyze(bundle, 16, diagnostics=False, gamut_names=('srgb',))
        self.assertEqual(detect.call_count, 1)
        self.assertEqual(refresh.call_count, 2)
        self.assertEqual(first.channel_fullwell, second.channel_fullwell)
        self.assertEqual(first.threshold - second.threshold, 12)
        for cid in first.channel_ids:
            self.assertEqual(first.channel_thresholds[cid] - second.channel_thresholds[cid], 12)

    def test_analysis_gets_independent_mutable_copies(self):
        evidence = _evidence()
        bundle = _bundle(evidence)
        with patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect, _small_analysis():
            first, _, _ = an.analyze(bundle, 4, diagnostics=False, gamut_names=('srgb',))
            expected = {name: dict(getattr(first, name)) for name in
                        ('ceilings', 'ceil_spike_counts', 'ceil_near_counts', 'ceil_spike_ok',
                         'saturation_levels', 'channel_fullwell')}
            ids, fullwell_ids = list(first.channel_ids), list(first.fullwell_channel_ids)
            first.channel_ids.append(99)
            first.fullwell_channel_ids.append(99)
            for name in expected:
                getattr(first, name)[0] = -999
            second, _, _ = an.analyze(bundle, 4, diagnostics=False, gamut_names=('srgb',))
        self.assertEqual(detect.call_count, 1)
        self.assertEqual(second.channel_ids, ids)
        self.assertEqual(second.fullwell_channel_ids, fullwell_ids)
        for name, value in expected.items():
            self.assertEqual(getattr(second, name), value)
            self.assertIsNot(getattr(second, name), getattr(first, name))

    def test_mutated_metadata_and_flattened_bundle_fields_do_not_reuse_old_values(self):
        evidence = _evidence(levels=[0.] * 4)
        bundle = _bundle(evidence)
        with patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect:
            original = _summary(evidence)
            self.assertIs(_summary(evidence), original)
            evidence.camera_white_levels[0] = 1200.
            updated = _summary(evidence)
            self.assertEqual(dict(updated.saturation_levels)[0], 1200)
            self.assertEqual(detect.call_count, 2)
            # load_raw copies these fields into the bundle; that copy may diverge.
            changed = summarize_sensor(bundle.raw_image, bundle.raw_colors, 2000,
                                       bundle.camera_white_levels, evidence=evidence)
            self.assertEqual(dict(changed.saturation_levels)[0], 2000)
            bundle.camera_white_levels[0] = 1500.
            changed = summarize_sensor(bundle.raw_image, bundle.raw_colors, bundle.white_level,
                                       bundle.camera_white_levels, evidence=evidence)
            self.assertEqual(dict(changed.saturation_levels)[0], 1500)
            self.assertEqual(detect.call_count, 4)

    def test_replaced_or_view_arrays_do_not_inherit_identity_cache(self):
        for change in ('raw-copy', 'colors-copy', 'raw-view', 'colors-view'):
            with self.subTest(change=change):
                evidence = _evidence()
                overrides = {}
                name = 'raw_image' if change.startswith('raw') else 'raw_colors'
                value = getattr(evidence, name)
                overrides[name] = _immutable_sensor_copy(value) if change.endswith('copy') else value.view()
                with patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect:
                    _summary(evidence)
                    _summary(evidence, **overrides)
                    _summary(evidence, **overrides)
                self.assertEqual(detect.call_count, 3)

    def test_in_place_layout_and_dtype_changes_invalidate(self):
        for change in ('shape', 'dtype'):
            with self.subTest(change=change):
                evidence = _evidence()
                with patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect:
                    _summary(evidence)
                    # Deliberately exercise this still-supported mutation hole;
                    # new arrays/views would test a different identity instead.
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', DeprecationWarning)
                        if change == 'shape':
                            evidence.raw_image.shape = (32, 128)
                            evidence.raw_colors.shape = (32, 128)
                        else:
                            evidence.raw_image.dtype = np.int16
                    actual = _summary(evidence)
                    self.assertEqual(detect.call_count, 2)
                self.assertEqual(dict(actual.ceilings), {0: 990, 1: 980, 2: 500, 3: 995})

    def test_provenance_and_policy_changes_invalidate(self):
        for name, value in (('provider', 'other'), ('provider_version', 'new-runtime'),
                            ('sample_kind', 'linear-camera-rgb')):
            with self.subTest(provenance=name):
                evidence = _evidence()
                with patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect:
                    _summary(evidence)
                    object.__setattr__(evidence, name, value)
                    _summary(evidence)
                self.assertEqual(detect.call_count, 2)
        for name, value in (('CEILING_PLAUSIBLE_FRACTION', .999), ('CEILING_MIN_PILE_PIXELS', 2000),
                            ('CEILING_MIN_PILE_FRACTION', .9), ('CEILING_NEAR_WINDOW_SCALE', 512)):
            with self.subTest(policy=name):
                evidence = _evidence()
                with patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect:
                    _summary(evidence)
                    with patch.object(an, name, value):
                        actual = _summary(evidence)
                    self.assertEqual(detect.call_count, 2)
                with patch.object(an, name, value):
                    expected = _legacy_oracle(evidence.raw_image, evidence.raw_colors, 1000, [1000.] * 4)
                self.assertEqual({field.name: getattr(actual, field.name) for field in fields(actual)}, expected)
        from dngscan import sensor_summary
        evidence = _evidence()
        with patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect:
            _summary(evidence)
            with patch.object(sensor_summary, '_SENSOR_SUMMARY_SCHEMA_VERSION', -1):
                _summary(evidence)
        self.assertEqual(detect.call_count, 2)

    def test_public_helper_patch_is_observed_after_warm_cache(self):
        evidence = _evidence()
        _summary(evidence)
        for name in ('channel_saturation_levels', 'detect_ceilings', 'resolve_fullwell'):
            with self.subTest(helper=name), patch.object(an, name, wraps=getattr(an, name)) as helper:
                _summary(evidence)
                self.assertEqual(helper.call_count, 1)

    def test_untrusted_readonly_or_writable_arrays_never_memoize(self):
        for kind in ('writable', 'owned-readonly', 'writable-base', 'bytearray-memoryview'):
            for unsafe in ('raw_image', 'raw_colors'):
                with self.subTest(kind=kind, unsafe=unsafe):
                    evidence = _evidence()
                    source = np.array(getattr(evidence, unsafe), copy=True)
                    if kind == 'owned-readonly':
                        source.flags.writeable = False
                    elif kind == 'writable-base':
                        source = source.view()
                        source.flags.writeable = False
                    elif kind == 'bytearray-memoryview':
                        source = np.frombuffer(memoryview(bytearray(source.tobytes())).toreadonly(),
                                               dtype=source.dtype).reshape(source.shape)
                    evidence = replace(evidence, **{unsafe: source})
                    with patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect:
                        _summary(evidence)
                        _summary(evidence)
                    self.assertEqual(detect.call_count, 2)

    def test_dataclass_replace_resets_capture_memo(self):
        evidence = _evidence()
        with patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect:
            _summary(evidence)
            copied = replace(evidence)
            _summary(copied)
            _summary(copied)
        self.assertEqual(detect.call_count, 2)

    def test_copy_and_pickle_drop_memo_and_preserve_sensor_values(self):
        for name, clone in (('copy', copy.copy), ('deepcopy', copy.deepcopy),
                            ('pickle4', lambda value: pickle.loads(pickle.dumps(value, protocol=4))),
                            ('pickle5', lambda value: pickle.loads(pickle.dumps(value, protocol=5)))):
            with self.subTest(clone=name):
                evidence = _evidence()
                expected = _summary(evidence)
                self.assertIsNotNone(evidence._sensor_summary_cache)
                restored = clone(evidence)
                self.assertIsNone(restored._sensor_summary_cache)
                np.testing.assert_array_equal(restored.raw_image, evidence.raw_image)
                np.testing.assert_array_equal(restored.raw_colors, evidence.raw_colors)
                self.assertEqual(restored.camera_white_levels, evidence.camera_white_levels)
                with patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect:
                    self.assertEqual(_summary(restored), expected)
                    self.assertEqual(_summary(restored), expected)
                if restored.raw_image.flags.writeable or restored.raw_colors.flags.writeable:
                    self.assertEqual(detect.call_count, 2)
                    self.assertIsNone(restored._sensor_summary_cache)
                else:
                    # NumPy protocol/version determines whether restored readonly
                    # storage is bytes-backed or merely a reversible owned buffer.
                    self.assertIn(detect.call_count, (1, 2))

    def test_failed_computation_is_not_cached_or_partially_published(self):
        evidence = _evidence()
        original = an.resolve_fullwell
        count = 0
        def sometimes_fail(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                raise RuntimeError('injected fullwell failure')
            return original(*args, **kwargs)
        with patch.object(an, 'resolve_fullwell', side_effect=sometimes_fail), \
             patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect:
            with self.assertRaisesRegex(RuntimeError, 'injected fullwell failure'):
                _summary(evidence)
            self.assertIsNone(evidence._sensor_summary_cache)
            result = _summary(evidence)
            self.assertIs(_summary(evidence), result)
        self.assertEqual(detect.call_count, 2)

    def test_metadata_changed_during_computation_does_not_publish_stale_memo(self):
        evidence = _evidence()
        original = an.resolve_fullwell
        def mutate_metadata(*args, **kwargs):
            evidence.camera_white_levels[0] = 1200.
            return original(*args, **kwargs)
        with patch.object(an, 'resolve_fullwell', side_effect=mutate_metadata), \
             patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect:
            first = _summary(evidence)
            self.assertIsNone(evidence._sensor_summary_cache)
            second = _summary(evidence)
            self.assertIs(_summary(evidence), second)
        self.assertEqual(detect.call_count, 2)
        self.assertEqual(dict(first.saturation_levels)[0], 1000)
        self.assertEqual(dict(second.saturation_levels)[0], 1200)

    def test_metadata_aba_change_uses_entry_snapshot_not_mutable_list(self):
        evidence = _evidence()
        original = an.channel_saturation_levels
        def transient_metadata_change(*args, **kwargs):
            evidence.camera_white_levels[0] = 1200.
            try:
                return original(*args, **kwargs)
            finally:
                evidence.camera_white_levels[0] = 1000.
        with patch.object(an, 'channel_saturation_levels', side_effect=transient_metadata_change), \
             patch.object(an, 'detect_ceilings', wraps=an.detect_ceilings) as detect:
            actual = _summary(evidence)
            self.assertEqual(dict(actual.saturation_levels)[0], 1000)
            self.assertIs(_summary(evidence), actual)
        self.assertEqual(detect.call_count, 1)

    def test_proxy_and_cached_bundle_do_not_keep_capture_arrays_alive(self):
        from dngscan.gui.preview_cache import build_proxy_entry, _bundle_metadata, _bundle_from_cache
        evidence = _evidence()
        bundle = _bundle(evidence)
        with _small_analysis():
            analysis, _, _ = an.analyze(bundle, 4, diagnostics=False, gamut_names=('srgb',))
        summary = _summary(evidence)
        memo = evidence._sensor_summary_cache
        raw_ref, colors_ref, evidence_ref = weakref.ref(bundle.raw_image), weakref.ref(bundle.raw_colors), weakref.ref(evidence)
        proxy = build_proxy_entry(bundle, analysis)
        restored = _bundle_from_cache(bundle.path, _bundle_metadata(bundle),
                                      bundle.scene_rec2020_render.copy(), None, None)
        del bundle, evidence
        gc.collect()
        self.assertIsNone(raw_ref())
        self.assertIsNone(colors_ref())
        self.assertIsNone(evidence_ref())
        self.assertIsNone(proxy.bundle.evidence)
        self.assertIsNone(restored.evidence)
        self.assertEqual(summary.channel_ids, (0, 1, 2, 3))
        self.assertIs(memo.summary, summary)

    def test_absent_sensor_arrays_use_decoded_only_even_with_stale_evidence(self):
        evidence = _evidence()
        _summary(evidence)
        bundle = _bundle(evidence)
        bundle.raw_image = bundle.raw_colors = None
        with patch.object(an, 'detect_ceilings', side_effect=AssertionError('sensor path forbidden')):
            result, _, _ = an.analyze(bundle, 4, diagnostics=False, gamut_names=('srgb',))
        self.assertEqual(result.channel_ids, [])
        self.assertEqual(result.channel_fullwell, {})
        self.assertTrue(math.isnan(result.fullwell))
        self.assertEqual(result.noise_evidence_status, 'unavailable')


if __name__ == '__main__':
    unittest.main()
