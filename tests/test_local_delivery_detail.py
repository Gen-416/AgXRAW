# SPDX-License-Identifier: GPL-3.0-or-later
"""Actual delivery entry points must see supported local detail loss."""
from dataclasses import replace
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from dngscan import gainmap, _fast
from dngscan.auto_encode import coding_metrics, additional_error_acceptable
from dngscan.delivery import ARCHIVE_TOLERANCES, SHARE_TOLERANCES, SHARE_HEIC_TOLERANCES, resolve_delivery_profile
from dngscan.local_detail import local_detail_loss, detail_is_acceptable


class LocalDetailTests(unittest.TestCase):
    @staticmethod
    def checker(size, *, hdr=False, dark=False, colour=False, position=(100, 100), shape=(257, 261)):
        base = .025 if dark else .25
        amplitude = .02 if dark else .125
        expected = np.full(shape + (3,), base, np.float32)
        yy, xx = np.indices((size, size))
        pattern = np.where((xx + yy) % 2, -amplitude, amplitude)
        patch_rgb = np.repeat(pattern[..., None], 3, axis=2)
        if colour:
            # Opposing R/G colour structure; brightness remains nearly constant.
            patch_rgb[..., 1] *= -.35
            patch_rgb[..., 2] = 0
        y, x = position
        expected[y:y + size, x:x + size] += patch_rgb
        decoded = np.full_like(expected, base)
        if hdr:
            return decoded.astype(np.float16), expected.astype(np.float16)
        return np.rint(decoded * 255).astype(np.uint8), np.rint(expected * 255).astype(np.uint8)

    def test_erased_small_dark_and_mid_luma_regions_fail_all_absolute_profiles(self):
        for size in (8, 16, 32):
            for dark in (False, True):
                for position in ((100, 100), (257-size, 261-size)):
                    with self.subTest(size=size, dark=dark, position=position):
                        for hdr in (False, True):
                            a, e = self.checker(size, hdr=hdr, dark=dark, position=position)
                            metrics = (gainmap._roundtrip_error_arrays(a, e) if hdr
                                       else gainmap._base_roundtrip_error_arrays(a, e))
                            gate = gainmap._hdr_roundtrip_is_acceptable if hdr else gainmap._base_roundtrip_is_acceptable
                            for tolerance in (ARCHIVE_TOLERANCES, SHARE_TOLERANCES, SHARE_HEIC_TOLERANCES):
                                self.assertFalse(gate(metrics, tolerance))

    def test_multiple_frequencies_and_phase_reversal_are_observed(self):
        yy, xx = np.indices((64, 64))
        for period in (2, 4, 8, 16):
            e = np.repeat((64 + np.where((xx // (period // 2)) % 2, -32, 32))[..., None], 3, axis=2).astype(np.uint8)
            a = 128 - e
            self.assertGreater(local_detail_loss(a, e), .9)
            # Reversal keeps all variances identical; signed-position errors still reject it.
            self.assertEqual(float(a.var()), float(e.var()))

    def test_bad_pixel_quantization_and_dither_do_not_trigger_worst_pixel_gate(self):
        rng = np.random.default_rng(3)
        e = np.full((65, 67, 3), 128, np.uint8)
        for point in ((0, 0), (35, 35), (64, 66)):
            a = e.copy()
            a[point] = (0, 255, 0)
            self.assertLessEqual(local_detail_loss(a, e), 3 / 16)
            self.assertTrue(gainmap._base_roundtrip_is_acceptable(gainmap._base_roundtrip_error_arrays(a, e)))
        a = (e.astype(np.int16) + rng.integers(-1, 2, e.shape[:2] + (1,))).astype(np.uint8)
        self.assertLessEqual(local_detail_loss(a, e), .25)
        self.assertTrue(additional_error_acceptable(coding_metrics(a, e), coding_metrics(e, e)))
        eh = e.astype(np.float16) / 255
        ah = (eh.astype(np.float32) + rng.uniform(-.0005, .0005, eh.shape)).astype(np.float16)
        self.assertLess(local_detail_loss(ah, eh, linear_hdr=True), .1)
        ah[35, 35] = (0, 100, 0)
        self.assertLess(local_detail_loss(ah, eh, linear_hdr=True), .3)

    def test_hdr_quantization_floor_uses_same_eight_code_budget_as_sdr(self):
        from dngscan.local_detail import _local_detail_loss_numpy, _HDR_EIGHT_CODE_LINEAR
        # The original .01 linear floor was stricter than 8 SDR codes in the
        # middle tones. Give both outputs the same explicit quantization budget;
        # do not move the .90 gate or treat image variations as sensor noise.
        yy, xx = np.indices((64, 64))
        e = np.repeat((.125 + np.where((xx+yy) % 2, -.0075, .0075))[..., None], 3, axis=2).astype(np.float32)
        a = np.full_like(e, .125)
        loss = _local_detail_loss_numpy(a, e, linear_hdr=True)
        floor = float(_HDR_EIGHT_CODE_LINEAR * np.sqrt(np.float32(.125)))
        self.assertAlmostEqual(loss, (.015/floor)**2, places=5)
        self.assertAlmostEqual(local_detail_loss(a, e, linear_hdr=True), loss, delta=2e-7)
        self.assertLess(loss, .4)
        # A supported complete erasure remains catastrophic in dark, middle
        # and above-reference-white regions under the exact same floor.
        for mean, amplitude in ((.025, .02), (.25, .125), (2., .75)):
            for size in (8, 16, 32):
                e = np.full((128, 128, 3), mean, np.float32)
                pattern = np.where(np.indices((size,size)).sum(axis=0) % 2,
                                   -amplitude, amplitude)
                e[48:48+size, 48:48+size] += pattern[..., None]
                a = np.full_like(e, mean)
                self.assertEqual(_local_detail_loss_numpy(a, e, linear_hdr=True), 1.)

    def test_auto_rejects_eight_pixel_region_against_clean_reference(self):
        a, e = self.checker(8, position=(96, 96), shape=(256, 256))
        metrics = coding_metrics(a, e)
        self.assertEqual(metrics['coding_luma_rmse'], 1.)
        self.assertEqual(metrics['coding_local_luma_p99'], 0.)
        self.assertEqual(metrics['coding_local_detail_loss'], 1.)
        self.assertFalse(additional_error_acceptable(metrics, coding_metrics(e, e)))

    def test_equal_variance_unrelated_noise_cannot_impersonate_detail(self):
        a, e = self.checker(32, position=(96, 96), shape=(256, 256))
        rng = np.random.default_rng(14)
        a[96:128, 96:128] = rng.permutation(e[96:128, 96:128].reshape(-1, 3)).reshape(32, 32, 3)
        self.assertEqual(float(a.var()), float(e.var()))
        self.assertGreater(local_detail_loss(a, e), .9)

    def test_local_luma_gate_makes_no_pure_chroma_texture_guarantee(self):
        # 4:2:0 intentionally loses fine chroma. Keep the scope explicit rather
        # than treating all colour-noise/subsampling changes as catastrophic.
        from dngscan.local_detail import _SDR_LUMA, _HDR_LUMA
        for hdr in (False, True):
            weights = _HDR_LUMA if hdr else _SDR_LUMA
            for size in (8, 16, 32):
                for period in (2, 4, 8, 16):
                    e = np.full((128, 128, 3), .25, np.float32)
                    yy, xx = np.indices((size, size))
                    pattern = np.where((xx // (period//2)) % 2, -.1, .1)
                    e[48:48+size, 48:48+size, 0] += pattern
                    e[48:48+size, 48:48+size, 1] -= pattern * float(weights[0] / weights[1])
                    a = np.full_like(e, .25)
                    if hdr:
                        e, a = e.astype(np.float16), a.astype(np.float16)
                    else:
                        e, a = np.rint(e*255).astype(np.uint8), np.rint(a*255).astype(np.uint8)
                    self.assertLess(local_detail_loss(a, e, linear_hdr=hdr), .1)

    def test_numpy_native_workspace_and_fused_returns_have_identical_detail(self):
        for hdr in (False, True):
            a, e = self.checker(8, hdr=hdr)
            values = []
            for mode in (['0', '1'] if _fast._load_extension() is not None else ['0']):
                with patch.dict(os.environ, DNGSCAN_FAST=mode):
                    if hdr:
                        for workspace in (None, gainmap._new_hdr_metrics_workspace()):
                            values.append(gainmap._roundtrip_error_arrays(a, e, _workspace=workspace)['local_detail_loss'])
                    else:
                        values.append(gainmap._base_roundtrip_error_arrays(a, e)['base_local_detail_loss'])
                        values.append(gainmap._base_and_coding_metrics_arrays(a, e)['base_local_detail_loss'])
                        values.append(gainmap._base_and_coding_metrics_arrays(a, e)['coding_local_detail_loss'])
            self.assertEqual(values, [1.] * len(values))

    def test_alpha_stride_nonfinite_and_missing_metrics(self):
        a, e = self.checker(8, hdr=True)
        aa = np.concatenate((a, np.full(a.shape[:2] + (1,), np.nan, np.float16)), axis=2)
        ee = np.concatenate((e, np.full(e.shape[:2] + (1,), np.inf, np.float16)), axis=2)
        self.assertEqual(local_detail_loss(aa[::-1, ::-1], ee[::-1, ::-1], linear_hdr=True), 1.)
        aa[0, 0, 1] = np.nan
        self.assertEqual(local_detail_loss(aa, ee, linear_hdr=True), float('inf'))
        self.assertFalse(detail_is_acceptable({}, 'local_detail_loss'))
        self.assertFalse(detail_is_acceptable({'local_detail_loss': np.nan}, 'local_detail_loss'))

    def test_unsupported_storage_overflow_and_invalid_tail_reject(self):
        good = np.full((129, 17, 3), .25, np.float32)
        for bad in (good.astype(np.int32), good.astype('>f4')):
            self.assertEqual(local_detail_loss(bad, bad, linear_hdr=True), float('inf'))
        unaligned = np.ndarray(good.shape, np.float32, bytearray(good.nbytes+1), offset=1)
        unaligned[:] = good
        self.assertEqual(local_detail_loss(unaligned, unaligned, linear_hdr=True), float('inf'))
        # Finite inputs can overflow differences or normalized errors.
        huge = np.full_like(good, np.finfo(np.float32).max)
        for a, e in ((huge, -huge), (huge, np.zeros_like(huge))):
            self.assertEqual(local_detail_loss(a, e, linear_hdr=True), float('inf'))
        a, e = self.checker(32, hdr=True, position=(0, 0), shape=(129, 131))
        self.assertEqual(local_detail_loss(a, e, linear_hdr=True), 1.)
        a[-1, -1, 2] = np.nan
        self.assertEqual(local_detail_loss(a, e, linear_hdr=True), float('inf'))

    def test_real_jpeg_high_quality_preserves_luma_patterns(self):
        # Use actual codec output: the gate must not equate ordinary q95/q97 loss
        # with complete texture erasure. Fine chroma loss is tested separately.
        a, e = self.checker(32, position=(96, 96), shape=(256, 256))
        for quality in (95, 97, 100):
            for chroma in (0, 1, 2):
                buffer = io.BytesIO()
                Image.fromarray(e).save(buffer, format='JPEG', quality=quality, subsampling=chroma)
                buffer.seek(0)
                with Image.open(buffer) as image:
                    decoded = np.asarray(image.convert('RGB'))
                self.assertLess(local_detail_loss(decoded, e), .9)

    def test_manual_jpeg_checks_pixels_and_preserves_existing_file_on_failure(self):
        from dngscan.export import export_srgb_jpeg, save_jpeg_array
        a, e = self.checker(8, position=(96, 96), shape=(256, 256))
        profile = resolve_delivery_profile('share', quality=97, chroma='420')
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'existing.jpg'
            out.write_bytes(b'previous good image')
            def damaged_encoder(rgb, path, quality, gamut, subsampling):
                return save_jpeg_array(a, path, quality, gamut, subsampling)
            with patch('dngscan.export.render_output_u8', return_value=e), \
                 patch('dngscan.export.save_jpeg_array', side_effect=damaged_encoder), \
                 patch('dngscan.export.carry_capture_metadata', return_value=False):
                with self.assertRaisesRegex(RuntimeError, '局部细节'):
                    export_srgb_jpeg(Path('source.DNG'), out, 97, None, None, delivery=profile, subsampling=2)
            self.assertEqual(out.read_bytes(), b'previous good image')

    def test_manual_heif_checks_supported_detail_before_publish(self):
        from dngscan.heif_delivery import save_sdr_heif
        from tests.test_sdr_heif import SdrHeifFinalReadbackTests
        # Keep the real HEIF writer, pixel metrics and gate. Only the unavailable
        # codec/container I/O is replaced by the existing test fixture.
        fixture = SdrHeifFinalReadbackTests()
        e = np.full((256, 256, 3), 100, np.uint8)
        yy, xx = np.indices((8, 8))
        e[96:104, 96:104] = np.where((xx+yy) % 2, 68, 132)[..., None]
        profile = replace(resolve_delivery_profile('share', container='heic'), heif_encoder='x265')
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'existing.heic'
            out.write_bytes(b'previous')
            with fixture.codec_fixture(out, rgb=e):
                with self.assertRaisesRegex(RuntimeError, '门限'):
                    save_sdr_heif(e, out, profile)
            self.assertEqual(out.read_bytes(), b'previous')

    def test_packaged_hdr_writer_rejects_small_sdr_and_hdr_detail_losses(self):
        from tests.test_gainmap_staged_delivery import GainmapStagedWriterTests
        fixture_owner = GainmapStagedWriterTests()
        for damage_hdr in (False, True):
            with self.subTest(damage_hdr=damage_hdr), tempfile.TemporaryDirectory() as td:
                directory = Path(td)
                out = directory / 'existing.heic'
                out.write_bytes(b'previous')
                with fixture_owner.writer_fixture(directory) as fixture:
                    fixture.base = np.full((256, 256, 3), 180, np.uint8)
                    fixture.hdr = np.full((256, 256, 4), 2., np.float16)
                    fixture.hdr[..., 3] = 1
                    fixture.overrides.update(width=256, height=256)
                    fixture.read.return_value = fixture.base.copy()
                    fixture.hdr_read.return_value = fixture.hdr.copy()
                    yy, xx = np.indices((8, 8))
                    if damage_hdr:
                        fixture.hdr[96:104, 96:104, :3] = np.where((xx+yy) % 2, .125, .375)[..., None]
                        fixture.hdr_read.return_value[96:104, 96:104, :3] = .25
                    else:
                        fixture.base[96:104, 96:104] = np.where((xx+yy) % 2, 148, 212)[..., None]
                    with self.assertRaisesRegex(RuntimeError, '局部细节'):
                        fixture_owner.write(fixture, out)
                self.assertEqual(out.read_bytes(), b'previous')

    def test_real_heif_luma_survives_without_claiming_fine_chroma_preservation(self):
        from dngscan import heif_encoder
        from dngscan.heif_gainmap import _parse
        if not heif_encoder.available():
            self.skipTest('libheif/x265 unavailable')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'probe.heic'
            for colour in (False, True):
                _, e = self.checker(32, colour=colour, position=(96, 96), shape=(256, 256))
                for quality in (95, 97, 100):
                    for chroma in ('420', '444'):
                        heif_encoder.encode(e, path, quality, chroma, bit_depth=10,
                                            preset='fast', tune='ssim', output_gamut='srgb')
                        decoded = heif_encoder.read_rgb_item(path, _parse(path.read_bytes())[2])
                        loss = local_detail_loss(decoded, e)
                        self.assertLess(loss, .9)


@unittest.skipUnless(_fast._load_extension() is not None
                     and hasattr(_fast._load_extension(), 'local_detail_loss'),
                     'local-detail native kernel unavailable')
class NativeLocalDetailParityTests(unittest.TestCase):
    def test_custom_weights_aliases_invalid_weights_and_error_budgets(self):
        from dngscan.local_detail import _local_detail_loss_numpy
        extension = _fast._load_extension()
        yy, xx = np.indices((17, 19))
        e = np.full((17, 19, 3), 128, np.uint8)
        e[..., 1] = np.where((xx+yy) % 2, 64, 192)
        a = np.full_like(e, 128)
        # Optional weights have observable, precise semantics, not metadata only.
        self.assertEqual(extension.local_detail_loss(a, e, False, [1., 0., 0.]), 0.)
        self.assertEqual(extension.local_detail_loss(a, e), 1.)
        self.assertEqual(extension.local_detail_loss(e, e), 0.)
        for weights in ([np.nan, 0., 1.], [-1., 1., 1.], [0., 0., 0.], [1., 0.]):
            with self.assertRaises((TypeError, ValueError)):
                extension.local_detail_loss(a, e, False, weights)
        with patch.dict(os.environ, DNGSCAN_FAST='1', DNGSCAN_FAST_SKIP='local_detail_loss'):
            self.assertEqual(local_detail_loss(a, e), _local_detail_loss_numpy(a, e))
        self.assertTrue(detail_is_acceptable({'score': .90}, 'score'))
        self.assertFalse(detail_is_acceptable({'score': .9001}, 'score'))
        for reference, limit in ((.1, .35), (.4, .47)):
            self.assertTrue(additional_error_acceptable(
                {'coding_local_detail_loss': limit}, {'coding_local_detail_loss': reference}))
            self.assertFalse(additional_error_acceptable(
                {'coding_local_detail_loss': limit+.001}, {'coding_local_detail_loss': reference}))

    def test_partial_tiles_strides_alpha_mixed_dtypes_and_thread_budgets(self):
        from dngscan.local_detail import _local_detail_loss_numpy
        rng = np.random.default_rng(54)
        extension = _fast._load_extension()
        self.addCleanup(extension.set_thread_budget, 0)
        for hdr in (False, True):
            for shape in ((1, 1, 3), (1, 17, 4), (7, 13, 3), (17, 31, 4), (129, 67, 3)):
                e = rng.uniform(.1, 2., shape)
                a = e * .98 + rng.uniform(-.005, .005, shape)
                if hdr:
                    types = ((np.float16, np.float16), (np.float32, np.float64),
                             (np.uint8, np.float64))
                else:
                    e, a = e*100, a*100
                    types = ((np.uint8, np.uint8),)
                for actual_type, expected_type in types:
                    aa, ee = a.astype(actual_type), e.astype(expected_type)
                    if hdr and shape[2] == 4 and actual_type != np.uint8:
                        aa[..., 3], ee[..., 3] = np.nan, np.inf
                    for select in (lambda x: x, lambda x: x[::-1, ::-1],
                                   lambda x: x.transpose(1, 0, 2)):
                        x, y = select(aa), select(ee)
                        x.flags.writeable = y.flags.writeable = False
                        before = (x.tobytes(), y.tobytes())
                        expected = _local_detail_loss_numpy(x, y, linear_hdr=hdr)
                        for workers in (1, 3):
                            extension.set_thread_budget(workers)
                            with patch.dict(os.environ, DNGSCAN_FAST='1', DNGSCAN_FAST_SKIP=''):
                                actual = local_detail_loss(x, y, linear_hdr=hdr)
                            self.assertAlmostEqual(actual, expected, delta=2e-7,
                                                   msg=(hdr, shape, actual_type, expected_type))
                        self.assertEqual(before, (x.tobytes(), y.tobytes()))

    def test_invalid_and_derived_overflow_are_rejected_by_both_backends(self):
        from dngscan.local_detail import _local_detail_loss_numpy
        e = np.full((129, 17, 3), .25, np.float32)
        cases = [(e, -e), (e.astype(np.float64), e.copy())]
        a = e.copy()
        a[-1, -1, 2] = np.nan
        cases.append((a, e))
        huge = np.full_like(e, np.finfo(np.float32).max)
        cases.extend(((huge, -huge), (huge, np.zeros_like(huge))))
        for x, y in cases:
            expected = _local_detail_loss_numpy(x, y, linear_hdr=True)
            with patch.dict(os.environ, DNGSCAN_FAST='1', DNGSCAN_FAST_SKIP=''):
                actual = local_detail_loss(x, y, linear_hdr=True)
            if math.isfinite(expected):
                self.assertAlmostEqual(actual, expected, delta=2e-7)
            else:
                self.assertEqual(actual, float('inf'))


if __name__ == '__main__':
    unittest.main()
