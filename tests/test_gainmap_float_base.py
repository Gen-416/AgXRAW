# SPDX-License-Identifier: GPL-3.0-or-later
"""A 10-bit HDR HEIF template, donor and gates share one floating SDR master."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dngscan import gainmap, heif_encoder
from dngscan.delivery import resolve_delivery_profile
from dngscan.gainmap_session import PrimarySearchSession
from tests import test_gainmap_staged_delivery as staged


class GainmapFloatBaseTests(unittest.TestCase):
    @contextmanager
    def fixture(self, directory):
        with staged.GainmapStagedWriterTests().writer_fixture(directory) as fixture:
            fixture.quartz.kCIFormatRGBAf = 'rgbaf'
            fixture.quartz.kCIContextWorkingFormat = 'working-format'
            def write(_image, path, _space, options, _error):
                Path(path).write_text(json.dumps({
                    'primary': 'native-ten-bit',
                    'aux': round(options[fixture.quartz.kCGImageDestinationLossyCompressionQuality] * 100)}))
                return True, None
            fixture.context.writeHEIF10RepresentationOfImage_toURL_colorSpace_options_error_ = Mock(side_effect=write)
            fixture.quartz.CIContext.writeHEIF10RepresentationOfImage_toURL_colorSpace_options_error_ = write
            fixture.float_base = (fixture.base.astype(np.float32) + np.float32(.25)) / np.float32(255.)
            with patch.object(gainmap, 'read_primary_rgb_float', return_value=fixture.float_base.copy()) as read:
                fixture.float_read = read
                yield fixture

    def test_prepared_float_rgba_owner_preserves_fractional_codes_without_aliases(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            source = fixture.float_base[::-1, ::-1]
            expected = source.copy()
            contexts = fixture.quartz.CIContext.contextWithOptions_
            with patch.object(fixture.quartz.CIContext, 'contextWithOptions_', wraps=contexts) as context:
                prepared = gainmap._PreparedGainmapMaster(source, fixture.hdr, 1., verify_capability=False)
            np.testing.assert_array_equal(prepared.base, expected)
            self.assertEqual(prepared.base.dtype, np.float32)
            self.assertFalse(prepared.base.flags.c_contiguous)
            self.assertTrue(np.shares_memory(prepared.base, prepared.base_rgba))
            self.assertEqual(prepared.base.strides[-2:], (16, 4))
            self.assertTrue(np.all(prepared.base_rgba[..., 3] == 1.))
            self.assertEqual(fixture.image_builder.call_args_list[0].args[1], 'rgbaf')
            self.assertEqual(context.call_args.args[0]['working-format'], 'rgbaf')
            for owner in (prepared.base, prepared.base_rgba):
                with self.assertRaises(ValueError):
                    owner.flags.writeable = True
            source.fill(0)
            np.testing.assert_array_equal(prepared.base, expected)
            prepared.validate(prepared.base, prepared.hdr, 1.)
            prepared.close()
            self.assertIsNone(prepared.base)
            self.assertIsNone(prepared.base_rgba)

    def test_manual_float_template_donor_and_verification_never_use_uint8(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            path = Path(td) / 'float.heic'
            with patch('dngscan.heif_encoder.encode', wraps=heif_encoder.encode) as encode:
                info = gainmap.write_apple_gainmap_file(fixture.float_base, fixture.hdr, path, 1.,
                    delivery=fixture.profile, _verify_roundtrip_capability=False, _gainmap_quality=100)
            native = fixture.context.writeHEIF10RepresentationOfImage_toURL_colorSpace_options_error_
            native.assert_called_once()
            fixture.context.writeHEIFRepresentationOfImage_toURL_format_colorSpace_options_error_.assert_not_called()
            self.assertEqual(native.call_args.args[3]['request-options']['base-format'], int.from_bytes(b'xf44', 'big'))
            self.assertEqual(encode.call_count, 1)
            coded = encode.call_args.args[0]
            self.assertEqual(coded.dtype, np.float32)
            np.testing.assert_array_equal(coded, fixture.float_base)
            self.assertFalse(coded.flags.writeable)
            self.assertTrue(encode.call_args.kwargs['dither_quantization'])
            fixture.read.assert_not_called()
            fixture.float_read.assert_called_once()
            fixture.hdr_read.assert_called_once()
            self.assertEqual(info['readback_precision'], 'float32')
            self.assertEqual(info['sdr_master_precision'], 'float32')
            self.assertEqual(info['base_mean_code_error'], 0.)
            self.assertEqual(info['coding_luma_rmse'], 0.)

    def test_auto_templates_and_donors_reuse_the_same_immutable_float_master(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            sources = []
            def encode(source, path, quality, chroma, **kwargs):
                sources.append(source)
                self.assertTrue(kwargs['dither_quantization'])
                path.write_text(json.dumps({'primary': f'q{quality}-{chroma}'}))
                return {'encoder': 'fake x265'}
            def search(path, encode, **kwargs):
                first = encode(95, '444', 100, path)
                second = encode(90, '444', 100, path)
                self.assertEqual(first['readback_precision'], 'float32')
                return second
            with patch('dngscan.heif_encoder.encode', side_effect=encode), \
                 patch('dngscan.auto_encode.select_heif_encoding', side_effect=search):
                info = gainmap.write_apple_gainmap_file(fixture.float_base, fixture.hdr,
                    Path(td) / 'auto.heic', 1., delivery=replace(fixture.profile, name='auto'),
                    _verify_roundtrip_capability=False)
            self.assertEqual(info['delivery_quality'], 90)
            self.assertEqual(len(sources), 2)
            self.assertIs(sources[0], sources[1])
            np.testing.assert_array_equal(sources[0], fixture.float_base)
            fixture.context.writeHEIF10RepresentationOfImage_toURL_colorSpace_options_error_.assert_called_once()
            self.assertEqual(fixture.image_builder.call_count, 2)
            self.assertEqual(fixture.float_read.call_count, 2)
            fixture.read.assert_not_called()
            self.assertEqual(fixture.hdr_read.call_count, 2)

    def test_float_primary_session_reuses_tpdf_donor_and_rejects_8bit_profile(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = PrimarySearchSession(root)
            base = np.frombuffer(np.full((8, 8, 3), .5, np.float32).tobytes(), np.float32).reshape(8, 8, 3)
            profile = replace(resolve_delivery_profile('share', container='heic'), heif_encoder='x265')
            def encode(source, path, *_args, **kwargs):
                self.assertIs(source, base)
                self.assertTrue(kwargs['dither_quantization'])
                path.write_bytes(b'donor')
                return {'bit_depth': 10}
            with patch('dngscan.heif_encoder.encode', side_effect=encode) as encoder:
                first, _ = session.primary(base, profile)
                again, _ = session.primary(base, profile)
                self.assertEqual(first, again)
                encoder.assert_called_once()
                with self.assertRaisesRegex(ValueError, '10-bit'):
                    session.primary(base, replace(profile, heif_bit_depth=8))
                self.assertEqual(list(root.iterdir()), [first])

    def test_apple_only_auto_keeps_requested_420_and_existing_quality_search(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            fixture.overrides['chroma_subsampling'] = '4:2:0'
            seen = []
            def search(path, encode, *, qualities):
                self.assertEqual(tuple(qualities), (99, 98, 97, 96, 95))
                for quality in (99, 95):
                    info = encode(quality, path)
                    seen.append((info['delivery_quality'], info['delivery_chroma_requested']))
                return info
            profile = replace(fixture.profile, name='auto', chroma='420', heif_encoder='apple')
            with patch('dngscan.auto_encode.select_encoding', side_effect=search), \
                 patch('dngscan.heif_encoder.encode', side_effect=AssertionError('x265 unavailable')):
                info = gainmap.write_apple_gainmap_file(fixture.float_base, fixture.hdr,
                    Path(td) / 'apple-auto.heic', 1., delivery=profile,
                    _verify_roundtrip_capability=False)
            self.assertEqual(seen, [(99, '420'), (95, '420')])
            self.assertEqual(info['delivery_chroma_requested'], '420')
            self.assertEqual(fixture.float_read.call_count, 2)
            fixture.read.assert_not_called()

    def test_apple_only_manual_420_does_not_silently_accept_native_444(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            path = Path(td) / 'existing.heic'
            path.write_bytes(b'previous')
            profile = replace(fixture.profile, quality=100, chroma='420', heif_encoder='apple')
            with self.assertRaisesRegex(RuntimeError, '采样与请求不符'):
                gainmap.write_apple_gainmap_file(fixture.float_base, fixture.hdr,
                    path, 1., delivery=profile, _verify_roundtrip_capability=False)
            self.assertEqual(path.read_bytes(), b'previous')
            fixture.float_read.assert_not_called()
            fixture.hdr_read.assert_not_called()

    def test_fractional_metrics_are_cached_without_uint8_readback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = staged._SessionSeam(root)
            first, variant = root / 'first.heic', root / 'variant.heic'
            for path, aux in ((first, 100), (variant, 95)):
                path.write_text(json.dumps({'primary': 'same', 'aux': aux}))
            intended = np.full((16, 24, 3), .5, np.float32)
            decoded = intended + np.float32(.25 / 255.)
            with patch.object(gainmap, 'read_primary_rgb_float', return_value=decoded) as read, \
                 patch.object(gainmap, 'read_primary_rgb_u8', side_effect=AssertionError('u8 read')):
                metrics = gainmap._search_sdr_metrics(first, intended, session)
                reused = gainmap._search_sdr_metrics(variant, intended, session)
            read.assert_called_once()
            self.assertEqual(metrics, reused)
            self.assertAlmostEqual(metrics['base_mean_code_error'], .25, places=5)

    def test_invalid_float_domain_and_legacy_targets_fail_before_system_setup(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            for value in (np.nan, -.001, 1.001):
                base = fixture.float_base.copy()
                base[-1, -1, 2] = value
                with self.subTest(value=value), self.assertRaises(ValueError):
                    gainmap.write_apple_gainmap_file(base, fixture.hdr, Path(td) / 'invalid.heic',
                        1., delivery=fixture.profile, _verify_roundtrip_capability=False)
            for profile in (replace(fixture.profile, container='jpeg'), replace(fixture.profile, heif_bit_depth=8)):
                with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, '10-bit HEIF'):
                    gainmap.write_apple_gainmap_file(fixture.float_base, fixture.hdr,
                        Path(td) / 'invalid', 1., delivery=profile, _verify_roundtrip_capability=False)
            fixture.api_status.assert_not_called()

    def test_missing_tenbit_api_and_eightbit_templates_fail_without_publication(self):
        for mode in ('missing-api', 'new-eightbit-template', 'reused-eightbit-template'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
                out = Path(td) / 'existing.heic'
                out.write_bytes(b'previous')
                kwargs = {}
                if mode == 'missing-api':
                    del fixture.quartz.CIContext.writeHEIF10RepresentationOfImage_toURL_colorSpace_options_error_
                else:
                    fixture.overrides['bit_depth'] = 8
                    if mode == 'reused-eightbit-template':
                        fixture.template.write_text(json.dumps({'primary': 'old-eight-bit', 'aux': 100}))
                        kwargs['_template_path'] = fixture.template
                with self.assertRaisesRegex(RuntimeError, '10-bit|8-bit'):
                    gainmap.write_apple_gainmap_file(fixture.float_base, fixture.hdr, out,
                        1., delivery=fixture.profile, _verify_roundtrip_capability=False, **kwargs)
                self.assertEqual(out.read_bytes(), b'previous')
                self.assertFalse(any('.tmp' in file.name for file in Path(td).iterdir()))
                fixture.float_read.assert_not_called()
                fixture.hdr_read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
