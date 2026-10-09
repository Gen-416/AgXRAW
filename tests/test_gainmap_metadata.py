# SPDX-License-Identifier: GPL-3.0-or-later
"""ImageIO's explicit ISO declarations, independent of decoded pixel peaks."""
from __future__ import annotations

import math
from contextlib import nullcontext
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from dngscan.gainmap import _ISO_HEADROOM_NAMESPACE, inspect_gainmap_file


class GainMapMetadataTests(unittest.TestCase):
    def inspect(self, suffix='.jpg', *, fields=None, details=None, primary=None,
                listed=True, auxiliary=True, namespace=_ISO_HEADROOM_NAMESPACE):
        tags = [(namespace, key, value) for key, value in (fields or {}).items()]
        properties = {'ProfileName': 'Display P3', 'PixelWidth': 32,
                      'PixelHeight': 16, 'Depth': 8, **(primary or {})}
        desc = {'PixelFormat': int.from_bytes(b'444f', 'big'), 'Width': 32, 'Height': 16}
        first = {'ChromaSubsampling': '4:4:4'}
        if listed:
            first['AuxiliaryData'] = [{'AuxiliaryDataType': 'iso', **desc}]
        if details is not None:
            first['DerivationDetails'] = details
        info = {'Metadata': tags, 'Description': desc} if auxiliary else None
        self.auxiliary_calls = 0
        def read_auxiliary(*args):
            self.auxiliary_calls += 1
            return info
        quartz = types.SimpleNamespace(
            kCGImagePropertyFileContentsDictionary='FileContents',
            kCGImagePropertyAuxiliaryData='AuxiliaryData',
            kCGImageAuxiliaryDataTypeISOGainMap='iso',
            kCGImageAuxiliaryDataInfoMetadata='Metadata',
            kCGImageAuxiliaryDataInfoDataDescription='Description',
            kCGImagePropertyProfileName='ProfileName',
            kCGImagePropertyPixelWidth='PixelWidth',
            kCGImagePropertyPixelHeight='PixelHeight',
            kCGImagePropertyDepth='Depth',
            CGImageSourceCreateWithURL=lambda *args: object(),
            CGImageSourceCopyPropertiesAtIndex=lambda *args: properties,
            CGImageSourceCopyProperties=lambda *args: {'FileContents': {'Images': [first]}},
            CGImageSourceCopyAuxiliaryDataInfoAtIndex=read_auxiliary,
            CGImageMetadataCopyTags=lambda metadata: metadata,
            CGImageMetadataTagCopyNamespace=lambda tag: tag[0],
            CGImageMetadataTagCopyName=lambda tag: tag[1],
            CGImageMetadataTagCopyValue=lambda tag: tag[2],
        )
        foundation = types.SimpleNamespace(NSURL=types.SimpleNamespace(fileURLWithPath_=lambda x: x))
        objc = types.SimpleNamespace(autorelease_pool=nullcontext)
        with mock.patch.dict(sys.modules, {'Quartz': quartz, 'Foundation': foundation, 'objc': objc}):
            return inspect_gainmap_file(Path('encoded' + suffix))

    def test_iso_auxiliary_explicit_stops_are_primary_for_jpeg_and_heic(self):
        for suffix in ('.jpg', '.heic'):
            with self.subTest(suffix=suffix):
                result = self.inspect(suffix, fields={'Version': '1', 'BaseHeadroom': '0.000000',
                                                     'AlternateHeadroom': '1.000000'})
                self.assertEqual(result['headroom'], 2.)
                self.assertEqual(result['base_headroom'], 1.)
                self.assertEqual(result['headroom_source'], 'iso-auxiliary')
                self.assertEqual(result['headroom_status'], 'declared')
                self.assertTrue(result['has_iso_gainmap'])

    def test_iso_auxiliary_api_can_prove_map_without_file_contents_listing(self):
        result = self.inspect(listed=False, fields={'BaseHeadroom': '0', 'AlternateHeadroom': '2'})
        self.assertTrue(result['has_iso_gainmap'])
        self.assertEqual(result['gainmap_width'], 32)
        self.assertEqual(result['gainmap_pixel_format'], '444f')
        self.assertEqual(result['headroom'], 4.)

    def test_linear_derivation_is_a_declaration_when_auxiliary_api_metadata_is_absent(self):
        result = self.inspect(fields={}, auxiliary=False, details=[
            {'TonemapBaseHDRHeadroom': 1., 'TonemapAlternateHDRHeadroom': 4.}])
        self.assertEqual(result['headroom'], 4.)
        self.assertEqual(result['headroom_source'], 'iso-derivation')
        self.assertEqual(result['base_headroom'], 1.)

    def test_consistent_declarations_allow_only_metadata_representation_rounding(self):
        result = self.inspect(fields={'BaseHeadroom': '0', 'AlternateHeadroom': '2.349834'},
                              details=[{'TonemapBaseHDRHeadroom': 1.,
                                        'TonemapAlternateHDRHeadroom': 5.097656726837158}],
                              primary={'Headroom': 5.097656726837158})
        self.assertAlmostEqual(result['headroom'], 5.097656726837158, delta=2e-6)
        self.assertEqual(result['headroom_source'], 'iso-derivation')
        self.assertEqual(self.auxiliary_calls, 0)

    def test_explicit_derivation_does_not_materialize_auxiliary_pixels_per_candidate(self):
        for suffix in ('.jpg', '.heic'):
            with self.subTest(suffix=suffix):
                result = self.inspect(suffix, details=[
                    {'TonemapBaseHDRHeadroom': 1., 'TonemapAlternateHDRHeadroom': 2.}])
                self.assertEqual(result['headroom'], 2.)
                self.assertEqual(result['headroom_source'], 'iso-derivation')
                self.assertEqual(self.auxiliary_calls, 0)

    def test_legacy_primary_headroom_is_preserved_when_iso_declarations_are_unavailable(self):
        result = self.inspect(auxiliary=False, primary={'Headroom': 2.})
        self.assertEqual(result['headroom'], 2.)
        self.assertIsNone(result['base_headroom'])
        self.assertEqual(result['headroom_source'], 'legacy-primary')

    def test_ordinary_sdr_is_distinct_from_missing_iso_headroom(self):
        result = self.inspect(listed=False, auxiliary=False)
        self.assertEqual(result['headroom'], 1.)
        self.assertEqual(result['headroom_status'], 'not-declared')
        self.assertEqual(result['headroom_source'], 'sdr-no-gainmap')
        self.assertFalse(result['has_iso_gainmap'])
        with self.assertRaisesRegex(RuntimeError, '缺少明确'):
            self.inspect()

    def test_wrong_namespace_is_not_an_iso_declaration(self):
        with self.assertRaisesRegex(RuntimeError, '缺少明确'):
            self.inspect(fields={'BaseHeadroom': '0', 'AlternateHeadroom': '1'}, namespace='unknown')

    def test_partial_unknown_and_malformed_iso_metadata_cannot_fall_back_to_legacy(self):
        for fields in (
            {'BaseHeadroom': '0'}, {'AlternateHeadroom': '1'},
            {'BaseHeadroom': '0', 'AlternateHeadroom': 'unknown'},
            {'BaseHeadroom': '0', 'AlternateHeadroom': None},
            {'BaseHeadroom': '0', 'AlternateHeadroom': True},
            {'BaseHeadroom': '0', 'AlternateHeadroom': 'nan'},
            {'BaseHeadroom': '0', 'AlternateHeadroom': 'inf'},
            {'BaseHeadroom': '0', 'AlternateHeadroom': '-1'},
            {'BaseHeadroom': '0', 'AlternateHeadroom': '1024'},
            {'Version': '2', 'BaseHeadroom': '0', 'AlternateHeadroom': '1'},
        ):
            with self.subTest(fields=fields), self.assertRaises(RuntimeError):
                self.inspect(fields=fields, primary={'Headroom': 2.})

    def test_base_must_be_sdr_and_alternate_must_be_hdr(self):
        for base, alternate in (('1', '2'), ('0', '0'), ('-1', '1'), ('nan', '1')):
            with self.subTest(base=base, alternate=alternate), self.assertRaises(RuntimeError):
                self.inspect(fields={'BaseHeadroom': base, 'AlternateHeadroom': alternate})

    def test_malformed_linear_derivation_cannot_fall_back_to_legacy(self):
        for detail in (
            {'TonemapBaseHDRHeadroom': 1.},
            {'TonemapAlternateHDRHeadroom': 2.},
            {'TonemapBaseHDRHeadroom': 1., 'TonemapAlternateHDRHeadroom': math.nan},
            {'TonemapBaseHDRHeadroom': 1., 'TonemapAlternateHDRHeadroom': math.inf},
            {'TonemapBaseHDRHeadroom': 1., 'TonemapAlternateHDRHeadroom': 'unknown'},
            {'TonemapBaseHDRHeadroom': 2., 'TonemapAlternateHDRHeadroom': 4.},
            {'TonemapBaseHDRHeadroom': 1., 'TonemapAlternateHDRHeadroom': 1.},
        ):
            with self.subTest(detail=detail), self.assertRaises(RuntimeError):
                self.inspect(details=[detail], primary={'Headroom': 2.})

    def test_conflicting_primary_or_derivation_headroom_is_rejected(self):
        fields = {'BaseHeadroom': '0', 'AlternateHeadroom': '1'}
        for primary, details in (
            ({'Headroom': 1.}, None), ({'Headroom': 4.}, None),
            ({'Headroom': 4.}, [{'TonemapBaseHDRHeadroom': 1., 'TonemapAlternateHDRHeadroom': 2.}]),
            ({}, [{'TonemapBaseHDRHeadroom': 1., 'TonemapAlternateHDRHeadroom': 2.},
                  {'TonemapBaseHDRHeadroom': 1., 'TonemapAlternateHDRHeadroom': 4.}]),
        ):
            with self.subTest(primary=primary, details=details), self.assertRaises(RuntimeError):
                self.inspect(fields=fields, primary=primary, details=details)

    def test_invalid_legacy_headroom_is_not_defaulted_to_sdr(self):
        for value in (None, False, math.nan, math.inf, 0., -1., 'unknown'):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.inspect(auxiliary=False, primary={'Headroom': value})


if __name__ == '__main__':
    unittest.main()
