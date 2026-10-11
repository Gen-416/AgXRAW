# SPDX-License-Identifier: GPL-3.0-or-later
"""Encoded-primary gain-map references retain the independent delivery masters.

Platform I/O uses the existing staged fixture. These tests establish writer and
failure contracts; the small ratio probe is not a camera image-quality result.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dngscan import gainmap, heif_encoder
from dngscan.delivery import resolve_delivery_profile
from dngscan import heif_gainmap as heif
from tests.test_codec_repack import heif_fixture
from tests import test_gainmap_float_base as floating


class EncodedPrimaryReferenceTests(unittest.TestCase):
    @contextmanager
    def fixture(self, directory):
        """Keep the real writer, donor session, metrics and publication gates."""
        with floating.GainmapFloatBaseTests().fixture(directory) as fixture:
            fixture.profile = replace(fixture.profile, quality=99, chroma='420')
            fixture.overrides['chroma_subsampling'] = '4:2:0'
            fixture.out = directory / 'existing.heic'
            fixture.out.write_bytes(b'previous delivery')
            fixture.encoded_images, fixture.donors, fixture.sources, fixture.references = [], [], [], []

            class DonorImage:
                def __init__(self, path):
                    self.path = Path(path)

                def extent(self):
                    return SimpleNamespace(
                        origin=SimpleNamespace(x=0., y=0.),
                        size=SimpleNamespace(width=24., height=16.))

                def imageBySettingContentHeadroom_(self, value):
                    self.headroom = value
                    return self

            def decode_donor(path):
                self.assertTrue(Path(path).exists())
                return DonorImage(path)

            def write(image, path, _space, options, _error):
                self.assertEqual(fixture.out.read_bytes(), b'previous delivery')
                auxiliary = round(options['quality'] * 100)
                fixture.encoded_images.append((image, auxiliary))
                reference = (json.loads(image.path.read_text())['primary']
                             if isinstance(image, DonorImage) else 'master')
                fixture.references.append((reference, auxiliary))
                Path(path).write_text(json.dumps({
                    'primary': 'apple', 'aux': auxiliary,
                    'reference': reference}))
                return True, None

            def encode(base, path, quality, chroma, **_kwargs):
                self.assertEqual(fixture.out.read_bytes(), b'previous delivery')
                fixture.sources.append(base)
                fixture.donors.append(path)
                path.write_text(json.dumps({'primary': f'q{quality}-{chroma}'}))
                return {'encoder': 'x265'}

            fixture.context.writeHEIF10RepresentationOfImage_toURL_colorSpace_options_error_.side_effect = write
            fixture.quartz.CIImage = SimpleNamespace(imageWithContentsOfURL_=Mock(side_effect=decode_donor))
            fixture.donor_decode = fixture.quartz.CIImage.imageWithContentsOfURL_
            with patch.object(heif_encoder, 'encode', side_effect=encode) as encoder, \
                 patch.object(gainmap, '_decoded_primary_reference_image',
                              side_effect=lambda path, shape: decode_donor(path), create=True) as decoded, \
                 patch.object(gainmap, '_heif_primary_reference_identity',
                              side_effect=lambda path: json.loads(Path(path).read_text())['primary'],
                              create=True) as identity:
                fixture.encoder = encoder
                fixture.decoded_reference, fixture.reference_identity = decoded, identity
                yield fixture

    @staticmethod
    def write(fixture, **kwargs):
        return gainmap.write_apple_gainmap_file(
            fixture.float_base, fixture.hdr, fixture.out, 1.,
            delivery=fixture.profile, _verify_roundtrip_capability=False, **kwargs)

    def test_manual_ladder_precedes_one_reanchor_and_preserves_both_masters(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            original_base_gate = gainmap._base_and_coding_metrics_float
            original_hdr_gate = gainmap._hdr_roundtrip_is_acceptable

            def hdr_gate(metrics, tolerance):
                self.assertTrue(original_hdr_gate(metrics, tolerance))
                return hasattr(fixture.encoded_images[-1][0], 'path')

            with patch.object(gainmap, '_hdr_roundtrip_is_acceptable', side_effect=hdr_gate), \
                 patch.object(gainmap, '_base_and_coding_metrics_float',
                              wraps=original_base_gate) as base_gate, \
                 patch.object(gainmap, '_roundtrip_error', wraps=gainmap._roundtrip_error) as hdr_read:
                result = self.write(fixture)

            self.assertEqual([q for _, q in fixture.encoded_images], [95, 97, 98, 99, 100, 100])
            fixture.encoder.assert_called_once()
            fixture.decoded_reference.assert_called_once()
            self.assertEqual(result['gainmap_reference'], 'decoded-primary')
            self.assertEqual(result['gainmap_encoding_quality'], 100)
            self.assertEqual((result['delivery_quality'], result['delivery_chroma_requested']), (99, '420'))
            self.assertEqual(json.loads(fixture.out.read_text())['primary'], 'q99-420')
            self.assertEqual(json.loads(fixture.out.read_text())['reference'], 'q99-420')
            # Metrics retain the original immutable master, independently of
            # the image used to calculate the auxiliary's denominator.
            for call in base_gate.call_args_list:
                np.testing.assert_array_equal(call.args[1], fixture.float_base)
                self.assertIs(call.args[1], fixture.sources[0])
            for call in hdr_read.call_args_list:
                np.testing.assert_array_equal(call.args[1], fixture.hdr)
            self.assertEqual(hdr_read.call_count, 6)
            self.assertFalse(fixture.donors[0].exists())

    def test_explicit_auxiliary_quality_is_retained_by_reanchoring(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            with patch.object(gainmap, '_hdr_roundtrip_is_acceptable', side_effect=[False, True]):
                result = self.write(fixture, _gainmap_quality=97)
            self.assertEqual([q for _, q in fixture.encoded_images], [97, 97])
            self.assertEqual(result['gainmap_encoding_quality'], 97)
            fixture.encoder.assert_called_once()
            fixture.decoded_reference.assert_called_once()

    def test_sdr_or_container_rejection_cannot_reanchor(self):
        for failure in ('sdr', 'container'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as td, \
                 self.fixture(Path(td)) as fixture:
                if failure == 'sdr':
                    fixture.float_read.return_value = fixture.float_base + np.float32(.1)
                else:
                    fixture.overrides['width'] = 99
                with patch.object(gainmap, '_hdr_roundtrip_is_acceptable') as hdr_gate:
                    with self.assertRaises(RuntimeError):
                        self.write(fixture)
                hdr_gate.assert_not_called()
                fixture.decoded_reference.assert_not_called()
                self.assertEqual([q for _, q in fixture.encoded_images], [95])
                self.assertEqual(fixture.out.read_bytes(), b'previous delivery')

    def test_failed_reanchor_keeps_destination_and_releases_export_resources(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            closed = []
            close = gainmap._PreparedGainmapMaster.close

            def observe(master):
                close(master)
                closed.append(master)

            with patch.object(gainmap, '_hdr_roundtrip_is_acceptable', return_value=False), \
                 patch.object(gainmap._PreparedGainmapMaster, 'close', autospec=True,
                              side_effect=observe):
                with self.assertRaises(gainmap.HdrRoundtripError):
                    self.write(fixture, _gainmap_quality=100)
            self.assertEqual([q for _, q in fixture.encoded_images], [100, 100])
            self.assertEqual(fixture.out.read_bytes(), b'previous delivery')
            self.assertEqual(set(Path(td).iterdir()), {fixture.out, fixture.template})
            self.assertEqual(len(closed), 1)
            for field in ('base', 'hdr', 'base_image', 'hdr_image', 'context'):
                self.assertIsNone(getattr(closed[0], field))

    def test_auto_reanchors_only_quality100_and_never_shares_a_donor_template(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            fixture.profile = replace(fixture.profile, name='auto')
            latest = {}
            roundtrip = gainmap._roundtrip_error

            def read_hdr(path, hdr, **kwargs):
                latest.clear()
                latest.update(json.loads(Path(path).read_text()))
                return roundtrip(path, hdr, **kwargs)

            def replace_auxiliary(path, donor, _item):
                payload = json.loads(path.read_text())
                payload['aux'] = int(json.loads(donor.read_text())['primary'].split('-')[0][1:])
                path.write_text(json.dumps(payload))

            def search(path, encode, **_kwargs):
                candidate = path.with_name('.search-candidate.heic')
                first = encode(99, '420', 100, candidate)
                self.assertEqual(first['gainmap_reference'], 'decoded-primary')
                # The high-quality original-master template still describes
                # the original master. A lower auxiliary candidate must fail,
                # rather than silently switch to Apple q100 and report q95.
                with self.assertRaises(gainmap.HdrRoundtripError):
                    encode(97, '420', 95, candidate)
                fixture.overrides['chroma_subsampling'] = '4:2:2'
                result = encode(98, '422', 100, candidate)
                candidate.replace(path)
                return result

            with patch('dngscan.auto_encode.select_heif_encoding', side_effect=search), \
                 patch('dngscan.heif_gainmap.iso_gainmap_item', return_value=2), \
                 patch('dngscan.heif_gainmap.replace_image_item', side_effect=replace_auxiliary), \
                 patch.object(heif_encoder, 'read_rgb_item', return_value=np.zeros((16, 24, 3), np.uint8)), \
                 patch.object(gainmap, '_roundtrip_error', side_effect=read_hdr), \
                 patch.object(gainmap, '_hdr_roundtrip_is_acceptable',
                              side_effect=lambda *_args: latest['reference'] == latest['primary']):
                result = self.write(fixture)
            self.assertEqual(fixture.references, [('master', 100), ('q99-420', 100), ('q98-422', 100)])
            self.assertEqual(fixture.decoded_reference.call_count, 2)
            self.assertEqual(result['gainmap_encoding_quality'], 100)
            self.assertEqual(result['gainmap_encoder'], 'Apple ImageIO')
            self.assertEqual((result['delivery_quality'], result['delivery_chroma_requested']), (98, '422'))
            self.assertEqual(json.loads(fixture.out.read_text())['reference'], 'q98-422')
            # One encoder call per primary, plus one independent q95 auxiliary.
            self.assertEqual(fixture.encoder.call_count, 4)
            self.assertEqual(set(Path(td).iterdir()), {fixture.out, fixture.template})

    def test_non_x265_or_changed_installed_primary_cannot_publish_reanchor(self):
        for failure in ('encoder', 'installed-primary'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as td, \
                 self.fixture(Path(td)) as fixture:
                fixture.encoder.side_effect = None

                def encode(base, path, quality, chroma, **kwargs):
                    path.write_text(json.dumps({'primary': f'q{quality}-{chroma}'}))
                    fixture.donors.append(path)
                    return {'encoder': 'foreign codec' if failure == 'encoder' else 'x265'}

                fixture.encoder.side_effect = encode
                if failure == 'installed-primary':
                    def identity(path):
                        payload = json.loads(Path(path).read_text())
                        # The installed file still passes the separate geometry
                        # inspection seam; its primary identity does not match.
                        return payload['primary'] + ('changed' if payload.get('reference') == payload['primary'] else '')
                    fixture.reference_identity.side_effect = identity
                with patch.object(gainmap, '_hdr_roundtrip_is_acceptable', return_value=False):
                    with self.assertRaisesRegex(RuntimeError, 'HDR rendition|保留参考主图'):
                        self.write(fixture, _gainmap_quality=100)
                self.assertEqual(fixture.out.read_bytes(), b'previous delivery')
                self.assertEqual(fixture.decoded_reference.call_count, 0 if failure == 'encoder' else 1)
                self.assertEqual(set(Path(td).iterdir()), {fixture.out, fixture.template})

    def test_reanchor_scope_does_not_expand_other_deliveries(self):
        profile = replace(resolve_delivery_profile('share', container='heic'), heif_encoder='x265')
        prepared = SimpleNamespace(base=np.zeros((2, 2, 3), np.float32))
        session = object()
        eligible = gainmap._decoded_primary_reference_eligible
        self.assertTrue(eligible(prepared, profile, True, session, True))
        cases = (
            (SimpleNamespace(base=prepared.base.astype(np.uint8)), profile, True, session, True),
            (prepared, replace(profile, heif_bit_depth=8), True, session, True),
            (prepared, replace(profile, name='archive'), True, session, True),
            (prepared, replace(profile, heif_encoder='apple'), False, session, True),
            (prepared, replace(profile, container='jpeg'), True, session, True),
            (prepared, profile, True, None, True),
            (prepared, profile, True, session, False),
        )
        for case in cases:
            with self.subTest(case=case[1:]):
                self.assertFalse(eligible(*case))

    def test_primary_identity_covers_payload_and_decode_properties_not_the_gainmap(self):
        with tempfile.TemporaryDirectory() as td:
            target, donor = Path(td) / 'pair.heic', Path(td) / 'donor.heic'
            target.write_bytes(heif_fixture())
            donor.write_bytes(heif_fixture(hdr=False))
            top, children, primary, infos, refs, props, assocs, payloads = heif._parse(donor.read_bytes())
            props.extend([(b'colr', b'profDisplay-P3'), (b'pixi', bytes(4) + b'\3\12\12\12')])
            assocs[primary].extend([(True, len(props) - 1), (True, len(props))])
            heif._write_items(donor, children, infos, refs, props, assocs, payloads, top[0][1])
            before = heif._parse(target.read_bytes())
            heif.replace_primary(target, donor)
            installed = heif._parse(target.read_bytes())
            donor_identity = gainmap._heif_primary_reference_identity(donor)
            self.assertIsNotNone(donor_identity)
            self.assertEqual(gainmap._heif_primary_reference_identity(target), donor_identity)
            self.assertEqual(installed[7][installed[2]], payloads[primary])
            self.assertEqual(installed[7][2], before[7][2])
            self.assertEqual(installed[7][3], before[7][3])

            def alter(*, property_kind=None, primary_payload=False, auxiliary=False):
                t, c, p, i, r, pr, a, values = heif._parse(target.read_bytes())
                if property_kind is not None:
                    idx = next(index for _, index in a[p] if pr[index - 1][0] == property_kind)
                    kind, value = pr[idx - 1]
                    pr[idx - 1] = (kind, value + b'changed')
                if primary_payload:
                    values[p] += b'changed'
                if auxiliary:
                    values[2] += b'changed gainmap'
                    values[3] += b'changed ISO tone map'
                heif._write_items(target, c, i, r, pr, a, values, t[0][1])

            pristine = target.read_bytes()
            alter(auxiliary=True)
            self.assertEqual(gainmap._heif_primary_reference_identity(target), donor_identity)
            for change in ({'primary_payload': True}, {'property_kind': b'hvcC'},
                           {'property_kind': b'colr'}, {'property_kind': b'pixi'},
                           {'property_kind': b'ispe'}):
                target.write_bytes(pristine)
                alter(**change)
                with self.subTest(change=change):
                    self.assertNotEqual(gainmap._heif_primary_reference_identity(target), donor_identity)

    def test_ci_reference_is_sdr_url_decode_with_exact_extent_and_no_bitmap_clipping(self):
        decode = gainmap._decoded_primary_reference_image
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            fixture.quartz.kCIImageExpandToHDR = 'expand-hdr'
            sys.modules['Foundation'].NSDictionary = SimpleNamespace(
                dictionaryWithDictionary_=lambda values: dict(values))
            extent = SimpleNamespace(origin=SimpleNamespace(x=0., y=0.),
                                     size=SimpleNamespace(width=24., height=16.))
            image = SimpleNamespace(extent=lambda: extent)
            image.imageBySettingContentHeadroom_ = Mock(return_value=image)
            load = Mock(return_value=image)
            fixture.quartz.CIImage.imageWithContentsOfURL_options_ = load
            donor = Path(td) / 'encoded-donor.heic'
            self.assertIs(decode(donor, (16, 24, 3)), image)
            load.assert_called_once_with(str(donor), {'expand-hdr': False})
            image.imageBySettingContentHeadroom_.assert_called_once_with(1.)
            fixture.float_read.assert_not_called()
            fixture.image_builder.assert_not_called()
            # A shifted origin cannot be silently normalized or resized into
            # apparent agreement with the fixed formation masters.
            extent.origin.y = 1.
            with self.assertRaisesRegex(RuntimeError, '尺寸或原点'):
                decode(donor, (16, 24, 3))

    def test_ratio_uses_the_final_primary_without_redefining_the_hdr_master(self):
        base = np.full((64, 64, 3), .25, np.float32)
        hdr = np.ones((64, 64, 4), np.float16)
        checker = np.indices((8, 8)).sum(axis=0) % 2
        base[24:32, 24:32] = np.where(checker[..., None], .375, .125)
        hdr[24:32, 24:32, :3] = base[24:32, 24:32] * 4.
        primary = np.full_like(base, .25)
        before = (base.copy(), hdr.copy(), primary.copy())

        # Exact, uncompressed multiplication isolates which base the gain
        # ratio describes; it does not claim a real codec removes this patch.
        old = np.ones_like(hdr)
        old[..., :3] = primary * (hdr[..., :3].astype(np.float32) / base)
        reanchored = np.ones_like(hdr)
        reanchored[..., :3] = primary * (hdr[..., :3].astype(np.float32) / primary)
        tolerances = resolve_delivery_profile('share', container='heic').tolerances
        old_error = gainmap._roundtrip_error_arrays(old[..., :3], hdr[..., :3])
        new_error = gainmap._roundtrip_error_arrays(reanchored[..., :3], hdr[..., :3])
        self.assertEqual(old_error['local_detail_loss'], 1.)
        self.assertFalse(gainmap._hdr_roundtrip_is_acceptable(old_error, tolerances))
        self.assertEqual(new_error['local_detail_loss'], 0.)
        self.assertTrue(gainmap._hdr_roundtrip_is_acceptable(new_error, tolerances))
        for actual, expected in zip((base, hdr, primary), before):
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
