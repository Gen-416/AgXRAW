"""Late delivery failure cannot replace an existing user's file."""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dngscan import export, jpeg_gainmap
from dngscan.delivery import resolve_delivery_profile
from dngscan.delivery_transaction import DeliveryTransaction


class DeliveryTransactionTests(unittest.TestCase):
    def test_commit_and_failure_cleanup_share_destination_filesystem(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'photo.jpg'
            destination.write_bytes(b'old')
            with self.assertRaisesRegex(RuntimeError, 'late check'):
                with DeliveryTransaction(destination) as transaction:
                    self.assertEqual(transaction.path.parent.parent, destination.parent)
                    transaction.path.write_bytes(b'new')
                    raise RuntimeError('late check')
            self.assertEqual(destination.read_bytes(), b'old')
            self.assertEqual(list(Path(directory).iterdir()), [destination])
            with DeliveryTransaction(destination) as transaction:
                transaction.path.write_bytes(b'new')
                transaction.commit()
            self.assertEqual(destination.read_bytes(), b'new')
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_sdr_auto_final_readback_failure_preserves_previous_delivery(self):
        rgb = np.zeros((2, 3, 3), np.uint8)
        profile = resolve_delivery_profile('auto')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'photo.jpg'
            target.write_bytes(b'old')

            def selected(path, *_args, **_kwargs):
                path.write_bytes(b'private winner')
                return {'auto_saved_pct': 17.0}

            with mock.patch.object(export, 'render_output_u8', return_value=rgb), \
                 mock.patch('dngscan.auto_encode.select_encoding', side_effect=selected), \
                 mock.patch.object(export, 'carry_capture_metadata', return_value=False), \
                 mock.patch('PIL.Image.open', side_effect=OSError('final readback')):
                with self.assertRaisesRegex(RuntimeError, 'final readback'):
                    export.export_srgb_jpeg(Path('capture.dng'), target, 99, None, None,
                                            delivery=profile, return_rgb=True)
            self.assertEqual(target.read_bytes(), b'old')
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_sdr_auto_metadata_best_effort_does_not_reselect_or_change_savings(self):
        profile = resolve_delivery_profile('auto')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'photo.jpg'

            def selected(path, *_args, **_kwargs):
                path.write_bytes(b'verified')
                return {'auto_saved_pct': 17., 'file_size_bytes': 8}

            with mock.patch.object(export, 'render_output_u8', return_value=np.zeros((2, 3, 3), np.uint8)), \
                 mock.patch('dngscan.auto_encode.select_encoding', side_effect=selected) as select, \
                 mock.patch.object(export, 'carry_capture_metadata', return_value=False):
                info = export.export_srgb_jpeg(Path('capture.dng'), target, 99, None, None,
                                              delivery=profile)
            select.assert_called_once()
            self.assertEqual(info['auto_saved_pct'], 17.)
            self.assertFalse(info['exif_carried'])
            self.assertEqual(info['file_size_bytes'], target.stat().st_size)
            self.assertEqual(info['output_path'], str(target))
            self.assertEqual(target.read_bytes(), b'verified')

    def test_hdr_readback_is_private_until_success_and_metadata_remains_best_effort(self):
        from tests.test_hdr_native import _scene_plan
        plan = _scene_plan()
        tone = SimpleNamespace(rendered_headroom_ev=2., peak_linear=4., display_headroom_ev=2.,
                               requested_headroom_ev=2., reliable_tail_ev=3., shoulder_start_ev=1.,
                               white_ev=2., shoulder_alpha=.5, shoulder_segments=())
        hdr_plan = SimpleNamespace(tone=tone, color=SimpleNamespace(channel_separation=.5))
        base = np.zeros((2, 3, 3), np.uint8)
        hdr = np.full((2, 3, 3), 2., np.float32)
        for fail in (True, False):
            with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                target = Path(directory) / 'photo.jpg'
                target.write_bytes(b'old')

                def encode(_pair, path, _profile):
                    self.assertNotEqual(path, target)
                    self.assertEqual(target.read_bytes(), b'old')
                    path.write_bytes(b'verified HDR')
                    return {'auto_saved_pct': 5.}

                def read(path, _gamut):
                    self.assertEqual(target.read_bytes(), b'old')
                    self.assertEqual(path.read_bytes(), b'verified HDR')
                    if fail:
                        raise RuntimeError('final HDR readback')
                    return base

                stack.enter_context(mock.patch.object(export, 'apple_gainmap_backend_status', return_value=(True, '')))
                stack.enter_context(mock.patch('dngscan.hdr_agx_plan.compile_hdr_agx_plan', return_value=hdr_plan))
                stack.enter_context(mock.patch('dngscan.hdr_agx_plan.describe_hdr_plan', return_value='test plan'))
                stack.enter_context(mock.patch('dngscan.hdr_agx.render_ultrahdr_agx_pair_packed',
                    return_value=(base, np.ones((2, 3, 4), np.float16), 1.)))
                stack.enter_context(mock.patch.object(export, 'encode_finished_pair', side_effect=encode))
                stack.enter_context(mock.patch.object(export, 'carry_capture_metadata_hdr', return_value=False))
                stack.enter_context(mock.patch('dngscan.gainmap.read_primary_rgb_u8', side_effect=read))
                invoke = lambda: export.export_ultrahdr_jpeg(
                    Path('capture.dng'), target, 99, SimpleNamespace(scene_decoder='test'), None,
                    tone_plan=plan, return_rgb=True)
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'final HDR readback'):
                        invoke()
                    self.assertEqual(target.read_bytes(), b'old')
                else:
                    info = invoke()
                    self.assertIs(info['_decoded_rgb'], base)
                    self.assertFalse(info['exif_carried'])
                    self.assertEqual(info['file_size_bytes'], target.stat().st_size)
                    self.assertEqual(target.read_bytes(), b'verified HDR')
                self.assertEqual(list(Path(directory).iterdir()), [target])


class JpegPrimarySessionTests(unittest.TestCase):
    def test_retry_reuses_codestream_and_repack_stays_byte_exact(self):
        from tests.test_codec_repack import jpeg_template
        rgb = np.random.default_rng(20).integers(0, 256, (19, 23, 3), dtype=np.uint8)
        rgb.flags.writeable = False
        session = jpeg_gainmap.PrimaryCodestreamSession(rgb)
        with mock.patch.object(jpeg_gainmap, 'encode_primary_codestream', wraps=jpeg_gainmap.encode_primary_codestream) as encode:
            a = session.primary(rgb, 99, '422')
            self.assertIs(session.primary(rgb, 99, '422'), a)
            self.assertIsNot(session.primary(rgb, 98, '422'), a)
            self.assertEqual(encode.call_count, 2)
        template, _aux = jpeg_template(rgb)
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / 'old.jpg', Path(directory) / 'new.jpg'
            old.write_bytes(template)
            new.write_bytes(template)
            jpeg_gainmap.replace_primary(old, rgb, 99, '422')
            jpeg_gainmap.replace_primary_codestream(new, a)
            self.assertEqual(new.read_bytes(), old.read_bytes())

    def test_session_rejects_mutable_or_changed_master(self):
        source = np.zeros((2, 3, 3), np.uint8)
        with self.assertRaises(ValueError):
            jpeg_gainmap.PrimaryCodestreamSession(source)
        source.flags.writeable = False
        session = jpeg_gainmap.PrimaryCodestreamSession(source)
        with self.assertRaises(ValueError):
            session.primary(source.copy(), 99, '422')
