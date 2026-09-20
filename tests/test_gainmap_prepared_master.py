# SPDX-License-Identifier: GPL-3.0-or-later
"""One immutable master and CI owner set spans every candidate of one export."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from dngscan import gainmap
from tests import test_gainmap_staged_delivery as staged


class PreparedGainmapTests(unittest.TestCase):
    def fixture(self, directory):
        return staged.GainmapStagedWriterTests().writer_fixture(directory)

    @staticmethod
    def enable_template_write(fixture):
        def write(_image, path, _format, _space, options, _error):
            quality = options[fixture.quartz.kCGImageDestinationLossyCompressionQuality]
            Path(path).write_text(json.dumps({"aux": round(quality * 100)}))
            return True, None
        fixture.context.writeHEIFRepresentationOfImage_toURL_format_colorSpace_options_error_.side_effect = write

    def test_immutable_snapshots_identity_bounds_and_explicit_release(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            base, hdr = fixture.base[::-1, ::-1], fixture.hdr[::-1, ::-1]
            expected = (base.tobytes(), hdr.tobytes())
            prepared = gainmap._PreparedGainmapMaster(base, hdr, 1.,
                verify_capability=False, metrics_workspace=object())
            self.assertEqual((prepared.base.tobytes(), prepared.hdr.tobytes()), expected)
            self.assertEqual(prepared.actual_headroom, float(np.max(hdr[..., :3])))
            self.assertTrue(prepared.base.flags.c_contiguous and prepared.hdr.flags.c_contiguous)
            for array in (prepared.base, prepared.hdr, prepared.base_rgba):
                with self.assertRaises(ValueError):
                    array.flags.writeable = True
            base.fill(0)
            hdr.fill(4)
            self.assertEqual((prepared.base.tobytes(), prepared.hdr.tobytes()), expected)
            prepared.validate(prepared.base, prepared.hdr, 1.)
            for wrong_base, wrong_hdr, ev in ((prepared.base.copy(), prepared.hdr, 1.),
                    (prepared.base, prepared.hdr.view(), 1.), (prepared.base, prepared.hdr, 2.)):
                with self.assertRaises(ValueError):
                    prepared.validate(wrong_base, wrong_hdr, ev)
            del array, wrong_base, wrong_hdr
            source_refs = [weakref.ref(prepared.base), weakref.ref(prepared.hdr)]
            fixture.image_builder.reset_mock()
            prepared.close()
            gc.collect()
            self.assertTrue(all(ref() is None for ref in source_refs))
            with self.assertRaises(ValueError):
                prepared.validate(None, None, 1.)
            self.assertIsNone(prepared.context)
            self.assertIsNone(prepared.workspace)

    def test_auto_candidates_share_one_setup_but_keep_all_verification_gates(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            self.enable_template_write(fixture)
            context_factory = fixture.quartz.CIContext.contextWithOptions_
            seen = []

            def search(out, encode, **_kwargs):
                for quality in (95, 90):
                    info = encode(quality, '444', 100, out)
                    seen.append(info['delivery_quality'])
                return info

            with patch('dngscan.auto_encode.select_heif_encoding', side_effect=search), \
                 patch.object(fixture.quartz.CIContext, 'contextWithOptions_', wraps=context_factory) as contexts, \
                 patch.object(gainmap, '_new_hdr_metrics_workspace', return_value=None), \
                 patch.object(gainmap, 'write_apple_gainmap_file', wraps=gainmap.write_apple_gainmap_file) as public:
                info = gainmap.write_apple_gainmap_file(fixture.base, fixture.hdr,
                    Path(td) / 'auto.heic', 1., delivery=replace(fixture.profile, name='auto'),
                    _verify_roundtrip_capability=False)
            public.assert_called_once()
            contexts.assert_called_once()
            fixture.api_status.assert_called_once()
            self.assertEqual(fixture.image_builder.call_count, 2)
            self.assertEqual(fixture.inspector.call_count, 2)
            self.assertEqual(fixture.hdr_read.call_count, 2)
            self.assertEqual(fixture.absolute_gate.call_count, 2)
            fixture.context.writeHEIFRepresentationOfImage_toURL_format_colorSpace_options_error_.assert_called_once()
            self.assertEqual(seen, [95, 90])
            self.assertEqual(info['gainmap_encoding_quality'], 100)

    def test_manual_auxiliary_retry_is_nonrecursive_and_reuses_same_ci_master(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            self.enable_template_write(fixture)
            with patch.object(gainmap, '_hdr_roundtrip_is_acceptable', side_effect=[False, False, True]), \
                 patch.object(gainmap, '_new_hdr_metrics_workspace', return_value=None), \
                 patch.object(gainmap, 'write_apple_gainmap_file', wraps=gainmap.write_apple_gainmap_file) as public:
                info = gainmap.write_apple_gainmap_file(fixture.base, fixture.hdr,
                    Path(td) / 'manual.heic', 1., delivery=fixture.profile,
                    _verify_roundtrip_capability=False)
            public.assert_called_once()
            fixture.api_status.assert_called_once()
            self.assertEqual(fixture.image_builder.call_count, 2)
            self.assertEqual(fixture.inspector.call_count, 3)
            self.assertEqual(fixture.hdr_read.call_count, 3)
            options = [call.args[4] for call in
                fixture.context.writeHEIFRepresentationOfImage_toURL_format_colorSpace_options_error_.call_args_list]
            self.assertEqual([round(option['quality'] * 100) for option in options], [95, 97, 98])
            self.assertEqual(info['gainmap_encoding_quality'], 98)

    def test_failed_candidate_closes_prepared_resources_and_keeps_destination(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            out = Path(td) / 'existing.heic'
            out.write_bytes(b'previous')
            fixture.overrides['width'] = 99
            closed = []
            close = gainmap._PreparedGainmapMaster.close

            def observe(master):
                close(master)
                closed.append(master)

            with patch.object(gainmap._PreparedGainmapMaster, 'close', side_effect=observe, autospec=True):
                with self.assertRaisesRegex(RuntimeError, '尺寸'):
                    gainmap.write_apple_gainmap_file(fixture.base, fixture.hdr, out, 1.,
                        delivery=fixture.profile, _verify_roundtrip_capability=False,
                        _template_path=fixture.template, _gainmap_quality=100)
            self.assertEqual(len(closed), 1)
            self.assertTrue(closed[0]._closed)
            self.assertIsNone(closed[0].base)
            self.assertIsNone(closed[0].hdr)
            self.assertIsNone(closed[0].context)
            self.assertEqual(out.read_bytes(), b'previous')
            self.assertFalse(any('.tmp' in file.name for file in Path(td).iterdir()))

    def test_partial_prepare_failure_closes_created_owners(self):
        for stage in ('hdr_image', 'context'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
                closed = []
                close = gainmap._PreparedGainmapMaster.close

                def observe(master):
                    close(master)
                    closed.append(master)

                if stage == 'hdr_image':
                    fixture.image_builder.side_effect = [fixture.image_builder.return_value,
                                                         RuntimeError('second CIImage failed')]
                    failing = patch.object(gainmap, '_new_hdr_metrics_workspace', return_value=None)
                else:
                    failing = patch.object(fixture.quartz.CIContext, 'contextWithOptions_', return_value=None)
                with failing, patch.object(gainmap._PreparedGainmapMaster, 'close',
                        side_effect=observe, autospec=True):
                    with self.assertRaises(RuntimeError):
                        gainmap._PreparedGainmapMaster(fixture.base, fixture.hdr, 1., verify_capability=False)
                self.assertEqual(len(closed), 1)
                self.assertTrue(closed[0]._closed)
                for name in ('base', 'hdr', 'base_rgba', 'base_image', 'hdr_image', 'base_data', 'hdr_data', 'context'):
                    self.assertIsNone(getattr(closed[0], name), name)

    def test_banded_validation_includes_alpha_and_last_partial_band(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td)) as fixture:
            base = np.zeros((1025, 3, 3), np.uint8)
            hdr = np.full((1025, 3, 4), 1., np.float16)
            hdr[-1, -1, 1] = 4.
            prepared = gainmap._PreparedGainmapMaster(base, hdr, 2.,
                verify_capability=False, metrics_workspace=object())
            self.assertEqual(prepared.actual_headroom, 4.)
            prepared.close()
            fixture.api_status.reset_mock()
            hdr[-1, -1, 3] = np.nan
            with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
                gainmap._PreparedGainmapMaster(base, hdr, 2., verify_capability=False)
            fixture.api_status.assert_not_called()


if __name__ == '__main__':
    unittest.main()
