# SPDX-License-Identifier: GPL-3.0-or-later
"""Runtime capability degradation must remain usable and truthful end to end."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import coreimage_decode as ci
from dngscan.analysis import analyze
from dngscan.evidence import EvidenceAcquisitionError
from dngscan.raw_io import load_raw
from dngscan.report import summary_lines, csv_row
from dngscan.gui import preview_cache as pc


class RuntimeLadderTests(unittest.TestCase):
    def _decode(self, version, failures):
        opened = []
        def open_filter(path):
            obj = {}
            opened.append(obj)
            return obj
        def configure(obj, **kw):
            obj.update(kw)
            return dict(kw, color_noise_reduction_amount=0, color_noise_cleared=True)
        def render(obj, **kw):
            if obj['version'] in failures:
                raise RuntimeError('injected render failure')
            return np.ones((24, 32, 3), dtype=np.float16)
        with ExitStack() as stack:
            stack.enter_context(patch.object(ci, 'supported_versions', return_value=('9', '8', '7')))
            stack.enter_context(patch.object(ci, '_open_filter', side_effect=open_filter))
            stack.enter_context(patch.object(ci, 'configure_linear_filter', side_effect=configure))
            stack.enter_context(patch.object(ci, '_render_linear_rec2020', side_effect=render))
            stack.enter_context(patch.object(ci, 'decoder_runtime_id', return_value='test'))
            rgb, info = ci.decode_scene_rec2020(Path('test.dng'), half_size=False, version=version)
        return opened, rgb, info

    def test_render_failure_retries_fresh_older_decoder(self):
        opened, rgb, info = self._decode('auto', {'9'})
        self.assertEqual([o['version'] for o in opened], ['9', '8'])
        self.assertIsNot(opened[0], opened[1])
        self.assertEqual(info['version'], '8')
        self.assertIn('RAW 9', info['fallback_errors'][0])
        self.assertEqual(rgb.shape, (24, 32, 3))

    def test_explicit_version_does_not_fallback(self):
        with self.assertRaisesRegex(RuntimeError, 'injected render failure'):
            self._decode('9', {'9'})

    def test_exact_internal_token_is_not_reselected_as_bare_version(self):
        self.assertEqual(ci.resolve_decoder_version('9.dng', ('9', '9.dng', '8')), '9.dng')
        with self.assertRaisesRegex(RuntimeError, 'not offered'):
            ci.resolve_decoder_version('9.dng', ('9', '8'))

    def test_all_versions_fail_with_attempt_details(self):
        with self.assertRaisesRegex(RuntimeError, 'RAW 9.*RAW 8.*RAW 7'):
            self._decode('auto', {'9', '8', '7'})


class AppleOnlyContractTests(unittest.TestCase):
    def test_export_pins_auto_preview_decoder_and_refuses_changed_evidence(self):
        from tests.test_preview_cache import _bundle
        from dngscan.gui.service import _load_export_scene
        preview = replace(_bundle(), scene_decoder='coreimage', scene_decoder_version='9.dng',
                          scene_decoder_fallback='RAW 9 failed', scene_reliability_source='sensor-reference')
        with patch('dngscan.gui.service.dg.load_raw', return_value=replace(preview)) as decode:
            result = _load_export_scene(preview.path, 'reconstruct', 'camera', 'coreimage', 'auto', 'auto', 'aligned', preview)
            self.assertEqual(decode.call_args.kwargs['coreimage_version'], '9.dng')
            self.assertEqual(result.scene_decoder_fallback, preview.scene_decoder_fallback)
        changed = replace(preview, scene_reliability_source='decoded-image-estimate')
        with patch('dngscan.gui.service.dg.load_raw', return_value=changed):
            with self.assertRaisesRegex(RuntimeError, '刷新预览'):
                _load_export_scene(preview.path, 'reconstruct', 'camera', 'coreimage', 'auto', 'auto', 'aligned', preview)
        preview = replace(preview, scene_decoder='libraw', scene_decoder_version=None)
        with patch('dngscan.gui.service.dg.load_raw', return_value=replace(preview)) as decode:
            _load_export_scene(preview.path, 'reconstruct', 'camera', 'coreimage', 'auto', 'auto', 'aligned', preview)
            self.assertEqual(decode.call_args.kwargs['decoder'], 'libraw')

    def test_preview_decoder_contract_reaches_export_child_with_explicit_seed(self):
        from tests.test_preview_cache import _bundle
        from types import SimpleNamespace
        from unittest.mock import Mock
        from dngscan.gui import service
        captured = {}
        class Process:
            def __init__(self, target, args, name):
                captured.update(args[0])
            def start(self):
                raise RuntimeError('captured child payload')
        context = SimpleNamespace(Queue=lambda maxsize: Mock(), Process=Process)
        preview = replace(_bundle(), scene_decoder='coreimage', scene_decoder_version='8')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'image.dng'
            path.touch()
            params = {'input': str(path), 'decoder': 'coreimage', 'coreimageVersion': 'auto',
                      'filmOpticsSeed': 1234}
            with patch.object(service.PREVIEW_STORE, 'peek', return_value=SimpleNamespace(bundle=preview)), \
                 patch.object(service.mp, 'get_context', return_value=context):
                with self.assertRaisesRegex(RuntimeError, 'captured child payload'):
                    service.run_export_isolated(params)
        self.assertEqual(captured['_previewDecode']['scene_decoder_version'], '8')
        self.assertEqual(captured['filmOpticsSeed'], 1234)

    def test_evidence_failure_keeps_scene_analysis_report_and_cache_usable(self):
        # This is fault injection, not a claim about any real private RAW codec.
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            path = Path(tmp) / 'opaque.raw'
            path.write_bytes(b'fixture')
            stack.enter_context(patch('dngscan.raw_io.acquire_raw_evidence', side_effect=EvidenceAcquisitionError('injected unsupported sensor codec')))
            stack.enter_context(patch.object(ci, 'runtime_available', return_value=True))
            image = np.full((32, 48, 3), .18, np.float16)
            image[:2] = 4.0
            stack.enter_context(patch.object(ci, 'decode_scene_rec2020', return_value=(image, {'version': '8', 'baseline_exposure_cleared': True})))
            bundle = load_raw(path, decoder='coreimage', wb_mode='daylight')
            self.assertIsNone(bundle.raw_image)
            self.assertIsNone(bundle.white_level)
            self.assertEqual(bundle.camera_wb, [])
            self.assertEqual(bundle.evidence_provider, 'unavailable')
            self.assertEqual(bundle.scene_reliability_source, 'decoded-image-estimate')
            self.assertEqual(bundle.wb_mode, 'camera')
            self.assertIsNotNone(bundle.wb_degradation)
            analysis, y, ev = analyze(bundle, 4)
            self.assertEqual(analysis.channel_ids, [])
            self.assertTrue(np.isnan(analysis.cell_union_pct))
            self.assertEqual(analysis.noise_evidence_status, 'unavailable')
            self.assertIn('不可用', '\n'.join(summary_lines(bundle, analysis)))
            row = csv_row(bundle, analysis, None)
            self.assertEqual(row['fullwell_reference'], '')
            self.assertIsNone(row['container_bits_est'])
            from dngscan.tone import build_render_plan
            from dngscan.hdr_agx_plan import compile_hdr_agx_plan
            plan = build_render_plan(bundle, analysis, 'agx', 'p3')
            hdr = compile_hdr_agx_plan(plan, analysis=analysis, scene_decoder='coreimage')
            self.assertLessEqual(hdr.tone.rendered_headroom_ev, 1.0)
            self.assertEqual(hdr.color.channel_separation, 0.0)
            proxy = pc.build_proxy_entry(bundle, analysis)
            cache = Path(tmp) / 'cache.npz'
            pc._write_disk_entry(cache, proxy)
            restored = pc._read_disk_entry(cache, path, False)
            self.assertIsNotNone(restored)
            self.assertIsNone(restored.bundle.white_level)
            self.assertEqual(restored.bundle.evidence_error, bundle.evidence_error)
            self.assertEqual(restored.bundle.scene_reliability_source, bundle.scene_reliability_source)
            from dngscan.plot import plot_dashboard
            plot_dashboard(bundle, analysis, y, ev, Path(tmp) / 'scan.png')
            self.assertTrue((Path(tmp) / 'scan.png').is_file())

    def test_reference_empty_and_missing_survive_disk_distinctly(self):
        from tests.test_preview_cache import _bundle, _analysis
        for reference in (None, np.empty((0, 3), np.float32), np.ones((100, 3), np.float32)):
            with self.subTest(reference=None if reference is None else reference.shape), tempfile.TemporaryDirectory() as tmp:
                bundle = replace(_bundle(), scene_reliable_reference_rec2020=reference,
                                 scene_reliability_source='sensor-reference', scene_reliable_reference_pct=42.)
                analysis = replace(_analysis(), color_clip_k_of_all_pct={0: 90., 1: 10., 2: 0., 3: 0.})
                entry = pc.build_proxy_entry(bundle, analysis)
                path = Path(tmp) / 'cache.npz'
                pc._write_disk_entry(path, entry)
                restored = pc._read_disk_entry(path, bundle.path, False)
                self.assertIsNotNone(restored)
                self.assertEqual(restored.analysis.color_clip_k_of_all_pct, analysis.color_clip_k_of_all_pct)
                if reference is None:
                    self.assertIsNone(restored.bundle.scene_reliable_reference_rec2020)
                else:
                    np.testing.assert_array_equal(restored.bundle.scene_reliable_reference_rec2020, reference)
                self.assertEqual(restored.bundle.scene_reliable_reference_pct, 42.)
