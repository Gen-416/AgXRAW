# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared singleton ownership, unchanged rendering, and versioned boundaries."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dngscan.application.api import Application
from dngscan.application.contracts import API_VERSION, default_job


def request(command, params=None):
    return {'version': API_VERSION, 'id': 'contract-test', 'command': command, 'params': params or {}}


class ApplicationContractTests(unittest.TestCase):
    def test_compatibility_imports_share_exact_module_and_singletons(self):
        for name in ('service', 'preview_cache', 'preview_scheduler', 'scheduler', 'constants', 'histogram', 'calibration_service'):
            self.assertIs(importlib.import_module('dngscan.gui.'+name), importlib.import_module('dngscan.application.'+name))

    def test_shared_application_has_no_gui_or_transport_dependency(self):
        import ast
        root = Path(__file__).resolve().parents[1] / 'dngscan/application'
        for path in root.glob('*.py'):
            if path.name == '__main__.py':
                continue # Optional launcher is an adapter entry.
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn('gui', (node.module or '').split('.'), path.name)
                    self.assertNotIn('transports', (node.module or '').split('.'), path.name)

    def test_dispatch_preserves_parameters_and_identity_without_mutating_caller(self):
        class Recording:
            def prepare_preview(self, params):
                self.params = params.copy()
                params['ev'] = 9
                return {'ok': True, 'prepared': True}
        service = Recording()
        app = Application(service=service, calibrations=object())
        params = {**default_job(), 'input': '/sample.DNG', 'selectionEpoch': 4, 'generation': 8}
        result = app.execute(request('preview.prepare', params))
        self.assertEqual(result['id'], 'contract-test')
        self.assertTrue(result['ok'])
        self.assertEqual(service.params, params)
        self.assertEqual(params['ev'], 0.)

    def test_invalid_contracts_do_not_dispatch_or_enable_processing(self):
        class Recording:
            def prepare_preview(self, params):
                raise AssertionError('must not dispatch')
        app = Application(Recording(), object())
        inputs = [None, {}, {**request('preview.prepare'), 'version': 99},
                  request('unknown'), request('preview.prepare', {'input': '/x', 'ev': float('nan')}),
                  request('preview.prepare', {'input': '/x', 'evAuto': 'false'}),
                  request('preview.prepare', {'input': '/x', 'typoExposure': 1}),
                  request('preview.prepare', {})]
        for value in inputs:
            with self.subTest(value=value):
                result = app.execute(value)
                self.assertFalse(result['ok'])
                self.assertEqual(result['error']['code'], 'invalid-request')

    def test_error_categories_and_superseded_are_preserved(self):
        class Service:
            def run_preview(self, _params):
                return {'ok': True, 'superseded': True, 'generation': 9}
            def prepare_preview(self, _params):
                raise FileNotFoundError('missing RAW')
        app = Application(Service(), object())
        self.assertEqual(app.execute(request('preview.render', {'input': '/x'}))['result']['generation'], 9)
        self.assertEqual(app.execute(request('preview.prepare', {'input': '/x'}))['error']['code'], 'source-not-found')

    def test_new_defaults_match_fresh_legacy_page_controls(self):
        from html.parser import HTMLParser
        from dngscan.gui.page import PAGE
        class Controls(HTMLParser):
            def __init__(self):
                super().__init__(); self.values = {}; self.select = None
            def handle_starttag(self, tag, attrs):
                a = dict(attrs)
                if tag == 'input' and 'id' in a:
                    self.values[a['id']] = ('checked' in a) if a.get('type') == 'checkbox' else a.get('value', '')
                if tag == 'select': self.select = a.get('id')
                if tag == 'option' and self.select and ('selected' in a or self.select not in self.values):
                    self.values[self.select] = a.get('value')
            def handle_endtag(self, tag):
                if tag == 'select': self.select = None
        controls = Controls(); controls.feed(PAGE)
        defaults = default_job()
        for key, value in defaults.items():
            page_key = 'autoExposure' if key == 'evAuto' else key
            if page_key not in controls.values:
                continue # 'none' options are injected by the existing grade registries.
            actual = controls.values[page_key]
            if key in ('hdrRho', 'hdrWhiteMargin', 'hdrShoulderStart') and actual == '':
                actual = 'auto' # Legacy payload() translates empty HDR dials to auto.
            if isinstance(value, (float, int)) and not isinstance(value, bool): actual = float(actual)
            self.assertEqual(actual, value, key)
        self.assertNotIn('quality', defaults)
        self.assertNotIn('chroma', defaults)

    def test_generated_types_are_current(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root/'tools/generate_application_types.py'), '--check'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_shared_command_cli_returns_structured_error_without_a_gui(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, '-m', 'dngscan.application', '--request', '-'],
                                input=json.dumps(request('unknown')), cwd=root,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)['error']['code'], 'invalid-request')

    def test_actual_raw_new_and_legacy_preview_pixels_and_plan_match(self):
        from tests.test_spectral_fallback_pipeline import write_noise_dng
        from dngscan.application import service
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'source.dng'
            write_noise_dng(path, file_profile=True)
            params = {**default_job(), 'input': str(path), 'includeMetrics': False}
            before = path.read_bytes()
            expected = service.run_preview(params)
            actual = Application().execute(request('preview.render', params))
            self.assertTrue(actual['ok'], actual)
            self.assertEqual(expected['preview'], actual['result']['preview'])
            self.assertEqual(expected['ev_auto'], actual['result']['ev_auto'])
            self.assertEqual(expected['scene_histogram'], actual['result']['scene_histogram'])
            self.assertEqual(path.read_bytes(), before)


class ApplicationHttpTests(unittest.TestCase):
    def test_versioned_http_and_static_frontend_preserve_auth(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest('optional GUI dependencies absent')
        from dngscan.transports.http import create_app
        from tests.test_gui_http import RecordingService
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'index.html').write_text('<html>Independent workspace</html>')
            app = create_app(service_module=RecordingService(), session_token='private-token', expected_port=12345, frontend_dir=root)
            with TestClient(app, base_url='http://127.0.0.1:12345') as client:
                self.assertIn('Independent workspace', client.get('/workspace/').text)
                payload = request('preview.prepare', {'input': '/x'})
                self.assertEqual(client.post('/api/v1/commands', json=payload).status_code, 403)
                headers = {'X-DngScan-Token': 'private-token'}
                result = client.post('/api/v1/commands', json=payload, headers=headers).json()
                self.assertEqual(result['result']['kind'], 'prepare')
                self.assertEqual(client.post('/api/v1/commands', json=payload, headers={**headers, 'Origin': 'http://evil.example'}).status_code, 403)
                self.assertEqual(client.post('/api/v1/commands', content='not json', headers=headers).status_code, 400)
                self.assertEqual(client.get('/api/v1/bootstrap', headers={'Host': 'evil.example'}).status_code, 403)


if __name__ == '__main__':
    unittest.main()
