# SPDX-License-Identifier: GPL-3.0-or-later
"""Transport-independent command facade over the shared application services."""
from __future__ import annotations

from pathlib import Path
import json
import os
import sys
from typing import Any

from .contracts import API_VERSION, COMMANDS, CALIBRATION_COMMANDS, schema, validate_request


class Application:
    def __init__(self, service=None, calibrations=None):
        if service is None:
            from . import service
        if calibrations is None:
            from . import calibration_service as calibrations
        self.service, self.calibrations = service, calibrations

    def bootstrap(self) -> dict[str, Any]:
        import dngscan as dg
        from .constants import RAW_EXTS
        from dngscan import coreimage_decode

        available, reason = dg.apple_gainmap_backend_status()
        folder = Path.home() / 'Pictures'
        return {**schema(), 'default_directory': str(folder if folder.is_dir() else Path.home()),
                'raw_extensions': sorted(RAW_EXTS),
                'capabilities': {'coreimage': coreimage_decode.available(),
                                 'hdr': available, 'hdr_reason': reason,
                                 'dependency_errors': list(dg.IMPORT_ERRORS)}}

    def execute(self, request: Any) -> dict[str, Any]:
        identifier = request.get('id') if isinstance(request, dict) else None
        try:
            identifier, command, params = validate_request(request)
        except (ValueError, TypeError) as exc:
            return self._error(identifier, 'invalid-request', str(exc))
        try:
            owner = self.service if command in COMMANDS else self.calibrations
            method = (COMMANDS | CALIBRATION_COMMANDS)[command]
            self._trace(identifier, command, 'started')
            result = getattr(owner, method)(params)
            if result.get('ok') is False:
                self._trace(identifier, command, 'failed')
                return self._error(identifier, 'operation-failed', str(result.get('error', '操作失败')))
            self._trace(identifier, command, 'superseded' if result.get('superseded') else 'succeeded')
            return {'version': API_VERSION, 'id': identifier, 'ok': True, 'result': result}
        except FileNotFoundError as exc:
            self._trace(identifier, command, 'failed')
            return self._error(identifier, 'source-not-found', str(exc))
        except ValueError as exc:
            self._trace(identifier, command, 'failed')
            return self._error(identifier, 'invalid-input', str(exc))
        except Exception as exc:
            self._trace(identifier, command, 'failed')
            from dngscan.debug_util import maybe_print_exc
            maybe_print_exc()
            return self._error(identifier, 'engine-error', str(exc))

    @staticmethod
    def _trace(identifier, command, state):
        if os.environ.get('AGXRAW_TRACE_COMMANDS') == '1':
            # Developer observability: no source paths, parameters, image data
            # or session tokens are included in these structured trace lines.
            print(json.dumps({'agxraw_command': command, 'state': state, 'id': identifier}),
                  file=sys.stderr, flush=True)

    @staticmethod
    def _error(identifier, code, message):
        return {'version': API_VERSION, 'id': identifier, 'ok': False,
                'error': {'code': code, 'message': message}}
