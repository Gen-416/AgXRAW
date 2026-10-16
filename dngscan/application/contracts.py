# SPDX-License-Identifier: GPL-3.0-or-later
"""Versioned transport contract. Render validation remains in the engine.

This module has no optional GUI dependencies. Its field declarations also
generate the desktop TypeScript request types; clients never invent defaults.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

API_VERSION = 1
JOB_FIELDS = {
    **dict.fromkeys(('input', 'outdir', 'highlight', 'gamut', 'wb', 'decoder',
                    'coreimageVersion', 'coreimageScale', 'demosaic', 'lensFilter',
                    'format', 'deliveryProfile', 'chroma', 'heifEncoder', 'heifPreset',
                    'heifTune', 'toneCore', 'lumNorm', 'agxPrimaries', 'grade',
                    'sceneTransform', 'endpointMode', 'chromaCleanup', 'previewClient',
                    'previewSession'), 'string'),
    **dict.fromkeys(('ev', 'hdrHeadroom', 'gradeStrength', 'sceneTransformStrength',
                    'punch', 'midtoneBrightness', 'midtoneContrast', 'shadowTransition',
                    'highlightTransition', 'highlightFade', 'toeEndOffset',
                    'shoulderWhiteOffset', 'chromaNr'), 'number'),
    **dict.fromkeys(('quality', 'heifBitDepth', 'clipMargin', 'generation', 'selectionEpoch'), 'integer'),
    **dict.fromkeys(('evAuto', 'png', 'includeMetrics'), 'boolean'),
    **dict.fromkeys(('hdrRho', 'hdrWhiteMargin', 'hdrShoulderStart'), 'number-or-auto'),
    'captureDeclaration': 'object',
}
# Explicitly identical to a fresh legacy GUI's declaration, including neutral
# user controls. Auto delivery intentionally omits quality and chroma.
DEFAULT_JOB = {
    'highlight': 'clip', 'gamut': 'srgb', 'wb': 'camera', 'decoder': 'libraw',
    'coreimageVersion': 'auto', 'coreimageScale': 'aligned', 'demosaic': 'auto',
    'lensFilter': 'none', 'clipMargin': 4, 'chromaNr': 0., 'chromaCleanup': 'off',
    'format': 'sdr', 'deliveryProfile': 'auto', 'heifEncoder': 'auto',
    'heifBitDepth': 10, 'heifPreset': 'slow', 'heifTune': 'ssim',
    'toneCore': 'agx', 'lumNorm': 'y', 'agxPrimaries': 'base',
    'grade': 'none', 'gradeStrength': 1., 'sceneTransform': 'none',
    'sceneTransformStrength': 1., 'punch': 1., 'midtoneBrightness': 0.,
    'midtoneContrast': 0., 'shadowTransition': 0., 'highlightTransition': 0.,
    'highlightFade': 0., 'endpointMode': 'adaptive', 'toeEndOffset': 0.,
    'shoulderWhiteOffset': 0., 'hdrHeadroom': 3., 'hdrRho': 'auto',
    'hdrWhiteMargin': 'auto', 'hdrShoulderStart': 'auto', 'evAuto': True,
    'ev': 0., 'png': False,
}
COMMANDS = {
    'source.support': 'raw9_support',
    'preview.prepare': 'prepare_preview',
    'preview.render': 'run_preview',
    'preview.clip-overlay': 'clip_overlay',
    'delivery.export': 'run_export_isolated',
}
CALIBRATION_COMMANDS = {
    'calibration.list': 'list_calibrations', 'calibration.inspect': 'inspect_calibration',
    'calibration.import': 'import_calibration', 'calibration.remove': 'remove_calibration',
    'calibration.active': 'set_calibration_active', 'library.status': 'public_library_status',
    'library.install': 'install_public_library', 'library.select': 'select_public_library',
}


def default_job() -> dict[str, Any]:
    return deepcopy(DEFAULT_JOB)


def validate_request(request: Any) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(request, dict) or set(request) != {'version', 'id', 'command', 'params'}:
        raise ValueError('请求需要 version、id、command、params 四个字段')
    if type(request['version']) is not int or request['version'] != API_VERSION:
        raise ValueError('接口版本不匹配，请重启或更新前端')
    identifier, command, params = request['id'], request['command'], request['params']
    if not isinstance(identifier, str) or not 1 <= len(identifier) <= 160:
        raise ValueError('请求 id 无效')
    if not isinstance(command, str) or command not in COMMANDS | CALIBRATION_COMMANDS:
        raise ValueError('未知应用命令')
    if not isinstance(params, dict):
        raise ValueError('params 必须是对象')
    if command in COMMANDS:
        for key, value in params.items():
            kind = JOB_FIELDS.get(key)
            valid = (
                (kind == 'string' and isinstance(value, str))
                or (kind == 'boolean' and type(value) is bool)
                or (kind == 'integer' and type(value) is int)
                or (kind == 'number' and type(value) in (int, float) and math.isfinite(value))
                or (kind == 'number-or-auto' and (value == 'auto' or
                    (type(value) in (int, float) and math.isfinite(value))))
                or (kind == 'object' and isinstance(value, dict))
            )
            if not valid:
                raise ValueError(f'不支持或类型错误的参数：{key}')
        if not params.get('input'):
            raise ValueError('请先选择 RAW 文件')
    return identifier, command, deepcopy(params)


def schema() -> dict[str, Any]:
    return {'version': API_VERSION, 'job_fields': JOB_FIELDS,
            'commands': list(COMMANDS | CALIBRATION_COMMANDS), 'defaults': default_job()}
