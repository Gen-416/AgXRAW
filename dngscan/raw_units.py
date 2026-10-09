# SPDX-License-Identifier: GPL-3.0-or-later
"""File coding endpoints, distinct from sensor linear-response thresholds."""
from __future__ import annotations

import math


def coding_endpoints(path, white_level, camera_white_levels, black_levels, spatial_black=None):
    """Capture the DNG encoding white and maximum black once, before rendering.

    LibRaw's per-channel linear-validity white can include LinearResponseLimit.
    The file's WhiteLevel remains the DN coding endpoint after linearization.
    Non-DNG files retain LibRaw's per-channel endpoint convention.
    """
    from .spatial_black import sensor_tags

    tags = sensor_tags(path, {50717})
    if tags:
        whites = list(tags.get(50717, ())) or [white_level]
    else:
        whites = list(camera_white_levels if camera_white_levels is not None else ()) or [white_level]
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in whites):
        raise ValueError("invalid RAW coding white level")
    blacks = (list(spatial_black.max_black) if spatial_black is not None else list(black_levels))
    return [float(v) for v in whites], [float(v) for v in blacks]


def normalized_raw_span(bundle, cid):
    """The common denominator used by calibration and decoder variance transfer.

    Empty coding fields preserve compatibility with legacy in-memory fixtures.
    Production acquisition always records explicit coding endpoints.
    """
    whites = getattr(bundle, "coding_white_levels", None)
    if whites:
        white = float(whites[min(cid, len(whites) - 1)])
    else:
        levels = getattr(bundle, "camera_white_levels", ())
        white = (float(levels[cid]) if cid < len(levels) and levels[cid] > 0
                 else float(bundle.white_level))
    blacks = getattr(bundle, "coding_black_levels", None) or bundle.black_levels
    black = float(blacks[min(cid, len(blacks) - 1)]) if blacks else 0.0
    return white - black
