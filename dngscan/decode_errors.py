# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared LibRaw failure categories; a filename is never codec evidence."""
from __future__ import annotations

from ._deps import rawpy


def classify_libraw_error(exc: Exception) -> dict[str, str]:
    def is_rawpy(name):
        cls = getattr(rawpy, name, None)
        return isinstance(cls, type) and isinstance(exc, cls)

    if is_rawpy("LibRawFileUnsupportedError"):
        status, kind = "unsupported_format", "unsupported-codec"
    elif is_rawpy("LibRawDataError"):
        status, kind = "corrupt_data", "corrupt-data"
    elif isinstance(exc, OSError) or is_rawpy("LibRawIOError"):
        status, kind = "io_error", "io"
    else:
        status, kind = "runtime_failure", "runtime"
    return {"status": status, "failure_kind": kind,
            "exception_type": type(exc).__name__, "detail": str(exc)}


LIBRAW_USABLE_STATUSES = frozenset({
    "full", "dng_calibrated", "fallback_matrix", "no_color_calibration",
})
