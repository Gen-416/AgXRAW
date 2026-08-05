# SPDX-License-Identifier: GPL-3.0-or-later
"""Small helpers for developer-facing diagnostics."""

from __future__ import annotations

import os
import sys
import traceback


def debug_traceback_enabled() -> bool:
    raw = os.environ.get("DNGSCAN_DEBUG", "").strip().lower()
    return raw not in {"", "0", "false", "off", "no"}


def maybe_print_exc(file: object | None = None) -> None:
    if debug_traceback_enabled():
        traceback.print_exc(file=file or sys.stderr)
