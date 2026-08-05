# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional runtime dependencies (numpy, rawpy, matplotlib)."""
from __future__ import annotations

IMPORT_ERRORS: list[str] = []

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    np = None  # type: ignore[assignment]
    IMPORT_ERRORS.append(f"numpy: {exc}")

try:
    import rawpy
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    rawpy = None  # type: ignore[assignment]
    IMPORT_ERRORS.append(f"rawpy: {exc}")
else:
    from .libraw_policy import rawpy_runtime_problem

    if problem := rawpy_runtime_problem(rawpy):
        IMPORT_ERRORS.append(f"rawpy: {problem}")

# matplotlib is only needed by the diagnostic dashboard (plot.py), yet importing
# pyplot + font_manager costs ~0.27s — paid by every spawned export worker if it
# happens at package import. Availability is checked cheaply here; the actual
# import is deferred to plot._matplotlib() at first dashboard render.
try:
    import importlib.util as _importlib_util

    if _importlib_util.find_spec("matplotlib") is None:
        raise ImportError("matplotlib is not installed")
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    IMPORT_ERRORS.append(f"matplotlib: {exc}")
