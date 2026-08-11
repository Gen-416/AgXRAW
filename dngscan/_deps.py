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
# happens at package import. A8 item 8: it is a true EXTRA — a plain
# conversion must not fail because the dashboard dependency is absent, so
# its availability is tracked separately from the required deps and only
# the dashboard path enforces it (plot._matplotlib() at first render).
DASHBOARD_IMPORT_ERRORS: list[str] = []
try:
    import importlib.util as _importlib_util

    if _importlib_util.find_spec("matplotlib") is None:
        raise ImportError("matplotlib is not installed")
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    DASHBOARD_IMPORT_ERRORS.append(f"matplotlib: {exc}")
