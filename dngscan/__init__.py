# SPDX-License-Identifier: GPL-3.0-or-later
"""AgXRAW public surface.

A8 item 8: the package previously re-exported nearly everything through a
wildcard, so any internal symbol looked public. ``__all__`` now declares
the SUPPORTED api; ``dngscan.core`` stays the legacy compatibility surface
for the GUI/tools and internal modules that still rely on its import side
effects — new code should import from the named modules instead.
"""
from .core import *  # noqa: F401,F403

__all__ = [
    # decode + analysis
    "load_raw", "analyze", "RawBundle", "Analysis",
    # render planning + execution
    "build_render_plan", "apply_render_adjustments", "RenderPlan",
    "render_output_u8", "apply_tone_core",
    # environment
    "require_dependencies", "IMPORT_ERRORS",
]
