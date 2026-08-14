# SPDX-License-Identifier: GPL-3.0-or-later
"""GUI service guards that mirror CLI contracts (review R2 items 2/3)."""
from __future__ import annotations

import unittest


class GatedCoreimageGuardTests(unittest.TestCase):
    """R2 item 2: the service refuses toneCore=gated + decoder=coreimage the
    same way the CLI does, instead of letting gated silently decay to
    raw_permission≈0 on the maskless pipeline."""

    def test_gated_with_coreimage_is_refused(self) -> None:
        from dngscan.gui.service import reject_gated_coreimage

        with self.assertRaises(ValueError):
            reject_gated_coreimage("gated", "coreimage")

    def test_every_other_combination_passes(self) -> None:
        from dngscan.gui.service import reject_gated_coreimage

        for core in ("agx", "lum", "neutral"):
            reject_gated_coreimage(core, "coreimage")
        for core in ("agx", "gated", "lum", "neutral"):
            reject_gated_coreimage(core, "libraw")


class DaylightResetRemovalTests(unittest.TestCase):
    """R2 item 3: the GUI no longer silently rewrites wb=daylight to camera
    when RAW 9 is selected — raw_io implements the daylight anchor through
    the hot-WB matrix path (A9), so GUI, CLI and API expose one capability."""

    def test_page_source_carries_no_daylight_reset(self) -> None:
        from pathlib import Path

        import dngscan.gui.page as page

        src = Path(page.__file__).read_text("utf-8")
        self.assertNotIn('$("#wb").value==="daylight"', src)


if __name__ == "__main__":
    unittest.main()
