# SPDX-License-Identifier: GPL-3.0-or-later
"""Ground-truth cross-validation against the Jiangtherapee ideal-image
pair (synthetic DNGs with declared black level, white level, gain and
noise model). The assets live outside the repo (~2 MB each); the suite
runs them only when DNGSCAN_IDEAL_IMAGE_DIR points at the archive —
see tools/validate_ideal_image.py for the checks and provenance."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

_DIR = os.environ.get("DNGSCAN_IDEAL_IMAGE_DIR", "")


@unittest.skipUnless(_DIR, "DNGSCAN_IDEAL_IMAGE_DIR not set")
class IdealImageValidationTests(unittest.TestCase):
    def test_declared_truth_recovered(self) -> None:
        from tools.validate_ideal_image import validate

        rows = validate(Path(_DIR).expanduser())
        failures = [f"{name}: {detail}" for name, ok, detail in rows if not ok]
        self.assertGreaterEqual(len(rows), 7)
        self.assertFalse(failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
