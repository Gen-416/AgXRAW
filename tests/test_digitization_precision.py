# SPDX-License-Identifier: GPL-3.0-or-later
"""Gate: chart-digitized assets keep sampling ambiguity within declared
read-off uncertainty (see tools/audit_digitization.py for the metric and
the 2026-08-25 audit that introduced it)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))


class DigitizationPrecisionTests(unittest.TestCase):
    def test_sampling_ambiguity_within_declared_uncertainty(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("scipy not available")
        from audit_digitization import run

        for name, value, gate, passed in run():
            with self.subTest(curve=name):
                self.assertTrue(
                    passed,
                    f"{name}: sampling ambiguity {value:.4f} exceeds the "
                    f"declared uncertainty {gate} — add anchors (see "
                    "tools/audit_digitization.py)",
                )


if __name__ == "__main__":
    unittest.main()
