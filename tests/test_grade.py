# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for unified grade selection."""

from __future__ import annotations

import unittest

from dngscan.grade import (
    grade_choices,
    grade_id_for_filter,
    grade_id_for_look,
    parse_grade_id,
    resolve_grade,
    resolve_grade_params,
)


class GradeTests(unittest.TestCase):
    def test_public_choices_exclude_missing_vendor_luts(self) -> None:
        self.assertFalse(any(choice.startswith("filter:") for choice in grade_choices()))

    def test_mutually_exclusive_legacy_params(self) -> None:
        with self.assertRaises(ValueError):
            resolve_grade_params({"look": "optic_warm_cyan", "filter": "kodak_2383_d65"})

    def test_filter_grade(self) -> None:
        look, ls, filt, fs = resolve_grade(grade_id_for_filter("kodak_2383_d65"), 0.8)
        self.assertEqual(look, "none")
        self.assertEqual(filt, "kodak_2383_d65")
        self.assertAlmostEqual(fs, 0.8)

    def test_filter_grade_bare_name(self) -> None:
        look, ls, filt, fs = resolve_grade("kodak_2383_d65", 0.8)
        self.assertEqual(filt, "kodak_2383_d65")

    def test_look_grade(self) -> None:
        look, ls, filt, fs = resolve_grade(grade_id_for_look("optic_warm_cyan"), 1.0)
        self.assertEqual(look, "optic_warm_cyan")
        self.assertEqual(filt, "none")

    def test_look_grade_bare_name(self) -> None:
        look, ls, filt, fs = resolve_grade("optic_warm_cyan", 1.0)
        self.assertEqual(look, "optic_warm_cyan")

    def test_colliding_bare_id_raises(self) -> None:
        from dngscan import look

        orig = look.LOOK_FIELDS.get("kodak_2383_d65")
        try:
            look.LOOK_FIELDS["kodak_2383_d65"] = look.LOOK_FIELDS["optic_warm_cyan"]
            with self.assertRaises(ValueError):
                parse_grade_id("kodak_2383_d65")
        finally:
            if orig is None:
                look.LOOK_FIELDS.pop("kodak_2383_d65", None)
            else:
                look.LOOK_FIELDS["kodak_2383_d65"] = orig


if __name__ == "__main__":
    unittest.main()
