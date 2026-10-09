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
            resolve_grade_params({"look": "optic_warm_cyan", "filter": "red_ipp2_rec709_medium"})

    def test_filter_grade(self) -> None:
        look, ls, filt, fs = resolve_grade(grade_id_for_filter("red_ipp2_rec709_medium"), 0.8)
        self.assertEqual(look, "none")
        self.assertEqual(filt, "red_ipp2_rec709_medium")
        self.assertAlmostEqual(fs, 0.8)

    def test_filter_grade_bare_name(self) -> None:
        look, ls, filt, fs = resolve_grade("red_ipp2_rec709_medium", 0.8)
        self.assertEqual(filt, "red_ipp2_rec709_medium")

    def test_look_grade(self) -> None:
        look, ls, filt, fs = resolve_grade(grade_id_for_look("optic_warm_cyan"), 1.0)
        self.assertEqual(look, "optic_warm_cyan")
        self.assertEqual(filt, "none")

    def test_look_grade_bare_name(self) -> None:
        look, ls, filt, fs = resolve_grade("optic_warm_cyan", 1.0)
        self.assertEqual(look, "optic_warm_cyan")

    def test_colliding_bare_id_raises(self) -> None:
        from dngscan import look

        orig = look.LOOK_FIELDS.get("red_ipp2_rec709_medium")
        try:
            look.LOOK_FIELDS["red_ipp2_rec709_medium"] = look.LOOK_FIELDS["optic_warm_cyan"]
            with self.assertRaises(ValueError):
                parse_grade_id("red_ipp2_rec709_medium")
        finally:
            if orig is None:
                look.LOOK_FIELDS.pop("red_ipp2_rec709_medium", None)
            else:
                look.LOOK_FIELDS["red_ipp2_rec709_medium"] = orig


if __name__ == "__main__":
    unittest.main()
