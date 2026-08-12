# SPDX-License-Identifier: GPL-3.0-or-later
"""CBLD advisory black-level priors: lookup contract, channel order,
clipping/mismatch surfacing, and the advisory-only doctrine."""
from __future__ import annotations

import json
import unittest

from dngscan import cbld


class LookupTests(unittest.TestCase):
    def test_known_camera_exact_iso(self) -> None:
        hit = cbld.find_black_levels("NIKON CORPORATION", "NIKON D810", 400)
        self.assertIsNotNone(hit)
        self.assertTrue(hit["measured"])
        # channel order contract: R, G1, B, G2 (CBLD's own order)
        self.assertEqual(hit["values"], (601.3, 601.16, 601.9, 601.56))
        self.assertFalse(hit["clipping"])

    def test_nearest_iso_and_clipping_flag(self) -> None:
        hit = cbld.find_black_levels("NIKON CORPORATION", "NIKON D810", 1000)
        self.assertEqual(hit["iso_matched"], 800)
        self.assertTrue(hit["clipping"])

    def test_unknown_camera_returns_none(self) -> None:
        self.assertIsNone(cbld.find_black_levels("SONY", "ILCE-7M3", 100))
        self.assertIsNone(cbld.find_black_levels(None, None, 100))

    def test_asset_provenance_and_contract(self) -> None:
        payload = json.loads(cbld.DATA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["channel_order"], ["R", "G1", "B", "G2"])
        self.assertIn("姜尧耕", payload["author"])
        self.assertIn("仅供参考", payload["advisory"])
        self.assertGreaterEqual(len(payload["cameras"]), 6)


class ReportLineTests(unittest.TestCase):
    def test_mismatch_against_metadata_is_surfaced(self) -> None:
        line = cbld.report_line(
            "NIKON CORPORATION", "NIKON D810", 800, [600, 600, 600, 600],
            color_desc="RGBG",
        )
        self.assertIn("削底", line)
        self.assertIn("最大差", line)
        self.assertIn("姜尧耕", line)

    def test_close_metadata_stays_quiet_about_mismatch(self) -> None:
        line = cbld.report_line(
            "NIKON CORPORATION", "NIKON D810", 64,
            [601.3, 601.3, 601.5, 601.4],
            color_desc="RGBG",
        )
        self.assertIsNotNone(line)
        self.assertNotIn("最大差", line)

    def test_non_rgbg_order_skips_channel_comparison(self) -> None:
        # upstream warns some cameras report RG1G2B-style orders; a
        # channel-wise diff across a mismatched order would invent a
        # spurious mismatch, so the comparison must be gated on RGBG.
        for desc in ("RGGB", "", None):
            line = cbld.report_line(
                "NIKON CORPORATION", "NIKON D810", 800,
                [600, 600, 600, 600], color_desc=desc,
            )
            self.assertIsNotNone(line)
            self.assertNotIn("最大差", line)
            self.assertIn("通道顺序非 RGBG", line)

    def test_no_match_yields_no_line(self) -> None:
        self.assertIsNone(
            cbld.report_line("SONY", "ILCE-1", 100, None, color_desc="RGBG")
        )


if __name__ == "__main__":
    unittest.main()
