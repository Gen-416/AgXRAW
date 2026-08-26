# SPDX-License-Identifier: GPL-3.0-or-later
"""CBLD advisory black-level priors: lookup contract, channel order,
clipping/mismatch surfacing, and the advisory-only doctrine.

Review R1 item 8: the upstream database carries no explicit
redistribution license, so it is NOT shipped in the repo or wheel — the
loader reads a user-imported file ($DNGSCAN_CBLD or
~/.config/dngscan/cbld.json) and is silent without one. These tests run
against a SYNTHETIC fixture with the same schema; no upstream data is
reproduced here.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from dngscan import cbld

FIXTURE = {
    "schema": 1,
    "source": "synthetic test fixture (no upstream data)",
    "page": "https://y-g-jiang.github.io/CBLD.html",
    "author": "知乎@姜尧耕 (y-g-jiang.github.io)",
    "advisory": "仅供参考，最好以自己机器的当次拍摄为准（上游原话）",
    "channel_order": ["R", "G1", "B", "G2"],
    "fetched": "2026-08-13",
    "cameras": [
        {
            "id": "testcam-a",
            "name": "TestCam A(实测)",
            "measured": True,
            "libraw_make_contains": "TESTMAKE",
            "libraw_model_contains": ["TESTCAM A"],
            "shootingModes": [
                {
                    "modeName": "single",
                    "data": [
                        {"iso": 100,
                         "r": {"avg": 512.25}, "g1": {"avg": 512.10},
                         "b": {"avg": 512.80}, "g2": {"avg": 512.40},
                         "clipping": False},
                        {"iso": 800,
                         "r": {"avg": 513.00}, "g1": {"avg": 513.10},
                         "b": {"avg": 513.60}, "g2": {"avg": 513.20},
                         "clipping": True},
                    ],
                },
                {"modeName": "burst", "data": [
                    {"iso": 100,
                     "r": {"avg": 500.0}, "g1": {"avg": 500.0},
                     "b": {"avg": 500.0}, "g2": {"avg": 500.0},
                     "clipping": False},
                ]},
            ],
        },
        {
            "id": "testcam-b",
            "name": "TestCam B(推荐值)",
            "measured": False,
            "libraw_make_contains": "TESTMAKE",
            "libraw_model_contains": ["TESTCAM B"],
            "shootingModes": [
                {"modeName": "single", "data": [
                    {"iso": 200,
                     "r": {"avg": 64.0}, "g1": {"avg": 64.0},
                     "b": {"avg": 64.0}, "g2": {"avg": 64.0},
                     "clipping": False},
                ]},
            ],
        },
    ],
}


class _FixtureBase(unittest.TestCase):
    """Point the loader at a temp fixture via DNGSCAN_CBLD; the loader's
    cache is keyed by resolved path, so a fresh temp file reloads."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name) / "cbld.json"
        path.write_text(json.dumps(FIXTURE, ensure_ascii=False), "utf-8")
        cls._old = os.environ.get("DNGSCAN_CBLD")
        os.environ["DNGSCAN_CBLD"] = str(path)
        cbld._db_cached.cache_clear()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._old is None:
            os.environ.pop("DNGSCAN_CBLD", None)
        else:
            os.environ["DNGSCAN_CBLD"] = cls._old
        cbld._db_cached.cache_clear()
        cls._tmp.cleanup()


class LookupTests(_FixtureBase):
    def test_known_camera_exact_iso(self) -> None:
        hit = cbld.find_black_levels("TESTMAKE INC.", "TESTCAM A", 100)
        self.assertIsNotNone(hit)
        self.assertTrue(hit["measured"])
        # channel order contract: R, G1, B, G2 (CBLD's own order)
        self.assertEqual(hit["values"], (512.25, 512.10, 512.80, 512.40))
        self.assertFalse(hit["clipping"])
        # first shooting mode contract: burst rows must not be picked
        self.assertEqual(hit["mode"], "single")

    def test_nearest_iso_and_clipping_flag(self) -> None:
        hit = cbld.find_black_levels("TESTMAKE INC.", "TESTCAM A", 1000)
        self.assertEqual(hit["iso_matched"], 800)
        self.assertTrue(hit["clipping"])

    def test_unknown_camera_returns_none(self) -> None:
        self.assertIsNone(cbld.find_black_levels("SONY", "ILCE-7M3", 100))
        self.assertIsNone(cbld.find_black_levels(None, None, 100))

    def test_bad_channel_order_fails_closed(self):
        """Audit R11: cbld._load_db empties the whole library when the
        fixture's channel_order is not R,G1,B,G2 — previously untested."""
        import json
        import os
        import tempfile
        from dngscan import cbld

        with tempfile.TemporaryDirectory() as td:
            bad = dict(FIXTURE)
            bad["channel_order"] = ["G1", "R", "B", "G2"]
            path = os.path.join(td, "cbld.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(bad, fh, ensure_ascii=False)
            prev = os.environ.get("DNGSCAN_CBLD")
            os.environ["DNGSCAN_CBLD"] = path
            cbld._db_cached.cache_clear()
            try:
                self.assertEqual(cbld._db(), {"cameras": []},
                                 "bad channel_order must fail closed")
            finally:
                if prev is None:
                    os.environ.pop("DNGSCAN_CBLD", None)
                else:
                    os.environ["DNGSCAN_CBLD"] = prev
                cbld._db_cached.cache_clear()

    def test_fixture_contract(self) -> None:
        payload = json.loads(cbld.data_path().read_text(encoding="utf-8"))
        self.assertEqual(payload["channel_order"], ["R", "G1", "B", "G2"])
        self.assertIn("仅供参考", payload["advisory"])


class ReportLineTests(_FixtureBase):
    def test_mismatch_against_metadata_is_surfaced(self) -> None:
        line = cbld.report_line(
            "TESTMAKE INC.", "TESTCAM A", 800, [512, 512, 512, 512],
            color_desc="RGBG",
        )
        self.assertIn("削底", line)
        self.assertIn("最大差", line)
        self.assertIn("姜尧耕", line)
        # matching honesty (R1 item 8): the heuristics are named
        self.assertIn("启发式匹配", line)
        self.assertIn("最近档", line)
        self.assertIn("首个", line)
        self.assertIn("实测", line)

    def test_unmeasured_entry_is_not_labelled_measured(self) -> None:
        line = cbld.report_line(
            "TESTMAKE INC.", "TESTCAM B", 200, None, color_desc="RGBG"
        )
        self.assertIsNotNone(line)
        self.assertIn("推荐值", line)
        self.assertNotIn("(实测", line)

    def test_close_metadata_stays_quiet_about_mismatch(self) -> None:
        line = cbld.report_line(
            "TESTMAKE INC.", "TESTCAM A", 100,
            [512.25, 512.10, 512.80, 512.40],
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
                "TESTMAKE INC.", "TESTCAM A", 800,
                [512, 512, 512, 512], color_desc=desc,
            )
            self.assertIsNotNone(line)
            self.assertNotIn("最大差", line)
            self.assertIn("通道顺序非 RGBG", line)

    def test_no_match_yields_no_line(self) -> None:
        self.assertIsNone(
            cbld.report_line("SONY", "ILCE-1", 100, None, color_desc="RGBG")
        )


class AbsentDatabaseTests(unittest.TestCase):
    """R1 item 8: without a user import the module is silent — no line,
    no exception, no warning."""

    def test_missing_file_is_silent(self) -> None:
        old = os.environ.get("DNGSCAN_CBLD")
        os.environ["DNGSCAN_CBLD"] = "/nonexistent/cbld.json"
        cbld._db_cached.cache_clear()
        try:
            self.assertIsNone(
                cbld.find_black_levels("TESTMAKE INC.", "TESTCAM A", 100)
            )
            self.assertIsNone(
                cbld.report_line("TESTMAKE INC.", "TESTCAM A", 100, None)
            )
        finally:
            if old is None:
                os.environ.pop("DNGSCAN_CBLD", None)
            else:
                os.environ["DNGSCAN_CBLD"] = old
            cbld._db_cached.cache_clear()


if __name__ == "__main__":
    unittest.main()
