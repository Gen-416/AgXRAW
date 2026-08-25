# SPDX-License-Identifier: GPL-3.0-or-later
"""Gates for the literature-anchor and external-dataset registry files.

Both are declaration-heavy data files (docs/INTERIMAGE_LITERATURE.zh-CN.md):
these tests pin the transcribed numbers against the sources quoted in the
provenance fields, so a silent edit that drifts a number from its patent
quote fails here.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

DATA = Path(__file__).parents[1] / "dngscan" / "data"


class InterimageLiteratureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = json.loads((DATA / "interimage_literature.json").read_text())

    def test_format_and_provenance(self):
        self.assertEqual(self.doc["format"], "dngscan-interimage-literature-1")
        self.assertIn("Hanson & Horton", self.doc["provenance"]["method_definition"]["note"])
        for e in self.doc["entries"]:
            self.assertTrue(e.get("source_url"), e["id"])

    def test_us5942381_pins(self):
        e = {x["id"]: x for x in self.doc["entries"]}["US5942381-example"]
        rows = {r["film"][:1]: r for r in e["data"]}
        self.assertEqual(rows["A"]["receiver_gamma_red"], 0.99)
        self.assertEqual(rows["C"]["receiver_gamma_red"], 0.65)
        self.assertEqual(e["derived"]["receiver_gamma_suppression"]["A1"], 0.34)

    def test_us6004737_pins(self):
        e = {x["id"]: x for x in self.doc["entries"]}["US6004737-tables-III-VI"]
        ratios = [r["R"] for r in e["data"]]
        self.assertEqual(min(ratios), 0.70)
        self.assertEqual(max(ratios), 0.85)

    def test_us4830954_pins(self):
        e = {x["id"]: x for x in self.doc["entries"]}["US4830954-claims-and-examples"]
        claim = e["data"][0]
        self.assertEqual((claim["yellow_iie_min"], claim["magenta_iie_min"],
                          claim["cyan_iie_min"]), (0.10, 0.25, 0.15))
        v1b = e["data"][2]
        self.assertEqual((v1b["yellow_iie"], v1b["magenta_iie"], v1b["cyan_iie"]),
                         (0.15, 0.35, 0.30))

    def test_beta_table_comparison_is_documented_not_gated(self):
        """The beta table is NOT range-gated against the literature: the
        mapping between IIE% and the bounded-map beta needs the derivation
        recorded in docs/INTERIMAGE_LITERATURE.zh-CN.md section 2 first.
        This test just keeps the doc's promise discoverable."""
        doc = (Path(__file__).parents[1] / "docs" /
               "INTERIMAGE_LITERATURE.zh-CN.md").read_text()
        self.assertIn("等效 IIE% 数值复现", doc)


class ExternalDatasetRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = json.loads((DATA / "external_datasets.json").read_text())

    def test_every_entry_declares_status_and_use(self):
        self.assertEqual(self.doc["format"], "dngscan-external-datasets-1")
        for e in self.doc["entries"]:
            for field in ("status", "allowed_use", "urls"):
                self.assertTrue(e.get(field), f"{e['id']}: {field}")

    def test_haldclut_stays_taste_material(self):
        e = {x["id"]: x for x in self.doc["entries"]}["rawtherapee-haldclut"]
        self.assertIn("外观层", e["allowed_use"])
        self.assertIn("不得当任何形式的真值", e["allowed_use"])

    def test_cinestill_is_watch_not_fetch(self):
        e = {x["id"]: x for x in self.doc["entries"]}["cinestill800t-paired"]
        self.assertIn("watch", e["allowed_use"])
        self.assertIn("未发布", e["status"])


if __name__ == "__main__":
    unittest.main()
