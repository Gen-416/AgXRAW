# SPDX-License-Identifier: GPL-3.0-or-later
"""External review batch 24 (2026-09-02 handoff): technical-debt fixes.

1. R-P2-7: the auto grain seed was minted per PreviewEntry, so a memory-LRU
   eviction, a disk-cache reload or a restart re-minted it and an export
   that peeked after eviction minted yet another — preview and export grain
   could disagree. The realization is now derived from the cache identity
   digest and the export fallback asks the store for that same value.
2. R-P2-3: export naming and the fingerprint read the raw ``grade`` key
   while the render resolved the legacy look/filter pair — a legacy payload
   rendered a look under a name and fingerprint claiming grade=none.
3. R-P2-2: the fingerprint carried the input's path and size only; a
   same-size in-place replacement collided. It now carries mtime too.
4. R-P3-2: the bit-exact native/NumPy contracts are only verified on NumPy 2
   (the CI lock); the declared floor said 1.24.
5. R-P3-3 / R-P3-4 / R-P3-5: ARCHITECTURE's "HDR never uses the completed SDR
   pixels", ENGINEERING_NOTES' "halation is not transported" and the optics
   plan's "P0–P5 all merged" were each over-broad against the code; the
   documents now state the actual boundaries.
"""
from __future__ import annotations

import inspect
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]




class GradeIdResolution(unittest.TestCase):
    def test_legacy_look_and_filter_payloads_name_what_they_render(self) -> None:
        from dngscan.grade import (
            grade_id_for_filter,
            grade_id_for_look,
            resolve_grade_id,
            resolve_grade_params,
        )

        self.assertEqual(resolve_grade_id({}), ("none", 1.0))
        self.assertEqual(resolve_grade_id({"grade": "none"}), ("none", 1.0))
        gid, strength = resolve_grade_id({"grade": "look:x", "gradeStrength": "0.7"})
        self.assertEqual((gid, strength), ("look:x", 0.7))
        look = next(iter(__import__("dngscan.grade", fromlist=["LOOK_CHOICES"]).LOOK_CHOICES))
        if look == "none":
            look = list(__import__("dngscan.grade", fromlist=["LOOK_CHOICES"]).LOOK_CHOICES)[1]
        legacy = {"look": look, "lookStrength": 0.8}
        rendered_look, rendered_strength, _, _ = resolve_grade_params(legacy)
        gid, strength = resolve_grade_id(legacy)
        self.assertEqual(gid, grade_id_for_look(rendered_look))
        self.assertAlmostEqual(strength, rendered_strength)
        self.assertNotEqual(gid, "none")
        # a filter payload resolves the same way through the filter id
        from dngscan.grade import FILTER_CHOICES, filter_available

        for filt in FILTER_CHOICES:
            if filt != "none" and filter_available(filt):
                gid, _ = resolve_grade_id({"filter": filt})
                self.assertEqual(gid, grade_id_for_filter(filt))
                break

    def test_service_names_from_the_resolved_id(self) -> None:
        from dngscan.gui import service

        src = inspect.getsource(service.run_export)
        self.assertIn("grade_id, grade_strength = resolve_grade_id(params)", src)
        self.assertNotIn('params.get("grade", "none")', src)


class FingerprintCarriesMtime(unittest.TestCase):
    def test_input_mtime_is_a_fingerprinted_parameter(self) -> None:
        from dngscan.gui import service

        src = inspect.getsource(service.run_export)
        call = src[src.find("fingerprint = export_plan_fingerprint("):]
        call = call[: call.find("\n    )")]
        for needle in ("input_path", "input_size", "input_mtime_ns"):
            self.assertIn(needle, call)


class DeclaredBoundaries(unittest.TestCase):
    def test_numpy_floor_matches_the_verified_contract(self) -> None:
        self.assertIn('"numpy>=2.0"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertIn("numpy>=2.0", (ROOT / "requirements.txt").read_text(encoding="utf-8"))

    def test_documents_state_independent_hdr_formation(self) -> None:
        arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        self.assertIn("HDR never uses the completed SDR pixels", arch)
        self.assertNotIn("render_ultrahdr_film_pair", arch)
        arch_zh = (ROOT / "docs" / "ARCHITECTURE.zh-CN.md").read_text(encoding="utf-8")
        self.assertIn("HDR 不会把已经完成的 SDR 像素当作 tone-map 输入", arch_zh)


if __name__ == "__main__":
    unittest.main()
