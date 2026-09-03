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


class IdentityDerivedRealization(unittest.TestCase):
    def test_helper_is_deterministic_odd_and_32_bit(self) -> None:
        from dngscan.gui.preview_cache import _realization_id_for

        a = _realization_id_for("0f1e2d3c4b5a69788796a5b4c3d2e1f0")
        self.assertEqual(a, _realization_id_for("0f1e2d3c4b5a69788796a5b4c3d2e1f0"))
        self.assertEqual(a & 1, 1)
        self.assertLess(a, 1 << 32)
        self.assertNotEqual(a, _realization_id_for("ffffffff0000000011111111"))

    def test_store_stamps_the_entry_and_exposes_the_same_value(self) -> None:
        from dngscan.gui import preview_cache

        src = inspect.getsource(preview_cache.PreviewCache.get)
        self.assertIn("built.realization_id = _realization_id_for(digest)", src)
        self.assertTrue(hasattr(preview_cache.PreviewCache, "realization_id_for"))
        src = inspect.getsource(preview_cache.PreviewCache.realization_id_for)
        self.assertIn("_cache_identity(", src)
        self.assertIn("return _realization_id_for(digest)", src)

    def test_export_fallbacks_no_longer_mint_random_seeds(self) -> None:
        from dngscan.gui import service

        for fn in (service.run_export, service.run_export_isolated):
            src = inspect.getsource(fn)
            with self.subTest(fn=fn.__name__):
                for needle in ("secrets", "randbits"):
                    self.assertNotIn(needle, src)
                self.assertIn("_identity_realization(", src)


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

    def test_documents_state_the_film_pair_and_optics_dial_exceptions(self) -> None:
        arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        self.assertIn("On the AgX path\nHDR never uses the completed SDR pixels", arch)
        self.assertIn("render_ultrahdr_film_pair", arch)
        arch_zh = (ROOT / "docs" / "ARCHITECTURE.zh-CN.md").read_text(encoding="utf-8")
        self.assertIn("AgX 路径上 HDR 不会把已经完成的 SDR 像素当作 tone-map 输入", arch_zh)
        self.assertIn("render_ultrahdr_film_pair", arch_zh)
        notes = (ROOT / "docs" / "ENGINEERING_NOTES.zh-CN.md").read_text(encoding="utf-8")
        self.assertIn("默认不运送", notes)
        self.assertIn("可选拨盘", notes)
        plan = (ROOT / "docs" / "FILM_OPTICS_V2_PLAN.zh-CN.md").read_text(encoding="utf-8")
        self.assertIn("R-P3-5", plan)
        self.assertIn("尚未接入 halation 源", plan)


if __name__ == "__main__":
    unittest.main()
