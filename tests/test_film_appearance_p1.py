# SPDX-License-Identifier: GPL-3.0-or-later
"""Appearance layer P1 gates (FILM_APPEARANCE_RECIPE_PLAN §16 P1).

P1 ships infrastructure and must change nothing: the exit gate is that every
frozen byte in the repo stays identical (the wider suites hold that), that
`technical` never touches an asset, and that everything wrong about a recipe
fails CLOSED — schema, hash, pairing, non-identity fields, unknown modes,
hand-built plans claiming reference without a compiled object.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from dngscan.film_appearance import (
    APPEARANCE_DIR,
    FilmAppearancePlan,
    apply_film_appearance,
    compile_appearance_plan,
    load_recipe,
    medium_family,
)


def _compile_render_plan(**kw):
    from dngscan.tone import build_render_plan
    from tests.golden_support import all_scenes

    scene = all_scenes()["daylight_wide_dr"]
    base = dict(film_curve="portra400", film_mode="full", film_crossover="off")
    base.update(kw)
    return build_render_plan(scene.bundle, scene.analysis, "agx", "srgb", **base)


class CompileTests(unittest.TestCase):
    def test_technical_is_the_default_and_touches_no_asset(self) -> None:
        plan = _compile_render_plan()
        app = plan.film[-1]
        self.assertIsInstance(app, FilmAppearancePlan)
        self.assertEqual(app.mode, "technical")
        self.assertIsNone(app.recipe)
        self.assertEqual(app.asset_sha256, "")

    def test_reference_resolves_and_pins_the_recipe(self) -> None:
        plan = _compile_render_plan(film_appearance="reference")
        app = plan.film[-1]
        self.assertEqual(app.mode, "reference")
        self.assertEqual(app.recipe_id, "portra400__endura_reference_v1")
        self.assertEqual(len(app.asset_sha256), 64)
        self.assertIsNotNone(app.recipe)
        # the same compiled object rides the tone plan for the runtime
        self.assertIs(plan.tone.film_appearance_compiled, app)

    def test_reference_without_a_recipe_fails_closed(self) -> None:
        # gold200 has no authored recipe (E1 gave velvia100 one)
        with self.assertRaises(ValueError):
            _compile_render_plan(film_curve="gold200", film_appearance="reference")

    def test_reference_outside_full_mode_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            _compile_render_plan(film_mode="observe", film_appearance="reference")

    def test_unknown_mode_and_strength_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            compile_appearance_plan("banana", 1.0, stock_id="x", medium_id="y")
        # 3.1 = just past STRENGTH_MAX (raised to 3.0 by the owner's
        # taste-to-dial policy; the ceiling itself is math-gated in P2)
        for bad in (-0.1, 3.1, float("nan")):
            with self.subTest(strength=bad):
                with self.assertRaises(ValueError):
                    compile_appearance_plan(
                        "reference", bad,
                        stock_id="portra400",
                        medium_id="kodak_portra_endura__translated",
                    )

    def test_medium_family_collapse(self) -> None:
        self.assertEqual(medium_family("kodak_portra_endura__translated"), "endura")
        self.assertEqual(medium_family("kodak_supra_endura__translated"), "endura")
        self.assertEqual(medium_family("kodak_2383__native"), "print2383")
        self.assertEqual(medium_family("direct__velvia100"), "direct")


class LoaderFailClosedTests(unittest.TestCase):
    """Every corrupted-asset axis refuses to load. Corruptions are written to
    a scratch copy; the shipped asset is never touched."""

    def _tamper(self, tmpdir: Path, mutate) -> Path:
        src = APPEARANCE_DIR / "portra400__endura_reference_v1.npz"
        with np.load(src, allow_pickle=False) as z:
            data = {k: np.asarray(z[k]).copy() for k in z.files}
        mutate(data)
        out = tmpdir / src.name
        np.savez_compressed(out, **data)
        return out

    def test_wrong_pairing_fails(self) -> None:
        with self.assertRaises(ValueError):
            load_recipe(
                "portra400__endura_reference_v1",
                stock_id="ektar100",
                medium_id="kodak_portra_endura__translated",
            )
        with self.assertRaises(ValueError):
            load_recipe(
                "portra400__endura_reference_v1",
                stock_id="portra400",
                medium_id="kodak_2383__native",
            )

    def test_hash_drift_fails(self) -> None:
        import tempfile
        from unittest import mock

        from dngscan import film_appearance as fa

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            def bump(data):
                d = json.loads(str(data["meta"]))
                d["note"] = "tampered"
                data["meta"] = np.asarray(json.dumps(d))

            path = self._tamper(tmp, bump)
            with mock.patch.object(fa, "APPEARANCE_DIR", tmp), \
                 mock.patch.object(fa, "MANIFEST_PATH", fa.MANIFEST_PATH):
                with self.assertRaises(ValueError) as ctx:
                    fa.load_recipe(
                        "portra400__endura_reference_v1",
                        stock_id="portra400",
                        medium_id="kodak_portra_endura__translated",
                    )
                self.assertIn("哈希", str(ctx.exception))

    # test_non_identity_recipe_is_refused_in_p1 was removed WITH the P1
    # loader gate when the P2 kernel landed, exactly as both documented.
    # Non-identity recipes are now exercised by tests/test_film_appearance_p2.

    def test_missing_manifest_fails(self) -> None:
        import tempfile
        from unittest import mock

        from dngscan import film_appearance as fa

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = APPEARANCE_DIR / "portra400__endura_reference_v1.npz"
            (tmp / src.name).write_bytes(src.read_bytes())
            with mock.patch.object(fa, "APPEARANCE_DIR", tmp), \
                 mock.patch.object(fa, "MANIFEST_PATH", tmp / "MANIFEST.json"):
                with self.assertRaises(ValueError) as ctx:
                    fa.load_recipe(
                        "portra400__endura_reference_v1",
                        stock_id="portra400",
                        medium_id="kodak_portra_endura__translated",
                    )
                self.assertIn("清单", str(ctx.exception))


class RuntimeTests(unittest.TestCase):
    def test_reference_engages_and_technical_stays_frozen(self) -> None:
        """P1's original assertion (reference == technical) was predicated
        on the identity placeholders; P4 authored the shipped recipes, so
        the wiring contract flips: reference must CHANGE the render and
        technical must not have moved. The strict identity behaviour lives
        on with synthetic identity recipes in test_film_appearance_p2."""
        from dngscan.render import apply_tone_core

        pt = _compile_render_plan()
        pr = _compile_render_plan(film_appearance="reference",
                                  film_crossover="off")
        arr = (np.random.default_rng(7).random((1024, 3)) * 0.6).astype(np.float32)
        a = np.asarray(apply_tone_core(arr, pt.tone, pt.color))
        b = np.asarray(apply_tone_core(arr, pr.tone, pr.color))
        self.assertFalse(np.array_equal(a, b),
                         "an authored recipe must change the render")

    def test_technical_apply_returns_the_same_object(self) -> None:
        arr = np.ones((4, 3), dtype=np.float32)
        self.assertIs(apply_film_appearance(arr, FilmAppearancePlan()), arr)
        self.assertIs(apply_film_appearance(arr, None), arr)

    def test_handwritten_reference_without_compiled_object_fails(self) -> None:
        from types import SimpleNamespace

        from dngscan.film_develop import apply_film_core

        plan = SimpleNamespace(
            curve_preset="portra400", film_mode="full", film_crossover="off",
            film_exposure_ev=0.0, film_print_timing="fixed",
            film_print_medium="", film_print_exposure_ev=0.0,
            color_head_y=0.0, color_head_m=0.0,
            film_development="measured_default",
            film_dev_contrast=0.0, film_dev_fog=0.0, film_dev_density=0.0,
            film_compression=0.0, film_compression_knee=2.0,
            film_highlight_density=0.0,
            film_grain=0.0, film_halation=0.0, film_bloom=0.0,
            film_optics_seed=0, film_appearance="reference",
        )
        with self.assertRaises(ValueError):
            apply_film_core(np.full((4, 3), 0.18, dtype=np.float32), plan)


class ServiceWiringTests(unittest.TestCase):
    def test_interpretation_group_parses_and_guards_mode(self) -> None:
        from dngscan.gui.service import parse_film_params

        base = {"film": "portra400", "filmMode": "full"}
        out = parse_film_params({**base, "filmInterimage": "off",
                                 "filmAppearance": "reference",
                                 "filmAppearanceStrength": 0.8})
        # positions 13:16 are the interpretation trio; the tail grew the
        # P6 custom modifiers (richness/color-density/neutral-bias)
        self.assertEqual(out[13:16], ("off", "reference", 0.8))
        with self.assertRaises(ValueError):
            parse_film_params({**base, "filmAppearance": "banana"})
        with self.assertRaises(ValueError):
            parse_film_params({"film": "portra400", "filmMode": "observe",
                               "filmInterimage": "off"})

    def test_the_fingerprint_forks_on_the_interpretation_group(self) -> None:
        from dngscan.gui.service import export_plan_fingerprint

        base = dict(wb="5500k", film_grain=0.0)
        a = export_plan_fingerprint(**base, film_interimage="declared",
                                    film_appearance="technical",
                                    film_appearance_strength=1.0)
        b = export_plan_fingerprint(**base, film_interimage="off",
                                    film_appearance="technical",
                                    film_appearance_strength=1.0)
        c = export_plan_fingerprint(**base, film_interimage="declared",
                                    film_appearance="reference",
                                    film_appearance_strength=1.0)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
