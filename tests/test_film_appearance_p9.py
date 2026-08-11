# SPDX-License-Identifier: GPL-3.0-or-later
"""Appearance E2 gates: the extended interpretation (plan §10 item 5).

``extended`` is the scan/telecine counter-reading of a family's reference
recipe: the same hue/richness direction at 0.6x amplitude, no shadow
density block, and a DIGITALLY NEUTRAL grey axis (the recipe declares
technical-neutral and the compiler's None-default follows the declaration
— E2's refinement of the A5 single-resolution-point rule).

Shipped for vision3250d only; every other stock fails closed. Black point
and gamut width stay OUT of the palette's powers (they belong to tone and
gamut fit; §7's paper warp remains closed by measurement).

Measured at authoring: ext-vs-tech mid aggregate 0.41 vs reference 2.17;
ext grey axis vs bounded technical 0.001 dE00; ext-vs-ref mid median 2.04.
"""
from __future__ import annotations

import unittest

import numpy as np

from dngscan import film_appearance as fa
from dngscan import film_palette_diag as pal

MID_BAND = (-2.0, 4.0)
RID_REF = "vision3250d__print2383_reference_v1"
RID_EXT = "vision3250d__print2383_extended_v1"
MEDIUM = "kodak_2383__translated"


def _render(vol, **kw):
    from tests.golden_support import all_scenes
    from dngscan.render import apply_tone_core
    from dngscan.tone import build_render_plan

    scene = all_scenes()["daylight_wide_dr"]
    plan = build_render_plan(
        scene.bundle, scene.analysis, "agx", "srgb",
        film_curve="vision3250d", film_mode="full", **kw,
    )
    return np.asarray(
        __import__("dngscan.render", fromlist=["apply_tone_core"]).apply_tone_core(
            np.asarray(vol, np.float32).reshape(-1, 3), plan.tone, plan.color
        ),
        np.float64,
    ), plan


class VariantPlumbingTests(unittest.TestCase):
    def test_the_variant_selects_the_asset_and_the_policy(self) -> None:
        _, plan = _render(pal.palette_volume()[0][:8],
                          film_appearance="reference",
                          film_appearance_variant="extended")
        app = plan.tone.film_appearance_compiled
        self.assertEqual(app.recipe_id, RID_EXT)
        self.assertEqual(app.variant, "extended")
        # recipe-declared technical-neutral resolves the None default
        self.assertEqual(plan.tone.film_crossover, "off")
        self.assertEqual(plan.tone.film_appearance_variant, "extended")

    def test_reference_default_still_resolves_print_balanced(self) -> None:
        _, plan = _render(pal.palette_volume()[0][:8],
                          film_appearance="reference")
        self.assertEqual(plan.tone.film_appearance_compiled.recipe_id, RID_REF)
        self.assertEqual(plan.tone.film_crossover, "print")

    def test_strength_zero_keeps_the_declared_grey_axis(self) -> None:
        """A6 item 2, DECIDED semantics: strength scales the PALETTE only.
        The neutralization policy is the interpretation's own declared
        property, so reference+strength=0 keeps print-balanced (continuous
        in strength — collapsing to technical at 0 would jump the grey
        axis). Full technical identity is spelled film_appearance=technical
        or an explicit neutralization choice; the CLI help says so."""
        _, plan = _render(pal.palette_volume()[0][:8],
                          film_appearance="reference",
                          film_appearance_strength=0.0)
        self.assertEqual(plan.tone.film_crossover, "print")
        _, plan = _render(pal.palette_volume()[0][:8],
                          film_appearance="reference",
                          film_appearance_strength=0.0,
                          film_crossover="off")
        self.assertEqual(plan.tone.film_crossover, "off")

    def test_an_explicit_crossover_still_wins(self) -> None:
        _, plan = _render(pal.palette_volume()[0][:8],
                          film_appearance="reference",
                          film_appearance_variant="extended",
                          film_crossover="print")
        self.assertEqual(plan.tone.film_crossover, "print")

    def test_fail_closed_paths(self) -> None:
        vol = pal.palette_volume()[0][:8]
        with self.assertRaises(ValueError):   # unknown variant
            fa.compile_appearance_plan(
                "reference", 1.0, stock_id="vision3250d",
                medium_id=MEDIUM, variant="banana",
            )
        with self.assertRaises(ValueError):   # technical + non-default variant
            fa.compile_appearance_plan(
                "technical", 1.0, stock_id="vision3250d",
                medium_id=MEDIUM, variant="extended",
            )
        with self.assertRaises(ValueError):   # no extended asset for this stock
            from tests.golden_support import all_scenes
            from dngscan.tone import build_render_plan

            scene = all_scenes()["daylight_wide_dr"]
            build_render_plan(
                scene.bundle, scene.analysis, "agx", "srgb",
                film_curve="velvia100", film_mode="full",
                film_appearance="reference",
                film_appearance_variant="extended",
            )

    def test_service_parses_and_guards_the_variant(self) -> None:
        from dngscan.gui.service import parse_film_params

        base = {"film": "vision3250d", "filmMode": "full"}
        out = parse_film_params({
            **base, "filmAppearance": "reference",
            "filmAppearanceVariant": "extended",
        })
        self.assertEqual(out[-1], "extended")
        with self.assertRaises(ValueError):
            parse_film_params({**base, "filmAppearanceVariant": "extended"})
        with self.assertRaises(ValueError):
            parse_film_params({
                **base, "filmAppearance": "reference",
                "filmAppearanceVariant": "banana",
            })


class ExtendedSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vol, cls.idx = pal.palette_volume()
        cls.tech, _ = _render(cls.vol, film_appearance="technical")
        cls.ref, _ = _render(cls.vol, film_appearance="reference")
        cls.ext, _ = _render(cls.vol, film_appearance="reference",
                             film_appearance_variant="extended")

    def _mid(self):
        return (
            (self.idx.kind == "wheel")
            & (self.idx.ev >= MID_BAND[0]) & (self.idx.ev <= MID_BAND[1])
        )

    def test_the_grey_axis_is_digitally_neutral(self) -> None:
        """The point of the interpretation: with technical-neutral declared,
        extended's grey ramp matches the bounded technical chain to float
        noise (measured 0.001 dE00) — no crossover survives."""
        neutral = self.idx.kind == "neutral"
        d = pal.compare(self.tech, self.ext)["delta_e00"]
        self.assertLess(float(np.nanmax(d[neutral])), 0.05)

    def test_extended_is_the_milder_reading(self) -> None:
        mid = self._mid()
        d_ext = pal.compare(self.tech, self.ext)["delta_e00"]
        d_ref = pal.compare(self.tech, self.ref)["delta_e00"]
        self.assertLess(float(np.nanmedian(d_ext[mid])),
                        float(np.nanmedian(d_ref[mid])))

    def test_extended_is_still_a_distinct_interpretation(self) -> None:
        mid = self._mid()
        d = pal.compare(self.ref, self.ext)["delta_e00"]
        self.assertGreaterEqual(float(np.nanmedian(d[mid])), 1.0)

    def test_the_family_direction_is_preserved_by_construction(self) -> None:
        """extended hue/richness = 0.6 x reference AS BYTES (authored that
        way; proportionality keeps every sign and band position), and the
        density field drops the shadow block: its -3 EV row carries less
        than a third of reference's."""
        ref = fa.load_recipe(RID_REF, stock_id="vision3250d", medium_id=MEDIUM)
        ext = fa.load_recipe(RID_EXT, stock_id="vision3250d", medium_id=MEDIUM)
        np.testing.assert_allclose(
            ext["hue_delta_deg"],
            (0.6 * ref["hue_delta_deg"]).astype(np.float32), atol=1e-6,
        )
        np.testing.assert_allclose(
            ext["log_chroma_gain"],
            (0.6 * ref["log_chroma_gain"]).astype(np.float32), atol=1e-6,
        )
        self.assertLess(
            float(np.abs(ext["density_ev"][1]).max()),
            float(np.abs(ref["density_ev"][1]).max()) / 3.0,
        )
        self.assertEqual(ext["neutralization_policy"], "technical-neutral")


if __name__ == "__main__":
    unittest.main()
