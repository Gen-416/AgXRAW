# SPDX-License-Identifier: GPL-3.0-or-later
"""Appearance P6 gates: the custom controls (plan §13/§16 P6).

Three bounded modifiers, multiplicative about the recipe's own values so the
recipe stays the centre of every control range (§7.3: no unitless free
constants): richness (chroma-gain field ×(1+r)), colour density (density
field ×(1+d)), neutral-bias strength (×s). The load-bearing identity:
custom with 0/0/1 equals reference BIT FOR BIT.
"""
from __future__ import annotations

import unittest

import numpy as np

from dngscan import film_palette_diag as pal


def _probe(**kw):
    from tools.film_palette_probe import render_probe

    vol, idx = pal.palette_volume()
    return vol, idx, render_probe(vol, "portra400", "full", **kw)


class CustomControlTests(unittest.TestCase):
    def test_custom_at_defaults_is_reference_bit_for_bit(self) -> None:
        vol, _, ref = _probe(film_appearance="reference")
        _, _, cus = _probe(film_appearance="custom")
        np.testing.assert_array_equal(np.asarray(ref), np.asarray(cus))

    def test_richness_moves_chroma_not_hue(self) -> None:
        from tests.golden_support import all_scenes
        from dngscan.render import apply_tone_core
        from dngscan.tone import build_render_plan

        scene = all_scenes()["daylight_wide_dr"]
        vol, idx = pal.palette_volume()
        outs = {}
        for r in (-0.5, 0.0, 0.5):
            plan = build_render_plan(
                scene.bundle, scene.analysis, "agx", "srgb",
                film_curve="portra400", film_mode="full",
                film_crossover="print", film_appearance="custom",
                film_richness=r,
            )
            outs[r] = np.asarray(
                apply_tone_core(vol.reshape(-1, 3), plan.tone, plan.color),
                dtype=np.float64,
            )
        wheel = idx.kind == "wheel"
        lo = pal.compare(outs[0.0], outs[-0.5])
        hi = pal.compare(outs[0.0], outs[0.5])
        lo_g = np.nanmedian(lo["log2_colorfulness_ratio"][wheel])
        hi_g = np.nanmedian(hi["log2_colorfulness_ratio"][wheel])
        self.assertLess(float(lo_g), -0.005)
        self.assertGreater(float(hi_g), 0.005)
        for d in (lo, hi):
            hh = np.abs(d["d_hue_deg"][wheel])
            self.assertLess(float(np.nanmedian(hh[np.isfinite(hh)])), 1.0)

    def test_color_density_moves_luminance_of_colours_only(self) -> None:
        from tests.golden_support import all_scenes
        from dngscan.render import apply_tone_core
        from dngscan.tone import build_render_plan

        scene = all_scenes()["daylight_wide_dr"]
        vol, idx = pal.palette_volume()
        outs = {}
        for d in (0.0, 0.8):
            plan = build_render_plan(
                scene.bundle, scene.analysis, "agx", "srgb",
                film_curve="portra400", film_mode="full",
                film_crossover="print", film_appearance="custom",
                film_color_density=d,
            )
            outs[d] = np.asarray(
                apply_tone_core(vol.reshape(-1, 3), plan.tone, plan.color),
                dtype=np.float64,
            )
        cmp_ = pal.compare(outs[0.0], outs[0.8])
        wheel = idx.kind == "wheel"
        neutral = idx.kind == "neutral"
        # the recipe's density fields are mostly positive, so +density darkens
        self.assertLess(
            float(np.nanmedian(cmp_["d_output_ev"][wheel])), 0.0
        )
        # Under print-balanced the print's near-neutrals carry the grey
        # scale's own crossover cast — to the kernel that residual chroma IS
        # colour (w_c ~ 0.03-0.1), so a density modifier grazes it at the
        # hundredth-of-a-stop scale. The gate bounds that graze rather than
        # denying it: a truly chromatic patch moves an order more.
        self.assertLess(
            float(np.nanmax(np.abs(cmp_["d_output_ev"][neutral]))), 0.03,
            "colour density moved the greys beyond the crossover graze",
        )

    def test_modifiers_fail_closed_outside_custom(self) -> None:
        from tests.golden_support import all_scenes
        from dngscan.tone import build_render_plan

        scene = all_scenes()["daylight_wide_dr"]
        for kw in (
            dict(film_appearance="reference", film_richness=0.3),
            dict(film_appearance="technical", film_color_density=0.2),
            dict(film_appearance="reference", film_neutral_bias=0.5),
        ):
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError):
                    build_render_plan(
                        scene.bundle, scene.analysis, "agx", "srgb",
                        film_curve="portra400", film_mode="full", **kw,
                    )

    def test_out_of_range_modifiers_fail_closed(self) -> None:
        from dngscan.film_appearance import compile_appearance_plan

        for kw in (
            dict(richness_delta=1.5), dict(color_density_delta=-1.2),
            dict(neutral_bias_strength=2.5),
            dict(richness_delta=float("nan")),
        ):
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError):
                    compile_appearance_plan(
                        "custom", 1.0,
                        stock_id="portra400",
                        medium_id="kodak_portra_endura__translated",
                        **kw,
                    )

    def test_service_guards_the_modifiers(self) -> None:
        from dngscan.gui.service import parse_film_params

        base = {"film": "portra400", "filmMode": "full"}
        out = parse_film_params({
            **base, "filmAppearance": "custom", "filmRichness": 0.4,
        })
        self.assertEqual(out[16:19], (0.4, 0.0, 1.0))
        with self.assertRaises(ValueError):
            parse_film_params({**base, "filmRichness": 0.4})

    def test_the_compiled_plan_carries_the_deltas(self) -> None:
        from tests.golden_support import all_scenes
        from dngscan.tone import build_render_plan

        scene = all_scenes()["daylight_wide_dr"]
        plan = build_render_plan(
            scene.bundle, scene.analysis, "agx", "srgb",
            film_curve="portra400", film_mode="full",
            film_appearance="custom", film_richness=0.25,
            film_color_density=-0.1,
        )
        app = plan.film[-1]
        self.assertEqual(app.mode, "custom")
        self.assertEqual(app.richness_delta, 0.25)
        self.assertEqual(app.color_density_delta, -0.1)
        self.assertEqual(app.neutral_bias_strength, 1.0)


if __name__ == "__main__":
    unittest.main()
