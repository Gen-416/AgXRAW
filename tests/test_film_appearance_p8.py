# SPDX-License-Identifier: GPL-3.0-or-later
"""Appearance E1 gates: the first cross-family expansion (plan §10 items 3-4).

Two single-stock families join the Endura pair:

- ``vision3250d__print2383_reference_v1`` — the cine print reading: dense
  dark colour, warm skin, cyan-cold shadow blues, highlights drifting
  warm-green before the ±6 EV envelope walks them back to neutral;
- ``velvia100__direct_reference_v1`` — the reversal landmark: green/cyan/
  magenta separation, high colour density, the skin arc protected.

No within-family pair exists for either, so there is no identity-increment
gate; what this file pins instead:

- the recipe VISIBILITY floor per declared target region (ref-vs-tech
  peak-row medians) and the mid-band aggregate;
- DIRECTIONAL structure, not just magnitude (review lesson: assert the
  sign) — Vision3's shadow blues move toward cyan, Velvia's greens toward
  emerald;
- Velvia's skin protection: the skin arc is its LEAST-moved target region;
- the kernel-isolated neutral gate (same print-balanced chain with and
  without the recipe), same 0.5 dE00 bound as P4;
- the §15.2 caps and protected ±6 EV rows on the shipped assets;
- the CROSS-FAMILY identity floor: every pair among the four reference
  renders keeps a mid-band median >= 2 dE00 (§15.2 absolute floor);
- theatrical presets fail closed in reference mode until their recipes are
  authored (family identity alone must not silently borrow the translated
  recipe for a different stock id).
"""
from __future__ import annotations

import unittest

import numpy as np

from dngscan import film_appearance as fa
from dngscan import film_palette_diag as pal

MID_BAND = (-2.0, 4.0)
STOCKS = {
    "vision3250d": ("vision3250d__print2383_reference_v1",
                    "kodak_2383__translated"),
    "velvia100": ("velvia100__direct_reference_v1", "direct__velvia100"),
}


def _print_balanced_technical(vol, stock):
    from tests.golden_support import all_scenes
    from dngscan.render import apply_tone_core
    from dngscan.tone import build_render_plan

    scene = all_scenes()["daylight_wide_dr"]
    plan = build_render_plan(
        scene.bundle, scene.analysis, "agx", "srgb",
        film_curve=stock, film_mode="full",
        film_crossover="print", film_appearance="technical",
    )
    return np.asarray(
        apply_tone_core(np.asarray(vol, dtype=np.float32).reshape(-1, 3),
                        plan.tone, plan.color),
        dtype=np.float64,
    )


class ExpansionGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from tools.film_palette_probe import render_probe

        cls.vol, cls.idx = pal.palette_volume()
        cls.tech = {}
        cls.ref = {}
        for stock in STOCKS:
            cls.tech[stock] = render_probe(cls.vol, stock, "full")
            cls.ref[stock] = render_probe(
                cls.vol, stock, "full", film_appearance="reference"
            )

    def _mid(self):
        return (self.idx.ev >= MID_BAND[0]) & (self.idx.ev <= MID_BAND[1])

    def test_every_target_region_clears_the_visibility_floor(self) -> None:
        """Measured at authoring (E1): vision peaks 3.0-6.5, velvia
        3.8-6.4 across the four regions. The floor holds the recipes
        visible without pinning taste."""
        from tools.film_palette_probe import TARGET_REGIONS, _region_mask

        for stock in STOCKS:
            d = pal.compare(self.tech[stock], self.ref[stock])["delta_e00"]
            strong = 0
            for name in TARGET_REGIONS:
                m0 = _region_mask(self.idx, name)
                rows = [float(np.nanmedian(d[m0 & (self.idx.ev == ev)]))
                        for ev in (-2.0, 0.0, 2.0, 4.0)]
                peak = max(rows)
                with self.subTest(stock=stock, region=name):
                    self.assertGreaterEqual(peak, 1.5,
                                            f"{stock}/{name} peak {peak:.2f}")
                if peak >= 2.5:
                    strong += 1
            with self.subTest(stock=stock):
                self.assertGreaterEqual(strong, 2)

    def test_the_mid_band_aggregate(self) -> None:
        wheel = self.idx.kind == "wheel"
        mid = self._mid()
        for stock in STOCKS:
            d = pal.compare(self.tech[stock], self.ref[stock])["delta_e00"]
            with self.subTest(stock=stock):
                self.assertGreaterEqual(
                    float(np.nanmedian(d[wheel & mid])), 1.5
                )

    def test_vision_shadow_blues_move_toward_cyan(self) -> None:
        """§10 item 4's declared direction, asserted as a SIGN: at the -2 EV
        row the sky/blue region's hue delta is negative (toward cyan)."""
        from tools.film_palette_probe import _region_mask

        d = pal.compare(self.tech["vision3250d"], self.ref["vision3250d"])
        m = _region_mask(self.idx, "sky_cyan") & (self.idx.ev == -2.0)
        med = float(np.nanmedian(d["d_hue_deg"][m]))
        self.assertLess(med, -0.5, f"shadow blues moved {med:+.2f} deg")

    def test_velvia_greens_move_toward_emerald(self) -> None:
        from tools.film_palette_probe import _region_mask

        d = pal.compare(self.tech["velvia100"], self.ref["velvia100"])
        m = _region_mask(self.idx, "foliage_green") & (self.idx.ev == 0.0)
        med = float(np.nanmedian(d["d_hue_deg"][m]))
        self.assertGreater(med, 0.5, f"greens moved {med:+.2f} deg")

    def test_velvia_skin_is_the_least_moved_region(self) -> None:
        """§10 item 3: skin protection is a DECLARED property — at EV0 the
        skin arc's median move is strictly the smallest of the four target
        regions."""
        from tools.film_palette_probe import TARGET_REGIONS, _region_mask

        d = pal.compare(self.tech["velvia100"], self.ref["velvia100"])["delta_e00"]
        meds = {}
        for name in TARGET_REGIONS:
            m = _region_mask(self.idx, name) & (self.idx.ev == 0.0)
            meds[name] = float(np.nanmedian(d[m]))
        others = min(v for k, v in meds.items() if k != "skin_warm")
        self.assertLess(meds["skin_warm"], others,
                        f"skin {meds['skin_warm']:.2f} vs min-other {others:.2f}")

    def test_neutral_ramp_untouched_by_the_kernel(self) -> None:
        """Same kernel-isolation gate as P4: tech(bounded)-vs-ref greys are
        the CROSSOVER (Velvia is the flagship crossover stock — ~8 dE00 of
        print character); the kernel itself must stay under half a JND."""
        neutral = self.idx.kind == "neutral"
        for stock in STOCKS:
            d = pal.compare(
                _print_balanced_technical(self.vol, stock), self.ref[stock]
            )
            with self.subTest(stock=stock):
                self.assertLess(
                    float(np.nanmax(d["delta_e00"][neutral])), 0.5
                )

    def test_caps_and_protected_extremes(self) -> None:
        for stock, (rid, medium) in STOCKS.items():
            r = fa.load_recipe(rid, stock_id=stock, medium_id=medium)
            with self.subTest(recipe=rid):
                self.assertLessEqual(
                    float(np.abs(r["hue_delta_deg"]).max()), 12.0
                )
                self.assertLessEqual(
                    float(np.abs(r["density_ev"]).max()), 0.35
                )
                for f in ("hue_delta_deg", "log_chroma_gain", "density_ev"):
                    self.assertEqual(float(np.abs(r[f][0]).max()), 0.0)
                    self.assertEqual(float(np.abs(r[f][-1]).max()), 0.0)

    def test_the_cross_family_identity_floor(self) -> None:
        """§15.2: every pair among the four shipped reference renders keeps
        a mid-band median >= 2 dE00. Measured at authoring: 3.7-6.5."""
        from tools.film_palette_probe import render_probe

        wheel = self.idx.kind == "wheel"
        mid = self._mid()
        renders = dict(self.ref)
        for stock in ("portra400", "ektar100"):
            renders[stock] = render_probe(
                self.vol, stock, "full", film_appearance="reference"
            )
        names = sorted(renders)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = pal.compare(renders[a], renders[b])["delta_e00"]
                with self.subTest(pair=(a, b)):
                    self.assertGreaterEqual(
                        float(np.nanmedian(d[wheel & mid])), 2.0
                    )

    def test_theatrical_reference_fails_closed(self) -> None:
        """The theatrical presets share the print2383 FAMILY but are their
        own stock ids; until their recipes are authored, reference mode must
        refuse rather than silently borrow the translated recipe."""
        from tests.golden_support import all_scenes
        from dngscan.tone import build_render_plan

        scene = all_scenes()["daylight_wide_dr"]
        with self.assertRaises(ValueError):
            build_render_plan(
                scene.bundle, scene.analysis, "agx", "srgb",
                film_curve="vision3250d_theatrical", film_mode="full",
                film_appearance="reference",
            )


if __name__ == "__main__":
    unittest.main()
