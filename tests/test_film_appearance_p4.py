# SPDX-License-Identifier: GPL-3.0-or-later
"""Appearance P4 gates: the first authored recipe pair (plan §10, A4 contract).

The pair under test is the shipped Portra 400 / Ektar 100 on Endura, authored
as a common base plus per-stock residuals. What this file pins:

- the IDENTITY INCREMENT (§15.2 as recalibrated with P4's measurements): the
  pair's dE00 distance with recipes minus without, in each declared target
  region. Measured on the region's PEAK EV row and on the visible mid-band —
  an all-EV median is structurally unreachable because black-on-black cannot
  differ and the shoulder is protected by the EV envelope by declaration;
- the increment survives stripping the global brightness difference (it must
  be hue-path and colour-density identity, not a disguised global grade);
- no differential richness: the pair's chroma-gain fields are equal by
  construction (the purity axis belongs to the inter-image beta);
- neutrality, the hue cap, and the highlight/toe protection;
- reference mode defaults to the recipes' declared print-balanced
  neutralization when the user made no explicit choice.

Look approval is the OWNER's: these gates hold the floor under the numbers,
the samples decide the taste. Joint A/B pending — neither recipe counts as
finished before it (A4).
"""
from __future__ import annotations

import unittest

import numpy as np

from dngscan import film_appearance as fa
from dngscan import film_palette_diag as pal

MID_BAND = (-2.0, 4.0)


def _renders():
    from tools.film_palette_probe import render_probe

    vol, idx = pal.palette_volume()
    out = {}
    for stock in ("portra400", "ektar100"):
        out[stock] = {
            "tech": render_probe(vol, stock, "full"),
            "ref": render_probe(vol, stock, "full", film_appearance="reference"),
        }
    return vol, idx, out


def _print_balanced_technical(vol, stock):
    """technical appearance on the print-balanced chain: the reference
    pipeline minus the recipe — the kernel-isolation baseline."""
    from tests.golden_support import all_scenes
    from dngscan.render import apply_tone_core
    from dngscan.tone import build_render_plan

    scene = all_scenes()["daylight_wide_dr"]
    plan = build_render_plan(
        scene.bundle, scene.analysis, "agx", "srgb",
        film_curve=stock, film_mode="full",
        film_crossover="print", film_appearance="technical",
    )
    import numpy as np

    return np.asarray(
        apply_tone_core(np.asarray(vol, dtype=np.float32).reshape(-1, 3),
                        plan.tone, plan.color),
        dtype=np.float64,
    )


class IdentityIncrementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vol, cls.idx, r = _renders()
        cls.d_tech = pal.compare(r["portra400"]["tech"], r["ektar100"]["tech"])["delta_e00"]
        cls.d_ref = pal.compare(r["portra400"]["ref"], r["ektar100"]["ref"])["delta_e00"]
        cls.renders = r

    def _region_rows(self, d, name):
        from tools.film_palette_probe import _region_mask

        m0 = _region_mask(self.idx, name)
        return {
            ev: float(np.nanmedian(d[m0 & (self.idx.ev == ev)]))
            for ev in (-2.0, 0.0, 2.0, 4.0)
        }

    def test_every_region_gains_identity_at_its_peak(self) -> None:
        """Recalibrated §15.2: peak-row increment >= +1.0 in every target
        region, >= +1.5 in at least two. Measured v3 (post-A5 kernel: scene
        EV coordinate, S gates, density on L and C, true print-balanced
        pipeline): skin +1.90, foliage +2.38, sky +3.01, magenta +1.69."""
        from tools.film_palette_probe import TARGET_REGIONS

        strong = 0
        for name in TARGET_REGIONS:
            rows_t = self._region_rows(self.d_tech, name)
            rows_r = self._region_rows(self.d_ref, name)
            peak = max(rows_r[ev] - rows_t[ev] for ev in rows_t)
            with self.subTest(region=name):
                self.assertGreaterEqual(peak, 1.0,
                                        f"{name} peak increment {peak:+.2f}")
            if peak >= 1.5:
                strong += 1
        self.assertGreaterEqual(strong, 2, "fewer than two strong regions")

    def test_the_visible_band_aggregate_holds_the_floor(self) -> None:
        wheel = self.idx.kind == "wheel"
        mid = (self.idx.ev >= MID_BAND[0]) & (self.idx.ev <= MID_BAND[1])
        inc = (float(np.nanmedian(self.d_ref[wheel & mid]))
               - float(np.nanmedian(self.d_tech[wheel & mid])))
        self.assertGreaterEqual(inc, 0.5, f"aggregate increment {inc:+.2f}")

    def test_the_increment_survives_stripping_global_brightness(self) -> None:
        """Ektar's density fields darken it overall; if the identity were
        only that global difference, normalizing luminance would erase it.
        A5 item 8: BOTH pairs are normalized the same way (the first cut
        normalized only the reference pair, inflating its increment) and
        the normalization is Rec.2020 LUMA, not a channel sum."""
        from tools.film_palette_probe import TARGET_REGIONS

        luma = np.array([0.2627, 0.6780, 0.0593])
        wheel = self.idx.kind == "wheel"
        mid = (self.idx.ev >= MID_BAND[0]) & (self.idx.ev <= MID_BAND[1])

        def norm_pair(a, b):
            ys = float(np.median(a[wheel & mid] @ luma)
                       / np.median(b[wheel & mid] @ luma))
            return pal.compare(a, (b * ys).astype(np.float64))["delta_e00"]

        d_tech_n = norm_pair(self.renders["portra400"]["tech"],
                             self.renders["ektar100"]["tech"])
        d_ref_n = norm_pair(self.renders["portra400"]["ref"],
                            self.renders["ektar100"]["ref"])
        for name in TARGET_REGIONS:
            rows_t = self._region_rows(d_tech_n, name)
            rows_n = self._region_rows(d_ref_n, name)
            peak = max(rows_n[ev] - rows_t[ev] for ev in rows_t)
            with self.subTest(region=name):
                self.assertGreaterEqual(
                    peak, 0.9,
                    f"{name} normalized peak {peak:+.2f} — the identity "
                    "collapsed into a global brightness difference",
                )

    def test_no_differential_richness(self) -> None:
        """The A4 rule as bytes: both recipes carry the SAME chroma-gain
        field. The purity axis belongs to beta."""
        a = fa.load_recipe(
            "portra400__endura_reference_v1",
            stock_id="portra400", medium_id="kodak_portra_endura__translated",
        )
        b = fa.load_recipe(
            "ektar100__endura_reference_v1",
            stock_id="ektar100", medium_id="kodak_portra_endura__translated",
        )
        np.testing.assert_array_equal(a["log_chroma_gain"], b["log_chroma_gain"])

    def test_the_hue_cap_and_the_protected_extremes(self) -> None:
        for rid, stock in (
            ("portra400__endura_reference_v1", "portra400"),
            ("ektar100__endura_reference_v1", "ektar100"),
        ):
            r = fa.load_recipe(
                rid, stock_id=stock,
                medium_id="kodak_portra_endura__translated",
            )
            with self.subTest(recipe=rid):
                self.assertLessEqual(
                    float(np.abs(r["hue_delta_deg"]).max()), 12.0
                )
                for f in ("hue_delta_deg", "log_chroma_gain", "density_ev"):
                    self.assertEqual(float(np.abs(r[f][0]).max()), 0.0,
                                     "the -6 EV row must stay zero")
                    self.assertEqual(float(np.abs(r[f][-1]).max()), 0.0,
                                     "the +6 EV row must stay zero")

    def test_neutral_ramp_untouched_by_the_kernel(self) -> None:
        """Since A5 item 6 the reference pipeline runs print-balanced, so
        tech(bounded)-vs-ref greys legitimately differ by the CROSSOVER —
        that difference is the print character, not a leak. The gate
        therefore isolates the KERNEL: same print-balanced chain with and
        without the recipe must leave the grey ramp alone."""
        neutral = self.idx.kind == "neutral"
        for stock in ("portra400", "ektar100"):
            d = pal.compare(
                _print_balanced_technical(self.vol, stock),
                self.renders[stock]["ref"],
            )
            with self.subTest(stock=stock):
                # Not zero: print-balanced greys carry the crossover cast,
                # and to the S-gated kernel that residual chroma IS colour
                # (S near c0 puts w_c around 0.4), so heavy density fields
                # graze them — the same bounded graze P6 pins at the EV
                # scale. The bound is well under one JND.
                self.assertLess(
                    float(np.nanmax(d["delta_e00"][neutral])), 0.5
                )


class NeutralizationDefaultTests(unittest.TestCase):
    def test_recipes_declare_print_balanced(self) -> None:
        r = fa.load_recipe(
            "portra400__endura_reference_v1",
            stock_id="portra400", medium_id="kodak_portra_endura__translated",
        )
        self.assertEqual(r["neutralization_policy"], "print-balanced")

    def test_the_compiler_resolves_the_default_policy(self) -> None:
        """A5 item 6: ONE resolution point. The service forwards None when
        the user made no explicit choice; the compiler resolves it from the
        appearance mode; explicit choices win everywhere."""
        from dngscan.gui.service import parse_film_params
        from dngscan.tone import build_render_plan
        from tests.golden_support import all_scenes

        base = {"film": "portra400", "filmMode": "full"}
        self.assertIsNone(parse_film_params(
            {**base, "filmAppearance": "reference"})[3])
        self.assertEqual(parse_film_params({
            **base, "filmAppearance": "reference",
            "filmNeutralization": "technical-neutral",
        })[3], "off")
        scene = all_scenes()["daylight_wide_dr"]
        for appearance, crossover, expect in (
            ("reference", None, "print-balanced"),
            ("custom", None, "print-balanced"),
            ("technical", None, "technical-neutral"),
            ("reference", "off", "technical-neutral"),
        ):
            with self.subTest(appearance=appearance, crossover=crossover):
                plan = build_render_plan(
                    scene.bundle, scene.analysis, "agx", "srgb",
                    film_curve="portra400", film_mode="full",
                    film_crossover=crossover, film_appearance=appearance,
                )
                self.assertEqual(
                    plan.film[2].neutralization_policy, expect
                )


if __name__ == "__main__":
    unittest.main()
