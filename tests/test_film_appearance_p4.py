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
        region, >= +1.5 in at least two. Measured v2 draft: skin +1.56,
        foliage +2.41, sky +2.67, magenta +1.11."""
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
        Measured: every region's normalized peak stays within a tenth of
        the raw one."""
        from tools.film_palette_probe import TARGET_REGIONS

        wheel = self.idx.kind == "wheel"
        mid = (self.idx.ev >= MID_BAND[0]) & (self.idx.ev <= MID_BAND[1])
        pr = self.renders["portra400"]["ref"]
        er = self.renders["ektar100"]["ref"]
        ys = float(np.median(pr[wheel & mid].sum(1))
                   / np.median(er[wheel & mid].sum(1)))
        d_norm = pal.compare(pr, (er * ys).astype(np.float64))["delta_e00"]
        for name in TARGET_REGIONS:
            rows_t = self._region_rows(self.d_tech, name)
            rows_n = self._region_rows(d_norm, name)
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

    def test_neutral_ramp_untouched_by_both(self) -> None:
        neutral = self.idx.kind == "neutral"
        for stock in ("portra400", "ektar100"):
            d = pal.compare(
                self.renders[stock]["tech"], self.renders[stock]["ref"]
            )
            with self.subTest(stock=stock):
                self.assertLess(
                    float(np.nanmax(d["delta_e00"][neutral])), 0.05
                )


class NeutralizationDefaultTests(unittest.TestCase):
    def test_recipes_declare_print_balanced(self) -> None:
        r = fa.load_recipe(
            "portra400__endura_reference_v1",
            stock_id="portra400", medium_id="kodak_portra_endura__translated",
        )
        self.assertEqual(r["neutralization_policy"], "print-balanced")

    def test_service_reference_defaults_to_print_balanced(self) -> None:
        from dngscan.gui.service import parse_film_params

        base = {"film": "portra400", "filmMode": "full"}
        out = parse_film_params({**base, "filmAppearance": "reference"})
        self.assertEqual(out[3], "print")
        # explicit choice still wins
        out2 = parse_film_params({
            **base, "filmAppearance": "reference",
            "filmNeutralization": "technical-neutral",
        })
        self.assertEqual(out2[3], "off")
        # technical keeps the frozen default
        out3 = parse_film_params(base)
        self.assertEqual(out3[3], "off")


if __name__ == "__main__":
    unittest.main()
