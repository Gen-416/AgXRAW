# SPDX-License-Identifier: GPL-3.0-or-later
"""Mainline A gates: inter-image amplification.

The 2026-08-10 review located "full looks weak" precisely: not tone (system
gamma measured equal to observe, inside the classic 1.5-1.8 neg x paper
range), not the spatial operators — colour separation. The C-41 print-through
chain delivered an EV0 saturation transfer of 0.76 against observe's 1.16,
because the chain's declared-absent DIR inter-image effects are exactly the
mechanism that amplifies inter-layer differences in real material.

The modelled term:  D'_c = D_c + beta * (D_c - C_c(mean logE))

Everything this file pins follows from that one line:

- the neutral reference is the stock's own response to a grey patch at the
  same exposure, so a grey ramp is EXACTLY invariant — the naive D - mean(D)
  form shifted the neutral axis by up to 48%, because a C-41 neutral has
  strongly unequal channel densities (the orange mask);
- amplification is along each pixel's own colour direction in dye space, so
  hue must not rotate;
- beta is ALSO the within-family identity lever: Portra and Ektar sat 0.46
  dE00 apart before it, a hue whisper with no chroma difference at all.
"""
from __future__ import annotations

import unittest

import numpy as np

from dngscan import film_palette_diag as pal
from dngscan.film_develop import INTERIMAGE_BETA, interimage_beta

LUMA = np.array([0.2627, 0.6780, 0.0593])


def _chroma_sweep() -> np.ndarray:
    rows = []
    for h in (30.0, 140.0, 230.0):
        ray = float(pal._ray_chroma(np.array([h]))[0])
        for f in np.linspace(0.15, 0.95, 9):
            c = f * ray
            lab = np.array([[0.7, c * np.cos(np.radians(h)),
                             c * np.sin(np.radians(h))]])
            rgb = np.maximum(pal.oklab_to_rec2020(lab)[0], 0)
            y = rgb @ LUMA
            rows.append(rgb * (0.18 / max(y, 1e-9)))
    return np.asarray(rows, dtype=np.float32)


def _s_transfer(stock: str) -> float:
    from tools.film_palette_probe import render_probe

    arr = _chroma_sweep()
    out = render_probe(arr, stock, "full")
    din = pal.decompose(arr.astype(np.float64))
    dout = pal.decompose(out)
    return float(np.nanmean(dout["S"] / din["S"]))


class TableDeclarationTests(unittest.TestCase):
    def test_reversals_declare_no_interimage(self) -> None:
        """The direct-B2 reversal chain measured 1.16 already — its
        separation is baked in the measured response. A beta on top would
        double what the data records."""
        for stock in ("velvia100", "provia100f", "ektachrome100", "kodachrome64"):
            with self.subTest(stock=stock):
                self.assertEqual(interimage_beta(stock), 0.0)

    def test_the_c41_family_is_deliberately_spread(self) -> None:
        """beta is the within-family identity lever, so the table must not
        collapse the family to one value: Ektar (the vivid outlier) above the
        consumer stocks, above Portra, above Pro 400H (the airy soft one)."""
        self.assertGreater(interimage_beta("ektar100"), interimage_beta("gold200"))
        self.assertGreater(interimage_beta("gold200"), interimage_beta("portra400"))
        self.assertGreater(interimage_beta("portra400"), interimage_beta("pro400h"))
        self.assertGreater(interimage_beta("pro400h"), 0.0)

    def test_unknown_stock_gets_identity(self) -> None:
        self.assertEqual(interimage_beta("not_a_stock"), 0.0)

    def test_every_key_names_a_shipped_stock(self) -> None:
        from tests.test_film_v2_assets import _stock_files

        shipped = set(_stock_files())
        for key in INTERIMAGE_BETA:
            with self.subTest(key=key):
                self.assertIn(key, shipped, f"beta for unshipped stock {key}")


class NeutralInvarianceTests(unittest.TestCase):
    def test_a_grey_ramp_is_exactly_invariant(self) -> None:
        """The load-bearing property. The neutral reference is the same
        (possibly recipe-perturbed) table the pixel used at the same mean
        logE, so on the neutral axis the term is zero by construction — the
        orange mask, the tau(0) timing anchor and both neutralization
        families never see it."""
        from tools.film_palette_probe import render_probe
        import dngscan.film_develop as fd

        evs = np.linspace(-6.0, 4.0, 21)
        ramp = (0.18 * np.exp2(evs))[:, None].repeat(3, 1).astype(np.float32)
        with_beta = render_probe(ramp, "portra400", "full")
        saved = dict(fd.INTERIMAGE_BETA)
        try:
            fd.INTERIMAGE_BETA.clear()
            without = render_probe(ramp, "portra400", "full")
        finally:
            fd.INTERIMAGE_BETA.update(saved)
        np.testing.assert_array_equal(with_beta, without)

    def test_tone_is_untouched_on_the_neutral_axis(self) -> None:
        """Colour separation must not ride in on a contrast change: the
        review's whole point was that tone was already right."""
        from tools.film_palette_probe import render_probe

        evs = np.linspace(-4.0, 3.0, 29)
        ramp = (0.18 * np.exp2(evs))[:, None].repeat(3, 1).astype(np.float32)
        out = render_probe(ramp, "portra400", "full")
        y = np.maximum(out @ LUMA, 1e-9)
        slope = np.gradient(np.log10(y), np.log10(0.18 * np.exp2(evs)))
        g0 = float(slope[np.argmin(np.abs(evs))])
        self.assertGreater(g0, 1.2)
        self.assertLess(g0, 1.8)


class SaturationRecoveryTests(unittest.TestCase):
    def test_the_c41_negatives_recover_print_level_saturation(self) -> None:
        """The review's headline number, inverted: 0.76 becomes >= 1.0, with
        Ektar meaningfully above Portra."""
        portra = _s_transfer("portra400")
        ektar = _s_transfer("ektar100")
        self.assertGreater(portra, 1.0, f"portra S transfer {portra:.3f}")
        self.assertGreater(ektar, portra + 0.15, f"ektar {ektar:.3f} vs {portra:.3f}")
        self.assertLess(ektar, 1.5, "vivid, not neon")

    def test_the_reversal_is_byte_stable(self) -> None:
        """velvia declares beta 0, so mainline A must not have moved it at
        all — its chain is untouched code."""
        s = _s_transfer("velvia100")
        self.assertAlmostEqual(s, 1.157, delta=0.02)

    def test_hue_does_not_rotate(self) -> None:
        """Amplification runs along each pixel's own colour direction in dye
        space; a hue shift would mean the neutral reference is off-axis."""
        from tools.film_palette_probe import render_probe
        import dngscan.film_develop as fd

        arr = _chroma_sweep()
        with_beta = render_probe(arr, "portra400", "full")
        saved = dict(fd.INTERIMAGE_BETA)
        try:
            fd.INTERIMAGE_BETA.clear()
            without = render_probe(arr, "portra400", "full")
        finally:
            fd.INTERIMAGE_BETA.update(saved)
        d = pal.compare(without, with_beta)
        self.assertLess(
            float(np.nanmean(np.abs(d["d_hue_deg"]))), 2.5,
            "the term must not act as a hue rotation",
        )

    def test_within_family_identity_exists_now(self) -> None:
        """Portra vs Ektar was 0.46 dE00 median — indistinguishable. The
        spread table has to buy real separation, in the direction the stocks
        are known for (Ektar more saturated)."""
        from tools.film_palette_probe import render_probe

        vol, idx = pal.palette_volume()
        a = render_probe(vol, "portra400", "full")
        b = render_probe(vol, "ektar100", "full")
        d = pal.compare(a, b)
        wheel = idx.kind == "wheel"
        self.assertGreater(
            float(np.nanmedian(d["delta_e00"][wheel])), 1.2,
            "family separation collapsed back toward 0.46",
        )
        self.assertGreater(
            float(np.nanmedian(d["log2_saturation_ratio"][wheel])), 0.1,
            "Ektar must read more saturated than Portra, not merely different",
        )

    def test_editorial_developer_recipes_still_compose(self) -> None:
        """The neutral reference uses the recipe-perturbed table, so an
        editorial developer must keep its grey-axis contract under beta."""
        from tools.film_palette_probe import reference_plan
        from dngscan.film_develop import apply_film_core
        from types import SimpleNamespace

        plan, _, _, _ = reference_plan("portra400", "full")
        tweaked = SimpleNamespace(
            **{f: getattr(plan.tone, f) for f in (
                "curve_preset", "film_mode", "film_crossover",
                "film_exposure_ev", "film_print_timing", "film_print_medium",
                "film_print_exposure_ev", "color_head_y", "color_head_m",
                "film_compression", "film_compression_knee",
                "film_highlight_density", "film_grain", "film_halation",
                "film_bloom", "film_optics_seed",
            )},
            film_development="editorial_custom",
            film_dev_contrast=0.3, film_dev_fog=0.0, film_dev_density=0.0,
        )
        evs = np.linspace(-3.0, 3.0, 13)
        ramp = (0.18 * np.exp2(evs))[:, None].repeat(3, 1).astype(np.float32)
        out = np.asarray(apply_film_core(ramp, tweaked), dtype=np.float64)
        lab = pal.rec2020_to_oklab(out)
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        self.assertLess(
            float(chroma.max()), 0.02,
            "a grey ramp through an editorial recipe must stay near-neutral",
        )


if __name__ == "__main__":
    unittest.main()
