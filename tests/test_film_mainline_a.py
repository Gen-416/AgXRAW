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

        evs = np.linspace(-6.0, 4.0, 21)
        ramp = (0.18 * np.exp2(evs))[:, None].repeat(3, 1).astype(np.float32)
        with_beta = render_probe(ramp, "portra400", "full")
        without = render_probe(ramp, "portra400", "full", film_interimage="off")
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
        """velvia declares beta 0, so mainline A must not have moved it —
        its interimage chain is untouched code. The pinned value itself was
        re-measured 2026-08-26 after route C's Stage A chromaticity field
        (1.157 under the 3x3 observer -> 1.2625 under the field): a
        DECLARED Stage A change, not mainline-A drift, moved it."""
        s = _s_transfer("velvia100")
        self.assertAlmostEqual(s, 1.2625, delta=0.02)

    def test_the_hue_path_is_bounded_and_does_not_worsen_folds(self) -> None:
        """Honest claims, round three (A4 item 1).

        Round two compared only each ring's WORST step, which lets a new
        shallow fold at spoke B hide behind a deeper pre-existing one at
        spoke A, skipped the 345->0 wrap seam, and ran Portra only. The gate
        is now PER ADJACENT EDGE — step_declared[i] >= min(step_off[i], 0) -
        tolerance — with the periodic closing edge included, over EVERY
        stock that declares a beta. A4's own stricter scan found no actual
        violation; this pins that state so one cannot appear unnoticed.

        Bounds (mean/p95 of the hue path) stay asserted on the two extremes
        of the beta table, Portra (family default) and Ektar (largest beta).
        """
        from dngscan.film_develop import INTERIMAGE_BETA
        from tools.film_palette_probe import render_probe

        vol, idx = pal.palette_volume()
        wheel = idx.kind == "wheel"
        spoke = 360.0 / pal.HUE_COUNT

        def ring_edges(dec_off, dec_on, ring, order):
            """Aligned per-edge hue steps for both renderings.

            Common validity mask (perceptual 0.01 Oklab C floor, in BOTH
            renderings — same reasoning as round two), adjacency filter on
            the input-hue gap, and the wrap edge from the last surviving
            spoke back to the first (+360) appended so the 345->0 seam is a
            first-class edge instead of a blind spot.
            """
            valid = (
                (dec_off["C"][ring][order] > 1e-2)
                & (dec_on["C"][ring][order] > 1e-2)
                # Signed Stage A contract (review 2026-08-27, F4): a probe
                # on the gamut boundary has a channel EXACTLY zero, and a
                # pure-channel observer layer then receives zero light — its
                # log geometry is a cliff by construction (the old clamp
                # fabricated ~1e-9 of light and interpolated a moderate
                # exposure out of the LUT's clipped edge cells). Such inputs
                # are outside the film chain's positive domain; the fold
                # ratchet measures the hue path INSIDE it.
                & positive_in[ring][order]
            )
            if valid.sum() < 3:
                return None
            hin = idx.hue_deg[ring][order][valid]
            gaps = np.diff(np.concatenate([hin, [hin[0] + 360.0]]))
            adjacent = gaps <= 2.0 * spoke + 1.0

            def steps(dec):
                h = np.radians(dec["h_deg"][ring][order][valid])
                u = np.unwrap(np.concatenate([h, [h[0]]]))
                return np.diff(u)

            return steps(dec_off)[adjacent], steps(dec_on)[adjacent]

        positive_in = (np.asarray(vol, dtype=np.float64).reshape(-1, 3) > 0.0).all(axis=1)
        for stock in sorted(k for k, v in INTERIMAGE_BETA.items() if v > 0):
            on = render_probe(vol, stock, "full")
            off = render_probe(vol, stock, "full", film_interimage="off")
            dec_on = pal.decompose(on)
            dec_off = pal.decompose(off)
            if stock in ("portra400", "ektar100"):
                d = pal.compare(off, on)
                hh = np.abs(d["d_hue_deg"][wheel])
                hh = hh[np.isfinite(hh)]
                with self.subTest(stock=stock, gate="bounds"):
                    self.assertLess(float(np.mean(hh)), 6.5)
                    self.assertLess(float(np.percentile(hh, 95)), 22.0)
            for ev in pal.PROBE_EVS:
                for cf in pal.CHROMA_LEVELS:
                    ring = wheel & (idx.ev == ev) & (idx.chroma_frac == cf)
                    order = np.argsort(idx.hue_deg[ring])
                    edges = ring_edges(dec_off, dec_on, ring, order)
                    if edges is None:
                        continue
                    base, got = edges
                    if base.size == 0:
                        continue  # every edge of this ring lies on the gamut boundary
                    worst = float(np.min(got - np.minimum(base, 0.0)))
                    # The gamut-ray endpoint ring (chroma fraction 1.0) sits
                    # by construction on the Rec.2020 boundary, where the
                    # honest Stage A hands over from the training-hull field
                    # to the 3x3 (review 2026-08-27, F1/F4: the field weight
                    # tapers to zero at the edge so the hand-over is exact).
                    # Its spokes therefore straddle the field/3x3 seam with
                    # weights that vary spoke to spoke, and a hue step there
                    # measures the seam, not beta. The ratchet keeps that
                    # ring under a separate WATCH bound (10 deg, measured
                    # worst -9.05 deg at EV -4) instead of the in-hull bound;
                    # the old smooth band had hidden the seam by extrapolating
                    # the field past the edge — fabricated light, not a gate.
                    tol = 0.035 if cf < 1.0 else 0.18
                    with self.subTest(stock=stock, ev=ev, chroma=cf):
                        # 0.035 rad (was 0.03), re-pinned 2026-08-26 with
                        # route C's Stage A field: two edges (c200 and
                        # gold200, +4 EV chroma 1.0 — gamut-ray extremes
                        # whose rings run mostly OUTSIDE the training hull,
                        # through the field/observer blend band) measured
                        # -1.81/-1.95 deg against the old 1.72 deg bound.
                        # Widening the blend band (sigma 2->5) recovered the
                        # other eight edges; a stronger fit ridge was
                        # measured and REFUSED (it costs up to 0.12 stop of
                        # held-out p99). The ratchet stays a ratchet — this
                        # is its declared new baseline, not a silencing.
                        self.assertGreaterEqual(
                            worst, -tol,
                            "a fold appeared or deepened on an edge: "
                            f"declared vs off margin {np.degrees(worst):.2f}deg",
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

    def test_a_compiled_plan_is_immune_to_table_mutation(self) -> None:
        """A3 item 1. The compiler writes the effective beta into the plan
        and the runtime consumes THAT — editing the module table after
        compile must not move a single pixel (measured 0.0726 max drift
        when the runtime still consulted the table)."""
        import dngscan.film_develop as fd
        from dngscan.render import apply_tone_core
        from tools.film_palette_probe import reference_plan

        plan, _, _, _ = reference_plan("portra400", "full")
        self.assertEqual(
            plan.tone.film_interimage_beta, fd.INTERIMAGE_BETA["portra400"]
        )
        arr = _chroma_sweep()
        before = apply_tone_core(arr, plan.tone, plan.color)
        saved = dict(fd.INTERIMAGE_BETA)
        try:
            fd.INTERIMAGE_BETA["portra400"] = 0.0
            after = apply_tone_core(arr, plan.tone, plan.color)
        finally:
            fd.INTERIMAGE_BETA.update(saved)
        np.testing.assert_array_equal(
            np.asarray(before), np.asarray(after),
            err_msg="compiled plan output moved with the module table",
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
