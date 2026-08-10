# SPDX-License-Identifier: GPL-3.0-or-later
"""Mainline A2 acceptance matrix (review 2026-08-11).

The A1 review found that the raw linear inter-image form pushed 18-26% of
probe samples past the Stage B dye domain and let `amounts_to_unit` clip them
silently — part of the measured "saturation recovery" was channels riding the
LUT rails, indistinguishable from real separation. This module is the matrix
that makes that class of failure structurally visible:

- ZERO domain overflow, for every stock that declares a beta, across the
  full probe volume, film exposures spanning the declared envelope, and the
  editorial developer corners (the DECLARED corners: probing beyond them is
  itself refused fail-closed at developer_perturbation since the A2
  envelope-gap follow-up);
- grey-axis byte identity everywhere the term is active;
- saturation gain continuous in exposure (no cliff where a rail used to be);
- the compiled plan is the audit surface: `FilmDevelopmentPlan` carries the
  effective beta, the oracle switch is a validated plan field, and unknown
  values fail closed.
"""
from __future__ import annotations

import unittest

import numpy as np

from dngscan import film_palette_diag as pal
from dngscan.film_develop import INTERIMAGE_BETA

LUMA = np.array([0.2627, 0.6780, 0.0593])

# Editorial developer corners: the DECLARED envelope's extremes (film_plans
# validation and the developer_perturbation runtime guard share the bounds
# via film_v2_math), where the perturbed characteristic tables have the least
# headroom to the rails. The first version of this matrix probed contrast +1
# / density +1 believing only the physical poles (1+x > 0) bounded the
# domain; those corners are OUTSIDE the declared editorial domain, the baked
# shaper cube never claimed to cover them, and they measured a 12% excursion
# that was the probe's own doing — the bypass now fails closed
# (test_out_of_domain_recipe_fails_closed_at_runtime below).
from dngscan.film_v2_math import (
    EDITORIAL_CONTRAST_LIMIT as _C,
    EDITORIAL_DENSITY_LIMIT as _D,
    EDITORIAL_FOG_MAX as _F,
)

DEV_CORNERS = (
    dict(film_development="measured_default"),
    dict(film_development="editorial_custom", film_dev_contrast=_C,
         film_dev_fog=0.0, film_dev_density=_D),
    dict(film_development="editorial_custom", film_dev_contrast=-_C,
         film_dev_fog=_F, film_dev_density=-_D),
)


def _overflow_spy():
    """Patch amounts_to_unit with a counter; returns (counter, restore)."""
    import dngscan.film_v2_math as fvm

    counter = {"rows": 0, "bad": 0, "deepest": 0.0}
    original = fvm.amounts_to_unit

    def spy(amounts, lo, hi):
        a = np.asarray(amounts, dtype=np.float64)
        lo_ = np.asarray(lo, dtype=np.float64)
        hi_ = np.asarray(hi, dtype=np.float64)
        span = np.maximum(hi_ - lo_, 1e-12)
        excess = np.maximum(np.maximum(lo_ - a, a - hi_), 0.0) / span
        counter["rows"] += a.shape[0]
        counter["bad"] += int((excess > 1e-3).any(axis=-1).sum())
        counter["deepest"] = max(counter["deepest"], float(excess.max()))
        return original(amounts, lo, hi)

    fvm.amounts_to_unit = spy

    def restore() -> None:
        fvm.amounts_to_unit = original

    return counter, restore


class DomainMatrixTests(unittest.TestCase):
    """The reviewer's matrix: every beta stock, the whole probe, the
    exposure envelope and the developer corners — zero overflow."""

    def test_no_beta_stock_overflows_anywhere(self) -> None:
        from tools.film_palette_probe import render_probe

        volume, _ = pal.palette_volume()
        counter, restore = _overflow_spy()
        try:
            for stock in sorted(INTERIMAGE_BETA):
                with self.subTest(stock=stock):
                    render_probe(volume, stock, "full")
        finally:
            restore()
        self.assertGreater(counter["rows"], 0)
        self.assertEqual(
            counter["bad"], 0,
            f"dye-domain overflow returned (deepest {counter['deepest']:.4f})",
        )

    def test_exposure_envelope_and_developer_corners_stay_in_domain(self) -> None:
        """The A1 review measured near-80% overflow depth at a legal
        editorial extreme. The bounded map must hold the whole declared
        envelope — EVERY beta stock x its exposure bounds x the developer
        corners (A3 item 5: the first cut claimed this scope but ran three
        stocks; now the claim and the loop match)."""
        from types import SimpleNamespace

        from dngscan.film_develop import _load_v2, apply_film_core

        volume, _ = pal.palette_volume()
        flat = np.asarray(volume, dtype=np.float32)
        counter, restore = _overflow_spy()
        try:
            for stock in sorted(INTERIMAGE_BETA):
                st, _media = _load_v2(stock)
                for ev in (st["exp_lo"], 0.0, st["exp_hi"]):
                    for corner in DEV_CORNERS:
                        fields = dict(
                            curve_preset=stock, film_mode="full",
                            film_crossover="off",
                            film_exposure_ev=float(ev),
                            film_print_timing="fixed", film_print_medium="",
                            film_print_exposure_ev=0.0,
                            color_head_y=0.0, color_head_m=0.0,
                            film_dev_contrast=0.0, film_dev_fog=0.0,
                            film_dev_density=0.0,
                            film_compression=0.0, film_compression_knee=2.0,
                            film_highlight_density=0.0,
                            film_grain=0.0, film_halation=0.0, film_bloom=0.0,
                            film_optics_seed=0,
                            film_interimage="declared",
                        )
                        fields.update(corner)
                        with self.subTest(stock=stock, ev=ev, corner=corner):
                            # In-domain recipes stay inside the baked cube by
                            # construction (the bake covers exactly the
                            # declared envelope), so the gate is absolute
                            # ZERO overflow — term off AND on. The earlier
                            # baseline-relative form existed only to tolerate
                            # the out-of-domain +1/+1 corner this matrix used
                            # to probe.
                            for mode in ("off", "declared"):
                                counter.update(rows=0, bad=0, deepest=0.0)
                                fields["film_interimage"] = mode
                                apply_film_core(flat, SimpleNamespace(**fields))
                                self.assertEqual(
                                    counter["bad"], 0,
                                    f"interimage={mode}: dye-domain overflow "
                                    f"(deepest {counter['deepest']:.4f})",
                                )
        finally:
            restore()

    def test_out_of_domain_recipe_fails_closed_at_runtime(self) -> None:
        """The envelope-gap review path: a raw plan object that bypasses
        validate_film_plans must NOT get silent rail clipping — the runtime
        guard in developer_perturbation refuses beyond the declared domain."""
        from types import SimpleNamespace

        from dngscan.film_develop import apply_film_core

        plan = SimpleNamespace(
            curve_preset="ektar100", film_mode="full", film_crossover="off",
            film_exposure_ev=0.0, film_print_timing="fixed",
            film_print_medium="", film_print_exposure_ev=0.0,
            color_head_y=0.0, color_head_m=0.0,
            film_development="editorial_custom",
            film_dev_contrast=1.0, film_dev_fog=0.0, film_dev_density=1.0,
            film_compression=0.0, film_compression_knee=2.0,
            film_highlight_density=0.0,
            film_grain=0.0, film_halation=0.0, film_bloom=0.0,
            film_optics_seed=0, film_interimage="off",
        )
        with self.assertRaises(ValueError):
            apply_film_core(np.full((4, 3), 0.18, dtype=np.float32), plan)


class PlanCompilationTests(unittest.TestCase):
    """The switch is a real, validated, auditable plan field now — not a
    phantom getattr plus a mutable module dict (review item 2)."""

    def _compile(self, **kw):
        from dngscan.tone import build_render_plan
        from tests.golden_support import all_scenes

        scene = all_scenes()["daylight_wide_dr"]
        return build_render_plan(
            scene.bundle, scene.analysis, "agx", "srgb",
            film_curve="portra400", film_mode="full", film_crossover="off",
            **kw,
        )

    def test_the_effective_beta_is_compiled_into_the_plan(self) -> None:
        plan = self._compile()
        development = plan.film[1]
        self.assertEqual(development.interimage_mode, "declared")
        self.assertEqual(development.interimage_beta, INTERIMAGE_BETA["portra400"])
        self.assertEqual(plan.tone.film_interimage, "declared")

    def test_the_two_compiled_copies_cannot_diverge(self) -> None:
        """A4 item 2: the compiler resolves beta ONCE and stamps both the
        tone plan (runtime) and FilmDevelopmentPlan (audit); the two must be
        the same object-level value, or the audit surface lies."""
        plan = self._compile()
        self.assertEqual(
            plan.tone.film_interimage_beta, plan.film[1].interimage_beta
        )
        off = self._compile(film_interimage="off")
        self.assertEqual(off.tone.film_interimage_beta, 0.0)
        self.assertEqual(off.film[1].interimage_beta, 0.0)

    def test_handwritten_off_with_stray_beta_fails_closed(self) -> None:
        """A4 item 3: the runtime must reject the same contradiction the
        validator rejects — "off" with a nonzero compiled beta — instead of
        silently zeroing it, or the two entry points disagree about what
        off means."""
        from types import SimpleNamespace

        import numpy as np

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
            film_optics_seed=0,
            film_interimage="off", film_interimage_beta=0.62,
        )
        with self.assertRaises(ValueError):
            apply_film_core(np.full((4, 3), 0.18, dtype=np.float32), plan)

    def test_off_compiles_to_zero_beta(self) -> None:
        plan = self._compile(film_interimage="off")
        self.assertEqual(plan.film[1].interimage_mode, "off")
        self.assertEqual(plan.film[1].interimage_beta, 0.0)

    def test_unknown_mode_fails_closed_at_compile(self) -> None:
        with self.assertRaises(ValueError):
            self._compile(film_interimage="banana")

    def test_unknown_mode_fails_closed_at_runtime(self) -> None:
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
            film_optics_seed=0, film_interimage="banana",
        )
        with self.assertRaises(ValueError):
            apply_film_core(np.full((4, 3), 0.18, dtype=np.float32), plan)


class ContinuityTests(unittest.TestCase):
    def test_saturation_gain_is_continuous_in_exposure(self) -> None:
        """A rail-clip shows up as a kink: saturation rises with exposure
        and then flatlines the moment a channel hits the boundary. The
        bounded map saturates smoothly instead, so the S transfer across a
        fine exposure sweep must have no step."""
        from tools.film_palette_probe import render_probe

        lab = np.array([[0.7, 0.12, 0.05]])
        base = np.maximum(pal.oklab_to_rec2020(lab)[0], 0)
        evs = np.linspace(-2.0, 2.0, 17)
        arr = np.asarray(
            [base * (0.18 * 2.0 ** ev / max(float(base @ LUMA), 1e-9))
             for ev in evs],
            dtype=np.float32,
        )
        out = render_probe(arr, "ektar100", "full")
        s = pal.decompose(out)["S"]
        steps = np.abs(np.diff(s))
        self.assertLess(
            float(steps.max()), 0.06,
            "saturation transfer has a cliff across the exposure sweep",
        )

    def test_the_family_spread_is_not_one_multiplier(self) -> None:
        """Portra vs Ektar must differ in SHAPE, not only in a uniform
        chroma multiplier (review item 5's honest framing: beta is a purity
        lever, so the probe must show at least hue-dependent structure in the
        difference)."""
        from tools.film_palette_probe import render_probe

        vol, idx = pal.palette_volume()
        wheel = idx.kind == "wheel"
        a = render_probe(vol, "portra400", "full")
        b = render_probe(vol, "ektar100", "full")
        d = pal.compare(a, b)
        ratios = d["log2_saturation_ratio"][wheel]
        by_octant = []
        for k in range(8):
            m = (idx.hue_deg[wheel] >= k * 45.0) & (idx.hue_deg[wheel] < (k + 1) * 45.0)
            vals = ratios[m]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                by_octant.append(float(np.median(vals)))
        self.assertGreater(
            float(np.ptp(by_octant)), 0.08,
            "the stock difference collapses to one uniform saturation gain",
        )


if __name__ == "__main__":
    unittest.main()
