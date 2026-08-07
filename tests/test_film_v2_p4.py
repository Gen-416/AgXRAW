# SPDX-License-Identifier: GPL-3.0-or-later
"""film v2 P4 gates (FILM_PRINT_RENDERING_PLAN §6/§8/§12 P4).

Editorial developer recipe: bounded analytic perturbation of the
characteristic tables, mid-grey anchored by construction for the contrast and
colour-density terms; measured_default locks the deltas; retimed timing and
bounded neutralization are refused (their solutions assume measured
development). Film Compression: C1 at the knee, strict identity at impact 0,
luminance-EV geometry (no hue rotation), highlight colour density toward the
luminance-preserved neutral.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from tests.test_film_v2_assets import _stock_files


def _plan(preset: str, **kw):
    base = dict(
        curve_preset=preset, film_mode="full", film_crossover="datasheet",
        film_exposure_ev=0.0, film_print_timing="fixed",
        film_print_medium="", film_print_exposure_ev=0.0,
        color_head_y=0.0, color_head_m=0.0,
        film_development="measured_default",
        film_dev_contrast=0.0, film_dev_fog=0.0, film_dev_density=0.0,
        film_compression=0.0, film_compression_knee=2.0,
        film_highlight_density=0.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _negative_stock() -> str:
    for stock in _stock_files():
        if stock.startswith(("portra", "pro400h", "c200", "gold")):
            return stock
    return _stock_files()[0]


def _patches(evs) -> np.ndarray:
    return (0.18 * np.exp2(np.asarray(evs, dtype=np.float64)))[:, None].repeat(
        3, axis=1
    ).astype(np.float32)


class FilmCompressionMathTests(unittest.TestCase):
    def test_identity_below_knee_and_at_zero_impact(self) -> None:
        from dngscan.film_v2_math import film_compression_ev

        rgb = np.array([[0.05, 0.2, 0.4], [0.18, 0.18, 0.18]], dtype=np.float64)
        out0 = film_compression_ev(rgb, impact=0.0, knee_ev=1.0)
        np.testing.assert_array_equal(out0, rgb)
        # Below the knee the map is the identity branch exactly.
        out = film_compression_ev(rgb, impact=1.0, knee_ev=4.0)
        np.testing.assert_allclose(out, rgb, rtol=1e-12)

    def test_c1_continuity_at_the_knee(self) -> None:
        from dngscan.film_v2_math import film_compression_ev

        # Numeric derivative of x' (luminance EV out vs in) across the knee:
        # a C1 join means no jump in the difference quotient.
        evs = np.linspace(1.9, 2.1, 401)
        rgb = _patches(evs).astype(np.float64)
        out = film_compression_ev(rgb, impact=0.7, knee_ev=2.0)
        x_out = np.log2(out[:, 1] / 0.18)
        d = np.diff(x_out) / np.diff(evs)
        self.assertLess(np.max(np.abs(np.diff(d))), 5e-3, "knee join is not C1")

    def test_highlights_compress_monotonically_and_mid_stays(self) -> None:
        from dngscan.film_v2_math import film_compression_ev

        rgb = _patches([0.0, 3.0, 5.0]).astype(np.float64)
        out = film_compression_ev(rgb, impact=0.8, knee_ev=1.0)
        np.testing.assert_allclose(out[0], rgb[0], rtol=1e-12)
        self.assertLess(out[1, 1], rgb[1, 1])
        self.assertLess(out[2, 1], rgb[2, 1])
        # Order preserved: still monotone in exposure.
        self.assertLess(out[1, 1], out[2, 1])

    def test_highlight_colour_density_pulls_chroma_not_hue(self) -> None:
        from dngscan.film_v2_math import film_compression_ev

        luma = np.array([0.2627, 0.6780, 0.0593])
        rgb = np.array([[4.0, 2.0, 1.0]], dtype=np.float64)  # warm highlight
        plain = film_compression_ev(rgb, impact=0.9, knee_ev=0.5)
        dense = film_compression_ev(
            rgb, impact=0.9, knee_ev=0.5, highlight_color_density=1.5
        )
        y_plain = float((plain @ luma)[0])
        y_dense = float((dense @ luma)[0])
        self.assertAlmostEqual(y_plain, y_dense, places=9, msg="rho must not move Y")
        spread = np.ptp(dense / y_dense)
        self.assertLess(spread, np.ptp(plain / y_plain), "chroma must shrink")
        # Hue direction preserved: channel ordering unchanged.
        self.assertTrue(np.all(np.argsort(dense[0]) == np.argsort(plain[0])))


class DeveloperRecipeTests(unittest.TestCase):
    def test_measured_default_is_the_untouched_table(self) -> None:
        from dngscan.film_v2_math import developer_perturbation

        le = np.linspace(-2, 2, 41)
        t = np.stack([0.2 + 1.1 / (1 + np.exp(-2.0 * le))] * 3, axis=1)
        out = developer_perturbation(le, t)
        np.testing.assert_array_equal(out, t)

    def test_contrast_and_density_keep_the_mid_anchor(self) -> None:
        from dngscan.film_v2_math import developer_perturbation

        le = np.linspace(-2, 2, 81)
        t = np.stack(
            [0.2 + (1.0 + 0.1 * c) / (1 + np.exp(-2.2 * (le + 0.1 * c)))
             for c in range(3)],
            axis=1,
        )
        out = developer_perturbation(le, t, contrast_delta=0.3, color_density=-0.2)
        for c in range(3):
            self.assertAlmostEqual(
                float(np.interp(0.0, le, out[:, c])),
                float(np.interp(0.0, le, t[:, c])),
                places=9,
                msg="mid-grey amount must be anchored by construction",
            )
        # Monotone preserved.
        self.assertTrue(np.all(np.diff(out, axis=0) >= -1e-12))

    def test_contrast_steepens_and_fog_lifts(self) -> None:
        from dngscan.film_v2_math import developer_perturbation

        le = np.linspace(-2, 2, 81)
        t = np.stack([0.2 + 1.1 / (1 + np.exp(-1.8 * le))] * 3, axis=1)
        hi = developer_perturbation(le, t, contrast_delta=0.4)
        spread = lambda a: (
            np.interp(1.0, le, a[:, 0]) - np.interp(-1.0, le, a[:, 0])
        )
        self.assertGreater(spread(hi), spread(t))
        foggy = developer_perturbation(le, t, fog_delta=0.2)
        np.testing.assert_allclose(foggy, t + 0.2, rtol=0, atol=1e-12)


class P4RuntimeTests(unittest.TestCase):
    def test_compression_identity_at_zero_and_effect_above_knee(self) -> None:
        from dngscan.film_develop import apply_film_core

        stock = _negative_stock()
        rgb = _patches([-1.0, 0.0, 1.0, 3.0, 4.5])
        base = apply_film_core(rgb, _plan(stock))
        again = apply_film_core(rgb, _plan(stock, film_compression=0.0))
        np.testing.assert_array_equal(base, again)
        squeezed = apply_film_core(
            rgb, _plan(stock, film_compression=0.8, film_compression_knee=1.0)
        )
        luma = np.array([0.2627, 0.6780, 0.0593])
        # Below/at the knee untouched. Compression pulls extreme scene EV
        # down -> a LESS dense negative -> more paper exposure -> the print
        # recovers density (darker, detailed) where it was paper-white.
        np.testing.assert_allclose(base[:2], squeezed[:2], rtol=1e-5)
        self.assertLess(
            float(squeezed[4] @ luma), float(base[4] @ luma) + 1e-6
        )
        self.assertLess(
            abs(float(np.log2(max(float(squeezed[4] @ luma), 1e-9)))
                - float(np.log2(max(float(base[2] @ luma), 1e-9)))),
            abs(float(np.log2(max(float(base[4] @ luma), 1e-9)))
                - float(np.log2(max(float(base[2] @ luma), 1e-9)))),
            "compressed +4.5 EV must land closer to the +1 EV print level",
        )

    def test_editorial_contrast_changes_render_but_keeps_mid(self) -> None:
        from dngscan.film_develop import apply_film_core

        stock = _negative_stock()
        rgb = _patches([-1.5, 0.0, 1.5])
        base = apply_film_core(rgb, _plan(stock))
        pushed = apply_film_core(
            rgb,
            _plan(
                stock,
                film_development="editorial_custom",
                film_dev_contrast=0.3,
            ),
        )
        luma = np.array([0.2627, 0.6780, 0.0593])
        mid_shift = abs(
            np.log2(max(float(pushed[1] @ luma), 1e-9))
            - np.log2(max(float(base[1] @ luma), 1e-9))
        )
        self.assertLess(mid_shift, 0.08, "mid-grey print level must hold")
        spread = lambda o: np.log2(max(float(o[2] @ luma), 1e-9)) - np.log2(
            max(float(o[0] @ luma), 1e-9)
        )
        self.assertGreater(
            abs(spread(pushed)), abs(spread(base)),
            "push development must widen the printed spread",
        )

    def test_fog_lightens_the_negative_print(self) -> None:
        from dngscan.film_develop import apply_film_core

        stock = _negative_stock()
        rgb = _patches([-1.0, 0.0])
        base = apply_film_core(rgb, _plan(stock))
        foggy = apply_film_core(
            rgb,
            _plan(stock, film_development="editorial_custom", film_dev_fog=0.15),
        )
        luma = np.array([0.2627, 0.6780, 0.0593])
        for i in range(2):
            self.assertGreater(
                float(foggy[i] @ luma), float(base[i] @ luma),
                "fog adds negative density -> lighter print",
            )


class P4PlanContractTests(unittest.TestCase):
    def _compile(self, **kw):
        from dngscan.film_plans import (
            AnalogFinishPlan,
            FilmDevelopmentPlan,
            FilmExposurePlan,
            FilmPrintPlan,
            validate_film_plans,
        )

        dev = {k[4:]: v for k, v in kw.items() if k.startswith("dev_")}
        prn = {k[6:]: v for k, v in kw.items() if k.startswith("print_")}
        fin = {k[7:]: v for k, v in kw.items() if k.startswith("finish_")}
        validate_film_plans(
            FilmExposurePlan(stock_id="portra400"),
            FilmDevelopmentPlan(**dev),
            FilmPrintPlan(medium_id="print_paper", **prn),
            AnalogFinishPlan(**fin),
        )

    def test_measured_default_locks_deltas(self) -> None:
        with self.assertRaises(ValueError):
            self._compile(dev_contrast_delta=0.1)

    def test_editorial_bounds_are_hard(self) -> None:
        ok = dict(
            dev_recipe_id="editorial_custom",
            print_neutralization_policy="datasheet",
        )
        self._compile(**ok, dev_contrast_delta=0.5)
        for bad in (
            dict(dev_contrast_delta=0.6),
            dict(dev_fog_delta=-0.1),
            dict(dev_fog_delta=0.4),
            dict(dev_color_density=0.7),
        ):
            with self.assertRaises(ValueError):
                self._compile(**ok, **bad)

    def test_editorial_refuses_retimed_and_bounded(self) -> None:
        with self.assertRaises(ValueError):
            self._compile(
                dev_recipe_id="editorial_custom",
                print_neutralization_policy="datasheet",
                print_timing_policy="retimed",
            )
        with self.assertRaises(ValueError):
            self._compile(dev_recipe_id="editorial_custom")  # bounded default

    def test_compression_domains(self) -> None:
        self._compile(finish_compression=0.5, finish_compression_knee_ev=2.0)
        for bad in (
            dict(finish_compression=1.5),
            dict(finish_compression=0.5, finish_compression_knee_ev=7.0),
            dict(finish_highlight_color_density=2.5),
        ):
            with self.assertRaises(ValueError):
                self._compile(**bad)

    def test_tone_compiler_gates_the_dials(self) -> None:
        from tests.golden_support import build_daylight_wide_dr
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        args = (scene.bundle, scene.analysis, "agx", "srgb")
        # observe mode refuses the P4 dials; so does the no-preset case.
        for kw in (
            dict(film_curve="portra400", film_mode="observe",
                 film_development="editorial_custom"),
            dict(film_curve="portra400", film_mode="observe",
                 film_compression=0.3),
            dict(film_compression=0.3),
        ):
            with self.assertRaises(ValueError):
                build_render_plan(*args, **kw)
        # the valid full-mode declaration compiles and stamps.
        plan = build_render_plan(
            *args,
            film_curve="portra400", film_mode="full",
            film_crossover="datasheet",
            film_development="editorial_custom",
            film_dev_contrast=0.2,
            film_compression=0.4,
            film_highlight_density=0.5,
        )
        self.assertEqual(plan.tone.film_development, "editorial_custom")
        self.assertEqual(plan.tone.film_dev_contrast, 0.2)
        self.assertEqual(plan.tone.film_compression, 0.4)
        self.assertEqual(plan.film[1].recipe_id, "editorial_custom")
        self.assertEqual(plan.film[1].provenance, "editorial")
        self.assertEqual(plan.film[3].compression, 0.4)
        # editorial under the bounded default fails at the plan gate.
        with self.assertRaises(ValueError):
            build_render_plan(
                *args,
                film_curve="portra400", film_mode="full",
                film_development="editorial_custom",
                film_dev_contrast=0.2,
            )


if __name__ == "__main__":
    unittest.main()
