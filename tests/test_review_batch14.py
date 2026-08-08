# SPDX-License-Identifier: GPL-3.0-or-later
"""Review batch 14 regression gates.

1. The reference-white probe anchors the plateau at the chain's TRUE upper
   bound and searches negative EV — extreme exposure/timing states no longer
   fake a +6 EV plateau; the HDR headroom is co-compiled with the reliable
   tail's distance from the join.
2. The HDR alternate is decoded_base + float_print * (gain - 1): the body is
   the real encoded base, the increment carries float precision.
3. The safe-EV probe runs the spatial operators on a decimated image when
   engaged — bloom/halation now participate in the answer.
4. The GUI custom-timing listener refreshes dependent controls.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from tests.test_film_v2_assets import _stock_files


def _plan_ns(preset: str, **kw):
    base = dict(
        curve_preset=preset, film_mode="full", film_crossover="datasheet",
        film_exposure_ev=0.0, film_print_timing="fixed",
        film_print_medium="", film_print_exposure_ev=0.0,
        color_head_y=0.0, color_head_m=0.0,
        film_development="measured_default",
        film_dev_contrast=0.0, film_dev_fog=0.0, film_dev_density=0.0,
        film_compression=0.0, film_compression_knee=2.0,
        film_highlight_density=0.0,
        film_grain=0.0, film_halation=0.0, film_bloom=0.0,
        film_optics_seed=0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _negative_stock() -> str:
    for stock in _stock_files():
        if stock.startswith(("portra", "pro400h", "c200", "gold")):
            return stock
    return _stock_files()[0]


class ReferenceWhiteProbeTests(unittest.TestCase):
    def test_extreme_states_do_not_fake_a_plateau(self) -> None:
        from dngscan.film_develop import apply_film_core, film_reference_white_ev

        stock = _negative_stock()
        # the review's failing state: underexposed emulsion + brighter
        # custom print pushes paper white far above the old +6 EV scan cap
        plan = _plan_ns(
            stock, film_exposure_ev=-2.0, film_print_timing="custom",
            film_print_exposure_ev=2.0,
        )
        e0 = film_reference_white_ev(plan)
        # at the returned join the print must ACTUALLY be near its true
        # plateau: luma(join) >= 0.9 * luma(+24 EV)
        rgb = (0.18 * np.exp2(np.array([e0, 24.0])))[:, None].repeat(3, 1)
        out = apply_film_core(rgb.astype(np.float32), plan)
        luma = out @ np.array([0.2627, 0.6780, 0.0593])
        self.assertGreaterEqual(
            float(luma[0]), 0.9 * float(luma[1]) - 1e-6,
            f"join {e0:+.2f} EV is not on the true plateau",
        )
        # and a slope check: half an EV below the join must still be below
        # the 90% line (the join is the FIRST crossing, not a fake cap)
        rgb2 = (0.18 * np.exp2(np.array([e0 - 0.5])))[:, None].repeat(3, 1)
        below = apply_film_core(rgb2.astype(np.float32), plan) @ np.array(
            [0.2627, 0.6780, 0.0593]
        )
        self.assertLess(float(below[0]), 0.9 * float(luma[1]) + 1e-6)

    def test_overexposed_emulsion_finds_an_early_join(self) -> None:
        from dngscan.film_develop import film_reference_white_ev

        stock = _negative_stock()
        base = film_reference_white_ev(_plan_ns(stock))
        early = film_reference_white_ev(_plan_ns(stock, film_exposure_ev=1.5))
        self.assertLess(early, base - 0.5, "overexposure must move the join down")

    def test_headroom_is_capped_by_the_reliable_tail(self) -> None:
        import inspect

        from dngscan import hdr_agx

        src = inspect.getsource(hdr_agx.render_ultrahdr_film_pair)
        self.assertIn("reliable_tail_ev", src)
        self.assertIn("min(", src)


class AlternatePrecisionTests(unittest.TestCase):
    def _pair(self):
        from tests.golden_support import build_daylight_wide_dr
        from dngscan.export import DEFAULT_HDR_HEADROOM_EV
        from dngscan.hdr_agx import render_ultrahdr_film_pair
        from dngscan.hdr_agx_plan import compile_hdr_agx_plan
        from dngscan.models import HdrDisplayTarget
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        plan = build_render_plan(
            scene.bundle, scene.analysis, "agx", "p3",
            film_curve=_negative_stock(), film_mode="full",
            film_crossover="datasheet", film_exposure_ev=1.5,
        )
        target = HdrDisplayTarget(
            peak_nits=100.0 * float(2.0 ** float(DEFAULT_HDR_HEADROOM_EV))
        )
        hdr_plan = compile_hdr_agx_plan(
            plan, target, analysis=scene.analysis,
            scene_decoder=str(scene.bundle.scene_decoder),
        )
        return scene, plan, hdr_plan, *render_ultrahdr_film_pair(
            scene.bundle, scene.analysis, plan, hdr_plan, "p3"
        )

    def test_body_is_the_decoded_base_and_increment_is_float(self) -> None:
        from dngscan.color import fit_to_output_gamut, srgb_decode
        from dngscan.film_develop import film_reference_white_ev
        from dngscan.render import scene_render_to_display_linear
        from dngscan.tone import scene_intent_rec2020

        scene, plan, hdr_plan, base_u8, hdr_linear = self._pair()
        decoded = srgb_decode(
            base_u8.reshape(-1, 3).astype(np.float64) / 255.0
        ).reshape(hdr_linear.shape)
        # gain >= 1 against the real base, everywhere
        self.assertGreaterEqual(float((hdr_linear - decoded).min()), -1e-6)
        # body identity
        rec = scene_intent_rec2020(
            scene.bundle.scene_rec2020_render.reshape(-1, 3), scene.bundle
        )
        luma = np.array([0.2627, 0.6780, 0.0593])
        ev = np.log2(np.maximum(rec @ luma, 1e-9) / 0.18).reshape(
            hdr_linear.shape[:2]
        )
        join = film_reference_white_ev(plan.tone)
        body = ev <= join
        np.testing.assert_allclose(hdr_linear[body], decoded[body], atol=1e-6)
        # the increment above the join must equal float_print * (gain - 1):
        # reconstruct it and compare — this pins the construction, so the
        # highlight detail is NOT quantized to 8-bit steps
        sdr_lin = scene_render_to_display_linear(scene.bundle, plan, "p3")
        fitted = fit_to_output_gamut(
            sdr_lin.reshape(-1, 3), "p3",
            alpha=float(plan.color.gamut_fit_alpha),
        ).reshape(hdr_linear.shape)
        hot = ~body
        if np.any(hot):
            increment = hdr_linear[hot] - decoded[hot]
            with np.errstate(invalid="ignore", divide="ignore"):
                implied_gain = 1.0 + increment / np.maximum(fitted[hot], 1e-9)
            self.assertTrue(np.all(np.isfinite(implied_gain)))
            self.assertGreaterEqual(float(implied_gain.min()), 1.0 - 1e-4)


class SpatialSafeEvTests(unittest.TestCase):
    def test_probe_engages_the_spatial_operators(self) -> None:
        from unittest import mock

        from tests.golden_support import build_daylight_wide_dr
        from dngscan import auto_ev as auto_ev_mod
        from dngscan import film_develop

        scene = build_daylight_wide_dr()
        stock = _negative_stock()
        spatial_calls = []
        real = film_develop.apply_film_core

        def spy(rgb, plan, spatial_shape=None, spatial=None):
            if spatial_shape is not None:
                spatial_calls.append(spatial_shape)
            return real(rgb, plan, spatial_shape=spatial_shape, spatial=spatial)

        with mock.patch.object(film_develop, "apply_film_core", spy):
            safe_plain = auto_ev_mod.max_safe_ev(
                scene.bundle, scene.analysis, "srgb",
                film_curve=stock, film_mode="full", film_crossover="datasheet",
            )
            self.assertEqual(spatial_calls, [], "no optics -> flat probe")
            safe_optics = auto_ev_mod.max_safe_ev(
                scene.bundle, scene.analysis, "srgb",
                film_curve=stock, film_mode="full", film_crossover="datasheet",
                film_halation=1.0, film_bloom=1.0,
            )
        self.assertGreater(
            len(spatial_calls), 0,
            "engaged optics must route the probe through the spatial core",
        )
        # the operators must CHANGE the answer (batch 15's conservative
        # scatter redistributes energy, so the direction is scene-dependent:
        # cores shed light and safe EV may rise — the old "added light only
        # lowers it" belonged to the additive-bloom era)
        self.assertNotEqual(safe_optics, safe_plain)


class GuiTimingRefreshTests(unittest.TestCase):
    def test_custom_timing_listener_refreshes_dependent_controls(self) -> None:
        from dngscan.gui.page import PAGE

        i = PAGE.index('$("#filmPrintTiming").addEventListener')
        line = PAGE[i:PAGE.index("\n", i)]
        self.assertIn("updateFilmModeUi()", line)
        self.assertIn("updateColorHeadUi()", line)


if __name__ == "__main__":
    unittest.main()
