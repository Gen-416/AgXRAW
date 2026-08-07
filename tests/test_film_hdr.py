# SPDX-License-Identifier: GPL-3.0-or-later
"""film v2 P6 gates (FILM_PRINT_RENDERING_PLAN §10/§12 P6).

胶片印相 + scene HDR 扩展: the SDR base is byte-identical to the standalone
film print, gain >= 1 everywhere with no negative gain in deep shadows, the
body below the reference-white join is untouched exactly, the gain curve is
C1, and the ceiling is the plan's solved reliable headroom. The observe HDR
path and the Ultra HDR file contract stay untouched (their own freezes).
"""
from __future__ import annotations

import unittest

import numpy as np

from tests.test_film_v2_assets import _stock_files


def _negative_stock() -> str:
    for stock in _stock_files():
        if stock.startswith(("portra", "pro400h", "c200", "gold")):
            return stock
    return _stock_files()[0]


class GainCurveTests(unittest.TestCase):
    def test_gain_contract_by_construction(self) -> None:
        from dngscan.film_v2_math import film_hdr_gain_log2

        ev = np.linspace(-6.0, 10.0, 3201)
        lg = film_hdr_gain_log2(ev, headroom_ev=2.0, join_ev=3.0, span_ev=3.0)
        # gain >= 1 everywhere; exactly 1 at/below the join (body invariance,
        # deep shadows included); capped at the headroom.
        self.assertTrue(np.all(lg >= 0.0))
        self.assertTrue(np.all(lg[ev <= 3.0] == 0.0))
        self.assertTrue(np.all(lg <= 2.0 + 1e-12))
        self.assertAlmostEqual(float(lg[-1]), 2.0, places=12)
        # monotone and C1: numeric derivative has no jumps at either end.
        d = np.diff(lg) / np.diff(ev)
        self.assertTrue(np.all(d >= -1e-12))
        # C1 = no derivative JUMP: successive difference-quotient steps are
        # bounded by the curve's own curvature times the sample step
        # (max |d2/dev2| = 6*h/span^2 here), never a discontinuity.
        step = float(ev[1] - ev[0])
        curvature_bound = 6.0 * 2.0 / (3.0 ** 2)
        self.assertLess(
            np.max(np.abs(np.diff(d))), 1.5 * curvature_bound * step,
            "join must be C1",
        )

    def test_reference_white_probe_is_sane(self) -> None:
        from types import SimpleNamespace

        from dngscan.film_develop import film_reference_white_ev

        base = dict(
            film_mode="full", film_crossover="off", film_exposure_ev=0.0,
            film_print_timing="fixed", film_print_medium="",
            film_print_exposure_ev=0.0, color_head_y=0.0, color_head_m=0.0,
            film_development="measured_default", film_dev_contrast=0.0,
            film_dev_fog=0.0, film_dev_density=0.0, film_compression=0.0,
            film_compression_knee=2.0, film_highlight_density=0.0,
            film_grain=0.0, film_halation=0.0, film_bloom=0.0,
            film_optics_seed=0,
        )
        for stock in (_negative_stock(), "velvia100"):
            e0 = film_reference_white_ev(
                SimpleNamespace(curve_preset=stock, **base)
            )
            self.assertGreater(e0, 1.0, stock)
            self.assertLess(e0, 5.5, stock)


class FilmPairTests(unittest.TestCase):
    def _pair(self, **plan_kw):
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
            film_crossover="datasheet", film_exposure_ev=1.5, **plan_kw,
        )
        target = HdrDisplayTarget(
            peak_nits=100.0 * float(2.0 ** float(DEFAULT_HDR_HEADROOM_EV))
        )
        hdr_plan = compile_hdr_agx_plan(
            plan, target, analysis=scene.analysis,
            scene_decoder=str(scene.bundle.scene_decoder),
        )
        base_u8, hdr_linear = render_ultrahdr_film_pair(
            scene.bundle, scene.analysis, plan, hdr_plan, "p3"
        )
        return scene, plan, hdr_plan, base_u8, hdr_linear

    def test_sdr_base_is_the_standalone_print(self) -> None:
        from dngscan.render import render_output_u8

        scene, plan, _, base_u8, hdr_linear = self._pair()
        standalone = render_output_u8(scene.bundle, scene.analysis, "p3", plan)
        np.testing.assert_array_equal(
            base_u8, standalone,
            "the pair's SDR base must be byte-identical to the SDR export",
        )
        self.assertFalse(np.isnan(hdr_linear).any())
        self.assertTrue(np.all(hdr_linear >= 0.0))

    def test_body_matches_sdr_and_highlights_gain(self) -> None:
        from dngscan.color import srgb_decode
        from dngscan.film_develop import film_reference_white_ev
        from dngscan.tone import scene_intent_rec2020

        scene, plan, hdr_plan, base_u8, hdr_linear = self._pair()
        # The §10 contract holds against the REAL encoded base (decoded from
        # the 8-bit dithered pixels the file carries), not the float print
        # (review batch 13: quantization noise broke gain >= 1 there).
        decoded = srgb_decode(
            base_u8.reshape(-1, 3).astype(np.float64) / 255.0
        ).reshape(hdr_linear.shape)
        self.assertGreaterEqual(
            float((hdr_linear - decoded).min()), -1e-6,
            "gain >= 1 must hold against the decoded real base everywhere "
            "(HDR never darker than the encoded print, black stays black)",
        )
        flat_scene = scene.bundle.scene_rec2020_render.reshape(-1, 3)
        rec = scene_intent_rec2020(flat_scene, scene.bundle)
        luma = np.array([0.2627, 0.6780, 0.0593])
        ev = np.log2(np.maximum(rec @ luma, 1e-9) / 0.18).reshape(hdr_linear.shape[:2])
        join = film_reference_white_ev(plan.tone)
        body = ev <= join
        np.testing.assert_allclose(
            hdr_linear[body], decoded[body], rtol=0, atol=1e-6,
            err_msg="below the reference-white join the HDR body IS the "
                    "decoded print",
        )
        hot = ev >= join + 0.5
        self.assertTrue(
            np.any(hot),
            "test scene must carry highlights above the join (the +1.5 EV "
            "emulsion overexposure moves the print's reference white down)",
        )
        hot_ratio = (
            hdr_linear[hot] / np.maximum(decoded[hot], 1e-9)
        )[decoded[hot] > 1e-3]
        self.assertGreater(
            float(hot_ratio.min()), 1.0,
            "reliable scene highlights must gain above reference white",
        )
        # never above the solved headroom
        self.assertTrue(np.all(
            hdr_linear <= decoded * (
                2.0 ** float(hdr_plan.tone.rendered_headroom_ev)
            ) + 1e-5
        ))

    def test_export_paths_accept_full_plus_hdr(self) -> None:
        # The CLI/service/export exclusions are gone: export_jpeg no longer
        # raises for full+ultrahdr (backend availability permitting).
        import inspect

        from dngscan import export as export_mod
        from dngscan.gui import service as service_mod

        self.assertNotIn("暂仅支持 SDR", inspect.getsource(export_mod))
        self.assertNotIn("暂仅支持 SDR", inspect.getsource(service_mod))


if __name__ == "__main__":
    unittest.main()
