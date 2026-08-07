# SPDX-License-Identifier: GPL-3.0-or-later
"""film v2 P5a gates (FILM_PRINT_RENDERING_PLAN §9/§12 P5, first batch).

The film-space grain contract: one deterministic band-limited field in
negative mm coordinates; preview/crop/full-size renderings sample the SAME
realization by area integration (half resolution == block mean of full, crop
== the corresponding region). Halation reinjects red-heavy backscatter into
layer exposure before the curves; medium bloom spreads the print's own
highlights after formation. Everything at 0 is a strict identity, including
through the full renderer (chunk-stream fast path preserved).
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


class GrainFieldContractTests(unittest.TestCase):
    def test_field_statistics_and_determinism(self) -> None:
        from dngscan.film_optics import MODELLED_DEFAULT, _band_limited_field

        f = _band_limited_field(MODELLED_DEFAULT, 7)
        rms = np.sqrt(np.mean(np.square(f, dtype=np.float64), axis=(0, 1)))
        np.testing.assert_allclose(rms, 1.0, atol=1e-3)
        flat = f.reshape(-1, 3).astype(np.float64)
        corr = np.corrcoef(flat.T)
        self.assertAlmostEqual(
            corr[0, 1], MODELLED_DEFAULT.grain_layer_corr, delta=0.05,
            msg="declared cross-layer covariance",
        )
        # band-limited, not white: strong positive neighbour correlation
        lag1 = np.mean(flat[:-1, 0] * flat[1:, 0])
        self.assertGreater(lag1, 0.5)
        f2 = _band_limited_field(MODELLED_DEFAULT, 7)
        np.testing.assert_array_equal(f, f2)
        f3 = _band_limited_field(MODELLED_DEFAULT, 8)
        self.assertFalse(np.allclose(f, f3))

    def test_shared_coordinate_sampling(self) -> None:
        from dngscan.film_optics import (
            FilmGeometry,
            MODELLED_DEFAULT,
            grain_field_for,
            sample_field,
        )

        field = grain_field_for(MODELLED_DEFAULT, 0)
        full = sample_field(field, FilmGeometry(400, 600))
        # half-resolution preview == block mean of the full sampling
        half = sample_field(field, FilmGeometry(200, 300))
        block = full.reshape(200, 2, 300, 2, 3).mean(axis=(1, 3))
        np.testing.assert_allclose(half, block, atol=2e-6)
        # a crop covers the corresponding region of the full frame
        crop = sample_field(
            field, FilmGeometry(200, 300, x0_mm=9.0, y0_mm=6.0, w_mm=18.0)
        )
        np.testing.assert_allclose(crop, full[100:300, 150:450], atol=2e-6)


class SpatialOperatorTests(unittest.TestCase):
    def test_identity_at_zero_amount(self) -> None:
        from dngscan.film_optics import (
            FilmGeometry,
            MODELLED_DEFAULT,
            apply_density_grain,
            halation_reinject,
            medium_bloom,
        )

        g = FilmGeometry(20, 30)
        rng = np.random.default_rng(0)
        a = rng.uniform(0.2, 1.5, (600, 3))
        self.assertIs(
            apply_density_grain(a, np.zeros(3), np.ones(3) * 2, g,
                                MODELLED_DEFAULT, 0.0, 0), a,
        )
        le = rng.uniform(-1, 1, (600, 3))
        self.assertIs(
            halation_reinject(le, np.zeros(600), g, MODELLED_DEFAULT, 0.0), le
        )
        img = rng.uniform(0, 1, (600, 3))
        self.assertIs(medium_bloom(img, g, MODELLED_DEFAULT, 0.0), img)

    def test_halation_is_red_dominant_and_spreads(self) -> None:
        from dngscan.film_optics import FilmGeometry, MODELLED_DEFAULT, halation_reinject

        h, w = 64, 96
        g = FilmGeometry(h, w)
        le = np.full((h * w, 3), -0.5)
        ev_y = np.full(h * w, -2.0)
        # one hot highlight in the centre
        centre = (h // 2) * w + w // 2
        ev_y[centre] = 5.0
        out = halation_reinject(le, ev_y, g, MODELLED_DEFAULT, 1.0)
        delta = (out - le).reshape(h, w, 3)
        # the neighbourhood (not just the source pixel) gains exposure
        near = delta[h // 2 - 2, w // 2, :]
        self.assertGreater(near[0], 0.0, "halation must spread beyond the source")
        # red layer gains most, blue least (red-sensitive backscatter)
        self.assertGreater(near[0], near[1])
        self.assertGreater(near[1], near[2])

    def test_bloom_spreads_highlights_softly(self) -> None:
        from dngscan.film_optics import FilmGeometry, MODELLED_DEFAULT, medium_bloom

        h, w = 64, 96
        g = FilmGeometry(h, w)
        img = np.full((h * w, 3), 0.05, dtype=np.float32)
        img[(h // 2) * w + w // 2] = 1.0
        out = medium_bloom(img, g, MODELLED_DEFAULT, 1.0).reshape(h, w, 3)
        base = img.reshape(h, w, 3)
        # neighbourhood brightens; far corner nearly untouched
        self.assertGreater(out[h // 2 + 3, w // 2, 1], base[h // 2 + 3, w // 2, 1])
        self.assertLess(out[2, 2, 1] - base[2, 2, 1], 5e-3)

    def test_grain_modulates_density_at_mid_not_extremes(self) -> None:
        from dngscan.film_optics import (
            FilmGeometry,
            MODELLED_DEFAULT,
            apply_density_grain,
        )

        h, w = 40, 60
        g = FilmGeometry(h, w)
        lo, hi = np.zeros(3), np.full(3, 2.0)
        mid = np.full((h * w, 3), 1.0)
        toe = np.full((h * w, 3), 0.0)
        out_mid = apply_density_grain(mid, lo, hi, g, MODELLED_DEFAULT, 1.0, 0)
        out_toe = apply_density_grain(toe, lo, hi, g, MODELLED_DEFAULT, 1.0, 0)
        self.assertGreater(np.std(out_mid), 1e-3, "grain must act at mid density")
        self.assertLess(np.std(out_toe), 1e-9, "no grain at film base")
        # deterministic: same seed, same result
        again = apply_density_grain(mid, lo, hi, g, MODELLED_DEFAULT, 1.0, 0)
        np.testing.assert_array_equal(out_mid, again)


class SpatialRuntimeTests(unittest.TestCase):
    def _rgb_image(self, h: int = 48, w: int = 72) -> np.ndarray:
        rng = np.random.default_rng(3)
        img = rng.uniform(0.02, 0.5, (h, w, 3)).astype(np.float32)
        img[h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2] = 6.0
        return img

    def test_flat_caller_and_zero_amounts_are_identity(self) -> None:
        from dngscan.film_develop import apply_film_core

        stock = _negative_stock()
        img = self._rgb_image()
        flat = img.reshape(-1, 3)
        base = apply_film_core(flat, _plan(stock))
        # amounts on but NO spatial shape: optics inert by contract
        probe = apply_film_core(flat, _plan(stock, film_grain=0.5))
        np.testing.assert_array_equal(base, probe)
        # spatial shape given but all amounts zero: same bytes
        spatial = apply_film_core(flat, _plan(stock), spatial_shape=img.shape[:2])
        np.testing.assert_array_equal(base, spatial)

    def test_spatial_ops_engage_with_shape(self) -> None:
        from dngscan.film_develop import apply_film_core

        stock = _negative_stock()
        img = self._rgb_image()
        flat = img.reshape(-1, 3)
        base = apply_film_core(flat, _plan(stock))
        for kw in (
            dict(film_grain=0.6),
            dict(film_halation=0.8),
            dict(film_bloom=0.8),
        ):
            with self.subTest(**kw):
                out = apply_film_core(
                    flat, _plan(stock, **kw), spatial_shape=img.shape[:2]
                )
                self.assertFalse(np.array_equal(base, out))
                self.assertFalse(np.isnan(out).any())
        # grain seed reproducibility at the core level
        a = apply_film_core(
            flat, _plan(stock, film_grain=0.6, film_optics_seed=5),
            spatial_shape=img.shape[:2],
        )
        b = apply_film_core(
            flat, _plan(stock, film_grain=0.6, film_optics_seed=5),
            spatial_shape=img.shape[:2],
        )
        np.testing.assert_array_equal(a, b)
        c = apply_film_core(
            flat, _plan(stock, film_grain=0.6, film_optics_seed=6),
            spatial_shape=img.shape[:2],
        )
        self.assertFalse(np.array_equal(a, c))

    def test_renderer_routes_full_frame_and_keeps_fast_path(self) -> None:
        from tests.golden_support import build_daylight_wide_dr
        from dngscan.render import render_output_u8
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        stock = _negative_stock()

        def render(**kw):
            plan = build_render_plan(
                scene.bundle, scene.analysis, "agx", "srgb",
                film_curve=stock, film_mode="full",
                film_crossover="datasheet", **kw,
            )
            return render_output_u8(
                scene.bundle, scene.analysis, "srgb", tone_plan=plan
            )

        base = render()
        again = render(film_grain=0.0, film_halation=0.0, film_bloom=0.0)
        np.testing.assert_array_equal(base, again, "all-off must stay byte-identical")
        grained = render(film_grain=0.7)
        self.assertFalse(np.array_equal(base, grained))
        grained2 = render(film_grain=0.7)
        np.testing.assert_array_equal(grained, grained2, "fixed seed reproducible")


class P5PlanContractTests(unittest.TestCase):
    def test_tone_compiler_gates_optics(self) -> None:
        from tests.golden_support import build_daylight_wide_dr
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        args = (scene.bundle, scene.analysis, "agx", "srgb")
        for kw in (
            dict(film_curve="portra400", film_mode="observe", film_grain=0.5),
            dict(film_bloom=0.5),
        ):
            with self.subTest(**kw), self.assertRaises(ValueError):
                build_render_plan(*args, **kw)
        plan = build_render_plan(
            *args, film_curve="portra400", film_mode="full",
            film_grain=0.4, film_halation=0.3, film_bloom=0.2,
            film_optics_seed=11,
        )
        finish = plan.film[3]
        self.assertEqual(finish.grain_profile, "modelled_default")
        self.assertEqual(finish.grain_amount, 0.4)
        self.assertEqual(finish.halation_profile, "modelled_default")
        self.assertEqual(finish.bloom_amount, 0.2)
        self.assertEqual(finish.seed, 11)
        from dngscan.film_plans import is_identity_finish
        self.assertFalse(is_identity_finish(finish))

    def test_amount_domains_are_hard(self) -> None:
        from tests.golden_support import build_daylight_wide_dr
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        with self.assertRaises(ValueError):
            build_render_plan(
                scene.bundle, scene.analysis, "agx", "srgb",
                film_curve="portra400", film_mode="full", film_grain=1.5,
            )


if __name__ == "__main__":
    unittest.main()
