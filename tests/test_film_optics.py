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

# P1 §7.1: the operators take the specific asset they implement, so the tests
# pull the same declared assets the renderer compiles rather than a shared
# profile struct that no longer exists.
from dngscan.film_optics_assets import (  # noqa: E402
    DEFAULT_PRINT_OPTICS,
    DEFAULT_STOCK_OPTICS,
    load_print_optics,
    load_stock_optics,
)

_GRAIN = load_stock_optics(DEFAULT_STOCK_OPTICS).grain
_HALATION = load_stock_optics(DEFAULT_STOCK_OPTICS).halation
_SCATTER = load_print_optics(DEFAULT_PRINT_OPTICS).print_scatter

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
        from dngscan.film_optics import _band_limited_field

        f = _band_limited_field(_GRAIN, 7)
        rms = np.sqrt(np.mean(np.square(f, dtype=np.float64), axis=(0, 1)))
        np.testing.assert_allclose(rms, 1.0, atol=1e-3)
        flat = f.reshape(-1, 3).astype(np.float64)
        corr = np.corrcoef(flat.T)
        self.assertAlmostEqual(
            corr[0, 1], _GRAIN.layer_corr, delta=0.05,
            msg="declared cross-layer covariance",
        )
        # band-limited, not white: strong positive neighbour correlation
        lag1 = np.mean(flat[:-1, 0] * flat[1:, 0])
        self.assertGreater(lag1, 0.5)
        f2 = _band_limited_field(_GRAIN, 7)
        np.testing.assert_array_equal(f, f2)
        f3 = _band_limited_field(_GRAIN, 8)
        self.assertFalse(np.allclose(f, f3))

    def test_shared_coordinate_sampling(self) -> None:
        from dngscan.film_optics import (
            FilmGeometry,
            grain_field_for,
            integral_from_field,
            sample_field,
        )

        # sample_field takes an INTEGRAL image by contract (batch 19): a raw
        # field silently produced different values (RMS 0.425 apart), and a
        # test that only compares scale/crop RELATIONS could not see it —
        # both sides were equally wrong (review batch 20).
        field = integral_from_field(grain_field_for(_GRAIN, 0))
        full = sample_field(field, FilmGeometry(400, 600))
        # half-resolution preview == block mean of the full sampling
        half = sample_field(field, FilmGeometry(200, 300))
        block = full.reshape(200, 2, 300, 2, 3).mean(axis=(1, 3))
        # float32 integral storage (review batch 14): identity holds to
        # integral precision, far below the grain sigma it modulates
        np.testing.assert_allclose(half, block, atol=2e-4)
        # a crop covers the corresponding region of the full frame
        crop = sample_field(
            field, FilmGeometry(200, 300, x0_mm=9.0, y0_mm=6.0, w_mm=18.0)
        )
        np.testing.assert_allclose(crop, full[100:300, 150:450], atol=2e-4)

    def test_sampling_matches_a_direct_area_mean(self) -> None:
        """Absolute truth, not just internal consistency: sampling a whole
        landscape frame at the grid's own resolution must reproduce the
        field itself. A raw field passed where an integral belongs fails
        this immediately (review batch 20 — the relation-only test above
        passed with both sides equally wrong)."""
        from dngscan.film_optics import (
            GATE_H_MM,
            GATE_W_MM,
            FilmGeometry,
            grain_field_for,
            integral_from_field,
            sample_field,
        )

        field = grain_field_for(_GRAIN, 0)
        gh, gw = field.shape[:2]
        got = sample_field(
            integral_from_field(field),
            FilmGeometry(gh, gw, w_mm=GATE_W_MM, h_mm=GATE_H_MM),
        )
        np.testing.assert_allclose(got, field, atol=2e-4)


class SpatialOperatorTests(unittest.TestCase):
    def test_identity_at_zero_amount(self) -> None:
        from dngscan.film_optics import (
            FilmGeometry,
            apply_density_grain,
            bloom_apply_rows,
            halation_reinject_rows,
        )

        g = FilmGeometry(20, 30)
        rng = np.random.default_rng(0)
        a = rng.uniform(0.2, 1.5, (600, 3))
        self.assertIs(
            apply_density_grain(a, np.zeros(3), np.ones(3) * 2, g,
                                _GRAIN, 0.0, 0), a,
        )
        le = rng.uniform(-1, 1, (600, 3))
        self.assertIs(
            halation_reinject_rows(
                le, None, np.full(3, 0.18), 0, 20, 20, 30, _HALATION, 0.0
            ),
            le,
        )
        img = rng.uniform(0, 1, (600, 3))
        self.assertIs(
            bloom_apply_rows(img, None, 0, 20, 20, 30, _SCATTER, 0.0), img
        )

    def test_halation_is_red_dominant_and_spreads(self) -> None:
        from dngscan.film_optics import (
            GATE_W_MM,
            halation_reinject_rows,
            halation_spread_map,
        )

        h, w = 64, 96
        # Layer exposure now, not one photometric luminance (R1 §5.2): the
        # source gate is per layer and the transfer matrix decides the colour.
        e_ref = np.full(3, 0.18, dtype=np.float32)
        e_lin = np.full((h, w, 3), np.float32(0.18 * 2 ** -2.0))
        e_lin[h // 2, w // 2] = np.float32(0.18 * 2 ** 6.0)
        le = np.log10(np.maximum(e_lin, 1e-12)).reshape(-1, 3)
        spread = halation_spread_map(e_lin, e_ref, GATE_W_MM, _HALATION)
        out = halation_reinject_rows(
            le, spread, e_ref, 0, h, h, w, _HALATION, 1.0
        )
        delta = (out - le).reshape(h, w, 3)
        near = delta[h // 2 - 2, w // 2, :]
        self.assertGreater(near[0], 0.0, "halation must spread beyond the source")
        self.assertGreater(near[0], near[1])
        self.assertGreater(near[1], near[2])
        # The residual form takes what it gives: the core loses light.
        core = delta[h // 2, w // 2, :]
        self.assertLess(float(core[0]), 0.0)

    def test_bloom_redistributes_highlights_conservatively(self) -> None:
        from dngscan.film_optics import (
            integral_from_field,
            bloom_apply_rows,
        )

        h, w = 64, 96
        img = np.full((h * w, 3), 0.05, dtype=np.float32)
        img[(h // 2) * w + w // 2] = 1.0
        from dngscan.film_optics import scatter_source, scatter_spread

        spread = scatter_spread(
            scatter_source(img.reshape(h, w, 3), _SCATTER), _SCATTER
        )
        ii = integral_from_field(spread).astype(np.float32)
        out = bloom_apply_rows(
            img, ii, 0, h, h, w, _SCATTER, 1.0
        ).reshape(h, w, 3)
        base = img.reshape(h, w, 3)
        # neighbourhood brightens, the CORE darkens (energy redistribution),
        # far corner nearly untouched, nothing negative
        self.assertGreater(out[h // 2 + 3, w // 2, 1], base[h // 2 + 3, w // 2, 1])
        self.assertLess(out[h // 2, w // 2, 1], base[h // 2, w // 2, 1])
        self.assertLess(abs(out[2, 2, 1] - base[2, 2, 1]), 5e-3)
        self.assertGreaterEqual(float(out.min()), 0.0)

    def test_grain_modulates_density_at_mid_not_extremes(self) -> None:
        from dngscan.film_optics import (
            FilmGeometry,
            apply_density_grain,
        )

        h, w = 40, 60
        g = FilmGeometry(h, w)
        lo, hi = np.zeros(3), np.full(3, 2.0)
        mid = np.full((h * w, 3), 1.0)
        toe = np.full((h * w, 3), 0.0)
        out_mid = apply_density_grain(mid, lo, hi, g, _GRAIN, 1.0, 0)
        out_toe = apply_density_grain(toe, lo, hi, g, _GRAIN, 1.0, 0)
        self.assertGreater(np.std(out_mid), 1e-3, "grain must act at mid density")
        self.assertLess(np.std(out_toe), 1e-9, "no grain at film base")
        again = apply_density_grain(mid, lo, hi, g, _GRAIN, 1.0, 0)
        np.testing.assert_array_equal(out_mid, again)


class RowBandEquivalenceTests(unittest.TestCase):
    """§9.3: the sequential row-band path must match the full-frame oracle
    exactly at every band split (shared spread-grid definition, coordinate
    grain, zero-halo seams)."""

    def test_bands_match_full_frame_for_every_operator_mix(self) -> None:
        from dngscan.film_develop import (
            apply_film_core,
            prepare_film_spatial,
        )
        from dngscan.film_optics import area_decimate, spread_grid_shape

        stock = _negative_stock()
        h, w = 48, 72
        rng = np.random.default_rng(9)
        img = rng.uniform(0.02, 0.5, (h, w, 3)).astype(np.float32)
        img[20:26, 30:38] = 6.0
        flat = img.reshape(-1, 3)
        plan = _plan(stock, film_grain=0.6, film_halation=0.7, film_bloom=0.5)
        full = apply_film_core(flat, plan, spatial_shape=(h, w))
        ctx = prepare_film_spatial(plan, h, w)
        dh, dw = spread_grid_shape(h, w)
        scene_dec = area_decimate(img, dh, dw)
        # P3 lifecycle: one scene pass. Bloom's finest rung is gated at full
        # resolution, then the halation source reads the BLOOMED scene.
        if ctx.bloom > 0.0:
            ctx.begin_bloom_source()
            ctx.accumulate_bloom_source(flat, 0, h)
            ctx.finish_bloom_map(scene_dec)
        if ctx.halation > 0.0:
            ctx.finish_maps(scene_dec, plan, stock)
        for band_rows in (5, 16, 48):
            out = np.empty_like(full)
            for y0 in range(0, h, band_rows):
                y1 = min(y0 + band_rows, h)
                out[y0 * w:y1 * w] = apply_film_core(
                    flat[y0 * w:y1 * w], plan, spatial=(ctx, y0, y1)
                )
            np.testing.assert_allclose(
                out, full, atol=2e-6,
                err_msg=f"band_rows={band_rows} must match the full-frame oracle",
            )


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


class RendererBandInvarianceTests(unittest.TestCase):
    def test_band_split_does_not_change_bytes(self) -> None:
        from unittest import mock

        from tests.golden_support import build_daylight_wide_dr
        from dngscan import render as render_mod
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        stock = _negative_stock()
        plan = build_render_plan(
            scene.bundle, scene.analysis, "agx", "srgb",
            film_curve=stock, film_mode="full", film_crossover="datasheet",
            film_grain=0.6, film_halation=0.5, film_bloom=0.4,
        )
        h, w = scene.bundle.scene_rec2020_render.shape[:2]
        rng = np.random.default_rng(42)
        noise = (
            rng.random((h, w, 3), dtype=np.float32),
            rng.random((h, w, 3), dtype=np.float32),
        )
        one_band = render_mod.render_output_u8(
            scene.bundle, scene.analysis, "srgb", tone_plan=plan,
            dither_noise=noise,
        )
        with mock.patch.object(render_mod, "_optics_band_rows", return_value=16):
            banded = render_mod.render_output_u8(
                scene.bundle, scene.analysis, "srgb", tone_plan=plan,
                dither_noise=noise,
            )
        np.testing.assert_array_equal(
            one_band, banded,
            "budget-solved band size must not change output bytes",
        )

    def test_display_linear_path_runs_banded(self) -> None:
        from unittest import mock

        from tests.golden_support import build_daylight_wide_dr
        from dngscan import render as render_mod
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        stock = _negative_stock()
        plan = build_render_plan(
            scene.bundle, scene.analysis, "agx", "srgb",
            film_curve=stock, film_mode="full", film_crossover="datasheet",
            film_grain=0.6,
        )
        full = render_mod.scene_render_to_display_linear(scene.bundle, plan)
        with mock.patch.object(render_mod, "_optics_band_rows", return_value=16):
            banded = render_mod.scene_render_to_display_linear(scene.bundle, plan)
        np.testing.assert_array_equal(full, banded)


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


class P5cServiceTests(unittest.TestCase):
    def _parse(self, **params):
        from dngscan.gui.service import parse_film_params

        base = {"filmCurve": "portra400", "filmMode": "full"}
        base.update(params)
        return parse_film_params(base)

    def test_tiers_map_to_declared_amounts(self) -> None:
        for tier, expect in (
            ("off", (0.0, 0.0, 0.0)),
            ("light", (0.25, 0.2, 0.15)),
            ("standard", (0.5, 0.4, 0.3)),
        ):
            got = self._parse(filmOptics=tier)[-3:]
            self.assertEqual(got, expect, tier)
        got = self._parse(
            filmOptics="custom", filmGrain=0.7, filmHalation=0.1, filmBloom=0.0
        )[-3:]
        self.assertEqual(got, (0.7, 0.1, 0.0))

    def test_service_contract_failures(self) -> None:
        with self.assertRaises(ValueError):
            self._parse(filmOptics="heavy")
        with self.assertRaises(ValueError):
            self._parse(filmOptics="custom", filmGrain=1.5)
        with self.assertRaises(ValueError):
            # observe mode must carry no optics payload
            self._parse(filmMode="observe", filmOptics="standard")

    def test_preview_plan_carries_the_film_state(self) -> None:
        """Regression: the preview's _cached_render_plan call sites truncated
        at film_crossover, so the P2/P3 exposure/timing/medium dials (and now
        the optics) never reached the preview plan while the export honoured
        them."""
        import inspect

        from dngscan.gui import service

        src = inspect.getsource(service)
        starts = [i for i in range(len(src)) if src.startswith("_cached_render_plan(", i)]
        call_sites = [s_ for s_ in starts if not src[:s_].rstrip().endswith("def")]
        self.assertGreaterEqual(len(call_sites), 2)
        for i in call_sites:
            window = src[i:i + 1400]
            for field in (
                "film_exposure_ev", "film_print_timing", "film_print_medium",
                "film_print_exposure_ev", "film_grain", "film_halation",
                "film_bloom",
            ):
                self.assertIn(
                    field, window,
                    f"_cached_render_plan call site must pass {field}",
                )

    def test_pixel_cache_key_covers_the_film_state(self) -> None:
        from dngscan.gui.service import _preview_pixel_key

        class _B:
            lens_filter = "none"

        common = dict(
            bundle=_B(), gamut="srgb", ev=0.0, look="none", look_strength=1.0,
            display_filter="none", filter_strength=1.0, scene_transform="none",
            scene_transform_strength=1.0, punch_scale=1.0, tone_core="agx",
            lum_norm="y", agx_primaries="base", lens_filter="none",
            film_curve="portra400", adjustments=None, film_mode="full",
        )
        base = _preview_pixel_key(**common)
        for kw in (
            dict(film_exposure_ev=1.0),
            dict(film_print_timing="retimed"),
            dict(film_print_medium="kodak_supra_endura__translated"),
            dict(film_grain=0.5),
            dict(film_halation=0.4),
            dict(film_bloom=0.3),
        ):
            self.assertNotEqual(
                base, _preview_pixel_key(**common, **kw),
                f"pixel cache key must vary with {kw}",
            )

    def test_page_contract(self) -> None:
        from dngscan.gui.page import render_page

        html = render_page("").decode("utf-8")
        for needle in (
            'id="filmOptics"', 'id="filmOpticsBlock"', 'id="filmOpticsCustom"',
            'id="filmGrain"', 'id="filmHalation"', 'id="filmBloom"',
            'value="light"', 'value="standard"',
        ):
            self.assertIn(needle, html)

    def test_export_suffix_names_the_optics(self) -> None:
        from dngscan.gui.service import export_suffix_parts

        clean = export_suffix_parts(
            "clip", "srgb", "sdr", film_mode="full",
        )
        optics = export_suffix_parts(
            "clip", "srgb", "sdr", film_mode="full",
            film_grain=0.5, film_halation=0.4, film_bloom=0.3,
        )
        self.assertNotEqual(clean, optics)
        self.assertIn("optics", optics)


if __name__ == "__main__":
    unittest.main()
