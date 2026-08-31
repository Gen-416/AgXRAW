# SPDX-License-Identifier: GPL-3.0-or-later
"""Halation fine-source gates (P5f, review 2026-08-31).

The residual reinject subtracts its pointwise A@U at FULL resolution while
the spread map used to gate the DECIMATED proxy. The gate is a smootherstep
on log exposure, so gate(decimate(E)) != decimate(gate(E)): a source smaller
than a decimated cell fell below the gate on the map side while the
subtraction still charged it — measured 100% of the given energy destroyed
(no halo anywhere) and the source pixel down 7.7%/3.0% of its own R/G layer
exposure at amount 1, on exports only (identity grids balanced to 0.1%).
P5f gates at full resolution inside pass A's band loop and area-decimates
the per-component transferred sources, exactly the bloom fine-rung pattern;
the spread stays on the decimated grid, and identity grids keep the classic
path byte for byte.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from dngscan.film_optics_assets import DEFAULT_STOCK_OPTICS, load_stock_optics

_HALATION = load_stock_optics(DEFAULT_STOCK_OPTICS).halation


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
    from tests.test_film_v2_assets import _stock_files

    for stock in _stock_files():
        if stock.startswith(("portra", "pro400h", "c200", "gold")):
            return stock
    return _stock_files()[0]


class _ForcedDecimation:
    """Shrink the spread grid so a small test frame decimates (the defect
    only exists when the grid is smaller than the image)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def __enter__(self):
        from dngscan import film_optics

        self._saved = film_optics.spread_max_dim
        film_optics.spread_max_dim = lambda: self.limit
        return self

    def __exit__(self, *exc):
        from dngscan import film_optics

        film_optics.spread_max_dim = self._saved
        return False


class FineSourceEnergyTests(unittest.TestCase):
    def test_give_and_take_balance_for_a_sub_cell_source(self) -> None:
        """The energy the pointwise subtraction charges a sub-cell source
        must come back through the spread map — the classic decimated gate
        destroyed all of it."""
        from dngscan.film_optics import (
            area_decimate,
            area_decimate_rows,
            halation_component_source,
            halation_pointwise_return,
            halation_spread_map,
            halation_spread_map_from_sources,
            upsample_rows,
        )

        ref = np.ones(3, dtype=np.float32)
        h = w = 448
        geo_w_mm = 36.0 * (w / 6016)
        factor = 7
        dh, dw = h // factor, w // factor
        e = np.ones((h, w, 3), dtype=np.float32)
        e[h // 2, w // 2, :] = 2.0 ** 6.5  # 1-px white source, above every gate

        give = np.minimum(halation_pointwise_return(e, ref, _HALATION), e)
        g = give.reshape(-1, 3).sum(axis=0)
        self.assertGreater(g[0], 1.0, "the probe source must be charged")

        # classic decimated gate: the averaged cell falls below the gate
        classic = halation_spread_map(
            area_decimate(e, dh, dw), ref, geo_w_mm, _HALATION
        )
        t_classic = classic.reshape(-1, 3).sum(axis=0) * factor * factor
        self.assertLess(
            t_classic[0], 0.01 * g[0],
            "pinning the defect: the decimated gate must not see the source",
        )

        # fine path: gate at full resolution, decimate, blur — in bands
        sources = []
        for comp in _HALATION.components:
            if float(np.abs(np.asarray(comp.transfer)).sum()) == 0.0:
                sources.append(None)
                continue
            acc = np.zeros((dh, dw, 3), dtype=np.float32)
            for y0, y1 in ((0, 150), (150, 301), (301, h)):
                src = halation_component_source(e[y0:y1], ref, comp)
                area_decimate_rows(src, y0, h, w, dh, dw, acc)
            sources.append(acc)
        spread = halation_spread_map_from_sources(
            sources, geo_w_mm, _HALATION, (dh, dw)
        )
        take = upsample_rows(spread, 0, h, h, w).reshape(-1, 3).sum(axis=0)
        for c in range(2):  # blue transfer is zero in the default asset
            self.assertLess(
                abs(take[c] - g[c]) / max(g[c], 1e-9), 0.01,
                f"channel {c}: fine-path take must return the given energy",
            )

    def test_band_split_is_deterministic(self) -> None:
        from dngscan.film_optics import (
            area_decimate_rows,
            halation_component_source,
        )

        ref = np.ones(3, dtype=np.float32)
        h, w, dh, dw = 96, 144, 24, 36
        rng = np.random.default_rng(11)
        e = (0.2 + 8.0 * rng.random((h, w, 3))).astype(np.float32)
        e[10, 20] = 90.0
        comp = _HALATION.components[0]
        one = np.zeros((dh, dw, 3), dtype=np.float32)
        area_decimate_rows(
            halation_component_source(e, ref, comp), 0, h, w, dh, dw, one
        )
        split = np.zeros((dh, dw, 3), dtype=np.float32)
        for y0, y1 in ((0, 7), (7, 40), (40, 41), (41, h)):
            src = halation_component_source(e[y0:y1], ref, comp)
            area_decimate_rows(src, y0, h, w, dh, dw, split)
        np.testing.assert_array_equal(one, split)


class ContextLifecycleTests(unittest.TestCase):
    def _frame(self, h: int = 96, w: int = 144) -> np.ndarray:
        img = np.full((h, w, 3), 0.18 * 0.25, dtype=np.float32)
        # one bright pixel: above the full-resolution gate, averaged below
        # it on a 2.25x-decimated grid
        img[40, 60] = 0.18 * 2.0 ** 5.0
        return img

    def test_identity_grid_keeps_the_classic_path(self) -> None:
        from dngscan.film_develop import prepare_film_spatial
        from dngscan.film_optics import halation_spread_map

        stock = _negative_stock()
        plan = _plan(stock, film_halation=0.7)
        img = self._frame()
        h, w = img.shape[:2]
        ctx = prepare_film_spatial(plan, h, w)
        ctx.begin_halation_source(plan, stock)
        self.assertIsNone(
            ctx.hal_fine,
            "an undecimated grid must not open the fine accumulators",
        )
        ctx.finish_maps(img, plan, stock)
        e_lin = ctx._layer_exposure_f32(img, ctx._hal_prep_from(plan, stock))
        np.testing.assert_array_equal(
            ctx.hal_map,
            halation_spread_map(e_lin, ctx.hal_ref, ctx.geometry.region()[2],
                                ctx.optics.stock.halation),
            err_msg="classic path must stay byte-identical",
        )

    def test_decimated_grid_recovers_the_sub_cell_halo(self) -> None:
        from dngscan.film_develop import prepare_film_spatial
        from dngscan.film_optics import area_decimate, light_source

        stock = _negative_stock()
        plan = _plan(stock, film_halation=0.7)
        img = self._frame()
        h, w = img.shape[:2]
        with _ForcedDecimation(64):
            from dngscan.film_optics import spread_grid_shape

            dh, dw = spread_grid_shape(h, w)
            self.assertLess(dh, h, "the fixture must actually decimate")
            light = light_source(img)
            scene_dec = area_decimate(light, dh, dw)

            classic = prepare_film_spatial(plan, h, w)
            classic.finish_maps(scene_dec, plan, stock)

            fine = prepare_film_spatial(plan, h, w)
            fine.begin_halation_source(plan, stock)
            self.assertIsNotNone(fine.hal_fine)
            self.assertIsNone(
                fine.hal_fine[2],
                "the zero-transfer aura component must not pay for a grid",
            )
            for y0, y1 in ((0, 33), (33, h)):
                fine.accumulate_halation_source(light[y0:y1], y0, y1)
            fine.finish_maps(scene_dec, plan, stock)
            self.assertIsNone(fine.hal_fine, "accumulators must be released")

        classic_sum = float(np.abs(classic.hal_map).sum())
        fine_sum = float(np.abs(fine.hal_map).sum())
        self.assertLess(
            classic_sum, 0.02 * max(fine_sum, 1e-12),
            "pinning the defect: the classic map cannot see the source",
        )
        self.assertGreater(fine_sum, 0.0, "the fine map must hold the halo")

    def test_bands_match_full_frame_oracle_under_decimation(self) -> None:
        """The §9.3 row-band contract extends to the fine path: the
        renderer-mirror band lifecycle must reproduce the full-frame oracle
        (which drives begin/accumulate itself) at every split."""
        from dngscan.film_develop import apply_film_core, prepare_film_spatial
        from dngscan.film_optics import (
            area_decimate_rows,
            light_source,
            spread_grid_shape,
        )

        stock = _negative_stock()
        plan = _plan(stock, film_halation=0.7, film_bloom=0.4)
        img = self._frame()
        h, w = img.shape[:2]
        flat = img.reshape(-1, 3)
        with _ForcedDecimation(64):
            full = apply_film_core(flat, plan, spatial_shape=(h, w))
            ctx = prepare_film_spatial(plan, h, w)
            dh, dw = spread_grid_shape(h, w)
            acc = np.zeros((dh, dw, 3), dtype=np.float64)
            ctx.begin_bloom_source()
            ctx.begin_halation_source(plan, stock)
            for y0 in range(0, h, 17):
                y1 = min(y0 + 17, h)
                light = light_source(flat[y0 * w:y1 * w])
                ctx.accumulate_bloom_source(light, y0, y1)
                ctx.accumulate_halation_source(light, y0, y1)
                area_decimate_rows(
                    light.reshape(-1, w, 3), y0, h, w, dh, dw, acc
                )
            scene_dec = acc.astype(np.float32)
            ctx.finish_bloom_map(scene_dec)
            ctx.finish_maps(scene_dec, plan, stock)
            out = np.empty_like(full)
            for band_rows in (13,):
                for y0 in range(0, h, band_rows):
                    y1 = min(y0 + band_rows, h)
                    out[y0 * w:y1 * w] = apply_film_core(
                        flat[y0 * w:y1 * w], plan, spatial=(ctx, y0, y1)
                    )
                np.testing.assert_allclose(out, full, atol=2e-6)

    def test_bloom_correction_only_adds_source(self) -> None:
        """The decimated-grid bloom bracket is non-negative: the glow only
        adds light, so a bloomed plan's spread map must carry at least the
        unbloomed plan's energy."""
        from dngscan.film_develop import prepare_film_spatial
        from dngscan.film_optics import (
            area_decimate,
            light_source,
            spread_grid_shape,
        )

        stock = _negative_stock()
        img = self._frame()
        img[40:44, 100:110] = 0.18 * 2.0 ** 6.0  # a proxy-visible source too
        h, w = img.shape[:2]

        def build(**kw):
            plan = _plan(stock, film_halation=0.7, **kw)
            with _ForcedDecimation(64):
                dh, dw = spread_grid_shape(h, w)
                light = light_source(img)
                scene_dec = area_decimate(light, dh, dw)
                ctx = prepare_film_spatial(plan, h, w)
                if ctx.bloom > 0.0:
                    ctx.begin_bloom_source()
                    ctx.accumulate_bloom_source(light, 0, h)
                    ctx.finish_bloom_map(scene_dec)
                ctx.begin_halation_source(plan, stock)
                ctx.accumulate_halation_source(light, 0, h)
                ctx.finish_maps(scene_dec, plan, stock)
            return float(ctx.hal_map.sum())

        plain = build()
        bloomed = build(film_bloom=0.6)
        self.assertGreaterEqual(bloomed, plain - 1e-4 * abs(plain))


if __name__ == "__main__":
    unittest.main()
