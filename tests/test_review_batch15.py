# SPDX-License-Identifier: GPL-3.0-or-later
"""Review batch 15 acceptance gates.

Conservative medium scatter: energy is REDISTRIBUTED inside the declared
frame, never added — cores shed what neighbourhoods gain, hue-preserving
source extraction, area-preserving upsample, band-split invariance.

Grain master/realization split: one expensive master field per profile,
per-RAW randomness is a spatial phase on its periodic extension — identical
statistics, different arrangement, no rebuild, no full-field copies.
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


def _negative_stock() -> str:
    for stock in _stock_files():
        if stock.startswith(("portra", "pro400h", "c200", "gold")):
            return stock
    return _stock_files()[0]


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


class ConservativeScatterTests(unittest.TestCase):
    def _delta(self, img):
        from dngscan.film_optics import bloom_delta_map

        return bloom_delta_map(img, _SCATTER)

    def test_uniform_field_passes_through(self) -> None:
        img = np.full((48, 64, 3), 0.9, dtype=np.float32)  # above threshold
        delta = self._delta(img)
        self.assertLess(float(np.abs(delta).max()), 1e-5,
                        "a uniform bright field must scatter into itself")

    def test_impulse_conserves_energy_and_redistributes(self) -> None:
        img = np.full((64, 96, 3), 0.05, dtype=np.float32)
        img[32, 48] = 1.0
        delta = self._delta(img)
        # per-channel total energy error < 1e-5 (interior impulse)
        sums = delta.sum(axis=(0, 1), dtype=np.float64)
        self.assertLess(float(np.abs(sums).max()), 1e-5)
        # core loses, neighbourhood gains
        self.assertLess(float(delta[32, 48].mean()), 0.0)
        self.assertGreater(float(delta[34, 48].mean()), 0.0)

    def test_edge_impulse_never_gains_energy(self) -> None:
        img = np.full((33, 35, 3), 0.05, dtype=np.float32)
        img[0, 0] = 1.0
        delta = self._delta(img)
        sums = delta.sum(axis=(0, 1), dtype=np.float64)
        self.assertLessEqual(float(sums.max()), 1e-5)

    def test_coloured_highlight_keeps_its_rgb_ratio(self) -> None:
        img = np.full((48, 64, 3), 0.05, dtype=np.float32)
        img[24, 32] = (0.9, 0.6, 0.3)  # warm highlight above Y threshold
        delta = self._delta(img)
        # the source (negative part at the core) must be RGB-proportional to
        # the pixel, so the scattered energy carries the highlight's hue
        core = -delta[24, 32]
        self.assertGreater(float(core.min()), 0.0)
        ratio = core / core[0]
        want = np.array([0.9, 0.6, 0.3]) / 0.9
        np.testing.assert_allclose(ratio, want, rtol=0.02)

    @staticmethod
    def _spread_ii(img):
        from dngscan.film_optics import (
            integral_from_field,
            scatter_source,
            scatter_spread,
        )

        return integral_from_field(
            scatter_spread(scatter_source(img, _SCATTER), _SCATTER)
        ).astype(np.float32)

    def test_band_split_matches_full_frame_bytes(self) -> None:
        from dngscan.film_optics import bloom_apply_rows

        h, w = 64, 96
        rng = np.random.default_rng(2)
        img = rng.uniform(0.0, 1.0, (h, w, 3)).astype(np.float32)
        spread_ii = self._spread_ii(img)
        flat = img.reshape(-1, 3)
        full = bloom_apply_rows(flat, spread_ii, 0, h, h, w, _SCATTER, 0.8)
        banded = np.empty_like(full)
        for y0 in range(0, h, 11):
            y1 = min(y0 + 11, h)
            banded[y0 * w:y1 * w] = bloom_apply_rows(
                flat[y0 * w:y1 * w], spread_ii, y0, y1, h, w, _SCATTER, 0.8
            )
        np.testing.assert_array_equal(full, banded)

    def test_output_sum_conserved_and_clean(self) -> None:
        from dngscan.film_optics import bloom_apply_rows

        h, w = 64, 96
        rng = np.random.default_rng(3)
        img = rng.uniform(0.0, 1.1, (h, w, 3)).astype(np.float32)
        spread_ii = self._spread_ii(img)
        out = bloom_apply_rows(
            img.reshape(-1, 3), spread_ii, 0, h, h, w, _SCATTER, 1.0
        )
        self.assertTrue(np.isfinite(out).all())
        self.assertGreaterEqual(float(out.min()), 0.0)
        drift = abs(float(out.sum(dtype=np.float64))
                    - float(img.sum(dtype=np.float64)))
        self.assertLess(drift / float(img.sum(dtype=np.float64)), 1e-4)

    def test_amount_zero_is_byte_identity_through_the_core(self) -> None:
        from dngscan.film_develop import apply_film_core

        stock = _negative_stock()
        rng = np.random.default_rng(4)
        img = rng.uniform(0.02, 0.6, (48, 72, 3)).astype(np.float32)
        flat = img.reshape(-1, 3)
        a = apply_film_core(flat, _plan(stock))
        b = apply_film_core(flat, _plan(stock, film_bloom=0.0),
                            spatial_shape=(48, 72))
        np.testing.assert_array_equal(a, b)


class GrainRealizationTests(unittest.TestCase):
    def test_same_realization_repeats_and_autos_differ(self) -> None:
        from dngscan.film_develop import apply_film_core

        stock = _negative_stock()
        rng = np.random.default_rng(5)
        img = rng.uniform(0.05, 0.4, (48, 72, 3)).astype(np.float32)
        flat = img.reshape(-1, 3)
        a1 = apply_film_core(flat, _plan(stock, film_grain=0.7,
                                         film_optics_seed=1234),
                             spatial_shape=(48, 72))
        a2 = apply_film_core(flat, _plan(stock, film_grain=0.7,
                                         film_optics_seed=1234),
                             spatial_shape=(48, 72))
        np.testing.assert_array_equal(a1, a2)
        b = apply_film_core(flat, _plan(stock, film_grain=0.7,
                                        film_optics_seed=5678),
                            spatial_shape=(48, 72))
        self.assertFalse(np.array_equal(a1, b))

    def test_master_is_built_once_across_seeds(self) -> None:
        from unittest import mock

        from dngscan import film_optics
        from dngscan.film_optics import _grain_ii_for

        film_optics._FIELD_CACHE.clear()
        calls = []
        real = film_optics._band_limited_field

        def spy(profile, seed):
            calls.append(seed)
            return real(profile, seed)

        with mock.patch.object(film_optics, "_band_limited_field", spy):
            _grain_ii_for(_GRAIN, film_optics.MASTER_SEED)
            from dngscan.film_optics import (
            FilmGeometry,
            realization_phases,
            sample_field,
        )

            master = _grain_ii_for(_GRAIN, film_optics.MASTER_SEED)
            gh, gw = master.shape[0] - 1, master.shape[1] - 1
            for seed in (111, 222, 333):
                sample_field(
                    master, FilmGeometry.fit(60, 90),
                    phase=realization_phases(seed, gh, gw),
                )
        self.assertEqual(calls, [film_optics.MASTER_SEED],
                         "switching realizations must never rebuild the master")

    def test_realization_creation_is_cheap_and_copy_free(self) -> None:
        import time

        from dngscan.film_optics import realization_phases

        t0 = time.perf_counter()
        for seed in range(1, 2001):
            realization_phases(seed, 2000, 3000)
        per = (time.perf_counter() - t0) / 2000
        self.assertLess(per, 1e-3, "realization must cost <1 ms")

    def test_periodic_boundary_has_no_seam(self) -> None:
        from dngscan.film_optics import (
            FilmGeometry,
            _grain_ii_for,
            sample_field,
        )

        master = _grain_ii_for(_GRAIN, 0)
        gh, gw = master.shape[0] - 1, master.shape[1] - 1
        # a phase that forces every footprint to wrap
        got = sample_field(
            master, FilmGeometry.fit(300, 450), phase=(gh - 1, gw - 1)
        )
        self.assertTrue(np.isfinite(got).all())
        row_rms = np.sqrt(np.mean(np.square(got, dtype=np.float64), axis=(1, 2)))
        col_rms = np.sqrt(np.mean(np.square(got, dtype=np.float64), axis=(0, 2)))
        for rms in (row_rms, col_rms):
            self.assertGreater(float(rms.min()), 0.3 * float(rms.max()),
                               "wrap seam: dead or doubled band at the boundary")

    def test_half_resolution_is_still_the_area_mean(self) -> None:
        from dngscan.film_optics import (
            FilmGeometry,
            _grain_ii_for,
            realization_phases,
            sample_field,
        )

        master = _grain_ii_for(_GRAIN, 0)
        gh, gw = master.shape[0] - 1, master.shape[1] - 1
        phase = realization_phases(97, gh, gw)
        full = sample_field(master, FilmGeometry(400, 600), phase=phase)
        half = sample_field(master, FilmGeometry(200, 300), phase=phase)
        block = full.reshape(200, 2, 300, 2, 3).mean(axis=(1, 3))
        np.testing.assert_allclose(half, block, atol=2e-4)

    def test_seed_zero_is_the_master_realization(self) -> None:
        from dngscan.film_optics import realization_phases

        self.assertEqual(realization_phases(0, 2000, 3000), (0, 0))


class SeedLifecycleTests(unittest.TestCase):
    def test_plan_and_pixel_caches_key_on_the_seed(self) -> None:
        from dngscan.gui.service import _preview_pixel_key

        class _B:
            lens_filter = "none"

        common = dict(
            bundle=_B(), gamut="srgb", ev=0.0, look="none", look_strength=1.0,
            display_filter="none", filter_strength=1.0, scene_transform="none",
            scene_transform_strength=1.0, punch_scale=1.0, tone_core="agx",
            lum_norm="y", agx_primaries="base", lens_filter="none",
            film_curve="portra400", adjustments=None, film_mode="full",
            film_grain=0.5,
        )
        a = _preview_pixel_key(**common, film_optics_seed=1)
        b = _preview_pixel_key(**common, film_optics_seed=2)
        self.assertNotEqual(a, b, "seed change must invalidate preview pixels")

    def test_preview_entry_mints_one_realization(self) -> None:
        import dataclasses

        from dngscan.gui.preview_cache import PreviewEntry

        fields = {f.name for f in dataclasses.fields(PreviewEntry)}
        self.assertIn("realization_id", fields)

    def test_cli_resolves_auto_once_and_reports_it(self) -> None:
        import inspect

        from dngscan import cli, report

        src = inspect.getsource(cli)
        self.assertIn('== "auto"', src)
        self.assertIn("secrets", src)
        # The report must PRINT the resolved seed. Asserted on the rendered
        # text rather than by grepping report.py for a field name: the seed
        # now reaches the line through the compiled optics plan, and a source
        # grep would have called that refactor a regression while the user-
        # visible behaviour was unchanged.
        from types import SimpleNamespace

        from tests.golden_support import all_scenes

        scene = all_scenes()["daylight_wide_dr"]
        note = report.jpeg_tone_plan_cn(
            scene.bundle, scene.analysis, "agx",
            tone_plan=SimpleNamespace(
                film_mode="full", curve_preset="portra400",
                film_grain=0.5, film_halation=0.0, film_bloom=0.0,
                film_optics_seed=20260810, film_crossover="datasheet",
            ),
        )
        self.assertIn("seed=20260810", note)

    def test_build_render_plan_never_randomizes(self) -> None:
        import inspect

        from dngscan import tone

        src = inspect.getsource(tone.build_render_plan)
        for needle in ("secrets", "random", "randbits"):
            self.assertNotIn(needle, src)


if __name__ == "__main__":
    unittest.main()
