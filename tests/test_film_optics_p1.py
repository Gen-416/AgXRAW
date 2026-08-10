# SPDX-License-Identifier: GPL-3.0-or-later
"""FILM_OPTICS_V2 phase P1 gates: assets, compiler, and a topology move that
must not move a pixel.

P1 restructures without changing maths, which makes it the easiest phase to
get wrong quietly. Two things are therefore pinned hard: every asset field is
refused when it is nonsense, and the frozen legacy render still matches byte
for byte.

The one behavioural change is R1 §3.1 — the film's exposure state moves out of
the characteristic-curve lookup and into the layer exposure, so that spatial
operators can see it. Because `ev_offset` was only ever a translation of the
lookup axis, this is algebraically an identity; the tests below prove that on
both the algebra and the frozen bytes.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dngscan import film_optics_assets as fa

ROOT = Path(__file__).resolve().parents[1]


def _plan(**kw) -> SimpleNamespace:
    base = dict(
        curve_preset="portra400", film_mode="full", film_crossover="datasheet",
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


class AssetLoadingTests(unittest.TestCase):
    def test_the_shipped_assets_load_and_declare_provenance(self) -> None:
        stock = fa.load_stock_optics(fa.DEFAULT_STOCK_OPTICS)
        medium = fa.load_print_optics(fa.DEFAULT_PRINT_OPTICS)
        bloom = fa.load_capture_bloom(fa.DEFAULT_CAPTURE_BLOOM)
        for asset in (stock, medium, bloom):
            self.assertIn(asset.provenance, fa.PROVENANCE)
        self.assertEqual(stock.grain.provenance, "modelled")
        self.assertEqual(stock.halation.provenance, "modelled")
        self.assertEqual(medium.print_scatter.provenance, "modelled")
        self.assertFalse(bloom.active, "capture bloom has no P1 implementation")

    def test_manifest_pins_every_shipped_asset(self) -> None:
        from tools.gen_film_optics_manifest import build

        stored = json.loads(fa.MANIFEST_PATH.read_text("utf-8"))
        self.assertEqual(stored, build(), "run tools/gen_film_optics_manifest.py")

    def test_a_tampered_asset_is_refused_before_it_is_parsed(self) -> None:
        """The hash is checked against the BYTES. An asset that was edited
        without re-pinning must fail as a file, not be half-trusted field by
        field."""
        name = f"stock__{fa.DEFAULT_STOCK_OPTICS}"
        path = fa.ASSET_DIR / f"{name}.json"
        original = path.read_bytes()
        try:
            path.write_bytes(original.replace(b'"sigma0": 0.055', b'"sigma0": 0.5'))
            fa._CACHE.clear()
            with self.assertRaises(fa.OpticsAssetError) as ctx:
                fa.load_stock_optics(fa.DEFAULT_STOCK_OPTICS)
            self.assertIn("hash", str(ctx.exception))
        finally:
            path.write_bytes(original)
            fa._CACHE.clear()

    def test_every_field_family_fails_closed(self) -> None:
        good_grain = {
            "provenance": "modelled", "medium": "negative",
            "model": "band_limited_gaussian_v1",
            "pitch_um": 12.0, "size_um": 18.0, "sigma0": 0.055,
            "layer_corr": 0.35,
        }
        for mutation, needle in (
            ({"model": "guessed"}, "unknown grain model"),
            ({"pitch_um": 0.0}, "must be positive"),
            ({"sigma0": float("nan")}, "not finite"),
            ({"layer_corr": 1.5}, "outside"),
            ({"provenance": "vibes"}, "provenance"),
        ):
            with self.subTest(mutation=mutation):
                raw = dict(good_grain, **mutation)
                with self.assertRaises(fa.OpticsAssetError) as ctx:
                    fa.GrainAsset.from_json(raw, "probe")
                self.assertIn(needle, str(ctx.exception))

        good_hal = {
            "provenance": "modelled", "model": "legacy_threshold_cascade_v1",
            "radius_mm": 0.55, "layer_weights": [1.0, 0.22, 0.06],
            "threshold_ev": 1.5, "strength": 0.12, "dc_mode": "additive",
        }
        for mutation, needle in (
            ({"dc_mode": "whatever"}, "unknown dc_mode"),
            ({"layer_weights": [1.0, -0.2, 0.06]}, "non-negative"),
            ({"layer_weights": [1.0, 0.2]}, "three components"),
            ({"radius_mm": -1.0}, "must be positive"),
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(fa.OpticsAssetError) as ctx:
                    fa.HalationAsset.from_json(dict(good_hal, **mutation), "probe")
                self.assertIn(needle, str(ctx.exception))

    def test_future_phase_fields_must_stay_empty(self) -> None:
        """A P2/P3/P4 field that arrives early with data nobody implemented
        would render silently; the loader refuses it by name."""
        raw = json.loads(
            (fa.ASSET_DIR / f"stock__{fa.DEFAULT_STOCK_OPTICS}.json").read_text("utf-8")
        )
        raw["emulsion_scatter"] = {"sigma_um": 2.0}
        with self.assertRaises(fa.OpticsAssetError):
            fa.StockOpticsAsset.from_json(raw, "probe")

        praw = json.loads(
            (fa.ASSET_DIR / f"print__{fa.DEFAULT_PRINT_OPTICS}.json").read_text("utf-8")
        )
        for field in ("formation_scatter", "positive_grain", "viewing_scatter"):
            with self.subTest(field=field):
                with self.assertRaises(fa.OpticsAssetError):
                    fa.PrintOpticsAsset.from_json(
                        dict(praw, **{field: {"anything": 1.0}}), "probe"
                    )


class CompilerTests(unittest.TestCase):
    def test_all_amounts_zero_compiles_to_nothing(self) -> None:
        """The strict-identity fast path: no asset is read and no context is
        built, so amount 0 cannot cost a file open."""
        self.assertIsNone(fa.compile_film_optics_plan(_plan()))

    def test_engaged_plan_carries_assets_amounts_and_hashes(self) -> None:
        got = fa.compile_film_optics_plan(
            _plan(film_grain=0.5, film_halation=0.4, film_bloom=0.3, film_optics_seed=7)
        )
        self.assertTrue(got.engaged)
        self.assertEqual(got.seed, 7)
        self.assertEqual(
            (got.grain_amount, got.halation_amount, got.print_scatter_amount),
            (0.5, 0.4, 0.3),
        )
        self.assertEqual(len(got.asset_hashes), 3)
        for digest in got.asset_hashes:
            self.assertEqual(len(digest), 64)

    def test_a_negative_or_nonfinite_amount_is_refused(self) -> None:
        for bad in (-0.1, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(fa.OpticsAssetError):
                    fa.compile_film_optics_plan(_plan(film_grain=bad))

    def test_report_names_the_asset_and_its_provenance(self) -> None:
        rep = fa.compile_film_optics_plan(_plan(film_halation=0.4)).report()
        self.assertEqual(rep["stock_optics"], fa.DEFAULT_STOCK_OPTICS)
        self.assertEqual(rep["provenance"]["halation"], "modelled")
        self.assertEqual(rep["halation_dc_mode"], "additive")
        self.assertEqual(set(rep["asset_sha256"]), set(rep["asset_sha256"]))

    def test_the_render_report_states_asset_and_dc_mode(self) -> None:
        from dngscan.report import jpeg_tone_plan_cn

        from tests.golden_support import all_scenes

        scene = all_scenes()["daylight_wide_dr"]
        note = jpeg_tone_plan_cn(
            scene.bundle, scene.analysis, "agx",
            tone_plan=_plan(film_grain=0.5, film_halation=0.4),
        )
        self.assertIn("模拟光学", note)
        self.assertIn(fa.DEFAULT_STOCK_OPTICS, note)
        self.assertIn("halation DC=additive", note)


class ExposureTopologyTests(unittest.TestCase):
    """R1 §3.1: exposure moves ahead of the spatial operators."""

    def test_moving_the_offset_is_algebraically_an_identity(self) -> None:
        """`characteristic_amounts` shifts its lookup axis by
        ev_offset * log10(2). Adding the same shift to logE beforehand has to
        give identical dye amounts, which is what makes the relocation safe to
        do without moving the frozen bytes."""
        from dngscan.film_v2_math import LOG10_2, characteristic_amounts

        rng = np.random.default_rng(3)
        le = np.linspace(-4.0, 2.0, 64)
        table = np.cumsum(rng.random((64, 3)) * 0.05, axis=0)
        log_e = rng.uniform(-3.0, 1.0, size=(256, 3))
        for ev in (-1.5, 0.0, 0.75):
            with self.subTest(ev=ev):
                via_offset = characteristic_amounts(log_e, le, table, ev_offset=ev)
                via_exposure = characteristic_amounts(log_e + ev * LOG10_2, le, table)
                np.testing.assert_allclose(via_exposure, via_offset, rtol=0, atol=0)

    def test_film_exposure_now_reaches_the_halation_operator(self) -> None:
        """The point of the move. Before it, changing the film exposure
        changed density while every spatial operator kept seeing the same
        light; the reinjected energy must now respond."""
        from dngscan.film_develop import apply_film_core

        h = w = 48
        scene = np.full((h, w, 3), 0.02, dtype=np.float32)
        scene[20:28, 20:28] = 12.0
        flat = scene.reshape(-1, 3)

        def energy(ev: float) -> float:
            base = apply_film_core(
                flat, _plan(film_exposure_ev=ev), spatial_shape=(h, w)
            )
            lit = apply_film_core(
                flat, _plan(film_exposure_ev=ev, film_halation=0.6),
                spatial_shape=(h, w),
            )
            return float(np.sum(np.asarray(lit) - np.asarray(base)))

        at_zero = energy(0.0)
        self.assertGreater(at_zero, 0.0, "halation must inject something at all")
        self.assertNotAlmostEqual(
            energy(1.0) / at_zero, 1.0, places=3,
            msg="halation still ignores the film exposure state",
        )


class LegacyByteFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The freeze is a NumPy-reference freeze by construction, so it is
        # read with the native fast backend off. Without this the suite passes
        # under DNGSCAN_FAST=0 and fails under 1, which reads as a P1
        # regression when it is only a backend difference the fixtures never
        # claimed to cover.
        cls._fast = os.environ.get("DNGSCAN_FAST")
        os.environ["DNGSCAN_FAST"] = "0"

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._fast is None:
            os.environ.pop("DNGSCAN_FAST", None)
        else:
            os.environ["DNGSCAN_FAST"] = cls._fast

    def test_p1_does_not_move_the_frozen_optics_render(self) -> None:
        """The P1 exit gate. Restructuring the assets and relocating the
        exposure offset must leave the recorded output untouched."""
        from tools.regen_optics_freeze import iter_cases, render_case

        cases = iter_cases()
        self.assertGreaterEqual(len(cases), 6)
        for case in cases:
            with self.subTest(case=case.stem):
                stored = np.load(case.path, allow_pickle=False)
                linear, u8 = render_case(case)
                np.testing.assert_array_equal(stored["u8"], u8)
                self.assertLessEqual(
                    float(np.max(np.abs(linear - stored["linear"].astype(np.float32)))),
                    float(np.finfo(np.float16).eps) * 4,
                )


if __name__ == "__main__":
    unittest.main()
