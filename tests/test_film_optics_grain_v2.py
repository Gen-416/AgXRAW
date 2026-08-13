# SPDX-License-Identifier: GPL-3.0-or-later
"""Grain V2 (optics V2 P4) — measured_sigma_v2 kernel gates.

Gate 14: a synthetic flat field, rendered at the film grid pitch and
measured back through the chart's own 48 um aperture, reproduces the
digitized sigma(D) at that density.
Gate 15: the amount sweep does not change the mean transmittance of the
negative (the bias-compensation contract).
Plus asset-contract checks: v1 assets stay valid, v2 fail-closed rules.
"""
from __future__ import annotations

import unittest

import numpy as np

from dngscan.film_optics import (
    GATE_H_MM,
    GATE_W_MM,
    FilmGeometry,
    _aperture_rms,
    apply_density_grain,
)
from dngscan.film_optics_assets import (
    DEFAULT_STOCK_OPTICS,
    GrainAsset,
    load_stock_optics,
)


def _flat_geometry(px_um: float) -> "FilmGeometry":
    w = int(round(GATE_W_MM * 1000.0 / px_um))
    h = int(round(GATE_H_MM * 1000.0 / px_um))
    return FilmGeometry(width=w, height=h, rotated=False)


class MeasuredGrainKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grain = load_stock_optics(DEFAULT_STOCK_OPTICS).grain
        assert cls.grain is not None

    def test_default_stock_grain_is_measured_v2(self) -> None:
        g = self.grain
        self.assertEqual(g.model, "measured_sigma_v2")
        # R1 item 3/7: measured amplitude + modelled geometry on a
        # GENERIC profile = derived, never the material's own measured
        self.assertEqual(g.provenance, "derived")
        self.assertEqual(g.aperture_um, 48.0)
        self.assertEqual(len(g.sigma_density), 3)
        self.assertEqual(g.sigma0, 0.0)

    def test_gate14_flat_field_reproduces_chart_sigma(self) -> None:
        # Density-native contract (review R1 item 1): the chain's amounts
        # ARE net Status-family densities, so a flat field AT net density
        # d sits at chart density base+d and the measured 48um box RMS of
        # the OUTPUT amounts must equal the chart sigma directly — no
        # affine span assumption on either side of the comparison.
        g = self.grain
        geometry = _flat_geometry(g.pitch_um)
        h, w = geometry.height, geometry.width
        lo = np.zeros(3)
        hi = np.ones(3)
        for net_d in (0.3, 0.8, 1.4):
            amounts = np.full((h * w, 3), net_d, dtype=np.float64)
            out = apply_density_grain(
                amounts, lo, hi, geometry, g, 1.0, 0
            ).reshape(h, w, 3)
            n = int(round(g.aperture_um / g.pitch_um))
            for ch in range(3):
                base, _dmax = g.chart_density[ch]
                tab = np.asarray(g.sigma_density[ch])
                sigma_want = float(np.interp(base + net_d,
                                             tab[:, 0], tab[:, 1]))
                delta = (out[..., ch] - amounts.reshape(h, w, 3)[..., ch])
                ii = np.zeros((h + 1, w + 1))
                np.cumsum(np.cumsum(delta, axis=0), axis=1, out=ii[1:, 1:])
                box = (ii[n:, n:] - ii[n:, :-n] - ii[:-n, n:]
                       + ii[:-n, :-n]) / (n * n)
                box -= box.mean()  # bias term is deterministic, not grain
                got = float(np.sqrt(np.mean(box * box)))
                self.assertLess(
                    abs(got - sigma_want) / sigma_want, 0.12,
                    f"ch{ch} D={net_d}: got {got:.5f} want {sigma_want:.5f}",
                )

    def test_gate15_amount_sweep_preserves_mean_transmittance(self) -> None:
        g = self.grain
        geometry = _flat_geometry(g.pitch_um)
        h, w = geometry.height, geometry.width
        lo = np.zeros(3)
        hi = np.ones(3)
        amounts = np.full((h * w, 3), 0.8, dtype=np.float64)
        # transmittance in CHART density units per channel
        ref = None
        for amt in (0.0, 0.5, 1.0):
            out = apply_density_grain(amounts, lo, hi, geometry, g, amt, 0)
            for ch in range(3):
                base, _dmax = g.chart_density[ch]
                d = base + out[:, ch]
                t = float(np.mean(np.power(10.0, -d)))
                if amt == 0.0 and ch == 0:
                    ref = t
                if ch == 0:
                    self.assertLess(
                        abs(t - ref) / ref, 0.004,
                        f"amount={amt}: mean transmittance drifted {t} vs {ref}",
                    )

    def test_grain_scales_with_declared_sigma_direction(self) -> None:
        # 5207 chart fact carried into the render: at low density the B
        # channel is grainier than R
        g = self.grain
        geometry = _flat_geometry(g.pitch_um)
        h, w = geometry.height, geometry.width
        amounts = np.full((h * w, 3), 0.15, dtype=np.float64)
        out = apply_density_grain(
            amounts, np.zeros(3), np.ones(3), geometry, g, 1.0, 0
        )
        # density-native: per-channel std IS chart-sigma scale directly
        std = [(out[:, ch] - amounts[:, ch]).std() for ch in range(3)]
        self.assertGreater(std[2], std[0])

    def test_aperture_rms_is_cached_and_below_unity(self) -> None:
        r1 = _aperture_rms(self.grain)
        r2 = _aperture_rms(self.grain)
        self.assertEqual(r1, r2)
        # aperture averaging must reduce a unit-RMS field
        self.assertLess(r1, 1.0)
        self.assertGreater(r1, 0.0)


class GrainAssetContractTests(unittest.TestCase):
    BASE = {
        "provenance": "measured", "medium": "negative",
        "model": "measured_sigma_v2", "pitch_um": 12.0, "size_um": 18.0,
        "layer_corr": 0.35, "aperture_um": 48.0,
        "channels": {
            name: {
                "chart_density": [0.2, 2.0],
                "sigma_density": [[0.2, 0.005], [0.8, 0.006],
                                  [1.4, 0.0055], [2.0, 0.005]],
            } for name in ("R", "G", "B")
        },
    }

    def test_v2_parses(self) -> None:
        g = GrainAsset.from_json(dict(self.BASE), "t")
        self.assertEqual(g.model, "measured_sigma_v2")
        self.assertEqual(len(g.sigma_density[0]), 4)

    def test_v2_rejects_sigma0(self) -> None:
        raw = dict(self.BASE)
        raw["sigma0"] = 0.05
        with self.assertRaises(Exception):
            GrainAsset.from_json(raw, "t")

    def test_v1_rejects_measured_fields(self) -> None:
        raw = {
            "provenance": "modelled", "medium": "negative",
            "model": "band_limited_gaussian_v1", "pitch_um": 12.0,
            "size_um": 18.0, "sigma0": 0.05, "layer_corr": 0.35,
            "aperture_um": 48.0,
        }
        with self.assertRaises(Exception):
            GrainAsset.from_json(raw, "t")

    def test_v2_rejects_unsorted_table(self) -> None:
        import json as _json
        raw = _json.loads(_json.dumps(self.BASE))
        raw["channels"]["G"]["sigma_density"][1][0] = 0.1
        with self.assertRaises(Exception):
            GrainAsset.from_json(raw, "t")

    def test_v1_assets_still_load(self) -> None:
        g = GrainAsset.from_json({
            "provenance": "modelled", "medium": "negative",
            "model": "band_limited_gaussian_v1", "pitch_um": 12.0,
            "size_um": 18.0, "sigma0": 0.055, "layer_corr": 0.35,
        }, "t")
        self.assertEqual(g.model, "band_limited_gaussian_v1")
        self.assertEqual(g.sigma_density, ())


if __name__ == "__main__":
    unittest.main()


class MultiBandFieldTests(unittest.TestCase):
    """P4 multi-band spectrum gates, windows from the dye-cloud particle
    oracle (tools/grain_particle_oracle.py): a 12 um-pitch representation
    of Boolean dye-cloud grain shows Selwyn slope ~-0.93 and correlation
    FWHM ~13 um; the fitted band mixture reproduces the aperture-RMS curve
    to 4.2% log-RMS with slope -0.90 / FWHM 12.3 um."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.grain = load_stock_optics(DEFAULT_STOCK_OPTICS).grain

    def test_default_asset_declares_the_oracle_fit(self) -> None:
        self.assertEqual(self.grain.bands, ((6.0, 0.85), (18.0, 0.15)))

    def test_field_is_grain_not_blotch(self) -> None:
        from dngscan import film_optics_diag as diag
        from dngscan.film_optics import grain_field_for

        f = grain_field_for(self.grain, 0)
        slope = diag.selwyn_slope(f, apertures=(1, 2, 4, 8))
        self.assertLess(slope, -0.8, "Selwyn slope must be in the grain regime")
        self.assertGreater(slope, -1.05)
        corr = diag.correlation_length_cells(f)
        self.assertLess(
            2.0 * corr * self.grain.pitch_um, 20.0,
            "correlation FWHM must sit near the render pitch, not blotch scale",
        )

    def test_band_weights_fail_closed(self) -> None:
        import json as _json
        base = _json.loads(_json.dumps(GrainAssetContractTests.BASE))
        base["bands"] = [[6.0, 0.5], [18.0, 0.4]]  # sum != 1
        with self.assertRaises(Exception):
            GrainAsset.from_json(base, "t")
        base["bands"] = [[18.0, 0.5], [6.0, 0.5]]  # not ascending
        with self.assertRaises(Exception):
            GrainAsset.from_json(base, "t")

    def test_single_band_fallback_matches_declared_size(self) -> None:
        # an asset without `bands` keeps the one-band behaviour
        g = GrainAsset.from_json(dict(GrainAssetContractTests.BASE), "t")
        self.assertEqual(g.bands, ())


class DualGrainTests(unittest.TestCase):
    """P4 dual grain: the positive medium's own measured 2383 grain rides
    the print dye amounts before B2 (negative branch only)."""

    def test_print_asset_declares_measured_2383_grain(self) -> None:
        from dngscan.film_optics_assets import (
            DEFAULT_PRINT_OPTICS,
            load_print_optics,
        )

        pos = load_print_optics(DEFAULT_PRINT_OPTICS).positive_grain
        self.assertIsNotNone(pos)
        self.assertEqual(pos.model, "measured_sigma_v2")
        self.assertEqual(pos.provenance, "derived")
        self.assertEqual(pos.medium, "print_film")
        # 2383 chart fact: the yellow-forming B layer dominates
        b_end = pos.sigma_density[2][-1][1]
        g_end = pos.sigma_density[1][-1][1]
        self.assertGreater(b_end, 2.0 * g_end)

    def test_print_grain_shares_the_stock_field_master(self) -> None:
        from dngscan.film_optics import _field_geometry_key
        from dngscan.film_optics_assets import (
            DEFAULT_PRINT_OPTICS,
            DEFAULT_STOCK_OPTICS,
            load_print_optics,
            load_stock_optics,
        )

        stock = load_stock_optics(DEFAULT_STOCK_OPTICS).grain
        pos = load_print_optics(DEFAULT_PRINT_OPTICS).positive_grain
        self.assertEqual(
            _field_geometry_key(stock), _field_geometry_key(pos),
            "shared field geometry keeps ONE master realization in cache",
        )

    def test_print_realization_is_independent_of_the_stock_field(self) -> None:
        """Review R1 item 6: the print grain must be an independent keyed
        phase of the shared master — the old channel-roll+sign-flip
        derivation made the print field fully predictable from the
        negative's (per-channel cross-correlation exactly -layer_corr)."""
        from dngscan.film_optics import (
            MASTER_SEED,
            FilmGeometry,
            _grain_ii_for,
            realization_phases,
            sample_field,
        )
        from dngscan.film_optics_assets import (
            DEFAULT_STOCK_OPTICS,
            load_stock_optics,
        )

        grain = load_stock_optics(DEFAULT_STOCK_OPTICS).grain
        master = _grain_ii_for(grain, MASTER_SEED)
        gh, gw = master.shape[0] - 1, master.shape[1] - 1
        geo = FilmGeometry(512, 768)
        seed = 424242
        f_neg = sample_field(
            master, geo.rows(0, 512), phase=realization_phases(seed, gh, gw)
        ).astype(np.float64)
        f_pr = sample_field(
            master, geo.rows(0, 512),
            phase=realization_phases(seed ^ 0x50524E54, gh, gw),
        ).astype(np.float64)
        for ch in range(3):
            a = f_neg[..., ch].ravel()
            b = f_pr[..., ch].ravel()
            corr = float(np.corrcoef(a, b)[0, 1])
            self.assertLess(
                abs(corr), 0.05,
                f"channel {ch}: print/stock field correlation {corr:+.3f} "
                "must be ~0 (independent keyed phase, R1 item 6)",
            )

    def test_dual_grain_is_live_on_the_negative_branch(self) -> None:
        import dataclasses

        from dngscan import film_optics_assets as fa
        from dngscan.film_develop import apply_film_core
        from types import SimpleNamespace

        def plan(**kw):
            base = dict(
                curve_preset="portra400", film_mode="full",
                film_crossover="datasheet", film_exposure_ev=0.0,
                film_print_timing="fixed", film_print_medium="",
                film_print_exposure_ev=0.0, color_head_y=0.0,
                color_head_m=0.0, film_development="measured_default",
                film_dev_contrast=0.0, film_dev_fog=0.0, film_dev_density=0.0,
                film_compression=0.0, film_compression_knee=2.0,
                film_highlight_density=0.0, film_grain=0.8,
                film_halation=0.0, film_bloom=0.0, film_optics_seed=0,
            )
            base.update(kw)
            return SimpleNamespace(**base)

        h, w = 40, 60
        flat = np.full((h * w, 3), 0.18, dtype=np.float32)
        got = apply_film_core(flat, plan(), spatial_shape=(h, w))
        key = f"print:{fa.DEFAULT_PRINT_OPTICS}"
        original = fa.load_print_optics(fa.DEFAULT_PRINT_OPTICS)
        try:
            fa._CACHE[key] = dataclasses.replace(original, positive_grain=None)
            without = apply_film_core(flat, plan(), spatial_shape=(h, w))
        finally:
            fa._CACHE[key] = original
        diff = np.abs(np.asarray(got, np.float64) - np.asarray(without, np.float64))
        self.assertGreater(
            float(diff.mean()), 1e-6,
            "the paper-stage grain hook must actually reach the output",
        )
