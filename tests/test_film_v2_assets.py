# SPDX-License-Identifier: GPL-3.0-or-later
"""film v2 P1 gates: plan objects, Stage A math, schema-v6 modular assets.

Acceptance per FILM_PRINT_RENDERING_PLAN §13: Stage A reproduces the observer
inverse + characteristic curves analytically; the deployed Stage B bytes
reproduce the direct spectral chain within the declared gate; the per-medium
plan contract fails closed.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from dngscan.film_plans import (
    AnalogFinishPlan,
    FilmDevelopmentPlan,
    FilmExposurePlan,
    FilmPrintPlan,
    is_identity_finish,
    validate_film_plans,
)
from dngscan.film_v2_math import (
    LOG10_2,
    SCENE_MID,
    amounts_to_unit,
    layer_log_exposure,
    stage_a_amounts,
)

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "dngscan" / "data" / "film_v2"
PILOT = ("portra400", "velvia100", "vision3250d")


def _stock_files() -> tuple[str, ...]:
    return tuple(sorted(
        p.stem for p in ASSET_DIR.glob("*.npz")
        if not p.name.startswith(("print__", "b2__"))
    ))


def _print_files() -> tuple[Path, ...]:
    return tuple(sorted(ASSET_DIR.glob("print__*.npz")))


def _b2_files() -> tuple[Path, ...]:
    return tuple(sorted(ASSET_DIR.glob("b2__*.npz")))


# Deployed-bytes gate (plan §13). The fixed-path bound loosened slightly in
# P3: the factorized chain stacks TWO f16 volumes plus the paper-table hop,
# and the worst pairing (vision350d x 2383) measured max 0.0413 stop.
ORACLE_P99_GATE = 0.03
ORACLE_MAX_GATE = 0.05


def _load(stock: str) -> dict:
    path = ASSET_DIR / f"{stock}.npz"
    if not path.is_file():
        raise unittest.SkipTest(f"missing {path}; run tools/build_film_v2_assets.py")
    return dict(np.load(path, allow_pickle=False))


def _load_print(stock: str) -> dict:
    z = _load(stock)
    medium = str(np.asarray(z["default_medium"]))
    path = ASSET_DIR / f"print__{stock}__{medium}.npz"
    if not path.is_file():
        raise unittest.SkipTest(f"missing {path}")
    return dict(np.load(path, allow_pickle=False))


class FilmPlanContractTests(unittest.TestCase):
    def _valid(self, **overrides) -> tuple:
        plans = dict(
            exposure=FilmExposurePlan(stock_id="portra400"),
            development=FilmDevelopmentPlan(),
            print_plan=FilmPrintPlan(medium_id="Endura"),
            finish=AnalogFinishPlan(),
        )
        plans.update(overrides)
        return plans

    def test_identity_defaults_validate(self) -> None:
        validate_film_plans(**self._valid())

    def test_exposure_domain_hard_rejects(self) -> None:
        for ev in (-2.01, 2.01, 5.0):
            with self.subTest(ev=ev), self.assertRaises(ValueError):
                validate_film_plans(**self._valid(
                    exposure=FilmExposurePlan(stock_id="portra400", exposure_ev=ev)
                ))
        validate_film_plans(**self._valid(
            exposure=FilmExposurePlan(stock_id="portra400", exposure_ev=-2.0)
        ))

    def test_measured_default_locks_developer_parameters(self) -> None:
        with self.assertRaises(ValueError):
            validate_film_plans(**self._valid(
                development=FilmDevelopmentPlan(contrast_delta=0.1)
            ))
        # P4: an editorial recipe must pair with datasheet neutralization
        # (the bounded casts are solved against measured development).
        validate_film_plans(**self._valid(
            development=FilmDevelopmentPlan(
                recipe_id="editorial_custom", contrast_delta=0.1,
                provenance="editorial",
            ),
            print_plan=FilmPrintPlan(
                medium_id="print_paper", neutralization_policy="datasheet",
            ),
        ))

    def test_reversal_direct_fails_closed(self) -> None:
        cases = (
            dict(medium_id="reversal_direct", timing_policy="retimed"),
            dict(medium_id="reversal_direct", timing_policy="custom"),
            dict(medium_id="reversal_direct", printer_y_cc=5.0),
            dict(medium_id="reversal_direct", printer_m_cc=5.0),
            dict(medium_id="reversal_direct", print_exposure_ev=0.5),
        )
        for kwargs in cases:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                validate_film_plans(**self._valid(print_plan=FilmPrintPlan(**kwargs)))
        validate_film_plans(**self._valid(
            print_plan=FilmPrintPlan(medium_id="reversal_direct")
        ))

    def test_negative_joint_timing_owns_the_print(self) -> None:
        with self.assertRaises(ValueError):
            validate_film_plans(**self._valid(
                print_plan=FilmPrintPlan(medium_id="Endura", printer_y_cc=10.0)
            ))
        validate_film_plans(**self._valid(
            print_plan=FilmPrintPlan(
                medium_id="Endura", timing_policy="custom", printer_y_cc=10.0
            )
        ))

    def test_identity_finish_predicate(self) -> None:
        self.assertTrue(is_identity_finish(AnalogFinishPlan()))
        self.assertFalse(is_identity_finish(AnalogFinishPlan(grain_amount=0.5)))
        self.assertFalse(is_identity_finish(AnalogFinishPlan(bloom_amount=0.1)))


class StageAMathTests(unittest.TestCase):
    def test_grey_ramp_maps_to_the_loge_axis_exactly(self) -> None:
        """Neutral anchoring: a grey ramp's layer logE equals scene EV *
        log10(2) in every layer, for any positive observer matrix."""
        rng = np.random.default_rng(7)
        observer = rng.uniform(0.05, 1.0, (3, 3))
        ev = np.linspace(-6.0, 4.0, 21)
        grey = SCENE_MID * np.exp2(ev)[:, None].repeat(3, axis=1)
        log_e = layer_log_exposure(grey, observer)
        expected = (ev * LOG10_2)[:, None].repeat(3, axis=1)
        np.testing.assert_allclose(log_e, expected, atol=1e-9)

    def test_film_exposure_ev_is_a_scene_gain(self) -> None:
        """stage_a_amounts(x, Efilm=+1) == stage_a_amounts(2x, Efilm=0):
        the film exposure state IS light on the emulsion (plan §5.1)."""
        rng = np.random.default_rng(11)
        observer = rng.uniform(0.05, 1.0, (3, 3))
        le = np.linspace(-2.0, 2.0, 41)
        table = np.stack([
            0.2 + 1.1 / (1 + np.exp(-2.2 * (le + 0.1 * c))) for c in range(3)
        ], axis=1)
        rgb = rng.uniform(0.001, 2.0, (64, 3))
        a1 = stage_a_amounts(rgb, observer, le, table, film_exposure_ev=1.0)
        a2 = stage_a_amounts(rgb * 2.0, observer, le, table, film_exposure_ev=0.0)
        np.testing.assert_allclose(a1, a2, atol=1e-9)


class FilmV2AssetTests(unittest.TestCase):
    def test_schema_and_provenance_per_kind(self) -> None:
        """Every file in the modular family carries schema 9, its declared
        kind, sane domains and full-length source hashes (plan §7.1; 9 =
        Stage A provenance keys joined the ABI, review 2026-08-27 F8)."""
        stocks = _stock_files()
        self.assertGreaterEqual(len(stocks), 25)
        for stock in stocks:
            with self.subTest(kind="stock", name=stock):
                z = _load(stock)
                self.assertEqual(int(z["schema"]), 9)
                self.assertEqual(str(np.asarray(z["kind"])), "stock")
                self.assertTrue(np.all(z["amount_hi"] > z["amount_lo"]))
                self.assertEqual(float(z["exposure_ev_min"]), -2.0)
                self.assertEqual(float(z["exposure_ev_max"]), 2.0)
                for digest in z["source_sha256"]:
                    self.assertEqual(len(str(digest)), 64)
                media = [str(m) for m in z["media"]]
                self.assertIn(str(np.asarray(z["default_medium"])), media)
        self.assertGreaterEqual(len(_print_files()), 20)
        for path in _print_files():
            with self.subTest(kind="print_state", name=path.stem):
                z = dict(np.load(path, allow_pickle=False))
                self.assertEqual(int(z["schema"]), 9)
                self.assertEqual(str(np.asarray(z["kind"])), "print_state")
                n = int(z["n"])
                self.assertEqual(z["b1_volume"].shape, (n, n, n, 3))
                nodes = z["tau_nodes"]
                self.assertTrue(np.all(np.diff(nodes) > 0))
                self.assertEqual(z["tau"].shape, (nodes.size, 3))
                self.assertLessEqual(float(z["retimed_ev_min"]), 0.0)
                self.assertGreaterEqual(float(z["retimed_ev_max"]), 0.0)
                self.assertIn("premix refuted", str(np.asarray(z["premix_refuted_note"])))
        self.assertGreaterEqual(len(_b2_files()), 8)
        for path in _b2_files():
            with self.subTest(kind="b2", name=path.stem):
                z = dict(np.load(path, allow_pickle=False))
                self.assertEqual(int(z["schema"]), 9)
                self.assertEqual(str(np.asarray(z["kind"])), "b2")
                n = int(z["n"])
                vol = z["volume"].astype(np.float32)
                self.assertEqual(vol.shape, (n, n, n, 3))
                self.assertTrue(np.all(np.isfinite(vol)))
                self.assertTrue(np.all(vol >= 0.0))
                self.assertTrue(np.all(z["dye_hi"] > z["dye_lo"]))

    def test_characteristic_curves_are_monotone_in_trend(self) -> None:
        """Measured curves carry digitization wiggle (0.11-0.56% of span on
        the pilot trio); the gate is one dominant direction per layer with
        counter-motion under 1% of span."""
        for stock in _stock_files():
            with self.subTest(stock=stock):
                z = _load(stock)
                table = z["char_amounts"]
                d = np.diff(table, axis=0)
                for c in range(3):
                    dc = d[:, c]
                    span = float(table[:, c].max() - table[:, c].min())
                    pos = float(dc[dc > 0].sum())
                    neg = float(-dc[dc < 0].sum())
                    self.assertGreater(span, 0.0)
                    self.assertLess(min(pos, neg) / span, 0.01)

    def test_b2_is_shared_across_stocks(self) -> None:
        """§7.1: the print medium's B2 is one file reused by every stock
        printing on it — not re-baked per stock."""
        users: dict[str, set[str]] = {}
        for stock in _stock_files():
            z = _load(stock)
            for medium in z["media"]:
                users.setdefault(str(medium), set()).add(stock)
        shared = [m for m, u in users.items() if len(u) >= 3]
        self.assertTrue(shared, "no shared print medium found")
        for medium in users:
            self.assertTrue((ASSET_DIR / f"b2__{medium}.npz").is_file(), medium)


class FilmV2RuntimeTests(unittest.TestCase):
    """Fixed-path end-to-end: apply_film_core (factorized default) against
    the stock's shipped direct-chain oracle, both neutralization variants."""

    def _plan(self, preset: str, crossover: str = "off", **kw):
        from types import SimpleNamespace

        base = dict(
            curve_preset=preset, film_mode="full", film_crossover=crossover,
            film_exposure_ev=0.0, film_print_timing="fixed",
            film_print_medium="", film_print_exposure_ev=0.0,
            color_head_y=0.0, color_head_m=0.0,
            # These oracles certify the MEASURED spectral chain; the modelled
            # inter-image beta (mainline A) has its own gates in
            # test_film_mainline_a and must not be compared against baked
            # spectral truth.
            film_interimage="off",
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_runtime_matches_the_direct_chain_oracle(self) -> None:
        from dngscan.film_develop import apply_film_core

        for stock in _stock_files():
            z = _load(stock)
            rgb = (SCENE_MID * np.exp2(z["oracle_ev"].astype(np.float64))).astype(np.float32)
            truth = z["oracle_truth"].astype(np.float64)
            out = apply_film_core(rgb, self._plan(stock, "datasheet"))
            vis = truth > 5e-3
            err = np.abs(np.log2(
                np.maximum(out[vis].astype(np.float64), 1e-9)
                / np.maximum(truth[vis], 1e-9)
            ))
            with self.subTest(stock=stock):
                self.assertLessEqual(float(np.percentile(err, 99)), ORACLE_P99_GATE)
                self.assertLessEqual(float(err.max()), ORACLE_MAX_GATE)

    def test_ev0_neutral_prints_mid_grey(self) -> None:
        from dngscan.film_develop import apply_film_core

        for stock in PILOT:
            grey = np.full((1, 3), SCENE_MID, dtype=np.float32)
            out = apply_film_core(grey, self._plan(stock, "datasheet")).astype(np.float64)
            y = float(out[0] @ np.array([0.2627, 0.678, 0.0593]))
            with self.subTest(stock=stock):
                self.assertLess(abs(np.log2(y / SCENE_MID)), 0.02)

    def test_out_of_domain_exposure_hard_fails(self) -> None:
        from dngscan.film_develop import apply_film_core

        with self.assertRaises(ValueError):
            apply_film_core(
                np.full((4, 3), 0.18, dtype=np.float32),
                self._plan("portra400", film_exposure_ev=3.0),
            )

    def test_unknown_medium_fails_closed(self) -> None:
        from dngscan.film_develop import apply_film_core

        with self.assertRaises(ValueError):
            apply_film_core(
                np.full((2, 3), 0.18, dtype=np.float32),
                self._plan("portra400", film_print_medium="kodak_imaginary__translated"),
            )


class FilmV2RetimedTests(unittest.TestCase):
    """P2/P3 gates: retimed nodes hold EV0 neutrality, the deployed
    factorized runtime reproduces the midpoint oracle, the exposure axis is
    continuous, cross-medium pairings work without double tone mapping, and
    custom timing behaves as declared per-layer delta-tau."""

    RETIMED = ("portra400", "vision3250d")

    def _plan(self, preset, exposure=0.0, timing="retimed", crossover="off", **kw):
        from types import SimpleNamespace

        base = dict(
            curve_preset=preset, film_mode="full", film_crossover=crossover,
            film_exposure_ev=exposure, film_print_timing=timing,
            film_print_medium="", film_print_exposure_ev=0.0,
            color_head_y=0.0, color_head_m=0.0,
            # Spectral-oracle certification runs without the modelled
            # inter-image beta — see the runtime tests' plan helper above.
            film_interimage="off",
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_deployed_runtime_matches_midpoint_oracle(self) -> None:
        from dngscan.film_develop import apply_film_core

        for stock in self.RETIMED:
            ps = _load_print(stock)
            rgb = (SCENE_MID * np.exp2(ps["oracle_ev"].astype(np.float64))).astype(np.float32)
            for i, e_mid in enumerate(ps["oracle_exposures"].tolist()):
                truth = ps["oracle_truth"][i].astype(np.float64)
                out = apply_film_core(
                    rgb, self._plan(stock, exposure=float(e_mid), crossover="datasheet")
                )
                vis = truth > 5e-3
                err = np.abs(np.log2(
                    np.maximum(out[vis].astype(np.float64), 1e-9)
                    / np.maximum(truth[vis], 1e-9)
                ))
                with self.subTest(stock=stock, exposure=e_mid):
                    self.assertLessEqual(float(np.percentile(err, 99)), 0.03)
                    self.assertLessEqual(float(err.max()), 0.05)

    def test_retimed_nodes_hold_ev0_neutrality(self) -> None:
        from dngscan.film_develop import apply_film_core

        for stock in self.RETIMED:
            ps = _load_print(stock)
            for e in ps["tau_nodes"].tolist()[::4]:
                grey = np.full((1, 3), SCENE_MID, dtype=np.float32)
                out = apply_film_core(
                    grey, self._plan(stock, exposure=float(e), crossover="datasheet")
                ).astype(np.float64)
                y = float(out[0] @ np.array([0.2627, 0.678, 0.0593]))
                with self.subTest(stock=stock, node=e):
                    self.assertLess(abs(np.log2(y / SCENE_MID)), 0.02)
                    mx, mn = float(out[0].max()), float(out[0].min())
                    self.assertLess((mx - mn) / max(y, 1e-9), 0.02)

    def test_exposure_axis_is_continuous(self) -> None:
        from dngscan.film_develop import apply_film_core

        probe = np.array([[0.18, 0.18, 0.18], [0.6, 0.3, 0.15]], dtype=np.float32)
        evs = np.linspace(-2.0, 2.0, 81)
        outs = np.stack([
            apply_film_core(probe, self._plan("portra400", exposure=float(e)))
            for e in evs
        ])
        self.assertLess(float(np.abs(np.diff(outs, axis=0)).max()), 0.06)

    def test_fixed_timing_keeps_the_enlarger_setting(self) -> None:
        from dngscan.film_develop import apply_film_core

        grey = np.full((1, 3), SCENE_MID, dtype=np.float32)
        luma = np.array([0.2627, 0.678, 0.0593])
        fixed = apply_film_core(
            grey, self._plan("portra400", exposure=2.0, timing="fixed",
                             crossover="datasheet")
        ).astype(np.float64)
        retimed = apply_film_core(
            grey, self._plan("portra400", exposure=2.0, timing="retimed",
                             crossover="datasheet")
        ).astype(np.float64)
        self.assertGreater(np.log2(float(fixed[0] @ luma) / SCENE_MID), 0.5)
        self.assertLess(abs(np.log2(float(retimed[0] @ luma) / SCENE_MID)), 0.02)

    def test_reversal_rejects_retiming(self) -> None:
        from dngscan.film_develop import apply_film_core

        with self.assertRaises(ValueError):
            apply_film_core(
                np.full((2, 3), 0.18, dtype=np.float32),
                self._plan("velvia100", timing="retimed"),
            )

    def test_cross_medium_swap_without_double_tone_mapping(self) -> None:
        """§12 P3: the same negative on a different paper renders through the
        SAME Stage A + B1/tau/paper/B2 topology — mid-grey stays anchored
        (tone mapped exactly once) while the papers genuinely differ."""
        from dngscan.film_develop import apply_film_core

        grey = np.full((1, 3), SCENE_MID, dtype=np.float32)
        luma = np.array([0.2627, 0.678, 0.0593])
        chroma = np.array([[0.6, 0.3, 0.15], [0.05, 0.2, 0.5]], dtype=np.float32)
        default = apply_film_core(
            chroma, self._plan("portra400", timing="fixed", crossover="datasheet")
        )
        alt = apply_film_core(
            chroma, self._plan(
                "portra400", timing="fixed", crossover="datasheet",
                film_print_medium="kodak_supra_endura__translated",
            )
        )
        self.assertGreater(float(np.abs(alt - default).max()), 1e-3)
        for medium in ("", "kodak_supra_endura__translated"):
            out = apply_film_core(
                grey, self._plan("portra400", timing="fixed",
                                 crossover="datasheet", film_print_medium=medium)
            ).astype(np.float64)
            y = float(out[0] @ luma)
            with self.subTest(medium=medium or "default"):
                self.assertLess(abs(np.log2(y / SCENE_MID)), 0.02)

    def test_custom_timing_delta_tau_semantics(self) -> None:
        """+Y CC attenuates the blue-sensitive layer (print moves away from
        yellow: b axis down); +1 EV manual print exposure DARKENS the print
        (positive paper: more light, more density); custom + bounded
        neutralization is refused."""
        from dngscan.film_develop import apply_film_core

        grey = np.full((1, 3), SCENE_MID, dtype=np.float32)
        base = apply_film_core(
            grey, self._plan("portra400", timing="custom", crossover="datasheet")
        ).astype(np.float64)
        y30 = apply_film_core(
            grey, self._plan("portra400", timing="custom", crossover="datasheet",
                             color_head_y=30.0)
        ).astype(np.float64)
        # b* proxy: blue channel rises relative to red+green when yellow drops.
        def b_axis(rgb):
            return float(rgb[0, 2] - 0.5 * (rgb[0, 0] + rgb[0, 1]))

        self.assertGreater(b_axis(y30), b_axis(base))
        darker = apply_film_core(
            grey, self._plan("portra400", timing="custom", crossover="datasheet",
                             film_print_exposure_ev=1.0)
        ).astype(np.float64)
        luma = np.array([0.2627, 0.678, 0.0593])
        self.assertLess(float(darker[0] @ luma), float(base[0] @ luma))
        with self.assertRaises(ValueError):
            apply_film_core(
                grey, self._plan("portra400", timing="custom", crossover="off")
            )

    def test_plan_compiler_enforces_full_only_and_reversal_contract(self) -> None:
        from tests.golden_support import build_daylight_wide_dr
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        with self.assertRaises(ValueError):
            build_render_plan(
                scene.bundle, scene.analysis, "agx", "srgb",
                film_curve="portra400", film_mode="observe",
                film_exposure_ev=1.0,
            )
        with self.assertRaises(ValueError):
            build_render_plan(
                scene.bundle, scene.analysis, "agx", "srgb",
                film_curve="velvia100", film_mode="full",
                film_print_timing="retimed",
            )
        plan = build_render_plan(
            scene.bundle, scene.analysis, "agx", "srgb",
            film_curve="portra400", film_mode="full",
            film_exposure_ev=-1.5, film_print_timing="retimed",
        )
        self.assertEqual(float(plan.tone.film_exposure_ev), -1.5)
        self.assertEqual(plan.tone.film_print_timing, "retimed")
        # custom unlocks the colour head in full (modelled), datasheet only.
        plan = build_render_plan(
            scene.bundle, scene.analysis, "agx", "srgb",
            film_curve="portra400", film_mode="full",
            film_print_timing="custom", film_crossover="datasheet",
            color_head_y=15.0,
        )
        self.assertEqual(plan.tone.film_print_timing, "custom")
        self.assertEqual(float(plan.tone.color_head_y), 15.0)

    def test_gui_service_rejects_exposure_outside_full(self) -> None:
        from dngscan.gui.service import parse_film_params

        with self.assertRaises(ValueError):
            parse_film_params({
                "filmCurve": "portra400", "filmMode": "observe",
                "filmExposure": 1.0,
            })
        parsed = parse_film_params({
            "filmCurve": "portra400", "filmMode": "full",
            "filmExposure": -0.5, "filmPrintTiming": "retimed",
        })
        self.assertEqual(parsed[6], -0.5)
        self.assertEqual(parsed[7], "retimed")
        # neutralization alias contract: both keys together hard-fail.
        with self.assertRaises(ValueError):
            parse_film_params({
                "filmCurve": "portra400", "filmMode": "full",
                "filmNeutralization": "bounded", "filmCrossover": "off",
            })
        parsed = parse_film_params({
            "filmCurve": "portra400", "filmMode": "full",
            "filmNeutralization": "datasheet", "filmPrintTiming": "custom",
            "colorHeadY": 10.0,
        })
        self.assertEqual(parsed[3], "datasheet")
        self.assertEqual(parsed[7], "custom")


if __name__ == "__main__":
    unittest.main()
