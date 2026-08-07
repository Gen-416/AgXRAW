# SPDX-License-Identifier: GPL-3.0-or-later
"""film v2 P1 gates: plan objects, Stage A math, schema-v5 two-stage assets.

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


def _all_stocks() -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in ASSET_DIR.glob("*.npz")))


# Deployed-bytes gate (plan §13): tighter than the v1 scene-EV LUT's ~0.03
# stop because the steep characteristic curves left the 3D grid. Measured at
# build time across all 25 assets: p99 0.001-0.015, max 0.001-0.035 — the max
# outlier is superia400 (0.0343), the stock whose datasheet self-consistency
# is documented as the roster's weakest; the gate is set just above it
# rather than pretending the data is cleaner than it is.
ORACLE_P99_GATE = 0.02
ORACLE_MAX_GATE = 0.04


def _load(stock: str) -> dict:
    path = ASSET_DIR / f"{stock}.npz"
    if not path.is_file():
        raise unittest.SkipTest(f"missing {path}; run tools/build_film_v2_assets.py")
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
        validate_film_plans(**self._valid(
            development=FilmDevelopmentPlan(
                recipe_id="editorial_custom", contrast_delta=0.1,
                provenance="editorial",
            )
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
    def test_schema_and_provenance(self) -> None:
        stocks = _all_stocks() or PILOT
        self.assertGreaterEqual(len(stocks), 25)
        for stock in stocks:
            with self.subTest(stock=stock):
                z = _load(stock)
                self.assertEqual(int(z["schema"]), 5)
                self.assertEqual(str(z["input_space"]), "scene_rec2020_via_amounts")
                self.assertEqual(str(z["timing_policy"]), "fixed_q0")
                self.assertEqual(int(z["n"]), 65)
                self.assertEqual(z["volume"].shape, (65, 65, 65, 3))
                self.assertEqual(z["volume"].dtype, np.float16)
                for digest in z["source_sha256"]:
                    self.assertEqual(len(str(digest)), 64)
                self.assertTrue(np.all(z["amount_hi"] > z["amount_lo"]))
                self.assertEqual(float(z["exposure_ev_min"]), -2.0)
                self.assertEqual(float(z["exposure_ev_max"]), 2.0)
                self.assertTrue(np.all(np.isfinite(z["volume"].astype(np.float32))))
                self.assertTrue(np.all(z["volume"].astype(np.float32) >= 0.0))

    def test_characteristic_curves_are_monotone_in_trend(self) -> None:
        """The MEASURED curves carry digitization-noise wiggles (measured:
        0.11-0.56% of span across the pilot trio), so strict monotonicity is
        not the honest gate. The gate: one dominant direction per layer with
        counter-motion under 1% of the layer's span — enough to catch a
        corrupted table or a sign flip without denying the data's texture."""
        for stock in PILOT:
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
                    self.assertLess(
                        min(pos, neg) / span, 0.01,
                        f"{stock} layer {c}: counter-motion exceeds 1% of span",
                    )

    def test_deployed_bytes_reproduce_the_direct_chain(self) -> None:
        """The shipped f16 volume + analytic Stage A vs the float64 direct
        spectral chain truth baked into the asset (plan §13)."""
        from dngscan.film_develop import _tetrahedral

        for stock in _all_stocks() or PILOT:
            with self.subTest(stock=stock):
                z = _load(stock)
                rgb = SCENE_MID * np.exp2(z["oracle_ev"].astype(np.float64))
                amounts = stage_a_amounts(
                    rgb, z["observer"], z["char_le"], z["char_amounts"],
                    film_exposure_ev=0.0,
                    anchor_ev_offset=float(z["anchor_ev_offset"]),
                )
                u = amounts_to_unit(amounts, z["amount_lo"], z["amount_hi"])
                out = _tetrahedral(
                    z["volume"].astype(np.float32), u.astype(np.float32), int(z["n"])
                )
                truth = z["oracle_truth"].astype(np.float64)
                vis = truth > 5e-3
                err = np.abs(np.log2(
                    np.maximum(out[vis], 1e-9) / np.maximum(truth[vis], 1e-9)
                ))
                self.assertLessEqual(float(np.percentile(err, 99)), ORACLE_P99_GATE)
                self.assertLessEqual(float(err.max()), ORACLE_MAX_GATE)
                # And the build-time record matches what we just measured.
                self.assertLessEqual(
                    abs(float(np.percentile(err, 99)) - float(z["oracle_p99_stop"])),
                    5e-3,
                )

    def test_stage_b_neutral_axis_hits_mid_grey_at_ev0(self) -> None:
        """Fixed timing q(0): the two-stage composite maps neutral EV0 to a
        neutral 0.18 (plan §7.2, before any runtime neutralization)."""
        from dngscan.film_develop import _tetrahedral

        for stock in PILOT:
            with self.subTest(stock=stock):
                z = _load(stock)
                grey = np.full((1, 3), SCENE_MID)
                amounts = stage_a_amounts(
                    grey, z["observer"], z["char_le"], z["char_amounts"],
                    anchor_ev_offset=float(z["anchor_ev_offset"]),
                )
                u = amounts_to_unit(amounts, z["amount_lo"], z["amount_hi"])
                out = _tetrahedral(
                    z["volume"].astype(np.float32), u.astype(np.float32), int(z["n"])
                )
                y = float(out[0] @ np.array([0.2627, 0.6780, 0.0593]))
                self.assertLess(abs(np.log2(y / SCENE_MID)), 0.02)


class FilmV2RuntimeTests(unittest.TestCase):
    """End-to-end: apply_film_core's v2 default against the asset's own
    direct-chain oracle, both neutralization variants (plan §12 P1: the
    two-stage runtime reproduces full v1's physical semantics within the
    direct-chain tolerance — not by inverting final images)."""

    def _plan(self, preset: str, crossover: str):
        from types import SimpleNamespace

        return SimpleNamespace(
            curve_preset=preset, film_mode="full", film_crossover=crossover,
            film_exposure_ev=0.0,
        )

    def test_runtime_matches_the_direct_chain_oracle(self) -> None:
        from dngscan.film_develop import apply_film_core

        for stock in PILOT:
            z = _load(stock)
            rgb = (SCENE_MID * np.exp2(z["oracle_ev"].astype(np.float64))).astype(np.float32)
            truth = z["oracle_truth"].astype(np.float64)
            out = apply_film_core(rgb, self._plan(stock, "datasheet"))
            vis = truth > 5e-3
            err = np.abs(np.log2(
                np.maximum(out[vis].astype(np.float64), 1e-9)
                / np.maximum(truth[vis], 1e-9)
            ))
            with self.subTest(stock=stock, variant="datasheet"):
                self.assertLessEqual(float(np.percentile(err, 99)), ORACLE_P99_GATE)
                self.assertLessEqual(float(err.max()), ORACLE_MAX_GATE)
            # Bounded variant: divide the truth by the shipped cast exactly as
            # the runtime does — the quotient's visible error equals the
            # datasheet's by construction.
            cast_ev = z["cast_ev"].astype(np.float64)
            cast = z["cast_bounded"].astype(np.float64)
            ev_y = np.log2(np.maximum(
                rgb.astype(np.float64) @ np.array([0.2627, 0.678, 0.0593]), 1e-9
            ) / SCENE_MID)
            truth_nz = truth.copy()
            for c in range(3):
                truth_nz[:, c] /= np.interp(ev_y, cast_ev, cast[:, c])
            out_nz = apply_film_core(rgb, self._plan(stock, "off"))
            err = np.abs(np.log2(
                np.maximum(out_nz[vis].astype(np.float64), 1e-9)
                / np.maximum(truth_nz[vis], 1e-9)
            ))
            with self.subTest(stock=stock, variant="bounded"):
                self.assertLessEqual(float(np.percentile(err, 99)), ORACLE_P99_GATE)
                self.assertLessEqual(float(err.max()), ORACLE_MAX_GATE)

    def test_out_of_domain_exposure_hard_fails(self) -> None:
        from dngscan.film_develop import apply_film_core
        from types import SimpleNamespace

        plan = SimpleNamespace(
            curve_preset="portra400", film_mode="full", film_crossover="off",
            film_exposure_ev=3.0,
        )
        with self.assertRaises(ValueError):
            apply_film_core(np.full((4, 3), 0.18, dtype=np.float32), plan)

    def test_legacy_backend_still_serves_v1_bytes(self) -> None:
        import os

        from dngscan.film_develop import apply_film_core, _LUT_DIR

        prev = os.environ.get("DNGSCAN_FILM_LEGACY_LUT")
        os.environ["DNGSCAN_FILM_LEGACY_LUT"] = "1"
        try:
            with np.load(_LUT_DIR / "portra400.npz", allow_pickle=False) as z:
                rgb = (0.18 * np.exp2(z["oracle_ev"].astype(np.float64))).astype(np.float32)
                want = z["oracle_datasheet"].astype(np.float64)
            out = apply_film_core(rgb, self._plan("portra400", "datasheet"))
            vis = want > 5e-3
            err = np.abs(np.log2(
                np.maximum(out[vis].astype(np.float64), 1e-9)
                / np.maximum(want[vis], 1e-9)
            ))
            self.assertLessEqual(float(np.percentile(err, 99)), 0.05)
        finally:
            if prev is None:
                os.environ.pop("DNGSCAN_FILM_LEGACY_LUT", None)
            else:
                os.environ["DNGSCAN_FILM_LEGACY_LUT"] = prev


class FilmV2RetimedTests(unittest.TestCase):
    """P2 gates (plan §13): retimed nodes hit EV0 neutrality, the deployed
    factorized runtime reproduces the midpoint direct-chain oracle, the
    exposure axis is continuous, and the per-medium contract fails closed."""

    RETIMED = ("portra400", "vision3250d")

    def _plan(self, preset, exposure=0.0, timing="retimed", crossover="off"):
        from types import SimpleNamespace

        return SimpleNamespace(
            curve_preset=preset, film_mode="full", film_crossover=crossover,
            film_exposure_ev=exposure, film_print_timing=timing,
        )

    def test_deployed_runtime_matches_midpoint_oracle(self) -> None:
        from dngscan.film_develop import apply_film_core

        for stock in self.RETIMED:
            z = _load(stock)
            rgb = (SCENE_MID * np.exp2(z["retimed_oracle_ev"].astype(np.float64))).astype(np.float32)
            for i, e_mid in enumerate(z["retimed_oracle_exposures"].tolist()):
                truth = z["retimed_oracle_truth"][i].astype(np.float64)
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
        """plan §13: every retimed node prints neutral mid-grey back to
        Y=0.18 with near-zero chroma (DeltaE00 <= 0.5 stand-in: Oklab
        distance x100 <= 0.5)."""
        from dngscan.film_develop import apply_film_core

        for stock in self.RETIMED:
            z = _load(stock)
            for e in z["retimed_nodes"].tolist():
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
        """plan §13: the slider must not jump across q nodes."""
        from dngscan.film_develop import apply_film_core

        stock = "portra400"
        probe = np.array([[0.18, 0.18, 0.18], [0.6, 0.3, 0.15]], dtype=np.float32)
        evs = np.linspace(-2.0, 2.0, 81)
        outs = np.stack([
            apply_film_core(probe, self._plan(stock, exposure=float(e)))
            for e in evs
        ])
        step = np.abs(np.diff(outs, axis=0))
        self.assertLess(float(step.max()), 0.06)

    def test_fixed_timing_keeps_the_enlarger_setting(self) -> None:
        """fixed: +2 EV on the emulsion prints BRIGHTER through the same
        q(0); retimed prints back near mid-grey. The two recipes must
        actually differ (plan §5.3)."""
        from dngscan.film_develop import apply_film_core

        grey = np.full((1, 3), SCENE_MID, dtype=np.float32)
        fixed = apply_film_core(
            grey, self._plan("portra400", exposure=2.0, timing="fixed",
                             crossover="datasheet")
        ).astype(np.float64)
        retimed = apply_film_core(
            grey, self._plan("portra400", exposure=2.0, timing="retimed",
                             crossover="datasheet")
        ).astype(np.float64)
        luma = np.array([0.2627, 0.678, 0.0593])
        y_fixed = float(fixed[0] @ luma)
        y_retimed = float(retimed[0] @ luma)
        self.assertGreater(np.log2(y_fixed / SCENE_MID), 0.5)
        self.assertLess(abs(np.log2(y_retimed / SCENE_MID)), 0.02)

    def test_retimed_without_assets_fails_closed(self) -> None:
        from dngscan.film_develop import apply_film_core

        with self.assertRaises(ValueError):
            apply_film_core(
                np.full((2, 3), 0.18, dtype=np.float32),
                self._plan("gold200", exposure=1.0, timing="retimed"),
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


if __name__ == "__main__":
    unittest.main()
