# SPDX-License-Identifier: GPL-3.0-or-later
"""Appearance P3 gates: the print-balanced neutralization policy (plan §8).

Three policies, one construction property each:

- technical-neutral (historical "bounded", frozen): per-pixel
  exposure-indexed cast division — the whole grey scale is digitally
  neutral;
- print-balanced: the SAME cast table evaluated ONCE at the EV0 anchor — a
  constant per-channel balance. Mid grey prints neutral BY CONSTRUCTION
  (identical to technical-neutral at EV0), and away from EV0 the grey
  scale keeps the medium's own crossover, which is precisely the print
  character the per-pixel form erases;
- native (historical "datasheet"): no correction.

The sharp identity that pins print-balanced: its ratio to native is a
constant per channel at every exposure.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from dngscan.film_develop import apply_film_core


def _plan(crossover: str, **kw) -> SimpleNamespace:
    base = dict(
        curve_preset="portra400", film_mode="full", film_crossover=crossover,
        film_exposure_ev=0.0, film_print_timing="fixed", film_print_medium="",
        film_print_exposure_ev=0.0, color_head_y=0.0, color_head_m=0.0,
        film_development="measured_default", film_dev_contrast=0.0,
        film_dev_fog=0.0, film_dev_density=0.0, film_compression=0.0,
        film_compression_knee=2.0, film_highlight_density=0.0,
        film_grain=0.0, film_halation=0.0, film_bloom=0.0, film_optics_seed=0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _ramp(evs: np.ndarray) -> np.ndarray:
    return (0.18 * np.exp2(evs))[:, None].repeat(3, 1).astype(np.float32)


class ConstructionTests(unittest.TestCase):
    EVS = np.linspace(-5.0, 3.0, 17)

    def _triple(self, stock: str):
        ramp = _ramp(self.EVS)
        return tuple(
            np.asarray(apply_film_core(ramp, _plan(c, curve_preset=stock)),
                       np.float64)
            for c in ("off", "print", "datasheet")
        )

    def test_ev0_is_anchored_exactly(self) -> None:
        """At the anchor, print-balanced IS technical-neutral: both divide
        by the same cast value there. No tolerance games — the same
        interpolation on the same table at the same point."""
        for stock in ("portra400", "vision3250d", "velvia100"):
            with self.subTest(stock=stock):
                tech, prnt, _ = self._triple(stock)
                i0 = int(np.argmin(np.abs(self.EVS)))
                np.testing.assert_allclose(prnt[i0], tech[i0], rtol=1e-6)

    def test_the_ratio_to_native_is_a_constant(self) -> None:
        """The whole definition in one assertion: print-balanced divides
        native by ONE number per channel, so their ratio cannot vary with
        exposure. A per-pixel term sneaking back in breaks this first."""
        for stock in ("portra400", "velvia100"):
            with self.subTest(stock=stock):
                _, prnt, natv = self._triple(stock)
                good = natv > 1e-3   # the deep toe divides noise by noise
                ratio = np.where(good, prnt / np.maximum(natv, 1e-12), np.nan)
                i0 = int(np.argmin(np.abs(self.EVS)))
                # float32 through two 65^3 volumes leaves ~1e-3 relative
                # noise; the property being pinned is "no EXPOSURE trend",
                # which a per-pixel term would break at percent scale.
                sel = ratio[good.all(axis=1)]
                np.testing.assert_allclose(
                    sel, np.broadcast_to(ratio[i0], sel.shape), rtol=3e-3,
                    err_msg="print-balanced must be a constant balance",
                )

    def test_the_ends_keep_their_crossover(self) -> None:
        """The reason the policy exists: away from EV0 the grey scale must
        NOT be digitally neutral — the medium's exposure-dependent cast
        survives, visibly and continuously."""
        tech, prnt, _ = self._triple("portra400")
        self.assertGreater(float(np.abs(prnt[0] - tech[0]).max()), 1e-4)
        self.assertGreater(float(np.abs(prnt[-1] - tech[-1]).max()), 1e-4)
        # continuity: the deviation from technical grows smoothly from the
        # anchor, no step
        dev = np.abs(prnt - tech).max(axis=1)
        self.assertLess(float(np.abs(np.diff(dev)).max()), 0.05)

    def test_technical_neutral_is_byte_frozen(self) -> None:
        """P3 must not have moved the default: "off" is the frozen
        historical behaviour, asserted against the appearance freeze by the
        wider suites; here the cheap invariant — unknown values still fail
        closed and 'off' still runs the per-pixel form (ratio to native is
        NOT constant)."""
        tech, _, natv = self._triple("portra400")
        ratio = tech / np.maximum(natv, 1e-12)
        spread = float(np.abs(ratio - ratio[0]).max())
        self.assertGreater(spread, 1e-3,
                           "technical-neutral must stay per-pixel")
        with self.assertRaises(ValueError):
            apply_film_core(_ramp(self.EVS), _plan("banana"))

    def test_custom_timing_rejects_digital_neutralization(self) -> None:
        for crossover in ("off", "print"):
            with self.subTest(crossover=crossover):
                with self.assertRaises(ValueError):
                    apply_film_core(
                        _ramp(self.EVS),
                        _plan(crossover, film_print_timing="custom"),
                    )


class NamingMigrationTests(unittest.TestCase):
    def test_cli_maps_canonical_and_deprecated_names(self) -> None:
        from dngscan.cli import NEUTRALIZATION_TO_CROSSOVER

        self.assertEqual(
            NEUTRALIZATION_TO_CROSSOVER,
            {
                "technical-neutral": "off", "bounded": "off",
                "print-balanced": "print",
                "native": "datasheet", "datasheet": "datasheet",
            },
        )

    def test_service_maps_names_and_rejects_unknown(self) -> None:
        from dngscan.gui.service import parse_film_params

        base = {"film": "portra400", "filmMode": "full"}
        for name, expect in (
            ("technical-neutral", "off"), ("bounded", "off"),
            ("print-balanced", "print"), ("native", "datasheet"),
        ):
            with self.subTest(name=name):
                out = parse_film_params({**base, "filmNeutralization": name})
                self.assertEqual(out[3], expect)
        with self.assertRaises(ValueError):
            parse_film_params({**base, "filmNeutralization": "banana"})

    def test_the_compiled_plan_records_canonical_names(self) -> None:
        from dngscan.tone import build_render_plan
        from tests.golden_support import all_scenes

        scene = all_scenes()["daylight_wide_dr"]
        for crossover, canonical in (
            ("off", "technical-neutral"),
            ("print", "print-balanced"),
            ("datasheet", "native"),
        ):
            with self.subTest(crossover=crossover):
                plan = build_render_plan(
                    scene.bundle, scene.analysis, "agx", "srgb",
                    film_curve="portra400", film_mode="full",
                    film_crossover=crossover,
                )
                self.assertEqual(
                    plan.film[2].neutralization_policy, canonical
                )


if __name__ == "__main__":
    unittest.main()
