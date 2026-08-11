# SPDX-License-Identifier: GPL-3.0-or-later
"""Review A8 regression gates: full-well plausibility, deep recipe
immutability, unclipped gamut metrics, plan finiteness, report topology,
and the Core Image runtime probe contract."""
from __future__ import annotations

import copy
import dataclasses
import pickle
import unittest

import numpy as np

from dngscan import film_appearance as fa


class FullwellPlausibilityTests(unittest.TestCase):
    def test_an_ordinary_plateau_does_not_become_the_full_well(self) -> None:
        """A8 item 1 repro: 300 px stuck at 8000 under WhiteLevel 16383
        must NOT override the metadata — that deflated clip %, DR, the CFA
        clip mask and HDR headroom in one stroke."""
        from dngscan.analysis import detect_ceilings, resolve_fullwell

        raw = np.full((1000, 1000), 3000, np.uint16)
        raw.flat[:300] = 8000
        colors = np.zeros_like(raw, np.int32)
        sat = {0: 16383}
        ce, _, _, ok = detect_ceilings(raw, colors, [0], sat)
        fw, _, note, _ = resolve_fullwell([0], ce, ok, sat)
        self.assertEqual(fw, 16383)
        self.assertIn("metadata", note)

    def test_plateaus_at_three_quarters_and_nine_tenths_stay_rejected(self) -> None:
        """A9 item 2: A8's 0.75 gate still let a 13000/16383 plateau
        through. Metadata is authoritative; only a pile within the narrow
        0.95 tolerance may override. Regression at 0.76/0.80/0.90."""
        from dngscan.analysis import detect_ceilings, resolve_fullwell

        colors = np.zeros((1000, 1000), np.int32)
        sat = {0: 16383}
        for frac in (0.76, 0.80, 0.90):
            raw = np.full((1000, 1000), 3000, np.uint16)
            raw.flat[:300] = int(16383 * frac)
            ce, _, _, ok = detect_ceilings(raw, colors, [0], sat)
            fw, _, _, _ = resolve_fullwell([0], ce, ok, sat)
            with self.subTest(frac=frac):
                self.assertEqual(fw, 16383)

    def test_a_genuine_near_white_pile_still_overrides(self) -> None:
        from dngscan.analysis import detect_ceilings, resolve_fullwell

        raw = np.full((1000, 1000), 3000, np.uint16)
        raw.flat[:300] = 16200
        colors = np.zeros_like(raw, np.int32)
        sat = {0: 16383}
        ce, _, _, ok = detect_ceilings(raw, colors, [0], sat)
        fw, _, _, _ = resolve_fullwell([0], ce, ok, sat)
        self.assertEqual(fw, 16200)


class DeepImmutabilityTests(unittest.TestCase):
    def _plan(self):
        return fa.compile_appearance_plan(
            "reference", 1.0, stock_id="portra400",
            medium_id="kodak_portra_endura__translated",
        )

    def test_every_demonstrated_mutation_hole_is_closed(self) -> None:
        """A8 item 2: the A7 dict subclass leaked through |=, the dict
        C-methods and a re-invoked __init__ — each swapped the payload
        under EVERY cached plan. The Mapping wrapper has no inherited
        mutation surface."""
        r = self._plan().recipe
        with self.assertRaises(TypeError):
            r["x"] = 1                     # type: ignore[index]
        with self.assertRaises(TypeError):
            r |= {"x": 1}                  # type: ignore[operator]
        with self.assertRaises(TypeError):
            dict.__setitem__(r, "x", 1)    # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            r.__init__({"x": 1})
        with self.assertRaises(TypeError):
            object.__getattribute__(r, "__setattr__")("_data", {})
        self.assertEqual(r["provenance"], "editorial-authored")

    def test_serialization_contracts_survive(self) -> None:
        plan = self._plan()
        p2 = pickle.loads(pickle.dumps(plan))
        self.assertEqual(p2.recipe_id, plan.recipe_id)
        self.assertIs(copy.copy(plan.recipe), plan.recipe)
        self.assertIn("chroma_knee", dataclasses.asdict(plan)["recipe"])

    def test_the_cache_still_shares_one_object(self) -> None:
        self.assertIs(self._plan().recipe, self._plan().recipe)


class GamutMetricsTests(unittest.TestCase):
    def test_saturated_rec2020_blue_is_counted(self) -> None:
        """A8 item 3: the packed uint16 XYZ clipped Z (~1.061 for pure
        blue) at the representable ceiling, under-reporting exactly the
        saturated blues this metric exists to count. The scene Rec.2020
        buffer is now the unclipped source."""
        from dngscan.analysis import compute_gamut_metrics

        scene = np.zeros((8, 1, 3), np.uint16)
        scene[..., 2] = 65535
        y = np.full((8, 1), 0.06, np.float32)
        pct, _ = compute_gamut_metrics(scene, 65535.0, y, ("sRGB",))
        self.assertEqual(pct["sRGB"], 100.0)


class PlanFinitenessTests(unittest.TestCase):
    def test_nan_inf_and_absurd_cc_fail_closed(self) -> None:
        from dngscan.film_plans import (
            AnalogFinishPlan, FilmDevelopmentPlan, FilmExposurePlan,
            FilmPrintPlan, validate_film_plans,
        )

        base = dict(
            exposure=FilmExposurePlan(stock_id="portra400"),
            development=FilmDevelopmentPlan(
                interimage_mode="off", interimage_beta=0.0
            ),
            print_plan=FilmPrintPlan(
                medium_id="kodak_portra_endura__translated"
            ),
            finish=AnalogFinishPlan(),
        )
        validate_film_plans(**base)   # the clean plan must stay legal
        medium = "kodak_portra_endura__translated"
        for kw in (
            dict(finish=AnalogFinishPlan(grain_amount=float("nan"))),
            dict(print_plan=FilmPrintPlan(
                medium_id=medium, timing_policy="custom",
                print_exposure_ev=float("nan"))),
            dict(print_plan=FilmPrintPlan(
                medium_id=medium, timing_policy="custom",
                printer_y_cc=float("inf"))),
            dict(print_plan=FilmPrintPlan(
                medium_id=medium, timing_policy="custom",
                printer_y_cc=9000.0)),
        ):
            with self.subTest(kw=str(kw)[:60]):
                with self.assertRaises(ValueError):
                    validate_film_plans(**{**base, **kw})


class TopologyNarrativeTests(unittest.TestCase):
    def test_the_report_describes_the_factorised_chain(self) -> None:
        """A8 item 5: the run report must describe the chain that RUNS —
        reviews were being steered by a deleted monolithic-LUT story."""
        from dngscan.report import jpeg_policy_cn

        line = jpeg_policy_cn("agx", "srgb", "portra400", "full")
        self.assertNotIn("65³", line)
        for stage in ("Stage A", "B1", "B2", "中性化"):
            self.assertIn(stage, line)


class RuntimeProbeTests(unittest.TestCase):
    def test_runtime_probe_never_raises_and_is_cached(self) -> None:
        from dngscan import coreimage_decode as ci

        first = ci.runtime_available()
        self.assertIsInstance(first, bool)
        self.assertEqual(ci.runtime_available(), first)
        if not ci.available():
            self.assertFalse(first)

    def test_the_probe_renders_the_real_workload_parameters(self) -> None:
        """A11 item 2: pin the probe's render call — RGBAh, rowBytes 8,
        extended linear Rec.2020 — with a fake Quartz, so a host (or CI
        runner) without Core Image still guards the parameter set."""
        import sys
        import unittest.mock as mock

        from dngscan import coreimage_decode as ci

        calls = {}

        class _Ctx:
            def render_toBitmap_rowBytes_bounds_format_colorSpace_(
                self, img, buf, row_bytes, bounds, fmt, cs
            ):
                calls.update(row_bytes=row_bytes, fmt=fmt, cs=cs,
                             buflen=len(buf))

        fake = mock.MagicMock()
        fake.kCIFormatRGBAh = "RGBAh"
        fake.kCGColorSpaceExtendedLinearITUR_2020 = "ext2020-name"
        fake.CGColorSpaceCreateWithName = lambda name: f"cs:{name}"
        fake.CIContext.contextWithOptions_ = lambda opts: _Ctx()
        img = mock.MagicMock()
        img.imageByCroppingToRect_ = lambda rect: img
        fake.CIImage.imageWithColor_ = lambda color: img

        with mock.patch.dict(sys.modules, {"Quartz": fake,
                                           "Foundation": mock.MagicMock()}),              mock.patch.object(ci, "available", return_value=True),              mock.patch.dict(ci._RUNTIME_AVAILABLE, {}, clear=True),              mock.patch.dict(ci._CONTEXTS, {}, clear=True):
            self.assertTrue(ci.runtime_available())
        self.assertEqual(calls["fmt"], "RGBAh")
        self.assertEqual(calls["row_bytes"], 8)
        self.assertEqual(calls["buflen"], 8)
        self.assertEqual(calls["cs"], "cs:ext2020-name")


if __name__ == "__main__":
    unittest.main()
