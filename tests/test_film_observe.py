# SPDX-License-Identifier: GPL-3.0-or-later
"""Observe film simulation (dngscan/film): asset, plan transform, tone core,
grain and the CLI/GUI contracts."""
from __future__ import annotations

import dataclasses
import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np

from dngscan import film
from dngscan.film import FilmSelection, FilmSpatial, apply_film_plan, film_tone_core
from dngscan.film.color import REC2020_LUMA
from dngscan.film.grain import (FilmGeometry, GrainProfile, apply_observe_grain, master,
                                sample_field)
from dngscan.models import ToneCompressionPlan
from dngscan.render import apply_agx_core, apply_tone_core

STOCKS = ("portra400", "velvia100", "superia400", "vision3500t")


def _plan(**kw) -> ToneCompressionPlan:
    base = dict(target_gamut="Rec2020", luma_p1=0.01, luma_p50=0.18, luma_p99=2.0,
                luma_p999=4.0, black_ev=-7.0, white_ev=5.0, dynamic_range_ev=12.0,
                contrast=1.8, toe_power=1.5, shoulder_power=1.5, chroma_p95=0.1,
                negative_rgb_pct=0.0, over_rgb_pct=0.0)
    base.update(kw)
    return ToneCompressionPlan(**base)


def _film_plan(stock: str, grain: float = 0.0) -> ToneCompressionPlan:
    return apply_film_plan(_plan(punch_strength=0.4, view_brightness=1.2, agx_primaries="punchy"),
                           FilmSelection(stock, grain, 7))


class FilmAssetTests(unittest.TestCase):
    def test_asset_schema_and_stocks(self):
        a = film._asset()
        self.assertEqual(a["schema"], film.ASSET_SCHEMA)
        self.assertEqual(len(film.film_choices()), 21)
        for key, s in a["stocks"].items():
            self.assertIn(s["process"], ("negative", "reversal"))
            if s["process"] == "negative":
                self.assertEqual(s["exposure_offset_ev"], 0.0)
            else:
                self.assertLess(s["exposure_offset_ev"], 0.0)
            self.assertGreater(s["grain_scale"], 0.0)

    def test_cc_by_sa_attribution_travels_with_the_data(self):
        lic = film._asset()["license"]
        self.assertIn("Andrea Volpato", lic["attribution"])
        self.assertIn("https://github.com/andreavolpato/spektrafilm", lic["attribution"])
        self.assertIn("CC BY-SA 4.0", lic["license"])
        here = Path(film.__file__).resolve().parent
        text = (here / "SPEKTRAFILM_LICENSE.txt").read_text(encoding="utf-8")
        self.assertIn("Attribution-ShareAlike 4.0", text)
        self.assertIn("Modified by", (here / "CHANGELOG.txt").read_text(encoding="utf-8"))

    def test_master_field_rebuilds_the_frozen_realization(self):
        a = film._asset()["grain"]
        ii, rms48 = master(GrainProfile.from_asset(a))
        self.assertAlmostEqual(rms48, a["check"]["aperture_rms"], places=7)
        probe = (ii[1:9, 1:9] - ii[1:9, :8] - ii[:8, 1:9] + ii[:8, :8]).astype(np.float64)
        np.testing.assert_allclose(probe, np.array(a["check"]["field_probe_8x8"]), atol=2e-6)

    def test_unknown_stock_and_grain_domain_fail_closed(self):
        with self.assertRaises(ValueError):
            film.validate_film("kodak_imaginary")
        for bad in (-0.1, 3.5, float("nan"), "x"):
            with self.assertRaises(ValueError):
                film.validate_film_grain(bad)


class FilmPlanTests(unittest.TestCase):
    def test_none_is_identity(self):
        p = _plan()
        self.assertIs(apply_film_plan(p, None), p)
        self.assertIs(apply_film_plan(p, FilmSelection("none")), p)
        rgb = np.random.default_rng(0).uniform(0, 2, (500, 3)).astype(np.float32)
        np.testing.assert_array_equal(apply_tone_core(rgb, p), apply_agx_core(rgb, p))

    def test_preset_pins_curve_and_neutralizes_scene_adaptive_looks(self):
        p = _film_plan("portra400", 1.0)
        curve = film._asset()["stocks"]["portra400"]["curve"]
        self.assertEqual(p.black_ev, curve["black_ev"])
        self.assertEqual(p.contrast, curve["contrast"])
        self.assertEqual((p.punch_strength, p.view_brightness, p.agx_primaries), (0.0, 1.0, "base"))
        self.assertEqual((p.film_stock, p.film_grain, p.target_black_linear), ("portra400", 1.0, 0.0))

    def test_non_agx_core_is_refused(self):
        with self.assertRaises(ValueError):
            apply_film_plan(_plan(tone_core="lum"), FilmSelection("portra400"))


class FilmToneCoreTests(unittest.TestCase):
    def _grey(self, plan, y):
        rgb = np.full((1, 3), y, dtype=np.float32)
        return float(apply_tone_core(rgb, plan)[0] @ REC2020_LUMA)

    def test_negative_mid_grey_is_018_and_reversals_meter_darker(self):
        self.assertAlmostEqual(self._grey(_film_plan("portra400"), 0.18), 0.18, delta=0.004)
        for stock in ("velvia100", "provia100f", "ektachrome100", "kodachrome64"):
            y = self._grey(_film_plan(stock), 0.18)
            self.assertLess(y, 0.16, stock)
            self.assertGreater(y, 0.10, stock)

    def test_black_point_compensated(self):
        for stock in STOCKS:
            self.assertLess(self._grey(_film_plan(stock), 0.18 * 2.0 ** -9), 2e-4, stock)

    def test_crossover_preserves_luminance_and_tints_the_toe(self):
        eng = film._engine("superia400")
        out = np.full((3, 3), 0.05, dtype=np.float32)
        scene = (0.18 * 2.0 ** np.array([-3.0, 0.0, 2.0]))[:, None].repeat(3, 1).astype(np.float32)
        tinted = eng.crossover.apply(out, scene)
        np.testing.assert_allclose(tinted @ REC2020_LUMA, out @ REC2020_LUMA, rtol=1e-5)
        self.assertLess(np.ptp(tinted[1]), 1e-5)            # mid grey stays neutral
        self.assertGreater(np.ptp(tinted[0]), 1e-3)         # Fuji toe is not

    def test_hue_rendition_leaves_neutrals_neutral(self):
        eng = film._engine("velvia100")
        g = np.repeat(np.array([[0.02], [0.18], [0.5], [0.9]], dtype=np.float32), 3, axis=1)
        out = eng.hue.apply(g)
        # Oklab neutrals carry ~1e-4 residual chroma, so the operator moves them
        # by < 0.1 8-bit code; no visible tint, luminance kept.
        self.assertLess(float((np.ptp(out, axis=1) / g[:, 0]).max()), 1e-3)
        np.testing.assert_allclose(out @ REC2020_LUMA, g @ REC2020_LUMA, rtol=1.5e-3)

    def test_probe_path_renders_grain_free(self):
        p = _film_plan("portra800", 1.0)
        rgb = np.random.default_rng(1).uniform(0.01, 1.0, (600, 3)).astype(np.float32)
        no_grain = film_tone_core(rgb, dataclasses.replace(p, film_grain=0.0), apply_agx_core,
                                  FilmSpatial(0, 20, 30))
        np.testing.assert_array_equal(apply_tone_core(rgb, p), no_grain)


class FilmGrainTests(unittest.TestCase):
    def setUp(self):
        self.g = GrainProfile.from_asset(film._asset()["grain"])
        self.eng = film._engine("portra400")

    def test_band_split_equals_whole_frame(self):
        h, w = 60, 90
        rgb = np.random.default_rng(2).uniform(0.02, 1.5, (h * w, 3)).astype(np.float32)
        whole = apply_observe_grain(rgb, 0, h, w, self.eng.grain, self.g, 1.0, 11)
        cut = 1234                                             # not on a row boundary
        parts = np.vstack([
            apply_observe_grain(rgb[:cut], 0, h, w, self.eng.grain, self.g, 1.0, 11),
            apply_observe_grain(rgb[cut:], cut, h, w, self.eng.grain, self.g, 1.0, 11),
        ])
        np.testing.assert_allclose(parts, whole, rtol=1e-6, atol=1e-9)

    def test_grain_is_mean_preserving_on_a_flat_field(self):
        h, w = 200, 300
        rgb = np.full((h * w, 3), 0.18, dtype=np.float32)
        out = apply_observe_grain(rgb, 0, h, w, self.eng.grain, self.g, 1.0, 5)
        self.assertGreater(float(np.log2(out / rgb).std()), 0.02)
        self.assertLess(abs(float(np.log2(out.mean(0) / 0.18).mean())), 0.02)

    def test_seed_moves_arrangement_not_statistics(self):
        h, w = 120, 180
        rgb = np.full((h * w, 3), 0.18, dtype=np.float32)
        a = apply_observe_grain(rgb, 0, h, w, self.eng.grain, self.g, 1.0, 3)
        b = apply_observe_grain(rgb, 0, h, w, self.eng.grain, self.g, 1.0, 4)
        self.assertLess(abs(np.corrcoef(a[:, 1], b[:, 1])[0, 1]), 0.2)
        self.assertAlmostEqual(float(np.log2(a / rgb).std()), float(np.log2(b / rgb).std()), delta=0.006)

    def test_portrait_geometry_turns_the_gate(self):
        geo = FilmGeometry.fit(300, 200)
        self.assertTrue(geo.rotated)
        ii, _ = master(self.g)
        self.assertEqual(sample_field(ii, geo).shape, (300, 200, 3))

    def test_seed_from_file_is_stable_and_path_independent(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a.dng", Path(td) / "b.dng"
            data = bytes(range(256)) * 9000
            a.write_bytes(data)
            b.write_bytes(data)
            self.assertEqual(film.film_seed_for_file(a), film.film_seed_for_file(b))
            b.write_bytes(data + b"x")
            self.assertNotEqual(film.film_seed_for_file(a), film.film_seed_for_file(b))


class FilmContractTests(unittest.TestCase):
    def _cli_error(self, argv):
        from dngscan.cli import parse_args

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(argv)

    def test_cli_refuses_incompatible_combinations(self):
        self._cli_error(["x.dng", "--film", "nope"])
        self._cli_error(["x.dng", "--film", "portra400", "--tone-core", "lum"])
        self._cli_error(["x.dng", "--film", "portra400", "--agx-primaries", "punchy"])
        self._cli_error(["x.dng", "--film", "portra400", "--film-grain", "5"])

    def test_cli_defaults(self):
        from dngscan.cli import parse_args

        args = parse_args(["x.dng", "--film", "velvia100"])
        self.assertEqual((args.film, args.film_grain), ("velvia100", film.DEFAULT_FILM_GRAIN))
        self.assertEqual(parse_args(["x.dng"]).film, "none")

    def test_gui_parse_film(self):
        import tempfile

        from dngscan.gui.service import parse_film

        with tempfile.NamedTemporaryFile(suffix=".dng") as fh:
            fh.write(b"raw" * 1000)
            fh.flush()
            self.assertIsNone(parse_film({}, "jpeg", fh.name, "agx", "none", "base"))
            sel = parse_film({"film": "ektar100", "filmGrain": 0.5}, "jpeg", fh.name, "agx", "none", "base")
            self.assertEqual((sel.stock, sel.grain), ("ektar100", 0.5))
            with self.assertRaises(ValueError):
                parse_film({"film": "ektar100"}, "jpeg", fh.name, "lum", "none", "base")
            with self.assertRaises(ValueError):
                parse_film({"film": "ektar100"}, "jpeg", fh.name, "agx", "arri_skin_d55", "base")


if __name__ == "__main__":
    unittest.main()
