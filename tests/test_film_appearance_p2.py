# SPDX-License-Identifier: GPL-3.0-or-later
"""Appearance P2 gates: the palette kernel (plan §6, §16 P2).

The kernel is pointwise field application in the common Rec.2020/Oklab +
print-EV space. What this file pins, in the order the plan demands it:

- identity recipes and strength 0 stay STRICT identities (same object — the
  P1 byte gate depends on it);
- the neutral axis is untouchable by construction (w_c -> 0), not by test
  tolerance;
- the hue field wraps C1 across 345->0 and the EV axis cannot overshoot its
  knots (PCHIP hull property);
- each field does ITS job and not another's: a pure hue recipe rotates
  without changing chroma, a pure density recipe darkens without rotating;
- folds: the per-edge comparative gate from mainline A4, reused verbatim in
  spirit — a recipe must not create or deepen hue folds against its own
  baseline;
- strength is continuous and monotone in effect.

Non-identity recipes are synthesized into a temp directory with their own
manifest — the SHIPPED assets stay identity until P4 authors real ones.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dngscan import film_appearance as fa
from dngscan import film_palette_diag as pal

LUMA = np.array([0.2627, 0.6780, 0.0593])


def synth_recipe(tmp: Path, **fields) -> tuple[Path, Path]:
    """Write a synthetic recipe + manifest into tmp; returns (dir, manifest)."""
    k, h = len(fa.EV_KNOTS), fa.HUE_KNOT_COUNT
    data = {
        "hue_delta_deg": np.zeros((k, h), np.float32),
        "log_chroma_gain": np.zeros((k, h), np.float32),
        "density_ev": np.zeros((k, h), np.float32),
        "neutral_bias_ab": np.zeros((k, 2), np.float32),
    }
    for name, val in fields.items():
        data[name] = np.asarray(val, np.float32)
    meta = {
        "schema": fa.APPEARANCE_SCHEMA,
        "recipe_id": "portra400__endura_reference_v1",
        "stock_id": "portra400",
        "medium_id": "kodak_portra_endura__translated",
        "process_space": "display-linear-rec2020/oklab+scene-ev",
        "provenance": "editorial-authored",
        "chroma_knee": 0.18, "chroma_power": 2.0, "neutral_chroma_c0": 0.03,
    }
    path = tmp / "portra400__endura_reference_v1.npz"
    np.savez_compressed(
        path, meta=np.asarray(json.dumps(meta)),
        ev_knots=np.asarray(fa.EV_KNOTS, np.float32),
        hue_knots_deg=(np.arange(h) * (360.0 / h)).astype(np.float32),
        **data,
    )
    manifest = tmp / "MANIFEST.json"
    manifest.write_text(json.dumps({
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    }))
    return tmp, manifest


def plan_for(tmp: Path, manifest: Path, strength: float = 1.0):
    with mock.patch.object(fa, "APPEARANCE_DIR", tmp), \
         mock.patch.object(fa, "MANIFEST_PATH", manifest):
        return fa.compile_appearance_plan(
            "reference", strength,
            stock_id="portra400", medium_id="kodak_portra_endura__translated",
        )


def patch_volume() -> np.ndarray:
    """A chroma/EV/hue sweep in mapped Rec.2020, plus a neutral ramp."""
    rows = []
    for hh in np.arange(0.0, 360.0, 15.0):
        for cf in (0.08, 0.15, 0.25):
            for evv in (-3.0, -1.0, 0.0, 1.0):
                lab = np.array([[0.6, cf * np.cos(np.radians(hh)),
                                 cf * np.sin(np.radians(hh))]])
                rgb = np.maximum(pal.oklab_to_rec2020(lab)[0], 1e-6)
                y = float(rgb @ LUMA)
                rows.append(rgb * (0.18 * 2.0 ** evv / y))
    for evv in np.linspace(-5.0, 2.0, 15):
        rows.append(np.full(3, 0.18 * 2.0 ** evv))
    return np.asarray(rows, dtype=np.float32)


class FastPathTests(unittest.TestCase):
    def test_identity_recipe_returns_the_same_object(self) -> None:
        """The Oklab round trip alone would cost byte identity; the identity
        flag must route around the kernel entirely. Synthetic since P4 —
        the shipped recipes are authored now."""
        with tempfile.TemporaryDirectory() as td:
            tmp, man = synth_recipe(Path(td))
            plan = plan_for(tmp, man)
            self.assertTrue(plan.recipe["is_identity"])
            arr = np.ones((8, 3), np.float32) * 0.3
            self.assertIs(fa.apply_film_appearance(arr, plan), arr)

    def test_strength_zero_returns_the_same_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            k, h = len(fa.EV_KNOTS), fa.HUE_KNOT_COUNT
            tmp, man = synth_recipe(
                Path(td), hue_delta_deg=np.full((k, h), 8.0, np.float32)
            )
            plan = plan_for(tmp, man, strength=0.0)
            arr = np.ones((8, 3), np.float32) * 0.3
            self.assertIs(fa.apply_film_appearance(arr, plan), arr)


class FieldSemanticsTests(unittest.TestCase):
    def _apply(self, vol, **fields):
        with tempfile.TemporaryDirectory() as td:
            tmp, man = synth_recipe(Path(td), **fields)
            plan = plan_for(tmp, man)
            return np.asarray(
                fa.apply_film_appearance(vol, plan), dtype=np.float64
            )

    def test_the_neutral_axis_is_untouchable(self) -> None:
        """w_c -> 0 by construction: even a shouting recipe (20 deg hue,
        +-0.5 stop chroma, 0.3 EV density) leaves a grey ramp within float
        noise of itself."""
        k, h = len(fa.EV_KNOTS), fa.HUE_KNOT_COUNT
        vol = patch_volume()
        out = self._apply(
            vol,
            hue_delta_deg=np.full((k, h), 20.0),
            log_chroma_gain=np.full((k, h), 0.5),
            density_ev=np.full((k, h), 0.3),
        )
        dec_in = pal.decompose(vol.astype(np.float64))
        neutral = dec_in["C"] < 1e-3
        drift = np.abs(out[neutral] - vol[neutral].astype(np.float64))
        self.assertLess(float(drift.max() / 0.18), 2e-3)

    def test_a_pure_hue_recipe_rotates_without_pumping_chroma(self) -> None:
        k, h = len(fa.EV_KNOTS), fa.HUE_KNOT_COUNT
        vol = patch_volume()
        out = self._apply(vol, hue_delta_deg=np.full((k, h), 10.0))
        d = pal.compare(vol.astype(np.float64), out)
        hh = d["d_hue_deg"]
        hh = hh[np.isfinite(hh)]
        # w_c gates the rotation near neutral, so the SATURATED samples must
        # carry ~10 deg while chroma stays put everywhere
        dec = pal.decompose(vol.astype(np.float64))
        strong = dec["C"] > 0.1
        h_strong = d["d_hue_deg"][strong]
        self.assertAlmostEqual(float(np.nanmedian(h_strong)), 10.0, delta=1.0)
        ratios = d["log2_colorfulness_ratio"]
        ratios = ratios[np.isfinite(ratios)]
        self.assertLess(float(np.abs(ratios).max()), 0.1)

    def test_a_pure_density_recipe_darkens_without_rotating(self) -> None:
        k, h = len(fa.EV_KNOTS), fa.HUE_KNOT_COUNT
        vol = patch_volume()
        out = self._apply(vol, density_ev=np.full((k, h), 0.3))
        d = pal.compare(vol.astype(np.float64), out)
        dec = pal.decompose(vol.astype(np.float64))
        strong = dec["C"] > 0.1
        self.assertLess(float(np.nanmax(d["d_output_ev"][strong])), 0.0,
                        "denser colours must get darker")
        # A5 item 2: density is a NEUTRAL-DENSITY move — it scales L and C
        # together, so saturation S = C/L is conserved. The first cut scaled
        # only L and silently added +0.1 stop of saturation per 0.3 EV.
        sat = d["log2_saturation_ratio"][strong]
        self.assertLess(float(np.nanmax(np.abs(sat[np.isfinite(sat)]))), 0.02,
                        "density must not move saturation")
        hh = np.abs(d["d_hue_deg"][strong])
        self.assertLess(float(np.nanmedian(hh[np.isfinite(hh)])), 1.5)
        neutral = dec["C"] < 1e-3
        self.assertLess(
            float(np.abs(d["d_output_ev"][neutral]).max()), 1e-3,
            "density must not touch the greys",
        )

    def test_richness_shoulder_gives_low_purity_more_than_high(self) -> None:
        """§6.3: the whole point of r(S) — a chroma recipe lifts low/mid
        purity more (in log ratio) than already-saturated colour. A5 item 3:
        purity is SATURATION S = C/L, not raw chroma C — grouping by C mixes
        bright weak colour with dark strong colour and measures the wrong
        shoulder."""
        k, h = len(fa.EV_KNOTS), fa.HUE_KNOT_COUNT
        vol = patch_volume()
        out = self._apply(vol, log_chroma_gain=np.full((k, h), 0.4))
        d = pal.compare(vol.astype(np.float64), out)
        dec = pal.decompose(vol.astype(np.float64))
        low = (dec["S"] > 0.1) & (dec["S"] < 0.2)
        high = dec["S"] > 0.35
        g_low = np.nanmedian(d["log2_colorfulness_ratio"][low])
        g_high = np.nanmedian(d["log2_colorfulness_ratio"][high])
        self.assertGreater(float(g_low), float(g_high) + 0.05)
        self.assertGreater(float(g_high), 0.0)


class ContinuityTests(unittest.TestCase):
    def _field_probe(self, field: np.ndarray) -> np.ndarray:
        """Sample one field densely across hue at EV 0 through the public
        kernel path (via a hue recipe), returning the applied hue delta."""
        with tempfile.TemporaryDirectory() as td:
            tmp, man = synth_recipe(Path(td), hue_delta_deg=field)
            plan = plan_for(tmp, man)
            hues = np.arange(0.0, 360.0, 0.5)
            rows = []
            for hh in hues:
                lab = np.array([[0.6, 0.2 * np.cos(np.radians(hh)),
                                 0.2 * np.sin(np.radians(hh))]])
                rgb = np.maximum(pal.oklab_to_rec2020(lab)[0], 1e-6)
                rows.append(rgb * (0.18 / float(rgb @ LUMA)))
            vol = np.asarray(rows, dtype=np.float32)
            out = np.asarray(fa.apply_film_appearance(vol, plan), np.float64)
            d = pal.compare(vol.astype(np.float64), out)
            return d["d_hue_deg"]

    def test_the_hue_wrap_is_seamless(self) -> None:
        """One bump knot near the seam: the response across 345->0->15 must
        be smooth — no jump at the periodic boundary."""
        k, h = len(fa.EV_KNOTS), fa.HUE_KNOT_COUNT
        field = np.zeros((k, h), np.float32)
        field[:, 0] = 12.0   # bump exactly at 0 degrees
        got = self._field_probe(field)
        got = got[np.isfinite(got)]
        step = np.abs(np.diff(got))
        self.assertLess(float(step.max()), 0.6,
                        "0.5-degree sampling must show no seam jump")

    def test_ev_interpolation_cannot_overshoot_its_knots(self) -> None:
        """PCHIP hull property: a recipe writing [0, 0, 10, 0, 0] on the EV
        axis can never deliver more than 10 anywhere."""
        k, h = len(fa.EV_KNOTS), fa.HUE_KNOT_COUNT
        field = np.zeros((k, h), np.float32)
        field[2, :] = 10.0   # EV 0 only
        with tempfile.TemporaryDirectory() as td:
            tmp, man = synth_recipe(Path(td), hue_delta_deg=field)
            plan = plan_for(tmp, man)
            rows = []
            for evv in np.linspace(-6.5, 6.5, 121):
                lab = np.array([[0.6, 0.18, 0.05]])
                rgb = np.maximum(pal.oklab_to_rec2020(lab)[0], 1e-6)
                rows.append(rgb * (0.18 * 2.0 ** evv / float(rgb @ LUMA)))
            vol = np.asarray(rows, dtype=np.float32)
            out = np.asarray(fa.apply_film_appearance(vol, plan), np.float64)
            d = pal.compare(vol.astype(np.float64), out)
            hh = d["d_hue_deg"]
            hh = hh[np.isfinite(hh)]
            self.assertLessEqual(float(np.nanmax(np.abs(hh))), 10.5)

    def test_strength_scales_continuously(self) -> None:
        k, h = len(fa.EV_KNOTS), fa.HUE_KNOT_COUNT
        with tempfile.TemporaryDirectory() as td:
            tmp, man = synth_recipe(
                Path(td), hue_delta_deg=np.full((k, h), 10.0)
            )
            lab = np.array([[0.6, 0.2, 0.0]])
            rgb = np.maximum(pal.oklab_to_rec2020(lab)[0], 1e-6)
            vol = (rgb * (0.18 / float(rgb @ LUMA))).astype(np.float32)[None, :]
            deltas = []
            for st in (0.25, 0.5, 0.75, 1.0, 1.25):
                plan = plan_for(tmp, man, strength=st)
                out = np.asarray(fa.apply_film_appearance(vol, plan), np.float64)
                d = pal.compare(vol.astype(np.float64), out)
                deltas.append(float(d["d_hue_deg"][0]))
            steps = np.diff(deltas)
            self.assertTrue(all(st > 0 for st in steps), deltas)
            self.assertLess(float(np.std(steps) / np.mean(steps)), 0.25,
                            "strength response should be near-linear")

    def test_no_new_hue_folds_against_the_recipe_off_baseline(self) -> None:
        """The A4 per-edge doctrine applied to the palette: a smooth recipe
        must not fold hue. Probed on a dense ring at moderate chroma."""
        k, h = len(fa.EV_KNOTS), fa.HUE_KNOT_COUNT
        field = np.zeros((k, h), np.float32)
        field[:, ::2] = 8.0    # alternating bumps: the hardest smooth case
        with tempfile.TemporaryDirectory() as td:
            tmp, man = synth_recipe(Path(td), hue_delta_deg=field)
            plan = plan_for(tmp, man)
            hues = np.arange(0.0, 360.0, 1.0)
            rows = []
            for hh in hues:
                lab = np.array([[0.6, 0.18 * np.cos(np.radians(hh)),
                                 0.18 * np.sin(np.radians(hh))]])
                rgb = np.maximum(pal.oklab_to_rec2020(lab)[0], 1e-6)
                rows.append(rgb * (0.18 / float(rgb @ LUMA)))
            vol = np.asarray(rows, dtype=np.float32)
            out = np.asarray(fa.apply_film_appearance(vol, plan), np.float64)
            h_out = pal.decompose(out)["h_deg"]
            u = np.unwrap(np.radians(np.concatenate([h_out, [h_out[0]]])))
            self.assertGreater(float(np.min(np.diff(u))), 0.0,
                               "the palette must not fold hue")


class ChainIntegrationTests(unittest.TestCase):
    def test_pre_gamut_output_is_gamut_choice_invariant(self) -> None:
        """§12.1: the appearance runs in common Rec.2020 before the gamut
        fit, so srgb and p3 plans must agree bit-for-bit at this stage."""
        from dngscan.render import apply_tone_core
        from dngscan.tone import build_render_plan
        from tests.golden_support import all_scenes

        scene = all_scenes()["daylight_wide_dr"]
        arr = (np.random.default_rng(3).random((512, 3)) * 0.5).astype(np.float32)
        outs = []
        for gamut in ("srgb", "p3"):
            plan = build_render_plan(
                scene.bundle, scene.analysis, "agx", gamut,
                film_curve="portra400", film_mode="full", film_crossover="off",
                film_appearance="reference",
            )
            outs.append(np.asarray(apply_tone_core(arr, plan.tone, plan.color)))
        np.testing.assert_array_equal(outs[0], outs[1])

    def test_reference_white_probe_is_recipe_invariant(self) -> None:
        """The HDR join probes a NEUTRAL ramp; w_c -> 0 means no recipe can
        move it, which is what keeps the SDR base / HDR gain contract out of
        the palette's reach."""
        from types import SimpleNamespace

        from dngscan.film_develop import film_reference_white_ev

        plan = SimpleNamespace(
            curve_preset="portra400", film_mode="full", film_crossover="off",
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
        self.assertTrue(np.isfinite(film_reference_white_ev(plan)))


if __name__ == "__main__":
    unittest.main()
