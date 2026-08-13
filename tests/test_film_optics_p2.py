# SPDX-License-Identifier: GPL-3.0-or-later
"""FILM_OPTICS_V2 phase P2 gates: halation stops double-counting the DC,
stops guessing colour from luminance, and comes back to a physical size.

The three defects P0 measured, each with the gate that now holds it shut:

    +0.95% frame-wide red energy   -> gate 11, DC neutrality
    a blue source returning the
    same red halo as a white one   -> gate 16, per-layer trigger
    0.66-0.72 mm half-energy
    radius against a declared 0.55 -> the radius assertions below

Gate 13 (the transfer budget) has nothing to check in P2: no explicit
emulsion scatter ships, so `MTF_explicit` is identity and the residual is the
measured MTF unchanged. The test that would enforce it arrives with the
operator, in P5; asserting it now would be theatre.
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import numpy as np

from dngscan import film_optics_assets as fa
from dngscan import film_optics_charts as charts
from dngscan import film_optics_diag as diag
from dngscan.film_develop import apply_film_core

GATE_W_MM = 36.0
H, W = 640, 960
SCALE = GATE_W_MM / W


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


def _develop(scene: np.ndarray, **kw) -> np.ndarray:
    h, w = scene.shape[:2]
    out = apply_film_core(
        np.asarray(scene, dtype=np.float32).reshape(-1, 3),
        _plan(**kw), spatial_shape=(h, w),
    )
    return np.asarray(out, dtype=np.float64).reshape(h, w, 3)


def _halo(color: str = "white", background_ev: float = -4.0, amount: float = 0.4):
    scene, (cy, cx) = charts.single_emitter(
        H, W, diameter_mm=0.04, exposure_ev=7.0,
        background_ev=background_ev, color=color,
    )
    base = _develop(scene)
    got = _develop(scene, film_halation=amount)
    delta = diag.isolate(got, base)
    radii, prof, _ = diag.radial_profile(delta, (cy, cx), max_radius_px=H / 2)
    return base, delta, radii, np.clip(prof, 0.0, None)


def _ring(radii, prof, lo_mm, hi_mm) -> np.ndarray:
    mm = radii * SCALE
    m = (mm > lo_mm) & (mm < hi_mm)
    e = (prof[m] * radii[m, None]).sum(axis=0)
    return e / max(float(np.max(np.abs(e))), 1e-30)


class DcNeutralityTests(unittest.TestCase):
    """Gate 11. The characteristic curve was measured on large uniform
    patches, where the light scattered out of a patch is replaced by light
    scattered in from its neighbours. That DC gain is already inside the
    curve; reinjecting it again is a warm cast, not a halo."""

    def test_a_uniform_field_is_an_identity(self) -> None:
        for ev in (-2.0, 0.0, 2.0, 5.0):
            with self.subTest(ev=ev):
                flat = charts.uniform_patch(96, 128, ev)
                base = _develop(flat)
                lit = _develop(flat, film_halation=1.0)
                self.assertLess(
                    float(np.max(np.abs(lit - base))), 1e-6,
                    "residual reinjection must vanish on a flat field",
                )

    def test_frame_wide_energy_stays_inside_the_gate(self) -> None:
        base, delta, _, _ = _halo(amount=0.4)
        ratio = np.abs(diag.energy_ratio(delta, base))
        self.assertLess(
            float(ratio.max()), 1e-3,
            f"gate 11: frame energy moved by {ratio} (P0 baseline was 9.5e-3)",
        )

    def test_additive_mode_still_renders_what_it_declares(self) -> None:
        """An asset that declares the old DC handling must get the old maths,
        not silently inherit the new. Otherwise `dc_mode` is decoration."""
        import dataclasses

        stock = fa.load_stock_optics(fa.DEFAULT_STOCK_OPTICS)
        additive = dataclasses.replace(
            stock, halation=dataclasses.replace(stock.halation, dc_mode="additive")
        )
        # +4 EV: comfortably past the R layer's 1.0-2.2 EV gate, so the
        # comparison is about DC handling and not about the gate.
        flat = charts.uniform_patch(64, 96, 4.0)
        base = _develop(flat)
        try:
            fa._CACHE[f"stock:{fa.DEFAULT_STOCK_OPTICS}"] = additive
            lit = _develop(flat, film_halation=1.0)
        finally:
            fa._CACHE.clear()
        self.assertGreater(
            float(np.max(np.abs(lit - base))), 1e-4,
            "additive mode must still add on a flat field",
        )


class PerLayerTriggerTests(unittest.TestCase):
    """Gate 16. A saturated blue source has a low photometric Y and an
    enormous blue-layer exposure; a luminance gate cannot see it."""

    def test_a_blue_source_does_not_return_a_white_source_halo(self) -> None:
        _, _, r_w, p_w = _halo("white")
        _, _, r_b, p_b = _halo("blue")
        white = _ring(r_w, p_w, 0.03, 0.10)
        blue = _ring(r_b, p_b, 0.03, 0.10)
        self.assertGreater(
            abs(blue[1] / blue[0] - white[1] / white[0]), 0.1,
            f"blue {blue} and white {white} return the same halo colour",
        )

    def test_a_blue_source_produces_halation_at_all(self) -> None:
        _, delta, _, _ = _halo("blue")
        self.assertGreater(
            float(np.abs(delta).max()), 1e-4,
            "a saturated blue source must reach the emulsion's blue layer",
        )

    def test_the_gate_is_c1_not_a_step(self) -> None:
        """A hard threshold draws a visible contour at the onset. Sweeping a
        source through the gate must change the returned energy smoothly."""
        energies = []
        for ev in np.linspace(0.5, 4.0, 15):
            scene, _ = charts.single_emitter(
                160, 240, diameter_mm=0.4, exposure_ev=float(ev),
                background_ev=-4.0, color="white",
            )
            base = _develop(scene)
            got = _develop(scene, film_halation=0.6)
            energies.append(float(np.sum(np.abs(got - base))))
        second = np.abs(np.diff(energies, n=2))
        span = max(float(np.max(energies)), 1e-12)
        self.assertLess(
            float(second.max()) / span, 0.35,
            f"second difference {second / span} suggests a hard threshold",
        )


class RadiusAndColourTests(unittest.TestCase):
    def test_the_halo_is_physically_sized(self) -> None:
        """P0 measured 0.66-0.72 mm against a declared 0.55. Published
        backing-reflection geometry puts the first bounce near 65 um and the
        wider ones out to ~110 um."""
        _, _, radii, prof = _halo("white")
        for c in range(3):
            r50 = diag.half_energy_radius(radii, prof[:, c], baseline=0.0) * SCALE
            with self.subTest(channel=c):
                self.assertGreater(r50, 0.02, "halo collapsed to nothing")
                self.assertLess(r50, 0.20, f"halo half-energy radius {r50:.3f} mm")

    def test_the_inner_ring_is_warmer_than_the_outer_one(self) -> None:
        """The exit-gate look criterion, and the reason components exist: one
        kernel with a fixed weight vector returns the same hue at every
        radius, so these two numbers would be equal."""
        _, _, radii, prof = _halo("white")
        inner = _ring(radii, prof, 0.03, 0.10)
        outer = _ring(radii, prof, 0.30, 1.00)
        self.assertGreater(
            inner[1] / inner[0], outer[1] / outer[0] + 0.1,
            f"inner {inner} is not warmer than outer {outer}",
        )
        self.assertAlmostEqual(outer[0], 1.0, places=6, msg="outer ring must be red-led")

    def test_daylight_skin_does_not_pick_up_a_global_red_cast(self) -> None:
        """The other half of the exit gate. A normal daylight frame has no
        source above the trigger, so halation must be close to inert."""
        scene = np.zeros((128, 192, 3), dtype=np.float32)
        skin = np.array([0.36, 0.26, 0.20], dtype=np.float32)
        scene[:] = skin
        scene[:, :64] = skin * 0.5
        scene[:, 128:] = skin * 1.8
        base = _develop(scene)
        got = _develop(scene, film_halation=0.6)
        shift = diag.energy_ratio(diag.isolate(got, base), base)
        self.assertLess(float(np.abs(shift).max()), 5e-4, f"skin drifted by {shift}")


class ProfileAssetTests(unittest.TestCase):
    def test_both_declared_profiles_load(self) -> None:
        strong = fa.load_stock_optics("35mm_strong_ah_modelled")
        remjet = fa.load_stock_optics("35mm_no_remjet_editorial")
        self.assertEqual(strong.halation.anti_halation_class, "strong")
        self.assertEqual(remjet.halation.anti_halation_class, "none")
        self.assertEqual(strong.provenance, "modelled")
        self.assertEqual(remjet.provenance, "editorial")

    def test_only_the_no_remjet_profile_has_an_aura(self) -> None:
        """A modern stock with an intact anti-halation layer does not produce
        a large-radius return. The component is present but zero so the two
        assets stay directly comparable."""
        strong = fa.load_stock_optics("35mm_strong_ah_modelled").halation
        remjet = fa.load_stock_optics("35mm_no_remjet_editorial").halation
        s_aura = next(c for c in strong.components if c.name == "aura")
        r_aura = next(c for c in remjet.components if c.name == "aura")
        self.assertEqual(float(np.abs(s_aura.transfer).max()), 0.0)
        self.assertGreater(float(r_aura.transfer.max()), 0.0)
        self.assertGreater(
            float(remjet.total_return().max()),
            float(strong.total_return().max()) * 1.5,
        )

    def test_a_zero_width_gate_is_refused(self) -> None:
        raw = {
            "name": "probe", "radius_mm": 0.1,
            "gate_ev": [[1.0, 1.0], [1.0, 1.5], [1.0, 1.5]],
            "transfer": [[0.01, 0, 0], [0, 0, 0], [0, 0, 0]],
        }
        with self.assertRaises(fa.OpticsAssetError) as ctx:
            fa.HalationComponent.from_json(raw, "probe")
        self.assertIn("smootherstep width", str(ctx.exception))

    def test_a_negative_transfer_is_refused(self) -> None:
        raw = {
            "name": "probe", "radius_mm": 0.1,
            "gate_ev": [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
            "transfer": [[0.01, -0.01, 0], [0, 0, 0], [0, 0, 0]],
        }
        with self.assertRaises(fa.OpticsAssetError):
            fa.HalationComponent.from_json(raw, "probe")


class NonNegativityTests(unittest.TestCase):
    def test_the_core_can_give_away_its_light_but_not_more(self) -> None:
        """The residual takes energy out of a highlight core. Capping the
        subtraction at the pixel's own layer exposure makes non-negativity
        structural instead of a clamp applied after the fact."""
        scene, _ = charts.single_emitter(
            160, 240, diameter_mm=0.6, exposure_ev=8.0,
            background_ev=-6.0, color="white",
        )
        for amount in (0.5, 1.0):
            with self.subTest(amount=amount):
                got = _develop(scene, film_halation=amount)
                self.assertTrue(bool(np.isfinite(got).all()))
                self.assertGreaterEqual(float(got.min()), 0.0)


class FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._fast = os.environ.get("DNGSCAN_FAST")
        os.environ["DNGSCAN_FAST"] = "0"

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._fast is None:
            os.environ.pop("DNGSCAN_FAST", None)
        else:
            os.environ["DNGSCAN_FAST"] = cls._fast

    def test_the_regenerated_freeze_still_reproduces(self) -> None:
        """P2 deliberately moves the frozen bytes — that is the point of the
        phase. What must hold is that the NEW behaviour is reproducible."""
        from tools.regen_optics_freeze import iter_cases, render_case

        for case in iter_cases():
            with self.subTest(case=case.stem):
                stored = np.load(case.path, allow_pickle=False)
                linear, u8 = render_case(case)
                np.testing.assert_array_equal(stored["u8"], u8)


if __name__ == "__main__":
    unittest.main()


class ExposureDirectionTests(unittest.TestCase):
    """Review 2026-08-10 F1. The first gate reference multiplied the absolute
    grey layer exposure back into an already-normalized coordinate and scaled
    it by 10^ev instead of 2^ev: each added stop of film exposure pushed the
    halation trigger 2.32 EV the WRONG way — measured 400x LESS injected
    energy at +1 EV. The P2 gate only asserted that exposure CHANGED the
    energy, which a backwards response satisfies; this one pins the sign.
    """

    def test_more_film_exposure_means_more_halation(self) -> None:
        scene, _ = charts.single_emitter(
            320, 480, diameter_mm=0.3, exposure_ev=5.0, background_ev=-4.0
        )
        energies = []
        # Media scatter off on BOTH renders (review R1 item 4): the base
        # render has no engaged optics and therefore no spatial context,
        # while the halation render used to drag both scatter stages in
        # with it — the isolated delta then measured scatter + halation
        # and the exposure ordering inverted.
        for ev in (-1.0, 0.0, 1.0):
            base = _develop(
                scene, film_exposure_ev=ev, film_media_scatter="off"
            )
            lit = _develop(
                scene, film_exposure_ev=ev, film_halation=0.6,
                film_media_scatter="off",
            )
            energies.append(float(np.sum(np.abs(lit - base))))
        self.assertLess(energies[0], energies[1])
        self.assertLess(energies[1], energies[2])
        self.assertGreater(
            energies[2] / max(energies[1], 1e-12), 1.5,
            "a stop of overexposure must meaningfully widen the trigger",
        )

    def test_the_gate_reference_is_unity(self) -> None:
        """layer_log_exposure is neutral-anchored, so the reference the gates
        measure EV against must be exactly 1 — anything else re-introduces
        the absolute-exposure double count."""
        from dngscan.film_develop import prepare_film_spatial

        ctx = prepare_film_spatial(_plan(film_halation=0.5), 64, 96)
        flat = charts.uniform_patch(64, 96, 0.0)
        ctx.finish_maps(
            np.asarray(flat, dtype=np.float32), _plan(film_halation=0.5),
            "portra400",
        )
        np.testing.assert_array_equal(ctx.hal_ref, np.ones(3, dtype=np.float32))
