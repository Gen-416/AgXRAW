# SPDX-License-Identifier: GPL-3.0-or-later
"""FILM_OPTICS_V2 phase P3 gates: bloom moves to the domain where it can work.

P0's measurement of the old operator was not "the strength is wrong" but "the
domain is": it ran after B2, where the print offered **0.73 EV** of overrange,
on scenes that had offered **6.00**. No setting recovers six stops. The
editorial capture bloom now runs on the scene, before the observer.

Two plan gates land here:

    gate 18  three source sizes must receive three different halo sizes
    gate 17  Save Lights must stay continuous — no hard ring, no hollow halo

`film_bloom` now drives the capture bloom. The old post-B2 conservative
operator is `legacy_print_scatter`, unreachable from any user amount, and the
tests below check that the two cannot share a field.
"""
from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace

import numpy as np

from dngscan import film_optics as fo
from dngscan import film_optics_assets as fa
from dngscan import film_optics_charts as charts
from dngscan import film_optics_diag as diag
from dngscan.film_develop import apply_film_core

H, W = 640, 960
SCALE = 36.0 / W


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


def _glow(diameter_mm: float, amount: float = 0.5):
    scene, (cy, cx) = charts.single_emitter(
        H, W, diameter_mm=diameter_mm, exposure_ev=7.0, background_ev=-4.0
    )
    base = _develop(scene)
    got = _develop(scene, film_bloom=amount)
    delta = diag.isolate(got, base)
    radii, prof, _ = diag.radial_profile(delta, (cy, cx), max_radius_px=H / 2)
    return base, delta, radii, np.clip(prof, 0.0, None)


class DomainTests(unittest.TestCase):
    def test_a_scene_with_no_bright_source_is_untouched(self) -> None:
        for ev in (-2.0, 0.0, 1.0):
            with self.subTest(ev=ev):
                flat = charts.uniform_patch(96, 128, ev)
                self.assertLess(
                    float(np.max(np.abs(
                        _develop(flat, film_bloom=1.0) - _develop(flat)
                    ))), 1e-7,
                )

    def test_the_glow_reaches_the_shadows_the_old_operator_could_not(self) -> None:
        """The point of the move. A +7 EV source on a -4 EV field must lift
        its surroundings measurably — that ratio is what the display domain
        did not have."""
        _, delta, radii, prof = _glow(0.25)
        mm = radii * SCALE
        far = prof[(mm > 0.5) & (mm < 1.5), 1]
        self.assertGreater(float(far.mean()), 1e-4, "no reach into the surround")

    def test_bloom_no_longer_darkens_the_source_core_to_conserve(self) -> None:
        """The old operator was conservative by subtracting from the highlight
        core. An editorial glow adds light; §16 P3's exit gate says so."""
        base, delta, _, _ = _glow(0.6)
        self.assertGreaterEqual(
            float(delta.min()), -1e-3,
            "capture bloom must not take light out of the source",
        )
        self.assertGreater(float(diag.energy_ratio(delta, base).max()), 0.0)


class ScaleSpaceTests(unittest.TestCase):
    """Gate 18. One detector radius cannot tell a filament from a window
    however its threshold is set; each rung carries its own diffusion."""

    def test_three_source_sizes_get_three_halo_sizes(self) -> None:
        radii_mm = []
        for d in (0.04, 0.25, 2.0):
            _, _, radii, prof = _glow(d)
            radii_mm.append(
                diag.half_energy_radius(radii, prof[:, 1], baseline=0.0) * SCALE
            )
        for small, large in zip(radii_mm, radii_mm[1:]):
            self.assertGreater(
                large, small * 2.0,
                f"halo radii {radii_mm} are not separated by source size",
            )

    def test_the_finest_rung_must_detect_at_full_resolution(self) -> None:
        """A point source averaged into a decimated cell falls under the gate
        before anything looks at it — the same non-commutation review batch 18
        measured on the old bloom. The asset refuses a ladder that starts
        coarse."""
        raw = {
            "kind": "capture_bloom", "asset_id": "probe",
            "provenance": "editorial", "active": True,
            "scales": [
                {"detect_um": 30.0, "diffuse_um": 40.0,
                 "gate_ev": [2.0, 3.0], "weight": 1.0},
                {"detect_um": 60.0, "diffuse_um": 200.0,
                 "gate_ev": [2.0, 3.0], "weight": 1.0},
            ],
        }
        with self.assertRaises(fa.OpticsAssetError) as ctx:
            fa.CaptureBloomAsset.from_json(raw, "probe")
        self.assertIn("full resolution", str(ctx.exception))

    def test_diffusion_must_grow_with_the_detected_size(self) -> None:
        raw = {
            "kind": "capture_bloom", "asset_id": "probe",
            "provenance": "editorial", "active": True,
            "scales": [
                {"detect_um": 0.0, "diffuse_um": 400.0,
                 "gate_ev": [2.0, 3.0], "weight": 1.0},
                {"detect_um": 60.0, "diffuse_um": 40.0,
                 "gate_ev": [2.0, 3.0], "weight": 1.0},
            ],
        }
        with self.assertRaises(fa.OpticsAssetError):
            fa.CaptureBloomAsset.from_json(raw, "probe")


class SaveLightsTests(unittest.TestCase):
    """Gate 17. `max(G - k*S, 0)` is only C0 and can leave a hard ring or a
    hollow halo around a large bright area."""

    def _operator(self, save: float) -> np.ndarray:
        asset = dataclasses.replace(
            fa.load_capture_bloom("modelled_default"), save_lights=save
        )
        n, w = 1, 64
        img = np.full((n, w, 3), 0.02, dtype=np.float32)
        img[0, 24:40] = 12.0
        glow = np.zeros((8, 16, 3), dtype=np.float32)
        glow[:, 3:9] = 0.5
        return fo.capture_bloom_apply_rows(
            img.reshape(-1, 3), glow, 0, n, n, w, asset, 1.0
        ).reshape(n, w, 3)

    def test_the_core_suppression_is_monotone_and_continuous(self) -> None:
        outs = [self._operator(s) for s in np.linspace(0.0, 1.0, 11)]
        core = np.array([float(o[0, 32, 1]) for o in outs])
        self.assertTrue(
            bool(np.all(np.diff(core) <= 1e-6)),
            f"core response is not monotone in save_lights: {core}",
        )
        second = np.abs(np.diff(core, n=2))
        self.assertLess(
            float(second.max()), 0.1 * max(float(np.ptp(core)), 1e-9) + 1e-6,
            "save_lights response has a kink",
        )

    def test_the_surround_keeps_its_glow_at_full_save_lights(self) -> None:
        """save_lights=1 protects the SOURCE, not the halo. A version that
        flattens the surround too is a strength control wearing a costume."""
        off = self._operator(0.0)
        full = self._operator(1.0)
        self.assertGreater(float(full[0, 12, 1]), float(off[0, 12, 1]) * 0.9)
        self.assertLess(float(full[0, 32, 1]), float(off[0, 32, 1]))

    def test_the_halo_stays_radially_monotone(self) -> None:
        """A hollow halo shows up as a profile that rises again with radius."""
        for save in (0.0, 0.5, 1.0):
            with self.subTest(save=save):
                asset = dataclasses.replace(
                    fa.load_capture_bloom("modelled_default"), save_lights=save
                )
                try:
                    fa._CACHE["bloom:modelled_default"] = asset
                    _, _, radii, prof = _glow(1.0, amount=0.6)
                finally:
                    fa._CACHE.clear()
                mm = radii * SCALE
                inside = prof[mm < 1.0, 1]
                after = inside[int(np.argmax(inside)):]
                self.assertTrue(
                    bool(np.all(np.diff(after) <= 1e-6)),
                    "halo rises again with radius: hollow ring",
                )


class RenameTests(unittest.TestCase):
    def test_film_bloom_drives_the_capture_bloom_not_the_print_scatter(self) -> None:
        got = fa.compile_film_optics_plan(_plan(film_bloom=0.4))
        self.assertEqual(got.capture_bloom_amount, 0.4)
        self.assertTrue(got.capture_bloom.active)
        self.assertFalse(hasattr(got, "print_scatter_amount"))

    def test_the_legacy_print_scatter_is_deleted(self) -> None:
        """P5e closed the §12.2 ledger: the legacy operator and its asset
        are GONE, not merely unreachable — the acceptance comparison it was
        retained for has served its purpose. A print asset that still
        carries the block is refused, so it can never quietly return."""
        from dngscan import film_optics

        self.assertFalse(hasattr(film_optics, "bloom_apply_rows"))
        self.assertFalse(hasattr(film_optics, "scatter_spread"))
        self.assertFalse(hasattr(fa, "PrintScatterAsset"))
        self.assertFalse(
            hasattr(fa.load_print_optics(fa.DEFAULT_PRINT_OPTICS),
                    "print_scatter"))
        with self.assertRaises(fa.OpticsAssetError):
            fa.PrintOpticsAsset.from_json({
                "kind": "print_optics", "asset_id": "t",
                "provenance": "modelled",
                "legacy_print_scatter": {"threshold": 0.6},
            }, "t")

    def test_the_report_names_the_capture_bloom_and_its_provenance(self) -> None:
        rep = fa.compile_film_optics_plan(_plan(film_bloom=0.4)).report()
        self.assertEqual(rep["capture_bloom"], "modelled_default")
        self.assertEqual(rep["provenance"]["capture_bloom"], "editorial")


class OrderingTests(unittest.TestCase):
    def test_halation_sees_the_bloomed_scene(self) -> None:
        """The two operators must not disagree about the same photograph: a
        source bright enough to bloom has to be able to halate too."""
        scene, _ = charts.single_emitter(
            256, 384, diameter_mm=0.3, exposure_ev=6.0, background_ev=-4.0
        )
        only_hal = _develop(scene, film_halation=0.5)
        both = _develop(scene, film_halation=0.5, film_bloom=0.8)
        bloom_only = _develop(scene, film_bloom=0.8)
        base = _develop(scene)
        # If halation read the raw scene, `both` would be exactly the sum of
        # the two isolated deltas. It is not, because the glow feeds it.
        additive = base + (only_hal - base) + (bloom_only - base)
        self.assertGreater(float(np.max(np.abs(both - additive))), 1e-5)


if __name__ == "__main__":
    unittest.main()
