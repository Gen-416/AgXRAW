# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase-1 SceneScaleContract equivalence and concurrency gates."""
from __future__ import annotations

import threading
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from dngscan.models import RawBundle, SceneScaleContract
from dngscan.raw_io import baseline_exposure_gain, libraw_scene_scale
from dngscan.scene_scale import (
    calibration_confidence_for_mode,
    scene_scale_contract_from_bundle,
    with_intent_exposure,
)
from dngscan.tone import compute_exposure_gain, exposure_mode_for_tone_core, scene_rec2020_to_float


def _tiny_bundle(
    *,
    scene_scale: float = 4.0,
    exposure_gain: float = 1.0,
    decoder: str = "libraw",
    scale_mode: str | None = None,
    baseline: float | None = None,
    baseline_baked: bool = False,
    rgb: np.ndarray | None = None,
) -> RawBundle:
    if rgb is None:
        rgb = np.array([[[100.0, 200.0, 50.0], [10.0, 20.0, 30.0]]], dtype=np.float32)
    return RawBundle(
        path=Path("synthetic.dng"),
        raw_image=np.zeros((2, 2), dtype=np.uint16),
        raw_colors=np.zeros((2, 2), dtype=np.uint8),
        xyz_render=rgb.copy(),
        render_scale=scene_scale,
        scene_rec2020_render=rgb.copy(),
        scene_scale=scene_scale,
        white_level=16383,
        black_levels=[0.0, 0.0, 0.0, 0.0],
        camera_wb=[1.0, 1.0, 1.0, 1.0],
        color_desc="RGBG",
        raw_pattern=[[0, 1], [1, 2]],
        camera_white_levels=[16383.0, 16383.0, 16383.0],
        exposure_gain=exposure_gain,
        baseline_exposure=baseline,
        baseline_exposure_baked_in=baseline_baked,
        scene_decoder=decoder,
        scene_scale_mode=scale_mode,
    )


class SceneScaleContractTest(unittest.TestCase):
    def test_legacy_and_contract_paths_are_pixel_identical(self) -> None:
        bundle = _tiny_bundle(
            scene_scale=3.5,
            exposure_gain=compute_exposure_gain(exposure_mode_for_tone_core("agx"), 0.75),
        )
        stored = bundle.scene_rec2020_render.astype(np.float32)
        legacy = scene_rec2020_to_float(stored, bundle.scene_scale, bundle.exposure_gain)
        contract = scene_scale_contract_from_bundle(bundle, tone_core="agx")
        via_contract = scene_rec2020_to_float(stored, 1.0, contract=contract)
        self.assertTrue(np.array_equal(legacy, via_contract))
        self.assertAlmostEqual(contract.legacy_exposure_gain, bundle.exposure_gain, places=12)
        self.assertAlmostEqual(
            contract.total_render_gain, bundle.exposure_gain, places=12
        )

    def test_user_ev_plus_one_multiplies_pre_drt_rgb_by_two(self) -> None:
        base = _tiny_bundle(scene_scale=2.0, exposure_gain=1.0)
        ev0 = with_intent_exposure(base, user_ev=0.0, tone_core="agx")
        ev1 = with_intent_exposure(base, user_ev=1.0, tone_core="agx")
        stored = base.scene_rec2020_render
        rgb0 = scene_rec2020_to_float(stored, ev0.scene_scale, ev0.exposure_gain)
        rgb1 = scene_rec2020_to_float(stored, ev1.scene_scale, ev1.exposure_gain)
        self.assertTrue(np.allclose(rgb1, rgb0 * 2.0, rtol=0.0, atol=0.0))

    def test_baseline_exposure_applied_once_via_libraw_scene_scale(self) -> None:
        baseline = 1.0
        scale = libraw_scene_scale(
            encoded_max=16383.0,
            highlight_mode_name="clip",
            wb_values=[1.0, 1.0, 1.0],
            baseline_exposure=baseline,
        )
        scale_no_baseline = libraw_scene_scale(
            encoded_max=16383.0,
            highlight_mode_name="clip",
            wb_values=[1.0, 1.0, 1.0],
            baseline_exposure=None,
        )
        # Divisor already includes BaselineExposure; contract must not multiply it again.
        bundle = _tiny_bundle(
            scene_scale=scale,
            baseline=baseline,
            decoder="libraw",
            exposure_gain=compute_exposure_gain(exposure_mode_for_tone_core("agx"), 0.0),
        )
        contract = scene_scale_contract_from_bundle(bundle, user_ev=0.0)
        self.assertFalse(contract.baseline_baked_in)
        self.assertEqual(contract.baseline_render_gain, 1.0)
        self.assertAlmostEqual(
            float(scale) * baseline_exposure_gain(baseline),
            float(scale_no_baseline),
            places=9,
        )
        # Core Image now clears BaselineExposure in CIRAWFilter and folds it into the
        # scale divisor, so its normal handoff is not marked baked either.
        ci = _tiny_bundle(
            scene_scale=1.0 / baseline_exposure_gain(baseline),
            baseline=baseline,
            decoder="coreimage",
            scale_mode="aligned",
        )
        ci_contract = scene_scale_contract_from_bundle(ci, user_ev=0.0)
        self.assertFalse(ci_contract.baseline_baked_in)
        self.assertEqual(ci_contract.baseline_render_gain, 1.0)

        # An API fallback that could not clear the property remains explicit and still
        # must not multiply the baseline a second time in the contract.
        legacy_ci = _tiny_bundle(
            scene_scale=1.0,
            baseline=baseline,
            baseline_baked=True,
            decoder="coreimage",
            scale_mode="unity",
        )
        legacy_contract = scene_scale_contract_from_bundle(legacy_ci, user_ev=0.0)
        self.assertTrue(legacy_contract.baseline_baked_in)
        self.assertEqual(legacy_contract.baseline_render_gain, 1.0)

    def test_fixed_gains_ignore_scene_content(self) -> None:
        bright = _tiny_bundle(
            rgb=np.full((2, 2, 3), 8000.0, dtype=np.float32),
            scene_scale=4.0,
        )
        dark = _tiny_bundle(
            rgb=np.full((2, 2, 3), 5.0, dtype=np.float32),
            scene_scale=4.0,
        )
        c_bright = scene_scale_contract_from_bundle(bright, user_ev=0.25, tone_core="agx")
        c_dark = scene_scale_contract_from_bundle(dark, user_ev=0.25, tone_core="agx")
        self.assertEqual(c_bright.fixed_midgray_gain, c_dark.fixed_midgray_gain)
        self.assertEqual(c_bright.user_ev_gain, c_dark.user_ev_gain)
        self.assertEqual(c_bright.total_render_gain, c_dark.total_render_gain)

    def test_aligned_coreimage_is_relative_confidence(self) -> None:
        self.assertEqual(
            calibration_confidence_for_mode("coreimage", "aligned"), "relative"
        )
        self.assertEqual(
            calibration_confidence_for_mode("coreimage", "unity"), "decoder-native"
        )
        self.assertEqual(calibration_confidence_for_mode("libraw", None), "calibrated")
        bundle = _tiny_bundle(decoder="coreimage", scale_mode="aligned")
        contract = scene_scale_contract_from_bundle(bundle, user_ev=0.0)
        self.assertEqual(contract.calibration_confidence, "relative")

    def test_concurrent_preview_does_not_share_mutable_exposure(self) -> None:
        shared = _tiny_bundle(scene_scale=2.0, exposure_gain=1.0)
        results: dict[str, float] = {}
        errors: list[BaseException] = []

        def worker(name: str, ev: float) -> None:
            try:
                local = with_intent_exposure(shared, user_ev=ev, tone_core="agx")
                # Simulate render reading exposure_gain after a scheduling gap.
                results[name] = float(local.exposure_gain)
                self.assertNotEqual(local.exposure_gain, shared.exposure_gain)
            except BaseException as exc:  # noqa: BLE001 — collect for main thread
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=("a", 0.0)),
            threading.Thread(target=worker, args=("b", 1.0)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertAlmostEqual(results["b"] / results["a"], 2.0, places=12)
        # Original shared bundle must remain untouched.
        self.assertEqual(shared.exposure_gain, 1.0)

    def test_contract_is_frozen(self) -> None:
        contract = SceneScaleContract(storage_scale=1.0)
        with self.assertRaises(Exception):
            contract.storage_scale = 2.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
