# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the optional C++ AgX backend."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from dngscan import _fast as fast_backend
from dngscan.agx import apply_core, formation_matrices
from dngscan.color import (
    encode_display_linear,
    fit_to_output_gamut,
    rec2020_to_output,
)
from dngscan.fast_plan import (
    NATIVE_ABI_VERSION,
    NATIVE_OUTPUT_GAMUT_FIT_ITERS,
    NATIVE_OUTPUT_GAMUT_TOLERANCE,
)
from dngscan.models import ToneCompressionPlan
from dngscan.punch import apply_punch_rec2020
from dngscan.render import (
    apply_agx_core,
    dither_quantize_u8_with_noise,
    dither_quantize_u8_with_tpdf,
)


def _sample_plan(**overrides) -> ToneCompressionPlan:
    base = dict(
        target_gamut="Rec2020",
        luma_p1=0.01,
        luma_p50=0.18,
        luma_p99=1.0,
        luma_p999=2.0,
        black_ev=-8.0,
        white_ev=5.0,
        dynamic_range_ev=13.0,
        contrast=3.0,
        toe_power=1.5,
        shoulder_power=2.9,
        chroma_p95=0.5,
        negative_rgb_pct=0.0,
        over_rgb_pct=0.0,
        tone_core="agx",
        use_c1_endpoints=True,
        punch_strength=0.0,
        hue_restore=0.6,
        agx_primaries="base",
    )
    base.update(overrides)
    return ToneCompressionPlan(**base)


def _reference_agx_core(rgb: np.ndarray, plan: ToneCompressionPlan) -> np.ndarray:
    inset, outset = formation_matrices(plan)
    mapped = apply_core(rgb, plan, inset, outset)
    return apply_punch_rec2020(mapped, float(plan.punch_strength))


@unittest.skipUnless(fast_backend.available(), "native extension not built")
class NativeAgxParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = os.environ.get("DNGSCAN_FAST")
        os.environ["DNGSCAN_FAST"] = "1"

    def tearDown(self) -> None:
        if self._env is None:
            os.environ.pop("DNGSCAN_FAST", None)
        else:
            os.environ["DNGSCAN_FAST"] = self._env

    def test_basis_colors_match_reference(self) -> None:
        plan = _sample_plan(hue_restore=1.0)
        rgb = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        ref = _reference_agx_core(rgb, plan)
        out = apply_agx_core(rgb, plan)
        np.testing.assert_allclose(out, ref, rtol=0.0, atol=2e-5)

    def test_punch_zero_is_identity(self) -> None:
        plan = _sample_plan(punch_strength=0.0)
        rng = np.random.default_rng(3)
        rgb = rng.uniform(0.0, 1.0, size=(128, 3)).astype(np.float32)
        ref = _reference_agx_core(rgb, plan)
        out = apply_agx_core(rgb, plan)
        np.testing.assert_allclose(out, ref, rtol=0.0, atol=1e-5)

    def test_punch_matches_python(self) -> None:
        plan = _sample_plan(punch_strength=0.35)
        rgb = np.asarray([[0.28, 0.16, 0.10], [0.55, 0.40, 0.32]], dtype=np.float32)
        ref = _reference_agx_core(rgb, plan)
        out = apply_agx_core(rgb, plan)
        np.testing.assert_allclose(out, ref, rtol=0.0, atol=2e-5)

    def test_synthetic_scene_matches_reference(self) -> None:
        rng = np.random.default_rng(11)
        rgb = rng.uniform(0.0, 1.5, size=(4096, 3)).astype(np.float32)
        for primaries in ("smooth", "base", "punchy", "muted"):
            for hue_restore in (0.0, 0.4, 0.6, 1.0):
                plan = _sample_plan(agx_primaries=primaries, hue_restore=hue_restore)
                ref = _reference_agx_core(rgb, plan)
                out = apply_agx_core(rgb, plan)
                np.testing.assert_allclose(out, ref, rtol=0.0, atol=2e-5, err_msg=primaries)

    def test_parallel_kernel_matches_reference(self) -> None:
        plan = _sample_plan(agx_primaries="punchy", hue_restore=0.7)
        rng = np.random.default_rng(29)
        # Deliberately exceeds the native kernel's parallel threshold.
        rgb = rng.uniform(0.0, 2.0, size=(140_000, 3)).astype(np.float32)
        ref = _reference_agx_core(rgb, plan)
        out = apply_agx_core(rgb, plan)
        np.testing.assert_allclose(out, ref, rtol=0.0, atol=2e-5)

    def test_view_brightness_both_sides_match_reference(self) -> None:
        rgb = np.asarray([[0.02, 0.08, 0.25], [0.8, 0.45, 0.12]], dtype=np.float32)
        for brightness in (0.64, 1.25):
            plan = _sample_plan(view_brightness=brightness)
            ref = _reference_agx_core(rgb, plan)
            out = apply_agx_core(rgb, plan)
            np.testing.assert_allclose(out, ref, rtol=0.0, atol=2e-5)

    def test_extended_target_white_matches_reference(self) -> None:
        plan = _sample_plan(target_white_linear=8.0)
        rgb = np.asarray([[0.18, 0.18, 0.18], [8.0, 4.0, 1.0]], dtype=np.float32)
        ref = _reference_agx_core(rgb, plan)
        out = apply_agx_core(rgb, plan)
        self.assertGreater(float(np.max(ref)), 1.0)
        np.testing.assert_allclose(out, ref, rtol=0.0, atol=2e-5)

    def test_fast_does_not_mutate_input(self) -> None:
        plan = _sample_plan()
        rgb = np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32)
        before = rgb.copy()
        apply_agx_core(rgb, plan)
        np.testing.assert_array_equal(rgb, before)

    def test_nan_inf_contract_matches_python(self) -> None:
        # Audit R11: with non-finite input the dispatcher and the reference
        # share the SAME NumPy path, so comparing them alone is a tautology.
        # The real contract is the ROUTING: non-finite frames must never
        # reach the native kernel, even when the backend claims support.
        plan = _sample_plan(punch_strength=0.8)
        rgb = np.asarray([[np.nan, 0.3, np.inf], [-np.inf, 0.2, 0.1]], dtype=np.float32)
        ref = _reference_agx_core(rgb, plan)
        kernel = mock.Mock(side_effect=AssertionError("native kernel saw non-finite input"))
        with mock.patch.object(fast_backend, "supports_agx", return_value=True), \
                mock.patch.object(fast_backend, "apply_agx_core_f32", kernel):
            out = apply_agx_core(rgb, plan)
        kernel.assert_not_called()
        np.testing.assert_allclose(out, ref, rtol=0.0, atol=2e-5)


@unittest.skipUnless(fast_backend.available(), "native extension not built")
class NativeOutputParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = os.environ.get("DNGSCAN_FAST")
        os.environ["DNGSCAN_FAST"] = "1"

    def tearDown(self) -> None:
        if self._env is None:
            os.environ.pop("DNGSCAN_FAST", None)
        else:
            os.environ["DNGSCAN_FAST"] = self._env

    @staticmethod
    def _samples(random_count: int = 20_000) -> np.ndarray:
        boundary = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1e-4, 0.5, 1.0001],
                [-2.0, 0.5, 4.0],
                [np.nan, np.inf, -np.inf],
            ],
            dtype=np.float32,
        )
        rng = np.random.default_rng(20260802)
        random = rng.uniform(-2.0, 4.0, size=(random_count, 3)).astype(np.float32)
        return np.ascontiguousarray(np.concatenate((boundary, random), axis=0))

    def test_abi_and_algorithm_constants_are_fixed(self) -> None:
        from dngscan import _dngscan_fast as ext

        self.assertEqual(ext.native_abi_version(), NATIVE_ABI_VERSION)
        # v9: peak-proximity chroma convergence in the HDR gated blend
        # (two-route doctrine item B).
        self.assertEqual(NATIVE_ABI_VERSION, 9)
        self.assertEqual(NATIVE_OUTPUT_GAMUT_FIT_ITERS, 16)
        self.assertEqual(NATIVE_OUTPUT_GAMUT_TOLERANCE, 1e-4)
        for gamut in ("srgb", "p3"):
            plan = fast_backend.compile_output_plan(gamut, 0.05)
            self.assertEqual(plan.gamut_fit_iters, 16)
            self.assertEqual(plan.gamut_tolerance, 1e-4)

    def test_float_gamut_fit_matches_reference(self) -> None:
        rgb = self._samples()
        before = rgb.copy()
        for gamut in ("srgb", "p3"):
            for alpha in (0.045, 0.05, 0.075):
                plan = fast_backend.compile_output_plan(gamut, alpha)
                ref = fit_to_output_gamut(rgb, gamut, alpha=alpha, iters=16)
                out = fast_backend.fit_output_gamut_f32(rgb, plan)
                delta = np.abs(out - ref)
                self.assertLessEqual(float(delta.max()), 1e-4)
                self.assertLessEqual(float(np.percentile(delta, 99.99)), 5e-5)
                self.assertTrue(np.isfinite(out).all())
                self.assertGreaterEqual(float(out.min()), 0.0)
                self.assertLessEqual(float(out.max()), 1.0)
        np.testing.assert_array_equal(rgb, before)

    def test_in_gamut_values_are_not_changed_before_clipping(self) -> None:
        rgb = np.asarray(
            [[0.0, 0.25, 1.0], [1e-5, 0.5, 0.99999], [-1e-4, 1.0001, 0.4]],
            dtype=np.float32,
        )
        for gamut in ("srgb", "p3"):
            plan = fast_backend.compile_output_plan(gamut, 0.05)
            out = fast_backend.fit_output_gamut_f32(rgb, plan)
            np.testing.assert_array_equal(out, np.clip(rgb, 0.0, 1.0))

    def test_fused_u8_matches_reference_with_same_noise(self) -> None:
        """R2 item 6 tightened this gate: the Rec.2020 -> output stage is the
        exact two-stage NumPy graph now (float64 accumulate, float32 per-stage
        materialization), so IN-GAMUT pixels are bit-exact — the old merged
        matrix's 8.6e-5 deviation moved ~1% of codes by one. Out-of-gamut
        pixels go through the native gamut fit, which keeps its documented
        1e-4 float tolerance (its matrices remain pre-merged; unmerging them
        belongs to the reference-mode program), so a residual <=1-code
        difference is permitted there only."""
        from dngscan.color import _rgb_rows_in_unit_gamut

        rng = np.random.default_rng(41)
        rec2020 = rng.uniform(-1.0, 2.5, size=(200_000, 3)).astype(np.float32)
        noise_a = rng.random(rec2020.shape, dtype=np.float32)
        noise_b = rng.random(rec2020.shape, dtype=np.float32)
        for gamut in ("srgb", "p3"):
            plan = fast_backend.compile_output_plan(gamut, 0.05)
            linear = rec2020_to_output(rec2020, gamut)
            fitted = fit_to_output_gamut(linear, gamut, alpha=0.05, iters=16)
            ref = dither_quantize_u8_with_noise(
                encode_display_linear(fitted, gamut), noise_a, noise_b
            )
            out = fast_backend.finalize_rec2020_u8_f32(
                rec2020, noise_a, noise_b, plan
            )
            delta = np.abs(out.astype(np.int16) - ref.astype(np.int16))
            sanitized = np.nan_to_num(
                linear, nan=0.0, posinf=1e6, neginf=-1e6
            ).astype(np.float32, copy=False)
            in_gamut = np.asarray(
                _rgb_rows_in_unit_gamut(sanitized, np.float32(1e-4))
            )
            self.assertTrue(bool(np.any(in_gamut)))
            self.assertEqual(int(delta[in_gamut].max()), 0)
            self.assertLessEqual(int(delta.max()), 1)
            self.assertLessEqual(float(np.mean(delta != 0)), 0.0005)

    def test_parallel_finalizer_is_deterministic(self) -> None:
        rng = np.random.default_rng(79)
        rgb = rng.uniform(-1.0, 2.0, size=(140_000, 3)).astype(np.float32)
        noise_a = rng.random(rgb.shape, dtype=np.float32)
        noise_b = rng.random(rgb.shape, dtype=np.float32)
        plan = fast_backend.compile_output_plan("srgb", 0.05)
        first = fast_backend.finalize_output_u8_f32(rgb, noise_a, noise_b, plan)
        second = fast_backend.finalize_output_u8_f32(rgb, noise_a, noise_b, plan)
        np.testing.assert_array_equal(first, second)

    def test_combined_noise_finalizer_matches_reference(self) -> None:
        rng = np.random.default_rng(97)
        rec2020 = rng.uniform(-1.0, 2.5, size=(140_000, 3)).astype(np.float32)
        noise_a = rng.random(rec2020.shape, dtype=np.float32)
        noise_b = rng.random(rec2020.shape, dtype=np.float32)
        noise = noise_a - noise_b
        plan = fast_backend.compile_output_plan("srgb", 0.05)
        linear = rec2020_to_output(rec2020, "srgb")
        fitted = fit_to_output_gamut(linear, "srgb", alpha=0.05, iters=16)
        reference = dither_quantize_u8_with_tpdf(
            encode_display_linear(fitted, "srgb"), noise
        )
        actual = fast_backend.finalize_rec2020_u8_noise_f32(
            rec2020, noise, plan
        )
        previous = fast_backend.finalize_rec2020_u8_f32(
            rec2020, noise_a, noise_b, plan
        )
        previous_delta = np.abs(
            actual.astype(np.int16) - previous.astype(np.int16)
        )
        self.assertLessEqual(int(previous_delta.max()), 1)
        self.assertEqual(float(np.percentile(previous_delta, 99)), 0.0)
        self.assertLess(float(np.mean(previous_delta != 0)), 0.0001)
        delta = np.abs(actual.astype(np.int16) - reference.astype(np.int16))
        self.assertLessEqual(int(delta.max()), 1)
        self.assertEqual(float(np.percentile(delta, 99)), 0.0)

class NativeDispatchTests(unittest.TestCase):
    def test_fast_unavailable_falls_back(self) -> None:
        plan = _sample_plan()
        rgb = np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32)
        with mock.patch.object(fast_backend, "available", return_value=False):
            ref = _reference_agx_core(rgb, plan)
            out = apply_agx_core(rgb, plan)
        np.testing.assert_allclose(out, ref, rtol=0.0, atol=1e-6)

    def test_fast_rejects_non_agx_cores(self) -> None:
        for core in ("neutral", "lum", "gated"):
            plan = _sample_plan(tone_core=core)
            self.assertFalse(fast_backend.supports_agx(plan))

    def test_fast_accepts_c_contiguous_float32(self) -> None:
        plan = _sample_plan()
        rgb = np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32)
        self.assertTrue(fast_backend.can_use_agx(rgb, plan) or not fast_backend.available())
        rgb_f64 = rgb.astype(np.float64)
        self.assertFalse(fast_backend.can_use_agx(rgb_f64, plan))

    def test_abi_mismatch_is_unavailable(self) -> None:
        with mock.patch.object(fast_backend, "_load_extension", return_value=None):
            with mock.patch.object(fast_backend, "_extension_error", "native ABI mismatch"):
                self.assertFalse(fast_backend.available())


if __name__ == "__main__":
    unittest.main()
