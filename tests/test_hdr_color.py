# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 3 gates: colour geometry that changes chroma and nothing else."""
from __future__ import annotations

import unittest

import numpy as np

from dngscan.agx import formation_matrices
from dngscan.constants import REC2020_LUMA, RGB_TO_XYZ
from dngscan.hdr_color import (
    blend_native_hdr_paths,
    fit_hdr_color_volume,
    neutral_axis_lambda,
    output_luma_weights,
    formation_luma_weights,
    raw_gated_channel_separation,
)

P3_LUMA = output_luma_weights("p3")


def _formation(n: int = 40000, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)


class NativePathBlendTests(unittest.TestCase):
    """Tone/color separation identities of the independent HDR formation."""

    def test_equal_paths_are_bit_identical_at_every_rho(self) -> None:
        f = _formation()
        for rho in (0.0, 0.5, 1.0):
            with self.subTest(rho=rho):
                out = blend_native_hdr_paths(f, f, rho, P3_LUMA)
                self.assertTrue(bool(np.array_equal(out, f)))

    def test_rho_zero_uses_reference_chromaticity_at_native_luminance(self) -> None:
        reference = np.asarray([[0.30, 0.10, 0.10]], dtype=np.float32)
        native = np.asarray([[1.20, 0.20, 0.10]], dtype=np.float32)
        out = blend_native_hdr_paths(reference, native, 0.0, P3_LUMA)
        np.testing.assert_allclose(out @ P3_LUMA, native @ P3_LUMA, atol=2e-7, rtol=0.0)
        ref_ratio = reference[0] / reference[0, 1]
        out_ratio = out[0] / out[0, 1]
        np.testing.assert_allclose(out_ratio, ref_ratio, atol=2e-6, rtol=0.0)

    def test_rho_one_returns_native_path_exactly(self) -> None:
        reference, native = _formation(seed=2), _formation(seed=8) * np.float32(3.0)
        out = blend_native_hdr_paths(reference, native, 1.0, P3_LUMA)
        self.assertTrue(bool(np.array_equal(out, native)))

    def test_rho_does_not_move_luminance(self) -> None:
        """The whole point: rho is a colour control, not a second tone control."""
        reference, native = _formation(seed=2), _formation(seed=8) * np.float32(3.0)
        target = native @ P3_LUMA
        for rho in (0.25, 0.5, 0.75, 1.0):
            with self.subTest(rho=rho):
                y = blend_native_hdr_paths(reference, native, rho, P3_LUMA) @ P3_LUMA
                rel = np.abs(y - target) / np.maximum(target, 1e-4)
                self.assertLess(float(np.median(rel)), 1e-6)
                self.assertLess(float(np.max(rel)), 1e-2)

    def test_neutral_input_stays_neutral_at_every_rho(self) -> None:
        reference = np.repeat(np.linspace(0.01, 1.0, 2000, dtype=np.float32)[:, None], 3, axis=1)
        native = np.repeat(np.linspace(0.01, 4.0, 2000, dtype=np.float32)[:, None], 3, axis=1)
        for rho in (0.0, 0.5, 1.0):
            with self.subTest(rho=rho):
                out = blend_native_hdr_paths(reference, native, rho, P3_LUMA)
                self.assertEqual(float(np.max(out.max(1) - out.min(1))), 0.0)

    def test_higher_rho_moves_toward_native_highlight_chroma(self) -> None:
        """Otherwise rho would be an inert parameter that only looks like a control."""
        rng = np.random.default_rng(11)
        reference = rng.uniform(0.2, 0.8, size=(20000, 3)).astype(np.float32)
        reference = reference * np.float32(0.35) + np.mean(reference, axis=1, keepdims=True) * np.float32(0.65)
        native = rng.uniform(0.2, 4.0, size=(20000, 3)).astype(np.float32)
        native[:, 2] *= np.float32(0.15)
        sats = []
        for rho in (0.0, 0.5, 1.0):
            out = blend_native_hdr_paths(reference, native, rho, P3_LUMA)
            hi, lo = out.max(1), out.min(1)
            sats.append(float(np.median((hi - lo) / np.maximum(hi, 1e-6))))
        self.assertLess(sats[0], sats[1])
        self.assertLess(sats[1], sats[2])

    def test_cfa_masks_gate_the_corresponding_channel_and_multiclip(self) -> None:
        masks = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
        rho = raw_gated_channel_separation(0.5, masks)
        np.testing.assert_allclose(rho[0], [0.25, 0.5, 0.5], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(rho[1], [0.0, 0.0, 0.0], atol=0.0, rtol=0.0)

    def test_formation_luma_uses_the_actual_outset_not_inverse_inset(self) -> None:
        class Plan:
            agx_primaries = "base"

        inset, outset = formation_matrices(Plan())
        got = formation_luma_weights(outset)
        expected = np.asarray(REC2020_LUMA, dtype=np.float64) @ outset
        expected /= expected.sum()
        wrong = np.asarray(REC2020_LUMA, dtype=np.float64) @ np.linalg.inv(inset)
        wrong /= wrong.sum()
        np.testing.assert_allclose(got, expected, atol=5e-8, rtol=0.0)
        self.assertGreater(float(np.max(np.abs(got - wrong))), 0.05)


class GamutProjectorTests(unittest.TestCase):
    def test_ceiling_is_enforced(self) -> None:
        rng = np.random.default_rng(7)
        rgb = rng.uniform(-2.0, 14.0, size=(60000, 3)).astype(np.float32)
        out = fit_hdr_color_volume(rgb, 8.0, "p3")
        self.assertLessEqual(float(np.max(out)), 8.0 + 1e-5)

    def test_floor_is_enforced_by_neutral_axis_projection(self) -> None:
        rgb = np.array([[-0.05, 0.4, 0.2]], dtype=np.float32)
        out = fit_hdr_color_volume(rgb, 8.0, "p3")
        self.assertGreaterEqual(float(np.min(out)), 0.0)
        self.assertFalse(bool(np.array_equal(out, np.clip(rgb, 0.0, 8.0))))

    def test_in_gamut_pixels_are_bit_identical(self) -> None:
        """Reconstructing them as y + 1.0*(arr-y) is not exact in float32."""
        rng = np.random.default_rng(5)
        rgb = rng.uniform(0.0, 3.0, size=(50000, 3)).astype(np.float32)
        self.assertTrue(bool(np.array_equal(fit_hdr_color_volume(rgb, 8.0, "p3"), rgb)))

    def test_neutral_axis_is_preserved_exactly(self) -> None:
        neutral = np.repeat(np.linspace(0.0, 12.0, 500, dtype=np.float32)[:, None], 3, axis=1)
        out = fit_hdr_color_volume(neutral, 8.0, "p3")
        self.assertEqual(float(np.max(out.max(1) - out.min(1))), 0.0)

    def test_luminance_survives_the_projection(self) -> None:
        """Tone decided the luminance; the gamut fit must not be able to overrule it."""
        rng = np.random.default_rng(9)
        rgb = rng.uniform(0.0, 14.0, size=(40000, 3)).astype(np.float32)
        out = fit_hdr_color_volume(rgb, 8.0, "p3")
        y_in, y_out = rgb @ P3_LUMA, out @ P3_LUMA
        inside = (y_in > 1e-3) & (y_in < 8.0)
        rel = np.abs(y_out[inside] - y_in[inside]) / y_in[inside]
        self.assertLess(float(np.max(rel)), 2e-6)

    def test_luma_weights_keep_the_neutral_axis_exact(self) -> None:
        self.assertAlmostEqual(float(np.sum(P3_LUMA)), 1.0, places=7)

    def test_luma_weight_normalisation_does_not_mutate_sdr_matrices(self) -> None:
        before = np.array(RGB_TO_XYZ["P3"], copy=True)
        _ = output_luma_weights("p3")
        self.assertTrue(bool(np.array_equal(RGB_TO_XYZ["P3"], before)))

    def test_projection_reduces_chroma_rather_than_clipping_channels(self) -> None:
        """The output-RGB opponent direction is stable, unlike per-channel clipping."""
        rgb = np.array([[20.0, 4.0, 1.0]], dtype=np.float32)
        out = fit_hdr_color_volume(rgb, 8.0, "p3")
        clipped = np.minimum(rgb, 8.0)
        self.assertFalse(bool(np.allclose(out, clipped)))
        lam = float(neutral_axis_lambda(rgb, 8.0, P3_LUMA)[0])
        self.assertGreater(lam, 0.0)
        self.assertLess(lam, 1.0)


if __name__ == "__main__":
    unittest.main()
