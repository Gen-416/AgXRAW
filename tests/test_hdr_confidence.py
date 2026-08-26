# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-pixel HDR confidence contracts (two-route doctrine item B, 2026-08-26).

Luminance and chroma are authorized separately: the HDR curve owns brightness,
while CFA clip evidence decides how much of that brightness may carry
independent colour. Two mechanisms are pinned here:

1. AgX pair — peak-proximity convergence: clip-compromised pixels lose their
   remaining chroma authority continuously as the native formation luminance
   climbs from reference white to the content peak
   (hdr_color.raw_gated_channel_separation keyword path).
2. Film pair — chroma-authorized increment: the scene-EV luminance gain is
   untouched, but the print colour it amplifies is lerped toward its own luma
   axis where CFA evidence says the hue is reconstructed
   (hdr_agx.render_ultrahdr_film_pair).
"""
from __future__ import annotations

import unittest

import numpy as np

from dngscan.hdr_color import output_luma_weights, raw_gated_channel_separation


class PeakProximityConvergenceTests(unittest.TestCase):
    RHO = 0.5

    def test_unclipped_pixels_keep_full_permission_at_any_luminance(self) -> None:
        masks = np.zeros((4, 3), dtype=np.float32)
        y = np.asarray([0.1, 1.0, 4.0, 8.0], dtype=np.float32)
        rho = raw_gated_channel_separation(self.RHO, masks, y_native=y, peak=8.0)
        np.testing.assert_allclose(rho, np.full((4, 3), self.RHO), rtol=0, atol=1e-7)

    def test_below_reference_white_is_identity_with_legacy_gating(self) -> None:
        masks = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.6, 0.0]], dtype=np.float32)
        legacy = raw_gated_channel_separation(self.RHO, masks)
        y = np.asarray([0.5, 1.0], dtype=np.float32)
        gated = raw_gated_channel_separation(self.RHO, masks, y_native=y, peak=8.0)
        np.testing.assert_allclose(gated, legacy, rtol=0, atol=1e-7)

    def test_fully_clipped_channel_at_peak_withdraws_all_separation(self) -> None:
        masks = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
        y = np.asarray([8.0], dtype=np.float32)
        rho = raw_gated_channel_separation(self.RHO, masks, y_native=y, peak=8.0)
        np.testing.assert_allclose(rho, np.zeros((1, 3)), rtol=0, atol=1e-7)

    def test_convergence_is_monotone_in_luminance(self) -> None:
        masks = np.tile(
            np.asarray([[0.7, 0.2, 0.0]], dtype=np.float32), (8, 1)
        )
        y = np.linspace(1.0, 8.0, 8).astype(np.float32)
        rho = raw_gated_channel_separation(self.RHO, masks, y_native=y, peak=8.0)
        for ch in range(3):
            diffs = np.diff(rho[:, ch])
            self.assertTrue(bool(np.all(diffs <= 1e-7)), f"channel {ch} not monotone")

    def test_unit_peak_disables_the_proximity_term(self) -> None:
        # peak <= 1 means no rendered headroom: the reference table is the
        # native table and the blend never runs, so the keyword path must
        # degrade to the legacy gating rather than divide by zero.
        masks = np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32)
        legacy = raw_gated_channel_separation(self.RHO, masks)
        gated = raw_gated_channel_separation(
            self.RHO, masks, y_native=np.asarray([5.0], dtype=np.float32), peak=1.0
        )
        np.testing.assert_allclose(gated, legacy, rtol=0, atol=1e-7)


class FilmPairChromaAuthorizationTests(unittest.TestCase):
    """The increment formula itself, replicated on the same operations the
    render uses, so the contract is checked without a full film render:

        hdr = decoded + fitted_eff * (gain - 1)
        fitted_eff = fitted * p + Y(fitted) * (1 - p)
        p = (1 - 0.5*max(m)) * (1 - second(m))
    """

    @staticmethod
    def _neutralize(fitted: np.ndarray, masks: np.ndarray, gamut: str) -> np.ndarray:
        w_out = output_luma_weights(gamut).astype(np.float32)
        m = np.clip(masks, 0.0, 1.0)
        second = np.partition(m, 1, axis=-1)[..., 1]
        p = (
            (np.float32(1.0) - np.float32(0.5) * np.max(m, axis=-1))
            * (np.float32(1.0) - second)
        )[..., None]
        y = np.tensordot(fitted, w_out, axes=([-1], [0]))[..., None]
        return fitted * p + y * (1.0 - p)

    def test_multi_clipped_increment_is_neutral_and_luma_preserving(self) -> None:
        gamut = "p3"
        w_out = output_luma_weights(gamut).astype(np.float32)
        fitted = np.asarray([[1.4, 0.3, 0.2]], dtype=np.float32)
        masks = np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32)
        eff = self._neutralize(fitted, masks, gamut)
        # Two clipped channels: the hue is reconstruction, the increment
        # collapses to the luma axis (R == G == B) ...
        self.assertAlmostEqual(float(eff[0, 0]), float(eff[0, 1]), places=6)
        self.assertAlmostEqual(float(eff[0, 1]), float(eff[0, 2]), places=6)
        # ... at exactly the print's own luminance (w_out is normalized).
        y_before = float(np.tensordot(fitted, w_out, axes=([-1], [0]))[0])
        y_after = float(np.tensordot(eff, w_out, axes=([-1], [0]))[0])
        self.assertAlmostEqual(y_after, y_before, places=5)

    def test_unclipped_increment_is_untouched(self) -> None:
        fitted = np.asarray([[1.4, 0.3, 0.2]], dtype=np.float32)
        masks = np.zeros((1, 3), dtype=np.float32)
        eff = self._neutralize(fitted, masks, "p3")
        np.testing.assert_array_equal(eff, fitted)

    def test_single_clip_retains_half_the_chroma(self) -> None:
        gamut = "p3"
        w_out = output_luma_weights(gamut).astype(np.float32)
        fitted = np.asarray([[1.4, 0.3, 0.2]], dtype=np.float32)
        eff = self._neutralize(
            fitted, np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), gamut
        )
        y = np.tensordot(fitted, w_out, axes=([-1], [0]))[..., None]
        np.testing.assert_allclose(eff, fitted * 0.5 + y * 0.5, rtol=0, atol=1e-6)

    def test_render_path_wires_the_authorization(self) -> None:
        """End-to-end on the real render. build_staggered_clip's reliable tail
        (~3.0 EV) sits exactly at the portra400 print join, which compiles the
        film-pair headroom to zero and leaves the gain inert — so the scene is
        brightened +2 EV to earn real headroom, and a block of the brightest
        rows is forced to two-channel clip evidence. Wherever that evidence
        holds AND the luminance gain engaged, the HDR increment the file would
        carry (hdr - decoded_base) must be neutral; bright unclipped pixels
        must keep a chromatic increment. A formula-only test could not catch
        the production path silently dropping the masks."""
        import dataclasses as _dc

        from tests.golden_support import build_staggered_clip
        from dngscan.color import srgb_decode
        from dngscan.hdr_agx import render_ultrahdr_film_pair
        from dngscan.hdr_agx_plan import compile_hdr_agx_plan
        from dngscan.tone import build_render_plan

        scene = build_staggered_clip()
        bright = np.asarray(
            scene.bundle.scene_rec2020_render, dtype=np.float32
        ) * np.float32(4.0)
        luma = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
        ev = np.log2(np.maximum(bright @ luma, 1e-9) / np.float32(0.18))
        forced = np.array(scene.bundle.clip_masks, dtype=np.float32, copy=True)
        # Only the top ~1% of pixels (by rank — the brightest values sit on a
        # clipped plateau, so a strict percentile threshold selects nothing):
        # the reliable-tail selection needs a healthy unclipped evidence pool
        # (masks < 0.10) or the whole plan compiles to zero headroom.
        n_top = max(ev.size // 100, 16)
        target = np.zeros(ev.shape, dtype=bool)
        target.ravel()[np.argsort(ev.ravel())[-n_top:]] = True
        forced[target] = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
        bundle = _dc.replace(
            scene.bundle, scene_rec2020_render=bright, clip_masks=forced
        )
        plan = build_render_plan(
            bundle, scene.analysis, "agx", "srgb",
            film_curve="portra400", film_mode="full", film_crossover="datasheet",
        )
        hdr_plan = compile_hdr_agx_plan(plan, analysis=scene.analysis)
        self.assertGreater(
            float(hdr_plan.tone.rendered_headroom_ev), 0.0,
            "brightened fixture no longer earns HDR headroom",
        )
        base_u8, hdr = render_ultrahdr_film_pair(
            bundle, scene.analysis, plan, hdr_plan, "srgb"
        )
        decoded = srgb_decode(base_u8.astype(np.float32) / 255.0)
        increment = hdr - decoded
        inc_max = np.max(increment, axis=-1)
        inc_span = inc_max - np.min(increment, axis=-1)
        engaged = target & (inc_max > 1e-3)
        self.assertTrue(
            bool(np.any(engaged)),
            "no forced multi-clip pixel engaged the gain — fixture drifted",
        )
        self.assertLess(
            float(np.max(inc_span[engaged])), 1e-4,
            "multi-clipped HDR increment is not neutral — the render path "
            "is not consuming the chroma authorization",
        )
        masks_max = np.max(forced, axis=-1)
        unclipped = (masks_max < 1e-6) & (inc_max > 1e-3)
        if bool(np.any(unclipped)):
            self.assertGreater(float(np.max(inc_span[unclipped])), 1e-4)


if __name__ == "__main__":
    unittest.main()
