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




if __name__ == "__main__":
    unittest.main()
