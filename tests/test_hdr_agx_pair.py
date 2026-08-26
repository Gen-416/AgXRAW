# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure-render HDR pair contracts (no Core Image, CI-runnable).

R11 item 1: the old "SDR base == plain export" test compared
render_output_u8 against itself (a determinism tautology) and lived in a
sample-gated class CI never runs. This is the real claim, on a synthetic
scene: the SDR leg of render_ultrahdr_agx_pair must be pixel-identical to
a standalone render_output_u8 export of the same plan — the docstring's
promise ("the native AgX kernel and dither grouping match a standalone
render_output_u8 export"), pinned.
"""
from __future__ import annotations

import unittest

import numpy as np


class UltrahdrPairSdrLegTests(unittest.TestCase):
    def _assert_pair_matches(self, scene) -> None:
        from dngscan.hdr_agx import render_ultrahdr_agx_pair
        from dngscan.hdr_agx_plan import compile_hdr_agx_plan
        from dngscan.tone import build_render_plan
        from dngscan.render import render_output_u8

        plan = build_render_plan(scene.bundle, scene.analysis, "agx", "p3")
        hdr_plan = compile_hdr_agx_plan(plan, analysis=scene.analysis)
        sdr_leg, hdr_linear = render_ultrahdr_agx_pair(
            scene.bundle, scene.analysis, plan, hdr_plan
        )
        plain = render_output_u8(scene.bundle, scene.analysis, "p3", plan)
        self.assertEqual(sdr_leg.shape, plain.shape)
        self.assertTrue(
            bool(np.array_equal(sdr_leg, plain)),
            "the pair's SDR leg diverged from a standalone export: "
            f"{int(np.count_nonzero(sdr_leg != plain))} px differ",
        )
        self.assertTrue(np.all(np.isfinite(hdr_linear)))

    def test_pair_sdr_leg_matches_standalone_export(self) -> None:
        from tests.golden_support import build_daylight_wide_dr

        self._assert_pair_matches(build_daylight_wide_dr())

    def test_pair_parity_holds_with_clipped_highlights(self) -> None:
        """Clip masks are where the two paths could plausibly diverge: the
        staggered-clip scene drives per-channel saturation handling, which
        the daylight scene never touches."""
        from tests.golden_support import build_staggered_clip

        self._assert_pair_matches(build_staggered_clip())


if __name__ == "__main__":
    unittest.main()
