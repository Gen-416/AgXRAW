# SPDX-License-Identifier: GPL-3.0-or-later
"""Chroma-only NR (digitization repair, chroma_nr.py) contract gates.

The retained sensor noise is honest texture; this operator may remove ONLY
the low-frequency colour mottle in its declared sensor-pixel band. The
gates pin what the design promises structurally: luminance untouched,
pixel-scale chroma speckle untouched at identity grids, real colour
surviving shrinkage, mottle-band energy genuinely removed, amount 0 a
strict identity through the render entry, and (v2, 2026-09-17) the HDR
entries reading the same repaired scene as the SDR export.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from dngscan.chroma_nr import (
    LUMA_W,
    _atrous_smooth,
    apply_chroma_correction_flat,
    apply_chroma_correction_rows,
    atrous_levels_for,
    chroma_correction_map,
)


def _zero_luma(field: np.ndarray) -> np.ndarray:
    return field - (field @ LUMA_W)[..., None]


def _fixture(rng: np.random.Generator, dh: int = 256) -> dict:
    scene = np.full((dh, dh, 3), 0.02, dtype=np.float32)
    scene[60:130, 60:130] = [0.06, 0.015, 0.012]  # real colour patch
    mottle = rng.standard_normal((dh, dh, 3)).astype(np.float32)
    m = mottle
    for lv in (0, 1, 2):
        m = _atrous_smooth(m, lv)
    m = _zero_luma(m)
    m *= 0.004 / max(float(np.abs(m).std()), 1e-9)
    speckle = _zero_luma(
        rng.standard_normal((dh, dh, 3)).astype(np.float32)
    ) * np.float32(0.003)
    return {"scene": scene, "mottle": m, "speckle": speckle}


class OperatorContractTests(unittest.TestCase):
    def test_band_anchoring_in_sensor_pixels(self) -> None:
        # export-scale grids start at level 0; identity grids skip the
        # levels that would touch pixel-scale speckle
        self.assertEqual(atrous_levels_for(6.8)[0], 0)
        self.assertGreaterEqual(atrous_levels_for(1.0)[0], 3)

    def test_luma_is_untouched(self) -> None:
        rng = np.random.default_rng(3)
        f = _fixture(rng)
        noisy = f["scene"] + f["mottle"] + f["speckle"]
        for factor in (1.0, 6.8):
            corr = chroma_correction_map(noisy, 1.0, decimation_factor=factor)
            self.assertLess(float(np.abs(corr @ LUMA_W).max()), 1e-7)

    def test_identity_grid_keeps_pixel_speckle(self) -> None:
        rng = np.random.default_rng(5)
        f = _fixture(rng)
        noisy = f["scene"] + f["mottle"] + f["speckle"]
        corr = chroma_correction_map(noisy, 1.0, decimation_factor=1.0)
        out = noisy + corr

        def level0(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            return x - _atrous_smooth(x, 0)

        kept = float(
            np.sum(level0(out) * level0(f["speckle"]))
            / max(np.sum(level0(f["speckle"]) ** 2), 1e-12)
        )
        self.assertGreater(kept, 0.98, "pixel speckle is the kept texture")

    def test_mottle_removed_and_colour_kept(self) -> None:
        rng = np.random.default_rng(7)
        f = _fixture(rng)
        noisy = f["scene"] + f["mottle"] + f["speckle"]
        corr = chroma_correction_map(noisy, 1.0, decimation_factor=6.8)
        out = noisy + corr

        def band_rms(x: np.ndarray, levels: set) -> float:
            s = np.asarray(x, dtype=np.float32)
            e = 0.0
            for lv in range(max(levels) + 1):
                c = _atrous_smooth(s, lv)
                if lv in levels:
                    e += float(np.mean((s - c) ** 2))
                s = c
            return float(np.sqrt(e))

        band = {2, 3, 4}
        residual = band_rms(out - f["scene"] - f["speckle"], band) / band_rms(
            f["mottle"], band
        )
        self.assertLess(residual, 0.6, "the mottle band must lose most of "
                        "its amplitude at amount 1")
        patch = (slice(75, 115), slice(75, 115))

        def patch_chroma(x: np.ndarray) -> float:
            c = x[patch] - (x[patch] @ LUMA_W)[..., None]
            return float(np.abs(c).mean())

        self.assertGreater(
            patch_chroma(out) / patch_chroma(noisy), 0.9,
            "the garrote must not desaturate a real colour patch",
        )

    def test_amount_scales_monotonically(self) -> None:
        rng = np.random.default_rng(9)
        f = _fixture(rng)
        noisy = f["scene"] + f["mottle"] + f["speckle"]
        removed = [
            float(np.abs(chroma_correction_map(
                noisy, amount, decimation_factor=6.8
            )).sum())
            for amount in (0.25, 0.5, 1.0)
        ]
        self.assertLess(removed[0], removed[1])
        self.assertLess(removed[1], removed[2])

    def test_amount_zero_is_refused_at_operator_level(self) -> None:
        with self.assertRaises(ValueError):
            chroma_correction_map(np.zeros((8, 8, 3), np.float32), 0.0)

    def test_flat_apply_matches_row_apply_across_cuts(self) -> None:
        rng = np.random.default_rng(11)
        h, w = 64, 96
        rows = rng.uniform(0.0, 0.4, (h, w, 3)).astype(np.float32)
        corr = _zero_luma(
            rng.standard_normal((16, 24, 3)).astype(np.float32) * 0.01
        )
        whole = apply_chroma_correction_rows(rows, corr, 0, h, h, w)
        flat = rows.reshape(-1, 3)
        got = np.empty_like(flat)
        # chunk cuts that split rows mid-way
        for s, e in ((0, 1000), (1000, 3500), (3500, h * w)):
            got[s:e] = apply_chroma_correction_flat(flat[s:e], corr, s, e, h, w)
        np.testing.assert_array_equal(got, whole)


class RenderEntryTests(unittest.TestCase):
    def test_plan_domain_fails_closed(self) -> None:
        from tests.golden_support import build_daylight_wide_dr
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        with self.assertRaisesRegex(ValueError, "chroma_nr"):
            build_render_plan(
                scene.bundle, scene.analysis, "agx", "srgb", chroma_nr=1.5
            )
        plan = build_render_plan(
            scene.bundle, scene.analysis, "agx", "srgb", chroma_nr=0.6
        )
        self.assertEqual(float(plan.tone.chroma_nr), 0.6)

    def test_amount_zero_is_a_strict_identity(self) -> None:
        from tests.golden_support import build_daylight_wide_dr
        from dngscan.render import scene_render_to_display_linear
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        base_plan = build_render_plan(scene.bundle, scene.analysis, "agx", "srgb")
        self.assertEqual(float(base_plan.tone.chroma_nr), 0.0)
        base = scene_render_to_display_linear(scene.bundle, base_plan, "srgb")
        again = scene_render_to_display_linear(scene.bundle, base_plan, "srgb")
        np.testing.assert_array_equal(base, again)

    def test_engaged_render_changes_chroma_not_luma_ranking(self) -> None:
        """A render with the dial on differs from the base render, and the
        difference at the SCENE stage is chroma-only by construction — here
        we just pin that the end-to-end path engages at all."""
        from tests.golden_support import build_night_sparse_lamps
        from dngscan.render import scene_render_to_display_linear
        from dngscan.tone import build_render_plan

        scene = build_night_sparse_lamps()
        base = scene_render_to_display_linear(
            scene.bundle,
            build_render_plan(scene.bundle, scene.analysis, "agx", "srgb"),
            "srgb",
        )
        treated = scene_render_to_display_linear(
            scene.bundle,
            build_render_plan(
                scene.bundle, scene.analysis, "agx", "srgb", chroma_nr=1.0
            ),
            "srgb",
        )
        self.assertFalse(np.array_equal(base, treated),
                         "the dial must reach the pixels")

    def _hdr_pair(self, scene, amount: float):
        from dngscan.hdr_agx import render_ultrahdr_agx_pair
        from dngscan.hdr_agx_plan import compile_hdr_agx_plan
        from dngscan.tone import build_render_plan

        plan = build_render_plan(
            scene.bundle, scene.analysis, "agx", "p3", chroma_nr=amount
        )
        hdr_plan = compile_hdr_agx_plan(plan, analysis=scene.analysis)
        return plan, hdr_plan, render_ultrahdr_agx_pair(
            scene.bundle, scene.analysis, plan, hdr_plan
        )

    def test_hdr_pair_sdr_leg_matches_the_standalone_repaired_export(self) -> None:
        """v2 (2026-09-17): the pair's two legs read ONE repaired scene. The
        SDR leg must stay pixel-identical to a standalone export of the same
        plan — the v1 refusal existed because an unrepaired HDR leg would
        have made the gain map encode the repair."""
        from tests.golden_support import build_night_sparse_lamps
        from dngscan.render import render_output_u8

        scene = build_night_sparse_lamps()
        plan, _hdr_plan, (sdr_leg, hdr_linear) = self._hdr_pair(scene, 1.0)
        plain = render_output_u8(scene.bundle, scene.analysis, "p3", plan)
        self.assertTrue(bool(np.array_equal(sdr_leg, plain)))
        self.assertTrue(np.all(np.isfinite(hdr_linear)))

    def test_hdr_legs_engage_and_the_standalone_formation_agrees(self) -> None:
        """The dial reaches the HDR pixels, and the standalone HDR formation
        (the gain-map-less HDR entry) renders the SAME repaired scene as the
        pair's HDR leg."""
        from tests.golden_support import build_night_sparse_lamps
        from dngscan.hdr_agx import scene_render_to_hdr_display_linear

        scene = build_night_sparse_lamps()
        _p0, _h0, (_sdr0, hdr0) = self._hdr_pair(scene, 0.0)
        plan, hdr_plan, (_sdr1, hdr1) = self._hdr_pair(scene, 1.0)
        self.assertFalse(np.array_equal(hdr0, hdr1), "the dial must reach the HDR leg")
        alone = scene_render_to_hdr_display_linear(scene.bundle, plan, hdr_plan)
        np.testing.assert_array_equal(alone, hdr1)

    def test_hdr_amount_zero_is_a_strict_identity(self) -> None:
        from tests.golden_support import build_daylight_wide_dr

        scene = build_daylight_wide_dr()
        _p, _h, (sdr_a, hdr_a) = self._hdr_pair(scene, 0.0)
        _p, _h, (sdr_b, hdr_b) = self._hdr_pair(scene, 0.0)
        np.testing.assert_array_equal(sdr_a, sdr_b)
        np.testing.assert_array_equal(hdr_a, hdr_b)


if __name__ == "__main__":
    unittest.main()
