# SPDX-License-Identifier: GPL-3.0-or-later
"""Scheduler plan S4 acceptance gates: staged buffer lifetimes and bounded
decode-stage transients.

The two decode hot spots are banded, and the gates prove the banding is
BIT-IDENTICAL to the whole-frame arithmetic it replaces — a memory fix that
changed a pixel would be a silent regression in every rendered file.
"""
from __future__ import annotations

import unittest

import numpy as np


class FeatherBandingTests(unittest.TestCase):
    def test_banded_feather_is_bit_identical(self) -> None:
        from dngscan.raw_io import _feather_masks, _feather_masks_f16

        rng = np.random.default_rng(0)
        for shape in ((400, 601, 3), (37, 41, 3), (5, 5, 3), (1, 9, 3), (9, 1, 3)):
            mask = rng.uniform(0.0, 1.0, shape).astype(np.float32)
            whole = _feather_masks(mask).astype(np.float16, copy=False)
            banded = _feather_masks_f16(mask)
            np.testing.assert_array_equal(
                banded, whole,
                f"{shape}: banded feather diverged from the whole-frame filter",
            )

    def test_feather_still_clamps_to_unit_range(self) -> None:
        from dngscan.raw_io import _feather_masks_f16

        mask = np.full((32, 48, 3), 2.0, dtype=np.float32)
        out = _feather_masks_f16(mask)
        self.assertEqual(out.dtype, np.float16)
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertGreaterEqual(float(out.min()), 0.0)


class GainMapBandingTests(unittest.TestCase):
    """The 1x1-pitch GainMap path is the single biggest decode transient;
    the banded form must reproduce the historical expression exactly."""

    def _legacy(self, img, colors, m, blacks, white_level):
        h, w = img.shape
        rows = np.arange(m["top"], min(m["bottom"], h), m["row_pitch"])
        cols = np.arange(m["left"], min(m["right"], w), m["col_pitch"])
        gains_grid = np.mean(np.asarray(m["gains"], dtype=np.float64), axis=2)
        iv = np.clip(
            ((rows + 0.5) / h - m["origin_v"]) / max(m["spacing_v"], 1e-9),
            0, m["points_v"] - 1,
        )
        ih = np.clip(
            ((cols + 0.5) / w - m["origin_h"]) / max(m["spacing_h"], 1e-9),
            0, m["points_h"] - 1,
        )
        v0 = np.clip(np.floor(iv).astype(int), 0, m["points_v"] - 2)
        h0 = np.clip(np.floor(ih).astype(int), 0, m["points_h"] - 2)
        fv = (iv - v0)[:, None]
        fh = (ih - h0)[None, :]
        h1 = np.minimum(h0 + 1, m["points_h"] - 1)
        rows_lo = gains_grid[v0]
        rows_hi = gains_grid[np.minimum(v0 + 1, m["points_v"] - 1)]
        gains = (
            rows_lo[:, h0] * (1 - fv) * (1 - fh) + rows_lo[:, h1] * (1 - fv) * fh
            + rows_hi[:, h0] * fv * (1 - fh) + rows_hi[:, h1] * fv * fh
        )
        view = img[m["top"]:min(m["bottom"], h):m["row_pitch"],
                   m["left"]:min(m["right"], w):m["col_pitch"]]
        cidx = colors[m["top"]:min(m["bottom"], h):m["row_pitch"],
                      m["left"]:min(m["right"], w):m["col_pitch"]]
        sub = view.astype(np.float32)
        b = blacks[np.clip(cidx, 0, blacks.size - 1)]
        view[...] = np.clip(
            b + (sub - b) * gains, 0.0, float(white_level)
        ).astype(img.dtype)

    def test_banded_gain_map_is_bit_identical(self) -> None:
        from types import SimpleNamespace

        from dngscan.raw_io import _apply_gain_maps_mosaic

        rng = np.random.default_rng(7)
        h, w = 301, 407
        base = rng.integers(0, 16000, (h, w), dtype=np.uint16)
        colors = (np.indices((h, w)).sum(axis=0) % 4).astype(np.uint8)
        gains = rng.uniform(0.9, 1.4, (9, 11, 1))
        spec = dict(
            top=1, bottom=h, left=2, right=w, row_pitch=1, col_pitch=1,
            gains=gains, origin_v=0.0, origin_h=0.0,
            spacing_v=1.0 / 8, spacing_h=1.0 / 10, points_v=9, points_h=11,
        )
        blacks = np.asarray([512.0, 514.0, 512.0, 513.0], dtype=np.float32)

        want = base.copy()
        self._legacy(want, colors, spec, blacks, 16383)

        got = base.copy()
        raw = SimpleNamespace(raw_image_visible=got, raw_colors_visible=colors)
        _apply_gain_maps_mosaic(
            raw, [SimpleNamespace(**spec)], [512.0, 514.0, 512.0, 513.0], 16383
        )
        np.testing.assert_array_equal(
            got, want,
            "banded GainMap application diverged from the whole-frame form",
        )


class StagedReleaseTests(unittest.TestCase):
    def test_release_drops_only_the_analysis_buffer(self) -> None:
        from tests.golden_support import build_daylight_wide_dr
        from dngscan.raw_io import release_analysis_buffers

        scene = build_daylight_wide_dr()
        released = release_analysis_buffers(scene.bundle)
        self.assertIsNone(released.xyz_render)
        # the render path's inputs survive untouched
        self.assertIsNotNone(released.scene_rec2020_render)
        self.assertIs(released.raw_image, scene.bundle.raw_image)
        self.assertIs(released.raw_colors, scene.bundle.raw_colors)
        self.assertIs(
            released.scene_rec2020_render, scene.bundle.scene_rec2020_render
        )

    def test_export_path_releases_before_exporting(self) -> None:
        import inspect

        from dngscan import cli

        src = inspect.getsource(cli.main)
        release = src.find("release_analysis_buffers")
        export = src.find("export_result = export_jpeg")
        self.assertGreater(release, 0, "the CLI export path must stage the release")
        self.assertLess(release, export, "release must precede the export")


if __name__ == "__main__":
    unittest.main()
