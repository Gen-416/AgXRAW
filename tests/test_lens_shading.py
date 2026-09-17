# SPDX-License-Identifier: GPL-3.0-or-later
"""DNG dark-field correction gates: opcode parsing and application semantics."""
from __future__ import annotations

import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dngscan.metadata import (
    DngVignetteRadial,
    _parse_gain_map_payload,
    read_dng_shading_ops,
)
from dngscan.raw_io import _apply_gain_maps_mosaic, _apply_vignette_render, _orient_like_libraw

SAMPLE_FP = Path.home() / "Pictures" / "AgXRAW样张" / "_SDI0150.DNG"
SAMPLE_IPHONE = Path.home() / "Pictures" / "Original RAW 26-05-11 193721820.dng"


def _synthetic_gain_map_payload(gain: float, pv: int = 2, ph: int = 2) -> bytes:
    head = struct.pack(
        ">4L2L2L2L", 0, 0, 8, 8, 0, 1, 1, 1, pv, ph
    ) + struct.pack(">4d", 1.0, 1.0, 0.0, 0.0) + struct.pack(">L", 1)
    gains = struct.pack(f">{pv * ph}f", *([gain] * (pv * ph)))
    return head + gains


class GainMapParserTests(unittest.TestCase):
    def test_synthetic_payload_round_trips(self) -> None:
        m = _parse_gain_map_payload(_synthetic_gain_map_payload(1.25))
        self.assertIsNotNone(m)
        self.assertEqual((m.points_v, m.points_h, m.map_planes), (2, 2, 1))
        np.testing.assert_allclose(np.asarray(m.gains), 1.25)

    def test_truncated_payload_is_refused(self) -> None:
        self.assertIsNone(_parse_gain_map_payload(b"\x00" * 40))

    def test_uniform_gain_applies_around_black(self) -> None:
        m = _parse_gain_map_payload(_synthetic_gain_map_payload(2.0))
        img = np.full((8, 8), 300, dtype=np.uint16)
        raw = SimpleNamespace(
            raw_image_visible=img,
            raw_colors_visible=np.zeros((8, 8), dtype=np.uint8),
        )
        _apply_gain_maps_mosaic(raw, [m], [100.0], 4000)
        # (300 - 100) * 2 + 100 = 500, uniformly
        np.testing.assert_allclose(img, 500)

    def test_gains_clip_at_white_level(self) -> None:
        m = _parse_gain_map_payload(_synthetic_gain_map_payload(10.0))
        img = np.full((8, 8), 3900, dtype=np.uint16)
        raw = SimpleNamespace(
            raw_image_visible=img,
            raw_colors_visible=np.zeros((8, 8), dtype=np.uint8),
        )
        _apply_gain_maps_mosaic(raw, [m], [0.0], 4000)
        np.testing.assert_allclose(img, 4000)


class VignetteTests(unittest.TestCase):
    def test_correction_commutes_with_all_libraw_orientations(self) -> None:
        v = DngVignetteRadial(k=(1.0, -0.1, 0.05, 0.0, 0.0), cx_hat=0.25, cy_hat=0.6)
        rng = np.random.default_rng(38)
        for shape in ((63, 97, 3), (32, 49, 3)):
            raw = rng.integers(100, 12000, shape, dtype=np.uint16)
            corrected = _apply_vignette_render(raw.copy(), v)
            for flip in range(8):
                with self.subTest(shape=shape, flip=flip):
                    oriented = _orient_like_libraw(raw.copy(), flip)
                    out = _apply_vignette_render(oriented, v, flip)
                    self.assertIs(out, oriented)
                    np.testing.assert_array_equal(out, _orient_like_libraw(corrected, flip))

    def test_invalid_orientation_does_not_mutate_input(self) -> None:
        img = np.ones((10, 12, 3), dtype=np.uint16)
        v = DngVignetteRadial(k=(1.0, 0.0, 0.0, 0.0, 0.0), cx_hat=0.3, cy_hat=0.4)
        with self.assertRaises(ValueError):
            _apply_vignette_render(img, v, 8)
        np.testing.assert_array_equal(img, 1)

    def test_radial_gain_grows_toward_corners_and_centre_is_unity(self) -> None:
        v = DngVignetteRadial(k=(1.0, 0.0, 0.0, 0.0, 0.0), cx_hat=0.5, cy_hat=0.5)
        img = np.full((64, 96, 3), 1000, dtype=np.uint16)
        out = _apply_vignette_render(img, v)
        centre = float(out[32, 48, 0])
        corner = float(out[0, 0, 0])
        self.assertLess(abs(centre - 1000), 15)  # r~0 -> g~1
        self.assertGreater(corner, 1900)  # r/m -> 1 at true corner, g -> 2


@unittest.skipUnless(SAMPLE_FP.is_file() and SAMPLE_IPHONE.is_file(), "samples unavailable")
class RealFileShadingTests(unittest.TestCase):
    def test_each_camera_reports_its_correction_form(self) -> None:
        fp_ops = read_dng_shading_ops(SAMPLE_FP)
        ip_ops = read_dng_shading_ops(SAMPLE_IPHONE)
        self.assertEqual(len(fp_ops["gain_maps"]), 1)
        self.assertIsNone(fp_ops["vignette"])
        self.assertEqual(len(ip_ops["gain_maps"]), 0)
        self.assertIsNotNone(ip_ops["vignette"])

    def test_bundle_records_applied_correction(self) -> None:
        from dngscan.raw_io import load_raw

        fp = load_raw(SAMPLE_FP, scene_half_size=True)
        self.assertEqual(fp.lens_shading, "gainmap")


if __name__ == "__main__":
    unittest.main()
