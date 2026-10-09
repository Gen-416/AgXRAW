# SPDX-License-Identifier: GPL-3.0-or-later
"""SDR HEIF reaches its encoder without an earlier 8-bit quantization."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan.color import encode_display_linear, srgb_decode
from dngscan.delivery import resolve_delivery_profile
from dngscan.export import export_jpeg
from dngscan.render import (finalize_output_linear, render_output_encoded_float,
                            render_output_linear)


def _bundle_and_plan(*args, **kwargs):
    from tests.test_stream_render import StreamRenderTest
    return StreamRenderTest()._bundle_and_plan(*args, **kwargs)


class SdrFloatRenderTests(unittest.TestCase):
    def test_float_master_applies_output_transfer_once_without_quantization(self):
        encoded = np.broadcast_to(np.linspace(0, 1, 1024, dtype=np.float32)[None, :, None],
                                  (2, 1024, 3)).copy()
        linear = srgb_decode(encoded)
        with patch("dngscan.render.render_output_linear", return_value=linear) as render:
            actual = render_output_encoded_float(None, None, "p3")
        self.assertEqual(render.call_count, 1)
        self.assertIs(actual, linear)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, encoded, atol=1.2e-7, rtol=0)
        self.assertEqual(np.unique(np.rint(actual * 1023)).size, 1024)
        with patch("dngscan.render.render_output_linear",
                   return_value=np.full((1, 1, 3), .18, np.float32)):
            middle = render_output_encoded_float(None, None)
        self.assertEqual(int(np.rint(middle[0, 0, 0] * 1023)), 472)

    def test_real_formation_matches_linear_reference_for_both_gamuts_and_cores(self):
        bundle, plan = _bundle_and_plan(41, 53, seed=89)
        for fast in ("0", "1"):
            from dngscan import _fast
            if fast == "1" and not _fast.available():
                continue
            for gamut in ("srgb", "p3"):
                for core in ("agx", "neutral", "lum"):
                    with self.subTest(fast=fast, gamut=gamut, core=core), \
                            patch.dict(os.environ, DNGSCAN_FAST=fast):
                        selected = replace(plan, tone_core=core)
                        expected = encode_display_linear(
                            render_output_linear(bundle, object(), gamut, selected), gamut)
                        actual = render_output_encoded_float(bundle, object(), gamut, selected)
                        np.testing.assert_array_equal(actual, expected)
                        self.assertTrue(np.isfinite(actual).all())
                        self.assertGreaterEqual(float(actual.min()), 0.)
                        self.assertLessEqual(float(actual.max()), 1.)

    def test_dense_scene_gradient_survives_to_more_than_256_encoder_levels(self):
        bundle, plan = _bundle_and_plan(8, 4096)
        bundle.scene_rec2020_render[:] = np.linspace(0, 65535, 4096, dtype=np.uint16)[None, :, None]
        actual = render_output_encoded_float(bundle, object(), "srgb", plan)
        # The DRT need not occupy every output code; it must preserve levels
        # beyond the earlier 256-code handoff for this smooth scene gradient.
        self.assertGreater(np.unique(np.rint(actual[..., 1] * 1023)).size, 256)

    def test_owned_linear_master_can_be_finalized_without_another_raster(self):
        source = np.random.default_rng(41).uniform(-.2, 1.3, (17, 31, 3)).astype(np.float32)
        reference = finalize_output_linear(source, "p3", "optic_warm_cyan", .7)
        owned = source.copy()
        actual = finalize_output_linear(owned, "p3", "optic_warm_cyan", .7, out=owned)
        self.assertTrue(np.shares_memory(actual, owned))
        np.testing.assert_array_equal(actual, reference)

    def test_all_10bit_sdr_heif_profiles_use_float_and_8bit_keeps_bytes(self):
        floating = np.linspace(0, 1, 1024 * 3, dtype=np.float32).reshape(1, 1024, 3)
        byte = np.zeros((1, 1024, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            for name in ("auto", "share", "archive"):
                for depth in (8, 10):
                    with self.subTest(profile=name, depth=depth):
                        profile = replace(resolve_delivery_profile(name, container="heic"),
                                          heif_bit_depth=depth)
                        with patch("dngscan.export.render_output_encoded_float", return_value=floating) as f, \
                                patch("dngscan.export.render_output_u8", return_value=byte) as u, \
                                patch("dngscan.heif_delivery.save_sdr_heif", return_value={}) as save:
                            export_jpeg(Path("source.dng"), Path(directory) / "out.heic", profile.quality,
                                        None, None, output_format="sdr-heic", delivery=profile)
                        self.assertIs(save.call_args.args[0], floating if depth == 10 else byte)
                        self.assertEqual((f.call_count, u.call_count), (1, 0) if depth == 10 else (0, 1))


if __name__ == "__main__":
    unittest.main()
