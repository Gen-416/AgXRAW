# SPDX-License-Identifier: GPL-3.0-or-later
"""Encoding endpoints and LibRaw's decoder slope are distinct unit contracts."""
from __future__ import annotations

import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from dngscan.dng_opcodes import OpcodePlan
from dngscan.noise_model import model_from_prior
from dngscan.raw_io import _libraw_noise_decode, libraw_scene_scale, load_raw
from dngscan.raw_units import coding_endpoints, normalized_raw_span
from tests.test_pipeline_corrections import write_sensor_dng


class CodingEndpointTests(unittest.TestCase):
    def test_uniform_black_keeps_native_decode_with_file_encoding_span(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scalar-black.dng"
            write_sensor_dng(path, limit=.8, signal=1000, black_pattern=[[100]])
            bundle = load_raw(path, scene_half_size=True)
            self.assertIsNone(bundle.evidence.spatial_black)
            self.assertEqual(bundle.coding_white_levels, [4095.])
            self.assertEqual(bundle.coding_black_levels, [100.] * 4)
            self.assertEqual(bundle.camera_white_levels, [3296.] * 4)
            for cid in range(4):
                self.assertEqual(normalized_raw_span(bundle, cid), 3995.)
            self.assertTrue(bundle.noise_decode["supported"])

    def test_spatial_maximum_black_defines_normalized_plane_span(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spatial.dng"
            for kwargs, expected in (({"black_pattern": [[100, 200], [300, 400]]}, 400.),
                                     ({"spatial_black": True}, 64.)):
                with self.subTest(kwargs=kwargs):
                    write_sensor_dng(path, limit=.8, signal=1000, **kwargs)
                    bundle = load_raw(path, scene_half_size=True)
                    self.assertEqual(bundle.coding_white_levels, [4095.])
                    self.assertEqual(bundle.coding_black_levels, [expected])
                    self.assertIsNotNone(bundle.evidence.spatial_black)
                    for cid in range(4):
                        self.assertEqual(normalized_raw_span(bundle, cid), 4095. - expected)
                    # The endpoint is known; the spatial decoder covariance
                    # is still unestablished and must not become active.
                    self.assertFalse(bundle.noise_decode["supported"])
                    self.assertIn("spatial-black", bundle.noise_decode["reason"])

    def test_encoding_white_remains_in_unpacked_linearization_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linearized.dng"
            write_sensor_dng(path, lut=True, limit=.8, signal=1000)
            bundle = load_raw(path, scene_half_size=True)
            np.testing.assert_array_equal(bundle.raw_image, np.full((128, 128), 2000, np.uint16))
            self.assertEqual(bundle.white_level, 8190)
            self.assertEqual(bundle.coding_white_levels, [8190.])
            self.assertEqual(bundle.camera_white_levels, [6552.] * 4)
            for cid in range(4):
                self.assertEqual(normalized_raw_span(bundle, cid), 8190.)
            self.assertTrue(bundle.noise_decode["supported"])

    def test_missing_dng_white_uses_encoding_maximum_not_response_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-white.dng"
            write_sensor_dng(path)
            data = bytearray(path.read_bytes())
            ifd, = struct.unpack_from("<L", data, 4)
            count, = struct.unpack_from("<H", data, ifd)
            for index in range(count):
                offset = ifd + 2 + 12 * index
                tag, = struct.unpack_from("<H", data, offset)
                if tag == 50717:
                    struct.pack_into("<H", data, offset, 65000)
            path.write_bytes(data)
            white, black = coding_endpoints(path, 65535, [32767.] * 4, [100.] * 4)
            self.assertEqual(white, [65535.])
            self.assertEqual(black, [100.] * 4)

    def test_non_dng_preserves_per_channel_libraw_endpoint_convention(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor.raw"
            path.write_bytes(b"not a TIFF container")
            whites = [4000., 4010., 4020., 4010.]
            blacks = [100., 110., 120., 110.]
            actual = coding_endpoints(path, 4095, whites, blacks)
            self.assertEqual(actual, (whites, blacks))
            self.assertIsNot(actual[0], whites)
            self.assertIsNot(actual[1], blacks)
            self.assertEqual(coding_endpoints(path, 4095, None, blacks), ([4095.], blacks))

    def test_unequal_pedestals_keep_decoder_and_normalized_scales_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix-contract.dng"
            write_sensor_dng(path)
            # The minimum is deliberately the second green: LibRaw uses
            # every channel when separating its common black pedestal.
            black = [100., 200., 300., 50.]
            wb = [2., 1., 1.5, 1.]
            evidence = SimpleNamespace(
                path=path, raw_image=np.zeros((128, 128), np.uint16),
                raw_pattern=[[0, 1], [3, 2]], camera_wb=wb,
                spatial_black=None, black_levels=black, white_level=4095,
                coding_white_levels=[4095.], coding_black_levels=black,
                camera_white_levels=[3000.] * 4, color_desc="RGBG", shot_iso=200,
                orientation_flip=0,
            )
            raw = SimpleNamespace(color_matrix=np.eye(3, 4), rgb_xyz_matrix=np.eye(4, 3))
            recipe = OpcodePlan(crop=(0., 0., 128., 128.), white_levels=(4095.,))
            # This small source-contract fixture isolates unequal channel
            # pedestals without bypassing the production spatial-black guard.
            # Pinned LibRaw scales every plane by maximum - common MIN black.
            rec = np.asarray(((.627452, .329249, .043299),
                              (.069109, .919531, .011360),
                              (.016398, .088030, .895572)), np.float32)
            spans = 4095. - np.asarray(black[:3])
            expected = rec.astype(np.float64) @ np.diag(np.asarray(wb[:3]) * spans / (4095. - 50.))
            for mode in ("clip", "blend", "reconstruct"):
                with self.subTest(mode=mode):
                    scale = libraw_scene_scale(65535., mode, wb)
                    descriptor = _libraw_noise_decode(raw, evidence, recipe, mode, scale, True)
                    self.assertTrue(descriptor["supported"])
                    np.testing.assert_allclose(descriptor["normalized_raw_to_scene"], expected, rtol=1e-12)
            prior = {
                "id": "pedestal calibration", "base_iso": 200,
                "fwc_e": 7790., "reference_dn_range": 3895.,
                "gain_log2iso_log2epd": [[math.log2(200), 1.]],
                "read_noise_log2iso_log2e": [[math.log2(200), math.log2(3.)]],
            }
            model = model_from_prior(evidence, {cid: 4095. for cid in range(4)}, prior)
            self.assertEqual(model.status, "valid")
            for cid, label in enumerate(("R", "G1", "B", "G2")):
                span = 4095. - black[cid]
                self.assertEqual(normalized_raw_span(evidence, cid), span)
                a, b = model.coefficients(label)
                self.assertAlmostEqual(a, 1 / (2. * span), places=14)
                self.assertAlmostEqual(b, (3. / (2. * span)) ** 2, places=14)


if __name__ == "__main__":
    unittest.main()
