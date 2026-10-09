# SPDX-License-Identifier: GPL-3.0-or-later
"""Uncertified decoder support is qualification, not measured RGB clipping."""
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from dngscan import decoder_loss, raw_io
from dngscan._deps import rawpy
from dngscan.scene_reference import reliable_reference_samples
from tests.test_pipeline_corrections import write_sensor_dng


def _red_point(value):
    pixels = np.full((128, 128), 1000, np.uint16)
    pixels[64, 64] = value
    return pixels


class UnknownSupportContractTests(unittest.TestCase):
    def test_unknown_support_has_reason_but_no_fabricated_spatial_mask(self):
        colors = np.tile([[0, 1], [3, 2]], (16, 16)).astype(np.uint8)
        loss = np.zeros(colors.shape, np.uint8)
        loss[16, 16] = 1
        for mode, algorithm, half_size in (
                ("clip", "DHT", False), ("clip", "unknown", False),
                ("reconstruct", "AHD", False), ("reconstruct", "DHT", True)):
            with self.subTest(mode=mode, algorithm=algorithm, half_size=half_size):
                mask, reason = decoder_loss.propagate_mosaic_loss(
                    loss, colors, "RGBG", (32, 32), half_size=half_size,
                    demosaic=algorithm, highlight=mode, is_bayer=True)
                self.assertIsNone(mask)
                self.assertTrue(decoder_loss.support_is_untrusted(reason))
        # Missing loss evidence and audited local support remain distinct.
        mask, reason = decoder_loss.propagate_mosaic_loss(
            None, colors, "RGBG", (32, 32), half_size=False,
            demosaic="DHT", highlight="clip", is_bayer=True)
        self.assertIsNone(mask)
        self.assertIsNone(reason)
        self.assertFalse(decoder_loss.support_is_untrusted(reason))

    def test_audited_local_support_remains_spatial_and_qualified(self):
        colors = np.tile([[0, 1], [3, 2]], (16, 16)).astype(np.uint8)
        loss = np.zeros(colors.shape, np.uint8)
        loss[16, 16] = 1
        for algorithm, half_size, shape in (("AHD", False, (32, 32)),
                                            ("DHT", True, (16, 16))):
            mask, reason = decoder_loss.propagate_mosaic_loss(
                loss, colors, "RGBG", shape, half_size=half_size,
                demosaic=algorithm, highlight="clip", is_bayer=True)
            self.assertFalse(decoder_loss.support_is_untrusted(reason))
            self.assertGreater(np.count_nonzero(mask), 0)
            self.assertLess(np.count_nonzero(mask), mask.size)
            np.testing.assert_array_equal(mask[0, 0], 0)

    def test_untrusted_reference_stays_present_empty_without_a_spatial_mask(self):
        # Early rejection must not infer trust from processing_loss=None or
        # require full-resolution sensor/scene allocations to reject evidence.
        for reason in ("global-conservative: uninstrumented DHT",
                       "global-conservative: spatial highlight reconstruction"):
            reference, percent = reliable_reference_samples(
                None, None, 1., None, SimpleNamespace(loss_support=reason))
            self.assertEqual(reference.shape, (0, 3))
            self.assertEqual(reference.dtype, np.float32)
            self.assertEqual(percent, 0.)

    def test_actual_dht_overflow_preserves_known_local_masks_and_separate_flag(self):
        dht = getattr(rawpy.DemosaicAlgorithm, "DHT", None)
        if dht is None or not dht.isSupported:
            self.skipTest("LibRaw build does not support DHT")
        with TemporaryDirectory() as td:
            path = Path(td) / "wb.dng"
            bundles = []
            for value in (2000, 3000):
                write_sensor_dng(path, signal=_red_point(value), neutral=(.5, 1., 1.))
                bundles.append(raw_io.load_raw(path, demosaic="dht"))
            self.assertFalse(bundles[0].scene_loss_support_untrusted)
            self.assertTrue(bundles[1].scene_loss_support_untrusted)
            self.assertTrue(bundles[1].noise_decode["loss_support_untrusted"])
            self.assertEqual(bundles[1].scene_reliability_source, "decoder-support-untrusted")
            np.testing.assert_array_equal(bundles[0].scene_rec2020_render[16, 16],
                                          bundles[1].scene_rec2020_render[16, 16])
            for bundle in bundles:
                np.testing.assert_array_equal(bundle.clip_masks[16, 16], 0)
                if bundle.processing_clip_masks is not None:
                    np.testing.assert_array_equal(bundle.processing_clip_masks[16, 16], 0)
                self.assertLess(bundle.scene_processing_loss_pct, 1.)
                # Immutable RAW saturation facts do not inherit decoder distrust.
                self.assertTrue(np.all(bundle.raw_image < bundle.white_level))
            # The known uint16 ceiling still contributes local evidence.
            self.assertGreater(np.count_nonzero(bundles[1].processing_clip_masks), 0)

    def test_actual_gainmap_reconstruction_is_untrusted_without_global_colour_mask(self):
        gain = struct.pack(">10L4dL4f", 0, 0, 128, 128, 0, 1, 1, 1, 2, 2,
                           1., 1., 0., 0., 1, *([1.384] * 4))
        with TemporaryDirectory() as td:
            path = Path(td) / "gainmap.dng"
            for value, expected in ((2000, False), (3000, True)):
                write_sensor_dng(path, signal=_red_point(value),
                                 opcodes={51009: [(9, gain)]})
                bundle = raw_io.load_raw(path, scene_highlight_mode="reconstruct")
                self.assertEqual(bundle.scene_loss_support_untrusted, expected)
                self.assertEqual(bundle.noise_decode["loss_support_untrusted"], expected)
                np.testing.assert_array_equal(bundle.clip_masks[16, 16], 0)
                if bundle.processing_clip_masks is not None:
                    np.testing.assert_array_equal(bundle.processing_clip_masks[16, 16], 0)
                self.assertLess(bundle.scene_processing_loss_pct, 1.)
                self.assertTrue(np.all(bundle.raw_image < bundle.white_level))


if __name__ == "__main__":
    unittest.main()
