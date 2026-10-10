# SPDX-License-Identifier: GPL-3.0-or-later
"""Late crop must preserve evidence of inputs that formed retained RGB.

The source clip location is outside the output. Colour-loss masks may therefore
remain empty, while demosaicing or a later warp can still make retained output
depend on that source. Only the independent reliability exclusion follows that
dependency; Bayer half-size without a warp supplies the no-dependency control.
"""
from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

from dngscan._deps import np, rawpy
from dngscan.analysis import analyze
from dngscan.guidance import build_raw_guidance_maps
from dngscan.raw_io import load_raw
from dngscan.scene_reference import reliable_reference_samples
from dngscan.tone import reliable_scene_ev_selection
from tests.test_odd_raw_clip_masks import write_odd_dng
from tests.test_pipeline_corrections import write_sensor_dng
from tests.test_readout_contract import edit_ifd


@unittest.skipIf(rawpy is None, "rawpy unavailable")
class SensorDependencyCropTests(unittest.TestCase):
    def pair(self, path, *, algorithm, half=False, point=(124, 124), **settings):
        bundles = []
        for value in (3000, 4095):
            pixels = np.full((125, 127), 1000, np.uint16)
            pixels[point] = value
            write_odd_dng(path, pixels, crop=(126, 124), **settings)
            bundles.append(load_raw(path, demosaic=algorithm, scene_half_size=half))
        return bundles

    def selection(self, bundle):
        result, _, _ = analyze(bundle, 4)
        _, _, reliable, _ = reliable_scene_ev_selection(bundle, result)
        return result, reliable.reshape(bundle.scene_rec2020_render.shape[:2])

    def assert_no_color_loss_evidence(self, bundle, result):
        # The raw source is truly saturated, but sits outside the visible
        # sensor cells. Dependency qualification must not fabricate clipping
        # in retained colour channels or open the path-to-white permission.
        self.assertGreater(max(result.clip_pct.values()), 0.)
        self.assertFalse(np.any(bundle.clip_masks))
        self.assertIsNone(bundle.processing_clip_masks)
        maps = build_raw_guidance_maps(bundle, result)
        self.assertFalse(np.any(maps.clip_class))
        self.assertFalse(np.any(maps.raw_permission))
        self.assertTrue(np.all(maps.headroom > .7))

    def test_dht_full_cropped_source_withdraws_reliability_without_color_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-dht.dng"
            for point in ((124, 124), (122, 126)):
                with self.subTest(point=point):
                    before, clipped = self.pair(path, algorithm="dht", point=point)
                    self.assertEqual(clipped.noise_decode["demosaic_algorithm"], "DHT")
                    changed = np.any(before.scene_rec2020_render != clipped.scene_rec2020_render, axis=2)
                    self.assertGreater(int(np.count_nonzero(changed)), 0)
                    self.assertFalse(before.scene_loss_support_untrusted)
                    self.assertTrue(clipped.scene_loss_support_untrusted)
                    self.assertIsNone(clipped.scene_reliability_exclusion)
                    result, reliable = self.selection(clipped)
                    self.assertFalse(np.any(reliable))
                    self.assert_no_color_loss_evidence(clipped, result)

    def test_ahd_full_cropped_source_has_local_dependency_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-ahd.dng"
            for point in ((124, 124), (122, 126)):
                with self.subTest(point=point):
                    before, clipped = self.pair(path, algorithm="ahd", point=point)
                    self.assertEqual(clipped.noise_decode["demosaic_algorithm"], "AHD")
                    changed = np.any(before.scene_rec2020_render != clipped.scene_rec2020_render, axis=2)
                    self.assertGreater(int(np.count_nonzero(changed)), 0)
                    self.assertFalse(clipped.scene_loss_support_untrusted)
                    exclusion = clipped.scene_reliability_exclusion
                    self.assertEqual(exclusion.dtype, np.uint8)
                    self.assertEqual(exclusion.shape, changed.shape)
                    self.assertTrue(np.all(exclusion[changed] == 1))
                    self.assertLess(float(np.mean(exclusion)), .05)
                    result, reliable = self.selection(clipped)
                    self.assertFalse(np.any(reliable[changed]))
                    self.assertTrue(np.any(reliable))
                    self.assert_no_color_loss_evidence(clipped, result)

    def test_half_without_warp_discards_the_source_and_keeps_reliability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-half.dng"
            for algorithm in ("auto", "dht", "ahd"):
                for point in ((124, 124), (122, 126)):
                    with self.subTest(algorithm=algorithm, point=point):
                        before, clipped = self.pair(path, algorithm=algorithm, half=True, point=point)
                        self.assertEqual(clipped.noise_decode["demosaic_algorithm"], "half-size")
                        np.testing.assert_array_equal(before.scene_rec2020_render, clipped.scene_rec2020_render)
                        self.assertFalse(clipped.scene_loss_support_untrusted)
                        exclusion = clipped.scene_reliability_exclusion
                        self.assertTrue(exclusion is None or not np.any(exclusion))
                        result, reliable = self.selection(clipped)
                        self.assertTrue(np.all(reliable))
                        self.assert_no_color_loss_evidence(clipped, result)

    def test_source_dependency_follows_scale_warp_and_orientation_before_crop(self):
        configurations = (
            (6, None, False),
            (2, (1.03, 1.), False),
            (8, (1., 1.03), False),
            (4, None, True),
            (6, (1.03, 1.), True),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-geometry.dng"
            for half in (False, True):
                for orientation, scale, warp in configurations:
                    with self.subTest(half=half, orientation=orientation, scale=scale, warp=warp):
                        before, clipped = self.pair(path, algorithm="ahd", half=half,
                                                   orientation=orientation, scale=scale, warp=warp)
                        changed = np.any(before.scene_rec2020_render != clipped.scene_rec2020_render, axis=2)
                        _, reliable = self.selection(clipped)
                        self.assertFalse(clipped.scene_loss_support_untrusted)
                        exclusion = clipped.scene_reliability_exclusion
                        if not half or warp:
                            self.assertTrue(np.any(changed))
                            self.assertEqual(exclusion.dtype, np.uint8)
                            self.assertEqual(exclusion.shape, changed.shape)
                            self.assertTrue(np.all(exclusion[changed] == 1))
                            self.assertFalse(np.any(reliable[changed]))
                            self.assertTrue(np.any(reliable))
                        else:
                            self.assertFalse(np.any(changed))
                            if scale is None:
                                # Identity half-size has an exact discarded
                                # cell here; DefaultScale instead uses a
                                # conservative interpolation footprint. Equal
                                # decoded values alone cannot certify absence
                                # of dependencies in that wider footprint.
                                self.assertTrue(exclusion is None or not np.any(exclusion))
                                self.assertTrue(np.all(reliable))
                            elif exclusion is not None:
                                self.assertEqual(exclusion.shape, changed.shape)
                                self.assertFalse(np.any(reliable[exclusion != 0]))
                                self.assertLess(float(np.mean(exclusion)), .05)
                            self.assertTrue(np.any(reliable))

    def test_terminal_trim_and_actual_post_orientation_box_share_dependency_support(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-trim.dng"
            for flip, orientation in enumerate((1, 2, 4, 3, 5, 8, 6, 7)):
                pairs = {}
                for half in (False, True):
                    bundles = []
                    for value in (3000, 4095):
                        image = np.full((128, 128), 1000, np.uint16)
                        image[120, 114] = value  # Just outside the terminal trim.
                        write_sensor_dng(path, signal=image, opcodes={
                            51022: [(6, struct.pack(">4l", 11, 13, 120, 116))]})
                        edit_ifd(path, {274: (3, 1, struct.pack("<H", orientation))})
                        bundles.append(load_raw(path, demosaic="ahd", scene_half_size=half))
                    pairs[half] = bundles
                    before, clipped = bundles
                    changed = np.any(before.scene_rec2020_render != clipped.scene_rec2020_render, axis=2)
                    _, reliable = self.selection(clipped)
                    exclusion = clipped.scene_reliability_exclusion
                    with self.subTest(flip=flip, half=half):
                        self.assertFalse(clipped.scene_loss_support_untrusted)
                        self.assertEqual(exclusion.shape, changed.shape)
                        self.assertTrue(np.all(exclusion[changed] == 1))
                        self.assertFalse(np.any(reliable[changed]))
                        self.assertTrue(np.any(reliable))
                        # The affected last row is discarded only when it
                        # remains the last axis after orientation and box crop.
                        self.assertEqual(bool(np.any(changed)), not half or bool(flip & 2))
                full, half = pairs[False][1], pairs[True][1]
                h, w = half.scene_rec2020_render.shape[:2]
                expected = full.scene_reliability_exclusion[:2*h, :2*w].reshape(
                    h, 2, w, 2).max(axis=(1, 3))
                np.testing.assert_array_equal(half.scene_reliability_exclusion, expected)

    def test_real_libraw_reference_uses_source_qualification_in_its_own_geometry(self):
        from dngscan.raw_io import _decode_corrected_libraw

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-reference.dng"
            for algorithm in ("ahd", "dht"):
                with self.subTest(algorithm=algorithm):
                    _, clipped = self.pair(path, algorithm=algorithm)
                    chosen = getattr(rawpy.DemosaicAlgorithm, algorithm.upper())
                    with rawpy.imread(str(path)) as raw:
                        scene, loss, recipe, _ = _decode_corrected_libraw(
                            raw, path, clipped.evidence, "clip", False, chosen)
                    np.testing.assert_array_equal(scene, clipped.scene_rec2020_render)
                    samples, percent = reliable_reference_samples(
                        clipped.evidence, scene, clipped.scene_scale, loss, recipe)
                    if algorithm == "dht":
                        self.assertEqual(samples.shape, (0, 3))
                        self.assertEqual(percent, 0.)
                    else:
                        retained = recipe.scene_reliability_exclusion == 0
                        self.assertGreater(np.count_nonzero(retained), 256)
                        expected = scene[retained].astype(np.float32) / np.float32(clipped.scene_scale)
                        np.testing.assert_array_equal(samples, expected)
                        self.assertAlmostEqual(percent, float(np.mean(retained)) * 100.)

    def test_fullwell_replay_cannot_reuse_stale_bound_sample_permissions(self):
        from dngscan.prepared_sample import PreparedSceneSample
        from dngscan.raw_io import refresh_clip_masks_from_fullwell

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-bound-replay.dng"
            point = (64, 65)
            pixels = np.full((125, 127), 1000, np.uint16)
            pixels[124, 124] = 4095  # Existing exclusion outside the crop.
            pixels[point] = 3000    # A newly resolved green endpoint below it.
            write_odd_dng(path, pixels, crop=(126, 124))
            bundle = load_raw(path, demosaic="ahd")
            analysis, _, _ = analyze(bundle, 4)
            bound = PreparedSceneSample.from_bundle(bundle).bind(bundle)
            _, _, before, _ = reliable_scene_ev_selection(bound, analysis)
            self.assertTrue(before.reshape(bundle.scene_rec2020_render.shape[:2])[point])
            levels = dict(analysis.channel_fullwell)
            levels[1] = 3000
            self.assertTrue(refresh_clip_masks_from_fullwell(bound, levels))
            self.assertEqual(int(bound.scene_reliability_exclusion[point]), 1)
            _, _, after, _ = reliable_scene_ev_selection(bound, analysis)
            self.assertFalse(after.reshape(bundle.scene_rec2020_render.shape[:2])[point])


if __name__ == "__main__":
    unittest.main()
