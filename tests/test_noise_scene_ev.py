# SPDX-License-Identifier: GPL-3.0-or-later
"""File decoding exposure translates noise evidence into the rendered scene."""
from __future__ import annotations

import math
import json
from dataclasses import replace
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from dngscan.analysis import analyze, noise_floor_ev_estimate, reanalyze_balanced_scene, snr_ev_coordinates
from dngscan.gui.preview_cache import _analysis_from_json, _analysis_to_json
from dngscan.hdr_agx_plan import compile_hdr_agx_plan
from dngscan.raw_io import load_raw
from dngscan.scene_scale import with_intent_exposure
from dngscan.tone import build_color_geometry_plan, build_render_plan, scene_intent_rec2020
from tests.test_pipeline_corrections import write_sensor_dng


def write_scene_noise_dng(path, baseline):
    """Write actual TIFF tags without patching metadata or LibRaw decoding."""
    pixels = np.ones((128, 128), np.uint16)
    pixels[80:120, 80:120] = 3500
    write_sensor_dng(path, signal=pixels)
    data = bytearray(path.read_bytes())
    old_ifd, = struct.unpack_from("<L", data, 4)
    count, = struct.unpack_from("<H", data, old_ifd)
    entries = {}
    for index in range(count):
        entry = bytes(data[old_ifd + 2 + 12 * index:old_ifd + 14 + 12 * index])
        tag, = struct.unpack_from("<H", entry)
        entries[tag] = entry
    replacements = {
        34855: (3, 1, struct.pack("<H", 200)),
        50730: (10, 1, struct.pack("<ll", int(baseline * 1000), 1000)),
        51041: (12, 2, struct.pack("<dd", .0001, .000004)),
    }
    offset = len(data)
    payload_offset = offset + 2 + 12 * len(set(entries) | set(replacements)) + 4
    payload = bytearray()
    for tag, (kind, size, value) in replacements.items():
        field = (value.ljust(4, b"\0") if len(value) <= 4
                 else struct.pack("<L", payload_offset + len(payload)))
        entries[tag] = struct.pack("<HHL", tag, kind, size) + field
        if len(value) > 4:
            payload.extend(value)
    data[4:8] = struct.pack("<L", offset)
    data.extend(struct.pack("<H", len(entries)))
    data.extend(b"".join(entries[tag] for tag in sorted(entries)))
    data.extend(struct.pack("<L", 0))
    data.extend(payload)
    path.write_bytes(data)


class NoiseSceneEvPipelineTests(unittest.TestCase):
    def _decoded_pair(self, directory, *, highlight="clip", half=False):
        results = []
        for baseline in (0., 1.):
            path = Path(directory) / f"baseline-{baseline}.dng"
            write_scene_noise_dng(path, baseline)
            bundle = load_raw(path, scene_highlight_mode=highlight, scene_half_size=half)
            analysis, _, _ = analyze(bundle, 4)
            results.append((bundle, analysis))
        return results

    def test_fixed_baseline_moves_noise_and_tone_coordinates_together(self):
        for highlight in ("clip", "blend", "reconstruct"):
            for half in (False, True):
                with self.subTest(highlight=highlight, half=half):
                    self._check_fixed_baseline(highlight, half)

    def _check_fixed_baseline(self, highlight, half):
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for bundle, analysis in self._decoded_pair(directory, highlight=highlight, half=half):
                plans = {
                    mode: build_render_plan(bundle, analysis, "agx", endpoint_mode=mode)
                    for mode in ("adaptive", "evidence")
                }
                hdr = {mode: compile_hdr_agx_plan(plan, analysis=analysis)
                       for mode, plan in plans.items()}
                results.append((bundle, analysis, plans, hdr))
            (bundle0, analysis0, plans0, hdr0), (bundle1, analysis1, plans1, hdr1) = results
            np.testing.assert_array_equal(bundle0.raw_image, bundle1.raw_image)
            np.testing.assert_array_equal(bundle0.scene_rec2020_render, bundle1.scene_rec2020_render)
            np.testing.assert_array_equal(scene_intent_rec2020(bundle1.scene_rec2020_render, bundle1),
                                          scene_intent_rec2020(bundle0.scene_rec2020_render, bundle0) * 2)
            self.assertEqual(analysis0.noise_model.channel_variance, analysis1.noise_model.channel_variance)
            self.assertEqual(analysis0.noise_floor, .002)
            self.assertEqual(analysis0.noise_floor, analysis1.noise_floor)
            self.assertEqual(analysis0.usable_dr_ev, analysis1.usable_dr_ev)
            for channel in analysis0.snr_curves:
                for key in ("stops", "snr_db", "count"):
                    np.testing.assert_array_equal(analysis0.snr_curves[channel][key],
                                                  analysis1.snr_curves[channel][key])
            floor0, source0 = noise_floor_ev_estimate(analysis0)
            floor1, source1 = noise_floor_ev_estimate(analysis1)
            self.assertEqual((source0, source1), ("model", "model"))
            self.assertAlmostEqual(floor0, 3 + math.log2(.002))
            self.assertAlmostEqual(floor1 - floor0, 1.)
            coords0, coords1 = snr_ev_coordinates(analysis0), snr_ev_coordinates(analysis1)
            for tier in ("snr1", "snr10", "snr20"):
                self.assertAlmostEqual(coords1[tier] - coords0[tier], 1.)
            for mode in ("adaptive", "evidence"):
                self.assertAlmostEqual(plans1[mode].tone.black_ev - plans0[mode].tone.black_ev, 1.)
                self.assertAlmostEqual(hdr1[mode].tone.black_ev - hdr0[mode].tone.black_ev, 1.)

    def test_user_ev_does_not_rewrite_fixed_noise_coordinates_or_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            _, (bundle, analysis) = self._decoded_pair(directory)
            user_bundle = with_intent_exposure(bundle, user_ev=2.)
            user_analysis, _, _ = analyze(user_bundle, 4)
            self.assertEqual(noise_floor_ev_estimate(user_analysis), noise_floor_ev_estimate(analysis))
            self.assertEqual(snr_ev_coordinates(user_analysis), snr_ev_coordinates(analysis))
            for mode in ("adaptive", "evidence"):
                plan = build_render_plan(bundle, analysis, "agx", endpoint_mode=mode)
                user_plan = build_render_plan(user_bundle, user_analysis, "agx", endpoint_mode=mode)
                self.assertEqual(plan.tone, user_plan.tone)
            ev0_bundle = with_intent_exposure(bundle, user_ev=0.)
            np.testing.assert_array_equal(scene_intent_rec2020(user_bundle.scene_rec2020_render, user_bundle),
                                          scene_intent_rec2020(ev0_bundle.scene_rec2020_render, ev0_bundle) * 4)

    def test_cached_and_balanced_analysis_preserve_coordinate_context(self):
        with tempfile.TemporaryDirectory() as directory:
            _, (bundle, analysis) = self._decoded_pair(directory)
            restored = _analysis_from_json(json.loads(json.dumps(_analysis_to_json(analysis))))
            self.assertEqual(restored.sensor_to_scene_ev_offset, 1.)
            self.assertEqual(noise_floor_ev_estimate(restored), noise_floor_ev_estimate(analysis))
            self.assertEqual(snr_ev_coordinates(restored), snr_ev_coordinates(analysis))
            balanced = reanalyze_balanced_scene(analysis, bundle)
            self.assertEqual(balanced.sensor_to_scene_ev_offset, 1.)
            self.assertEqual(snr_ev_coordinates(balanced), snr_ev_coordinates(analysis))
            legacy = _analysis_to_json(analysis)
            del legacy["sensor_to_scene_ev_offset"]
            self.assertEqual(_analysis_from_json(legacy).sensor_to_scene_ev_offset, 0.)

    def test_dr_only_fallback_uses_same_scene_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            plans, colour_floors = [], []
            for bundle, analysis in self._decoded_pair(directory):
                analysis = replace(analysis, noise_model=None, noise_floor_e=None,
                                   prior_read_noise_e=None, usable_dr_ev=float("nan"),
                                   usable_dr_eff_ev=8.)
                self.assertTrue(math.isnan(noise_floor_ev_estimate(analysis)[0]))
                plans.append(build_render_plan(bundle, analysis, "agx"))
                colour_floors.append(build_color_geometry_plan(analysis, "srgb", "gated").gated_noise_ev_floor)
            self.assertAlmostEqual(plans[1].tone.black_ev - plans[0].tone.black_ev, 1.)
            self.assertAlmostEqual(colour_floors[1] - colour_floors[0], 1.)

    def test_legacy_prior_and_frame_floors_use_recorded_offset(self):
        from tests.test_preview_cache import _analysis

        base = replace(_analysis(), noise_floor=.002, noise_floor_e=2.,
                       prior_read_noise_e=3.)
        shifted = replace(base, sensor_to_scene_ev_offset=1.)
        self.assertEqual(noise_floor_ev_estimate(base)[1], "prior")
        self.assertAlmostEqual(noise_floor_ev_estimate(shifted)[0] - noise_floor_ev_estimate(base)[0], 1.)
        for key, value in snr_ev_coordinates(base).items():
            self.assertAlmostEqual(snr_ev_coordinates(shifted)[key] - value, 1.)
        frame = replace(base, noise_floor_e=None, prior_read_noise_e=None)
        shifted_frame = replace(frame, sensor_to_scene_ev_offset=1.)
        self.assertEqual(noise_floor_ev_estimate(frame)[1], "frame")
        self.assertAlmostEqual(noise_floor_ev_estimate(shifted_frame)[0] - noise_floor_ev_estimate(frame)[0], 1.)


if __name__ == "__main__":
    unittest.main()
