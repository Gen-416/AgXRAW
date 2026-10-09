# SPDX-License-Identifier: GPL-3.0-or-later
"""Real decoder uncertainty withdraws authority without colouring distant pixels."""
from dataclasses import replace
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dngscan._deps import rawpy
from dngscan.analysis import analyze
from dngscan.hdr_agx import render_ultrahdr_agx_pair, scene_render_to_hdr_display_linear
from dngscan.hdr_agx_plan import compile_hdr_agx_plan
from dngscan.noise_propagation import calibrated_chroma_variance
from dngscan.raw_io import load_raw
from dngscan.render import _prepare_chroma_nr_map, render_output_encoded_float, render_output_u8
from dngscan.tone import build_render_plan, reliable_scene_ev_selection
from tests.test_pipeline_corrections import write_sensor_dng


def _add_noise_profile(path):
    """Append an actual DNG NoiseProfile, retaining sample/metadata offsets."""
    data = bytearray(path.read_bytes())
    old_ifd, = struct.unpack_from("<L", data, 4)
    count, = struct.unpack_from("<H", data, old_ifd)
    entries = [bytes(data[old_ifd + 2 + 12 * i:old_ifd + 14 + 12 * i])
               for i in range(count)]
    offset = len(data)
    payload_offset = offset + 2 + 12 * (count + 1) + 4
    entries.append(struct.pack("<HHLL", 51041, 12, 2, payload_offset))
    entries.sort(key=lambda entry: struct.unpack_from("<H", entry)[0])
    data[4:8] = struct.pack("<L", offset)
    data.extend(struct.pack("<H", len(entries)) + b"".join(entries) + bytes(4))
    data.extend(struct.pack("<dd", .0001, .000004))
    path.write_bytes(data)


class UnknownSupportRenderTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        env = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "calibration"))
        env.start()
        self.addCleanup(env.stop)

    def decoded_pair(self, kind):
        if kind == "dht-wb":
            dht = getattr(rawpy.DemosaicAlgorithm, "DHT", None)
            if dht is None or not dht.isSupported:
                self.skipTest("LibRaw build does not support DHT")
            options, neutral, opcodes = {"demosaic": "dht"}, (.5, 1., 1.), None
        else:
            gain = struct.pack(">10L4dL4f", 0, 0, 128, 128, 0, 1, 1, 1, 2, 2,
                               1., 1., 0., 0., 1, *([1.384] * 4))
            options = {"scene_highlight_mode": "reconstruct"}
            neutral, opcodes = (.5, 1., .75), {51009: [(9, gain)]}
        bundles, analyses = [], []
        for value in (2000, 3000):
            pixels = np.full((128, 128), 1000, np.uint16)
            pixels[64, 64] = value
            path = self.root / f"{kind}-{value}.dng"
            write_sensor_dng(path, signal=pixels, neutral=neutral, opcodes=opcodes)
            _add_noise_profile(path)
            bundle = load_raw(path, **options)
            analysis, _, _ = analyze(bundle, 4, diagnostics=False)
            bundles.append(bundle)
            analyses.append(analysis)
        self.assertFalse(bundles[0].scene_loss_support_untrusted)
        self.assertTrue(bundles[1].scene_loss_support_untrusted)
        for bundle in bundles:
            self.assertTrue(np.all(bundle.raw_image < bundle.white_level))
        for analysis in analyses:
            self.assertEqual(analysis.noise_model.status, "valid")
            self.assertEqual(analysis.noise_model.source, "DNG NoiseProfile")
        self.assertEqual(analyses[0].noise_model.channel_variance,
                         analyses[1].noise_model.channel_variance)
        return bundles, analyses

    def test_fixed_plan_distant_sdr_and_hdr_pair_pixels_do_not_retreat(self):
        for kind in ("dht-wb", "reconstruct-gain"):
            with self.subTest(kind=kind):
                bundles, analyses = self.decoded_pair(kind)
                analysis = analyses[0]
                plan = build_render_plan(bundles[0], analysis, "agx", "p3")
                compiled = compile_hdr_agx_plan(plan, analysis=analysis)
                # Compare the same conservative HDR colour formation across
                # files; the second file additionally withdraws rho authority.
                hdr = replace(compiled, color=replace(compiled.color, channel_separation=0.))
                sdr_bytes, sdr_floats, pair_bytes, pair_floats, pair_hdr = [], [], [], [], []
                for bundle in bundles:
                    sdr_bytes.append(render_output_u8(bundle, analysis, "p3", plan))
                    sdr_floats.append(render_output_encoded_float(bundle, analysis, "p3", plan))
                    byte, alternate = render_ultrahdr_agx_pair(bundle, analysis, plan, hdr)
                    floating, float_alternate = render_ultrahdr_agx_pair(
                        bundle, analysis, plan, hdr, sdr_float=True)
                    pair_bytes.append(byte)
                    pair_floats.append(floating)
                    pair_hdr.append(alternate)
                    np.testing.assert_array_equal(byte, sdr_bytes[-1])
                    np.testing.assert_allclose(floating, sdr_floats[-1], atol=2e-7, rtol=0)
                    np.testing.assert_array_equal(alternate, float_alternate)
                    np.testing.assert_array_equal(bundle.clip_masks[16, 16], 0)
                far = np.s_[:32, :32]
                np.testing.assert_array_equal(bundles[0].scene_rec2020_render[far],
                                              bundles[1].scene_rec2020_render[far])
                self.assertGreater(float(np.ptp(sdr_floats[0][16, 16])), .01)
                for outputs in (sdr_bytes, sdr_floats, pair_bytes, pair_floats, pair_hdr):
                    np.testing.assert_array_equal(outputs[0][far], outputs[1][far])

    def test_supplied_nonzero_hdr_separation_is_qualified_in_both_entrypoints(self):
        for kind in ("dht-wb", "reconstruct-gain"):
            with self.subTest(kind=kind):
                bundles, analyses = self.decoded_pair(kind)
                analysis = analyses[0]
                plan = build_render_plan(bundles[0], analysis, "agx", "p3")
                # Deliberately pass a fixed, previously open HDR plan. A new
                # decode's qualification must still win over caller-supplied rho.
                open_scene = replace(plan.scene, reliable_tail_ev_p9999=8.)
                open_plan = replace(plan, scene=open_scene)
                hdr = compile_hdr_agx_plan(open_plan, analysis=analysis)
                hdr = replace(hdr, color=replace(hdr.color, channel_separation=.8, snr_gate=1.))
                zero = replace(hdr, color=replace(hdr.color, channel_separation=0.))
                self.assertGreater(hdr.tone.rendered_headroom_ev, 0.)
                # User exposure brings the genuine decoded coloured signal into
                # the extra range so that an erroneously open rho changes pixels.
                bundle = replace(bundles[1], exposure_gain=16.)
                qualified = scene_render_to_hdr_display_linear(bundle, plan, hdr)
                manual_zero = scene_render_to_hdr_display_linear(bundle, plan, zero)
                np.testing.assert_array_equal(qualified, manual_zero)
                trusted_control = replace(bundle, scene_loss_support_untrusted=False)
                wrongly_open = scene_render_to_hdr_display_linear(trusted_control, plan, hdr)
                self.assertGreater(float(np.max(np.abs(wrongly_open - manual_zero))), 1e-4)
                for floating in (False, True):
                    actual_base, actual_hdr = render_ultrahdr_agx_pair(
                        bundle, analysis, plan, hdr, sdr_float=floating)
                    expected_base, expected_hdr = render_ultrahdr_agx_pair(
                        bundle, analysis, plan, zero, sdr_float=floating)
                    np.testing.assert_array_equal(actual_base, expected_base)
                    np.testing.assert_array_equal(actual_hdr, expected_hdr)
                    np.testing.assert_allclose(actual_hdr, manual_zero, atol=2e-6, rtol=0)

    def test_physical_noise_model_survives_but_propagation_and_reliability_close(self):
        for kind in ("dht-wb", "reconstruct-gain"):
            with self.subTest(kind=kind):
                bundles, analyses = self.decoded_pair(kind)
                model = analyses[1].noise_model
                scene = bundles[1].scene_rec2020_render / bundles[1].scene_scale
                before = dict(model.channel_variance)
                # The calibrated approximation requires a coarse cell with
                # at least four native sensels, independent of qualification.
                coarse_scene = scene[::2, ::2]
                variance, reason = calibrated_chroma_variance(bundles[1], model, coarse_scene)
                self.assertIsNone(variance)
                self.assertIn("uncertified", reason)
                baseline_variance, _ = calibrated_chroma_variance(
                    bundles[0], analyses[0].noise_model, coarse_scene)
                self.assertIsNotNone(baseline_variance)
                correction = _prepare_chroma_nr_map(
                    bundles[1], SimpleNamespace(chroma_nr=1.), None,
                    scene.reshape(-1, 3), None, *scene.shape[:2], "none", 0., None,
                    analysis=analyses[1])
                self.assertIsNone(correction)
                self.assertEqual(bundles[1].chroma_nr_status, "skipped")
                self.assertIn("uncertified", bundles[1].chroma_nr_reason)
                self.assertEqual(model.status, "valid")
                self.assertEqual(model.channel_variance, before)
                _, _, reliable, evidence_ok = reliable_scene_ev_selection(bundles[1], analyses[1])
                self.assertFalse(np.any(reliable))
                self.assertFalse(evidence_ok)
                plan = build_render_plan(bundles[1], analyses[1], "agx", "p3")
                hdr = compile_hdr_agx_plan(plan, analysis=analyses[1])
                self.assertEqual(hdr.color.channel_separation, 0.)
                self.assertEqual(hdr.tone.rendered_headroom_ev, 0.)
                self.assertEqual(hdr.reliability_source, "decoder-support-untrusted")


if __name__ == "__main__":
    unittest.main()
