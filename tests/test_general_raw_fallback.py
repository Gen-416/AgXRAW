# SPDX-License-Identifier: GPL-3.0-or-later
"""Uncalibrated cameras remain usable without inventing sensor noise evidence."""
from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan.analysis import analyze, noise_floor_ev_estimate, snr_ev_coordinates
from dngscan.auto_ev import compute_auto_ev
from dngscan.gui.preview_cache import build_proxy_entry, _read_disk_entry, _write_disk_entry
from dngscan.gui.service import detected_scene_params
from dngscan.hdr_agx import render_ultrahdr_agx_pair
from dngscan.hdr_agx_plan import compile_hdr_agx_plan, compile_tail_snr_gate
from dngscan.raw_io import load_raw
from dngscan.render import render_output_encoded_float
from dngscan.scene_scale import with_intent_exposure
from dngscan.tone import build_render_plan
from tests.test_pipeline_corrections import write_sensor_dng


def write_uncalibrated_dng(path, *, profile=False, reduction=None):
    """Write a real unknown-camera Bayer DNG, deliberately without an ISO tag.

    The embedded colour matrix makes the file decodable independently of a
    camera database. All variation is scene signal; tests never estimate a
    noise model from it or replace the decoder/metadata/model implementations.
    """
    y, x = np.indices((128, 128))
    signal = (120 + 8 * x + 4 * y + 45 * np.sin(x / 7) * np.cos(y / 9))
    signal[88:116, 88:116] = 3400
    write_sensor_dng(path, signal=np.rint(signal).astype(np.uint16))
    if not profile and reduction is None:
        return
    data = bytearray(path.read_bytes())
    old_ifd, = struct.unpack_from("<L", data, 4)
    count, = struct.unpack_from("<H", data, old_ifd)
    entries = {}
    for index in range(count):
        entry = bytes(data[old_ifd + 2 + 12 * index:old_ifd + 14 + 12 * index])
        tag, = struct.unpack_from("<H", entry)
        entries[tag] = entry
    additions = {}
    if profile:
        additions[51041] = (12, 2, struct.pack("<dd", 1e-4, 4e-6))
    if reduction is not None:
        additions[50935] = (5, 1, struct.pack("<LL", round(reduction * 1000), 1000))
    offset = len(data)
    payload_offset = offset + 2 + 12 * len(set(entries) | set(additions)) + 4
    payload = bytearray()
    for tag, (kind, size, value) in additions.items():
        field = value.ljust(4, b"\0") if len(value) <= 4 else struct.pack("<L", payload_offset + len(payload))
        entries[tag] = struct.pack("<HHL", tag, kind, size) + field
        if len(value) > 4:
            payload.extend(value)
    data[4:8] = struct.pack("<L", offset)
    data.extend(struct.pack("<H", len(entries)))
    data.extend(b"".join(entries[tag] for tag in sorted(entries)))
    data.extend(bytes(4))
    data.extend(payload)
    path.write_bytes(data)


class GeneralRawFallbackTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "calibration"))
        environment.start()
        self.addCleanup(environment.stop)

    def decoded(self, **kwargs):
        path = self.root / "uncalibrated.dng"
        write_uncalibrated_dng(path, **kwargs)
        # Actual Bayer half decoding provides the coarse noise propagator's
        # four sensels per pixel; missing models cannot hide behind that guard.
        bundle = load_raw(path, scene_half_size=True)
        self.assertEqual((bundle.shot_make, bundle.shot_model), ("Review", "Synthetic"))
        self.assertIsNone(bundle.shot_iso)
        self.assertEqual(bundle.scene_decoder, "libraw")
        self.assertTrue(bundle.noise_decode["supported"])
        analysis, _, _ = analyze(bundle, 4, diagnostics=False)
        self.assertIsNone(analysis.prior_id)
        self.assertTrue(np.all(np.isfinite(bundle.scene_rec2020_render)))
        self.assertEqual(analysis.cell_union_pct, 0.)
        return bundle, analysis

    def formed(self, bundle, analysis):
        # Explicitly request the optional NR: ordinary rendering must remain
        # possible even when that one capability lacks independent evidence.
        plan = build_render_plan(bundle, analysis, "agx", "p3", chroma_nr=1.)
        automatic = compute_auto_ev(bundle, analysis, "p3", chroma_nr=1.)
        self.assertTrue(math.isfinite(automatic.ev))
        self.assertTrue(math.isfinite(automatic.highlight_cap_ev))
        self.assertGreaterEqual(automatic.ev, 0.)
        self.assertTrue(math.isfinite(plan.tone.black_ev))
        self.assertGreater(plan.tone.white_ev, plan.tone.black_ev)
        exposed = with_intent_exposure(bundle, user_ev=automatic.ev)
        hdr_plan = compile_hdr_agx_plan(plan, analysis=analysis)
        sdr = render_output_encoded_float(exposed, analysis, "p3", plan)
        base, hdr = render_ultrahdr_agx_pair(exposed, analysis, plan, hdr_plan, sdr_float=True)
        for output in (sdr, base, hdr):
            self.assertEqual(output.shape, bundle.scene_rec2020_render.shape)
            self.assertTrue(np.all(np.isfinite(output)))
            self.assertGreater(float(np.ptp(output)), .1)
        self.assertTrue(np.all((sdr >= 0) & (sdr <= 1)))
        np.testing.assert_allclose(base, sdr, atol=2e-7, rtol=0)
        return exposed, plan, hdr_plan, automatic, sdr, hdr

    def assert_missing_noise(self, analysis):
        self.assertTrue(math.isnan(analysis.noise_floor))
        self.assertTrue(math.isnan(analysis.usable_dr_ev))
        self.assertIsNone(analysis.gain_e_per_dn)
        self.assertIsNone(analysis.noise_floor_e)
        self.assertIsNone(analysis.prior_read_noise_e)
        self.assertEqual(noise_floor_ev_estimate(analysis)[1], "none")
        self.assertIsNone(snr_ev_coordinates(analysis))
        for curve in analysis.snr_curves.values():
            self.assertTrue(np.all(np.isnan(curve["snr_db"])))

    def assert_nr_skipped_without_pixel_change(self, rendered, analysis):
        exposed, plan, _, _, sdr, _ = rendered
        self.assertEqual(exposed.chroma_nr_status, "skipped")
        self.assertIn("calibration unavailable", exposed.chroma_nr_reason)
        off = replace(plan, tone=replace(plan.tone, chroma_nr=0.))
        np.testing.assert_array_equal(render_output_encoded_float(replace(exposed), analysis, "p3", off), sdr)

    def test_unknown_camera_without_iso_or_noise_profile_forms_sdr_and_hdr(self):
        bundle, analysis = self.decoded()
        self.assertEqual(analysis.noise_model.status, "unavailable")
        self.assertEqual(analysis.noise_evidence_status, "unavailable")
        self.assert_missing_noise(analysis)
        self.assertEqual(compile_tail_snr_gate(analysis), 1.)
        rendered = self.formed(bundle, analysis)
        self.assert_nr_skipped_without_pixel_change(rendered, analysis)
        detected = detected_scene_params(rendered[0], analysis, rendered[1])
        evidence = detected["processing_evidence"]
        self.assertEqual(evidence["mode"], "general-image")
        self.assertEqual(evidence["noise_status"], "unavailable")
        self.assertEqual(evidence["hdr_reliability_source"], rendered[2].reliability_source)
        self.assertEqual(detected["reliability_source"], rendered[2].reliability_source)
        self.assertEqual(evidence["chroma_nr_status"], "skipped")

    def test_file_noise_profile_does_not_require_iso_or_external_calibration(self):
        bundle, analysis = self.decoded(profile=True)
        self.assertEqual(analysis.noise_model.status, "valid")
        self.assertEqual(analysis.noise_model.source, "DNG NoiseProfile")
        self.assertEqual(analysis.noise_model.coefficients("G1"), (1e-4, 4e-6))
        self.assertAlmostEqual(analysis.noise_floor, .002)
        self.assertIsNone(analysis.gain_e_per_dn)
        self.assertIsNone(analysis.noise_floor_e)
        self.assertTrue(all(np.all(np.isfinite(curve["snr_db"]))
                            for curve in analysis.snr_curves.values()))
        exposed, *_ = self.formed(bundle, analysis)
        self.assertEqual(exposed.chroma_nr_status, "active-approximate")

    def test_declared_raw_processing_keeps_noise_rejected_but_allows_image_formation(self):
        bundle, analysis = self.decoded(profile=True, reduction=.5)
        self.assertEqual(analysis.noise_model.status, "rejected")
        self.assertEqual(analysis.noise_model.reason, "raw-noise-reduction-declared")
        self.assertEqual(analysis.noise_evidence_status, "model-rejected")
        self.assert_missing_noise(analysis)
        self.assertEqual(compile_tail_snr_gate(analysis), 0.)
        rendered = self.formed(bundle, analysis)
        self.assertEqual(rendered[2].color.snr_gate, 0.)
        self.assert_nr_skipped_without_pixel_change(rendered, analysis)

    def test_disk_preview_roundtrip_keeps_uncalibrated_rendering_and_missing_snr(self):
        bundle, analysis = self.decoded()
        original = self.formed(bundle, analysis)
        entry = build_proxy_entry(bundle, analysis, include_guidance=True)
        cache_path = self.root / "preview.npz"
        _write_disk_entry(cache_path, entry)
        restored = _read_disk_entry(cache_path, bundle.path, require_guidance=True)
        self.assertIsNotNone(restored)
        self.assertIsNone(restored.bundle.raw_image)
        self.assertEqual(restored.analysis.noise_model.status, "unavailable")
        self.assert_missing_noise(restored.analysis)
        self.assertIsNone(restored.bundle.raw_guidance.snr_confidence)
        cached = self.formed(restored.bundle, restored.analysis)
        self.assert_nr_skipped_without_pixel_change(cached, restored.analysis)
        before = detected_scene_params(original[0], analysis, original[1])["processing_evidence"]
        after = detected_scene_params(cached[0], restored.analysis, cached[1])["processing_evidence"]
        # Releasing RAW pixels is a memory policy, not a loss of the immutable
        # sensor facts already captured in the persisted full analysis.
        for key in ("mode", "noise_status", "sensor_evidence", "label", "hdr_reliability_source"):
            self.assertEqual(after[key], before[key], key)
        self.assertEqual(cached[3].ev, original[3].ev)
        self.assertEqual(cached[1].tone, original[1].tone)
        np.testing.assert_allclose(cached[4], original[4], atol=2e-7, rtol=0)
        np.testing.assert_allclose(cached[5], original[5], atol=2e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
