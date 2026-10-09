# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent spectral constraints survive real DNG variance fallback."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import priors
from dngscan.analysis import analyze
from dngscan.calibration import import_calibration
from dngscan.hdr_agx_plan import compile_tail_snr_gate
from dngscan.raw_io import load_raw
from dngscan.render import _prepare_chroma_nr_map
from tests.test_pipeline_corrections import write_sensor_dng
from tests.test_user_calibration import collect_profile, unresolved_collect_profile


def write_noise_dng(path, *, file_profile=True):
    """Add real calibration endpoints, ISO and NoiseProfile to a sensor DNG.

    Keep the original sample strip and pointed-to payloads in place, then
    append a new IFD. No metadata reader or decoder is patched in these tests.
    """
    rng = np.random.default_rng(692)
    y, x = np.indices((128, 128))
    pixels = np.rint(2400 + 180 * np.sin(x / 8) + 100 * np.sin(y / 12)
                    + rng.normal(0, 20, (128, 128))).astype(np.uint16)
    write_sensor_dng(path, signal=pixels)
    original = bytearray(path.read_bytes())
    old_ifd, = struct.unpack_from("<L", original, 4)
    count, = struct.unpack_from("<H", original, old_ifd)
    entries = {}
    for index in range(count):
        entry = bytes(original[old_ifd + 2 + 12 * index:old_ifd + 14 + 12 * index])
        tag, = struct.unpack_from("<H", entry)
        entries[tag] = entry
    replacements = {
        271: (2, 6, b"SIGMA\0"),
        272: (2, 3, b"fp\0"),
        34855: (3, 1, struct.pack("<H", 200)),
        50714: (5, 1, struct.pack("<LL", 1024, 1)),
        50717: (4, 1, struct.pack("<L", 16383)),
    }
    if file_profile:
        replacements[51041] = (12, 2, struct.pack("<dd", 1e-4, 1e-8))
    ifd_offset = len(original)
    payload_offset = ifd_offset + 2 + 12 * len(set(entries) | set(replacements)) + 4
    payload = bytearray()
    for tag, (kind, size, value) in replacements.items():
        field = value.ljust(4, b"\0") if len(value) <= 4 else struct.pack("<L", payload_offset + len(payload))
        entries[tag] = struct.pack("<HHL", tag, kind, size) + field
        if len(value) > 4:
            payload.extend(value)
    original[4:8] = struct.pack("<L", ifd_offset)
    original.extend(struct.pack("<H", len(entries)))
    original.extend(b"".join(entries[tag] for tag in sorted(entries)))
    original.extend(struct.pack("<L", 0))
    original.extend(payload)
    path.write_bytes(original)


class SpectralFallbackPipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        env = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "store"))
        env.start()
        self.addCleanup(env.stop)

    def run_pipeline(self, profile, *, file_profile=True):
        path = self.root / "profile.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        # The synthetic file has no proprietary shutter tag. Explicitly make
        # this test calibration applicable to that unknown readout mode.
        imported = import_calibration(path, shutter_override="any")
        raw_path = self.root / "noise.dng"
        write_noise_dng(raw_path, file_profile=file_profile)
        # Half-size LibRaw decoding supplies four native sensels per render
        # pixel, so this small fixture reaches the calibrated chroma kernel
        # instead of its unrelated minimum-sample-count guard.
        bundle = load_raw(raw_path, scene_half_size=True)
        self.assertEqual((bundle.shot_make, bundle.shot_model, bundle.shot_iso), ("SIGMA", "fp", 200))
        self.assertEqual(bundle.scene_decoder, "libraw")
        self.assertTrue(bundle.noise_decode["supported"])
        result, _, _ = analyze(bundle, 4)
        self.assertEqual(result.prior_id, profile["id"])
        self.assertEqual(priors.find_priors("SIGMA", "fp", iso=200)["calibration_id"], imported["id"])
        scene = bundle.scene_rec2020_render
        height, width = scene.shape[:2]
        correction = _prepare_chroma_nr_map(
            bundle, SimpleNamespace(chroma_nr=1), None,
            scene.reshape(-1, scene.shape[-1]), None, height, width,
            "none", 0., None, analysis=result,
        )
        return bundle, result, correction

    def test_dng_fallback_keeps_independent_spectral_constraint(self):
        profile = unresolved_collect_profile()
        profile["noise_whiteness_h_log2iso"] = [[math.log2(200), .1]]
        bundle, result, correction = self.run_pipeline(profile)
        model = result.noise_model
        self.assertEqual(model.status, "valid")
        self.assertEqual(model.source, "DNG NoiseProfile")
        self.assertEqual(model.reason, "file-declared-model-after-unresolved-prior")
        self.assertEqual(model.fallback_reason, "read-noise-unresolved")
        self.assertEqual(model.coefficients("G1"), (1e-4, 1e-8))
        self.assertEqual(result.gain_e_per_dn, 2.)
        self.assertIsNone(result.prior_read_noise_e)
        self.assertEqual(model.spectral_ratios, {"h": .1})
        self.assertEqual(model.correlation, "measured-spectral-imbalance")
        self.assertEqual(result.noise_correlation_status, "measured-spectral-imbalance")
        self.assertEqual(compile_tail_snr_gate(result), 0.)
        self.assertIsNone(correction)
        self.assertEqual(bundle.chroma_nr_status, "skipped")
        self.assertIn("correlated-noise", bundle.chroma_nr_reason)

    def test_valid_and_interpolated_read_noise_keep_same_spectral_constraint(self):
        for read_state in ("measured", "interpolated"):
            profile = collect_profile()
            if read_state == "interpolated":
                profile["read_noise_log2iso_log2e"] = [
                    point for point in profile["read_noise_log2iso_log2e"]
                    if point[0] != math.log2(200)]
            profile["noise_whiteness_h_log2iso"] = [[math.log2(200), .1]]
            with self.subTest(read_state=read_state):
                bundle, result, correction = self.run_pipeline(profile)
                self.assertEqual(result.noise_model.status, "valid")
                self.assertEqual(result.noise_model.reason, "matched-shot-read-model")
                self.assertEqual(result.noise_model.spectral_ratios, {"h": .1})
                self.assertAlmostEqual(result.prior_read_noise_e,
                                       2. if read_state == "measured" else math.sqrt(4.5))
                self.assertEqual(compile_tail_snr_gate(result), 0.)
                self.assertIsNone(correction)
                self.assertEqual(bundle.chroma_nr_status, "skipped")

    def test_unresolved_without_replacement_stays_blocked(self):
        profile = unresolved_collect_profile()
        profile["noise_whiteness_h_log2iso"] = [[math.log2(200), .1]]
        bundle, result, correction = self.run_pipeline(profile, file_profile=False)
        self.assertEqual(result.noise_model.status, "unresolved")
        self.assertIsNone(result.noise_model.coefficients("G1"))
        self.assertEqual(result.noise_model.spectral_ratios, {"h": .1})
        self.assertEqual(compile_tail_snr_gate(result), 0.)
        self.assertIsNone(correction)
        self.assertEqual(bundle.chroma_nr_status, "skipped")

    def test_out_of_domain_spectrum_does_not_block_dng_replacement(self):
        profile = unresolved_collect_profile()
        profile["noise_whiteness_h_log2iso"] = [[math.log2(100), .1]]
        bundle, result, correction = self.run_pipeline(profile)
        self.assertEqual(result.noise_model.status, "valid")
        self.assertEqual(result.noise_model.source, "DNG NoiseProfile")
        self.assertEqual(result.noise_model.spectral_ratios, {})
        self.assertEqual(result.noise_model.correlation, "unknown")
        self.assertEqual(compile_tail_snr_gate(result), 1.)
        self.assertEqual(bundle.chroma_nr_status, "active-approximate")
        self.assertIsNotNone(correction)
        self.assertGreater(float(np.max(np.abs(correction))), 0.)


if __name__ == "__main__":
    unittest.main()
