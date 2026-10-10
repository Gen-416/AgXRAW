# SPDX-License-Identifier: GPL-3.0-or-later
"""LinearResponseLimit is a validity threshold, never a DN storage rescale."""
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dngscan.analysis import analyze
from dngscan.calibration import import_calibration
from dngscan.hdr_agx_plan import compile_tail_snr_gate
from dngscan.noise_propagation import calibrated_chroma_variance
from dngscan.raw_io import load_raw
from dngscan.render import _prepare_chroma_nr_map
from tests.test_noise_scene_ev import write_scene_noise_dng
from tests.test_user_calibration import single_profile


def write_coding_dng(path, limit, *, file_profile=False, white=4095):
    write_scene_noise_dng(path, 0.)
    data = bytearray(path.read_bytes())
    ifd, = struct.unpack_from("<L", data, 4)
    count, = struct.unpack_from("<H", data, ifd)
    for index in range(count):
        offset = ifd + 2 + 12 * index
        tag, = struct.unpack_from("<H", data, offset)
        pointer, = struct.unpack_from("<L", data, offset + 8)
        if tag == 50734:
            struct.pack_into("<LL", data, pointer, round(limit * 1000000), 1000000)
        elif tag == 50717:
            struct.pack_into("<L", data, offset + 8, white)
        elif tag == 51041 and not file_profile:
            struct.pack_into("<H", data, offset, 65000)
        elif tag == 273:
            rng = np.random.default_rng(819)
            pixels = np.rint(1000 + rng.normal(0, 20, (128, 128))).astype("<u2")
            data[pointer:pointer + pixels.nbytes] = pixels.tobytes()
    path.write_bytes(data)


class NoiseCodingRangeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        env = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "store"))
        env.start()
        self.addCleanup(env.stop)

    def install(self, *, spectrum=None):
        profile = single_profile()
        profile.update(brand="Review", model="Synthetic", iso=200,
                       gain_e_per_dn=2., read_noise_e=2., fwc_e=8190.,
                       white_level_used=4095, black_level_g1=0)
        if spectrum is not None:
            # Collect profiles carry independent spectral curves.
            profile = {"format": "dngscan-jptc-collect-1", "id": "coding ruler",
                       "make": "Review", "model_candidates": ["Synthetic"], "shutter": "any",
                       "ptc_anchor": {"iso": 200, "gain_e_per_dn": 2, "quality": "ok"},
                       "fwc_e": 8190., "gain_log2iso_log2epd": [[math.log2(200), 1.]],
                       "read_noise_log2iso_log2e": [[math.log2(200), 1.]],
                       "noise_whiteness_h_log2iso": [[math.log2(200), spectrum]]}
        path = self.root / "calibration.json"
        path.write_text(json.dumps(profile))
        import_calibration(path, shutter_override="any")

    def decoded(self, limit, **kwargs):
        path = self.root / "sensor.dng"
        write_coding_dng(path, limit, **kwargs)
        bundle = load_raw(path, scene_half_size=True)
        result, _, _ = analyze(bundle, 4)
        scene = bundle.scene_rec2020_render
        correction = _prepare_chroma_nr_map(
            bundle, SimpleNamespace(chroma_nr=1), None, scene.reshape(-1, 3), None,
            *scene.shape[:2], "none", 0., None, analysis=result)
        variance, _ = calibrated_chroma_variance(bundle, result.noise_model, scene)
        return bundle, result, variance, correction

    def test_response_limit_cannot_change_coefficients_transfer_or_nr(self):
        self.install()
        reference = self.decoded(1.)
        for limit in (.97, .8, .5):
            with self.subTest(limit=limit):
                bundle, analysis, variance, correction = self.decoded(limit)
                original, expected, expected_variance, expected_correction = reference
                self.assertEqual(bundle.coding_white_levels, [4095.])
                self.assertLess(bundle.camera_white_levels[0], original.camera_white_levels[0])
                np.testing.assert_array_equal(bundle.raw_image, original.raw_image)
                np.testing.assert_array_equal(bundle.scene_rec2020_render, original.scene_rec2020_render)
                self.assertEqual(float(bundle.clip_masks.max()), 0.)
                self.assertEqual(analysis.noise_model.status, "valid")
                self.assertEqual(analysis.noise_model.channel_variance, expected.noise_model.channel_variance)
                self.assertEqual(analysis.noise_model.coefficients("G1"),
                                 (1 / 8190., (2 / 8190.) ** 2))
                # LRL changes the resolved sensor endpoint, not the DN encoding
                # or covariance transfer. Keep its new reliability stamp separate.
                self.assertEqual(
                    {key: value for key, value in bundle.noise_decode.items() if key != "source_loss_fullwell"},
                    {key: value for key, value in original.noise_decode.items() if key != "source_loss_fullwell"},
                )
                self.assertEqual(bundle.noise_decode["source_loss_fullwell"],
                                 {str(cid): level for cid, level in analysis.channel_fullwell.items()})
                np.testing.assert_array_equal(variance, expected_variance)
                np.testing.assert_array_equal(correction, expected_correction)
                self.assertEqual(bundle.chroma_nr_status, "active-approximate")
                self.assertGreater(float(np.max(np.abs(correction))), 0.)

    def test_response_limit_cannot_revoke_applicable_spectral_constraint(self):
        self.install(spectrum=.1)
        for limit in (1., .97, .8, .5):
            with self.subTest(limit=limit):
                bundle, analysis, variance, correction = self.decoded(limit, file_profile=True)
                model = analysis.noise_model
                self.assertEqual(model.status, "valid")
                self.assertIn("User JPTC calibration", model.source)
                self.assertEqual(model.spectral_ratios, {"h": .1})
                self.assertEqual(model.correlation, "measured-spectral-imbalance")
                self.assertEqual(compile_tail_snr_gate(analysis), 0.)
                self.assertIsNone(variance)
                self.assertIsNone(correction)
                self.assertEqual(bundle.chroma_nr_status, "skipped")

    def test_actual_coding_span_mismatch_is_still_rejected(self):
        self.install()
        _, analysis, variance, correction = self.decoded(1., white=3000)
        self.assertEqual(analysis.noise_model.status, "rejected")
        self.assertEqual(analysis.noise_model.reason, "unmatched-dn-scale")
        self.assertIsNone(variance)
        self.assertIsNone(correction)

    def test_global_maximum_cannot_revoke_valid_per_plane_prior(self):
        self.install()
        bundle, expected, _, _ = self.decoded(.8)
        # Sony ARW can publish a global LibRaw maximum unrelated to the
        # per-plane coding endpoints. Isolate that metadata handoff on a
        # real decoded DNG while keeping sensor samples and coding unchanged.
        changed, _, _ = analyze(replace(bundle, white_level=6000), 4)
        self.assertEqual(changed.noise_model.channel_variance,
                         expected.noise_model.channel_variance)
        self.assertEqual(changed.gain_e_per_dn, expected.gain_e_per_dn)
        self.assertEqual(changed.gain_e_per_dn, 2.)
        self.assertEqual(changed.prior_read_noise_e, 2.)
        self.assertEqual(changed.prior_quality_status, expected.prior_quality_status)


if __name__ == "__main__":
    unittest.main()
