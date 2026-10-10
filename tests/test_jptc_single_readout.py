# SPDX-License-Identifier: GPL-3.0-or-later
"""Official single-point PTC conversion retains its measured readout scope."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import calibration, priors
from dngscan.analysis import analyze
from dngscan.raw_io import load_raw
from tests.test_spectral_fallback_pipeline import write_noise_dng


REPO = Path(__file__).resolve().parents[1]


class SingleJptcReadoutTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "store"))
        environment.start()
        self.addCleanup(environment.stop)

    def csv(self, *, raw_size="128x128", compression="lossless", jpeg_geometry=(64, 32)):
        header = ["#Format: JPTC/2", "#BlackLevel: 1024,1024,1024,1024"]
        if raw_size is not None:
            header.append(f"#RawSize: {raw_size}")
        if compression is not None:
            header.append(f"#Compression: {compression}")
        if jpeg_geometry is not None:
            header.extend((f"#ImageWidth: {jpeg_geometry[0]}", f"#ImageHeight: {jpeg_geometry[1]}"))
        rows = []
        for signal in np.geomspace(2., 15359.*1.15, 80):
            mean = min(1024.+signal, 16383.)
            std = math.sqrt(signal/4.+(3./4.)**2) if mean < 16383. else 0.
            rows.append(f"{mean:.17g},{std:.17g}")
        path = self.root / "ptc-iso200.csv"
        path.write_text("\n".join(header + ["G1_Mean,G1_Std"] + rows), encoding="utf-8")
        return path

    def convert(self, **declarations):
        output = self.root / "single.json"
        process = subprocess.run(
            [sys.executable, str(REPO / "tools/import_jptc.py"), str(self.csv(**declarations)),
             "--brand", "SIGMA", "--model", "fp", "--iso", "200",
             "--shutter", "electronic", "--out", str(output)],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        return output, json.loads(output.read_text(encoding="utf-8"))

    def pipeline(self, *, file_profile=False, **declarations):
        output, payload = self.convert(**declarations)
        # Exercise the shipped installation CLI as well as the converter.
        process = subprocess.run(
            [sys.executable, "-m", "dngscan", "calibration", "import", str(output)],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        installed = json.loads(process.stdout)
        path = self.root / "capture.dng"
        write_noise_dng(path, file_profile=file_profile)
        bundle = load_raw(path, scene_half_size=True)
        result, _, _ = analyze(bundle, 4)
        diagnostic = calibration.calibration_diagnostics(
            bundle.shot_make, bundle.shot_model, bundle.shot_shutter,
            bundle.shot_iso, bundle.capture_readout,
        )[0]
        self.assertEqual(result.prior_id, payload["id"])
        self.assertEqual(diagnostic["id"], installed["id"])
        return payload, bundle, result, diagnostic

    def assert_external_quantities_unavailable(self, result):
        self.assertIsNone(result.gain_e_per_dn)
        self.assertIsNone(result.prior_read_noise_e)
        self.assertIsNone(result.prior_pdr_ev)

    def test_official_cli_install_and_real_dng_match_raw_mosaic_not_jpeg_size(self):
        payload, bundle, result, diagnostic = self.pipeline()
        self.assertEqual(payload["readout_contract"], {"version": 1, "libraw_raw_geometry": [128, 128]})
        self.assertEqual(payload["geometry"], ["64", "32"])
        self.assertEqual(payload["source"]["geometry"], ["64", "32"])
        self.assertEqual(payload["source"]["raw_size"], "128x128")
        self.assertEqual(payload["source"]["compression"], "lossless")
        self.assertEqual(payload["compression"], "lossless")
        self.assertEqual(payload["acquisition_contract"], {"geometry_domain": "camera-jpeg-output"})
        self.assertEqual(bundle.capture_readout["libraw_raw_geometry"], [128, 128])
        self.assertEqual(bundle.scene_rec2020_render.shape[:2], (64, 64))
        self.assertEqual(diagnostic["readout_match_status"], "matched")
        self.assertEqual(result.noise_model.status, "valid")
        self.assertIn("declared-readout-constraints-matched", result.noise_model.approximation)
        self.assertAlmostEqual(result.gain_e_per_dn, 4., places=6)
        self.assertAlmostEqual(result.prior_read_noise_e, 3., places=6)
        a, b = result.noise_model.coefficients("G1")
        self.assertAlmostEqual(a, 1./(15359.*4.), places=14)
        self.assertAlmostEqual(b, (3./(15359.*4.))**2, places=16)
        # A spatial single-frame PTC cannot establish a paired-dark total.
        self.assertNotIn("stored_dark_variance_dn2_log2iso", payload)
        self.assertNotIn("stored_dark_variance_measurement", payload["acquisition_contract"])
        self.assertNotIn("sigma_clip_correction", payload["acquisition_contract"])
        self.assertIn("single-frame spatial std", payload["noise_aperture"])

    def test_declared_mosaic_mismatch_rejects_every_external_quantity(self):
        _, _, result, diagnostic = self.pipeline(raw_size="256x128")
        self.assertEqual(diagnostic["readout_match_status"], "mismatch")
        self.assertEqual(result.noise_model.status, "rejected")
        self.assertEqual(result.noise_model.reason, "file-libraw-raw-geometry-mismatch")
        self.assertEqual(result.prior_quality_status, result.noise_model.reason)
        self.assert_external_quantities_unavailable(result)
        self.assertIsNone(result.noise_model.coefficients("G1"))

    def test_uninterpreted_compression_declaration_remains_visible_and_unverified(self):
        payload, _, result, diagnostic = self.pipeline(compression="RAW HQ 14-bit")
        self.assertEqual(payload["compression"], "RAW HQ 14-bit")
        self.assertEqual(payload["source"]["compression"], "RAW HQ 14-bit")
        self.assertEqual(diagnostic["readout_match_status"], "unverified")
        self.assertEqual(result.noise_model.status, "rejected")
        self.assertEqual(result.noise_model.reason, "measurement-compression-declaration-unverified")
        self.assert_external_quantities_unavailable(result)

    def test_readout_failure_allows_independent_dng_profile_without_borrowing_gain(self):
        for declarations, reason in (({"raw_size": "256x128"}, "file-libraw-raw-geometry-mismatch"),
                                     ({"compression": "RAW HQ"}, "measurement-compression-declaration-unverified")):
            with self.subTest(declarations=declarations):
                _, _, result, _ = self.pipeline(file_profile=True, **declarations)
                self.assertEqual(result.noise_model.status, "valid")
                self.assertEqual(result.noise_model.source, "DNG NoiseProfile")
                self.assertEqual(result.noise_model.coefficients("G1"), (1e-4, 1e-8))
                self.assertEqual(result.noise_model.fallback_reason, reason)
                self.assertEqual(result.prior_quality_status, reason)
                self.assert_external_quantities_unavailable(result)
                # Keep this subcase independent of matching-prior recency.
                for record in calibration.list_calibrations():
                    calibration.remove_calibration(record["id"])

    def test_jpeg_only_declaration_does_not_become_a_sensor_raster_match(self):
        _, _, result, diagnostic = self.pipeline(raw_size=None)
        self.assertEqual(diagnostic["readout_match_status"], "unverified")
        self.assertEqual(result.noise_model.reason, "measurement-raw-geometry-unavailable")
        self.assert_external_quantities_unavailable(result)

    def test_undeclared_legacy_input_keeps_existing_scope(self):
        _, _, result, diagnostic = self.pipeline(raw_size=None, compression=None, jpeg_geometry=None)
        self.assertEqual(diagnostic["readout_match_status"], "not-declared")
        self.assertEqual(result.noise_model.status, "valid")
        self.assertIn("sub-readout-not-declared", result.noise_model.approximation)

    def test_malformed_raw_size_is_not_silently_discarded(self):
        for value in ("128", "128x0", "active 128x128"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "RawSize"):
                calibration.import_csv(self.csv(raw_size=value), "SIGMA", "fp", 200, None, "electronic")

    def test_packaged_single_point_loader_preserves_the_same_contract(self):
        output, _ = self.convert(raw_size="256x128")
        original_glob = Path.glob
        packaged = Path(priors.__file__).parent / "data/priors/jptc"
        collect = packaged.parent / "jptc_collect"

        def files(path, pattern):
            if path == packaged:
                return iter([output])
            if path == collect:
                return iter([])
            return original_glob(path, pattern)

        path = self.root / "capture.dng"
        write_noise_dng(path, file_profile=False)
        bundle = load_raw(path, scene_half_size=True)
        with patch.object(Path, "glob", files), patch.object(priors, "_JPTC_CACHE", None), \
                patch.object(priors, "PRIOR_TABLE", []):
            entry = priors._jptc_entries()[0]
            self.assertEqual(entry["readout_contract"]["libraw_raw_geometry"], [256, 128])
            self.assertTrue(entry["readout_contract"]["storage_lossless"])
            self.assertEqual(entry["readout_informational"], {"camera_jpeg_geometry": [64, 32]})
            result, _, _ = analyze(bundle, 4)
        self.assertEqual(result.noise_model.status, "rejected")
        self.assertEqual(result.noise_model.reason, "file-libraw-raw-geometry-mismatch")
        self.assert_external_quantities_unavailable(result)


if __name__ == "__main__":
    unittest.main()
