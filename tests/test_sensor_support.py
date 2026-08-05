# SPDX-License-Identifier: GPL-3.0-or-later
"""Opened-interface gates: new-body priors, fallback matrices, WB degradation.

The contract: every camera renders. Missing per-body data downgrades loudly —
sensor priors fall back to single-frame measurement with a completeness note,
fixed-Kelvin WB falls through the calibration ladder (DNG tags -> LibRaw matrix
-> fallback matrix table) and, at the bottom, degrades to as-shot with a
reported warning instead of refusing the export.
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from dngscan import priors as sensor_priors
from dngscan.camera_matrices import fallback_xyz_to_cam
from dngscan.raw_io import camera_data_support_note, solve_wb_for_mode
from dngscan.wb import kelvin_camera_multipliers

# (make, model) pairs as the cameras report themselves in EXIF.
NEW_BODIES = {
    "Sony ILCE-7M5 (A7 V)": ("SONY", "ILCE-7M5"),
    "Sony ILCE-7SM3 (A7S III)": ("SONY", "ILCE-7SM3"),
    "Sony ILCE-7RM6 (A7R VI)": ("SONY", "ILCE-7RM6"),
    "Ricoh GR IV": ("RICOH IMAGING COMPANY, LTD.", "RICOH GR IV"),
    "Nikon Z f": ("NIKON CORPORATION", "Z f"),
    "Fujifilm X100VI": ("FUJIFILM", "X100VI"),
    "Fujifilm X-E5": ("FUJIFILM", "X-E5"),
}


class PriorsTableTests(unittest.TestCase):
    def test_every_new_body_has_chart_priors(self) -> None:
        for prior_id, (make, model) in NEW_BODIES.items():
            with self.subTest(camera=prior_id):
                entry = sensor_priors.find_priors(make, model)
                self.assertIsNotNone(entry, f"no priors for {make} {model}")
                self.assertEqual(entry["id"], prior_id)

    def test_prior_curves_interpolate_to_physical_values(self) -> None:
        for prior_id, (make, model) in NEW_BODIES.items():
            with self.subTest(camera=prior_id):
                entry = sensor_priors.find_priors(make, model)
                for iso in (100, 800, 6400):
                    pdr = sensor_priors.pdr_ev(entry, iso)
                    rn = sensor_priors.read_noise_e(entry, iso)
                    gain = sensor_priors.gain_e_per_dn(entry, iso)
                    self.assertTrue(pdr is not None and 2.0 < pdr < 14.0, (iso, pdr))
                    self.assertTrue(rn is not None and 0.2 < rn < 40.0, (iso, rn))
                    self.assertTrue(gain is not None and math.isfinite(gain))

    def test_pdr_decreases_with_iso(self) -> None:
        """The one shape every real sensor obeys; a mis-parsed series breaks it."""
        for prior_id, (make, model) in NEW_BODIES.items():
            with self.subTest(camera=prior_id):
                entry = sensor_priors.find_priors(make, model)
                self.assertGreater(
                    sensor_priors.pdr_ev(entry, 200),
                    sensor_priors.pdr_ev(entry, 12800),
                )

    def test_unknown_camera_degrades_to_none(self) -> None:
        self.assertIsNone(sensor_priors.find_priors("ACME", "IMAGINARY-1"))


class FallbackMatrixTests(unittest.TestCase):
    def test_known_new_bodies_have_matrices(self) -> None:
        for make, model in (
            ("SONY", "ILCE-7M5"), ("SONY", "ILCE-7SM3"),
            ("NIKON CORPORATION", "Z f"), ("FUJIFILM", "X100VI"),
        ):
            with self.subTest(model=model):
                hit = fallback_xyz_to_cam(make, model)
                self.assertIsNotNone(hit)
                matrix, note = hit
                self.assertEqual(np.asarray(matrix).shape, (3, 3))
                self.assertIn("LibRaw master", note)

    def test_xe5_matrix_declares_its_borrow(self) -> None:
        matrix, note = fallback_xyz_to_cam("FUJIFILM", "X-E5")
        self.assertIn("X100VI", note)

    def test_kelvin_solve_works_through_fallback_matrix(self) -> None:
        matrix, _note = fallback_xyz_to_cam("SONY", "ILCE-7M5")
        mult = kelvin_camera_multipliers(5500.0, matrix)
        self.assertEqual(mult[1], 1.0)
        self.assertTrue(all(0.2 < m < 8.0 for m in mult))

    def test_a7r6_is_deliberately_absent(self) -> None:
        """No published coefficients located: absence is the declared state, the
        degradation path covers the body (see camera_matrices.py)."""
        self.assertIsNone(fallback_xyz_to_cam("SONY", "ILCE-7RM6"))


class WbDegradationTests(unittest.TestCase):
    def test_missing_calibration_degrades_with_note(self) -> None:
        result, note = solve_wb_for_mode(
            "5500k", Path("/nonexistent/fake.arw"), None,
            make="ACME", model="IMAGINARY-1",
        )
        self.assertIsNone(result)
        self.assertIn("退化为相机 AsShot", note)
        self.assertIn("可用", note)

    def test_fallback_matrix_rescues_the_solve(self) -> None:
        result, note = solve_wb_for_mode(
            "5500k", Path("/nonexistent/fake.raf"), None,
            make="FUJIFILM", model="X-E5",
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        self.assertIn("回退矩阵表", note)

    def test_camera_mode_is_untouched(self) -> None:
        result, note = solve_wb_for_mode("camera", Path("/nonexistent/x.arw"), None)
        self.assertIsNone(result)
        self.assertIsNone(note)


class DecodeSupportProbeTests(unittest.TestCase):
    """The per-file two-decoder tier report: deterministic labels per tier."""

    def _probe(self, libraw_tier, ci_probe):
        from unittest import mock

        from dngscan import decode_support

        with mock.patch.object(decode_support, "_libraw_tier", return_value=libraw_tier), \
             mock.patch("dngscan.coreimage_decode.probe_raw9_support",
                        return_value=ci_probe), \
             mock.patch("dngscan.dng_metadata.read_dng_shot_info") as shot, \
             mock.patch("dngscan.priors.find_priors", return_value=None), \
             mock.patch.object(Path, "is_file", return_value=True):
            shot.return_value.make = "NIKON CORPORATION"
            shot.return_value.model = "Z50_2"
            return decode_support.probe_decode_support(Path("/x/DSC_0001.NEF"))

    def test_format_gap_blocks_coreimage_and_says_so(self) -> None:
        report = self._probe(
            {"status": "unsupported_format", "detail": "TicoRAW"},
            {"coreimage_available": True, "raw9_supported": True,
             "versions_offered": ("9",), "fallback_version": None, "error": None},
        )
        self.assertTrue(report["coreimage"]["blocked_by_libraw"])
        joined = "\n".join(report["lines"])
        self.assertIn("格式缺口", joined)
        self.assertIn("统一 Evidence 策略要求 LibRaw", joined)
        self.assertEqual(report["evidence"]["provider"], "libraw")

    def test_full_support_reads_clean(self) -> None:
        report = self._probe(
            {"status": "full", "detail": "机型颜色矩阵在 LibRaw 表内"},
            {"coreimage_available": True, "raw9_supported": True,
             "versions_offered": ("9", "8"), "fallback_version": None, "error": None},
        )
        joined = "\n".join(report["lines"])
        self.assertIn("✓ 完整支持", joined)
        self.assertIn("RAW 9", joined)
        self.assertFalse(report["coreimage"]["blocked_by_libraw"])

    def test_downgrade_tier_names_the_available_version(self) -> None:
        report = self._probe(
            {"status": "fallback_matrix", "detail": "回退表"},
            {"coreimage_available": True, "raw9_supported": False,
             "versions_offered": ("8",), "fallback_version": "8", "error": None},
        )
        joined = "\n".join(report["lines"])
        self.assertIn("仅 RAW 8", joined)
        self.assertIn("△ 可用", joined)


class UnsupportedFormatGuidanceTests(unittest.TestCase):
    """Format gaps (files LibRaw cannot open) get a precise diagnosis, not a
    generic 'unsupported' — the canonical case being Nikon HE/HE* TicoRAW NEFs."""

    def _shot(self, make="NIKON CORPORATION", model="Z50_2"):
        from types import SimpleNamespace

        return SimpleNamespace(make=make, model=model)

    def test_nef_guidance_names_the_cause_and_the_outs(self) -> None:
        from dngscan.raw_io import _unsupported_format_guidance

        msg = _unsupported_format_guidance(
            Path("/x/DSC_0001.NEF"), self._shot(), RuntimeError("Unsupported file format")
        )
        self.assertIn("TicoRAW", msg)
        self.assertIn("DNG Converter", msg)
        self.assertIn("无损压缩", msg)
        self.assertIn("Z50_2", msg)

    def test_other_formats_point_to_the_master_upgrade(self) -> None:
        from dngscan.raw_io import _unsupported_format_guidance

        msg = _unsupported_format_guidance(
            Path("/x/photo.arw"), self._shot("SONY", "ILCE-9M4"),
            RuntimeError("Unsupported file format"),
        )
        self.assertIn("build_libraw_master", msg)
        self.assertNotIn("TicoRAW", msg)

    def test_load_raw_surfaces_the_guidance(self) -> None:
        import tempfile
        from unittest import mock

        import rawpy

        from dngscan import raw_io

        with tempfile.NamedTemporaryFile(suffix=".NEF") as tmp:
            tmp.write(b"II*\x00")  # minimal TIFF magic so shot-info parsing no-ops
            tmp.flush()
            with mock.patch.object(
                rawpy, "imread",
                side_effect=rawpy.LibRawFileUnsupportedError("Unsupported file format"),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    raw_io.load_raw(Path(tmp.name))
        self.assertIn("TicoRAW", str(ctx.exception))


class DataSupportMarkerTests(unittest.TestCase):
    """The consolidated per-body marker: truthful label, never a gate."""

    def test_any_real_calibration_means_full_support(self) -> None:
        self.assertIsNone(camera_data_support_note(True, False, False, False, "A", "B"))
        self.assertIsNone(camera_data_support_note(False, True, False, True, "A", "B"))

    def test_fallback_only_declares_partial_coverage(self) -> None:
        note = camera_data_support_note(False, False, True, True, "SONY", "ILCE-XX")
        self.assertIn("回退", note)
        self.assertIn("功能照常执行", note)
        self.assertIn("ILCE-XX", note)

    def test_nothing_at_all_declares_unpredictable_deviation(self) -> None:
        note = camera_data_support_note(False, False, False, False, "ACME", "IMAGINARY-1")
        self.assertIn("无法预测的偏差", note)
        self.assertIn("功能照常执行", note)
        self.assertIn("传感器先验", note)

    def test_missing_priors_alone_does_not_raise_the_marker(self) -> None:
        """Priors degrade analysis numbers, not the render; the priors report line
        carries that honestly without escalating to the accuracy marker."""
        self.assertIsNone(
            camera_data_support_note(False, True, False, False, "A", "B")
        )


if __name__ == "__main__":
    unittest.main()
