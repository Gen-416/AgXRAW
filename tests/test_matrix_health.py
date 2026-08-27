# SPDX-License-Identifier: GPL-3.0-or-later
"""Route-E matrix-health diagnostic: condition number of the active calibration.

Diagnostic only — a suspicious matrix never refuses a render, it changes what
the report claims about gamut-pressure confidence. Thresholds are pinned to
the measured fleet (evidence-shell DNG matrices 2.40..4.23, fallback table
2.56..3.18): >6 reads 偏高, >10 reads 异常.
"""
from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace

import numpy as np

from dngscan.wb import interpolated_color_matrix, matrix_health


def _dual_calibration():
    return SimpleNamespace(
        cct1=2856.0,
        matrix1=[[0.9, -0.3, -0.08], [-0.35, 1.1, 0.25], [-0.01, 0.09, 0.59]],
        cct2=6504.0,
        matrix2=[[0.8, -0.25, -0.06], [-0.3, 1.05, 0.2], [-0.02, 0.08, 0.6]],
        cct3=None,
        matrix3=None,
    )


class MatrixHealthTests(unittest.TestCase):
    def test_dual_calibration_reports_interpolated_kappa(self) -> None:
        cal = _dual_calibration()
        health = matrix_health(cal, None, cct=6500.0)
        self.assertEqual(health["source"], "dng-dual")
        expected = float(
            np.linalg.cond(np.asarray(interpolated_color_matrix(cal, 6500.0)))
        )
        self.assertAlmostEqual(health["kappa"], expected, places=9)
        self.assertEqual(health["status"], "正常")

    def test_fleet_band_matrices_read_normal(self) -> None:
        from dngscan.camera_matrices import _FALLBACK_MATRICES

        for entry in _FALLBACK_MATRICES:
            m = np.asarray(entry["matrix"], dtype=np.float64) / 10000.0
            health = matrix_health(None, m)
            with self.subTest(label=entry["label"]):
                self.assertEqual(health["source"], "single-matrix")
                self.assertEqual(health["status"], "正常")
                self.assertLess(health["kappa"], 6.0)

    def test_elevated_and_anomalous_bands(self) -> None:
        base = np.diag([1.0, 1.0, 1.0])
        elevated = np.diag([7.0, 1.0, 1.0])  # kappa exactly 7
        self.assertEqual(matrix_health(None, elevated)["status"], "偏高")
        anomalous = np.diag([50.0, 1.0, 1.0])  # kappa 50
        self.assertEqual(matrix_health(None, anomalous)["status"], "异常")
        singular = np.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(matrix_health(None, singular)["status"], "异常")
        self.assertEqual(matrix_health(None, base)["status"], "正常")

    def test_no_calibration_returns_none(self) -> None:
        self.assertIsNone(matrix_health(None, None))


class MatrixHealthReportLineTests(unittest.TestCase):
    def test_line_states_source_kappa_and_degradation(self) -> None:
        from tests.golden_support import build_daylight_wide_dr
        from dngscan.report import matrix_health_line_cn

        scene = build_daylight_wide_dr()
        line = matrix_health_line_cn(scene.bundle)
        if getattr(scene.bundle, "wb_xyz_to_cam", None) is None:
            self.assertIn("缺失", line)
        else:
            self.assertIn("κ=", line)
            self.assertNotIn("置信度降低", line)
        degraded = dataclasses.replace(
            scene.bundle, wb_degradation="missing calibration"
        )
        line2 = matrix_health_line_cn(degraded)
        if getattr(degraded, "wb_xyz_to_cam", None) is not None:
            self.assertIn("置信度降低", line2)

    def test_kappa_is_evaluated_on_the_matrix_the_render_targets(self) -> None:
        """Third review F4: the line must come from the SAME ladder the hot-WB
        stage resolves. Declared Kelvin without DNG tags -> the evidence
        matrix at the declared CCT; daylight and camera -> target == decode
        matrix (no CCT); with DNG dual-illuminant tags a Kelvin mode reads
        the interpolated matrix and says so. All three branches run on a
        constructed bundle, none skips."""
        from unittest import mock

        from tests.golden_support import build_daylight_wide_dr
        from dngscan.camera_matrices import _FALLBACK_MATRICES
        from dngscan.report import matrix_health_line_cn

        scene = build_daylight_wide_dr()
        m = np.asarray(_FALLBACK_MATRICES[0]["matrix"], dtype=np.float64) / 10000.0
        base = dataclasses.replace(scene.bundle, wb_xyz_to_cam=m)
        with mock.patch("dngscan.raw_io.dng_metadata.read_dng_color_calibration", return_value=None):
            line = matrix_health_line_cn(dataclasses.replace(base, wb_mode="3200k"))
            self.assertIn("证据矩阵", line)
            self.assertIn("@ 3200K(声明)", line)
            day = matrix_health_line_cn(dataclasses.replace(base, wb_mode="daylight"))
            self.assertIn("(daylight: 目标=解码矩阵)", day)
            cam = matrix_health_line_cn(dataclasses.replace(base, wb_mode="camera"))
            self.assertIn("(camera: 目标=解码矩阵)", cam)
        cal = _dual_calibration()
        with mock.patch("dngscan.raw_io.dng_metadata.read_dng_color_calibration", return_value=cal):
            line = matrix_health_line_cn(dataclasses.replace(base, wb_mode="5500k"))
            self.assertIn("DNG双光源插值", line)
            self.assertIn("@ 5500K(声明)", line)
            # Self-review 2026-08-27 (P1): the evidence+cct rung targets the
            # D65-row-normalised interpolated matrix (the same normalisation
            # the decode side carries), so the line reports κ of THAT matrix,
            # not of the raw interpolation (2.33 vs 2.81 on the fallback set).
            from dngscan.raw_io import d65_row_normalize

            expected = float(np.linalg.cond(np.asarray(
                d65_row_normalize(interpolated_color_matrix(cal, 5500.0))
            )))
            self.assertIn(f"κ={expected:.2f}", line)


if __name__ == "__main__":
    unittest.main()
