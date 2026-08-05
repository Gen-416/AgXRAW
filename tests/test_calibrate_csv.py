# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for spectral calibration CSV ingestion."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import tempfile
import unittest
from pathlib import Path


def load_calibrator():
    path = Path(__file__).resolve().parents[1] / "tools" / "calibrate_skin_matrix.py"
    spec = importlib.util.spec_from_file_location("calibrate_skin_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CalibrationCsvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cal = load_calibrator()

    def test_curve_csv_with_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curve.csv"
            path.write_text(
                "wavelength_nm,R,G,B\n"
                "400,0.1,0.2,0.3\n"
                "700,0.7,0.8,0.9\n",
                encoding="utf-8",
            )
            data = self.cal.load_curve_csv(path)
        self.assertEqual(data.shape, (31, 3))
        self.assertAlmostEqual(float(data[0, 0]), 0.1)
        self.assertAlmostEqual(float(data[-1, 2]), 0.9)

    def test_spectra_wide_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wide.csv"
            rows = ["wavelength_nm,a,b"]
            for wl in self.cal.WL:
                rows.append(f"{wl:.0f},{wl / 1000:.4f},{wl / 2000:.4f}")
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            spectra = self.cal.load_spectra_csv(path)
        self.assertEqual(spectra.shape, (2, 31))
        self.assertAlmostEqual(float(spectra[0, 0]), 0.4)
        self.assertAlmostEqual(float(spectra[1, -1]), 0.35)

    def test_spectra_tidy_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tidy.csv"
            rows = ["sample,wavelength_nm,reflectance"]
            for name, scale in (("skin_a", 1000.0), ("skin_b", 2000.0)):
                for wl in self.cal.WL:
                    rows.append(f"{name},{wl:.0f},{wl / scale:.4f}")
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            spectra = self.cal.load_spectra_csv(path)
        self.assertEqual(spectra.shape, (2, 31))
        self.assertAlmostEqual(float(spectra[0, 0]), 0.4)
        self.assertAlmostEqual(float(spectra[1, -1]), 0.35)

    def test_spectra_dir_skips_non_spectral_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mask_to_surface.csv").write_text("mask,surface\n1,skin\n", encoding="utf-8")
            rows = ["wavelength_nm,reflectance"]
            for wl in self.cal.WL:
                rows.append(f"{wl:.0f},{wl / 1000:.4f}")
            (root / "skin.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                spectra = self.cal.load_spectra_dir(root)
        self.assertEqual(spectra.shape, (1, 31))
        self.assertAlmostEqual(float(spectra[0, -1]), 0.7)


if __name__ == "__main__":
    unittest.main()
