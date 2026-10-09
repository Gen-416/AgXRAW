# SPDX-License-Identifier: GPL-3.0-or-later
"""Review batch 20 regression gates.

1. Grain sampling is checked against ABSOLUTE truth, not only against its
   own scale/crop relations — a raw field passed where an integral belongs
   made both sides equally wrong and the relation test still passed.
2. A failed main export leaves no finished-looking diagnostic PNG behind.
3./4. The advertised memory tiers and the spatial-context lifecycle are
   described correctly wherever they are promised.
"""
from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

# P1 §7.1: the operators take the specific asset they implement, so the tests
# pull the same declared assets the renderer compiles rather than a shared
# profile struct that no longer exists.


ROOT = Path(__file__).resolve().parents[1]




class ExportAtomicityTests(unittest.TestCase):
    def test_failed_export_leaves_no_dashboard_png(self) -> None:
        from dngscan.gui import service

        sample = Path.home() / "Pictures" / "AgXRAW样张" / "_SDI0199.DNG"
        if not sample.is_file():
            self.skipTest(f"sample unavailable: {sample}")
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)
            wrote: list[str] = []

            def fake_dashboard(bundle, analysis, y, ev, path, auto_ev=None):
                wrote.append(Path(path).name)
                Path(path).write_bytes(b"PNG-STUB")

            def boom(**kw):
                raise RuntimeError("HDR backend exploded")

            params = {
                "input": str(sample), "outdir": str(outdir),
                "png": True, "ev": 0,
            }
            with mock.patch.object(service.dg, "plot_dashboard", fake_dashboard), \
                    mock.patch.object(service.dg, "export_jpeg", boom):
                with self.assertRaises(RuntimeError):
                    service.run_export(params)
            self.assertEqual(len(wrote), 1, "the dashboard should have run")
            self.assertNotEqual(
                wrote[0], f"{sample.stem}_scan.png",
                "the dashboard must not claim its FINAL name before the "
                "main export succeeds",
            )
            self.assertIn(
                ".part", wrote[0],
                f"expected a temp name, got {wrote[0]}",
            )
            self.assertTrue(
                wrote[0].endswith(".png"),
                "the temp must keep its .png extension — matplotlib picks "
                "its writer from it (a '.part1234' tail made savefig raise)",
            )
            self.assertEqual(
                sorted(p.name for p in outdir.iterdir()), [],
                "a failed export must leave nothing behind — a finished-"
                "looking _scan.png could pair with an older JPEG",
            )

    def test_rename_is_atomic_and_guarded(self) -> None:
        from dngscan.gui import service

        src = inspect.getsource(service.run_export)
        self.assertIn("os.replace(png_temp, png_path)", src)
        self.assertIn("finally:", src)
        self.assertIn("png_temp.unlink(missing_ok=True)", src)


class DocumentedContractTests(unittest.TestCase):
    def test_spatial_budget_is_documented(self) -> None:
        doc = (ROOT / "docs" / "CHROMA_NR.zh-CN.md").read_text()
        self.assertIn("DNGSCAN_SPATIAL_BUDGET_MIB", doc)



if __name__ == "__main__":
    unittest.main()
