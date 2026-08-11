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
from dngscan.film_optics_assets import (  # noqa: E402
    DEFAULT_PRINT_OPTICS,
    DEFAULT_STOCK_OPTICS,
    load_print_optics,
    load_stock_optics,
)

_GRAIN = load_stock_optics(DEFAULT_STOCK_OPTICS).grain
_HALATION = load_stock_optics(DEFAULT_STOCK_OPTICS).halation
_SCATTER = load_print_optics(DEFAULT_PRINT_OPTICS).print_scatter

ROOT = Path(__file__).resolve().parents[1]


class SamplingTruthTests(unittest.TestCase):
    def test_raw_field_input_is_measurably_wrong(self) -> None:
        """Pins WHY the contract matters: the wrong input is not a subtle
        drift, it is a different image (RMS ~1.0 on a unit-RMS field)."""
        from dngscan.film_optics import (
            GATE_H_MM,
            GATE_W_MM,
            FilmGeometry,
            grain_field_for,
            integral_from_field,
            sample_field,
        )

        field = grain_field_for(_GRAIN, 0)
        gh, gw = field.shape[:2]
        geo = FilmGeometry(gh, gw, w_mm=GATE_W_MM, h_mm=GATE_H_MM)
        right = sample_field(integral_from_field(field), geo)
        wrong = sample_field(field, geo)  # raw-field-on-purpose
        np.testing.assert_allclose(right, field, atol=2e-4)
        rms = float(np.sqrt(np.mean(np.square(wrong - field, dtype=np.float64))))
        self.assertGreater(
            rms, 0.1,
            "a raw field must NOT accidentally resemble the truth — that is "
            "what let the relation-only test pass",
        )

    def test_no_test_passes_a_raw_field_to_sample_field(self) -> None:
        """Every call site in the suite must pass a value that was DECLARED
        an integral image — either inline or through a variable bound to
        one earlier in the same file."""
        import re

        producers = ("integral_from_field", "_grain_ii_for")
        offenders = []
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            lines = path.read_text().splitlines()
            declared: set[str] = set()
            for idx, line in enumerate(lines, 1):
                bind = re.match(r"\s*(\w+)\s*=\s*(.+)$", line)
                if bind and any(fn in bind.group(2) for fn in producers):
                    declared.add(bind.group(1))
                # bare name only: the optics sampler. The appearance
                # kernel's fa._sample_field takes recipe tables, not
                # integral images (A6: the P2 overshoot gate calls it).
                call = re.search(r"(?<![\w.])sample_field\(\s*(\w+)?", line)
                if not call or "def " in line:
                    continue
                arg = call.group(1)
                if arg is None:      # argument on the next line
                    nxt = lines[idx].strip() if idx < len(lines) else ""
                    arg = re.match(r"(\w+)", nxt)
                    arg = arg.group(1) if arg else ""
                nxt_line = lines[idx] if idx < len(lines) else ""
                context = line + nxt_line
                if (
                    arg in declared
                    or any(fn in context for fn in producers)
                    or "raw-field-on-purpose" in context
                ):
                    continue
                offenders.append(f"{path.name}:{idx}")
        self.assertEqual(
            offenders, [],
            "these call sites pass something never declared an integral "
            "image to sample_field",
        )


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
    def test_design_doc_matches_the_shipped_tiers(self) -> None:
        from dngscan.render import OPTICS_BUDGET_TIERS_MIB

        doc = (ROOT / "docs" / "FILM_PRINT_RENDERING_PLAN.zh-CN.md").read_text()
        self.assertNotIn(
            "256/512/1024 MiB", doc,
            "the design doc still promises a tier the runtime removed",
        )
        self.assertIn("512/1024 MiB", doc)
        self.assertEqual(tuple(OPTICS_BUDGET_TIERS_MIB), (512, 1024))

    def test_context_lifecycle_is_documented(self) -> None:
        from dngscan.film_develop import FilmSpatialContext, prepare_film_spatial

        doc = FilmSpatialContext.__doc__ or ""
        for needle in ("finish_maps", "begin_bloom_source",
                       "accumulate_bloom_source", "finish_bloom_map"):
            self.assertIn(needle, doc, f"lifecycle must name {needle}")
        self.assertIn("HALATION ONLY", doc)
        self.assertIn("pass", (prepare_film_spatial.__doc__ or "").lower())


if __name__ == "__main__":
    unittest.main()
