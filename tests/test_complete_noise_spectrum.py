# SPDX-License-Identifier: GPL-3.0-or-later
"""Collector PSD units and persisted evidence, tested against spatial signals."""
from __future__ import annotations

from copy import deepcopy
import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import calibration, priors
from dngscan.noise_spectrum import (
    SCHEMA, PHASES, complete_axis_spectrum, dark_phase_mapping, validate_spectrum,
)
from dngscan.spectral_variance import measured_phase_detail_variance


def line_spectrum(plane, axis, window="none"):
    n = plane.shape[axis]
    centered = plane - plane.mean(axis=axis, keepdims=True)
    weights = np.hanning(n) if window == "hann" else np.ones(n)
    shape = [1, 1]
    shape[axis] = n
    transformed = np.fft.rfft(centered * weights.reshape(shape), axis=axis)
    powers = np.mean(np.abs(transformed) ** 2, axis=1 - axis) / (n * np.square(weights).sum())
    return powers, float(np.square(centered).mean())


def collector_fixture(*, n=128, window="none"):
    rng = np.random.default_rng(9164)
    planes = {}
    scalar_rows = []
    for phase, cid in zip(PHASES, (0, 1, 3, 2)):
        # Large row offsets prove whole-plane variance is the wrong h reference.
        single = rng.normal(0., 3., (96, n)) + rng.normal(0., 50., (96, 1))
        diff = rng.normal(0., 3., (96, n)) - rng.normal(0., 3., (96, n))
        planes[phase] = {"single": single, "diff": diff}
        scalar = {"ISO": "200", "Channel": phase, "ColorIndex": str(cid),
                  "BlackA": "100", "BlackB": "100", "StdDiffClipped": str(diff.std())}
        for direction, axis in (("Row", 1), ("Col", 0)):
            for source, plane in planes[phase].items():
                _, reference = line_spectrum(plane, axis, window)
                scalar[f"Within{direction}Var{source.title()}"] = str(reference)
        scalar_rows.append(scalar)
    axes = {}
    for name, axis in (("h", 1), ("v", 0)):
        length = next(iter(planes.values()))["single"].shape[axis]
        header = {"Format": "JPTC-SPECTRUM/1", "CfaPattern": "RGBG",
                  "Axis": "horizontal" if name == "h" else "vertical",
                  "TransformLength": str(length), "Window": window,
                  "Normalisation": "|Y(k)|^2 / (N * sum(w^2)), averaged over lines",
                  "OneSided": "bins 1..N/2-1 are NOT doubled. To integrate a column back to a",
                  "DiffPowerFactor": "2   (the difference spectrum is NOT halved here)",
                  "FreqUnit": f"cycles per channel-plane pixel (bin k -> k/{length}); x2 for sensor pixels"}
        rows = [{"bin": str(k), "freq": format(k / length, ".6g")}
                for k in range(length // 2 + 1)]
        for phase, sources in planes.items():
            for source, plane in sources.items():
                power, _ = line_spectrum(plane, axis, window)
                for row, p in zip(rows, power):
                    row[f"iso200_{phase}_{source}"] = format(float(p), ".6g")
        axes[name] = (header, rows)
    return axes, scalar_rows, planes


def measured_fixture(**kwargs):
    axes, scalars, planes = collector_fixture(**kwargs)
    spectrum = {"schema": SCHEMA, "mapping": dark_phase_mapping({"CfaPattern": "RGBG"}, scalars),
                "axes": {name: {"ratios_log2iso": {},
                                **complete_axis_spectrum(header, rows, name, scalar_rows=scalars)}
                         for name, (header, rows) in axes.items()}}
    return spectrum, axes, scalars, planes


def csv_text(header, rows):
    stream = io.StringIO()
    for key, value in header.items():
        stream.write(f"#{key}: {value}\n")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_collect(directory):
    axes, scalars, _ = collector_fixture()
    dark_header = {"Format": "JPTC-DARK/1", "Camera": "Review Sensor",
                   "CfaPattern": "RGBG", "ClipVarianceFactor": "1", "AdcStep": "0"}
    (directory / "dark-scalars.csv").write_text(csv_text(dark_header, scalars))
    for name, (header, rows) in axes.items():
        (directory / f"spectrum-{name}.csv").write_text(csv_text(header, rows))
    return axes, scalars


class CompleteNoiseSpectrumTests(unittest.TestCase):
    def test_rectangular_integral_matches_within_line_reference_even_and_odd(self):
        for n in (127, 128):
            spectrum, _, _, planes = measured_fixture(n=n)
            with self.subTest(length=n):
                validate_spectrum(spectrum)
                for name in ("h", "v"):
                    record = spectrum["axes"][name]
                    np.testing.assert_allclose(record["frequencies_sensor"],
                                               np.asarray(record["frequencies_channel_plane"]) / 2.)
                    for phase, item in record["measurements"][0]["phases"].items():
                        self.assertEqual(item["integration_status"], "verified-rectangular")
                        self.assertAlmostEqual(item["parseval_ratio_single"], 1., delta=2e-6)
                        self.assertAlmostEqual(item["parseval_ratio_diff"], 1., delta=2e-6)
                        self.assertAlmostEqual(item["temporal_variance_dn2"],
                                               item["reference_diff_within_line_variance_dn2"] / 2., delta=2e-5)
                        if name == "h":
                            self.assertGreater(planes[phase]["single"].var() /
                                               item["single_variance_dn2"], 50.)

    def test_import_registry_retains_powers_headers_mapping_and_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = directory / "collect"
            source.mkdir()
            axes, scalars = write_collect(source)
            with patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(directory / "store")):
                summary = calibration.import_calibration(source)
                stored = json.loads(Path(summary["path"]).read_text())["payload"]
                spectrum = stored["noise_spectrum"]
                self.assertEqual(spectrum["mapping"]["phases"]["C10"]["color"], "G")
                for name, (header, rows) in axes.items():
                    record = spectrum["axes"][name]
                    self.assertEqual(record["source_headers"], header)
                    self.assertEqual(record["source_sha256"], stored["source"]["inputs"][f"spectrum-{name}.csv"])
                    np.testing.assert_array_equal(record["measurements"][0]["phases"]["C00"]["diff_power"],
                                                  [float(row["iso200_C00_diff"]) for row in rows])
                # Runtime conversion is independent of whether a gain anchor exists.
                record, prior = calibration._read_record(Path(summary["path"]))
                self.assertEqual(prior["noise_spectrum"], spectrum)
                self.assertFalse(calibration.list_calibrations()[0].get("error"))

    def test_missing_contract_is_retained_but_cannot_qualify_detail_variance(self):
        spectrum, axes, scalars, _ = measured_fixture()
        for key in ("TransformLength", "Normalisation", "OneSided", "DiffPowerFactor"):
            header, rows = deepcopy(axes["h"])
            header.pop(key)
            spectrum["axes"]["h"] = {"ratios_log2iso": {},
                **complete_axis_spectrum(header, rows, "h", scalar_rows=scalars)}
            with self.subTest(missing=key):
                self.assertEqual(spectrum["axes"]["h"]["normalization_status"], "incomplete-contract")
                self.assertIn("diff_power", spectrum["axes"]["h"]["measurements"][0]["phases"]["C00"])
                with self.assertRaisesRegex(ValueError, "normalization"):
                    measured_phase_detail_variance(spectrum, "C00", 200, assume_separable=True)

    def test_factor_of_two_error_cannot_pass_parseval(self):
        spectrum, axes, scalars, _ = measured_fixture()
        header, rows = deepcopy(axes["h"])
        for row in rows:
            row["iso200_C00_diff"] = str(float(row["iso200_C00_diff"]) * 2.)
        spectrum["axes"]["h"] = {"ratios_log2iso": {},
            **complete_axis_spectrum(header, rows, "h", scalar_rows=scalars)}
        item = spectrum["axes"]["h"]["measurements"][0]["phases"]["C00"]
        self.assertEqual(item["integration_status"], "reference-mismatch")
        self.assertAlmostEqual(item["parseval_ratio_diff"], 2., delta=4e-6)
        with self.assertRaisesRegex(ValueError, "not qualified"):
            measured_phase_detail_variance(spectrum, "C00", 200, assume_separable=True)

    def test_missing_difference_reference_cannot_borrow_single_reference(self):
        spectrum, axes, scalars, _ = measured_fixture()
        for row in scalars:
            row.pop("WithinRowVarDiff")
        header, rows = axes["h"]
        spectrum["axes"]["h"] = {"ratios_log2iso": {},
            **complete_axis_spectrum(header, rows, "h", scalar_rows=scalars)}
        with self.assertRaisesRegex(ValueError, "difference spectrum.*reference"):
            measured_phase_detail_variance(spectrum, "C00", 200, assume_separable=True)

    def test_window_and_unmeasured_iso_need_explicit_bounded_choices(self):
        spectrum, _, _, _ = measured_fixture(window="hann")
        with self.assertRaisesRegex(ValueError, "window"):
            measured_phase_detail_variance(spectrum, "C00", 200, assume_separable=True)
        bands = measured_phase_detail_variance(spectrum, "C00", 200, assume_separable=True, allow_hann=True)
        self.assertEqual(set(bands), {0, 1, 2})
        self.assertTrue(all(value > 0 for value in bands.values()))
        with self.assertRaisesRegex(ValueError, "measured ISO"):
            measured_phase_detail_variance(spectrum, "C00", 201, assume_separable=True, allow_hann=True)

    def test_persisted_interpretation_cannot_be_tampered_separately_from_power(self):
        original, _, _, _ = measured_fixture()
        changes = {
            "sensor units": lambda axis: axis["frequencies_sensor"].__setitem__(1, .25),
            "transform": lambda axis: axis.__setitem__("transform_length", 64),
            "normalization": lambda axis: axis["source_headers"].__setitem__("Normalisation", "|Y(k)|^2 / sum(w^2)"),
            "headers type": lambda axis: axis.__setitem__("source_headers", "invalid"),
            "variance": lambda axis: axis["measurements"][0]["phases"]["C00"].__setitem__("temporal_variance_dn2", 123.),
            "ratio": lambda axis: axis["measurements"][0]["phases"]["C00"].__setitem__("parseval_ratio_diff", 123.),
            "qualification": lambda axis: axis["measurements"][0]["phases"]["C00"].__setitem__("integration_status", "unknown"),
        }
        for name, change in changes.items():
            spectrum = deepcopy(original)
            change(spectrum["axes"]["h"])
            with self.subTest(change=name), self.assertRaises(ValueError):
                validate_spectrum(spectrum)

    def test_bad_frequency_axis_and_ambiguous_scalar_reference_are_rejected(self):
        axes, scalars, _ = collector_fixture()
        header, rows = deepcopy(axes["h"])
        rows[2]["freq"] = rows[1]["freq"]
        with self.assertRaisesRegex(ValueError, "increasing"):
            complete_axis_spectrum(header, rows, "h", scalar_rows=scalars)
        header, rows = axes["h"]
        with self.assertRaisesRegex(ValueError, "axis declaration"):
            complete_axis_spectrum(header, rows, "v", scalar_rows=scalars)
        bad = dict(scalars[0], WithinRowVarDiff="999")
        with self.assertRaisesRegex(ValueError, "cannot identify"):
            complete_axis_spectrum(header, rows, "h", scalar_rows=scalars + [bad])

    def test_combined_record_size_rejected_before_writing_unreadable_store(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = directory / "collect"
            source.mkdir()
            write_collect(source)
            limit = max(path.stat().st_size for path in source.iterdir()) + 1
            with patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(directory / "store")), \
                    patch.object(calibration, "_MAX_JSON_BYTES", limit):
                with self.assertRaisesRegex(ValueError, "combined calibration record"):
                    calibration.import_calibration(source)
            self.assertFalse((directory / "store").exists())


if __name__ == "__main__":
    unittest.main()
