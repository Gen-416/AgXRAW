# SPDX-License-Identifier: GPL-3.0-or-later
"""Full-resolution analysis survives JSON metadata from actual LibRaw bundles."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dngscan.analysis import analyze
from dngscan.gui import preview_cache as pc, service
from dngscan.raw_io import load_raw
from tests.test_pipeline_corrections import write_sensor_dng


class AnalysisCacheGeometryRoundtripTests(unittest.TestCase):
    def make_entry(self, path):
        write_sensor_dng(path, signal=1000, limit=.8)
        source = load_raw(path)
        self.assertIsInstance(source.scene_sensor_window_shape, tuple)
        self.assertEqual(source.coding_white_levels, [4095.])
        self.assertEqual(source.coding_black_levels, [0.] * 4)
        self.assertNotEqual(source.coding_white_levels, source.camera_white_levels)
        analysis, _, _ = analyze(source, 4)
        self.assertEqual(analysis.sensor_to_scene_ev_offset, 1.)
        entry = pc.build_proxy_entry(source, analysis)
        self.assertEqual(entry.bundle.coding_white_levels, source.coding_white_levels)
        self.assertEqual(entry.bundle.coding_black_levels, source.coding_black_levels)
        _, entry.cache_digest = pc._cache_identity(
            path, "clip", "camera", "libraw", "auto", "auto", "aligned", 4)
        return source, entry

    def cached_analysis(self, path, decoded, envelope=None):
        return service._cached_full_analysis(
            path, "clip", "camera", "libraw", "auto", "auto",
            decoded_bundle=decoded, envelope=envelope)

    def assert_same_analysis(self, expected, actual):
        self.assertIsNotNone(actual)
        self.assertEqual(actual.sensor_to_scene_ev_offset, 1.)
        # Cache deserialization intentionally restores SNR arrays as float32;
        # compare the actual persisted representation, including all fields.
        expected = pc._analysis_from_json(pc._analysis_to_json(expected))
        self.assertEqual(
            json.dumps(pc._analysis_to_json(expected), sort_keys=True),
            json.dumps(pc._analysis_to_json(actual), sort_keys=True))

    def test_actual_decode_hits_envelope_before_disk_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sensor.dng"
            source, entry = self.make_entry(path)
            envelope = service._preview_analysis_envelope(entry)
            decoded = load_raw(path)
            with patch.object(pc, "_cache_dir", return_value=root / "cache"):
                self.assert_same_analysis(
                    entry.analysis, self.cached_analysis(path, decoded, envelope))
                for field in ("coding_white_levels", "coding_black_levels"):
                    original = getattr(decoded, field)
                    setattr(decoded, field, [original[0] + 1, *original[1:]])
                    self.assertIsNone(self.cached_analysis(path, decoded, envelope))
                    setattr(decoded, field, original)
                # Geometry still qualifies the envelope: normalizing sequence
                # types must not make a genuinely different window reusable.
                decoded.scene_sensor_window_shape = (
                    source.scene_sensor_window_shape[0] - 2,
                    source.scene_sensor_window_shape[1])
                self.assertIsNone(self.cached_analysis(path, decoded, envelope))

    def test_actual_decode_hits_disk_and_restores_runtime_tuple(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sensor.dng"
            source, entry = self.make_entry(path)
            cache_path = root / "cache" / f"{entry.cache_digest}.npz"
            pc._write_disk_entry(cache_path, entry)
            restored = pc._read_disk_entry(cache_path, path, False)
            self.assertIsNotNone(restored)
            self.assertIsInstance(restored.bundle.scene_sensor_window_shape, tuple)
            self.assertEqual(restored.bundle.scene_sensor_window_shape,
                             source.scene_sensor_window_shape)
            self.assertEqual(restored.bundle.coding_white_levels, source.coding_white_levels)
            self.assertEqual(restored.bundle.coding_black_levels, source.coding_black_levels)
            self.assertEqual(restored.source_metadata, entry.source_metadata)
            decoded = load_raw(path)
            with patch.object(pc, "_cache_dir", return_value=cache_path.parent):
                self.assert_same_analysis(
                    entry.analysis, self.cached_analysis(path, decoded))
                for field in ("coding_white_levels", "coding_black_levels"):
                    original = getattr(decoded, field)
                    setattr(decoded, field, [original[0] + 1, *original[1:]])
                    self.assertIsNone(self.cached_analysis(path, decoded))
                    setattr(decoded, field, original)
                decoded.scene_sensor_window_shape = (
                    source.scene_sensor_window_shape[0],
                    source.scene_sensor_window_shape[1] - 2)
                self.assertIsNone(self.cached_analysis(path, decoded))


if __name__ == "__main__":
    unittest.main()
