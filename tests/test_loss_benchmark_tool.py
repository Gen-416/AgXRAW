# SPDX-License-Identifier: GPL-3.0-or-later
"""Loss benchmark report/CLI contracts without RAW decoding or large buffers."""
from contextlib import redirect_stderr, redirect_stdout
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from dngscan import _fast
from tools import benchmark_loss_pipeline as bench


ROOT = Path(__file__).resolve().parents[1]


class LossBenchmarkArrayTests(unittest.TestCase):
    def test_noncontiguous_hash_preserves_every_bit_across_bands(self):
        bits = np.resize(np.array([
            0, 0x80000000, 0x3F800000, 0x7FC00001, 0x7FC12345,
            0x7F800000, 0xFF800000, 0x00800001,
        ], dtype=np.uint32), (137, 7, 3))
        for view in (bits.view(np.float32)[::-1, ::2, ::-1],
                     bits.view(np.float32).transpose(1, 0, 2)):
            self.assertFalse(view.flags.c_contiguous)
            record = bench.array_record(view)
            self.assertEqual(record["shape"], list(view.shape))
            self.assertEqual(record["dtype"], str(view.dtype))
            self.assertEqual(record["sha256"], hashlib.sha256(view.tobytes(order="C")).hexdigest())

    def test_hash_distinguishes_nan_payload_signed_zero_dtype_and_shape(self):
        bits = np.array([0x7FC00001, 0x80000000], np.uint32)
        original = bench.array_record(bits.view(np.float32))
        for replacement in ((0x7FC00002, 0x80000000), (0x7FC00001, 0)):
            different = np.array(replacement, np.uint32).view(np.float32)
            self.assertNotEqual(original["sha256"], bench.array_record(different)["sha256"])
        self.assertNotEqual(original, bench.array_record(bits))
        self.assertNotEqual(original, bench.array_record(bits.view(np.float32).reshape(1, 2)))

    def test_none_scalar_and_empty_have_unambiguous_records(self):
        self.assertIsNone(bench.array_record(None))
        for array in (np.float16(-0.), np.empty((0, 3), np.float32)):
            value = np.asarray(array)
            record = bench.array_record(array)
            self.assertEqual(record["shape"], list(value.shape))
            self.assertEqual(record["sha256"], hashlib.sha256(value.tobytes()).hexdigest())

    def test_json_nonfinite_scalars_are_stable_strict_json(self):
        @dataclass
        class Decision:
            unavailable: float
            limits: tuple
            count: object

        value = {
            1: Decision(float("nan"), (float("inf"), np.float32(-np.inf)), np.int64(3)),
            "path": Path("report.json"),
            "array": np.array([np.nan, -0.], np.float16),
        }
        converted = bench.json_value(value)
        text = json.dumps(converted, allow_nan=False, sort_keys=True)
        self.assertEqual(json.loads(text), converted)
        self.assertEqual(converted["1"], {"unavailable": "nan", "limits": ["inf", "-inf"], "count": 3})
        self.assertEqual(text, json.dumps(bench.json_value(copy.deepcopy(value)), allow_nan=False, sort_keys=True))

    def test_tiny_synthetic_covers_all_six_cases(self):
        record = {}
        with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "0"}):
            bench.synthetic(SimpleNamespace(synthetic=(7, 9), repeats=1), record)
        expected = {operation + "_" + dtype for operation in ("crop", "merge_same", "merge_resized")
                    for dtype in ("float16", "float32")}
        self.assertEqual(set(record["identity"]), expected)
        self.assertEqual(set(record["cases"]), expected)
        for key in expected:
            self.assertEqual(record["identity"][key]["shape"], [9, 7, 3])
            self.assertEqual(len(record["cases"][key]["seconds"]), 1)


class LossBenchmarkCliTests(unittest.TestCase):
    def invoke(self, args, *, identity=None, compute=None):
        def measured(parsed, record):
            self.assertEqual(os.environ["DNGSCAN_FAST"], "1")
            self.assertEqual(os.environ["DNGSCAN_FAST_SKIP"],
                             ",".join(bench.LOSS_KERNELS) if parsed.reference else "")
            record.update(identity=copy.deepcopy(identity or {"mask": "same"}), cases={})
            if compute is not None:
                compute(parsed, record)

        stream = io.StringIO()
        with mock.patch.object(sys, "argv", ["benchmark_loss_pipeline.py", *args]), \
             mock.patch.object(sys, "path", sys.path[:]), \
             mock.patch.dict(os.environ, {}), \
             mock.patch.object(_fast, "_load_extension", return_value=SimpleNamespace(native_abi_version=lambda: 15)) as ext, \
             mock.patch.object(bench.subprocess, "run", return_value=SimpleNamespace(stdout="fixture-commit\n")), \
             mock.patch.object(bench.platform, "platform", return_value="test-platform"), \
             mock.patch.object(bench, "synthetic", side_effect=measured) as synthetic, \
             mock.patch.object(bench, "pipeline", side_effect=measured) as pipeline, \
             redirect_stdout(stream), redirect_stderr(io.StringIO()):
            try:
                result = bench.main()
            finally:
                self.last_extension_calls = ext.call_count
                self.last_compute_calls = synthetic.call_count + pipeline.call_count
        return result, stream.getvalue()

    def args(self, out, *extra):
        return ["--repo", str(ROOT), "--synthetic", "7", "9", "--repeats", "1",
                "--out", str(out), *extra]

    def test_same_identity_passes_and_different_identity_returns_two(self):
        original = {"mask": bench.array_record(np.array([0., -0.], np.float16)),
                    "decisions": {"unavailable": "nan", "headroom": 1.}}
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            reference = directory / "reference.json"
            reference.write_text(json.dumps({"identity": original}, allow_nan=False))
            for changed in (False, True):
                identity = copy.deepcopy(original)
                if changed:
                    identity["mask"] = bench.array_record(np.array([0., 0.], np.float16))
                out = directory / f"result-{changed}.json"
                status, _ = self.invoke(self.args(out, "--compare", str(reference)), identity=identity)
                report = json.loads(out.read_text())
                self.assertEqual(status, 2 if changed else 0)
                self.assertEqual(report["comparison"]["exact"], not changed)
                self.assertEqual(report["identity"], identity)
            self.assertEqual(json.loads(reference.read_text()), {"identity": original})

    def test_reference_mode_disables_only_named_loss_kernels(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "reference.json"
            status, output = self.invoke(self.args(out, "--reference"))
            self.assertEqual(status, 0)
            self.assertTrue(json.loads(output)["reference"])
            self.assertTrue(json.loads(out.read_text())["reference"])
            self.assertEqual(self.last_compute_calls, 1)

    def test_existing_output_or_source_alias_is_rejected_before_compute(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.dng"
            source.write_bytes(b"source must remain unchanged")
            cases = (
                self.args(source),
                ["--repo", str(ROOT), "--source", str(source), "--out", str(source)],
            )
            for args in cases:
                with self.subTest(args=args), self.assertRaises(SystemExit) as caught:
                    self.invoke(args)
                self.assertEqual(caught.exception.code, 2)
                self.assertEqual(self.last_compute_calls, 0)
                self.assertEqual(self.last_extension_calls, 0)
                self.assertEqual(source.read_bytes(), b"source must remain unchanged")

    def test_bad_cli_values_fail_before_extension_or_compute(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.json"
            for extra in (("--repeats", "0"), ("--synthetic", "3", "9"),
                          ("--repo", str(Path(td) / "missing-repo"))):
                with self.subTest(extra=extra), self.assertRaises(SystemExit) as caught:
                    self.invoke(self.args(out, *extra))
                self.assertEqual(caught.exception.code, 2)
                self.assertEqual(self.last_compute_calls, 0)
                self.assertEqual(self.last_extension_calls, 0)
                self.assertFalse(out.exists())

    def test_invalid_reference_is_rejected_before_compute(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            reference = directory / "reference.json"
            for content in (None, "invalid json", "{}", '{"identity": null}'):
                with self.subTest(content=content):
                    reference.unlink(missing_ok=True)
                    if content is not None:
                        reference.write_text(content)
                    out = directory / "out.json"
                    with self.assertRaises(SystemExit) as caught:
                        self.invoke(self.args(out, "--compare", str(reference)))
                    self.assertEqual(caught.exception.code, 2)
                    self.assertEqual(self.last_compute_calls, 0)
                    self.assertFalse(out.exists())

    def test_dangling_output_symlink_is_rejected_without_creating_target(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "must-not-create.json"
            out = Path(td) / "out.json"
            out.symlink_to(target)
            with self.assertRaises(SystemExit) as caught:
                self.invoke(self.args(out))
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(self.last_compute_calls, 0)
            self.assertTrue(out.is_symlink())
            self.assertFalse(target.exists())

    def test_output_created_during_measurement_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.json"

            def create_output(parsed, record):
                parsed.out.write_bytes(b"created by another process")

            with self.assertRaises(FileExistsError):
                self.invoke(self.args(out), compute=create_output)
            self.assertEqual(self.last_compute_calls, 1)
            self.assertEqual(out.read_bytes(), b"created by another process")


if __name__ == "__main__":
    unittest.main()
