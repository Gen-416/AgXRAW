# SPDX-License-Identifier: GPL-3.0-or-later
"""RGB clip benchmark contracts using tiny inputs and mocked RAW rendering."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

from dngscan import _fast, analysis, raw_io
from tools import benchmark_loss_pipeline as pipeline_helper
from tools import benchmark_sensor_rgb as bench


ROOT = Path(__file__).resolve().parents[1]
METRICS = {1: 12.5, 2: 6.25, 3: 0.0}


class SensorRgbBenchmarkTests(unittest.TestCase):
    def invoke(self, args, *, identity=None, hook=None, fail_first=False,
               extension_available=True, kernel_available=True, inspect_input=None):
        native = mock.Mock(return_value=METRICS)
        if fail_first:
            native.side_effect = [RuntimeError("fixture kernel error"), METRICS]
        other = object()

        def kernel(name):
            if name != bench.KERNEL:
                return other
            return (native if kernel_available and
                    bench.KERNEL not in os.environ["DNGSCAN_FAST_SKIP"].split(",") else None)

        def compute(*a, **kw):
            if inspect_input is not None:
                inspect_input(*a, **kw)
            fn = _fast.kernel(bench.KERNEL)
            return fn() if fn is not None else METRICS.copy()

        def pipeline(parsed, record):
            self.assertEqual(os.environ["DNGSCAN_FAST"], "1")
            self.assertEqual(os.environ["DNGSCAN_FAST_SKIP"], bench.KERNEL if parsed.reference else "")
            self.assertIs(_fast.kernel("some_other_kernel"), other)
            raw_io.load_raw(parsed.source)
            raw_io.load_raw(parsed.source, _defer_clip_masks=False)
            if fail_first:
                with self.assertRaisesRegex(RuntimeError, "fixture kernel error"):
                    analysis.compute_color_clip_metrics()
            analysis.compute_color_clip_metrics()
            record["identity"] = copy.deepcopy(identity if identity is not None else {
                "masks_loaded": None, "masks_analyzed": "mask", "scene": "same"})
            if hook is not None:
                hook(parsed, record)

        extension = (SimpleNamespace(native_abi_version=lambda: 15)
                     if extension_available else None)
        clock = itertools.count(step=0.25)
        stream = io.StringIO()
        with mock.patch.object(sys, "argv", ["benchmark_sensor_rgb.py", *args]), \
             mock.patch.object(sys, "path", sys.path[:]), \
             mock.patch.dict(os.environ, {"DNGSCAN_FAST": "0", "DNGSCAN_FAST_SKIP": "other"}), \
             mock.patch.object(_fast, "_load_extension", return_value=extension) as ext_mock, \
             mock.patch.object(_fast, "kernel", side_effect=kernel), \
             mock.patch.object(analysis, "compute_color_clip_metrics", side_effect=compute), \
             mock.patch.object(raw_io, "load_raw", return_value=object()) as load, \
             mock.patch.object(pipeline_helper, "pipeline", side_effect=pipeline) as render, \
             mock.patch.object(bench.subprocess, "run", return_value=SimpleNamespace(stdout="fixture-commit\n")), \
             mock.patch.object(bench.platform, "platform", return_value="test-platform"), \
             mock.patch.object(bench.time, "perf_counter", side_effect=lambda: next(clock)), \
             redirect_stdout(stream), redirect_stderr(io.StringIO()):
            try:
                status = bench.main()
            finally:
                self.render_count = render.call_count
                self.extension_count = ext_mock.call_count
                self.load_calls = load.call_args_list
        return status, stream.getvalue()

    def source(self, directory):
        source = directory / "fixture.dng"
        source.write_bytes(b"fixture RAW identity; never decoded")
        return source

    def args(self, source, output, *extra):
        return ["--repo", str(ROOT), "--source", str(source), "--out", str(output), *extra]

    def test_single_kernel_ablation_preserves_deferred_load_on_both_sides(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            for reference in (False, True):
                with self.subTest(reference=reference):
                    output = directory / f"{reference}.json"
                    status, stdout = self.invoke(self.args(
                        source, output, *(("--reference",) if reference else ())))
                    self.assertEqual(status, 0)
                    self.assertEqual([call.kwargs for call in self.load_calls], [
                        {"_defer_clip_masks": True}, {"_defer_clip_masks": False}])
                    record = json.loads(output.read_text())
                    self.assertEqual(record["native_abi"], 15)
                    self.assertEqual(record["commit"], "fixture-commit")
                    self.assertEqual(record["environment"], {"DNGSCAN_FAST": "1",
                        "DNGSCAN_FAST_SKIP": bench.KERNEL if reference else ""})
                    calls = record["sensor_rgb_calls"]
                    self.assertEqual(calls["compute_color_clip_metrics"]["count"], 1)
                    self.assertEqual("native." + bench.KERNEL in calls, not reference)
                    self.assertEqual(json.loads(stdout)["sensor_rgb_calls"], calls)
                    self.assertEqual(record["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_failed_kernel_and_analysis_calls_are_counted(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            output = directory / "out.json"
            status, _ = self.invoke(self.args(self.source(directory), output), fail_first=True)
            self.assertEqual(status, 0)
            calls = json.loads(output.read_text())["sensor_rgb_calls"]
            for name in ("compute_color_clip_metrics", "native." + bench.KERNEL):
                self.assertEqual(calls[name]["count"], 2)
                self.assertGreater(calls[name]["wall_s"], 0)

    def test_complete_identity_including_loaded_masks_and_source_digest_is_compared(self):
        identity = {"masks_loaded": None, "masks_analyzed": "mask", "scene": "same"}
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            previous = directory / "previous.json"
            for index, (changed, previous_digest) in enumerate((
                    (False, digest), (True, digest), (False, "0" * 64), (False, None))):
                with self.subTest(index=index):
                    reference = {"identity": copy.deepcopy(identity)}
                    if changed:
                        reference["identity"]["masks_loaded"] = "different"
                    if previous_digest is not None:
                        reference["source_sha256"] = previous_digest
                    previous.write_text(json.dumps(reference))
                    output = directory / f"{index}.json"
                    status, _ = self.invoke(self.args(source, output, "--compare", str(previous)), identity=identity)
                    expected = not changed and previous_digest in (None, digest)
                    result = json.loads(output.read_text())
                    self.assertEqual(status, 0 if expected else 2)
                    self.assertEqual(result["comparison"]["exact"], expected)
                    self.assertNotIn("excluded_fields", result["comparison"])
                    self.assertEqual(json.loads(previous.read_text()), reference)

    def test_synthetic_inputs_are_deterministic_and_linear_colors_have_zero_strides(self):
        observed = []

        def inspect(raw, colors, thresholds, labels, pattern):
            self.assertEqual(raw.dtype.name, "uint16")
            self.assertEqual(colors.dtype.name, "uint8")
            self.assertEqual(colors.shape, raw.shape)
            self.assertEqual(set(label[0] for label in labels.values()), {"R", "G", "B"})
            self.assertTrue(set(labels).issubset(thresholds))
            if raw.ndim == 3:
                self.assertEqual(colors.strides[:2], (0, 0))
                self.assertFalse(colors.flags.writeable)
            observed.append((raw.shape, len(pattern), len(pattern[0])))

        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            first = directory / "reference.json"
            second = directory / "native.json"
            base = ["--repo", str(ROOT), "--synthetic", "12", "12", "--repeats", "2"]
            status, _ = self.invoke([*base, "--reference", "--out", str(first)], inspect_input=inspect)
            self.assertEqual(status, 0)
            self.assertEqual(self.render_count, 0)
            self.assertEqual(self.load_calls, [])
            status, _ = self.invoke([*base, "--out", str(second), "--compare", str(first)], inspect_input=inspect)
            self.assertEqual(status, 0)
            self.assertEqual(len(observed), 12)
            record = json.loads(second.read_text())
            self.assertEqual(set(record["identity"]), {"bayer", "xtrans", "linear_rgb"})
            self.assertEqual(record["sensor_rgb_calls"]["compute_color_clip_metrics"]["count"], 6)
            self.assertEqual(record["sensor_rgb_calls"]["native." + bench.KERNEL]["count"], 6)
            for case in record["synthetic_cases"].values():
                self.assertEqual(len(case["wall_s"]), 2)
                self.assertGreater(case["median_s"], 0)
            self.assertEqual(record["identity"]["linear_rgb"]["inputs"]["color_strides"][:2], [0, 0])

    def test_native_requires_kernel_but_reference_accepts_previous_abi_without_it(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            output = directory / "native.json"
            with self.assertRaisesRegex(RuntimeError, "native RGB clip kernel"):
                self.invoke(self.args(source, output), kernel_available=False)
            self.assertFalse(output.exists())
            self.assertEqual(self.render_count, 0)
            status, _ = self.invoke(self.args(source, output, "--reference"), kernel_available=False)
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.read_text())["native_abi"], 15)

    def test_matching_extension_is_required_even_for_reference(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            output = directory / "out.json"
            with self.assertRaisesRegex(RuntimeError, "matching native extension"):
                self.invoke(self.args(self.source(directory), output, "--reference"), extension_available=False)
            self.assertFalse(output.exists())
            self.assertEqual(self.render_count, 0)

    def test_existing_output_and_source_alias_are_rejected_before_native_loading(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            existing = directory / "existing.json"
            existing.write_text("keep me")
            for output in (source, existing):
                with self.subTest(output=output), self.assertRaises(SystemExit) as raised:
                    self.invoke(self.args(source, output))
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(self.extension_count, 0)
            self.assertEqual(existing.read_text(), "keep me")
            self.assertEqual(source.read_bytes(), b"fixture RAW identity; never decoded")

    def test_dangling_symlink_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            output = directory / "out.json"
            output.symlink_to(directory / "missing.json")
            with self.assertRaises(SystemExit):
                self.invoke(self.args(self.source(directory), output))
            self.assertTrue(output.is_symlink())
            self.assertFalse(output.exists())
            self.assertEqual(self.extension_count, 0)

    def test_racing_output_is_never_overwritten(self):
        def create_racer(parsed, _record):
            parsed.out.write_text("concurrent writer")

        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            output = directory / "out.json"
            with self.assertRaises(FileExistsError):
                self.invoke(self.args(self.source(directory), output), hook=create_racer)
            self.assertEqual(output.read_text(), "concurrent writer")

    def test_invalid_comparison_and_nonpositive_dimensions_fail_before_compute(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            previous = directory / "previous.json"
            output = directory / "out.json"
            for text in ("not json", "[]", '{"identity": null}', '{}'):
                previous.write_text(text)
                with self.subTest(text=text), self.assertRaises(SystemExit):
                    self.invoke(self.args(source, output, "--compare", str(previous)))
                self.assertEqual(self.extension_count, 0)
            for extra in (("--synthetic", "0", "12"), ("--synthetic", "12", "-1"),
                          ("--synthetic", "12", "12", "--repeats", "0")):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    self.invoke(["--repo", str(ROOT), "--out", str(output), *extra])
                self.assertEqual(self.extension_count, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
