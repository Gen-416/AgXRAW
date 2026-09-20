# SPDX-License-Identifier: GPL-3.0-or-later
"""Channel benchmark contracts without RAW rendering or large allocations."""
from contextlib import redirect_stderr, redirect_stdout
import copy
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

from dngscan import _fast, analysis, raw_io
from tools import benchmark_loss_pipeline as pipeline_helper
from tools import benchmark_sensor_channels as bench


ROOT = Path(__file__).resolve().parents[1]


class SensorChannelBenchmarkTests(unittest.TestCase):
    def invoke(self, args, *, identity=None, hook=None, fail_index=None,
               extension_available=True, missing_kernels=(), inspect_input=None,
               threshold_cost=0.0):
        natives = {name: mock.Mock(return_value=None) for name in bench.KERNELS}
        if fail_index is not None:
            natives[bench.KERNELS[fail_index]].side_effect = [RuntimeError("fixture scan error"), None]
        rgb = object()
        clock = [0.0]

        def timer():
            clock[0] += .001
            return clock[0]

        def kernel(name):
            if name not in bench.KERNELS:
                return rgb
            return (natives[name] if name not in missing_kernels and
                    name not in os.environ["DNGSCAN_FAST_SKIP"].split(",") else None)

        def compute(index, *a, **kw):
            if inspect_input is not None:
                inspect_input(index, *a, **kw)
            fn = _fast.kernel(bench.KERNELS[index])
            if fn is not None:
                fn()
            ids = a[2] if len(a) > 2 else [0, 1, 2]
            if index == 0:
                return ({cid: 3 if cid == 0 else 16383 for cid in ids},
                        {cid: 17 for cid in ids}, {cid: 23 for cid in ids},
                        {cid: False for cid in ids})
            return {cid: 12.5 for cid in ids}

        def pipeline(parsed, record):
            self.assertEqual(os.environ["DNGSCAN_FAST"], "1")
            self.assertEqual(os.environ["DNGSCAN_FAST_SKIP"],
                             ",".join(bench.KERNELS) if parsed.reference else "")
            self.assertIs(_fast.kernel("sensor_rgb_clip_counts_u16"), rgb)
            raw_io.load_raw(parsed.source)
            raw_io.load_raw(parsed.source, _defer_clip_masks=False)
            if fail_index is not None:
                with self.assertRaisesRegex(RuntimeError, "fixture scan error"):
                    getattr(analysis, bench.ENTRYPOINTS[fail_index])()
            for name in bench.ENTRYPOINTS:
                getattr(analysis, name)()
            record["identity"] = copy.deepcopy(identity if identity is not None else {
                "masks_loaded": None, "masks_analyzed": "mask", "scene": "same"})
            if hook is not None:
                hook(parsed, record)

        make_thresholds = analysis.channel_clip_thresholds

        def thresholds(*a, **kw):
            clock[0] += threshold_cost
            return make_thresholds(*a, **kw)

        extension = (SimpleNamespace(native_abi_version=lambda: 16)
                     if extension_available else None)
        stream = io.StringIO()
        with mock.patch.object(sys, "argv", ["benchmark_sensor_channels.py", *args]), \
             mock.patch.object(sys, "path", sys.path[:]), \
             mock.patch.dict(os.environ, {"DNGSCAN_FAST": "0", "DNGSCAN_FAST_SKIP": "unrelated"}), \
             mock.patch.object(_fast, "_load_extension", return_value=extension) as ext_mock, \
             mock.patch.object(_fast, "kernel", side_effect=kernel), \
             mock.patch.object(analysis, "detect_ceilings", side_effect=lambda *a, **kw: compute(0, *a, **kw)), \
             mock.patch.object(analysis, "compute_clip_pct_by_thresholds", side_effect=lambda *a, **kw: compute(1, *a, **kw)), \
             mock.patch.object(analysis, "channel_clip_thresholds", side_effect=thresholds) as prep, \
             mock.patch.object(raw_io, "load_raw", return_value=object()) as load, \
             mock.patch.object(pipeline_helper, "pipeline", side_effect=pipeline) as render, \
             mock.patch.object(bench.subprocess, "run", return_value=SimpleNamespace(stdout="fixture-commit\n")), \
             mock.patch.object(bench.platform, "platform", return_value="test-platform"), \
             mock.patch.object(bench.time, "perf_counter", side_effect=timer), \
             redirect_stdout(stream), redirect_stderr(io.StringIO()):
            try:
                status = bench.main()
            finally:
                self.render_count = render.call_count
                self.extension_count = ext_mock.call_count
                self.load_calls = load.call_args_list
                self.threshold_calls = prep.call_count
        return status, stream.getvalue()

    def source(self, directory):
        source = directory / "fixture.dng"
        source.write_bytes(b"fixture RAW identity; never decoded")
        return source

    def args(self, source, output, *extra):
        return ["--repo", str(ROOT), "--source", str(source), "--out", str(output), *extra]

    def test_two_kernel_ablation_preserves_rgb_native_and_deferred_load(self):
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
                    self.assertEqual(record["native_abi"], 16)
                    self.assertEqual(record["commit"], "fixture-commit")
                    self.assertEqual(record["environment"], {"DNGSCAN_FAST": "1",
                        "DNGSCAN_FAST_SKIP": ",".join(bench.KERNELS) if reference else ""})
                    calls = record["sensor_channel_calls"]
                    for name in bench.ENTRYPOINTS:
                        self.assertEqual(calls[name]["count"], 1)
                    for name in bench.KERNELS:
                        self.assertEqual("native." + name in calls, not reference)
                    self.assertEqual(json.loads(stdout)["sensor_channel_calls"], calls)
                    self.assertEqual(record["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_failed_native_and_entry_calls_are_counted_for_each_kernel(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            for failed in (0, 1):
                with self.subTest(failed=failed):
                    output = directory / f"{failed}.json"
                    status, _ = self.invoke(self.args(source, output), fail_index=failed)
                    self.assertEqual(status, 0)
                    calls = json.loads(output.read_text())["sensor_channel_calls"]
                    for index, (entry, native) in enumerate(zip(bench.ENTRYPOINTS, bench.KERNELS)):
                        for name in (entry, "native." + native):
                            self.assertEqual(calls[name]["count"], 2 if index == failed else 1)
                            self.assertGreater(calls[name]["wall_s"], 0)

    def test_complete_identity_and_optional_source_digest_control_comparison(self):
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
                    record = json.loads(output.read_text())
                    self.assertEqual(status, 0 if expected else 2)
                    self.assertEqual(record["comparison"]["exact"], expected)
                    self.assertNotIn("excluded_fields", record["comparison"])
                    self.assertEqual(json.loads(previous.read_text()), reference)

    def test_synthetic_inputs_are_exact_and_threshold_preparation_is_not_timed(self):
        observed = []

        def inspect(index, raw, colors, ids, levels):
            self.assertEqual(raw.dtype.name, "uint16")
            self.assertEqual(colors.dtype.name, "uint8")
            self.assertEqual(colors.shape, raw.shape)
            self.assertEqual(ids, sorted(levels))
            if raw.ndim == 3:
                self.assertEqual(colors.strides[:2], (0, 0))
                self.assertFalse(colors.flags.writeable)
            if index == 0:
                self.assertEqual(levels, dict.fromkeys(ids, 16383))
            else:
                self.assertEqual(levels, {cid: 0 if cid == 0 else 16379 for cid in ids})
            observed.append(index)

        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            first, second = directory / "reference.json", directory / "native.json"
            base = ["--repo", str(ROOT), "--synthetic", "12", "12", "--repeats", "2"]
            status, _ = self.invoke([*base, "--reference", "--out", str(first)],
                                    inspect_input=inspect, threshold_cost=100)
            self.assertEqual(status, 0)
            self.assertEqual(self.render_count, 0)
            self.assertEqual(self.load_calls, [])
            self.assertEqual(self.threshold_calls, 3)
            status, _ = self.invoke([*base, "--out", str(second), "--compare", str(first)],
                                    inspect_input=inspect, threshold_cost=100)
            self.assertEqual(status, 0)
            self.assertEqual(observed, [0, 0, 1, 1] * 6)
            record = json.loads(second.read_text())
            self.assertEqual(set(record["identity"]), {"bayer", "xtrans", "linear_rgb"})
            for name in (*bench.ENTRYPOINTS, *("native." + k for k in bench.KERNELS)):
                self.assertEqual(record["sensor_channel_calls"][name]["count"], 6)
            for case in record["synthetic_cases"].values():
                self.assertEqual(set(case), set(bench.ENTRYPOINTS))
                for timing in case.values():
                    self.assertEqual(len(timing["wall_s"]), 2)
                    self.assertGreater(timing["median_s"], 0)
                    self.assertLess(timing["median_s"], 1)
            for case in record["identity"].values():
                self.assertTrue({"ceilings", "exact_counts", "near_counts", "spike_ok",
                                 "clip_pct_by_channel", "thresholds", "inputs"}.issubset(case))
                self.assertEqual(case["thresholds"]["0"], 0)

    def test_native_requires_each_kernel_but_old_reference_can_omit_both(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            output = directory / "out.json"
            for missing in bench.KERNELS:
                with self.subTest(missing=missing), self.assertRaisesRegex(RuntimeError, missing):
                    self.invoke(self.args(source, output), missing_kernels=(missing,))
                self.assertFalse(output.exists())
                self.assertEqual(self.render_count, 0)
            status, _ = self.invoke(self.args(source, output, "--reference"), missing_kernels=bench.KERNELS)
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.read_text())["native_abi"], 16)

    def test_subperiod_synthetic_sizes_select_only_visible_channel_ids(self):
        import numpy as np

        def inspect(_index, _raw, colors, ids, _levels):
            self.assertEqual(ids, sorted(int(cid) for cid in np.unique(colors)))

        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            for width, height in ((1, 1), (1, 2), (2, 1)):
                with self.subTest(width=width, height=height):
                    output = directory / f"{width}x{height}.json"
                    status, _ = self.invoke([
                        "--repo", str(ROOT), "--synthetic", str(width), str(height),
                        "--repeats", "1", "--reference", "--out", str(output)],
                        inspect_input=inspect)
                    self.assertEqual(status, 0)
                    identity = json.loads(output.read_text())["identity"]
                    if width == height == 1:
                        self.assertEqual(identity["bayer"]["inputs"]["channel_ids"], [0])
                        self.assertEqual(identity["xtrans"]["inputs"]["channel_ids"], [1])
                        self.assertEqual(identity["linear_rgb"]["inputs"]["channel_ids"], [0, 1, 2])

    def test_reference_requires_matching_extension(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            output = directory / "out.json"
            with self.assertRaisesRegex(RuntimeError, "matching native extension"):
                self.invoke(self.args(self.source(directory), output, "--reference"), extension_available=False)
            self.assertFalse(output.exists())
            self.assertEqual(self.render_count, 0)

    def test_existing_output_source_alias_and_dangling_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            existing = directory / "existing.json"
            existing.write_text("keep me")
            link = directory / "link.json"
            link.symlink_to(directory / "missing.json")
            for output in (source, existing, link):
                with self.subTest(output=output), self.assertRaises(SystemExit) as raised:
                    self.invoke(self.args(source, output))
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(self.extension_count, 0)
            self.assertEqual(existing.read_text(), "keep me")
            self.assertEqual(source.read_bytes(), b"fixture RAW identity; never decoded")
            self.assertTrue(link.is_symlink())
            self.assertFalse(link.exists())

    def test_racing_output_is_not_overwritten(self):
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
            previous, output = directory / "previous.json", directory / "out.json"
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
