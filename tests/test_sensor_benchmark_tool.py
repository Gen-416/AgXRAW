# SPDX-License-Identifier: GPL-3.0-or-later
"""Sensor-summary benchmark CLI contracts without loading or rendering a RAW."""
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

from dngscan import _fast, analysis
from tools import benchmark_loss_pipeline as pipeline_helper
from tools import benchmark_sensor_summary as bench


ROOT = Path(__file__).resolve().parents[1]


class SensorBenchmarkToolTests(unittest.TestCase):
    def invoke(self, args, *, identity=None, compute=None, missing_summary=None,
               fail_first_detect=False, extension_available=True):
        cache = {}
        signature = mock.Mock(return_value="immutable-source")
        summary = SimpleNamespace(_cache_signature=signature)

        def summarize(*_args, **_kwargs):
            key = summary._cache_signature("sensor")
            if key is not None and key in cache:
                return cache[key]
            analysis.channel_saturation_levels()
            result = analysis.detect_ceilings()
            analysis.resolve_fullwell()
            if key is not None:
                cache[key] = result
            return result

        summary.summarize_sensor = summarize

        def import_summary(name):
            self.assertEqual(name, "dngscan.sensor_summary")
            if missing_summary is not None:
                raise ModuleNotFoundError("synthetic missing module", name=missing_summary)
            return summary

        def pipeline(parsed, record):
            self.assertEqual(os.environ["DNGSCAN_FAST"], "1")
            self.assertEqual(os.environ["DNGSCAN_FAST_SKIP"], "")
            if missing_summary is None:
                # Both modes instrument entry calls; only the reference replaces
                # cache eligibility. All native dispatch remains enabled above.
                self.assertIsNot(summary.summarize_sensor, summarize)
                if parsed.reference:
                    self.assertIsNot(summary._cache_signature, signature)
                else:
                    self.assertIs(summary._cache_signature, signature)
            record["identity"] = copy.deepcopy(identity if identity is not None else {"scene": "same"})
            if compute is not None:
                compute(parsed, record, summary)
            elif missing_summary is None:
                summary.summarize_sensor("sensor")
                summary.summarize_sensor("sensor")
            else:
                analysis.detect_ceilings()

        native = SimpleNamespace(native_abi_version=lambda: 15) if extension_available else None
        detect_effect = [RuntimeError("synthetic failed reduction"), {"ceiling": 42}] if fail_first_detect else None
        clock = itertools.count(step=0.25)
        stream = io.StringIO()
        with mock.patch.object(sys, "argv", ["benchmark_sensor_summary.py", *args]), \
             mock.patch.object(sys, "path", sys.path[:]), \
             mock.patch.dict(os.environ, {"DNGSCAN_FAST": "0", "DNGSCAN_FAST_SKIP": "unrelated_kernel"}), \
             mock.patch.object(_fast, "_load_extension", return_value=native) as extension, \
             mock.patch.object(bench.importlib, "import_module", side_effect=import_summary), \
             mock.patch.object(bench.subprocess, "run", return_value=SimpleNamespace(stdout="fixture-commit\n")), \
             mock.patch.object(bench.platform, "platform", return_value="test-platform"), \
             mock.patch.object(bench.time, "perf_counter", side_effect=lambda: next(clock)), \
             mock.patch.object(analysis, "channel_saturation_levels", return_value={}), \
             mock.patch.object(analysis, "detect_ceilings", return_value={"ceiling": 42}, side_effect=detect_effect), \
             mock.patch.object(analysis, "resolve_fullwell", return_value={}), \
             mock.patch.object(pipeline_helper, "pipeline", side_effect=pipeline) as compute_mock, \
             redirect_stdout(stream), redirect_stderr(io.StringIO()):
            try:
                status = bench.main()
            finally:
                self.last_compute_calls = compute_mock.call_count
                self.last_extension_calls = extension.call_count
                self.last_signature_calls = signature.call_count
        self.assertIs(summary.summarize_sensor, summarize)
        self.assertIs(summary._cache_signature, signature)
        return status, stream.getvalue()

    def args(self, source, out, *extra, repo=ROOT):
        return ["--repo", str(repo), "--source", str(source), "--out", str(out), *extra]

    def source(self, directory):
        source = directory / "fixture.dng"
        source.write_bytes(b"small synthetic RAW identity; never decoded")
        return source

    def test_reference_bypasses_only_summary_cache_and_keeps_all_native_kernels(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            for reference in (False, True):
                with self.subTest(reference=reference):
                    out = directory / f"report-{reference}.json"
                    extra = ("--reference",) if reference else ()
                    status, stdout = self.invoke(self.args(source, out, *extra))
                    self.assertEqual(status, 0)
                    record = json.loads(out.read_text())
                    self.assertEqual(record["reference"], reference)
                    self.assertTrue(record["summary_available"])
                    self.assertEqual(record["native_abi"], 15)
                    self.assertEqual(record["commit"], "fixture-commit")
                    calls = record["sensor_calls"]
                    self.assertEqual(calls["summarize_sensor"]["count"], 2)
                    for name in ("channel_saturation_levels", "detect_ceilings", "resolve_fullwell"):
                        self.assertEqual(calls[name], {"count": 2 if reference else 1,
                                                     "wall_s": 0.5 if reference else 0.25})
                    self.assertEqual(calls["summarize_sensor"]["wall_s"], 3.5 if reference else 2.0)
                    self.assertEqual(self.last_signature_calls, 0 if reference else 2)
                    self.assertEqual(json.loads(stdout)["sensor_calls"], calls)

    def test_failed_reduction_and_failed_summary_still_increment_counts(self):
        def recover(_args, _record, summary):
            with self.assertRaisesRegex(RuntimeError, "synthetic failed reduction"):
                summary.summarize_sensor("sensor")
            summary.summarize_sensor("sensor")

        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            out = directory / "report.json"
            status, _ = self.invoke(self.args(self.source(directory), out), compute=recover,
                                    fail_first_detect=True)
            self.assertEqual(status, 0)
            calls = json.loads(out.read_text())["sensor_calls"]
            self.assertEqual(calls["summarize_sensor"], {"count": 2, "wall_s": 3.0})
            self.assertEqual(calls["detect_ceilings"], {"count": 2, "wall_s": 0.5})
            self.assertEqual(calls["channel_saturation_levels"]["count"], 2)
            self.assertEqual(calls["resolve_fullwell"]["count"], 1)

    def test_identity_and_optional_raw_digest_both_control_comparison_exit_code(self):
        identity = {"scene": "same", "decision": {"headroom": 2.0, "unavailable": "nan"}}
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            reference = directory / "reference.json"
            for case, changed_identity, previous_digest in (
                ("same", False, digest), ("different-buffer", True, digest),
                ("different-raw", False, "0" * 64), ("legacy-no-hash", False, None),
            ):
                with self.subTest(case=case):
                    previous = {"identity": identity}
                    if previous_digest is not None:
                        previous["source_sha256"] = previous_digest
                    reference.write_text(json.dumps(previous))
                    current = copy.deepcopy(identity)
                    if changed_identity:
                        current["scene"] = "different"
                    out = directory / f"{case}.json"
                    status, _ = self.invoke(self.args(source, out, "--compare", str(reference)), identity=current)
                    expected = not changed_identity and previous_digest in (None, digest)
                    record = json.loads(out.read_text())
                    self.assertEqual(status, 0 if expected else 2)
                    self.assertEqual(record["comparison"]["exact"], expected)
                    self.assertEqual(record["source_sha256"], digest)
                    self.assertEqual(json.loads(reference.read_text()), previous)

    def test_existing_output_and_raw_source_alias_are_rejected_before_compute(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            out = directory / "report.json"
            out.write_bytes(b"existing report")
            for target in (out, source):
                with self.subTest(target=target):
                    original = target.read_bytes()
                    with self.assertRaises(SystemExit) as caught:
                        self.invoke(self.args(source, target))
                    self.assertEqual(caught.exception.code, 2)
                    self.assertEqual(self.last_compute_calls, 0)
                    self.assertEqual(self.last_extension_calls, 0)
                    self.assertEqual(target.read_bytes(), original)

    def test_dangling_output_symlink_is_rejected_before_compute(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            target = directory / "must-not-create.json"
            out = directory / "report.json"
            out.symlink_to(target)
            with self.assertRaises(SystemExit) as caught:
                self.invoke(self.args(self.source(directory), out))
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(self.last_compute_calls, 0)
            self.assertEqual(self.last_extension_calls, 0)
            self.assertTrue(out.is_symlink())
            self.assertFalse(target.exists())

    def test_output_created_during_measurement_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            for symlink in (False, True):
                with self.subTest(symlink=symlink):
                    out = directory / f"report-{symlink}.json"
                    target = directory / f"target-{symlink}.json"

                    def create_output(parsed, _record, _summary):
                        if symlink:
                            parsed.out.symlink_to(target)
                        else:
                            parsed.out.write_bytes(b"created by another process")

                    with self.assertRaises(FileExistsError):
                        self.invoke(self.args(source, out), compute=create_output)
                    self.assertEqual(self.last_compute_calls, 1)
                    if symlink:
                        self.assertTrue(out.is_symlink())
                        self.assertFalse(target.exists())
                    else:
                        self.assertEqual(out.read_bytes(), b"created by another process")

    def test_invalid_comparison_is_rejected_before_loading_extension_or_computing(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            reference = directory / "reference.json"
            for content in (None, "invalid json", "[]", "{}", '{"identity": null}', '{"identity": []}'):
                with self.subTest(content=content):
                    reference.unlink(missing_ok=True)
                    if content is not None:
                        reference.write_text(content)
                    out = directory / "report.json"
                    with self.assertRaises(SystemExit) as caught:
                        self.invoke(self.args(source, out, "--compare", str(reference)))
                    self.assertEqual(caught.exception.code, 2)
                    self.assertEqual(self.last_compute_calls, 0)
                    self.assertEqual(self.last_extension_calls, 0)
                    self.assertFalse(out.exists())

    def test_pre_summary_repo_is_accepted_only_as_reference(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            old_repo = directory / "old-repo"
            (old_repo / "dngscan").mkdir(parents=True)
            (old_repo / "dngscan" / "raw_io.py").write_text("# old checkout fixture\n")
            out = directory / "reference.json"
            status, _ = self.invoke(self.args(source, out, "--reference", repo=old_repo),
                                    missing_summary="dngscan.sensor_summary")
            self.assertEqual(status, 0)
            record = json.loads(out.read_text())
            self.assertFalse(record["summary_available"])
            self.assertNotIn("summarize_sensor", record["sensor_calls"])
            self.assertEqual(record["sensor_calls"]["detect_ceilings"]["count"], 1)
            normal = directory / "normal.json"
            with self.assertRaises(ModuleNotFoundError) as caught:
                self.invoke(self.args(source, normal, repo=old_repo), missing_summary="dngscan.sensor_summary")
            self.assertEqual(caught.exception.name, "dngscan.sensor_summary")
            self.assertEqual(self.last_compute_calls, 0)
            self.assertFalse(normal.exists())

    def test_reference_does_not_hide_missing_dependency_inside_summary(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            out = directory / "report.json"
            with self.assertRaises(ModuleNotFoundError) as caught:
                self.invoke(self.args(self.source(directory), out, "--reference"),
                            missing_summary="unrelated_dependency")
            self.assertEqual(caught.exception.name, "unrelated_dependency")
            self.assertEqual(self.last_compute_calls, 0)
            self.assertFalse(out.exists())

    def test_both_modes_require_matching_native_extension(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            for reference in (False, True):
                with self.subTest(reference=reference):
                    out = directory / f"report-{reference}.json"
                    extra = ("--reference",) if reference else ()
                    with self.assertRaisesRegex(RuntimeError, "matching native extension is required"):
                        self.invoke(self.args(source, out, *extra), extension_available=False)
                    self.assertEqual(self.last_compute_calls, 0)
                    self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
