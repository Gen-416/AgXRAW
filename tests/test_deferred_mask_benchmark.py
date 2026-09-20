# SPDX-License-Identifier: GPL-3.0-or-later
"""Deferred-mask benchmark contracts without loading or rendering a RAW."""
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

from dngscan import _fast, raw_io
from tools import benchmark_deferred_masks as bench
from tools import benchmark_loss_pipeline as pipeline_helper


ROOT = Path(__file__).resolve().parents[1]


class DeferredMaskBenchmarkTests(unittest.TestCase):
    def invoke(self, args, *, identity=None, compute=None, legacy=False,
               fallback=False, fail_first_build=False, extension_available=True):
        load_flags = []

        def load(_source, *, decoder="libraw", **kw):
            if legacy and "_defer_clip_masks" in kw:
                raise TypeError("old load has no deferred keyword")
            deferred = kw.get("_defer_clip_masks", False)
            load_flags.append(deferred)
            if fallback and decoder == "coreimage":
                return raw_io.load_raw(_source, decoder="libraw", **kw)
            bundle = SimpleNamespace(scene_decoder="libraw", clip_masks=None)
            if not legacy:
                bundle._clip_masks_pending = deferred
            if not deferred:
                bundle.clip_masks = raw_io.build_clip_masks()
            return bundle

        def refresh(bundle, _fullwell):
            bundle.clip_masks = raw_io.build_clip_masks()
            if hasattr(bundle, "_clip_masks_pending"):
                bundle._clip_masks_pending = False
            return True

        def pipeline(parsed, record):
            self.assertEqual(os.environ["DNGSCAN_FAST"], "1")
            self.assertEqual(os.environ["DNGSCAN_FAST_SKIP"], "")
            if compute is not None:
                compute(parsed, record)
            else:
                bundle = raw_io.load_raw(parsed.source, decoder=parsed.decoder)
                masks_loaded = bundle.clip_masks
                raw_io.refresh_clip_masks_from_fullwell(bundle, {0: 4095})
                record["identity"] = {"masks_loaded": masks_loaded,
                                      "masks_analyzed": bundle.clip_masks,
                                      "decisions": {"headroom": 2.0}, "sdr": "pixels"}
            if identity is not None:
                record["identity"] = copy.deepcopy(identity)

        native = SimpleNamespace(native_abi_version=lambda: 15) if extension_available else None
        build_effect = [RuntimeError("failed mask build"), "mask"] if fail_first_build else None
        clock = itertools.count(step=0.25)
        stream = io.StringIO()
        with mock.patch.object(sys, "argv", ["benchmark_deferred_masks.py", *args]), \
             mock.patch.object(sys, "path", sys.path[:]), \
             mock.patch.dict(os.environ, {"DNGSCAN_FAST": "0", "DNGSCAN_FAST_SKIP": "unrelated_kernel"}), \
             mock.patch.object(_fast, "_load_extension", return_value=native) as extension, \
             mock.patch.object(bench.subprocess, "run", return_value=SimpleNamespace(stdout="fixture-commit\n")), \
             mock.patch.object(bench.platform, "platform", return_value="test-platform"), \
             mock.patch.object(bench.time, "perf_counter", side_effect=lambda: next(clock)), \
             mock.patch.object(raw_io, "load_raw", side_effect=load), \
             mock.patch.object(raw_io, "build_clip_masks", return_value="mask", side_effect=build_effect), \
             mock.patch.object(raw_io, "refresh_clip_masks_from_fullwell", side_effect=refresh), \
             mock.patch.object(pipeline_helper, "pipeline", side_effect=pipeline) as compute_mock, \
             redirect_stdout(stream), redirect_stderr(io.StringIO()):
            try:
                status = bench.main()
            finally:
                self.last_compute_calls = compute_mock.call_count
                self.last_extension_calls = extension.call_count
                self.last_load_flags = load_flags
        return status, stream.getvalue()

    def args(self, source, out, *extra, repo=ROOT):
        return ["--repo", str(repo), "--source", str(source), "--out", str(out), *extra]

    def source(self, directory):
        source = directory / "fixture.dng"
        source.write_bytes(b"synthetic RAW identity; never decoded")
        return source

    def test_only_optimized_load_defers_and_report_preserves_actual_handoff(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            for reference in (False, True):
                with self.subTest(reference=reference):
                    out = directory / f"report-{reference}.json"
                    extra = ("--reference",) if reference else ()
                    status, stdout = self.invoke(self.args(source, out, *extra))
                    self.assertEqual(status, 0)
                    report = json.loads(out.read_text())
                    self.assertEqual(self.last_load_flags, [not reference])
                    self.assertEqual(report["native_abi"], 15)
                    self.assertEqual(report["commit"], "fixture-commit")
                    self.assertEqual(report["mask_calls"]["build_clip_masks"],
                                     {"count": 2 if reference else 1,
                                      "wall_s": 0.5 if reference else 0.25})
                    self.assertEqual(report["mask_calls"]["refresh_clip_masks_from_fullwell"],
                                     {"count": 1, "wall_s": 0.75})
                    initial = {"decoder": "libraw", "pending": not reference, "mask_present": reference}
                    final = {"decoder": "libraw", "pending": False, "mask_present": True}
                    self.assertEqual(report["mask_load_states"], [{"depth": 0, "state": initial}])
                    self.assertEqual(report["mask_refresh_states"], [{"before": initial, "after": final,
                                                                     "completed": True, "rebuilt": True}])
                    self.assertEqual(report["identity"]["masks_loaded"], "mask" if reference else None)
                    self.assertEqual(json.loads(stdout)["mask_calls"], report["mask_calls"])

    def test_recursive_fallback_retains_deferred_flag_and_records_both_loads(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            out = directory / "report.json"
            status, _ = self.invoke(self.args(self.source(directory), out, "--decoder", "coreimage"),
                                    fallback=True)
            self.assertEqual(status, 0)
            self.assertEqual(self.last_load_flags, [True, True])
            report = json.loads(out.read_text())
            self.assertEqual([event["depth"] for event in report["mask_load_states"]], [1, 0])
            self.assertTrue(all(event["state"]["pending"] for event in report["mask_load_states"]))
            self.assertEqual(report["mask_calls"]["build_clip_masks"]["count"], 1)

    def test_old_checkout_is_accepted_as_reference_without_supplying_new_keyword(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            out = directory / "reference.json"
            status, _ = self.invoke(self.args(source, out, "--reference"), legacy=True)
            self.assertEqual(status, 0)
            report = json.loads(out.read_text())
            self.assertEqual(report["mask_load_states"][0]["state"],
                             {"decoder": "libraw", "pending": False, "mask_present": True})
            normal = directory / "normal.json"
            with self.assertRaisesRegex(TypeError, "old load has no deferred keyword"):
                self.invoke(self.args(source, normal), legacy=True)
            self.assertFalse(normal.exists())

    def test_failed_build_records_count_and_unfinished_refresh_before_retry(self):
        def recover(parsed, record):
            bundle = raw_io.load_raw(parsed.source)
            with self.assertRaisesRegex(RuntimeError, "failed mask build"):
                raw_io.refresh_clip_masks_from_fullwell(bundle, {0: 4095})
            raw_io.refresh_clip_masks_from_fullwell(bundle, {0: 4095})
            record["identity"] = {"masks_analyzed": bundle.clip_masks}

        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            out = directory / "report.json"
            status, _ = self.invoke(self.args(self.source(directory), out), compute=recover,
                                    fail_first_build=True)
            self.assertEqual(status, 0)
            report = json.loads(out.read_text())
            self.assertEqual(report["mask_calls"]["build_clip_masks"], {"count": 2, "wall_s": 0.5})
            self.assertEqual(report["mask_calls"]["refresh_clip_masks_from_fullwell"],
                             {"count": 2, "wall_s": 1.5})
            failed, retry = report["mask_refresh_states"]
            self.assertFalse(failed["completed"])
            self.assertNotIn("rebuilt", failed)
            self.assertEqual(failed["before"], failed["after"])
            self.assertTrue(failed["after"]["pending"])
            self.assertTrue(retry["completed"])
            self.assertFalse(retry["after"]["pending"])

    def test_comparison_excludes_only_loaded_masks_and_checks_source_digest(self):
        identity = {"masks_loaded": "old mask", "masks_analyzed": "final mask",
                    "processing_loss": "loss", "sdr": "pixels", "decisions": {"ev": 1.0}}
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            reference = directory / "reference.json"
            cases = [("loaded-only", None, digest), ("legacy-no-digest", None, None),
                     ("different-source", None, "0" * 64)]
            cases.extend((key, key, digest) for key in identity if key != "masks_loaded")
            for case, changed_key, previous_digest in cases:
                with self.subTest(case=case):
                    previous = {"identity": identity}
                    if previous_digest is not None:
                        previous["source_sha256"] = previous_digest
                    reference.write_text(json.dumps(previous))
                    current = copy.deepcopy(identity)
                    current["masks_loaded"] = None
                    if changed_key:
                        current[changed_key] = "changed"
                    out = directory / f"{case}.json"
                    status, _ = self.invoke(self.args(source, out, "--compare", str(reference)),
                                            identity=current)
                    expected = changed_key is None and previous_digest in (digest, None)
                    self.assertEqual(status, 0 if expected else 2)
                    report = json.loads(out.read_text())
                    self.assertEqual(report["comparison"]["exact"], expected)
                    self.assertEqual(report["comparison"]["excluded_fields"], ["identity.masks_loaded"])
                    self.assertIsNone(report["identity"]["masks_loaded"])
                    self.assertEqual(report["source_sha256"], digest)
                    self.assertEqual(json.loads(reference.read_text()), previous)

    def test_existing_output_source_alias_and_dangling_symlink_fail_before_compute(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            out = directory / "existing.json"
            out.write_bytes(b"keep report")
            link = directory / "dangling.json"
            missing = directory / "must-not-create.json"
            link.symlink_to(missing)
            for target in (out, source, link):
                with self.subTest(target=target):
                    original = target.read_bytes() if target.exists() else None
                    with self.assertRaises(SystemExit) as caught:
                        self.invoke(self.args(source, target))
                    self.assertEqual(caught.exception.code, 2)
                    self.assertEqual(self.last_compute_calls, 0)
                    self.assertEqual(self.last_extension_calls, 0)
                    if original is not None:
                        self.assertEqual(target.read_bytes(), original)
            self.assertTrue(link.is_symlink())
            self.assertFalse(missing.exists())

    def test_output_created_during_measurement_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            for symlink in (False, True):
                with self.subTest(symlink=symlink):
                    out = directory / f"report-{symlink}.json"
                    target = directory / f"target-{symlink}.json"

                    def create_output(parsed, record):
                        record["identity"] = {}
                        if symlink:
                            parsed.out.symlink_to(target)
                        else:
                            parsed.out.write_bytes(b"created elsewhere")

                    with self.assertRaises(FileExistsError):
                        self.invoke(self.args(source, out), compute=create_output)
                    self.assertEqual(self.last_compute_calls, 1)
                    if symlink:
                        self.assertTrue(out.is_symlink())
                        self.assertFalse(target.exists())
                    else:
                        self.assertEqual(out.read_bytes(), b"created elsewhere")

    def test_invalid_comparison_fails_preflight_without_loading_native_extension(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = self.source(directory)
            reference = directory / "reference.json"
            for content in (None, "not json", "[]", "{}", '{"identity": null}', '{"identity": []}'):
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
