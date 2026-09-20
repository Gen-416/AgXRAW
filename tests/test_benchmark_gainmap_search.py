# SPDX-License-Identifier: GPL-3.0-or-later
"""The fixed-master benchmark's measurement/oracle contract, without a codec."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools import benchmark_gainmap_search as bench

ROOT = Path(__file__).resolve().parents[1]


def report():
    return {"delivery": {"delivery_quality": 90, "delivery_chroma_requested": "444",
                         "gainmap_encoding_quality": 80, "coding_luma_rmse": .2,
                         "auto_attempts": [
                             {"quality": 95, "chroma": "444", "gainmap_quality": 100,
                              "accepted": True, "bytes": 100, "metrics": {"coding_luma_rmse": .1}},
                             {"quality": 70, "chroma": "444", "gainmap_quality": 100,
                              "accepted": False, "bytes": 80,
                              "metrics": {"coding_luma_rmse": 2., "chroma_error": .5}},
                         ]}}


class BenchmarkGainmapSearchTests(unittest.TestCase):
    def test_nested_self_time_excludes_children_and_logs_errors(self):
        ticks = iter((0., 1., 3., 5.))
        trace = bench.CallTrace(clock=lambda: next(ticks))
        child = trace.wrap(lambda: (_ for _ in ()).throw(ValueError("bad candidate")), "child", "child")
        parent = trace.wrap(child, "parent", "parent")
        with self.assertRaisesRegex(ValueError, "bad candidate"):
            parent()
        summary = trace.summary()
        self.assertEqual(summary["child"], {"count": 1, "wall_s": 2., "self_s": 2.})
        self.assertEqual(summary["parent"], {"count": 1, "wall_s": 5., "self_s": 3.})
        self.assertEqual(trace.local.stack, [])
        self.assertTrue(all(event["error"]["type"] == "ValueError" for event in trace.events))

    def test_comparison_allows_only_rejected_metric_omissions(self):
        baseline = report()
        candidate = copy.deepcopy(baseline)
        del candidate["delivery"]["auto_attempts"][1]["metrics"]["chroma_error"]
        result = bench.compare_reports(candidate, baseline)
        self.assertTrue(result["matches"])
        self.assertEqual(result["early_reject_omitted_metrics"], [{"attempt": 1, "metric": "chroma_error"}])
        incomplete = copy.deepcopy(candidate)
        del incomplete["delivery"]["auto_attempts"][1]["metrics"]["coding_luma_rmse"]
        self.assertFalse(bench.compare_reports(incomplete, baseline)["matches"])
        del candidate["delivery"]["auto_attempts"][0]["metrics"]["coding_luma_rmse"]
        self.assertFalse(bench.compare_reports(candidate, baseline)["matches"])

    def test_comparison_detects_decision_input_and_payload_changes(self):
        baseline = report()
        baseline.update(inputs={"base": {"file_sha256": "abc"}},
                        artifact={"payload_signature": ["signature"]}, headroom_ev=3.)
        for change in (
            lambda d: d["delivery"].update(delivery_quality=85),
            lambda d: d["delivery"]["auto_attempts"][1].update(accepted=True),
            lambda d: d["inputs"]["base"].update(file_sha256="def"),
            lambda d: d["artifact"].update(payload_signature=["changed"]),
            lambda d: d.update(headroom_ev=4.),
        ):
            candidate = copy.deepcopy(baseline)
            change(candidate)
            self.assertFalse(bench.compare_reports(candidate, baseline)["matches"])

    def test_attempt_bytes_are_exact_except_missing_early_reject_measurements(self):
        baseline = report()
        self.assertTrue(bench.compare_reports(copy.deepcopy(baseline), baseline)["matches"])
        for index in (0, 1):
            with self.subTest(changed_attempt=index):
                candidate = copy.deepcopy(baseline)
                candidate["delivery"]["auto_attempts"][index]["bytes"] += 1
                result = bench.compare_reports(candidate, baseline)
                self.assertFalse(result["matches"])
                self.assertIn(f"attempt {index} bytes changed", result["failures"])
        accepted_missing = copy.deepcopy(baseline)
        del accepted_missing["delivery"]["auto_attempts"][0]["bytes"]
        result = bench.compare_reports(accepted_missing, baseline)
        self.assertFalse(result["matches"])
        self.assertIn("attempt 0 bytes missing", result["failures"])
        rejected_missing = copy.deepcopy(baseline)
        del rejected_missing["delivery"]["auto_attempts"][1]["bytes"]
        self.assertTrue(bench.compare_reports(rejected_missing, baseline)["matches"])
        # Older reports without a measurement cannot be a byte-count oracle.
        self.assertTrue(bench.compare_reports(baseline, accepted_missing)["matches"])

    def test_signature_keeps_bytes_and_ignores_file_layout_for_payload(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td)/"a.heic", Path(td)/"b.heic"
            a.write_bytes(b"layout-a")
            b.write_bytes(b"layout-b")
            signature = lambda *args: (b"hvc1", b"\x00\xff", ((True, b"ICC"),))
            first = bench.artifact_record(a, signature)
            second = bench.artifact_record(b, signature)
            self.assertEqual(first["payload_signature_sha256"], second["payload_signature_sha256"])
            self.assertNotEqual(first["file_sha256"], second["file_sha256"])
            self.assertEqual(first["payload_signature"][1], {"bytes_hex": "00ff"})

    def test_dry_run_does_not_import_codec_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            repo = folder/"repo"
            (repo/"dngscan").mkdir(parents=True)
            (repo/"dngscan"/"gainmap.py").write_text('raise AssertionError("codec must not import")')
            np.save(folder/"base.npy", np.zeros((2, 3, 3), np.uint8))
            np.save(folder/"hdr.npy", np.ones((2, 3, 4), np.float16))
            out = folder/"result.heic"
            result = subprocess.run([
                sys.executable, str(ROOT/"tools"/"benchmark_gainmap_search.py"),
                "--repo", str(repo), "--base", str(folder/"base.npy"),
                "--hdr", str(folder/"hdr.npy"), "--out", str(out),
                "--headroom", "3", "--dry-run",
            ], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["dry_run"])
            self.assertFalse(out.exists())
            self.assertFalse(out.with_suffix(".benchmark.json").exists())

    def test_preflight_protects_reference_and_nonfinite_headroom(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            base, hdr = folder/"base.npy", folder/"hdr.npy"
            base.touch(); hdr.touch()
            reference = folder/"reference.heic"
            reference.write_bytes(b"reference")
            args = bench.parser().parse_args([
                "--repo", str(ROOT), "--base", str(base), "--hdr", str(hdr),
                "--out", str(reference), "--headroom", "3", "--overwrite",
                "--reference-file", str(reference),
            ])
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                bench.validate(args)
            args.out = folder/"new.heic"
            args.headroom = float("nan")
            with self.assertRaisesRegex(ValueError, "positive finite"):
                bench.validate(args)
            self.assertEqual(reference.read_bytes(), b"reference")

    def test_mocked_search_records_real_entry_counts_and_restores_hooks(self):
        from dngscan import gainmap, heif_encoder, delivery_integrity

        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            base, hdr = folder/"base.npy", folder/"hdr.npy"
            np.save(base, np.zeros((2, 3, 3), np.uint8))
            np.save(hdr, np.ones((2, 3, 4), np.float16))
            out = folder/"new.heic"

            def write(b, h, path, headroom, **kwargs):
                self.assertFalse(b.flags.writeable)
                self.assertFalse(h.flags.writeable)
                heif_encoder.encode(b, path, 90, "444")
                gainmap.read_primary_rgb_u8(path)
                gainmap._read_expanded_hdr_rgba_half(path)
                Path(path).write_bytes(b"mock-heif")
                return report()["delivery"]

            original_path = sys.path[:]
            try:
                with mock.patch.object(gainmap, "apple_gainmap_backend_status", return_value=(True, "mock")), \
                     mock.patch.object(heif_encoder, "available", return_value=True), \
                     mock.patch.object(heif_encoder, "encode") as encoder, \
                     mock.patch.object(gainmap, "read_primary_rgb_u8"), \
                     mock.patch.object(gainmap, "_read_expanded_hdr_rgba_half"), \
                     mock.patch.object(gainmap, "write_apple_gainmap_file", side_effect=write), \
                     mock.patch.object(delivery_integrity, "encoded_content_signature", return_value=(b"hvc1",)), \
                     mock.patch.dict("os.environ", {}), redirect_stdout(io.StringIO()):
                    self.assertEqual(bench.main([
                        "--repo", str(ROOT), "--base", str(base), "--hdr", str(hdr),
                        "--out", str(out), "--headroom", "3", "--native-mode", "auto",
                    ]), 0)
                    self.assertIs(heif_encoder.encode, encoder)
            finally:
                sys.path[:] = original_path
            result = json.loads(out.with_suffix(".benchmark.json").read_text())
            self.assertEqual(result["selected"]["delivery_quality"], 90)
            for key in ("primary_encode", "sdr_readback", "hdr_readback"):
                self.assertEqual(result["stages"][key]["count"], 1)
            self.assertEqual(result["artifact"]["bytes"], 9)
            self.assertEqual(result["inputs"]["base"]["file_sha256"], bench.file_sha256(base))
            self.assertNotIn("error", result)


if __name__ == "__main__":
    unittest.main()
