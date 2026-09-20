# SPDX-License-Identifier: GPL-3.0-or-later
"""Contracts for the tree benchmark, without RAW decode or render workloads."""
from contextlib import contextmanager, redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from tools import benchmark_pipeline_concurrency as bench


class PipelineConcurrencyBenchmarkTests(unittest.TestCase):
    def test_distribution_uses_nearest_rank_and_records_actual_intervals(self):
        self.assertEqual(bench.distribution([3, 1, 2])["median"], 2)
        self.assertEqual(bench.distribution([])["p95"], None)
        samples = [{"monotonic_s": t, "read_s": .002, "rss_bytes": n,
                    "threads": 2, "processes": [{}], "missing_pids": []}
                   for t, n in [(0., 1000), (1., 200), (1.05, 300), (1.14, 250), (3., 2000)]]
        summary = bench._sample_summary(samples, {"started_s": 1., "finished_s": 2.})
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["tree_rss_peak_bytes"], 300)
        self.assertAlmostEqual(summary["interval_s"]["max"], .09)

    def test_identity_retains_decisions_and_hashes_data_only_excludes_cache_flags(self):
        result = bench.response_identity({"preview": "data:image/jpeg;base64,YWJj",
            "cache_hit": False, "pixel_cache_hit": True, "generation": 2,
            "delivery": {"quality": 97, "chroma": "4:2:2", "path": "/tmp/task/out.jpg"}}, Path("/tmp/task"))
        self.assertEqual(result["generation"], 2)
        self.assertEqual(result["delivery"], {"quality": 97, "chroma": "4:2:2", "path": "<temporary>/out.jpg"})
        self.assertEqual(result["preview"]["sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertNotIn("cache_hit", result)
        bare = bench.response_identity({"preview": "YWJj"}, Path("/tmp/task"))
        self.assertEqual(bare["preview"]["sha256"], result["preview"]["sha256"])
        self.assertEqual(bare["preview"]["bytes"], 3)

    @unittest.skipUnless(sys.platform == "darwin" or sys.platform.startswith("linux"), "sampler platform")
    def test_real_self_sample_is_small_and_nonempty(self):
        result = bench.ProcessSampler().sample(os.getpid())
        self.assertGreater(result["rss_bytes"], 0)
        self.assertGreaterEqual(result["threads"], 1)
        self.assertIn(os.getpid(), [p["pid"] for p in result["processes"]])

    @unittest.skipUnless(sys.platform == "darwin" or sys.platform.startswith("linux"), "sampler platform")
    def test_real_child_is_included_in_tree(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            result = bench.ProcessSampler().sample(os.getpid())
            self.assertIn(child.pid, [p["pid"] for p in result["processes"]])
            self.assertEqual(result["rss_bytes"], sum(p["rss_bytes"] for p in result["processes"]))
        finally:
            child.terminate()
            child.wait()

    def test_tree_sampler_recurses_and_handles_disappeared_process(self):
        sampler = object.__new__(bench.ProcessSampler)
        with mock.patch.object(sampler, "children", side_effect=lambda pid: {1: [2], 2: [3], 3: []}[pid]), \
             mock.patch.object(sampler, "process", side_effect=lambda pid: None if pid == 3 else
                               {"pid": pid, "rss_bytes": pid * 100, "threads": pid}):
            result = sampler.sample(1)
        self.assertEqual(result["rss_bytes"], 300)
        self.assertEqual(result["threads"], 3)
        self.assertEqual(result["missing_pids"], [3])

    def test_worker_uses_three_real_entrypoints_keeps_quota_and_hashes_after_window(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source_a, source_b = folder / "a.dng", folder / "b.dng"
            source_a.write_bytes(b"source a")
            source_b.write_bytes(b"source b")
            (folder / "export").mkdir()
            active, peak, calls = [], [], []

            @contextmanager
            def slot(kind):
                active.append(kind)
                peak.append(len(active))
                try:
                    yield
                finally:
                    active.remove(kind)

            scheduler = SimpleNamespace(slot=slot, snapshot=lambda: {"active": list(active)})

            def invoke(kind, params):
                calls.append((kind, dict(params)))
                with scheduler.slot(kind):
                    time.sleep(.004)
                    if kind == "export":
                        output = Path(params["outdir"]) / "saved.jpg"
                        output.write_bytes(b"encoded")
                        return {"ok": True, "saved": [str(output)], "ev_auto": {"selected_ev": 1.25}}
                    if kind == "preview":
                        return {"ok": True, "preview": "data:image/jpeg;base64,YWJj",
                                "ev": params["ev"], "generation": params["generation"], "cache_hit": False}
                    return {"ok": True, "prepared": True, "detected": {"black_ev": -12.}}

            service = SimpleNamespace(SCHEDULER=scheduler,
                prepare_preview=lambda params: invoke("prepare", params),
                run_preview=lambda params: invoke("preview", params),
                run_export_isolated=lambda params: invoke("export", params))
            cache = SimpleNamespace(PREVIEW_STORE=SimpleNamespace(clear_memory=lambda: None,
                                         memory_snapshot=lambda: {"bytes": 100}),
                                    DISK_WRITER=SimpleNamespace(snapshot=lambda: {"pending_items": 0},
                                                               flush=lambda timeout: True))
            fake_dng = ModuleType("dngscan")
            fake_dng._fast = SimpleNamespace(_load_extension=lambda: SimpleNamespace(native_abi_version=lambda: 17))
            fake_dng.cpu_budget = SimpleNamespace(TOTAL=4)
            fake_gui = ModuleType("dngscan.gui")
            fake_gui.service, fake_gui.preview_cache = service, cache
            fake_gui.scheduler = SimpleNamespace(_QUOTAS={"prepare": 1, "preview": 1, "export": 1})
            args = SimpleNamespace(repo=folder, workdir=folder, source_a=source_a, source_b=source_b,
                                   frames=2, preview_period_ms=0., decoder="libraw", format="sdr")
            original_file_record = bench.file_record

            def file_record(path):
                self.assertFalse(active, "hashing must not run during a service call")
                return original_file_record(path)

            with mock.patch.dict(sys.modules, {"dngscan": fake_dng, "dngscan.gui": fake_gui}), \
                 mock.patch.object(sys, "path", sys.path[:]), \
                 mock.patch.object(bench.platform, "platform", return_value="fixture-platform"), \
                 mock.patch.object(bench.subprocess, "check_output", return_value="fixture\n"), \
                 mock.patch.object(bench, "file_record", side_effect=file_record):
                self.assertEqual(bench._worker(args), 0)
            result = json.loads((folder / "worker.json").read_text())
            self.assertEqual(len(result["slot_events"]), 6)
            self.assertEqual(result["class_quotas"], {"prepare": 1, "preview": 1, "export": 1})
            self.assertEqual(len(result["jobs"]["previews"]), 2)
            self.assertGreaterEqual(max(peak), 2)
            export_params = next(params for kind, params in calls if kind == "export")
            self.assertNotIn("ev", export_params)
            self.assertNotIn("quality", export_params)
            self.assertEqual(export_params["toneCore"], "agx")
            self.assertEqual(result["identity"]["artifacts"][0]["relative_path"], "export/saved.jpg")
            self.assertLessEqual(result["window"]["started_s"], result["jobs"]["cold_prepare"]["started_s"])

    def test_cli_rejects_overwrite_same_source_and_bad_intervals_before_worker(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            a, b, out = folder / "a", folder / "b", folder / "out"
            a.write_bytes(b"a")
            b.write_bytes(b"b")
            common = ["--repo", str(root), "--source-a", str(a), "--source-b", str(b), "--out", str(out)]
            with mock.patch.object(bench, "_parent") as parent, redirect_stderr(io.StringIO()):
                for extra in (["--sample-ms", "19"], ["--sample-ms", "nan"],
                              ["--source-b", str(a)], ["--frames", "0"], ["--preview-period-ms", "inf"]):
                    with self.subTest(extra=extra), self.assertRaises(SystemExit):
                        bench.main(common + extra)
                out.write_bytes(b"keep")
                with self.assertRaises(SystemExit):
                    bench.main(common)
                self.assertEqual(out.read_bytes(), b"keep")
                parent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
