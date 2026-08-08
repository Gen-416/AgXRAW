# SPDX-License-Identifier: GPL-3.0-or-later
"""Scheduler plan S1 acceptance gates: per-key single-flight cache builds.

Same key concurrently requested -> ONE build, everyone shares it. Different
keys -> parallel builds up to the quota. A failed build propagates to that
flight's waiters only and the next request retries.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock


class SingleFlightTests(unittest.TestCase):
    def _cache(self):
        from dngscan.gui.preview_cache import PreviewCache

        return PreviewCache()

    def _run(self, cache, n_threads, keys, build_log, build_time=0.05,
             fail_first=False):
        import dngscan.gui.preview_cache as pc

        failures = {"armed": fail_first}
        lock = threading.Lock()

        def fake_read_disk(*a, **k):
            return None

        def fake_load_raw(path, *a, **k):
            with lock:
                build_log.append(str(path))
                if failures["armed"]:
                    failures["armed"] = False
                    raise RuntimeError("injected decode failure")
            time.sleep(build_time)
            return mock.Mock(name="bundle")

        def fake_analyze(source, *a, **k):
            return mock.Mock(name="analysis"), None, None

        def fake_proxy(source, analysis, require_guidance):
            entry = mock.Mock()
            entry.bundle.raw_guidance = object()
            entry.get_or_build_balance = lambda wb, builder: entry
            return entry

        results, errors = [], []

        def worker(key_path):
            try:
                results.append(cache.get(key_path, "clip", "camera"))
            except Exception as exc:
                errors.append(exc)

        with mock.patch.object(pc, "_read_disk_entry", fake_read_disk), \
                mock.patch.object(pc, "_write_disk_entry", lambda *a: None), \
                mock.patch.object(pc, "build_proxy_entry", fake_proxy), \
                mock.patch.object(pc, "_cache_identity",
                                  lambda path, *a, **k: ((str(path),), "d" + str(path))), \
                mock.patch.object(pc.dg, "load_raw", fake_load_raw), \
                mock.patch.object(pc.dg, "analyze", fake_analyze):
            threads = [
                threading.Thread(target=worker, args=(keys[i % len(keys)],))
                for i in range(n_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        return results, errors

    def test_same_key_builds_once(self) -> None:
        from pathlib import Path

        cache = self._cache()
        log: list[str] = []
        results, errors = self._run(cache, 6, [Path("/tmp/a.dng")], log)
        self.assertEqual(errors, [])
        self.assertEqual(len(log), 1, f"six concurrent requests built {len(log)}x")
        self.assertEqual(len(results), 6)
        self.assertTrue(all(r is results[0] for r in results))

    def test_different_keys_build_in_parallel(self) -> None:
        from pathlib import Path

        cache = self._cache()
        log: list[str] = []
        t0 = time.perf_counter()
        _, errors = self._run(
            cache, 2, [Path("/tmp/a.dng"), Path("/tmp/b.dng")], log,
            build_time=0.25,
        )
        wall = time.perf_counter() - t0
        self.assertEqual(errors, [])
        self.assertEqual(sorted(set(log)), ["/tmp/a.dng", "/tmp/b.dng"])
        self.assertLess(
            wall, 0.45,
            f"two distinct cold starts took {wall:.2f}s — they serialized "
            "(the old global build_lock behaviour)",
        )

    def test_failure_propagates_and_next_request_retries(self) -> None:
        from pathlib import Path

        cache = self._cache()
        log: list[str] = []
        results, errors = self._run(
            cache, 3, [Path("/tmp/c.dng")], log, fail_first=True,
        )
        # every request eventually resolves: the failed flight surfaces its
        # error to at most its own waiters, and retries rebuild
        self.assertGreaterEqual(len(log), 2, "a retry build must happen")
        self.assertEqual(len(results) + len(errors), 3)
        self.assertGreaterEqual(len(results), 1)

    def test_build_quota_bounds_concurrency(self) -> None:
        from pathlib import Path

        import dngscan.gui.preview_cache as pc

        cache = self._cache()
        active = {"now": 0, "peak": 0}
        lock = threading.Lock()

        def fake_load_raw(path, *a, **k):
            with lock:
                active["now"] += 1
                active["peak"] = max(active["peak"], active["now"])
            time.sleep(0.1)
            with lock:
                active["now"] -= 1
            return mock.Mock(name="bundle")

        def worker(p):
            cache.get(p, "clip", "camera")

        with mock.patch.object(pc, "_read_disk_entry", lambda *a, **k: None), \
                mock.patch.object(pc, "_write_disk_entry", lambda *a: None), \
                mock.patch.object(
                    pc, "build_proxy_entry",
                    lambda s, an, g: mock.Mock(
                        bundle=mock.Mock(raw_guidance=object()),
                        get_or_build_balance=lambda wb, b: mock.Mock(),
                    )), \
                mock.patch.object(pc, "_cache_identity",
                                  lambda path, *a, **k: ((str(path),), "q" + str(path))), \
                mock.patch.object(pc.dg, "load_raw", fake_load_raw), \
                mock.patch.object(pc.dg, "analyze",
                                  lambda *a, **k: (mock.Mock(), None, None)):
            threads = [
                threading.Thread(
                    target=worker, args=(Path(f"/tmp/q{i}.dng"),)
                ) for i in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertLessEqual(
            active["peak"], pc.PreviewCache.MAX_CONCURRENT_BUILDS,
            "cold builds must respect the memory quota",
        )
        self.assertGreaterEqual(active["peak"], 2, "quota should allow parallelism")


if __name__ == "__main__":
    unittest.main()
