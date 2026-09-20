# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional operator pools spend the supplied share and finish on errors."""
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dngscan import cpu_budget, gated_drt, scene_transform


class OperatorBudgetShareTests(unittest.TestCase):
    def test_region_map_bounds_concurrency_propagates_share_and_orders_results(self):
        gate = threading.Barrier(2)
        active, peak, shares = [0], [0], []
        lock = threading.Lock()

        def compute(value):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
                shares.append(cpu_budget.current_inner())
            gate.wait(timeout=2)
            with lock:
                active[0] -= 1
            return value * 2

        with patch("dngscan._fast.set_thread_budget") as publish, \
             ThreadPoolExecutor(max_workers=8) as pool, cpu_budget.inner(2):
            result = cpu_budget.ordered_budget_map(pool, compute, range(8), max_workers=8)
            self.assertEqual(cpu_budget.current_inner(), 2)
        self.assertEqual(result, list(range(0, 16, 2)))
        self.assertEqual(peak[0], 2)
        self.assertEqual(shares, [1] * 8)
        self.assertEqual([call.args[0] for call in publish.call_args_list], [1, 0])

    def test_single_share_does_not_submit_and_empty_work_does_not_claim(self):
        pool = Mock()
        pool.submit.side_effect = AssertionError("must remain on caller")
        caller = threading.get_ident()
        with patch("dngscan._fast.set_thread_budget") as publish, cpu_budget.inner(1):
            self.assertEqual(cpu_budget.ordered_budget_map(
                pool, lambda _: threading.get_ident(), range(3), max_workers=8), [caller] * 3)
            self.assertEqual(cpu_budget.ordered_budget_map(pool, lambda x: x, [], max_workers=8), [])
        self.assertEqual([call.args[0] for call in publish.call_args_list], [1, 0])

    def test_each_batch_receives_its_actual_fraction_of_the_parent_share(self):
        with patch("dngscan._fast.set_thread_budget"), \
             ThreadPoolExecutor(max_workers=8) as pool, cpu_budget.inner(8):
            shares = cpu_budget.ordered_budget_map(
                pool, lambda _: cpu_budget.current_inner(), range(7), max_workers=3)
        self.assertEqual(shares, [2] * 7)

    def _failure_waits(self, fail_submit):
        started, release, returned = threading.Event(), threading.Event(), threading.Event()
        errors, published = [], []

        def compute(value):
            if value == 0:
                started.set()
                release.wait(timeout=3)
                return 0
            started.wait(timeout=3)
            raise RuntimeError("worker failed")

        with ThreadPoolExecutor(max_workers=2) as pool, patch(
                "dngscan._fast.set_thread_budget", side_effect=published.append):
            submit = pool.submit
            submitted = []

            def observed_submit(*a, **kw):
                if fail_submit and submitted:
                    raise RuntimeError("submit failed")
                result = submit(*a, **kw)
                submitted.append(result)
                return result

            def run():
                try:
                    with cpu_budget.inner(2), patch.object(pool, "submit", side_effect=observed_submit):
                        cpu_budget.ordered_budget_map(pool, compute, (0, 1), max_workers=2)
                except RuntimeError as error:
                    errors.append(str(error))
                finally:
                    returned.set()

            caller = threading.Thread(target=run)
            caller.start()
            try:
                self.assertTrue(started.wait(timeout=2))
                self.assertFalse(returned.wait(timeout=.02))
                self.assertNotIn(0, published)
            finally:
                release.set()
                caller.join(timeout=3)
            self.assertFalse(caller.is_alive())
            self.assertTrue(all(future.done() for future in submitted))
        self.assertEqual(errors, ["submit failed" if fail_submit else "worker failed"])
        self.assertEqual(published[-1], 0)

    def test_compute_failure_drains_all_futures_before_releasing_claim(self):
        self._failure_waits(False)

    def test_submit_failure_drains_already_submitted_futures(self):
        self._failure_waits(True)

    def test_interrupted_result_still_drains_running_future_and_keeps_first_error(self):
        started, interrupted = threading.Event(), threading.Event()
        release, returned = threading.Event(), threading.Event()
        errors, published, futures, waits = [], [], [], []

        def compute(value):
            if value == 0:
                started.set()
                release.wait(timeout=3)
                return 0
            raise RuntimeError("later worker error")

        with ThreadPoolExecutor(max_workers=2) as pool, patch(
                "dngscan._fast.set_thread_budget", side_effect=published.append):
            submit = pool.submit

            def observed_submit(*args, **kwargs):
                future = submit(*args, **kwargs)
                if not futures:
                    wait = future.result

                    def result(*a, **kw):
                        waits.append(1)
                        if len(waits) == 1:
                            interrupted.set()
                            raise KeyboardInterrupt("caller interrupted")
                        return wait(*a, **kw)

                    future.result = result
                futures.append(future)
                return future

            def run():
                try:
                    with cpu_budget.inner(2), patch.object(pool, "submit", side_effect=observed_submit):
                        cpu_budget.ordered_budget_map(pool, compute, (0, 1), max_workers=2)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    returned.set()

            caller = threading.Thread(target=run)
            caller.start()
            try:
                self.assertTrue(started.wait(timeout=2))
                self.assertTrue(interrupted.wait(timeout=2))
                self.assertFalse(returned.wait(timeout=.02))
                self.assertNotIn(0, published)
            finally:
                release.set()
                caller.join(timeout=3)
            self.assertFalse(caller.is_alive())
            self.assertTrue(all(future.done() for future in futures))
        self.assertEqual(len(waits), 2)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], KeyboardInterrupt)
        self.assertEqual(str(errors[0]), "caller interrupted")
        self.assertEqual(published, [1, 0])

    def test_scene_transform_keeps_exact_math_under_two_worker_share(self):
        name = next(name for name, preset in scene_transform.SCENE_TRANSFORMS.items()
                    if len(preset.regions) > 1)
        rgb = np.random.default_rng(51).uniform(.02, 2, (24, 3)).astype(np.float32)
        expected = scene_transform._apply_scene_transform_rec2020_reference(rgb, name)
        original = scene_transform._region_weight_from_chroma
        shares = []

        def observed(*args, **kwargs):
            shares.append(cpu_budget.current_inner())
            return original(*args, **kwargs)

        with patch("dngscan._fast.set_thread_budget"), \
             patch.object(scene_transform, "SCENE_TRANSFORM_REGION_PARALLEL_MIN_PIXELS", 0), \
             patch.object(scene_transform, "_region_weight_from_chroma", side_effect=observed), \
             cpu_budget.inner(2):
            result = scene_transform.apply_scene_transform_rec2020(rgb, name)
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(shares)
        self.assertEqual(set(shares), {1})

    def test_gated_branches_share_allowance_and_keep_the_blend(self):
        rgb = np.full((64 * 1024, 3), .25, dtype=np.float32)
        shares = []

        def mapped(values, *_):
            shares.append(cpu_budget.current_inner())
            return values * np.float32(.5)

        with patch("dngscan._fast.set_thread_budget"), \
             patch.object(gated_drt.agx_engine, "formation_matrices", return_value=(None, None)), \
             patch.object(gated_drt.agx_engine, "apply_core_parallel", side_effect=mapped), \
             patch.object(gated_drt.lum_engine, "apply_lum_core", side_effect=mapped), \
             patch.object(gated_drt.punch_engine, "apply_punch_rec2020", side_effect=lambda rgb, _: rgb), \
             patch.object(gated_drt.guidance_engine, "color_path_weight", return_value=np.zeros(len(rgb), np.float32)), \
             cpu_budget.inner(6):
            result = gated_drt.apply_gated_core(rgb, SimpleNamespace())
        self.assertEqual(shares, [3, 3])
        np.testing.assert_array_equal(result, rgb * np.float32(.5))


if __name__ == "__main__":
    unittest.main()
