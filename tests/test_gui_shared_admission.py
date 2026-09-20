# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared-flight subscribers return admission until their owner's result exists."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from dngscan.gui import preview_cache as pc, scheduler, service
from dngscan.gui.preview_scheduler import PreviewCoordinator, PreviewSuperseded
from tests.test_preview_cache import _analysis, _bundle


class SharedAdmissionTests(unittest.TestCase):
    def assert_idle(self, slots):
        self.assertEqual(slots.snapshot()['active'], dict(preview=0, prepare=0, export=0))
        for sem in slots._slots.values():
            self.assertTrue(sem.acquire(blocking=False))
            self.assertFalse(sem.acquire(blocking=False))
            sem.release()

    @contextmanager
    def parked_wait(self, parked):
        with scheduler.shared_flight_wait():
            parked.set()
            yield

    def test_metrics_separate_wait_and_reacquisition_and_complete_once(self):
        slots = scheduler.RenderScheduler()
        with mock.patch.object(scheduler.time, 'monotonic', side_effect=[1., 2., 4., 9., 10., 13.]):
            with slots.slot('preview'):
                with scheduler.shared_flight_wait():
                    self.assertEqual(slots.snapshot()['active']['preview'], 0)
                self.assertEqual(slots.snapshot()['active']['preview'], 1)
        state = slots.snapshot()
        self.assertEqual(state['queue_seconds']['preview'], 2.)
        self.assertEqual(state['execute_seconds']['preview'], 5.)
        self.assertEqual(state['shared_wait_seconds']['preview'], 5.)
        self.assertEqual(state['completed']['preview'], 1)
        self.assert_idle(slots)

    def test_reentrant_and_nested_waits_preserve_export_admission(self):
        slots = scheduler.RenderScheduler()
        with scheduler.shared_flight_wait():
            self.assert_idle(slots)
        with slots.slot('preview'):
            with slots.slot('preview'):
                self.assertEqual(slots.snapshot()['active']['preview'], 1)
                with scheduler.shared_flight_wait():
                    self.assertEqual(slots.snapshot()['active'], dict(preview=0, prepare=0, export=0))
                    with scheduler.shared_flight_wait():
                        self.assertEqual(slots.snapshot()['active']['preview'], 0)
                    with self.assertRaisesRegex(RuntimeError, 'shared-flight wait'):
                        with slots.slot('preview'):
                            self.fail('computation cannot run while its admission is paused')
                self.assertEqual(slots.snapshot()['active']['preview'], 1)
        with slots.slot('export'):
            with scheduler.shared_flight_wait():
                self.assertEqual(slots.snapshot()['active']['export'], 1)
        self.assertEqual(slots.snapshot()['completed'], dict(preview=1, prepare=0, export=1))
        self.assertEqual(slots.snapshot()['shared_wait_seconds']['export'], 0.)
        self.assert_idle(slots)

    def test_cross_category_and_cross_scheduler_nesting_fail_without_waiting(self):
        slots, other = scheduler.RenderScheduler(), scheduler.RenderScheduler()
        with slots.slot('prepare'):
            for target, kind in ((slots, 'preview'), (slots, 'export'), (other, 'prepare')):
                with self.assertRaisesRegex(RuntimeError, 'same scheduler and kind'):
                    with target.slot(kind):
                        self.fail('nested unrelated lease entered')
        self.assert_idle(slots)
        self.assert_idle(other)

    def test_failed_wait_reacquires_before_enclosing_handler_continues(self):
        slots = scheduler.RenderScheduler()
        error = ValueError('owner failed')
        with slots.slot('prepare'):
            with self.assertRaises(ValueError) as raised:
                with scheduler.shared_flight_wait():
                    raise error
            self.assertIs(raised.exception, error)
            self.assertEqual(slots.snapshot()['active']['prepare'], 1)
        self.assert_idle(slots)

    def test_interrupted_reacquisition_finishes_without_double_release(self):
        slots = scheduler.RenderScheduler()
        sem = slots._slots['preview']
        acquire = sem.acquire
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt('resume interrupted')
            return acquire(*args, **kwargs)
        with mock.patch.object(sem, 'acquire', side_effect=interrupted):
            with slots.slot('preview'):
                with self.assertRaises(KeyboardInterrupt):
                    with scheduler.shared_flight_wait():
                        pass
                # Catching a resume interruption inside slot() must not turn
                # the remainder of this request into unadmitted computation.
                self.assertEqual(slots.snapshot()['active']['preview'], 1)
                self.assertFalse(acquire(blocking=False))
        self.assertEqual(slots.snapshot()['completed']['preview'], 1)
        self.assert_idle(slots)

    def test_interrupted_resume_registration_returns_permit_before_retry(self):
        slots = scheduler.RenderScheduler()
        # Initial queue/start, suspend, resume queue, then interrupt AFTER the
        # semaphore was acquired. The retry must not wait for its own permit.
        times = [1., 2., 3., 4., KeyboardInterrupt('clock interrupted'), 5., 6., 7.]
        with mock.patch.object(scheduler.time, 'monotonic', side_effect=times):
            with slots.slot('preview'):
                with self.assertRaises(KeyboardInterrupt):
                    with scheduler.shared_flight_wait():
                        pass
                self.assertEqual(slots.snapshot()['active']['preview'], 1)
        self.assertEqual(slots.snapshot()['completed']['preview'], 1)
        self.assert_idle(slots)

    def test_unrecoverable_resume_failure_exits_without_busy_retry(self):
        slots = scheduler.RenderScheduler()
        sem = slots._slots['preview']
        acquire = sem.acquire
        calls = 0
        def failed(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise RuntimeError('broken admission')
            return acquire(*args, **kwargs)
        with mock.patch.object(sem, 'acquire', side_effect=failed):
            with self.assertRaisesRegex(RuntimeError, 'broken admission'):
                with slots.slot('preview'):
                    with scheduler.shared_flight_wait():
                        pass
        self.assertEqual(calls, 2)
        self.assert_idle(slots)
        with slots.slot('preview'):
            self.assertEqual(slots.snapshot()['active']['preview'], 1)
        self.assert_idle(slots)

    def test_plan_balance_and_auto_ev_subscribers_do_not_block_other_preview(self):
        for kind in ('plan', 'balance', 'auto_ev'):
            with self.subTest(kind=kind):
                slots = scheduler.RenderScheduler()
                entry = pc.PreviewEntry(_bundle(), _analysis())
                entered, release, parked = threading.Event(), threading.Event(), threading.Event()
                other_entered, other_release = threading.Event(), threading.Event()
                value = object()
                def build():
                    entered.set()
                    if not release.wait(2):
                        raise RuntimeError('owner gate did not release')
                    return value
                builder = mock.Mock(side_effect=build)
                def owner():
                    with slots.slot('prepare'):
                        return entry.get_or_compute(kind, 'same', builder)
                def subscriber():
                    with slots.slot('preview'):
                        result = entry.get_or_compute(kind, 'same', builder)
                        self.assertEqual(slots.snapshot()['active']['preview'], 1)
                        return result
                def unrelated():
                    with slots.slot('preview'):
                        other_entered.set()
                        if not other_release.wait(2):
                            raise RuntimeError('other preview gate did not release')
                with mock.patch.object(pc, 'shared_flight_wait', side_effect=lambda: self.parked_wait(parked)), \
                        ThreadPoolExecutor(max_workers=3) as pool:
                    first = pool.submit(owner)
                    self.assertTrue(entered.wait(2))
                    follower = pool.submit(subscriber)
                    try:
                        self.assertTrue(parked.wait(2))
                        other = pool.submit(unrelated)
                        self.assertTrue(other_entered.wait(2), 'subscriber kept the preview permit')
                        release.set()
                        self.assertIs(first.result(timeout=2), value)
                        self.assertFalse(follower.done(), 'subscriber continued before reacquiring its slot')
                    finally:
                        release.set()
                        other_release.set()
                    other.result(timeout=2)
                    self.assertIs(follower.result(timeout=2), value)
                builder.assert_called_once()
                self.assert_idle(slots)

    def test_cold_raw_subscriber_releases_prepare_for_another_cached_file(self):
        slots, cache = scheduler.RenderScheduler(), pc.PreviewCache()
        entry = pc.PreviewEntry(_bundle(), _analysis())
        cached = pc.PreviewEntry(_bundle(), _analysis())
        cache.entries[('b',)] = cached
        entered, release, parked = threading.Event(), threading.Event(), threading.Event()
        def load(*args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('decode gate did not release')
            return entry.bundle
        def prepare(path):
            with slots.slot('prepare'):
                return cache.get(Path(path), 'clip', 'camera')
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(pc, '_cache_identity', side_effect=lambda path, *a, **kw: ((path.stem,), path.stem)))
            stack.enter_context(mock.patch.object(pc, '_read_disk_entry', return_value=None))
            stack.enter_context(mock.patch.object(pc, 'build_proxy_entry', return_value=entry))
            stack.enter_context(mock.patch.object(pc.DISK_WRITER, 'submit'))
            stack.enter_context(mock.patch.object(pc.dg, 'release_analysis_buffers', side_effect=lambda bundle: bundle))
            decode = stack.enter_context(mock.patch.object(pc.dg, 'load_raw', side_effect=load))
            analyze = stack.enter_context(mock.patch.object(pc.dg, 'analyze', return_value=(entry.analysis, None, None)))
            stack.enter_context(mock.patch.object(pc, 'shared_flight_wait', side_effect=lambda: self.parked_wait(parked)))
            with ThreadPoolExecutor(max_workers=3) as pool:
                owner = pool.submit(cache.get, Path('a.dng'), 'clip', 'camera')
                self.assertTrue(entered.wait(2))
                follower = pool.submit(prepare, 'a.dng')
                try:
                    self.assertTrue(parked.wait(2))
                    self.assertIs(pool.submit(prepare, 'b.dng').result(timeout=2), cached)
                finally:
                    release.set()
                self.assertIs(owner.result(timeout=2), entry)
                self.assertIs(follower.result(timeout=2), entry)
        decode.assert_called_once()
        analyze.assert_called_once()
        self.assert_idle(slots)

    def test_failed_entry_flight_returns_permits_and_next_request_retries(self):
        slots = scheduler.RenderScheduler()
        entry = pc.PreviewEntry(_bundle(), _analysis())
        entered, release, parked = threading.Event(), threading.Event(), threading.Event()
        error = RuntimeError('plan failed')
        def build():
            entered.set()
            if not release.wait(2):
                raise RuntimeError('owner gate did not release')
            raise error
        def participant(kind):
            with slots.slot(kind):
                return entry.get_or_build_plan('same', build)
        with mock.patch.object(pc, 'shared_flight_wait', side_effect=lambda: self.parked_wait(parked)), \
                ThreadPoolExecutor(max_workers=2) as pool:
            owner = pool.submit(participant, 'prepare')
            self.assertTrue(entered.wait(2))
            follower = pool.submit(participant, 'preview')
            try:
                self.assertTrue(parked.wait(2))
            finally:
                release.set()
            for future in (owner, follower):
                with self.assertRaises(RuntimeError) as raised:
                    future.result(timeout=2)
                self.assertIs(raised.exception, error)
        self.assertEqual(entry._runtime_inflight, {})
        self.assertEqual(entry.get_or_build_plan('same', lambda: 7), 7)
        self.assert_idle(slots)

    def test_prepare_rechecks_selection_after_shared_plan_reacquires(self):
        slots, coordinator = scheduler.RenderScheduler(), PreviewCoordinator()
        entry = pc.PreviewEntry(_bundle(), _analysis())
        entered, release, parked = threading.Event(), threading.Event(), threading.Event()
        held, release_holder = threading.Event(), threading.Event()
        def build():
            entered.set()
            if not release.wait(2):
                raise RuntimeError('plan gate did not release')
            return object()
        def owner():
            return entry.get_or_build_plan('plan', build)
        def holder():
            with slots.slot('prepare'):
                held.set()
                if not release_holder.wait(2):
                    raise RuntimeError('prepare holder did not release')
        with tempfile.NamedTemporaryFile(suffix='.dng') as source, ExitStack() as stack:
            stack.enter_context(mock.patch.object(service, 'SCHEDULER', slots))
            stack.enter_context(mock.patch.object(service, 'PREVIEW_COORDINATOR', coordinator))
            stack.enter_context(mock.patch.object(service.PREVIEW_STORE, 'get', return_value=entry))
            stack.enter_context(mock.patch.object(service, '_cached_render_plan', side_effect=lambda *a, **kw: entry.get_or_build_plan('plan', build)))
            detected = stack.enter_context(mock.patch.object(service, 'detected_scene_params'))
            stack.enter_context(mock.patch.object(pc, 'shared_flight_wait', side_effect=lambda: self.parked_wait(parked)))
            with ThreadPoolExecutor(max_workers=3) as pool:
                first = pool.submit(owner)
                self.assertTrue(entered.wait(2))
                follower = pool.submit(service.prepare_preview, dict(input=source.name, previewClient='client', selectionEpoch=1))
                try:
                    self.assertTrue(parked.wait(2))
                    holding = pool.submit(holder)
                    self.assertTrue(held.wait(2))
                    coordinator.register_selection('client', 2)
                    release.set()
                    first.result(timeout=2)
                    self.assertFalse(follower.done())
                finally:
                    release.set()
                    release_holder.set()
                holding.result(timeout=2)
                self.assertTrue(follower.result(timeout=2)['superseded'])
        detected.assert_not_called()
        self.assert_idle(slots)

    def test_preview_pixel_hit_rechecks_selection_after_plan_wait(self):
        entry = pc.PreviewEntry(_bundle(), _analysis())
        current = threading.Event()
        current.set()
        def became_stale(*args, **kwargs):
            current.clear()
            return object()
        with mock.patch.object(entry, 'get_pixels', return_value=entry.bundle.scene_rec2020_render), \
                mock.patch.object(service, '_cached_render_plan', side_effect=became_stale), \
                mock.patch.object(service, 'scene_ev_histogram') as histogram, \
                mock.patch.object(service, 'preview_b64_from_u8') as encode:
            with self.assertRaises(PreviewSuperseded):
                service.export_preview_jpeg(Path('a.dng'), 'clip', 'srgb', 0., 95,
                                            cached=entry, is_current=current.is_set)
        histogram.assert_not_called()
        encode.assert_not_called()


if __name__ == '__main__':
    unittest.main()
