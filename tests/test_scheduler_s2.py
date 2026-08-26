# SPDX-License-Identifier: GPL-3.0-or-later
"""Scheduler plan S2 acceptance gates: independent per-class slots replace
RENDER_LOCK — a slow export cannot block previews, stale previews are
dropped at the slot boundary and the drop is observable."""
from __future__ import annotations

import threading
import time
import unittest


class SchedulerSlotTests(unittest.TestCase):
    def _scheduler(self):
        from dngscan.gui.scheduler import RenderScheduler

        return RenderScheduler()

    def test_export_does_not_block_preview(self) -> None:
        sched = self._scheduler()
        export_running = threading.Event()
        release_export = threading.Event()
        preview_done = threading.Event()

        def slow_export():
            with sched.slot("export"):
                export_running.set()
                self.assertTrue(release_export.wait(timeout=5),
                                "export gate never released")

        def quick_preview():
            self.assertTrue(export_running.wait(timeout=5),
                            "export never started")
            with sched.slot("preview"):
                preview_done.set()

        t1 = threading.Thread(target=slow_export)
        t2 = threading.Thread(target=quick_preview)
        t1.start(); t2.start()
        finished = preview_done.wait(timeout=1.0)
        release_export.set()
        t1.join(); t2.join()
        self.assertTrue(
            finished,
            "a preview must complete WHILE an export holds its slot — the "
            "retired RENDER_LOCK serialized them",
        )

    def test_same_class_serializes_and_slot_releases_on_error(self) -> None:
        sched = self._scheduler()
        order: list[str] = []

        errors: list[BaseException] = []

        def job(tag, fail=False):
            # R11 item 6: an exception escaping a bare Thread target is
            # printed by the interpreter and unittest still passes — the
            # test must CATCH it and assert on it in the main thread.
            try:
                with sched.slot("preview"):
                    order.append(tag)
                    time.sleep(0.05)
                    if fail:
                        raise RuntimeError("boom")
            except BaseException as exc:  # noqa: BLE001 - re-asserted below
                errors.append(exc)

        t1 = threading.Thread(target=lambda: job("a", fail=True))
        t1.start(); t1.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        # the slot must be free again despite the exception
        t2 = threading.Thread(target=lambda: job("b"))
        t2.start(); t2.join(timeout=2)
        self.assertFalse(t2.is_alive(), "slot was not released after the error")
        self.assertEqual(order, ["a", "b"])
        self.assertEqual(len(errors), 1)
        snap = sched.snapshot()
        self.assertEqual(snap["active"]["preview"], 0)
        self.assertEqual(snap["completed"]["preview"], 2)

    def test_stale_drop_is_observable(self) -> None:
        sched = self._scheduler()
        sched.note_dropped()
        sched.note_dropped()
        self.assertEqual(sched.snapshot()["dropped_stale"], 2)

    def test_service_no_longer_uses_a_global_render_lock(self) -> None:
        import inspect

        from dngscan.gui import service

        src = inspect.getsource(service)
        self.assertNotIn("RENDER_LOCK = threading.Lock()", src)
        for fn, kind in (
            (service.export_preview_jpeg, "preview"),
            (service.prepare_preview, "prepare"),
            (service.run_export_isolated, "export"),
        ):
            self.assertIn(
                f'SCHEDULER.slot("{kind}")', inspect.getsource(fn),
                f"{fn.__name__} must hold the {kind} slot",
            )
        # the drop point sits at the preview slot boundary
        self.assertIn("note_dropped", inspect.getsource(service.export_preview_jpeg))


if __name__ == "__main__":
    unittest.main()
