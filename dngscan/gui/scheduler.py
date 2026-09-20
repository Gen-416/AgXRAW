# SPDX-License-Identifier: GPL-3.0-or-later
"""RenderScheduler (RENDER_SCHEDULER_PLAN S2): the single owner of GUI
render concurrency.

Task classes hold INDEPENDENT bounded slots — a slow export no longer
blocks previews, because they never share one mutex the way the retired
RENDER_LOCK forced them to:

- ``preview``:  interactive frames, one at a time (stale generations are
  dropped at the slot boundary and counted, see ``note_dropped``);
- ``prepare``:  session warm-up / detected-plan compiles, one at a time —
  together with ``preview`` this bounds interactive renders to two;
- ``export``:   the isolated export process' parent-side slot; its deadline
  handling lives with the export code (batch 17) and simply releases the
  slot on timeout like any other exit.

The class quotas are the memory contract for concurrent renders; raising
them is a decision about RAM, not a code detail. Cancellation semantics
stay with PreviewCoordinator (generation registry) — the scheduler only
adds the drop point and its observability.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

_QUOTAS = {"preview": 1, "prepare": 1, "export": 1}
_local = threading.local()


class _SlotLease:
    """One request's admission, temporarily returnable while another owner works."""

    def __init__(self, scheduler, kind):
        self.scheduler, self.kind = scheduler, kind
        self.sem = scheduler._slots[kind]
        self.held = False
        self.started = None
        self.paused = None

    def acquire(self):
        queued = time.monotonic()
        if self.paused is not None:
            with self.scheduler._lock:
                self.scheduler.shared_wait_seconds[self.kind] += queued - self.paused
            self.paused = None
        try:
            self.sem.acquire()
        except BaseException:
            with self.scheduler._lock:
                self.scheduler.queue_seconds[self.kind] += time.monotonic() - queued
            raise
        try:
            started = time.monotonic()
            with self.scheduler._lock:
                active = self.scheduler.active[self.kind]
                elapsed = self.scheduler.queue_seconds[self.kind]
                try:
                    self.scheduler.active[self.kind] = active + 1
                    self.scheduler.queue_seconds[self.kind] = elapsed + started - queued
                    self.started = started
                    self.held = True
                except BaseException:
                    self.scheduler.active[self.kind] = active
                    self.scheduler.queue_seconds[self.kind] = elapsed
                    self.held = False
                    raise
        except BaseException:
            # Acquisition succeeded but clock/accounting registration failed.
            # Never retry while still owning the semaphore's sole permit.
            if self.held:
                with self.scheduler._lock:
                    self.scheduler.active[self.kind] -= 1
                    self.scheduler.queue_seconds[self.kind] -= started - queued
                self.held = False
            self.sem.release()
            raise

    def release(self, *, pause=False):
        finished = time.monotonic()
        with self.scheduler._lock:
            self.scheduler.active[self.kind] -= 1
            self.scheduler.execute_seconds[self.kind] += finished - self.started
        self.held = False
        self.paused = finished if pause else None
        self.sem.release()

    def finish(self):
        if self.held:
            self.release()
        elif self.paused is not None:
            with self.scheduler._lock:
                self.scheduler.shared_wait_seconds[self.kind] += time.monotonic() - self.paused
            self.paused = None
        with self.scheduler._lock:
            self.scheduler.completed[self.kind] += 1

    def resume(self):
        # A handler may catch a cancellation inside its outer slot and keep
        # working. Restore admission before propagating it, just as a lock must
        # be restored after a condition wait. No computation runs in this loop.
        interrupted = None
        while True:
            try:
                self.acquire()
                break
            except (KeyboardInterrupt, InterruptedError, SystemExit) as exc:
                if interrupted is None:
                    interrupted = exc
        if interrupted is not None:
            raise interrupted


@contextmanager
def shared_flight_wait():
    """Return only interactive admission while waiting for an existing flight.

    Callers must release their cache locks before entering, and may only wait
    inside this scope. Reacquisition completes before they can use the result.
    Export deadlines and admission are never changed. Outside a scheduler slot
    (including normal AutoEV subscriptions), this is a no-op.
    """
    leases = [lease for lease in getattr(_local, "leases", ())
              if lease.held and lease.kind in ("preview", "prepare")]
    for lease in reversed(leases):
        lease.release(pause=True)
    try:
        yield
    finally:
        # Even a failed flight must restore admission before an enclosing
        # handler can catch its exception and continue work. Reacquisition
        # interruptions propagate only after that same admission is restored.
        for lease in leases:
            lease.resume()


class RenderScheduler:
    def __init__(self) -> None:
        self._slots = {
            kind: threading.BoundedSemaphore(quota)
            for kind, quota in _QUOTAS.items()
        }
        self._lock = threading.Lock()
        self.active: dict[str, int] = {kind: 0 for kind in _QUOTAS}
        self.completed: dict[str, int] = {kind: 0 for kind in _QUOTAS}
        self.queue_seconds: dict[str, float] = {kind: 0.0 for kind in _QUOTAS}
        self.execute_seconds: dict[str, float] = {kind: 0.0 for kind in _QUOTAS}
        self.shared_wait_seconds: dict[str, float] = {kind: 0.0 for kind in _QUOTAS}
        self.dropped_stale = 0

    @contextmanager
    def slot(self, kind: str):
        leases = getattr(_local, "leases", None)
        if leases is None:
            leases = _local.leases = []
        existing = next((lease for lease in leases
                         if lease.scheduler is self and lease.kind == kind), None)
        if existing is not None:
            if not existing.held:
                raise RuntimeError("cannot start computation inside a shared-flight wait")
            # Reentrant helpers borrow this request's existing admission.
            yield
            return
        if leases:
            # No production path nests categories. Reject an accidental
            # cross-category/scheduler nesting before acquiring anything:
            # restoring two leases could otherwise create an ABBA wait cycle.
            raise RuntimeError("nested scheduler slots must use the same scheduler and kind")
        lease = _SlotLease(self, kind)
        lease.acquire()
        leases.append(lease)
        try:
            yield
        finally:
            try:
                lease.finish()
            finally:
                leases.remove(lease)

    def note_dropped(self) -> None:
        """A queued preview found itself stale when its slot arrived and was
        dropped instead of rendering — the observable S2 acceptance signal."""
        with self._lock:
            self.dropped_stale += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "active": dict(self.active),
                "completed": dict(self.completed),
                "queue_seconds": dict(self.queue_seconds),
                "execute_seconds": dict(self.execute_seconds),
                "shared_wait_seconds": dict(self.shared_wait_seconds),
                "dropped_stale": self.dropped_stale,
            }


SCHEDULER = RenderScheduler()
