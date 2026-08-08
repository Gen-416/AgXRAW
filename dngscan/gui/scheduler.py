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
from contextlib import contextmanager

_QUOTAS = {"preview": 1, "prepare": 1, "export": 1}


class RenderScheduler:
    def __init__(self) -> None:
        self._slots = {
            kind: threading.BoundedSemaphore(quota)
            for kind, quota in _QUOTAS.items()
        }
        self._lock = threading.Lock()
        self.active: dict[str, int] = {kind: 0 for kind in _QUOTAS}
        self.completed: dict[str, int] = {kind: 0 for kind in _QUOTAS}
        self.dropped_stale = 0

    @contextmanager
    def slot(self, kind: str):
        sem = self._slots[kind]
        sem.acquire()
        with self._lock:
            self.active[kind] += 1
        try:
            yield
        finally:
            with self._lock:
                self.active[kind] -= 1
                self.completed[kind] += 1
            sem.release()

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
                "dropped_stale": self.dropped_stale,
            }


SCHEDULER = RenderScheduler()
