# SPDX-License-Identifier: GPL-3.0-or-later
"""Generation tracking for latest-wins realtime preview requests."""
from __future__ import annotations

import threading
from collections import OrderedDict


class PreviewSuperseded(RuntimeError):
    """This subscriber has moved to another selection or frame."""


class PreviewCoordinator:
    """Remember only the newest generation for a bounded set of browser sessions.

    HTTP handlers may still be queued on a RenderScheduler slot, but stale handlers
    become constant-time no-ops before they compile a plan or touch pixels.  A running render
    checks the same generation before metrics/encoding, so it can never publish late.
    """

    def __init__(self, max_sessions: int = 64) -> None:
        self._max_sessions = max(1, int(max_sessions))
        self._latest: OrderedDict[str, int] = OrderedDict()
        self._selections: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()

    def register(self, session: str, generation: int) -> bool:
        if generation <= 0:
            return True
        with self._lock:
            current = self._latest.get(session, 0)
            if generation < current:
                return False
            self._latest[session] = generation
            self._latest.move_to_end(session)
            while len(self._latest) > self._max_sessions:
                self._latest.popitem(last=False)
            return True

    def is_current(self, session: str, generation: int) -> bool:
        if generation <= 0:
            return True
        with self._lock:
            return self._latest.get(session) == generation

    def clear(self) -> None:
        with self._lock:
            self._latest.clear()
            self._selections.clear()

    def register_selection(self, client: str, epoch: int) -> bool:
        if not client or epoch <= 0:
            return True
        with self._lock:
            if epoch < self._selections.get(client, 0):
                return False
            self._selections[client] = epoch
            self._selections.move_to_end(client)
            while len(self._selections) > self._max_sessions:
                self._selections.popitem(last=False)
            return True

    def selection_is_current(self, client: str, epoch: int) -> bool:
        if not client or epoch <= 0:
            return True
        with self._lock:
            return self._selections.get(client) == epoch


PREVIEW_COORDINATOR = PreviewCoordinator()
