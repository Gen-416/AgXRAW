# SPDX-License-Identifier: GPL-3.0-or-later
"""Generation tracking for latest-wins realtime preview requests."""
from __future__ import annotations

import threading
from collections import OrderedDict


class PreviewCoordinator:
    """Remember only the newest generation for a bounded set of browser sessions.

    HTTP handlers may still be waiting on the render lock, but stale handlers become
    constant-time no-ops before they compile a plan or touch pixels.  A running render
    checks the same generation before metrics/encoding, so it can never publish late.
    """

    def __init__(self, max_sessions: int = 64) -> None:
        self._max_sessions = max(1, int(max_sessions))
        self._latest: OrderedDict[str, int] = OrderedDict()
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


PREVIEW_COORDINATOR = PreviewCoordinator()
