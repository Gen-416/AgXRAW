# SPDX-License-Identifier: GPL-3.0-or-later
"""Capture-scoped metadata reuse; no persistent path or decoder-result cache."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
import hashlib
import os
from pathlib import Path
from threading import RLock
from typing import Any, Callable


def _path_key(path: Path) -> str:
    # Keep the lexical path: stat must observe replacement of a symlink too.
    return os.path.abspath(os.fspath(path))


def _stat_key(stat) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True)
class SourceIdentity:
    path: str
    stat: tuple[int, int, int, int, int]
    header_sha256: str


def _source_identity(path: Path) -> SourceIdentity | None:
    try:
        with path.open("rb") as source:
            before = _stat_key(os.fstat(source.fileno()))
            header = source.read(4096)
            after = _stat_key(os.fstat(source.fileno()))
        if before != after or after != _stat_key(path.stat()):
            return None
        return SourceIdentity(_path_key(path), before, hashlib.sha256(header).hexdigest())
    except (OSError, ValueError, AttributeError):
        # A missing/unsupported identity disables reuse, never the original reader.
        return None


class SourceMetadataSession:
    """One load's identity and isolated copies of successfully read metadata.

    A source change invalidates the whole session permanently. The caller can
    use is_current before and after evidence acquisition to authorize reuse of
    that same evidence during decoder fallback; this object never owns RAW data.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.identity = _source_identity(self.path)
        self._invalid = self.identity is None
        self._memo: dict[Callable, Any] = {}
        self._lock = RLock()

    def is_current(self, path: Path | None = None) -> bool:
        candidate = self.path if path is None else Path(path)
        with self._lock:
            if self._invalid or self.identity is None:
                return False
            if _path_key(candidate) != self.identity.path:
                return False
            try:
                matches = _stat_key(candidate.stat()) == self.identity.stat
            except OSError:
                matches = False
            if not matches:
                self._invalid = True
                self._memo.clear()
            return matches

    def read(self, reader: Callable, path: Path, *, cacheable: Callable | None = None):
        with self._lock:
            if not self.is_current(path):
                return reader(path)
            if reader in self._memo:
                return deepcopy(self._memo[reader])
            value = reader(path)
            if self.is_current(path) and (cacheable is None or cacheable(value)):
                # Parsed OpcodePlan is intentionally mutable during execution.
                # Store a private snapshot and give every reuse its own object.
                self._memo[reader] = deepcopy(value)
            return value


_SESSION: ContextVar[SourceMetadataSession | None] = ContextVar("agxraw_source_metadata", default=None)


@contextmanager
def source_metadata_session(path: Path):
    existing = _SESSION.get()
    if existing is not None and _path_key(path) == _path_key(existing.path):
        # Retain an invalidated session as invalid. A nested decoder fallback
        # must not silently establish a fresh identity for old RAW evidence.
        yield existing
        return
    session = SourceMetadataSession(Path(path))
    token = _SESSION.set(session)
    try:
        yield session
    finally:
        _SESSION.reset(token)
        session._memo.clear()


def cached_source_metadata(reader=None, *, cacheable: Callable | None = None):
    """Reuse a path-only reader inside a source_metadata_session, otherwise pass through."""
    if reader is None:
        return lambda fn: cached_source_metadata(fn, cacheable=cacheable)

    @wraps(reader)
    def wrapped(path, *args, **kwargs):
        session = _SESSION.get()
        if session is None or args or kwargs:
            return reader(path, *args, **kwargs)
        return session.read(reader, path, cacheable=cacheable)

    return wrapped
