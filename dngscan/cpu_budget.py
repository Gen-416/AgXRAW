# SPDX-License-Identifier: GPL-3.0-or-later
"""Process CPU budget (RENDER_SCHEDULER_PLAN S3): one owner for how much
parallelism a render may spend, rented DOWN the stack instead of each layer
optimizing locally.

- The outer streaming pools ask ``outer_workers`` for their size.
- Before a pooled section runs, the renderer publishes each worker's INNER
  budget (``TOTAL // outer``); operators and the native kernels read it via
  ``current_inner`` / the C++ atomic and go serial (or narrow) when their
  share is 1 — the review measured dozens of concurrent threads from the
  old three-layer nesting (Python pool x operator pools x C++ per-chunk
  threads).
- ``inner`` is thread-local, so a sequential caller (probes, tests, the
  spatial band path) keeps the whole machine.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager

TOTAL = max(1, os.cpu_count() or 4)

_tls = threading.local()


def outer_workers(wanted: int) -> int:
    """Clamp an outer pool's size to the machine budget."""
    return max(1, min(int(wanted), TOTAL))


def split_for(native_core: bool) -> tuple[int, int]:
    """(outer workers, inner budget) for a chunk-streaming render.

    MEASURED (10-core M-series, 16 MP frame, thread peak sampled from a
    parent process so the GIL cannot starve the sampler):

        native AgX core   unbudgeted 49 thr / 1.43 s
                          6x1  7 thr / 2.63 s     2x5  15 thr / 1.50 s
        NumPy film core   unbudgeted 14 thr / 3.09 s
                          6x1  7 thr / 3.22 s     6x2   8 thr / 3.07 s

    The two paths want OPPOSITE splits: when the C++ kernel does the heavy
    lifting, few outer workers with a wide native budget wins; when the core
    is NumPy (film takeover, gated, lum), the outer pipeline is what scales
    and the native budget only serves the output finalizer. Dividing the
    machine evenly (TOTAL // outer -> 6x1) is the worst of both, so the
    split follows WHO DOES THE WORK.
    """
    if native_core:
        outer = max(2, min(3, TOTAL // 4))
    else:
        outer = max(2, min(6, TOTAL - 2))
    return outer, max(2, TOTAL // outer)


def inner_budget_for(outer: int) -> int:
    """Each outer worker's fair share of the machine."""
    return max(1, TOTAL // max(1, int(outer)))


def set_inner(budget: int) -> None:
    _tls.inner = max(1, int(budget))


def current_inner() -> int:
    """The parallelism THIS thread may still spend (default: the machine)."""
    return int(getattr(_tls, "inner", TOTAL))


@contextmanager
def inner(budget: int):
    prev = getattr(_tls, "inner", None)
    set_inner(budget)
    try:
        yield
    finally:
        if prev is None:
            try:
                del _tls.inner
            except AttributeError:
                pass
        else:
            _tls.inner = prev


_native_lock = threading.Lock()
_native_claims: list[int] = []


@contextmanager
def native_budget(budget: int):
    """Publish a native-kernel thread budget for the duration of a pooled
    section. Concurrent claims are safe: the TIGHTEST wins while it is held,
    and releasing restores the tightest remaining claim (or "unlimited" when
    none is left) — a plain set/reset pair would let one render's cleanup
    hand the whole machine back to another render still inside its pool."""
    from . import _fast

    budget = max(1, int(budget))
    with _native_lock:
        _native_claims.append(budget)
        _fast.set_thread_budget(min(_native_claims))
    try:
        yield
    finally:
        with _native_lock:
            _native_claims.remove(budget)
            _fast.set_thread_budget(min(_native_claims) if _native_claims else 0)


@contextmanager
def claim(share: int):
    """The whole S3 claim for one pooled section: the caller's own inner
    budget plus the native one. Worker threads receive the same share via
    the pool initializer."""
    with native_budget(share), inner(share):
        yield share


def ordered_budget_map(pool, fn, values, *, max_workers: int):
    """Run independent items within this caller's share and join before return.

    The existing executor may be wider than the caller's budget. Submit only
    bounded batches, propagate each batch's inner share, and preserve item order.
    A failed submit or computation still drains every successfully submitted
    future before releasing the native claim or the input owners.
    """
    values = tuple(values)
    if not values:
        return []
    available = max(1, current_inner())
    workers = min(len(values), max(1, int(max_workers)), available)
    if workers == 1:
        with claim(available):
            return [fn(value) for value in values]
    width = (len(values) + workers - 1) // workers
    batches = [values[start:start + width] for start in range(0, len(values), width)]
    share = max(1, available // len(batches))

    def run(batch):
        with inner(share):
            return [fn(value) for value in batch]

    futures, results = [], []
    failure = None
    with native_budget(share):
        try:
            for batch in batches:
                futures.append(pool.submit(run, batch))
        except BaseException as exc:
            failure = exc
        for future in futures:
            while True:
                try:
                    results.extend(future.result())
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                    # A caller interrupt can abort result() while its worker
                    # is still running. Keep this claim and block on that same
                    # future again; terminal worker errors need no retry.
                    if not future.done():
                        continue
                break
        if failure is not None:
            raise failure
    return results
