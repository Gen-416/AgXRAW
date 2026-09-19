# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, deterministic scene samples without a periodic pixel lattice."""
from ._deps import np


def sample_indices(pixel_count: int, max_samples: int = 800_000):
    """One uniformly placed pixel per equal-area stratum, in source order.

    A fixed stride can miss an entire periodic highlight population. Integer
    hashing gives each stratum a different offset; neither image content nor a
    process-global random seed changes the sample or its matching evidence.
    """
    n = int(pixel_count)
    count = min(n, max(1, int(max_samples)))
    if n <= count:
        return np.arange(n, dtype=np.intp)
    k = np.arange(count, dtype=np.uint64)
    lo = k * np.uint64(n) // np.uint64(count)
    hi = (k + np.uint64(1)) * np.uint64(n) // np.uint64(count)
    z = k + np.uint64(0x9e3779b97f4a7c15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    z ^= z >> np.uint64(31)
    return (lo + z % (hi - lo)).astype(np.intp)
