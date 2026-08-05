# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared 3D LUT (.cube) load + trilinear sample."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._deps import np

_CUBE_CACHE: dict[Path, np.ndarray] = {}


def load_cube(path: Path) -> np.ndarray:
    size = None
    rows: list[list[float]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("TITLE"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
                continue
            if line.startswith(("DOMAIN_MIN", "DOMAIN_MAX", "LUT_1D", "LUT_3D_INPUT_RANGE")):
                continue
            parts = line.split()
            if len(parts) == 3:
                rows.append([float(v) for v in parts])
    if size is None or len(rows) != size**3:
        raise RuntimeError(f"bad cube {path}: size={size} rows={len(rows)}")
    return np.asarray(rows, dtype=np.float64).reshape(size, size, size, 3).transpose(2, 1, 0, 3)


def get_cube(path: Path) -> np.ndarray:
    key = path.resolve()
    cached = _CUBE_CACHE.get(key)
    if cached is None:
        cached = load_cube(key)
        _CUBE_CACHE[key] = cached
    return cached


def sample_cube(lut: np.ndarray, rgb01: np.ndarray) -> np.ndarray:
    n = lut.shape[0]
    coords = np.clip(rgb01, 0.0, 1.0) * (n - 1)
    lo = np.floor(coords).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = coords - lo
    out = np.zeros_like(rgb01)
    for corner in range(8):
        ix = hi[:, 0] if corner & 1 else lo[:, 0]
        iy = hi[:, 1] if corner & 2 else lo[:, 1]
        iz = hi[:, 2] if corner & 4 else lo[:, 2]
        wx = frac[:, 0] if corner & 1 else 1.0 - frac[:, 0]
        wy = frac[:, 1] if corner & 2 else 1.0 - frac[:, 1]
        wz = frac[:, 2] if corner & 4 else 1.0 - frac[:, 2]
        out += lut[ix, iy, iz] * (wx * wy * wz)[:, None]
    return out.astype(np.float32)
