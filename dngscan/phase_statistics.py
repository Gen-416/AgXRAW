# SPDX-License-Identifier: GPL-3.0-or-later
"""Job-local exact CFA tile reductions with bounded corrected-DN storage."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ._deps import np


@dataclass
class PhaseStatistics:
    snr: dict
    noise: list
    health: tuple[float, float]

    def noise_floor(self, bundle, fullwell):
        estimates = []
        for cid, sigma in self.noise:
            black = float(bundle.black_levels[cid]) if cid < len(bundle.black_levels) else 0.0
            denom = max(float(fullwell.get(cid, bundle.white_level)) - black, 1.0)
            estimates.append(max(sigma, 1.0 / math.sqrt(12.0)) / denom)
        return float(np.median(estimates)) if estimates else float('nan')


def _corrected_band(bundle, position, period, start, stop):
    """The original corrected_plane expression, restricted to phase rows."""
    yoff, xoff = position
    ph, pw = period
    y0, y1 = yoff + start * ph, min(bundle.raw_image.shape[0], yoff + stop * ph)
    band = bundle.raw_image[y0:y1:ph, xoff::pw].astype(np.float32)
    model = getattr(getattr(bundle, 'evidence', None), 'spatial_black', None)
    if model is not None:
        cid = int(bundle.raw_colors[yoff, xoff])
        base = float(bundle.black_levels[cid])
        band -= model.band(y0, y1, bundle.raw_image.shape[1])[::ph, xoff::pw] - base
    return band


def _health_tiles(first, second, width, tile):
    difference = first[:, :width] - second[:, :width]
    return difference.reshape(len(difference) // tile, tile, width // tile, tile).transpose(
        0, 2, 1, 3).reshape(-1, tile, tile)


def _histogram_percentile(histogram, percentile):
    """NumPy int64 linear percentile, using its exact rank and lerp order."""
    count = int(histogram.sum())
    virtual = (count - 1) * (np.float64(percentile) / np.float64(100))
    lo, hi = int(np.floor(virtual)), int(np.ceil(virtual))
    cumulative = np.cumsum(histogram)
    a = np.int64(np.searchsorted(cumulative, lo, side='right'))
    b = np.int64(np.searchsorted(cumulative, hi, side='right'))
    fraction = virtual - lo
    return float(b - (b - a) * (1 - fraction) if fraction >= .5
                 else a + (b - a) * fraction)


def _histogram_health(histogram):
    if not histogram.sum():
        return float('nan')
    p05, p60 = (_histogram_percentile(histogram, p) for p in (5., 60.))
    lo, hi = int(p05), int(max(p60, p05 + 32))
    # Clipping only adds counts to the two excluded endpoints. Empty values
    # above uint16's range remain present in the old histogram's interior.
    interior = np.zeros(max(0, hi - lo - 1), dtype=np.bool_)
    end = min(hi, len(histogram))
    if end > lo + 1:
        interior[:end - lo - 1] = histogram[lo + 1:end] != 0
    return float(np.mean(~interior) * 100.) if interior.size else float('nan')


def _health_lag(bundle, positions, period, height, width, variances, tile):
    if not variances:
        return float('nan')
    values = np.concatenate(variances)
    count = max(16, int(math.ceil(values.size * .25)))
    selected = np.argsort(values)[:min(count, 768)]
    # Preserve global argsort's tie order. Reload only selected tiles after
    # selection; never retain the complete green difference image.
    tiles = np.empty((len(selected), tile, tile), dtype=np.float32)
    tile_columns = width // tile
    phase_rows = (selected // tile_columns) * tile
    starts = (phase_rows // 128) * 128
    for start in np.unique(starts):
        start = int(start)
        stop = min(start + 128, height)
        first = _corrected_band(bundle, positions[0], period, start, stop)
        second = _corrected_band(bundle, positions[1], period, start, stop)
        band_tiles = _health_tiles(first, second, width, tile)
        take = np.flatnonzero(starts == start)
        local = selected[take] - (start // tile) * tile_columns
        tiles[take] = band_tiles[local]
    sel = tiles - tiles.mean(axis=(1, 2), keepdims=True)
    num_h = np.sum(sel[:, :, :-1] * sel[:, :, 1:], dtype=np.float64)
    den_h = math.sqrt(float(np.sum(sel[:, :, :-1] ** 2, dtype=np.float64))
                      * float(np.sum(sel[:, :, 1:] ** 2, dtype=np.float64)))
    num_v = np.sum(sel[:, :-1, :] * sel[:, 1:, :], dtype=np.float64)
    den_v = math.sqrt(float(np.sum(sel[:, :-1, :] ** 2, dtype=np.float64))
                      * float(np.sum(sel[:, 1:, :] ** 2, dtype=np.float64)))
    return float(.5 * (num_h / den_h + num_v / den_v)) if den_h > 0 and den_v > 0 else float('nan')


def build_phase_statistics(bundle, channel_ids, labels) -> PhaseStatistics | None:
    """Supported sensor storage shares noise/SNR/health in one bounded walk.

    Public standalone computations and unusual layouts keep their original
    oracle. No statistics survive this analyze() invocation.
    """
    raw = bundle.raw_image
    if (type(raw) is not np.ndarray or raw.ndim != 2 or raw.dtype != np.uint16
            or not raw.flags.c_contiguous or not raw.size):
        return None
    try:
        pattern = np.asarray(bundle.raw_pattern)
    except (TypeError, ValueError):
        return None
    if pattern.ndim != 2 or not pattern.size or min(pattern.shape) <= 0:
        return None
    from .analysis import cfa_positions_for_channel, SNR_TILE

    period = tuple(pattern.shape)
    ph, pw = period
    # analyze() supplies the complete, sorted SensorSummary channel IDs.
    # Do not sort/copy the full color plane again for this second consumer.
    configurations = [(int(cid), y, x) for cid in channel_ids
                      for y, x in cfa_positions_for_channel(bundle, int(cid))]
    green = [position for cid in channel_ids if labels[cid].startswith('G')
             for position in cfa_positions_for_channel(bundle, cid)]
    shapes = {key: raw[key[1]::ph, key[2]::pw].shape for key in configurations}
    # Very narrow phases have the historical adaptive noise tile (4..15),
    # independently of SNR's fixed 16. Keep the oracle when a 128-row shared
    # band would cut that tile; ordinary sensors always use 16 here.
    if any(h > 128 and 4 <= min(SNR_TILE, h, w) < SNR_TILE
           and 128 % min(SNR_TILE, h, w) for h, w in shapes.values()):
        return None
    reductions = {key: {'mean': [], 'sigma': [], 'signal': [], 'std': []} for key in configurations}
    health_height = health_width = 0
    if len(green) >= 2:
        h, w = map(min, zip(*(raw[y::ph, x::pw].shape for y, x in green[:2])))
        health_height, health_width = h // SNR_TILE * SNR_TILE, w // SNR_TILE * SNR_TILE
    variances = []
    histogram = np.zeros(65536, dtype=np.int64) if green else None
    for start in range(0, max((s[0] for s in shapes.values()), default=0), 128):
        green_bands = {}
        for key in configurations:
            cid, yoff, xoff = key
            h, w = shapes[key]
            stop = min(start + 128, h)
            if stop <= start:
                continue
            band = _corrected_band(bundle, (yoff, xoff), period, start, stop)
            if (yoff, xoff) in green[:2]:
                green_bands[(yoff, xoff)] = band
            if green and (yoff, xoff) == green[0]:
                raw_band = raw[yoff + start * ph:min(raw.shape[0], yoff + stop * ph):ph, xoff::pw]
                histogram += np.bincount(raw_band.reshape(-1), minlength=65536)
            tile = min(SNR_TILE, h, w)
            target = reductions[key]
            if tile < 4:
                continue
            h2, w2 = h // tile * tile, w // tile * tile
            rows = min(stop, h2) - start
            if rows <= 0:
                continue
            cells = band[:rows, :w2].reshape(rows // tile, tile, w2 // tile, tile)
            means = cells.mean(axis=(1, 3), dtype=np.float64).ravel()
            residual = np.diff(np.diff(cells, axis=1), axis=3)
            sigma = (np.median(np.abs(residual), axis=(1, 3)) / 1.3489795).ravel()
            target['mean'].append(means)
            target['sigma'].append(sigma)
            if tile == SNR_TILE:
                black = float(bundle.black_levels[cid]) if cid < len(bundle.black_levels) else 0.0
                target['signal'].append(np.maximum(means.astype(np.float32) - np.float32(black), 0.0))
                target['std'].append(np.maximum(cells.std(axis=(1, 3), dtype=np.float64).ravel().astype(np.float32), 0.0))
        rows = min(start + 128, health_height) - start
        if rows > 0 and health_width and len(green_bands) >= 2:
            tiles = _health_tiles(green_bands[green[0]][:rows], green_bands[green[1]][:rows], health_width, SNR_TILE)
            variances.append(tiles.var(axis=(1, 2)))
    noise, snr = [], {}
    for key, data in reductions.items():
        if data['mean']:
            means, sigma = np.concatenate(data['mean']), np.concatenate(data['sigma'])
            count = max(1, int(math.ceil(means.size * .10)))
            dark = np.argpartition(means, count - 1)[:count]
            noise.append((key[0], float(np.median(sigma[dark]))))
        snr[key] = tuple(np.concatenate(data[field]) if data[field] else np.asarray([], dtype=np.float32)
                         for field in ('signal', 'std'))
    lag = (_health_lag(bundle, green, period, health_height, health_width, variances, SNR_TILE)
           if len(green) >= 2 and health_height and health_width else float('nan'))
    hist = _histogram_health(histogram) if histogram is not None else float('nan')
    return PhaseStatistics(snr, noise, (lag, hist))
