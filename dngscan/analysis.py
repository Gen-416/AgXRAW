# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-frame RAW sensor analysis and metrics."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from dataclasses import replace
from typing import Any

from ._deps import np
from . import priors as sensor_priors
from .color import clamp_float, XYZ_TO_RGB
from .constants import (
    CEILING_MIN_PILE_FRACTION, CEILING_MIN_PILE_PIXELS, CEILING_NEAR_WINDOW_SCALE,
    CEILING_PLAUSIBLE_FRACTION, EPS, EV_REPORT_FLOOR, GAMUT_EPS,
    GRAY_EV, MIDGRAY_HEADROOM_STOPS, NOISE_DR_EPS, SNR_BRIGHT_UNRELIABLE_STOP,
    SNR_LOW_PERCENTILE, SNR_TILE,
)
from .models import Analysis, RawBundle

def channel_labels(color_desc: str, channel_ids: list[int]) -> dict[int, str]:
    chars = list(color_desc)
    max_id = max(channel_ids) if channel_ids else -1
    while len(chars) <= max_id:
        chars.append("")

    totals: dict[str, int] = {}
    for ch in chars:
        base = ch if ch and ch.isprintable() else "C"
        totals[base] = totals.get(base, 0) + 1

    seen: dict[str, int] = {}
    labels: dict[int, str] = {}
    for idx, ch in enumerate(chars):
        base = ch if ch and ch.isprintable() else "C"
        seen[base] = seen.get(base, 0) + 1
        if totals.get(base, 0) > 1:
            labels[idx] = f"{base}{seen[base]}"
        else:
            labels[idx] = base if base != "C" else f"C{idx}"

    return {cid: labels.get(cid, f"C{cid}") for cid in channel_ids}


def channel_color(label: str) -> str:
    base = label[:1].upper()
    if base == "R":
        return "#d23"
    if base == "G":
        return "#2a8"
    if base == "B":
        return "#36c"
    return "#7f7f7f"


def format_channel_values(
    values: dict[int, Any], labels: dict[int, str], channel_ids: list[int], fmt: str = "{}"
) -> str:
    parts = []
    for cid in channel_ids:
        val = values.get(cid, "")
        parts.append(f"{labels[cid]}={fmt.format(val)}")
    return " ".join(parts)


def format_pct(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if value == 0.0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.6f}"
    if abs(value) < 1.0:
        return f"{value:.4f}"
    return f"{value:.3f}"



def format_snr_dr(values: dict[str, float]) -> str:
    parts = []
    for group in ("R", "G", "B"):
        val = values.get(group, float("nan"))
        if math.isfinite(val):
            parts.append(f"{group}={val:.2f}")
        else:
            parts.append(f"{group}=n/a")
    return " ".join(parts)


def fullwell_note_cn(note: str) -> str:
    mapping = {
        "weak ceiling channels excluded from fullwell": "弱满阱通道已排除",
        "all channels have ceiling pile": "所有通道都有可靠满阱堆积",
        "no strong ceiling pile; fullwell is uncertain": "没有可靠满阱堆积，满阱估计不确定",
        "no ceiling pile; fullwell from metadata white_level": "无满阱堆积，改用元数据 white_level 估计满阱",
    }
    return mapping.get(note, note)


def channel_list(ids: list[int], labels: dict[int, str]) -> str:
    return "/".join(labels[cid] for cid in ids)


def format_wb_values(values: dict[int, float], labels: dict[int, str], channel_ids: list[int]) -> str:
    g_values = [values[cid] for cid in channel_ids if labels[cid].startswith("G") and values[cid] > 0]
    parts = []
    for cid in channel_ids:
        val = values[cid]
        label = labels[cid]
        if label.startswith("G") and val == 0.0 and g_values:
            parts.append(f"{label}=meta0")
        else:
            parts.append(f"{label}={val:.3g}")
    return " ".join(parts)


def padded_channel_values(values: list[float], channel_ids: list[int]) -> dict[int, float]:
    out: dict[int, float] = {}
    for cid in channel_ids:
        out[cid] = float(values[cid]) if cid < len(values) else 0.0
    return out


def detect_ceilings(
    raw_image: Any, raw_colors: Any, channel_ids: list[int],
    sat: dict[int, int] | None = None,
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, bool]]:
    ceilings: dict[int, int] = {}
    exact_counts: dict[int, int] = {}
    near_counts: dict[int, int] = {}
    spike_ok: dict[int, bool] = {}
    for cid in channel_ids:
        vals = raw_image[raw_colors == cid]
        if vals.size == 0:
            raise RuntimeError(f"no visible pixels for raw color channel {cid}")
        ceil = int(np.max(vals))
        exact = int(np.count_nonzero(vals == ceil))
        # A8 item 1: the near-window scales with bit depth — a fixed 2 DN
        # is a different tolerance at 12 and 16 bits. ~2 DN per 14-bit
        # scale, floor 2.
        level = int((sat or {}).get(cid, 0)) or int(np.iinfo(raw_image.dtype).max
                                                    if np.issubdtype(raw_image.dtype, np.integer)
                                                    else 65535)
        window = max(2, int(round(level / CEILING_NEAR_WINDOW_SCALE)))
        near = int(np.count_nonzero(vals >= max(ceil - window, 0)))
        min_pile = max(CEILING_MIN_PILE_PIXELS, int(math.ceil(vals.size * CEILING_MIN_PILE_FRACTION)))
        ceilings[cid] = ceil
        exact_counts[cid] = exact
        near_counts[cid] = near
        spike_ok[cid] = exact >= min_pile or near >= min_pile
    return ceilings, exact_counts, near_counts, spike_ok


def channel_value_map(raw_colors: Any, values: dict[int, Any], default: float, dtype: Any) -> Any:
    max_color = int(np.max(raw_colors)) if raw_colors.size else -1
    levels = np.full(max_color + 1, default, dtype=dtype)
    for cid, value in values.items():
        cid_i = int(cid)
        if 0 <= cid_i <= max_color:
            levels[cid_i] = value
    return levels[raw_colors]


def channel_threshold_map(raw_colors: Any, thresholds: dict[int, int]) -> Any:
    default = int(min(thresholds.values())) if thresholds else 0
    return channel_value_map(raw_colors, thresholds, default, np.int32)


def channel_fullwell_map(raw_colors: Any, fullwell_by_channel: dict[int, int]) -> Any:
    default = float(min(fullwell_by_channel.values())) if fullwell_by_channel else 1.0
    return channel_value_map(raw_colors, fullwell_by_channel, default, np.float32)


def compute_clip_pct_by_thresholds(
    raw_image: Any, raw_colors: Any, channel_ids: list[int], thresholds: dict[int, int]
) -> dict[int, float]:
    out: dict[int, float] = {}
    for cid in channel_ids:
        vals = raw_image[raw_colors == cid]
        threshold = int(thresholds.get(cid, 0))
        out[cid] = float(np.mean(vals >= threshold) * 100.0) if vals.size else 0.0
    return out


def compute_cell_metrics(
    raw_image: Any, raw_colors: Any, thresholds: dict[int, int]
) -> tuple[float, float, dict[int, float], dict[int, float]]:
    h, w = raw_image.shape
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    if h2 == 0 or w2 == 0:
        return 0.0, 0.0, {k: 0.0 for k in range(1, 5)}, {k: 0.0 for k in range(1, 5)}

    threshold_map = channel_threshold_map(raw_colors[:h2, :w2], thresholds)
    clipped = raw_image[:h2, :w2] >= threshold_map
    per_cell = clipped.reshape(h2 // 2, 2, w2 // 2, 2).sum(axis=(1, 3))
    total = int(per_cell.size)
    clipped_cells = per_cell >= 1
    clipped_count = int(np.count_nonzero(clipped_cells))
    union_pct = clipped_count / total * 100.0 if total else 0.0

    k_all: dict[int, float] = {}
    k_clipped: dict[int, float] = {}
    for k in range(1, 5):
        count = int(np.count_nonzero(per_cell == k))
        k_all[k] = count / total * 100.0 if total else 0.0
        k_clipped[k] = count / clipped_count * 100.0 if clipped_count else 0.0

    ge2 = int(np.count_nonzero(per_cell >= 2))
    ge2_of_clipped = ge2 / clipped_count * 100.0 if clipped_count else 0.0
    return union_pct, ge2_of_clipped, k_clipped, k_all


def nan_cell_metrics() -> tuple[float, float, dict[int, float], dict[int, float]]:
    nan = float("nan")
    return nan, nan, {k: nan for k in range(1, 5)}, {k: nan for k in range(1, 5)}


def compute_generic_cell_union(
    raw_image: Any, raw_colors: Any, thresholds: dict[int, int],
    raw_pattern: list[list[int]],
) -> float:
    """Clipped-sensel-area fraction for an arbitrary CFA period (R3 item 1).

    A "cell" is one CFA pattern period (6x6 for X-Trans). The union metric —
    the share of cells containing at least one clipped photosite — is
    meaningful for any period and is what the RAW9 rank-domain highlight trim
    consumes. The 2x2 multi-channel decomposition (ge2 / k-of-cell) stays
    Bayer-only: on other periods those fields remain NaN and their consumers
    must treat "no topology" as no evidence, never as zero clipping.
    """
    try:
        pattern = np.asarray(raw_pattern)
        ph, pw = int(pattern.shape[0]), int(pattern.shape[1])
    except Exception:
        return float("nan")
    if ph <= 0 or pw <= 0:
        return float("nan")
    h, w = raw_image.shape
    h2 = (h // ph) * ph
    w2 = (w // pw) * pw
    if h2 <= 0 or w2 <= 0:
        return float("nan")
    threshold_map = channel_threshold_map(raw_colors[:h2, :w2], thresholds)
    clipped = raw_image[:h2, :w2] >= threshold_map
    per_cell = clipped.reshape(h2 // ph, ph, w2 // pw, pw).sum(axis=(1, 3))
    total = int(per_cell.size)
    if total == 0:
        return float("nan")
    return float(np.count_nonzero(per_cell >= 1) / total * 100.0)


def is_2x2_cfa(raw_pattern: list[list[int]]) -> bool:
    try:
        pattern = np.asarray(raw_pattern)
    except Exception:
        return False
    return pattern.shape == (2, 2)


def channel_saturation_levels(
    channel_ids: list[int], camera_white_levels: list[float], white_level: int
) -> dict[int, int]:
    """Per-channel saturation from metadata: camera_white_level_per_channel when the DNG
    provides it, else the global white_level. Zeros (libraw left them unset) fall back too."""
    out: dict[int, int] = {}
    for cid in channel_ids:
        level = int(camera_white_levels[cid]) if cid < len(camera_white_levels) else 0
        out[cid] = level if level > 0 else int(white_level)
    return out


def resolve_fullwell(
    channel_ids: list[int], ceilings: dict[int, int], spike_ok: dict[int, bool], sat: dict[int, int]
) -> tuple[int, list[int], str, dict[int, int]]:
    """Prefer the measured saturation pile when the scene actually clips; otherwise fall back
    to the metadata white level so an unclipped frame's brightest pixel is not mistaken for the
    full well (which would deflate clip %, usable DR, and the tone plan)."""
    # A8 item 1: a pile only overrides metadata when it is PLAUSIBLY the
    # sensor's full well — near the metadata white level. An ordinary
    # highlight plateau (measured: 300 px at 8000 under WhiteLevel 16383)
    # must not become the frame's "full well" and poison clip %, DR, the
    # clip mask and HDR headroom downstream.
    strong = [
        cid for cid in channel_ids
        if spike_ok.get(cid, False)
        and ceilings[cid] >= CEILING_PLAUSIBLE_FRACTION * max(sat[cid], 1)
    ]
    channel_fullwell = {
        cid: int(ceilings[cid]) if cid in strong else int(sat[cid])
        for cid in channel_ids
    }
    if strong:
        weak = [cid for cid in channel_ids if cid not in strong]
        fullwell = int(min(ceilings[cid] for cid in strong))
        note = "weak ceiling channels excluded from fullwell" if weak else "all channels have ceiling pile"
        return fullwell, strong, note, channel_fullwell
    fullwell = int(min(sat[cid] for cid in channel_ids))
    return fullwell, channel_ids, "no ceiling pile; fullwell from metadata white_level", channel_fullwell


def channel_clip_thresholds(channel_ids: list[int], fullwell_by_channel: dict[int, int], margin: int) -> dict[int, int]:
    return {cid: int(max(fullwell_by_channel[cid] - margin, 0)) for cid in channel_ids}


def luminance_from_xyz_render(xyz_render: Any, render_scale: float) -> Any:
    y = xyz_render[..., 1].astype(np.float32, copy=False)
    y = y / np.float32(render_scale)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(y, 0.0, None)


def compute_ev_metrics(y: Any) -> tuple[Any, float, float, float, float, float, float, float, float, float]:
    ev = np.log2(np.clip(y, EPS, None)).astype(np.float32, copy=False)
    ev_report = np.maximum(ev, EV_REPORT_FLOOR)
    raw_p1 = float(np.percentile(ev, 1))
    p1, p50, p99, p999 = [float(x) for x in np.percentile(ev_report, [1, 50, 99, 99.9])]
    floor_hit_pct = float(np.mean(ev <= EV_REPORT_FLOOR) * 100.0)
    return ev, raw_p1, p1, p50, p99, p999, p999 - p1, floor_hit_pct, p50 - GRAY_EV


def estimate_noise_floor(y: Any, rows: int = 32, cols: int = 48) -> float:
    h, w = y.shape
    rows = min(rows, h)
    cols = min(cols, w)
    if rows <= 0 or cols <= 0:
        return 0.0

    y_edges = np.linspace(0, h, rows + 1, dtype=int)
    x_edges = np.linspace(0, w, cols + 1, dtype=int)
    means: list[float] = []
    stds: list[float] = []
    for r in range(rows):
        y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
        if y1 <= y0:
            continue
        for c in range(cols):
            x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
            if x1 <= x0:
                continue
            tile = y[y0:y1, x0:x1]
            if tile.size:
                means.append(float(np.mean(tile, dtype=np.float64)))
                stds.append(float(np.std(tile, dtype=np.float64)))

    if not means:
        return 0.0
    means_arr = np.asarray(means)
    stds_arr = np.asarray(stds)
    count = max(1, int(math.ceil(len(means) * 0.10)))
    darkest = np.argsort(means_arr)[:count]
    return float(np.median(stds_arr[darkest]))


def black_map(raw_colors: Any, black_levels: list[float]) -> Any:
    max_color = int(np.max(raw_colors)) if raw_colors.size else -1
    levels = np.zeros(max_color + 1, dtype=np.float32)
    for cid in range(max_color + 1):
        levels[cid] = float(black_levels[cid]) if cid < len(black_levels) else 0.0
    return levels[raw_colors]


def normalized_raw_signal(
    raw_image: Any, raw_colors: Any, black_levels: list[float], fullwell_by_channel: dict[int, int]
) -> Any:
    bmap = black_map(raw_colors, black_levels)
    fmap = channel_fullwell_map(raw_colors, fullwell_by_channel)
    denom = np.maximum(fmap - bmap, np.float32(1.0))
    signal = raw_image.astype(np.float32, copy=False) - bmap
    signal = np.clip(signal, 0.0, None) / denom
    return np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=0.0)


def estimate_raw_noise_floor(bundle: RawBundle, fullwell_by_channel: dict[int, int]) -> float:
    """Tile-local port of estimate_noise_floor over the NORMALIZED raw
    signal. The old path materialized full-frame black/fullwell maps and a
    normalized copy (three frame-sized float arrays) only to reduce them to
    32x48 tile means/stds (review batch 17); normalization is pointwise, so
    doing it per tile yields the identical statistic with tile-sized
    temporaries only."""
    raw = bundle.raw_image
    colors = bundle.raw_colors
    h = min(raw.shape[0], colors.shape[0])
    w = min(raw.shape[1], colors.shape[1])
    rows, cols = min(32, h), min(48, w)
    if rows <= 0 or cols <= 0:
        return 0.0
    fallback = max(fullwell_by_channel.values()) if fullwell_by_channel else 1
    max_color = int(np.max(colors)) if colors.size else -1
    levels = np.zeros(max_color + 1, dtype=np.float32)
    denoms = np.ones(max_color + 1, dtype=np.float32)
    for cid in range(max_color + 1):
        black = float(bundle.black_levels[cid]) if cid < len(bundle.black_levels) else 0.0
        levels[cid] = black
        denoms[cid] = max(float(fullwell_by_channel.get(cid, fallback)) - black, 1.0)
    y_edges = np.linspace(0, h, rows + 1, dtype=int)
    x_edges = np.linspace(0, w, cols + 1, dtype=int)
    means: list[float] = []
    stds: list[float] = []
    for r in range(rows):
        y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
        if y1 <= y0:
            continue
        for c in range(cols):
            x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
            if x1 <= x0:
                continue
            cid_tile = colors[y0:y1, x0:x1]
            tile = (
                raw[y0:y1, x0:x1].astype(np.float32) - levels[cid_tile]
            )
            np.clip(tile, 0.0, None, out=tile)
            tile /= denoms[cid_tile]
            np.nan_to_num(tile, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
            if tile.size:
                means.append(float(np.mean(tile, dtype=np.float64)))
                stds.append(float(np.std(tile, dtype=np.float64)))
    if not means:
        return 0.0
    means_arr = np.asarray(means)
    stds_arr = np.asarray(stds)
    count = max(1, int(math.ceil(len(means) * 0.10)))
    darkest = np.argsort(means_arr)[:count]
    return float(np.median(stds_arr[darkest]))


def noise_floor_ev_estimate(analysis: Analysis) -> tuple[float, str]:
    """Scene EV of the sensor noise floor under the fixed mid-gray exposure anchor.

    Tone planning renders with the constant anchor (mid gray = clip / 2**headroom at
    EV 0), which puts the RAW full well at exactly +MIDGRAY_HEADROOM_STOPS scene EV.
    A noise floor expressed as a fraction ``f`` of the full well therefore sits at
    ``MIDGRAY_HEADROOM_STOPS + log2(f)`` scene EV — the same convention every scene
    tone metric already uses.

    Two sources, following the existing prior/degradation philosophy:

    - ``("prior", ...)``: the sensor prior's published read noise in electrons over
      the frame's own electron-domain full well (recovered from the stored
      ``noise_floor_e / noise_floor`` ratio, so no extra fields are needed). This is
      the SNR=1 engineering floor and is unbiased by scene photon noise.
    - ``("frame", ...)``: single-frame tile-σ estimate (``usable_dr_ev``) when no
      prior is available. It includes photon noise of the darkest scene tiles and
      therefore reads conservatively shallow; callers should note the degradation.
    - ``("none", nan)``: no usable estimate at all.
    """
    noise_e = analysis.noise_floor_e
    read_e = analysis.prior_read_noise_e
    nf = float(analysis.noise_floor)
    if (
        noise_e is not None
        and read_e is not None
        and nf > 0.0
        and float(noise_e) > 0.0
        and float(read_e) > 0.0
    ):
        fullwell_e = float(noise_e) / nf
        if math.isfinite(fullwell_e) and fullwell_e > 0.0:
            floor_fraction = float(read_e) / fullwell_e
            if 0.0 < floor_fraction < 1.0:
                return MIDGRAY_HEADROOM_STOPS + math.log2(floor_fraction), "prior"
    if math.isfinite(analysis.usable_dr_ev):
        return MIDGRAY_HEADROOM_STOPS - float(analysis.usable_dr_ev), "frame"
    return float("nan"), "none"


def cfa_positions_for_channel(bundle: RawBundle, cid: int) -> list[tuple[int, int]]:
    try:
        pattern = np.asarray(bundle.raw_pattern)
    except Exception:
        pattern = np.asarray([])
    if pattern.ndim != 2 or pattern.size == 0:
        return []

    ph, pw = pattern.shape
    visible_pattern = np.asarray(bundle.raw_colors[:ph, :pw])
    positions = np.argwhere(visible_pattern == cid)
    if positions.size == 0:
        positions = np.argwhere(pattern == cid)
    return [(int(y), int(x)) for y, x in positions]


def tile_signal_noise_from_plane(plane: Any, black: float, tile_size: int = SNR_TILE) -> tuple[Any, Any]:
    h, w = plane.shape
    h2 = (h // tile_size) * tile_size
    w2 = (w // tile_size) * tile_size
    if h2 <= 0 or w2 <= 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    tiles = plane[:h2, :w2].astype(np.float32, copy=False).reshape(
        h2 // tile_size, tile_size, w2 // tile_size, tile_size
    )
    means = tiles.mean(axis=(1, 3), dtype=np.float64).reshape(-1).astype(np.float32)
    stds = tiles.std(axis=(1, 3), dtype=np.float64).reshape(-1).astype(np.float32)
    signal = np.maximum(means - np.float32(black), 0.0)
    return signal, np.maximum(stds, 0.0)


def group_tile_signal_noise(
    bundle: RawBundle, fullwell_by_channel: dict[int, int], ids: list[int]
) -> tuple[Any, Any, Any]:
    signals: list[Any] = []
    noises: list[Any] = []
    denoms: list[Any] = []
    fallback_fullwell = max(fullwell_by_channel.values()) if fullwell_by_channel else 1
    for cid in ids:
        black = float(bundle.black_levels[cid]) if cid < len(bundle.black_levels) else 0.0
        fullwell_f = float(fullwell_by_channel.get(cid, fallback_fullwell))
        denom = max(fullwell_f - black, 1.0)
        for yoff, xoff in cfa_positions_for_channel(bundle, cid):
            pattern = np.asarray(bundle.raw_pattern)
            ph, pw = pattern.shape
            plane = bundle.raw_image[yoff::ph, xoff::pw]
            sig, noise = tile_signal_noise_from_plane(plane, black)
            if sig.size:
                signals.append(sig)
                noises.append(noise)
                denoms.append(np.full(sig.shape, denom, dtype=np.float32))
    if not signals:
        return (
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
        )
    return np.concatenate(signals), np.concatenate(noises), np.concatenate(denoms)


def interpolate_zero_db_stop(stops: Any, snr_db: Any) -> float:
    valid = np.isfinite(stops) & np.isfinite(snr_db) & (stops <= SNR_BRIGHT_UNRELIABLE_STOP)
    x = stops[valid]
    y = snr_db[valid]
    if x.size < 2:
        return float("nan")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if np.all(y > 0):
        return float(x[0])
    if np.all(y < 0):
        return float("nan")
    for i in range(len(x) - 1):
        y0 = float(y[i])
        y1 = float(y[i + 1])
        if y0 == 0.0:
            return float(x[i])
        if (y0 < 0.0 <= y1) or (y0 > 0.0 >= y1):
            x0 = float(x[i])
            x1 = float(x[i + 1])
            if y1 == y0:
                return x0
            return x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
    return float("nan")


def compute_snr_curves(
    bundle: RawBundle, channel_ids: list[int], labels: dict[int, str], fullwell_by_channel: dict[int, int]
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, float]]:
    curves: dict[str, dict[str, Any]] = {}
    snr1_dr: dict[str, float] = {}
    snr1_stop: dict[str, float] = {}
    groups = rgb_channel_groups(channel_ids, labels)
    bins = np.linspace(EV_REPORT_FLOOR, 0.0, 85)
    centers = (bins[:-1] + bins[1:]) * 0.5

    for group_name, ids in groups:
        sig, noise, denom = group_tile_signal_noise(bundle, fullwell_by_channel, ids)
        if sig.size == 0:
            snr_db = np.full(centers.shape, np.nan, dtype=np.float32)
            counts = np.zeros(centers.shape, dtype=np.int32)
        else:
            valid = (sig > 0) & (noise > 0) & (denom > 0)
            stops = np.log2(np.clip(sig[valid] / denom[valid], 2.0 ** EV_REPORT_FLOOR, None))
            sig_valid = sig[valid]
            noise_valid = noise[valid]
            snr_db = np.full(centers.shape, np.nan, dtype=np.float32)
            counts = np.zeros(centers.shape, dtype=np.int32)
            # single sort + boundary slices instead of 84 whole-array boolean
            # scans (review batch 17): identical per-bin members, one pass
            order = np.argsort(stops, kind="stable")
            stops_sorted = stops[order]
            sig_sorted = sig_valid[order]
            noise_sorted = noise_valid[order]
            bounds = np.searchsorted(stops_sorted, bins)
            for i in range(len(centers)):
                lo, hi = int(bounds[i]), int(bounds[i + 1])
                n = hi - lo
                counts[i] = n
                if n < 8:
                    continue
                noise_est = float(np.percentile(noise_sorted[lo:hi], SNR_LOW_PERCENTILE))
                signal_med = float(np.median(sig_sorted[lo:hi]))
                if noise_est <= 0.0 or signal_med <= 0.0:
                    continue
                snr_db[i] = np.float32(20.0 * math.log10(signal_med / max(noise_est, 1e-9)))
        zero_stop = interpolate_zero_db_stop(centers, snr_db)
        snr1_stop[group_name] = zero_stop
        snr1_dr[group_name] = -zero_stop if math.isfinite(zero_stop) else float("nan")
        curves[group_name] = {
            "stops": centers.copy(),
            "snr_db": snr_db,
            "count": counts,
            "ids": ids,
        }
    return curves, snr1_dr, snr1_stop


def compute_gamut_metrics(
    scene_rec2020: Any,
    scene_scale: float,
    y: Any,
    gamut_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[dict[str, float], float]:
    """Out-of-gamut pressure from the SCENE Rec.2020 buffer, converted to
    float XYZ per chunk. A8 item 3: the packed uint16 XYZ render clips Z at
    the representable ceiling, and Rec.2020's pure blue carries Z ~= 1.061
    — reading it back under-reported exactly the saturated blues/cyans this
    metric exists to count (round-tripped blue fell to ~0.943 and grew
    ~0.015 of red). The scene buffer is the unclipped source of truth."""
    names = tuple(gamut_names) if gamut_names is not None else tuple(XYZ_TO_RGB.keys())
    names = tuple(name for name in names if name in XYZ_TO_RGB)
    if not names:
        names = tuple(XYZ_TO_RGB.keys())
    flat_y = y.reshape(-1)
    median_y = float(np.median(flat_y))
    bright_flat = flat_y > median_y
    if not np.any(bright_flat):
        bright_flat = (flat_y >= median_y) & (flat_y > EPS)

    bright_total = int(np.count_nonzero(bright_flat))
    total_pixels = int(flat_y.size)
    if bright_total == 0:
        return {name: 0.0 for name in names}, 0.0

    from .color import rec2020_to_xyz

    counts = {name: 0 for name in names}
    flat_scene = scene_rec2020.reshape(-1, scene_rec2020.shape[-1])
    inv_scale = np.float32(1.0 / scene_scale)
    chunk = 1_000_000

    for start in range(0, total_pixels, chunk):
        end = min(start + chunk, total_pixels)
        mask = bright_flat[start:end]
        if not np.any(mask):
            continue
        linear = flat_scene[start:end, :3][mask].astype(np.float32) * inv_scale
        xyz = rec2020_to_xyz(linear).astype(np.float32, copy=False)
        xyz = np.nan_to_num(xyz, nan=0.0, posinf=1.0, neginf=0.0)
        x = xyz[:, 0]
        y_chan = xyz[:, 1]
        z = xyz[:, 2]
        for name in names:
            matrix = XYZ_TO_RGB[name]
            rgb = np.empty((xyz.shape[0], 3), dtype=np.float32)
            rgb[:, 0] = matrix[0, 0] * x + matrix[0, 1] * y_chan + matrix[0, 2] * z
            rgb[:, 1] = matrix[1, 0] * x + matrix[1, 1] * y_chan + matrix[1, 2] * z
            rgb[:, 2] = matrix[2, 0] * x + matrix[2, 1] * y_chan + matrix[2, 2] * z
            rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
            denom = np.max(rgb, axis=1)
            valid = denom > EPS
            if not np.any(valid):
                continue
            norm = rgb[valid] / denom[valid, None]
            counts[name] += int(np.count_nonzero(np.any(norm < -GAMUT_EPS, axis=1)))

    pct = {name: counts[name] / bright_total * 100.0 for name in names}
    bright_pct = bright_total / total_pixels * 100.0 if total_pixels else 0.0
    return pct, bright_pct


def estimate_container_bits(white_level: int, raw_image: Any) -> int:
    if white_level > 0:
        return int(math.ceil(math.log2(white_level + 1)))
    if np.issubdtype(raw_image.dtype, np.integer):
        return int(np.iinfo(raw_image.dtype).bits)
    return 0


def raw_health_metrics(bundle: RawBundle, channel_ids: list[int], labels: dict[int, str]) -> tuple[float, float]:
    """Demosaic-independent checks for in-camera processing baked into the raw.

    lag-1 correlation: residuals of the darkest CFA green tiles should be ~white noise;
    clearly positive correlation means spatial filtering was applied before writing the
    file. histogram emptiness: missing DN codes in the dense value range indicate the
    data was rescaled/requantized in camera. Heuristics — reported, never acted on."""
    green_ids = [cid for cid in channel_ids if labels[cid].startswith("G")]
    if not green_ids:
        return float("nan"), float("nan")
    pattern = np.asarray(bundle.raw_pattern)
    if pattern.ndim != 2:
        return float("nan"), float("nan")
    ph, pw = pattern.shape
    positions = [pos for cid in green_ids for pos in cfa_positions_for_channel(bundle, cid)]
    if not positions:
        return float("nan"), float("nan")
    yoff, xoff = positions[0]
    plane = bundle.raw_image[yoff::ph, xoff::pw].astype(np.float32, copy=False)

    # Scene cancellation: the two green CFA planes sample (nearly) the same image, so
    # their difference is almost pure sensor noise. White noise -> lag-1 ~ 0; in-camera
    # spatial filtering leaves the residual noise correlated. Measuring on the raw plane
    # itself would pick up scene texture and always read "smoothed".
    lag1 = float("nan")
    if len(positions) >= 2:
        y2, x2 = positions[1]
        plane2 = bundle.raw_image[y2::ph, x2::pw].astype(np.float32, copy=False)
        h = min(plane.shape[0], plane2.shape[0])
        w = min(plane.shape[1], plane2.shape[1])
        diff = plane[:h, :w] - plane2[:h, :w]
        tile = SNR_TILE
        h2 = (h // tile) * tile
        w2 = (w // tile) * tile
        if h2 >= tile and w2 >= tile:
            tiles = diff[:h2, :w2].reshape(h2 // tile, tile, w2 // tile, tile).transpose(0, 2, 1, 3)
            tiles = tiles.reshape(-1, tile, tile)
            # Prefer flat tiles (low |diff| variance) to sidestep edge/aliasing residue.
            variances = tiles.var(axis=(1, 2))
            count = max(16, int(math.ceil(variances.size * 0.25)))
            flattest = np.argsort(variances)[: min(count, 768)]
            sel = tiles[flattest] - tiles[flattest].mean(axis=(1, 2), keepdims=True)
            num_h = np.sum(sel[:, :, :-1] * sel[:, :, 1:], dtype=np.float64)
            den_h = math.sqrt(
                float(np.sum(sel[:, :, :-1] ** 2, dtype=np.float64))
                * float(np.sum(sel[:, :, 1:] ** 2, dtype=np.float64))
            )
            num_v = np.sum(sel[:, :-1, :] * sel[:, 1:, :], dtype=np.float64)
            den_v = math.sqrt(
                float(np.sum(sel[:, :-1, :] ** 2, dtype=np.float64))
                * float(np.sum(sel[:, 1:, :] ** 2, dtype=np.float64))
            )
            if den_h > 0 and den_v > 0:
                lag1 = float(0.5 * (num_h / den_h + num_v / den_v))

    vals = plane.reshape(-1).astype(np.int64)
    p05, p60 = np.percentile(vals, [5.0, 60.0])
    lo, hi = int(p05), int(max(p60, p05 + 32))
    hist_empty = float("nan")
    if hi - lo >= 32:
        counts = np.bincount(np.clip(vals, lo, hi) - lo, minlength=hi - lo + 1)
        interior = counts[1:-1]
        if interior.size:
            hist_empty = float(np.mean(interior == 0) * 100.0)
    return lag1, hist_empty


def raw_health_verdict_cn(lag1: float, hist_empty: float) -> str:
    if not math.isfinite(lag1):
        return "n/a"
    if lag1 < 0.08:
        verdict = "干净(近白噪声)"
    elif lag1 < 0.20:
        verdict = "轻度空间处理迹象"
    else:
        verdict = "明显平滑(疑似机内降噪)"
    if math.isfinite(hist_empty) and hist_empty > 5.0:
        verdict += "; 直方图有梳齿(疑似机内缩放)"
    return verdict


def analyze(
    bundle: RawBundle,
    margin: int,
    *,
    diagnostics: bool = True,
    gamut_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[Analysis, Any, Any]:
    raw_image = bundle.raw_image
    raw_colors = bundle.raw_colors
    channel_ids = [int(x) for x in sorted(np.unique(raw_colors).tolist())]
    labels = channel_labels(bundle.color_desc, channel_ids)

    sat = channel_saturation_levels(channel_ids, bundle.camera_white_levels, bundle.white_level)
    ceilings, exact_counts, near_counts, spike_ok = detect_ceilings(
        raw_image, raw_colors, channel_ids, sat
    )
    fullwell, fullwell_ids, fullwell_note, channel_fullwell = resolve_fullwell(
        channel_ids, ceilings, spike_ok, sat
    )
    # load_raw can only seed the soft headroom mask from metadata. If this frame contains
    # a trustworthy saturation pile, bring the render-time mask onto the same resolved
    # per-channel full-well endpoints used by hard clip statistics.
    from .raw_io import refresh_clip_masks_from_fullwell

    refresh_clip_masks_from_fullwell(bundle, channel_fullwell)
    threshold = int(max(fullwell - margin, 0))
    channel_thresholds = channel_clip_thresholds(channel_ids, channel_fullwell, margin)
    clip_pct = compute_clip_pct_by_thresholds(raw_image, raw_colors, channel_ids, channel_thresholds)
    cell_supported = is_2x2_cfa(bundle.raw_pattern)
    if cell_supported:
        cell_union, cell_ge2, cell_k_clipped, cell_k_all = compute_cell_metrics(
            raw_image, raw_colors, channel_thresholds
        )
    else:
        # R3 item 1: "not applicable" used to be spelled NaN across the board,
        # and NaN flowed into the RAW9 highlight rank trim (clamping to a 0%
        # trim) and the HDR channel-separation compile (NaN rho). The union
        # metric generalises to any CFA period, so measure it; only the 2x2
        # multi-channel decomposition stays NaN, and its consumers now treat
        # missing topology as zero confidence rather than zero clipping.
        _, cell_ge2, cell_k_clipped, cell_k_all = nan_cell_metrics()
        cell_union = compute_generic_cell_union(
            raw_image, raw_colors, channel_thresholds, bundle.raw_pattern
        )

    y = luminance_from_xyz_render(bundle.xyz_render, bundle.render_scale)
    ev, raw_p1, p1, p50, p99, p999, dr, floor_hit_pct, vs_gray = compute_ev_metrics(y)
    nf = estimate_raw_noise_floor(bundle, channel_fullwell)
    usable_dr = math.log2(1.0 / max(nf, NOISE_DR_EPS))
    # Review R2 item 1: the SNR curve is a RENDER input now, not a diagnostic.
    # compile_channel_separation's design lists a tail-SNR confidence factor,
    # and gating this behind `diagnostics` made the same photograph render
    # differently depending on whether --csv/--scan happened to be passed.
    # Always-on costs ~0.2 s at 61 MP (vectorized tile means/stds), which the
    # one analyze() per loaded file absorbs.
    snr_curves, snr1_dr, snr1_stop = compute_snr_curves(
        bundle, channel_ids, labels, channel_fullwell
    )
    gamut_pct, bright_pct = compute_gamut_metrics(
        bundle.scene_rec2020_render, bundle.scene_scale, y, gamut_names
    )

    # Priors layer: electron-domain calibration from public measurements (best-effort).
    prior = sensor_priors.find_priors(bundle.shot_make, bundle.shot_model)
    prior_id = prior["id"] if prior else None
    gain_e = sensor_priors.gain_e_per_dn(prior, bundle.shot_iso) if prior and bundle.shot_iso else None
    prior_rn_e = sensor_priors.read_noise_e(prior, bundle.shot_iso) if prior and bundle.shot_iso else None
    prior_pdr = sensor_priors.pdr_ev(prior, bundle.shot_iso) if prior and bundle.shot_iso else None
    noise_e = None
    if gain_e is not None:
        mean_black = float(np.mean([bundle.black_levels[c] for c in channel_ids if c < len(bundle.black_levels)] or [0.0]))
        noise_e = float(nf * max(fullwell - mean_black, 1.0) * gain_e)
    # Effective DR for downstream tone planning: the empirical single-frame estimate,
    # gently bounded by the published PDR when available (never replaced by it).
    if prior_pdr is not None and math.isfinite(usable_dr):
        usable_dr_eff = clamp_float(usable_dr, prior_pdr - 1.5, prior_pdr + 1.5)
    else:
        usable_dr_eff = usable_dr
    if diagnostics:
        health_lag1, health_hist = raw_health_metrics(bundle, channel_ids, labels)
    else:
        health_lag1 = health_hist = float("nan")

    survivor_id = min(channel_ids, key=lambda cid: clip_pct.get(cid, float("inf")))
    analysis = Analysis(
        channel_ids=channel_ids,
        labels=labels,
        ceilings=ceilings,
        ceil_spike_counts=exact_counts,
        ceil_near_counts=near_counts,
        ceil_spike_ok=spike_ok,
        fullwell_channel_ids=fullwell_ids,
        fullwell_note=fullwell_note,
        saturation_levels=sat,
        channel_fullwell=channel_fullwell,
        channel_thresholds=channel_thresholds,
        fullwell=fullwell,
        threshold=threshold,
        clip_pct=clip_pct,
        cfa_cell_supported=cell_supported,
        cell_union_pct=cell_union,
        cell_ge2_of_clipped_pct=cell_ge2,
        cell_k_of_clipped_pct=cell_k_clipped,
        cell_k_of_all_pct=cell_k_all,
        ev_p1=p1,
        ev_raw_p1=raw_p1,
        ev_median=p50,
        ev_p99=p99,
        ev_p999=p999,
        ev_dr_p1_p999=dr,
        ev_floor_hit_pct=floor_hit_pct,
        median_vs_gray_ev=vs_gray,
        median_y=float(np.median(y)),
        noise_floor=nf,
        usable_dr_ev=usable_dr,
        snr_curves=snr_curves,
        snr1_dr=snr1_dr,
        snr1_stop=snr1_stop,
        gamut_out_pct=gamut_pct,
        bright_pixel_pct=bright_pct,
        survivor_channel=labels[survivor_id],
        container_bits_est=estimate_container_bits(bundle.white_level, raw_image),
        prior_id=prior_id,
        gain_e_per_dn=gain_e,
        noise_floor_e=noise_e,
        prior_read_noise_e=prior_rn_e,
        prior_pdr_ev=prior_pdr,
        usable_dr_eff_ev=usable_dr_eff,
        health_lag1_corr=health_lag1,
        health_hist_empty_pct=health_hist,
    )
    return analysis, y, ev


def reanalyze_balanced_scene(
    capture: Analysis,
    bundle: RawBundle,
    *,
    gamut_names: tuple[str, ...] | list[str] | None = None,
) -> Analysis:
    """Refresh only WB-dependent scene facts on an invariant capture analysis.

    Ceiling/noise/clip/CFA/SNR facts come from the RAW and are intentionally reused.
    EV and gamut facts come from the newly balanced scene.  Preview BalanceContexts run
    this over their fixed 1920px proxy; export still runs ``analyze`` on the full scene.
    Both paths share the identical WB matrix and pixel pipeline, while avoiding every
    sensor-domain scan on an interactive balance change.
    """
    y = luminance_from_xyz_render(bundle.xyz_render, bundle.render_scale)
    # EV percentiles and gamut occupancy are independent read-only reductions over the
    # same luminance plane.  NumPy releases the GIL for their heavy kernels, so running
    # the two original functions concurrently removes their serialized wall time while
    # preserving every operation, dtype, percentile definition, and result bit.
    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="dngscan-balance-analysis"
    ) as pool:
        ev_future = pool.submit(compute_ev_metrics, y)
        gamut_future = pool.submit(
            compute_gamut_metrics,
            bundle.scene_rec2020_render,
            bundle.scene_scale,
            y,
            gamut_names,
        )
        (
            _ev,
            raw_p1,
            p1,
            p50,
            p99,
            p999,
            dr,
            floor_hit_pct,
            vs_gray,
        ) = ev_future.result()
        gamut_pct, bright_pct = gamut_future.result()
    return replace(
        capture,
        ev_p1=p1,
        ev_raw_p1=raw_p1,
        ev_median=p50,
        ev_p99=p99,
        ev_p999=p999,
        ev_dr_p1_p999=dr,
        ev_floor_hit_pct=floor_hit_pct,
        median_vs_gray_ev=vs_gray,
        median_y=float(np.median(y)),
        gamut_out_pct=gamut_pct,
        bright_pixel_pct=bright_pct,
    )


def downsample_mean(arr: Any, max_dim: int = 900) -> Any:
    h, w = arr.shape
    step = max(1, int(math.ceil(max(h, w) / max_dim)))
    if step <= 1:
        return arr
    h2 = (h // step) * step
    w2 = (w // step) * step
    if h2 <= 0 or w2 <= 0:
        return arr[::step, ::step]
    return arr[:h2, :w2].reshape(h2 // step, step, w2 // step, step).mean(axis=(1, 3))


def downsample_any(mask: Any, max_dim: int = 900) -> Any:
    h, w = mask.shape
    step = max(1, int(math.ceil(max(h, w) / max_dim)))
    if step <= 1:
        return mask
    h2 = (h // step) * step
    w2 = (w // step) * step
    if h2 <= 0 or w2 <= 0:
        return mask[::step, ::step]
    return mask[:h2, :w2].reshape(h2 // step, step, w2 // step, step).any(axis=(1, 3))


def rgb_channel_groups(channel_ids: list[int], labels: dict[int, str]) -> list[tuple[str, list[int]]]:
    groups: list[tuple[str, list[int]]] = []
    for base in ("R", "G", "B"):
        ids = [cid for cid in channel_ids if labels[cid].startswith(base)]
        if ids:
            groups.append((base, ids))
    return groups
