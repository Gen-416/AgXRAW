# SPDX-License-Identifier: GPL-3.0-or-later
"""Matplotlib diagnostic dashboard."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ._deps import np

_MPL: tuple[Any, Any, Any, Any] | None = None


def _matplotlib() -> tuple[Any, Any, Any, Any]:
    """Deferred matplotlib import: (plt, font_manager, ListedColormap, Patch).

    Only the dashboard needs matplotlib; loading pyplot at package import made
    every spawned export worker pay ~0.27s for a module it never used.
    """
    global _MPL
    if _MPL is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        _MPL = (plt, font_manager, ListedColormap, Patch)
    return _MPL
from .analysis import (
    black_map, channel_color, channel_fullwell_map, channel_threshold_map,
    downsample_any, downsample_mean, format_pct, format_snr_dr, rgb_channel_groups,
)
from .constants import EPS, EV_REPORT_FLOOR, GRAY_EV, SNR_BRIGHT_UNRELIABLE_STOP, XYZ_TO_RGB
from .models import Analysis, AutoEvResult, RawBundle
from .report import summary_lines

def configure_plot_fonts() -> None:
    try:
        plt, font_manager, _, _ = _matplotlib()
    except Exception:
        return
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Songti SC",
        "STHeiti",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)
    if chosen:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def exposure_zone_map(y: Any, nf: float) -> Any:
    y_ds = downsample_mean(y)
    ev_ds = np.log2(np.clip(y_ds, EPS, None))
    noise_ev = math.log2(max(nf, EPS))
    near_noise_cut = max(noise_ev + 1.0, -12.0)

    zones = np.zeros(y_ds.shape, dtype=np.uint8)
    zones[ev_ds > near_noise_cut] = 1
    zones[ev_ds > -5.0] = 2
    zones[ev_ds > -2.0] = 3
    zones[(ev_ds >= -0.05) | (y_ds >= 0.98)] = 4
    return zones


def clipped_rgb_map(bundle: RawBundle, analysis: Analysis) -> Any:
    raw_clip = bundle.raw_image >= channel_threshold_map(bundle.raw_colors, analysis.channel_thresholds)
    groups = {"R": [], "G": [], "B": []}
    for cid in analysis.channel_ids:
        base = analysis.labels[cid][:1].upper()
        if base in groups:
            groups[base].append(cid)

    channels = []
    for base in ("R", "G", "B"):
        ids = groups[base]
        if ids:
            color_match = np.isin(bundle.raw_colors, ids)
            mask = raw_clip & color_match
            channels.append(downsample_any(mask))
        else:
            channels.append(np.zeros_like(downsample_any(raw_clip), dtype=bool))
    return np.dstack(channels).astype(np.float32)


def gaussian_kernel1d(sigma: float = 1.2, radius: int | None = None) -> Any:
    if radius is None:
        radius = max(1, int(math.ceil(sigma * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / np.sum(kernel)


def gaussian_smooth(counts: Any, sigma: float = 1.2) -> Any:
    if counts.size < 3:
        return counts.astype(np.float64, copy=False)
    kernel = gaussian_kernel1d(sigma)
    return np.convolve(counts.astype(np.float64, copy=False), kernel, mode="same")


def quantile_from_histogram(counts: Any, edges: Any, q: float) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return float("nan")
    target = total * float(q)
    cumulative = np.cumsum(counts)
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = max(0, min(index, len(edges) - 2))
    return float((edges[index] + edges[index + 1]) * 0.5)


def rgb_ev_histograms_from_xyz(
    xyz_render: Any, render_scale: float, ev_floor: float, ev_high: float, bins: int = 160
) -> tuple[Any, dict[str, Any], dict[str, float], dict[str, float]]:
    edges = np.linspace(ev_floor, ev_high, bins + 1, dtype=np.float32)
    counts = {name: np.zeros(bins, dtype=np.int64) for name in ("R", "G", "B")}
    floor_counts = {name: 0 for name in ("R", "G", "B")}
    matrix = XYZ_TO_RGB["sRGB"]
    flat_xyz = xyz_render.reshape(-1, xyz_render.shape[-1])
    inv_scale = np.float32(1.0 / render_scale)
    total = int(flat_xyz.shape[0])
    chunk = 1_000_000

    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        xyz = flat_xyz[start:end, :3].astype(np.float32, copy=False) * inv_scale
        xyz = np.nan_to_num(xyz, nan=0.0, posinf=1.0, neginf=0.0)
        x = xyz[:, 0]
        y_chan = xyz[:, 1]
        z = xyz[:, 2]
        rgb_values = {
            "R": matrix[0, 0] * x + matrix[0, 1] * y_chan + matrix[0, 2] * z,
            "G": matrix[1, 0] * x + matrix[1, 1] * y_chan + matrix[1, 2] * z,
            "B": matrix[2, 0] * x + matrix[2, 1] * y_chan + matrix[2, 2] * z,
        }
        for name, values in rgb_values.items():
            floor_counts[name] += int(np.count_nonzero(values <= EPS))
            ev_values = np.log2(np.clip(values, EPS, None)).astype(np.float32, copy=False)
            ev_values = np.clip(ev_values, ev_floor, ev_high)
            counts[name] += np.histogram(ev_values, bins=edges)[0]

    medians = {name: quantile_from_histogram(hist, edges, 0.5) for name, hist in counts.items()}
    floor_pct = {name: floor_counts[name] / max(total, 1) * 100.0 for name in counts}
    centers = (edges[:-1] + edges[1:]) * 0.5
    return centers, counts, medians, floor_pct


def stops_for_channel_ids(
    bundle: RawBundle, ids: list[int], fullwell_by_channel: dict[int, int], floor: float
) -> Any:
    masks = [bundle.raw_colors == cid for cid in ids]
    mask = masks[0] if len(masks) == 1 else np.logical_or.reduce(masks)
    raw_vals = bundle.raw_image[mask].astype(np.float32, copy=False)
    color_vals = bundle.raw_colors[mask]
    bmap = black_map(color_vals, bundle.black_levels)
    fmap = channel_fullwell_map(color_vals, fullwell_by_channel)
    denom = np.maximum(fmap - bmap, np.float32(1.0))
    signal = np.maximum(raw_vals - bmap, 0.0)
    min_ratio = np.float32(2.0 ** floor)
    ratio = np.maximum(signal / denom, min_ratio)
    stops = np.log2(ratio).astype(np.float32, copy=False)
    return np.nan_to_num(stops, nan=floor, posinf=0.0, neginf=floor)


def clip_pct_for_channel_ids(bundle: RawBundle, ids: list[int], thresholds: dict[int, int]) -> float:
    masks = [bundle.raw_colors == cid for cid in ids]
    mask = masks[0] if len(masks) == 1 else np.logical_or.reduce(masks)
    vals = bundle.raw_image[mask]
    threshold_vals = channel_threshold_map(bundle.raw_colors[mask], thresholds)
    return float(np.mean(vals >= threshold_vals) * 100.0) if vals.size else 0.0


def normalized_stop_density(stops: Any, x_min: float, x_max: float, bins: int = 180) -> tuple[Any, Any]:
    clipped_stops = np.clip(stops, x_min, x_max)
    counts, edges = np.histogram(clipped_stops, bins=bins, range=(x_min, x_max))
    density = gaussian_smooth(counts, sigma=1.25)
    peak = float(np.max(density)) if density.size else 0.0
    if peak > 0:
        density = density / peak * 100.0
    centers = (edges[:-1] + edges[1:]) * 0.5
    return centers, density


def threshold_stop_for_channel_ids(
    bundle: RawBundle,
    ids: list[int],
    thresholds: dict[int, int],
    fullwell_by_channel: dict[int, int],
    floor: float,
) -> float:
    stops = []
    fallback_fullwell = max(fullwell_by_channel.values()) if fullwell_by_channel else 1
    for cid in ids:
        black = float(bundle.black_levels[cid]) if cid < len(bundle.black_levels) else 0.0
        denom = max(float(fullwell_by_channel.get(cid, fallback_fullwell)) - black, 1.0)
        threshold = float(thresholds.get(cid, 0))
        ratio = max((threshold - black) / denom, 2.0 ** floor)
        stops.append(math.log2(ratio))
    return float(min(stops)) if stops else 0.0


def plot_snr_panel(ax: Any, analysis: Analysis) -> None:
    x_min = EV_REPORT_FLOOR
    x_max = 0.25
    finite_snr1_stops = [v for v in analysis.snr1_stop.values() if math.isfinite(v)]
    usable_floor = max(finite_snr1_stops) if finite_snr1_stops else None

    if usable_floor is not None:
        ax.axvspan(x_min, usable_floor, color="#f5d6d2", alpha=0.34)
        ax.axvspan(usable_floor, SNR_BRIGHT_UNRELIABLE_STOP, color="#f3e5c8", alpha=0.30)
        ax.axvline(usable_floor, color="#b00020", ls=":", lw=1.3, alpha=0.95)
    else:
        ax.axvspan(x_min, SNR_BRIGHT_UNRELIABLE_STOP, color="#f3e5c8", alpha=0.24)
    ax.axvspan(SNR_BRIGHT_UNRELIABLE_STOP, x_max, color="#dddddd", alpha=0.48)
    ax.axvspan(-0.08, x_max, color=channel_color("R"), alpha=0.06)

    for y_ref, label in [
        (0.0, "SNR=1 噪声=信号"),
        (20.0, "SNR=10 可用"),
        (20.0 * math.log10(32.0), "SNR=32 干净"),
    ]:
        ax.axhline(y_ref, color="#666666", ls="--", lw=0.8, alpha=0.7)
        ax.text(x_min + 0.35, y_ref + 0.7, label, fontsize=7.3, color="#555555", va="bottom", ha="left")

    any_curve = False
    for group_name in ("R", "G", "B"):
        curve = analysis.snr_curves.get(group_name)
        if not curve:
            continue
        x = curve["stops"]
        y = curve["snr_db"]
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        any_curve = True
        color = channel_color(group_name)
        low_snr = valid & (y < 0.0) & (x <= SNR_BRIGHT_UNRELIABLE_STOP)
        reliable = valid & (y >= 0.0) & (x <= SNR_BRIGHT_UNRELIABLE_STOP)
        unreliable = valid & (x > SNR_BRIGHT_UNRELIABLE_STOP)
        ax.plot(x[low_snr], y[low_snr], color=color, lw=1.4, ls=":", alpha=0.45)
        ax.plot(x[reliable], y[reliable], color=color, lw=2.0)
        ax.plot(x[unreliable], y[unreliable], color=color, lw=1.3, ls="--", alpha=0.45)
        if np.any(reliable):
            last = np.flatnonzero(reliable)[-1]
            ax.text(float(x[last]) + 0.08, float(y[last]), group_name, color=color, fontsize=8, va="center")

    if not any_curve:
        ax.text(0.5, 0.5, "SNR 估计不可用", ha="center", va="center", transform=ax.transAxes)

    finite_values = []
    for curve in analysis.snr_curves.values():
        y = curve["snr_db"]
        finite = y[np.isfinite(y)]
        if finite.size:
            finite_values.append(finite)
    if finite_values:
        all_y = np.concatenate(finite_values)
        y_top = min(70.0, max(35.0, float(np.percentile(all_y, 98)) + 6.0))
    else:
        y_top = 40.0

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-5.0, y_top)
    ax.set_xlabel("距剪切的档数: log2(signal / fullwell signal)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("SNR 曲线: 暗部可拉余量")
    ax.grid(True, alpha=0.18)
    if usable_floor is not None:
        ax.text(
            usable_floor - 0.08,
            0.10,
            "SNR=1 可用边界",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7.3,
            color="#9b0000",
        )
    ax.text(
        x_min + 1.6,
        0.84,
        "噪声主导",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=7.5,
        color="#9b0000",
    )
    ax.text(
        -6.0,
        0.96,
        "可拉但会发糙",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.5,
        color="#986300",
    )
    ax.text(
        0.02,
        0.96,
        "SNR=1 可用DR\n" + format_snr_dr(analysis.snr1_dr),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.82},
    )
    ax.text(
        SNR_BRIGHT_UNRELIABLE_STOP + 0.08,
        0.94,
        "亮端估计不可靠",
        transform=ax.get_xaxis_transform(),
        fontsize=7.5,
        color="#555555",
        va="top",
    )
    ax.text(
        0.98,
        0.04,
        "单帧估计；灰区不可靠",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#555555",
        transform=ax.transAxes,
    )


def plot_rgb_ev_panel(ax: Any, bundle: RawBundle, analysis: Analysis, ev: Any) -> None:
    ev_high = 1.0
    x, counts, medians, floor_pct = rgb_ev_histograms_from_xyz(
        bundle.xyz_render, bundle.render_scale, EV_REPORT_FLOOR, ev_high, bins=160
    )
    total = max(int(bundle.xyz_render.shape[0] * bundle.xyz_render.shape[1]), 1)
    colors = {"R": channel_color("R"), "G": channel_color("G"), "B": channel_color("B")}

    grey_counts, grey_edges = np.histogram(
        np.clip(ev.reshape(-1), EV_REPORT_FLOOR, ev_high), bins=160, range=(EV_REPORT_FLOOR, ev_high)
    )
    grey_x = (grey_edges[:-1] + grey_edges[1:]) * 0.5
    grey_y = gaussian_smooth(grey_counts, sigma=1.0) / total * 100.0
    ax.fill_between(grey_x, 0, grey_y, color="#d7d7d7", alpha=1.0, linewidth=0, label="Y 整体亮度", zorder=1)
    ax.plot(grey_x, grey_y, color="#b0b0b0", lw=0.8, alpha=1.0, zorder=2)

    for name in ("R", "G", "B"):
        y = gaussian_smooth(counts[name], sigma=1.0) / total * 100.0
        label = f"{name} 中位 {medians[name]:.2f} EV"
        ax.plot(x, y, color=colors[name], lw=2.2, alpha=0.98, label=label, zorder=4)
        if math.isfinite(medians[name]):
            ax.axvline(medians[name], color=colors[name], ls="-", lw=0.9, alpha=0.55, zorder=5)

    ax.axvline(GRAY_EV, color="#777777", ls="--", lw=1.0, label="18% 灰", zorder=5)
    ax.axvline(0.0, color="#333333", ls=":", lw=1.0, label="渲染白点", zorder=5)
    ax.set_xlim(EV_REPORT_FLOOR, ev_high)
    ax.set_xlabel("相对线性 sRGB 白点的 EV")
    ax.set_ylabel("每档像素占比 (%)")
    ax.set_title("RGB 曝光分布 + Y 亮度背景")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.78)
    ax.grid(True, alpha=0.2)
    floor_note = (
        f"Y p1->p99.9 {analysis.ev_dr_p1_p999:.2f} 档，左端压底 {analysis.ev_floor_hit_pct:.2f}%\n"
        + "通道左端压底: "
        + " ".join(f"{name}={floor_pct[name]:.2f}%" for name in ("R", "G", "B"))
    )
    ax.text(
        0.02,
        0.04,
        floor_note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )


def plot_dashboard(
    bundle: RawBundle,
    analysis: Analysis,
    y: Any,
    ev: Any,
    out_path: Path,
    auto_ev: AutoEvResult | None = None,
) -> None:
    plt, _, ListedColormap, Patch = _matplotlib()
    configure_plot_fonts()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=120, constrained_layout=True)
    ax_raw, ax_ev, ax_gamut, ax_zone, ax_clip, ax_text = axes.ravel()

    plot_snr_panel(ax_raw, analysis)

    plot_rgb_ev_panel(ax_ev, bundle, analysis, ev)

    gamut_names = ["sRGB", "P3", "Rec2020"]
    gamut_vals = [analysis.gamut_out_pct[name] for name in gamut_names]
    bars = ax_gamut.barh(gamut_names, gamut_vals, color=["#e45756", "#72b7b2", "#54a24b"])
    xmax = max(1.0, max(gamut_vals) * 1.25)
    ax_gamut.set_xlim(0, xmax)
    ax_gamut.set_xlabel("高亮像素色域越界比例 (%)")
    ax_gamut.set_title("色域风险: 导出色彩空间")
    for bar, val in zip(bars, gamut_vals):
        ax_gamut.text(val + xmax * 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.3f}%", va="center", fontsize=9)
    ax_gamut.grid(True, axis="x", alpha=0.2)

    zones = exposure_zone_map(y, analysis.noise_floor)
    cmap = ListedColormap(["#080812", "#15376d", "#238b45", "#f3c04d", "#d7191c"])
    ax_zone.imshow(zones, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax_zone.set_title("曝光区域: 空间 EV 图")
    ax_zone.set_axis_off()
    handles = [
        Patch(facecolor="#080812", label="接近噪声底"),
        Patch(facecolor="#15376d", label="阴影"),
        Patch(facecolor="#238b45", label="中间调"),
        Patch(facecolor="#f3c04d", label="高光"),
        Patch(facecolor="#d7191c", label="剪切"),
    ]
    ax_zone.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.75)
    ax_zone.text(
        0.01,
        0.02,
        "伪色: 红色=剪切，不代表红光",
        transform=ax_zone.transAxes,
        fontsize=7,
        color="white",
        ha="left",
        va="bottom",
        bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 2},
    )

    clip_rgb = clipped_rgb_map(bundle, analysis)
    ax_clip.imshow(clip_rgb, interpolation="nearest")
    ax_clip.set_title("剪切通道: 高光修复地图")
    ax_clip.set_axis_off()
    clip_handles = [
        Patch(facecolor=channel_color("R"), label="R 剪切"),
        Patch(facecolor=channel_color("G"), label="G 剪切"),
        Patch(facecolor=channel_color("B"), label="B 剪切"),
        Patch(facecolor="#ffffff", edgecolor="#999999", label="RGB 全剪切"),
    ]
    ax_clip.legend(handles=clip_handles, fontsize=7, loc="lower right", framealpha=0.75)

    ax_text.set_axis_off()
    ax_text.set_title("摘要: Darktable 修图提示与关键指标")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(summary_lines(bundle, analysis)),
        va="top",
        ha="left",
        fontsize=6.6,
        transform=ax_text.transAxes,
    )

    fig.suptitle(f"RAW 物理诊断: {bundle.path.name}", fontsize=14)
    if auto_ev is not None:
        limit_note = (
            f" · 高光限制（目标 {auto_ev.ev_median_target:+.2f} EV）"
            if auto_ev.highlight_limited
            else ""
        )
        fig.text(
            0.5,
            0.995,
            f"全图亮度参考 {auto_ev.ev_boost:+.2f}（应用 {auto_ev.ev:+.2f} EV）{limit_note}",
            ha="center",
            va="top",
            fontsize=11,
            color="#1a1a1a",
            bbox={"facecolor": "#f3c04d", "alpha": 0.92, "edgecolor": "none", "pad": 4},
        )
    fig.savefig(out_path)
    plt.close(fig)


def default_png_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_scan.png")
