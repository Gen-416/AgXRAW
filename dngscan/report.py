# SPDX-License-Identifier: GPL-3.0-or-later
"""CLI/GUI text reports and CSV export."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from .analysis import (
    channel_list, format_channel_values, format_pct, format_snr_dr, format_wb_values,
    fullwell_note_cn, padded_channel_values, raw_health_verdict_cn,
)
from .color import clamp_float, output_gamut_label
from .constants import EPS, EV_REPORT_FLOOR
from .grade import grade_label
from .scene_transform import scene_transform_label
from .models import Analysis, AutoEvResult, RawBundle, ToneCompressionPlan
from .raw_io import highlight_mode_cn
from .tone import plan_for_mode

def darktable_guidance_lines(bundle: RawBundle, analysis: Analysis) -> list[str]:
    lines = ["Darktable 修图建议:"]
    max_clip = max(analysis.clip_pct.values()) if analysis.clip_pct else 0.0
    if analysis.ev_p999 > -0.10 or max_clip > 0.05:
        lines.append("曝光: 避免全局加曝光；高光已贴近白点，优先局部提暗部。")
    elif analysis.ev_p999 < -0.70 and max_clip < 0.01:
        lines.append(f"曝光: 仍有约 {abs(analysis.ev_p999):.1f} EV 高光余量，可少量全局加曝光。")
    else:
        lines.append("曝光: 高光余量有限；全局曝光先小步调整，再看剪切图。")

    if analysis.cfa_cell_supported and math.isfinite(analysis.cell_union_pct):
        if analysis.cell_union_pct <= 0.01:
            lines.append("高光: RAW 剪切很少，高光重建风险低。")
        elif analysis.cell_ge2_of_clipped_pct >= 50.0:
            lines.append(
                f"高光: {format_pct(analysis.cell_union_pct)}% CFA cell 剪切，且多通道占比高；细节修复有限。"
            )
        else:
            lines.append(
                f"高光: {format_pct(analysis.cell_union_pct)}% CFA cell 剪切，多为单通道；可尝试高光重建。"
            )
    else:
        lines.append("高光: 非 2x2 CFA，剪切结构无法按 Bayer cell 判断。")

    finite_dr = [v for v in analysis.snr1_dr.values() if math.isfinite(v)]
    if finite_dr:
        limiting_dr = min(finite_dr)
        lines.append(f"暗部: 三通道共同可用边界约 {limiting_dr:.2f} 档；更暗处主要是在放大噪声。")
        if limiting_dr < 8.7 or analysis.ev_floor_hit_pct > 1.0:
            lines.append("降噪: 深阴影建议先用配置文件降噪 denoise(profiled)，再控制暗部拉升。")
        else:
            lines.append("降噪: 暗部余量尚可，局部拉阴影时仍需观察色噪。")
    else:
        lines.append("暗部: SNR=1 边界不可用，暗部拉升需以目视噪声为准。")

    srgb_risk = analysis.gamut_out_pct.get("sRGB", 0.0)
    if srgb_risk >= 5.0:
        lines.append(f"色彩: sRGB 高亮越界 {srgb_risk:.2f}%；导出前降低高光色度/饱和度。")
    elif srgb_risk >= 1.0:
        lines.append(f"色彩: sRGB 高亮越界 {srgb_risk:.2f}%；鲜艳高光需留意断层。")
    else:
        lines.append(f"色彩: sRGB 越界 {srgb_risk:.2f}%，常规导出风险较低。")

    wb = padded_channel_values(bundle.camera_wb, analysis.channel_ids)
    boosted = [(analysis.labels[cid], wb[cid]) for cid in analysis.channel_ids if wb[cid] > 1.8]
    if boosted:
        label, gain = max(boosted, key=lambda item: item[1])
        lines.append(f"白平衡: {label} 增益 {gain:.2g} 较高，提亮后该通道噪声/剪切更显眼。")
    return lines


def priors_line_cn(bundle: RawBundle, analysis: Analysis) -> str:
    ident = f"{bundle.shot_make or '?'} {bundle.shot_model or '?'}".strip()
    iso = f"ISO{bundle.shot_iso}" if bundle.shot_iso else "ISO?"
    if analysis.prior_id is None:
        return (
            f"机型/先验: {ident} @ {iso}（无先验表条目，全部使用单帧实测；"
            "传感器标定数据不完整，绝对档位/动态范围数字可能有偏差，渲染仍可正常使用）"
        )
    parts = [f"机型/先验: {analysis.prior_id} @ {iso}"]
    if analysis.gain_e_per_dn is not None:
        parts.append(f"增益≈{analysis.gain_e_per_dn:.2f} e⁻/DN")
    if analysis.noise_floor_e is not None:
        parts.append(f"实测噪声底≈{analysis.noise_floor_e:.1f} e⁻")
    if analysis.prior_read_noise_e is not None:
        parts.append(f"读出噪声先验={analysis.prior_read_noise_e:.2f} e⁻")
    if analysis.prior_pdr_ev is not None:
        parts.append(f"PDR先验={analysis.prior_pdr_ev:.2f} EV")
    if math.isfinite(analysis.usable_dr_eff_ev) and abs(analysis.usable_dr_eff_ev - analysis.usable_dr_ev) > 0.01:
        parts.append(f"计划用DR={analysis.usable_dr_eff_ev:.2f}(先验收敛)")
    return "  ".join(parts)


def health_line_cn(analysis: Analysis) -> str:
    if not math.isfinite(analysis.health_lag1_corr):
        return "RAW 健康度: n/a"
    return (
        f"RAW 健康度: 暗部lag1相关={analysis.health_lag1_corr:.3f}"
        f"  直方图空码={analysis.health_hist_empty_pct:.1f}%"
        f" → {raw_health_verdict_cn(analysis.health_lag1_corr, analysis.health_hist_empty_pct)}"
    )


def describe_wb_mode(wb_mode: str) -> str:
    """Honest WB label for reports: the declared reference, not a stale binary.

    The daylight/camera binary predates the fixed-Kelvin modes; letting 5500k print
    as "相机白平衡" mislabelled a declared standard as AsShot — the report-side twin
    of a hidden white balance.
    """
    from .wb import KELVIN_WB_MODES

    if wb_mode == "daylight":
        return "日光固定配平"
    entry = KELVIN_WB_MODES.get(str(wb_mode))
    if entry is not None:
        return f"固定 {entry[0]:.0f}K 白平衡（{entry[1]}）"
    return "相机白平衡"


def wb_line_cn(bundle: RawBundle) -> str:
    mode = (
        "相机 AsShot" if bundle.wb_mode == "camera" else describe_wb_mode(bundle.wb_mode)
    )
    line = f"白平衡: {mode}"
    degradation = getattr(bundle, "wb_degradation", None)
    if degradation:
        line += f"\n白平衡警示: {degradation}"
    support = getattr(bundle, "camera_data_support", None)
    if support:
        line += f"\n机型数据支撑: {support}"
    cam = bundle.camera_wb
    day = bundle.daylight_wb
    if cam and day and len(cam) >= 3 and len(day) >= 3 and all(v > 0 for v in (cam[0], cam[2], day[0], day[2], cam[1], day[1])):
        # AsShot vs daylight, normalized to green: the scene's own light-source testimony.
        dev_r = math.log2((cam[0] / cam[1]) / (day[0] / day[1]))
        dev_b = math.log2((cam[2] / cam[1]) / (day[2] / day[1]))
        line += f"  AsShot相对日光: R{dev_r:+.2f}EV B{dev_b:+.2f}EV"
        if max(abs(dev_r), abs(dev_b)) > 0.8:
            line += "（明显偏离日光：人工/混合光源）"
    return line


def summary_lines(bundle: RawBundle, analysis: Analysis) -> list[str]:
    black = padded_channel_values(bundle.black_levels, analysis.channel_ids)
    wb = padded_channel_values(bundle.camera_wb, analysis.channel_ids)
    spike_flags = {
        cid: f"{'可靠' if analysis.ceil_spike_ok[cid] else '弱'}({analysis.ceil_spike_counts[cid]})"
        for cid in analysis.channel_ids
    }
    clip_parts = " ".join(
        f"{analysis.labels[cid]}={format_pct(analysis.clip_pct[cid])}" for cid in analysis.channel_ids
    )
    if analysis.cfa_cell_supported:
        cell_line = (
            f"2x2 剪切 cell: {format_pct(analysis.cell_union_pct)}%  "
            f"剪切 cell 中 >=2通道: {format_pct(analysis.cell_ge2_of_clipped_pct)}%"
        )
        cell_mix = "2x2 剪切通道数分布: " + " ".join(
            f"{k}={format_pct(analysis.cell_k_of_clipped_pct[k])}%" for k in range(1, 5)
        )
    else:
        cell_line = "2x2 剪切 cell: n/a (非 2x2 CFA)"
        cell_mix = "2x2 剪切通道数分布: n/a"
    return darktable_guidance_lines(bundle, analysis) + [
        "",
        "关键 RAW 指标:",
        f"文件: {bundle.path.name}",
        f"可见传感器: {bundle.raw_image.shape[1]} x {bundle.raw_image.shape[0]}",
        f"方向标记: flip={bundle.orientation_flip}  JPEG 导出: 按相机方向自动转正",
        f"white_level 标签: {bundle.white_level}  容器位深估计: {analysis.container_bits_est}",
        "黑电平 DN: " + format_channel_values(black, analysis.labels, analysis.channel_ids, "{:.1f}"),
        "相机白平衡: " + format_wb_values(wb, analysis.labels, analysis.channel_ids),
        "元数据白电平 DN: "
        + format_channel_values(analysis.saturation_levels, analysis.labels, analysis.channel_ids, "{}"),
        "观测 ceiling DN: " + format_channel_values(analysis.ceilings, analysis.labels, analysis.channel_ids, "{}"),
        "满阱堆积: " + format_channel_values(spike_flags, analysis.labels, analysis.channel_ids, "{}"),
        "采用满阱 DN: " + format_channel_values(analysis.channel_fullwell, analysis.labels, analysis.channel_ids, "{}"),
        "剪切阈值 DN: " + format_channel_values(analysis.channel_thresholds, analysis.labels, analysis.channel_ids, "{}"),
        f"保守满阱摘要={analysis.fullwell} 来自 {channel_list(analysis.fullwell_channel_ids, analysis.labels)}",
        "满阱说明: " + fullwell_note_cn(analysis.fullwell_note),
        "剪切 %: " + clip_parts,
        cell_line,
        cell_mix,
        f"EV p1/中位/p99/p99.9 下限 {EV_REPORT_FLOOR:.0f}: {analysis.ev_p1:.2f} / {analysis.ev_median:.2f} / "
        f"{analysis.ev_p99:.2f} / {analysis.ev_p999:.2f}",
        f"画面 DR p1->p99.9: {analysis.ev_dr_p1_p999:.2f} 档  左端压底: {analysis.ev_floor_hit_pct:.2f}% 原始p1={analysis.ev_raw_p1:.2f}",
        f"中位亮度相对 18% 灰: {analysis.median_vs_gray_ev:+.2f} EV",
        f"RAW 噪声底: {analysis.noise_floor:.6g}  可用 DR 上限: {analysis.usable_dr_ev:.2f} 档",
        priors_line_cn(bundle, analysis),
        health_line_cn(analysis),
        wb_line_cn(bundle),
        "SNR=1 可用 DR: " + format_snr_dr(analysis.snr1_dr),
        "高亮色域越界 %: "
        + (
            " ".join(
                f"{name}={analysis.gamut_out_pct[name]:.3f}"
                for name in ("sRGB", "P3", "Rec2020")
                if name in analysis.gamut_out_pct
            )
            or "（本次仅分析输出色域子集）"
        ),
        f"高亮采样比例: {analysis.bright_pixel_pct:.2f}% 像素",
        f"最不易剪切通道: {analysis.survivor_channel}",
        "注: SNR/噪声为单帧估计，不是光子转移测量。",
    ]


def print_report(
    bundle: RawBundle,
    analysis: Analysis,
    out_path: Path | None,
    csv_path: Path | None,
    jpeg_path: Path | None,
    jpeg_quality: int,
    jpeg_mode: str,
    jpeg_icc_embedded: bool,
    jpeg_ev: float = 0.0,
    tone_plan: ToneCompressionPlan | None = None,
    output_gamut: str = "srgb",
    auto_ev: AutoEvResult | None = None,
    jpeg_grade: str = "none",
    jpeg_grade_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
) -> None:
    for line in summary_lines(bundle, analysis):
        print(line)
    if out_path is not None:
        print(f"PNG 图像: {out_path}")
    if csv_path is not None:
        print(f"CSV 指标: {csv_path}")
    if jpeg_path is not None:
        print(f"JPEG 图像: {jpeg_path}")
        reported_mode = (
            str(getattr(tone_plan, "tone_core", jpeg_mode))
            if jpeg_mode in ("agx", "gated")
            else jpeg_mode
        )
        wb_label = describe_wb_mode(bundle.wb_mode)
        wb_degradation = getattr(bundle, "wb_degradation", None)
        if wb_degradation:
            wb_label += f"；警示：{wb_degradation}"
        data_support = getattr(bundle, "camera_data_support", None)
        if data_support:
            print(f"机型数据支撑: {data_support}")
        decoder = getattr(bundle, "scene_decoder", "libraw") or "libraw"
        decoder_version = getattr(bundle, "scene_decoder_version", None)
        decoder_runtime = getattr(bundle, "scene_decoder_runtime", None)
        if decoder == "coreimage":
            opcodes = tuple(getattr(bundle, "scene_opcode_names", ()) or ())
            opcode_note = f"，已执行 DNG opcode: {'/'.join(opcodes)}" if opcodes else ""
            scale_mode = getattr(bundle, "scene_scale_mode", None)
            align = getattr(bundle, "scene_align_factor", 1.0)
            align_err = getattr(bundle, "scene_align_error", None)
            if scale_mode == "aligned":
                if align_err:
                    scale_note = f"，逐文件尺度对齐失败（{align_err}），已回退 1×"
                else:
                    scale_note = (
                        f"，尺度对齐={align:.4f}×（逐文件解码绿色中位比；"
                        "confidence=relative，非自动曝光/非绝对辐射标定）"
                    )
            elif scale_mode == "measured":
                scale_note = "，尺度=旧版 Sigma fp 固定实测倍率（仅供复现）"
            else:
                scale_note = "，尺度=Core Image 原生单位（unity）"
            decoder_label = (
                f"Core Image/{decoder_version or '?'}"
                f"{f' · {decoder_runtime}' if decoder_runtime else ''}"
                "（独立管线：无逐像素 CFA 证据"
                f"{opcode_note}{scale_note}）"
            )
            highlight_note = (
                "高光处理=Core Image 高光重建（保留镜面余量）"
                f"（--highlight-mode {bundle.scene_highlight_mode} 不作用于此管线）"
            )
        else:
            decoder_label = "LibRaw"
            highlight_note = f"高光处理={highlight_mode_cn(bundle.scene_highlight_mode)}"
        ev_label = "全图亮度参考" if auto_ev is not None else "EV 补偿"
        ev_note = (
            f"{ev_label}={jpeg_ev:+.2f}（参考提升 {auto_ev.ev_boost:+.2f} EV）"
            if auto_ev is not None
            else f"EV 补偿={jpeg_ev:+.2f}，固定常数非自适应"
        )
        brighten_note = "全图亮度参考" if auto_ev is not None else "手动 EV / 固定锚点"
        # BaselineExposure moves every pixel and can be scene-dependent in ProRAW. It is
        # file-authored baseline rendering compensation, not an auto-gray decision or the
        # shutter/aperture/ISO measurement, so name it without calling it capture exposure.
        baseline_exposure = getattr(bundle, "baseline_exposure", None)
        baseline_exposure_note = (
            f"文件 BaselineExposure={baseline_exposure:+.3f} EV"
            f"（{'解码器内应用' if getattr(bundle, 'baseline_exposure_baked_in', False) else '线性交接后单次应用'}）；"
            if baseline_exposure is not None
            else ""
        )
        print(
            f"JPEG 设置: scene-linear Rec.2020 起点（解码={decoder_label}）；"
            f"8-bit {output_gamut_label(output_gamut)}（TPDF 抖动）；"
            f"{wb_label}；{brighten_note}；"
            f"曝光锚定增益={bundle.exposure_gain:.3f}（{ev_note}）；"
            f"{baseline_exposure_note}"
            f"模式={reported_mode}；{highlight_note}；"
            f"AgX 前馈={scene_transform_label(scene_transform)}（强度={scene_transform_strength:.2f}）；"
            f"成片风格={grade_label(jpeg_grade)}（强度={jpeg_grade_strength:.2f}）；"
            f"质量={jpeg_quality}；"
            f"ICC={'已嵌入' if jpeg_icc_embedded else '未嵌入'}"
        )
        if auto_ev is not None:
            anchored = auto_ev.anchored_median_ev
            print(f"亮度参考校验: 可靠 scene body 中位相对 18% 灰 {anchored:+.2f} EV")
        else:
            anchored = analysis.median_vs_gray_ev + math.log2(max(bundle.exposure_gain, EPS))
            print(
                f"RAW 统计校验: 解码场景中位亮度相对 18% 灰 {anchored:+.2f} EV"
                + ("（暗调场景，符合拍摄意图即可）" if anchored < -1.0 else "")
            )
        if auto_ev is not None:
            limit_note = (
                f"；高光限制，参考目标 {auto_ev.ev_median_target:+.2f} EV"
                if auto_ev.highlight_limited
                else ""
            )
            print(
                f"全图亮度参考：提升 {auto_ev.ev_boost:+.2f} EV（相对 EV 0）"
                f"{limit_note}；应用 EV={auto_ev.ev:+.2f}"
            )
        print(f"JPEG 策略: {jpeg_policy_cn(reported_mode, output_gamut, getattr(tone_plan, 'curve_preset', 'none'))}")
        plan_line = jpeg_tone_plan_cn(
            bundle,
            analysis,
            reported_mode,
            tone_plan,
            output_gamut,
            scene_transform,
            scene_transform_strength,
        )
        if plan_line:
            print(f"JPEG 自动计划: {plan_line}")


def jpeg_policy_cn(
    mode: str, output_gamut: str = "srgb", curve_preset: str = "none"
) -> str:
    label = output_gamut_label(output_gamut)
    if mode == "agx":
        # The formation order must be reported as actually executed: a film preset
        # inserts the measured channel-ratio stage between hue restore and outset.
        from .film_curve import channel_ratio_field

        ratio_stage = (
            "→胶片逐通道比率（实测层饱和差异，中性轴恒等）"
            if channel_ratio_field(str(curve_preset or "")) is not None
            else ""
        )
        return f"agx: scene-linear Rec.2020 工作空间；白平衡按导出选项；无隐式自动增亮；高光重建属于所选解码器；AgX inset→端点归一化 C1→hue restore{ratio_stage}→outset；可靠 scene Y 只编译黑白范围与 toe/shoulder；逐像素 CFA mask 存在时才驱动曲线前褪白；最后转 {label}；4:4:4 色度采样"
    if mode == "lum":
        return f"lum: scene-linear Rec.2020 工作空间；逐像素 CFA mask 存在时驱动曲线前褪白；场景编译 C1 作用于标量亮度/norm，RGB 比例保持；显示白附近再温和褪色；无 AgX inset/outset，最后转 {label} 并做输出色域 fit"
    if mode == "gated":
        return f"gated: 仅限 LibRaw；场景编译 C1 的亮度结果是唯一亮度权威；AgX 色彩候选先对齐到同一 Y，再由 CFA 剪切、余量与 SNR 权限逐像素混合；最后转 {label} 并做输出色域 fit"
    if mode == "neutral":
        return f"neutral: 固定 Y 比例诊断曲线；不读取场景编译 endpoint，不进入 AgX inset/outset；逐像素 CFA mask 存在时仍可做剪切褪白；最后转 {label} 并做输出色域 fit"
    return ""


def jpeg_tone_plan_cn(
    bundle: RawBundle,
    analysis: Analysis,
    mode: str,
    tone_plan: ToneCompressionPlan | None = None,
    output_gamut: str = "srgb",
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
) -> str:
    if mode == "neutral":
        from .neutral import NEUTRAL_BLACK_EV, NEUTRAL_WHITE_EV

        return (
            f"neutral: fixed diagnostic curve (Y ratio, black={NEUTRAL_BLACK_EV:.1f}EV "
            f"white=+{NEUTRAL_WHITE_EV:.1f}EV); no AgX; delivery={output_gamut}"
        )
    if mode in ("agx", "lum", "gated"):
        plan = tone_plan if tone_plan is not None else plan_for_mode(
            bundle, analysis, "agx", output_gamut, scene_transform, scene_transform_strength, tone_core=mode
        )
        label = {"agx": "AgX", "gated": "Gated", "lum": f"lum({plan.lum_norm})", "neutral": "neutral"}.get(
            mode, mode
        )
        extras = []
        if abs(plan.pivot_ev_offset) > 1e-3:
            extras.append(f"pivot={plan.pivot_ev_offset:+.2f}EV")
        if abs(plan.hue_restore - 0.6) > 1e-3:
            extras.append(f"hue_restore={plan.hue_restore:.2f}")
        if plan.target_black_linear > 1e-4:
            extras.append(f"lift黑={plan.target_black_linear:.3f}")
        if plan.target_white_linear < 1.0 - 1e-4:
            extras.append(f"褪白={plan.target_white_linear:.3f}")
        if getattr(plan, "agx_primaries", "base") != "base":
            extras.append(f"primaries={plan.agx_primaries}")
        if getattr(plan, "endpoint_mode", "adaptive") != "adaptive":
            note = getattr(plan, "endpoint_note", None)
            extras.append(
                f"endpoint={plan.endpoint_mode}" + (f"（{note}）" if note else "")
            )
        extra_text = ("；" + "，".join(extras)) if extras else ""
        return (
            f"{label} endpoint black={plan.black_ev:.2f}EV / toe接回={plan.toe_start_ev:.2f}EV / "
            f"shoulder起点={plan.shoulder_start_ev:+.2f}EV / white=+{plan.white_ev:.2f}EV，"
            f"DR={plan.dynamic_range_ev:.2f}档；Y p1/p50/p99.9={plan.luma_p1:.4f}/{plan.luma_p50:.4f}/{plan.luma_p999:.4f}；"
            f"pivot=0EV→18%（darktable 默认 gamma）；contrast={plan.contrast:.2f}, toe={plan.toe_power:.2f}, shoulder={plan.shoulder_power:.2f}, view brightness={plan.view_brightness:.2f}；"
            f"纯度补偿={plan.punch_strength:.2f}；"
            f"{plan.target_gamut} 负通道={plan.negative_rgb_pct:.2f}%"
            f"{extra_text}"
        )
    return ""


def csv_row(
    bundle: RawBundle,
    analysis: Analysis,
    out_path: Path | None,
    jpeg_path: Path | None = None,
    jpeg_quality: int | None = None,
    jpeg_mode: str = "",
    jpeg_icc_embedded: bool = False,
    jpeg_ev: float = 0.0,
    tone_plan: ToneCompressionPlan | None = None,
    output_gamut: str = "srgb",
    auto_ev: AutoEvResult | None = None,
    jpeg_grade: str = "none",
    jpeg_grade_strength: float = 1.0,
    scene_transform: str = "none",
    scene_transform_strength: float = 1.0,
) -> dict[str, Any]:
    reported_mode = (
        str(getattr(tone_plan, "tone_core", jpeg_mode))
        if jpeg_mode in ("agx", "gated")
        else jpeg_mode
    )
    row: dict[str, Any] = {
        "file": str(bundle.path),
        "filename": bundle.path.name,
        "width": int(bundle.raw_image.shape[1]),
        "height": int(bundle.raw_image.shape[0]),
        "orientation_flip": int(bundle.orientation_flip),
        "white_level": int(bundle.white_level),
        "container_bits_est": int(analysis.container_bits_est),
        "fullwell_reference": int(analysis.fullwell),
        "fullwell_min_channel_ceil": int(analysis.fullwell),
        "fullwell_channels": channel_list(analysis.fullwell_channel_ids, analysis.labels),
        "fullwell_note": analysis.fullwell_note,
        "threshold_reference": int(analysis.threshold),
        "unified_threshold_thr": int(analysis.threshold),
        "cfa_cell_supported": analysis.cfa_cell_supported,
        "cell_clip_union_pct": analysis.cell_union_pct,
        "cell_clip_ge2_of_clipped_pct": analysis.cell_ge2_of_clipped_pct,
        "ev_raw_p1": analysis.ev_raw_p1,
        "ev_p1": analysis.ev_p1,
        "ev_median": analysis.ev_median,
        "ev_p99": analysis.ev_p99,
        "ev_p99_9": analysis.ev_p999,
        "frame_dr_p1_to_p99_9_stops": analysis.ev_dr_p1_p999,
        "ev_report_floor": EV_REPORT_FLOOR,
        "ev_floor_hit_pct": analysis.ev_floor_hit_pct,
        "median_vs_18pct_gray_ev": analysis.median_vs_gray_ev,
        "raw_noise_floor_single_frame": analysis.noise_floor,
        "usable_dr_noise_limited_upper_bound_stops": analysis.usable_dr_ev,
        "camera_make": bundle.shot_make or "",
        "camera_model": bundle.shot_model or "",
        "iso": bundle.shot_iso if bundle.shot_iso else "",
        "wb_mode": bundle.wb_mode,
        "scene_decoder": getattr(bundle, "scene_decoder", "libraw") or "libraw",
        "scene_decoder_version": getattr(bundle, "scene_decoder_version", None) or "",
        "scene_decoder_runtime": getattr(bundle, "scene_decoder_runtime", None) or "",
        "scene_geometry_corr": (
            getattr(bundle, "scene_geometry_corr", None)
            if getattr(bundle, "scene_geometry_corr", None) is not None
            else ""
        ),
        "prior_id": analysis.prior_id or "",
        "gain_e_per_dn": analysis.gain_e_per_dn if analysis.gain_e_per_dn is not None else "",
        "noise_floor_e": analysis.noise_floor_e if analysis.noise_floor_e is not None else "",
        "prior_read_noise_e": analysis.prior_read_noise_e if analysis.prior_read_noise_e is not None else "",
        "prior_pdr_ev": analysis.prior_pdr_ev if analysis.prior_pdr_ev is not None else "",
        "usable_dr_eff_ev": analysis.usable_dr_eff_ev,
        "health_lag1_corr": analysis.health_lag1_corr,
        "health_hist_empty_pct": analysis.health_hist_empty_pct,
        "snr1_dr_R": analysis.snr1_dr.get("R", float("nan")),
        "snr1_dr_G": analysis.snr1_dr.get("G", float("nan")),
        "snr1_dr_B": analysis.snr1_dr.get("B", float("nan")),
        "snr1_stop_R": analysis.snr1_stop.get("R", float("nan")),
        "snr1_stop_G": analysis.snr1_stop.get("G", float("nan")),
        "snr1_stop_B": analysis.snr1_stop.get("B", float("nan")),
        # Defensive .get: the CSV path requests full diagnostics, but a subset-gamut
        # analysis (GUI / plain export) must degrade to blanks, never KeyError.
        "gamut_out_srgb_bright_pct": analysis.gamut_out_pct.get("sRGB", ""),
        "gamut_out_p3_bright_pct": analysis.gamut_out_pct.get("P3", ""),
        "gamut_out_rec2020_bright_pct": analysis.gamut_out_pct.get("Rec2020", ""),
        "bright_pixel_pct": analysis.bright_pixel_pct,
        "survivor_channel": analysis.survivor_channel,
        "png": str(out_path) if out_path is not None else "",
        "jpeg": str(jpeg_path) if jpeg_path is not None else "",
        "jpeg_mode": reported_mode if jpeg_path is not None else "",
        "jpeg_output_gamut": output_gamut if jpeg_path is not None else "",
        "jpeg_output_gamut_label": output_gamut_label(output_gamut) if jpeg_path is not None else "",
        "jpeg_highlight_mode": bundle.scene_highlight_mode if jpeg_path is not None else "",
        "jpeg_highlight_mode_cn": highlight_mode_cn(bundle.scene_highlight_mode) if jpeg_path is not None else "",
        "jpeg_quality": int(jpeg_quality) if jpeg_quality is not None else "",
        "jpeg_ev": jpeg_ev if jpeg_path is not None else "",
        "jpeg_grade": jpeg_grade if jpeg_path is not None else "",
        "jpeg_grade_label": grade_label(jpeg_grade) if jpeg_path is not None else "",
        "jpeg_grade_strength": jpeg_grade_strength if jpeg_path is not None else "",
        "jpeg_scene_transform": scene_transform if jpeg_path is not None else "",
        "jpeg_scene_transform_label": scene_transform_label(scene_transform) if jpeg_path is not None else "",
        "jpeg_scene_transform_strength": scene_transform_strength if jpeg_path is not None else "",
        "jpeg_ev_auto_boost": auto_ev.ev_boost if jpeg_path is not None and auto_ev is not None else "",
        "jpeg_ev_auto_limited": auto_ev.highlight_limited if jpeg_path is not None and auto_ev is not None else "",
        "jpeg_ev_auto_median_target": auto_ev.ev_median_target if jpeg_path is not None and auto_ev is not None else "",
        "jpeg_exposure_gain": bundle.exposure_gain if jpeg_path is not None else "",
        "jpeg_icc_embedded": jpeg_icc_embedded if jpeg_path is not None else "",
        "jpeg_srgb_icc_embedded": jpeg_icc_embedded if jpeg_path is not None and output_gamut == "srgb" else "",
        "jpeg_policy_cn": jpeg_policy_cn(reported_mode, output_gamut, getattr(tone_plan, "curve_preset", "none")) if jpeg_path is not None else "",
        "jpeg_tone_plan_cn": jpeg_tone_plan_cn(
            bundle,
            analysis,
            reported_mode,
            tone_plan,
            output_gamut,
            scene_transform,
            scene_transform_strength,
        ) if jpeg_path is not None else "",
        "note": "SNR/噪声为单帧估计，不是光子转移测量；位深不等于可用动态范围。",
        "darktable_guidance_cn": " | ".join(darktable_guidance_lines(bundle, analysis)[1:]),
        "fullwell_note_cn": fullwell_note_cn(analysis.fullwell_note),
    }

    black = padded_channel_values(bundle.black_levels, analysis.channel_ids)
    wb = padded_channel_values(bundle.camera_wb, analysis.channel_ids)
    for cid in analysis.channel_ids:
        label = analysis.labels[cid]
        row[f"black_{label}"] = black[cid]
        row[f"camera_wb_{label}"] = wb[cid]
        row[f"metadata_white_{label}"] = analysis.saturation_levels[cid]
        row[f"ceil_{label}"] = analysis.ceilings[cid]
        row[f"fullwell_{label}"] = analysis.channel_fullwell[cid]
        row[f"clip_threshold_{label}"] = analysis.channel_thresholds[cid]
        row[f"ceil_spike_exact_count_{label}"] = analysis.ceil_spike_counts[cid]
        row[f"ceil_spike_near_count_{label}"] = analysis.ceil_near_counts[cid]
        row[f"ceil_spike_ok_{label}"] = analysis.ceil_spike_ok[cid]
        row[f"clip_pct_{label}"] = analysis.clip_pct[cid]
    for k in range(1, 5):
        row[f"cell_clip_{k}_of_clipped_pct"] = analysis.cell_k_of_clipped_pct[k]
        row[f"cell_clip_{k}_of_all_pct"] = analysis.cell_k_of_all_pct[k]
    return row


def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
