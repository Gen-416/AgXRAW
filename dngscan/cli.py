# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .debug_util import maybe_print_exc

from ._deps import IMPORT_ERRORS
from .agx import AGX_PRIMARIES_CLI_CHOICES, resolve_agx_primaries
from .analysis import analyze
from .auto_ev import AutoEvResult, compute_auto_ev, is_ev_auto, parse_ev_value, resolve_export_ev
from .color import output_gamut_space
from .constants import (
    CHROMA_CHOICES, COREIMAGE_SCALE_CHOICES, COREIMAGE_SCALE_DEFAULT_MODE,
    COREIMAGE_SCALE_MEASURED_RATIO, COREIMAGE_VERSION_CHOICES, DECODER_CHOICES,
    DEFAULT_HDR_DRT, DEFAULT_HDR_HEADROOM_EV, DEMOSAIC_CHOICES, HDR_DRT_CHOICES,
    JPEG_OUTPUT_FORMATS, MAX_HDR_HEADROOM_EV, WB_CHOICES,
)
from .delivery import (
    ARCHIVE_CHROMA,
    ARCHIVE_JPEG_QUALITY,
    DEFAULT_DELIVERY_PROFILE,
    DELIVERY_PROFILE_CHOICES,
    container_for_output_format,
    is_hdr_output_format,
    profile_from_encode_settings,
    resolve_delivery_profile,
)
from .export import chroma_to_subsampling, export_jpeg
from .film_curve import FILM_CURVE_CHOICES, FILM_CURVE_PRESETS
from .scene_transform import SCENE_TRANSFORMS
from .lens_filter import LENS_FILTER_CHOICES, validate_lens_filter
from .grade import RENDER_MODE, grade_choices, resolve_grade
from .plot import default_png_path, plot_dashboard
from .raw_io import load_raw
from .report import csv_row, print_report, write_csv
from .scene_transform import SCENE_TRANSFORM_CHOICES
from .models import RenderAdjustments
from .scene_scale import with_intent_exposure
from .tone import (
    ENDPOINT_MODE_CHOICES, LUM_NORM_CHOICES, TONE_CORE_CHOICES,
    apply_render_adjustments, build_render_plan,
)


def require_dependencies() -> None:
    if IMPORT_ERRORS:
        joined = "\n  ".join(IMPORT_ERRORS)
        raise RuntimeError(
            "Missing or broken dependency. Install only the required packages "
            "(rawpy, numpy, matplotlib) and rerun.\n  " + joined
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgX RAW/DNG → JPEG；可选六面板诊断 PNG。"
    )
    parser.add_argument("path", type=Path, help="RAW/DNG 文件路径")
    parser.add_argument(
        "--margin",
        type=int,
        default=4,
        help="每通道满阱剪切阈值的 DN 回退量 (默认: 4)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="导出六面板诊断 PNG；纯 JPEG 转换默认不画图",
    )
    parser.add_argument("--out", type=Path, default=None, help="诊断 PNG 输出路径；设置后隐含 --scan")
    parser.add_argument("--csv", type=Path, default=None, help="可选指标 CSV 路径")
    parser.add_argument(
        "--jpeg",
        type=Path,
        default=None,
        help="可选 8-bit JPEG 输出路径",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="JPEG 质量 1-100；默认跟随 --delivery-profile（archive=100，share=90）",
    )
    parser.add_argument(
        "--chroma",
        choices=CHROMA_CHOICES,
        default=None,
        help=(
            "色度采样: 444/422/420；默认跟随 --delivery-profile（archive=444，share=420）。"
            "Ultrahdr 的主图采样由 Core Image 按 quality 决定；share 档通常为 4:2:0。"
        ),
    )
    parser.add_argument(
        "--delivery-profile",
        choices=DELIVERY_PROFILE_CHOICES,
        default=None,
        help=(
            "交付编码档: archive=q100/4:4:4 严格 round-trip；"
            "share=q90/4:2:0 倾向，体积更小、门禁放宽。只影响最后编码，不重算 AgX/HDR。"
            "缺省时：未显式给 --jpeg-quality/--chroma 则为 archive；"
            "给了则按参数值推断门禁档（q>=98 且 444 走 archive，其余走 share）。"
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=JPEG_OUTPUT_FORMATS,
        default="sdr",
        help=(
            "输出格式: sdr=普通 JPEG；ultrahdr=Apple ISO gain-map JPEG；"
            "ultrahdr-heic=同内容 HEIC 容器（实测在本机 Core Image 下并不更小，"
            "share 档误差也更大；主要用于需要 HEIC 的下游）"
        ),
    )
    parser.add_argument(
        "--hdr-headroom",
        type=float,
        default=DEFAULT_HDR_HEADROOM_EV,
        help=(
            f"HDR display capacity（EV，相对 100 nit reference white）；"
            f"默认 {DEFAULT_HDR_HEADROOM_EV}（800 nit），上限 {MAX_HDR_HEADROOM_EV:.6f}（4000 nit）。"
            "实际内容余量由成片决定，不是归一化目标。"
        ),
    )
    parser.add_argument(
        "--hdr-drt",
        choices=HDR_DRT_CHOICES,
        default=DEFAULT_HDR_DRT,
        help="HDR display rendering transform（当前仅 agx=dngscan 对 darktable AgX formation 的 HDR 扩展）",
    )
    parser.add_argument(
        "--hdr-debug-dir",
        type=Path,
        default=None,
        help="可选：写出 HDR 诊断中间结果目录",
    )
    parser.add_argument(
        "--ev",
        default="0",
        help="手动曝光补偿（档），或 auto=按可靠 scene body 中位计算 18%% 灰参考（高光保护；仅显式指定时应用）",
    )
    parser.add_argument(
        "--highlight-mode",
        choices=("clip", "blend", "reconstruct"),
        default="clip",
        help="仅 LibRaw：高光处理 clip/blend/reconstruct；RAW 9 固定使用 Apple 高光重建",
    )
    parser.add_argument(
        "--grade",
        choices=grade_choices(),
        default="none",
        help="可选内置色彩风格；本地 LUT 仅在文件存在时显示",
    )
    parser.add_argument(
        "--grade-strength",
        type=float,
        default=1.0,
        help="成片风格强度 0-1.5（默认 1.0；0=关闭效果）",
    )
    parser.add_argument(
        "--scene-transform",
        choices=SCENE_TRANSFORM_CHOICES,
        default="none",
        help="AgX 前 scene-linear Rec.2020 前馈变换；none=关闭，arri_skin_d55=demo ARRI 式肤色前馈",
    )
    parser.add_argument(
        "--scene-transform-strength",
        type=float,
        default=None,
        help=(
            "scene transform 强度 0-3（默认 1.0；0=关闭效果；>1 用于诊断/强化 A/B）。"
            "显式给出的值总是生效——包括显式的 1.0（胶片风格配对只填充未给出的层）"
        ),
    )
    parser.add_argument(
        "--punch",
        type=float,
        default=1.0,
        help="AgX 纯度补偿倍率 0-1.5（默认 1.0=场景自动值；0=关闭；夜景自动为 0）",
    )
    # Bounded post-plan biases, identical in meaning and range to the GUI sliders, so a
    # render dialled in there can be reproduced from the command line. 0 is exact
    # identity; the automatic endpoints and RAW evidence decisions stay authoritative.
    for _flag, _help in (
        ("midtone-brightness", "中间调亮度偏置 -1..1（显示端内部提升，不改曝光与端点）"),
        ("midtone-contrast", "中间调对比偏置 -1..1"),
        ("shadow-transition", "暗部过渡 -1..1（正=趾部更开）"),
        ("highlight-transition", "高光过渡 -1..1（正=肩部更柔）"),
        ("highlight-fade", "高光褪色 -1..1（显示端色度退让）"),
    ):
        parser.add_argument(f"--{_flag}", type=float, default=0.0, help=_help)
    parser.add_argument(
        "--endpoint-mode",
        choices=ENDPOINT_MODE_CHOICES,
        default="adaptive",
        help=(
            "曲线端点策略：adaptive=场景百分位自适应（默认，现状）；"
            "evidence=端点钉在证据界——黑端点=实测噪声底 EV（有传感器先验用先验读出噪声，"
            "无先验用单帧估计并注记），白端点只信可靠 RAW 尾部（保留最低白点地板；"
            "证据缺席时如实回退自适应并注记）。pivot 锚定不变（0EV→18%%）"
        ),
    )
    parser.add_argument(
        "--toe-end-offset",
        type=float,
        default=0.0,
        help=(
            "趾部收黑点 EV 偏移 -3..+0.5（0=现状）。负值把曲线落到近黑的 EV 下移，"
            "让更深的阴影保持可读、更晚坠向黑点；通过重解 toe 形状实现，"
            "不移动黑点、白点与 pivot 锚"
        ),
    )
    parser.add_argument(
        "--shoulder-white-offset",
        "--shoulder-start-offset",  # 旧名别名：兼容既有脚本/设置
        dest="shoulder_white_offset",
        type=float,
        default=0.0,
        help=(
            "肩部收白点 EV 偏移 -2..+3（0=现状）。控制曲线升到近白参考"
            "（黑地板到白点跨度的 90%%）的场景 EV：正值推迟收白——高光层次更晚合并、"
            "滚降更柔；负值提早收白、肩部更硬。通过重解肩部曲率实现，"
            "不移动黑点、白点、肩部起点与曝光锚；超出可达范围的请求钳到最软/最硬"
            "合法肩部，编译事实回报实际收白点"
        ),
    )
    parser.add_argument(
        "--agx-primaries",
        choices=AGX_PRIMARIES_CLI_CHOICES,
        default=None,
        help=(
            "仅 tone-core=agx 的 AgX 原色几何：base=固定版本 darktable scene 默认；"
            "smooth=darktable smooth；punchy/muted=纯度变化参考。默认 base；显式给出"
            "的值总是生效——包括显式的 base（胶片风格配对只填充未给出的层）"
        ),
    )
    parser.add_argument(
        "--tone-core",
        choices=TONE_CORE_CHOICES,
        default="agx",
        help="tone 核: agx=默认全图 AgX；gated=RAW 门控实验；lum=对照·场景 C1 仅亮度；neutral=诊断·固定 Y 比例曲线",
    )
    parser.add_argument(
        "--lum-norm",
        choices=LUM_NORM_CHOICES,
        default="y",
        help="lum 核 norm: y=Rec.2020 Y；power=power norm；max=max RGB",
    )
    parser.add_argument(
        "--wb",
        choices=WB_CHOICES,
        default="camera",
        help=(
            "白平衡: camera=相机 AsShot（默认）；daylight=相机日光标定（兼容保留）；"
            "固定色温声明: 6500k=D65 显示标准白点，5500k=摄影日光/日光卷，"
            "3400k=Type A 钨丝卷，3200k=Type B 钨丝卷（影棚钨丝灯），"
            "9300k=日本广播电视传统白点。固定色温经文件自身的颜色标定求解"
            "（DNG 双光源插值优先），两种解码器都支持"
        ),
    )
    parser.add_argument(
        "--film-curve",
        choices=FILM_CURVE_CHOICES,
        default="none",
        help=(
            "胶片曲线预设：整条 AgX 曲线锁定到具名胶片坐标（数据手册特性曲线最小二乘解），"
            "场景自适应关闭、整卷一致；portra400=Kodak Portra 400 + Endura 相纸，"
            "superia400=Fujifilm Superia X-TRA 400 + Crystal Archive 相纸。"
            "相纸 Dmax 的阴影地板随预设声明；明暗微调仍可在预设坐标上叠加"
        ),
    )
    parser.add_argument(
        "--color-head-y",
        type=float,
        default=0.0,
        metavar="CC",
        help=(
            "放大机色头黄（Y）滤镜档位，真实暗房单位：CC 密度 0-200、步进 5"
            "（30CC=0.30 光学密度≈相纸蓝敏层 1 档印相曝光衰减）。暗房口诀：成片偏"
            "什么色加什么滤镜——加 Y 去黄。仅负片曲线预设有效（反转片无印相环节）；"
            "每次改档位后按暗房惯例自动重解曝光时间，中灰亮度不变。0=预设的中性印相"
            "决定（逐字节现状）"
        ),
    )
    parser.add_argument(
        "--color-head-m",
        type=float,
        default=0.0,
        metavar="CC",
        help=(
            "放大机色头品（M）滤镜档位，单位与 --color-head-y 相同"
            "（品滤镜衰减相纸绿敏层曝光）：加 M 去品。仅负片曲线预设有效"
        ),
    )
    parser.add_argument(
        "--lens-filter",
        choices=LENS_FILTER_CHOICES,
        default="none",
        help=(
            "镜前转换滤镜（Wratten，按柯达出版的 mired 位移推导）："
            "85b=日光转钨丝(+131)，85=日光转TypeA(+112)，80a=钨丝转日光(-131)，"
            "81a=轻度暖化(+18)，82a=轻度冷化(-21)。作用于 scene-linear、前馈之前；"
            "可靠尾部与 HDR 预算都透过滤镜测量——胶片也是这样看世界的"
        ),
    )
    parser.add_argument(
        "--film",
        choices=FILM_CURVE_CHOICES,
        default="none",
        help=(
            "胶片观察位置组合预设：一次展开三层独立声明（WB 5500k + 对应光谱前馈 + "
            "对应曲线预设）。任何显式给出的单层参数优先于组合展开；没有烘焙，"
            "三层随时可单独调整"
        ),
    )
    parser.add_argument(
        "--support",
        action="store_true",
        help=(
            "只探测不解码：逐档报告此文件在 LibRaw 与 Apple RAW 两条解码线上的"
            "支持程度（格式/颜色标定/RAW 9 版本/传感器先验），然后退出"
        ),
    )
    parser.add_argument(
        "--film-mode",
        choices=("observe", "full"),
        default="observe",
        help=(
            "胶片分工模式（仅在胶片曲线激活时有意义）。observe=胶片声明观察者看见了"
            "什么（WB/分离/音调签名），颜色由 AgX 显影（默认，已验证路径）；"
            "full=胶片显影模型整体接管（离线烘焙光谱链 65³ LUT：观察者逆矩阵→"
            "三层乳剂→特性曲线→印相/幻灯观看链；实验性），AgX 只保留交付端色域"
            "安全。full 仅支持 SDR、仅 AgX tone core、不支持放大机色头"
        ),
    )
    parser.add_argument(
        "--film-crossover",
        choices=("off", "datasheet"),
        default="off",
        help=(
            "胶片层间漂移（crossover）声明开关，仅 --film-mode full 有意义"
            "（其余组合零改变）。off=数字中性化变体（默认）：接管 LUT 的输出按"
            "像素亮度曝光除以随包的有界中性染色曲线，介质灰阶偏中性两档以内严格"
            "中性；datasheet=光谱链原样：中灰由印相求解锚定，暗部/亮部按层间"
            "数据漂移（如 Velvia 阴影温和偏冷）——量级未经外部 oracle 裁决"
        ),
    )
    parser.add_argument(
        "--film-exposure",
        type=float,
        default=0.0,
        metavar="EV",
        help=(
            "film v2:胶片乳剂相对标称 EI 的曝光状态(不是输出曝光),仅 "
            "--film-mode full。负片配 --film-print-timing retimed 时按暗房惯例"
            "重新印相(总体亮度接近不变,颜色/对比/趾肩变化);fixed 保持同一放大机"
            "设置。域 [-2,+2](资产声明,越域报错)"
        ),
    )
    parser.add_argument(
        "--film-print-timing",
        choices=("fixed", "retimed"),
        default="fixed",
        help=(
            "film v2 印相 timing:fixed=沿用 EV0 联合求解的 q(0)(默认);"
            "retimed=随胶片曝光重解 q(E)(因式分解 Stage B,目前试点负片 "
            "portra400/vision3250d;反转片无印相环节,一律拒绝)"
        ),
    )
    parser.add_argument(
        "--demosaic",
        choices=DEMOSAIC_CHOICES,
        default="auto",
        help="仅 LibRaw：解拜耳插值算法。RAW 9 使用 Apple 的 CoreML 解拜耳+降噪模型",
    )
    parser.add_argument(
        "--decoder",
        choices=DECODER_CHOICES,
        default="libraw",
        help="scene-linear RGB 解码器: libraw=默认；coreimage=macOS CIRAWFilter（证据层仍为 LibRaw；与 --wb daylight 不兼容）",
    )
    parser.add_argument(
        "--coreimage-version",
        choices=COREIMAGE_VERSION_CHOICES,
        default="auto",
        help="仅 --decoder coreimage：auto=选文件支持的最高版本(优先9)；显式 9/8/7 在不支持时直接报错",
    )
    parser.add_argument(
        "--coreimage-scale",
        choices=COREIMAGE_SCALE_CHOICES,
        default=None,
        help=(
            "仅 --decoder coreimage：scene-linear 尺度策略。"
            "aligned=逐文件对齐 LibRaw 解码绿色中位（默认，非自动曝光）；"
            "unity=保留 Core Image 原生单位；"
            f"measured=旧版固定倍率 1/{COREIMAGE_SCALE_MEASURED_RATIO:.4f}，仅供复现"
        ),
    )
    parser.add_argument(
        "--output-gamut",
        choices=("srgb", "p3"),
        default="srgb",
        help="JPEG 输出色彩空间: srgb=兼容优先；p3=Display P3 并嵌入 ICC",
    )
    args = parser.parse_args(argv)
    if args.coreimage_scale is not None and args.decoder != "coreimage":
        parser.error(
            f"--coreimage-scale {args.coreimage_scale} 仅作用于 --decoder coreimage，"
            f"当前解码器是 {args.decoder}"
        )
    if args.coreimage_scale is None:
        args.coreimage_scale = COREIMAGE_SCALE_DEFAULT_MODE
    if args.agx_primaries is not None:
        args.agx_primaries = resolve_agx_primaries(args.agx_primaries)
    if args.margin < 0:
        parser.error("--margin must be >= 0")
    # A film combo expands to three independent declarations; a non-default value on
    # any single layer wins over the expansion. Nothing is baked: the expanded values
    # are ordinary per-layer settings the user could have typed.
    if args.film != "none":
        combo = FILM_CURVE_PRESETS.get(args.film, {}).get("combo", {})
        if args.wb == "camera":
            args.wb = str(combo.get("wb", "5500k"))
        combo_st = str(combo.get("scene_transform", "none"))
        if args.scene_transform == "none" and combo_st in SCENE_TRANSFORMS:
            args.scene_transform = combo_st
        if args.film_curve == "none":
            args.film_curve = args.film
        # Editorial style pairing (observe mode's declared look layer): applied only
        # to layers the user did not give — encoded as a None SENTINEL, never as
        # value equality. "The value happens to equal the default" and "the user did
        # not set it" are different intents: the old equality test silently turned an
        # explicit ×1.0 into the pairing's ×1.6, which mislabeled two documentation
        # plates before a diff map caught it. Full mode ignores the pairing: the
        # film development model owns its own character there.
        if args.film_mode == "observe":
            from .film_curve import film_style_pairing

            strength, primaries = film_style_pairing(args.film)
            if args.scene_transform_strength is None:
                args.scene_transform_strength = strength
            if args.agx_primaries is None:
                args.agx_primaries = primaries
    # Sentinel resolution: anything the combo/pairing did not fill falls back to
    # the documented defaults here, in one place, before validation.
    if args.scene_transform_strength is None:
        args.scene_transform_strength = 1.0
    # Full mode's input-domain contract, applied at the SOURCE so the plan,
    # the histograms, auto-EV probes, cache keys, filenames, reports and the
    # pixel render all read the same value (review batch 10).
    from .scene_transform import effective_scene_transform

    args.scene_transform = effective_scene_transform(
        args.scene_transform, args.film_mode, args.film_curve
    )
    if args.agx_primaries is None:
        args.agx_primaries = "base"
    args.agx_primaries = resolve_agx_primaries(args.agx_primaries)
    if args.color_head_y != 0.0 or args.color_head_m != 0.0:
        # Enlarger colour head: validate dial semantics (0-200 CC, detents of 5)
        # and the physical precondition — a print stage only exists for negatives.
        from .film_curve import film_process, validate_color_head_cc

        try:
            args.color_head_y = validate_color_head_cc(args.color_head_y, "--color-head-y")
            args.color_head_m = validate_color_head_cc(args.color_head_m, "--color-head-m")
        except ValueError as exc:
            parser.error(str(exc))
        if args.film_curve == "none":
            parser.error(
                "放大机色头需要负片胶片曲线预设：请先用 --film 或 --film-curve 选择负片"
            )
        if film_process(args.film_curve) != "negative":
            parser.error(
                f"放大机色头仅对负片预设有效：{args.film_curve} 是反转片，"
                "物理上没有印相环节（幻灯片自身就是显示介质）"
            )
    if args.film_mode == "full" and (
        float(args.color_head_y) > 0.0 or float(args.color_head_m) > 0.0
    ):
        parser.error(
            "--film-mode full 暂不支持放大机色头（接管核心是完整烘焙的光谱链）；"
            "请改用 observe 模式或把 --color-head-y/--color-head-m 归零"
        )
    if (
        args.film_mode == "full"
        and args.film_curve != "none"
        and str(args.output_format).startswith("ultrahdr")
    ):
        parser.error(
            "--film-mode full 暂仅支持 SDR：胶片接管显影没有 HDR 对应物"
            "（AgX 的 HDR 色彩机制在该模式下已让位）。请改用 --output-format sdr "
            "或 --film-mode observe"
        )
    if args.jpeg_quality is not None and not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if not 0 <= args.hdr_headroom <= MAX_HDR_HEADROOM_EV + 1e-9:
        parser.error(
            f"--hdr-headroom must be between 0 and {MAX_HDR_HEADROOM_EV:.6f} EV "
            "(4000 nit @ 100 nit reference white)"
        )
    if not 0.0 <= args.grade_strength <= 1.5:
        parser.error("--grade-strength must be between 0 and 1.5")
    if not 0.0 <= args.scene_transform_strength <= 3.0:
        parser.error("--scene-transform-strength must be between 0 and 3")
    if not 0.0 <= args.punch <= 1.5:
        parser.error("--punch must be between 0 and 1.5")
    for _name in (
        "midtone_brightness", "midtone_contrast", "shadow_transition",
        "highlight_transition", "highlight_fade",
    ):
        if not -1.0 <= getattr(args, _name) <= 1.0:
            parser.error(f"--{_name.replace('_', '-')} must be between -1 and 1")
    if not -3.0 <= args.toe_end_offset <= 0.5:
        parser.error("--toe-end-offset must be between -3 and 0.5")
    if not -2.0 <= args.shoulder_white_offset <= 3.0:
        parser.error("--shoulder-white-offset must be between -2 and 3")
    if is_hdr_output_format(args.output_format) and args.grade != "none":
        parser.error(
            "Ultrahdr 第一版不支持 display look/filter；请使用 --grade none"
        )
    if is_hdr_output_format(args.output_format) and abs(float(args.highlight_fade)) > 1e-9:
        parser.error(
            "HDR 尚未定义 SDR 显示侧的高光褪白算子；"
            "请使用 --highlight-fade 0"
        )
    if is_hdr_output_format(args.output_format) and args.tone_core != "agx":
        parser.error("HDR 输出当前只实现 AgX tone core；请使用 --tone-core agx")
    if args.film_mode == "full" and args.tone_core != "agx":
        parser.error(
            "胶片接管显影（--film-mode full）只在 AgX tone core 上运行；"
            "请使用 --tone-core agx 或切回 observe"
        )
    try:
        container = container_for_output_format(args.output_format)
        if args.delivery_profile is None and (
            args.jpeg_quality is not None or args.chroma is not None
        ):
            # Explicit encode knobs without a named profile keep working as before the
            # profiles existed: honour them, and infer which engineering gates apply.
            # Missing knobs fill from the historical CLI defaults (q100 / 4:4:4).
            args.delivery = profile_from_encode_settings(
                ARCHIVE_JPEG_QUALITY if args.jpeg_quality is None else args.jpeg_quality,
                ARCHIVE_CHROMA if args.chroma is None else args.chroma,
                container=container,
            )
        else:
            args.delivery = resolve_delivery_profile(
                args.delivery_profile or DEFAULT_DELIVERY_PROFILE,
                quality=args.jpeg_quality,
                chroma=args.chroma,
                container=container,
            )
    except ValueError as exc:
        parser.error(str(exc))
    if is_hdr_output_format(args.output_format) and args.chroma is not None:
        # The HDR container's primary-image subsampling is emergent from quality inside
        # Core Image; an explicit --chroma the encoder cannot honour must fail loudly
        # instead of writing a file that contradicts the request.
        if args.chroma == "422":
            parser.error(
                "HDR gain-map 容器不提供 4:2:2 主图采样；"
                "Core Image 按 quality 决定采样（q100→4:4:4，share→通常 4:2:0）"
            )
        if args.chroma == "444" and not args.delivery.is_archive:
            parser.error(
                "HDR 容器的 4:4:4 只在 q100（archive 档）下产生并被门禁验证；"
                "请用 --delivery-profile archive（或去掉 --chroma）"
            )
        if args.chroma == "420" and args.delivery.is_archive:
            parser.error(
                "archive 档（q100）的 HDR 主图为 4:4:4；"
                "要 4:2:0 请用 --delivery-profile share"
            )
    args.delivery_profile = str(args.delivery.name)
    args.jpeg_quality = int(args.delivery.quality)
    args.chroma = str(args.delivery.chroma)
    if args.decoder == "coreimage" and args.tone_core == "gated":
        # gated is defined as "RAW evidence gates the colour path"; the Core Image
        # pipeline has no per-pixel CFA evidence, so the combination is meaningless
        # rather than merely degraded.
        parser.error(
            "--tone-core gated 需要逐像素 CFA 证据，而 --decoder coreimage 是独立管线"
            "（Core Image 执行 DNG opcode，几何与 LibRaw 不可对齐）。"
            "请改用 --tone-core agx/lum/neutral，或改回 --decoder libraw"
        )
    if args.decoder == "coreimage":
        # CIRAWFilter exposes one calibrated reconstruction path, not LibRaw's three
        # highlight policies. Keep cache keys and reports honest about what was run.
        args.highlight_mode = "reconstruct"
        args.demosaic = "auto"
    return args


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        if not args.path.exists():
            raise FileNotFoundError(f"Input file does not exist: {args.path}")
        if not args.path.is_file():
            raise FileNotFoundError(f"Input path is not a file: {args.path}")
        require_dependencies()
        if args.support:
            from .decode_support import probe_decode_support

            for line in probe_decode_support(args.path)["lines"]:
                print(line)
            return 0
        if is_hdr_output_format(args.output_format):
            from .gainmap import apple_gainmap_backend_status

            available, reason = apple_gainmap_backend_status()
            if not available:
                raise RuntimeError(reason)
        if args.decoder == "coreimage":
            from . import coreimage_decode

            probe = coreimage_decode.probe_raw9_support(args.path)
            if not probe["coreimage_available"]:
                raise RuntimeError("Apple Core Image RAW 解码器在此系统不可用")
            if probe["error"]:
                raise RuntimeError(f"Apple RAW 无法探测这个文件：{probe['error']}")
            if not probe["raw9_supported"]:
                fallback = probe["fallback_version"]
                offered = ", ".join(str(value) for value in probe["versions_offered"]) or "none"
                if args.coreimage_version == "9":
                    raise RuntimeError(
                        f"此文件不支持 Apple RAW 9（系统报告版本：{offered}）；"
                        "请改用 --decoder libraw，或显式选择可用的 --coreimage-version"
                    )
                if args.coreimage_version == "auto":
                    if fallback is None:
                        raise RuntimeError(
                            f"此文件不支持 Apple RAW 9，且没有 RAW 8/7 降级路径"
                            f"（系统报告版本：{offered}）"
                        )
                    print(
                        f"warning: 此文件不支持 Apple RAW 9；将明确降级到 Apple RAW {fallback}。"
                        f"可用 --coreimage-version 9 禁止降级，或改用 --decoder libraw。",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"warning: 此文件不支持 Apple RAW 9；当前显式使用 Apple RAW "
                        f"{args.coreimage_version}。",
                        file=sys.stderr,
                    )
        scan_requested = bool(args.scan or args.out is not None or (args.jpeg is None and args.csv is None))
        out_path = args.out if args.out is not None else (default_png_path(args.path) if scan_requested else None)

        bundle = load_raw(
            args.path,
            args.highlight_mode,
            demosaic=args.demosaic,
            wb_mode=args.wb,
            decoder=args.decoder,
            coreimage_version=args.coreimage_version,
            coreimage_scale=args.coreimage_scale,
        )
        # Render intent, not capture data: the declared filter rides the bundle so the
        # tail, HDR budget and every formation see the scene through the glass.
        bundle.lens_filter = validate_lens_filter(args.lens_filter)
        diagnostics_requested = bool(scan_requested or args.csv is not None)
        analysis, y, ev = analyze(
            bundle,
            args.margin,
            diagnostics=diagnostics_requested,
            gamut_names=None
            if diagnostics_requested
            else (output_gamut_space("p3" if is_hdr_output_format(args.output_format) else args.output_gamut),),
        )
        look, look_strength, display_filter, filter_strength = resolve_grade(
            args.grade, args.grade_strength
        )

        ev_input = parse_ev_value(args.ev)
        cli_adjustments = RenderAdjustments(
            midtone_brightness=args.midtone_brightness,
            midtone_contrast=args.midtone_contrast,
            shadow_transition=args.shadow_transition,
            highlight_transition=args.highlight_transition,
            highlight_fade=args.highlight_fade,
            toe_end_offset=args.toe_end_offset,
            shoulder_white_offset=args.shoulder_white_offset,
        )
        auto_ev_result: AutoEvResult | None = None
        jpeg_output_gamut = "p3" if is_hdr_output_format(args.output_format) else args.output_gamut
        if is_ev_auto(ev_input):
            if args.jpeg is None and not scan_requested:
                raise ValueError("--ev auto 需要同时导出 JPEG（--jpeg）或诊断图（--scan / --out）")
            resolved_ev, auto_ev_result = resolve_export_ev(
                ev_input,
                bundle,
                analysis,
                jpeg_output_gamut,
                look,
                look_strength,
                display_filter,
                filter_strength,
                args.scene_transform,
                args.scene_transform_strength,
                args.punch,
                args.tone_core,
                args.lum_norm,
                args.agx_primaries,
                # The declared lens filter already rides the bundle (set above); the
                # curve-shaping choices — manual tone adjustments and the full film
                # declaration included — must reach the reference plan explicitly.
                adjustments=cli_adjustments,
                endpoint_mode=args.endpoint_mode,
                film_curve=args.film_curve,
                film_mode=args.film_mode,
                film_crossover=args.film_crossover,
                film_exposure_ev=args.film_exposure,
                film_print_timing=args.film_print_timing,
                color_head_y=args.color_head_y,
                color_head_m=args.color_head_m,
            )
        else:
            resolved_ev = float(ev_input)

        bundle = with_intent_exposure(
            bundle, user_ev=resolved_ev, tone_core=args.tone_core
        )
        if out_path is not None:
            plot_dashboard(bundle, analysis, y, ev, out_path, auto_ev=auto_ev_result)

        jpeg_path = args.jpeg
        if jpeg_path is not None and args.output_format == "ultrahdr-heic":
            if jpeg_path.suffix.lower() in {".jpg", ".jpeg", ""}:
                jpeg_path = jpeg_path.with_suffix(".heic")
        jpeg_icc_embedded = False
        render_plan = (
            build_render_plan(
                bundle,
                analysis,
                RENDER_MODE,
                jpeg_output_gamut,
                args.scene_transform,
                args.scene_transform_strength,
                args.punch,
                args.tone_core,
                args.lum_norm,
                agx_primaries=args.agx_primaries,
                film_curve=args.film_curve,
                film_mode=args.film_mode,
                film_crossover=args.film_crossover,
                film_exposure_ev=args.film_exposure,
                film_print_timing=args.film_print_timing,
                endpoint_mode=args.endpoint_mode,
                color_head_y=args.color_head_y,
                color_head_m=args.color_head_m,
            )
            if jpeg_path is not None
            else None
        )
        if render_plan is not None:
            render_plan = apply_render_adjustments(render_plan, cli_adjustments)
        if jpeg_path is not None:
            export_result = export_jpeg(
                path=args.path,
                out_path=jpeg_path,
                quality=args.jpeg_quality,
                bundle=bundle,
                analysis=analysis,
                tone_plan=render_plan,
                output_gamut=jpeg_output_gamut,
                output_format=args.output_format,
                hdr_headroom=args.hdr_headroom,
                hdr_drt=args.hdr_drt,
                subsampling=chroma_to_subsampling(args.chroma),
                look=look,
                look_strength=look_strength,
                display_filter=display_filter,
                filter_strength=filter_strength,
                scene_transform=args.scene_transform,
                scene_transform_strength=args.scene_transform_strength,
                tone_core=args.tone_core,
                lum_norm=args.lum_norm,
                agx_primaries=args.agx_primaries,
                punch_scale=args.punch,
                delivery=args.delivery,
                chroma=args.chroma,
            )
            jpeg_icc_embedded = (
                str(export_result.get("profile", "")) == "Display P3"
                if isinstance(export_result, dict)
                else bool(export_result)
            )

        if args.csv is not None:
            # Only built on demand: without --scan/--csv the analysis deliberately
            # computes a gamut subset, which the full CSV schema must not read.
            row = csv_row(
                bundle,
                analysis,
                out_path,
                jpeg_path,
                args.jpeg_quality if jpeg_path is not None else None,
                RENDER_MODE if jpeg_path is not None else "",
                jpeg_icc_embedded,
                resolved_ev,
                render_plan.tone if render_plan is not None else None,
                jpeg_output_gamut,
                auto_ev_result,
                args.grade,
                args.grade_strength,
                args.scene_transform,
                args.scene_transform_strength,
            )
            write_csv(args.csv, row)
        print_report(
            bundle,
            analysis,
            out_path,
            args.csv,
            jpeg_path,
            args.jpeg_quality,
            RENDER_MODE if jpeg_path is not None else "",
            jpeg_icc_embedded,
            resolved_ev,
            render_plan.tone if render_plan is not None else None,
            jpeg_output_gamut,
            auto_ev_result,
            args.grade,
            args.grade_strength,
            args.scene_transform,
            args.scene_transform_strength,
        )
        if jpeg_path is not None and is_hdr_output_format(args.output_format):
            container = "HEIC" if args.output_format == "ultrahdr-heic" else "JPEG"
            print(
                f"{container} HDR: Apple Core Image ISO 21496-1；Display P3 SDR 底图；"
                f"darktable 式 HDR AgX RGB gain map；capacity=+{args.hdr_headroom:.2f}EV；"
                f"delivery={args.delivery_profile}"
            )
        return 0
    except Exception as exc:
        maybe_print_exc()
        print(f"error: {exc}", file=sys.stderr)
        return 1
