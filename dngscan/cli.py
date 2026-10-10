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
    resolve_hdr_chroma,
)
from .export import chroma_to_subsampling, export_jpeg
from .lens_filter import LENS_FILTER_CHOICES, validate_lens_filter
from .grade import RENDER_MODE, grade_choices, resolve_grade
from .plot import default_png_path, plot_dashboard
from .raw_io import load_raw, release_analysis_buffers
from .report import csv_row, print_report, write_csv
from .scene_transform import SCENE_TRANSFORM_CHOICES
from .models import RenderAdjustments
from .scene_scale import with_intent_exposure
from .tone import (
    ENDPOINT_MODE_CHOICES, LUM_NORM_CHOICES, TONE_CORE_CHOICES,
    apply_render_adjustments, build_render_plan,
)


def require_dependencies(*, dashboard: bool = False) -> None:
    """Core deps (numpy, rawpy) gate every run; matplotlib gates only the
    diagnostic dashboard (A8 item 8: it is an extra, and a plain
    conversion must not fail without it)."""
    from ._deps import DASHBOARD_IMPORT_ERRORS

    errors = list(IMPORT_ERRORS)
    if dashboard:
        errors += DASHBOARD_IMPORT_ERRORS
    if errors:
        joined = "\n  ".join(errors)
        raise RuntimeError(
            "Missing or broken dependency. Install the required packages "
            "(numpy, rawpy; matplotlib only for the dashboard) and rerun."
            "\n  " + joined
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgX RAW/DNG → JPEG；可选六面板诊断 PNG。",
        epilog="实测标定管理：python -m dngscan calibration --help（导入 JPTC JSON 或 Collect 目录）。",
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
    parser.add_argument(
        "--report",
        action="store_true",
        help="打印完整分析报告（证据、曲线端点、色彩矩阵健康度等）；"
        "默认只打印写出的文件。--scan/--csv 的诊断运行自动附带报告",
    )
    parser.add_argument("--out", type=Path, default=None, help="诊断 PNG 输出路径；设置后隐含 --scan")
    parser.add_argument("--csv", type=Path, default=None, help="可选指标 CSV 路径")
    parser.add_argument(
        "--jpeg",
        type=Path,
        default=None,
        help="输出路径（JPEG 或 HEIF，由 --output-format 选择）",
    )
    parser.add_argument("--heif-encoder", choices=("auto","apple","x265"), default="auto",
                        help="HEIF 编码器；auto 优先采用可调 libheif/x265")
    parser.add_argument("--heif-bit-depth", type=int, choices=(8,10), default=10)
    parser.add_argument("--heif-preset", choices=("fast","medium","slow","slower"), default="slow")
    parser.add_argument("--heif-tune", choices=("ssim","psnr","grain"), default="ssim")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="JPEG 质量 1-100；默认跟随 --delivery-profile（auto=95–99 自动选择，archive=100，share=95，share-hq=97）",
    )
    parser.add_argument(
        "--chroma",
        choices=CHROMA_CHOICES,
        default=None,
        help=(
            "色度采样: 444/422/420；默认跟随 --delivery-profile（archive=444，share=420）。"
            "JPEG 与可调 HEIF 主图采样独立于 quality；实际文件必须通过回读。"
        ),
    )
    parser.add_argument(
        "--delivery-profile",
        choices=DELIVERY_PROFILE_CHOICES,
        default=None,
        help=(
            "交付编码档: archive=q100/4:4:4 严格 round-trip；"
            "auto=JPEG q95–99 / HEIF 独立刻度，按回读误差选择；share=手动，默认q95/4:2:0。"
            "share-hq=仅 JPEG，固定q97/4:2:0、原尺寸，超过20 MB仅提示。"
            "缺省时：未显式给 --jpeg-quality/--chroma 则为 auto；"
            "给了则按参数值推断门禁档（恰好 q100 且 444 走 archive——严格档合同"
            "只在其标定过的编码点成立，其余组合走 share）。"
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=JPEG_OUTPUT_FORMATS,
        default="sdr",
        help=(
            "输出格式: sdr=SDR JPEG；sdr-heic=SDR HEIF；"
            "ultrahdr=ISO gain-map JPEG；ultrahdr-heic=ISO gain-map HEIF"
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
    # HDR 高级选项（taste-to-dial,2026-08-14 owner 决策）:默认 auto 保持
    # 数学化政策默认逐位不变;显式值覆盖 normal 与 sparse-emitter 两档。证据
    # 门控(多通道剪切/尾部SNR/色域压力/解码器cap)是测量逻辑,不开旋钮。
    parser.add_argument(
        "--hdr-rho",
        default="auto",
        metavar="R|auto",
        help=(
            "HDR 高光逐通道高光保色基准 [0,1](仅 HDR 格式):证据置信满格时"
            "允许的 rho;auto(默认)=政策值 0.5。证据门控仍然乘算,不受此旋钮绕过"
        ),
    )
    parser.add_argument(
        "--hdr-white-margin",
        default="auto",
        metavar="EV|auto",
        help=(
            "HDR 白端点在可靠尾部之上的余量 EV [0,2](仅 HDR 格式):auto(默认)="
            "普通 0.30 / 稀疏光源 0.50;显式值同时覆盖两档"
        ),
    )
    parser.add_argument(
        "--hdr-shoulder-start",
        default="auto",
        metavar="EV|auto",
        help=(
            "HDR 肩部离开 SDR 主体的起点 EV [-1,3](仅 HDR 格式):auto(默认)="
            "普通 0.20 / 稀疏光源 0.00;显式值同时覆盖两档"
        ),
    )
    parser.add_argument(
        "--hdr-debug-dir",
        type=Path,
        default=None,
        help="可选：写出 HDR 诊断中间结果目录",
    )
    parser.add_argument(
        "--ev",
        default="auto",
        help="默认 auto：按可靠场景统计计算曝光并保护高光；指定数字可改为手动曝光补偿",
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
        default=None,
        help="AgX 前 scene-linear Rec.2020 前馈变换；none=关闭，arri_skin_d55=demo ARRI 式肤色前馈",
    )
    parser.add_argument(
        "--scene-transform-strength",
        type=float,
        default=None,
        help=(
            "scene transform 强度 0-3（默认 1.0；0=关闭效果；>1 用于诊断/强化 A/B）。"
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
        ("highlight-fade", "高光褪色 -1..1（显示端高光降饱和）"),
    ):
        parser.add_argument(f"--{_flag}", type=float, default=0.0, help=_help)
    parser.add_argument(
        "--endpoint-mode",
        choices=ENDPOINT_MODE_CHOICES,
        default="adaptive",
        help=(
            "曲线端点策略：adaptive=场景百分位自适应（默认，现状）；"
            "evidence=黑端点使用独立噪声模型的读出噪声底 EV，"
            "白端点只信可靠 RAW 尾部（保留最低白点地板；"
            "证据缺席时如实回退自适应并注记）。pivot 锚定不变（0EV→18%%）"
        ),
    )
    parser.add_argument(
        "--toe-end-offset",
        type=float,
        default=0.0,
        help=(
            "暗部收黑点 EV 偏移 -3..+0.5（0=现状）。负值把曲线落到近黑的 EV 下移，"
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
            "高光收白点 EV 偏移 -2..+3（0=现状）。控制曲线升到近白参考"
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
            "仅 tone-core=agx 的 AgX 色彩浓淡：base=固定版本 darktable scene 默认；"
            "smooth=darktable smooth；punchy/muted=纯度变化参考。默认 base"
        ),
    )
    parser.add_argument(
        "--tone-core",
        choices=TONE_CORE_CHOICES,
        default="agx",
        help="tone 核: agx=默认全图 AgX；gated=RAW 门控·保真(逐像素 CFA 证据门控色彩路径)；lum=对照·场景 C1 仅亮度；neutral=诊断·固定 Y 比例曲线",
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
        default=None,
        help=(
            "白平衡: camera=相机 AsShot（默认）；daylight=相机日光标定（兼容保留）；"
            "固定色温声明: 6500k=D65 显示标准白点，5500k=摄影日光/日光卷，"
            "3400k=Type A 钨丝卷，3200k=Type B 钨丝卷（影棚钨丝灯），"
            "9300k=日本广播电视传统白点。固定色温经文件自身的颜色标定求解"
            "（DNG 双光源插值优先），两种解码器都支持"
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
            "可靠尾部与 HDR 预算都透过滤镜测量"
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
        "--chroma-nr",
        type=float,
        default=0.0,
        metavar="0..1",
        help=(
            "基于独立噪声标定的可选色度平滑，0=关闭；在约 8–128 传感器像素频带"
            "按传播后的噪声方差收缩细节。需要有效标定与解码传播，缺失时跳过并报告；"
            "支持 SDR/HDR，强度越高真实颜色细节损失风险越大"
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
        help="scene-linear RGB 解码器: libraw=默认；coreimage=macOS CIRAWFilter（证据层仍为 LibRaw；--wb daylight 经固定 AsShot 解码后的项目 hot-WB 实现）",
    )
    parser.add_argument(
        "--coreimage-version",
        choices=COREIMAGE_VERSION_CHOICES,
        default="auto",
        help="仅 --decoder coreimage：auto=优先9，渲染失败按文件支持版本回退；显式 9/8/7/6 在不支持时直接报错",
    )
    parser.add_argument(
        "--coreimage-scale",
        choices=COREIMAGE_SCALE_CHOICES,
        default=None,
        help=(
            "仅 --decoder coreimage：scene-linear 尺度策略。"
            "aligned=逐文件对齐 LibRaw 解码绿色中位（默认，非自动曝光）；"
            "unity=保留 Core Image Apple 原始数值；"
            f"measured=旧版固定倍率 1/{COREIMAGE_SCALE_MEASURED_RATIO:.4f}，仅供复现"
        ),
    )
    parser.add_argument(
        "--output-gamut",
        choices=("srgb", "p3"),
        default=None,
        help=(
            "JPEG 输出色彩空间: srgb=兼容优先；p3=Display P3 并嵌入 ICC。"
            "缺省时:SDR 为 srgb,HDR 容器为 p3(合同固定)。HDR 下显式指定 "
            "srgb 会报错而不是被静默改写"
        ),
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
    if args.wb is None:
        args.wb = "camera"
    if args.scene_transform is None:
        args.scene_transform = "none"
    if args.scene_transform_strength is None:
        args.scene_transform_strength = 1.0

    if args.agx_primaries is None:
        args.agx_primaries = "base"
    args.agx_primaries = resolve_agx_primaries(args.agx_primaries)
    if args.jpeg_quality is not None and not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if not 0 <= args.hdr_headroom <= MAX_HDR_HEADROOM_EV + 1e-9:
        parser.error(
            f"--hdr-headroom must be between 0 and {MAX_HDR_HEADROOM_EV:.6f} EV "
            "(4000 nit @ 100 nit reference white)"
        )
    # HDR latitude dials: "auto" -> None (policy defaults, byte-identical);
    # explicit values are range-checked here and rejected outright when the
    # output format has no HDR leg — a dial that silently does nothing
    # teaches the user it is broken (same contract as the sibling HDR flags).
    for _flag, _attr, _lo, _hi in (
        ("--hdr-rho", "hdr_rho", 0.0, 1.0),
        ("--hdr-white-margin", "hdr_white_margin", 0.0, 2.0),
        ("--hdr-shoulder-start", "hdr_shoulder_start", -1.0, 3.0),
    ):
        _raw = str(getattr(args, _attr))
        if _raw == "auto":
            setattr(args, _attr, None)
            continue
        try:
            _val = float(_raw)
        except ValueError:
            parser.error(f"{_flag} 需要数值或 auto")
        if not (_lo <= _val <= _hi):
            parser.error(f"{_flag} 域为 [{_lo}, {_hi}]")
        if not is_hdr_output_format(args.output_format):
            parser.error(
                f"{_flag} 属于 HDR 编码(ultrahdr/ultrahdr-heic);"
                "SDR 输出没有 HDR 肩部/色度模型"
            )
        setattr(args, _attr, _val)
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
    if is_hdr_output_format(args.output_format):
        try:
            args.delivery = resolve_hdr_chroma(args.delivery, explicit_chroma=args.chroma)
        except ValueError as exc:
            parser.error(str(exc))
    from dataclasses import replace
    args.delivery = replace(args.delivery, heif_encoder=args.heif_encoder,
                            heif_bit_depth=args.heif_bit_depth, heif_preset=args.heif_preset,
                            heif_tune=args.heif_tune)
    args.delivery_profile = str(args.delivery.name)
    args.jpeg_quality = int(args.delivery.quality)
    args.chroma = str(args.delivery.chroma)
    # R4: the HDR base-image gamut is a container contract (Display P3), and
    # an explicit srgb used to be silently coerced — unlike every sibling
    # contradiction, which fails loudly. The None default makes an explicit
    # request distinguishable at last.
    if is_hdr_output_format(args.output_format):
        if args.output_gamut == "srgb":
            parser.error(
                "HDR gain-map 容器的底图合同固定为 Display P3；"
                "--output-gamut srgb 与 HDR 输出互斥,请去掉该参数或改用 "
                "--output-format sdr"
            )
        args.output_gamut = "p3"
    elif args.output_gamut is None:
        args.output_gamut = "srgb"
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


NEUTRALIZATION_TO_CROSSOVER = {
    "technical-neutral": "off", "bounded": "off",
    "print-balanced": "print",
    "native": "datasheet", "datasheet": "datasheet",
}


def calibration_main(argv: list[str]) -> int:
    """Manage reusable user measurements without decoding a photograph."""
    import json

    from . import calibration

    parser = argparse.ArgumentParser(
        prog="dngscan calibration",
        description="导入并管理 JPTC 实测标定；匹配相机、RAW 读出模式和 ISO 后自动用于成像分析。",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("import", help="导入 JPTC JSON 或 Collect CSV 目录")
    install.add_argument("path", type=Path)
    install.add_argument("--inactive", action="store_true", help="导入后先停用")
    install.add_argument("--shutter-mode", choices=("any", "electronic", "mechanical", "efcs"),
                         default=None, help="明确覆盖测量适用的读出模式；any 声明适用全部模式，默认遵循文件")
    commands.add_parser("list", help="列出已安装标定及有效 ISO 范围")
    for command, description in (("remove", "删除标定"), ("enable", "启用标定"),
                                 ("disable", "停用标定")):
        subparser = commands.add_parser(command, help=description)
        subparser.add_argument("id", help="list 输出的标定 ID")
    args = parser.parse_args(argv)
    if args.command == "import":
        result = calibration.import_calibration(args.path.expanduser(), active=not args.inactive,
                                                shutter_override=args.shutter_mode)
    elif args.command == "list":
        result = calibration.list_calibrations()
    elif args.command == "remove":
        result = calibration.remove_calibration(args.id)
    else:
        result = calibration.set_calibration_active(args.id, args.command == "enable")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str]) -> int:
    try:
        if argv and argv[0] == "calibration":
            return calibration_main(argv[1:])
        args = parse_args(argv)
        if not args.path.exists():
            raise FileNotFoundError(f"Input file does not exist: {args.path}")
        if not args.path.is_file():
            raise FileNotFoundError(f"Input path is not a file: {args.path}")
        if args.support:
            # R4: the support probe never plots — it must not demand
            # matplotlib through the "no --jpeg/--csv means dashboard"
            # default below (its own core deps still gate).
            require_dependencies()
            from .decode_support import probe_decode_support

            for line in probe_decode_support(args.path)["lines"]:
                print(line)
            return 0
        # Mirrors scan_requested below: the six-panel dashboard also runs
        # by default when neither --jpeg nor --csv was asked for.
        require_dependencies(
            dashboard=bool(
                args.scan or args.out is not None
                or (args.jpeg is None and args.csv is None)
            )
        )
        if is_hdr_output_format(args.output_format):
            from .gainmap import apple_gainmap_backend_status

            available, reason = apple_gainmap_backend_status()
            if not available:
                raise RuntimeError(reason)
        if args.decoder == "coreimage" and args.coreimage_version != "auto":
            from . import coreimage_decode

            probe = coreimage_decode.probe_raw9_support(args.path)
            if not probe["coreimage_available"]:
                raise RuntimeError("Apple Core Image RAW 解码器在此系统不可用")
            if probe["error"]:
                raise RuntimeError(f"Apple RAW 无法探测这个文件：{probe['error']}")
            if not probe["raw9_supported"]:
                offered = ", ".join(str(value) for value in probe["versions_offered"]) or "none"
                if args.coreimage_version == "9":
                    raise RuntimeError(
                        f"此文件不支持 Apple RAW 9（系统报告版本：{offered}）；"
                        "请改用 --decoder libraw，或显式选择可用的 --coreimage-version"
                    )
                coreimage_decode.resolve_decoder_version(args.coreimage_version,
                    tuple(probe["versions_offered"]))
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
            _defer_clip_masks=True,
            _analysis_luminance_only=not scan_requested,
        )
        # Render intent, not capture data: the declared filter rides the bundle so the
        # tail, HDR budget and every formation see the scene through the glass.
        bundle.lens_filter = validate_lens_filter(args.lens_filter)
        diagnostics_requested = bool(scan_requested or args.csv is not None)
        analysis, y, ev = analyze(
            bundle,
            args.margin,
            diagnostics=diagnostics_requested,
            _return_planes=scan_requested,
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
        auto_ev_plan: list = []
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
                # declaration included — must reach the reference plan explicitly.
                adjustments=cli_adjustments,
                endpoint_mode=args.endpoint_mode,
                chroma_nr=args.chroma_nr,
                _plan_sink=auto_ev_plan,
            )
        else:
            resolved_ev = float(ev_input)

        bundle = with_intent_exposure(
            bundle, user_ev=resolved_ev, tone_core=args.tone_core
        )
        if out_path is not None:
            plot_dashboard(bundle, analysis, y, ev, out_path, auto_ev=auto_ev_result)

        jpeg_path = args.jpeg
        if jpeg_path is not None and container_for_output_format(args.output_format) == "heic":
            if jpeg_path.suffix.lower() in {".jpg", ".jpeg", ""}:
                jpeg_path = jpeg_path.with_suffix(".heic")
        jpeg_icc_embedded = False
        export_result = None
        render_plan = (
            auto_ev_plan[0] if auto_ev_plan and jpeg_path is not None else
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
                chroma_nr=args.chroma_nr,
                endpoint_mode=args.endpoint_mode,
            )
            if jpeg_path is not None
            else None
        )
        if render_plan is not None and not auto_ev_plan:
            render_plan = apply_render_adjustments(render_plan, cli_adjustments)
        if jpeg_path is not None:
            # Staged ownership (scheduler plan S4): analysis and the optional
            # dashboard are done, so the XYZ analysis buffer they owned is
            # released for the export stage to reuse.
            bundle = release_analysis_buffers(bundle)
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
                hdr_rho=args.hdr_rho,
                hdr_white_margin=args.hdr_white_margin,
                hdr_shoulder_start=args.hdr_shoulder_start,
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
                bool(export_result.get("icc_embedded", str(export_result.get("profile", "")) == "Display P3"))
                if isinstance(export_result, dict)
                else bool(export_result)
            )
            # R4: the HDR exporter may rewrite the suffix to match the actual
            # container (any non-heic -> .heic for ultrahdr-heic, .heic ->
            # .jpg for ultrahdr) beyond the narrow rewrite above. The report
            # and CSV must name the file that EXISTS, not the one requested.
            if isinstance(export_result, dict) and export_result.get("output_path"):
                jpeg_path = Path(str(export_result["output_path"]))
            if isinstance(export_result, dict) and export_result.get("delivery_quality"):
                args.jpeg_quality = int(export_result["delivery_quality"])
                if export_result.get("delivery_profile") == "auto":
                    args.chroma = str(export_result["delivery_chroma_requested"])
                    print(f"自动编码: q{args.jpeg_quality}/{export_result['chroma_subsampling']}，"
                          f"{export_result['file_size_bytes']/1048576:.2f} MiB，"
                          f"较 q{export_result['auto_reference_quality']} 参考节省 {export_result['auto_saved_pct']:.1f}%")
                elif export_result.get("delivery_profile") == "share-hq":
                    print(f"高质量分享: q97/4:2:0，{export_result['file_size_bytes']/1_000_000:.2f} MB")
                if export_result.get("size_warning"):
                    print(f"warning: {export_result['size_warning']}", file=sys.stderr)
            if args.chroma_nr > 0:
                print("色度降噪: " + getattr(bundle, "chroma_nr_status", "skipped")
                      + "；" + (getattr(bundle, "chroma_nr_reason", None) or "处理状态未提供"))

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
                chroma=args.chroma if jpeg_path is not None else "444",
                export_info=export_result if isinstance(export_result, dict) else None,
                scene=render_plan.scene if render_plan is not None else None,
            )
            write_csv(args.csv, row)
        # The report is a standalone deliverable (owner 2026-08-27): a plain
        # conversion prints the files it wrote and nothing else; the full
        # analysis report comes with --report or with a diagnostic run
        # (--scan / --csv), where it documents what the PNG/CSV contain.
        if args.report or diagnostics_requested:
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
                chroma=args.chroma if jpeg_path is not None else "444",
                export_info=export_result if isinstance(export_result, dict) else None,
                scene=render_plan.scene if render_plan is not None else None,
            )
        else:
            output_label = "HEIF 图像" if container_for_output_format(args.output_format) == "heic" else "JPEG 图像"
            for label, path in ((output_label, jpeg_path), ("PNG 图像", out_path)):
                if path is not None:
                    print(f"{label}: {path}")
        if jpeg_path is not None and is_hdr_output_format(args.output_format):
            container = "HEIC" if args.output_format == "ultrahdr-heic" else "JPEG"
            leg = "darktable 式 HDR AgX RGB gain map"
            print(
                f"{container} HDR: Apple Core Image ISO 21496-1；Display P3 SDR 底图；"
                f"{leg}；capacity=+{args.hdr_headroom:.2f}EV；"
                f"delivery={args.delivery_profile}"
            )
        return 0
    except Exception as exc:
        maybe_print_exc()
        print(f"error: {exc}", file=sys.stderr)
        return 1
