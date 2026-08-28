# 文档索引 / Documentation Index

按用途分类的全部项目文档。计划书均标注落地状态;基线与记录类文档是历史依据,
不随代码更新。

## 入门与使用 / Getting Started

GUI 上的描述性文案只有最基础的一句;每个控件的来龙去脉、可核对的数字与
判断方法都放在下面的教程和使用说明里。

| 文档 | 内容 |
|---|---|
| [EDITING_TUTORIAL.zh-CN.md](EDITING_TUTORIAL.zh-CN.md) | 修图教程:从导入到导出的完整流程,逐个控件讲用法,含 RAW 满阱层的读法 |
| [FILM_TUTORIAL.zh-CN.md](FILM_TUTORIAL.zh-CN.md) | 胶片教程:每个胶片滑条与选择的作用,配实测样张 |
| [USER_GUIDE.md](USER_GUIDE.md) / [USER_GUIDE.zh-CN.md](USER_GUIDE.zh-CN.md) | 使用说明:支持的相机、界面字段、RAW 满阱显示与选项置灰规则、导出选择、latitude 旋钮 |
| [SENSOR_SUPPORT.zh-CN.md](SENSOR_SUPPORT.zh-CN.md) | 机型支持:传感器数据、降级策略与 LibRaw 升级路线 |

## 架构 / Architecture

| 文档 | 内容 |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) / [ARCHITECTURE.zh-CN.md](ARCHITECTURE.zh-CN.md) | 技术架构:解码、证据层、tone 管线、双解码器边界 |
| [PRODUCT_ARCHITECTURE.md](PRODUCT_ARCHITECTURE.md) / [PRODUCT_ARCHITECTURE.zh-CN.md](PRODUCT_ARCHITECTURE.zh-CN.md) | 产品架构:模块职责与扩展边界 |
| [ENGINEERING_NOTES.zh-CN.md](ENGINEERING_NOTES.zh-CN.md) | 工程笔记:跨模块的实现约定与教训 |

## 计划书(合同 + 实施记录)/ Plans

均为"先立合同,批准后分批实施"的原始合同文本,文首标注落地状态,
实施记录以引用块追加在对应章节。

| 文档 | 状态 |
|---|---|
| [FILM_PRINT_RENDERING_PLAN.zh-CN.md](FILM_PRINT_RENDERING_PLAN.zh-CN.md) | 已落地(film v2 P0–P7) |
| [FILM_APPEARANCE_RECIPE_PLAN.zh-CN.md](FILM_APPEARANCE_RECIPE_PLAN.zh-CN.md) | 已落地(外观层;配方覆盖仍窄) |
| [FILM_OPTICS_V2_PLAN.zh-CN.md](FILM_OPTICS_V2_PLAN.zh-CN.md) | 已落地(光学 V2 P0–P5 + R1 整改;§11.1 已闭账) |
| [FILM_OBSERVATION_PLAN.zh-CN.md](FILM_OBSERVATION_PLAN.zh-CN.md) | 已落地(observe 模式) |
| [INTERIMAGE_LITERATURE.zh-CN.md](INTERIMAGE_LITERATURE.zh-CN.md) | 层间效应文献锚:专利定量转录、与 β 表对照分析、等效 IIE% 复现路线 |
| [HDR_AGX_V2_IMPLEMENTATION_PLAN.zh-CN.md](HDR_AGX_V2_IMPLEMENTATION_PLAN.zh-CN.md) | 已落地(HDR tone/color v2) |
| [RENDER_SCHEDULER_PLAN.zh-CN.md](RENDER_SCHEDULER_PLAN.zh-CN.md) | 已落地(S1–S4) |
| [HOT_WHITE_BALANCE_MIGRATION.zh-CN.md](HOT_WHITE_BALANCE_MIGRATION.zh-CN.md) | 已落地(固定 Kelvin 热 WB) |
| [REALTIME_PREVIEW_PLAN.zh-CN.md](REALTIME_PREVIEW_PLAN.zh-CN.md) | 已落地(实时预览;profile 数字为历史记录) |
| [PIPELINE_PERFORMANCE_EQUIVALENCE_PLAN.zh-CN.md](PIPELINE_PERFORMANCE_EQUIVALENCE_PLAN.zh-CN.md) | 部分落地(两处数值缺口已修;reference 执行模式程序待排期) |

## 基线与记录 / Baselines & Records

历史测量依据,展示当时缺陷与修复前状态,不随代码更新。

| 文档 | 内容 |
|---|---|
| [FILM_V2_P0_BASELINE.zh-CN.md](FILM_V2_P0_BASELINE.zh-CN.md) | film v2 起点基线 |
| [FILM_APPEARANCE_P0_BASELINE.zh-CN.md](FILM_APPEARANCE_P0_BASELINE.zh-CN.md) | 外观层起点基线 |
| [FILM_OPTICS_V2_P0_BASELINE.zh-CN.md](FILM_OPTICS_V2_P0_BASELINE.zh-CN.md) | 光学 V2 起点基线 |
| [PERF_REVIEW_2026-08.zh-CN.md](PERF_REVIEW_2026-08.zh-CN.md) | 2026-08 性能审查记录 |
| [archived/](archived) | 更早的审查发现与 HDR 对比记录 |

数据文件:`film_v2_p0_decomposition*.json` 是 P0 基线的分解测量数据。
