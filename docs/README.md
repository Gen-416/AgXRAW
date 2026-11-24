# 文档索引 / Documentation

[返回首页](../README.zh-CN.md) · [English README](../README.md)

- **第一次使用**：从 [使用说明](USER_GUIDE.zh-CN.md) / [User guide](USER_GUIDE.md) 与三篇中文实拍教程读起；
- **阅读代码与开发**：从 [开发指南与仓库地图](DEVELOPMENT.zh-CN.md) 开始。

> **阅读口径**：本索引严格区分**当前行为**、**设计合同**与**历史测量**。当前使用方法以使用说明和 `python -m dngscan --help` 为准；历史报告里的耗时、ABI、默认值与问题状态只对应其注明的版本，不覆盖现行文档。

---

## 一、使用与出片 / Use AgXRAW

| 文档 | 内容概览 |
| --- | --- |
| [使用说明](USER_GUIDE.zh-CN.md) / [User guide](USER_GUIDE.md) | 支持的相机、界面读数、选项置灰规则、SDR/HDR 与导出档位 |
| [修图教程](EDITING_TUTORIAL.zh-CN.md) | 从导入到导出：范围 → 亮度 → 色彩补偿与 RAW 过曝标记实拍对照 |
| [胶片教程](FILM_TUTORIAL.zh-CN.md) | 20 卷胶片的实拍对照：三类胶片、暗部与黑、颜色还原与颗粒 |
| [HDR 教程](HDR_TUTORIAL.zh-CN.md) | 参考白、可信高光、导出与验证，以及可直接观看的真 HDR 样张 |
| [机型支持](SENSOR_SUPPORT.zh-CN.md) | 传感器数据来源、能力降级与 LibRaw / Apple RAW 边界 |
| [胶片模拟技术说明](FILM.zh-CN.md) | 20 卷固定数据驱动胶片的用法、成像模型、资产与验收 |
| [胶片拆分与回并记录](FILM_SPLIT.zh-CN.md) | AgXFilm 研发档案、迁移边界、旧集成恢复与回并主仓记录 |
| [实测噪声标定](NOISE_CALIBRATION.zh-CN.md) | GUI / CLI 导入 JPTC 测量、启停、匹配范围及成像模型来源 |
| [JPEG / HEIF 质量与体积实测](DELIVERY_QUALITY_STUDY.zh-CN.md) | 自动编码参数的选择依据（表中数据对应指定样张与版本） |

---

## 二、架构与开发 / Architecture & Development

| 文档 | 内容概览 |
| --- | --- |
| [开发指南与仓库地图](DEVELOPMENT.zh-CN.md) | 目录职责、按管线阅读代码、环境搭建与双模式验证命令 |
| [产品架构与领域模型](PRODUCT_ARCHITECTURE.zh-CN.md) / [English](PRODUCT_ARCHITECTURE.md) | 软件四层架构、三大边界上下文、领域对象与扩展边界 |
| [成像管线与技术细节](ARCHITECTURE.zh-CN.md) / [English](ARCHITECTURE.md) | 解码、证据、分析、影调/色彩形成、交付及双解码器契约 |
| [HDR 编码回读验证](HDR_DELIVERY_VALIDATION.zh-CN.md) | 有损压缩回读指标、容差门限与能力边界 |
| [色度降噪与纹理保护](CHROMA_NR.zh-CN.md) | 独立噪声模型、处理域近似、BayesShrink 收缩与跳过条件 |
| [开发与测量工具](../tools/README.md) | 校准、基线生成、编码对照与性能 benchmark 命令目录 |

---

## 三、设计合同与实施记录 / Design Contracts

这些文件保留原路径，既记录实施过程，也承载当前代码依赖的数学定义与验收合同。阅读时请先看文首状态；文内历史任务清单不作为当前待办。

| 文档 | 当前状态 |
| --- | --- |
| [HDR AgX v2](HDR_AGX_V2_IMPLEMENTATION_PLAN.zh-CN.md) | 已落地；HDR 影调/色彩数学与生产合同 |
| [渲染调度器](RENDER_SCHEDULER_PLAN.zh-CN.md) | 已落地 S1–S4；后续缓存演进见性能报告 |
| [热白平衡迁移](HOT_WHITE_BALANCE_MIGRATION.zh-CN.md) | 已落地固定 Kelvin 热白平衡 |
| [实时预览](REALTIME_PREVIEW_PLAN.zh-CN.md) | 已落地；文中 profile 数字为历史测量 |
| [管线性能等价方案](PIPELINE_PERFORMANCE_EQUIVALENCE_PLAN.zh-CN.md) | 含已实施项与长期提案；本轮实施边界见性能完成记录 |

---

## 四、测量、审查与历史档案 / Evidence & History

- **总入口**：[报告索引](reports/README.md) 汇总各版本性能批次、冻结基线、工程决策与审查记录。
- **公共文档资产**：[assets/](assets) 保存文档插图、公开样张与机器可读验收 JSON（原始 RAW 与个人导出不入库）。胶片研究过程与光谱基线归档在 AgXFilm，详见 [胶片拆分与回并记录](FILM_SPLIT.zh-CN.md)。
- **近期关键性能基线**：
  - [管线完成记录](reports/performance/performance-pipeline-completion.zh-CN.md)（区分已实施、未采用与后续候选；不把单核加速直接等同于整条管线提速）
  - [导出专项记录（2026-10）](reports/performance/performance-export-reuse.zh-CN.md)（手动 HDR 重试主图复用、Core Image 释放与验收）

<details>
<summary>展开 2026-10 专题研究、标定与 PR 验收档案（15 项历史记录）</summary>

| 专题 | 文档与说明 |
| --- | --- |
| 噪声与纹理研究（2026-10-09） | [源码研究与反例](reviews/NOISE_TEXTURE_RESEARCH_2026-10-09.zh-CN.md)（基准 `738da0a`；第 13/14 节记有 10-bit 浮点母版实施，当前用法见 [实测噪声标定](NOISE_CALIBRATION.zh-CN.md) 与 [色度降噪](CHROMA_NR.zh-CN.md)） |
| 标定与原生 RAW 任务 | [标定与原生 RAW 任务记录](reviews/CALIBRATION_RAW_PLAN_2026-10-09.zh-CN.md) · [传感器精度研究](reviews/SENSOR_PRECISION_RESEARCH_2026-10-09.zh-CN.md)（区分生产接入与实验原型） |
| 实测校正与数值边界 | [2026-10-10 实施记录](reviews/MEASURED_CORRECTIONS_2026-10-10.zh-CN.md) · [克制色度清理](reviews/RESTRAINED_CHROMA_2026-10-09.zh-CN.md) · [镜头相对光谱接入](reviews/LENS_SPECTRAL_INTEGRATION_2026-10-09.zh-CN.md) · [Oklab 数值边界](reviews/OKLAB_ROUNDOFF_2026-10-10.zh-CN.md) |
| Apple RAW 9 适配与研究 | [窗口锚点范围](reviews/RAW9_ANCHOR_SCOPE_2026-10-10.zh-CN.md) · [解码配方状态](reviews/COREIMAGE_DECODE_RECIPE_2026-10-10.zh-CN.md) · [实机矩阵/输出残差研究](reviews/COREIMAGE_RAW_RESEARCH_2026-10-10.zh-CN.md) · [RAW9 实施记录](reviews/COREIMAGE_RAW9_IMPLEMENTATION_2026-10-10.zh-CN.md) |
| v3 管线合同与读回定位 | [v3 管线合同与验收](reviews/V3_PIPELINE_CONTRACTS_2026-10-11.zh-CN.md) · [Core Image 读回研究](reviews/COREIMAGE_HALF_READBACK_2026-10-11.zh-CN.md) · [最终 SDR 主图与增益图参考](reviews/FP_HDR_PRIMARY_REFERENCE_2026-10-11.zh-CN.md) |
| PR178 / PR179 标定与收尾 | [显式限制与新旧格式组合](reviews/CALIBRATION_SCOPE_LEGACY_2026-10-11.zh-CN.md) · [有效快门范围与 RAW9 工具合同](reviews/CALIBRATION_SCOPE_LEGACY_2026-10-11.zh-CN.md#pr179-后续收尾2026-10-11) |
| `fb91815` 后续修复与验收 | [镜头与重建范围](ARCHITECTURE.zh-CN.md) · [局部亮度纹理门禁](HDR_DELIVERY_VALIDATION.zh-CN.md#局部亮度纹理门禁2026-10-09) · [管线验收原始数据](assets/delivery-quality/pipeline-repair-20261009.json) |

</details>
