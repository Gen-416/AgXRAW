# 文档索引 / Documentation

第一次使用从 [使用说明](USER_GUIDE.zh-CN.md) / [User guide](USER_GUIDE.md) 开始；阅读代码从 [开发指南与仓库地图](DEVELOPMENT.zh-CN.md) 开始。中文教程包含更完整的操作解释和实拍对照。

本文区分**当前行为、设计合同、历史测量**。计划中的提案不等于已经实现；报告里的时间、ABI、默认值和问题状态只对应其注明的版本。当前使用方法以使用说明和 CLI `--help` 为准，当前代码结构以架构说明为准。

## 使用与出片 / Use the app

| 文档 | 从这里了解什么 |
| --- | --- |
| [使用说明](USER_GUIDE.zh-CN.md) / [User guide](USER_GUIDE.md) | 支持的相机、界面读数、选项置灰、SDR/HDR 和导出档位 |
| [修图教程](EDITING_TUTORIAL.zh-CN.md) | 从导入到导出，曝光、曲线和 RAW 过曝标记的实拍示例 |
| [HDR 教程](HDR_TUTORIAL.zh-CN.md) | 参考白、可信高光、HDR 导出与可直接观看的样张 |
| [胶片教程](FILM_TUTORIAL.zh-CN.md) | 风格模式、完整冲印和各个胶片控件 |
| [机型支持](SENSOR_SUPPORT.zh-CN.md) | 传感器数据来源、能力降级和 LibRaw 支持边界 |
| [JPEG / HEIF 质量与体积实测](DELIVERY_QUALITY_STUDY.zh-CN.md) | 编码参数的选择依据；表中数字属于指定样张和版本 |

## 当前实现与开发 / Current implementation

| 文档 | 内容 |
| --- | --- |
| [开发指南与仓库地图](DEVELOPMENT.zh-CN.md) | 目录职责、按管线阅读代码、依赖与测试入口 |
| [技术架构](ARCHITECTURE.zh-CN.md) / [Architecture](ARCHITECTURE.md) | 解码、证据、分析、成像、交付及双解码器边界 |
| [产品架构](PRODUCT_ARCHITECTURE.zh-CN.md) / [Product architecture](PRODUCT_ARCHITECTURE.md) | 模块职责、领域模型与扩展边界 |
| [HDR 编码回读验证](HDR_DELIVERY_VALIDATION.zh-CN.md) | 有损压缩后的检查、容差与不能保证的部分 |
| [色度降噪](CHROMA_NR.zh-CN.md) | 色度 NR 的层位、尺度与约束 |
| [开发与测量工具](../tools/README.md) | 校准、基线生成、编码对照和性能测试的具体命令 |

## 设计合同与实施记录 / Design contracts

这些文件保留原路径，既记录实施过程，也包含当前代码依赖的数学定义与验收合同。阅读时先看文首状态；历史任务清单不作为当前待办清单。

| 文档 | 阅读口径 |
| --- | --- |
| [HDR AgX v2](HDR_AGX_V2_IMPLEMENTATION_PLAN.zh-CN.md) | 已落地；HDR tone/color 数学及生产合同，任务拆分为历史记录 |
| [胶片完整冲印](FILM_PRINT_RENDERING_PLAN.zh-CN.md) | 已落地 film v2 P0–P7 |
| [胶片外观层](FILM_APPEARANCE_RECIPE_PLAN.zh-CN.md) | 已落地；配方覆盖范围仍有限 |
| [胶片光学 V2](FILM_OPTICS_V2_PLAN.zh-CN.md) | 已落地 P0–P5 与 R1；冻结基线的变更合同 |
| [胶片风格模式](FILM_OBSERVATION_PLAN.zh-CN.md) | 已落地 observe 模式 |
| [胶片 Stage A 色度场](FILM_STAGE_A_CHROMA_FIELD.zh-CN.md) | 实际 shipped 算子与交叉验证；保留被撤回光源分档的测量依据 |
| [层间效应文献](INTERIMAGE_LITERATURE.zh-CN.md) | 专利定量转录、β 表对照与等效 IIE% 复现路线 |
| [渲染调度器](RENDER_SCHEDULER_PLAN.zh-CN.md) | 已落地 S1–S4；后续缓存变化另见性能报告 |
| [热白平衡迁移](HOT_WHITE_BALANCE_MIGRATION.zh-CN.md) | 已落地固定 Kelvin 热 WB |
| [实时预览](REALTIME_PREVIEW_PLAN.zh-CN.md) | 已落地；文中的 profile 数字为历史测量 |
| [管线性能等价方案](PIPELINE_PERFORMANCE_EQUIVALENCE_PLAN.zh-CN.md) | 同时含已实施项和长期提案；本轮非胶片实施边界见下方完成记录 |

## 测量、审查与历史 / Evidence and history

[报告索引](reports/README.md) 汇集性能批次、冻结基线、工程决策和旧审查。近期非胶片效率工作从 [管线完成记录](reports/performance/performance-pipeline-completion.zh-CN.md) 读起，它逐项区分已实施、未采用和后续候选；不要把单核提速相加当作整条管线的收益。

文档图片和机器可读测量保留在 [assets/](assets)。P0 分解数据 `film_v2_p0_decomposition*.json`、色度场 `chroma_field_cv.json` 和光源分档 `illuminant_tier_cv.json` 保留原路径，便于现有工具和引用复现。原始 RAW 和个人导出不属于公共文档资产。
