# 测量与历史记录 / Reports

[返回文档索引](../README.md)

这里保存决策依据与可复现证据，不覆盖历史数字来模拟当前状态。每份记录的样张、软件环境、参考版本和测量范围都影响结论；最新行为看 [使用说明](../USER_GUIDE.zh-CN.md) 和 [架构](../ARCHITECTURE.zh-CN.md)。

## 非胶片性能

优先阅读 [管线完成记录](performance/performance-pipeline-completion.zh-CN.md)，再按瓶颈进入专项。`assets/performance` 中的 JSON 是对应报告的数据来源；不同批次的改善幅度不能直接相加。

| 记录 | 测量或实施边界 |
| --- | --- |
| [2026-08 性能审查](performance/PERF_REVIEW_2026-08.zh-CN.md) | 早期性能快照，不能作为现行耗时承诺 |
| [交付第一批](performance/performance-delivery.zh-CN.md) | HEIF 主图复用、分阶段验收与固定 master 对照 |
| [Loss](performance/performance-loss.zh-CN.md) | ABI 15 裁切／合并、逐位合同与测量方法 |
| [SensorSummary](performance/performance-sensor-summary.zh-CN.md) | RAW 只读所有权、摘要复用与缓存失效 |
| [延迟 mask](performance/performance-deferred-masks.zh-CN.md) | 分析后一次构建最终 mask、浮点白点与公开加载合同 |
| [RGB 过曝分组](performance/performance-sensor-rgb.zh-CN.md) | ABI 16 精确计数、步长借用与完整管线对照 |
| [传感器逐通道统计](performance/performance-sensor-channels.zh-CN.md) | ABI 17 ceiling／逐通道扫描、完整感光点计数 |
| [相位统计](performance/performance-phase-statistics.zh-CN.md) | noise/SNR/health 的有界工作区、24/60MP 对照 |
| [形成与交付](performance/performance-render-delivery.zh-CN.md) | ABI 18、分块 HDR、B3、交付指标与未采用方案 |
| [GUI 缓存](performance/performance-gui-cache.zh-CN.md) | 单 key 请求合并、异步落盘、可信分析与所有权计费 |
| [并发](performance/performance-concurrency.zh-CN.md) | 预览、冷解码与导出的延迟、任务树内存及线程 |
| [本轮完成记录](performance/performance-pipeline-completion.zh-CN.md) | 分析和 AutoEV、全路径文件等价、最终门禁与明确保留的候选 |

## 胶片冻结基线

这些是引入对应实现前的起点，正文中的问题描述不能直接用于判断当前版本。机器可读冻结数据与回归测试的位置写在各文首；更改冻结数据须遵守对应设计合同。

| 记录 | 口径 |
| --- | --- |
| [Film v2 P0](film/FILM_V2_P0_BASELINE.zh-CN.md) | film v2 起点及分解测量 |
| [外观层 P0](film/FILM_APPEARANCE_P0_BASELINE.zh-CN.md) | 外观层起点；包含后续校准口径说明 |
| [光学 V2 P0](film/FILM_OPTICS_V2_P0_BASELINE.zh-CN.md) | 旧空间算子测量及后续阶段复测 |

## 工程决策与审查

这些文档保留稳定路径，方便已有讨论和提交记录引用。日期早于当前代码的开放问题需要重新核对，不能直接当作当前 bug 清单。

| 记录 | 口径 |
| --- | --- |
| [工程决策记录](../ENGINEERING_NOTES.zh-CN.md) | 2026-07 集中开发期的证据、推理与教训 |
| [数据驱动 AgX / 胶片设计审查](../DATA_DRIVEN_AGX_FILM_DESIGN_REVIEW.zh-CN.md) | 2026-08-27 快照与处置，含已撤回设计 |
| [2026-09-02 审查交接](../reviews/CODE_REVIEW_HANDOFF_2026-09-02.zh-CN.md) | 指定 commit 的问题状态与集成接缝清单 |
| [更早的 HDR 审查](../archived/REVIEW_FINDINGS.md) | HDR v2 之前的历史合同 |
| [早期 HDR 对比](../archived/HDR_COMPARISONS.md) | 已退役算法的对比图，不作当前像素参考 |
