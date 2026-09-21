# Prepare、预览与导出的进程树测量

本页记录实施计划 §8.3 的最终资源验收。最终三对测量包含有界相位统计、prepared delivery master、GUI 字节统计、shared-flight 等待归还槽位和分段 finalizer。216 对交互帧、三对导出及完整 service 决策身份一致；冷 prepare 和导出更快，预览 p95 未出现持续回退。当前工作集没有提供新增全局 CPU 或内存入场限制的直接证据，保留现有类别配额。这一结论限于本页的样张、机器与并发场景，不代表跨进程硬预算已经实现。

## 测量边界

`tools/benchmark_pipeline_concurrency.py` 在独立 coordinator 进程中调用真实 GUI service。先预热 Sigma `_SDI0150.DNG` 的 prepare 与首帧，并等磁盘 writer 排空；随后通过 barrier 同时启动 Sony `DSC00225.ARW` 的冷 prepare、Sigma 的连续预览，以及 Sigma 的 `run_export_isolated`。导出仍由真实短生命周期 spawn 子进程完成。

测试机器为 10 个逻辑 CPU、16 GiB 物理内存的 macOS arm64 主机。Sigma 解码图为 24.00 MP，Sony 为 32.95 MP。两侧均使用 LibRaw、默认 AgX、sRGB SDR JPEG、固定 seed 371，导出使用默认自动曝光和编码参数选择。预览走两个独立 client 的正常 selection/generation 协议，72 个互不重复的手动曝光值从 0 向 +1.5 EV 递增。每帧计划间隔为 100 ms；实际 service 时延更长时，下一帧紧接上一帧完成后执行。

每次运行使用新进程与独立临时磁盘缓存。baseline 为干净的 `9c8a447`、ABI 17；组合工作树为 ABI 18。三对顺序为 before→after、after→before、before→after，两侧运行同一份 schema 2 工具。原有 prepare/preview/export 类别配额均为 1。最终 runner 在首次运行时记录两侧全部 Python 文件、native 共享库和工具的 SHA256，六次运行前后逐一复核，确认生产身份没有变化。

外部 monitor 通过 `libproc` 递归采样 coordinator、export child 和 multiprocessing resource tracker。记录每个 PID 的 RSS/线程数、采样时间与读取耗时，同时记录每次 scheduler slot 的排队/执行时间和每帧 service 时延。`proc_listchildpids` 的返回值按 PID **数量**解析，已有真实子进程合同测试；最早误按字节解析的 pilot 已单独标记无效，不参与统计。

RSS 是采样时各进程 resident pages 的总和，共享页可能重复计数，也可能遗漏两个采样点之间的尖峰，不等于物理 footprint 或内存硬上限。线程数包含闲置线程，不能直接当作 CPU 饱和度。基准暂存 72 个预览 JPEG/base64 响应，直到并发窗结束后核验身份，106.73 MiB 的这一观测持有量在两侧相同，不属于产品缓存。HTTP 传输、浏览器绘制和报告写入不在 service 计时内。

## 最终结果

2026-09-20 的三对中位数如下。每个 run 先对自己的 72 帧计算 p50/p95，再取三次 run 的统计量中位数，没有把六次运行的帧混成一个总体。p95 使用 nearest-rank，72 帧对应排序后第 69 个值，即第 4 慢帧。完整逐帧时延、采样汇总、各对变化、source/产物 SHA256、源码清单及原始报告 SHA256 见 [concurrency.json](../../assets/performance/concurrency.json)。

| 指标 | before | after | 变化 |
|---|---:|---:|---:|
| 冷 prepare | 14.962 s | 14.047 s | −6.12% |
| 完整导出 | 15.190 s | 14.454 s | −4.85% |
| 预览 p50 | 172.69 ms | 169.78 ms | −1.69% |
| 预览 p95 | 256.00 ms | 256.81 ms | +0.32% |
| 采样任务树 RSS 峰值 | 5.331 GiB | 5.358 GiB | +0.51% |
| 采样线程峰值 | 33 | 36 | +9.09% |

216 对交互帧的 JPEG 字节一致，三对导出产物一致；完整 service 响应中的直方图、检测信息、自动曝光和编码决策等也一致。比较只排除 `cache_hit`、`pixel_cache_hit` 两项缓存观测字段，并归一临时工作目录。schema 2 将裸 base64 和 data URL 都转为字节数/SHA256 保存。scheduler 的计时观测单独保存在资源快照中，不参与输出身份。

冷 prepare、导出和预览 p50 三对均改善。p95 的逐对变化为 −7.43%、+2.91%、−0.16%，绝对变化分别为 −20.61、+7.46、−0.38 ms；中位数差为 +0.81 ms。before 范围为 244.25–277.42 ms，after 为 243.86–263.46 ms。因此没有证据认定最终版本持续损害交互尾延迟，也不能从三对测量推导普遍提速或严格统计显著性。六次运行的 72 个帧中点均位于 prepare 与 export 同时在途的窗口，没有因后台任务提前结束而混入大量无竞争帧。

RSS 范围为 before 4.991–5.390 GiB、after 5.219–5.419 GiB，方向不一致，不认定为稳定下降或上升。线程峰值范围分别为 30–35、36–37；本次线程数增加，不能宣称整体线程数量减少。请求 30 ms 采样，实际间隔的中位数为 39.21–39.96 ms，p95 为 46.84–48.12 ms；原始报告保留全部实际间隔与偶发的进程退出读取失败。

## 排队与资源决策

外层 slot wrapper 记录 context 的墙钟时间，包括可能发生的 shared-flight 等待和重新入场；新 scheduler 快照的 `execute_seconds` 增量才是实际持槽时间，`shared_wait_seconds` 另计共享等待。最终资产同时保留两者。baseline 未提供的观测记为 null，没有伪装成零。本工作集每类同时只有一个任务，新版本 shared wait 增量为零，因此这项实测不替代同 key 合并请求和等待归还槽位的合同测试。

新版本 72 帧累计入场排队为 0.107–0.162 ms，prepare 和 export 的排队也只有微秒量级。这表明当前场景的交互时延主要不在类别信号量排队中；并不证明任务内部不存在 CPU、内存带宽或 codec 竞争，也不能将“每类 quota 为 1”当作整个进程树已有统一 CPU 上限。

在这台 16 GiB 机器上，24MP 预览/导出与 32.95MP 冷 prepare 的采样任务树峰值约 5.4 GiB，最终三对未见持续的 p95 回退，prepare 与导出均改善。因此本轮保留原类别配额和缓存额度，没有加入未经收益验证的固定全局配额或内存入场门。资源阶段的结论是完成测量后选择不新增限制，而非遗漏该阶段。

这仍不能保证 61MP、低内存机器、HDR/HEIF、多导出突发或其他应用占用大量内存时的资源边界。采样没有测系统实时可用内存或所有应用的压力，约 5.4 GiB 也不是容量承诺；这些情况若出现压力，需要单独测量后验证动态入场或交互优先级策略。当前数据不足以承诺所有机器均无需进一步资源控制。

## 历史阶段

[concurrency-phase.json](../../assets/performance/concurrency-phase.json) 保留补齐有界相位统计和 prepared delivery master 之前的三对阶段数据。该阶段 p95 中位数为 +3.93%，促使最终版本重新完整验收；原始报告名为 `concurrency-final-*`，仅是当时预定的批次名，不代表最终生产版本。最终报告使用 `concurrency-complete-*`，本页表格仅采用这一批。

更早的 12 帧三对和 72 帧单对报告也保留在外部性能记录目录中，说明回退的发现过程。12 帧只覆盖前约两秒，不能代表完整形成/编码阶段的持续竞争。这些历史数据和无效 pilot 均不计入最终中位数。
