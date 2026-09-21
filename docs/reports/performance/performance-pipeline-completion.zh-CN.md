# 非胶片管线效率方案完成记录

本批沿用 `9c8a447` 的成像和编码规则。优化以同一环境下旧生产实现为参考；现有 Rust/NumPy 之间的历史容差不用于批准新的像素变化。新的实现、测试和测量覆盖实施方案中的分析、形成与交付、GUI、I/O 和资源调度步骤。

## 实施范围与取舍

| 原批次 | 完成的边界 | 记录 |
| --- | --- | --- |
| 0：参考与观测 | 独立进程 RAW 矩阵、固定 master 编码、真实 CLI、GUI 工作集及任务树采样 | 本文与 `tools/README.md` |
| 1：交付结构 | HEIF donor/共享回读/早拒绝已发布；本批补 JPEG 重试复用、输入转换及最终文件事务 | [交付首批](performance-delivery.zh-CN.md)、[渲染与交付](performance-render-delivery.zh-CN.md) |
| 2：loss | footprint 裁切和 half 合并核已发布 | [loss](performance-loss.zh-CN.md) |
| 3：RAW 与分析 | 只读采集、SensorSummary、延迟 mask、精确传感器扫描已发布；本批完成 Y-only、相位统计、AutoEV/sample 去重 | [sensor channels](performance-sensor-channels.zh-CN.md) 与本文 |
| 4：形成与指标 | 分块 mask、量化、HDR half、读回 owner、B3、guidance、base/coding 合扫及 HDR workspace | [渲染与交付](performance-render-delivery.zh-CN.md) |
| 5：GUI | 单 key 合并、选择代际、异步落盘、可信分析传递、共享噪声及字节额度 | [GUI 缓存](performance-gui-cache.zh-CN.md) |
| 6：I/O | 有界 opcode 摘要、任务内 metadata session、回退 evidence 身份核对 | 本文 |
| 7：资源 | 实际 operator share、有界任务提交、失败与中断时完整 join、并发观测 | [并发](performance-concurrency.zh-CN.md) |
| 8：验收 | 两种执行模式、精度、安装产物、真实 RAW/压缩结果、文档与 CI | 下文最终门禁 |

这里的完成指实施方案中有实测依据且满足等价合同的改动完成。native plan handle 的调用成本不足块计算的 0.004%，不增加 handle 生命周期；更深的 intent/retreat 融合仅约占 AgX+output 核耗时的 2.6%，保留现有分块边界。gated 内部换用现有 native AgX 的实验改变了 3 个最终 u8 码值，未启用。完整通道 MAD 和既有 NR 强度策略继续原定义。详见渲染专项中的测量，不将这些候选描述为已迁移。

持续 executor、常驻大图导出进程、HEIF 候选并行和新的跨进程内存入场策略均不是无条件目标。最终资源结论依赖并发实测范围；现有配额不等于整个进程树的 CPU/内存硬上限。DHT 的 OpenMP 仍关闭，未改变编码 quality/chroma/preset、系统解码或最终验收分辨率。

HEIF parser 的 offset/view 与模板解析缓存、跨编码复用 Pillow/heif_image 对象，以及 HDR 8×8 block 首遍扫描的进一步融合均保留为后续候选。原 HEIF profile 中重封装约 0.70 s、解析约 0.105 s；当前不以增加对象生命周期复杂度换取未经测量的整体收益。本文只将已经落地的合扫、多秩选择与 workspace 计为完成，不声称所有提案都已迁移到 Rust。

## 分析与自动曝光

内部 CLI/GUI 在不生成诊断图时，通过 `load_raw(..., _analysis_luminance_only=True)` 只保留分析需要的 Y 存储平面。Y 仍由原矩阵入口以百万像素块计算，保留 float64 运算、float32 落点、Apple half 再舍入，以及旧整数存储的归一化、clip 和截断；没有用另一组近似亮度系数。公开 `load_raw` 默认仍返回完整 XYZ；公开 `analyze` 默认仍返回 `(analysis, y, ev)`。内部 `_return_planes=False` 只改变临时数组的返回与生命周期，不抽样 EV 分位数。

Apple 对齐只将绿色通道转成 float32，再除实际 scale、筛选 finite 且大于 `1e-4` 的值、求全体 median；不会将除法移到 median 后。已经是 half 的输入独立复制并按原规则处理 NaN/±Inf，不再展开整个 RGB 到 float32。半精度全部 65,536 种位模式及所有权测试覆盖这条分支。

普通连续 uint16 mosaic 由 `PhaseStatistics` 按最多 128 个相位行共享校正 DN，同时生成 noise、SNR 和 green-pair health 的 tile 摘要。noise 的二次差分、MAD、暗部 argpartition，SNR 的 float64 mean/std 后转 float32、排序与分位数不变。health 保留全局原顺序的 variance 小数组，用同一 argsort（包括 tie 顺序）选择最多 768 个 tile，再只重取这些 tile 做原 f32 去均值与 f64 相关规约，不保存完整 green difference。第一绿色相位的原始 DN 用精确 65,536 桶直方图保留原 int64 percentile 的秩、插值和空桶定义，不把 spatial-black 校正值用于这个检查。相位摘要只存活于本次分析。

非 uint16、非连续输入以及 128 行不能整除自适应 noise tile 的极窄图保留原计算；小图的 noise tile 与固定 16×16 SNR tile 仍各自处理。Linear RGB 不因此获得独立感光点噪声资格。公开独立统计函数仍保留旧体，供兼容调用与逐位 oracle 对照。

AutoEV 不再重复计算 baseline 分位数，也不重复渲染搜索循环最后一个 high。CLI/GUI 的同一次导出可以接收其完整参考 RenderPlan，避免用同一声明再次采样和编译；公开 AutoEvResult 仍只含原有标量。此复用限于局部同一任务，不按文件路径缓存 plan。曝光依旧作用于 render，计划固定在原来的零 EV 锚点，adjustments 只应用一次。tone 和 AutoEV 的 mask 只提取同一组样本后升成 float32，避免在形成前就留下完整 float32 mask cache。

局部 `PreparedSceneSample` 让非胶片、非 gated 的全分辨率 AutoEV 共用原始 800k 以内的 storage RGB 和对应 mask；220k probe 继续在这个总体内二次抽样，不能直接重抽整图。GUI 已有完整源样本时直接使用既有数据。transform 后的样本在 plan 内共享，但不跨 EV 缓存：乘曝光、lens filter、scene transform 的运算和 float32 舍入顺序仍逐次保留。对象不挂全局缓存，结束 AutoEV 后释放。

## 形成与交付

HDR pair 的 SDR 路径复用已有两阶段矩阵 finalizer。连续且同尺寸的 half/float mask 按 chunk 转换；裁切和 resize 保留原 PIL/bilinear 语义。生产 AgX HDR 出口逐 chunk 写最终 RGBA half，同时精确保留 packing 前 RGB max 的 p99.99 所需上尾秩；旧 float32 HDR pair API 保留给诊断和调用者。packing 前分位数与 packing 后 half absolute max 始终分开。

HDR 回读返回持有原 bytearray owner 的只读 view，避免再复制整帧；HEIF uint8→8 bit 直接复制，uint8→10 bit 使用按原 float32 `/255`、`*1023`、`rint` 生成的完整 256 项表。浮点 HEIF 输入保留原转换。JPEG 重试在同一不可变母版内复用相同 quality/chroma 的主图 codestream。

自动交付的候选顺序、参考档、误差门限、选中档和 metadata 前大小比较不变。新文件在同目录私有事务中完成编码验证、metadata best-effort 和所需最终回读，之后才原子替换目标。失败保留原有成品；不会因 metadata 携带失败重新选择质量。

HDR 每次导出只建立一份不可变母版快照，分带验证 finite/half peak，并准备一次 RGBA、CIImage/NSData 与 CIContext；自动候选和手动 auxiliary 重试不再递归进入完整 writer。每个候选仍按原规则重建容器、回读并验收。这个输入准备会话带来一份新的 half HDR 快照（8 bytes/pixel），因此单独记录最终峰值，不把减少准备次数等同于降低 RSS。

## GUI、I/O 与预算

GUI 的 AutoEV、plan、balance 采用每 key single-flight：锁只管理查询、发布和 LRU，builder 在锁外执行。请求使用稳定 previewClient 和递增 selectionEpoch；共享同一个 RAW flight 时按所有订阅者判断是否仍有兴趣，不能因一个客户端切图而取消另一个客户端。

冷 proxy 先发布到内存，再进入有界单 writer 落盘队列。父进程将可信的小型 Analysis envelope 传给立即启动的导出子进程，匹配文件/cache/native ABI、完整解码元数据和实际几何后才复用；外部请求传入的同名内部字段先被删除。命中 analysis 后仍刷新新 scene 的 full-well mask。cache schema 升至 20，旧版本显式拒用。

同一几何的 WB 子 entry 共享两张原 seed-0 TPDF 源平面，子 entry 不保留已用完的分析缓冲。缓存按共享 backing owner 只计一次字节，并保留原条目上限；磁盘待写和活跃计算的持有量另外观察。scheduler 分别累计 queue 与 execute 时间，未改变超时和原任务类别配额。

共享 RAW/plan/WB/AutoEV flight 的订阅者若已占 preview/prepare 槽，仅在锁外等待期间归还槽；结果就绪后重新入场并检查是否过期，之后才继续计算。等待、重新排队与实际持槽时间分别记录。中断恢复必须先取回许可，许可取得后的登记失败会回滚归还，export 槽与原超时不受影响。这一步在 RAW/CLI 对照完成后补齐，未改变其成图或编码代码；最终并发及完整测试覆盖包含该修复的代码。

Apple opcode 摘要改为有界 seek，保留原 all-IFD/best-effort 策略，不替换为 LibRaw 的 required-opcode 解析规则。同一次 load 的 metadata session 按 path/dev/inode/size/mtime/ctime/头部摘要建立身份，文件变化使该会话永久失效。缓存的可变 OpcodePlan 每次返回独立副本。Apple auto 回退仅在读前、读后及递归入口身份一致时复用传感器 evidence；scene 仍新开 LibRaw 句柄。

Warp/base 核的预算 1 分支直接串行。可选 scene-transform/gated pool 按调用者实际 share 分配工作，保持结果声明顺序，并在异常或提交失败后 join 所有已提交工作。不把这些项目可控线程的额度称为对系统、NumPy、codec 或整个进程树的绝对线程上限。

## 测量和最终门禁

`tools/benchmark_pipeline_completion.py` 在独立进程中只形成一种 SDR、float HDR 或生产 packed HDR；哈希不计入阶段计时，进程 RSS 高水位包含验证。旧版按 CLI 的实际生命周期同时保留返回的 Y/EV 平面，新版内部 summary-only 不返回它们。相同母版的编码另由 `benchmark_gainmap_search.py` 验证，未编码 master 一致不替代压缩文件验证。

本机为 macOS 27.2 arm64、10 个逻辑 CPU、16 GiB 物理内存，Python 3.14.4、NumPy 2.5.2，严格 native。旧版 `9c8a447` / ABI 17 与本批 ABI 18 使用相同依赖，所有重测试和基准串行。每条路径、每种形成方式做三对交替测量，共 48 个新进程；RAW evidence、颜色、scene、mask、Analysis、AutoEV、RenderPlan、SDR 与 packed HDR/headroom 的身份在同一路径六次运行中全部一致。

| RAW / decoder | 形成方式 | 墙钟中位数，旧→新 | 峰值 RSS 中位数，旧→新 |
| --- | --- | ---: | ---: |
| Sigma 24MP / LibRaw | SDR | 7.301→6.941 s | 2878.6→2069.4 MiB |
| Sigma 24MP / LibRaw | packed HDR | 8.682→7.925 s | 2991.5→2085.4 MiB |
| Sony 32.95MP / LibRaw | SDR | 10.641→10.195 s | 3608.5→3258.6 MiB |
| Sony 32.95MP / LibRaw | packed HDR | 12.669→11.286 s | 3475.1→3269.3 MiB |
| Fujifilm 40MP / LibRaw | SDR | 23.767→22.906 s | 3623.7→3591.7 MiB |
| Fujifilm 40MP / LibRaw | packed HDR | 25.509→24.064 s | 4120.8→3794.7 MiB |
| Sigma / Apple RAW | SDR | 6.495→6.388 s | 2748.3→2100.3 MiB |
| Sigma / Apple RAW | packed HDR | 7.753→7.362 s | 2726.5→2101.3 MiB |

这些是 RAW→分析→AutoEV→形成，不含编码。墙钟中位数减少约 1.6%–10.9%，但多组取值范围交叠，不能承诺每次都按该幅度加速；CPU 总时间也没有一致下降。Fujifilm SDR 的 RSS 范围交叠，不宣称该项有稳定峰值收益。逐次数据、范围、阶段时间和精确身份见 [pipeline-completion.json](../../assets/performance/pipeline-completion.json)。Fujifilm 此样例没有获准扩展 HDR，因此它检验 pair 路径等价，不代表正 headroom HDR 交付。

独立统计段见 [有界 CFA 相位统计](performance-phase-statistics.zh-CN.md)：24MP 三对中位数 383.8→262.9 ms，进程峰值 434.5→146.3 MiB，全部公开统计精确一致。其 synthetic 60MP 单对用于检查内存规模，不能替代真实 61MP 全任务资源验收。形成/交付核、GUI 工作集和并发测量分别见上表链接，阶段收益不可直接相加。

`benchmark_cli_delivery.py` 另从 Sigma 原 RAW 运行真实 CLI，包含自动曝光、形成、编码候选、metadata 和最终发布。自动/手动各四种格式，共八对；全部决策、内容签名、主图回读和**完整文件 SHA-256**一致。手动 JPEG 固定 q98/4:2:2，HEIF 固定 q90/4:4:4；自动模式不传 quality/chroma。

| 模式 / 格式 | 最终主图参数 | 最终字节数，旧新相同 | 墙钟，旧→新 | 峰值 RSS，旧→新 |
| --- | --- | ---: | ---: | ---: |
| 自动 SDR JPEG | q98 / 4:2:2 | 13,337,371 | 9.839→9.944 s | 3057.1→2204.4 MiB |
| 自动 HDR JPEG | q98 / 4:2:2 | 20,147,085 | 14.476→12.977 s | 3668.4→3034.8 MiB |
| 自动 SDR HEIF | q90 / 4:4:4 | 24,498,093 | 121.830→125.672 s | 3789.4→2861.7 MiB |
| 自动 HDR HEIF | q90 / 4:4:4 | 46,247,403 | 191.398→193.145 s | 4156.8→3533.0 MiB |
| 手动 SDR JPEG | q98 / 4:2:2 | 13,337,371 | 8.122→7.561 s | 2978.0→2173.6 MiB |
| 手动 HDR JPEG | q98 / 4:2:2 | 20,147,085 | 10.179→9.451 s | 3435.9→3073.7 MiB |
| 手动 SDR HEIF | q90 / 4:4:4 | 24,498,093 | 22.142→21.672 s | 3633.9→2766.4 MiB |
| 手动 HDR HEIF | q90 / 4:4:4 | 55,093,418 | 86.592→84.689 s | 4190.1→3889.1 MiB |

这八对是最终交付正确性与峰值观察，每项仅一对，不据此宣称稳定的端到端加速。尤其自动 HEIF 墙钟没有下降，编码仍占主导；自动与手动 HDR HEIF 的 auxiliary quality 分别为 80、100，差异来自原有选择/重试规则，两边各自保持一致。额外输入哈希和交付回读不计入墙钟，但包含在进程 RSS 高水位中。

四路径的公开 float32 HDR pair、开启 gated 与非零 NR 也各做一次最终代码回归，身份全部精确一致；这些补充仅证明相应样例正确性，不作为稳定性能结论。

固定 24MP SDR/HDR 母版的 HEIF 搜索保留全部 14 次候选的指标与判定，选中 q90/4:4:4、aux80，最终 46,437,841 bytes 完整文件相同。主图编码 10 次、aux 编码 4 次、SDR 回读 10 次、HDR 回读 7 次均不变；新的母版图像准备只调用两次。base 与 coding 合扫的十次累计时间为 1.024 s，旧两个指标合计 1.363 s；HDR 指标 1.903→1.812 s。整次搜索反而为 183.864→188.366 s，其中主图编码 110.589→117.474 s、aux 编码 55.741→58.492 s。这一对用于证明候选/文件一致和调用边界，不能把局部减少扫描宣称成整个 HEIF 搜索更快。

最终冻结 288 份生产/测试文件，在每套验证前后检查 SHA-256：严格 native 完整套件 **1,904 项，6 skip，258.915 s**；NumPy 完整套件 **1,904 项，46 skip，715.043 s**，均通过。显式 digitization precision 1/1 通过、零 skip；Cargo 15/15 通过。GUI shared-admission 和缓存专项 108 项在两种执行模式下均通过。

纯 Python wheel 从隔离源码副本构建，确认不含 `.so/.dylib`，安装到独立目录后核对实际 import 路径、两级传感器 priors、镜头库及 film optics 资源。此本地检查使用已有 review venv 的依赖；CI 的 wheel job 另从全新 venv 验证依赖安装，不能把两种环境混为同一个结果。源文件与日志指纹保存在上述 JSON 的 `validation` 中。

最后冻结的 GUI/形成/分析组合完成三对并发复测：冷 prepare、72 帧连续预览和真实独立导出子进程同时运行。216 对预览与三次导出、完整 service 决策身份均相同。三对中位数 prepare **14.962→14.047 s**，导出 **15.190→14.454 s**，预览 p50 **172.69→169.78 ms**，p95 **256.00→256.81 ms**；p95 的逐对方向不同，不能宣称尾延迟改善，也没有持续回退证据。任务树采样 RSS **5.331→5.358 GiB**、范围交叠，不宣称全树内存降低。完整数据和资源边界见 [最终并发记录](performance-concurrency.zh-CN.md) 与 [concurrency.json](../../assets/performance/concurrency.json)，早期 `concurrency-phase.json` 仅作历史对照。

发布时沿用仓库 CI 的 macOS 14 / Python 3.11、3.12 两条完整 NumPy→native→precision 路径，以及独立的纯 Python wheel 安装任务。本文的本机通过记录与 GitHub 对提交运行的结果分别保留，不用本机 Python 3.14 结果代替其他版本 CI。
