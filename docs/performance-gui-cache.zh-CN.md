# GUI 调度与缓存：保持输出，减少重复计算

本批对应实施计划 §7。RAW 解码、分析、AgX 与编码参数的数学政策不在这里改变。GUI 冷分析使用内部的仅 Y 工作区，并在建立 proxy 前释放分析平面；WB child 完成 scene reanalysis 后同样释放 XYZ/Y。全分辨率分析仍来自完整 RAW，tone plan 仍使用原定的全分辨率采样规则。

`PreviewEntry` 中的 plan、WB 和自动曝光按实际 key 合并同一轮计算。锁内只检查缓存、登记 Future 和发布结果，builder 与等待者均在锁外运行；失败只通知本轮等待者，下一请求可以重试。自动曝光 key 使用解析后的参数，省略默认值与显式默认值相同，输出色域、格式、调整参数、白平衡所属 entry 等依赖仍区分。

共享 flight 的订阅者还会通过 `shared_flight_wait()` 临时归还当前 preview/prepare 槽，等待已有 owner 后再取得同类槽，之后才继续计算。这样，正在等待预览冷解码 A 的 prepare 不会挡住另一文件的 prepare；正在等待 prepare 编译计划 A 的 preview，也不会挡住另一文件已经就绪的预览。两处接入分别是 RAW 的 Event 等待与 entry 的 Future 等待，owner 和缓存命中路径保持原样。正常 RAW/AutoEV 预览订阅本来就在 preview 槽外，此时该上下文不做额外入场。export 槽不暂停，原导出截止时间不变。

槽由请求线程的 lease 记录；同一 scheduler/类别重入借已有 lease，跨类别或跨 scheduler 的同线程嵌套明确拒绝，避免恢复多个槽时出现循环等待。flight 失败后先恢复槽再传播异常；恢复过程中被中断，也先完成同一槽的重取，再传播中断，保证上层即使捕获异常继续工作仍受原配额约束。plan 与 histogram 返回、prepare 恢复后重新检查选择是否过期；订阅者退出仍不取消其他客户端需要的 owner。

页面发送稳定的 `previewClient` 和随选择递增的 `selectionEpoch`，保留 session 内的 generation。排队的 prepare、RAW build 和发布边界检查选择是否仍有效。共享同一 RAW flight 的请求各自登记兴趣：旧页面退出不会取消另一个客户端仍需要的解码。正在运行的系统 decoder 不被强行中断；在可控边界跳过后续过期分析或渲染。

冷 proxy 先进入内存、唤醒等待者，再交给一个后台 writer。writer 最多保留两个待写项、合计 256 MiB；正在写的一项另计，最大同为 256 MiB。过大的可再生项直接跳过，队列满时淘汰旧待写项。快照仅持磁盘所需的 proxy、采样与分析，不保留 live entry 后续的 WB、frame、pixel、plan 或 dither 缓存。内部像素输入遵循已有的只读共享合同，避免为排队再复制大数组；写入仍使用临时文件和原子替换，失败不阻断预览。

立即导出不依赖 writer 已经落盘：父进程从 camera WB base 构建小型 analysis envelope，剥除外部请求伪造的内部字段后交给现有短生命周期子进程。子进程仍重新解码完整 scene，核对文件身份、缓存 schema、native ABI、实际 decoder/version/runtime、storage dtype、完整尺度、可靠性和几何元数据，匹配后才复用 Analysis。新 schema 为 **20**，19 及更早的磁盘分析不再接受旧的局部元数据比较。full scene shape/dtype 与 proxy shape/dtype 分别保存；不匹配、非 camera WB 或 dashboard 请求保持重算。命中后仍根据 fullwell 刷新新 decode 的空间 clip mask，并释放分析平面。

base 和同几何 WB child 共用一个 dither owner，保持原 seed、RNG、生成顺序和两张独立只读源平面。内部独占 RGB8 输出可直接转移给 pixel cache；外部或别名不明的输入继续防御性复制。缓存按 ndarray backing owner 去重统计字节，保留原 item 上限，并增加 512 MiB 的工作集预算；先驱逐旧 base，再驱逐唯一剩余 base 的可重算 runtime 项。活跃请求持有的对象不强制销毁。这个数值需结合真实 WB 工作集确认，不代表系统 RSS 上限或普遍最优值。

观测接口是 `PREVIEW_STORE.memory_snapshot()`、`DISK_WRITER.snapshot()` 与 `SCHEDULER.snapshot()`。缓存 bytes 主要表示被缓存持有的数组 backing buffer 与字符串/字节 payload，忽略 Python 对象开销，不等于 RSS。scheduler 的 `queue_seconds` 累计首次入场及恢复时的排队，`execute_seconds` 只累计持槽区间，`shared_wait_seconds` 单独记录已归还交互槽的共享等待；每个外层请求的 `completed` 只增加一次。父进程的 export 占槽时间包含等待子进程，不能与子进程 CPU 时间直接相加。原并发配额、导出 timeout 和跨客户端取消边界保持不变。

轻量合同测试覆盖同 key 合并、无关缓存锁可用、失败后重试、过期排队任务、两个客户端共享 flight、cold/memory/disk、v19 拒用、完整 source 几何、立即 export 的字段信任边界、mask 刷新与临时平面释放、共享 dither、字节驱逐重算和 writer 队列上限。运行：

```sh
DNGSCAN_FAST=1 python -m unittest tests.test_gui_shared_admission tests.test_gui_pipeline_cache tests.test_scheduler_s1 tests.test_scheduler_s2 tests.test_preview_cache tests.test_preview_realtime tests.test_gui_page tests.test_gui_guards -q
```

`DNGSCAN_FAST=0` 可验证 NumPy 路径。真实 RAW 工作集使用单独进程、临时磁盘缓存和 `tests/benchmark_gui_cache_workset.py`，显式传入 `--repo`、`--source`、`--decoder`、`--out`。它测量 GUI service API 的 cold prepare、自动曝光首帧及重复命中、七种 WB 两轮、清内存后的 disk prepare；记录 decode/analyze/plan/balance/dither/AutoEV 次数、缓存字节、调度统计、预览 JPEG 哈希和进程峰值 RSS。计数器不记录函数参数，避免 benchmark 自身保留完整 RAW 干扰 RSS；HTTP 传输与浏览器绘制不在该指标内。真实对照结果应另随测量报告给出。

2026-09-20 的缓存阶段轻量回归：加入 lease 前的七个模块共 96 项，ABI 18 严格 native 与 NumPy 路径均通过。后续 `test_gui_shared_admission` 新增 12 项，覆盖 RAW/plan/WB/AutoEV 订阅让出槽、无关任务完成、恢复配额、错误重试、中断后的槽恢复、取得 permit 后登记失败的回滚、不可恢复错误退出、重入/嵌套边界与恢复后过期检查。冻结后的八个模块共 108 项，严格 native 与 NumPy 路径均通过。该检查证明调度/缓存合同，不作为真实 RAW 墙钟性能或 512 MiB 工作集是否足够的证据。

实施计划的 GUI 必做代码项已接入，尚须真实工作集与并发测量决定是否调整默认字节额度。scene EV histogram 仍复用以 scene transform/intent 等为依赖的 base，encoded display histogram 仍来自对应 RGB8，未将两种统计混用。跨进程动态 CPU 份额、按 megapixel/dtype/codec 余量估算的内存入场控制属于 §8.3 的后续条件项：先观察三类任务同时运行的任务树峰值，再判断现有配额是否需要收紧或新增排队政策。本批的缓存驻留上限不覆盖活跃解码/编码的资源。持久 export worker、session executor 和 native plan handle 也仍取决于新 profile 是否证明启动/创建/解析成本显著；不以本批同 key 合并或 native atomic 预算代替该证据。

## 真实 WB 工作集验收

2026-09-20，在同机 Sigma `_SDI0150.DNG`、LibRaw、默认 AgX/P3/SDR 下，以新进程、独立临时缓存执行三对交替 `before → after`。before 为 `9c8a447` 主目录，after 为本批组合工作树（ABI 18）。每次包含冷 prepare、手动 EV 首帧、自动曝光首帧及重复、七种 WB 两轮，以及清内存后从磁盘恢复 camera 预览。计时范围为 service API，不含 HTTP 与浏览器绘制。

这组 WB 测量早于后续 PreparedSceneSample、有界相位统计、HDR 母版会话及 shared-wait lease 的最后接入，用于验证当时的缓存工作集与复用，不代表这些后续改动的独立收益。最终冻结版本的 RAW/交付和并发验收分别见 [管线完成记录](performance-pipeline-completion.zh-CN.md) 与 [并发测量](performance-concurrency.zh-CN.md)。

| 指标 | before 中位数 | after 中位数 |
|---|---:|---:|
| 全工作集累计墙钟 | 8.440 s | 8.182 s |
| 冷 prepare | 5.797 s | 5.699 s |
| 自动曝光首帧 | 0.381 s | 0.364 s |
| daylight 首次预览 | 0.317 s | 0.287 s |
| 6500K 首次预览 | 0.323 s | 0.296 s |
| disk prepare | 0.0708 s | 0.0710 s |
| 进程峰值 RSS | 3518.94 MiB | 2392.28 MiB |

冷 prepare 的三次范围分别为 5.680–5.809 s 与 5.684–5.759 s，互有重叠，不将其小幅中位数变化认定为稳定首帧加速。全工作集约减少 3.1%，RSS 约减少 32.0%；这是组合管线结果，包含分析工作区等同时启用的改动，不能全部归因于 GUI 缓存或共享 dither。

54 对对应预览 JPEG 哈希完全一致，camera 内存与 disk 重载的哈希一致，自动曝光首帧与重复命中一致。每轮 load/analyze 各一次、plan 七次、非 camera balance 六次、AutoEV 一次保持不变；dither 生成由七次降为一次。七种 WB 的第二轮全部命中，没有重新 decode/analyze/plan/balance/dither/AutoEV。优化工作树在该工作集中的最高缓存持有量为 **422.47 MiB**，低于 512 MiB，所以保留此默认预算；该证据仅覆盖所测工作集，不声称两个 RAW 的全部 WB/plan 组合同时驻留，或证明此额度普遍最优。

原始六份报告与逐项核验汇总保存在本机性能记录目录的 `completion/gui-workset-{before,after}-{0,1,2}.json`、`completion/gui-workset-summary.json`。三任务同时运行的交互延迟与任务树峰值由独立并发基准记录，不能从这个串行工作集外推。

## 连续预览的字节统计开销

后续并发基准发现持续调 EV 的延迟回退。`tests/benchmark_gui_frame_profile.py` 用真实 Sigma proxy，预热 plan/dither 后逐帧改变 EV，记录每次 `_trim_memory` 的调用来源和耗时；每轮清空 frame/pixel LRU，分别按正常通知、禁用通知、禁用通知、正常通知运行 36 帧。禁用通知仅为定位成本的进程内消融，不是生产缓存策略。测量已使用本批最终的 ABI 18 delivery/quantization 构建，修复前后只改变缓存统计的执行成本。

原实现每帧进行了三次全树字节统计，其中 dither 命中也重复通知。满 24 帧的 frame cache 包含大量 histogram 数值，这些数值从来不计入 buffer bytes，但每次递归仍执行 import、对象身份登记和类型判断。正常通知两轮的单帧中位数为 139.25/140.91 ms，禁用通知为 123.91/124.39 ms；字节统计本身的中位数约 15.42 ms，满缓存后的末 12 帧约 20.76–21.26 ms，确认了主要额外成本。

修复把 dataclass helpers 移到模块导入，对 exact Python bool/int/float/complex 提前返回零，并让相同 dither owner 的重复命中不再触发通知。第一次连接共享 owner 的 WB child 仍通知；数值子类、dataclass 内的数组/文本、共享 backing owner 和原有驱逐规则保持原合同。修复后正常通知的两轮中位数为 127.50/128.51 ms，禁用通知为 124.96/124.85 ms；统计中位数约 2.37 ms，比修前降低 84.6%，整帧降低约 8.6%。预算检查仍在像素和输出 payload 插入时执行。

两次 profile 共 288 帧在所有轮次和修复前后均逐一哈希一致。常驻 base 的 `_clip_masks_resized` 始终为空，符合每次曝光生成临时 bundle 的生命周期，因而没有把延迟回退归咎于一个未实际复用的旧 mask 缓存。报告为 `completion/gui-frame-accounting-{before,after}.json`，精简汇总并入 `docs/assets/performance/gui-cache.json`。这项串行消融解释并消除了大部分统计开销；最终并发延迟仍由修复后的长窗并发基准验收。
