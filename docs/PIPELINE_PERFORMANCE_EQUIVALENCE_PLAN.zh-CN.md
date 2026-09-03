## 管线性能优化与零效果变化方案

本文是性能改造的规范性门禁。`REALTIME_PREVIEW_PLAN.zh-CN.md` 记录的 profile 数字用于定位瓶颈；若其中早期实验允许数值容差，以本文的“最终可见结果逐字节相同”要求为准。

### 目标与等价性契约

优化可以改变任务调度、缓存位置、内存生命周期和执行设备，但不能改变解码器语义、输入像素、矩阵、阈值、percentile、分支、运算顺序、16 轮 Oklab gamut-fit、随机序列或量化顺序。LibRaw 与 Apple RAW 本来就是两个独立的解码效果；每条路径分别与自己在相同依赖和运行环境下的串行参考实现对比，不要求二者互相相同。

“效果不变”分三层验收：

1. RAW evidence、mask、分析标量、RenderPlan 字段和所有影响分支的中间结果位级相同；
2. 预览/导出的 RGB8/RGB16、alpha、HDR base/gain map 逐字节相同；同一编码器和依赖版本下 JPEG/HEIF 也逐字节相同；
3. 缓存 miss/memory hit/disk hit、串行/并行、不同 worker 数和进程重启的结果相同，取消的旧任务永远不能发布。

浮点最大误差、p99 或感知指标只作为定位报告，不能替代上述发布门禁。若设备浮点实现无法做到相同输出，该 backend 保持关闭并回退参考路径。

### 先冻结可执行参考

在继续优化前新增 `reference` 执行模式，强制关闭本轮新增的并行调度、capture cache、analysis/output/device 快路，并固定 Python、NumPy、LibRaw、Core Image、编译器与色彩配置版本。优化前已经属于算法实现的 native AgX 等组件仍保持原样，reference 表示“当前产品串行行为”，而不是另造一个算法。每个用例输出一份 reference bundle：

- RAW evidence、sensor facts、ceiling/noise/clip/CFA 指标和 full-size mask；
- full-resolution scene、1920px scene/mask、所有 analysis 标量和 RenderPlan；
- tone core 后的线性结果、gamut-fit、transfer、两组 TPDF 噪声及最终像素；
- cache key、依赖版本、阶段耗时、shape/dtype/stride，以及每个数组和最终文件的 SHA-256。

Apple RAW 的 reference 必须在同一 macOS/Core Image build 上产生和比较；升级系统或 Core Image 时显式更新 reference ABI，不能把平台升级产生的变化记成性能优化。

### 解决方案

#### 1. 冷路径改成依赖图，而不是一条串行函数

首先只重排本来独立的工作，不移动算法边界：

```text
CaptureSeed(file bytes + immutable metadata)
  ├─ LibRaw evidence / sensor facts ──> CaptureInvariant
  └─ selected decoder scene decode ──> BalanceContext
CaptureInvariant + BalanceContext ────> analysis -> proxy -> RenderPlan
```

- Apple RAW：Core Image scene decode 与 LibRaw evidence 使用独立对象和有界 worker 并行，二者完成后再汇合。LibRaw evidence 不接触 Core Image 输出。
- LibRaw：evidence 与 scene decode 只有在两个独立 `RawProcessor`/文件句柄、无共享可变全局状态且实测 wall time 获益时才并行；否则共享一次 open/unpack 的串行实现可能更快。两种调度必须产生同一个 reference bundle。
- 文件读取由 OS page cache 共享，但 decoder 对象、错误状态和输出 buffer 不共享；worker 数受内存预算约束，避免两份全尺寸 RAW 同时物化导致 RSS 峰值失控。

#### 2. 把 capture invariant 与 WB/decoder 结果分开缓存

`CaptureInvariant` 只缓存真正与 WB 无关的原始域结果：evidence、sensor facts、ceiling/noise/clip/CFA 指标和 full-size mask。key 至少包含文件内容摘要、活动区/方向、LibRaw 与 sensor DB/evidence ABI；值不可变并带校验和。

`DecodeContext` 包含 selected decoder、decoder/Core Image 版本、高光、解拜耳和固定 AsShot 解拜耳预条件；`BalanceContext` 只包含用户 WB 矩阵、WB 后 scene/analysis/proxy。decoder 切换可以复用同一 `CaptureInvariant`，不能误用另一 decoder 的 scene。

2026-08-02 的产品决策把 WB 从 decoder-coupled 参数迁移为项目自有热阶段：LibRaw 与 Apple RAW 都只按固定 AsShot 预条件重建一次，再用 `C·Gtarget·Gdecode^-1·C^-1` 在 Rec.2020 中实现 camera-linear 重平衡。该迁移相对旧 LibRaw/Apple 内置 WB 是一次明确的算法版本变化，旧 oracle 只用于迁移 A/B 与审片，不能要求逐 bit 相同；新 oracle 冻结后，preview/export、cache hit/miss 和后续 native/Metal/CUDA 优化仍执行本文的逐字节门禁。详细边界见 `HOT_WHITE_BALANCE_MIGRATION.zh-CN.md`。

#### 3. 全分辨率 analysis 做 exact native 融合

保留 full-resolution 输入、当前矩阵阶段、阈值、NaN/Inf 处理和 percentile 定义，把 Rec.2020→XYZ、gamut test、EV/luminance 统计的多次 NumPy 扫描合并为分块 native pass。第一版只融合访存和调度：每个原有 float32 materialization/舍入点仍显式保留，percentile 继续调用同一参考选择算法；矩阵不得预合并，reduction 不得因线程数改变结合顺序。

确认逐 bit 相同后，才逐个尝试确定性的并行 histogram/select。任何改变 percentile、mask 或 RenderPlan 位模式的版本不启用。该阶段不以 1920px 样本替代全分辨率分析，因此不会制造预览/导出分叉。

#### 4. 胶片路径消除重复计算，但不假设错误的代数等价

RenderPlan 先生成一次 transformed sample，`scene_tone_metrics` 与 tone-plan 共享同一只读数组，替代当前两次相同前馈。稳定热帧的 scene transform 改为 exact native/Metal/CUDA kernel，严格复现当前逐阶段 float32 运算顺序。

不默认缓存“与 EV 无关的权重”或把曝光移过非线性变换：这类代数变形在实数域看似等价，在 float32、阈值和分支处不保证位级相同。只有 reference 证明权重本来就在曝光前生成且操作顺序不变时才缓存；否则仅缓存常量矩阵/只读参数，逐帧运行 exact kernel。

#### 5. 修正输出快路的两个已知非严格等价点

- 矩阵预合并去掉了原 NumPy 图中的一次 float32 舍入。**已修正（R2 项 6,ABI v8）**:
  `NativeOutputPlan` 携带 float64 的 `rec2020_to_xyz`/`xyz_to_output` 两阶段矩阵,
  kernel 逐阶段 float64 累加、float32 materialization,与 NumPy 表达式树逐位一致
  （`-ffp-contract=off` 早已在 CMake 保证无 FMA 合并）。默认路径即精确路径,不再有
  严格/优化之分;门禁收紧为 **in-gamut 像素 memcmp 相同**（旧口径 ~1% 像素差 1 code,
  现仅 gamut-fit 路径上 ≤0.05% 残差,见下一条遗留）。gamut fit 内部的
  `output_to_lms`/`lms_to_output` 仍是预合并矩阵——它只作用于 out-of-gamut 像素并
  受 1e-4 浮点容差门约束,拆分归属完整 reference-mode 程序。
- HDR 快路的输出级（`NativeHdrPlan` 的 rec2020_to_xyz/xyz_to_output）曾是 float32 链,
  头文件却写着"与 NumPy 运算顺序一致"（NumPy 在 float64 累加）。**已修正（批 25,
  ABI v10）**:与 SDR v8 同一合同——float64 两阶段、逐级 float32 materialization。
  实测（test_hdr_native 的 10 组 60k 像素扫描）max |Δ| 8.46e-5→8.27e-5、p99 5.2e-6 不变、
  逐位相同像素 21.7%→21.9%:这一级只是 HDR 残差的来源之一,其余来自曲线表插值、Oklab
  punch 路径与 gamut-fit（预合并矩阵,1e-4 容差）等 float32 级,HDR 门禁仍为
  2e-4 / 2e-5(p99),未收紧;逐级拆分归属完整 reference-mode 程序。
  **2026-09-03 数学审查(ABI v11)**:上面"其余来自曲线表插值、Oklab punch"的判断
  只对了一半——inset/outset 与 punch 的六个 Oklab 矩阵在 NumPy 里同样是 float64
  矩阵级(`agx._apply_matrix3`/`apply_rgb_matrix3`),两个核全部改为精确 f64 级后:
  SDR 逐位相同像素 47%→92%(punch 开启 7%→90%,max 1.55e-6→1.01e-6);HDR
  max |Δ| 8.27e-5→2.36e-5、p99 5.2e-6→2.0e-6、逐位相同 22%→81%。门禁随之收紧:
  HDR 6e-5 / 6e-6(p99),SDR atol 4e-6。剩余残差:曲线表插值、cbrt/hypot/
  smoothstep 等 float32 逐元素级与预合并 gamut-fit。
- 单平面 TPDF cache 把 `(value + noise_a) - noise_b` 改成 `value + (noise_a - noise_b)`。
  **已修正（批 20 期间）**:生产路径全部走双平面
  （`deterministic_dither_planes` / native finalize 的 noise_a/noise_b 入参,
  预览缓存 `get_or_build_dither_noise` 返回平面对）;合并单平面仅剩
  `deterministic_dither_plane` 兼容 shim,不在任何生产调用链上。

transfer、dither/quantize 和 gamut-fit 可以融合到同一 native 调用，但 kernel 内仍要保留上述语义阶段。最终 RGB 必须 `memcmp` 相同；“最大 1 code value”不再是发布标准。

#### 6. Metal（及未来任何 GPU 后端）只作为通过 exact gate 的执行后端

> 注：本节的 CUDA 表述写于跨平台设想期；项目现已声明 macOS-only，CUDA 无实现
> 计划。门禁条款对未来任何平台后端一体适用。

场景、mask、噪声和 plan 在设备常驻，每帧只上传参数并回读 RGB8。Metal/CUDA 禁用 fast-math、收缩 FMA 和不确定 reduction，显式复现 float32 舍入、NaN/Inf、分支、16 轮二分和量化顺序。若平台的 `pow/cbrt` 无法与参考位级一致，使用经穷举验证的 LUT/软件实现，或保留该阶段在 CPU；不能降低门禁。

Mac/Metal、CUDA 和 CPU 各自独立 feature flag、ABI 和回退。ANE/NPU 只有在能表达同一确定性算子并通过相同门禁时才考虑；不为使用硬件而改写成近似模型。

### 验证方案

#### 测试集

- 真实相机：现有 Nikon NEF、SONY ILCE-7M5 ARW，再覆盖 Apple RAW 支持的 Bayer/DNG、不同方向/尺寸、压缩模式、活动区和黑白电平；
- 参数笛卡尔抽样：LibRaw/Apple、全部解拜耳/高光/WB、AgX/neutral/lum/gated、sRGB/P3、film/look/filter、EV 边界和预览/导出；
- 合成输入：0/1、阈值两侧、基色/灰阶、clip/noise、NaN/Inf、随机广色域以及恰落在 quantize 边界的值；
- 状态场景：cold miss、memory hit、disk hit、进程重启、损坏 cache、版本失效、decoder 往返、快速滑动/cancel 和并发请求。

#### 分层门禁

1. **cache key 真值表**：逐个改变依赖，验证应命中的节点仍命中、应失效的节点全部失效；值的 SHA-256 与 reference 相同。
2. **冷 DAG**：串行与并行各重复至少 100 次并随机化任务完成顺序；所有 evidence、scene、mask、analysis 和异常类型/错误信息逐字段、逐 bit 相同，并用线程/地址检测器检查共享状态。
3. **WB 与 decoder**：LibRaw 和 Apple 分别比较新 WB cold miss、预热命中、LRU 回访和 decoder 切换；不得跨 backend 误用 scene，取消帧不得发布。
4. **analysis/RenderPlan**：reference 与 native、单线程与多线程逐数组 `memcmp`，所有 percentile、分支和 plan 字段位级相同；共享 sample 与原两次调用的两个消费者结果分别相同。
5. **胶片与输出**：固定两组原始噪声，逐阶段 hash 并最终 `memcmp` RGB8/RGB16/HDR；不同 chunk、worker 数、Metal/CUDA backend 和重复运行都相同。
6. **端到端文件**：同一依赖环境下预览 payload 与导出文件逐字节比较；若容器含时间戳，先固定或剥离非像素元数据并同时单独校验像素与必要元数据。
7. **完整回归与产物**：reference/optimized 两种模式跑全量测试；从 wheel/app 安装产物重跑，防止源码树旧扩展掩盖 ABI 问题。

任一正确性层失败即停止性能验收并回退该 feature flag。golden 更新不能和性能改造放在同一提交；确需改变效果时必须作为独立需求、独立评审和独立基线迁移。

#### 性能验收

正确性全绿后，在相同电源模式、线程数和缓存状态下报告 p50/p95/p99、CPU/GPU wall、RSS/显存、I/O、能耗和各阶段 exclusive wall time。至少覆盖：首次选 RAW、首次新 WB、WB 回访、普通热调参、首次胶片、胶片连续调参、预览与导出。

基于现有 profile，第一批优化分别以移除每个新 WB 重复的约 0.49/1.27 秒 capture-invariant 工作、重叠 Apple scene decode 与 LibRaw evidence、消除首次胶片约 80–110 ms 的第二次 sample 为目标；实际收益以改造后测量为准，不把理论可并行时间相加。全分辨率 analysis 和胶片 exact kernel 分别单独出报告，便于判断 macOS/Metal 与 CUDA 的真实收益。

### 实施顺序与发布条件

1. reference 模式、bundle/hash 工具、golden corpus 和 CI 门禁；
2. `CaptureSeed/CaptureInvariant/BalanceContext` 数据结构先串行落地，证明零 diff；
3. Apple RAW 并行 evidence/decode，再独立评估 LibRaw 双实例并行；
4. capture-invariant memory/disk cache 与严格失效；
5. RenderPlan 单次 transformed sample；
6. exact native analysis 与 exact scene-transform；
7. 修正矩阵舍入和双平面 dither 后启用 exact output fast path；
8. Metal（CUDA 已随 macOS-only 声明搁置）；以冻结后的项目 hot-WB oracle 为准。

每一项必须是可独立关闭、可独立回退的提交。合入条件同时满足：严格等价门禁全绿、目标平台有真实 profile 收益、峰值内存和取消语义达标；三者缺一不可。
