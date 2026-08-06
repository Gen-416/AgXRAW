## 实时预览技术方案

> **状态：混合文档**。"桌面与移动端 UI 契约"一节是**活合同**（发布门禁仍按其
> 四档视口执行）；各 profile 表与冷热路径方案记录的是 2026-08-02 前后的改造前
> 基线与已落地工作（P0 冷路径拆分、共享 RenderPlan sample 已完成，台账见
> [PERF_REVIEW_2026-08.zh-CN.md](PERF_REVIEW_2026-08.zh-CN.md)）。文末
> Metal/CUDA 一节写于跨平台设想期；项目现已声明 **macOS-only**，CUDA 部分
> 无实现计划，仅当未来放开平台时按同一门禁重启。

实时预览与导出继续使用同一套颜色算法和参数语义。优化只拆分计算阶段、改变缓存边界和预览载体质量，不减少 AgX gamut fit 的 16 轮，也不建立一套“预览专用”的颜色数学。

性能改造的规范性等价门禁见 `PIPELINE_PERFORMANCE_EQUIVALENCE_PLAN.zh-CN.md`。本文下方早期实验记录中的误差容限不能作为发布条件；最终可见结果必须与各自的 LibRaw/Apple 串行参考逐字节相同。

### 分辨率与画质

- 预览固定为 1920 像素长边。1920 可被 3 整除，常见 3:2 照片得到 1920×1280；其他画幅严格按照解码后原图比例四舍五入，不裁切、不拉伸，也不强制改成 3:2。
- 冷路径用用户选择的解拜耳算法完成全分辨率 RAW 解码，再在线性 Rec.2020 场景数据上用 Lanczos 缩到目标尺寸。旧实现的半尺寸解码会绕过所选解拜耳算法，再叠加 BOX 缩放，容易让细节发软。
- 浏览器预览固定使用 JPEG q95、4:4:4。导出质量和色度采样仍由交付档独立控制。

### 桌面与移动端 UI 契约

移动适配只改变控件的编排与导航，不建立移动端专用的颜色、预览或导出算法。桌面与移动端必须提交同一套参数，使用同一张 1920px 预览和同一条服务端管线；断点切换不得重置当前 RAW、参数或渲染结果。

1. **适用断点**：宽度不超过 767px 的视口进入手机布局；宽度不超过 900px 且高度不超过 500px 的短横屏也进入手机横屏布局。平板和桌面继续使用桌面仪表盘。
2. **竖屏结构**：预览固定在上方、当前控制分区位于中间、五页导航固定在底部。预览在任何控制页都必须保持正的可见宽高，不能被控制列挤成 0px。
3. **横屏结构**：预览在左、当前控制分区在右、五页导航纵向排列在最右；三者必须同时位于一个视口内。
4. **功能可达性**：手机导航固定为“解码 / 曝光 / 明暗 / 成像 / 色彩”，分别覆盖全部现有成像控件；同一时刻只展示当前分区，不能通过裁切把控件永久藏在视口外。输出参数继续由导出弹窗承载。
5. **无可见滚动条**：页面、控制区和导航均不得产生横向或纵向滚动条。输出弹窗内容超出短屏时可以内部滚动，但滚动条必须隐藏，所有输出字段仍可访问。
6. **触控与安全区**：可操作目标的命中高度至少为 44px；导航目标至少为 48px。页面四边必须使用 `safe-area-inset-*`，避免被刘海、圆角或 Home Indicator 遮挡。
7. **可访问性与状态**：导航使用 `tablist` / `tab`、`aria-controls` 和唯一的 `aria-selected=true`；支持方向键、Home、End，并记忆最后一个手机分区。

发布门禁固定覆盖 `375×667`、`393×852`、`412×915` 三档竖屏和 `852×393` 横屏。每档都要逐页验证：document/body 尺寸等于 viewport、预览和导航可见、仅目标分区可见、活动卡片及其控件不越界、触控目标达标。桌面端同时回归 `1440×900` 与 `1920×1080`，保证手机规则不会改变桌面单视口仪表盘。

### 冷热路径与缓存依赖

冷路径包含 RAW evidence、解码器及其版本、Core Image 版本、白平衡、高光恢复、解拜耳、全分辨率场景解码和 Lanczos 缩放。它们共同组成磁盘代理缓存键；任意依赖变化都会构建新代理。进程内保留最近两个冷配置，允许用户切换回来时直接复用，旧项按 LRU 淘汰。

热路径按以下层次复用：

1. 1920 线性场景代理和分析结果；
2. 与场景/色调结构有关的不可变 RenderPlan；
3. 与所有可见图像参数有关的 RGB8 像素（最近两帧）；
4. 包含 metrics 开关和 JPEG 表示的完整响应帧。

metrics 的延迟/补算不再重复颜色渲染，只读取同一份不可变 RGB8 像素。像素缓存不用于带自动曝光文字叠层的临时帧，避免把 UI 标注混入正常预览。

内存上限是两个冷代理、32 个小型 RenderPlan、两个 RGB8 像素帧和 24 个压缩响应帧；磁盘代理仍按文件数与总容量双重淘汰。

### Native 优化顺序

AgX 核心已经是 C++17/pybind native 实现，无需先整体重写。完成上述拆分后重新 profile；若非缓存热帧仍主要耗在 16 轮色域拟合，再在一个 native chunk kernel 内融合 Oklab 往返、二分和 gamut test。原参考图中的矩阵阶段和 float32 舍入点必须保留，不能通过预合并矩阵改变运算图。之后才考虑融合 transfer+dither 和复用常驻 worker pool。每一步都必须通过逐字节像素回归，保持预览与导出的算法一致。

### 输出后处理 native 优化方案

#### 计算边界

新增独立于 AgX 的 `NativeOutputPlan` 和两种入口，所有 tone core 共用，不把输出优化绑到 AgX：

- `finalize_rec2020_u8_f32`：输入 tone core 产生的 Rec.2020 float32，融合 Rec.2020→输出空间、16 轮 Oklab gamut-fit、sRGB/P3 OETF 和 TPDF dither/uint8 quantize；普通 baseline 走这条最短路径。
- `finalize_output_u8_f32`：输入已经执行 display filter、look 或 highlight chroma retreat 的输出空间 float32，融合后续 gamut-fit、OETF 和 quantize；这些功能仍保留原有 Python 色彩操作，但不再退回 NumPy 的 16 轮拟合。
- `fit_output_gamut_f32`：只暴露同一 native gamut 实现的 float32 结果，用于与 NumPy 参考逐点验证，不在产品路径额外增加一次计算。

`NativeOutputPlan` 在 Python 侧按输出色域和 gamut alpha 缓存原参考图的三组常量矩阵：Rec.2020→目标 RGB、目标 RGB→Oklab LMS、Oklab LMS→目标 RGB。矩阵阶段之间保留与 NumPy 相同的 float32 materialization，不能预合并。kernel 每个像素只在寄存器中保存 RGB、L/a/b、L0、lo、hi；越界像素严格执行 16 次二分，容差固定为现有 `1e-4`，in-gamut 像素保持原值后只做最终 `[0,1]` 夹取。

TPDF dither 的随机序列仍由 `np.random.default_rng(0)` 按当前顺序生成两组 float32 值并传入 native。第一版不替换 RNG，确保流式 chunk、预览和导出继续消费完全相同的噪声；native 严格执行 `(value + noise_a) - noise_b`、floor、clip 和 uint8 转换。这样先消除颜色中间数组和 16 轮内存往返，同时把随机算法变更隔离到未来独立决策。

native 输出 kernel 在一个 quantize group 上独占自身的像素并行；Python 外层继续按顺序提交 tone chunk，最终帧顺序和 RNG 消费顺序不变。先 profile 线程创建成本，只有它成为可测瓶颈后才引入常驻 pool，避免在同一变更里同时改变数值实现和调度生命周期。

#### 回退与失效

- `DNGSCAN_FAST=0` 永远走原 NumPy 参考；`auto` 在扩展缺失或 native 调用失败时用已经生成的同一组噪声回退；`DNGSCAN_FAST=1` 继续把 native 失败视为错误。
- native ABI 升级，旧扩展不会被静默加载。
- sRGB/P3、gamut alpha、look/display filter/highlight retreat、所有 tone core 都进入已有 plan/frame cache key；输出 plan 只缓存不可变小矩阵，不缓存像素结果。
- `render_output_linear`、HDR/gain-map 和分析路径保持原 float reference；本次只替换 SDR uint8 的最终交付路径。

### 一致性验证方案

按以下顺序设门禁，前一层失败不得进入性能验收：

1. **算法常量门禁**：native 迭代数只能是编译期 16；测试检查 ABI、sRGB/P3 plan、`alpha` 与 `1e-4` tolerance。预览和导出不得传入不同迭代数。
2. **float gamut parity**：对 sRGB/P3 和实际计划范围内的多个 alpha，覆盖 RGB 基色、灰阶、`-1e-4/0/1/1.0001` 边界、负值/高值、随机广色域值及 NaN/Inf。所有影响分支和 RenderPlan 的中间结果要求位级相同；预合并矩阵造成的 `8.6e-5` 误差只保留为历史诊断，不得启用。
3. **编码/量化 parity**：向参考与 native 注入完全相同的两组 float32 噪声并保持原加减顺序；相同线程数和不同线程数的最终 uint8 必须逐字节一致。最大 1 code value 不再作为可接受发布标准。
4. **端到端统一性**：AgX/neutral/lum/gated，sRGB/P3，EV、look、scene transform、highlight retreat 各取代表组合，对照 `DNGSCAN_FAST=0/1`；跑 stream ordering、golden、SDR freeze 和现有 native parity。预览与导出继续调用同一 finalizer，不增加预览专用近似。
5. **完整回归**：`DNGSCAN_FAST=0` 与 `DNGSCAN_FAST=1` 全量测试都通过；构建 wheel 后再从安装产物验证 ABI 和 self-test，避免只测试源码树中的旧 `.so`。
6. **性能门禁**：真实 1920px NEF 与 A7M5 各记录 output matrix、gamut-fit、transfer、dither、pixel pipeline 和端到端 p50/p95。融合阶段 p50 至少降低 25%，连续热帧 p50 不得回退超过 5%；若不满足，保留参考实现并不启用 native dispatch。

### 输出融合 profile 结果

2026-08-02 在 Apple Silicon 上用 `tools/benchmark_realtime_preview.py --output-backend numpy/native --iterations 40` 测量。两组都保持 `DNGSCAN_FAST=1`，因此 AgX native core 完全相同，只切换本次输出 finalizer；输入命中相同的磁盘冷代理，帧缓存通过逐轮改变 EV 避免命中。

| RAW | 尺寸 | NumPy 连续 p50 / p95 | Native 连续 p50 / p95 | NumPy→Native 像素管线 p50 | 首帧 |
| --- | --- | --- | --- | --- | --- |
| Nikon NEF | 1920×1275 | 153.91 / 159.23 ms | 73.87 / 75.63 ms | 143.82→62.91 ms（-56.3%） | 169.78→89.66 ms |
| SONY ILCE-7M5 compressed ARW | 1920×1281 | 120.08 / 147.75 ms | 74.23 / 80.97 ms | 109.98→64.76 ms（-41.1%） | 137.14→92.62 ms |

融合 finalizer 在每个 quantize group 上的 p50 / p95：NEF 为 5.49 / 9.26 ms，A7M5 为 4.39 / 7.47 ms。连续热帧 p50 分别降低 52.0% 和 38.2%，通过 ≥25% 性能门禁；测试仍固定 16 轮拟合并注入原 NumPy TPDF 噪声序列。

### 第二阶段瓶颈诊断与优化

同日继续在 M4 Max（12P+4E）上对 40 个连续变参帧记录进程资源计数和逐阶段耗时。两份 RAW 的 `filesystem_input_blocks` 与 major page fault 都为 0；session/frame/plan cache lookup 仅约 0.05 ms。磁盘 I/O 只属于代理准备的冷路径，连续预览明确是计算与内存流量瓶颈。

阶段耗时存在两个 Python render worker 与各 native kernel 内部 8 路并行的重叠，因此不能直接相加。去掉重叠后，端到端热帧仍主要由以下工作组成：AgX tone core 每帧累计约 54–60 ms CPU、clip retreat 约 11–18 ms、native output finalizer 约 17–20 ms，最后 JPEG q95/4:4:4 占 11–13 ms wall time。进程平均使用约 6.6–7.1 个 CPU 核；继续增加 CPU 线程会先遇到调度和内存带宽，而不是 I/O。

seed-0 TPDF 噪声只依赖预览尺寸和固定的 quantize-group 顺序。曾实验每个冷代理缓存只读 float32 单平面 `noise_a-noise_b`，完整导出仍用原流式双平面。每个 1920px 代理增加约 28 MiB，两个 LRU 代理上限约 56 MiB。单平面把浮点求值从 `(value + noise_a) - noise_b` 改为 `value + (noise_a - noise_b)`，实测 p99 差值为 0、最大 1 code value、变化通道低于 0.01%。在严格等价契约下该实验不合格，产品快路必须改为双平面缓存/精确 RNG 重放并保持原运算顺序，否则关闭该快路。

| RAW | 缓存前连续 p50 / p95 | 缓存 TPDF 后 p50 / p95 | p50 变化 | 热帧 CPU / wall |
| --- | --- | --- | --- | --- |
| Nikon NEF 1920×1275 | 78.99 / 82.22 ms | 64.68 / 67.25 ms | -18.1% | 461.87 ms / 7.13× |
| SONY ILCE-7M5 ARW 1920×1281 | 76.52 / 80.93 ms | 63.83 / 66.27 ms | -16.6% | 423.86 ms / 6.64× |

两个 CPU 实验未进入产品：把 AgX 到 finalizer 合成单个全帧 CPU 调用后，NEF p50 为 74.37 ms，破坏了现有 chunk 间的流水重叠；改用 libdispatch 常驻全局池后连续 p50 没有实质变化（64.68→64.45 ms），p95 与首帧反而变为 68.16 和 270.94 ms。它们说明 CPU 下一步不是继续扩大并行度或做全帧串行融合。

### 完整管线 profile：白平衡与首次胶片模拟

下面记录的是 hot-WB 迁移前的基线：当时页面在白平衡变化时调用 `preparePreview()`；选择胶片组合还会先把 WB 改成 5500K/3200K，再调用同一个冷准备入口。WB 曾是代理缓存键的一部分，因此一个尚未缓存的新 WB 会完整执行 evidence、全分辨率解拜耳、分析、Lanczos 代理和 RenderPlan。现已改为固定 AsShot DecodeContext + 项目热 WB，WB 不再进入 decode/cache identity；迁移设计与门禁见 `HOT_WHITE_BALANCE_MIGRATION.zh-CN.md`。下表保留为改造前性能对照，不能当作当前实现说明。

以下为 M4 Max 上空代理缓存的单次实测。时间是嵌套调用的 wall time；表内大阶段可相加，子阶段仅解释父阶段，不能重复相加。

| 空缓存首次操作 | Nikon NEF 1920×1275 | A7M5 ARW 1920×1281 | 判断 |
| --- | ---: | ---: | --- |
| WB 5500K：完整 prepare | 2999.58 ms | 7756.76 ms | 冷路径 |
| ├ RAW load/decode | 1543.32 ms | 4020.99 ms | 51.5% / 51.8% |
| ├ capture analysis | 1217.91 ms | 3304.26 ms | 40.6% / 42.6% |
| ├ 1920px proxy build | 170.44 ms | 356.02 ms | 5.7% / 4.6% |
| ├ baseline RenderPlan | 59.80 ms | 67.25 ms | 2.0% / 0.9% |
| └ proxy disk write | 7.74 ms | 7.94 ms | 可忽略 |
| WB 5500K：第一帧 | 103.25 ms | 105.03 ms | prepare 完成后才执行 |
| **WB 到可见图像** | **3102.83 ms** | **7861.79 ms** | 与「数秒」体感吻合 |
| Portra 400：完整 prepare | 3153.34 ms | 7915.92 ms | 同样先创建 5500K 冷代理 |
| Portra 400：第一帧 | 274.07 ms | 279.26 ms | 含光谱前馈与胶片曲线 |
| **首次胶片到可见图像** | **3427.41 ms** | **8195.18 ms** | 当前最差交互路径 |

冷准备的资源计数进一步排除了 I/O：WB profile 的 CPU 时间分别为 2996.27 / 7695.20 ms，CPU/wall 为 0.997 / 0.991；两份样张的 filesystem input blocks、major page faults 都为 0。这里的文件页已经在系统缓存中，因此不能代表机械盘首次读取，但它能说明应用内这 3–8 秒几乎全部是单核计算与数组分配，不是等待磁盘。

#### RAW load/decode 子阶段

| 子阶段 | NEF | A7M5 | 分析 |
| --- | ---: | ---: | --- |
| evidence 读取/复制 | 113.63 ms | 311.48 ms | WB 无关，当前每个 WB 重做 |
| LibRaw 解拜耳 + postprocess | 987.01 ms | 2518.20 ms | decode 最大项；需要保留所选算法与高光语义 |
| Rec.2020→XYZ 分析缓冲 | 98.99 ms | 254.28 ms | 大型全帧矩阵与分配 |
| 全分辨率 clip mask | 236.89 ms | 618.93 ms | evidence 派生，WB 无关，当前每个 WB 重做 |
| WB 求解 | 0.17 ms | 0.13 ms | 不是瓶颈 |
| 其余元数据/尺度/分配 | 106.69 ms | 318.22 ms | 父阶段减去已列子阶段 |

直接可复用的是 evidence、full-well/clip 证据和 mask；真正必须随 WB 改变的是场景颜色解码。短期只缓存磁盘代理会隐藏第二次等待，但不能解决第一次。

#### Capture analysis 子阶段

| 子阶段 | NEF | A7M5 | 分析 |
| --- | ---: | ---: | --- |
| 全分辨率 gamut metrics | 732.51 ms | 2052.23 ms | analysis 最大项，约 60–62%；应以融合 native kernel 保留同一全分辨率算法 |
| EV percentiles | 160.15 ms | 336.85 ms | 大数组 `log2` + percentile |
| RAW noise floor | 74.22 ms | 180.20 ms | WB 无关，可按 capture 复用 |
| clip percentage | 59.10 ms | 159.81 ms | WB 无关，可按 capture 复用 |
| ceiling detection | 55.77 ms | 151.15 ms | WB 无关，可按 capture 复用 |
| CFA cell metrics | 38.07 ms | 101.18 ms | WB 无关，可按 capture 复用 |
| refresh clip masks | 22.37 ms | 60.12 ms | 与前面的全尺寸 mask 构建重复接触大数组 |
| luminance buffer | 17.91 ms | 48.06 ms | 场景相关 |
| 其余/对象构建 | 58.80 ms | 214.72 ms | 父阶段余量 |

`diagnostics=False` 已经跳过 SNR 曲线和 RAW health，仍有 1.22 / 3.30 秒，说明下一步应拆分 analysis 数据依赖，而不是再关闭诊断项。传感器 evidence 派生结果按文件缓存；WB 相关的全分辨率 gamut/EV 则用一个并行 native pass 融合 Rec.2020→XYZ、luminance/EV、gamut test 和统计，避免 NumPy 为每个指标重扫及物化大数组。百分位定义、输入像素与最终 RenderPlan 都保持不变，不用 1920px 近似替换导出依据。

#### 代理与 RenderPlan

1920px 代理构建由全分辨率线性 Rec.2020 Lanczos（100.98 / 204.85 ms）和 clip-mask resize（68.93 / 150.67 ms）几乎完全组成。画质要求决定保留 Lanczos；在解码/分析拆分完成前，这里不是最高收益项。

已有磁盘代理时，普通计划约 60–75 ms；Portra 400 计划仍为 216.29 / 224.27 ms，其中 `scene_tone_metrics` 为 107.34 / 110.47 ms，tone-plan 内的第二次 scene sample 为 79.11 / 79.23 ms。两者对同一最多 80 万像素样本重复执行光谱前馈。计划编译应先生成一次不可变 transformed sample，scene metrics 与 tone plan 共享；这项不改变颜色算法，只消除重复计算。

#### 稳定热路径

普通 AgX 连续变参 40 帧的端到端 p50/p95 为 NEF 65.53/68.85 ms、A7M5 63.57/65.36 ms。同一帧内两个 render worker 与 native kernel 的线程会重叠，所以下表的「累计」是 CPU 工作量，不能横向相加；`pixel pipeline` 和端到端才是关键路径 wall time。

| 热阶段 | NEF | A7M5 | 分析 |
| --- | ---: | ---: | --- |
| cache/session/key/plan lookup | ≤0.06 ms | ≤0.06 ms | 已不是瓶颈 |
| scene intent（累计） | 7.93 ms | 7.77 ms | 存储尺度/曝光展开 |
| clip retreat（累计） | 10.71 ms | 18.15 ms | 内容相关 |
| native AgX（累计） | 60.44 ms | 53.00 ms | 其中隔离测量的 base/C1 为约 29–30 ms，hue restore 为约 6–13 ms |
| native output（累计） | 19.82 ms | 15.40 ms | NEF 的 16 轮 output gamut-fit 更重 |
| **pixel pipeline wall p50/p95** | **53.66/57.34 ms** | **49.79/51.82 ms** | 主关键路径 |
| JPEG q95 4:4:4 | 10.98 ms | 13.61 ms | pixel 后串行 |
| ICC | 0.18 ms | 0.18 ms | 可忽略 |
| **端到端 p50/p95** | **65.53/68.85 ms** | **63.57/65.36 ms** | 无 I/O；CPU/wall 约 7.0/6.6 核 |

Portra 400 的稳定热帧明显更差：NEF p50/p95 230.74/233.28 ms，A7M5 为 237.09/239.75 ms；pixel pipeline 分别为 221.58/224.46 ms。scene transform 的跨 chunk 累计 CPU 工作约 298.49 / 299.87 ms，是首要瓶颈；tone core 约 47.32 / 51.15 ms，clip retreat 10.14 / 16.90 ms，native finalizer 约 5.5 ms，JPEG 8.36 / 12.07 ms。胶片场景前馈按每个 EV 帧重新计算色度窗口、高斯权重和区域矩阵，解释了「首次之后调整也仍慢」。

精确帧回访缓存命中只需 0.13–0.16 ms；它只能覆盖参数完全相同的帧，无法帮助连续调 EV 或胶片强度。因此继续增加最终帧 cache 不是主要解法。

#### 按收益排序的改造边界

1. **P0 冷路径拆分**：建立以文件/decoder/demosaic/highlight 为键的 capture evidence cache，把 ceiling、noise、clip、CFA metrics 与 full-size mask 从 WB 代理中移出；WB 变化不得再执行这些 0.49 / 1.27 秒的工作。
2. **P0 全分辨率 analysis native 融合**：保留同一像素集、矩阵、阈值与 percentile 定义，把 gamut/EV 的多次 NumPy 扫描和中间数组合成一个并行 native pass；目标是压缩当前 0.9 / 2.4 秒工作，而不是用低分辨率近似制造预览/导出分叉。
3. **P0 胶片 transformed-scene cache/native kernel**：缓存与 EV 无关的色度窗口权重，或将 scene transform 与 exposure 融合为 native/Metal/CUDA kernel；不能缓存已经乘过 EV 的最终像素。目标是把约 300 ms 累计 CPU 工作移出每帧。
4. **P1 共享 RenderPlan sample**：一次 transformed sample 同时供 scene metrics 与 tone plan，降低首次胶片计划的约 80–110 ms 重复工作。
5. **P1 设备常驻整条热管线**：Metal/CUDA 常驻 scene/mask/noise，只传小参数、只回读 RGB8；普通 AgX 的目标是移除 50 ms 左右 CPU pixel critical path。JPEG 仍是约 8–14 ms 的下限，后续单独评估平台编码器。
6. **不作为主方案**：扩大 proxy/frame LRU 可以改善来回切换，预热 5500K/3200K 可以隐藏部分等待，但都会增加内存/后台计算，且不能降低首次真实计算量。

所有改造继续遵守同一算法契约：预览与导出共享公式、输入像素、16 轮 gamut-fit、边界与量化顺序；允许改变数据依赖、缓存位置和执行设备，不允许改变采样载体或建立另一套预览颜色数学。

### Metal / CUDA 异构热路径

下一阶段把冷代理作为统一设备边界，而不是只把 finalizer 搬到 GPU。仅 offload 输出阶段仍会保留约 50 ms CPU 像素管线，并为每帧增加约 28 MiB float 输入传输，收益上限太低。Metal 与 CUDA 共用以下逻辑契约：

1. `prepare_device_session` 将 1920px 线性 Rec.2020 代理、clip/gated guidance、固定 TPDF、常量矩阵和 tone/output plan 放入设备常驻只读 buffer；冷配置变化才重建。
2. 每次交互只上传 EV、look/filter、tone core 等小型参数块。scene intent、clip retreat、所选 tone core、Rec.2020→输出、固定 16 轮 Oklab gamut-fit、transfer、dither/quantize 在同一设备队列执行；只回读 RGB8，JPEG 先保留 CPU 编码。
3. macOS 使用 Metal compute 和 Apple Silicon unified-memory `MTLBuffer`，复用 command queue/buffer，避免 CPU↔GPU 拷贝；CUDA 使用 CUDA C++ kernel、常驻 device buffer、独立 stream，并在参数/shape 固定后用 CUDA Graph 重放。二者共享 plan 结构、测试向量和 CPU reference，不共享平台 kernel 源码。
4. ANE/NPU 不作为这段确定性色彩数学的执行器：它适合受支持算子组成的模型，而这里有分支、固定 16 轮二分和严格的 NaN/Inf/量化语义；强行转成 Core ML 会引入算子覆盖和一致性风险。CPU 仍用于 RAW 冷解码、回退与 JPEG，GPU 承担热像素管线。

实现按 Metal→CUDA 两个可独立验收的 backend 落地。每个 backend 都必须覆盖 AgX/neutral/lum/gated、sRGB/P3、边界/NaN/Inf 和固定双平面噪声；float 误差只作诊断，所有影响分支的中间结果以及最终 RGB8/RGB16/HDR 必须与 CPU reference 逐 bit/逐字节一致。性能报告必须同时给出 kernel、参数上传、RGB8 回读、JPEG 和端到端 p50/p95；CUDA 只在真实目标 GPU 上出数字，本次 macOS 机器不推测 CUDA 性能。

### 验收

- 3:2 横竖图分别得到 1920×1280 / 1280×1920，非 3:2 图保持自身比例；
- 更改解码器、版本、白平衡、高光或解拜耳会失效冷代理，更改 EV/色调参数只进入热路径；
- 同一参数的完整帧命中压缩帧缓存，metrics 表示变化命中 RGB8 缓存；
- JPEG 采样为 4:4:4，颜色回归和 16 轮 gamut fit 不变；
- 用真实 RAW 分别记录冷准备、首次热渲染、RGB8 复用和完整帧命中耗时。
