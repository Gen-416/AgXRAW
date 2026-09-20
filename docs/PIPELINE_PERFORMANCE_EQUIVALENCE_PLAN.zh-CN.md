## 管线性能优化与零效果变化方案

本文是性能改造的规范性门禁。`REALTIME_PREVIEW_PLAN.zh-CN.md` 记录的 profile 数字用于定位瓶颈；若其中早期实验允许数值容差，以本文的“最终可见结果逐字节相同”要求为准。

2026-09-20 的非胶片效率实施与测量见 [管线完成记录](performance-pipeline-completion.zh-CN.md)。该批按各解码路径的既有生产实现验证新旧等价；本文中的长期冷路径并行、胶片与设备后端提案仍需各自的实现和发布门禁，不能从该批完成推断它们已经启用。

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
  **2026-09-15 原生层迁移到 Rust(rust/,PyO3 + setuptools-rust)**:四个内核与绑定层
  逐句转写自 C++,ABI v11 与模块 API 不变;验收口径是 C++ 构建在 26 组输入(含
  NaN/Inf 边缘)上的黄金输出与 Rust 构建逐位相同,性能持平(6 MP:AgX 38 ns/px
  同、finalize 43 vs 49、HDR 49 vs 47)。Rust 不会自行把 a*b+c 合成 FMA,
  C++ 时代靠 -ffp-contract=off 保证的性质在 Rust 里是语言默认;libm 调用
  (cbrtf/hypotf/atan2f/powf/expf/exp2f/log2f/fmodf)与 std::min/max 的 NaN 语义
  (第一参数保 NaN)按 C++ 语义显式复刻(rust/src/pixel.rs cmax/cmin)。
  **Stage 1(同日)**:解码侧证据与指标搬进 Rust——掩码羽化(`feather_masks_f16`)、
  DNG GainMap opcode(`apply_gain_map_mosaic`)、色域计数(`gamut_counts`)、HDR/底图
  回读校验(`hdr_roundtrip_metrics`/`base_roundtrip_metrics`)。NumPy 体保留为参考实现,
  `_fast.kernel(name)` 按同一策略分派;NumPy 的 float32 中位数((a+b)/2)、
  百分位(gamma 转 float32 的 _lerp,≥0.5 用反向式)与 8×8 块均值(64 个样本按 (i,j)
  行主序顺序 float32 累加再 /64)由实验钉死并在 tests/test_rust_stage1.py 里复刻;
  底图回读的两项 float64 求和按 band 顺序累加而非 NumPy 的 pairwise,是声明的末位差。
  实测 24 MP:SDR 10 s→7 s,HDR 15.7 s→约 11.5 s,导出 JPEG 逐字节不变。
  **2026-09-17 回读验证补强（ABI v12）**:HDR 扫描增加块内绝对亮度误差和局部
  高光最大误差，计算融合进已有 Rust 扫描，不分配全图亮度平面。绑定额外接收
  项目 P3 亮度权重；旧 ABI 自动拒用。公式、反例和实测门槛见
  [HDR 编码回读验证](HDR_DELIVERY_VALIDATION.zh-CN.md)。
  **Stage 2(同日)**:film_optics 的空间算子搬进 Rust——面积降采样/上采样、slab 高斯与
  5-tap 小 σ 模糊、halation 门/逐点回注/分量源、bloom 门/源/应用、散射混合、颗粒场
  采样(不复制主场,旋转几何按转置索引)、密度颗粒 v1/v2、halation 回注。参考平台
  语义由实验钉死并作为测试保留:Accelerate 的 (n,3)@(3,3) matmul(f32/f64)= k 顺序
  的 FMA 链;(n,3)@(3,) matvec 与 einsum "cj,...j->...c" 是顺序乘加(无 FMA);
  np.interp = fma(slope, x−xp[j], fp[j])(arm64 编译合并);np.sum(float64)= NumPy
  pairwise 顺序;float32 log2/exp2/exp/log/cbrt/hypot/atan2/pow 与系统 libm 逐位一致
  (sin/cos 不一致,NumPy 自带 SIMD 实现——本阶段无此依赖)。逐位验收:每个算子对 NumPy
  体随机输入逐位相同,固定 `--film-optics-seed` 后胶片+光学导出 JPEG 与主树逐字节相同
  (不给种子时 CLI 每次随机铸种,逐字节比较无意义)。`DNGSCAN_FAST_SKIP=a,b` 可按内核名
  回退 NumPy 做二分。实测 24 MP 胶片+光学导出 41 s→35 s;剩余大头是胶片核逐像素链
  (四面体 LUT 9 s、色度场 7.5 s、interp 2.5 s),为 Stage 3。
  **Stage 3(同日)**:胶片核逐像素链搬进 Rust(rust/src/film_core.rs)——层曝光与色度场
  对数曝光(`layer_log_exposure`/`chroma_field_log_exposure`,目前只接 float32 场景;halation
  预处理 slab 路径喂的压缩后 float64 场景保留 NumPy)。精度更正:observer 系数是 float64,
  float32 场景值并不保证乘积精确;小批次或尾行的 BLAS 累加也可能与 Rust FMA 链不同。
  macOS 27 / NumPy 2.5.2 的 Velvia 100 七像素样本可复现 logE 相差 8.33e-17。
  原先“所有平台逐位一致”的解释不成立;源文件的旧注释也不能作为此保证。
  `film_v2_math.py` 属于已发布资产的 builder-source 哈希输入,本轮不为注释重烘焙资产。
  特性曲线 interp(`characteristic_amounts`)、
  层间效应(`interimage_amplify`,含逐像素中性点)、四面体 LUT(`tetrahedral`)、高光预压缩
  (`film_compression_ev`)、技术中性 cast 逐像素除法(`cast_divide`)。`amounts_to_unit`
  留在 NumPy(0.4 s,不值一个内核)。新钉住的语义:np.mean(axis=1) 三元是 ((a+b)+c)/3;
  `_tetrahedral` 的 (g−i0) 先升 float64 再转 float32,四个权重 float32 左结合;float32 数组
  `/=` float64 插值结果在 float64 里除后回存 float32;float64 log10/exp2 走 libm;(n,3)@(3,3)
  float64 原生实现采用 k 序 FMA 链,BLAS 的所有尺寸不一定如此。内核按 budget::workers_for 切像素块并行(逐像素映射,切分精确)。
  验收:tests/test_rust_stage3.py 每个内核对 NumPy 原体随机输入 + 全部出厂 stock 的 B1/B2 LUT
  逐位相同,apply_film_core 负片/反转片、压缩、层间效应 custom、off/print crossover、retimed
  全路径 array_equal;固定种子胶片+光学导出 JPEG 与主树 sha256 相同。实测 24 MP:单独 Stage 3
  胶片 full 12.7→9.8 s、胶片+光学 41→25 s;三阶段叠加胶片+光学 15–17 s(Stage 1 前为 41 s),SDR 8 s。剩余大头是 LibRaw
  解码、`np.add.at` 的 float32 累加与积分图采样——已无单个 Python 阶段超过 2 s。
  **Stage 4(2026-09-17)**:剖析里剩下的两处 NumPy 残留。① halation 预处理 slab 路径
  (`FilmSpatialContext._layer_exposure_f32`)在未启用高光预压缩时不再先转 float64:float32 行
  原样交给 Stage A,由它自己升 float64——数值相同,而 float32 场景正是原生内核接受的输入
  (精度边界见上面的更正);启用压缩时场景是真 float64,仍留 NumPy。② `color.apply_rgb_matrix3`
  (float64 乘积、左结合 (a+b)+c、一次舍入到 float32)进 Rust,f32/f64 场景都接,float32 矩阵
  回落 NumPy。验收:tests/test_rust_stage4.py 逐位对拍(含 NaN/Inf/3e38 与 float64 场景);
  SDR、胶片+光学、带压缩的胶片导出 JPEG 与 ultrahdr 整个文件的 sha256 与 main 相同。实测
  24 MP:胶片+光学 16.9→14.6 s,ultrahdr 10.3→9.7 s,SDR 7.4→7.0 s。现在每条路径里最大的
  单项都是 LibRaw 解码(2.1 s,第三方),其后是 JPEG/HEIC 编码与 Apple gain-map 写出。
  **2026-09-17 Rust pipeline 内存修复**:GainMap opcode 直接借用可跨步的 mosaic/CFA
  视图,逐点写回并释放 GIL,删除全图 `(y,x,value)` 待写队列;临时空间只随行列坐标和小型
  gain 网格增长。clip-mask feather 改成每个线程复用一行 float32,直接输出交错 float16,
  遵守 native thread budget。HDR 回读借用 RGB/RGBA 的原始 strides,不再复制完整 RGB;
  工作线程共享 512 行临时预算并复用缓冲,每批立即合并精确 top-K,不保留历史行带。
  每个 RGB 分量在已有扫描中检查有限性,NaN/Inf 返回拒绝指标;NumPy 参考路径同样拒绝。
  SDR base 回读用 256 档整数误差直方图计算精确 p99,删除逐像素误差数组。
  Gaussian reflect 对单元素轴直接返回 0,其余用周期折叠,修复 1×N/N×1 死循环。

  回归见 `tests/test_rust_pipeline_memory.py`:包括 NaN/±Inf 各通道、反向/转置视图、
  重叠 GainMap、单元素轴、小图大半径、跨多批行带、不同线程预算及独立进程 RSS 门禁。
  Stage 3 新增全部出厂 stock 的短/奇数批次:float64 曝光中间量用 `rtol=atol=2e-14`,
  最终 float32 胶片输出仍要求 array_equal,原有精确回归不放宽。
  本机完整 `unittest discover -s tests -q` 两次通过: `DNGSCAN_FAST=1` 共 1435 项、
  跳过 36 项;`DNGSCAN_FAST=0` 共 1435 项、跳过 75 项。数字化精度门禁单独通过且未跳过。

  合成核基准(三次独立进程中位数,macOS 27.2 arm64、Python 3.14.4、NumPy 2.5.2,
  release 构建,thread budget=8;修复前 `7cb201c`)如下。峰值 RSS **包含输入与输出**,
  不是额外临时内存;输入为均匀图,不代表真实 RAW 全流程导出速度。

  | 核 | 24 MP 耗时 前→后 | 24 MP 峰值 MiB 前→后 | 60.2 MP 耗时 前→后 | 60.2 MP 峰值 MiB 前→后 |
  | --- | --- | --- | --- | --- |
  | GainMap | 0.137→0.038 s | 659→109 | 0.288→0.092 s | 1592→212 |
  | HDR 回读统计 | 0.302→0.224 s | 926→500 | 0.820→0.628 s | 2199→1158 |
  | SDR base 回读统计 | 0.051→0.017 s | 284→179 | 0.144→0.047 s | 644→391 |
  | mask feather | 0.127→0.035 s | 864→453 | 0.332→0.093 s | 2108→1074 |

  `tools/benchmark_native_memory.py --kernel gain --height 6336 --width 9504 --threads 8`
  可复测,`--repo` 指向另一份已编译 checkout 作对照。单线程预算下 feather 的 60.2 MP
  耗时为 0.340→0.392 s:旧实现无视预算固定启动三个线程,新实现遵守预算;默认多线程
  预算下更快。精确 HDR 中位数仍保留一份全图 float32 相对误差,空间不是 O(1)。

  **2026-09-17 空间核续修(基线 `d06d265`)**:Gaussian 与 small-sigma 卷积直接借用
  float32 源视图,每个线程只保存一行垂直滤波结果,随后写入独立输出;水平循环按 tap 遍历
  连续内存,保留每个像素的乘加顺序。`_blur_bounded` 的原生路径省去外层防写回拷贝,
  NumPy 路径仍保留该拷贝。scatter 在同一行池内按通道、分量顺序卷积并累加,删除全图
  通道、累加器和分量模糊图,也删除三个通道线程各自再启动 Gaussian 池的嵌套。
  area decimation 接受 float32/float64 及其 strided 源视图,在原 float64 累加位置逐样本
  转换,删除入口处整图升精度拷贝;两遍 np.add.at 的顺序保持不变,列降采样缓存仍保留。

  `tests/test_rust_spatial_streaming.py` 覆盖反向/转置/广播/只读视图、空图、单元素轴、
  超出图像尺寸的大半径、周期/反射边界、NaN/Inf 传播、不同源和累加器 dtype、分带累加
  以及源数组不被改写。独立进程 RSS 门禁新增四种空间核。父进程采样 scatter 的线程数:

  | native 预算 | 修复前进程峰值线程 | 修复后进程峰值线程 |
  | --- | --- | --- |
  | 1 | 4 | 1 |
  | 2 | 10 | 3 |
  | 4 | 16 | 5 |
  | 8 | 28 | 8 |

  完整测试:严格 native 与 NumPy 路径各 1443 项,分别跳过 36/75 项,均无失败;
  数字化精度门禁单独通过且未跳过。

  下面为相同环境、三次独立进程中位数,24 MP 均匀 float32 RGB 输入、预算 8。
  RSS 包括输入和输出;Gaussian sigma=1.7,small-sigma=0.7,area 输出 384×512,
  scatter 使用默认 emulsion asset、36/6000 mm/px。

  | 核 | 耗时 前→后 | 峰值 RSS MiB 前→后 |
  | --- | --- | --- |
  | Gaussian (`_blur_bounded`) | 0.186→0.047 s | 1143→594 |
  | small-sigma 通道视图 | 0.110→0.018 s | 593→410 |
  | area decimation | 0.121→0.076 s | 921→372 |
  | scatter mix | 0.240→0.093 s | 1692→594 |

  预算 1 下 scatter 为 0.387→0.425 s:旧实现仍占用三个通道线程,新实现真正串行。
  可用 `tools/benchmark_native_memory.py --kernel blur|small-blur|area|scatter` 复测
  (每次选择一个 kernel),`--repo` 对比 checkout,`--sigma` 设置 Gaussian 半径参数。
  60.2 MP、预算 8 时耗时分别为 0.613→0.112、0.278→0.048、0.457→0.175、
  1.066→0.313 s;核内存收益不等于完整导出的 RSS 收益。

  真实 `_SDI0150.DNG` 输出 4042×6064 JPEG,Portra 400 full、grain=0.5、halation=0.4、
  bloom=0.3、seed=42,前后交替各三次:main() 耗时中位数 15.37→14.28 s (约 −7.1%)。
  六个 JPEG 的 SHA-256 都是 `484598a1fbacb9bc7a22d26e564bad7005d70ebc711f7a800fc85bb3b91a6e1b`。
  本组完整进程 RSS 中位数 2094→2271 MiB,样本范围分别 1930–2279 与 2106–2298 MiB;
  因此本轮不声称整图峰值 RSS 已下降。
  另一次分阶段诊断通过 macOS `proc_pid_rusage(RUSAGE_INFO_V4)` 读取
  `ri_lifetime_max_phys_footprint`:前后为 3199→3086 MiB。两者在 load_raw 结束约
  1698 MiB、analyze 结束约 2168 MiB,高水位继续增长至输出结束附近。该数据来自一对
  带探针的诊断运行,用于区分测量口径与定位生命周期,不替代上面的三次无探针基准。

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
