# 第三批第四步：精确 ceiling 与逐通道剪切计数

本步将 `detect_ceilings` 的传感器代码值扫描和 `compute_clip_pct_by_thresholds` 的逐通道剪切计数迁移到 Rust，原生 ABI 升为 **17**。旧实现为每个通道反复构建颜色相等的布尔图，再 gather 对应 RAW 代码值进行统计。新核直接借用 RAW 与颜色索引，在扫描中维护 256 个可能通道的固定大小计数，避免这些完整栅格和逐通道像素副本。

这一步只替换精确扫描，不建立直方图，也不改写 full-well 判定、SensorSummary 的复用／失效政策或 RAW 证据的所有权。上一批 RGB 分组核、分析后一次构建最终 mask、已有 Rust 核均继续运行。RAW 解码与校准、自动曝光、AgX／HDR 形成及编码设置保持原流程。

## ceiling 的两遍扫描与合并合同

[`sensor_ceiling_counts_u16`](../rust/src/lib.rs) 接受原生字节序的 uint16 RAW、同形状 uint8 颜色索引，以及 256 项窗口 LUT，返回各 256 项的 ceiling、总数、精确峰值数量和邻近峰值数量。代码按实际 CID 统计，G1／G2 保持独立；二维与三维数组中的所有感光点都参与，包括不完整 CFA 周期的末端行列。本核不沿用 RGB 分组统计中裁去不足一个完整周期边缘的规则。

[`sensor.rs`](../rust/src/sensor.rs) 第一遍同时计算每个通道的 maximum、total 和 exact。遇到更大的代码值时，局部 maximum 更新为该值，exact 重置为 1；遇到相同 maximum 时增加 exact。合并 worker 结果时，总数始终相加，较大的 maximum 替换当前 maximum 和 exact，相同 maximum 才合并 exact。因此低峰值区域的局部 exact 不会被算入最终全局峰值的堆积数量。

第二遍在所有 worker 的结果合并后执行，使用每个通道最终的 **全局 maximum** 计算 `max(ceiling - window, 0)`，统计所有达到该下界的感光点。Rust 使用无符号饱和减法实现相同下界。near 不由各 worker 相对于自身 maximum 的局部窗口累加，否则一个局部较暗区域可能被错误地当成全局饱和堆积。两遍都只维护固定的逐通道标量数组，没有按代码值建立直方图，也不保留逐像素中间结果。

窗口及可信度政策仍由 [`analysis.py`](../dngscan/analysis.py) 的 Python 代码决定。原流程的整数元数据转换、0／缺失白点回退至 uint16 上限 65535、Python `round`、窗口最小值 2 均保持原样。传入 Rust 前将窗口上限收至 65535；对 uint16 输入，更大的窗口同样只会使下界变为 0，所以不会改变 near 计数。底层 API 接受 0–65535，Python 的生产入口仍使用至少 2 的原窗口政策。

`min_pile = max(CEILING_MIN_PILE_PIXELS, ceil(count * CEILING_MIN_PILE_FRACTION))` 继续在 Python 中按原顺序计算，可信堆积仍由 exact 或 near 达到门槛决定。后续 `resolve_fullwell` 仍执行原来的接近元数据白点、排除弱堆积及元数据回退规则。此处移入 Rust 的 maximum 不会直接取代可信 full-well，也不把普通场景亮部平台提升为饱和证据。

## 逐通道剪切与兼容行为

`sensor_channel_clip_counts_u16` 用独立的一遍扫描返回每个 CID 的 total 和 hit。RAW 值无损升为 int32 后执行 `raw >= threshold`，精确保留负阈值、零阈值和超过 65535 的 int32 阈值行为。阈值缺失时使用 **0**；这是 `compute_clip_pct_by_thresholds` 的原政策，与 RGB 分组核从整个阈值字典取最小值的政策不同，两者仍分别构造 LUT。

Python 按原 `channel_ids` 顺序输出字典，保留稀疏 CID 和重复 CID 的字典语义。剪切百分比保持 float64 先除以样本总数、再乘 100 的运算顺序，返回 Python float；请求的通道没有样本时仍返回 `0.0`。ceiling 入口对无样本通道则按原通道顺序抛出 `RuntimeError`，两种公开接口的缺失通道行为没有统一成同一种结果。

仅普通 ndarray 的 2D／3D uint16 RAW、同形状 uint8 颜色索引以及有效整数 CID 进入原生路径。元数据检查只针对 `channel_ids` 选中的值：普通 dict 中的 Python／NumPy 整数且落在 int32 范围内可以进入；不相关通道上的特殊值不会被提前求值。浮点、超出支持范围的整数、特殊类型、ndarray 子类、非原生字节序、广播形状或其他不满足合同的输入保留原 NumPy 路径。

窗口预计算遇到异常时在 dispatch 前回退，由原函数继续决定异常与顺序。例如前面的请求通道没有感光点、后面通道有非法元数据时，不能因为先批量准备 LUT 就改变原先的缺失通道错误。原生计算错误仍按既有 auto 模式回退、strict 模式报错政策处理；有效但超出原生能力的输入在资格检查阶段直接选择原实现。

## 借用、线程与空数组

两个入口复用既有 `sensor_view_shape`，在借用前验证维度、同形状、原生字节序、指针／步长对齐、元素数量、非零维度乘积和地址跨度。转置、反向和零步长广播视图都直接读取，无需整帧转为连续数组。Python 同步检查这些资格，包含 NumPy `ALIGNED` 可能忽略的 singleton 轴奇数步长，以及不能安全取负的最小有符号步长。

空数组在创建 ndarray 借用视图前直接返回 256 项零结果，包括三维零通道输入。这样不会在零长度负步长轴上触发不必要的指针调整；具体公开接口再按原规则返回 `0.0` 或抛出缺失通道错误。这里不沿用 RGB 低层 API 对零通道 H×W 像素的零组计数，因为本步统计总体是实际感光点，零通道数组没有感光点。

只读借用覆盖释放 GIL 后的完整计算。按 H 方向的行分区并遵守已有线程预算，小输入串行执行；每个 worker 仅维护 256 个 CID 的固定大小统计数组。每一遍结束都显式 join 所有已启动线程，线程创建失败或 worker panic 也在完成 join 后传播。计算不写 RAW、颜色索引或外部输出数组；第一遍失败不会进入第二遍，也不会发布部分统计。

## 可复现的测量方法

[`benchmark_sensor_channels.py`](../tools/benchmark_sensor_channels.py) 在两侧固定 `DNGSCAN_FAST=1`。`--reference` 仅通过 `DNGSCAN_FAST_SKIP` 跳过 `sensor_ceiling_counts_u16` 与 `sensor_channel_clip_counts_u16`，普通模式清空 skip。RGB 分组核、其他原生核、SensorSummary 复用和 deferred mask 始终启用；两侧必须有与各自 checkout 匹配的扩展。

真实 RAW 每次在独立进程中全分辨率解码、分析、自动曝光，再形成默认 AgX／P3 SDR 和目标 800 nit 的 HDR pair，**不执行编码**。测量覆盖 Sigma `_SDI0150.DNG` 的 LibRaw／Apple 两条路径、Sony `DSC00225.ARW`／LibRaw 和 Fuji `DSCF0214.RAF`／LibRaw 四条路径。新代码内的 reference/native 对照隔离两个扫描核的影响；改动前 `3847943`／ABI 16 则单独提供完整输出基线。

```sh
python tools/benchmark_sensor_channels.py \
  --source /path/to/photo.dng --decoder libraw --reference \
  --out /path/to/channels-reference-1.json

python tools/benchmark_sensor_channels.py \
  --source /path/to/photo.dng --decoder libraw \
  --compare /path/to/channels-reference-1.json \
  --out /path/to/channels-native-1.json

python /path/to/current/tools/benchmark_sensor_channels.py \
  --repo /path/to/checkout-at-3847943 \
  --source /path/to/photo.dng --decoder libraw --reference \
  --out /path/to/channels-original.json
```

Apple 使用 `--decoder coreimage`。各轮交替 reference→native 与 native→reference，保持输入、环境和参数相同，串行运行完整 RAW 测量，避免与测试或编译竞争 CPU。`--repeats` 仅用于合成模式，不在同一进程内重复真实 RAW 管线。

`--compare` 要求完整 `identity` 一致，**没有排除项**：加载后及分析后的 mask、processing loss、独立参考、完整分析与自动决策，以及 SDR／HDR base／HDR alternate 的形状、dtype、内容 SHA-256 均在范围内。参考报告有 `source_sha256` 时还检查输入文件身份。不一致返回退出码 2；已有输出、悬空符号链接和运行期间出现的同名文件不会被覆盖。未编码 master 一致不能推导出压缩文件逐字节相同。

合成模式运行 **24MP（6000×4000）** 和 **60MP（10000×6000）** 的 Bayer、6×6 X-Trans、三通道 LinearRGB。固定 seed 371 的 uint16 输入取值在 0–16383，饱和元数据设为 16383；LinearRGB 颜色索引为零步长广播视图。LinearRGB 的 MP 表示 H×W，每像素另含三个代码值。两个入口分别计时，分别记录各轮与中位数：先测 `detect_ceilings`，再以结果通过原 `channel_clip_thresholds` 和 margin 4 准备阈值，最后测 `compute_clip_pct_by_thresholds`。

```sh
python tools/benchmark_sensor_channels.py \
  --synthetic 6000 4000 --repeats 3 --reference \
  --out /path/to/channels-24mp-reference.json

python tools/benchmark_sensor_channels.py \
  --synthetic 6000 4000 --repeats 3 \
  --compare /path/to/channels-24mp-reference.json \
  --out /path/to/channels-24mp-native.json
```

60MP 将尺寸改为 `10000 6000`。输入生成、输入哈希和两个入口之间的阈值准备均在计时区间之外；报告保存输入、阈值、ceiling、exact／near、可信度和逐通道剪切百分比，检查多轮结果及两侧条件。合成测试只说明局部统计成本，不代表真实相机解码、完整分析、HDR 形成或编码总耗时。

`sensor_channel_calls` 记录两个 Python 入口与对应原生核的调用数及墙钟。原生时间已包含在入口时间中，不能相加；真实管线内的传感器统计也已包含在 analysis 等阶段内。SDR／HDR 两个阶段总和共享解码和分析，不能相加当作一次导出总时间。输出哈希与编码不在相关管线阶段计时内。

峰值 RSS 是整个进程的内存高水位，包括解码、形成、输入／输出身份检查和合成输入。它既不是扫描核的独占分配，也不是 GPU 内存；局部消除 gather 不自动保证进程峰值同比下降。最终结果需分别说明统计入口、完整阶段时间、RSS 与样片 HDR headroom 的适用范围。

## 实测结果

2026-09-20，macOS 27.2 arm64、10 个逻辑 CPU、Python 3.14.4、NumPy 2.5.2，release 扩展 ABI 17。先在未修改的 `3847943`／ABI 16 上生成四条路径的基线，再以新代码交替启用／跳过两个扫描核。**20 次新进程测量的完整 identity 与旧版一致**，包含 ceiling、exact／near、可信堆积、full-well、逐通道阈值和百分比、自动曝光、HDR 决策、mask 与 SDR／HDR master；源文件 SHA-256 也一致。

分轮计时、调用次数、输入／输出身份、关键传感器决策及代码指纹见 [`sensor-channels.json`](assets/performance/sensor-channels.json)。管线总和包含解码、分析、自动曝光和形成，**不含编码**。

| 路径 | 交替对照轮数 | 两项统计合计中位耗时：NumPy → Rust | SDR 中位秒数 | HDR 中位秒数 |
|---|---:|---:|---:|---:|
| Sigma `_SDI0150.DNG`／LibRaw | 3 | 225.00 → 10.07 ms | 7.443 → 7.179 | 8.453 → 8.136 |
| Sony `DSC00225.ARW`／LibRaw | 3 | 304.98 → 14.28 ms | 10.658 → 10.507 | 12.029 → 11.745 |
| Fuji `DSCF0214.RAF`／LibRaw | 3 | 297.50 → 24.99 ms | 22.791 → 22.515 | 23.749 → 23.460 |
| Sigma／Apple | 1 | 220.23 → 9.56 ms | 6.287 → 6.022 | 7.119 → 6.854 |

统计合计先对每轮的两个 Python 入口时间相加，再取中位数；不包含重复叠加其中的 Rust 子调用。四条路径两侧都只调用一次 ceiling 和一次逐通道剪切，说明 SensorSummary 仍复用同一份统计；原生侧各进入一次对应新核。

Sigma、Sony、Fuji 的未编码 SDR／HDR 中位数分别减少 **3.5%／3.8%**、**1.4%／2.4%**、**1.2%／1.2%**。局部统计在每对测量中均变快；整条管线则还包含解码和形成的波动：Sony 第二对 HDR 为 12.029→12.144 s，略慢，其余实拍对照均减少。不能据此把中位数降幅当作每次完整 JPEG／HEIF 导出必然达到的收益。

Apple 只测一对，用于独立 LibRaw 参考与完整路径正确性验证，不给稳定整体提速结论。Fuji 的 HDR headroom 为 0 EV，双路结果用于渲染器验证，不代表扩展动态范围交付。进程峰值 RSS 无一致下降；本步消除了统计的逐通道 gather，不宣称整条管线峰值内存降低。

以下为合成统计的三次进程内重复中位数。每种尺寸各用一个 reference 和一个 native 新进程，reference 在先；生成输入、哈希及两段计时之间的阈值准备均不计时。六种输入的 ceiling、exact／near、可信度与逐通道百分比完全相同。

| 像素数 | 布局 | ceiling：NumPy → Rust | 通道剪切：NumPy → Rust |
|---|---|---:|---:|
| 24MP | Bayer | 103.37 → 6.52 ms | 109.44 → 3.70 ms |
| 24MP | X-Trans | 82.96 → 10.98 ms | 87.72 → 5.98 ms |
| 24MP | LinearRGB | 458.98 → 18.43 ms | 479.03 → 9.60 ms |
| 60MP | Bayer | 258.46 → 14.80 ms | 270.78 → 8.01 ms |
| 60MP | X-Trans | 211.75 → 26.59 ms | 228.06 → 14.07 ms |
| 60MP | LinearRGB | 1173.18 → 44.86 ms | 1216.61 → 23.29 ms |

## 回归结果

release 构建、ABI 17 自检和 macOS 签名验证通过。Rust library 单元测试 **10 项全部通过**，其中本步新增 3 项验证跨 worker 的全局 maximum／exact／near 合并、转置／反向／广播视图、int32 阈值边界和零通道数组。

严格 native 完整 suite 运行 **1,767 项，1,761 通过、6 跳过**，无失败或错误。6 项均沿用原有条件：一项实拍裁切高光不足，三项缺少胶片 LUT，一项未配置 ideal-image 数据目录，一项旧胶片组合表未暴露。新增测试均通过。

NumPy 回退专项运行 **255 项，254 通过、1 跳过**，无失败或错误；唯一跳过项仍是实拍高光不足。新增 38 项在严格全套和 NumPy 专项中均通过。直接绑定合同测试在专项内会局部启用 native 以检查低层 API，不把它们描述为全程 NumPy 运算；其余回退及集成测试按 `DNGSCAN_FAST=0` 执行。

新增 [`test_sensor_channel_native.py`](../tests/test_sensor_channel_native.py) 的 28 项测试包含独立冻结的 NumPy oracle、原始整数计数和逐位百分比、Bayer／X-Trans／LinearRGB、全部感光点边缘、稀疏／重复 CID、0 白点、round ties、window 饱和、min-pile 规则、缺失通道／异常元数据顺序、前缀布尔索引、各种步长／dtype／子类回退、输入不变和 native 参数／结果预检。

新增 [`test_sensor_channel_benchmark.py`](../tests/test_sensor_channel_benchmark.py) 的 10 项测试验证两核消融且保留 RGB 核、两侧 deferred load、完整身份与输入 SHA、两入口／两核计时及异常计数、旧 ABI 16 reference、1×1 等不足周期的合成通道、小型确定输入、阈值准备不进入计时，以及已有／竞争输出不被覆盖。
