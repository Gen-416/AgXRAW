# 第三批第一步：复用同一份 RAW 的传感器统计

本步完成了证据采集后的不可写存储，以及随单份 `RawEvidence` 生存的标量统计缓存。Apple 管线先为独立 LibRaw 参考样本计算传感器满阱，随后 `analyze` 需要相同统计；现在两处共用一次计算结果。LibRaw 管线原本只有 `analyze` 的一次首算，本步不减少这一次必要计算，也不据此承诺两种解码器获得相同收益。

这只是第三批 SensorSummary 工作的第一步。统计仍调用原 NumPy／Python 函数，没有新增 Rust 核或改变 ABI；全套传感器分析、直方图扫描合并，以及满阱确定前后重复构建 mask 的优化均不属于本步。此前 ABI 15 的原生核继续按原规则运行。

## 证据所有权与统计范围

[`acquire_raw_evidence`](../../../dngscan/evidence.py) 原来将 rawpy 的可见传感器数组复制到 NumPy 拥有的可写内存。现在以一次 `tobytes(order="C")` 复制取得独立存储，再通过 `frombuffer` 构造只读数组；这份 bytes 存储替代原来的 ndarray 复制，不额外保留一帧 RAW。即使沿着 ndarray 的 `.base` 链访问，每一级也不能通过 `setflags(write=True)` 恢复写入。解码句柄关闭或工作解码器修改自己的 RAW 缓冲区，都不会改变已采集的证据。

CFA 的代码值与颜色索引分别复制，保留输入 dtype、字节序和逻辑形状，得到连续存储。LinearRGB 在复制前只选择前三个通道；颜色索引由不可变的三个字节 `0, 1, 2` 广播生成，不分配完整的 RGB 索引栅格，也不隐藏一个可写的 NumPy seed。白平衡、黑电平和通道白点等元数据继续保留原来的独立 list 接口。

[`SensorSummary`](../../../dngscan/sensor_summary.py) 只保存标量和不可变 tuple：通道 ID、元数据饱和值、观测 ceiling、精确与邻近堆积计数、堆积可信标记、全局及逐通道 fullwell，以及原有说明文字。实际计算仍依次调用 `channel_saturation_levels`、`detect_ceilings`、`resolve_fullwell`：

- 元数据先按原规则转为 `int`；通道白点缺失、为零或转换后非正时仍回退到全局白点。没有修改取整或 fallback 规则。
- ceiling 邻域、最小堆积数量和可信满阱的接近白点条件保持原值。普通亮部平台仍不能被直接当作传感器满阱。
- G1／G2 保持独立通道。稀疏通道 ID 仍按实际 ID 索引白点，不将 `{0, 2, 5}` 压缩成连续的三个白点位置。

`scene_reference.reliable_reference_samples` 与 `analysis.analyze` 取得缓存后，各自生成新的 list／dict 供旧接口使用，因此修改一次分析的输出不会污染下一次调用。缓存不包含饱和判定余量 `margin`、颜色标签、硬剪切比例、噪声、SNR、场景百分位、自动曝光或 HDR 决策。

满阱后的 `refresh_clip_masks_from_fullwell` 继续执行。它仍负责在必要时重建几何对齐的 sensor mask、合并 processing loss，并使 resized mask 与 RAW guidance 缓存失效。GUI 导出复用既有 `Analysis` 时，也仍须在新解码的 bundle 上重放该 refresh。方向、镜头矫正、DefaultCrop、feather 顺序和 Apple 独立参考样本的几何关系均保持原流程。

## 缓存资格、失效与生命周期

缓存只挂在当前 `RawEvidence` 的私有字段上，没有全局缓存或新的磁盘缓存。只有调用方的 RAW／颜色数组与 evidence 中的对象相同，而且整个 ndarray base 链只读、最终 owner 确实为 bytes，才具备复用资格。手工创建的可写 bundle、仅将可写 owner 的视图标成只读、替换了 RAW 数组或白点的 bundle，均执行原统计函数。

只读数据不意味着 ndarray 的元数据不可改变，因此签名还逐层记录对象身份、shape、dtype、strides、数据地址与布局属性。白点值、provider／版本、sample kind、统计策略常量和三个计算函数的身份也参与签名。数组布局、白点或算法发生变化时不会沿用旧统计；白平衡、曝光和目标显示亮度则不是这几项原始传感器统计的输入。

计算开始前的小型白点快照同时作为签名与实际计算输入，计算完成后再次检查资格和签名；失败或签名已变的计算不会发布缓存。缓存中的数组与函数关联使用弱引用，summary 本身不持有 RAW、scene 或 mask。它不会延长证据栅格的生命周期，预览 proxy 仍可释放完整 RAW。

`dataclasses.replace(evidence)`、`copy`、`deepcopy` 和 pickle 均丢弃私有 memo，重新使用时重新验证资格。复制或反序列化后的数组是否仍为 bytes-backed，只能以实际存储判定；可写或只有可逆 readonly 标记的副本正常回算。磁盘预览仍使用原来的显式字段，不将这个私有缓存写入 `Analysis` 或预览文件。

## 一致性验证

本步新增 40 项小型测试：[`test_sensor_summary.py`](../../../tests/test_sensor_summary.py) 的 22 项、[`test_evidence_ownership.py`](../../../tests/test_evidence_ownership.py) 的 8 项，以及 [`test_sensor_benchmark_tool.py`](../../../tests/test_sensor_benchmark_tool.py) 的 10 项。它们覆盖旧统计 oracle、G1／G2 和稀疏通道、可信堆积边界、reference 后 analysis 的复用与 refresh、可写／伪只读输入、布局与白点失效、计算失败、复制与释放，以及基准工具的比较和输出保护。

真实样片以改动前 `8cedc86` 的 master 为基线，Sigma `_SDI0150.DNG` 的 LibRaw／Apple 两条路径、Sony `DSC00225.ARW`／LibRaw，以及 Fuji `DSCF0214.RAF`／LibRaw 四条路径共 20 次测量均通过精确比较。检查范围包括加载后和分析后的 mask、processing loss、独立参考样本、完整分析与形成决策，以及 SDR、HDR base、HDR alternate 的 shape、dtype 和内容 SHA-256。比较止于未编码 master；本步不执行 JPEG／HEIF 编码，也不能由此推断压缩文件逐字节相同。

完整严格 native suite 运行 **1,668 项，1,662 通过、6 跳过**；NumPy 回退专项运行 **156 项，155 通过、1 跳过**。后者唯一跳过的是实拍样片裁切高光不足，无法判断 Core Image 高光恢复的测试；新增 40 项在两种模式下均通过。本步没有修改 Rust 源码或 ABI，继续使用已通过上一批验收的 ABI 15 扩展。

## 实测结果

2026-09-20，本机 macOS 27.2 arm64、10 个逻辑 CPU、Python 3.14.4、NumPy 2.5.2。原始分轮计时、调用数、输入及输出身份、源文件 SHA-256 汇总见 [`sensor-summary.json`](../../assets/performance/sensor-summary.json)。所有完整 RAW 测量串行运行；两侧均启用所有现有 Rust 核。

下表比较当前实现的“禁用摘要复用”与“启用摘要复用”，两侧都采用新的不可写采集存储。它隔离缓存收益；修前 `8cedc86` 则单独作为图像与决策一致性的基线。

| 路径 | 交替对照轮数 | 传感器统计次数 | SDR 中位秒数：禁用 → 启用 | HDR 中位秒数：禁用 → 启用 |
|---|---:|---:|---:|---:|
| Sigma／Apple | 5 | 2 → 1 | 6.841 → 6.656 | 7.635 → 7.470 |
| Sigma／LibRaw | 3 | 1 → 1 | 7.872 → 7.867 | 8.796 → 8.795 |
| Sony／LibRaw | 1 | 1 → 1 | 11.678 → 11.113 | 12.854 → 12.260 |
| Fuji／LibRaw | 1 | 1 → 1 | 23.581 → 23.450 | 24.581 → 24.393 |

Sigma／Apple 每次仍有两个 summary 入口，但三个旧统计函数从各执行两次变成一次。summary 累计墙钟中位数 **0.304 → 0.151 s**；其中 `detect_ceilings` 为 **0.217 → 0.108 s**，二者是包含关系。SDR／HDR 管线中位数分别减少约 **2.7%／2.2%**，五对测量的两项管线时间均下降；这是该样片的局部收益，不能外推到所有 RAW 或包含编码的导出总时间。

LibRaw 三条路径均只有一次必要首算，因此没有缓存命中带来的提速。Sigma 三轮中位数基本不变；Sony 和 Fuji 各一对只作为输出正确性及运行检查，时间差不能归因于缓存。Fuji 的 HDR 计划 headroom 为 0 EV，这一行验证现有回退结果保持一致，不代表有实际扩展动态范围的 HDR 交付性能。真实峰值 RSS 有明显波动，本步不宣称整条管线的内存高水位下降。

另以五对交替的合成数组复制，检查新采集存储没有引入额外整帧复制成本。24MP／60MP CFA 的旧复制与新存储中位数分别为 **1.589 → 1.636 ms／4.014 → 4.024 ms**，基本相同；24MP 三通道 LinearRGB 为 **6.487 → 3.264 ms**，四通道输入取前三通道为 **80.041 → 76.209 ms**。这只测数组采集表达式，排除文件读取、解码、输入生成和哈希；不能作为真实 Linear DNG 管线加速结论。首对的 shape、dtype 与逐字节输出均一致。

## 可复现的测量方法

[`benchmark_sensor_summary.py`](../../../tools/benchmark_sensor_summary.py) 复用 [`benchmark_loss_pipeline.py`](../../../tools/benchmark_loss_pipeline.py) 的真实 RAW 管线，每次以新进程执行全分辨率解码、分析、自动曝光及默认 AgX，形成 P3 SDR 与 800 nit 目标的 HDR pair。真实 RAW 只作为读取输入；结果写入新的 JSON 路径。

两侧均固定 `DNGSCAN_FAST=1`、`DNGSCAN_FAST_SKIP=""`，要求存在匹配的原生扩展。当前构建为 ABI 15。`--reference` 只令 SensorSummary 的缓存资格函数返回 `None`，不关闭任何 Rust 核，也不改变统计公式。这个对照隔离 memo 的收益；它仍使用当前采集存储实现，所以还需要另与改动前 checkout 比较，才能覆盖本步完整改动。

```sh
python tools/benchmark_sensor_summary.py \
  --source /path/to/photo.dng --decoder coreimage --reference \
  --out /path/to/sensor-reference-1.json

python tools/benchmark_sensor_summary.py \
  --source /path/to/photo.dng --decoder coreimage \
  --compare /path/to/sensor-reference-1.json \
  --out /path/to/sensor-memo-1.json
```

LibRaw 路径使用 `--decoder libraw`。默认 decoder 为 `coreimage`，实际选中的解码器及版本另存于决策身份中。工具没有合成图或进程内 repeats 参数；应至少运行三对独立进程，并交替先 reference、后 memo 与先 memo、后 reference 的顺序。保持机器、Python／NumPy、原生扩展和解码参数相同，避免同时运行其他整帧测试。

对旧 checkout 可从当前工具路径执行，并通过 `--repo` 选择待测代码：

```sh
python /path/to/current/tools/benchmark_sensor_summary.py \
  --repo /path/to/checkout-at-8cedc86 \
  --source /path/to/photo.dng --decoder coreimage --reference \
  --out /path/to/sensor-original-1.json
```

旧 checkout 必须提供既有 loss pipeline 工具和匹配的原生扩展。只有 `--reference` 接受缺少 `dngscan.sensor_summary` 的旧版本；普通模式缺少该模块，或模块内部缺少其他依赖，都会明确失败。

每份报告包含输入文件 `source_sha256`、commit、运行环境、原生 ABI、输出 `identity` 以及分阶段计时。`--compare` 要求 identity 完全一致；若参考报告提供输入 SHA，也必须相同。旧 loss 报告没有输入 SHA 时，只比较 identity。差异返回退出码 2。已有输出、指向不存在目标的符号链接，以及测量期间被其他进程创建的输出均不会被覆盖；无效比较报告在计算开始前拒绝。

`sensor_calls` 分别记录 summary 入口及三个旧统计函数的调用次数和墙钟。正常的 Apple reference→analysis 链路可用两次 summary 入口、一次 `detect_ceilings` 判断是否复用；禁用 memo 时两次入口各自计算。LibRaw 的一次首算仍保留。计数在 `finally` 中更新，失败调用也会计入；未捕获异常仍会中止测量，不输出成功报告。

这些计时存在嵌套：`summarize_sensor` 已包含它调用的三个旧函数，不能将四项时间相加。`pipeline_sdr_s`／`pipeline_hdr_s` 是对应阶段墙钟之和，二者共用解码、分析、自动曝光与 SDR plan，也不能相加作为一次导出总耗时。它们不是整个进程耗时，也不是独立 CPU 或 GPU 执行时间；输入与输出哈希在相关管线计时之外。

`peak_rss_mib` 是整个进程的内存高水位，包含解码、形成、输入读取和输出哈希，不能视为 summary 独占分配，也不能代表 GPU 内存。局部减少重复扫描不自动等于整条管线同比提速或峰值内存下降。报告应分别解释统计调用数、局部墙钟、完整阶段时间和 RSS；本步没有删除满阱 refresh 或合并其前后的 mask 构建。
