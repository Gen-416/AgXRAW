# 第三批第三步：精确 Rust RGB 过曝分组计数

本步将 `analysis.compute_color_clip_metrics` 中的 RGB 过曝分组计数迁移到 Rust，原生 ABI 升为 **16**。旧实现先生成完整的逐感光点阈值图与剪切布尔图，再分别生成 R／G／B 分组布尔图并归约；新核直接读取 RAW 代码值和实际颜色索引，在遍历中累计整数计数，避免这些完整栅格的临时分配。

本步只替换这一项统计的计算方式。满阱、ceiling 堆积、逐通道剪切百分比、噪声与 SNR 仍走原有算法；SensorSummary 复用和分析后一次构建最终 mask 的前两步优化继续保留。RAW 解码、镜头与空间校准、自动曝光、AgX、HDR 形成以及 JPEG／HEIF 编码参数不因本步改变。

## 数值合同与借用边界

[`sensor_rgb_clip_counts_u16`](../rust/src/lib.rs) 接受原生字节序的 `uint16` RAW 和同形状的 `uint8` 颜色索引数组，直接借用 ndarray 视图。连续、转置、负步长与零步长广播视图均可进入原生核，不要求调用方复制成连续数组。阈值和分组分别使用 256 项的小型 LUT；计算期间不构造完整阈值图、剪切图或颜色成员图。

[`sensor.rs`](../rust/src/sensor.rs) 对二维 CFA 按完整的 `period_h × period_w` 单元遍历，末端不足一个周期的行列不计入总体。周期只确定分组边界，实际通道始终读取 `raw_colors`，不假设颜色索引必然等于重复铺开的 pattern。三维 LinearRGB 则以一个 H×W 像素为单位，沿其所有通道归约，周期固定为 1×1；不会假定数组只能有三个通道。

每个感光点先按有符号整数阈值判断 `raw >= threshold`，满足时将对应的 R／G／B 位加入当前单元。G1 和 G2 共用绿色位，两个绿色感光点同时过曝仍只算一个颜色组。未被标签映射到 RGB 的实际通道对应位 0，不增加颜色组。核返回四个 `u64`，分别记录恰有 0、1、2、3 个颜色组过曝的单元数量；未过曝单元仍计入百分比的分母。

阈值策略保持 [`channel_threshold_map`](../dngscan/analysis.py) 的规则：缺失通道使用整个阈值字典的最小值，包括当前图像中未出现的通道；空字典的默认值为 0。它不同于逐通道剪切百分比的默认值，所以本步没有将两种统计强行合并。负阈值、零阈值、超过 uint16 上限但仍在 int32 范围内的阈值均保留精确比较。Python 将整数计数转换为 float64，先除以总单元数、再乘 100，保持原 `np.mean(bool) * 100.0` 的百分比算术及 Python float 输出。

Python 仅将满足精确输入合同的数组和整数元数据送入 Rust；其他 dtype、非原生字节序、未对齐输入、形状广播、空总体及特殊元数据继续执行原 NumPy 实现，保留其输出与异常语义。所有颜色标签未恰好覆盖 R／G／B 时，仍按原规则返回空字典。`DNGSCAN_FAST=0` 或显式跳过本核也使用原实现；原生调用异常继续遵守既有 auto 回退／strict 报错政策。

绑定在创建借用视图前验证维度、同形状、字节序、指针与全部步长的对齐、尺寸与地址跨度、LUT 长度和分组值，以及周期的有效性。NumPy 的 `ALIGNED` 标志会忽略长度为 1 的轴上未使用的奇数步长，因此 Python 也逐一检查步长，而非仅检查该标志。Rust 额外检查非零维度乘积和最小有符号步长，避免 ndarray 视图构造中的整数溢出。低层 API 对空数组直接返回计数，不先做负步长视图的指针调整；三维零通道输入的每个 H×W 像素归入零组。

只读借用保持到释放 GIL 的计算结束。计算按单元行划分，沿用进程的原生线程预算，小数组串行执行；每个 worker 只保存四个 `u64` 计数。线程创建失败或 worker panic 都会在显式 join 所有已启动线程后传播，返回前没有存活的后台工作，也没有对输入或外部输出数组的写入。

## 可复现的测量方法

[`benchmark_sensor_rgb.py`](../tools/benchmark_sensor_rgb.py) 提供真实 RAW 管线和合成统计两种模式。两侧都固定 `DNGSCAN_FAST=1`；`--reference` 只通过 `DNGSCAN_FAST_SKIP=sensor_rgb_clip_counts_u16` 关闭本核，普通模式清空 skip。其他 Rust 核始终启用，并要求 checkout 与其原生扩展的 ABI 匹配。当前实现要求 ABI 16；旧版基线使用自己的匹配扩展。

真实 RAW 模式复用现有 loss pipeline，每次以新进程执行全分辨率解码、分析、自动曝光、默认 AgX／P3 SDR，以及目标 800 nit 的 HDR pair。两侧都使用延迟 mask 构建，保留 SensorSummary 复用。真实 RAW 只读，输出为新的 JSON 报告；不执行 JPEG／HEIF 编码。

```sh
python tools/benchmark_sensor_rgb.py \
  --source /path/to/photo.dng --decoder libraw --reference \
  --out /path/to/rgb-reference-1.json

python tools/benchmark_sensor_rgb.py \
  --source /path/to/photo.dng --decoder libraw \
  --compare /path/to/rgb-reference-1.json \
  --out /path/to/rgb-native-1.json
```

Apple 路径使用 `--decoder coreimage`。以多个独立进程交替运行 reference→native 和 native→reference，保持输入、参数和运行环境一致，避免与其他整帧测试或编译并发。工具的 `--repeats` 只用于合成模式，不在一个进程里重复真实 RAW 管线。

改动前的 `eaa42a0` 单独提供完整输出基线。可从当前工具路径运行，并通过 `--repo` 指向旧 checkout、加 `--reference`；旧代码也以延迟 mask 方式加载，避免把上一步有意不同的中间状态混入本步比较。

```sh
python /path/to/current/tools/benchmark_sensor_rgb.py \
  --repo /path/to/checkout-at-eaa42a0 \
  --source /path/to/photo.dng --decoder libraw --reference \
  --out /path/to/rgb-original.json
```

本步比较完整 `identity`，**没有排除项**：包括加载后和分析后的 mask、processing loss、独立参考、完整分析与自动决策，以及 SDR、HDR base、HDR alternate 的形状、dtype 和内容 SHA-256。参考报告包含输入 `source_sha256` 时也必须一致；不一致返回退出码 2。比较止于未编码 master，不据此宣称压缩文件逐字节相同。已有输出、悬空符号链接和运行期间出现的同名输出均不会被覆盖；无效比较报告在计算前拒绝。

合成模式分别测试 **24MP（6000×4000）** 与 **60MP（10000×6000）**，每种尺寸依次运行 Bayer、6×6 X-Trans 和三通道 LinearRGB。输入以固定 seed 371 生成 uint16 代码值；LinearRGB 的颜色索引是零步长广播视图。报告保存输入身份、pattern、阈值、标签、各轮耗时与结果，以检查两侧条件和输出一致。

```sh
python tools/benchmark_sensor_rgb.py \
  --synthetic 6000 4000 --repeats 3 --reference \
  --out /path/to/rgb-24mp-reference.json

python tools/benchmark_sensor_rgb.py \
  --synthetic 6000 4000 --repeats 3 \
  --compare /path/to/rgb-24mp-reference.json \
  --out /path/to/rgb-24mp-native.json
```

60MP 将尺寸改为 `10000 6000`。合成模式只计 RGB 分组统计，输入生成和哈希不在计时区间；它可以说明该核的局部收益，不能替代真实相机的解码、分析和形成总时间。LinearRGB 的 MP 数表示 H×W 像素数，每像素另含三个代码值。

`sensor_rgb_calls` 同时记录 Python 统计入口和原生核的次数／墙钟；原生时间已包含在入口耗时中，不能相加。`pipeline_sdr_s` 与 `pipeline_hdr_s` 共享准备阶段，也不能相加作为一次导出的总耗时。分阶段时间不含输出哈希和编码；进程峰值 RSS 则包含整个进程的分配与哈希，不代表本核独占内存，也不代表 GPU 内存。是否降低整条管线的峰值内存，必须单独依据多轮结果判断。

## 实测结果

2026-09-20，macOS 27.2 arm64、10 个逻辑 CPU、Python 3.14.4、NumPy 2.5.2，release 扩展 ABI 16。先用未修改的 `eaa42a0` 和匹配的 ABI 15 扩展生成四条路径的基线，再以当前代码只开／关新增计数核。**20 次新进程测量的完整 identity 均与旧版一致**，没有排除任何中间状态或决策字段；原始样张 SHA-256 也一致。

全部分轮计时、调用次数、输入／输出身份、决策哈希与代码指纹见 [`sensor-rgb.json`](assets/performance/sensor-rgb.json)。以下管线总和包含解码、分析、自动曝光和形成，**不含编码**。

| 路径 | 交替对照轮数 | RGB 统计中位耗时：NumPy → Rust | SDR 中位秒数 | HDR 中位秒数 |
|---|---:|---:|---:|---:|
| Sigma `_SDI0150.DNG`／LibRaw | 3 | 358.04 → 4.14 ms | 7.791 → 7.399 | 8.735 → 8.325 |
| Sony `DSC00225.ARW`／LibRaw | 3 | 482.12 → 5.32 ms | 11.400 → 10.799 | 12.577 → 12.016 |
| Fuji `DSCF0214.RAF`／LibRaw | 3 | 475.83 → 4.09 ms | 24.282 → 24.023 | 25.212 → 25.015 |
| Sigma／Apple | 1 | 731.57 → 3.78 ms | 7.721 → 6.555 | 8.601 → 7.424 |

Sigma／Sony 的三对测量中，SDR 和 HDR 每一对均减少。管线中位数分别下降 **5.0%／4.7%**（Sigma，SDR／HDR）和 **5.3%／4.5%**（Sony）；这一比例只描述本机、本样张及未编码阶段，不能直接套用于完整 JPEG／HEIF 导出。Fuji 的三对也均减少，但解码占时较大，SDR／HDR 中位数只下降 **1.1%／0.8%**。此 Fuji 场景的 HDR headroom 为 0 EV，双路结果用于渲染器验证，不代表扩展动态范围的交付。

Apple 只测一对，用于独立参考与完整路径的正确性验证；不把这一对较大的整体时间差当作稳定收益。各路径两侧都只调用一次 RGB 分组统计，原生侧该次调用进入新核。进程峰值 RSS 没有一致下降，因此不宣称整条管线峰值内存降低。

合成统计的三次进程内重复中位数如下。每种尺寸的两侧都以新进程执行，先 reference 后 native；输入和全部百分比结果一致。它们是计数入口的局部耗时，不包含生成输入、哈希、解码和渲染。

| 像素数 | 布局 | NumPy | Rust |
|---|---|---:|---:|
| 24MP | Bayer | 356.06 ms | 3.99 ms |
| 24MP | X-Trans | 260.02 ms | 2.51 ms |
| 24MP | LinearRGB | 1334.18 ms | 8.98 ms |
| 60MP | Bayer | 869.67 ms | 9.84 ms |
| 60MP | X-Trans | 682.07 ms | 6.12 ms |
| 60MP | LinearRGB | 3450.59 ms | 21.71 ms |

## 回归结果

release 扩展构建、ABI 16 自检与 macOS 签名验证通过；Rust library 单元测试 **7 项全部通过**，其中新增 3 项覆盖实际颜色／双绿合并／残边、LinearRGB 有符号阈值及反向／广播步长、多线程分块与空通道总体。

严格 native 完整 suite 运行 **1,729 项，1,723 通过、6 跳过**，无失败或错误。6 项沿用已有条件：一项实拍裁切高光不足，三项缺少胶片 LUT，一项未配置 ideal-image 数据目录，一项旧胶片组合表未暴露；新增测试没有跳过。

NumPy 回退专项运行 **236 项，220 通过、16 跳过**，无失败或错误。其中 15 项是既有 AgX／输出核测试的装饰器因 `DNGSCAN_FAST=0` 主动禁用 native 而跳过，日志中的通用原因虽写作“native extension not built”，本机扩展实际已经构建；这些项已在严格全套中通过。另 1 项仍是实拍高光不足。新增 38 项在两种模式下均通过；其中直接绑定合同测试会局部启用 native，以验证低层 API，不把它们描述为全程 NumPy 计算。

新增 [`test_sensor_rgb_native.py`](../tests/test_sensor_rgb_native.py) 的 28 项测试，以冻结的 NumPy oracle 检查逐位百分比、RGB 数学、dispatch／fallback、稀疏及未知 CID、缺失阈值、整数边界、借用布局、输入不变、空总体、绑定预检与错误政策。新增 [`test_sensor_rgb_benchmark.py`](../tests/test_sensor_rgb_benchmark.py) 的 10 项测试覆盖单核消融、两侧延迟 load、完整身份与源文件哈希比较、旧版 reference、调用计时、合成输入和禁止覆盖输出。
