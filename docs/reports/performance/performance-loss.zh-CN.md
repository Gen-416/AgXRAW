# 第二批管线效率改进：loss evidence 的裁切与合并

本批将 RAW 可靠性掩码中的两处运算移入 Rust，原生 ABI 升至 **15**。优化对象是分数坐标裁切的 footprint maximum，以及处理损失向最终 half 掩码的合并。RAW 解码器选择、分析规则、自动曝光、AgX、HDR 形成和编码参数均沿用原流程。

`_merge_processing_loss` 原先已经通过 `np.maximum(..., out=masks)` 原地写回。本批消除的是 resize 中间栅格和 NumPy 运算开销，不能将其描述成“首次改为原地处理”。

## 实现与数值合同

| 原生入口 | 输入与输出 | 保留的运算顺序 |
|---|---|---|
| `crop_loss_footprint(values, ylo, yhi, xlo, xhi)` | 借用 H×W×3 的 float16／float32 输入；输出保持输入 dtype | 输出从正零开始；按旧实现的 128 行 band、dy 后 dx 的顺序取 maximum；无效 footprint 样本仍按正零参与 |
| `merge_processing_loss_f16_inplace(masks, processing, y_indices=None, x_indices=None)` | `masks` 为连续、可写的 float16；借用 float16／float32 processing；成功时返回原 `masks` 对象 | 同尺寸直接 maximum 后写入 half；有 resize 时先将最近邻样本写成 half，再与 half 掩码取 maximum |

裁切坐标继续在 Python 中生成，保留 `linspace`、`floor/ceil` 和 `1e-9` 边界处理。Rust 只执行索引后的取样与 maximum，不重新推导像素覆盖范围。它避免原实现逐轮创建广播索引结果、有效性数组和 `where` 临时数组；最终输出栅格仍然需要分配。

处理损失的几何缩放仍使用 Pillow 的最近邻选择。Python 用 mode-I 的一维行、列坐标条获得准确索引，再交给 Rust；没有用近似的坐标公式替代 Pillow。这样只需 O(H+W) 索引存储，不再为 resize 构造完整的 half RGB 中间栅格及逐通道 float32 图像。

生产入口只将同尺寸 float16 合并、不同尺寸 float16／float32 合并交给 Rust；同尺寸 float32 processing 继续使用原 NumPy。24 MP 的三次局部测量中，这条 NumPy SIMD 路径约 0.011 s，而包含特殊值预检查的 Rust 实现约 0.064 s；60 MP 同样为约 0.027 s 对 0.157 s。因此本批保留较快的既有实现。原生 API 仍支持该组合，独立测试继续检查它的数值合同。

两核都支持借用对齐的只读、负步长或广播输入，不为接入 Rust 强制复制整帧，也不将 half 输入统一扩成 float32。合并目的地保留调用方所有权。管线仍然**先 feather sensor loss，再合并 processing loss**，因此处理过程记录的裁切、外推或饱和损失不会被 feather 稀释。方向变换、DefaultCrop 与现有 RAW 证据坐标关系保持不变。

Rust 运算释放 GIL，使用现有线程预算。尺寸、dtype、索引范围、对齐、可写性和内存重叠在原生写入前检查；合并核也在写入前完成不支持数值的检查。线程启动失败不会留下部分写入的掩码。

## 精度、回退与故障

验收使用逐位比较，包括正负零和 NaN payload。`maximum` 的特殊值行为不能用视觉一致或 `allclose` 替代，也不能将某台机器上的 float32 SIMD／尾部行为硬编码为所有平台的规则。

float16 的直接 maximum 保留原始 half 位模式。对于 float32 NaN、部分负零组合，以及 resize 中涉及 NaN 的 half／float32 转换，原生核可以返回 `None`，表示该数值情形不受支持；合并目的地必须完全未变。Python 随后执行保留的旧 NumPy／Pillow 实现，严格 native 模式也允许这种明确回退。无 NaN 的正负无穷有独立测试。

不支持的 dtype、非本机字节序、未对齐输入、非连续或不可写的合并目的地，以及可能与目的地共享内存的 processing，继续按旧 Python 路径处理。直接调用原生 API 的非法参数会报错，并且不能先写入一部分结果。端序检查由 NumPy Rust 绑定的 dtype 等价检查完成，测试覆盖 swapped float16、float32 和 intp 索引。

显式不支持与执行异常是两个不同合同：`None` 允许回退；原生执行异常在自动模式记录警告并回退，在 `DNGSCAN_FAST=1` 下抛出 `NativeKernelError`。`DNGSCAN_FAST_SKIP=crop_loss_footprint,merge_processing_loss_f16_inplace` 可以只关闭本批两核，其他原生核继续运行。

## 一致性验证

[test_loss_native.py](../../../tests/test_loss_native.py) 保留了独立的旧 NumPy 裁切和 Pillow resize 基准，覆盖：

- 奇数裁切、epsilon 边界、单行／单列、窄图、空输出的原有结果或异常，以及 LibRaw 方向 0–7。
- float16／float32 逐位结果、NaN payload、正负零、无穷和 half 舍入顺序；subnormal、偶数舍入、溢出边界及原生明确回退后的公共入口结果。
- 只读负步长／广播视图、非本机端序、未对齐输入、相同数组和重叠视图，以及不同 Python owner 包装同一地址的别名。
- 非法参数与末尾才出现的不支持值均在合并写入前退出；线程预算 1、2、3、8 的结果一致。
- 实际 dispatch、skip key、dtype 与对象身份，以及自动／严格模式的异常处理。

独立 oracle 与 dispatch 共 22 项，另有 12 项基准工具测试；其中直接 API 合同测试始终调用扩展，公共入口按各自模式执行。完整严格 native suite 运行 **1,628 项，1,622 通过、6 跳过**；Cargo 4 项全部通过，其中 3 项属于新 `loss` 模块。

最终 RAW／证据专项两种模式各运行 182 项：NumPy 166 通过、16 跳过，严格 Rust 181 通过、1 跳过。共同跳过的是样片裁切高光不足的 Core Image 高光恢复测试；NumPy 额外跳过 15 项要求启用 native 的既有测试。本批新增的 34 项在两种模式下均通过。后续 SensorSummary 批次复核了该跳过条件；此前将其归因于缺少 reference-load 样片的说明不正确。

最终版本的四条真实解码路径共 16 次测量中，加载后与分析后的掩码、processing loss、独立参考样本、分析／计划决策以及 SDR／HDR master 的身份均与原 `fe13ba5`／ABI 14 一致。这项证据的范围止于形成结果：本次没有执行 JPEG／HEIF 编码，不能据此声称压缩文件逐字节相同。

## 可复现的测量方法

[benchmark_loss_pipeline.py](../../../tools/benchmark_loss_pipeline.py) 每次调用使用一个新进程。两侧都要求匹配的原生扩展；`--reference` 仅关闭本批两核，保留其他 Rust 核，从而将收益归因限制在本批改动。工具会设置对应的 `DNGSCAN_FAST` 和 `DNGSCAN_FAST_SKIP`，不能用外部环境变量将参考侧替换为全 NumPy 管线。

在仓库根目录使用同一个 Python 环境运行。例如，对一张 RAW 做 LibRaw 全分辨率对照：

```sh
python tools/benchmark_loss_pipeline.py \
  --source /path/to/photo.dng --decoder libraw --reference \
  --out /path/to/loss-libraw-reference-1.json

python tools/benchmark_loss_pipeline.py \
  --source /path/to/photo.dng --decoder libraw \
  --compare /path/to/loss-libraw-reference-1.json \
  --out /path/to/loss-libraw-native-1.json
```

Apple 路径改用 `--decoder coreimage`，实际解码器及版本保存在结果中。可通过 `--repo /path/to/AgXRAW` 指定 checkout。输出 JSON 必须是新路径，工具不会覆盖已有报告；`--compare` 要求全部 `identity` 一致，差异使进程以状态码 2 退出。

真实 RAW 模式执行完整分辨率解码、分析、自动曝光和默认 AgX，分别生成 P3 SDR 与 800 nit 目标的 HDR pair。报告保留掩码、参考样本、分析与形成决策及 master 的 shape、dtype、SHA-256，同时记录两处 Python 入口和对应原生入口的调用次数与时间。调用次数可以判断样片是否真的经过新核：没有命中的路径不能用来证明该核提速。

大尺寸局部测量使用合成模式，例如 6000×4000 的 24 MP：

```sh
python tools/benchmark_loss_pipeline.py \
  --synthetic 6000 4000 --repeats 3 --reference \
  --out /path/to/loss-24mp-reference-1.json

python tools/benchmark_loss_pipeline.py \
  --synthetic 6000 4000 --repeats 3 \
  --compare /path/to/loss-24mp-reference-1.json \
  --out /path/to/loss-24mp-native-1.json
```

`--synthetic` 的参数顺序是宽、高；10000×6000 为 60 MP。该模式分别测量 float16／float32 的 crop、同尺寸 merge 和 resize merge，使用可精确表示的普通非负样本。它验证常规路径的计算与内存成本，不代表特殊值回退成本。`--repeats` 只控制合成模式同一进程内的重复次数；真实 RAW 模式每次调用只执行一轮。

正式比较应至少做三轮独立进程的交替顺序测量，保持同一机器、Python／NumPy、原生构建和解码器设置，并避免同时运行其他整帧测试。每轮使用新报告名，并逐轮比较身份。报告记录的 `pipeline_sdr_s`／`pipeline_hdr_s` 是对应阶段计时之和，不是整个进程墙钟；两者共享解码与分析，不能相加为一次导出时间。数组哈希、合成输入生成和 merge 目的地初始化不计入各操作时间。

`peak_rss_mib` 是整个进程的内存高水位，包含输入准备和哈希过程，不是某一行代码的存活分配量，也不能解释为本批两核独占的峰值。裁切与合并的局部时间、真实管线阶段时间和 RSS 应分别报告；局部临时数组减少并不保证整条管线出现同等比例的内存下降。

## 最终测量结果

环境为 macOS 27.2 arm64、10 个逻辑 CPU、Python 3.14.4、NumPy 2.5.2。参考侧与优化侧使用同一份 ABI 15 release 扩展，只切换本批两个核；另与 `fe13ba5`／ABI 14 原实现的完整输出身份比较。所有测量串行进行。原始输入 SHA-256、每次计时、范围、调用数及输出身份见 [loss-kernels.json](../../assets/performance/loss-kernels.json)。

| 样张／解码器 | 次数 | SDR 原版 → 本批 | HDR pair 原版 → 本批 |
|---|---|---:|---:|
| Sigma `_SDI0150.DNG`／LibRaw，24 MP | 三轮交替，中位数 | 9.067 → 7.755 s | 9.967 → 8.685 s |
| 同一 Sigma／Apple | 三轮交替，中位数 | 7.262 → 6.938 s | 8.007 → 7.744 s |
| Sony `DSC00225.ARW`／LibRaw，约 33 MP | 单对 | 11.645 → 11.182 s | 12.762 → 12.346 s |
| Fuji `DSCF0214.RAF`／LibRaw，约 40 MP | 单对 | 23.782 → 23.506 s | 24.718 → 24.446 s |

Sigma／LibRaw 的 SDR 耗时减少 **14.5%**，HDR pair 减少 **12.9%**；SDR 原版范围 8.809–9.080 s，本批 7.649–7.848 s，HDR 原版 9.722–9.997 s，本批 8.572–8.753 s。Apple 路径分别减少约 4.5% 和 3.3%。Sony、Fuji 的单对数字仅作样本记录，不据此宣称稳定的整体百分比收益。此 Fuji 场景的 HDR 预算为 0，表内是双路渲染验证，不能视为实际 HDR 文件导出性能。

Sigma／LibRaw 每轮两次裁切的合计中位数从 0.869 s 降至 0.0324 s，两次合并从 0.485 s 降至 0.1149 s。Apple 每轮一次裁切从 0.230 s 降至 0.00742 s，合并从 0.0603 s 降至 0.01255 s。Sony／Fuji 样张没有命中分数裁切，不用它们证明裁切核提速。

下表是同进程重复三次的局部中位数；输入生成、目的地重置和哈希不计时。24 MP／60 MP 全部六组输出均逐位一致，局部倍率不能套用到整条 RAW 管线。

| 操作 | 24 MP 原版 → 本批 | 60 MP 原版 → 本批 |
|---|---:|---:|
| crop float16 | 2.140 → 0.0779 s | 5.117 → 0.1883 s |
| crop float32 | 1.691 → 0.0426 s | 4.190 → 0.0978 s |
| 同尺寸合并 float16 | 0.2426 → 0.0360 s | 0.6066 → 0.0875 s |
| resize 合并 float16 | 0.2475 → 0.0285 s | 0.6076 → 0.0644 s |
| resize 合并 float32 | 0.2533 → 0.0263 s | 0.6045 → 0.0696 s |
| 同尺寸合并 float32，保留 NumPy | 0.01085 → 0.01091 s | 0.02706 → 0.02714 s |

合成进程高水位在 24 MP 为 1110→886 MiB，60 MP 为 2676→1954 MiB，包含依次运行全部六组及输入准备。真实 Sigma／LibRaw 的高水位中位数为 2803→2808 MiB；Apple 为 2508→2588 MiB，且各次波动较大。**本批没有证明真实全管线峰值内存下降**，主要已确认收益是缩短遮罩处理时间。

后续的 SensorSummary 共享、full-well 前后重复 mask build、整帧生命周期及其他分析统计仍是独立批次；本批没有删除 full-well refresh，也没有改变 Apple 对 LibRaw 参考证据的依赖。
