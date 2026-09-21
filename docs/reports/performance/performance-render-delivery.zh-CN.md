# 第四批：渲染与交付的有界内存、精确可选核（ABI 18）

本记录针对非胶片 SDR/HDR 管线，原版对照为 `9c8a447` / ABI 17。
默认曝光、AgX 曲线、HDR 内容预算、编码质量搜索与验收门槛保持原定义。
完整实拍矩阵、GUI 缓存、分析与 I/O 改动汇总见
[管线收尾记录](performance-pipeline-completion.zh-CN.md)。此处记录渲染与交付部分的合同、独立测量和未采用方案的依据。

## 渲染与存储合同

HDR pair 的 SDR 分支在不需要 display highlight chroma retreat 时，直接把 Rec.2020 RGB
交给已有 `finalize_rec2020_u8_f32`。它合并原来的两次颜色矩阵与 output finalizer；
其余颜色分支保留原执行路线。量化仍以原来的 1M 像素组组织，每组完整生成 A 噪声后再生成
B 噪声；已有 native output plan 时，直接把每个 render chunk 与对应噪声切片交给
finalizer，省去将两个 500k RGB 临时块 concatenate 成 1M 块的复制。整组噪声生成顺序、
调用者提供的 A/B 或 TPDF 噪声及最后不足一组的尾部保持原定义。原 NumPy 与胶片空间
分带路线继续按原分组量化；每组结束释放噪声切片，避免跨组滞留。native 自动回退会先恢复 output
空间转换，再执行旧 NumPy 末端，strict 失败仍向上传播。

相同几何、连续的 float16/float32 masks 由 `clip_masks_for_render` 持有借用视图，
实际采样或 render chunk 才转换为 float32 并 clamp。它不创建完整 float32 mask 缓存。
需要 crop、resize 或非连续布局时，继续使用公开 `clip_masks_for_shape` 的原采样定义。
tone、AutoEV 和 GUI 的先行采样也须使用这一入口，才能避免在 render 开始前升精度整图。

生产 HDR 导出使用 `render_ultrahdr_agx_pair_packed`：保留每个 render chunk 的 float32 HDR，
直接 clip/转换到最终 float16 RGBA；不保留完整 float32 HDR。公开原 pair 仍返回 float32，
方便既有调用者与逐位对照。alpha 固定为 half 1。独立 `to_gainmap_alternate` 也改为 128 行
转换，避免整幅 clipped RGB 和临时 half RGB 同时存在。

实际 headroom 仍是 **打包前 float32 RGB 每像素最大值的 p99.99**。
`_HeadroomTail` 保留精确 order statistic 所需的上尾 K 个值，最终使用原 NumPy float32
百分位插值顺序；并不把 half 的最大值或容器容量当成这一统计量。容器的 half 统计及
roundtrip 指标仍走自己的验收路线。NaN、Inf 与部分最后一块有独立回归测试。

Core Image HDR 读回直接返回持有私有 bitmap owner 的只读 memoryview/NumPy view，
避免 `bytearray → ndarray.copy()` 的整图复制。不同读回不共享 owner；没有增加跨文件缓存。

## ABI 18 与可选计算

`atrous_smooth_f32(plane, level)` 只接受原生 endian、对齐的二维 float32 NumPy array。
绑定在构造借用 view 前验证形状、元素数、stride/span 和 level，支持负、零、转置步长；
空数组直接返回对应形状的空结果，不构造借用 view。持有只读 borrow 后释放 GIL 计算，
输出是独立 owner，输入不写入。生产入口对原有空/其他类型输入保留 NumPy 语义。

Rust B3 先处理 Y 轴，再处理 X 轴；每轴按 `[1,4,6,4,1]/16` 顺序执行五次 float32
乘法和累加，不使用 FMA 重排。reflect 索引覆盖单轴长度 1 和超过图像尺寸的 hole spacing。
内核按 CPU budget 划分行，预算 1 直接运行，多线程显式 join 后才返回。
仅替换 B3 平滑：整个 decimated channel 的 MAD、garrote、层级选择和最后零亮度投影
仍沿用 NumPy。amount=0 的默认路径仍完全不调用 NR。

`gamut_counts` 追加可选 `median=None` 参数，兼容原有位置调用。分析阶段已经持有同一
float32 luminance population 的 median 时可直接传入，省去 Rust 的完整 y copy 和再次
partition。没有共享统计量时仍使用原 median 路线，输入形状与矩阵验证不变。

RAW guidance 的普通二维 uint16 mosaic 路线，以 CFA 周期对齐的最多约 128 行构建证据，
直接输出 CFA-binned float32 map，不再创建完整 RAW 尺寸的 pseudo-RGB float32 map。
各 CID 的 fullwell、black、electrons、SNR 与 smoothstep 运算顺序和 dtype 保持不变；
G1/G2 仍归并到 G，unknown CID 不贡献。周期归并按同一 row-major sensel 顺序逐 offset
执行 minimum，避免跨交错 axes 的昂贵归约。方向、crop、geometry alignment、resize、
完整 CFA cell 与残边规则均保留。带 spatial black 的输入以及特殊表示继续使用原构建体。
存储为 half 后再计算 raw permission 的顺序不变，缺少可信 SNR prior 时仍返回原 scene
fallback，不伪造置信信息。

## 编码与交付合同

原计划 §4.2 的准备阶段现由私有 `_PreparedGainmapMaster` 负责。每次公开 writer 调用
建立一次 SDR uint8 / HDR half 的私有连续快照，以 immutable bytes 为 owner，阻止调用者
的可写别名在候选之间改变母版。按 512 行检查 HDR 所有四通道的有限性，同时取得 RGB
实际最大值；后续候选不重复扫描。一次构造 base RGBA、两个 CIImage/NSData、P3 色彩
空间和编码 CIContext，整个搜索复用这些 owner，保留原 `CacheIntermediates=False`。

公开 writer、auto 调度、manual candidate 和单次编码已经拆开。候选与辅助图精度重试
不会递归进入完整 writer；每次只生成当前质量/采样/辅助精度的 options，并继续执行
全部容器、SDR、HDR 验收。手动 HEIF 辅助精度仍为 95 后按 97/98/99/100 重试，JPEG
仍可从 95 升到 100；auto 首个 reference 的升级与参考模板策略不变。现有 HEIF primary
内容身份缓存和 JPEG 当前 codestream 缓存保持原范围，没有缓存所有候选或解码图像。

prepared token 校验母版对象、shape/dtype/strides/data pointer/只读属性和 headroom 身份，
拒绝外来母版及已经关闭的会话。构造中途失败也执行 close；正常返回或验收失败时关闭
CIImage/NSData/context，再释放其对应 NumPy owner 与指标 workspace。没有模块全局的
prepared cache。测试覆盖 alpha 非有限值、最后不足一带的最大值、可写源修改、外来/
过期 token、一次 setup 多次独立验证，以及 manual 重试与失败保留目标文件。

这一步减少的是 **候选之间重复准备**。为了冻结之前直接借用的 HDR 输入，新快照会
增加 8 字节/像素的常驻输入（24MP 为 192MB，约 183.1MiB）；不能单独宣称峰值内存降低。
本页前述 buffer/metric 微基准早于此 prepared refactor，不能作为其最终编码时长或
峰值数据。包含该改动的真实编码、候选和文件身份对照由完整管线记录登记。

HEIF 的 uint8 输入在 8-bit 编码中直接借用连续 band；10-bit 使用不可写的 256 项 LUT。
LUT 由旧 float32 `/255`、`*1023`、`rint` 生成，全部 256 个码值逐位相同，不换成另一个
整数舍入公式。其他 dtype 保持原 float32 量化，像素平面 stride 与 libheif 交接方式不变。

JPEG auto 交付会持有一个只读 SDR master，缓存当前 `(quality,chroma)` 的 primary
codestream。auxiliary quality 重试可复用同一主图；换质量或采样方式仍重新编码。
APP 元数据、ICC、ISO gain map、MPF index 的重定位和最终验收继续走原 repacker。
缓存只存在于一次导出，不保留多档 JPEG，也不跨 job 复用。

`DeliveryTransaction` 在目的目录所在文件系统内创建私有候选目录。
HDR 和 SDR auto 的候选搜索、原有 metadata 内容完整性门槛、可选最终像素读回均在私有
路径完成，成功后一次 `os.replace` 发布到用户路径。后期失败保留原文件并清理候选。
metadata 仍为原有 best effort：不改成硬性成功条件；质量搜索与 savings 仍采用原来的
编码阶段大小，不因事后 metadata 大小重新选择候选。手动 SDR 路线已有同类私有候选
替换机制，本步不改变其策略。事务不额外宣称断电后的持久化保证。

## 共享交付扫描与工作区（原计划 §6.5）

`base_and_coding_metrics_u8(decoded, intended, sum_buffer_size=None)` 同时返回原有五个
base 指标和三个 coding 指标。输入是只读借用的原生 uint8 H×W×C 数组（C≥3），
支持 RGB-of-RGBA、负/零/转置步长，忽略 alpha。普通 C 空间布局只扫描一次像素，
保持原 128 行 coding band、float32 luma/chroma 运算、float64 求和顺序、完整及残缺
8×8 块的均值规则；base 原整数直方图/偏差统计不改变。固定大小的 band scratch 替代
两次独立像素扫描，不构造整幅 RGB repack 或 float32 差图。特殊空间 Fortran 布局
保持原 Python coding 归约次序；该分支仍可使用原生 base 扫描。

NumPy 的 float32→float64 归约分段受到当前 `np.getbufsize()` 影响；新入口默认读当前值，
显式参数必须与之相同且为正，否则拒绝。不会临时修改全局 NumPy 设置。测试覆盖 32
和 8192 两档。coding 的上尾插值与旧 HDR 百分位 helper 有不同的舍入细节，分别保留
各自定义，避免复用错误公式引入一个 float32 ULP 的变化。

HDR 的 `HdrMetricsWorkspace(max_bytes=268435456)` 在一次导出内复用候选之间的
统计缓冲区容量。median/p95/p99/p99.9 收集其必需秩后共享递归 partition，避免多次
对同一数组重复选择；chroma p99 只选择保留上尾中的必要秩，不再排序整个上尾。
仍保持旧 HDR native 的每个 float32 运算、8×8 块、掩码、NaN/Inf 拒绝与全部 12 个
指标。原 `hdr_roundtrip_metrics` 留作独立对照入口；不改变既有 NumPy/native 的已知
尾差合同，也不改变任何候选验收门限。

workspace 保存的是 **容量，不是图像、结果或跨任务缓存**。每次调用前后清除有效长度
和累加状态；默认最多保留 256MiB，可设为 0，接口上限 512MiB。单次大图仍可临时使用
超过保留上限的 scratch，计算结束即释放。一次 export 的多个候选及 gain-map 重试复用
同一 workspace，函数退出后自然释放。HEIF `PrimarySearchSession` 的压缩内容身份与
结果缓存仍使用原合同；workspace 不参与身份判断，不跳过新候选的像素验证。

绑定在 borrow 前检查 native endian、data pointer/每轴 stride 对齐、形状、isize 范围
和地址 span；空输入不构造借用视图。计算期间只读借用持有源 owner，释放 GIL 后才
获取 workspace 锁，并在重新获取 GIL 前释放锁。共享 workspace 的并发调用串行隔离，
原生工作线程全部 join 后再返回。forcecast 适配器针对 NumPy 同 dtype 但未对齐的输入
单独 copy；普通对齐输入仍零复制，严格借用接口则拒绝未对齐输入。

生产入口在 gainmap 自动候选和 SDR HEIF 验证中合并 base/coding 扫描，Core Image 主图
读回可供内部指标借用私有 RGBA owner 的只读 RGB 视图；公开 `read_primary_rgb_u8` 默认
仍返回 C 连续 RGB。最终请求返回给调用者的 SDR HEIF RGB 仍保持 C 连续。
FAST=0、不适用类型/布局与指定 SKIP 保留原参考路线；strict 仅对已适用原生调用的失败
报错。`DNGSCAN_FAST_SKIP=base_and_coding_metrics_u8,HdrMetricsWorkspace` 可单独消融本步，
已有 `base_roundtrip_metrics`、`hdr_roundtrip_metrics` 的 SKIP 也会禁用对应合并路线。

## 独立测量

环境为本机 Apple Silicon、Python 3.14、NumPy 2.5.2；重任务串行。
下表都是三个重复的中位数。buffer 测试在每一侧使用新进程，24MP 为 6000×4000。
时间只覆盖相应操作，不含输入生成与输出 SHA。RSS 是进程累计高水位，包含输入与输出
owner、模块导入和一致性散列；不能当成单个 kernel 的精确工作集。

| 独立操作 | 原路线 | 新路线 | fresh-process RSS 原→新 |
|---|---:|---:|---:|
| HDR p99.99 headroom | 309.5 ms | 269.2 ms | 与下一行同进程 |
| HDR half RGBA packing | 132.6 ms | 79.9 ms | 1108.7→533.8 MiB |
| HDR readback owner | 12.87 ms | 0.0081 ms | 418.3→235.2 MiB |
| HEIF uint8→8-bit planes | 73.14 ms | 4.72 ms | 223.0→189.8 MiB |
| HEIF uint8→10-bit planes | 77.10 ms | 62.40 ms | 291.7→269.8 MiB |

以上四项全部输出数组 SHA 相同；packing 同时核对精确 headroom。
HDR packing 表采用固定完整 float32 输入；生产 pair 的逐 chunk packing 还能消除这个
完整 float32 owner，其实际导出收益应看完整管线记录，不能将两种测量直接相加。

同一 24MP、预算 8 下，交付指标使用新进程原/新入口，输入生成与 SHA 在计时之外，
一次 cold 调用后重复三次。base 参考包含实际 RGBA 读回所需的 RGB repack，再运行旧
native base 与 NumPy coding；新路线直接借用 RGBA。HDR 参考是原 native 独立调用，
新路线为同一 export workspace 的重复调用。所有源数组 SHA 和指标的 `float.hex()`
逐字段一致。精简原始记录及代码阶段见 [渲染与交付测量](../../assets/performance/render-delivery.json)。
本表是生产 dispatcher 接入前的最终数值核独立测量；后续增加了对齐/ambient buffer
边界检查，普通对齐输入的计算体不变。不能把独立 kernel 数据称为最终完整编码时长。

| 24MP 交付扫描 | 原路线 | 新路线 | 减少 | fresh-process RSS 原→新 |
|---|---:|---:|---:|---:|
| base + coding | 183.90 ms | 91.80 ms | 50.1% | 318.7→230.7 MiB |
| HDR 全部指标 | 266.99 ms | 227.32 ms | 14.9% | 572.7→572.8 MiB |

HDR workspace 在本例保留 149,053,056 字节（约 142.15MiB）；该项提升是减少分配/重复
partition，不宣称降低峰值 RSS。旧/新 cold 分别为 270.74/230.37ms。

量化分组单独测 1M 像素、两个 500k 输入块，每档预算使用两个独立进程，各三次重复。
完整 A/B 噪声预先生成，所有输入、噪声和输出 SHA 相同；两侧写入同大小预分配 u8
master。原始外部采样及身份见 [渲染与交付测量中的 quantize 项](../../assets/performance/render-delivery.json)。

| native 预算 | concatenate | 两个 slice | 外部采样 RSS 原→新 |
|---|---:|---:|---:|
| 1 | 228.84 ms | 227.95 ms | 103.94→90.41 MB |
| 5 | 58.69 ms | 58.53 ms | 103.87→90.51 MB |
| 8 | 42.46 ms | 42.54 ms | 104.09→90.64 MB |

延迟变化均在 ±0.4% 内；采用两 slice 的依据是免除 RGB concat 和缩小临时 u8 输出，
实测工作集减少约 13.5MB，而非宣称速度改善。外部采样 RSS 可能漏掉短暂峰值，与
`ru_maxrss` 高水位不是同一指标；MB 与前表 MiB 的单位也有区别。

可选路径用固定 synthetic fixture、单线程预算隔离算子。ChromaNR 输入为 1408×938
float32 RGB、amount=0.6；RAW guidance 输入为 3000×2000 uint16 Bayer，prior 已校准，
headroom/SNR 的目标 map 为 1500×1000。参考侧只替换 B3 与 guidance 构建体，其他 native
继续开启。结果同样按完整输出 SHA 判等。

| 可选入口 | 原 NumPy 体 | 新入口 | 减少 |
|---|---:|---:|---:|
| 完整 ChromaNR，decimation factor=1 | 142.71 ms | 122.89 ms | 13.9% |
| 完整 ChromaNR，factor=6000/1408 | 148.90 ms | 134.47 ms | 9.7% |
| RAW headroom map | 163.60 ms | 69.09 ms | 57.8% |
| RAW SNR map | 202.46 ms | 101.02 ms | 50.1% |

Rust B3 初版逐像素坐标计算曾使 NR 变慢，已被逐行实现替换；该初版结果不作为收益依据。
优化后的 NR 仍受到全通道 MAD 限制，不以逐 tile MAD 换取速度。

复现 buffer 消融，例如：

```sh
python tools/benchmark_delivery_buffers.py --operation pack --reference --out /tmp/pack-before.json
python tools/benchmark_delivery_buffers.py --operation pack --compare /tmp/pack-before.json --out /tmp/pack-after.json
```

`--operation` 另有 `readback`、`plane8`、`plane10`，每个操作须使用独立输出路径。
两侧维度必须相同；默认 24MP，可用 `--size WIDTH HEIGHT` 指定。输出路径必须不存在。

```sh
python tools/benchmark_optional_render.py --reference --out /tmp/optional-before.json
python tools/benchmark_optional_render.py --compare /tmp/optional-before.json --out /tmp/optional-after.json
```

optional 工具同时记录各入口独立计时、额外一次 cProfile 和完整 SHA；cProfile 不计入
计时重复。默认 `--repeats 3`，形状与随机种子记录在 JSON 中。它需要匹配的 ABI 18。

交付指标与量化分组复现：

```sh
python tools/benchmark_delivery_metrics.py --operation base --reference --out /tmp/base-before.json
python tools/benchmark_delivery_metrics.py --operation base --compare /tmp/base-before.json --out /tmp/base-after.json
python tools/benchmark_delivery_metrics.py --operation hdr --reference --out /tmp/hdr-before.json
python tools/benchmark_delivery_metrics.py --operation hdr --compare /tmp/hdr-before.json --out /tmp/hdr-after.json
python tools/benchmark_quantize_groups.py --out /tmp/quantize-groups.json
```

metrics 默认 `--size 6000 4000 --workers 8 --repeats 3`，workspace 可用
`--retention-mib` 控制保留容量；quantize 默认预算 1/5/8、各三次，不改变生产 dispatcher。
两工具均要求输出文件不存在。

本轮定向回归包括 15 个 Rust tests；渲染/交付/可选路径/staged writer/SDR HEIF 的
50 个 Python tests 通过（1 个平台集成条件跳过）。另外对原 native/HDR 合同执行了
77 项回归（8 项跳过）：初次仅 ABI 常量仍钉在 17 失败，修为 18 后该项单测通过。覆盖 tiny/partial chunks、
half/stride/owner 生命周期、全部 u8 LUT、交易失败保留原文件、JPEG donor 重试、B3
跨预算与反射边界、完整 NR、2×2/6×6 CFA、G1/G2、odd edges、方向、prior 和 half permission。
全项目及实拍最终结果由管线收尾记录统一登记。

## 测量后保留原路线的候选

以下取[测量记录](../../assets/performance/render-delivery.json)中
`optional / remaining-optional-abi18-final.json` 的同轮结果。每次空 native 调用的上界
包含 plan 解析、Python/FFI、输入校验与空输出分配。AgX/HDR/output 分别约
3.008/4.419/2.125 μs；对应 500k 像素块约 94.87/125.95/125.75 ms。
因此 plan handles 最多影响约 0.004% 的同类块耗时，本轮不增加其生命周期与 ABI 复杂度。

同轮将 gated 分支内部纯 AgX 换成 native 的实验为 181.17→117.03 ms，但 500k 像素中有 96,166 个
float32 通道值不同，固定噪声量化后仍有 3 个 u8 码值变化。这个实验只存在于工具里，
生产 gated 继续原计算体；不能把已有默认 native/NumPy 尾差扩大到原先未使用的分支。

更深的 input scale/sanitize/retreat 融合，其独立 NumPy proxy 在 500k 像素约 5.8 ms，
约为 AgX 加 output 两核总耗时的 2.6%。当前每块临时量已受既有分片限制；本轮先落实
能直接消除完整 owner 的 mask、HDR packing、readback 和 guidance，保留此处明确的
运算边界，不宣称性能已经没有后续空间。
