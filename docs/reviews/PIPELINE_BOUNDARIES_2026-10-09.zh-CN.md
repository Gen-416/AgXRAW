# RAW 管线交接修复（2026-10-09）

本轮承接针对 `fd8ca7c` 的两份独立复核，在 `797bf83` 上修复六项 P2。没有替换 LibRaw，没有改变默认关闭色度降噪的选择。

| 问题 | 当前处理 | 回归证据 |
| --- | --- | --- |
| GainMap 将 ActiveArea 坐标直接当成 LibRaw visible 坐标 | 保存原始 ActiveArea 与 visible 偏移；AreaSpec 求交保留 pitch 相位，网格插值仍使用原始尺寸和坐标；NumPy、Rust、损失掩膜与噪声增益矩使用同一几何 | 奇偶原点、分 CFA 相位恒定增益、非恒定水平网格、完整／半尺寸实际 DNG，与预先写入等效 DN 的文件逐位比较 |
| 高内存档位令 NR 网格细于当前噪声近似允许范围 | NR 单独规划网格，限制每格至少覆盖 4 个原生感光点；GUI 与渲染共用规划器；传播器仍独立检查 | 4000×3000 在 512 MiB／1024 MiB 下分别选 1408×1056／2000×1500；细长、奇数尺寸、旋转与释放 RAW 后的几何检查 |
| Stage 3 非线性点变换误用已经施加 WB 的数值域 | 根据固定解码 WB 与实际编码范围，在归一化未平衡相机域执行变换，再恢复 WB；保留浮点分数及合法超 65535 值 | 实际 LinearRAW 的 List2／List3／预先平方样本对照、非单位 WB、MapTable、半尺寸、通道白点、后续镜头操作与截断证据 |
| 双光源矩阵先相乘再插值 | 分别插值 CM、CC，再组合 `AB @ CC @ CM`；签名不适用时 CC 为单位矩阵 | 3400 K／5500 K 实际 DNG、相同 CC、签名不匹配、端点钳位、第三光源、仅 CM2 |
| optional WarpRectilinear2 的兼容跳过状态持续生效 | 只消费紧随其后的一个旧 warp；后续独立 warp 正常执行 | optional warp2＋兼容 warp＋独立非恒等 warp，与正确独立 warp 的实际 DNG 场景逐位相同 |
| ICC fallback 的生成时间戳使有效导出失败 | 每次 JPEG／SDR HEIF 导出确定一份 ICC 字节，供全部编码候选及最终回读使用；继续严格比较字节 | 不同有效 ICC 时间戳、实际 DNG→AgX→JPEG、真实 libheif 写入；真正替换 ICC 时仍拒绝交付并保留旧文件 |

ActiveArea 与 visible 的偏移也进入 DefaultCrop、点变换 AreaSpec 和终端 TrimBounds。行列变换表仍以作者声明的原始 AreaSpec 为索引，不能因求交而把表头移到新的第一行。含非零 DefaultCrop 的实际 DNG 与紧凑预览缓存往返包含在回归中。

原始 RAW 剪切统计、前级处理损失、解拜耳依赖资格继续分开保存。GainMap 修改工作缓冲，不能修改已经复制的传感器证据。Stage 3 的点变换仍拒绝 blend／reconstruct，尚未定义的扩展域语义没有因本次修复而被放行。早期整数交接仍可能产生分数舍入；List2 与浮点 List3 的比较需要区分这一量化边界，不能把正常的亚码值差异误判为同一种 WB 错误。

Rust GainMap 接口升级到 ABI 21；预览缓存升级到 version 30，避免复用旧的镜头几何或白平衡派生结果。10-bit SDR HEIF 和 HDR HEIF 的浮点底图交接保持现有合同。

另以真实可解码的 4000×3000 合成 DNG 执行 `load_raw → analyze → build_render_plan → _prepare_chroma_nr_map`。完整／半尺寸各在 512／1024 MiB 下执行一次，四组均为 `active-approximate`，校正图非零，GUI 的可用状态一致。参数、版本和数值见[验收记录](../assets/pipeline/geometry-grid-20261009.json)。它验证大尺寸生产路径实际启用，没有把仅检查网格尺寸当作降噪执行证据；没有执行容器编码，也不证明更细网格具有普遍画质收益。

本地 SIGMA fp（`_SDI0150.DNG`）、Sony ILCE-7M5（`DSC00225.ARW`）、Fujifilm X100VI（`DSCF0214.RAF`）和 iPhone 16 Pro（`Original RAW 26-07-11 202403093.dng`）也重新执行了 LibRaw 半尺寸加载、实际分析和默认 AgX SDR 浮点形成，四组输出均为有限值。这项检查覆盖基本成像可用性，不代表所有机型均可启用校准 NR，也没有把半尺寸形成当作全尺寸容器交付验收。

最终本地验证使用 pinned rawpy／LibRaw、Python 3.14 和重建后的 ABI 21 Rust 扩展：严格 native 全量 1,794 项中 1,791 通过、3 项条件跳过，0 失败；NumPy 路径的 96 项相关回归中 95 通过、1 项条件跳过，0 失败。新增 28 项回归包含在上述测试中，不另行累加。完整 native 回归允许实际 macOS Core Image／ImageIO 与 HEIF 编码回读执行；额外大尺寸及多机型探针单独统计。

这些反例验证坐标、数值域和交付合同，不代替实拍画质评价，也不将近似噪声传播解释为所有纹理与解码器上的质量保证。
