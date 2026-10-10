# 标定、噪声传播与 RAW 专项计划实施记录

实施日期：2026-10-09（America/Los_Angeles；UTC 2026-10-10）。任务来源为用户提供的 `AgXRAW_Codex_Review_and_RAW_Test_Plan.md`，范围包含缺陷修复、研究原型和现有本地 RAW 专项验证。

**结论：已复现的 F01–F06 已修复或复核关闭；差分 PTC、多锚点、逐相位方差和逐文件编码身份进入生产链路。个人像素黑位、signed CFA、完整相关噪声传递、固定图样校正和联合色度收缩完成有边界的研究原型，保持默认关闭。Sony、Fuji、fp、iPhone 与传统 Nikon NEF 已实际处理；缺少的实测标定、其他 RAW 模式及独立参考解码没有计作通过。**

任务书基线为 `797bf831579d4af57677af8474ff9e5bdc17b9f3`；开始实施时本地与远端为 `d954e435fff3526203c9dc629659bf1fbb1e83f2`。两者之间已有正确修复，尤其 Stage 3 WB 域，不重复推翻。固定运行库为 rawpy `0.27.0+libraw.e419de08`，LibRaw `0.22.0`，rawpy commit `cc7b4748c7b3e87da319198fdfcdb46e17c9c2a6`，LibRaw commit `e419de08001de28ae6988ecb22df47e52b9c5eaa`。本地为 macOS arm64、Python 3.14.4、NumPy 2.5；CI 使用仓库声明的 Python 矩阵。

## 逐项状态

“原型完成”表示已完成数值合同、实验、测试和默认集成判断，不表示已完成所有真实机身的产品验收。

| ID | 状态 | 实施与证据 | 尚未验证／集成边界 |
|---|---|---|---|
| F01 | 已修复 | `9ceb062`；paired-shutter 按组、相位构建 ISO 比值图，重复帧组内处理，保留权重、闭环残差和连通分量。标称快门误差不再混入组内增益比 | 真实灯光漂移仍需实测；auto-shutter 保留对标称时间的依赖；断链不补齐 |
| F02 | 已修复 | `9ceb062`；从 Channel/ColorIndex 与实际 color description 建立空间相位映射，四种 Bayer 与非 RGBG 描述有回归 | 缺失／矛盾映射不能冒充完整四相位频谱；非 Bayer 不套用 Bayer 传播 |
| F03 | 已修复 | `27d570c`、`cbf9e4c`；user/packaged/curated/bulk 统一先核对模式，再排序；Sony 机械先验结构化限制，direct matching API 同样保留 shutter | 文件没有可靠模式信息时明确未知；不根据机型猜测电子／机械 |
| F04 | 基线后已解决，扩充复核 | `d954e43` 已修正域；`0ae967a` 扩充真实 Bayer DNG、Linear DNG、MapPolynomial/MapTable、DeltaRow/Column、ScaleRow/Column、有色 WB、半尺寸回归 | 无法逆转的早期整数截断不宣称恢复；恢复高光与不兼容 point-op 组合继续明确拒绝 |
| F05 | 已修复 | `9ceb062`；逐相位扣黑后估计 gain；时域方差按定义聚合，避免先平均黑位／标准差 | 未测平面不伪造为测量值 |
| F06 | 已修复 | `9ceb062`；逐相位、逐方向频谱保留异常，0.1/1.9 不再平均成许可；回退方差来源不撤销独立负约束 | high/mid 比值接近 1 仍不构成白噪声证明 |
| C01 | 研究原型完成，缺实测 | `003e185`；`sensor_research.PersonalBlackCalibration` 与 `signed_black_correct`；机身／模式／ISO／曝光／温度／固件／时效匹配；absolute/residual 分开，保留元数据空间残差与身份 hash | 未接入默认像素校正和生产缓存；公共 CBLD 不覆盖个体黑位；需要匹配暗场和算子顺序验收 |
| C02 | 研究原型与真实合成 DNG 验证完成 | 20 组近黑统计；12 组真实 LibRaw DNG 解码，分别检查三档输出曝光的 SDR/HDR；signed bilinear 保留负尾并测性能／内存 | 不是 LibRaw 质量等价替代；默认整数兼容路线保留；缺真实匹配暗场／弱信号验收 |
| C03 | 已集成，缺实测平场 | `a3cf0d7`；消费成对均值与差分方差，明确差分 1/2 与 clip correction，拒绝明显均值漂移；保留空间 PTC 交叉检查与条件拟合区间 | 合成 Poisson/read/PRNU 验证不替代真实平场；拟合区间不包含全部采集系统误差 |
| C04 | 已集成，缺实测多 ISO | `a3cf0d7`、`cbf9e4c`；全部 PTC 拟合与 hash；连通段独立锚定，冲突报告，无外推；ISO、CFA、DN 参考范围冲突拒绝 | 无独立证据不跨 DCG 断点平滑；单一参考 DN 范围不接受相互冲突的锚点 |
| C05 | 逐相位方差已集成；完整协方差仍近似 | `a3cf0d7`、`cbf9e4c`；保留四相位 stored temporal variance、黑位、单位、质量／不确定度；独立 phase gain 需来源与 DN 参考；两绿按平方权重合并；Monte Carlo 和 GainMap 相位测试 | 共享 gain 必须标为近似；未知相位回退有理由；两绿不等 WB／归一化跨度继续拒绝未经验证的三输入传递 |
| C06 | PSD 持久化已集成；相关传播原型完成 | `4a9b9dd`；`noise_spectrum.py` 保存完整 single/diff PSD 和归一化证据；`spectral_variance.py` 验证固定核带域方差与 `detail_variance` 桥接 | h/v 不能唯一恢复任意二维协方差；可分离与独立相位须显式同意；不自动用于自适应解拜耳，频谱异常门控保留 |
| C07 | 研究原型完成，缺个体暗场 | 独立均值图、时域／差分方差、均值图不确定度、行列偏置与热像素候选；64 训练／32 留出合成暗场 | 产品默认不授权像素减图；温度／曝光／长曝 NR 条件不能跨用；候选不等于坏点修复 |
| C08 | 研究原型完成，不进入默认 | 完整二维零亮度色度协方差、白化与联合向量收缩；96 组合成方向／细线／织物／色边／亮度边；拒绝奇异／病态模型 | 仍明显削弱部分低 SNR 真结构；Y 不变不等于颜色纹理无偏；与多尺度生产核不是算法等价比较 |
| C09 | 量化盘点与原型完成，默认不变 | LUT、扣黑、GainMap、point-op、WB、重建、缩放及输出边界已盘点；逐次截断／最近舍入／延迟整数化误差预算 | 合成亚 DN 改善不能直接推断实拍可见改善；未修改默认 GainMap 的整数兼容边界 |
| D01 | 已集成，现有样本验证 | `3baedb4`；原生 RAW 主 CFA IFD 编码证据与 decoder 版本进入 readout；能解码、能显影、是否符合物理标定条件分开；未知 codec 不冒充无损 | ADC bits、快门、温度等不可知字段保持未知；模式实测资格仍需相应标定来源 |
| D02 | 已修复单位合同，真实 ARW6 基础验证 | `3baedb4`；coding endpoint 使用 LibRaw 解包／LUT 后 maximum，linear_max 保留为线性有效阈值；Sony 实际 39002/32800/black1024 分开 | 缺同模式受控暗场、平场及近饱和帧；没有宣称完整 ARW6 PTC 已实测验证；Nikon LUT 未实测 |
| D03 | 现有真实语料完成，部分缺样本 | `3baedb4`、`47619d4` 与专项工具；Sony、Nikon 全／半尺寸及 Fuji、fp、iPhone，共 7 场景 × NumPy/native 两路；15 项适用损坏案例明确拒绝；机器记录见下 | Nikon HE/HE* 与其他传统模式、Sony 其他 RAW 模式、独立厂商 mosaic 参考未覆盖，不据此宣称解包绝对精确或压缩无损 |

## 真实 RAW 与合成证据分开

[native-raw-20261010.json](../assets/delivery-quality/native-raw-20261010.json) 由 [validate_native_raw.py](../../tools/validate_native_raw.py) 生成，保存实际文件 hash、主 RAW 编码／几何／CFA、黑白位、模式资格、运行配方和分区指标。私有照片未上传，仓库没有新增完整像素缓冲。

本轮实际处理 Sony `DSC00225.ARW`（全／半尺寸）、Fujifilm `DSCF0214.RAF`、SIGMA fp DNG、iPhone 16 Pro Bayer DNG。5 组均完成分析及 SDR/HDR 形成，NumPy/native 共 10 次，有限值与数值一致性验收通过。Sony 全尺寸 scene 的最大路径差约 1.445 个当前存储单位，按 scene scale 归一化约 2.21×10⁻⁵；SDR 浮点最大差不超过 3.94×10⁻⁵。这是同一解码输入下两种计算后端的一致性，**不是独立解码参考误差上界，也不是 JPEG/HEIF 本轮封装验收**。

Sony 文件主 RAW IFD 是 7040×4688、14-bit、Compression 32766，当前识别为 ARW6/LLVC3。容器 14-bit 与 LUT 后编码域不同；cRAW HQ 不标为无损。native DefaultCrop 与 LibRaw formed raster 的差异保留为观察，没有在缺乏坐标证明时贸然裁图。

D03 的异常输入检查另存于 [raw-failures-20261010.json](../assets/delivery-quality/raw-failures-20261010.json)，由 [validate_raw_failures.py](../../tools/validate_raw_failures.py) 在上述 Sony、Fuji、fp、iPhone 原件的临时副本上执行。正常原件均能解码；损坏文件头、仅保留前 4096 字节、截断为原长度一半、将主 CFA IFD 压缩方式改为不支持值，共 15 项均由生产 `load_raw()` 明确失败，未产生可供分析的 bundle，未继续生成可信统计或导出图像。RAF 的 TIFF compression 修改不适用，单独记录，未计入通过。所有原件 SHA-256 前后相同；临时副本已删除。对应可生成 DNG 反例进入 [test_raw_failure_contract.py](../../tests/test_raw_failure_contract.py)，包含正常解码控制与独立子进程检查。这个结果覆盖选定的结构性破坏，不保证识别任何仍合法的像素篡改；没有认证完整性信息时，不能把合法黑场或合法低码值图像凭外观判作损坏。

另一次独立只读检查比较了 22 份本地 Sony、Fuji X-Trans、iPhone 与 fp 样张的 `raw_pattern`，以及 `raw_colors_visible` 起始处同尺寸 tile，全部一致；Fuji margins 为 (6, 0)、tile 为 6×6，其余当前样本 margins 为 (0, 0)、tile 为 2×2。6 份合成 DNG 将 ActiveArea 起点设置为 (0,0)、(0,1)、(1,0)、(1,1)、(2,3)、(3,2)，实际 LibRaw margins 分别为 (0,0)、(0,2)、(2,0)、(2,2)、(2,4)、(4,2)，同样未复现 compact pattern 与 visible tile 不一致。固定 [rawpy 属性实现](https://github.com/Gen-416/rawpy/blob/cc7b4748c7b3e87da319198fdfcdb46e17c9c2a6/rawpy/_rawpy.pyx#L720) 的 `raw_pattern` 使用 full raster 的 `raw_color(y,x)`，而 visible colors 按 margins 切片，因此本次一致性是当前 LibRaw 处理和样本条件下的**未复现观察**，不构成任意奇数裁切、其他 decoder 或所有 CFA 的原点正确性保证。该 22 份检查来自独立代理已执行的只读探针，本报告未把它冒充新的逐文件机器记录。

补充公开 Nikon 验证见 [nikon-nef-20261010.json](../assets/delivery-quality/nikon-nef-20261010.json)：[raw.pixls.us](https://raw.pixls.us/) 的 [repository 元数据](https://raw.pixls.us/json/getrepository.php?set=all) id 897 为 Nikon D750、12-bit lossless-compressed 传统 NEF，该行明确标有 [CC0](https://creativecommons.org/publicdomain/zero/1.0/)。临时下载 20,095,707 字节，SHA-256 为 `baae8dfc05a81a4a4034831a49ed99d129e797858ad876d31173a46d33b7d3ea`，与站点记录一致。全尺寸 6032×4032、半尺寸 3016×2016 各完成 NumPy/native 的分析与 SDR/HDR 形成，4 次均为有限值；两后端 scene 一致，SDR 最大差 1.63×10⁻⁶、HDR 最大差 4.77×10⁻⁷。生产元数据识别 Compression 34713，coding white 4095 与 linear validity 3827 保持分开；逐文件 `storage_lossless` 仍未知，外部物理噪声先验按 `file-storage-lossless-unverified` 拒绝，未利用语料站的模式描述放宽运行时门控。本次只证明这一传统 NEF 可用及计算后端一致，不覆盖 HE/HE*、其他 NEF 模式或独立解码准确度。来源 URL、许可、预期／实得 hash 与临时 RAW 删除状态已存入机器记录；原 RAW 和完整像素缓冲均未留在仓库。

[sensor-precision-20261009.json](../assets/pipeline/sensor-precision-20261009.json) 保存纯合成实验与实际解码的合成 DNG，包含基准 commit、工作区状态及关键源码 SHA-256。不能把它算作私有相机的测量数据。数值合同、代价和默认判断见[传感器精度研究](SENSOR_PRECISION_RESEARCH_2026-10-09.zh-CN.md)与[频谱传播原型](SPECTRAL_VARIANCE_PROTOTYPE_2026-10-09.zh-CN.md)。

## 兼容与回退

旧标定没有新增相位或完整 PSD 字段时，继续使用有明确来源的原有标尺；不能因字段缺失补造完整测量。明确 unresolved 的点／区间、失败 phase fit、未知读出条件保持其状态。允许适用的 DNG NoiseProfile 提供方差系数，同时保留独立频谱约束。

preview cache 升到 31，新增相位和原生 readout 身份，旧缓存自然重建。完整 PSD 增大导入记录；写入前按现有 16 MiB 限制校验，避免写成功后无法读取。超过限制应拆分标定集合，不能静默截断频谱。

本轮没有改 Rust 内核的算法或 ABI，没有为了通过测试降低质量容差。个人黑位、浮点 CFA、DSNU 图和联合色度原型均不从默认解码／渲染调用；已有不适用模型的门控继续有效。

## 验证记录

新增可生成反例进入普通测试，包括 paired ISO 图、四种 Bayer 映射、多锚点和差分 PTC、逐相位方差、readout 资格、固定谱核和 signed sensor 原型。外部素材仍由环境变量提供，缺失时显示跳过。

首次严格 native 全量运行 1890 项，发现两处旧测试预期仍要求缓存版本 30 与非 DNG 的旧四项白点表示；未发现新的像素计算失败。修正为新合同并增加 coding white 与 linear limit 的区分断言后，相关 63 项（包括本地 ideal-image）通过。

最终本机结果如下。两个全量进程均启用本地 `DNGSCAN_IDEAL_IMAGE_DIR`，在可实际创建 Core Image context 的运行环境中执行；跳过项未计为通过。

| 验证 | 通过 | 跳过 | 失败／错误 | 耗时 |
|---|---:|---:|---:|---:|
| 严格 native 最终全量，1905 项 | 1903 | 2 | 0 | 278.721 s |
| NumPy 全量，1890 项 | 1857 | 33 | 0 | 731.629 s |
| NumPy 最后新增／变更模块复跑，57 项 | 57 | 0 | 0 | 0.798 s |

NumPy 全量在最终 15 项研究／损坏输入回归添加前启动；其后的 57 项复跑覆盖这些新增用例与相应模块，不能与全量计数简单相加。严格 native 全量已经包含最终 1905 项。验证包含真实 macOS Core Image/ImageIO、HEIF 精度／gain-map 回读和 Rust 路径；不把计算后端一致性当成独立 RAW 解包准确度。`git diff --check` 及本轮文档链接检查通过。远端 Python 3.11／3.12 两路与 wheel 安装检查由[仓库 CI](https://github.com/Gen-416/AgXRAW/actions/workflows/ci.yml)在本轮推送后执行，最终状态另以对应提交的 Checks 为准。

以下仍需要新素材，当前未验证：个体真实暗场／成对平场、多 ISO／DCG 受控阶梯、Nikon HE/HE* 与其他传统 NEF 模式、Sony 多种 codec 与裁剪／高 ISO／复杂 tile、独立可比较的未 WB mosaic 参考，以及研究原型的真实阴影和弱彩纹理验收。普通实拍能验证管线可用，不能替代这些测量。
