# 噪声与纹理：现有方案及 AgXRAW 改进依据

[返回文档索引](../README.md) · [报告索引](../reports/README.md)

日期：2026-10-09。研究基准：`738da0a214c92244dc3edf42f369f691f0db144a`。第 1–8 节保留该版本的非胶片 RAW 噪声证据、HDR 置信度、可选色度降噪研究；研究阶段只读源码，没有修改运行代码或进行外部软件实拍画质对比。随后实施的内容另记于[第 9 节](#9-研究后的实施记录2026-10-09)，不能把前文的基准问题描述当作当前状态。当前使用方法见[实测标定接口](../NOISE_CALIBRATION.zh-CN.md)与[色度核合同](../CHROMA_NR.zh-CN.md)。

结论：优先修正噪声证据的来源与解释。Jiangtherapee 的配套测量工具通过跨帧采样分离随机噪声和稳定结构；darktable 用独立的相机噪声模型约束滤波；单帧稳健估计则明确处理纹理污染和模型假设。适合 AgXRAW 的方向是把**标定、单帧观测、降噪动作**分开，不能继续让待处理纹理同时定义噪声量、HDR 可信度和滤除强度。

## 1. 本项目需要解决的具体问题

基准版本存在三条相互关联的失效链：

| 入口 | 当前方法与失效 | 应恢复的语义 |
| --- | --- | --- |
| `analysis.py` / `phase_statistics.py` 的 SNR | CFA 块内标准差同时含噪声和真实结构；低分位不能保证有平坦样本 | 单帧局部变化量不自动等于物理噪声；输出覆盖率、适用条件和不确定性 |
| `raw_health_metrics()` | 空间错位的 G1/G2 相减残留纹理，相关性会误判为机内处理；继而撤销先验和 SNR 曲线 | 相关性线索与经验证的处理证据分开；单帧不可测不应自动推翻匹配的标定 |
| `chroma_nr.py` | 每尺度全图色度细节 MAD 设阈值；真实颜色纹理越多，滤波也可能越强 | 独立模型给噪声尺度，结构证据负责减少平滑；默认关闭保持不变 |

前一轮独立合成反例已覆盖这些链路：固定随机噪声加入纹理后，SNR 可从约 60 dB 降到约 4 dB；无空间滤波的周期结构可触发高绿色残差相关性；无噪声周期色度在强度 0.25 / 0.5 / 1.0 下只保留约 60% / 34% / 18% 振幅。它们证明估计方法的失效，不能用既有测试通过或 Python/Rust 数值相等排除。

HDR 还有状态传递问题：`compile_tail_snr_gate()` 当前在有效曲线缺失时返回 1。这个值原本表示该因子不施加额外限制，但“普通缺测”和“明确撤销的证据”经过清空字段后已无法区分。其他裁切、色域和解码器门控仍然存在；这里不是整条 HDR 管线无条件放行。

## 2. Jiangtherapee：最有价值的是采样设计

本次区分三个公开组件：Jiangtherapee Online、JPTC Collect（仓库名 JiangtherapeeTesterView）和 JoRaw。没有找到可确认的传统 RawTherapee/ART 桌面 fork 源码，不能据名称推定其继承了某种上游降噪实现。

### 2.1 同一像素的两帧差分

JPTC Collect 固定提交 `57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c`。它逐 CFA 相位取两张黑场或平场，在**相同物理坐标**计算 A−B。对独立、同分布的帧噪声，单帧标准差由差分标准差除以 √2 得到；两类测量的界面派生值均对 sigma clip 的截断作显式修正。平场保存 A、B、差分的独立统计，不把单帧空间标准差直接命名为随机噪声。[黑场实现](https://github.com/y-g-jiang/JiangtherapeeTesterView/blob/57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c/src/analysis/darkPair.mjs#L110)、[平场实现](https://github.com/y-g-jiang/JiangtherapeeTesterView/blob/57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c/src/analysis/ptcPair.mjs#L60)

这依赖重复观测条件匹配：ISO、曝光、读出模式、温度和照明稳定，且不能混入运动或闪烁。G1/G2 位于不同空间位置，不具备这种场景抵消保证。

固定图样与随机噪声也分开处理。代码用相同剔除掩码下的 `Var(A) − Var(A−B)/2` 估固定图样方差；差分本身会消掉稳定热像素偏置，所以不能单凭差分生成完整坏点图。[固定图样与掩码处理](https://github.com/y-g-jiang/JiangtherapeeTesterView/blob/57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c/src/analysis/darkPair.mjs#L163)

### 2.2 保留原始测量，避免过早解释

JPTC 把 `measured` 与 `derived` 分开：采集端保存需要像素才能得到的统计、剔除数量、原始差分标准差等；√2 和量化等修正保留为可重新计算的分析。其横纵频谱也区分单帧与差分，逐行/列平均 periodogram，并单独记录行间/列间变化。逐行去均值会隐藏整行偏置，因此“频谱没有峰”不能单独证明没有条纹。[测量合同](https://github.com/y-g-jiang/JiangtherapeeTesterView/blob/57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c/README.md#L19)、[频谱实现](https://github.com/y-g-jiang/JiangtherapeeTesterView/blob/57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c/src/dsp/spectrum.mjs#L24)

AgXRAW 已有 `tools/import_jptc_collect.py`，无需另造一个不兼容的采集格式。但导入器保留的 `WithinRow/ColVarDiff` 旧数据语义仍标为 `UNCONFIRMED`，不能因为新源码可读，就直接把已有 CSV 数值升级成条纹方差比例；需要核对生成版本与输出合同。

### 2.3 Online 的能力与本次问题的边界

Online 发布站固定提交 `504e119bcfb9ff2b58fc9e47c204ddb77eee5879`。指南及当前编译 bundle 展示 PTC 的 `a+bS+cS²` 拟合、信号分组、参数约束和稳健重加权。这适合标定数据分析，但分组取低方差或压低离群点，仍不能保证任意自然图像含有可识别的纯噪声样本。[Online 指南](https://y-g-jiang.github.io/JOindex.html)、[发布 bundle](https://github.com/y-g-jiang/y-g-jiang.github.io/blob/504e119bcfb9ff2b58fc9e47c204ddb77eee5879/assets/index-WaelY09P.js#L6072)

需避免两个混淆：指南的 “Jiang’s official NR” 在 SFR/LSF 测量中指自适应半径选择，不能当成主照片色度降噪；JoRaw 既可 `unpack()` 读取 mosaic，也有调用 LibRaw `dcraw_process()` 的显影路径，暴露 `threshold` / `fbddNoiserd` 不代表 PTC 已自动控制这些算法。Online 指南提及的原始 TS 目录在本次公开发布 tree 中未见，因此这部分核查以发布 bundle 为边界。[JoRaw 源码](https://github.com/y-g-jiang/JiangOnlineRaw-joraw-/blob/d4da562ae9e6d335254caa516431db0d2a22f1a1/libraw_wrapper.cpp#L48)

## 3. darktable：噪声模型与结构强弱各司其职

固定版本：darktable **5.6.2**，提交 `20891f6c6fa7be995e5bf9dff9d1ee5062051773`。[版本入口](https://github.com/darktable-org/darktable/releases/tag/release-5.6.2)

### 3.1 Profile 是独立输入，不是任意图像的细节 MAD

darktable 离线 profile 工具把专门的标定照片显影为线性 camera-RGB PFM，以 Haar LL 估亮度、HH-MAD 估噪声，按信号强度拟合 `variance = a × mean + b`。原理文章要求失焦以减少边缘泄漏，说明标定照片也需要控制结构；MAD 不能凭自身识别所有纹理。2012 年文章的操作步骤已过时，本次用它解释假设，用固定版本源码核对计算。[原理与假设](https://www.darktable.org/2012/12/profiling-sensor-and-photon-noise/)、[HH-MAD 源码](https://github.com/darktable-org/darktable/blob/20891f6c6fa7be995e5bf9dff9d1ee5062051773/tools/noise/noiseprofile.c#L223)、[标定出口与拟合工具](https://github.com/darktable-org/darktable/blob/20891f6c6fa7be995e5bf9dff9d1ee5062051773/tools/noise/darktable-gen-noiseprofile#L149)

实际 profiled denoise 通常在解拜耳后、输入颜色矩阵前的线性 camera RGB 工作，补偿白平衡放大，先做方差稳定变换（VST），再滤波。其 JSON 虽有三通道 a/b，当前波段与 NLM 路径主要取绿色 a/b 配合 WB，不能描述为完整的逐通道物理传播。[处理顺序](https://github.com/darktable-org/darktable/blob/20891f6c6fa7be995e5bf9dff9d1ee5062051773/src/common/iop_order.c#L298)、[处理域与参数](https://github.com/darktable-org/darktable/blob/20891f6c6fa7be995e5bf9dff9d1ee5062051773/src/iop/denoiseprofile.c#L1211)

### 3.2 纹理增强不应自动提高噪声阈值

波段路径的关键是：噪声尺度由 profile/VST 给定，图像细节方差用于估计信号强度。其 BayesShrink 形式可概括为：

\[
\sigma_x=\sqrt{\max(\operatorname{Var}(d)-\sigma_n^2,\epsilon)},\qquad
T=K\,\frac{\sigma_n^2}{\sigma_x}.
\]

`d` 是该尺度细节，`σn²` 是预期噪声方差，K 包含强度和经验系数。真实纹理增加时，估计的信号方差一般上升、阈值降低；AgXRAW 基准版本的全图 MAD 则可能让纹理增加被解释为噪声增大、阈值提高。值得借鉴的是这种分工，并非直接复制常数。[BayesShrink 实现](https://github.com/darktable-org/darktable/blob/20891f6c6fa7be995e5bf9dff9d1ee5062051773/src/iop/denoiseprofile.c#L1345)

à trous 分解还有颜色距离边缘权重，减少明显边界两侧混合；NLM 则比较邻域 patch 的相似性，支持重复结构。两者仍可能损伤低 SNR 细节。这里确认的是逐尺度收缩，不能根据旧文章引用推定当前实现使用跨尺度父子联合收缩。[边缘权重](https://github.com/darktable-org/darktable/blob/20891f6c6fa7be995e5bf9dff9d1ee5062051773/src/common/eaw.c#L226)、[NLM 核](https://github.com/darktable-org/darktable/blob/20891f6c6fa7be995e5bf9dff9d1ee5062051773/src/common/nlmeans_core.c#L428)

### 3.3 不应直接移植的部分

darktable profile 的 a/b 处在其标定显影与 PFM 尺度，不能直接当作 AgXRAW 的 e−/DN。新 VST 也包含阴影偏差修正和经验参数；变换后的“单位噪声”及滤波残差不能直接充当本项目的物理 SNR。官方说明明确把滤波强度视为噪声与细节的取舍，且 NLM 比波段模式更耗资源。[VST 与偏差说明](https://github.com/darktable-org/darktable/blob/20891f6c6fa7be995e5bf9dff9d1ee5062051773/src/iop/denoiseprofile.c#L969)、[5.6 官方手册](https://docs.darktable.org/usermanual/5.6/en/module-reference/processing-modules/denoise-profiled/)

遍历该提交完整 `noiseprofiles.json`，未找到 Sigma fp / fp L 的型号记录。缺 profile 时代码退回 `generic poissonian`，它是默认滤波参数，不是 fp 的实测标定。[profile 数据库](https://github.com/darktable-org/darktable/blob/20891f6c6fa7be995e5bf9dff9d1ee5062051773/data/noiseprofiles.json)、[generic 默认与匹配](https://github.com/darktable-org/darktable/blob/20891f6c6fa7be995e5bf9dff9d1ee5062051773/src/common/noiseprofiles.c#L26)

## 4. 其他方案：单帧估计更稳健，但仍有假设

| 方案 | 如何处理结构 | 对 AgXRAW 的启示及边界 |
| --- | --- | --- |
| Foi / Azzari 的信号相关噪声估计 | 分割构造均值—标准差散点，再用 Gaussian–Cauchy 混合似然稳健拟合，专门抵抗密集纹理造成的异常点 | 不只是把 std 换成 MAD；需处理结构污染、剪切和拟合诊断。不能据论文成功案例承诺所有单帧都可测 |
| PCA 噪声曲线 | 用块协方差的小特征值寻找噪声，筛选候选块，并按亮度分箱 | 适合对照原型；IPOL 实现明确承认整个亮度 bin 都有纹理时仍难分离，不应无条件喂给 HDR |
| Google HDR+ | 标定的 Bayer `Ax+B` 模型给噪声尺度；帧间差异相对噪声越大，其他帧权重越低，逐渐退向参考帧 | 借鉴独立模型与保守退化；单张 DNG 没有多帧观测，不能直接获得其收益 |
| BM3D | 将相似块分组，在联合变换域收缩并聚合，利用重复结构 | 仍需可信噪声尺度；相关噪声需要变换域方差，完整引入会扩大计算与接口范围 |

来源：[Azzari / Foi 2014 原论文](https://webpages.tuni.fi/foi/papers/ICASSP2014_GaussCauchy-AzzariFoi.pdf)、[Colom / Buades 2016，§3、§6.3](https://www.ipol.im/pub/art/2016/124/article_lr.pdf)、[HDR+ 2016，§5](https://people.csail.mit.edu/hasinoff/pubs/HasinoffEtAl16-hdrplus.pdf)、[BM3D 2007 原论文](https://webpages.tuni.fi/foi/GCF-BM3D/BM3D_TIP_2007.pdf)、[相关噪声协同滤波 2020](https://webpages.tuni.fi/foi/papers/Ymir-Collaborative_Filtering_of_Correlated_Noise-TIP.pdf)。本次没有整合这些外部实现。

## 5. Sigma fp 可利用的独立证据

### 5.1 PTC 与黑场各自提供什么

PTC/ISO 锚定可提供转换增益、读噪及适用范围；黑场对可验证时域噪声与相关性，均值堆栈可识别稳定偏置、热像素与固定条纹。单张全黑照片适合观察异常位置与黑电平，但没有重复观测，不能完整分离时域噪声与固定图样。

相同条件下令 `D=(F1−F2)/√2`：若帧噪声独立且同分布，D 保留单帧随机噪声协方差，稳定场景与固定图样抵消。再按 CFA 同相位测 ACF/PSD。另看均值堆栈中的固定成分；差分干净不代表没有 FPN。这个采样原则与 EMVA 的时域/空间拆分一致。[EMVA 1288 Linear 4.0](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf)

不能简单把 G1/G2 差分替换成某个单帧高通后继续以零相关为基准。直接由协方差可推得：独立白噪声做一阶差分后，相邻相关系数为 −1/2。相关性检查必须计入算子本身引入的相关性。

### 5.2 已发现文件内的 DNG NoiseProfile

本轮仅作 metadata 读取，本地 `_SDI0150.DNG` 的可复核信息为：

| 项目 | 值 |
| --- | --- |
| 相机 / ISO | SIGMA fp / 1600 |
| BlackLevel / WhiteLevel | 四项均为 1024 / 16383 |
| NoiseProfile（tag 51041） | `[8.957672433680315e-05, 1.7812016132784875e-08]` |
| NoiseReductionApplied | 未读取到该 tag；不能据此证明完全未经处理 |

DNG 规范把它定义为归一化线性信号 x 的 `variance = Sx + O`；一组参数可用于所有颜色平面，多组则按 CFAPlaneColor 排序。模型假定白噪声与空间独立，不包含完整 FPN/PRNU。Raw IFD 和 Enhanced IFD 可以有不同 profile，不能混用。[Adobe DNG 1.7.1，NoiseProfile，pp.59–61](https://helpx.adobe.com/content/dam/help/en/camera-raw/digital-negative/jcr_content/root/content/flex/items/position/position-par/download_section_733958301/download-1/DNG_Spec_1_7_1_0.pdf)

在基准版本生产代码中未找到 NoiseProfile 的读取/消费路径（后续实施见第 9 节）。它值得作为新的候选模型来源，与匹配的 PTC/黑场交叉验证；**tag 存在不代表值已经验证正确**，单个文件的发现也不能推断所有 fp DNG 都提供或正确填写该 tag。

### 5.3 进入管线前必须核对数据域

以下是本项目建议采用的单位推导，而非直接引用其他软件的参数。在线性、未剪切 DN 域，设扣除固定黑电平后的期望信号 `s=E[DN]−B`，转换增益 g 的单位为 e−/DN，读噪 r 的单位为 e−，则独立 shot/read 模型为：

\[
\operatorname{Var}(DN\mid s)\simeq s/g+(r/g)^2.
\]

令随机观测 `X=(DN−B)/R`、均值 `μ=s/R`，R 是对应编码范围，则 `a=1/(gR)`、`b=(r/(gR))²`，得到 `Var(X|μ)=aμ+b`。用观测代替均值只能是近似估计。这个 g 的单位不能与采用 DN/e− 约定的软件混淆。量化、剪切、非线性编码、增益切换及固定图样需要另行处理。

固定正线性增益 `Y=kX` 会使输出域参数变为 `a′=ka`、`b′=k²b`；固定颜色矩阵 M 传播协方差为 `C′=MCMᵀ`。白平衡、镜头暗场 GainMap、解拜耳、缩采样和几何插值都会改变噪声量或空间相关性。因此 raw-domain SNR 模型不能未经传播就拿去给解拜耳后的色度波段设阈值。文件内 NoiseProfile、PTC、darktable PFM profile 是不同来源，必须记录各自的归一化和处理层位。

## 6. 基准版本的建议改进顺序

以下保留研究时的建议；第 9 节分别记录已实施部分与仍未完成的边界。

1. **先修证据合同与状态传递。** 分开记录模型来源/适用性、单帧验证质量、相关性证据。来源可为匹配标定、文件 metadata、有条件单帧估计或缺失；观测至少区分有效、覆盖不足、结构污染疑似、模型不匹配。相关性单列未知/线索/经标定验证，不通过一个绿色残差相关系数认定机内降噪。不要通过清空数值抹掉失败原因。
2. **让独立模型提供噪声尺度。** 核对 PTC 的机型、ISO、快门/读出模式、DN 范围、黑电平和增益跳变；支持文件 NoiseProfile 的安全解析与一致性诊断。单帧纹理导致不可测时，保留仍然匹配的标定模型，不因这次观测失败就撤销它。已有 fp curated entry 优先于 JPTC 的规则也需明确，避免用户导入实测却没有被选中。
3. **补保守的单帧验证。** CFA 同相位、去除剪切/坏点、筛掉明显梯度和方向性结构，按信号强度分组，以稳健拟合及小规模 PCA 原型交叉检查。必须输出样本覆盖、拟合残差和不确定性；覆盖不足就不可测，不强行拟合满曲线。真实纹理与随机噪声不可识别时，程序应承认边界。
4. **让 HDR 显式消费状态。** 有适用模型时使用模型给出的尾部噪声证据；无可靠模型时，不再把不可信的局部方差解释为物理 SNR。区分普通缺测与明确模型不匹配，并为每种状态定义退化策略和诊断。不能机械地把所有缺测改成 0，也不能把撤销证据直接合并为 1；需验证其对通道分离、曝光和趾部的各自含义。
5. **最后改可选色度滤波。** 继续默认关闭。阈值取自处理域的预期噪声，纹理/边缘证据只减少收缩；模型缺失时采用明确的保守行为。先借鉴 model-driven BayesShrink 与边缘保护，不急于加入完整 NLM/BM3D、神经模型或替换 LibRaw。保留亮度只是必要约束，验收还要看真实色度幅度和颜色边界。
6. **算法成立后再优化 Rust。** 先冻结 Python 参考语义，再迁移合适的统计、滤波和频谱核。Python/Rust 相等只证明实现等价，必须另用已知信号/噪声检查估计偏差和纹理损失。不得把新的物理模型与纯性能等价修改混在同一验收里。

验证集应覆盖：无噪声周期和随机色度纹理；固定噪声模型加不同真实结构；平场、渐变、边缘；CFA 相位；shot/read/行列相关噪声；黑白剪切和低码值量化；低覆盖与型号/模式不匹配。HDR 要比较相同信号区间的已知噪声与门控，色度滤波要同时度量降噪收益、真实色度振幅、边缘和亮度。实拍以匹配的 fp PTC、黑场及纹理样张补充，不能只看总体 PSNR 或缩图观感。

## 7. 暂缓待办（用户已指定后置）

以下两项尚未修复，本轮仅登记；不是当前噪声研究的实施步骤。

- [ ] **SDR 10-bit HEIF 保留浮点输入到编码出口。** `export.py` 目前先 `render_output_u8()` 再分流 HEIF，10-bit 编码不能恢复提前丢失的层级。后续从已完成显示变换的浮点结果分流：`render_output_linear()` 返回 display-linear，进入期望非线性 RGB 的编码器前须正确应用输出 OETF，再做 10-bit 量化。同步补高精度主图回读与度量；现有 u8 指标不能简单接 float。验收包含真实渐变层级、传递函数、ICC/NCLX、自动/手动 HEIF 路径，JPEG 的既有 8-bit 合同保持稳定。
- [ ] **机型先验规范化精确匹配与显式别名。** `priors.find_priors()` 的 bulk fallback 使用包含关系，`Canon EOS 5D` 可误命中 `EOS 5D Mark II`。后续改成规范化后的完整型号匹配；别名显式登记，未知型号返回无先验，不删除有区分意义的代际/后缀。验收覆盖前缀碰撞、已知别名、未知型号、模式歧义及 fp 既有匹配。

## 8. 研究阶段的验证边界

已完成固定提交源码阅读、官方手册/原论文交叉核对、AgXRAW 生产入口检查，以及一张本地 DNG 的 tag-only 读取。外部资料与源码暂存于仓库外；未导入外部算法代码或权重，未更改生产代码，未运行新的编码/降噪画质基准。本记录中的方法比较是机制分析，尚不能宣称某个候选在 fp 实拍上优于现有输出。


## 9. 研究后的实施记录（2026-10-09）

以下记录本轮运行代码的实施范围，不改写前文基准反例。具体 GUI 入口、可执行命令、接受格式与适用条件见[实测标定接口](../NOISE_CALIBRATION.zh-CN.md)。

- 新增本机用户标定存储及公共管理接口，GUI / CLI 可导入 Collect CSV 目录、`dngscan-jptc-collect-1` 和 `dngscan-jptc-prior-1` JSON，启停或删除记录。适用用户数据优先于包内 curated；相机精确匹配、快门声明、测量 ISO 域、增益跳变、拟合质量及 DN 尺度仍须通过检查。`compression` / `geometry` 目前只保留测量声明并核对 Collect 内一致性，尚未逐文件验证。
- 生产噪声模型明确来源、有效性、原因及处理域。匹配 shot/read 模型优先，随后可用合法 Raw IFD DNG `NoiseProfile`；没有独立模型时明确不可用。照片局部变化不再定义物理噪声量，空间错位的 G1/G2 相关性只作线索，不单独撤销标定。模型 SNR、模型读噪底与 RAW 剪切/分布分开报告；缺噪声模型不抹去有效 RAW 证据。文件明确声明已降噪或声明非法时，兼容性约束覆盖外部标定的优先级，拒绝独立白噪声模型。
- HDR 尾部 SNR 门控区分明确拒绝与普通缺测；其他剪切、色域和解码器约束独立保留。独立测量频谱的高频/中频比小于 0.5 或大于 2 时另标不均衡，保留有效 gain/read 模型，但将尾部门控设为 0 并跳过当前粗网格 NR。阈值为保守启发式检查；摘要正常不证明白噪声，也不排除窄带峰或行列偏置。
- 色度核保持默认关闭，独立 `a×signal+b` 模型经低频 CFA 近似传播到处理域，以局部结构减少 BayesShrink 式收缩。计入已记录白平衡、颜色矩阵、曝光及受支持的 GainMap / 单个平滑 rectilinear warp；剪切/重建等无效支撑不处理。Apple RAW、未知变换或不适用模型跳过并诊断。scene-linear 亮度投影保留，但真实色度仍可能有损；细节与近似见[色度降噪与纹理保护](../CHROMA_NR.zh-CN.md)。
- 分析与预览缓存保存小型模型及传递描述符，不需继续持有 RAW 大图。标定内容指纹进入分析、预览及导出配方，启停/删除可使旧结果失效；界面显示实际来源、匹配原因与 NR 状态。

接口和缓存验证见 [test_user_calibration.py](../../tests/test_user_calibration.py)、[test_calibration_gui_cli.py](../../tests/test_calibration_gui_cli.py)，模型与滤波验证见 [test_noise_model.py](../../tests/test_noise_model.py)、[test_calibrated_chroma.py](../../tests/test_calibrated_chroma.py)。合成验证覆盖零噪声密集纹理、已知噪声与真实结构、处理域传播和失败跳过；不等于所有自然照片的画质保证。

仍未实施完整的稳健单帧噪声估计/PCA 验证、完整相关噪声与分频传播、行列条纹或固定图样修复。JPTC scalar-green、粗网格 CFA 与几何传播均有明确近似；自己的 fp PTC/黑场还未提供，不能宣称已完成个人相机标定。第 7 节的 SDR 10-bit HEIF 与包内 bulk 模糊机型匹配仍为待办，本轮没有实现。

## 10. 后续管线审查修复（基准 fb91815）

镜头操作不再把 LibRaw 已恢复的高光重新裁到重建前的通道上限。`blend` / `reconstruct` 的晚期暗角和畸变处理使用有符号 float32，保留超范围增益；默认 `clip` 量程不变。Rust ABI 20 同时传输重采样图像与损失掩膜，避免重复坐标扫描，并记录 clip 域实际发生的负插值截断。这个合同没有重排 LibRaw 内部的高光重建；stage-3 点变换与重建模式尚无可靠扩展域定义，明确拒绝该组合。详见[解码与校正架构](../ARCHITECTURE.zh-CN.md)。

Collect 明确无法分辨的读噪 ISO 及跨越失败点的插值区间，现在保留为 `model-unresolved`，不再生成有效读噪；有效 gain 独立保留。独立 DNG NoiseProfile 可作为替代，但报告保留原失败标定来源。`NoiseReductionApplied` 的 `0/0` 未知、`0/1` 未应用和非法零分母声明分别保留，不再统一变成数值零。

色度核采样从实际保留的传感器窗口与输出网格推导，计入半尺寸、DefaultScale、裁剪和方向；双轴均落在声明波段时才选择该尺度。半尺寸 box 缩小丢弃奇数边界后，保留窗口逆向映射回原传感器坐标，修正旋转/镜像时的轴与原点偏移。预览缓存升至 22。

交付新增[局部亮度纹理门禁](../HDR_DELIVERY_VALIDATION.zh-CN.md#局部亮度纹理门禁2026-10-09)，补齐暗部、中间调小区域在均值和百分位中被稀释的覆盖缺口。手动 JPEG 同样验证像素；自动编码和 HDR 的原有色差、亮度、采样及发布原子性约束保留。Rust 执行相同的有符号多尺度差分，NumPy 作为独立参考。局部纯色度纹理仍不在新增门禁的保证范围内，不能把“回读通过”表述为全部纹理完整。

三张 fp 样张的完整 clip scene 缓冲和缩放尺度与基准逐字节一致，原始 RAW 证据保持不变；blend/reconstruct 在此前被重新截断的位置有预期变化。合成 DNG 覆盖恒等镜头操作、非中性 WB、重建高光、全/半尺寸及全部八种方向。实拍与性能记录保存在[机器可读验收记录](../assets/delivery-quality/pipeline-repair-20261009.json)，其中速度仅代表新增指标的独立扫描，不是整条管线加速倍率。两项后置待办不变。

最终边界复查还确认：RAW 通道尚未饱和时，白平衡可能已让 LibRaw 的 uint16 相机 RGB 触及 65535。现在在镜头操作前将这类整数交接边界并入不可靠掩膜，供 HDR 与噪声传播使用；不改原始 RAW 饱和百分比，也不把后续浮点镜头增益产生的超范围值误判为截断。这个标记可能与传感器饱和重叠，并不能测量准确的截断幅度。完整 clip 场景缓冲不变，不代表置信度修正后最终 SDR/HDR 像素必然不变。

## 11. DNG 方差回退保留独立频谱（基准 c220019）

读噪未分辨时的提前返回曾使同一份标定的有效频谱丢失。若文件同时声明合法 `NoiseProfile`，替代系数被记为有效模型，但相关性退回 `unknown`，重新允许当前色度核并使 HDR 噪声因子回到 1。真实合成 DNG 的导入、LibRaw 解码、分析与色度核入口复现了这一组合缺口。

现在先独立验证标定的 DN 尺度并读取适用频谱，读噪未分辨或普通缺测只影响方差系数。DNG 替代后保留原标定来源、原因及横纵频谱；实测异常继续令 HDR 噪声因子为 0、色度核跳过。相机/快门、ISO 域与 DN 尺度不匹配的频谱不会借用；文件声明已降噪或非法仍优先拒绝。预览与分析缓存升级到 23，淘汰旧回退结果。

[真实 DNG 组合回归](../../tests/test_spectral_fallback_pipeline.py)同时覆盖有效读噪、普通插值、无 DNG 替代及频谱域外对照。域外对照仍实际启用色度核，避免把所有回退都禁用来掩盖问题；模型单元测试另覆盖低/零/高/正常频谱、双轴保留、DN 不匹配及缓存往返。10-bit SDR 母版与 bulk 机型匹配继续后置。
