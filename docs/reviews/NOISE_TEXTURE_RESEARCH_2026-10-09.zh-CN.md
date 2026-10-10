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

## 7. 原后置待办（完成情况见第 13 节）

以下为研究时登记的两项后置任务，后续已完成，实施范围与验证见第 13 节。

- [x] **SDR 10-bit HEIF 保留浮点输入到编码出口。** 原 `export.py` 先量化 uint8 的入口现已分流：完成显示变换及输出 OETF 的浮点母版直接进入 10-bit 量化，自动/手动 HEIF 都用高精度回读验收；JPEG 保留既有 8-bit 合同。
- [x] **机型先验规范化精确匹配与显式别名。** 包内 curated / JPTC / bulk 与用户数据均使用规范化完整型号，未知型号不借用另一代先验；显式别名保留必要兼容写法。

## 8. 研究阶段的验证边界

已完成固定提交源码阅读、官方手册/原论文交叉核对、AgXRAW 生产入口检查，以及一张本地 DNG 的 tag-only 读取。外部资料与源码暂存于仓库外；未导入外部算法代码或权重，未更改生产代码，未运行新的编码/降噪画质基准。本记录中的方法比较是机制分析，尚不能宣称某个候选在 fp 实拍上优于现有输出。


## 9. 研究后的实施记录（2026-10-09）

以下记录本轮运行代码的实施范围，不改写前文基准反例。具体 GUI 入口、可执行命令、接受格式与适用条件见[实测标定接口](../NOISE_CALIBRATION.zh-CN.md)。

- 新增本机用户标定存储及公共管理接口，GUI / CLI 可导入 Collect CSV 目录、`dngscan-jptc-collect-1` 和 `dngscan-jptc-prior-1` JSON，启停或删除记录。适用用户数据优先于包内 curated；相机精确匹配、快门声明、测量 ISO 域、增益跳变、拟合质量及 DN 尺度仍须通过检查。该阶段 `compression` / `geometry` 只保留测量声明并核对 Collect 内一致性；后续逐文件合同见第 14 节。
- 生产噪声模型明确来源、有效性、原因及处理域。匹配 shot/read 模型优先，随后可用合法 Raw IFD DNG `NoiseProfile`；没有独立模型时明确不可用。照片局部变化不再定义物理噪声量，空间错位的 G1/G2 相关性只作线索，不单独撤销标定。模型 SNR、模型读噪底与 RAW 剪切/分布分开报告；缺噪声模型不抹去有效 RAW 证据。文件明确声明已降噪或声明非法时，兼容性约束覆盖外部标定的优先级，拒绝独立白噪声模型。
- HDR 尾部 SNR 门控区分明确拒绝与普通缺测；其他剪切、色域和解码器约束独立保留。独立测量频谱的高频/中频比小于 0.5 或大于 2 时另标不均衡，保留有效 gain/read 模型，但将尾部门控设为 0 并跳过当前粗网格 NR。阈值为保守启发式检查；摘要正常不证明白噪声，也不排除窄带峰或行列偏置。
- 色度核保持默认关闭，独立 `a×signal+b` 模型经低频 CFA 近似传播到处理域，以局部结构减少 BayesShrink 式收缩。计入已记录白平衡、颜色矩阵、曝光及受支持的 GainMap / 单个平滑 rectilinear warp；剪切/重建等无效支撑不处理。Apple RAW、未知变换或不适用模型跳过并诊断。scene-linear 亮度投影保留，但真实色度仍可能有损；细节与近似见[色度降噪与纹理保护](../CHROMA_NR.zh-CN.md)。
- 分析与预览缓存保存小型模型及传递描述符，不需继续持有 RAW 大图。标定内容指纹进入分析、预览及导出配方，启停/删除可使旧结果失效；界面显示实际来源、匹配原因与 NR 状态。

接口和缓存验证见 [test_user_calibration.py](../../tests/test_user_calibration.py)、[test_calibration_gui_cli.py](../../tests/test_calibration_gui_cli.py)，模型与滤波验证见 [test_noise_model.py](../../tests/test_noise_model.py)、[test_calibrated_chroma.py](../../tests/test_calibrated_chroma.py)。合成验证覆盖零噪声密集纹理、已知噪声与真实结构、处理域传播和失败跳过；不等于所有自然照片的画质保证。

仍未实施完整的稳健单帧噪声估计/PCA 验证、完整相关噪声与分频传播、行列条纹或固定图样修复。JPTC scalar-green、粗网格 CFA 与几何传播均有明确近似；自己的 fp PTC/黑场还未提供，不能宣称已完成个人相机标定。在本节记录的实施阶段，第 7 节两项后置任务尚未完成；后续完成情况见第 13 节。

## 10. 后续管线审查修复（基准 fb91815）

镜头操作不再把 LibRaw 已恢复的高光重新裁到重建前的通道上限。`blend` / `reconstruct` 的晚期暗角和畸变处理使用有符号 float32，保留超范围增益；默认 `clip` 量程不变。Rust ABI 20 同时传输重采样图像与损失掩膜，避免重复坐标扫描，并记录 clip 域实际发生的负插值截断。这个合同没有重排 LibRaw 内部的高光重建；stage-3 点变换与重建模式尚无可靠扩展域定义，明确拒绝该组合。详见[解码与校正架构](../ARCHITECTURE.zh-CN.md)。

Collect 明确无法分辨的读噪 ISO 及跨越失败点的插值区间，现在保留为 `model-unresolved`，不再生成有效读噪；有效 gain 独立保留。独立 DNG NoiseProfile 可作为替代，但报告保留原失败标定来源。`NoiseReductionApplied` 的 `0/0` 未知、`0/1` 未应用和非法零分母声明分别保留，不再统一变成数值零。

色度核采样从实际保留的传感器窗口与输出网格推导，计入半尺寸、DefaultScale、裁剪和方向；双轴均落在声明波段时才选择该尺度。半尺寸 box 缩小丢弃奇数边界后，保留窗口逆向映射回原传感器坐标，修正旋转/镜像时的轴与原点偏移。预览缓存升至 22。

交付新增[局部亮度纹理门禁](../HDR_DELIVERY_VALIDATION.zh-CN.md#局部亮度纹理门禁2026-10-09)，补齐暗部、中间调小区域在均值和百分位中被稀释的覆盖缺口。手动 JPEG 同样验证像素；自动编码和 HDR 的原有色差、亮度、采样及发布原子性约束保留。Rust 执行相同的有符号多尺度差分，NumPy 作为独立参考。局部纯色度纹理仍不在新增门禁的保证范围内，不能把“回读通过”表述为全部纹理完整。

三张 fp 样张的完整 clip scene 缓冲和缩放尺度与基准逐字节一致，原始 RAW 证据保持不变；blend/reconstruct 在此前被重新截断的位置有预期变化。合成 DNG 覆盖恒等镜头操作、非中性 WB、重建高光、全/半尺寸及全部八种方向。实拍与性能记录保存在[机器可读验收记录](../assets/delivery-quality/pipeline-repair-20261009.json)，其中速度仅代表新增指标的独立扫描，不是整条管线加速倍率。该阶段两项后置任务尚未完成，后续见第 13 节。

最终边界复查还确认：RAW 通道尚未饱和时，白平衡可能已让 LibRaw 的 uint16 相机 RGB 触及 65535。现在在镜头操作前将这类整数交接边界并入不可靠掩膜，供 HDR 与噪声传播使用；不改原始 RAW 饱和百分比，也不把后续浮点镜头增益产生的超范围值误判为截断。这个标记可能与传感器饱和重叠，并不能测量准确的截断幅度。完整 clip 场景缓冲不变，不代表置信度修正后最终 SDR/HDR 像素必然不变。

## 11. DNG 方差回退保留独立频谱（基准 c220019）

读噪未分辨时的提前返回曾使同一份标定的有效频谱丢失。若文件同时声明合法 `NoiseProfile`，替代系数被记为有效模型，但相关性退回 `unknown`，重新允许当前色度核并使 HDR 噪声因子回到 1。真实合成 DNG 的导入、LibRaw 解码、分析与色度核入口复现了这一组合缺口。

现在先独立验证标定的 DN 尺度并读取适用频谱，读噪未分辨或普通缺测只影响方差系数。DNG 替代后保留原标定来源、原因及横纵频谱；实测异常继续令 HDR 噪声因子为 0、色度核跳过。相机/快门、ISO 域与 DN 尺度不匹配的频谱不会借用；文件声明已降噪或非法仍优先拒绝。预览与分析缓存升级到 23，淘汰旧回退结果。

[真实 DNG 组合回归](../../tests/test_spectral_fallback_pipeline.py)同时覆盖有效读噪、普通插值、无 DNG 替代及频谱域外对照。域外对照仍实际启用色度核，避免把所有回退都禁用来掩盖问题；模型单元测试另覆盖低/零/高/正常频谱、双轴保留、DN 不匹配及缓存往返。10-bit SDR 母版与 bulk 机型匹配在该阶段仍后置，后续见第 13 节。

## 12. 编码范围、截断支撑与噪声坐标（基准 049bfd6）

本轮修复阶段之间的三类证据错位：独立保存 DNG 编码白点及最大黑位，让标定 DN 适用性、a/b 与解码方差传递共用来源；`LinearResponseLimit` 继续描述线性区间，不改变编码尺度。记录 WB／前级校正的实际截断，再覆盖解拜耳及 DefaultScale 的来源支撑。固定文件 BE 进入噪声底、SNR 和黑端点的 scene-EV 转换，RAW 模型及用户 EV 的既有含义不变。

DHT 的原地坏点处理及全帧极值步骤尚无已审计的局部支撑。显式选择 DHT 时保留成像算法，存在前级损失则使用整帧保守权限。`auto + clip` 全尺寸 Bayer 此时选择支持的 AHD，并报告原因；AHD 用经固定 LibRaw 源码核对的半径 5，半尺寸 Bayer 用来源 2×2 最大值。DefaultScale 使用真实累加坐标及两个来源 tap；尺寸舍入后不变也执行。空间高光重建和其他未经核对的算法仍保守退让，不宣称完整局部支撑或精确噪声协方差。

预览缓存升至 24，窗口尺寸以 JSON 稳定列表保存，完整 Analysis 在 envelope 和磁盘往返后均能实际命中；编码范围、几何或固定噪声坐标不兼容仍失效。

三张 fp 实拍的 RAW 证据哈希保持不变，均触发自动 AHD。相对基准，最终 SDR 的逐通道平均绝对差约为 1.93–5.33 个 8-bit 码值，局部最大差更大；这是解拜耳、可靠尾部及黑端规划共同变化的结果，不是编码误差，也不代表量化出的画质提升。对应 HDR 使用量余量分别从 1.122／1.012／1.304 EV 变为 0.943／0.908／1.288 EV，保留正余量。六次全尺寸 SDR JPEG / HDR HEIF 导出通过实际回读及现有门限，独立 SDR 与 HDR base 母版逐字节一致。HEIF 测试使用固定 share q95，JPEG 使用 share-hq q97／4:2:0，未执行自动编码搜索，也没有保证 HEIF 小于 20 MB。

详见[机器可读验收记录](../assets/delivery-quality/pipeline-evidence-boundaries-20261009.json)和新增真实 DNG 回归：[编码范围](../../tests/test_noise_coding_range.py)、[黑位与线性化](../../tests/test_noise_coding_endpoints.py)、[解拜耳支撑](../../tests/test_demosaic_loss_support.py)、[同尺寸 DefaultScale 与 Linear DNG](../../tests/test_decoder_loss_edge_variants.py)、[BE 坐标](../../tests/test_noise_scene_ev.py)、[缓存往返](../../tests/test_analysis_cache_geometry_roundtrip.py)。实拍通过现有交付门限仍不等于全部弱纹理完整；10-bit SDR 母版和 bulk 机型匹配在该阶段仍后置，后续见第 13 节。

## 13. 真正的 10-bit SDR 与先验边界（基准 f10e664）

独立 SDR HEIF 在完成 AgX/其他受支持 SDR 核、输出颜色处理、色域拟合和 sRGB/P3 输出
OETF 后保留 float32，编码边界加入确定性 TPDF 抖动并量化到 10-bit。浮点母版的后处理与
OETF 复用同一张完整 raster，临时计算按块执行。JPEG 和显式 8-bit HEIF 继续走原 uint8
入口。手动、archive 与 auto 均接入新入口；自动搜索和写入元数据后的验收直接度量浮点
回读，不再先转 uint8。误差单位仍是等效 8-bit 码值，原预算和局部纹理阈值不变。

Core Image 使用 RGBAf 工作与输出格式，关闭 HDR 展开，输出非线性目标色域；原生
NSDictionary 解决实际 PyObjC 读取选项的桥接问题。实际 q100/444、无抖动的编码器探针
保留 1024 级渐变，相邻 512/1023 与 513/1023 灰阶在旧 u8 回读中同为 128，浮点回读
能正确区分。正式带抖动的 SDR 出口保留 959–979 个渐变层级，不能把无抖动探针的近乎
精确回读推广为有损 HEIF 的无损保证。ICC 必须精确匹配，NCLX 若存在则检查一致性；
本机 libheif 在嵌入 ICC 时未额外写出 NCLX，报告保持 `nclx_verified=False`。

三张 fp 全尺寸 q95/420、10-bit SDR HEIF 通过实际回读；日光样张 auto 选择 q90/444，
通过相同预算。这里没有给 HEIF 增加 20 MB 硬目标，高感手动输出和日光 auto 均可超限。
JPEG 97/420 与 HDR HEIF 95/420 另作实际回归。具体文件体积、门限和测试范围保存在
[机器可读验收记录](../assets/delivery-quality/sdr10-prior-boundaries-20261009.json)。

包内 curated / JPTC / bulk 统一采用规范化后的完整机型匹配与显式别名；Canon EOS 5D
不再误用 Mark II 先验，未知型号不借用同系列数据。制造商前缀与空格可规范化，但代际和
有区分意义的后缀保留；原用户优先级、快门和 DN 适用性检查保持。

Collect 新导入保留扣量化前的成对暗场总方差，与电子读噪分开。模型在原有适用性及物理
读噪 resolved 检查通过后，优先用声明明确的线性化 DN 总方差形成存储 RAW 的常数项。
旧转换器只有明确的 identity/ADC/Sheppard 合同且 sigma-clip 修正完成，才恢复对应量化项，
并标为绿色汇总近似；其他来源不统一加 `1/12`。导入在实测 ISO 交点用 DN 读噪或电子
读噪/增益检查总方差下界，运行时在当前 ISO 再检查插值结果，保留 5% 容差。总方差明显
低于物理读噪方差，或归一化模型系数非有限时，明确记为 `unresolved` 并保留原因，
不静默退回物理读噪常数项。
独立有效 DNG `NoiseProfile` 可替代系数，原标定的异常频谱仍约束 HDR 与色度核。
未分辨物理读噪不会因保留总方差而自动放行。
预览及完整分析缓存升至 25，避免复用旧机型或方差模型。

实际系统回归还修复了 Apple RAW 的独立参考：`f10e664` 中重建参考的全帧保守损失使可信
样本归零。现在保留该重建参考和原中位比作为亮度标尺，仅在支撑不可验证时另取校正后的
clip/auto 参考作为传感器证据。第二份参考使用自己的存储尺度，失败明确关闭 HDR，并不
撤销已验证的亮度对齐。fp 日光实测保留约 98.0955% 可信样本，相机白平衡 HDR 余量恢复
至约 0.97767 EV；原全/半尺寸对齐因子不变。新增[真实 DNG 参考回归](../../tests/test_coreimage_sensor_reference.py)
覆盖 BE、半尺寸、后置 Kelvin 白平衡、缓冲释放和失败关闭。

Apple RAW 日光的 10-bit SDR HEIF、HDR archive 与 HDR auto 均通过实际编码；HDR auto
选 q85/444，约 31.4 MB。手动 HDR q95/420 在局部纹理指标上为 1.0，未通过，目标文件被
丢弃。辅助质量 100 时，下采样请求 2 与不指定都实际生成全尺寸增益图，且失败指标相同；
不能把这个拒绝错误归因于增益图尺寸，也没有为使手动档通过而放宽预算。
同一母版与同一 gain map 的主图对照中，95/444 的局部指标为 0.39413，95/420 与
100/420 均为 1.0，NumPy/Rust 完全一致。红色细纹处的线性亮度差从 +0.04151 变为
约 −0.00276，满足指标要求的空间支撑；本反例确实是 4:2:0 的局部结构损失，提高 quality
到 100 也不能补回缺失的色度采样。默认 auto 保持 4:4:4 通过验收，手动设置不被静默改写。

回归见 [浮点形成与分流](../../tests/test_sdr_float_render.py)、
[高精度读取](../../tests/test_sdr_float_readback.py)、[浮点指标](../../tests/test_sdr_float_metrics.py)、
[HEIF 出口与实际渐变](../../tests/test_sdr_heif_precision.py)、
[精确机型](../../tests/test_prior_exact_matching.py)与[总方差合同](../../tests/test_calibration_stored_variance.py)。
该阶段 HDR gain-map 的 SDR base 仍为 uint8，独立 HDR alternate 保留 float16；
HDR HEIF 底图与子读出匹配的后续改动见第 14 节。Apple 独立 SDR 编码后备仍限手动
8-bit/420。完整相关噪声传播、FPN/PRNU/坏点与条纹修复、单帧稳健估计仍未完成。
自己的 fp PTC/重复黑场尚未提供，因此没有个人机身校正的实拍验收。

## 14. HDR HEIF 的浮点底图与逐文件采集约束（基准 e55835c）

HDR HEIF 10-bit 的配对形成现在直接输出完成 P3 传递函数的 float32 SDR 底图，
与原 float16 HDR alternate 共用一次场景处理；JPEG 与显式 8-bit HEIF 保留旧量化合同。
`FinishedPair` 明确只允许一个 SDR 母版，编码、搜索、模板和回读均消费同一份底图。
浮点私有 RGB 视图与不可变 RGBAf 输入共享存储，比另存 RGB 快照减少 12 bytes/pixel，
24 MP 对应 288 MB；这是缓冲大小核算，不是端到端峰值内存测量。

实际 API 探针发现，通用 HEIF 写入方法即使使用 RGBAf/h，也只输出 8-bit/256 级主图。
专用 `writeHEIF10RepresentationOfImage_toURL_colorSpace_options_error_` 才保留 10-bit
模板，探针回读 1024 级。生产路径检查新建及复用模板的真实位深，再从同一浮点底图
编码带确定性 TPDF 的 x265 主图。自动候选与 SDR 绝对门禁使用浮点回读，HDR 展开仍与
原 HDR 母版比较；误差预算、局部纹理门限和 headroom 检查不放宽。元数据搬运通过图像
载荷、颜色属性及 rendition 图关系的身份校验，不能用最终 GUI 的 uint8 缩略图代替验收。
[Apple 的双 rendition 说明](https://developer.apple.com/videos/play/wwdc2024/10177/)
解释了用 SDR 与 HDR 两张完成的图像计算 gain map 的合同，实际系统精度另由探针验证。

逐文件采集描述独立于镜头解析和成像裁剪，记录主 RAW IFD 的存储位深、几何、ActiveArea、
DefaultCrop、压缩过程，以及 LibRaw 完整 mosaic 几何。存储位深不冒充 ADC 精度，
DefaultCrop 不冒充像素合并。Compression=7 进一步读取 JPEG 过程及 point transform，
不能把预览 JPEG 或有损过程称为无损 RAW。fp 的电子快门按精确机型的厂商能力登记，
明确标注来源，不由此推导静态/视频、ADC 或 binning 的全部模式。

复核同时修正了增强 IFD 混入主 RAW 的问题：校准标签、镜头指令与采集描述共用默认
LibRaw 首个主 RAW 的选择顺序，排除增强图像、预览和 mask；BE 优先读实际 RAW IFD，
缺失才采用 IFD0。额外增强图像的位深、白点或曝光不能再改变主 RAW 的解释。

标定通过带版本的 `readout_contract` 约束可核对字段；声明冲突与文件证据不足分别报告。
无损压缩与无压缩可按样本保存等价处理；旧自由文本没有足够语义时不猜测相近模式。
新 Collect 的 RawSize 对应 LibRaw 完整 mosaic，ImageWidth/Height 为操作者声明的 JPEG
输出尺寸，后者保留作信息，不与传感器尺寸混比。噪声模型和电子域先验共用匹配结果，
失败的用户约束不会静默换成未经核验的包内相机参数。具体格式及范围见
[标定文档](../NOISE_CALIBRATION.zh-CN.md)。

没有声明更细模式的相机级先验仍保留原近似用途，并明确报告 `sub-readout-not-declared`；
没有可读标签的 ADC、binning、静态/视频或厂商 readout ID 仍为未知。这里完成的是可证
字段的传递、匹配和拒用机制，不能称为所有相机的完整采集模式识别。个人机身实测校正
按用户要求暂缓。

三张 fp 的全尺寸 HDR HEIF archive，以及日光的 LibRaw/Apple RAW HDR auto、JPEG
97/420 和独立 SDR HEIF 95/420，共七次实际导出通过原生产门限。所有 HDR HEIF 均报告
10-bit 主图、float32 SDR 母版与浮点回读；没有为通过样张而覆盖预算。日光 LibRaw auto
选 q90/444，相对本轮参考减小约 16%；文件大小和并发测试期间的耗时只作验收记录，
不能推广为所有场景的压缩率或性能保证。与基准 e55835c 使用完全相同 CLI 的 JPEG
97/420 对照，像素、编码载荷和整个文件 SHA256 都相同。
机器可读数据见[本轮验收记录](../assets/delivery-quality/hdr10-readout-20261009.json)，
实际系统渐变回归见[HDR HEIF 精度测试](../../tests/test_hdr_heif_precision_live.py)。
[逐文件合同测试](../../tests/test_readout_contract.py)覆盖真实 DNG/CSV 至分析与缓存，
[真实 SOF3 测试](../../tests/test_readout_lossless_jpeg_pipeline.py)通过 LibRaw 解码确认
Pt0 保码值、Pt1 丢低位；声明的 tile/strip 字节边界之外不能借用另一个 JPEG 头。

最终冻结树以 `DNGSCAN_FAST=1` 运行 1662 项完整测试：1659 通过、3 跳过、0 失败，
耗时 260.184 秒，包含本机实际 Core Image/ImageIO 与 Rust 核。读出相关 NumPy 回归
255 项中 252 通过、3 项 native-only 跳过；对应 native 255 项全部通过。各组有重叠，
不能相加成独立测试总数。系统渐变验收覆盖 Apple-only 与 x265 的手动、自动 HDR HEIF。

## 15. 解码资格与局部损失分离、奇数 CFA 边界（基准 c9adb84）

附件对 `f10e664` 的两个反例在后续 `c9adb84` 仍能复现。未经认证的 DHT 或空间高光
重建支撑，原先被表示为整帧 RGB 剪切，进而错误触发局部退色。本轮增加独立
`scene_loss_support_untrusted`：局部掩码只保留实际定位的源/解码损失；全局标记撤销
可靠尾部资格、跳过当前色度噪声传播、关闭 gated 的非 RAW 颜色许可，不伪造三通道饱和。
物理噪声系数、原始饱和统计和已有局部 RAW 颜色许可仍各自保留。

自动 HDR 规划没有可信尾部时不分配额外 headroom；两个直接 HDR 渲染入口即使收到
外部固定计划，也会将未认证帧的通道分离设为零。独立 LibRaw 参考明确返回“已测但为空”，
不会因缺少空间掩码恢复资格。新标记贯穿延迟掩码、WB 重分析、Prepared 样本、AutoEV、
proxy 与磁盘缓存，缓存版本升至 27。报告将已知局部损失覆盖率与全局资格分开表达。

两组 128×128 实际 LibRaw DNG 使用相同固定计划：DHT/WB 与非中性 WB 的 GainMap/
reconstruct 各有 16,375 个解码值不变的像素，误变色数均从 16,375 降为 0，可靠样本
仍为 0。浮点 SDR、8-bit SDR、配对 SDR/HDR 均有最终形成回归；固定非零 rho 另有
实际开放对照，排除本来就没有 HDR 色差的无效验收。

传感器剪切聚合、空间黑电平剪切与 headroom/SNR 指导现在保留残缺 CFA 周期中的实际
感光点。部分单元按真实传感器范围注册后再 warp/crop/orient，不能将 ceil 网格均匀
拉伸而把已舍弃的残行重新带入裁剪窗口。真正的解码后 2×2 box 缩小仍按其实际 floor
行为舍弃残边，偶数完整周期的算术保持逐位一致。缺失的颜色样本不会填成三通道剪切。

125×127 DNG 的最后绿色感光点仍产生原有全/半尺寸 scene 差值，但对应 soft mask 现在
分别为约 0.430 / 0.258，可靠性均为 false；G headroom 为 0，clip class 仅为 G。
测试另覆盖最后一列、空间黑电平、四种 Bayer、DefaultScale、非恒等 warp、旋转及实际
DefaultCrop 排除残边。实现不声称这些粗网格掩码等于精确解拜耳协方差。

fp 日光样张在与基准完全相同的 NumPy CLI 下导出原尺寸 JPEG 97/420，像素和整文件
SHA256 相同；全尺寸 HDR HEIF archive 通过生产回读门禁，实际主图为 10-bit/444，
ISO gain map 和 headroom 声明存在。生成图片仅用于临时验收，原文件未修改。
数值及最终测试记录见[本轮验收数据](../assets/delivery-quality/decoder-qualification-20261009.json)。

回归入口：[资格合同](../../tests/test_loss_support_contract.py)、
[实际 SDR/HDR 形成](../../tests/test_loss_support_render.py)、
[缓存及公开入口](../../tests/test_loss_qualification_handoff.py)、
[奇数传感器几何](../../tests/test_odd_raw_clip_masks.py)。DHT/空间高光重建的完整局部
损失支撑仍未认证，当前全局撤权继续保留；个人机身实测校正仍按用户要求暂缓。

最终冻结代码以 `DNGSCAN_FAST=1` 运行完整 1690 项：1687 通过、3 跳过、0 失败，
耗时 242.977 秒，包含本机实际 Core Image/ImageIO、10-bit HEIF 渐变和 Rust 核。
本轮重点 `DNGSCAN_FAST=0` 回归另有 77 项全部通过（2.144 秒），包含独立 partial
oracle 与旧问题的尺度/频谱/BE/缓存回归；这两组有重叠，不相加为独立总数。

## 16. 裁切前的饱和依赖与单点 PTC 读出范围（基准 8db60a3）

新增 `scene_reliability_exclusion`，以最终 scene 对齐的 H×W uint8 表达“这个输出依赖
饱和来源”。它从不可变 RAW、与分析一致的 resolved fullwell 出发，先传播解拜耳支撑，
再经过真实 DefaultScale、TCA/畸变 warp、保留窗口、orientation 和实际 box reduction。
它不写入视觉 clip mask，不增加 processing-loss 覆盖率，也不因源饱和改变自动解拜耳
选择。AHD 使用已审查的半径 5 保守支撑，Bayer half 使用 2×2 支撑；DHT、空间高光重建
及尚未审查的前级空间坏点修复，在存在源饱和时仍使用全局资格限制。

125×127、crop 126×124 的真实 LibRaw DNG 反例中，DHT 的 5 个、AHD 的 3 个保留输出
仍随窗口外 R 感光点改变，但获准进入可靠统计的数量均降为 0。AHD 仅排除 35 个输出，
其余 15,589 个仍可靠，视觉剪切仍全零，processing mask 仍不存在。无 warp 的 half
对照没有输出变化、没有局部排除，3,906 个样本保持可靠。加入径向 warp 后，half 的
6 个和 full AHD 的 20 个实际依赖输出也全部排除；warp 自身的边界处理损失单独保留。
真实 TrimBounds、八种 orientation、DefaultScale 和后置 floor box 另有交叉回归。

可靠尾部、独立 LibRaw reference、可选色度噪声传播和 HDR 逐像素颜色分离均消费新
依赖资格，视觉 retreat 继续只用原掩码。gated 的 `scene_eligibility` 是独立乘数，放在
原有 SNR／EV 噪声底门控之后；缺测 SNR 保持 None，不能用全 1 覆盖缺测回退，进而
放宽未排除区域。方差粗单元使用 ANY 覆盖，可选色度核再按各尺度的实际滤波支撑
扩张无效区域。它仍是低频近似，不宣称重建了解拜耳协方差。

Prepared 和缓存规划样本保留完整源图的精确 sampling indices 及逐行排除；预览像素
则按实际 Lanczos 半径 3 的完整输入支撑传播，不能仅做最近邻或面积投票。缓存升至
28，独立保存场景／样本排除及 gated 资格，缺失、错形或无效数据直接失效。满阱端点
更新时重算依赖，并使已绑定的旧样本一起失效；端点未变时不重复整帧传播。

官方单点 `tools/import_jptc.py` 现复用读出字段解析，保留 typed RawSize 和原始
Compression；JPEG 尺寸仍仅是输出信息。用户安装与包内单点加载均执行同一匹配，
尺寸冲突拒用外部 gain/RN，未知压缩保持不可核对；独立 DNG NoiseProfile 仍可回退。
单点空间 PTC 不继承 Collect 成对暗场的总方差语义。旧转换器已经删除的声明无法
自动恢复，应从原 CSV 重新转换并导入。

同一 fp 日光文件与基准使用相同 NumPy CLI，原尺寸 JPEG 97/420 的像素、字节和
SHA256 完全相同（9,228,234 bytes），默认曝光及 HDR headroom 计划亦相同。该文件
本来已选 AHD，新局部源排除与既有可靠性限制重叠，不能推断其他源饱和 DHT 文件也
不会改变 HDR 分配。当前真实 HDR HEIF archive 输出通过未放宽的生产门禁，主图
4000×6000、10-bit/444、Display P3，ISO gain map 与 headroom 声明均存在。

数值与验证范围见[本轮验收记录](../assets/delivery-quality/source-dependency-20261009.json)。
回归入口：[裁切外真实依赖](../../tests/test_sensor_dependency_crop.py)、
[消费者](../../tests/test_reliability_consumers.py)、
[缓存与采样交接](../../tests/test_reliability_handoff.py)、
[单点正式导入](../../tests/test_jptc_single_readout.py)。分尺度降噪控制、early denoise、
浮点 RAW 前端、Lensfun 扩展以及个人机身暗／平场修正仍为后续工作，本轮不扩展算法范围。

最终冻结代码以 `DNGSCAN_FAST=1` 运行完整 1729 项：1726 通过、3 跳过、0 失败，
耗时 250.722 秒，包含本机实际 Core Image/ImageIO、10-bit HEIF 与 Rust 核。
`DNGSCAN_FAST=0` 的 28 个定向模块另有 243 项：241 通过、2 跳过、0 失败，耗时
5.774 秒；覆盖源依赖、正式 PTC 导入、缺测 SNR 回退、缓存及上一轮尺度／频谱／BE。
两组有重叠，不能相加为独立总数；跳过项不计为通过。

## 17. 多相机通用成像、处理依据与编码尺度（基准 39306ec）

AgXRAW 的目标是处理解码器能够读取的多种相机 RAW，Sigma fp 实测路线不是普通成像的
前置条件。没有适用外部标定、ISO 或 DNG NoiseProfile 时，仍使用文件颜色标定与解码
图像统计完成默认 AgX、自动曝光及 SDR/HDR 形成；HDR 仍受实际高光余量和其他可靠性
限制。这种回退降低的是证据完整程度，不主动降低输出分辨率或编码位深，也不从纹理
方差虚构物理噪声、电子数或 SNR。缺测噪声保持缺测；可选模型色度降噪因无适用模型而
跳过，不能为了让降噪继续运行而采用未核对的相机标尺。

GUI、报告与 CSV 现在区分匹配噪声模型辅助、文件噪声声明辅助、噪声未标定的通用
成像，以及没有传感器证据时的图像统计回退，同时报告模型拒用原因、HDR 可靠性来源
和色度降噪实际状态。这些标签描述已有依据，不另选一套成像数学。普通缺测不会撤销
独立的传感器高光证据；明确 rejected/unresolved 或实测频谱异常仍保留对应限制。
合法 DNG NoiseProfile 可以在没有 ISO 的情况下提供归一化方差，不能据此声称获得
电子域标定；声明机内处理或解码传播不适用时，当前色度核仍按各自条件跳过。

本地非 fp 样张实际为 Sony ILCE-7M5（ISO 1000）和 Fujifilm X100VI（ISO 250）。两者
均执行文件内暗角、畸变与横向色差校正，分别采用 AHD 与 LibRaw 原生 X-Trans 解拜耳。
两张原尺寸 JPEG 97/420 通过 share-hq 生产回读门禁，分别为 14,853,931 和
11,688,618 bytes，均低于 20,000,000 bytes；这不是所有相机或场景的体积保证。Sony
HDR HEIF archive 的实际 10-bit/444 主图、float32 SDR 母版、HDR 展开与 gain map 回读
通过，文件 53,320,957 bytes。Fuji 的同一画面没有可用扩展白，显式 HDR 请求在编码前
被拒绝（可信尾部约 +0.89 EV）；没有为通过样张而放宽门禁。该图的真正 10-bit/444
SDR HEIF archive 正常通过浮点回读，文件 39,008,866 bytes。

两份实片都有包内近似噪声先验，子读出模式仍为 `sub-readout-not-declared`。因此，
实片通过本身不证明缺标定回退。Sony 表格镜头校正的噪声传递和 Fuji X-Trans 的噪声
传递尚不支持，不能将有效传感器方差等同于可用的解码后协方差。额外真实可解码的
未知相机合成 DNG 回归覆盖缺 ISO/缺模型、合法文件模型、明确拒用及缓存往返，检查
AgX、AutoEV、SDR/HDR 形成和可选降噪跳过。最终冻结工作区另用未知机型文件执行完整
CLI，10-bit SDR HEIF 实际编码回读及报告成功，退出码为 0；CSV 保持噪声不可用、SNR
不可用，并明确标为“通用成像（噪声未标定）”。它是缺标定路径的系统烟雾验收，不是
iPhone 或未知真实机型的画质验证。

Sony 实片同时暴露了一个旧尺度入口：LibRaw 全局 `white_level=39002`，各通道编码
白点为 32800，黑点为 1024。独立噪声模型使用正确编码跨度 31776，电子域先验入口却
使用全局跨度 37978，错误报告 `unmatched-dn-scale`。分析器现在与噪声模型共用绿色
通道 `normalized_raw_span`；增益恢复为 0.2167667999648069 e⁻/file-DN，PDR 恢复为
9.167623853802283 EV，模型 a/b 不变。缓存升至 29，避免复用旧尺度产生的分析报告。
同参数 Sony JPEG 与基准整个文件 SHA256 相同，因而像素和编码也相同；这次修正恢复
了证据，并没有改变这张样片的默认成像结果。

验收来源须分开理解：前四次实片调用来自 `39306ec` 基准；Fuji SDR HEIF 来自开发中
工作区；Sony 尺度修后 JPEG 也来自开发中工作区，当时实际导出和回读已成功，但后续
报告调用碰到正在同步的 `export_info` 参数，CLI 退出码为 1。该报告接口随后完成整合，
最终未知机型 CLI 验收为 0；不能将前六次调用统称为最终冻结源的完整 CLI 通过。报告
现在使用实际交付的容器、位深和母版精度，避免将 10-bit HEIF 仍描述成 8-bit JPEG。
数值、文件哈希和逐项源码阶段见[本轮紧凑验收记录](../assets/delivery-quality/multicamera-fallback-20261009.json)。

回归入口：[通用 RAW 回退](../../tests/test_general_raw_fallback.py)、
[处理依据与交付精度报告](../../tests/test_processing_evidence_summary.py)、
[编码尺度一致性](../../tests/test_noise_coding_range.py)。本轮未提供 iPhone standard RAW
或 ProRAW 实片，不把两份相机样张推广为 iPhone 验收；其逐文件噪声声明、颜色解释、
解码器回退和 HDR 余量仍需实际文件验证。未修改原始样张，原尺寸临时输出已清理，
私有缩略预览没有加入仓库。

最终冻结代码以 `DNGSCAN_FAST=1` 运行完整 1747 项：1744 通过、3 跳过、0 失败，
耗时 255.355 秒，包含本机实际 Rust、Core Image/ImageIO 和 10-bit HEIF 回归。
`DNGSCAN_FAST=0` 的 18 个定向模块另有 223 项：222 通过、1 跳过、0 失败，耗时
1.815 秒。两组有重叠，不能相加为独立总数；跳过项不计为通过。最终未知相机完整
CLI 编码／回读／报告退出码为 0，GUI JavaScript 语法检查通过。

## 18. iPhone Bayer DNG 与 ideal-image 验收（基准 38f5bc7）

本轮补充用户新提供的五张 iPhone RAW 和本地 `ideal-image` 合成样张，运行代码以
`38f5bc7` 为基准，没有发现需要新增生产修复的问题，也没有放宽交付门禁。这里只
扩展验收证据，不能用通过样张代替全部机型、解码模式或镜头校正的认证。
已按实际 CFA 标签更正教程中的 standard RAW／ProRAW 分类及旧暗角指令说明。

五张 iPhone 文件均为 Apple iPhone 16 Pro 的 BGGR CFA Bayer DNG，RAW IFD 为
4224×3024，默认裁剪为 4032×3024；16-bit 存储中 BlackLevel 为 528、WhiteLevel 为
4095，Compression 为 7。它们不是 Linear RGB ProRAW，不能仅由文件名认定捕获或
机内处理完全未经干预。ISO 分别为 500、200、800、80、200，BaselineExposure 从
+0.02748 到 +1.85261 EV。全部具有合法 DNG NoiseProfile；`NoiseReductionApplied`
的 0/0 正确保留为 unknown。因此，这批实片证明的是**无个人标定但有文件噪声声明**
的成像，不是完全无噪声模型的回退。电子域增益、读噪与个人机身标定仍不声明。

每张均实际执行 LibRaw 和 Apple RAW `9.dng` 的原尺寸默认 AgX、自动曝光与 JPEG
97/420 导出。LibRaw 执行文件内 `FixVignetteRadial`，全尺寸解拜耳使用已验证截断
支撑的 AHD。Apple 场景路径保留 LibRaw 证据，HDR 可靠性来源为 `sensor-reference`；
没有发生旧版解码器或 LibRaw 场景回退，也没有落入无传感器证据的 1 EV 图像统计
回退。两解码器的中位曝光分析接近，但这不能证明其纹理、噪声或成像结果相同；
Apple 输出较小也不能单独说明压缩效率或画质更好。

完整 CLI 实际编码／回读／报告共 16 次，退出码均为 0：五次 LibRaw JPEG、五次
Apple JPEG、一次额外开启 NR 的 LibRaw JPEG、三次 SDR 10-bit/444 HEIF，以及两次
HDR 10-bit/444 HEIF。JPEG 均低于 20,000,000 bytes；这只是本批文件的结果。三次 SDR
HEIF 为 ISO 800 的 LibRaw/Apple 双路径和 ISO 80 的 LibRaw 路径，均保留 10-bit
量化及 float32 回读。HDR 使用 ISO 200、BaselineExposure +1.85261 EV 的一张图：
LibRaw 与 Apple 实际母版范围分别为 +1.86440768 和 +1.93872468 EV，文件声明与母版
差各小于 0.0003 EV，SDR 主图与展开 HDR 均通过现有回读门禁。其文件分别为
23,831,373 和 4,846,278 bytes，未将 archive 的高质量文件强行压到 20 MB。

文件噪声模型 valid 不等于可选降噪已经适用。LibRaw 的暗角／点操作路径尚无对应
噪声方差传播，Apple 解码传递也仍不透明；ISO 800 显式 NR=1 均跳过。额外 LibRaw
NR=1 JPEG 与关闭 NR 的完整文件 SHA256 一致，验证了跳过没有静默修改像素。它是
保护条件正确生效的证据，不能描述成 iPhone 降噪成功。

`ideal-image` 的 A/B 是已知参数的合成 CFA，型号含 `ILCE-7RM2 synthetic ideal-sampled`
后缀。两份原始文件均没有 ISO 或 NoiseProfile，没有误借真实 Sony 型号的先验；
它们可直接覆盖通用无模型成像。真值工具在 NumPy 与严格 native 两路均为 7/7：
黑电平和元数据白点正确，PTC 增益恢复 5.7134 e⁻/ADU，相对真值偏差约 0.20%；
去量化读噪为 2.861 e⁻，相对 3 e⁻ 真值偏差约 4.6%。推导满阱检查与增益检查代数
相关，不是另一份独立测量。原始 A/B 的完整和半尺寸 SDR/HDR 在两路均形成有限
输出，模型降噪跳过，与关闭结果一致；成对 SDR 与独立 SDR 的差为 0。

为了验收可用模型下的色度核，另在临时 B 副本声明真值 shot/read 方差和 ADC 量化
方差，未改原始文件或场景样本。半尺寸近似传播下 NR 实际激活，两后端校正图最大
绝对值约 0.001326。五组后端对照中 scene 严格一致，非线性 SDR 最大差不超过
3.91×10⁻⁵，HDR 最大差不超过 1.65×10⁻⁵。使用同一曝光及 B 的 tone/color plan 与
无噪声 A 比较，这一个固定噪声实现的全图色度 RMSE 从 0.002617232 变为 0.002612760，
约改善 0.17%；天空渐变区域约改善 0.47%，人脸区域几乎不变。亮度 RMSE 基本不变。
这是幅度很小、受这份合成信号与尺度约束的结果，不支持广泛降噪收益或全部纹理
保真的结论。

机器可读的源文件／输出哈希、逐项编码与 HDR 回读指标、真值和后端误差见
[紧凑验收记录](../assets/delivery-quality/iphone-ideal-20261009.json)。本轮没有验证
Linear RGB ProRAW、Apple 单独可读文件、其他 iPhone 机型或旧版本解码回退。原始
RAW 与 ideal-image 文件未改，临时图片不提交仓库。本轮冻结回归计数保存在验收记录
的 `validation` 字段：启用本地 ideal-image 资产、`DNGSCAN_FAST=1`
运行 14 个定向模块共 135 项，134 通过、1 跳过、0 失败，耗时 42.776 秒。该组与
上述真值及生产调用覆盖有重叠，不能相加为独立总数；跳过项不计为通过。
