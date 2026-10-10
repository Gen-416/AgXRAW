# 完整频谱保存与带域噪声方差原型

[实测标定](../NOISE_CALIBRATION.zh-CN.md) · [噪声与纹理研究](NOISE_TEXTURE_RESEARCH_2026-10-09.zh-CN.md)

日期：2026-10-09。对应审查计划 F02/F06/C06。完整频谱已进入标定导入与保存；相关噪声的带域方差计算目前是显式调用的研究原型，没有自动接入普通照片的解拜耳／色度降噪，也不会因导入标定而开启降噪。

## 保存了什么，哪些信息可以约束生产模型

`noise_spectrum` 保留逐 CFA 位置、逐 ISO、横纵方向的 `single_power` 与 `diff_power`，原始 CSV header、文件 SHA-256、变换长度、窗口、频率、采样间距、归一化声明、对应 within-line 方差，以及积分检查结果。原始数组保留源 CSV 的数值精度，不从摘要重建频谱。没有足够合同的旧 CSV 仍可保存，但标记 `incomplete-contract`，不能据此获得带域方差资格。

`C00/C01/C10/C11` 表示 2×2 空间位置。导入器用 dark CSV 的 `Channel + ColorIndex` 和实际 colour description 建立颜色映射，并检查跨 ISO 一致性；不能把位置后缀当成绿色索引。上游 `CfaPattern` 字段来自 LibRaw `cdesc`，并非 RGGB 空间排列。[上游 binding](https://github.com/y-g-jiang/JiangtherapeeTesterView/blob/57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c/native/src/binding.cc)、[CSV 合同](https://github.com/y-g-jiang/JiangtherapeeTesterView/blob/57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c/src/output/darkCsv.mjs)

生产模型继续逐平面读取独立的高／中频比，任何实际参与 RGB 传播的平面异常，都不能被另一平面的相反偏差抵消。`0.1/1.9` 不再平均成 1。兼容的绿色单值摘要选择离 1 最远的对数比值；运行时保留各平面数值。摘要正常不证明白噪声，也不能排除窄峰或低频结构。完整 PSD 的存在不会自动解除现有异常门控。

## 归一化和频率单位

固定核对的上游提交为 `57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c`。其每行／列先去自身均值，再计算

\[
P(k)=\frac{\langle|\operatorname{FFT}((x-\bar x)w)_k|^2\rangle_{\rm lines}}
{N\sum_nw_n^2}.
\]

这份 CSV 的值是每个 bin 的方差贡献，单位 DN²/bin；不是还需要乘一次频率间距的密度。单边谱内部 bin 尚未翻倍，DC 和偶数长度的 Nyquist 只计一次。差分谱保留 A−B 的功率，等方差、独立曝光的单帧时域噪声还需除以 2。若转换成连续频率密度，需要先除以 `Δf`，积分时才再乘 `Δf`。[频谱归一化实现](https://github.com/y-g-jiang/JiangtherapeeTesterView/blob/57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c/src/dsp/spectrum.mjs#L99)、[单边谱与差分声明](https://github.com/y-g-jiang/JiangtherapeeTesterView/blob/57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c/src/output/darkCsv.mjs#L169)

矩形窗下，恢复完整单边权重后的积分应等于 `WithinRowVarSingle/Diff` 或 `WithinColVarSingle/Diff`。它不应与整平面的 `StdA²` 比较：后者还含行间／列间均值的变化。当前导入采用 20 ppm 的矩形窗积分容差，覆盖源 CSV 六位有效数字的舍入；倍数错误会保留原始数组并报告 `reference-mismatch`。Hann 窗与未加窗的空间参考只作比较，不宣称严格 Parseval 恒等式。[参考方差来源](https://github.com/y-g-jiang/JiangtherapeeTesterView/blob/57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c/src/analysis/darkPair.mjs#L210)

每个 Bayer 相位的相邻样本间距是两个传感器像素。因此

\[
f_{\rm sensor}=f_{\rm plane}/2.
\]

这与上游 `FreqUnit` 中的 “x2 for sensor pixels” 文字相反，故同时保存原始说明和独立推导的单位。回归使用传感器周期 16 像素的正弦：相位采样峰值为 0.125 cycles/plane-pixel，对应 0.0625 cycles/sensor-pixel。窗口／DefaultScale／解码网格的进一步变化需要另行传播，不能仅凭此除以 2 就假定已经映射到 scene 网格。

读取保存后的 JSON 会重新核对变换长度、频率、源合同、功率积分和诊断状态，不能单独修改 `integration_status` 或方差摘要就获得资格。整个本机记录还必须小于现有 16 MiB 读取上限；大目录在写入前明确拒绝，避免导入成功后无法再读。每方向另有 400 万功率值的计算上限。

## 固定线性核的可验证原型

实现位于 `dngscan/spectral_variance.py`。`measured_phase_detail_variance()` 只选择实际测得的 ISO，不插值／外推 PSD；默认需要 h/v 两方向、矩形窗和对应差分方差参考。Hann 需要显式 `allow_hann=True`，且其积分与空间参考仍须在所选方差容差内。调用者必须声明 `assume_separable=True`，因为两个一维边缘谱无法唯一确定任意二维协方差。

在平稳、可分离的假设下，用横纵归一化功率质量构造

\[
S_{xy}=S_xS_y/\sigma^2,\qquad
\operatorname{Var}(d_l)=\sum_{f_x,f_y}|H_l(f_x,f_y)|^2S_{xy}(f_x,f_y).
\]

这里 `H_l` 是固定 B3 spline à-trous 级联相邻尺度之差；单次 B3 响应为 `cos⁴(π·2ˡ·f)`。实现把二维求和分解成一维内积，时间与临时内存随 `Nh+Nv` 和层数增长，不分配 `Nh×Nv` 频谱网格。横纵积分或外部指定方差不一致时拒绝计算，不把缺失方向补成白噪声。

`project_independent_phase_bands()` 在显式承认相位间独立、并提供固定 3×phase 线性投影后，把各相位带域方差转换为 `chroma_nr.chroma_correction_map(..., detail_variance=...)` 已有的逐层 RGB 方差输入。回归实际连接了这两个接口，验证非零噪声校正、零亮度变化、常量图的零校正以及无效边界的保护。这里的矩阵必须明确包含 DN 到处理域的尺度和色度投影；不能直接套用到未知解拜耳传递。

`separable_linear_probe_variance()` 另对单个已知线性采样点作有界的协方差计算，允许最多 65×65 源样本的固定权重，以及同一足迹内已知的正 GainMap。它从独立 PSD 计算相位网格的相关函数；若先作空间增益 `G`，再作固定核 `H`，使用 `H G C Gᵀ Hᵀ`。变化的 GainMap 不能简化为输出像素中心增益的平方。此接口仅预测明确足迹内的方差，不推断边界规则或自适应核。

原型有以下适用限制：

- 去掉的行间／列间功率不能从 h/v PSD 自动恢复。低频／条纹占主导时，即使两方向积分接近，也不能据此证明二维可分离。
- 单帧谱和差分谱之差可以描述固定结构的功率，不提供其空间相位，不能生成逐像素 FPN／坏点校正图。
- 自适应解拜耳、几何重采样和非线性操作不能冒充固定线性卷积；它们需要经过验证的传递模型或处理后实测。
- 预测是平稳网格内部的方差。反射边界、缺失像素及局部变化的核需要独立处理；原型接入测试通过有效掩码保护边界。
- 相关性假设与窗口近似都是显式研究选择，不是对所有 RAW 的自动资格声明。生产独立噪声近似仍沿用原来的保守跳过条件。

## 数值验证与开销

新增测试覆盖完整 PSD 的 CSV→导入→本机记录→运行时 prior 往返；偶／奇变换长度、DC／Nyquist、差分除 2；错误归一化、错误单位、缺失合同、歧义参考、Hann 选择与禁止外推。按 C06 的验收项，当前覆盖如下：

| 验收输入／操作 | 预测与观测的对照 |
| --- | --- |
| 白噪声 | 白谱与现有独立 B3 方差在 float32 下相同，并与空间高斯噪声的实际带域方差比较 |
| 横向／纵向相关噪声 | 分别生成单方向 AR(1)，以及横纵不同参数的可分离二维高斯噪声，比较三个固定 B3 层 |
| 窄带峰 | 已知双方向谱峰的解析响应，与实际周期信号经 B3 级联后的能量一致 |
| 低频结构 | 可分离低频周期结构可预测；纯单方向条带、加性十字条带的逐行去均值缺失功率不能冒充完整模型，提供独立总方差时明确拒绝 |
| 固定 GainMap 与曝光／单位缩放 | 常量增益及线性尺度改变的带方差按总倍率平方变化，并与实际输出比较 |
| 变化的空间 GainMap | 8 万份独立相关高斯局部样本，在已知增益足迹前后，与完整 `G C Gᵀ` 预测比较；细节核用例还排除中心增益平方的错误简化 |
| 线性重采样 | 固定双线性权重、3×4 面积平均和 B3 细节核，分别比较 GainMap 前后的局部方差 |
| 固定线性解拜耳参考 | 四个独立相关相位、不同方差与 WB 增益，经过显式 signed bilinear Bayer 参考；红格点 RGB 方差与各相位平方权重预测比较 |
| 模型不适用 | 保留现有真实 DNG 的异常频谱→NR 跳过／HDR 门控回归；原型不会提供默认解码器的自动放行入口 |

这些合成输入是独立生成的算子验收证据，不是从待处理照片的纹理反推噪声。固定线性解拜耳参考的通过不证明 LibRaw 自适应解拜耳已完成同等协方差标定。

一份 512×512、横纵 AR(1) 参数为 0.75／0.35 的合成高斯噪声，固定核 Monte Carlo 的实测／预测带方差比为 0.9969、0.9981、0.9932。它验证所声明的可分离模型和固定核计算，没有验证真实相机解拜耳的协方差。

本机 Python／NumPy 微基准：h/v 变换长度均为 4096、8 个细节层、单相位，200 次平均约 **0.40 ms**；该函数调用的 tracemalloc 峰值约 **149 kB**。这些数字不含 JSON 验证、读取、图像运算或真实 RAW 显影，不能当作整体导出性能。

在本次执行中，六个相关模块的 **89 项聚焦测试通过**，包括真实合成 DNG 频谱门控和用户标定往返；其中完整频谱／方差原型的 26 项还通过 NumPy 路径复跑，不能将其重复相加成额外用例。相机实测 PSD、处理后二维协方差和可见细节收益仍需之后的独立标定／实拍证据。
