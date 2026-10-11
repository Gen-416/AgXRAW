# Core Image half / float 读回研究

执行日期：2026-10-10 America/Los_Angeles / 2026-10-11 UTC。产品基线：`1f3c8af6c3a9d09b188ebd47f6d88d418c07980a`。本轮只改研究工具和测试，默认解码、缓存、AgX、降噪及交付门限均未改变。

**结论：此前的超预算来自读回舍入合同与验收假设不一致。** 当前机器上，Core Image 直接写 RGBAh 的 RGB 结果全部等于同一 CIImage 的 RGBAf 结果向零舍入（RTZ）。旧预算按最近偶数舍入（RNE）建立；RTZ 可以超过半个 ULP，却仍小于一个 ULP。保留旧预算失败记录，同时增加舍入模式诊断，不能将放宽门限描述成提高画质。

本轮决定保留默认 half 读回。完整 RGBAf 读回后显式转 half 是可验证的数值改进候选，在本组数值比较中显著降低向零偏差，仍保留原 half 存储；但本轮没有证明可见细节改善，临时内存增加明显。分条 RGBAf 候选未达到完整读回的像素等价，暂不接入。没有理由因此把整个处理链改成 float32/float64，也不能在已经丢失尾数的 half 上统一加半个 ULP 来伪装最近舍入。

机器记录：[coreimage-half-readback-20261011.json](../assets/pipeline/coreimage-half-readback-20261011.json)。记录包括源文件 hash、实际 token、runtime、完整配方、冻结计划、所有值的误差统计、负结果和研究脚本身份。私人原片及完整浮点缓冲没有放入 Git。

后续已将本轮 fp HDR 负例定位为最终有损 SDR 主图与增益图参考不一致，并保留默认 half 修复，见 [fp HDR 主图参考](FP_HDR_PRIMARY_REFERENCE_2026-10-11.zh-CN.md)。下文失败结果保留其原执行基线；不将封装改进计为提高前端 half 精度的收益。

## 差异来源

实际环境为 Apple M4 / Mac16,1、macOS 27.2 build 26B5101f、Python 3.14.4、NumPy 2.5.0。相机形成和证据使用项目固定的 rawpy / LibRaw。本结论限定于这次实际 runtime，不是 Apple 对所有版本的舍入承诺。

先绕开 RAW：把已知 signed RGBAf 像素交给 CIImage，再分别写 RGBAf 和 RGBAh。控制覆盖正负值、精确 half 点、四分点、ties-even 中点、近黑 subnormal 和超出 half 有限范围的数值。五种 context 配置分别为生产配置、显式 RGBAf 工作格式、显式 RGBAh 工作格式、请求软件渲染、禁用色彩管理。有限 half 范围内每组 51,426 个 RGB 分量均逐位符合 RTZ，无超过 1 ULP 或额外 subnormal 清零。RGBAf 正常范围与输入一致；极小的约 `1e-40` float32 subnormal 在读回中归零，另行记录，不混成 half 缩窄误差。请求软件渲染不证明实际执行设备。

随后对三份真实全尺寸 RAW 使用同一个已配置 CIImage、context、extent/origin 和 extended-linear Rec.2020，读取 half、float、重复 float。没有改曝光、白平衡或 RAW decoder。

|文件|实际 token|尺寸|RGB 分量|RTZ 精确匹配|超过 1 ULP / 无法分类|
|---|---|---:|---:|---:|---:|
|Sony DSC00225.ARW|8|7008×4672|98,224,128|100%|0 / 0|
|Nikon D750 NEF|9|6016×4016|72,480,768|100%|0 / 0|
|Sigma fp _SDI0016.DNG|9|6000×4000|72,000,000|100%|0 / 0|

合计 242,704,896 个 RGB 分量全部匹配 RTZ；三个样本的重复 float RGBA 读回最大差均为零。fp 包含 706,599 个负分量和 27,571 个 subnormal 范围参考值，也全部符合 RTZ；其中 25 个合法缩窄到带符号零，没有额外 flush-to-zero。原 RNE 预算仍分别报告约 2,720 万、1,974 万、1,946 万个超限值；这与 RTZ 全匹配并不矛盾。

正常正值因此略偏低，负值略向零移动。三份样本正亮度平均差约 `−0.000506～−0.000508 EV`。这只是配对数值差，未做自由曝光拟合，不是曝光测量误差或传感器噪声指标。

本轮还保留 alpha：fp 为 1，Sony 在 1 附近有微小变化，Nikon 约为 0.9624～1.0381。对同一 Nikon CIImage 单独切换 context 的 `outputPremultiplied=True/False`，RGBAf 结果逐位相同，显式 True 与原生产默认也相同。不能仅凭 alpha 非 1 就套用预乘/反预乘公式；它也不能解释已被逐位 RTZ 完整解释的 RGB 格式差异。

Apple 的 bitmap API 指定目的格式，没有保证 RGBAh 等于 RGBAf 最后一次 RNE 转换；[workingFormat](https://developer.apple.com/documentation/coreimage/cicontextoption/workingformat) 控制中间缓冲，目的 RGBAf 也不代表 Apple 内部全为 float32。[Metal 4.1 规范](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) §1.6.7 区分纹理写出的 native/rte/rtz，但不能由此确定 Core Image 使用了哪个内部算子或设备。本报告的 RTZ 结论来自实际比特比较。

## 对最终成像的影响

主实验对每份 RAW 只分析一次，冻结 Analysis、AsShot WB、sceneScale、用户 EV、AgX SDR 计划和 HDR 计划；关闭额外色度处理及 scene transform。三种变体只替换同一位置的像素：直接 half、RGBAf 后 RNE 转 half、保留 RGBAf。后两种是研究交接，不是现有产品默认已支持的输出策略。SDR/HDR 使用生产形成函数，编码使用生产 JPEG97/420、10-bit HEIF99/420 和原交付门限。

|样本|直接 half：8-bit 通道变化率 / 最大差|RNE half：8-bit 通道变化率 / 最大差|
|---|---:|---:|
|Sony|2.3012% / 1 code|1.3673% / 1 code|
|Nikon|1.0976% / 1 code|0.5856% / 1 code|
|fp|1.4986% / 2 codes|0.8791% / 2 codes|

参照为相同冻结计划的 float 输出，量化抖动 seed 相同。这里是 RGB 通道占比，不是像素占比。RNE 使输入 RGB RMS 约减半，并大幅降低单向偏差，但不保证每个最终通道的最大误差都减半：非线性色域处理和输出传递曲线会放大靠近零的通道差异。

Sony 在约 3,274 万像素中，仅 18 / 2 / 1 个像素的 SDR 浮点差超过 1 / 2 / 4 个 10-bit RGB proxy codes。最大点的输出从 `[0,0,0.017123]` 变为 `[0,0.001226,0.011335]`，pre-DRT Y 只从 0.02273254 变到 0.02272859。fp 对应计数为 48,673 / 4,445 / 66（约 2,400 万像素）；其最大点从 `[0.997044,0.558341,0.011442]` 变到 `[0.997158,0.558090,0.005431]`。后者是明亮饱和颜色的近零蓝通道，不能统称“整像素近黑”。RNE 降低了这些尾部计数，但没有完全消除色域边界的敏感性。

三份样本的 18 个主实验 SDR JPEG/HEIF 全部通过原有门限。JPEG 跨变体回读最大差为 6～9 个 8-bit RGB codes；HEIF 约为 12 个 10-bit RGB proxy codes。这些是不同母版分别有损编码后的差异，不能当作 half 读回本身丢了 9/12 级。Sony 同一母版重复编码的 JPEG 与 HEIF 均字节相同、回读差为零，说明本次跨变体差异包含确定的编码阈值/预测变化，而不是观察到的随机编码波动。各文件同时对自己的编码前母版验证，没有只看两个成品的差值。

HDR 浮点形成中，直接 half 的最大绝对 RGB 差为 Sony 0.000855、Nikon 0.000941、fp 0.004053（显示线性 P3，参考白为 1）；RNE 候选相应降低。Nikon 三个实际 HDR10-bit HEIF 均通过生产回读；真实 packed 峰值及文件 headroom 都是 1.5859375。其展开 HDR 跨变体 RMS 约 0.00051～0.00077，包含编码/增益图及回读边界；不能全部归因于前端 half。

fp 的三个 HDR99/420 文件均被生产 HDR 回读门限拒绝并丢弃：局部细节分数约 0.970～0.971，其他误差也几乎一致。float 参照同样失败，因此这不是改成 float 就能解决的问题。本轮保留为独立的该样本/配方 HDR 交付负例，没有放宽门限、没有提交失败文件，也没有宣称完成该场景的 HDR 画质验收。

默认 aligned 路径另做三次重新分析，避免自动补偿混入固定计划实验。三机型三种变体的 autoEV 和白端点在本组样本中均不变；alignment 及黑端点有小变化，直接 half 与 float 的黑端点差约 `0.00016～0.00021 EV`。这不保证其他接近决策阈值的场景都保持同一个自动档位。

100% 同坐标裁图未显示明确的细节收益；这是有限人工观察，不是独立留出或盲测。此次没有传感器输出域的独立噪声标尺，不能将误差换成“几倍读噪”，也不能由普通照片证明真实纹理被恢复。

## 修法与资源取舍

fp 全尺寸、独立新进程、验证前进程 RSS：直接 half 约 794 MB，完整 float 后转 half 约 1,176 MB。单次首次读取耗时约 1.43 / 1.42 秒，不能据一次冷启动断言哪种更快；完整 float 策略的临时 bitmap 从 192 MB 增至 384 MB，最终 RGB half 仍为 144 MB。RSS 不是完整 GPU/ANE 内存预算，时间也不是整条导出耗时。

128/512 行条带候选已先用真实 identity CIImage 验证坐标合同：bitmap 首行对应矩形高 y 一端，条带 bounds 必须从高 y 到低 y 排列，条内不翻转；非零原点、末尾不足一条均覆盖。初版原型的低 y 拼接错误保留为研究失败，不计入产品缺陷。修正后实际 fp RAW 条带仍有约 0.191% 的 RGB half 值不同，最大差 0.005859375，因此拒绝候选。仅凭较少内存不构成可替换生产路径的证据；不猜测有限 overlap 一定能修复自适应 RAW graph 的区域行为。

追加同一 CIImage/context 的控制中，完整 float 读回后转 half 仍与既有完整参考逐位一致，两个条带候选分别仍有 138,198 / 137,787 个分量不同，排除了“仅由不同进程或不同滤镜实例导致”的解释；尚未定位到具体内部算子。候选资格检查失败后不再计入后续热缓存测速。直接 half 与完整 float 各做四轮热缓存读取，含 bitmap 分配、同步读取和 RGB half 交接的中位耗时分别为 0.125 / 0.111 秒。本样本未显示完整 float 候选的热读时间惩罚，但不能外推为整条导出加速；同进程峰值内存累计，不能按模式归因，内存取舍仍采用前述独立进程结果。

本轮实际改动只在 `tools/validate_sensor_precision.py` 和对应测试：原 RNE 预算保留，另报告四种转换模式的精确比特匹配、ULP、正负/subnormal/溢出及完整 RGBA 重复证据。25 项受影响定向测试通过，无跳过；扩展后的工具在本机 fp 全尺寸真实入口再次得到 RTZ 100% 与 RNE 预算失败同时成立。没有重跑与本次无关的全量测试，也没有把此前测试计数叠加为新增覆盖。

后续若提供高精度读取，应优先考虑“完整 RGBAf → 明确 RNE → 既有 half 存储”，记录新的读取配方和缓存身份，并重新验收预览、全尺寸及实际文件；不改变 RAW9 内建处理声明或放开其噪声传播门控。若继续研究条带，应在 float32 层保存同一 CIImage 内交错 full/strip/full 的结果，再物化 float 图像做 identity 对照，统计差异与条带边界的距离和移动关系。当前默认不采用尚未等价的条带，也不把一次实验的 RTZ 观察当成永久平台合同。

复核当前工具的基本命令（实际 RAW 路径由执行者提供）：

```sh
DNGSCAN_FAST=1 python tools/validate_sensor_precision.py --size 32 \
  --apple-raw /path/to/source.dng --apple-version 9 --apple-full \
  --output /tmp/coreimage-half-precision.json
DNGSCAN_FAST=1 python -m unittest tests.test_precision_budget \
  tests.test_coreimage_half_storage tests.test_scene_scale_contract -q
```

Sony 本文件的实际 token 是 8，不能改成 9 后静默回退并计作同一实验。工具拒绝覆盖源 RAW；精度比较不要求保存永久无损母版，也没有改动普通 JPEG/HEIF 的有损交付目标。
