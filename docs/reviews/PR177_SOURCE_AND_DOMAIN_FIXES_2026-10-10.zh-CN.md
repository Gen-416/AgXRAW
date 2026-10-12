# PR177：原片保护与标定数值域修复

复审基线：`4f7a4704602387217e215b31e2497b239a73a9b5`。本轮按用户选择修复 F01／F02；G01–G03 仅整理 [后续实施顺序](PR177_REQUIREMENTS_NEXT_STEPS_2026-10-10.zh-CN.md)，没有接入新的默认校正或降噪。

后续状态（2026-10-11）：[v3 实施记录](V3_PIPELINE_CONTRACTS_2026-10-11.zh-CN.md)已接入 G01 拍摄事实、G02 逐能力资格及单点 PTC 数值域合同；[作用域修复](CALIBRATION_SCOPE_LEGACY_2026-10-11.zh-CN.md)补齐显式限制、混合供应者和快门收窄。本文的 profile-wide 行为与“后续计划”仅对应历史基线。G03 的完整精度预算及真实留出仍未全面闭环，不以这些合同修复替代。

## F01：已知原始输入不得成为交付目的地

统一保护同时保存输入路径的规范身份和已存在文件的 inode 身份，并在处理前、最终提交前核对。显式源路径、bundle 路径及其 evidence 路径都受保护；请求目标与容器调整后缀后的最终目标分别检查。输入路径后来被替换时，最初记录的源身份仍保留。

JPEG 固定／自动、SDR HEIF、HDR gain-map、CLI 诊断 PNG／CSV 和 GUI 导出使用同一规则。CLI 在执行任何输出前检查全部目标，避免先写诊断再发现另一个目标指向原片。PNG／CSV 使用私有候选和原子提交；元数据复制以及自动编码候选不能吞掉原片冲突异常后报告成功。array-only writer 无法从像素推断来源，提供显式 `protected_sources` 合同，并将保护快照传到最终写入。

五个明确接收 RAW 输入的开发命令也接入保护：`validate_native_raw`、`validate_coreimage_raw`、`validate_raw_failures`、`validate_calibrated_pipeline` 和 `calibrate_raw9_anchors`。包括它们的报告、已知派生 PNG、worker 结果和 anchor 安装目标；没有声称所有任意研究资产写入都能自动推断 RAW 来源。

拒绝同路径、相对／绝对别名、目录链接、指向原片的符号链接或硬链接。这里保护的是已声明输入，不是禁止覆盖任意既有成品，也不是对恶意进程并发替换整个目录树的安全隔离承诺。测试使用一次性合成 DNG／临时占位文件；不以真实用户原片做破坏探针。

## F02：单位、解包域与实际 evidence 共同决定资格

固定版本 [Collect writer](https://github.com/y-g-jiang/JiangtherapeeTesterView/tree/57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c) 的受支持格式统计 LibRaw unpack 后、扣黑／WB／解拜耳前的整数码。信号单位规范化为 `libraw-unpacked-dn`，方差／功率为 `libraw-unpacked-dn2`；原始 A−B 差分和单帧方差的换算仍由对应拟合器处理。格式可以提供这项权威单位缺省，不能提供另一张 CSV 缺失的拍摄条件。

显式 electrons、electron-squared、normalized RAW、scene RGB 或未知 decoder 声明不再按 DN 默默解释。没有实现相应受验证的单位变换时，保留测量和拟合供检查，统计许可保持未验证，诊断显示原因。导入保存的资格摘要在 JSON 读取时重算，不能通过编辑摘要绕过。传统未声明这项新合同的 curated／手工数据保留既有策略；显式声明则必须参与检查。

不可变 RAW evidence 记录实际解包提供者、整数 CFA／线性 RGB 类型、LibRaw LUT 状态、通道描述、空间相位和编码跨度。Collect 的 `CfaPattern` 来自 LibRaw `cdesc`，不是 2×2 空间排列；两者分别与正确的证据字段核对。LinearResponseLimit 不是编码尺度，继续使用文件编码端点。

LibRaw 版本字符串保留为来源，不要求机械相等。当前兼容范围限于可验证的 LibRaw 整数 CFA unpack、恒等 LUT、声明的读出／编码约束及已有 DN 跨度传递；未知提供者、非恒等 LUT 或无法确认的域不取得许可。码值格点只能否定不相容的步长，不能测出物理 ADC、证明统计独立或证明两个 decoder 逐位一致。

格点按各 CFA 相位独立检查，避免把相位间的常量偏置误认为 ADC 步长。传统物理读噪条目若未声明这项新域合同，保留既有 fallback；其无效 ADC 仍不能用于恢复 stored variance。新 Collect／显式域合同中的 ADC 则必须有限、正且可表示。

合法 DNG NoiseProfile 和不依赖物理噪声模型的基础显影继续独立回退。本轮维持既有 profile-wide 保守资格，未实施 G02 的逐能力拆分，也不解除 RAW9 输出协方差门控。

## 缓存迁移与验收边界

预览缓存由 34 升至 **35**。旧 Analysis 可能依据不受支持的单位取得模型许可，因此旧代理及 Analysis 不复用；首次重新分析。新的解包域描述随 capture_readout 保存，JSON 往返和实际域变化均参与缓存一致性。

数值域资格不等于获得了用户机身的实测标定。官方 writer 夹具和合成 DNG 用于证明导入、匹配、模型及回退的连接；真实 RAW 冒烟与编码回读用于证明指定文件和配置可用。独立暗场／平场／弱色纹理留出验收、完整捕获条件、逐能力回退和精度预算仍按后续计划推进。

## 本轮执行结果

机器记录：[pr177-source-domain-fixes-20261010.json](../assets/pipeline/pr177-source-domain-fixes-20261010.json)。记录以 `4f7a470` 加本轮工作区修改为基线，保存 130 个处理源文件／资产的 SHA-256、测试及验证工具身份；测试前后哈希一致。没有把尚未提交时的工作区冒充为 `4f7a470` 原版。

本机为 macOS 27.2 / 26B5101f、Apple M4；Python 3.14.4、rawpy `0.27.0+libraw.e419de08` / LibRaw 0.22.0。固定依赖为 rawpy `cc7b4748c7b3e87da319198fdfcdb46e17c9c2a6`、LibRaw `e419de08001de28ae6988ecb22df47e52b9c5eaa`；native ABI 22，Apple recipe 2。

同一组 2,126 项全量回归分别执行，两个后端不相加：

|后端|通过|跳过|失败／错误|用时|
|---|---:|---:|---:|---:|
|Rust 严格路径|2,123|3|0|271.5 秒|
|NumPy|2,091|35|0|715.5 秒|

```sh
DNGSCAN_FAST=1 python -m unittest discover -s tests
DNGSCAN_FAST=0 python -m unittest discover -s tests
```

新回归位于 `tests/test_source_output_protection.py`、`tests/test_raw_tool_output_protection.py` 和 `tests/test_measurement_domain.py`，覆盖生产 API／CLI、提交前别名变更、自动候选终止、报告保护，以及官方 writer 包到真实合成 DNG 解码／匹配／分析。另执行 10 组数值域探针：正常与兼容声明取得资格，错误单位／未知 decoder 不生成外部模型，合法文件 NoiseProfile 保留回退；基础 scene 与正常基线逐位相同。合成统计不代表真实相机测量。

实际 Sony RAW8、Nikon RAW9、fp RAW9 共六个 SDR／HDR HEIF 出口通过既有生产回读门限；独立检查 10-bit、P3 ICC、方向、尺寸及 HDR gain-map／展开回读。使用原生预览、长边目标 1280，不是全尺寸验收。Sony／Nikon 实际文件的 GUI auto→固定 token，以及 envelope／磁盘完整 Analysis 复用均通过；全部源文件 hash 不变。纯 Python wheel 在隔离安装下可导入新增模块，28 项数据资产的 hash 与源一致。

对 fp 另做同源、同 RAW9／尺度／配置、同 Rust 后端的受控修前对照：从 Git 临时快照运行 `4f7a470`，母版 hash 和当前代码一致，SDR HEIF 浮点回读逐位一致；同版本重复解码／编码也一致。它只证明该配置的代码对照。六出口与历史运行相比仍有部分回读差异，数值保留在机器记录；历史运行差异的来源未单独确定，不宣称所有输出均逐位不变。

历史 42 格 Sony／Nikon 矩阵和上轮六出口仍见 [前轮记录](../assets/pipeline/pr177-contract-fixes-20261010.json)，未计入本轮新执行结果。私人原片、成品像素、原始绝对路径与明文机身序列号没有提交；保留源 hash、脱敏身份状态、参数和指标。Nikon HE／HE*、其他 Sony codec，以及独立暗场／平场／弱色纹理留出仍缺测。
