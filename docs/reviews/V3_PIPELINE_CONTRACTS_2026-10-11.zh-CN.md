# v3 管线合同、精度边界与全尺寸交付验收

起点：`c4e0c8e5432752bc2cdbcb43fafc0506a0696f7a`；本轮修改在 `codex/v3-pipeline-contracts`。PR #177 已合并，本轮单独提交。本文只记本轮执行，历史验收不加到新计数中。

## 已实现的合同修复

### V3-F01：单点 CSV 不再丢弃显式数值域

`tools/import_jptc.py → calibration.import_csv()` 使用现有 `measurement_domain` 的单位与 decoder 规则，保留原声明和规范化合同。原始 `DNDomain`、`VarianceDomain`、`Decoder`、`LinearisationCurve`、ADC/CFA 声明与 CSV 哈希进入 `single_point_measurement`，安装及运行时重新验证。电子单位、未知 decoder、非恒等 LUT 不再获得有效外部噪声模型资格；合法文件 NoiseProfile 可独立回退，基础显影继续可用。

单帧空间 PTC 的拟合仍是受限证据，不因此获得 paired-dark 或 ADC 校正能力。缺少声明保留明确的历史列合同兼容范围，不代表未知条件已测量。重复声明冲突拒绝导入；显式未知 Format 拒绝外部模型资格，仍可保留拟合供诊断。仅知道机型和快门不足以放行，当前文件还需提供合格的 LibRaw unpack evidence、恒等 LUT 和兼容 DN 编码尺度。

带转换器来源/hash、却没有新合同的旧单点产物保留统计数据，但报告 `single-point-domain-reimport-required`。原 CSV 存在时可重建：

```sh
python tools/import_jptc.py ORIGINAL.csv --rebuild-prior OLD.json --out REBUILT.json
```

只恢复原声明、不重新拟合历史数值时必须核对旧产物中的原 CSV 哈希：

```sh
python tools/import_jptc.py ORIGINAL.csv --rebuild-prior OLD.json --restore-contract-only --out RESTORED.json
```

包内 9 份历史单点资产均找到了固定上游提交、字节 SHA 一致的原 CSV。本轮只恢复可核实的声明，历史拟合和适用范围不变，不补造 RAW 尺寸、ADC、LUT 或读出条件。来源、前后资产 hash 与命令见 [恢复清单](../assets/calibration/single-contract-recovery-20261010.json)。用户自己的旧产物没有原 CSV 时仍需明确重导入，不能由 hash 反推丢失的 header；不删除全部历史测量。

### V3-G01：当前文件的不可变拍摄事实

新增版本化 `CaptureFacts`，与 evidence、场景加载共用同一次源事务。曝光秒数、f-number、焦距保留 TIFF rational 原值、单位、IFD/tag 来源和 verified/missing/invalid/conflict 状态。机身与镜头 serial 分开；标准 EXIF 镜头厂商/名称构成显式 namespace 的 `metadata-name` 身份，不冒充硬件 LensID。读不到的温度、对焦距离、转接环、滤镜及硬件 ID 保持缺测。

事实进入 bundle、readout 匹配、预览/完整 Analysis 缓存与诊断。匹配器可使用曝光/光圈/焦距范围及确切的 namespaced 镜头名称身份；不从所选标定反填事实。公开报告和 CSV 对两种 serial 脱敏，内部匹配仍保留精确值。没有接入新的默认个人像素校正。

本轮首批适配标准 TIFF/DNG 与 RAF 内的 EXIF；其他容器或私有标签仍可能缺测。只读实际元数据核对：Sony ARW 为 1/80 s、f/2.8、70 mm，提供标准 LensModel；Nikon D750 NEF 为 1/30 s、f/1.8、35 mm，未提供标准 EXIF 镜头身份；Fuji RAF 为 10/9000 s、f/5.6、23 mm，标准镜头名称缺测。不以 MakerNote 猜测填充缺失值。

### V3-G02：按真正的统计依赖授权

PTC 增益、相对 ISO 增益、绝对增益、DN 读噪、电子读噪、stored dark variance 和 h/v PSD 分别记录源文件、条件与状态。全包问题仍保留为诊断，但可选 PSD 的条件缺失不再撤销独立完整的 gain/RN/variance。依赖 dark 的产物仍要检查 dark；电子读噪还依赖同域合格增益；PTC 若使用自己的明确 pedestal，不借用坏 dark 的条件。

不把 PSD 不可用解释成噪声白化，不解除现有色度传播门控。适用的独立异常频谱在 DNG 方差回退后仍生效。跨来源组合继续检查实际依赖的单位、读出、ISO 和模式。

GUI/报告分别展示 found、qualified、matched、valid、applicable 和像素处理应用状态；统计分析采用与附加色度处理实际应用单列。`fit_exists` 只表示拟合存在，`valid` 按当前 ISO 的生产访问规则判断；独立 PTC 锚点不因此获得中间 ISO 插值资格。`consumers` 明确表示 profile 层的来源贡献，未解析为具体产物使用。缺标定仍能基本成像。

缓存版本从 35 升至 36；旧预览/分析缓存自动失效重建。新单点合同与公共资产内容改变也进入标定指纹。没有更改 native ABI。

## V3-V01：精度实验的合同

复用 `validate_sensor_precision.py`，增加 `--production-boundaries` 及固定 Apple token 的 `--apple-raw/--apple-full`。不增加产品默认高精度前端，不用换 demosaicer 的差值充当数值误差。

|边界|本轮隔离方法|解释限制|
|---|---|---|
|空间扣黑|同一精确空间场和缩放公式，float64 clipped 参考对生产 uint16|signed→clipped 政策损失另列；不是读噪改善|
|项目后置 warp|同系数、同 bicubic 支撑，float32 对 uint16；另测两次操作累计|不把第一次误差传播后的累计差当单次舍入界|
|CI bitmap|同一个 CIImage、context、extent/origin、Rec.2020；RGBAf→RGBAh→RGBAf 重复读回|格式路径差异不是 Apple 内部精度、传感器真值或画质损失证明|
|后续 half|固定增益与相同 PIL bilinear 源/目标分别对 cast 参考|条件性 legacy gain/resize 不冒充每次默认 full export 实际执行|
|最终编码|生产 exporter 与原质量门限，实际 HEIF primary/HDR 回读|编码差异与成像/修复差异分开|

预先固定预算：未截幅整数最近舍入 0.5 码，truncate 小于等于 1 码；half 最近舍入用 `abs(x)*2^-11+2^-25`（溢出另列）；同 kernel 单次 warp 存储预算 `1+2^-9` 码。完整 bias/RMS/max 使用全部像素对；分位数明确为最多 65536 点的确定性抽样。比较不自由拟合曝光、偏置或增益。

真实 Sony RAW8 / Nikon RAW9 全尺寸实验中，float 重复读回逐位一致，但 RGBAh 对 RGBAf 超出纯最近舍入预算。结果保留为“格式路径并非仅最近 cast”，不放宽预算宣称通过，不据此直接推断可见画质问题，也不修改默认 half。后续同 kernel half cast 实验符合对应舍入预算。详细数值及源码身份见本轮机器记录。

生产边界仍未完全闭环：特定私有 codec 解包的独立 mosaic 参考、所有 stage-2 opcode 组合、LibRaw 内部整数边界、冻结形成计划的完整同算法参考、独立输出噪声标尺与真实留出纹理仍需补测。不能把此研究称为 v3 全面精度验收。

## V3-V02：实文件与交付范围

扩展现有 `validate_coreimage_raw.py`，可用 `--deliver-to` 实际生成生产 SDR/HDR HEIF，保持本 case 的 Analysis、WB、曝光和 tone plan；门限不覆盖。独立检查完整尺寸、方向、P3 ICC、10-bit、ISO gain map、HDR 展开有限值以及原片 hash。不同 decoder case 仍按各自分析/自动曝光构建计划，分布差异不当作同计划数值误差。

`--inspection-crops` 只保存本地 8-bit master/回读裁切作观察，不把它当高精度参考或独立低噪声真值。私人 RAW、成品和裁切不进 Git。

可复用命令（路径由操作者替换）：

```sh
DNGSCAN_FAST=1 python tools/validate_coreimage_raw.py --files SONY.ARW \
  --matrix raw8 libraw --sizes full --scales native \
  --deliver-to PRIVATE_OUTPUT --inspection-crops --out REPORT.json
DNGSCAN_FAST=1 python tools/validate_coreimage_raw.py --files NIKON.NEF \
  --matrix raw9 libraw --sizes full --scales native \
  --deliver-to PRIVATE_OUTPUT --inspection-crops --out REPORT.json
DNGSCAN_FAST=1 python tools/validate_sensor_precision.py --size 128 \
  --include-pipeline --production-boundaries --apple-raw SONY.ARW \
  --apple-version 8 --apple-full --output PRECISION.json
```

现有 Sony 文件只提供 Apple RAW8，RAW9 不可用；不能替换版本后记为 RAW9 通过。Nikon 样本是传统 D750 NEF，不覆盖 HE/HE*。真实暗场/平场独立训练与留出、弱等 Y 色纹理参考、DCG 两侧及更多实际 codec 仍缺素材。普通照片可证明所列配置能够运行和编码，不证明新增随机降噪的保真收益。默认 `chroma_nr=0`、RAW9 opaque transfer 保护保持。

## 本轮最终结果

[机器记录](../assets/pipeline/v3-contracts-validation-20261011.json) 绑定冻结的处理源码、测试、工具与资产 SHA-256；执行时 Git 基线仍为 c4e0c8e，提交后的文件可据 hash 核验。macOS 27.2 / 26B5101f、Apple M4、Python 3.14.4、NumPy 2.5、rawpy 0.27.0+libraw.e419de08；LibRaw commit `e419de08001de28ae6988ecb22df47e52b9c5eaa`。

|完整回归|运行|通过|跳过|失败/错误|
|---|---:|---:|---:|---:|
|native-strict|2175|2172|3|0/0|
|numpy|2175|2140|35|0/0|

两种后端运行同一套用例，不叠加为独立覆盖数量。缺后端/素材跳过不计通过；对应测试源与标定指纹均绑定本次记录。

全尺寸交付使用 share HEIF、quality 99、4:2:0、10-bit；这是验收配置，不修改默认档位，也不保证 20 MB 以下。下面 MB 为十进制；时间在并行回归期间取得，仅作执行记录。

|相机 / 实际前端|完整尺寸 W×H|SDR MB / 秒|HDR MB / 秒|
|---|---:|---:|---:|
|ILCE-7M5 / raw8/full/native|7008×4672|22.28 / 47.58|28.10 / 48.30|
|ILCE-7M5 / libraw/full/libraw|7028×4688|24.54 / 41.21|30.20 / 39.62|
|NIKON D750 / raw9/full/native|6016×4016|1.92 / 17.99|2.72 / 26.26|
|NIKON D750 / libraw/full/libraw|6032×4032|10.75 / 33.04|13.29 / 27.68|

8 个文件均通过原生产回读门限及独立尺寸、方向、10-bit、ICC 检查，HDR 另核对 ISO gain map 与实际展开。原片 SHA 未变。8 个 native-preview / resized-full 配置另均通过；这证明相应尺度路径可用，未声称它们是冻结同计划的像素等价对照。

精度隔离结果（scene-linear Rec.2020 RGB，无自由归一化）：

|同 CIImage 的 RGBAh − RGBAf|bias|RMS|max abs|超最近舍入预算值数|
|---|---:|---:|---:|---:|
|sony / token 8|-2.1921341e-05|3.8251527e-05|0.0019519329|27197456 / 98224128|
|nikon / token 9|-8.6986352e-06|2.964327e-05|0.0019524097|19743644 / 72480768|

两份 float 重复读回最大差均为 0，所有检查值有限；half 格式路径超预算的结果保留，不通过放宽预算消除。后续固定增益及相同 resize 的纯 half cast 检查无超预算。空间扣黑与单次同 kernel warp 的存储舍入符合各自预算；截幅政策及两次 warp 的前级误差传播另列。上述结果不将 float 读回视为传感器真值，也不自动改变默认 dtype。

复用研究运行了 20 个近黑、96 个色度、18 个量化条件及 12 个真实 LibRaw 解码的合成 DNG 配置；它们不是实际机身标定或独立画质样本。

Sony / Nikon 的三组全尺寸修复对照固定本文件 Analysis、WB、EV 与形成计划。无合格个人像素图，system-only 与 off 的 SDR/HDR 最大差为 0。Sony 保守色度处理因独立模型不可用跳过，输出不变；Nikon 的近似粗尺度传播允许该可选处理，产生记录中的局部差异。未凭这些普通照片宣布降噪或纹理保真收益。已观察本地 master/回读的边缘及暗部裁切，但没有独立真值，裁图不代替门限或留出验收。

|任务|本轮完成定义|剩余限制|
|---|---|---|
|V3-F01|单点入口合同、工具→安装→真实合成 DNG 回归、九份公共资产可重现恢复|用户旧产物需 hash 匹配的原 CSV；缺失时保留重新导入边界|
|V3-G01|首批不可变拍摄事实、状态/来源、匹配、缓存和 GUI/报告|MakerNote 镜头硬件 ID、温度/光路等未知字段未补造；个人像素校正未启用|
|V3-G02|按依赖授权、当前 ISO 状态与适用频谱约束|不同来源组合仍受单位、模式、ISO、相关性限制|
|V3-V01|生产边界隔离工具和 Apple 同图像格式实验|完整形成/预览同计划参考、所有 opcode/codec 边界和误差的画质代价尚未全面闭环|
|V3-V02|现有 Sony / 传统 Nikon 全尺寸八出口、尺度路径与修复对照|HE/HE*、其他 Sony codec、独立暗场/平场/弱等 Y 纹理及 DCG 留出仍缺测|

本轮没有修改默认附加降噪、默认 half 策略或用户 RAW。私人原片、HEIF 与观察裁图不提交 Git；记录保留源/成品 hash、ROI、实际 recipe、消费者状态及缺测。
