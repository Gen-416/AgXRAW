# 实测噪声标定的导入与应用

[返回文档索引](README.md) · [色度降噪与纹理保护](CHROMA_NR.zh-CN.md)

AgXRAW 可以把 Jiangtherapee / JPTC 的实测结果保存为本机标定，匹配照片后用于噪声尺度、SNR、暗部及 HDR 分析，并为可选色度降噪提供输入。测量在前期完成，普通照片仍按单张 RAW 处理。默认 `--chroma-nr 0` 保持关闭；导入标定不会自动开启降噪。

当前接口接收测量统计及派生 JSON，不直接接收黑场 DNG，也不据此生成坏点图或减去固定图样。尚无自己的 Sigma fp 测量时可以先使用接口；其他相机的标定只能验证导入操作，不会因此套用到 fp。fp 照片仍使用适用的包内先验或文件内模型，实际来源见界面和分析报告。

## GUI 导入

在「基础」页的「RAW 解码」卡片右上角打开「实测噪声标定」。选择一个标定 JSON，或选择包含 Collect CSV 的测量目录。导入后可以查看相机、快门模式、ISO 范围、数据警示，并执行启用、停用或删除。

「本次导入的读出模式声明」默认遵循文件。电子快门、机械快门、电子前帘和适用全部模式均为显式覆盖；只在确认测量适用范围时使用。特定快门模式的标定不会自动应用到未记录快门模式的 RAW。「适用全部模式」表示用户明确作出这个声明，不是程序推断相机各模式等价。

导入成功默认启用。当前照片会重新分析，旧预览和分析缓存失效。管理页的「已启用」只表示允许参与选择；是否实际应用，以 RAW 解码卡下的「噪声模型」来源与当前照片的匹配结果为准。导入失败不会安装该记录。

浏览器只提交文件选择器选中的 JSON/CSV/TXT 文本，最多 128 个文件、合计 8 MiB；请选择单套测量目录。RAW 本体和其他二进制文件不进入标定包。

## CLI 管理

以下命令在项目根目录、已安装项目依赖的 Python 环境中执行。把 `/path/to/...` 换成自己的测量路径，把 `CALIBRATION_ID` 换成 `list` 输出中的 `id`。这里的 ID 是安装记录的内容 ID，不是相机名称。

```bash
python -m dngscan calibration --help
python -m dngscan calibration import /path/to/fp-measurements
python -m dngscan calibration import /path/to/fp-profile.json
python -m dngscan calibration list
python -m dngscan calibration disable CALIBRATION_ID
python -m dngscan calibration enable CALIBRATION_ID
python -m dngscan calibration remove CALIBRATION_ID
```

先安装但不启用时加 `--inactive`。需要明确覆盖适用快门模式时加 `--shutter-mode electronic`、`mechanical`、`efcs` 或 `any`，例如：

```bash
python -m dngscan calibration import /path/to/fp-profile.json --inactive
python -m dngscan calibration import /path/to/fp-profile.json --shutter-mode electronic
```

`--shutter-mode any` 明确声明标定适用于全部快门模式；它不会验证这种声明是否符合相机实际行为。覆盖声明和原始文件的快门信息都会保留。不同声明产生不同安装 ID，旧记录可单独停用或删除。

管理命令返回 JSON，不需要提供待显影的 RAW。标定默认保存在 `~/.config/dngscan/calibrations`；设置 `XDG_CONFIG_HOME` 时使用其下的 `dngscan/calibrations`，`DNGSCAN_CALIBRATION_DIR` 可覆盖目录。它们是本机用户数据，无需修改包内 `dngscan/data/priors`。

导入后照常转换照片；以下为手动开启色度降噪的例子，不代表建议所有照片都使用这个强度：

```bash
python -m dngscan /path/to/photo.DNG --jpeg /path/to/photo.jpg --chroma-nr 0.25 --report
```

## 接受的测量格式

| 输入 | 必要内容与适用范围 |
| --- | --- |
| Collect CSV 目录 | `dark-scalars.csv`（`JPTC-DARK/1`）；绝对增益需要 `ptc-iso*.csv`（`JPTC/2`）锚点。可附 `gain-levels.csv`（`JPTC-ISOGAIN/1`）及 `spectrum-h.csv` / `spectrum-v.csv`（`JPTC-SPECTRUM/1`）。相机、快门、压缩及尺寸声明冲突会拒绝导入。 |
| `dngscan-jptc-collect-1` JSON | Collect 的派生结果，包含明确机型、PTC 锚点或增益曲线、读噪曲线及测量来源。可记录多个 ISO 和增益跳变。 |
| `dngscan-jptc-prior-1` JSON | 单个 ISO 的 PTC 派生结果，包含增益、读噪、满阱及 DN 范围依据。用户导入的单点标定只适用于该 ISO，不向其他 ISO 外推。 |

一个完整目录通常如下：

```text
fp-measurements/
  dark-scalars.csv
  ptc-iso100.csv
  gain-levels.csv
  spectrum-h.csv
  spectrum-v.csv
```

只有黑场统计、没有绝对增益锚点时仍可保存供检查，但不能构成完整的 shot/read 成像噪声模型。增益和读噪的有效 ISO 范围都要覆盖当前照片；未解析的读噪不是零读噪。

Collect 的 `read_noise_unresolved_isos` 保存「测过但未能分辨」的 ISO。该列表随导入记录和包内 Collect 数据进入运行时；不能把它当成普通未采样间隔。DN 与电子读噪曲线都不在失败点求值，也不跨失败点插值或外推。失败列表包含非法 ISO，或与同一 ISO 的有效读噪点冲突时，导入会拒绝。单点 PTC 的零读噪同样表示未分辨，不是可用的零噪声模型。

现有离线工具也可生成上述两类 JSON，再由 GUI 或 CLI 安装。单个 `JPTC/2` CSV 不直接作为 GUI 标定文件输入：

```bash
python tools/import_jptc_collect.py /path/to/fp-measurements --out /path/to/fp-collect.json
python tools/import_jptc.py /path/to/ptc-iso100.csv --brand SIGMA --model fp --iso 100 --shutter electronic --out /path/to/fp-iso100.json
```

第二条命令使用 CSV 的 G1 空间统计进行带 PRNU 处理的 PTC 拟合，不能因此把它称为完整的跨帧随机噪声测量。Collect 黑场对可提供独立的时域读噪。来源、拟合状态和修正假设应随测量保留，尤其不能把增益估计方法间的分歧当成完整的统计置信区间。

## 何时参与成像

传感器先验选择顺序为：适用且启用的用户标定、包内 curated 先验、包内 JPTC、P2P bulk。同等读出适用范围内，多个用户记录优先选择最近导入的一份；需要固定另一份时，停用其余记录。用户标定可覆盖 fp 的包内 curated 数据，但「保存成功」不绕过适用性检查。

选中的先验能构成有效 shot/read 模型时，用其计算归一化 RAW 噪声方差和 SNR。否则尝试文件 Raw IFD 内合法的 DNG `NoiseProfile`；没有可用来源就明确报告不可用，不再用照片纹理的块内方差冒充物理噪声。文件模型是厂商声明，合法解析不等于已由自己的 PTC 验证。

标定尺度使用独立保存的 DN 编码端点：DNG 为线性化后的 `WhiteLevel` 减该分量的最大黑位，
空间黑图案和行列变化的最大值计入其中。该范围同时用于增益适用性、归一化 a/b 和解码方差传递；
`LinearResponseLimit` 只是线性有效区间，改变它不能改变编码尺度或绕过独立频谱限制。
真实编码范围不匹配仍按现有容差拒绝。LibRaw 解码器自身的缩放使用编码最大值减共同最小黑位，
此解码斜率与噪声模型的归一化分母不是同一个量。

读噪未分辨不会丢弃整份匹配标定并静默退回包内先验：独立有效的增益继续保留，匹配诊断显示 `gain-only`，分别报告增益与读噪状态。没有独立替代来源时，噪声模型为 `unresolved`，分析状态为 `model-unresolved`，不生成物理 SNR 或读噪底，HDR 尾部 SNR 门控为 0，色度核跳过。普通缺测仍为 `unavailable`；没有独立异常频谱等负面证据时，该 HDR 因子保持中性值 1，这不绕过剪切、色域及解码器限制。合法的独立 `NoiseProfile` 可以提供替代模型，此时来源明确为 DNG，原标定来源和未分辨原因也随报告、界面和缓存保留。

方差系数来源与独立适用性约束分别处理。读噪未分辨或普通缺测时，同一适用标定已经测得的横纵频谱仍然保留；DNG 替代系数不会将其重置为 `unknown`。例如 ISO 200 读噪未分辨、实测 `h = 0.1` 且 DNG 系数合法时，模型可以为 `valid` / `DNG NoiseProfile`，同时保持 `measured-spectral-imbalance`、HDR 噪声因子为 0、色度核跳过。系数来源及原标定来源、原因、频谱一起进入报告和缓存。频谱仍须满足相机、快门、ISO 域及 DN 尺度检查；不能借用其他 ISO 或不兼容读出条件的测量。普通缺测且无可用 DNG 系数时，独立异常频谱也不会被清空。这些修复已进入缓存版本 25；旧回退、编码范围、损失支撑和噪声坐标结果不会复用。

Collect 成对暗场现在分别保留物理读噪（e⁻）和扣除 Sheppard 量化项之前的时域总方差
`stored_dark_variance_dn2_log2iso`。相机、快门、ISO、DN 尺度及物理读噪 resolved 检查通过后，
模型优先采用声明为线性化 DN 域、且已完成 sigma-clip 修正的总方差，换算为归一化 RAW
常数项；物理读噪报告不变。旧 Collect 仅在旧转换器明确记录 identity 线性化、有效 ADC
步长及 Sheppard 扣除合同，且 sigma-clip 修正完成后，恢复对应量化项，仍属于绿色汇总
近似。缺少这些声明或有效步长时保留原物理读噪近似并报告原因。PTC、P2P 和 DNG
NoiseProfile 不统一加 `1/12`。读噪明确未分辨仍按原状态处理，不因保留总方差而自动放行。

导入时在相同实测 ISO 上核对总方差不低于物理读噪方差；可选 DN 读噪曲线缺失时，也用
电子读噪与增益核对。运行时在当前 ISO 再检查，防止稀疏总方差插值低于该 ISO 的物理
读噪；下界校验保留 5% 容差。明显矛盾的总方差或非有限模型系数明确记为 `unresolved`
并保留原因，不会静默生成零噪声模型或退回物理读噪常数项。合法独立 DNG `NoiseProfile`
仍可提供替代系数，原标定的适用频谱约束继续保留；已测异常频谱仍令 HDR 尾部噪声因子
为 0、色度核跳过。
缓存版本 26 同时淘汰旧模型、模糊机型匹配及缺少逐文件读出描述的结果。

## 逐文件读出约束

解码时独立读取采集描述，不再通过可选镜头配置取得快门信息。DNG 使用主 RAW IFD，
分别保存 RAW 栅格、ActiveArea 和 DefaultCrop；缩略图的位深与尺寸不参与匹配。
主 RAW 按固定 LibRaw 默认选帧规则选择第一个 `NewSubfileType=0` 的 CFA / LinearRAW
帧，不把更大的增强 IFD、预览或掩膜当作实际解码帧。噪声标签、校正配方及读出约束共用
这一来源；BaselineExposure 优先取实际 RAW 帧，缺少时才取 IFD0 全局值。
LibRaw 的完整马赛克尺寸另行保存，不用 `raw_image_visible` 或显影尺寸代替。
这些小型描述随分析、预览及导出缓存保存，参与完整来源比对。

JSON 可在现有测量内容之外添加明确的约束，例如与本地 fp 样张相同的存储配置：

```json
"readout_contract": {
  "version": 1,
  "sample_bits": 14,
  "raw_geometry": [6064, 4042],
  "storage_lossless": true
}
```

这是添加到完整标定对象的片段，不是独立可导入文件。数值须来自对应测量文件，不能
照抄示例证明自己的测量适用。`sample_bits` 是 TIFF 存储样本位数，不能据此声称已测得
ADC 有效位数。`raw_geometry` 也只限定文件 RAW 栅格，不独自证明没有像素合并或其他处理。

| 约束字段 | 当前逐文件依据 |
| --- | --- |
| `raw_geometry` | DNG 主 RAW IFD 的 ImageWidth / ImageHeight。 |
| `libraw_raw_geometry` | 当前文件的 LibRaw `raw_width` / `raw_height`；包含完整马赛克，不借用可见窗口。 |
| `active_geometry` | DNG 明确存在的 ActiveArea 宽高；缺 tag 时保持未知。 |
| `default_crop` | DNG 明确声明的 DefaultCropSize，可含合法分数；与 RAW 栅格分开。 |
| `sample_bits` | 主 RAW IFD 的 BitsPerSample。 |
| `storage_lossless` | 当前编码过程是否保留存储码值；无压缩和 Deflate 可确认，JPEG 7 在各 tile/strip 的声明 bytecount 范围内检查 SOF3、完整分量覆盖及 SOS point transform 为 0。缺少块长度、不完整或未支持过程保持未知。 |
| `sensor_bits`、`sensor_binning`、`capture_kind`、`readout_id` | 可以显式声明，但当前没有足够的通用文件字段映射；未知不会当作匹配。 |

每个明确声明的约束都必须有对应证据并相符。声明为 lossless、无损压缩、uncompressed
或无压缩的旧 `compression` 文本接受同一保码值约束；无损 JPEG 与无压缩容器不同，
本身不证明传感器读出不同。`12bit`、`14bit`、`RAW HQ` 等重载文本不会被猜成 ADC 位深
或特定读出模式。已确认有损、或 DNG 压缩过程尚不能确认保码值时，不应用外部 shot/read
先验；合法文件内 `NoiseProfile` 仍可独立提供系数。这里核对的是当前存储过程，不能证明
图像在更早阶段从未经历有损处理。

Collect 的 ImageWidth / ImageHeight 是操作者声明的相机 JPEG 输出尺寸，`RawSize`
才是 LibRaw 完整马赛克尺寸，二者的层位由[固定版本输出合同](https://github.com/y-g-jiang/JiangtherapeeTesterView/blob/57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c/src/output/darkCsv.mjs#L43)确认。
新 CSV 导入用 `RawSize` 形成 `libraw_raw_geometry` 约束，JPEG 尺寸作为信息保留；
Dark/PTC 等文件已声明的 RawSize 相互矛盾时拒绝导入。任何一份参与测量的 CSV 声明了
RawSize、另一份却缺少时，保留逐文件声明覆盖并报告不可核对；不拿暗场尺寸补齐 PTC，
也不反向补齐暗场，更不从 JPEG 尺寸推断。此规则同时用于离线 Collect 构建与用户导入。
旧 JSON 若只剩无数据域声明的 `geometry`，报告不可核对；
可重新导入保留完整头部的 CSV，或在有原始证据时明确 `acquisition_contract.geometry_domain`
为 `raw-ifd`、`active-area`、`default-crop` 或 `libraw-raw-mosaic`，不能通过猜尺寸解除限制。
仅 JPEG 尺寸、缺少 RAW 尺寸依据的 Collect 也不宣称已经验证传感器栅格。

单点 `tools/import_jptc.py` 也保留 JPTC/2 的 `RawSize` 与 `Compression`：前者约束
LibRaw 完整 mosaic，后者由运行时核对，不能解释的文本保持不可核对。单点 PTC 的
ImageWidth / ImageHeight 仍仅为 JPEG 输出信息；它不会因此取得 Collect 专属的
成对暗场总方差语义。用户安装与包内单点记录使用同一读出约束。旧转换器已经删除的
声明无法从 JPEG 尺寸或文件哈希恢复；已有单点 JSON 应从原 CSV 重新转换并导入。

同机型、快门及 ISO 有多份记录时，优先选择实际读出约束匹配的记录，再按原策略处理
未声明子模式的记录；同等适用范围内仍按导入时间选择。没有可用候选时，保留失败记录
和明确原因，不静默回退到同样未核对的 curated/bulk 标定。增益、读噪、PDR 与噪声模型
使用同一上下文，不能一项拒用而另一项继续借用。独立 DNG 模型可替代方差系数，但不能
因此借用不适用标定的增益或频谱。

fp 的电子快门来自[精确型号的厂商规格](https://www.sigma-global.com/en/cameras/fp/?local=table&tab=support&table_id=11934)，
来源标为 `manufacturer-capability:SIGMA-fp`，不冒充文件内快门 tag。这项能力只证明快门
类型，不把所有静态、视频、位深、裁切或像素合并模式视为相同。当前也不向 fp L 或其他
型号推广。未声明完整子模式的 curated/bulk 先验继续属于明确标记的近似；匹配已声明字段
不等于验证了全部传感器读出状态。

当前检查与限制如下：

- 用户标定与包内 curated / JPTC / bulk 均按规范化的完整制造商、型号匹配；保留代际和后缀，不用包含关系猜测另一代相机。已知不同写法通过显式别名登记，未知型号返回无先验。例如 `Canon EOS 5D` 不会借用 `EOS 5D Mark II` 的标定。
- 快门模式必须匹配或由用户明确声明 `any`。缺少照片快门信息时，不默认为机械或电子快门。
- 用户标定仅在测量 ISO 域内插值，不外推，不跨已声明的增益跳变或读噪失败点插值。单点只适用于该点；增益覆盖不代表读噪也已覆盖。失败点只阻断读噪证据，不自动撤销独立增益。
- 必须有可核对的 DN 范围，照片的编码范围须匹配标定或满足现有存储位移换算检查。拟合残差过高、未收敛或增益估计分歧过大等记录不会作为有效物理先验使用。
- `compression`、`geometry` 保留原测量声明；可明确解释的数据域和 typed `readout_contract` 逐文件核对。未知域、未知过程和不能映射的子模式明确报告，不当作匹配。单纯改变 DefaultCrop 不自动视为传感器读出变化；全子读出模式仍未覆盖，具体范围见上节。
- 新 Collect 输入保留各 CFA 位置的黑位、时域方差、读噪及未分辨点。只有独立 PTC 列与实际位置映射均可核对时，才保存该位置的物理增益；其他位置继续明确使用 scalar-green 增益近似，不冒充四通道实测。频谱摘要可触发保守限制，尚未自动变成分频降噪、条纹修复或固定图样校正；旧行列方差字段的语义也不升级为已验证条纹比例。

## 同快门阶梯、差分 PTC 与多个锚点

ISO 阶梯按实际 `ShutterGroup` 或容差内的同一快门设置形成配对边。每个 CFA 位置先扣除
自己的黑位，再由同快门信号比得到增益比；标称快门时间不进入这些比值。重复测量先在同组
平均，图上的闭环残差、断开的 ISO 组及被拒绝行分别保留。边的权重只表示相对重复次数，
不宣称为统计置信区间。`mixed` 输入只用配对边，auto-shutter 的结果另作诊断；纯
auto-shutter 保留依赖标称曝光时间的原方法并明确说明限制。

JPTC/2 的成对字段按[固定版本输出合同](https://github.com/y-g-jiang/JiangtherapeeTesterView/blob/57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c/src/output/entryCsv.mjs)
解释：`StdDiff` 是 A−B 的标准差，不是单帧标准差；`StdDiffClipped` 还需除以声明的
`ClipVarianceFactor`。单帧存储 DN 方差因此是 `StdDiff² / 2` 或
`StdDiffClipped² / (2 * ClipVarianceFactor)`，这里不统一追加或扣掉 `1/12`。
自变量使用两帧均值的平均值再扣黑。均值差超过信号的 1% 的对被剔除；这是防止明显漂移
污染拟合的工程限制，不能证明照明、行带或内部处理完全一致。

有效差分拟合优先用于 gain，单帧空间 PRNU 拟合作为交叉检查，保留二者分歧。缺少差分
字段、clip correction 无法确认且没有未裁切差分、或稳定样本不足时，显式记录回退原因。
差分拟合记录 HC3 条件标准误、由斜率的正态近似区间换算的条件 95% gain 区间、拟合窗口
和残差；斜率区间越过零时记录 gain 上界不可分辨，不伪造有限区间。这些区间不包含照明、
量化模型、裁切和采集系统误差，不能作为相机物理增益的完整置信区间。成对读噪截距仍是已存 RAW 的方差，
不是扣除 ADC 贡献后的纯模拟读噪。

Collect 构建拟合全部 PTC 文件，保存逐文件 hash、质量、拒绝原因与各列结果。无效的首份
文件不遮蔽后面的有效锚点；每个连通阶梯组须有自己的独立锚点。组内锚点以等权对数尺度
合并，并检查现有 5% 尺度容差；冲突保留在报告中，整份物理先验不可作为正常有效模型。
没有连接的锚点只适用于测量 ISO，不外推、不跨组补成一条连续曲线；交错的断开 ISO 组
只保留离散点，以免展平曲线后错误混用两个组。现有 gain-jump 候选把插值域分段，不跨
跳变间隔插值，也不把仅由阶梯发现的跳变自动命名为 DCG。

`CfaPattern` 是 LibRaw 的颜色索引描述，不是 Bayer 空间排列。各位置的颜色由暗场 CSV
的 `Channel`、`ColorIndex` 和原始描述共同确定；PTC 的 G1 是与红色位于同一行的绿色，
不能固定假定为 ColorIndex 1。没有位置证据时保留旧 scalar-green 兼容读法，不能据此
生成独立四相位增益。重复读噪在方差域合并后开方，未分辨点继续阻止跨点插值。

回归入口为 [test_calibration_gain_graph.py](../tests/test_calibration_gain_graph.py) 和
[test_calibration_temporal_anchors.py](../tests/test_calibration_temporal_anchors.py)。这些
合成信号验证单位、图连接与证据传递；实际平场的照明漂移和各 ISO 相位残差仍需实测数据验收。

模型状态、来源与单帧相关性线索分开记录。真实纹理导致 G1/G2 残差相关时，不会单凭它撤销匹配标定。HDR 使用模型 SNR；有效、普通缺测、未分辨和明确拒绝分别记录，剪切、色域和解码器约束仍独立存在。证据约束黑端点的读噪底来自独立模型，可靠 RAW 尾部单独约束白端点；缺模型时不把局部变化量改称实测读噪底，RAW 剪切分析也不会因此消失。

噪声底和 SNR 坐标转成 scene EV 时，计入有效文件 `BaselineExposure`；BE +1 EV 使这些坐标及
受噪声限制的 SDR/HDR 黑端点同步增加 1 EV，RAW a/b、RAW SNR 和传感器 DR 不变。
用户 EV 不重新塑造固定计划。BE 已由 Apple 烘焙在像素中或保存在 `scene_scale` 分母中，
都只转换一次证据坐标，不再次乘图像；这一转换不升级 Apple 的绝对辐射标定可信度。

频谱的横纵高频/中频功率比另作适用性检查：任一比值小于 0.5 或大于 2，标为 `measured-spectral-imbalance`；有效 gain/read 模型仍保留，但 HDR 尾部 SNR 门控为 0、当前粗网格色度核跳过。此阈值是保守启发式限制，不是相关噪声的完整统计检验。其他已测比值仅记为 `measured-spectrum-summary`，接近 1 不能证明白噪声，也不能排除窄带峰或整行/列偏置；没有摘要时为 `unknown`。界面显示该状态与横纵比值，原始测量仍需保留以便后续复核。

文件明确声明 `NoiseReductionApplied > 0`，或该声明非法时，独立白噪声假设不适用：即使外部标定匹配也拒绝该成像噪声模型，HDR 尾部 SNR 门控为 0、色度核跳过。`NoiseProfile` 的数值仍按前述来源优先级选择；处理声明属于对全部来源的兼容性限制。tag 缺失不证明 RAW 完全未经处理。

有理数声明保留原始状态：`0/1` 是明确未降噪（`none`），`0/0` 是未知（`unknown`），非零分子除以零是非法（`invalid`）。未知不等于未降噪，也不会单凭它拒绝匹配噪声模型；非法声明则拒绝。公共 TIFF 解析不再把零分母统一写成 0，模型与报告分别保留缺失、未知、明确无降噪、已降噪、非法和不可读状态。

## 色度降噪的适用边界

色度核需要把 RAW 模型传播到自己的处理层位，不能直接使用 e− 或 RAW DN 阈值。目前通过 LibRaw 记录的线性变换和 CFA 粗网格采样作低频近似，计入曝光、白平衡及支持的镜头修正；不宣称精确重现解拜耳和几何插值的全部空间协方差。

Apple RAW 的降噪、解拜耳和其他内部变换没有可用的协方差标定，因此项目自己的可选色度核会跳过并报告原因。这不关闭 Apple 引擎内部处理，也不妨碍可用的独立传感器证据参与其他分析。未知或不支持的变换、缺失模型以及无效区域同样不会退回旧的全图 MAD 阈值。

具体滤波层位、频带、BayesShrink 思路和细节损失边界见[色度降噪与纹理保护](CHROMA_NR.zh-CN.md)。实测提供独立的噪声尺度，不能逐像素证明哪些变化是真实纹理，开启降噪仍需检查颜色细节。

## 代码与验证入口

导入、校验及用户存储在 [calibration.py](../dngscan/calibration.py)，模型选择在 [noise_model.py](../dngscan/noise_model.py)，处理域传播在 [noise_propagation.py](../dngscan/noise_propagation.py)。GUI 及 CLI 使用同一公共导入接口，标定变动会改变预览、磁盘分析和导出配方指纹；导出分析期间变动则在写图前要求重试。

接口与缓存回归见 [test_user_calibration.py](../tests/test_user_calibration.py)、[test_calibration_gui_cli.py](../tests/test_calibration_gui_cli.py)；模型和滤波见 [test_noise_model.py](../tests/test_noise_model.py)、[test_calibrated_chroma.py](../tests/test_calibrated_chroma.py)。[test_rational_metadata_state.py](../tests/test_rational_metadata_state.py) 通过大小端真实 TIFF 标签检查未知、非法及有效声明；Collect 合成 CSV 和导入后的分析测试覆盖失败点不被重新插值、增益保留及 HDR/色度核行为。[test_spectral_fallback_pipeline.py](../tests/test_spectral_fallback_pipeline.py) 用带真实 ISO / NoiseProfile 标签的合成 DNG，实际执行导入、LibRaw 解码、分析和色度核入口，验证方差回退保留独立异常频谱，并用频谱域外输入验证正常启用的对照路径。合成测试和其他相机数据不能替代自己的 fp PTC、黑场及实拍纹理验收。

编码范围与频谱约束组合见 [test_noise_coding_range.py](../tests/test_noise_coding_range.py)，
黑位／线性化端点见 [test_noise_coding_endpoints.py](../tests/test_noise_coding_endpoints.py)，
实际 BE 解码至 SDR/HDR 计划的坐标一致性见 [test_noise_scene_ev.py](../tests/test_noise_scene_ev.py)。

逐文件读出与真实 DNG/CSV 导入回归见 [test_readout_contract.py](../tests/test_readout_contract.py)：
主 RAW 与缩略图区分、快门来源、无损过程、位深和几何约束、同 ISO 子模式选择、失败后
独立 DNG 替代，以及采集描述的缓存往返。
