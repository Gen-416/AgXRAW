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

传感器先验选择顺序为：适用且启用的用户标定、包内 curated 先验、包内 JPTC、P2P bulk。多个适用用户记录优先选择最近导入的一份；需要固定另一份时，停用其余记录。用户标定可覆盖 fp 的包内 curated 数据，但「保存成功」不绕过适用性检查。

选中的先验能构成有效 shot/read 模型时，用其计算归一化 RAW 噪声方差和 SNR。否则尝试文件 Raw IFD 内合法的 DNG `NoiseProfile`；没有可用来源就明确报告不可用，不再用照片纹理的块内方差冒充物理噪声。文件模型是厂商声明，合法解析不等于已由自己的 PTC 验证。

读噪未分辨不会丢弃整份匹配标定并静默退回包内先验：独立有效的增益继续保留，匹配诊断显示 `gain-only`，分别报告增益与读噪状态。没有独立替代来源时，噪声模型为 `unresolved`，分析状态为 `model-unresolved`，不生成物理 SNR 或读噪底，HDR 尾部 SNR 门控为 0，色度核跳过。普通缺测仍为 `unavailable`；没有独立异常频谱等负面证据时，该 HDR 因子保持中性值 1，这不绕过剪切、色域及解码器限制。合法的独立 `NoiseProfile` 可以提供替代模型，此时来源明确为 DNG，原标定来源和未分辨原因也随报告、界面和缓存保留。

方差系数来源与独立适用性约束分别处理。读噪未分辨或普通缺测时，同一适用标定已经测得的横纵频谱仍然保留；DNG 替代系数不会将其重置为 `unknown`。例如 ISO 200 读噪未分辨、实测 `h = 0.1` 且 DNG 系数合法时，模型可以为 `valid` / `DNG NoiseProfile`，同时保持 `measured-spectral-imbalance`、HDR 噪声因子为 0、色度核跳过。系数来源及原标定来源、原因、频谱一起进入报告和缓存。频谱仍须满足相机、快门、ISO 域及 DN 尺度检查；不能借用其他 ISO 或不兼容读出条件的测量。普通缺测且无可用 DNG 系数时，独立异常频谱也不会被清空。预览与分析缓存版本 23 淘汰旧的回退结果。

当前检查与限制如下：

- 用户标定按规范化的完整制造商、型号匹配；不会用型号包含关系猜测另一代相机。包内 bulk 的旧模糊匹配另有待办，本轮没有替换。
- 快门模式必须匹配或由用户明确声明 `any`。缺少照片快门信息时，不默认为机械或电子快门。
- 用户标定仅在测量 ISO 域内插值，不外推，不跨已声明的增益跳变或读噪失败点插值。单点只适用于该点；增益覆盖不代表读噪也已覆盖。失败点只阻断读噪证据，不自动撤销独立增益。
- 必须有可核对的 DN 范围，照片的编码范围须匹配标定或满足现有存储位移换算检查。拟合残差过高、未收敛或增益估计分歧过大等记录不会作为有效物理先验使用。
- `compression`、`geometry` 会保存为测量声明，Collect 内互相矛盾会拒绝；当前尚未逐文件验证它们是否匹配照片。因此这里只验证相机、快门、ISO 和 DN 尺度，不能称为覆盖全部子读出模式。
- 当前 JPTC 输入使用 G1 或绿色汇总形成 scalar-green 模型。后续把同一增益和读噪用于不同颜色平面是明确的近似，不是实测得到了完整 RGB 协方差。频谱摘要可触发保守限制，尚未自动变成分频降噪、条纹修复或固定图样校正；旧行列方差字段的语义也不升级为已验证条纹比例。

模型状态、来源与单帧相关性线索分开记录。真实纹理导致 G1/G2 残差相关时，不会单凭它撤销匹配标定。HDR 使用模型 SNR；有效、普通缺测、未分辨和明确拒绝分别记录，剪切、色域和解码器约束仍独立存在。证据约束黑端点的读噪底来自独立模型，可靠 RAW 尾部单独约束白端点；缺模型时不把局部变化量改称实测读噪底，RAW 剪切分析也不会因此消失。

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
