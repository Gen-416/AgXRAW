# 传感器支持与开放接口策略

> 2026-07-30 建立。回应"很多相机导出显示接口不支持"：两条解码接口全部开放，
> 缺数据的机型**降级并警示**，不再拒绝。本文记录策略、数据出处与逐机型状态。

AgXRAW 的目标是尽可能覆盖不同品牌、机型和 RAW 编码，不以 Sigma fp 或用户个人标定
作为使用前提。必须分别判断：文件能否解码、是否有可执行的镜头参数、是否有适用的
物理噪声模型。缺少第三项不等于前两项或最终成像不可用；相同扩展名也不保证其中的
压缩编码已被解码器支持。

## 无个人标定时怎样处理

| 可用证据 | 成像行为 |
|---|---|
| 匹配的包内或用户噪声模型 | 使用适用的噪声约束；可选色度降噪另核对解码传播 |
| 合法 CFA DNG NoiseProfile | 无需自己的黑场或 ISO 增益曲线，采用文件声明模型 |
| RAW 可解码，但没有适用噪声模型 | 通用成像：自动曝光、AgX、色彩和导出继续；不声明物理 SNR，不启用模型色度降噪 |
| Apple 可解码，但没有独立传感器证据 | 使用图像统计回退；HDR 余量限 1 EV，关闭 HDR 通道分离 |
| 已确认模型不适用或实测频谱异常 | 普通成像继续，保留相应噪声／HDR 限制及原因 |

无噪声模型的 LibRaw 路径仍可使用可靠 RAW 高光证据，不会被一律压成图像域 1 EV
回退。HDR 是否有额外亮度范围取决于这张照片，而不是品牌名称；没有可信余量时
应导出 SDR，不能靠放松噪声匹配凭空制造 HDR。X-Trans 当前可正常显影，但项目的
色度噪声传播尚未认证，因此即使有相机噪声模型，也不会自动启用该色度核。

GUI、文本报告和 CSV 共享同一份只读处理依据摘要。预览为节省内存释放 RAW 缓冲后，
已取得的传感器分析不会因此变成“无证据”。缺少噪声标定属于正常兼容路径，拒绝或
未分辨的测量仍保留原状态；不会用真实纹理的幅度填补噪声模型。

2026-10-09 新增五张 iPhone 16 Pro **Bayer DNG** 实片验收，ISO 80–800，
不是 Linear RGB ProRAW。文件均有合法 NoiseProfile，`NoiseReductionApplied=0/0`
保留为未知；无需个人黑场即可使用文件方差模型，电子域增益仍不声明。LibRaw 与
本机 Apple RAW `9.dng` 均完成原尺寸 JPEG 97/420 编码回读，另有三次 10-bit/444 SDR
HEIF 和两次 10-bit/444 HDR HEIF 通过浮点回读。LibRaw 执行文件内暗角校正；其校正
路径与 Apple 解码器的噪声传递均未建立，显式开启项目色度降噪仍跳过。Apple 路径
保留独立 LibRaw 传感器参考，未落入无证据的 1 EV 回退。这些结果仅覆盖本批文件与
本机版本，未覆盖其他 iPhone、Linear RGB ProRAW、Apple 单独可读或旧解码器回退。
尺寸、哈希与验收范围见[紧凑记录](assets/delivery-quality/iphone-ideal-20261009.json)
及[研究记录第 18 节](reviews/NOISE_TEXTURE_RESEARCH_2026-10-09.zh-CN.md#18-iphone-bayer-dng-与-ideal-image-验收基准-38f5bc7)。

## 策略：三级降级，永不静默

1. **颜色标定阶梯**（固定 Kelvin 白平衡求解，`raw_io.solve_wb_for_mode`）：
   文件自带 DNG 双光源标定 → 安装版 LibRaw 的机型矩阵 → 本项目回退矩阵表
   （`camera_matrices.py`，取自 LibRaw master 的 Adobe 系数，GPL 同族许可）→
   全部缺失时**退化为相机 AsShot 并在报告显式警示**（"结果可用，但白平衡声明
   与色彩精度可能有偏差"）。渲染照常出片——声明的降级可用，静默的降级等于
   隐藏白平衡。
2. **传感器先验与噪声模型**：适用且启用的用户 JPTC 标定优先于包内 curated、
   JPTC 和 P2P bulk 先验。选中的先验不能构成有效 shot/read 模型时，尝试合法的
   Raw IFD DNG `NoiseProfile`；没有模型就明确不可用，不把单帧局部纹理变化当作物理噪声。
   RAW 电平和剪切仍可独立分析，渲染也可继续。GUI/CLI 接口及机型、快门、ISO、DN
   尺度限制见[实测噪声标定](NOISE_CALIBRATION.zh-CN.md)。
3. **Apple RAW**：逐文件探测支持版本；自动模式按文件提供的 RAW 9/8/7/6
   尝试实际渲染，全部失败后才尝试 LibRaw，并报告实际版本与回退原因。
   显式指定版本保持严格，失败直接报错。

ARW/RAF/NEF 的扩展名不能证明具体编码可解包。新机型可能同时缺解码器、逐机型
颜色矩阵或统计标定，必须逐文件核对。回退矩阵只补颜色标定这一层。注意边界：
回退矩阵服务于 Kelvin 求解与报告，无法注入 LibRaw 内部的色彩转换；对 LibRaw
完全不认识的机型，Rec.2020 转换精度取决于 LibRaw 的内部回退，报告如实说明。

## 新机型适配状态（2026-07-30）

| 机型 | 先验（P2P） | 颜色矩阵 | 备注 |
|---|---|---|---|
| Sony A7 V (ILCE-7M5) | 机械快门 unityEv 8.7983 / FWC 71k；电子快门另选 JPTC | ✓ LibRaw pin | gain、读噪、PDR 的快门来源分别记录；快门未知不能命中机械条目 |
| Sony A7S III (ILCE-7SM3) | ✓ unityEv 10.15 / FWC 228k | ✓ LibRaw master | 大像素签名明显；unityEv 为 DxO 派生表锚定 |
| Sony A7R VI (ILCE-7RM6) | ✓ unityEv 7.80 / FWC 36k | **无**（刻意缺席） | 未找到已发布系数；矩阵宁缺毋猜，降级路径覆盖；unityEv 为 JPTC 一手实测锚定 |
| Ricoh GR IV | ✓ unityEv 7.37 / FWC 27k | 无需（DNG 自带 ColorMatrix） | |
| Nikon Zf | ✓ unityEv 9.06 / FWC 82k | ✓ LibRaw master | 同传感器 Z6II DxO unity 508.3 交叉印证 |
| Fujifilm X100VI | ✓ unityEv 7.61 / FWC 24k | ✓ LibRaw master | |
| Fujifilm X-E5 | ✓ unityEv 7.54 / FWC 23k | ✓ 借自 X100VI | 同款 40MP X-Trans CMOS 5 HR，声明的借用 |

数据出处：PhotonsToPhotos PDR.htm / RN_e.htm（2026-07-30 提取，含 P2P 直接
发布的 `fwc` 与 `unityEv` 字段；PDR 曲线只取实心点，三角标记起点记入
`suspect_iso_min`）；矩阵出处逐条记录在 `camera_matrices.py`。

> **2026-08-24 轴解码审计**：P2P 图表 x 轴的真实标签公式是 ISO = 3.125·2^x
> （对渲染出的坐标轴刻度逐一验证，并与八台机型的原生 ISO 区间精确对齐），
> 2026-07-30 的提取把 2^x 当成了 ISO，导致全部 curated 曲线 x、
> `suspect_iso_min` 与图表锚定的 `unity_gain_ev` 系统性低了
> log2(3.125)=1.6439 EV（unity 差 3.125 倍）。上表 unityEv 为修正后的值；
> 每台的证据链写在 `sensor_priors.json` / `priors.py` 的 `source` 字段，
> 回归钉在 `tests/test_priors_importers.py::TestCuratedAxisAudit`。
> bulk 层（`p2p_bulk.json`，y-g-jiang 转换）经查解码正确，不受影响。

## LibRaw 项目依赖（2026-08-01，已执行）

轮子版 rawpy 0.27.0 捆绑 LibRaw **0.22.1 发布版**。对本清单实测：0.22.1 已知
A7S III、X100VI、**Zf**（初判"缺 Zf"是 `strings` 默认 4 字符下限吃掉了
"Z f" 3 字符串的工具假象，已用 `-n 3` 复核更正——教训：用工具探测前先想清
工具自己的截断规则）；master（快照 2026-07-18，commit e419de08）对本清单的
增益是 **A7 V** 一台，外加约两年其他新机型表项。X-E5/GR IV/A7R VI 连 master
都没有——回退矩阵表对它们仍是必需层。

升级不能走 dylib 换装：master 把共享库 soname 从 25 升到 26（ABI 声明不兼容，
结构体布局可能变化，强行换装是内存踩踏不是升级）。项目现在把 rawpy fork 的
`cc7b4748` 精确提交同时写入 `requirements.txt` 和 `pyproject.toml`；该 fork 又把
`external/LibRaw` 精确锁在 `e419de08`。因此常规 `pip install -r requirements.txt`
会直接构建并安装验证过的组合，不再依赖某台机器事后运行升级脚本，也不会被 PyPI
的 0.27.0 发布轮子悄悄替换。`tools/build_libraw_master.sh` 仅保留为已有虚拟环境的
显式修复入口；换钉必须全套回归，若解码输出漂移则重基线 SDR 冻结/金标。

项目构建已验证：`rawpy.__version__ = 0.27.0+libraw.e419de08`，并公开记录完整
LibRaw 来源提交；`rawpy.libraw_version = (0, 22, 0)`（master 线），soname 26，
A7 V 入表；**当时的全套 454 项测试在 NumPy/原生两条路径零漂移通过**（2026-08-01 快照数字，测试规模随后续批次持续增长）——master 对既有机型
（fp/iPhone 样张）的解码逐字节兼容，SDR 冻结与金标均未失效。

两层的分工从此明确：**LibRaw 升级**解决"LibRaw 内部色彩转换缺矩阵"（回退表
够不到的那一半）；**回退矩阵表**覆盖"比 master 还新"的机型窗口期（当前：
X-E5 借 X100VI 矩阵、A7R VI 待上游、GR IV 走 DNG 自带标签）。0.22.1 已知
机型的回退条目（A7S III/X100VI/Zf）永不触发，保留作老构建环境的防御层。

## 解码格式支持：颜色表缺口 vs 格式缺口（2026-07-30）

新机型解码失败分两类，处置完全不同：

- **颜色表缺口**（文件能解包、缺机型矩阵）：走上文的标定阶梯优雅降级，回退表
  可补——A7 V、X-E5 这类属于此类。
- **格式缺口**（LibRaw 根本打不开文件）：回退表无能为力，升级 LibRaw 也未必
  有用。**典型案例：尼康高效压缩（HE/HE\*）NEF**——Z9/Z8/Z6III/Z50II 世代的
  HE 格式使用 intoPIX **TicoRAW** 编码。本项目固定的 LibRaw `e419de08` 中，
  JPEG XS 标记进入未实现分支，不能由机型矩阵或 `.NEF` 扩展名证明可读。
  同机身的无损压缩 NEF 是另一编码类别，也需要实际文件验收。截至本次复核，
  [LibRaw #826](https://github.com/LibRaw/LibRaw/pull/826) 仍开放；提案中的逆向
  HE 解码器没有进入本项目 pin，不能当作已安装能力，HE 与 HE* 也不能合并宣称支持。

  **已关闭的格式缺口：Sony cRAW HQ（ARW6/LLVC3，ILCE-7M5 世代）**——
  y-g-jiang 的逆向解码器 `sony_arw6_load_raw()` 已于 2026-07-18 合并进
  LibRaw master（[LibRaw#824](https://github.com/LibRaw/LibRaw/pull/824)，
  上游随后调整了 ARW6 黑/白/线性上限点 e419de08）。本项目的 LibRaw pin
  （`tools/libraw-pin.env`）自 e419de08 起即包含该解码器（venv 构建内
  `libraw_r.dylib` 携带 `sony_arw6_load_raw` 已验证），Compression=32766
  的 ARW6 IFD 自动分派。当前本地 `DSC00225.ARW` 已确认这个 codec，完成完整／
  半尺寸解码与 AgX SDR/HDR 的 NumPy／Rust 对照。作者对特定样本／版本与 Adobe
  的接近一致不能外推为所有 RAW 的误差上界，也不能证明有损编码前的 ADC 样本被
  恢复；LUT 展开后的一个内部码阶并不一定等于一个均匀 DN。
  [Sony 官方](https://helpguide.sony.net/ilc/2540/v1/en/contents/251h_raw_file_type.html)
  区分 Lossless Comp、Compressed(HQ)、Compressed。本项目当前只从该文件确定
  ARW6/LLVC3，未获得可核实的菜单子模式和快门信息，不能把它标成无损或完整读出模式。

格式缺口的处置（`raw_io._unsupported_format_guidance`，报错即引导）：

1. 用支持该具体编码的 Adobe DNG Converter 转成 DNG 后再次逐文件探测；转换结果
   是否保留 CFA、处理声明和合适的噪声模型，需要独立核对，不能承诺转换后全功能；
2. 相机内改用**无损压缩** RAW；
3. **尝试 Apple RAW 自动模式**：若系统能够解码该文件，可在 LibRaw 证据缺席时
   输出场景图像。传感器 CFA、剪切和 SNR 明确不可用；HDR 使用最多 1EV 的图像估计。
   是否支持具体压缩变体以实际渲染为准，相机兼容名单不能证明某个 HE 文件可用。

Apple 覆盖注记：Z50 II 在 macOS Sequoia/26 名单上（标准 NEF 确认；HE 变体的
Apple 原生解码覆盖待真实样张验证——第三方如 RAW Power 以自带扩展解码支持
HE/HE\*，说明系统级覆盖可能不完整）。

### 证据层的能力边界（局限的精确表述，2026-08-01）

格式缺口上"证据不可得"是**工具能力的客观交集为空**，与我们的实现选择无关：

> 马赛克数据在文件里存在（只是以 TicoRAW 压缩）。**LibRaw** 肯原样暴露马赛克
> （`raw_image_visible` 一族接口），但解不开该压缩；**Apple CIRAWFilter** 即使能解开
> 某个文件，也只输出解拜耳后的场景图像，API 不暴露
> 马赛克、CFA 排列或逐像素电平。能开锁的不给看原件，肯给看原件的开不了锁。

证据层支撑的能力（此类文件上因此不可得）：逐像素 CFA 剪切掩码（clip retreat、
gated core）、**有传感器证据支持的最亮高光**、满阱/黑白电平的绝对 EV
标尺、RAW 健康度（lag1/空码）、**RAW 码值直方图**（未解拜耳计数分布）。

**不**依赖证据层的能力（证据缺失也完整可用）：场景解码与全部渲染（AgX、
SDR 交付）、元数据级信息（机型/ISO/部分电平，自有 TIFF 解析）、以及**渲染侧
直方图**——scene-referred（EV 轴）与 display-referred（输出码值）两个口径都
只读渲染缓冲，未来做 GUI 实时直方图属于此侧，任何证据状态下均可实时；证据
只影响直方图上的**注记线**（RAW 过曝点、可靠尾部、满阱线）：有证据时叠加，
无证据时如实略去。

当前实现保留证据来源一致性，同时将证据获取、实际场景解码和可选校正参考分成
独立能力。Apple-only 保留 SDR/HEIF 导出及场景分析，不伪造传感器指标；HDR 的
`decoded-image-estimate` 最多 1EV，关闭分通道扩展，并明确标记为工程降级策略。
参考成功但没有足够可靠样本时，HDR 仍为零，不能将测得的否决重新解释为“未知”。
本轮缺证据分支由故障注入验证；私有压缩 RAW 的覆盖仍需对应真实样张验证。

## 逐文件支持探针（`--support`）

"支持"不再是一个模糊词：`dngscan <文件> --support` 输出该文件在两条解码线上的
**逐档预检报告**（`decode_support.probe_decode_support`，LibRaw 解包、校正配方与 Apple 版本探测，不执行完整场景渲染；
GUI 选中文件后同一报告显示在解码器控件下方）：

```text
机型：SIGMA SIGMA fp
Evidence（LibRaw）：✓ 解包/标定预检通过（文件自带 DNG 双光源标定）
LibRaw 场景解码：✓ 解包/标定预检通过；尚未验证实际渲染
Apple RAW：✓ 提供 RAW 9（自动模式会在渲染失败后重试旧版）；尚未验证实际渲染
传感器先验：✓ 有（PhotonsToPhotos 实测标尺）
```

分层定义——LibRaw：`✗ 格式缺口`（打不开，回退表无效，如 HE NEF）→
`△ 色彩无锚`（可解码、零标定）→ `△ 回退矩阵`（WB 已代偿、内部转换仍缺）→
`✓ DNG 自带标定` / `✓ 矩阵在表`；Apple RAW：`✗ 不支持` → `△ 仅 RAW 7/8`
（自动模式可使用）→ `✓ RAW 9`，另有传感器证据不可用标记。自动版本在实际渲染失败时
会继续尝试旧版，全部失败再尝试 LibRaw；显式版本失败即报错。传感器先验单列；它影响分析标尺，并可通过 SNR 门控和可选色度核影响成像，不能仅凭
探针成功就认为噪声模型或降噪传播适用。最终来源和状态以实际分析为准。格式缺口的报错
信息与本探针互链。

## iPhone 双解码对照

以下为历史 iPhone 16 Pro 同帧双解码示例。该系列 standard RAW 样张为 Bayer DNG，
文件内暗角指令为 `FixVignetteRadial`；LibRaw 执行该指令，Apple RAW 交由系统处理。
当前逐文件实测范围见上文，不能由这张对照图推定 ProRAW 支持或暗角校正的定量精度。

![iPhone 16 Pro 同帧双解码对照](assets/decoder-iphone-libraw-vs-raw9.jpg)

## 原生 RAW 的逐文件统计资格（2026-10-10）

项目现在分别记录「实际完成解码」「实际完成成像」与「噪声标定适用」。原生 TIFF
只读取唯一的主 CFA IFD；缩略图、增强图或多个候选 RAW 帧不能替代它。可读取的
信息包括编码、存储位数、原始几何与默认裁剪；拿不到的 ADC 位数、固件、连拍、
binning、快门和完整 readout ID 保持未知。Sony 32767 还需码流长度条件才能区分
ARW2 与 unpacked，Nikon 34713 本身不能区分无损／有损／HE。

已知有损或尚未确认无损的文件，当前不会直接套用来自其他读出／压缩模式的物理先验，
但通用曝光、AgX、SDR/HDR 成像继续。以后若有相同 codec、LUT 域和采集模式的独立
实测模型，可以在明确的噪声合同下扩展资格；当前拒绝的是未经验证的模型适用性，
不是把有损 RAW 永久排除在项目之外。

编码白点与线性响应阈值分开：DNG 使用文件 WhiteLevel；非 DNG 使用 LibRaw 的
展开后 maximum。当前 ARW6 pin 的 black=1024、maximum=39002、linear_max=32800，
对应编码跨度 37978 与线性有效跨度 31776，TIFF 声明的 14-bit 不替代展开后的码域。
标定 DN 尺度与解码方差传递共用编码跨度，原始剪切／非线性判断仍使用各自的有效
阈值。依据见固定版本的 [TIFF 分派](https://github.com/LibRaw/LibRaw/blob/e419de08001de28ae6988ecb22df47e52b9c5eaa/src/metadata/tiff.cpp)
与 [ARW6 端点修正](https://github.com/LibRaw/LibRaw/commit/e419de08001de28ae6988ecb22df47e52b9c5eaa)。

实际样本的 SHA-256、运行时版本、CFA 四相位、电平、固定局部区域，以及全图和局部
NumPy／Rust 差异保存在[机器可读记录](assets/delivery-quality/native-raw-20261010.json)。
复跑方式（不保存用户照片或完整像素缓冲）：

```sh
python tools/validate_native_raw.py "$SAMPLES/DSC00225.ARW" --sizes half full \
  --out /tmp/native-raw-acceptance.json
```

这份对照验证同一固定 LibRaw 下的实现一致性。当前缺少厂商／Adobe 的独立解码
参考、Sony 其他编码与 APS-C／快门／ISO 组合、同模式暗场／平场／近饱和标定对；
授权本地样本目录也没有 Nikon NEF，因此 Nikon 传统／HE／HE* 均未完成实片验收。
这些缺口记录为未测，不计为通过。

## iPhone 传感器声明的边界

历史文档把网络传闻中的 IMX903、ADC 位数、工艺、DCG 和满阱结构解释写成了确定
规格；本项目没有足够的一手证据支持这些具体断言，现撤回这些声明。DNG 的存储位数
也不能证明传感器 ADC 位数，或替代这台设备与相应拍摄模式的电子域标定。

iPhone standard Bayer DNG 与 Linear RGB ProRAW 按逐文件数据类型、ColorMatrix、
NoiseProfile 和处理声明决定能力；不能把一个代际名称、网上推测的传感器型号，或
另一代 iPhone 的测量借用为物理先验。当前五张 Bayer DNG 验收范围仍见本文开头。
