# 数据驱动 AgX 与胶片转换：设计审查与后续合同

> 审查快照：`64c1cc2`（`illuminant-tiers-runtime`）加当前未提交的 Route D
> 运行时改动。审查日期：2026-08-27。本文审查的是设计语义、数学连接、数据能力与
> 验收门，不把“代码存在”直接等同于“目标已经成立”。

> **处置记录（2026-08-27，PR #141/#142）**——审查正文保持原样，以下只记录处置。
> - **8.2/P1.1 的更大问题——Route D 的前提被撤回。** 复审（同日第二份审查）指出
>   阶段 1 的"assumed"误差用了跨光源无意义的绝对尺度，且 D55 场在未做色适应的
>   `xyz_i` 上求值；按运行时坐标（白平衡像素经 `inv(M_D55)`）与各模型自身白板归一
>   重算：A 光下 D55 模型 held-out p99 中位 0.699 vs 专档 0.733，LED-B3 0.674 vs
>   0.680——专档没有优势。本 PR 曾按第二版口径实现的钨丝档资产、`--film-illuminant`、
>   按白平衡插值与光源权重**全部撤回**；记录、工具与门改为钉住否定结论与两个口径
>   错误（`docs/illuminant_tier_cv.json`、`tests/test_illuminant_tier_cv.py`）。
> - P0.1 `FilmSpatialContext.__slots__`：该字段随分档一起撤回，崩溃路径不再存在；
>   真实 render seam（`test_hdr_confidence`）在最终树上通过。
> - P0.2 文档：`FILM_STAGE_A_CHROMA_FIELD` 路线 D 段重写为否定结果与三版口径历史；
>   `ARCHITECTURE(.zh-CN)` schema 8、README/USER_GUIDE 的"观察者逆矩阵"改为 Stage A
>   色度场/3×3 口径。
> - P0.3 报告：Stage A 行输出**实际运行的模型**及其 held-out 残差、3×3 基线，并把
>   同一模型在白平衡后钨丝/高显色 LED 场景下的 p99 并列打印（数字随资产烘入，
>   `tests/test_stage_a_report.py`）。
> - P0.4 README "测量的乳剂×相纸系统印出的样子"已改为"公开光谱数据约束的三刺激
>   重建 + 声明的 modelled 层间项，关闭编辑性外观"。
> - P1.2–P1.4（AsShot 语义、假设/权重写入报告、混光语义）：随分档撤回而消解；报告
>   固定声明"光源假设=D55（实测无需分档）"，窄带光源作为置信度降级依据记录在案。
> - 未做（不属本轮）：7.3 负 Rec.2020 域外语义（P2.1）、外部反射谱库（P2.2）、
>   逐 stock 误差分布发布（P2.3）、P3 AgX/HDR policy 标定、资产 provenance 扩展到
>   CV 记录/反射率库/CMF/光源数据、`fit_film_curve` 的 ResourceWarning、CV 测试缺
>   colour-science 时的提示改进。

## 1. 审查问题

这次审查回答两个问题。

第一，现有相机、传感器、RAW 与镜头数据，是否已经被正确用于 AgX SDR / AgX HDR
出图。这里的“正确”不是让数据库替照片决定曝光或风格，而是把数据编译成：信号可信
度、可用动态范围、趾肩边界、HDR 亮度预算，以及色度可以被保留到什么程度。

第二，在没有逐相机实拍标定、没有完整传感器 SSF、也不能从三个 scene RGB 数值重建
原始光谱的前提下，胶片转换能严肃到什么程度。目标不是宣称逐分子复原，而是让每一层
近似都有明确坐标、误差、退化路径和 provenance；测量底座、模型补全和编辑外观必须可
分离。

## 2. 总结判定

| 目标 | 当前判定 | 关键理由 |
|---|---|---|
| RAW 证据参与 SDR AgX | **基本成立，但默认路径只部分启用** | 固定曝光锚、可靠 body/tail、CFA clip、PDR/读噪声已经进入 plan；SNR 趾部绑定只在 `endpoint_mode=evidence`，逐像素电子 SNR 只在 `gated` 核。 |
| RAW 证据参与 HDR AgX | **结构成立，参数仍待显示语料标定** | HDR 亮度预算只读可靠尾部；亮度与色度授权分离；CFA clip 逐像素撤销色度自由；无可靠尾部即拒绝 HDR。rho、margin、SNR 门仍是项目策略常数。 |
| 相机数据保持拍摄意图 | **成立，但需限定 view adaptation 的措辞** | EV0 是固定标尺，不做中位数归一；暗场仍然暗。`view_brightness` 最多自动抬 30%，属于显示适应而非曝光，必须单独声明。 |
| 胶片 Stage A 的相机无关转换 | **已达到当前三刺激条件下较强的实现** | Route C 的 `E=Y*2^L(x,y)` 保持曝光齐次性，24/25 stock 通过 held-out 判据进入运行时；残差明确承认同色异谱边界。 |
| 胶片光源条件 | **端点模型已测，运行时接线尚未完成验收** | Route D 的 A / LED-B3 专档和 D55 基线已生成；当前未提交接线存在直接崩溃，且中间 CCT 的 log-exposure 插值尚无直接光谱 oracle。 |
| 胶片介质与印相链 | **结构成熟** | Stage A、B1、tau、相纸 1D、B2 已因式分解；技术链、modelled interimage、editorial recipe 可区分；资产 fail-closed。 |
| “严格 ALEV / 严格胶片复刻” | **不成立，也不应作为当前宣称** | 缺少目标机身完整 SSF、个体差异与实拍 oracle；scene RGB 已丢失同色异谱信息。当前应称“光谱数据约束的三刺激重建”。 |

总体方向是对的：相机数据负责告诉 DRT **哪里可信**，胶片数据负责描述 **可信的
scene tristimulus 应如何激励胶片层**。两类数据没有被混成一个“智能 look”。目前主要
风险已从管线拓扑转移到三处：Route D 未完成接线、Stage A 域外输入语义、以及报告仍
在输出旧模型的误差数字。

## 3. 不可破坏的设计合同

### 3.1 捕获意图合同

1. EV0 必须由固定相机/管线标尺与用户 EV 定义，不能令场景中位数等于 0.18。
2. 相机先验可以约束可信范围，不能直接制造自动曝光或自动白平衡。
3. CFA 剪切可以撤销颜色权限，不能凭空增加亮度预算。
4. 重建高光可以改善外观，但不能反向成为“传感器测到了更多 headroom”的证据。
5. 数据缺席时要回退到逐帧估计或保守路径，并把降级写进报告；不能填一个看似精确的
   默认数。

### 3.2 DRT 关注点分离合同

- Tone：black、white、pivot、toe、shoulder、contrast。
- Color geometry：inset/outset、hue path、chroma retreat、path-to-white、gamut fit。
- Capture evidence：full well、clip topology、可靠尾部、噪声、先验质量。
- Delivery：sRGB/P3、SDR/HDR、OETF、量化、dither、容器。

Capture evidence 只能编译前两层的**边界与置信度**。例如 P3 越界压力可以降低 HDR
通道分离，但不能移动黑白端点；场景 body 可以选择 shoulder 形状，但不能冒充 CFA
是否剪切。

### 3.3 胶片诚实性合同

真实层级必须始终可辨认：

- `measured/tabulated`：可追溯的 SSF、染料密度、相纸感度、光源 SPD、CMF。
- `derived`：由上述数据积分、拟合或烘焙出的 observer、色度场、B1/B2、tau。
- `modelled`：无直接数据的 interimage、滤色头近似、光源假设。
- `editorial`：reference recipe、palette、richness、用户显影扰动。

`technical` 只能表示“关闭 editorial appearance 后的技术链”，不能简写成“实测胶片
直出”。技术链里仍含三刺激反演、模型化层间项和公开资料数字化误差。

## 4. 数据能力清单：能做什么，不能做什么

### 4.1 传感器辐射计量数据

当前 curated / JPTC / P2P 先验给出 gain、read noise、PDR、unity gain、full-well
等量。它们适合：

- 把读噪声与 shot noise 解成 `EV_SNR1/10/20`；
- 给单帧噪声估计加合理边界；
- 决定暗部颜色是否仍有可信信号；
- 发现机身、ISO、快门模式不匹配并整体降级。

它们不包含波长维的 RGB SSF，不能用于相机到胶片的颜色仿真，也不能从 PDR 推断肤色
或高光 hue path。`JPTC-SPECTRUM/1` 是空间噪声功率谱，不是光谱敏感度；命名相似
不能连接到胶片 Stage A。

### 4.2 DNG 色彩矩阵与 RAW 元数据

DNG ColorMatrix、CalibrationIlluminant、AsShotNeutral、black/white level 能建立相机
RGB 到 scene colorimetry 的标尺，并检查矩阵健康。它们是 camera colorimetric
normalization 的依据，不是传感器完整 SSF。Route E 的矩阵健康报告有价值，但在没有
独立 target 的情况下应保持诊断身份，不自动“修正”矩阵。

### 4.3 CFA 像素证据

这是本项目相对通用中段 tone mapper 的结构优势。LibRaw 路径能在 demosaic 前保留：

- per-channel full well 与软 clip mask；
- 单通道/多通道剪切拓扑；
- reliable body 与 reliable tail；
- 有先验时的电子域 SNR map。

RAW9/Core Image 没有对齐的 CFA 几何，因此只能使用聚合证据和全局保守上限。现有实现
没有伪造空间 mask，这一点符合高保真原则。

### 4.4 光谱反射率、胶片 SSF 与介质数据

rawtoaces 反射率、胶片感色层、染料密度、相纸与观看数据能训练 scene tristimulus 到
胶片层曝光的确定性近似，并构建后续印相链。它们不能解除 metamerism：若两个材料在
输入 scene RGB 已相同，任何逐像素确定性函数都不能恢复它们在胶片上的不同响应。

### 4.5 镜头与滤镜透过率

118 条透过率曲线目前是可读取数据库，不是渲染输入。正确的使用位置是**离线联合光学
模型**：`illuminant * lens_T * camera_SSF` 或 `illuminant * filter_T * film_SSF`。
已经拍进 RAW 的镜头光谱效应不能在运行时再乘一次，否则是双重施加。没有镜头身份、
焦段、光圈与实拍标定时，保持诊断/研究数据比自动接线更诚实。

## 5. AgX SDR 审查

### 5.1 已正确实现的部分

`compute_exposure_gain()` 使用固定中灰标尺和手动 EV，不读取场景中位数；因此夜景不会
因为“自动正确曝光”被拉成灰。`scene_tone_metrics()` 将可靠 body 与完整/可靠 tail
分开，LibRaw 排除 CFA clip，RAW9 只能按聚合 clip share 对亮度排序做 trim。这个
分工是正确的：body 决定主体统计，可靠 tail 决定白端和 HDR 资格，重建尾部只可做
SDR 防御性回退。

传感器先验经过统一 `prior_usability()` 门，gain、read noise、PDR、electron noise
一起可用或一起失效，避免“只借一个看起来好用的数字”。PDR 只把单帧 DR 限在先验
周围 ±1.5 EV，没有完全覆盖本帧估计。

Route A 进一步把

`N_s = (s^2 + s*sqrt(s^2 + 4r^2))/2`

转换为 `EV_SNR1/10/20`。`endpoint_mode=evidence` 将黑端放在工程噪声底附近，并用同一
曲线求解器把 toe end 绑定到 `EV_SNR10`。公式与坐标约定正确，缺先验时 fail closed。

### 5.2 尚未完全达到的部分

默认 `endpoint_mode=adaptive` 不使用 SNR 坐标重解 toe；默认 `tone_core=agx` 也不消费
逐像素电子 SNR map。逐像素 SNR 目前属于 LibRaw-only 的 `gated` 实验核。因此准确的
表述应是：

> 默认 AgX 使用 RAW clip、可靠 tail、单帧 DR 和可用传感器先验来编译范围；只有
> evidence/gated 选择才进一步让电子域 SNR 直接塑造 toe 或逐像素颜色权限。

这不是必须马上把 `evidence` 设为默认。当前传感器库覆盖与模式匹配仍不均匀，保留
默认 adaptive 是保守选择。但 GUI/报告必须让用户看见“本次先验是否实际进入了 tone”
而不是只显示“数据库里有这台机身”。

`view_brightness` 会依据暗 body 与 DR 在显示域自动增加最多 30% 亮度。它不移动固定 EV
标尺，也不抬黑白端点，但会改变观看亮度。设计上可以保留为 bounded viewing
adaptation；文档不能把它和“0EV 完全不动”混为一谈。应继续用真实暗场语料验证场景
间亮度排序不被反转。

### 5.3 SDR 后续验收

1. 同一 RAW 改变先验质量状态时，只允许 `usable_dr_eff`、证据 endpoint、SNR 坐标和
   guidance 改变；固定 exposure anchor、WB、pivot 不得改变。
2. 对同一场景施加全局 `k` 倍曝光，除 clip/rail 状态变化外，SNR 坐标与 scene EV
   坐标要按解析关系平移。
3. 暗场 corpus 钉住：scene median 排序、主体肤色亮度、黑端命中率、view brightness
   增益，防止“无中位归一”被未来启发式间接破坏。
4. 报告分别写“先验存在”“先验通过质量门”“先验实际被哪个消费者使用”。

## 6. AgX HDR 审查

### 6.1 亮度路径

当前 HDR 是独立 formation，不从 SDR 像素抬 gain。`reliable_tail_ev_p9999` 是 headroom
唯一 scene 证据；display headroom 是容量，requested 是场景挣得的范围，rendered 是
曲线实际承载的范围，actual 是渲染后测量值。四者分开是正确的。

HDR body 与 shoulder 分离，headroom 只进入 shoulder。scene median 不参与 HDR 预算，
所以更亮的屏幕不会自动把夜景整体抬亮。C1/单调性、reference white、body invariance
已有数学测试覆盖。

### 6.2 色度路径

`rho` 只在 reference-white common path 与 native extended path 之间选择色度；最终
重新归一到 native HDR luminance。因此亮度由 HDR 曲线单独拥有，clip/SNR/gamut 只撤销
色度自由，不削掉 HDR 光感。

新加入的 peak-proximity convergence 符合目标：未剪切像素可一直保留 HDR 色度；剪切
像素越靠近内容峰值，越向公共白路径收敛。film full 的 HDR extension 也把 luminance
gain 和增量色度授权分开，双通道剪切时只扩展中性亮度。

### 6.3 仍需标定的部分

`RHO_BASE=0.5`、多通道剪切 10%、P3 压力 20%、RAW9 rho cap 0.25、white margin
0.30/0.50 EV、shoulder start 0/0.20 EV、tail SNR 2:1 到 10:1 等均是项目 policy，
不是 darktable、Blender、Apple、ACES 或 ISO 常数。代码已把它们登记到 policy register，
这是正确做法；下一步不是继续理论推导出“唯一值”，而是用 EDR 实屏语料标定。

建议每个候选 policy 记录四类指标：body shift、可靠高光 detail、clip 区 chroma error、
输出 gamut projector 的 pullback。必须按日景皮肤、夜景灯源、霓虹、金属反射、天空、
单/双/三通道 clip 分层统计，而不是只比较总体平均。

## 7. 胶片 Stage A 审查

### 7.1 当前数学上做对了什么

Stage A 不再把 Rec.2020 三通道直接当三层乳剂。它先把 scene-linear Rec.2020 映射到
胶片三层曝光，再过每层 1D characteristic curve 得到 dye amounts。中性锚与曝光轴
定义清楚，后续 Stage B 的 LUT 域是 dye amount，而不是含混的“RGB density”。

原 3x3 observer 在 held-out 反射谱上的 p99 为 0.47–1.13 stop；训练与 CV 接近，说明
主要是线性模型/同色异谱结构误差，不是简单过拟合。Route C 的形式

`E_layer = Y * 2^L_layer(x,y)`

是合适的下一阶模型：色度场只见 `(x,y)`，亮度 `Y` 外提，所以
`E(kX)=kE(X)` 由构造成立。三次模型在同折、同白板锚、同 190 反射谱上使 p95 中位降低
39%、p99 中位降低 31%；按 p95 至少改善 15% 且 p99 不退步的规则，24/25 stock 采用，
Pro 400H 保留 3x3。这个采纳逻辑比“一律换更复杂模型”严谨。

### 7.2 硬边界

色度场降低的是**非线性拟合不足**，不会消除 metamerism。当前剩余约 0.46–0.79 stop
的 held-out p99 不能被称为“严格胶片响应”。扩大同一训练集、继续升多项式阶数也不能
证明泛化；真正的提升需要独立反射谱库、发射体/窄带光源压力集，以及目标胶片更可靠
的 SSF/染料数据。

相机身份不应直接烘进 stock Stage A。正常拓扑是：

`camera RAW -> camera normalization -> common scene colorimetry -> film Stage A`

只有拿到某台相机完整 SSF 和独立 target 时，才可在 camera normalization 层增加
camera-conditioned residual；film stock 资产仍应保持相机无关。否则同一 Portra 资产
会随相机而变，既无法解释也无法复用。

### 7.3 当前发现的域外语义问题

`chroma_field_log_exposure()` 先对每个 Rec.2020 通道做 `max(channel, 1e-9)`，然后计算
XYZ/色度。对相机矩阵产生的负 Rec.2020 分量，这不是“回退”，而是先改颜色再进入
色度场。实测 Portra 400 对 `[-0.1, 0.2, 0.3]` 的红敏层与 3x3 路径可差约 12.75 stop；
虽然 characteristic floor 会掩盖一部分显示差异，数学语义仍不成立。

应二选一并写成合同：

1. **推荐**：原始 scene RGB 任一通道非正、非有限或落出声明 training domain 时，整
   像素回退 3x3 observer，不在进入模型前改色；或
2. 在 Stage A 之前定义一个共享、可逆/有界的 scene gamut compression，并用同一变换
   重建训练集和运行时。

当前“钳正使色度永远在 Rec.2020 原色三角形内”的测试只证明代码自洽，不能证明钳正
符合目标。需要增加含负分量的独立 oracle 与退化测试。

## 8. Route D 光源假设审查

### 8.1 已证明的部分

固定 D55 假设在钨丝、LED 和窄带 RGB 光源上会产生显著 Stage A 误差；专用同光源
field/observer 能把 p99 拉回与 D55 同级。因而“光源假设”确实是模型的一阶变量，
不是可忽略的小修。

LED 只允许显式选择是正确的：CCT 不能区分磷光 LED、荧光、RGB LED 和混光。
所有 tier 保持灰轴与曝光齐次性，也是正确合同。

### 8.2 尚未证明的数学

当前自动钨丝路径用 reciprocal CCT 得权重，再在 **log exposure** 域混合 D55 与 A
两档。这等价于对层曝光取几何插值。它保持 `E(kX)=kE(X)`，但并不由光谱积分推出：
若 SPD 是线性混合，层曝光应在线性 exposure 域混合；真实 3200K/3400K 黑体也不是
D55 与 A 的简单线性或几何混合。

因此 Route D phase 1 证明了“端点专档有价值”，还没有证明“中间 CCT 插值正确”。
上线前必须用 3000/3200/3400/4000/4500K 的直接 SPD oracle 比较：

- log-exposure interpolation；
- linear-exposure interpolation；
- 直接多锚点 field/observer 插值。

以 held-out p95/p99 和灰轴误差选择方法，不能只测 homogeneity。若两端插值达不到
既定门槛，应增加 3200K/4000K 锚点，而不是调一个经验 easing。

### 8.3 当前实现阻断

当前工作树在 `FilmSpatialContext.__init__` 写入 `self.illuminant_weights`，但该字段未加入
`__slots__`。任何走到该 context 的 film full render 会直接抛
`AttributeError: ... has no attribute 'illuminant_weights'`。定向测试已在
`tests.test_hdr_confidence.FilmPairChromaAuthorizationTests` 的真实 render seam 复现。
这是 Route D 合并前的 P0 阻断，不是测试误差。

另一个语义风险是 `wb_mode=camera` 当前直接使用 D55 fallback。AsShot 只说明相机白平衡，
不说明拍摄 SPD 是日光。可以保留 D55 作为兼容基线，但报告必须写“光源未知，使用 D55
回退，低置信度”，不能把 AsShot 与日光并列。若元数据和 matrix 能可靠估 CCT，可用于
日光/钨丝大类假设；仍不得从 CCT 自动猜 LED。

## 9. Stage B、外观层与“胶片味”边界

当前 full 链的因式分解是合理的：

`Stage A -> dye amounts -> B1 -> tau(E) -> paper 1D curves -> B2 -> neutralization -> appearance`

B1 与 tau 分离避免为每个曝光 timing 重烘整个显示 LUT；paper curve 保持 1D 物理位置；
B2 按介质与观看条件复用。technical、reference、custom 也形成了正确的产品语义：
technical 是可回退底座，reference/custom 承认编辑判断。

仍需修正两类表述/报告：

1. README 中“测量的乳剂×相纸系统印出的样子”过强。应改为“公开光谱数据约束的
   三刺激重建 + 声明的 modelled interimage；关闭 editorial appearance”。
2. 报告目前仍打印 `observer_p99_stop`，即旧 3x3 的误差。24/25 stock 实际使用色度场
   后，这个数不再代表当前运行时 Stage A。资产已有 `chroma_cv_note`，但 loader/report
   没有携带 selected model、selected p95/p99 和 tier residual。报告应输出本次实际路径：
   `D55 field / A 3x3 / LED field` 及对应 held-out 误差；旧 observer residual 只能作为
   baseline 对照。

“胶片味”不应通过夸大 technical 链来获得。technical 负责介质响应可解释，reference
负责可见的 palette/hue-density 选择，空间成像负责 halation/grain/scatter。三者可以
强，但必须各自在自己的层上强。

## 10. 测试体系审查

### 10.1 已有强项

- 固定曝光锚与夜景不重曝光有回归；
- evidence toe 绑定 SNR10，adaptive 路径保持不动；
- HDR 无可靠 tail 即零 headroom；body 与 HDR shoulder 分离；
- HDR clip 色度授权、峰值收敛、film HDR 增量中性化有公式与真实 render seam 测试；
- Stage A field 有 CV 决策、曝光齐次、灰轴、retained 3x3 bit identity 测试；
- Route D 有 plan 解析、LED explicit-only、tier 灰轴与 homogeneity 测试；
- film 资产 schema、hash、domain 与 oracle 使用 fail-closed。

### 10.2 必须补的门

1. Route D `FilmSpatialContext` 的真实 SDR full render、HDR film pair、halation on/off 三条
   seam 均需覆盖，防止只测试纯函数而漏掉 `__slots__`/plan transport。
2. 中间 CCT 必须对**直接光谱积分 oracle**，不能只断言权重公式和齐次性。
3. Stage A 增加负 Rec.2020 / 超域 scene RGB；测试应验证声明回退，而不是验证钳正后
   一定落进三角形。
4. 报告测试应钉“实际 selected model + 实际 residual”，并随 illuminant tier 改变。
5. 相机先验做消费者白名单测试：更换 prior 只允许改变规定字段。
6. EDR corpus 增加 policy sweep，不只测单调/finite，还测主体亮度、clip 色度误差和实际
   显示观感判词。

本地定向运行 142 项测试，出现 3 项错误：两组 CV 测试因当前 Python
未安装 `colour-science` 无法运行；一项 film HDR seam 暴露上述 `__slots__` 真缺陷。CI
的 test dependency group 已包含 `colour-science`，所以前两项是本地环境不完整，不是
数学失败；测试入口仍可改进错误信息，明确提示需要 `dngscan[calibration]`/test group。
另外 `fit_film_curve.py` 读取 profile 时产生未关闭文件的 `ResourceWarning`，属于低优先
工程清理项。

## 11. 分阶段后续方案

### P0：合并阻断与事实一致性

1. 修正 `FilmSpatialContext.__slots__`，跑 Route D 的 full SDR/HDR/空间算子真实 seam。
2. 更新 schema 8、Route D phase 2 与当前 Stage A 色度场拓扑文档；当前中文架构仍写
   schema 6/observer inverse，`FILM_STAGE_A_CHROMA_FIELD` 仍写 phase 2 未做。
3. 报告输出实际 selected Stage A model/residual，不再把 3x3 baseline 当当前误差。
4. 修正 technical 的 measured 过强措辞。

### P1：完成 Route D 数学验收

1. 直接烘 3000–4500K 中间光源 oracle，比较 log/linear/multi-anchor 插值。
2. 将 AsShot 标成 unknown + D55 fallback，或使用有 provenance 的 CCT 只做 daylight/
   tungsten 大类假设；LED 保持 explicit-only。
3. 将 illuminant assumption、tier、权重、selected model、CV residual 写入导出报告。
4. 决定混光语义：单一 CCT、显式 LED、或 unknown。不要把无法辨识的混光压成一个假
   精确 Kelvin。

### P2：Stage A 域外与泛化

1. 选择负 Rec.2020 的 fail-closed 回退或统一 scene gamut compression。
2. 引入与 rawtoaces 不重合的反射谱库做真正外部验证；加入肤色、植物、织物、颜料、
   高饱和塑料、荧光/发射体压力集。
3. 为每个 stock 发布 error distribution，不只 p99：按 hue/material/illuminant 分组，
   同时报告 out-of-domain share。
4. 只有拿到真实 camera SSF 和 target 时，才做 camera-conditioned residual；它属于
   capture normalization，不属于 stock asset。

### P3：AgX/AgX HDR policy 标定

1. 为 SDR 的 adaptive/evidence 建真实样张判词与量化对照，决定 evidence 是否有条件地
   自动启用，而不是仅凭“先验存在”。
2. 对 HDR rho、SNR gate、margin、shoulder start 做 EDR 实屏 blind A/B。
3. 把 RAW9 的无空间证据状态作为独立 stratum，不用 LibRaw 最优参数外推。
4. 保持 policy register/version/fingerprint 机制；任何常数变化都连同 corpus 指标提交。

## 12. 推荐的子代理审查工作流

以后做这类全量审查，建议保持五个独立角色，最后由一个集成者只做冲突裁决，不让同一
代理同时证明自己的前提和结论。

1. **Capture / SDR agent**：审 prior、full well、noise、body/tail、曝光不变量；输出
   “数据字段 -> 允许消费者”矩阵。
2. **HDR agent**：审 headroom、C1 shoulder、luminance/chroma authority、gamut/delivery；
   输出曲线不变量和 EDR policy 缺口。
3. **Film Stage A agent**：审 SSF/reflectance/illuminant、observer/field、metamer boundary；
   必须运行 held-out 与域外压力测试。
4. **Film Stage B/provenance agent**：审 characteristic、B1/tau/paper/B2、technical/modelled/
   editorial 标签和资产 schema。
5. **Adversarial tests/docs agent**：不接受注释为证据，只追真实 runtime seam、报告输出、
   依赖缺席和文档主张。

集成门要求每条结论至少包含：目标语义、代码位置、数值/测试证据、失败时回退、仍未知
项。若两个代理结论冲突，优先增加能区分两种解释的 oracle，不以投票决定数学。

本轮确实尝试了独立子代理，但所有代理在初始化后因账户子代理额度触顶退出，没有产生
可用报告。因此本文是本地逐层审查结果，不冒充多代理共识。上述 workflow 保留为下一
轮额度可用时的可重复审查合同。

## 13. 最终设计结论

项目已经跨过“把相机数据摆在报告里”和“给胶片套曲线”的阶段。AgX 侧最有价值的
成果是：固定曝光意图不变，同时让 RAW 证据决定可用范围和颜色权限；胶片侧最有价值的
成果是：从相机无关 scene colorimetry 出发，把 Stage A 与介质印相拆成可验证的数学层。

下一步不应继续堆更多数据库或更多 look。优先级应是：先让 Route D 通过真实 render，
再用直接光谱 oracle 选择光源插值，修正 Stage A 域外输入和实际残差报告，最后才扩展
相机/光源 profile。做到这些以后，可以严肃地称这套系统为“RAW 证据驱动的 AgX DRT，
以及光谱数据约束、明确承认同色异谱边界的胶片转换”；仍不应称为某台相机或某卷胶片
的严格物理复刻。
