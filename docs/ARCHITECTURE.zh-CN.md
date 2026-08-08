# dngscan 架构与技术细节

> 本文承载完整的管线展示与每个环节的设计理由，是 [README](../README.zh-CN.md) 的
> 技术下层。只想用起来看[使用说明](USER_GUIDE.zh-CN.md)；想看问题与解法的推理过程
> 看[工程决策记录](ENGINEERING_NOTES.zh-CN.md)；胶片观察功能群的生产合同在
> [FILM_OBSERVATION_PLAN.zh-CN.md](FILM_OBSERVATION_PLAN.zh-CN.md)；机型支持与
> 降级策略在 [SENSOR_SUPPORT.zh-CN.md](SENSOR_SUPPORT.zh-CN.md)。想先了解软件分层、
> 用例和领域模型，请看[产品架构与领域模型](PRODUCT_ARCHITECTURE.zh-CN.md)。本文中的
> Capture、Tone、Color geometry、Delivery 是像素处理阶段，不是表现层、应用层、领域层、
> 基础设施层四层软件架构。

先读[认识论基调](#认识论基调与声明纪律)——它解释这条管线的每个"为什么"共享的
那个前提。然后按四层往下读（胶片观察位置作为跨层功能群，单独成节排在三层之后）：

| 层 | 负责什么 | 不负责什么 |
|---|---|---|
| [Capture](#一层：capture--raw-证据从哪里来) | 从 RAW 读出可测量的事实 | 不决定观感 |
| [Tone](#二层：tone--曝光和曲线怎样确定) | 亮度关系与显示动态范围 | 不动色相与色度 |
| [Color geometry](#三层：color-geometry--agx-真正改变了什么) | 色相路径、色度压缩、向白过渡 | 不移动黑白端点 |
| [Delivery](#四层：delivery--sdr-与-hdr-交付) | 编码、容器、gain map | 不改变已成形的像素 |

[解码器](#解码器：libraw-与可选的-core-image--raw-9)是与这四层正交的一根轴：它决定 RAW
怎样变成 scene-linear 像素，不决定这些像素之后怎么被压缩。分层是刻意的：调整某个
环节时，至少能知道画面为什么发生变化。

## 认识论基调与声明纪律

管线搬运的是"某个声明观察者对场景的报告"——传感器的、或胶片乳剂的——数字部分
提供技术上的完善（精度、可验证、可复现），不掺自己的口味。这一立场源自数字音频界
"透明搬运模拟介质签名"的传统（创始故事见
[工程决策记录·前言](ENGINEERING_NOTES.zh-CN.md)）。

由此推出三条**声明纪律**，全库通用：

1. **公开出处**：每个常数要么有出版物/数据手册可引（mired 表、CIE 轨迹、
   Bartleson-Breneman 常数、spektrafilm 密度曲线），要么有可复现的标定脚本；
   "不知道哪来的 3×3 矩阵"（Siragusano 讥为 *magically derived from somewhere*）
   不准入库。
2. **固定管线位置**：每个变换声明它作用在链条的哪一点（滤镜在前馈前、surround 项
   在拟合目标里、色头 LMS 场在 outset 之后），位置本身是合同的一部分。
3. **可测量的残差**：每次拟合发布 rms/max 与钉界参数（`fit.pinned`=声明的域外
   外推）；残差是产品的一部分，不是要藏起来的尴尬。

两条来自色彩科学文献的界碑限定了野心：**同色异谱**（Luther-Ives 条件不满足）使
"复现观感"在原理上不可达，所以合同是"忠实翻译声明观察者的报告"而非"复刻眼睛
看到的"；而报告只有连同**阅读条件**才完整——把黑暗放映厅的密度数字原样搬进明
环境显示，搬运的是脱离校准的裸数字（同引用 Cineon 码值不带 95/1023 黑锚）。
翻译因此必须补上观看条件之间的经典 surround 项，且只补这一项——其余色貌现象
（Hunt、Stevens 等）需要绝对亮度，介质不提供，声明为边界。完整合同与逐条边界见
[FILM_OBSERVATION_PLAN.zh-CN.md](FILM_OBSERVATION_PLAN.zh-CN.md)。

同一纪律的另一面是**降级也要声明**：机型缺标定数据时照常渲染，但报告与 GUI 标明
"暂无足够数据支撑准确运算，输出可能有无法预测的偏差"——声明的降级可用，静默的
降级等于隐藏白平衡（见 [SENSOR_SUPPORT.zh-CN.md](SENSOR_SUPPORT.zh-CN.md)）。

## 为什么单独做这条管线

darktable 的 scene-referred 管线很像一间信号处理实验室，理解每个模块怎样改变信号正是
其中的重要部分。dngscan 从里面取出与目标最相关的路径：LibRaw 解释、scene-linear
Rec.2020，以及 darktable GPL `agx` 模块里的曲线构造与原色几何。AgX 本身来自 Troy
Sobotka，并在 Blender / EaryChow 生态里发展；这里主要通过 darktable 面向照片的实现来
继承它。

但如果这里只是把 darktable 的 AgX 模块单独拆出来，意义其实不大。dngscan 真正想做的，
是把 RAW 采集层的信息一直带到最终显示变换里。

darktable 的 AgX 模块工作在解拜耳、白平衡和曝光之后的浮点图像上。它能看到图像，却
看不到原始 CFA：不知道哪个通道真的在传感器上剪切了，也不知道一块平滑高光究竟来自
真实信号还是高光重建。dngscan 是一体化的小管线，可以在解拜耳前保存这些证据，再用
它们区分可靠的场景主体、传感器尾部和已经丢失的高光信息。

这里的“自动”也建立在同一原则上。自动判断不是替照片决定审美，而是把可以测量的东西交给
测量：黑白电平、逐通道 CFA 剪切、噪声底、可用动态范围、亮度主体和高光尾部。这些信息
可以决定曲线需要容纳多少 scene EV、什么时候允许色度向白退让，以及什么时候不应该相信
一个重建出来的像素。

曝光补偿、白平衡、风格和 LUT 是另一回事。它们表达的是拍摄意图或个人口味，因此留在
这套自动分析之外，作为明确的选择。曝光与白平衡不必永远不动；约束在于内容自适应算法
不能在没有说明的情况下把夜景拉成灰色，或者把现场光本来的颜色抹掉。

## 管线

第一张图从采集证据与解码像素开始，一直画到不可变的 render plan。实线表示图像数据流，
虚线表示证据或控制信息。

```mermaid
flowchart TB
    RAW["RAW / DNG"]

    subgraph EVIDENCE["1. Capture 证据 - 始终由 LibRaw 在解拜耳前读取"]
        direction TB
        CFA["可见 CFA 马赛克与颜色索引<br/>raw_image_visible / raw_colors_visible"]
        META["元数据<br/>black 与逐通道 white level<br/>camera/daylight WB、BaselineExposure、方向"]
    end

    subgraph DECODERS["2. Scene 像素形成 - 解码器是独立选择轴"]
        direction TB
        SELECT{"Scene decoder"}
        LR["LibRaw<br/>固定 AsShot 重建预条件<br/>解拜耳选择<br/>clip / blend / reconstruct"]
        LRRGB["带方向的 linear Rec.2020 uint16<br/>关闭 auto-bright"]
        CIPROBE["CIRAWFilter 能力探测<br/>RAW 9 或显式 RAW 8/7 回退"]
        CI["固定 AsShot Core Image RAW 配方<br/>RAW 9：CoreML 重建 + 降噪<br/>旧版本：对应系统解码器<br/>高光恢复、镜头校正、DNG opcode"]
        CIRGB["extended-linear Rec.2020 RGBAh<br/>保留负分量与 1 以上数值"]
        LRREF["仅 aligned 模式<br/>half-size LibRaw reconstruct 参考"]
        ALIGN["Core Image 尺度策略<br/>aligned：解码后 G 中位比<br/>或 unity / 旧 measured"]
        SELECT --> LR --> LRRGB
        SELECT --> CIPROBE --> CI --> CIRGB --> ALIGN
        LRREF --> ALIGN
    end

    subgraph CONTRACT["3. 统一 scene 契约与分析"]
        direction TB
        HOTWB["项目自有热白平衡<br/>ColorMatrix 恢复 camera-linear 增益<br/>preview / export 共用"]
        SCALE["Scene scale contract<br/>存储尺度与 WB 余量<br/>文件 BaselineExposure 配方<br/>可选 Core Image 对齐标量"]
        SCENE["RawBundle scene frame<br/>scene_rec2020_render + scene_scale<br/>scene-linear Rec.2020 交接"]
        ANALYSIS["Analysis<br/>按饱和堆积或 metadata 解析逐通道 full well<br/>硬 threshold、clip%、2x2 拓扑与 ceiling<br/>噪声底 / 可选诊断 SNR / 可用 DR<br/>解码后 XYZ-Y-EV 与输出色域压力"]
        SPATIAL["解析后的空间 RAW 证据 - 仅 LibRaw 几何<br/>95-99% mask 按实测 full well 刷新<br/>headroom / clip class / SNR guidance<br/>Core Image 几何不借用这些 mask"]
        EV["Intent exposure<br/>固定 EV0 中灰锚点 x 2^EV<br/>手动 EV 或显式亮度参考搜索"]
        SAMPLE["Plan 采样<br/>scene scale + intent exposure<br/>可选且随 WB 适配的 scene 前馈"]
        METRICS["SceneToneMetrics<br/>可靠主体与完整尾部分离<br/>LibRaw 按空间 mask 排除<br/>Core Image 按聚合比例 rank trim<br/>点状发光体分类"]
        CONTROLS["渲染意图<br/>输出色域、tone core、AgX primaries<br/>胶片观察位置（WB 声明+滤镜+分离+曲线预设）<br/>前馈、punch 与有界明暗微调"]
        COMPILE["分别编译<br/>SceneToneMetrics<br/>ToneCompressionPlan<br/>ColorGeometryPlan"]
        PLAN["不可变 RenderPlan"]
        REPORTS["可选六面板 / CSV / 文本报告"]

        HOTWB --> SCALE --> SCENE
        SCENE --> ANALYSIS
        ANALYSIS --> SPATIAL
        SCENE --> SAMPLE
        EV --> SAMPLE
        SAMPLE --> METRICS
        METRICS --> COMPILE
        CONTROLS --> COMPILE
        COMPILE --> PLAN
        ANALYSIS -.-> METRICS
        SPATIAL -.-> METRICS
        ANALYSIS -.-> COMPILE
        ANALYSIS -.-> EV
        ANALYSIS -.-> REPORTS
    end

    RAW --> CFA
    RAW --> META
    RAW --> SELECT
    RAW --> LRREF
    LRRGB --> HOTWB
    ALIGN --> HOTWB
    CFA -.-> ANALYSIS
    META -.-> ANALYSIS
    CFA -.-> SPATIAL
    META -.-> SCALE
    SCENE -.-> REPORTS

    classDef source fill:#5b21b6,stroke:#7c3aed,stroke-width:2px,color:#f5f3ff
    classDef evidence fill:#78350f,stroke:#b45309,stroke-width:1.5px,color:#fffbeb
    classDef libraw fill:#075985,stroke:#0284c7,stroke-width:1.5px,color:#f0f9ff
    classDef apple fill:#134e4a,stroke:#0d9488,stroke-width:1.5px,color:#f0fdfa
    classDef pixels fill:#1e3a5f,stroke:#3b82f6,stroke-width:1.5px,color:#eff6ff
    classDef contract fill:#374151,stroke:#9ca3af,stroke-width:1.5px,color:#f9fafb
    classDef intent fill:#7c2d12,stroke:#ea580c,stroke-width:1.5px,color:#fff7ed
    classDef plan fill:#14532d,stroke:#22c55e,stroke-width:2.5px,color:#f0fdf4
    classDef aside fill:#3f3f46,stroke:#71717a,stroke-width:1px,color:#fafafa,stroke-dasharray:4 3
    classDef choice fill:#78350f,stroke:#f59e0b,stroke-width:2px,color:#fffbeb

    class RAW source
    class SELECT choice
    class CFA,META evidence
    class LR,LRRGB,LRREF libraw
    class CIPROBE,CI,CIRGB,ALIGN apple
    class HOTWB,SCALE,SCENE pixels
    class ANALYSIS,SPATIAL,METRICS,SAMPLE contract
    class EV,CONTROLS intent
    class COMPILE,PLAN plan
    class REPORTS aside
```

配色标记的是**来源**，这是最容易在阅读中丢失的信息：琥珀色是解拜耳前读到的 RAW 证据，
蓝色是 LibRaw 解码器，青色是 Apple 的，灰色是两者共同汇入的契约层，橙色是人给出的意图，
绿色是编译完成、下游必须遵守的 plan。

第二张图展开真正的渲染过程。SDR 与 HDR 共享 capture、scene intent、曝光和可选前馈，随后在
显示形成之前分叉；HDR 不会把已经完成的 SDR 像素当作 tone-map 输入。

```mermaid
flowchart TB
    SCENE["存储态 scene-linear Rec.2020 frame"]
    SCALE["解释 scene 单位<br/>stored / scene_scale x 固定锚点 x 2^EV"]
    PREFEED["可选 scene-linear 前馈<br/>相机响应校正 或 胶片光谱分离（20 款）<br/>随 WB 声明适配；镜前滤镜先于此作用<br/>（film full 模式整体旁路）"]
    PLAN["RenderPlan<br/>+ 独立的 Analysis 证据"]
    MASKS["逐像素 CFA mask 与 guidance<br/>仅存在于 LibRaw 几何"]
    LOOKPOLICY["可选本地 look 的 plan override<br/>AgX hue restore 与 target black/white"]

    SCENE --> SCALE --> PREFEED

    subgraph SDR["4A. SDR 显示分支"]
        direction TB
        RETREAT["曲线前可选 RAW clip retreat<br/>只在存在空间 CFA 证据时作用"]
        CORE{"SDR tone core"}
        AGX["agx<br/>inset -> 逐通道 darktable 式 C1<br/>linearize -> hue restore -> outset -> punch"]
        GATED["gated - LibRaw 实验<br/>亮度 C1 是唯一亮度权威<br/>AgX 颜色候选先对齐到同一 Y<br/>按 RAW 余量 / 剪切 / SNR / gamut 加权混合"]
        LUM["lum 对照<br/>Y / power / max norm -> scene C1<br/>恢复原始 RGB 比例"]
        NEUTRAL["neutral 诊断<br/>固定 Y-ratio sigmoid<br/>不编译 scene endpoint，不使用 AgX 几何"]
        FORMED["显示形成后的 linear Rec.2020"]
        OUTPUT["Rec.2020 -> linear sRGB 或 Display P3"]
        FILTER["可选 display LUT renderer<br/>display-fed FPE 或并行 scene-fed 输出变换<br/>与本地 look 互斥"]
        GRADE["可选本地 Oklab 色度 look<br/>+ 可选显示侧高光褪色"]
        FIT["最终权威 Oklab 保色相 gamut fit"]
        ENCODE["sRGB/P3 OETF -> 确定性 TPDF 抖动 -> uint8"]

        RETREAT --> CORE
        CORE --> AGX --> FORMED
        CORE --> GATED --> FORMED
        CORE --> LUM --> FORMED
        CORE --> NEUTRAL --> FORMED
        FORMED --> OUTPUT
        FORMED --> FILTER
        OUTPUT --> GRADE
        FILTER --> GRADE
        GRADE --> FIT --> ENCODE
    end

    subgraph HDR["4B. 独立 HDR AgX 分支 - 仅支持 AgX"]
        direction TB
        HDRPLAN["编译 HdrAgxPlan<br/>可靠 RAW 尾部 -> 请求扩展白点<br/>K 以下固定 gamma 的 darktable body<br/>K 以上单调 log-stop Hermite shoulder<br/>（alpha<=3 单段，越界细分为 C1 链）"]
        HRETREAT["HDR 自有 RAW clip retreat"]
        HINSET["AgX inset<br/>原生扩展白逐通道 C1 formation"]
        PATH["HDR 色彩几何<br/>reference-white 与原生色度路径混合<br/>CFA 剪切会局部收回 rho<br/>原生曲线始终是唯一 Y 权威"]
        HFINISH["Hue restore + outset + punch"]
        HP3["Rec.2020 -> extended-linear Display P3"]
        HVOLUME["HDR color-volume fit<br/>可靠尾部限制峰值<br/>保持 linear Y 与 RGB opponent direction"]
        ALT["Float16 RGB HDR alternate rendition"]

        HDRPLAN --> HRETREAT --> HINSET --> PATH --> HFINISH --> HP3 --> HVOLUME --> ALT
    end

    PREFEED --> RETREAT
    PREFEED --> HRETREAT
    RETREAT --> FILTER
    PLAN -.-> RETREAT
    PLAN -.-> LOOKPOLICY
    LOOKPOLICY -.-> CORE
    PLAN -.-> HDRPLAN
    MASKS -.-> RETREAT
    MASKS -.-> GATED
    MASKS -.-> HRETREAT
    MASKS -.-> PATH

    ENCODE --> FORMAT{"输出格式"}
    FORMAT -->|SDR| SDRJPEG["SDR JPEG<br/>ICC + quality + 4:4:4 / 4:2:2 / 4:2:0"]
    FORMAT -->|HDR| BASE["HDR 模式的 SDR 底图<br/>Display P3；archive q100/4:4:4，share q90/4:2:0<br/>禁用 look / filter / highlight fade"]
    BASE --> PACKAGE["Core Image ISO 21496-1 写入<br/>RGB 辅助 gain map + content headroom<br/>JPEG 或 HEIC 容器"]
    ALT --> PACKAGE
    PACKAGE --> VERIFY["回读验证<br/>P3 profile、RGB gain map、声明余量、archive 要求 4:4:4<br/>SDR 码值误差 + HDR 块级与像素级色品门禁<br/>按档位与容器分别标定"]
    VERIFY --> HDRJPEG["原子替换后的 HDR gain-map JPEG / HEIC"]

    classDef shared fill:#374151,stroke:#9ca3af,stroke-width:1.5px,color:#f9fafb
    classDef sdrpath fill:#1e3a5f,stroke:#3b82f6,stroke-width:1.5px,color:#eff6ff
    classDef hdrpath fill:#4c1d95,stroke:#a78bfa,stroke-width:1.5px,color:#f5f3ff
    classDef choice fill:#78350f,stroke:#f59e0b,stroke-width:2px,color:#fffbeb
    classDef optional fill:#3f3f46,stroke:#71717a,stroke-width:1px,color:#fafafa,stroke-dasharray:4 3
    classDef deliver fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#f0fdf4
    classDef gate fill:#7f1d1d,stroke:#ef4444,stroke-width:2.5px,color:#fef2f2

    class SCENE,SCALE,PREFEED,PLAN,MASKS shared
    class RETREAT,AGX,GATED,LUM,NEUTRAL,FORMED,OUTPUT,FIT,ENCODE sdrpath
    class HDRPLAN,HRETREAT,HINSET,LIFT,HFINISH,HP3,HVOLUME,ALT hdrpath
    class CORE,FORMAT choice
    class FILTER,GRADE,LOOKPOLICY optional
    class SDRJPEG,BASE,PACKAGE,HDRJPEG deliver
    class VERIFY gate
```

紫色是 HDR 分支，蓝色是 SDR。两者只在左侧的灰色共享节点和右侧的封装处相遇——**没有任何
箭头从完成的 SDR 像素指向 HDR 分支**，这正是整个分叉要保证的性质。红色节点是唯一能否决
成品文件的关卡：它重新读回已写出的内容，与文件声称承载的 rendition 比对。

这几个层次是刻意分开的。Tone 层只负责亮度关系和显示动态范围；Color geometry 层负责
色相路径、色度压缩与向白过渡；Capture 层提供事实，但不直接决定口味。这样调整某个环节
时，至少能知道画面为什么发生变化。

Core Image 在图里出现两次，但用途完全不同：`CIRAWFilter` 是可选的 scene decoder，
`CIContext` 则是 HDR 容器写入器。选择 LibRaw 不妨碍使用 Apple gain-map 导出；选择 RAW 9
也不等于直接采用 Apple 原生成片作为 HDR DRT。两种情况下，dngscan 自己的 SDR/HDR
formation 都位于 scene 解码与 JPEG 交付之间。预览 proxy 与 C++/NumPy 分块只改变分辨率或
执行方式，不改变上述顺序；全分辨率导出沿用相同的 plan 语义。

### 架构契约

解码器与 tone core 是两个正交的选择轴。`libraw` / `coreimage` 决定 RAW 怎样变成
scene-linear 像素；`agx` / `gated` / `lum` / `neutral` 决定这些像素怎样进入显示域。
RAW 9 不是第五条 tone curve，`neutral` 也不是另一种 RAW 解码器。

以后修改这套管线时，下面几条应当保持不变：

- 原始 CFA、黑白电平、剪切比例与噪声统计始终由 LibRaw 在解拜耳前读取。只有
  LibRaw 的 scene frame 能携带对应的空间 mask；Core Image 执行了不同几何，只能接收
  聚合证据，不能借用逐像素 mask。
- 两种解码器都交接 scene-linear Rec.2020。负色彩分量和 diffuse white 以上的数值在
  DRT 前都是合法信号；输出色域 fit 发生在 tone 和可选 look 之后，不属于 capture。
- DNG [`BaselineExposure`](https://developer.apple.com/documentation/coreimage/cirawfilter/baselineexposure)
  是文件写入的基线显影补偿。它不是快门/光圈/ISO，不是传感器
  绝对标定，也不是内容自适应自动曝光；显式 `--ev` 调整发生在它之后。
- 场景亮度只编译 tone endpoint 与趾部/肩部；RAW 剪切和输出色域压力只编译颜色
  权限。颜色指标不能移动黑白端点，亮度百分位也不能冒充已经丢失的 CFA 色彩。
- `agx` 配 darktable `base` primaries 是成片默认。`lum`、`neutral` 是受控对照，
  `gated` 是仅限 LibRaw 的 RAW 证据实验。

## 一层：Capture — RAW 证据从哪里来

### 黑白电平与逐通道剪切

dngscan 从 `raw_image_visible` 和 `raw_colors_visible` 读取解拜耳前的 CFA 数据。黑电平
来自 metadata；full-well 则先检查每个通道顶端是否存在可信的饱和堆积，有就使用实测
ceiling，没有才回退到逐通道 metadata white level。它不会拿一个标量替所有 R/G/B，
剪切阈值因此也是一张按 CFA 颜色生成的 threshold map。没有任何通道出现可靠堆积时，
报告会明确把 full-well 标为 metadata fallback，而不是把估计值写成实测值。

这一点会影响的不只是报告里的 clip%。硬 clip%、2×2 cell 指标、高光分类和诊断剪切图
使用同一张逐通道 threshold map。渲染时的**软余量 mask**与它相关，但刻意不完全相同：
每个通道都按扣黑后的 full-well 从 95% 处的 0 平滑渐入到 99% 处的 1，让颜色能在插值形成
硬断层前开始退让。如果绿色比红色更早到满阱，硬统计与软权限图都会保留这个通道差别。

高光重建可以补出连续的亮度和看起来合理的颜色，但它不能重新获得传感器没有记录的信号。
因此剪切证据在重建之前保存，后面重建得再平滑，也不能反过来定义全图的 white endpoint。

### 解拜耳

全分辨率导出的 `auto` 顺序是 DHT → DCB → AHD，具体取当前 rawpy/LibRaw 构建实际支持的
最高优先级算法；X-Trans 等非 Bayer 数据继续走 LibRaw 对应路径。预览使用 half-size
2×2 超像素合并，所以预览适合看曝光、颜色和高光路径，不适合评价最终纹理。

dngscan 不做降噪，因此解拜耳也是主要的纹理选择。DHT 适合低 ISO 的干净信号；重噪声
夜景里，DCB、AAHD、VNG 或 PPG 有时比更激进的细节插值自然。标准 rawpy wheel 不一定包含
AMaZE、LMMSE、VCD、AFD 等 GPL demosaic pack 算法，实际可选项取决于本机 LibRaw 构建。
GUI/CLI 可手动指定 `dht / dcb / ahd / aahd / vng / ppg`；如果本机 LibRaw 还带有其他
算法，把它加入 `DEMOSAIC_CHOICES` 即可交给现有的可用性检测与回退逻辑。

### 白平衡

`camera` 使用文件里的 AsShot 测量，`daylight` 使用 LibRaw 的日光标定乘子。前者跟随拍摄
现场，后者适合让同一光线下的一组照片保持固定配平。固定色温模式（`6500k` D65 显示白点、
`5500k` 摄影日光/日光卷、`3400k`/`3200k` Type A/B 钨丝卷、`9300k` 日本广播传统白点）是
声明的标准参考而非肉眼调整：LibRaw 侧经**标定阶梯**求解——文件自身的 DNG 双光源
标定（ColorMatrix1/2 按倒数色温插值）→ LibRaw 的机型 Adobe 矩阵 → 本项目为
"比安装版 LibRaw 还新"的机型准备的回退矩阵表（`camera_matrices.py`）→ 全部缺失时
**退化为相机 AsShot 并显式警示**，渲染照常（声明的降级可用，静默的降级等于隐藏
白平衡）。RAW 9 侧通过 CIRAWFilter 原生的 neutralTemperature/neutralTint 接口接收
同一声明（tint 钉零，避免 AsShot tint 残留混入声明基准）。两种解码器各用自己的标定
兑现同一个声明；在适马 fp 参考帧上，求解的 6500K 乘数与厂商日光元数据吻合在 0.1%
以内。逐机型支持状态、传感器先验表（PhotonsToPhotos 实测曲线）与 LibRaw 升级路径
见 [SENSOR_SUPPORT.zh-CN.md](SENSOR_SUPPORT.zh-CN.md)。

日光、阴天和阴影大致落在可预测的日光轨迹上，机内测量通常足够有用；混合光、窄谱 LED、
荧光灯和钠灯则不是一个简单的色温问题。还有些看起来像“白平衡不对”的变化，实际来自
tone curve 对亮度与纯度的重新分配，所以 WB 与 DRT 在管线里保持独立。AsShot 相对日光
乘子的偏离也会写入分析结果，它既是白平衡数据，也是拍摄现场光源留下的信息。

显示器前已经适应环境的肉眼不能作为绝对白点测量。Hunt、
Stevens、Abney、Bezold–Brücke 等色貌效应还会让亮度和纯度变化被感知成色相或冷暖变化，
肤色、天空和植物这些记忆色也不是简单的色度学目标。看见“偏色”时，先区分它来自光源、
相机配平，还是 tone/color geometry，通常比直接转动色温更有用。

### 高光处理

LibRaw 的三种选择处理的是重建后的观感：

- `clip` 在饱和处直接截断，最接近传感器实际状态，但逐通道剪切可能留下色边。
- `blend` 在剪切边界混合，让过渡更平缓。
- `reconstruct` 根据幸存通道估算丢失通道，可以恢复连续结构，但色度属于推断。
- 重建色相往往会向幸存通道偏移，因此连续不等于色彩真实。

日常出片通常以 `reconstruct` 为实用默认，检查传感器和算法本身时则更适合用 `clip`。
无论选哪一个，RAW 剪切证据都不会改变。

LibRaw 会把 `blend` 和 `reconstruct` 的 uint16 整幅缩暗，倍数正好是归一化后的最大
白平衡增益，目的是给名义白点以上的重建值留容器码值。dngscan 现在把这段余量记进
`scene_scale`，不再把它当成整张照片的曝光下降。Sigma fp 样张上 `max WB = 2.33`，也就是
1.22 EV；修正后 clip 与 reconstruct 的可靠主体在 0.03 EV 内一致，而 reconstruct 仍保留更多
高光范围。

## 解码器：LibRaw 与可选的 Core Image / RAW 9

RAW 入口现在显式分成两层：`RawEvidence` 先且只由 LibRaw 获取传感器 mosaic、CFA、
黑白电平、白平衡证词与颜色矩阵；随后 scene decoder 才选择 LibRaw 或 Apple RAW 生成
scene-linear RGB。`acquire_raw_evidence(path)` 刻意没有 decoder 参数，因此切换 scene
decoder 不会改变 Evidence 的来源、数值、版本记录或失败条件。`RawBundle` 暂时保留旧的
扁平字段作为兼容外观，同时携带同一份 `evidence` 对象与 provider provenance。

两层各自打开解码句柄，LibRaw scene 的 GainMap / postprocess 不可能回写 Evidence 副本。
Apple 的 `aligned` 模式额外做的 half-size LibRaw RGB 渲染只是 scene scale A/B 对照，
不是 Evidence，也不会替换 `RawEvidence`。因此“Apple scene + LibRaw Evidence”就是默认
组合，而不是研究开关。

`--decoder coreimage` 是另一种 capture decoder，与 tone core 的选择彼此独立；它不是
默认画质升级。解码前，dngscan 会查询当前文件的
`CIRAWFilter.supportedDecoderVersions()`，不会把相机型号名单当作文件必然支持 RAW 9 的
依据。文件只支持 RAW 8/7 时，GUI 会先询问是否使用旧版解码器，CLI 则输出明确警告；显式
指定 `--coreimage-version 9` 会直接拒绝不支持的文件，不会静默降级。解码结果以 signed
RGBA half-float 渲染到 extended-linear Rec.2020。负色彩分量和 diffuse white 以上的值
会原样交给 AgX，不再经过 uint16 百分位缩放。look 类控制项按中性线性交接配置（RAW 9 的
moire 值刻意保留 Apple 更保细节的默认）；高光重建与镜头校正则显式开启。配置遵循 Apple
在 [WWDC21 session 10160](https://developer.apple.com/videos/play/wwdc2021/10160/) 中对线性 RAW 配方与可编辑显影配方的区分。

它是**独立管线，不是 LibRaw 的后端**。Core Image 会执行文件里的 DNG opcode：在
Sigma fp 的 DNG 上是逐平面 `WarpRectilinear` 加一张镜头阴影 `GainMap`。这个畸变校正
把画面角落移动了数十像素（24MP 实测约 70px），所以 LibRaw 的逐像素 CFA 掩码描述的
已经是另一批像素——它们被丢弃而不是重映射，因为沿用会让 clip retreat 作用在错误的
位置上。于是这条路径没有逐像素 CFA 证据：`--tone-core gated` 会被拒绝，clip retreat
不运行，`--highlight-mode` 也不适用（Core Image 有自己的高光重建）。而聚合型 RAW 事实
（黑白电平、剪切百分比、SNR、噪声底、白平衡证词）是分布而非像素位置，依然有效，仍由
LibRaw 提供。Tone plan 会用实测的剪切 cell 比例，从 RAW 9 亮度排序的最高端剔除等量样本。
这是聚合层面的对照启发式，不表示某个 RAW 9 像素能对应到某个 CFA site：重建高光仍可描述
尾部拓扑，但不能反过来定义全局白点。报告会写明解码器、版本，以及被执行的 opcode。

HDR 分支按同一套证据规则接入这条路径，并有专门测试钉住：色度自由度 `rho` 被压在 0.25
上限（没有逐像素 CFA 掩码可以在局部撤回它）、无掩码 formation 的渲染保持在 `[0, peak]`
体积内、rank-trim 后的 RAW 9 可靠尾部与 LibRaw 的 CFA 掩码测量在同一帧上对齐（日景参考
帧实测相差 0.09 EV，门限 0.3 EV）。两条解码线的成像差异属于相机诠释取向，不是 HDR
预算的泄漏。

反方向的暗场对等性：LibRaw 路径现在同样兑现 DNG 暗场 opcode——解拜耳前施加
`GainMap`（fp）、渲染后施加 `FixVignetteRadial`（iPhone ProRAW），证据副本取自
校正前的传感器真值。iPhone 主摄双解码对照（同一帧、同一 AgX plan）：

![iPhone 16 Pro 同帧双解码：LibRaw 施加 DNG GainMap 与 RAW 9 的 FixVignetteRadial，角部亮度一致](assets/decoder-iphone-libraw-vs-raw9.jpg)

![LibRaw 与 Apple RAW 9 走同一条 AgX plan：差异是相机诠释，不是管线漂移](assets/decoder-libraw-vs-raw9.jpg)

Core Image 与 LibRaw 并没有暴露同一个 scene unit，单一固定补偿也无法跨相机、跨场景成立。
所以默认改为 `--coreimage-scale aligned`：dngscan 会对同一文件快速做一次 half-size LibRaw
重建，再用两种解码结果的绿色通道中位比，对 RAW 9 整幅乘一个标量。以前解释里引入的 RAW
green 项会在分子分母中严格约掉；这里得到的是逐文件解码器 A/B 标尺，不是传感器绝对标定。
它不会把中位数拉到 18% 灰，也不改变画面内部的光比，但解码器色彩、几何和重建都会影响
这个统计量；它与上文的 Evidence 获取是两个独立调用和数据契约。

`--coreimage-scale unity` 会跳过该比较，保留 Core Image 原生单位；`measured` 只应用旧的
Sigma fp 固定 `1/1.0293` 倍率，用来复现早期 A/B。三个模式现在在效果上互斥，固定倍率不会
再被后续逐文件对齐抵消。

这类对比里有两种亮度口径，**不能互相引用**。**可靠主体**中位是 scene-linear 的，量在色调
曲线之前，且已剔除 RAW 剪切样本；**最终输出**中位量在渲染完成的图像上，此时 AgX 已经把
两端都压过。同一对解码器在 `_SDI0150` 上，前者相差 +0.123 EV，后者只有 +0.02~0.03 EV
——色调曲线吸收掉了 scene-linear 偏移的大部分，两个数字差了约 5 倍。它们都是正确答案，
只是回答的不是同一个问题；引用时必须写明是哪一个。

除对齐之外，差别主要来自相机解释本身——色彩分离、噪声重建和高光走向。另有三点行为差异
来自解码器本身而非口味：

- **亮度参考按钮在两条管线上可能给出不同 EV。** 它从当前解码器且经过所选
  scene transform 的结果里读取可靠主体中位，不再拿 LibRaw CFA 直方图代替亮度。然后用同一份
  已编译 plan 搜索最终输出的高光安全上限。这让按钮可以跨解码器工作，却不会把 EV 0 变成
  隐式自动曝光。要比较解码器本身，仍应固定 `--ev`。
- **Apple 缓冲保留 diffuse white 以上的镜面值，但重建结果不等于传感器测量。** 完整的
  RAW 9 尾部仍用来区分大面积高光与点状灯源；全局白点只读取减去全分辨率 CFA 剪切比例之后
  的可靠排序。这样既保留 Apple 的平滑重建，也不让它虚构已经丢失的传感器余量。
- **固定 `--ev` 依旧不能完全隔离解码器差异。** 两个缓冲可能编译出略有差别的 plan，Core
  Image 还执行了不同的几何。`tools/decode_ab.py` 会让每个缓冲分别走过两套 plan，把解码与
  plan 的影响拆开。当前 SD 卡抽样里，ISO 3200 的输出中位只差 +0.006 EV，明亮 ISO 100
  样张差 -0.020 EV；近乎全黑的 ISO 25600 样张则差 -0.413 EV，而且几乎全部来自 RAW 9
  解码本身。这也是它仍作为对照路径而非静默替换 LibRaw 的原因。

**RAW 9 的降噪来自架构本身。** Apple 把它描述为一个把解拜耳与降噪融合在一起的分块
CoreML 模型（[WWDC26 session 305](https://developer.apple.com/videos/play/wwdc2026/305/)），所以不存在"未处理模式"可以索取：重建本身就是解码器。
也因此 `luminanceNoiseReductionAmount` 为 0 **并不等于"不降噪"**——它只是在一个始终运行
的模型上选中了标定范围里最不平滑的一端。

dngscan 仍会清零暴露出来的 look 类控制项，包括 `sharpnessAmount`——它在版本 8 上无效、
版本 9 上生效，默认值随文件与版本而变（见过 0.485 和 0.954）。在全分辨率下逐项对着
Apple 默认值实测，版本 9 上真正起作用的只有三项，而当前配置已经处在 API 所能达到的
**最锐一端**：

| 控制项 | 相对本文所用设置的变化 |
| --- | --- |
| `colorNoiseReductionAmount`、`detailAmount` | 无——0/0.5/1.0 全程改变 0.00% 像素 |
| `sharpnessAmount` 取 Apple 默认 | 高频能量 +5.9% |
| `luminanceNoiseReductionAmount` 取默认 0.043 | −3.1%；取 1.0 则 −59.8% |
| `moireReductionAmount` 强制为 0 | −59.8% |

其中两行值得重读。第一行**订正了本文此前的一个论断**——先前写的是"这三者表现得像同一个
内部控制的别名，0.5 的默认值会改变 93.6% 的像素"；重测后不成立，在这两项上 Apple 的文档
是对的。而 `moireReductionAmount` 是**有意保留** Apple 的 0.55 而非清零：它的零点是这个
控制**最平滑**的一端而不是"关闭"，强行清零付出的细节代价与满强度亮度降噪相当。于是唯一
还能拿到的只剩 `sharpnessAmount`，而那是空间锐化，不该出现在 scene-referred 缓冲里。

所以 RAW 9 渲染里残留的柔化来自模型本身，不是某个没关掉的开关。没有可以再关的东西了。

即便如此，残余差异依然很大，而且差多少取决于场景。以画面最暗 30% 区域内 8×8 块局部标准差
的中位数为度量，两条路径都固定 `--ev 0`：

| 片子 | ISO | 亮度噪声 / LibRaw | 色度噪声 / LibRaw | 纯黑像素 |
| --- | --- | --- | --- | --- |
| 演出，近乎全黑 | 25600 | 15% | 15% | 9.5% 对 8.1% |
| 园林，阴天日光 | 12800 | 78% | 28% | 1.5% 对 1.4% |

色噪的清理是稳定的，亮噪不是。模型的优势主要来自信噪比真正糟糕的地方；在曝光正常的
片子上，最暗的 30% 只是**影调**暗而并不缺信号，两条路径于是几乎收敛。阴影并没有付出
代价：`shadowBias` 清零后（见下），纯黑像素比例与 LibRaw 路径相差约一个百分点。把
LibRaw 换成更平滑的解拜耳（VNG、PPG）并不能缩小差距，所以这是模型本身而非插值选择。
这一点与本工具"不做降噪、把纹理选择留给解拜耳"的立场需要各自权衡。

**解码严格采用 Apple 的线性提取结构。** `baselineExposure`、`shadowBias`、`boostAmount`、
`localToneMapAmount` 和 RAW `exposure` 在 CIRAWFilter 内部清零，EDR 与 gamut mapping 关闭，
结果渲染到 `extendedLinearITUR_2020`。文件原本的 `baselineExposure` 会在清零前记录，再像
LibRaw 路径一样通过 `scene_scale` 恢复一次。这样交接像素本身保持直接 scene-linear，文件的
显影意图也没有丢失，更不会和用户 EV 重复。

**aligned 是逐文件的实用解码器对照。** half-size LibRaw 参考使用与主 LibRaw 路径相同的
白平衡、高光重建与存储尺度契约；它的解码绿色中位除以 RAW 9 的解码绿色中位，得到整幅使用
的单一标量。旧解释中的 RAW mosaic 归一项在分子分母里完全相同，数学上会约掉，因此把结果
称作 raw→scene 传感器增益并不正确。

当前样张中 half-size 因子与全分辨率约在 0.02 EV 内。报告会写明因子；统计无效或超出可信
范围时会写明失败并回退为 1×。这一步不对齐几何，也不对齐 tone plan：两个缓冲仍各自编译
端点，RAW 9 也保留自己的重建、色彩分离与噪声行为。要检查这些原生尺度差异，应使用
`unity`。

这个标量比自动曝光窄得多：它没有外部亮度目标，也不会重排照片内部的光比，夜景仍然是夜景。
但两种解码器的颜色和几何并不完全一致，所以统计仍可能受内容影响；因此这里把它称作 A/B
标尺，而不是物理标定。

还有一处残余的不对称，看预览前值得知道。Core Image 的预览解码到 1280px 代理，而 LibRaw
的预览做 2×2 超像素合并，两者看到的噪声量不同，因此各自编译出的黑端可能与自己的导出略有
出入。三张片子实测，Core Image 的预览→导出黑端位移是 −0.01、−0.22、+0.01 EV，LibRaw
一侧是 −0.05、+0.02、−0.04 EV：除了噪声大的那张之外量级相当，而在那张上，代理把噪声
平均掉之后读出的阴影比导出实际会给的更干净。

**BaselineExposure 在两条管线上都被遵从。** Apple 明确把它定义为 RAW 文件请求的 baseline
exposure，默认值可以随相机设置变化；ProRAW 还会随场景动态范围写入逐图配方。它不是快门/
光圈/ISO 所描述的物理拍摄曝光，也不是要求把画面归一到某个中位亮度。LibRaw 不应用该标签，
所以 dngscan 在两条路径都把 gain 折进 `scene_scale`：Core Image 先读取并清零该属性，再在
线性交接后恢复，LibRaw 则直接从 DNG metadata 恢复。改变尺度而不放大存储缓冲，可以保留
高于名义白点的码值与精度。主观微调仍由 `--ev` 完成，报告会写出文件值及其应用位置。

`shadowBias` 是最容易漏掉的一项：默认值 **5.0**，作用是从阴影中减去一个量，本质是
display-referred 的黑电平基座，在 scene-linear 缓冲里没有立足之地。保留默认值会让
ISO 12800 那张有 1.4%、ISO 25600 那张有 21.0% 的分量被压到恰好为零——清零后分别是
0.006% 和 0.18%，**都低于** LibRaw 路径自身的 0.16% 和 1.9%——并且会把亮度的第 1 百分位
推成负值。看上去像被 CoreML 降噪吃掉的暗部细节，大部分其实是这个减法。

**清零控制项的边界在“重建”处。** 除此之外，“把控制项全部清零”是从 LibRaw 路径继承来的
规矩，把它不加分辨地套到一个假设完全不同的解码器上，得到的不是更纯净的解码，而是更错误
的解码。重建类控制会被显式设定，避免系统默认值变化悄悄改变数据契约：

- **`highlightRecoveryEnabled` 显式开启。** 它重建的是被剪切的通道，干的
  是 LibRaw 那边 `--highlight-mode reconstruct` 同样的活，不是 look 控制。关掉它，被剪
  高光返回时绿通道被钉在远低于红蓝的位置——近白均值 R 1.933 / G 0.681 / B 1.816，绿为
  最大通道的占比 0%——渲染出来就是品红色的高光核心和粉色光晕，导出 JPEG 上实测品红偏移
  +0.077，而 LibRaw 在同处是 −0.000。开启后同一批像素均值为 1.981 / 1.980 / 1.980，
  偏移降到 +0.002，同时镜面余量完好（p99.995 2.08、max 2.22）。这比 LibRaw 路径的
  "剪到同一白点"严格更好——后者的中性高光是靠丢掉滚降换来的。
- **`lensCorrectionEnabled` 显式开启。** RAW 9 与文件中的 DNG opcode 共同组成相机标定
  解码；这也正是 LibRaw 的逐像素掩码不能沿用的原因。
- **`gamutMappingEnabled` 保持关闭**。它是输出端钳位，位置应在视图变换之后。开启后
  p99.995 从 2.08 压到 1.07，所有负分量（Rec.2020 之外的真实场景色）被清零，14% 的像素
  发生变化。AgX 在下游有自己的色域处理，因此这一级的交接保持 scene-referred。

像素交接尽量直接照 Apple 的示例：`RGBAh`、extended-linear Rec.2020、signed half-float，
没有百分位归一化，也没有 unsigned clamp。预览通过 `CIRAWFilter.scaleFactor` 直接请求长边
1280px，而不是先解出约 6MP 再缩小；交互 `CIContext` 复用并启用
`cacheIntermediates=true`，全分辨率导出使用另一套复用 context，关闭中间缓存并给出 1024MB
memory target。一张 24MP Sigma fp 的实测中，RAW 9 解码在 1280px 为 1.24s、6000x4000 为
2.08s；完整全尺寸 decode + analyze + plan + render 在 JPEG 编码前约 5.1s。

`extendedDynamicRangeAmount` 被显式设为 0，避免 Apple 的显示侧 HDR 映射先于 AgX 进入
scene 缓冲。调到 1.0 确实在最顶端拉出更多分离度
（顶部像素极差 0.11 → 1.74），但高光区与默认渲染在 log2 上的相关系数是 0.996，说明基本
是同一批信息的重映射，而且会把峰值推到 20，远超这条管线预留的余量。

**选哪个解码版本，以及哪个变体。** 全新初始化的 filter 报告的是版本 8 而非 9，因此即使
文件支持 RAW 9 也必须显式 opt-in；macOS 27 上 `supportedCameraModels` 列出 921 个机型，
其中包含 Sigma fp。版本列表里还提供 `.dng` 变体（`9.dng` 与 `9` 并存），二者是真正不同的
解码，99.96% 的像素有差异。以 LibRaw 依据文件自带矩阵得到的色彩为基准，`9` 的色度距离是
0.015，`9.dng` 是 0.041 且明显偏蓝，因此本管线请求的是不带后缀的 `9`。

**需要留住的可调接口。** RAW 9 是计算导向的解码器，它的若干旋钮是对模型的**标定控制**，
而不是可以关掉的处理级。`exposure` 已接线。白平衡接口（`neutralTemperature` /
`neutralTint` / `neutralChromaticity` / `neutralLocation`）是移动白平衡的受支持途径，
也正是 `--wb daylight` 在这条路径上被拒绝而非近似的原因。`linearSpaceFilter` 是 Apple
自己提供的钩子，用于在图像仍处于线性状态时插入一个 CIFilter——任何想下沉进解码阶段的
scene-referred 操作，架构上都该放在这里。在那套白平衡接口的温度/色调映射得到验证之前，
`--wb daylight` 在这条路径上会被直接拒绝，而不是拿近似值糊弄过去。

RAW 9 随系统分发，一次 macOS 更新就可能换掉模型，而 `decoderVersion` 仍然回答 "9"。
现在报告会同时记录系统版本/build fingerprint，至少能把两次输出追溯到具体运行环境。金样本
回归仍只覆盖预解码后的稳定算法层；Core Image 解码测试断言性质而非固定字节，系统升级后仍需
用同一组 RAW 做显式 A/B，不能把相同的版本号当作相同模型。

## 二层：Tone — 曝光和曲线怎样确定

### 固定曝光锚点

整条管线以 scene-linear `0.18` 作为名义中灰。当前显影锚点是统一固定标量
`0.18 * 2^3` 再叠加手动 EV；它还不是逐机型标定，也不会把每张照片的中位数自动变成
18% 灰。文件存在 DNG `BaselineExposure` 时，会更早按文件显影配方遵从。常数缩放不会
破坏场景意图：暗场景进 AgX 前依然暗，明亮场景依然亮。

GUI 里的“亮度参考”是一个主动调用的对照读数，也就是 CLI 的 `--ev auto`。它会尝试把
可靠主体中位放到 18% 灰，并受新增高光剪切预算约束。这个中位来自当前解码器、当前
scene transform 之后的可靠 scene body，RAW 已剪切样本不会定义它。高光搜索会在同一份固定
plan 上尝试候选 EV，不会边测边改变目标。全图统计仍可能被占比很大的背景误导，所以它只是参考，
不是默认曝光。

### 场景统计不是简单 min/max

Tone plan 会把可靠主体和高光尾部分开。LibRaw 路径剔除空间 CFA clip mask 对应的样本；
Core Image 路径使用前文的聚合 rank trim。SNR 会约束黑端和 gated 颜色权限，但不是另一张
主体 mask。尾部只负责给肩部留出空间。点状灯源与大面积明亮表面也不是同一种高光：
前者可以进入 roll-off，后者如果被同样压到顶端，会让整张图显得又暗又刺眼。

因此 tone plan 里的几件事分别有自己的依据：

- `black point` 与 `toe` 参考噪声底、暗部可用范围和目标黑场。
- `white point` 与 `shoulder` 参考可靠亮度尾部、显示余量和发光体拓扑。
- `pivot` 当前固定在校准 EV 0，`contrast` 固定为 3.0；主体统计不会自动移动它们，GUI
  中的有限调整是明确的人为偏置。
- `view brightness` 只抬曲线内部，保持真黑与目标白端点，用于干净但整体偏暗的场景。

### GUI 中的四个明暗微调

GUI 不直接暴露校准支点或编译后的 black/white EV，而是在 tone plan 上提供四个有限
偏置。四个滑块的`自动`中心值就是分析结果，不是另一套 preset；全部归零时直接沿用原来的
render plan，输出不变。

| 选项 | 向左 | 向右 | 不会改变什么 |
| --- | --- | --- | --- |
| `中间调亮度` | 主体更沉、更暗 | 提亮主体和可见暗部 | 不移动 scene exposure、黑点或白点 |
| `中间调对比` | 中间调更柔和 | 拉开校准支点两侧的明暗距离 | 不移动支点本身 |
| `暗部过渡` | 趾部更深，更快沉入黑场 | 趾部更开放，阴影层次更容易看见 | 不移动黑点，也不会创造低 SNR 信号 |
| `高光过渡` | 肩部更直接，高光更有冲击力 | 肩部更柔和，更早保留亮部层次 | 不移动白点或 RAW 剪切位置 |

`中间调亮度`和曝光 EV 最容易混淆。曝光 EV 在 scene-linear 域缩放信号，会改变进入
肩部的位置并消耗高光余量；中间调亮度是显示侧的内部曲线调整，真黑和目标白保持不动。
`中间调对比`也不是另一个亮度控制：它围绕校准支点改变斜率，决定主体内部的明暗距离，
而不是把主体整体上下移动。

实际使用时，先用`中间调亮度`确定主体明暗，再用`中间调对比`确定立体感，最后分别调整
`暗部过渡`和`高光过渡`。打开暗部只能展示已经记录到的内容；低 SNR 场景开得过多，也会把
读出噪声和色噪一起带出来。`高光褪白`不属于这四个亮度控制，它只处理接近显示白的色度路径。

### 端点模式：自适应与证据界

EV 之所以感觉像“数字亮度”，是因为默认端点追随场景百分位、曲线随场景平移。
`endpoint_mode` 补上“界”这一轴：**adaptive**（默认）保持现状——黑端点参考主体 p1
与噪声下界，白端点参考可靠尾部加边距；**evidence** 把端点钉在传感器证据上——黑端点
= 实测噪声底 EV（固定曝光锚下 RAW 满阱位于 +`MIDGRAY_HEADROOM_STOPS` EV，噪声底占
满阱的比例 `f` 对应 `MIDGRAY_HEADROOM_STOPS + log2(f)`；有传感器先验时用先验读出
噪声的电子域口径，无先验时用单帧 tile-σ 估计并注记降级），白端点只允许可靠 RAW 尾部
（保留与自适应相同的边距与最低白点地板，防止肩部塌到主体上；重建尾部永远无权定义
白端点，证据缺席时如实回退并注记）。两种模式下校准支点都不动：EV 0 仍映射 18%，
端点变宽只是把曲线两端伸向证据边界，重编后的参数照常经过 C1 求解器的合法性钳制。

配套的两个有界偏移作用在编译后的 plan 上：`toe_end_offset` 移动曲线落到近黑
（display-linear 0.002）的 EV 位置，通过重解 toe power 实现——刻意不用 latitude 下移，
因为把线性 latitude 段向下延伸，会用中段直线替换本来抬起的趾部 sigmoid，实测反而压暗
深阴影；`shoulder_start_offset` 则是真正的 latitude 上移，让线性中段更晚交给肩部。
两者都不移动黑白端点与支点，编译后的实际值（含被钳制的请求）通过
`drt.compiled_curve_transitions` 回报给 GUI 与报告。

### 肩部自由度的几何：为什么有些照片的肩部收白纹丝不动

（本节承接《修图工作流说明书》"为什么样张一的肩部收白纹丝不动"的白话结论，
给出完整推导。）收白点定义为曲线升到"近白参考"——黑地板到白点跨度的 90%
——的场景 EV。肩部从校准支点出发能支配的显示空间等于中段斜率的投影升幅
减去实际需要的升幅：

```
contrast × white_ev / 16.5 − (1 − 0.18^(1/2.2))
```

对比度 3 时这个量在白点 ≈ 2.98 EV 处恰好归零。防御性白点下限把逆光桥样张
的白点钉在 3.0 EV（侧光人像样张 3.01），肩部因此被迫编译成一条贴着切线的
直线，合法曲线族里没有任何成员能把它掰弯：`shoulder_start_offset` 从 −1
拉到 +2，编译后的收白点仅从 2.726 移到 2.739 EV——0.013 EV，任何眼睛都
看不见。白点被真实高光推高时同一滑块立即恢复行程：灯盘烧毁样张的白点在
+5.15 EV，收白点从 +2.8 EV（−1）走到 +3.7（0）再到 +4.5（+2），共 1.7 EV
的真实行程。明暗卡实测行显示的正是这里的编译后收白点
（`drt.compiled_curve_transitions`）——数字不动是曲线几何无自由度的如实
回报，不是钳制故障。

### darktable 风格的 C1 曲线

现在的主曲线沿用 darktable AgX 的 C1 分段构造：趾部、线性 latitude 和肩部在连接点
同时保持数值与一阶导数连续。black/white EV、contrast、趾部/肩部 power 和 latitude
由 tone plan 提供，但 EV 0 到 18% 的校准锚点保持稳定。

采用这条结构，是因为只把场景 min/max 塞进一条普通 sigmoid 很容易让少数灯源定义
white EV，结果就是高光很刺眼而主体仍然偏暗。C1 端点和主体/尾部分离，让“场景有多宽”
与“主要内容应该落在哪里”成为两件不同的事。

## 三层：Color geometry — AgX 真正改变了什么

裸的逐通道 S 曲线会让 R/G/B 以不同速度进入趾部和肩部，高纯度颜色的色相因此会
随亮度漂移。AgX 不只是一条 sigmoid；它的关键是曲线前后的原色几何。

曲线前的 `inset` 把工作原色向中性轴收缩并做小幅旋转，避免极纯颜色直接撞上单通道上限，
给饱和高光留出平滑的 path-to-white。曲线后的 `outset` 再恢复纯度，但它刻意不是 inset
的严格逆矩阵。两者之间的差异，以及可选的 hue restore，共同构成 AgX 的颜色性格。
负片预设拨动色头时，outset 之后还施加一个**色头对角增益场**（Bradford-LMS
基，按场景亮度 EV 查表；见胶片观察一节）——放在 AgX 色彩几何之后是刻意的：
hue restore 约束的是通用曲线自身的色相摆动，而色头场是介质实测的报告内容，
不该被渲染器的色相纪律稀释。
这也在预先处理裸逐通道曲线的 notorious six：例如纯红随亮度走向橙黄、纯蓝走向 cyan；
inset 的小幅旋转同时承担一部分 Abney 式感知色相补偿。

dngscan 把这部分数学锁定到 darktable commit
`cf5e698c1a5afac52de785c3bf63fcbcb71707d3`。该版本 scene-referred 默认使用 `base`
几何和 0.6 hue restore，所以 dngscan 也以它为默认。矩阵构造按 darktable 的转置存储顺序
和 D50 ICC 连接空间复现；直接使用未适配的 D65 Rec.2020 坐标或把矩阵乘法顺序反过来，
都会改变色彩路径，甚至破坏中性轴。`smooth`、`punchy` 和 `muted` 保留为明确的几何对照，
不参与 RAW 分析，也不改变曝光。

hue restore 是**逐预设**的值而不是一个全局默认：编译器给 `base`、`punchy`、`muted` 写
0.6，给 `smooth` 写 0.0——后者的类 sigmoid 几何在上游就不需要恢复。能决定这个数字的地方
有三处：`ToneCompressionPlan` 的 dataclass 默认、编译器里的逐预设写入，以及只有重命名之前
的旧 plan 对象才会读到的 `AGX_HUE_RESTORE` 兜底，因此测试把三层全部钉住。只断言那个常量
是没有意义的：改掉它，全部 golden 渲染逐字节不变。

AgX 的代价也来自同一个结构。inset 在曲线前先降低纯度，而这份纯度主要由落入趾部的内容
通过逐通道扩张赚回来，因此高 ISO 夜景有时反而显得很浓，明亮宽 DR 日景却容易偏平。
Blender 生态常把 Base 与 Punchy look 配套使用，本质上也是在处理这件事。另一个代价是
色度与内容在曲线上的位置耦合：同一个物体换一个构图或曝光，落入不同曲线区间后可能得到
不同纯度。`punch`、`gated` 和 `lum` 都是为了把这些影响拆开观察，而不是否定 AgX。

### 四条压缩核心

四条核心共用同一个曝光锚点和交付端保护，方便在相同 EV 下拆开比较。它们并不都能拿到
相同的空间 CFA 证据：mask 只存在于 LibRaw 路径，各核心对现有证据的使用方式也不同。

| 核心 | 底层差别 |
| --- | --- |
| `agx` | 完整的 inset → 逐通道 C1 curve → hue path → outset。默认成片路径。 |
| `gated` | 仅限 LibRaw 的实验：同时计算 AgX 色彩与亮度保持结果，由 RAW 剪切、余量和噪声置信度逐像素混合。 |
| `lum` | 同一条场景编译的 C1 曲线只作用于亮度 norm，RGB 比例保持，不进入 AgX inset/outset。 |
| `neutral` | 固定 Y 比例诊断曲线，不使用场景编译 endpoint 或 AgX 几何；不是成片基线。 |

`gated` 不是另一条曝光曲线。它先把 AgX 候选归一到与 lum 候选相同的 Rec.2020 亮度，再
决定混入多少色度路径，所以亮度只有一个权威，mask 边界不会产生明暗接缝。它利用的是
darktable 模块本身看不到的 CFA 信息：某个颜色变化究竟来自有效通道，还是发生在已经
剪切并被重建的区域。

`lum` 则是刻意保留 RGB 比例。它能保住中频颜色纯度，但亮而饱和的颜色也更容易出现霓虹感，
因为颜色不会像 AgX 那样主动向白退让。`y`、`max` 和 `power` norm 分别在色度学亮度、最响
通道保护和两者折中之间选择。

`neutral` 连 tone window 也固定，因此适合单独检查场景 plan 与 AgX 几何分别带来了什么。
但保持 RGB 比例会把窄带高饱和高光直接推向 sRGB/P3 边界。最终 gamut fit 能保证输出合法，
却不会让它获得 AgX 那种逐渐向白过渡的路径。

### RAW clip retreat、punch 与 gamut fit

RAW headroom retreat 只在解拜耳前 CFA 表明通道接近或到达 full-well 时工作。95% 到 99%
的软渐变是保守权限信号：低端表示“开始不可靠”，不表示“已经剪切”。它在曲线前把颜色向
该亮度下的中性轴收回。它与 AgX 的全局 inset 不同：一个由传感器余量驱动，一个是显示变换
本身的颜色几何。

`punch`（GUI 中的`中频纯度`）用来补偿 AgX inset 在明亮宽动态场景里的整体去纯度。
它在 Oklab 中工作，自动值由主体亮度、可用 DR 和 tone window 共同门控；在中性轴、深影、
亮部、已经很浓的颜色和肤色
区域分别衰减。所有权重都乘在增益的增量上，因此 gain 始终 ≥ 1：它只补纯度，不会在某个
区域反向去饱和。夜景或高 ISO 场景可以精确归零并短路算子，避免放大暗部色噪声。GUI 强度
只是分析值的倍率，`1` 使用自动值，`0` 完全关闭。这仍然是基于有限样张调出的全局策略，
不是传感器测量本身。

`高光褪白`是另一层很轻的显示侧色度偏置。它不改亮度肩部，也不冒充 RAW 高光重建；
向右让接近显示白的颜色更早收向中性轴，向左则在最终 gamut fit 的保护下保留更多高光色度。

最后的 gamut fit 发生在 tone 和风格之后。它把无法装进目标 sRGB/P3 的颜色沿 Oklab 色度
方向压回边界，而不是简单逐通道 clip。这样 AgX 或 P3 保下来的高光颜色不会在最后一步突然
崩成硬原色。

## 胶片观察位置 — 跨四层的声明功能群

胶片模拟在这条管线里不是一张 LUT，而是**五个独立声明**，各自落在它物理上正确的层。
规范条款（验收门、边界、翻译规则）在
[设计合同](FILM_OBSERVATION_PLAN.zh-CN.md)；本节描述已建成的形态。

```mermaid
flowchart LR
    D1["WB 声明<br/>日光卷 5500K / 钨丝卷 3200K<br/>（一层 Capture）"]
    D2["镜前滤镜（可选）<br/>Wratten mired 位移<br/>（一层 Capture 光学）"]
    D3["光谱分离<br/>胶片感色层作为观察者<br/>（前馈层）"]
    D4["显影曲线 + 观看条件翻译<br/>AgX 具名坐标 · surround 项 · 介质地板<br/>（二层 Tone）"]
    D5["曝光依赖色彩<br/>observe: 后置色头(负片) · full: 烘焙光谱链 LUT<br/>（三层 formation 之后）"]
    D1 --> D2 --> D3 --> D4 --> D5
    D5 --> OUT["四层交付原样工作<br/>含 Ultra HDR gain map"]
```

### 五个声明

- **WB 声明**（一层）：固定色温是标准引用不是肉眼调整；求解走 Capture 一节描述的
  标定阶梯，两个解码器各用自己的标定兑现同一声明。
- **镜前滤镜**（一层，可选）：按柯达出版的 mired 位移构造（85B/85/80A/81A/82A），
  Bradford 对称锚对保证等值反号滤镜严格互逆。滤镜的第一语义是**把光源搬到胶片的
  校准点**（配对使用时中性轴在胶片参考系里不动）；单独使用是声明的创作性退化。
  作用于 scene-linear、前馈之前——可靠尾部与 HDR 预算都透过滤镜测量，胶片也是
  这样测光的。没有强度滑杆：玻璃没有半片。
- **光谱分离**（前馈层）：把胶片当作"另一台相机"喂给现有前馈标定器——数据手册的
  感色层曲线就是它的 SSF，分材料窗口矩阵、中性轴保持、置信度加权、von Kries 窗口
  搬运全部沿用。
- **显影曲线 + 观看条件翻译**（二层）：端到端目标由**光谱印相**构造——负片染料
  堆栈的透过谱 × 相纸自己的光谱感度 × TH-KG3 放大机（3400K 黑体 × Schott KG3
  热吸收玻璃）→ 相纸逐染料显影 → 印片光谱在声明观看光源下做相对比色——再
  最小二乘解到 AgX 参数空间的一个具名坐标；预设激活时整卷一致、场景自适应
  关闭，EV0→0.18 由全局曝光锚（测光表语义）保证，介质地板经
  `target_black_linear` 进入曲线。**标定零杂光**（介质原生语义，schema v4）：
  观看房间的闪光不进胶片曲线（保留常量留给未来的观看模拟层），黑位是介质
  自己的 Dmax；暗环境介质（反转片、Vision3→2383 放映链）统一按经典 surround
  常数（dark 1.5 / dim 1.2 / average 1.0，Bartleson-Breneman，Fairchild 2013）
  折算到明环境交付，v4 以 `black_policy` 诚实命名并同时发布翻译后与原生双地板；
  想要"影院拷贝原样上监视器"的是 `*_theatrical` 引用变体——合同区分**翻译**
  与**引用原文**，引用不占用胶片名，也不带观看翻译。
- **曝光依赖色彩**：observe 模式 = 负片的联合 Y×M 色头场（Bradford-LMS 对角
  增益，outset 后施加）；full 模式 = 离线烘焙的光谱链 65³ LUT 整体接管。两者
  的测量出处相同（同一光谱链），中性轴上恒为中性由构造保证。

**双模式分工**（`--film-mode`，合同 §光谱印相·双模式）：**observe（默认）**=
胶片声明观察者看见了什么（WB/分离/音调签名），颜色由 AgX 显影——分工画在
数据可信边界上，胶片预设在此模式下只是曲线参数，原生内核照常加速；
**full（实验）**= 胶片显影模型整体接管（`film_develop` 核，film v2 因式
分解链：Stage A 解析——观察者逆矩阵→三条 1D 特性曲线→染料密度，含胶片
曝光态、editorial 显影配方与 halation/颗粒挂点；Stage B 因式分解——B1
密度→相纸层曝光 65³、τ(E) 印相 timing 表、相纸 1D 显影曲线、B2 正介质
密度→观看 65³（按介质×观看条件键控、跨卷复用）；无 AgX 色彩几何，要求
AgX tone core；色头仅在 timing=custom 下以 modelled Δτ 解锁），AgX 只保留
交付端色域安全。SDR 之上，Ultra HDR 以"胶片印相 + scene HDR 扩展"参与
（SDR base=胶片印相逐字节不变，参考白之上按场景高光 C1 增益，封顶可靠
余量）。资产为 `dngscan/data/film_v2/` 三分家族（stock / print_state /
b2，schema 5，全部 fail-closed 加载）。关闭 `--film` 即纯 AgX。逐通道滚降本身就是颜色操作——这就是"AgX 只管拉伸滚降、颜色全给胶片"
无法折中切分、必须做成两极开关的结构原因。

### 曲线预设、色头与层间漂移的实现细节

（本节收纳《胶片调整说明书》只保留白话结论的三组技术事实。）

**曲线预设与端点模式的交互**：预设坐标由数据手册特性曲线到 AgX 参数空间的
最小二乘解给出（Velvia 100（v4 黑位语义:介质经声明环绕翻译的地板,原生
Dmax 并记录于 medium_floor_native_linear）：black −4.05 / white +7.59
——介质原生黑位下反转片不再顶满拟合域上界；此前 white 顶格 +8.5 是 v2 把
0.5% 投影杂光烘进目标后的副作用。theatrical 引用系仍可能顶界，属曲线族的
顶格表达，`fit.pinned` 字段逐项记录；残差见"预设库与出处"一节）。
预设激活时端点就是坐标本身，`endpoint_mode` 被静默改回自适应语义——实测
"证据界 + Velvia 曲线"与"自适应 + Velvia 曲线"逐字节相同，GUI/CLI 传入
evidence 不报错也不生效。趾/肩偏移仍在预设坐标上重解曲线形状（抬黑地板
相对化处理，预设声明的相纸 Dmax 地板保留）：后台样张实测 Velvia 曲线上
`toe_end_offset` −2 移动约 39% 的像素，而 `shoulder_start_offset` +2 只移动
约 4%——最硬的肩留给"更柔"的空间本来就小，与上文"肩部自由度的几何"同构。

**放大机色头的光谱印相实现（v4 场，SHA-256 内容钉扎）**：负片预设的 Y/M 滑杆不是平坦 RGB 乘数。
响应由该预设的拟合光谱印相模型推导——构建期把 CC 分色滤镜放进 TH-KG3
放大机光路重曝相纸、单次公共重定时恢复观看中灰 0.18（暗房试条约定），
场以 Rec.2020 基底发布。运行时在 **AgX outset 之后、输出 gamut fit 之前**
的 Rec.2020 域施加——对角增益不与基变换交换，所以增益必须回到它被标定的
基底；查表曝光取色调映射前 scene-linear 的单一亮度轴 EV_Y=log2(Y/0.18)
（显示值已过肩部，不再对应标定曝光；场也只沿中性轴测得——这是诚实的
中性轴推广，不是三层乳剂曝光的假想）。两档都在 0 时输出与不启用逐字节
相同；反转片物理上无印相环节，预设档案将其分类为无色头，CLI 传入直接
报错。第三阶段已落地**联合 Y×M 场**：全部 41×41 个真实档位逐一联合求解、
单次重定时（乘性可分离近似被联合重定时以 ~0.35 档 RMS 否证后废除），场以
Bradford-LMS 对角增益发布（极端滤镜下 Rec.2020 分量可能过零，LMS 保持数值
稳定），随包数据 `dngscan/data/color_head/*.npz`（float16，EV 轴为**自适应共享节点**
——在链条全分辨率上贪心插点直到全场线性插值误差低于 0.015 档，各卷实际
51–73 点；随机检片位 oracle 复测全 21 场最大 0.012 档 < 0.02 验收门。固定
128 点轴曾在随机抽查中量得 0.024 档，过线与否取决于抽签运气，故废除）；档位直接索引零 CC 插值，运行时
仅插值 EV。命名如实：这是**联合纸层曝光模型**——数据里尚无 Y/M 滤镜透射
光谱，不称完整色头光谱 oracle。

**层间漂移开关（`--film-crossover`，LUT 时代——已由 film v2 因式分解链
取代,本节保留为历史记录;现行资产为 `dngscan/data/film_v2/` 三分家族,
开关正名 `--film-neutralization`,详见 FILM_PRINT_RENDERING_PLAN）**：
当时 full 模式的接管核心是离线烘焙的光谱链 65³ 查找表（输入声明为
纯场景线性 Rec.2020（`input_space=scene_rec2020`——full 模式跳过光谱前馈，
观察者逆矩阵自担分色）、逐通道 log₂ 曝光整形、四面体插值、域外钳制），开关
选择中性轴的供给方式：**off**（默认）——数字中性化变体，各通道除以中性阶
染色在该像素亮度曝光处的取值（存于同一 npz 的**有界** cast 曲线：校正乘数
沿 h(t)=1+t·(1/cast−1) 直线取最大 t 使每通道落在 [0.25, 4]，该直线上任意
一点都严格保持中性轴亮度；不烘第二只表——有界除数逐像素精确求值时商的
档位误差等于 datasheet 表自身，烘复合函数反而把 EV_Y 折点摆到网格对角线
上，实测最差 1.73 EV），介质灰阶偏中性两档以内严格中性，更深的染色（如
Kodachrome 的洋红地板）在 t→0 处如实保留为介质性格；
**datasheet**——链原样，中灰由 q 求解锚定、其余按层间数据漂移。观察者逆
矩阵在 rawtoaces 训练反射谱上受约束拟合，同色异谱残差逐卷盖章
（observer_p99_stop 0.51–1.20 档）；DIR/层间效应数据缺席，如实声明。
Velvia 100 的实测中性场（v2 数据，仍随预设发布）：蓝通道在 −2 EV 处 1.16、
−4 EV 处 1.09，+1 EV 仅 1.05——漂移温和、住在中灰以下。早期版本此处蓝通
道高达 1.84/3.16，经查大半是染料谱带外零填充（表格域外被当成透明，紫外/
红外泄漏抬高 R/B 通道）的伪影；载入端现按底座声明对染料与基底密度执行
边缘保持（v2 时代曾以 0.5% 投影 / 1% IEC 观看杂光作同类修正，介质原生
标定落地后随零杂光合同一并撤销）。修正后四卷反转片的 Dmax 地板全部回到
±6% 内的中性黑，与富士数据手册 Status A 平台值（R 3.38/G 3.82/B 3.68，温和品
红倾向）方向一致。量级仍未经外部 oracle 裁决。observe 模式下 datasheet 与 off 逐字节相同：接管核心不参与，
开关自然惰性。重建后的一个实证：诚实光谱链下 observe 与 full 在公园样张
上近乎合流（1% 像素过阈、绿相差 4°）——旧 full 的 23° 绿移大半是被废除的
RGB 启发式的伪影；两条独立路径互为印证。

### 风格配对——observe 模式的 look 层

observe 模式的完整结构正是 FilmLight 的原型：**稳定的自家 DRT（AgX）坐镇渲染槽，
风格作为分离的 look 层叠加**——look 不改 DRT，DRT 不含 look。观感评估的结论是
"observe 稳而淡、full 浓而坏"，缺的味道由 look 层补足，且只用两个**已验证机制**
做词汇：

- **前馈分离超驱动**（`--scene-transform-strength` >1）：把胶片感色分离（二手
  datasheet 数据、有界矩阵、中性保持）推过校准强度——加的是"这卷胶片怎么分
  颜色"的味；
- **AgX 原色几何**（`--agx-primaries` punchy/muted）：AgX 自家被验证的纯度
  词汇——加的是密度与饱和的"浓"。

`FILM_STYLE_PAIRINGS`（film_curve.py）按卷口碑给每个预设一组
（强度, 几何）配对：Velvia→(×1.6, punchy)、Ektar→(×1.4, punchy)、
Kodachrome→(×1.4, muted)、Pro 400H→(×1.3, muted)、Vision3 系→(×1.2, muted)、
theatrical 系→(×1.4, punchy)、Portra 系→(×1.3, base) 等。三条属性：**编辑
决定，不冒充测量**（A/B 样张标定的初稿，用户观感为终裁）；combo 只覆盖仍处
默认值的层，显式给出的值优先，GUI 控件同步可见——没有烘焙；full 模式不吃
配对（胶片显影模型自带性格）。`--punch`（场景自适应纯度补偿）暂不在配对
词汇内，保持为独立控制。三管线对照与原型对应（observe≈FilmLight、
full≈Kelly）的完整论证见[设计合同](FILM_OBSERVATION_PLAN.zh-CN.md)。

两个杠杆的量感先说结论：**这是精修级不是风格再造级**——全幅缩略图上不可辨认
是量级的事实，不是展示的失败。因此文档采用两级仪器：全幅图看构图语境，
1:1 裁切 + 声明倍率差值图（×6，先例：HDR 亮度分配图）看差异本体与位置。

全幅语境（日光帧，Velvia 100；上排分离 ×1.0/×1.6/×2.2，下排 punch
0/0.75/1.5——punch 是场景自适应纯度补偿的倍率，夜景自动归零）：

![风格双轴全幅语境：分离强度与 punch 浓度各三档](assets/film-style-axes.jpg)

差异本体（室内密集色帧的最大差异区，1:1 裁切 + ×6 差值图，实测数字印在图内）：
分离轴 ×1.6 对 ×1.0 为 mean 3.3 / p99 33 / max 77 codes，×2.2 递进到
mean 6.4 / p99 64 / max 114——集中在瓶标的色彩分离；punch 轴 0→1.5 为
mean 1.3 / p99 28 / max 51，刻意的精修量级：

![风格双轴仪器图：1:1 最大差异区裁切与 ×6 差值图，含实测数字](assets/film-style-axes-indoor.jpg)

（勘误记录：本图的首两版把"×1.0"格错渲成了配对默认 ×1.6——配对曾用"值等于
默认即未设置"填充，显式 ×1.0 被静默改写；差值图全零暴露此 bug，已修为 None
哨兵语义并有回归测试，见工程决策记录案例 9 续三。）

同一帧 fp 样张过四个家族与影院引用（随机选自真实拍摄卡，全部现行管线渲染）：

![同帧对比：AgX 基线与七款胶片预设（柯达/富士负片、反转片、电影卷及影院引用）](assets/film-families.jpg)

同一组预设在非日光的室内复杂色场景（钨丝暖光 + 多色罐墙）：负片的宽容、反转片的
密度、影院引用的反差在混合人工光下走向各自的另一面：

![室内复杂色同帧对比：同一组胶片预设在钨丝混合光下](assets/film-families-indoor.jpg)

WB 声明与镜前滤镜层的独立效果（同帧，无胶片曲线）：

![同帧对比：AsShot 与三档固定色温，及 85B/80A 转换滤镜](assets/film-wb-and-filter.jpg)

### 预设库与出处

**二十款胶卷 + 五款影院变体**（Portra 全家含迫冲、Ektar、Gold、Ultramax、
Superia X-TRA、C200、Pro 400H、四款反转片、Vision3 全系与 Verita）以
`--film <名字>` 或 GUI 一键展开为上述声明组合；任何一层都可单独覆盖——**没有
烘焙**。每个预设携带 `source`（数据文件+模型描述）、`fit.rms_stop`（拟合残差）与
`fit.pinned`（钉界参数=声明的域外外推）；数据来自 spektrafilm 的 CC BY-SA 4.0
profile（出处链见 NOTICE.md 与 `dngscan_assets/spectral/spektrafilm/README.md`）。
当前拟合水平（介质原生 v4：TH-KG3 印相光源、原生网格梯形积分、q 联合求解、
零标定杂光、XYZ→CAT→Rec.2020 观看翻译）：25 支全部 rms ≤0.061——负片系
0.014–0.044、反转片系 0.022–0.053、theatrical 引用系 0.027–0.061。标定去掉
观看杂光后反转片全部回到无钉界的内点解（velvia white 回到 +7.59——v2 时代
的 +8.5 顶格是杂光垫高目标后的副作用）；仍在 `fit.pinned` 里的少数钉界
（gold200/vision3 系 white@8.5、verita toe@3.5）逐项如实记录——这是 C1
曲线族对硬肩介质的表达边界，不是数据缺陷。

### 外部验证

`tools/crosscheck_2383.py` 用独立实现（DiVERE 的 Kodak 2383 曲线，曲线域约定从其
源码逐行核对）交叉验证——这把外部尺子已用五次：第一轮定位了旧通道捷径的缺陷
（B 通道 rms 0.383，尺度阶梯随波长单调）；第二轮验证光谱印相升级（当时口径下
三通道 ≤0.086 stop 合流）。v2 底座重建后比较约定升级为**密度域对密度域**
（DiVERE 的曲线是逐通道印片密度，无比色主张；链 A 现供 pre-colorimetry 的
逐通道印片透过率——v2 的正规比色 CAT/基底会合法地混通道，不该进这把尺子的
裁决范围）。第三轮（五支 2383 卷,染料谱边缘保持修正前）：R rms 0.056–0.067、B
0.049–0.080 合流,G rms 0.11–0.15、B 密度尺度 ~0.68–0.76 偏离。第四轮
（边缘保持修正后重测）：B 密度尺度回升到 0.82–0.84（旧的 B 形状"合流"
部分是幻影紫外印相通道的巧合产物：负片染料表 380–405nm 缺口被零填充成
透明,而 2383 层 0 的本征紫外敏峰恰在 385nm）,但三通道形状 rms 上行。
**第五轮（ISO 5-3 密度计量投影,`--input divere-status`）结案**：把两条链
投影进同一密度计量域——链 A 取显影印片组的 **Status A 读数**,链 B 输入取
负片的 **Status M 读数**过 DiVERE 默认的 Cineon Status M→印相密度矩阵——
五卷全部合流：rms R 0.092–0.100 / G 0.039–0.043 / B 0.077–0.103 档,拟合
密度尺度 1.02–1.10（B 的"尺度偏离"彻底消失）。历史上的"G 深阴影背离"与
"B 密度尺度"两项立案都是**度量域伪影**：拿逐染料配量去对密度计读数,串扰
被记在了形状账上。柯达 H-1-2383 原始特性曲线作第三方裁决：中段 gamma≈4、
Dmax≈4.1,我方 Status A 重建（3.9–4.5）与 DiVERE 曲线（3.6–4.1）皆在其
域内；负片侧我方 Vision3 的 Status M gamma 实测 0.50–0.56/logE,与柯达
5207 出版值一致。残余的 ~1.1× G 尺度解读为 Cineon 时代通用矩阵（G 对角
0.89）与本卷配对光谱关系（1.08）之差,属约定差异而非缺陷。密度计响应度
数据入库 `dngscan_assets/spectral/densitometer/`（出处 Giorgianni/Madden/
Kriss 2009 p.335）。

与市面胶片工具的根本差异：全部声明发生在 scene-referred 侧，因此**胶片性格能进
Ultra HDR 交付**——"Portra 的身体 + 真实测量的高光余量"。曲线拟合的方法论与
历次拟合器缺陷的排查过程见[工程决策记录](ENGINEERING_NOTES.zh-CN.md)。

## 四层：Delivery — SDR 与 HDR 交付

SDR 输出是带确定性 TPDF 抖动的 8-bit JPEG，默认 quality 100、4:4:4。抖动发生在量化前，
用来减轻平滑渐变的断层；它不改变 tone plan。也可以选择 4:2:2 或 4:2:0 来减小文件，
代价是色度分辨率。Display P3 会嵌入 ICC profile，找不到 profile 就停止导出，不写未标记
的宽色域数据。

HDR 输出是可选的 Apple ISO 21496-1 gain-map 封装（JPEG 或 HEIC），目前只在
macOS/Core Image 后端可用，并且只接 AgX tone core。HEIC 与 JPEG 共用同一套 formation
masters，只换最后一跳编码——但本管线实测 HEIC 文件更大、回读误差也更大（见交付档
说明与使用指南），推荐默认仍是 JPEG 容器。它不是把 SDR 成片直接放大：同一份 scene-linear
Rec.2020 在 display formation 前分成 SDR AgX 与 HDR AgX 两条独立 DRT。两者共享拍摄曝光
意图和 RAW 分析，但 HDR 自己持有 tone curve、色彩几何和扩展 P3 投影，不要求任何像素区域
与 SDR 成片一致。

HDR 可用余量不是用户所选屏幕容量的同义词。屏幕容量只是上限；初始请求由
RAW 剪切证据筛过的可靠高光尾部决定。LibRaw 用逐像素 CFA mask，RAW9 则按全分辨率
剪切 cell 比例从亮度顶部做保守的 rank trim。没有足够 RAW 证据时 headroom 就是 0，导出会
明确失败，不会用重建高光或 SDR white endpoint 冒充传感器信息。

这个请求会编译成一条不改写 body 的 HDR 曲线。K 以下继续使用 darktable 式 AgX body，内部
gamma 固定为历史值 2.2；K 以上在 output-stop 坐标中接 cubic Hermite，从实际渲染 body
的数值与解析切线出发，在 W 到达场景挣得的峰值并以零导数结束。白端切线钉零时，单段
Hermite 单调的充要条件是归一化起点切线 `alpha <= 3`；显示容量 3 EV 下整个生产策略域都在
界内（用户可调 contrast=1.5-4.5 全范围的最坏值为 2.9306）。但显示容量独立于尾部驱动的 W
封顶 Z_peak，低容量显示配上很长的可靠尾部会把 `alpha` 合法地推过 3——这不是畸形请求，
只是一个压缩很强的 shoulder，与 Blender HDR AgX 在同一处境下加大肩部弯折是同一类事。
此时编译器细分为多段单调 Hermite 链，结构合同与单段完全一致：K 点锚定值与切线不动、
白端导数为零、段间 C1、逐段单调，由同一个验收函数把关；headroom 控制因此全程连续，
不会跳到"无 HDR"。严格 fail-closed 只留给真正退化的输入（空窗口、非正上升量、非有限
锚点）。管线里没有整体 gamma 抬升、曲线后 smootherstep gain、allocation window 或
lift-rate 启发式。

![HDR log-stop shoulder：+3 EV 容量单段、+1.5 EV 容量细分链，K 以下 body 完全一致](assets/hdr-shoulder-subdivision.png)

原生 HDR 曲线是唯一亮度权威。`rho` 只在 reference-white AgX 色度路径和扩展白原生路径之间
混合，两条路径先对齐到原生曲线决定的同一 Y。固定为 1.0 的 reference-white endpoint 不与
场景 W 耦合，因此这个辅助色度候选可以显式细分 Hermite 区间；它不进入权威 tone plan，也
不能改变输出 Y。逐像素 CFA 剪切 mask 会从不可信通道撤回原生路径。最后用保持 Y 的中性轴
投影收进扩展 P3 `[0, peak]` 色彩体积，不做逐通道硬裁。这里保持
的是线性 P3 的 opponent direction，不是严格的感知色相。ACES 2 在色貌模型 JMh 中完成更强
的色相约束；dngscan 当前投影器刻意更简单，这也是 HDR 仍需实机标定的边界之一。

Core Image 只把已完成的 SDR/HDR 两张 rendition 写成 RGB gain map。每个文件写完后
都会重新展开 HDR 像素，检查 P3 profile、RGB 辅助图、声明 headroom、SDR 底图码值误差，
以及 HDR 的块级与像素级色品误差；archive 档额外要求 4:4:4 底图。各容差集按投递档位
与容器分别在真实样张回归集上标定（share HEVC 在同样名义参数下的损失明显大于 share
JPEG）；任一门禁不过就不会保留输出文件。现在 HDR 不支持 display look/filter，
因为这些 SDR 算子还没有独立 HDR 定义。数学约束和验收线在
[`docs/HDR_AGX_V2_IMPLEMENTATION_PLAN.zh-CN.md`](HDR_AGX_V2_IMPLEMENTATION_PLAN.zh-CN.md)。

### 交付档的实测定位

两个交付档是两个被测量过的操作点，不是一根质量滑杆。在全分辨率回归样张（24.5 MP
Sigma fp）上：archive q100/4:4:4 约 60 MB——验证级母版，约为源 DNG 的两倍，因为去拜耳
后的三通道 q100 JPEG 加 gain map 本来就比无损压缩的 14-bit 拜耳马赛克大。share
q90/4:2:0 为 11–27 MB，gain map 与 content headroom 完整保留；最坏情况（高 ISO 舞台帧）
仍在微信原图 25 MB 上限之内。两档之间没有值得买的中间点：Core Image 的主图色度采样由
quality 涌现，实测只有恰好 q100 才输出 4:4:4，而 q90 到 q99 保真度几乎不变、体积单调
增长——损失由 4:2:0 主导，不由 quality 数字主导。

![archive 与 share 全分辨率裁切对比：q100/4:4:4 62.8 MB 对 q90/4:2:0 11.5 MB](assets/delivery-archive-vs-share.jpg)

share 操作点在普通内容上观感透明、体积小 5.5 倍；它的实测代价集中在像素级色品统计——
这正是按档位标定的门禁所盯的量。容器选择优先 JPEG：ISO 21496-1 gain-map JPEG 是
Apple 与 Google 收敛后的跨平台格式（iOS 18+、Android 15+、Chromium），在不认识它的
查看器里优雅退化为 SDR 底图；而国内通讯应用在多数路径上会把 HEIC 转码成 JPEG，gain
map 随之销毁。share HEVC 在本机实测也严格更差——文件更大、块级误差约 1.8 倍——所以
`ultrahdr-heic` 是为必须收 HEIC 的下游准备的，不是效率升级。

![舞台帧 share JPEG 对 share HEIC：HEIC 更大且涂抹感可见](assets/delivery-jpeg-vs-heic-share.jpg)

### HDR 对比

下面是 SDR 诊断图，不是 HDR 屏幕截图。每张依次是 SDR、按实测 headroom 降曝光后的原生
HDR，以及 curve expansion map。中间面板把 reference white 以上的细节压回普通网页范围，
所以理应更暗。右侧面板计算同一 HDR 分支中原生扩展白曲线与 reference-white 对照渲染的
log-stop 亮度比，因此只回答“DRT 把额外亮度放在哪里”，不会把色彩几何和 JPEG 封装混进来。

| 样张 | 编译余量 | 像素实际达到 | 超过 SDR 白的通道样本 | 主体亮度变化 |
|---|---:|---:|---:|---:|
| `_SDI0150` 混合光 | +1.36 EV | +1.20 EV | 1.75% | -0.00014 EV |
| `_SDI0199` 舞台 / ISO 25600 | +1.26 EV | +1.14 EV | 1.72% | -0.00004 EV |
| `_SDI0133` 暗光餐厅 | +1.32 EV | +1.17 EV | 3.13% | -0.00015 EV |

接近零的主体亮度变化是这里最重要的控制量：HDR 把显示余量花在高光上，没有把 headroom
变成隐蔽的整体曝光提升。表里的百分比统计超过 reference white 的 RGB 通道样本，不是整像素。
这些半分辨率诊断图由 [`tools/hdr_ab.py`](../tools/hdr_ab.py) 生成；正式 JPEG 仍按全分辨率渲染，
并在写入后展开 gain map 做 round-trip 校验。

**混合光室内人像——额外亮度留给线形灯具与反光**

![原生扩展白 HDR AgX 混合光诊断](assets/hdr-comparisons/_SDI0150_native_hdr_ab.jpg)

**舞台 / ISO 25600——稀疏光源获得余量，环境仍保持暗调**

![原生扩展白 HDR AgX 舞台灯诊断](assets/hdr-comparisons/_SDI0199_native_hdr_ab.jpg)

**暗光餐厅——小面积反光和皮肤高光展开，不整体抬升场景**

![原生扩展白 HDR AgX 餐厅高光诊断](assets/hdr-comparisons/_SDI0133_native_hdr_ab.jpg)

[早期对比页](archived/HDR_COMPARISONS.md)保留使用已删除 smootherstep allocator 时做的
RAW9/LibRaw 与 AgX/neutral 实验，作为开发记录，不再是当前像素参考。Core Image/ISO 在
macOS 上已经逐文件 round-trip；Android/Chrome 互认和项目自定色彩参数的 EDR 样张标定仍
需要真机完成。

这次 HDR 边界核对使用了 Apple 的 [Adaptive HDR 与 Core Image
流程](https://developer.apple.com/videos/play/wwdc2024/10177/)、Android 的
[libultrahdr gain-map 数学](https://android.googlesource.com/platform/external/libultrahdr/+/refs/heads/main/lib/include/ultrahdr/gainmapmath.h)、[darktable AgX
处理顺序](https://docs.darktable.org/usermanual/development/en/module-reference/processing-modules/agx/)，
以及 ACES 2 公布的 [chroma](https://docs.acescentral.com/system-components/output-transforms/technical-details/chroma-compression/)
和 [gamut](https://docs.acescentral.com/system-components/output-transforms/technical-details/gamut-compression/)
compression 说明。它们定义职责边界和参照方法，不会把 dngscan 自己的阈值变成上游常数。

## 附：保留的前馈实验

> 注：这套前馈机制现已承载胶片光谱分离预设（见"胶片观察位置"一节）——同一标定器、
> 同一窗口/置信度合同，拟合目标从数字化的 ALEV 猜测换成了数据手册里的胶片感色层曲线。
> 以下记述保留原 ALEV 实验的动机与边界。


这个实验始于“在进入 AgX 之前，先用测量数据补偿相机某些可重复缺陷”的想法。更进一步，
如果两套传感器与滤镜栈的光谱响应都测得足够清楚，也可以在原相机真正记录到的信息范围内，
近似另一台相机的部分响应关系。

项目里的 ARRI-like 前馈来自一个主观目标：让 Sigma fp 稍微靠近这个实验预期的 ARRI 肤色，
也就是血色撑起来的温润感，以及偏冷 cyan 环境带来的衬托。最初的猜想与 ALEV 滤镜栈
较宽松的红光/近红外响应有关，而 fp/IMX410 本身也有不同的滤镜和洋红行为。

现在这份实现把公开的相机 SSF、光源 SPD 和材料反射谱做光谱积分，对皮肤、植物、cyan、
中性与洋红等材料类别拟合受约束的 3×3 映射，再用 `(R/G, B/G)` 色度平面上的软窗口限制
每个映射的作用域。窗口会通过 von Kries 缩放随所选白平衡移动；中性轴约束避免它变成
隐性白平衡，逐类残差和跨类泄漏则进入置信度。

ALEV III SSF 数字化自 Leonhardt & Brendel 的 CIC23 论文。ARRI 在论文中对五台 ALEXA
的测量取平均，因为传感器叠层的干涉纹理会随个体变化。Sigma fp 一侧使用 AMPAS
`rawtoaces-data` 中由 Weta Digital 测量的 Sony A7 III 整机 SSF；它同样基于 IMX410，
但不能等同于 fp 自己的完整滤镜栈。相机到 Rec.2020 的 profile 使用 AMPAS 的 190 条训练
反射谱拟合。这里的来源和替代关系都保留在标定文件里，不把“同一块 CMOS”写成“同一台
相机”。

它有很明确的物理边界：如果两种材料在 fp 上已经成为同色异谱，逐像素矩阵不可能重新创造
它们在 ALEV 上本应有的区别。而且传感器滤镜栈存在个体差异，严肃标定应该针对实际使用的
每一台相机。目前缺少可控光源、标准靶和光谱设备，所以现有结果更接近一个克制的几何颜色映射，
离最初设定的 ARRI 肤色目标仍有距离。数据来源、假设、CSV 和拟合报告放在
[`dngscan_assets/spectral/`](../dngscan_assets/spectral/) 里。

## 附：风格与 LUT

仓库自带一个本地设计且日常使用的 `optic_warm_cyan`。它是 AgX 之后的 Oklab 色度场，
不是厂商 LUT，也不冒充相机前馈。

代码还留了 Kodak 2383、RED IPP2 和 Sony LC-709TypeA 的可选 `.cube` 槽位。合法拥有的
LUT 可以放进 `dngscan_assets/vendor_luts/` 下对应路径，GUI 会自动识别；仓库本身不分发
这些文件。前馈、AgX 几何和显示端 LUT 分属三个不同位置，效果即使相似，含义也不一样。

## 附：RAW 分析图

`--scan` 输出六面板报告，包括 SNR 对档数、分离的 R/G/B RAW 分布、曝光与色域压力、空间
曝光区与剪切通道图，并列出逐通道 full-well、clip、black level 和 WB 读数。RAW 分布横轴
使用距剪切的 stops，纵轴为峰值归一化线性密度；图上的密度曲线会轻微平滑以便阅读，
clip%、中位、分位和其他统计始终从未平滑的原始样本计算。SNR 与动态范围是单帧估计，不是
完整 photon-transfer 测量；容器 bit depth 也不等于可用动态范围。
