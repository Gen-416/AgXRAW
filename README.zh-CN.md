# AgXRAW

一套从传感器数据出发、用于生成和研究 SDR / HDR 影像的开源数字成像工作台。

它最早只是想解决一个很具体的问题：怎样不打开完整编辑器，也能用 AgX 把一张 RAW
显影好。真正做下去，问题很快变大了：传感器到底保留了多少高光？高光重建补出的像素
还能信多少？同一份场景数据怎样分别生成 SDR 和 HDR？胶片的白平衡、感色特性和显影
曲线，能不能拆开研究，而不是揉成一层滤镜？

AgXRAW 把这些问题放进同一套可测量的流程里。它在解拜耳前保存传感器数据，用
LibRaw 或 Core Image 形成场景线性 Rec. 2020 图像，再把测量结果与使用者的成像选择
汇总成渲染方案，最后生成经过颜色管理的 SDR 或 HDR 文件，并检查实际交付结果。

它现在可以直接当作本地 RAW 显影器使用，也已经长成了一座小型成像实验台。解码、
传感器分析、显示变换、胶片观察和交付编码各有边界；新的解码器、明暗压缩方法、胶片模型
或交付格式，可以在同一份 RAW 数据和同一套验证方法上比较，不必重新发明整条管线。

[English](README.md) · [许可证](LICENSE) · [第三方声明](NOTICE.md)

**文档**：
[修图教程](docs/EDITING_TUTORIAL.zh-CN.md)（从导入到导出的完整流程，逐个控件讲用法）·
[胶片教程](docs/FILM_TUTORIAL.zh-CN.md)（每个胶片滑条和选择到底有什么用，配样张）·
[使用说明](docs/USER_GUIDE.zh-CN.md)（支持的相机、界面字段、导出选择）·
[机型支持](docs/SENSOR_SUPPORT.zh-CN.md)（传感器数据、降级策略与 LibRaw 升级）·
[完整文档索引](docs/README.md)（架构、计划书落地状态、基线记录）

## 一张图看 HDR

![SDR、按余量归一化的 HDR 与 HDR 亮度分配图](docs/assets/hdr-comparisons/_SDI0150_native_hdr_ab.jpg)

从左到右分别是普通 SDR、单独生成的 HDR，以及 HDR 新增亮度的分布。右图越亮，表示
该区域用到的 HDR 余量越多；黑色表示没有增加，白色表示已经用满。

可以看到，额外亮度集中在灯具和反光处，房间本身并没有跟着变亮。支持 HDR 的设备会
显示这些亮部；普通屏幕则显示文件中的 SDR 底图。如果 RAW 里没有未过曝的高光，
AgXRAW 也不会凭空增加 HDR 余量。

## 一张图看胶片观察

![AgX 基线、Portra 400、Velvia 100 与 Vision3 250D 影院外观对比](docs/assets/film-observation-showcase.jpg)

同一张 RAW 的四种观察位置：AgX 基线（无胶片）、Kodak Portra 400（负片+相纸）、
Fujifilm Velvia 100（反转片）、Vision3 250D 影院放映外观（引用原文变体）。

**两种模式,两套语义**（A13 更正:此前的表述把它们混作一套默认底座）。默认的
`--film <卷>` 是 **observe（观察）模式**：胶片声明观察者看见了什么——白平衡
色温、感色层分离前馈、显影曲线——颜色仍由 AgX 显影;它**不经过**相纸模拟。
每款的默认观感另加两项**明确标注为编辑取向（非测量）**的声明：按胶卷口碑配对
的分离强度倍率与 AgX 原色几何,两者都在界面可见、随时可改,且永远不会覆盖你
亲手设置的值。**full（接管）模式**（`--film-mode full`,实验）才运行完整的
因式分解光谱链——Stage A 观察者→层曝光→特性密度→B1→印相 timing→相纸显影→
B2→灰阶中性化→可选参考印相外观层——没有单体、创意或不透明的 LUT：链被因式
分解为可追溯的 B1/B2 插值资产（同一声明数据离线求解的 65³ 体积,不是手调
观感）。层间放大与 reference recipe 只属于 full 模式。二十款胶卷与五款影院变体的完整清单见
[胶片教程](docs/FILM_TUTORIAL.zh-CN.md)，技术底座见
[架构文档](docs/ARCHITECTURE.zh-CN.md)。

## 一张图看三种胶片解释

![AgX 观察、胶片技术链、胶片参考印相对比](docs/assets/film-three-interpretations.jpg)

外观层落地后，full 模式下一卷胶片回答两个可分离的问题。**technical（技术链）**
是光谱底座加 modelled 层间放大——公开光谱数据（乳剂感度、染料密度、相纸、光源）
约束下的三刺激重建，加声明的 modelled 层间项，关闭编辑性外观，灰阶数字中性。它不是
"实测这张印相"的宣称：报告带着实际运行的 Stage A 模型及其 held-out 残差。
**reference（参考印相）**在其上叠加声明的参考印相外观：按卷 author 的 palette
（hue 路径与色密度，以 Endura 公共基调 + 每卷 residual 表达，从不是烘焙 LUT），
以及 print-balanced 灰阶策略——锚定中灰、让相纸自己的两端 crossover 呼吸。每一层
都声明来源（measured / modelled / editorial），`technical` 始终字节冻结、随时可回。
第三档 `custom` 暴露三个以 recipe 自身为中心的有界修饰（颜色丰度/色密度/灰阶
偏色）。GUI 出厂默认为 `reference` @ 强度 1.0（2026-08-12 一次性校准；依据
A/B 判词与真实照片可见性实测），`technical` 一键可回、无配方卷自动回退；
CLI/API 默认保持 `technical`（脚本口径以测量链为基准）。

## 一张图看光谱印相链

| 数字中性化（默认） | 数据手册层间漂移 |
|---|---|
| ![off](docs/assets/film-tutorial/crop_crossover_verita_off.jpg) | ![datasheet](docs/assets/film-tutorial/crop_crossover_verita_datasheet.jpg) |

实验性的胶片接管模式（full）不再是"逐通道曲线"的启发式：场景颜色经过受约束的
观察者逆矩阵折算为三层乳剂曝光，过各层特性曲线进入**因式分解的印相链**——负片
密度 → 相纸层曝光（B1）→ 印相 timing（τ）→ 相纸显影曲线 → 观看链（B2），印相
介质、timing（固定/随胶片曝光重定时/自定义色头）、灰阶中性化、编辑显影配方、
Film Compression 与模拟光学（颗粒/halation/bloom）都是这条链上可声明的真实状态；
Ultra HDR 下它以"胶片印相 + scene HDR 扩展"参与（SDR 底图=胶片印相，参考白之上
按场景高光平滑增益）。上图是它能渲染而调色滤镜渲染不出的东西——Verita 200D
印相链按数据手册实测的**层间漂移**（crossover）：雕花木门与石阶的阴影转出绿青，
白墙与受光的卵石地纹丝不动；全图亮度差中位数为 0。左边是默认的数字中性化变体
（灰阶严格保持中性，即 `--film-neutralization bounded`），右边是数据手册原样。

## 功能

- **读懂拍摄数据**：在解拜耳前测量黑电平、白电平、各颜色通道的过曝剪切和彩色滤光片
  阵列（CFA）排列，并据此估算噪声、可用动态范围和可靠的高光余量。
- **选择场景解码器**：LibRaw 与 Core Image / RAW 9 是两种独立选择，但都会向后续步骤
  交付场景线性 Rec. 2020 图像。
- **试验成像方法**：AgX 是默认选择，同时保留 RAW 门控、仅亮度和固定曲线等实验或诊断
  方法。曝光、白平衡、高光处理、场景变换、镜前滤镜和胶片观察都可以明确选择。
- **分别生成 SDR 与 HDR**：SDR 输出到 sRGB 或 Display P3；HDR 则从同一份场景图像
  重新显影，只使用 RAW 中未过曝高光真正支持的额外亮度。
- **观察并复现实验**：本地图形界面和命令行使用同一套设置，诊断图与 CSV 报告让测量
  结果可查、对照过程可复现；RAW 文件不会上传。
- **交付后再验证**：`archive` / `share` 档只改变编码，不改变成像。在 macOS 上，HDR
  可写成符合 ISO 21496-1 的增益图 JPEG 或 HEIC，随后回读检查颜色配置文件、增益图、
  文件声明的 HDR 余量和像素误差。

## 快速开始

需要 **Apple Silicon 的 macOS** 与 Python 3.11 或更新版本——原生内核 wheel
只在构建机的 macOS 版本上构建与验证（产物按构建机平台打标签,如
``macosx_.._arm64``）;更早的系统与 Intel Mac 未经测试,不作支持声明。纯
NumPy 路径（``DNGSCAN_FAST=0``）除 Python + NumPy 外无原生要求。工具与 Core Image /
RAW 9 解码器、HDR 增益图交付及其回读校验深度集成，支持平台限定为 macOS——这是声明的
边界，不是未测试的默认。项目已把验证过的 rawpy/LibRaw 源码版本锁为直接依赖，首次安装
会进行本机构建，因此还需要 Git 与 Xcode Command Line Tools。

### 图形界面

Python 包与命令行沿用引擎的历史名 `dngscan`。

```bash
git clone https://github.com/Gen-416/AgXRAW.git
cd AgXRAW
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m dngscan.gui
```

在浏览器中打开终端显示的本机地址（localhost）即可。可以先用 `EV 0`、`AgX`、`base` 原色、
`camera` 白平衡和 `reconstruct` 高光处理，再根据照片本身调整。

顶部的 RAW 入口使用浏览器原生文件选择器。选中的文件只会传给同一台电脑上的 localhost
服务，并保存在进程级临时目录中；退出 AgXRAW 后临时副本会自动清理，不会发送到外部服务。

### 命令行

```bash
# 默认 AgX JPEG
python -m dngscan photo.dng --jpeg photo.jpg

# 高光重建 + Display P3
python -m dngscan photo.dng --jpeg photo_p3.jpg \
  --highlight-mode reconstruct --output-gamut p3

# HDR 增益图 JPEG（macOS、Display P3、仅 AgX）
python -m dngscan photo.dng --jpeg photo_hdr.jpg \
  --output-format ultrahdr --hdr-headroom 3

# RAW 分析图和 CSV
python -m dngscan photo.dng --jpeg photo.jpg --scan --csv photo.csv

# 对比实验性的 RAW 门控明暗压缩
python -m dngscan photo.dng --jpeg photo_gated.jpg --tone-core gated

# 使用胶片观察位置
python -m dngscan photo.dng --jpeg photo_portra.jpg --film portra400
```

完整参数见 `python -m dngscan --help`。

### 可选的 C++ 加速

NumPy 是参考实现，不编译原生扩展也可以正常使用。可选的 pybind11 C++ 内核加速 AgX
核心与共享的 SDR 输出终段（16 轮 Oklab gamut fit、transfer、dither、量化）；
RAW 分析、渲染方案和出错时的回退处理仍由 Python 负责。

```bash
pip install pybind11 cmake
tools/build_native.sh
```

## 工作原理

AgXRAW 会先把传感器测到的数据与使用者主动选择的设置分开处理，到生成图片时再汇总成
一份渲染方案。

```mermaid
flowchart TB
    RAW["RAW / DNG"]
    E["1. 读取传感器数据<br/>解拜耳前记录 CFA 排列、黑/白电平<br/>测量过曝剪切、噪声与动态范围"]
    D["2. 形成场景图像<br/>LibRaw 或 Core Image<br/>场景线性 Rec. 2020"]
    I["使用者选择<br/>曝光 · 白平衡 · 风格<br/>输出色域"]
    P["3. 汇总分析，确定渲染方式<br/>画面主体 · 可靠高光 · 过曝区域<br/>曝光基准 · 明暗曲线 · 色彩 · HDR 余量"]
    S["4. 生成 SDR<br/>默认使用 AgX · 也可切换实验性的明暗压缩方法<br/>得到 sRGB 或 Display P3 底图"]
    H["5. 生成 HDR<br/>从同一场景图像单独显影<br/>亮度受 RAW 中未过曝高光约束"]
    V["6. 编码并校验成品<br/>SDR → JPEG<br/>HDR → 增益图 JPEG / HEIC，并回读检查"]
    OUT["可交付的 SDR JPEG<br/>或 HDR 增益图 JPEG / HEIC"]

    RAW --> E
    RAW --> D
    E -- "测量结果" --> P
    D -- "场景像素" --> P
    I -- "成像选择" --> P
    P --> S
    P --> H
    S --> V
    H --> V
    V --> OUT

    classDef source fill:#ede9fe,stroke:#7c3aed,color:#1f2937
    classDef process fill:#eff6ff,stroke:#2563eb,color:#1f2937
    classDef intent fill:#fff7ed,stroke:#ea580c,color:#1f2937
    classDef render fill:#ecfdf5,stroke:#059669,color:#1f2937
    classDef delivery fill:#f8fafc,stroke:#475569,color:#1f2937
    class RAW source
    class E,D,P process
    class I intent
    class S,H render
    class V,OUT delivery
```

1. **读取传感器数据。** 解拜耳之前，AgXRAW 会记录彩色滤光片阵列各通道发生过曝剪切的位置
   和满阱容量，并估算噪声水平。后面处理高光时，仍能分清哪些像素来自传感器，哪些像素
   是高光重建补出来的。
2. **形成场景图像。** LibRaw 或 Core Image 将 RAW 解码为场景线性 Rec. 2020 图像。
   解码器只决定 RAW 怎样变成像素；之后怎样压缩亮度、怎样处理颜色，由后面的步骤决定。
3. **把测量与选择汇总起来。** 分析会把画面主体、可靠高光和过曝区域分开，再与使用者
   选择的曝光、白平衡、风格和输出色域一起，确定这张照片的渲染方式。
4. **生成 SDR。** 默认由 AgX 完成显示变换，也可以切换其他明暗压缩方法做对照和诊断；
   结果是一张 sRGB 或 Display P3 底图。
5. **单独生成 HDR。** 这一支从同一张场景图像重新显影，不会把完成的 SDR 硬拉亮；能用
   多少额外亮度，只看 RAW 里还保留了多少未过曝的高光。
6. **编码并检查成品。** SDR 直接写成普通 JPEG；HDR 则把两版图像装进带增益图的 JPEG
   或 HEIC，写完后重新打开，确认文件里的图像和 HDR 余量都符合预期。

## 与常见 RAW 处理流程的区别

区别不在于少了多少按钮，而在于传感器数据从什么时候开始不再参与处理。

darktable 的 AgX 和许多显示变换模块接收的是解拜耳后的浮点图像；AgXRAW 则一直保留
解拜耳前的彩色滤光片阵列数据。这样，明暗曲线仍然知道哪些高光可信，颜色处理也能
降低对过曝剪切区域和重建区域的信任。

AgXRAW 还会把测量与审美分开。黑电平、白电平、过曝剪切、噪声、动态范围和高光范围交给
程序分析；曝光补偿、白平衡、风格和查找表（LUT）仍由使用者选择。自动分析只负责说明
照片里有什么，不替使用者决定照片应该是什么样子。

HDR 也不是更亮的 SDR。两版图像都从同一份场景线性图像出发，各自完成显影。文件写完后，
AgXRAW 还会重新打开成品，确认 SDR 底图、HDR 效果和增益图都符合预期。

这些边界也给项目留下了扩展空间。增加解码器时，只要交出约定好的场景图像，后面的分析
和成像就不用重写；新的明暗压缩方法或胶片模型可以复用同一份分析；新的交付格式只接收已经
形成的图像，不会反过来改变它们。测量与校验始终可见，因此新方法仍然能放在一起比较。

AgXRAW 目前不管理图库，也不做局部调整。这是现阶段产品的边界，不是项目想象力的边界。
它更大的潜力，是成为一套开放、可解释的数字成像工作台：既能直接用来出片，也能在同一
批照片上比较算法、验证标准，并继续发展新的成像方法。

## 面向开发者的技术文档

[产品架构与领域模型](docs/PRODUCT_ARCHITECTURE.zh-CN.md)（软件分层、用例、边界上下文与不变量）·
[架构与技术细节](docs/ARCHITECTURE.zh-CN.md)（完整管线与每个环节的设计理由）·
[工程决策记录](docs/ENGINEERING_NOTES.zh-CN.md)（问题、证据与解法的推理过程）·
[设计合同](docs/FILM_OBSERVATION_PLAN.zh-CN.md)（胶片观察的生产合同与计算边界）·
[胶片印相与模拟成像层计划](docs/FILM_PRINT_RENDERING_PLAN.zh-CN.md)（full v2 设计提案：曝光状态、印相、颗粒与 halation）

## 许可证

AgXRAW 以 [GPL-3.0-or-later](LICENSE) 发布。
