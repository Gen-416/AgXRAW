# Film Optics V2 实现计划：颗粒、Halation、Bloom 与介质散射

> 状态：设计冻结候选稿，仅用于指导后续实现；本文件本身不改变当前输出。
>
> 范围：`full v2` 胶片显影路径中的空间成像。AgX、胶片特性曲线、B1/timing/
> 相纸显影/B2、SDR/HDR 交付均不在本计划中重写。
>
> 修订 R1（组合语义评审）：初稿的每个算子单独成立，但**组合起来会系统性地重复
> 计入 DC、重复衰减 MTF、并把颗粒标定在错误的坐标里**。本次修订改了 §3 的曝光
> 拓扑合同、§4.3–§4.4 的颗粒坐标与 MTF 预算、§5.2–§5.3 的 halation 触发与回注
> 形式、§6.1 的 bloom 尺度空间与保高光，并在 §10.1 增加对应门禁。P0 的实测基线见
> [`FILM_OPTICS_V2_P0_BASELINE.zh-CN.md`](FILM_OPTICS_V2_P0_BASELINE.zh-CN.md)。

## 0. 结论

当前实现的层级位置大体正确，但三个名称覆盖了四种不同现象，并且全部由一份
`MODELLED_DEFAULT` 驱动：

1. 负片乳剂内部散射（emulsion scatter）；
2. 穿过乳剂和片基后返回的 halation（backing reflection）；
3. 正片/相纸在曝光与观看时的介质散射（print-medium scatter）；
4. 镜头杂散光与乳剂共同形成、以观感为目标的 bloom/glow。

V2 不再用一个阈值 blur 同时近似这些现象。最终拓扑定为：

```text
scene-linear Rec.2020
  -> [可选 editorial capture bloom]
  -> film observer -> 三层线性乳剂曝光 E-
  -> [负片乳剂内部散射，能量守恒]
  -> [片基反射 halation，按层回注]
  -> 特性曲线 -> 负片染料密度 D-
  -> [负片密度颗粒]
  -> B1 -> printer timing -> 正片/相纸线性曝光 E+
  -> [正介质 formation scatter，介质 profile 驱动]
  -> 正介质特性曲线 -> 正片/相纸染料密度 D+
  -> [正介质密度颗粒，有数据时启用]
  -> B2 -> display-linear Rec.2020
  -> [反射相纸 viewing scatter；首版默认关闭]
  -> gamut fit / delivery
```

用户仍然只需要理解“颗粒 / Halation / Bloom”。内部则必须按上述物理责任拆开。
`Bloom` 在 GUI 中表示可见的高光 glow；目前的守恒 `medium bloom` 应改名为
`print-medium scatter`，不再冒充 Dehancer 一类工具里的 Bloom。

## 1. 公开资料能证明什么

### 1.1 Dehancer：行为合同，不是可复刻公式

Dehancer 没有公开内部卷积核和方程，因此不能把其控制项反推成“官方算法”。公开
资料仍然给出了很有价值的行为合同：

- 颗粒由图像/密度形成，而不是覆盖一层扫描噪声；粒径会约束可见细节；负片与
  正片颗粒有不同调性分布；阴影和高光中的颗粒都不应被强制归零；控制维度包含
  Size、Amount、Film Resolution、Shadows/Midtones/Highlights 与 Chroma。
- Halation 同时依赖 Source Limiter、Background Gain、Smoothness、Local Diffusion、
  Global Diffusion、Amplify、Hue、Blue Compensation 和最终 Impact。公开说明明确描述
  内圈可偏橙、外圈转红，以及画幅越小，相对于画面的 halo 越大。
- Bloom 不是全局柔焦，而是最大曝光和高反差边缘附近的局部 glow；源的亮度、源
  的空间尺寸、扩散半径和高光保护是互相独立的变量。
- 工具 profile 按 8/16/35/65 mm 画幅组织；Halation 另分正常防光晕层与
  `No Remjet`。这说明“画幅”和“防光晕结构”应是 profile 轴，而不是一个 amount
  滑块的隐藏副作用。

公开资料：

- [Dehancer Desktop Manual](https://www.dehancer.com/learn/article/desktop-manual)
- [How does film grain work in Dehancer](https://blog.dehancer.com/articles/how-does-film-grain-work-in-dehancer-ofx-plugin/)
- [Halation and its simulation in Dehancer](https://blog.dehancer.com/articles/halation/)
- [Bloom: what it is and how it works](https://blog.dehancer.com/articles/bloom-what-it-is-and-how-it-works/)
- [Dehancer Tool Profiles](https://blog.dehancer.com/articles/dehancer-tool-profiles/)

Dehancer 的不同版本文档对 `Details` 方向有不一致表述：旧文章把低值对应小源，
新 Desktop Manual 把高值描述为更能识别细小点源。因此 AgXRAW 不复制其数值方向，
而把自己的参数直接定义成“最小源尺寸（film-plane um）”，并用测试图固定语义。

### 1.2 Filmbox：先测量空间行为，再做可用模型

Filmbox 的公开资料确认了两点：

- grain/halation 依赖 scene-referred 数值；改变进入负片前的曝光，应像改变现场曝光
  一样改变空间效果；
- 其空间特征来自 s35 Vision3 250D 测试图和 Arriscan 扫描，而不是只凭 datasheet
  曲线推断。颗粒还分 Softness、Roughness、Streaks、九区调性、RGB channel mix 和
  明/暗颗粒；Halation 把普通小半径 halo 与去除防光晕背层的大半径 `Aura` 分开。

资料：

- [Filmbox Technical FAQ](https://videovillage.com/filmbox/technical_faq)
- [Grading with Filmbox](https://videovillage.com/learn/filmbox/full-guide/grading-with-filmbox)
- [Filmbox Grain](https://videovillage.com/learn/filmbox/full-guide/negative/grain)
- [Filmbox Halation](https://videovillage.com/learn/filmbox/full-guide/negative/halation)

这给本项目的直接启示是：datasheet 能约束颗粒 RMS 与 MTF，却不能单独决定 halation
的径向轮廓；halation profile 最终仍需要点源/边缘实拍或胶片扫描标定。

### 1.3 可复核的开源与论文参考

- darktable 的 grain 是 display-referred 的轻量实现，不适合作为本项目的物理拓扑，
  但它使用多频 simplex noise、分辨率相关采样和“相纸响应”调性 LUT，说明单尺度
  Gaussian noise 即使在轻量视觉实现里也不够。
- Spektrafilm 在密度域用 binomial/Poisson 粒子、子层、不同通道粒径、dye-cloud blur
  与 log-normal microstructure；散射侧把能量守恒的乳剂 core/tail 和加性的多次
  背反射 halation 分开。这些是很好的开源结构参照，但仓库也明确包含 modelled 参数
  与未公开研究笔记，不能把其默认常数称为测量真值。
- Newson 等人的 IPOL 模型用随机银盐粒子和 Monte Carlo 像素积分获得与输出分辨率
  无关的颗粒，是“小图物理 oracle”的合适起点。

资料：

- [darktable grain source](https://raw.githubusercontent.com/darktable-org/darktable/master/src/iop/grain.c)
- [Spektrafilm grain.py](https://github.com/andreavolpato/spektrafilm/blob/main/src/spektrafilm/model/grain.py)
- [Spektrafilm diffusion.py](https://github.com/andreavolpato/spektrafilm/blob/main/src/spektrafilm/model/diffusion.py)
- [Realistic Film Grain Rendering, IPOL 2017](https://www.ipol.im/pub/art/2017/192/)

### 1.4 可测约束

- Kodak 相机负片 datasheet 给出三通道 sensitometry、MTF 和随密度变化的 diffuse RMS
  granularity；其 RMS 使用 48 um microdensitometer aperture。VISION3 500T 资料还明确
  说明影像颗粒取决于内容、密度、曝光、处理与扫描条件。
- ISO 10505:2009 定义透射材料 intrinsic RMS granularity，但明确不覆盖反射相纸和
  Wiener spectrum。因此一个 RMS 数字不能替代完整颗粒功率谱。
- 经典 sensitometry 文献区分乳剂内 irradiation 与片基全反射 halation，并指出 halo
  尺寸受片基厚度和折射率影响，防光晕背层负责吸收返回光。
- 摄影相纸研究表明散射同时作用于曝光阶段和观看/测量阶段；后者本质上非线性，
  不能被一个显示 RGB 高光 blur 完整代表。

资料：

- [Kodak VISION3 500T 5219/7219 technical information](https://www.kodak.com/content/products-brochures/motion-picture/KODAK-VISION3-5219-7219-technical-information.pdf)
- [Kodak VISION Color Print Film 2383/3383 technical information](https://www.kodak.com/content/products-brochures/motion-picture/KODAK-VISION-Color-Print-Film-2383-3383-technical-information.pdf)
- [ISO 10505:2009](https://www.iso.org/standard/50747.html)
- [NBS: Sensitometry of Photographic Emulsions](https://nvlpubs.nist.gov/nistpubs/ScientificPapers/nbsscientificpaper439vol18p1_A2b.pdf)
- [Paper Substrate Spread Function and the MTF of Photographic Paper](https://www.imaging.org/common/uploaded%20files/pdfs/Papers/2003/PICS-0-287/8514.pdf)

## 2. 当前实现的具体失配

当前 `dngscan/film_optics.py` 的数值和结构有以下问题：

| 模块 | 当前定义 | 结果 |
|---|---|---|
| Grain | 一份 unit-RMS Gaussian 场，经单一 Gaussian band-limit | 频谱窄、分布接近普通平滑噪声，没有粒子团簇与 dye-cloud 层次 |
| Grain tone | `sigma = sigma0 * 4*Dn*(1-Dn)` | 在 Dmin/Dmax 被构造为严格 0，也没有正介质颗粒 |
| Grain profile | 全部胶片共用 `18 um / sigma0=.055 / corr=.35` | 胶片感光度、画幅、负片/反转片和印相介质没有实质区别 |
| Halation source | scene Rec.2020 先压成单通道 Y，再硬减阈值 | 光源的层曝光与颜色信息在扩散前丢失，阈值处不光滑 |
| Halation color | 所有半径共用 `(1,.22,.06)` | 内圈和外圈只有强弱变化，不会自然从橙过渡到红 |
| Halation radius | 基准 `0.55 mm`，再用 0.5/1/2 倍 Gaussian | 36 mm 宽、6016 px 图像对应约 46/92/184 px sigma，容易成为宽泛红雾 |
| Bloom | `source -> pyramid blur -> spread-source` | 这是守恒介质散射；高光核心必然变暗，不是常见的可见 glow 语义 |
| Bloom source | display-linear Y 固定阈值 0.6 | 不识别光源尺寸、背景、曝光可靠性，也不区分相纸/正片/镜头 |
| Halation DC（R1） | `lin += gain * w_c * spread`，纯加性 | 均匀亮面也被抬亮；特性曲线里已含 DC 反射，等于算两遍。实测整帧红通道 +0.95% |
| 曝光拓扑（R1） | `ev_offset` 在 `characteristic_amounts` 里才加 | halation 在其之前算完，且 source 直接取 scene Y——改胶片曝光时密度变、空间效果不变 |

这些参数不是“略微没调好”，而是缺少可以分别调节失败模式的自由度。V2 的目标不是
增加更多任意 slider，而是让每个自由度对应一个明确的信号或物理量。

P0 已经把上表逐条转成带单位的数值并冻结，见
[`FILM_OPTICS_V2_P0_BASELINE.zh-CN.md`](FILM_OPTICS_V2_P0_BASELINE.zh-CN.md)。

## 3. 统一符号与不可破坏的合同

令：

- `S(x)`：WB、曝光和相机矩阵后的 scene-linear Rec.2020；
- `E-(x) = M_observer S(x)`：三层负片线性曝光；
- `D-(x) = C-(log10(E-))`：负片三层染料量/密度；
- `LEP2(x)`：B1 输出的正介质 `log2` 曝光；`P(x)=2^(LEP2+tau)`；
- `D+(x) = C+(log10(P))`：正片或相纸密度；
- `R(x) = B2(D+)`：display-linear Rec.2020。

所有空间半径以 film-plane `um` 或 `mm` 存储，运行时再根据 gate geometry 换算为像素。
不得把像素半径写入 profile。所有卷积核必须非负并归一：

```text
K(r) >= 0,    integral K(r) dr = 1
```

### 3.1 曝光拓扑：空间效果必须看见胶片曝光（R1）

当前实现把 `film_exposure_ev + stock anchor` 作为 `ev_offset` 传给特性曲线查表
（`characteristic_amounts` 内做 `x = log_e + ev_offset * log10(2)`），而 halation
在这之前就已经用未加曝光的 layer exposure 算完了；halation 的 source 更是直接来自
scene Rec.2020，连 observer 都没过。结果是**用户改胶片曝光时密度会变，但 halation
的触发点和强度纹丝不动**——这不符合任何感光材料的行为，也切断了「像改现场曝光一样
改空间效果」这条 §8 已经写死的原则。

V2 把曝光提前到空间效果之前：

```text
E_c = observer(scene) * 2^(film_exposure_ev + stock_anchor)
   -> emulsion scatter -> halation -> characteristic curve(E_c)   # 不再传 ev_offset
   -> D_c -> grain
```

因为 `ev_offset` 原本就只是查表轴上的一个平移，前移到 `log_e` 上是**代数恒等**。
所以这次重排带一条硬门禁：**空间效果全关时，输出必须与现行实现逐像素等价**
（§10.1 门 11）。

### 3.2 不可破坏的合同

1. amount=0 走严格恒等快路径，不建立空间上下文；
2. 预览/crop/全尺寸使用同一负片物理坐标；
3. 固定 seed 的颗粒与线程数、tile 顺序无关；
4. 不用图像最大值或百分位自动归一空间效果，曝光改变必须真实改变效果；
5. 60 MP 默认额外内存仍受 512 MiB 档约束；
6. profile 资产必须声明 `measured / derived / modelled / editorial`，未知版本 fail closed。

## 4. Grain V2

### 4.1 语义拆分

颗粒至少来自两个介质：

```text
negative grain: D- 形成后、B1 之前
positive grain: D+ 形成后、B2 之前
```

负片最终画面阴影处的颗粒不一定来自负片本身；透明负片使印相介质获得高曝光，正
介质的颗粒会进入最终阴影。V2 不再要求一条负片 `sigma(D)` 单独解释整个成片的颗粒。
反转片没有印相时只启用自身正片颗粒。

### 4.2 小图物理 oracle

先实现只服务校准和单测的粒子 oracle，不直接承担 60 MP 导出：

```text
candidate particle centers: Poisson process on film-plane coordinates
particle radius/amplitude: profile distributions (initially log-normal)
development probability: p_c(D), monotone in local layer density
particle cloud: normalized radial kernel k_c,l(r)
D'_c(x) = Dmin_c + sum_i a_i * I[u_i < p_c(D(x_i))] * k_i(x-x_i)
```

每个粒子由 counter-based RNG 的 `(profile_hash, seed, layer, cell_id)` 唯一决定。这样
任意 crop 和 tile 都能重建同一个物理区域，不需要依赖遍历顺序。oracle 需要支持：

- 每通道/子层不同粒径；
- 部分共享的低频 processing mottle；
- 非高斯 skew/kurtosis；
- 像素 footprint Monte Carlo 或精确面积积分；
- 对均匀 patch，以标定 densitometer aperture 内的**平均透射率**为守恒目标：通过
  profile 的染料谱/状态响应计算 `D_status=-log10(mean(T_status))`，并解一个随目标
  密度变化的微小 bias，使其等于原 characteristic curve 的目标值。不能简单要求
  `mean(D')=D`，否则 `-log10` 的 Jensen 偏差会改变平均影调。

### 4.3 生产快速模型

生产端不直接逐粒追踪，而拟合 oracle 或真实扫描的统计量：

```text
g_c(x) = sum_b w_c,b * F_b(x) + m_c * M(x)
delta_D_c(x) = sigma_c(D_c) * Q_c(g_c, D_c)
D'_c = D_c + delta_D_c
```

- `F_b` 是 film-space 多频带随机场，不少于 fine / cloud / mottle 三带；
- `M` 是跨层共享的低频显影不均匀场；
- `Q` 用低阶分位变换匹配目标 skew/kurtosis，不能只保留 Gaussian 分布；
- 三通道协方差由完整 3x3 矩阵 Cholesky 混合，不再只有一个相关系数；
- `sigma_c(D)` 为有界 PCHIP 表，不使用固定抛物线；端点是否为零由数据决定；
- 输出像素通过现有 film-space 面积积分采样，继续保持 preview=full block mean 合同。

#### 4.3.1 快速模型必须继承 oracle 的平均透过率约束（R1）

§4.2 的 oracle 已经写明：均匀 patch 的守恒目标是标定孔径内的**平均透过率**，因为
`D = -log10(T)` 是凸的，`mean(D') = D` 并不给出 `mean(T') = T`。但上面的快速模型写成
`D' = D + sigma(D) * Q(g, D)`，均值为零的密度扰动会因为 Jensen 偏差抬高平均透过率
——**颗粒强度会改变画面的亮度和颜色**，而不只是加噪声。

生产 profile 因此必须携带并验证：

```text
D'_c = D_c + bias_c(D_c) + sigma_c(D_c) * Q_c(g_c, D_c)
```

- `bias_c(D)` 是随密度变化的有界补偿量，由 oracle 或解析近似求解，使得标定孔径内的
  平均透过率（进而 status 密度）等于原特性曲线的目标值；
- `Q` 的 quantile 变换会改变通道协方差，协方差必须在变换**之后**重新校准；
- 验收测在**最终 status 密度或显示输出**的均值上，不能只验证噪声场自身均值为零
  （§10.1 门 12）。

#### 4.3.2 颗粒标定在哪个坐标系（R1）

Kodak 的 48 µm granularity 是状态密度计读出的**光学密度** RMS。而运行时的 `amounts`
是给光谱印相链用的**染料量**坐标（`characteristic_amounts` 的输出，B1 的输入）。
两者不是同一个变量，直接把 `sigma_status(D)` 加到 `amounts` 上，既不保证最终 status
密度的 RMS 正确，也容易造出错误的彩色颗粒（三层染料对三个状态通道不是对角映射）。

V2 明确坐标转换。设完整链的染料量到状态密度映射为 `F`：

```text
status = F(amounts)
J(amounts) = dF/d(amounts)                    # 每个密度节点上的 3x3
delta_amounts = regularized_inverse(J) * delta_status
```

`J` 按 stock 在若干密度节点上用有限差分预计算并入资产（它只依赖已烘焙的 B1/B2 和
染料谱，不依赖图像）。正则化是必需的：`J` 在 Dmin/Dmax 附近接近奇异，直接求逆会把
目标 RMS 放大成噪声爆炸。

允许的替代路线（首版优先）：**不解析求逆，直接在完整光谱链输出端拟合运行时颗粒
参数**，使 granularity、NPS 和通道协方差同时命中目标。它更慢但更可靠，且避免了对
`J` 的线性化假设；`J` 路线可作为拟合的初值。无论走哪条，资产里必须写明标定发生在
哪个坐标系，不能只写一个 `sigma0`。

颗粒强度的标定单位为 optical-density RMS，而不是 UI 百分比。若 profile 有 Kodak 曲线：

```text
G48_c(D) = datasheet 标示的 rms granularity 数值
sigma48_c(D) = G48_c(D) / 1000
```

多频 PSD `P_c(f,D)` 经 aperture transfer `A(f)` 后必须重现该 RMS：

```text
sigma_A^2(D) = integral P_c(f,D) * |A(f)|^2 df
```

只有 RMS 而没有 Wiener spectrum 时，PSD 形状必须标记为 `modelled`；不能把拟合结果
表述成厂家测量。

### 4.4 颗粒与清晰度

粒径变大但细节完全不变，会得到“锐图上叠噪点”。因此 profile 另带负片/正介质的
MTF 或 acutance 先验：

```text
E_filtered = K_residual * E         # 在相应介质曝光阶段
D_grained  = grain(C(E_filtered))
```

MTF 与 grain 共享物理画幅，但强度控制不得自动改变 profile 的测量 MTF。GUI 的
“颗粒大小”是 editorial 尺度修饰时，才按相同比例改变 grain PSD 和对应的细节截止。

#### 4.4.1 传递函数预算：不能重复衰减（R1）

本计划里会有多个东西同时模糊同一段链路：§5.1 的显式乳剂散射、这里的 `K_resolution`、
§6.2 的正介质 formation scatter，以及将来可能加入的正片 MTF。而厂家公布的 MTF
（如 [Kodak VISION3 500T 数据表](https://www.kodak.com/content/products-brochures/Film/VISION3_5219_7219_Technical-data.pdf)）
是**整块材料的系统响应**，本身已经包含乳剂内散射、显影扩散和颗粒结构的影响。把显式
散射和一条按实测 MTF 拟合的核相乘，等于把同一个物理过程算两遍，结果就是画面整体
偏软。

因此每种介质只有**一份传递函数预算**，显式项先扣除，剩下的才交给拟合核：

```text
MTF_residual(f) = MTF_measured(f) / max(MTF_explicit(f), eps)
MTF_explicit(f) = 该介质所有已显式建模的空间算子在同一阶段的联合响应
```

三条约束：

1. 若在任何频率上 `MTF_explicit(f) < MTF_measured(f)`，说明显式散射已经比实测更软，
   该 profile 判为**不可行**（构建期报错），而不是继续加模糊或把残差核当锐化用；
2. 真实胶片的 MTF 在低频可以**大于 1**（DIR 耦合的邻界效应）。这部分不属于
   `K_residual`，它由 §7 的 DIR/耦合模型承担；`K_residual` 按定义非负、归一、
   低通，超出部分必须显式记账，不能靠除法悄悄变成锐化；
3. 资产必须写清 `MTF_measured` 的测量阶段（曝光态还是显影后、含不含扫描器），
   否则残差没有意义。

### 4.5 Grain profile 字段

```yaml
grain:
  provenance: measured|derived|modelled
  medium: negative|reversal|print_film|paper
  gate_reference_mm: [36.0, 24.0]
  rms_aperture_um: 48.0
  rms_density_nodes: [...]          # [D, sigma_D] per channel
  psd_bands:                        # modelled unless scan-derived
    - {kind: fine, radius_um: ..., weight_rgb: [...]}
    - {kind: cloud, radius_um: ..., weight_rgb: [...]}
    - {kind: mottle, radius_um: ..., weight_rgb: [...]}
  covariance_rgb: [[...], [...], [...]]
  skew_density_nodes: [...]
  kurtosis_density_nodes: [...]
  mtf_cycles_per_mm: [...]
  mtf_response_rgb: [...]
```

## 5. Halation V2

### 5.1 乳剂内部散射

在 layer exposure 上先做能量守恒的 core/tail 混合：

```text
E^s_c = (1-s_c) E-_c
      + s_c [(1-w_c) (G_sigma_c * E-_c) + w_c (Exp_lambda_c * E-_c)]
```

`G` 和 `Exp` 都归一。该步骤不是红色 halo，而是乳剂内 wavelength-dependent
irradiation/分辨率损失；它作用于所有曝光，只因后续特性曲线而呈现非线性观感。
首版 profile 若无测量数据，`s/sigma/lambda` 全部标记为 modelled，默认应轻微。

### 5.2 背反射 source

不再从 Rec.2020 Y 生成一张灰度图。**触发门也必须是逐层的**（R1）：当前实现用
`Y_scene = flat @ REC2020_LUMA` 这一个标量决定「够不够亮到产生 halation」，而高饱和
的蓝光、绿光、霓虹和 LED 的 photometric Y 可能并不高，却会把某一层曝到极深。用亮度
开门等于让这类光源根本进不了 halation。

```text
EV_c = log2(E^s_c / E_ref_c)                  # 每层各自的归一化曝光
q_i,c = smootherstep(t0_i,c, t1_i,c, EV_c)
U_i,c = E^s_c * q_i,c
```

`E_ref_c` 是该层对 18% 灰的曝光，由 observer 和 stock anchor 决定，不是图像统计量。
返回哪些层由 §5.3 的 transfer matrix 决定——**是否触发**归各层自己管，**返回什么
颜色**归矩阵管，两件事不能由同一个亮度标量兼任。

使用宽度不小于 0.5 EV 的 smootherstep，保证 source gate 为 C1，不出现硬阈值轮廓。
阈值相对各层的 18% 参考曝光固定；禁止按每张照片的最大值重标定。

RAW clip 证据应进入 source confidence，而不是把重建后的死白当无限强光：

```text
U_reliable = U(scene estimate)
U_clipped  = U(conservative lower-bound/capped reconstruction)
U = lerp(U_reliable, U_clipped, cfa_clip_mask)
```

首版可以只做 cap，不尝试从剪切像素猜绝对光源亮度；报告必须注明 clipped-source 状态。

### 5.3 分层、多尺度回注

每个径向 component 拥有非负 layer transfer matrix：

```text
H_i,c(x) = sum_j A_i[c,j] * (K_i,j * U_i,j)(x)
```

回注必须是**相对均匀标定条件的空间残差**，不能是纯加性（R1）：

```text
H_dc,i,c = sum_j A_i[c,j] * U_i,j          # 同一算子在 DC 条件下的响应
E'_c = E^s_c + amount * sum_i (H_i,c - H_dc,i,c)
```

理由是标定口径。特性曲线来自真实胶片对**大块均匀 patch** 的感光测试，在那个条件下
乳剂内反射和片基背反射已经达到 DC 平衡——散射出去的光被邻域散射进来的光补上，这份
增益**已经烘进了曲线**。初稿的 `E' = E^s + amount * sum H` 把它再加一遍，等于同一个
物理过程既在曲线里、又作为空间效果叠加。

这不是理论顾虑，P0 已经量到：现行实现在 standard 档对整帧红通道注入 **+0.95%** 的
额外能量（`tests/optics_freeze/BASELINE.json` 的 `halation/standard/energy_ratio`）。
表现就是暖色漂移、大片亮部整体抬亮、画面发雾，而不只是边缘光晕。

残差形式下，均匀场恒等（`K * U = U` 时残差为 0），只有反差边界才产生回注。实现上还
需要两条约束：

- **非负下限**：残差在光源核心为负（核心把光散给了邻域）。层曝光必须保持非负，
  且负的部分不得超过该像素自身的 `E^s_c`；
- **返回能量守恒**：`sum_x (H_i,c - H_dc,i,c)` 在帧内应为 0（到代理精度），
  与 §6.2 的守恒散射同一口径。

如果确实需要一个整体抬亮的加性 glow，它属于 **editorial look**，走 §6.1 的 capture
bloom 通道并如此声明，不能挂在「胶片物理重建」名下。

至少有三类 component：

| component | 作用 | 颜色/半径要求 |
|---|---|---|
| local | 高反差边缘附近的主 halo | R 主导，G 次之；G 核更窄或门槛更高，使内圈偏橙 |
| global | 低幅、较大范围的红层 glare | 主要 R，不能把全画面染成均匀红雾 |
| aura | 去 remjet/弱防光晕结构的大半径返回 | 正常现代负片 profile 默认为 0 |

径向核可用归一的 Gaussian/exponential 混合；选择依据是径向拟合误差，不预设
“Gaussian 更物理”。普通 35 mm 强防光晕 profile 的 local 搜索范围先放在几十 um
量级；当前 `0.55 mm` 只允许成为 `aura` 候选，不能继续作为所有胶片的主 halo。
这里的范围是模型搜索区，不是测量结论。

### 5.4 背景与颜色控制的语义

物理默认让返回曝光自由传播，明背景上因相对反差低而自然不明显。高级
`background visibility` 是明确的 editorial gate：

```text
B = 1 - beta * smootherstep(b0, b1, log2((K_bg * Y_scene)/0.18))
E' = E^s + B * H
```

`beta=0` 是物理默认。`Hue` 不在输出 RGB 上旋转色相，而是约束 `A_i` 中 G 层相对
R 层的回注；Blue Compensation 只影响可见性 gate，不改 source energy。

## 6. Bloom 与 print-medium scatter 分家

### 6.1 Editorial capture bloom

GUI 的 `Bloom` 改为 scene exposure 驱动的局部 glow。它模拟的是镜头杂散光与乳剂
放大的综合观感，不声称是某一种相纸的测量属性。

先由场景亮度提取 source：

```text
q = smootherstep(source_ev0, source_ev1, log2(Y_scene/0.18))
S0 = S * q
```

“源尺寸”和“扩散半径”必须分开，而且**检测本身要做成尺度空间**（R1）。一个检测半径
加一组固定扩散核，无法同时合理处理灯丝/点状反光、霓虹管/细线高光、大窗户/天空过曝
这三类源——它们会套上尺寸相同的 glow，这正是「所有高光扩散尺度相近」的来源。

```text
S_i = scene_rgb * smootherstep(t0_i, t1_i, log2((G_sigma_i * Y_scene)/0.18))
G   = sum_i K_i * (w_i(E, i) * S_i)
```

- `G_sigma_i` 是一组递增的检测低通尺度（首版 3–4 级，成本可控）；小源只在最细的
  几级存活，大面积过曝在每一级都存活；
- `w_i(E, i)` 让**更亮、更大**的源平滑地激活更宽的尾部，而不是让所有光源共用一个
  半径。它是单调、有界、连续的，参数变化保持 C1；
- `K_i` 与 `S_i` 一一对应，仍然连续、非负、归一。

`G_sigma_0 = 0` 允许点源参与。GUI 名称直接写“最小光源尺寸”，避免复用 Dehancer
`Details` 的歧义；尺度空间是内部结构，不额外暴露 4 个滑块。

高光保护定义为可测试的 core protection，而不是随意 tone map。初稿写的
`G_surround = max(G - k*S, 0)` 虽然不会出负值，但 `max` 只有 C0，会在大面积亮区
边缘形成硬环甚至空心 halo，且与 §10.1 门 4 要求的「参数变化至少 C1」直接冲突。
改为按源与扩散场的**比例**做平滑核心抑制（R1）：

```text
r = S / (G + eps)                              # 源在本地扩散场中的占比
w_core = 1 - save_lights * smootherstep(r0, r1, r)
Delta = G * w_core
S_out = S_scene + impact * background_gate * Delta
```

- `save_lights=0`：`w_core = 1`，源核心和周围都可增亮；
- `save_lights=1`：源核心（`r` 高）被平滑压到接近 0，halo 形状保持连续；
- `saturation` 在 Rec.2020 luminance axis 上缩放 glow chroma，保持亮度与非负；
- 该算子在 film observer 前执行，让特性曲线自然压住被抬高的高光。

公式属于 editorial 模型，不要求全局能量守恒。其不变量是：无 source 时恒等、amount=0
恒等、输出有限非负、阈值和参数变化连续。

### 6.2 正介质 formation scatter

B1 与 timing 得到的 log paper exposure 必须先还原线性曝光：

```text
P_c = 2^(LEP2_c + tau_c)
P'_c = (1-s_c) P_c + s_c (K_form,c * P_c)
D+_c = C+_c(log2(P'_c))
```

这是能量守恒、归一 PSF 的介质属性；均匀 patch 必须不变，因此不会破坏现有 Stage B
中性/色头标定。它属于 `print_profile`，不由用户 Bloom slider 随意改变。

### 6.3 Viewing scatter

反射相纸在观看/测量时还有一次 substrate scatter，而且论文模型指出该阶段非线性。
首版只预留 `view_scatter_profile`，默认关闭；拿不到 paper PSF/CTF 数据前，不用当前
display-linear `spread-source` 假装已完成物理建模。电影正片的投影机 flare、扫描器
MTF 和观看环境 glare 继续属于外部系统，不写进 2383 介质 profile。

## 7. 资产与 plan 结构

### 7.1 资产拆分

不再有一份全局 `OpticsProfile`。至少拆成：

```text
stock optics asset
  - negative/reversal grain
  - emulsion scatter
  - backing halation
  - anti-halation class

print-medium optics asset
  - formation scatter
  - positive-medium grain
  - optional viewing scatter

editorial bloom preset
  - source gate
  - source-size detector
  - diffusion kernel
  - save-lights/saturation defaults
```

每份资产包含 schema version、单位、source URL、提取方式、原始文件 SHA-256、作者/许可、
`measured/derived/modelled/editorial` 字段。厂家 PDF 与第三方扫描原则上不入库，只保存
许可明确的派生数值、提取脚本和来源链接。

### 7.2 编译后的运行计划

资产与用户 modifier 在 plan 编译阶段合成不可变的 `FilmOpticsPlan`：

```python
FilmOpticsPlan(
    geometry,
    negative_grain,
    positive_grain,
    emulsion_scatter,
    halation_components,
    capture_bloom,
    print_formation_scatter,
    viewing_scatter,
    seed,
    provenance,
)
```

运行时代码只吃完整 plan，不查 GUI 字符串，不临时决定默认值。stock/medium 不支持某项
时写 `None` 并走严格恒等，不偷用别的 stock 的 profile。

### 7.3 用户 modifier

默认界面继续保持轻量：

- 颗粒：amount；
- Halation：amount；
- Bloom：amount；
- 效果隔离预览按钮。

高级面板只提供能对应失败模式的修饰：

| 效果 | 高级控制 | 修改的内部量 |
|---|---|---|
| 颗粒 | 尺寸、粗糙度、色度、阴/中/亮分布 | PSD 尺度、fine/cloud 比、通道协方差、`sigma(D)` modifier |
| Halation | 光源门槛、半径、global/aura、橙红比例 | source EV、component 半径、component amount、layer matrix |
| Bloom | 光源门槛、最小光源尺寸、扩散、保高光、饱和度 | source gate、detector radius、kernel scale、core protection、chroma |

所有 profile 参数以物理默认值为中心作乘法或 EV 偏移，不让 slider 的 0--1 直接成为
毫无单位的算法常数。

## 8. RAW 证据与自动判断的边界

空间效果不能做“把每张图调到看起来差不多”的内容归一化。可以使用的自动信息只有：

- CFA clip mask：标出光源估计不可靠区域；
- highlight reconstruction 模式：决定 clipped source 的 conservative cap；
- 真实输出 geometry：决定 film-plane 到像素的比例；
- stock/medium/gauge：选择 profile；
- 用户胶片曝光：像真实曝光一样进入 source 与颗粒密度。

不得使用整图 p99、最大值或主体中位数去自动改变 halo/bloom amount。相同的 scene exposure
进入同一 stock，必须产生同样的乳剂响应；这与 dngscan 的“尊重 RAW 数字化光学信息”
一致。

## 9. 诊断与标定工具

新增 `tools/calibrate_film_optics.py`，只处理公开/用户自备的标定数据，不进入运行依赖。

### 9.1 Grain 标定

输入：平场胶片扫描或 digitized datasheet granularity/MTF。

输出：

- 每通道 `sigma_D(D)`；
- 2D/径向 PSD、fine/cloud/mottle 拟合；
- RGB covariance、skew、kurtosis；
- 48 um aperture 回算误差；
- MTF 与建议粒径范围。

没有扫描时只做 datasheet RMS + modelled PSD，不生成虚假的“measured stock profile”。

### 9.2 Halation 标定

输入：同卷胶片拍摄的白/R/G/B 点源与硬边，至少覆盖 +1/+2/+4/+6 EV，并记录画幅、
stock、防光晕背层状态、扫描器与曝光。

流程：

1. 扣除 scanner/lens 基础 PSF；
2. 在暗背景上提取径向 excess；
3. 联合拟合 local/global/aura 核与 layer transfer；
4. 检查内圈/外圈 hue path；
5. 用未参与拟合的源尺寸和曝光验证。

输出径向图、半能量半径、每层积分能量与残差。只有一张普通照片不能升级为 measured。

### 9.3 Bloom 与介质散射标定

- editorial bloom 用合成 emitter chart 调整，不冒充测量；
- print formation scatter 需要介质曝光 edge/MTF 或 PSF 数据；
- reflective paper viewing scatter 需要 paper PSF/CTF，必须与 formation 阶段分开拟合。

## 10. 测试矩阵

### 10.1 数学测试

1. 所有 amount=0 返回同一对象/逐字节恒等；
2. 归一 scatter 对无限/周期均匀场严格恒等；
3. kernel 非负、积分为 1、半径换算与输出分辨率无关；
4. source gate 与参数变化至少 C1；
5. halation 回注非负，普通 profile 无 aura，No-Remjet profile 才允许大半径；
6. 白点源 halo 内圈 R+G、外圈 R 占比上升；蓝点源不能无条件生成与白点相同的灰源；
7. grain 在每个 density 节点重现目标 RMS、PSD、covariance 与高阶统计；
8. 48 um aperture 重算误差：measured profile <= 5%，modelled profile <= 10%；
9. 固定 seed 的 crop/旋转/线程/tile 一致；preview 是 full 的面积平均，统计容差沿用现有合同；
10. 所有阶段无 NaN/Inf、无负 exposure，关闭空间效果不新增全帧缓冲。

R1 新增：

11. **DC 中性**。任意均匀场经 halation 与乳剂散射后逐像素恒等；真实图上整帧
    per-channel `energy_ratio` 的绝对值 <= 1e-3（P0 基线是 +9.5e-3，见
    `tests/optics_freeze/BASELINE.json`）。
12. **曝光拓扑恒等**。`film_exposure_ev` 前移后，空间效果全关时输出与冻结的
    legacy 输出逐像素等价（`tests/optics_freeze/`）；空间效果开启时，改变
    `film_exposure_ev` 必须改变 halation 的触发面积与回注能量（单调、非零）。
13. **传递函数预算**。同一介质的 `MTF_explicit * K_residual` 在所有测量频点上复现
    `MTF_measured`，误差 <= 5%；`MTF_explicit < MTF_measured` 时构建期报错。
14. **颗粒坐标**。目标 granularity 在**最终 status 密度**上复算，而不是在 `amounts`
    扰动上；measured profile <= 5%，modelled <= 10%（与门 8 同口径，但明确了测点）。
15. **颗粒不改平均**。amount 从 0 扫到 1，均匀 patch 的最终显示输出均值漂移
    <= 0.2/255（每通道），证明 Jensen 偏差已被 `bias(D)` 吸收。
16. **逐层触发**。高饱和蓝光源（photometric Y 低、蓝层曝光高）必须产生 halation；
    与等 Y 的中性源相比，其 source mask 面积不得为零。
17. **保高光连续**。`save_lights` 在 [0,1] 上扫描时，输出对参数的一阶差分有界，
    大面积亮区边缘不出现空心环（径向轮廓单调不反转）。
18. **Bloom 尺度分离**。同一组参数下，点源、细线源、大面积源的 halo 半能量半径
    之比不小于 2:1（证明尺度空间检测生效，而不是所有源同一半径）。

### 10.2 合成视觉图

固定生成以下线性测试图，不能用手工 JPEG 作为唯一 golden：

- 白/R/G/B emitter，直径覆盖 1/2/4/8/32 px 对应的 film-plane 尺寸；
- emitter EV 为 +1/+2/+4/+6，背景 EV 为 -6/-3/0；
- 黑白硬边、斜边、Siemens star、细枝/窗格；
- Dmin 到 Dmax 的 21 阶三通道密度平场；
- 35 mm 与 16 mm 相同物理核的相对画面尺寸比较。

自动输出：source mask、local/global/aura、bloom source/diffusion、grain-only、radial profile、
PSD、MTF 和效果前后差值。没有隔离图的视觉验收不算完成。

### 10.3 真实照片验收

至少覆盖：

- 白天逆光人像：皮肤不能被 global halation 均匀染红；
- 夜间点光源：halo 贴边、内橙外红，不能形成半画幅红雾；
- 室内亮窗：Bloom 能表现大光源，但 Save Lights 不让窗核心硬剪；
- 霓虹/彩灯：source color 和层响应可区分，不能全部变成同一红圈；
- 平坦墙面/天空：颗粒有层次但不呈单频“磨砂”或周期纹；
- 高细节织物/树叶：粒径与细节截止协同，不是锐图叠噪点。

每张固定输出 `off / legacy / v2` 三联。`legacy` 只作为开发期 A/B oracle，不成为长期
公开模式；V2 通过验收后删除旧实现与旧 asset。

## 11. 流式、内存与性能

### 11.1 Spread 算子

保留当前最大长边 2048 的低频 spread grid，但改成多通道、可复用的 scale-space basis：

- area-decimate 线性 source；
- 同一 source 的 Gaussian/exponential basis 每个尺度只算一次；
- local/global/aura 通过 profile 权重组合，不各自持有完整临时图；
- 大 sigma 使用 O(N) IIR 或 FFT，小 sigma 使用 separable FIR；
- 禁止 nearest-neighbour pyramid expand，最终一律按面积/双线性合同回采样；
- 边界策略固定为 reflect/declared gate extension，不做逐张全局 renormalize 造成画框依赖。

### 11.2 Grain

- particle oracle 只在小图与校准工具运行；
- fast model 继续在物理坐标采样；
- 多频场不能简单把当前 144 MB master/integral 乘以频带数；优先共享基础随机场并按尺度
  派生，或用 counter-based tile field + halo；
- cache key 必须包含 profile hash、gate、seed model version；
- `seed=auto` 只改变空间 realization，不改变 RMS/PSD/profile。

### 11.3 性能门

- optics off：输出逐字节不变，速度回归 <= 2%；
- 默认 512 MiB 档：61 MP 三效果额外峰值仍 <= 512 MiB；
- V2 standard 在同机同图上不得慢于当前 standard 的 1.25 倍；
- 预览允许使用同一物理 source 的低分辨率 area version，但不得重新生成另一份颗粒；
- C++/Metal 只在 NumPy 小图 oracle 和统计门稳定后实现，不能先用近似 GPU kernel 固化错误。

## 12. GUI、报告与命名迁移

### 12.1 GUI

默认 `关闭 / 轻微 / 建模默认 / 自定义` 四档继续存在，但显示 profile 摘要，例如：

```text
35 mm · 强防光晕层 · modelled
```

自定义面板横置于预览旁，并提供四个隔离按钮：`原图 / 颗粒 / Halation / Bloom`。隔离态
只改变显示，不改变导出 plan。高级参数放折叠区，不把十几个控制全部铺在主界面。

### 12.2 CLI 与 plan 迁移

现有 `--film-grain/--film-halation/--film-bloom` 保持 amount 语义。新增 profile/compiler
字段时，旧设置按以下方式迁移：

- `film_grain` -> 同 amount 的 negative + available positive profile；
- `film_halation` -> stock 对应正常/No-Remjet profile 的 energy modifier；
- `film_bloom` -> editorial capture bloom；
- 旧 conservative bloom 不自动映射，改名为内部 `legacy_print_scatter`，开发验收后删除。

报告分别写出：stock optics、print-medium optics、editorial bloom、effective seed 和每项
provenance。不得只输出“模拟光学 standard”。

## 13. 分阶段实施

### P0：基线与诊断工具

- 冻结当前 `legacy` 输出与 5 张真实 RAW；
- 建 emitter/edge/density 合成图生成器；
- 增加 radial profile、PSD、MTF、mask-only 报告；
- 记录当前三个 amount 档的半径、能量、RMS 与 61 MP 性能。

退出门：现有视觉问题能被数字复现，而不只靠“看起来不好”。

### P1：asset/compiler 与语义拆分

- 引入 stock/print/editorial 三类 optics asset v2；
- 编译不可变 `FilmOpticsPlan`；
- 把当前 `bloom` 内部改称 `legacy_print_scatter`，但暂不改变公开输出；
- 所有未知 asset/version fail closed，amount=0 快路径不回归；
- **§3.1 曝光拓扑前移**（R1）：`film_exposure_ev + anchor` 从 `characteristic_amounts`
  的 `ev_offset` 移到 layer exposure 上。这是代数恒等的重排，放在 P1 是因为它必须
  先于任何 halation 改动落地，否则 P2 的 source 仍然看不见曝光。

退出门：尚未启用 V2 算子时，当前输出逐字节一致（`tests/optics_freeze/`，门 12）。

### P2：Halation V2（第一视觉优先级）

- layer-exposure source、**逐层** C1 source gate（R1 §5.2）；
- emulsion core/tail scatter；
- **残差回注**而非纯加性（R1 §5.3）；
- local/global/aura component 与 layer matrices；
- 传递函数预算：乳剂散射登记进 `MTF_explicit`（R1 §4.4.1）；
- RAW clip confidence/cap；
- 先发布 `35mm_strong_ah_modelled` 与 `35mm_no_remjet_editorial` 两个诚实 profile。

退出门：夜间点源内橙外红、普通 profile 无宽红雾、日景肤色无全局红漂，
门 11/13/16 通过。

### P3：Bloom 与 medium scatter 分家

- 新 scene-linear editorial bloom；
- **尺度空间 source 检测**（R1 §6.1）、diffusion、比例式 Save Lights、saturation；
- `film_bloom` 改指 capture bloom；旧的 post-B2 守恒算子更名
  `legacy_print_scatter`，保留作验收对照，**不再由任何用户档位可达**；
- viewing scatter 只留 asset/plan 空位，默认关闭。

> **P3 实施记录（2026-08-10）：§6.2 的 formation scatter 推迟到 P5**，与 §5.1 的
> 乳剂散射合并处理。两者是同一类算子：都在**全分辨率**的线性曝光上做小半径
> （2–12 µm）卷积，都无法用 ≤2048 的降采样扩散网格表示（其网格约 18 µm/格），
> 因此都需要行带路径支持 halo 行——那套基础设施 §11.3 本来就排在 P5。此外两者
> 的尺度都需要实测 MTF 才能定，而当前资产里没有任何一份 measured MTF。先建一次
> halo 行带路径、再把两个算子一起放上去，比在 P3 里为其中一个单独发明一套协议
> 更省也更诚实。门 13 因此在 P3/P4 期间为空转：没有显式散射，`MTF_explicit`
> 即恒等。

退出门：亮窗、点灯、树叶天空三种 source size 可独立控制（门 18）；Save Lights 全程
连续无空心环（门 17）；Bloom 不再通过暗化核心来“守恒”。

### P4：Grain V2

- particle oracle；
- multi-band fast model 与 `sigma(D)`/covariance；
- **染料量↔status 密度的坐标转换**（R1 §4.3.2）与 **`bias(D)` 平均透过率补偿**
  （R1 §4.3.1）；
- 负片 + 正介质双颗粒；
- Kodak 可得 stock 的 48 um RMS/MTF digitizer；
- preview/crop/full 物理坐标与统计一致。

退出门：合成平场达到 RMS/PSD/高阶统计门（门 14），amount 扫描不改变平均影调
（门 15），真实照片不再呈单尺度磨砂噪声。

### P5：GUI、性能与正式迁移

- profile 摘要、隔离视图和高级控制；
- **halo 行带路径**，然后在其上实现 §5.1 乳剂 core/tail 散射与 §6.2 正介质
  formation scatter，并接入 §4.4.1 的传递函数预算（门 13 到此才有可断言的内容）；
- IIR/FFT/C++ 优化；
- 全量真实 RAW A/B 与文档示例重渲；
- V2 默认后删除 legacy 算子、旧 `MODELLED_DEFAULT` 与临时开关。

退出门：功能、视觉、性能、内存、文档和 provenance 同时完成；不能只以“效果更明显”合并。

## 14. 明确暂不声称的内容

1. 不声称复刻 Dehancer 或 Filmbox 的内部算法；公开文档只用于行为和标定方法参考。
2. 没有实测扫描时，不声称某个半径/粒径是特定 stock 的真实值。
3. 不把镜头 bloom、片基 halation、相纸 scatter、扫描器 flare 合成一个“胶片 glow”。
4. 不用厂家一条 RMS granularity 曲线推断完整 PSD 后再称为 measured。
5. 不把视觉更强等同于更物理；默认 profile 可以克制，高级/editorial preset 可以明显。
6. （R1）不把厂家 MTF 当作可以与显式散射相乘的独立模块——它是整块材料的系统响应。
7. （R1）不声称 status 密度域的 granularity 数字可以直接加在染料量坐标上。

## 15. 许可边界

Dehancer 与 Filmbox 是闭源商业软件，只参考其公开说明和可观察行为，不复制参数表、
资产或反编译实现。Spektrafilm 与 IPOL 示例为 GPL-3.0 系，和本项目许可证兼容；若后续
直接移植任何代码，必须保留 SPDX、作者归属和来源提交，不能只在设计文档里提到。
Kodak/论文图表的数字化结果必须记录来源与提取方法，不把原 PDF、扫描样张或第三方
版权图像直接收入仓库。
