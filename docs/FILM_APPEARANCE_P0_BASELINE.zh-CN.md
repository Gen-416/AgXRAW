# 胶片外观层 — P0 基线测量

> 状态：已冻结。本文件记录 reference-print 外观层引入之前，`observe` 与
> `full technical` 的**实测**色彩行为。数字由 `tools/film_palette_probe.py` 生成，
> 钉在 `tests/appearance_freeze/BASELINE.json`，由 `tests/test_film_appearance_p0.py`
> 复算。计划正文见
> [`FILM_APPEARANCE_RECIPE_PLAN.zh-CN.md`](FILM_APPEARANCE_RECIPE_PLAN.zh-CN.md)。

## 0. P0 交付了什么

| 交付物 | 位置 |
|---|---|
| Palette probe 体（24 hue × 4 chroma × 7 EV + 29 阶中性 + 6 个命名 patch，共 743 样本） | `dngscan/film_palette_diag.py` |
| Oklab 分解、CIELAB、CIEDE2000、gamut 压力 | 同上 |
| 探针工具（observe / technical / 跨 stock 三组对比） | `tools/film_palette_probe.py` |
| technical 冻结（probe 精确 float32 + 4 场景渲染字节 + golden 树哈希） | `tests/appearance_freeze/`（`tools/regen_appearance_freeze.py`） |
| 门禁 | `tests/test_film_appearance_p0.py` |

退出门是「能量化证明『差异弱』发生在哪些 hue/EV/chroma 区」。下面每条结论都有值、
有坐标，并在测试里被断言。

测量口径：探针在 **B2 之后、gamut fit 之前**的公共 Rec.2020 里比较，这正是计划
§12.1 要求外观层运行的空间——否则 sRGB 和 P3 会对同一配方给出不同几何。色调计划
由一个**固定参考场景**编译，不由探针体编译：探针故意包含 +6 EV 样本和整圈色相，
让自动编译器看见它会使 observe 和 full 的曝光/端点不一致，之后测到的每一个色差都会
被一个没人问的色调差异污染。

CIEDE2000 实现对 Sharma et al. 公布的 8 组验证值逐条吻合（`places=3`），所以
§15.2 的 dE00 门槛与外部可比。

## 1. 最重要的一条：科内没有身份

| 组合（full technical） | dE00 中位 | 肤色 | 叶绿 | 青天 | 洋红 |
|---|---|---|---|---|---|
| **Portra 400 vs Ektar 100** | **0.68** | 0.56 | 0.79 | 0.79 | 0.72 |
| Portra 400 vs Vision3 250D | 4.88 | 5.50 | 5.81 | 4.04 | 3.93 |
| Portra 400 vs Velvia 100 | 6.03 | 5.94 | 6.00 | 6.66 | 7.33 |
| Velvia 100 vs Vision3 250D | 7.56 | 8.78 | 8.80 | 7.09 | 6.89 |
| Vision3 250D vs Ektar 100 | 4.77 | 5.06 | 5.73 | 4.53 | 4.11 |

Portra 400 和 Ektar 100 是两卷性格差异出名的 C-41 负片，现在**整体差 0.68 dE00**，
低于计划 §15.2 给 stock identity 定的 2.0 下限。拆开看更清楚：

```
色相差 中位 5.0°   p95 24.9°
log2 chroma 比 中位 -0.016     （几乎没有饱和度差异）
Oklab L 差 中位 0.003          （几乎没有明度差异）
```

也就是说，这两卷在 full 里只差一个很小的色相旋转，chroma 和 lightness 上**没有区别**。

而跨科（负片 vs 反转片 vs 电影负片）已经 4.8–7.6 dE00，远在门槛之上。**所以外观层
要解决的不是「整体不够强」，而是「同一科内没有可区分的身份」。** 一个全局加饱和的
配方在这个诊断下只会让所有卷一起变浓，两条曲线仍然重合。

## 2. 「full 偏弱」的准确含义：C-41 负片丢了约三分之一 chroma

observe → full 的 Oklab chroma 比（log2，负值表示 full 更淡）：

| stock | 总体中位 | c=0.25 | c=0.5 | c=0.75 | c=1.0 |
|---|---|---|---|---|---|
| Ektar 100 | **−0.751** | −0.828 | −0.846 | −0.633 | −0.592 |
| Portra 400 | **−0.629** | −0.785 | −0.683 | −0.557 | −0.528 |
| Vision3 250D | +0.060 | +0.193 | +0.014 | +0.003 | +0.004 |
| Velvia 100 | +0.394 | +0.264 | +0.364 | +0.351 | +0.493 |

`−0.63` 是 2^−0.63 ≈ **0.65 倍**——Portra 在 full 下只有 observe 约三分之二的
Oklab chroma，Ektar 只有 0.59 倍。而 **Vision3 和 Velvia 没有这个现象**（分别 +0.06
和 +0.39）。

这条把「偏弱」定位到了具体家族：问题在 C-41 负片印相链，不是整条 full。任何全局
strength 都会把已经不弱的 Velvia 一起推过头。

按区域看（Portra）：叶绿 −0.83、洋红 −0.98、肤色 −0.66、青天 −0.45。洋红和叶绿掉得
最多，青天最少。

## 3. 差异集中在中间调，高光两模式几乎一致

observe vs technical 的 dE00 中位，按 EV：

| stock | −6 | −4 | −2 | 0 | +2 | +4 | +6 |
|---|---|---|---|---|---|---|---|
| Portra 400 | 2.7 | 3.3 | 6.8 | **9.3** | 7.8 | 1.6 | 0.2 |
| Ektar 100 | 2.5 | 2.9 | 7.4 | **11.1** | 7.7 | 1.5 | 0.4 |
| Velvia 100 | 7.8 | 7.6 | 6.5 | 7.1 | 6.3 | 3.7 | 1.9 |
| Vision3 250D | 1.5 | 2.9 | 5.1 | 4.2 | 4.1 | 3.9 | 1.9 |

两条负片在 EV+4/+6 几乎重合（0.2–1.6 dE00），在 EV0 相差 9–11。这对配方设计是直接
的：**外观层有作用空间的是中间调，不是肩部**。在 +6 EV 上写强 hue path 只会被两条
链共同的高光压缩吃掉。

Velvia 是唯一在深阴影仍有 7.8 dE00 的，与它的反转片直接观看链一致。

## 4. 灰阶：两模式对中性轴的判断本来就不同

observe vs technical 在 29 阶中性 ramp 上的 dE00：

| stock | 中位 | 最大 |
|---|---|---|
| Portra 400 | 1.45 | 2.71 |
| Ektar 100 | 1.86 | 3.90 |
| Vision3 250D | 2.52 | 6.92 |
| Velvia 100 | 2.28 | 8.09 |

这不是缺陷，是两条链对灰阶 crossover 的不同处理（technical 走 `bounded` 逐像素
中性化）。记录在这里是因为计划 §8 要新增 `print-balanced` 常数 balance：改动的
基线就是这几个值，改完之后灰阶两端应当**变得更不中性**（保留 native crossover），
而 EV0 必须精确中性。

## 5. Gamut 压力：full 比 observe 保守得多

sRGB 之外的样本占比（gamut fit 之前）：

| stock | observe | full |
|---|---|---|
| Portra 400 | 0.374 | **0.106** |
| Ektar 100 | 0.347 | **0.112** |
| Vision3 250D | 0.207 | 0.166 |
| Velvia 100 | 0.444 | 0.424 |

两条 C-41 负片在 full 下只有 observe 三分之一不到的越域比例。这与 §2 的 chroma
结论互相印证，也给 §15.2 的「越域比例不得无上限增长」定了起点：Portra 从 0.106 出发，
配方把它推到 observe 的 0.374 之内都还在历史范围里。

## 6. observe 的配对（作为对照记录）

| stock | scene_transform | strength | agx_primaries |
|---|---|---|---|
| Portra 400 | `portra400_d55` | 1.3 | base |
| Ektar 100 | `ektar100_d55` | 1.4 | punchy |
| Velvia 100 | `velvia100_d55` | 1.6 | punchy |
| Vision3 250D | `vision3250d_d55` | 1.2 | muted |

值得注意：observe 给 Portra 和 Ektar 配了**不同的**前馈和 primaries，两者在 observe
下的 dE00 中位仍然只有 0.65——和 full 的 0.68 一样低。所以「科内无身份」不是 full
独有的问题，`FILM_STYLE_PAIRINGS` 那两个自由度也没能把它们分开。这为计划 §3.1
「style pairing 只是微调」提供了数值支撑。

## 7. 冻结了什么

| 内容 | 精度 |
|---|---|
| 4 个 stock × 2 模式的 probe 输出（743×3） | float32 逐位 |
| 4 个场景的 full technical 渲染（2 张真实裁切 + 2 个合成） | u8 逐位；linear 按 float16 存储精度 |
| `tests/golden` 树哈希 | sha256 |

计划 §16 P0 写的是「四张真实 RAW」；仓库只有两张真实照片裁切，另两个是合成场景，
清单和本文如实记录，不把合成图当照片。HDR golden 不重复冻结——它已由
`tests/test_film_hdr.py` 与 `tests/test_hdr_delivery.py` 覆盖，这里通过 golden 树
哈希连带钉住。

## 8. 对计划的三条修正建议

基于以上测量，建议在进入 P1 之前调整计划的优先级表述：

1. **§10 首批 recipe 的顺序应改。** 计划把 Vision3 250D + 2383 放在第一位，但
   measurement 显示电影负片家族已经和别的家族分开（4.8–7.6 dE00），而
   **Portra/Ektar 这一对是唯一失败的**。首个 recipe 应该做 C-41 负片科内分化，
   因为它是唯一能证明「配方确实制造身份，而不只是加浓」的案例。
2. **§15.2 需要补一条科内门。** 现有门只要求「两个不同 stock reference 在各自目标
   色区 dE00 至少 2」。应明确其中至少一对必须是**同科**（如 Portra vs Ektar），
   否则这条门用现成的跨科差异就能通过，配方一行不写也算达标。
3. **「整体强度」不是主要抓手。** §2 显示只有 C-41 负片丢 chroma，Velvia 反而更浓。
   若首版把 `strength` 做成全局标量，Portra 调好时 Velvia 会过。建议 recipe 资产
   自带家族级基准，`strength` 只在其上做相对缩放。

这三条属于测量结论而不是设计偏好，是否采纳由计划持有者决定；未采纳时应记录理由。
