# 胶片外观层：P0 基线测量

> 状态：已冻结。本文件记录 reference-print 外观层引入之前，`observe` 与
> 用户可见默认 `full technical` 的实测色彩行为。数字由
> `tools/film_palette_probe.py` 生成，钉在
> `tests/appearance_freeze/BASELINE.json`，由
> `tests/test_film_appearance_p0.py` 复算。计划正文见
> [`FILM_APPEARANCE_RECIPE_PLAN.zh-CN.md`](FILM_APPEARANCE_RECIPE_PLAN.zh-CN.md)。

## 0. 测量合同

| 交付物 | 位置 |
|---|---|
| 24 hue × 4 个 Rec.2020 边界色度比例 × 7 scene EV、29 阶中性与 6 个命名 patch，共 743 样本 | `dngscan/film_palette_diag.py` |
| Oklab 分解、CIELAB/CIEDE2000、dEOK、sRGB/P3 gamut 压力 | 同上 |
| observe / technical / 跨 stock 对比 | `tools/film_palette_probe.py` |
| probe float32 与 4 场景 technical 渲染冻结 | `tests/appearance_freeze/` |
| 再生与校验 | `tools/regen_appearance_freeze.py` |

探针在 **B2 之后、gamut fit 之前**的公共 Rec.2020 中比较。色调计划由固定的
`daylight_wide_dr` 场景编译，不让测试体自己的 +6 EV 与整圈色相改变 plan。

本轮审查修正了初版 P0 的两个口径错误：

1. 初版显式传了 `film_crossover=datasheet`，冻结的是 opt-in native 分支；现改为
   `off`，即 GUI/CLI 默认的 `bounded` full technical；
2. 初版把绝对 Oklab `C` 称为纯度。线性曝光缩放会让 `L` 与 `C` 同时按立方根
   变化，因此现在分列 colorfulness `C` 与曝光不变的 saturation `S=C/L`，并同时
   报告 mapped output EV。

CIEDE2000 对 Sharma et al. 验证集吻合。gamut fit 前约 0% 到 4% 样本会带负 XYZ；
这些点不再伪装成有效 dE00，而是报告 dEOK，并单列 CIE 有效覆盖率。四卷的
observe-vs-technical 覆盖率为 96.2% 到 100%。

## 1. 最重要的结果：C-41 科内仍没有身份

| 组合（bounded full technical） | dE00 中位 | 肤色 | 叶绿 | 青天 | 洋红 |
|---|---:|---:|---:|---:|---:|
| **Portra 400 vs Ektar 100** | **0.46** | 0.80 | 0.49 | 0.42 | 0.37 |
| Portra 400 vs Vision3 250D | 4.51 | 4.59 | 4.54 | 3.97 | 5.65 |
| Portra 400 vs Velvia 100 | 3.54 | 5.46 | 5.41 | 2.13 | 2.26 |
| Velvia 100 vs Vision3 250D | 4.05 | 4.44 | 3.80 | 4.01 | 5.10 |
| Vision3 250D vs Ektar 100 | 4.43 | 4.49 | 4.30 | 3.86 | 5.75 |

Portra 400 与 Ektar 100 整体只有 0.46 dE00，低于计划给 stock identity 定的 2.0
下限。拆分后的总体中位为：

```text
色相差                 1.60 degrees
log2 saturation 比     +0.022
log2 colorfulness 比   +0.002
Oklab |delta L|         0.0031
mapped output EV 差    -0.0007 EV
```

两卷的亮度、colorfulness 和 saturation 几乎重合；区别主要是很小的选择性色相
变化。跨工艺家族已经有 3.5 到 4.5 dE00 的中位差异，所以 reference recipe 的首个
验收必须是 Portra/Ektar 这类**同家族分化**。只提高所有胶片的统一强度无法通过。

## 2. “full 偏弱”的准确含义：C-41 saturation 明显下降

下表是 observe -> bounded full 的 `log2(S_full/S_observe)`，其中 `S=C/L`：

| stock | 总体中位 | s=0.25 | s=0.50 | s=0.75 | s=1.00 |
|---|---:|---:|---:|---:|---:|
| Portra 400 | **-0.778** | -0.917 | -0.888 | -0.620 | -0.594 |
| Ektar 100 | **-0.915** | -0.984 | -0.972 | -0.813 | -0.844 |
| Vision3 250D | -0.041 | -0.152 | -0.056 | +0.026 | +0.107 |
| Velvia 100 | -0.051 | -0.162 | -0.057 | -0.013 | +0.055 |

Portra 的 `2^-0.778` 约为 **0.58 倍**，Ektar 约为 **0.53 倍**。Vision3 与
Velvia 则在总体上接近不变。结论比初版更明确：问题集中在当前 C-41 负片/相纸
组合，不是整个 full 核，也不是由输出亮度差伪造的绝对 C 变化。

因此 recipe 必须有 stock/medium 自己的基准场。用户看到的 strength 只能在该配方
上做相对缩放，不能是一条全局 saturation。

## 3. 差异主体在中间调，但高光尾部没有消失

observe vs technical 的 dE00 中位，按输入 **scene EV**：

| stock | -6 | -4 | -2 | 0 | +2 | +4 | +6 | +6 p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Portra 400 | 0.10 | 2.39 | 6.56 | **8.71** | 8.02 | 1.48 | 0.28 | **27.58** |
| Ektar 100 | 0.06 | 2.01 | 6.68 | **9.74** | 7.88 | 1.61 | 0.37 | **30.49** |
| Velvia 100 | 0.02 | 0.62 | 3.74 | 6.49 | 6.29 | 2.48 | 0.18 | **31.33** |
| Vision3 250D | 0.41 | 1.62 | 5.07 | 4.15 | 3.73 | 2.91 | 0.80 | **29.12** |

多数 +6 EV patch 已向白收敛，所以中位差小；但最高纯度尾部仍有 28 到 31 dE00。
原先“高光两模式几乎一致”的表述只看中位，属于错误概括。reference recipe 的主要
可见作用区仍应从中间调开始，但高光必须保留专项 hue/path-to-white 验收，不能假设
shoulder 已经天然等价。

## 4. 灰阶：默认 bounded 已接近中性

observe vs bounded technical 在 29 阶中性 ramp 上的 dE00：

| stock | 中位 | 最大 |
|---|---:|---:|
| Portra 400 | 0.12 | 1.07 |
| Ektar 100 | 0.09 | 0.57 |
| Vision3 250D | 0.08 | 0.49 |
| Velvia 100 | 0.11 | 0.94 |

这与初版使用 datasheet/native 得到的 1.5 到 8.1 完全不同，也证明为什么冻结对象
必须是实际默认。后续 `print-balanced` 会有意只锚定 EV0、保留两端 crossover；其
基线应与 native 分支单独比较，不能再把 native 数字写成 technical 默认。

## 5. Gamut 压力

sRGB 之外的样本比例（gamut fit 前）：

| stock | observe | bounded full |
|---|---:|---:|
| Portra 400 | 0.374 | **0.153** |
| Ektar 100 | 0.347 | **0.096** |
| Vision3 250D | 0.207 | 0.198 |
| Velvia 100 | 0.444 | 0.300 |

Display P3 对应为：Portra `0.316/0.077`、Ektar `0.280/0.048`、Vision3
`0.122/0.118`、Velvia `0.354/0.215`。P3 统计在初版中因错误使用不存在的
`P3D65` 矩阵键而不可调用，现已纳入门禁。

这些数字是压力表，不是配额。recipe 不能只靠把颜色推到 gamut fitter 外面制造
“浓”，但也不应把 observe 当前的较高压力错误地当成绝对上限。

## 6. Observe 配对仍然只是微调

| stock | scene_transform | strength | agx_primaries |
|---|---|---:|---|
| Portra 400 | `portra400_d55` | 1.3 | base |
| Ektar 100 | `ektar100_d55` | 1.4 | punchy |
| Velvia 100 | `velvia100_d55` | 1.6 | punchy |
| Vision3 250D | `vision3250d_d55` | 1.2 | muted |

Portra 与 Ektar 在 observe 下的 dE00 中位仍只有 0.61。不同前馈与 primaries 没有
形成清晰科内身份，继续放大这两个自由度不是 reference-print 的替代方案。

## 7. 冻结范围

| 内容 | 精度 |
|---|---|
| 4 stock × 2 mode 的 probe 输出 | float32，跨 BLAS 绝对容差 `1e-6`，另钉 dE00 `<1e-3` |
| 2 张真实照片裁切 + 2 个合成场景的 bounded full technical | u8 逐字节；linear float32 `atol=1e-6` |
| HDR 与通用 DRT | 沿用各自专项测试，不与 appearance freeze 交叉哈希 |

初版把整个 `tests/golden` 树做单一 SHA-256。新增一个无关场景也会触发“胶片
technical 漂移”，是假阳性；现改为只钉 appearance 自己的 probe 与 render fixtures。

`tools/regen_appearance_freeze.py --check` 现在会在缺文件、probe/render/report 漂移时
返回非零。初版虽然打印 drift，最终始终返回 0，不能作为检查门。

## 8. 对后续阶段的约束

1. 首个纵向切片必须同时做 Portra 400 与 Ektar 100，先通过同家族 identity 门；
2. appearance 的 exposure 索引必须从 Stage A 显式携带，不能由 B2 输出亮度反推；
3. purity/richness 使用 `S=C/L`，绝对 `C` 只叫 colorfulness；
4. Color Density 改变线性能量时要同时缩放 Oklab `L/a/b`，保持 `S` 与 hue；
5. 中间调是主要目标，但 +6 EV 高纯度尾部必须保留独立 path-to-white 验收；
6. `technical` 永远指当前 bounded 默认；native/datasheet 是单独的研究分支。
