# 胶片模拟（observe）

[返回文档索引](README.md)

AgXRAW 内置 20 种胶片。每种胶片是一套**固定的、由数据决定的 AgX 渲染**：同一卷胶片在每张照片上按同一规则成像，没有逐步显影旋钮。研究过程、拟合数据和被否定的假设都保存在 AgXFilm 的 `docs/FILM_BPC_PRINT_VIEW.zh-CN.md`；本文只说明当前行为。

## 使用

```bash
python -m dngscan photo.dng --film portra400 --jpeg out.jpg
python -m dngscan photo.dng --film velvia100 --film-grain 0.5 --jpeg out.jpg
python -m dngscan photo.dng --film list          # 列出可用胶片
```

GUI 后端任务参数为 `film`（胶片名，默认 `none`）与 `filmGrain`（0–3，默认 1.0）。本轮尚未在 GUI 页面加入选择控件。

| 参数 | 含义 |
| --- | --- |
| `--film STOCK` | `none` 关闭；其余为 Kodak Portra 160/400/800（含增感 1/2 档）、Ektar 100、Gold 200、UltraMax 400、Vision3 50D/200T/250D/500T、Verita 200D、Ektachrome 100、Kodachrome 64，Fujifilm Velvia 100、Provia 100F、Superia X-TRA 400、Pro 400H、C200 |
| `--film-grain` | 颗粒强度。1.0 = 该胶片数据手册的颗粒度（48 µm 光孔 RMS）；0 关闭 |

约束（不满足时直接报错，不静默降级）：只用于 SDR 输出；需要 `--tone-core agx`；不叠加 `--scene-transform`；AgX 原色固定为 base。旧版胶片参数（`--film-mode`、`--film-curve`、色头、冲洗、印相等）仍被拒绝。

导出文件名带 `film-<胶片名>`，GUI 导出指纹在开启胶片时加入胶片名、颗粒强度和资产哈希；不开胶片时指纹与以前完全相同。

## 成像过程

胶片在编译好的 tone plan 上做一次替换，再由 tone core 执行：

1. **曲线**：AgX C1 曲线换成该胶片的固定坐标。拟合目标是冻结的光谱链（负片 → 相纸，或反转片自身）在原生观看下的中性灰阶，做亮度黑点适应（相纸/片基的最大密度映射到显示黑），EV0 → 0.18。每卷最大拟合误差 ≤ 2.12 个 sRGB 码值。自动 punch、暗场提亮按胶片设定关闭，AgX 原色固定为 base。
2. **反转片测光中灰**：负片的中灰由印放约定在 18%；反转片没有印放环节，按 ISO 2240（高于最低密度 0.20 与 2.00 两点曝光量的几何平均）确定标准测光下的中灰，作为进入曲线前的场景曝光偏移：Velvia −0.27、Provia −0.41、Ektachrome −0.25、Kodachrome −0.29 EV。
3. **颗粒**：在负片染料密度上施加，`D' = D + σ(D)·F(x,y)`，再经该胶片自己的特性曲线换回曝光，交给 AgX 曲线。σ(D) 是数字化的 Kodak 5207 漫射 RMS 颗粒度曲线；F 是 36×24 mm 片门上的带限周期颗粒场，按每个像素的面积精确积分取样，用 48 µm 光孔标定。各胶片强度乘数据手册颗粒度比例（Ektar 0.56 到 Portra 800 1.31；组间对接的假设见资产说明）。颗粒相位由 RAW 文件内容决定，同一张 RAW 的预览、导出与重导出颗粒排布一致。
4. **按色相的颜色还原**：在 Oklab 中按色相 × 亮度 × 色度修正色相旋转、色度比例和亮度，系数拟合自冻结的完整光谱链（526 个色样，色相残差 1–2.4°）。全部胶片的平均饱和度保持 AgX 水平，各卷相对差异来自数据。
5. **灰阶层间漂移**：完整光谱链自带的逐通道灰阶偏色，按“印相平衡”语义（中灰中性、两端保留），按场景像素亮度查表，保持亮度不变。

自动曝光仍在场景域把主体中位放到 18%（只提亮，受输出高光上限约束）。胶片模式下跳过有界 AgX 联合曲线规划——胶片曲线是固定的。曝光探测走同一个 tone core，看到相同的曲线与颜色，不带颗粒（颗粒按构造保持均值）。

## 资产与代码

| 位置 | 内容 |
| --- | --- |
| `dngscan/data/film/observe_v1.json` | 唯一资产（schema `agxraw-film-observe/1`）：每卷曲线、测光偏移、颗粒比例、色相系数、层间漂移表、Stage A 观察矩阵与特性曲线表；公共颗粒模型及其重建校验值。由 AgXFilm `tools/export_agxraw_film_asset.py` 从研究数据与冻结集成导出 |
| `dngscan/film/__init__.py` | 资产加载、plan 替换（`apply_film_plan`）、tone core 包装（`film_tone_core`） |
| `dngscan/film/grain.py` | 片门几何、颗粒主场、面积积分取样、σ(D) 施加与特性曲线求逆（从冻结集成移植的 NumPy 实现） |
| `dngscan/film/color.py` | 色相还原与灰阶层间漂移 |
| `tests/test_film_observe.py` | 资产、主场逐值重建、plan 替换、中灰与黑点、颗粒分块不变性与均值保持、CLI/GUI 合同 |

移植验证：同一组场景颜色经合并后的 tone core 与研究时的冻结链路逐像素一致（线性误差 < 1e-6）；颗粒在均匀灰场上与研究实现逐像素一致（相对误差 ~1e-6）；颗粒主场的 48 µm 光孔 RMS 与冻结实现相同到 1e-8。

## 尚未包含

- GUI 页面与桌面端的胶片选择控件（任务参数已可用）。
- HDR 输出。
- 印相介质自身的颗粒、halation、介质散射；颗粒纹理目前全部胶片共用 5207 一份，只有强度按胶片区分。
- 颗粒用 Stage A 的 3×3 观察矩阵定位密度；冻结集成里部分胶片采用的色度场修正没有带入（只影响颗粒读 σ 的密度位置，且经同一张表求逆）。
