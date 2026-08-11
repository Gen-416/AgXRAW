# spektrafilm 胶片 profile（原样引入）

来源：[spektrafilm / agx-emulsion](https://github.com/andreavolpato/agx-emulsion)
（Andrea Volpato），profile 数据许可为 **CC BY-SA 4.0**（见同目录
`SPEKTRAFILM_LICENSE.txt`），与本仓库代码的 GPL-3.0-or-later 分开适用。
上游从厂商数据手册与科学文献处理得到这些测量数据；原始数据版权属于各厂商。

本目录文件为**未修改的原样副本**（license 要求修改需在 CHANGELOG 中追踪；
当前无修改）。dngscan 的曲线拟合预设与前馈拟合目标由这些 profile 推导，
推导产物同样按 CC BY-SA 4.0 处理并在各自预设的 `source` 字段注明出处。

| 类别 | 文件 | 用途 |
|---|---|---|
| 拍摄负片（民用/专业） | portra_160/400/800（含 push1/2）、ektar_100、gold_200、ultramax_400、xtra_400、c200、pro_400h | 曲线预设（负片+配对相纸端到端）+ 前馈观察者 |
| 拍摄负片（电影） | vision3_50d/250d/200t/500t、verita_200d | 同上，配对 2383 印片；T 卷组合白平衡 3200K |
| 反转片 | provia_100f、velvia_100、ektachrome_100、kodachrome_64 | 曲线预设（直接正像模型）+ 前馈观察者 |
| 相纸/印片 | portra_endura、supra/ultra_endura、endura_premier、ektacolor_edge、crystal_archive_typeii、2383/2393 | 端到端合成的显示介质 |

全部 profile 为**未修改原样副本**；dngscan 的推导产物（曲线预设、前馈矩阵、SSF CSV）
按 CC BY-SA 4.0 处理，出处写入各预设 `source` 字段。反转片无相纸环节（自身即正像），
拟合残差系统性偏大并如实记录在 `fit.rms_stop`（AgX 曲线族对正片陡 S 的表达边界）。

`filters/schott_KG3.csv`：Schott KG3 热吸收滤镜透过率（同一上游 commit 的
`src/spektrafilm/data/filters/heat_absorbing/schott/KG3.csv` 原样副本），为
印相光源移植预置——上游 TH-KG3 = 3400K 黑体 × KG3（`model/illuminants.py`），
拟合器现用 3400K 黑体 × KG3 热吸收玻璃合成印片光源（A9 item 8 更正:此行曾停留在早期'纯 3200K 黑体'口径,与实现不符）。

引入日期：2026-07-29。上游 commit：`3bb2c2d2801ff68b92019cf1dbcbb133d60832bc`
（2026-08-05 复核：本目录 28 个 JSON 与该 commit 的
`src/spektrafilm/data/profiles/` 逐字节一致）。逐文件 SHA-256 见同目录
`MANIFEST.sha256`；重新同步时更新 commit、清单与本行。

## 发行边界（2026-08-05 立案）

wheel 只打包 `dngscan/`；本目录（及整个 `dngscan_assets/`）不进 wheel。当前
运行时已有两处对 `dngscan_assets` 的路径依赖（`display_filter` 的 vendor LUT、
`look.py` 的 look_fields）——源码运行正常、安装后的 wheel 缺文件，属先存缺口。
决策（第三阶段已执行）：项目自产的数据一律放入 `dngscan/data/`（随 wheel 发行）
——联合色头 npz 已就位，第四阶段 full LUT 同规。上述两处先存依赖经核查属**用户
自备数据**：vendor LUT 是专有文件（Resolve FPE 等）本就不可随包分发，look_fields
为用户本地测量；二者缺失时运行时均优雅降级（look 合并静默跳过、滤镜在使用时
明确报错），维持现状即为正确边界。
