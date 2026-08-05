# 固定重建与热白平衡迁移

## 决策

用户白平衡不再参与每次 RAW 重建。文件、decoder、Core Image 版本、解拜耳、高光与 opcode 策略不变时，管线只建立一次 `DecodeContext`：

```text
RAW / black / gain map
  -> 固定 AsShot 解拜耳预条件
  -> demosaic + decoder highlight/opcode
  -> 线性 Rec.2020 DecodeContext
  -> 项目 WB / tint（hot）
  -> analysis / tone / gamut / output
```

固定 AsShot 增益只是重建预条件，不再等同于用户当前选择的色温。用户 WB 使用文件或机型的 `XYZ -> camera` ColorMatrix。令 `Cdecode/Ctarget` 为固定重建/目标白点的 `camera -> Rec.2020`（DNG Kelvin 模式按双光源标定插值目标矩阵），G-normalized 增益分别为 `Gdecode`、`Gtarget`，热阶段矩阵为：

```text
Mwb = Ctarget * Gtarget * inverse(Cdecode * Gdecode)
```

实现直接把这个共轭矩阵施加在固定线性 Rec.2020 scene 上；它等价于恢复 camera-linear 通道、替换增益再回到 Rec.2020，但不需要额外保存一份全尺寸三通道缓冲。负值与大于容器白的 headroom 使用 float32 保留，不在 WB 阶段 clip/quantize。

LibRaw 和 Apple RAW 都执行这一项目 WB。Apple 的 `neutralTemperature` 固定为 AsShot；因此 daylight 也不再需要映射到 Core Image 的不透明 temperature/tint 实现。缺少 ColorMatrix 或声明 WB multiplier 时明确退化为 camera AsShot，并在 `wb_degradation` 报告，不能猜测矩阵。

## 缓存边界

- 磁盘和内存 `DecodeContext` key：文件签名、LibRaw evidence runtime、decoder/版本、demosaic、highlight；不含用户 WB。
- `CaptureInvariant`：ceiling、noise、clip、CFA、SNR、mask/guidance，随固定 decode 一次生成。
- `BalanceContext`：WB 后 proxy、scene EV/gamut 指标与 RenderPlan；每个 WB 在一个 DecodeContext 内按需生成并走有界 LRU。
- camera WB 保留 decoder 原始 codes，不经过浮点 identity round-trip。
- 当前 export 对非 camera WB 仍在全分辨率 scene 上重算 percentile/gamut；不能误用 proxy 指标。后续可用 exact fused native analysis 缓存替换，但不得改变采样、阈值或 percentile 定义。

## 一致性边界

这次迁移有两个不同门禁，不能混为一谈。

1. **旧 oracle -> 新 oracle 是一次算法迁移。** 旧 LibRaw 会在 DHT/AHD/VNG 与高光处理前应用用户 WB，无法与后乘矩阵逐 bit 等价。迁移需 A/B 检查细节、moire、噪声、极端色温、饱和高光和负通道，并由人工审片确认固定细节结构符合产品选择。差异必须如实记录，不能称为纯缓存优化。
2. **新 oracle 冻结后要求严格一致。** preview/export 使用同一 `hot_wb_matrix_rec2020` 与同一 WB multiplier solve；同一 DecodeContext 的 memory/disk hit、进程重启、执行顺序和未来 CPU/Metal/CUDA backend，分支中间量逐 bit、最终 RGB8/RGB16/HDR 逐字节一致。任何 backend 不满足即关闭并回退 reference。

## 自动验证

- 矩阵单测：验证 `C·Gtarget·Gdecode^-1·C^-1`、相同 WB identity、G2 缺失、病态/空矩阵退化。
- 缓存单测：WB 不参与 DecodeContext identity；camera -> daylight -> camera 只调用一次 `load_raw/analyze`；同一 BalanceContext 命中同一对象。
- scene 单测：float32 保留 NaN 清理前的负值与大于 65535 headroom，不在 WB 阶段量化。
- decoder 集成：LibRaw 与 Core Image 的 `decode_wb` 均保持 camera AsShot，`applied_wb` 反映用户选择；daylight/Kelvin 都走项目 WB。
- golden 集：至少覆盖 Bayer DHT/AHD/VNG、X-Trans、clip/blend/reconstruct、3200/5500/9300K、daylight、sRGB/P3、正常曝光与饱和高光。记录旧/新 scene float diff 与最终输出 diff，人工审片通过后生成新 golden。
- 性能：在同一进程内先建立 camera DecodeContext，再轮换全部 WB。分别报告 decode 次数、WB matrix、scene reanalysis、plan/render 与端到端 p50/p95；禁止把磁盘 cache hit 冒充首次 prepare。

## 裁决记录（2026-08-04，盲测通过）

**协议**：双盲 A/B。旧 = 迁移前 main（e32af14），新 = 迁移 + C₀ 阶梯修复（#9 squash `0cf6f48` + `fc0dbcb`，即 PR #10 分支）。每对 A/B 随机盲序（seed 20260804），密钥在裁决后揭盲。全尺寸 SDR 导出，其余参数默认。第一轮曾误以未含 #10 的 main 作"新"侧——fp 样张的固定色温在其上静默退化为 AsShot，导致 96% 像素级伪差异——作废重做；这也再次证明 #10 是迁移可用的前提。

**样张与逐对结果**（差异统计为 8bit 码值）：

| 对 | 场景 | max / p99.9 / 变化像素>2 | 用户判决 | 揭盲 |
|---|---|---|---|---|
| 1 | X100VI RAF（X-Trans，零剪切）· 5500K · clip | 63 / 5 / 8.6% | 看不出 | — |
| 2 | fp `_SDI0150` 混合光 · 5500K · clip | 48 / 5 / 2.9% | 看不出 | — |
| 3 | 同上 · reconstruct | 55 / 7 / 5.2% | 看不出 | — |
| 4 | fp `_SDI0199` 舞台 ISO25600 · 3200K · clip | 245 / 15 / 4.3% | A 好 | A = 旧 |
| 5 | 同上 · reconstruct | 255 / 24 / 6.7% | B 好 | B = 新 |

**用户总评**：整体几乎完全一样，颜色差距不可见，核心差距呈现为噪点形态不同。

**结论**：同一场景两种高光模式下偏好指向相反两侧（clip 选旧、reconstruct 选新），结合"几乎全同"总评，判定为**无系统性方向**——差异属阈值附近的噪点口味，不构成旧路径优势。迁移语义**通过**：配平精度获视觉确认（与交叉验证 1e-5 量级吻合）；高光重建与解拜耳的路径依赖差异（含 X-Trans）均在"看不出"或"无方向偏好"之列。

**金标**：现有 golden/SDR 冻结全部走 camera 路径，迁移前后逐字节不变（三方 sha 验证），无需重冻结。上节"golden 集"要求的非 camera 覆盖，在本裁决通过后按新语义生成即为基线。

## C₀ 锚定统一（缝 A 修复，2026-08-04）

**缺陷**：`resolve_hot_wb_c0` 阶梯 rung 1 直接把 evidence `rgb_xyz_matrix`（LibRaw 的 `cam_xyz`，DNG 上源自 ColorMatrix2，固定锚在其标定光源 ~D65）用作解码侧 C₀，而固定色温模式的目标侧从文件双光源标定在 target_cct 处插值——`Ctarget·Gtarget·(C₀·Gdecode)⁻¹` 两侧光源锚不一致（合成标定实测锚差 ~1500 K；fp 实拍标定上按帧可达 ~3000 K，见下）。#10 引入的 AsShot-CCT 不动点（`wb.asshot_reference_cct`）只在 rung 3 生效，携带可用 evidence 矩阵的健康 DNG 永远走不到。

**决策**（用户 2026-08-04 拍板）："后续管线可能调用 C₀，正确性优先"——修。rung 1 改为：文件存在 DNG 颜色标定时，解码侧 C₀ 也从标定在 AsShot CCT 处插值（与 rung 3 同机制），目标侧维持 target_cct 插值，两侧同为未归一 DNG 约定，锚一致（source 标签 `evidence+cct`）；无标定或非 Kelvin 目标（daylight）时 evidence 矩阵同时作两侧（source `evidence`，对角增益可交换，约定自洽）——绝不出现一侧插值一侧 evidence 的混用（那正是 rung 2 文档警告的隐性白平衡偏移）。AsShot-CCT 不动点解算失败时回退 evidence 两侧而非退化 camera。rung 2/3/4 不动。

**量化**（fp `_SDI0150` / `_SDI0199`，5500K 与 3200K）：

- **本仓库 fp 样张实测为零差**：fp 的 `rgb_xyz_matrix` evidence 全零，两张样张全部走 rung 2（`color_matrix`），修复对其为 no-op——半尺寸场景缓冲修复前后逐像素浮点差 p50/p99/p99.9/max 全部为 0（bit 相同），全尺寸输出 JPEG（camera/5500K/3200K 全部模式）sha256 逐字节不变。camera 路径 sha 不变的硬门禁同时满足。
- **rung 1 实际生效量级（模拟测量）**：以 ColorMatrix2（即 DNG 上 LibRaw `cam_xyz` 的来源，D65 锚）替身 evidence 矩阵、在同一 fp 半尺寸场景像素上对比修复前后变换：`_SDI0150`（AsShot-CCT 解得 3491 K，与 D65 锚差 3013 K）5500K 相对归一逐像素差 p50 0.0092 / p99 0.352 / p99.9 0.941 / max 1.447，3200K p50 0.0061 / p99 0.176 / p99.9 0.700 / max 0.700；`_SDI0199`（AsShot-CCT 6724 K，超出 cct2=6504 K 被 clamp 到 ColorMatrix2 本身）两模式差恒为 0——锚差越大、场景越暖，缝越宽。

**同批合入——rung 2 采纳门（缝 B）**：rawpy 的 `color_matrix` 读的是 LibRaw 采纳门*之前*的嵌入 `cmatrix`（钉扎 `identify.cpp`：仅 DNG 容器且 `cmatrix[0][0] > 0.125` 才 memcpy 进 `rgb_cam`，非 DNG 永不采纳，被拒时解码走恒等色彩 `raw_color=1`）。rung 2 现按同一判据设门（`metadata.is_dng_container` 判 IFD0 DNGVersion 标签 + 阈值 0.125），不满足则落到 rung 3/4 或显式降级，绝不用解码器从未施加过的矩阵建 C₀。fp/iPhone 等 DNG（`cmatrix[0][0]≈1.3–1.4`）通过采纳门，行为不变，由上面的 sha 验证顺带覆盖。

**与原裁决的关系**：上节盲测裁决（2026-08-04）针对的是修复前行为；其全部样张（fp 走 rung 2、X100VI RAF 无 DNG 标定走 rung 1 evidence 两侧）在本修复下逐字节不变，裁决对这些路径继续有效。但对真正命中 rung 1+标定的机型（evidence 矩阵非零且带双光源标定的 DNG，如 Adobe 转制 DNG），模拟量级（相对归一 p99.9 最高 ~0.94）明显超出原裁决所见差异量级（8bit p99.9 ≤ 24）；如后续在此类文件上量化出的数字超出原裁决量级，需用户重新过目——此判断留给用户在 PR review 时作出。
