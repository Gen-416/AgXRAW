# AgXRAW 代码审查交接文档

**生成日期**：2026-09-02  
**审查对象**：`main` @ `408abc8`（`origin/main` 同步）  
**审查方法**：multi-agent 并行 + 集成接缝交叉验证；**代码直读优先**，不以测试绿灯作正确性证明。  
**用途**：交给负责修代码的 CLI / 工程师，按优先级逐项处理。

---

## 1. 仓库与近期变更上下文

### 1.1 相关 PR / commit

| PR / 主题 | Commit | 说明 |
|-----------|--------|------|
| #158 halation 细源门 (P5f) | `59c0049` | gate 升全分辨率，修复亚格光源「只扣不还」 |
| #159 审查批 21（十项） | `6e8b3d0` | HDR 亮度腿、门控 evidence、导出命名、胶片计划契约、缓存键等 |
| #160 仅色度降噪 v1 | `408abc8` | digitization repair，`chroma_nr` CLI 拨盘 |

### 1.2 权威契约文档（审查时对照）

- `docs/ARCHITECTURE.md` / `docs/ARCHITECTURE.zh-CN.md`
- `docs/ENGINEERING_NOTES.zh-CN.md`
- `docs/HDR_AGX_V2_IMPLEMENTATION_PLAN.zh-CN.md`
- `docs/FILM_OPTICS_V2_PLAN.zh-CN.md`
- `docs/FILM_OBSERVATION_PLAN.zh-CN.md`
- **勿将** `docs/archived/REVIEW_FINDINGS.md` 当作现行 HDR 契约

### 1.3 已有审查测试锚点

- `tests/test_review_batch21.py` — 批 21 十项修复的 pin
- `tests/test_halation_fine_source.py` — P5f 能量守恒与 band 不变性
- `tests/test_chroma_nr.py` — 色度 NR 算子单测（不含 CLI/GUI 集成）

### 1.4 高风险集成接缝（修 bug 时优先对账）

1. Plan 编译不可变 ↔ 下游是否 mutate  
2. Preview plan ↔ Export plan 是否同一 `build_render_plan` 路径  
3. Preview pixels ↔ Export pixels（proxy 下采样、optics tier、chroma_nr）  
4. SDR ↔ HDR 参数对称（`view_brightness`、`punch`、formation 顺序）  
5. Fingerprint / cache key ↔ 实际 render 参数完备性  
6. NumPy oracle ↔ C++ native（矩阵精度、dispatch 排除表）  
7. 文档 ↔ 代码（尤其 film-full HDR、色度 NR 产品边界）

---

## 2. 审查批 21 — 已修复项（勿重复修，可作回归基准）

以下十项在 `main` 上**已落地**；`tests/test_review_batch21.py` 有对应 pin。

| # | 原问题 | 状态 | 关键代码 |
|---|--------|------|----------|
| 1 | Ultrahdr HDR 腿丢失 `view_brightness`，gain map 编码成「HDR 更暗」 | **FIXED** | `dngscan/hdr_curve.py:28-58` `body_brightness_power`；`121-123` `_apply_body_brightness`；`hdr_agx_plan.py:272+` shoulder 锚点链式法则 |
| 2 | gated linear 不传 `analysis`，与 u8 路径 guidance 不一致 | **FIXED**（主路径） | `render.py:574`、`745`；**残留见 §3.2** |
| 3 | `export_plan_fingerprint` 无 input；6 hex 碰撞 | **FIXED** | `gui/service.py:2482-2483` `input_path`+`input_size`；`2146` hash 12 hex |
| 4 | audit `medium_id` 硬编码 `"print_paper"` | **FIXED** | `tone.py:1256-1266` `_default_medium_for`；`film_develop.py:1061` |
| 5 | CLI `film_mode=full` 无 preset 静默变 observe | **FIXED** | `tone.py:920-926` compile fail-closed |
| 6 | 未烘焙 `--film-print-medium` 拖到像素路径才失败 | **FIXED** | `tone.py:1198-1210` compile 校验 |
| 7 | `auto_ev` 默认 `film_crossover="off"` vs compiler `None` sentinel | **FIXED** | `auto_ev.py:258` 等 `str \| None = None` |
| 8 | `ToneCompressionPlan` 可变，cached plan 可被污染 | **FIXED** | `models.py:251` `@dataclass(frozen=True)` |
| 9 | `scene_render_to_hdr_display_linear` 接受 full 计划 | **FIXED** | `hdr_agx.py:262-274`；`chroma_nr>0` 亦拒（275-282） |
| 10 | preview pixel key 漏 optics budget tier | **FIXED** | `gui/service.py:936-941` |

---

## 3. 仍开放 / 部分修复的问题

### 3.1 P0 — 阻断（必须先修）

#### R-P0-1：`--ev auto` 因 `chroma_nr` 签名不匹配而崩溃

- **现象**：`python -m dngscan ... --ev auto` → `TypeError: resolve_export_ev() got an unexpected keyword argument 'chroma_nr'`
- **根因**：PR #160 在 CLI 传入 `chroma_nr`，未贯通 auto-EV 链
- **证据**：
  - 调用方：`dngscan/cli.py:1159`（`resolve_export_ev(..., chroma_nr=args.chroma_nr)`）
  - 被调方：`dngscan/auto_ev.py:664-711` `resolve_export_ev` **无** `chroma_nr` 参数；内部 `compute_auto_ev` / `max_safe_ev` 亦无
  - 对比：`build_render_plan` 已支持 `chroma_nr`（`tone.py:913-987`）
- **测试缺口**：`tests/` 内无 `--ev auto` 端到端；`test_auto_ev.py` 自组 kwargs，看不到 CLI 漂移
- **修复建议**：
  1. `resolve_export_ev` → `compute_auto_ev` → `max_safe_ev` → reference `build_render_plan` 全链加 `chroma_nr: float = 0.0`
  2. 色度 NR 为 Y-neutral pre-pass，auto-EV probe 结果理论上不变，但仍应传同一值保 parity
  3. 新增测试：CLI kwargs 与 `resolve_export_ev` 签名绑定，或最小 `--ev auto` smoke

---

### 3.2 P1 — 高优先级

#### R-P1-1：色度 NR 工作集未纳入 §9.3 optics 内存 tier

- **位置**：`dngscan/render.py:485-539` `_prepare_chroma_nr_map`
- **现象**：1408/2048 spread grid 上额外 ~116–246 MiB peak（`chroma_nr.py` à-trous 中间态 + float64 accumulator）；在 pass A 前分配，贯穿 pass A/B
- **关联**：`_OPTICS_FIXED_MIB = 72+160+48+68`（`render.py:328`）未含 chroma_nr；`_optics_band_rows`（331-341） band 高度偏乐观
- **对比**：#158 更新了 batch-13 RSS；#160 **未**扩展 `tests/test_review_batch13.py`
- **修复建议**：更新 `_OPTICS_FIXED_MIB` + batch-13 独立进程 RSS 门

#### R-P1-2：色度 NR 与「无场景自适应降噪」产品契约冲突

- **位置**：`dngscan/chroma_nr.py:147-157`
- **现象**：阈值 = `amount * k * MAD(detail)`，每图 per-level per-channel 自适应；同噪声在不同构图下处理不同
- **产品表述**：README/架构强调 demosaic 仅插值、无降噪；即便称 digitization repair，仍是 content-adaptive
- **修复建议**：产品决策 — (a) 改文档/CLI 说明接受 adaptive repair，或 (b) 改算法为固定阈值/标定曲线

#### R-P1-3：色度 NR 无 GUI / preview 通路

- **位置**：
  - `dngscan/gui/service.py:700-830`、`1185-1230` — `build_render_plan` 未传 `chroma_nr`
  - `gui/service.py:836-941` `_preview_pixel_key` 未含
  - `gui/service.py:2481+` `export_plan_fingerprint` 未含
- **现象**：CLI `--chroma-nr` 可用；GUI 永远 0；API 传 `chromaNr` 被静默忽略
- **HDR 边界**：`hdr_agx.py:275-282`、`522-526` 已拒 `chroma_nr>0` 的 AgX HDR / pair（v1 仅 SDR）— 一致
- **修复建议**：GUI 拨盘 + service 转发 + fingerprint/preview key

#### R-P1-4：preview proxy 下 chroma_nr band 尺度错误

- **位置**：`render.py:532-538`；`preview_cache.py:549-566`
- **现象**：`decimation_factor = max(h,w)/max(dh,dw)` 用 **render 尺寸** vs spread grid；1600px proxy 把 proxy 像素当 sensor 像素，相对全尺寸 export 的「8–128px band」偏移
- **修复建议**：band 计算需含 sensor→render 比例，或 preview 标注「NR band 非 export 等价」

#### R-P1-5：preview 亚像素 halation 与 export 仍不等价（设计边界，非 regression）

- **位置**：`preview_cache.py:549` `downsample_mean` 在 pipeline 前；P5f 在 export 侧恢复亚像素 halo 能量
- **文档**：`docs/FILM_OPTICS_V2_PLAN.zh-CN.md` P5f 注 ⑤ 已记为信息边界
- **修复建议**：产品 sign-off — 接受 preview 不显示亚像素 halo，或改 preview 策略（代价高）

#### R-P1-6：halation P5f 典型帧性能仍差（数学正确，mask 收益有限）

- **位置**：`film_develop.py:613-688` 行级候选 mask
- **现象**：Portra400 `t0min=3.2 EV` → floor ≈ 0.909 scene-linear（~2.34 EV over 18%）；含近白像素的行仍全宽跑 Stage A；61MP ~14–18s/frame
- **对比**：512 行 band 内 40 点 adversarial 测试 favorable，不代表普通照片
- **修复建议**：像素级 mask（`e_lin` 仅对 candidate 索引计算，其余 src=0，byte-identical）；float32 `stage_a_log_exposure`；合并 per-component `area_decimate_rows`

---

### 3.3 P2 — 中优先级

#### R-P2-1：`scene_render_to_agx_u8` 仍不传 `analysis`

- **位置**：`render.py:672-694`
- **现象**：legacy 公开 API；gated plan 经此入口仍缺 sensor SNR guidance
- **修复**：一行加 `analysis=` 参数并下传（与 #2 主路径补洞一致）

#### R-P2-2：fingerprint 仍缺 mtime / content hash

- **位置**：`gui/service.py:2482-2483` 仅 `input_path` + `input_size`
- **现象**：同路径同大小原位替换 RAW → 撞名覆盖
- **对比**：preview cache 用更强 identity（`preview_cache.py:220-240` mtime/inode/header hash）

#### R-P2-3：legacy `look`/`filter` API 与 export 命名不一致

- **位置**：
  - 渲染：`parse_grade` → `resolve_grade_params`（`service.py:1322-1323`）
  - 命名/fingerprint：`grade_id = params.get("grade", "none")`（`2429-2430`、`2494-2495`）
- **现象**：只发 legacy `look` 时渲染有 look、指纹为 `grade=none`，路径碰撞
- **修复**：fingerprint/suffix 用 `resolve_grade_params` 解析结果

#### R-P2-4：色度 NR 宣称 8–128 sensor px band 实际上界可被突破

- **位置**：`chroma_nr.py:46-54` 文档 vs `58-73` `atrous_levels_for`
- **现象**：decimation 6.8 时 level 4 约 109–218 px；memory tier 改变 band
- **修复**：clamp top level 或全文案改为 tier-dependent

#### R-P2-5：零 luma chroma 修正经 film optics 后可能变 Y

- **位置**：`render.py:460-470`（NR 在 optics 前）；`chroma_nr.py:161-166` 点wise Y 投影
- **现象**：halation/bloom 非线性可把 chroma-only 修正转成亮度变化；单测只 cover map，未 cover 集成

#### R-P2-6：HDR native 矩阵仍为 float32，SDR 已 f64

- **位置**：
  - SDR：`cpp/src/output_core.cpp:43` `mat3_exact_f64`；`output_core.h:21-22`
  - HDR：`cpp/include/dngscan_fast/hdr_core.h:34-36` float；`hdr_core.cpp:201` `mat3`
  - NumPy oracle：`color.py:85-90` float64 累加
- **现象**：73% 像素 max abs diff ~2.4e-6（float32 链）；HDR test `MAX_ABS_TOL=2e-4` 不 fail
- **修复**：HDR plan 增 f64 矩阵字段，对齐 #107 SDR 路径

#### R-P2-7：auto grain seed 依赖 preview LRU，驱逐后 export 重 mint

- **位置**：`preview_cache.py:37-38`、`850-854`；`service.py:2290-2303`
- **现象**：preview 与 export 颗粒可分歧

#### R-P2-8：film-optics 按 band 独立 dither，budget tier 变则字节变

- **位置**：`render.py:958-964`
- **现象**：`DNGSCAN_OPTICS_BUDGET_MIB` 影响 band 边界与 RNG 顺序；linear→quantize parity 可破

#### R-P2-9：native gamut fit 与 NumPy 差 1 code

- **位置**：`fast_plan.py:122-139` 预合并 Oklab + 1e-4 容差

---

### 3.4 P3 — 低优先级 / 技术债

| ID | 问题 | 位置 |
|----|------|------|
| R-P3-1 | `chroma_nr` 未进 fingerprint（GUI 未接线前 latent） | `service.py:2481+` |
| R-P3-2 | NumPy ≥2 为 native bit-exact 隐式前提；依赖仍 `numpy>=1.24` | `color.py:85-90`；`pyproject.toml:14` |
| R-P3-3 | ARCHITECTURE 写「HDR 不消费 SDR 像素」；film-full HDR decode `base_u8` | `ARCHITECTURE.md:205-207`；`hdr_agx.py:414-488` |
| R-P3-4 | ENGINEERING_NOTES 仍写不传输 halation；film-full 已实现 | `ENGINEERING_NOTES.zh-CN.md:17-20` |
| R-P3-5 | FILM_OPTICS plan P0–P5 complete 但 clip-confidence 未接 halation source | `FILM_OPTICS_V2_PLAN.zh-CN.md` |
| R-P3-6 | HDR colour-head 注释三处不一致 | `hdr_agx.py:179-182` vs `189-195` vs `_fast.py:149-154` |
| R-P3-7 | 色头 exclusion 过宽（full 模式 / gain_lms None 时 operator 为 identity 仍拒 native） | `_fast.py:84-90`、`149-154` |
| R-P3-8 | `--punch` 与 preset 共用字段，preset 清零时用户 dial 一并丢失 | `tone.py:957` + `film_curve.py:337` |
| R-P3-9 | `export.py`/`render.py` fallback `build_render_plan` 丢 film kwargs（仅 `tone_plan=None` API） | `export.py:322+` |
| R-P3-10 | ABI changelog 落后 `NATIVE_ABI_VERSION=9` | `fast_plan.py:23-27` |
| R-P3-11 | archive HEIC 用 JPEG 校准 round-trip 容差（若仍适用需确认） | gainmap 路径 |
| R-P3-12 | `finish_maps` / `hal_prep` 无 assert 防 plan mismatch（fine path） | `film_develop.py:665-667` |
| R-P3-13 | auto-EV probe 在 decimated grid 上，export 用 P5f 全分辨率 halation | `auto_ev.py:378-385` |

---

## 4. 色度降噪 v1（PR #160）管线说明

**意图**：digitization repair — 去低频色斑，亮度与细彩噪按构造不动。

**运行时顺序**（`render.py`）：

```
scene intent → scene transform → clip retreat → chroma NR map 构建+应用 → film/AgX tone → optics pass A/B → quantize
```

**关键文件**：

| 文件 | 职责 |
|------|------|
| `dngscan/chroma_nr.py` | à-trous 色度收缩、零 luma 投影、`apply_chroma_correction_*` |
| `dngscan/render.py:485-539` | spread grid 上建 correction map |
| `dngscan/tone.py:913-987` | plan 字段 `chroma_nr` [0,1] |
| `dngscan/models.py:419-423` | `ToneCompressionPlan.chroma_nr` |
| `dngscan/cli.py` | `--chroma-nr` 拨盘 |
| `dngscan/hdr_agx.py:275+` | HDR 路径拒 chroma_nr（v1 SDR-only） |

**已有单测**：`tests/test_chroma_nr.py`（11 项，算子级；不含 CLI/GUI/内存 tier）

---

## 5. Halation 细源 P5f（PR #158）说明

**问题**：decimated spread grid 上 gate，全分辨率 residual 扣费 → 亚像素高光「只扣不还」。

**修复**：`begin_halation_source` / `accumulate_halation_source`（`film_develop.py`）在全分辨率 gate + area-decimate 源；`finish_maps` 只 blur。

**能量**：give/take 相对误差 <0.14%（bloom on 亚像素源）；flat field 严格守恒。

**接线**：`render.py:443-468` — bloom map 先于 halation finish；与 `film_optics.py:988-1009` API 一致。

**测试**：`tests/test_halation_fine_source.py`；`test_review_batch13.py` RSS 门（+34 MiB @ tier 512）。

---

## 6. 集成接缝矩阵（修复后快照）

| 接缝 | 状态 | 备注 |
|------|------|------|
| SDR ↔ HDR `view_brightness` | ✅ 已闭合 | `body_brightness_power` |
| SDR ↔ HDR formation_tail（punch/色头） | ✅ 已闭合 | |
| Preview ↔ Export plan kwargs | ✅ 批 21 | |
| Preview ↔ Export pixels（halation 亚像素） | ⚠️ 不等价 | 文档化边界 |
| Preview ↔ Export pixels（chroma_nr） | ❌ 未接线 | |
| Fingerprint ↔ Render | ⚠️ 大幅改善 | mtime、legacy grade 仍有缝 |
| `--ev auto` CLI | ❌ P0 断裂 | chroma_nr 签名 |
| Film-full ↔ HDR 文档 | ⚠️ drift | 代码有意，文档需分路径 |
| Py ↔ C++ HDR 矩阵 | ⚠️ OPEN | f64 债 |
| NumPy ↔ C++ 色头 exclusion | ⚠️ 过宽 | 性能非正确性 |

---

## 7. 建议修复批次（给 CLI 的执行顺序）

### Batch 22 — 阻断（预计 1 PR）

1. **R-P0-1**：贯通 `chroma_nr` → auto-EV 全链 + 签名绑定测试  
2. 跑：`tests/test_review_batch21.py`、`tests/test_chroma_nr.py`、新 `--ev auto` smoke

### Batch 23 — chroma_nr 可 ship（预计 1–2 PR）

3. **R-P1-1**：`_OPTICS_FIXED_MIB` + batch-13 RSS  
4. **R-P1-3** + **R-P3-1**：GUI 接线 + fingerprint/preview key  
5. **R-P1-2**：产品/文档决策 + 算法或文案  
6. **R-P1-4** + **R-P2-4**：band 尺度与文档一致  
7. **R-P2-5**：optics 后 Y 漂移 — 测试或顺序调整

### Batch 24 — 技术债（可拆分）

8. **R-P2-6**：HDR native f64 矩阵  
9. **R-P1-6**：halation 像素级 mask + float32 log  
10. **R-P2-2** + **R-P2-3**：fingerprint 加固  
11. **R-P3-3** + **R-P3-4**：ARCHITECTURE / ENGINEERING_NOTES 更新  
12. **R-P2-1**：`scene_render_to_agx_u8` analysis 参数

---

## 8. 关键文件索引（按子系统）

```
dngscan/
  tone.py              build_render_plan、film 契约、chroma_nr 字段
  auto_ev.py           resolve_export_ev / compute_auto_ev（P0 断点）
  cli.py               --ev auto、--chroma-nr
  hdr_curve.py         body_brightness_power（批 21 #1）
  hdr_agx.py           HDR/film pair、chroma_nr 拒绝
  render.py            chroma_nr map、optics、_OPTICS_FIXED_MIB
  chroma_nr.py         色度 NR 算子
  film_develop.py      P5f halation 累积
  film_optics.py       halation 源/ spread API
  gui/service.py       fingerprint、preview key、export
  gui/preview_cache.py proxy 下采样、cache identity
  models.py            frozen ToneCompressionPlan
  _fast.py             native dispatch 排除
  fast_plan.py         ABI、native plan 编译
cpp/src/
  hdr_core.cpp         HDR float32 矩阵（P2-6）
  output_core.cpp      SDR f64 参考实现
tests/
  test_review_batch21.py
  test_halation_fine_source.py
  test_chroma_nr.py
  test_review_batch13.py   RSS / optics tier
```

---

## 9. 审查方法论备忘（后续轮次复用）

1. **Phase 0**：固定 branch/commit，列权威文档，排除 archived  
2. **Phase 1**：10 路 specialist 并行（Plan / SDR / HDR / Native / Film / Optics / GUI / Decoder / Test / Doc）  
3. **Phase 2**：5 条集成接缝 cross-review  
4. **Phase 3**：去重、P0–P3 仲裁、输出 handoff 文档  
5. **证据优先级**：代码 > 调用链 > 测试线索 > 文档  

---

## 10. 统计摘要

| 级别 | 已修复（批 21） | 仍开放 |
|------|-----------------|--------|
| P0 | 0（当时无 P0） | **1** |
| P1 | 6 | **6** |
| P2 | 4 | **9** |
| P3 | — | **13** |

**当前最 urgent**：R-P0-1（`--ev auto` + `chroma_nr`）。

---

*本文档由 2026-09-02 multi-agent 完整审查合成。修完 Batch 22 后建议重跑 `tests/test_review_batch21.py` 并补 Batch 22 专用 pin。*
