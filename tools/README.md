# 工具一览 / Tools Index

均以 `python tools/<name>.py --help`(或文件头 docstring)为准。分组如下。

## 资产构建 / Asset Builders

| 脚本 | 作用 |
|---|---|
| `build_film_v2_assets.py` | 从光谱底座生成 film v2 stock/print/b2 资产 |
| `build_full_lut.py` | 烘焙 full 模式 65³ LUT(B1/B2)与观察者残差 |
| `build_film_appearance_recipes.py` | 生成外观层配方资产 |
| `build_film_appearance_identity.py` | 外观层 identity 参考资产 |
| `gen_film_v2_manifest.py` / `gen_film_optics_manifest.py` | 资产清单(哈希钉扎)再生成 |
| `fit_chroma_field.py` | 路线 C:Stage A 色度场 vs 3×3 的 held-out CV(`docs/chroma_field_cv.json`,资产选型依据)。runtime-faithful:每折用 `bake_lut`(与 builder 共用的纯烘焙)烘 LUT、经运行时调度求值;30 种子重复折按频率采纳;白板锚由参数化保证 |
| `fit_illuminant_tiers.py` | 路线 D:D55 光源假设的 held-out 代价 vs 同光源专档(`docs/illuminant_tier_cv.json`;运行时口径:白平衡像素经 D55 矩阵、按各模型自身白板归一;结论:不设档) |
| `export_film_ssf.py` | 导出胶片光谱敏感度数据 |
| `make_evidence_shell.py` | 证据壳:从 RAW 剥容器结构做 CI 元数据语料(CC0 来源,像素区剥除) |
| `build_native.sh` | 本地编译原生 kernel(产物勿入 wheel,见脚本头) |
| `build_libraw_master.sh` + `libraw-pin.env` | 固定版本 LibRaw 构建 |

## 数据导入 / Importers

| 脚本 | 作用 |
|---|---|
| `import_kodak_granularity.py` | Kodak 颗粒度图表数字化 → 颗粒 σ(D) 资产 |
| `import_kodak_mtf.py` | Kodak MTF 图表数字化 → 散射核拟合 |
| `import_cbld.py` | 用户本地 CBLD 黑电平参考导入(不随仓库分发) |
| `import_jptc.py` | JPTC/2 一手实测 CSV → PTC 拟合 priors 条目(`--self-test` 合成传感器门禁) |
| `import_p2p_pdr.py` | P2P 批量传感器表 → `data/priors/p2p_bulk.json`(135 台,许可状态见 NOTICE) |
| `import_dngshell.py` | 上游 DNGSHL1 壳 → dngscan evshell(一手拍摄语料接入,来源块入清单) |
| `import_jptc_collect.py` | JPTC collect 套件(dark/isogain/spectrum/ptc)→ 增益曲线+读噪曲线+白度/条带证据 |
| `import_lens_transmittance.py` | 一手镜头/滤镜光谱透过率 → `data/lens_transmittance.json`(118 条,380–755nm@1nm) |

## 校准与拟合 / Calibration & Fitting

| 脚本 | 作用 |
|---|---|
| `fit_film_curve.py` | 特性曲线拟合 |
| `fit_skin_window.py` / `calibrate_skin_matrix.py` | 肤色前馈窗口/矩阵标定 |
| `regenerate_material_presets.py` | 按各预设记录的目标 SSF 重生成全部材质/胶片分离前馈预设(窗口=实测反射率的色度真值) |
| `calibrate_raw9_anchors.py` | RAW9 对齐锚点标定 |
| `grain_particle_oracle.py` | 颗粒粒子 oracle(多带频谱拟合依据) |
| `spectral_base.py` | 光谱底座共享库 |

## 冻结与门禁 / Freezes & Gates

| 脚本 | 作用 |
|---|---|
| `sync_film_optics_from_charts.py` | 图表数字化 → 渲染资产编译器(`--check` 防陈旧;配 test_film_optics_chart_sync 门禁) |
| `audit_digitization.py` | 图表数字化采样充分性审计(线性 vs PCHIP 歧义 ≤ 声明误差;测试门禁共用) |
| `regen_appearance_freeze.py` | 外观冻结再生成/校验(`--check`) |
| `regen_optics_freeze.py` | 光学冻结 + BASELINE 再生成/校验(`--check`) |
| `regen_sdr_freeze.py` | SDR 冻结再生成/校验 |
| `regen_golden.py` | golden 语料再生成 |
| `regen_showcases.py` | 文档展示图整表重渲清单(NCC 裁切恢复、拼板重建) |

## 报告与探针 / Reports & Probes

| 脚本 | 作用 |
|---|---|
| `film_optics_report.py` | 光学算子逐项测量报告(§10.2 图表;`--perf` 61MP 计时) |
| `film_visibility_report.py` | 胶片可见性分级报告 |
| `film_palette_probe.py` | 外观层调色板探针 |
| `hdr_policy_probe.py` | HDR latitude 常数逐帧门控证据(重钉值的依据) |
| `corpus_report.py` | 样张语料批量报告 |
| `scan_drt_geometry.py` | DRT 几何扫描 |
| `crosscheck_2383.py` | 2383 印片资产交叉校验 |
| `validate_ideal_image.py` | 理想图像验证 |
| `pipeline_impact.py` | 管线改动影响评估 |

## A/B 与基准 / A/B & Benchmarks

| 脚本 | 作用 |
|---|---|
| `decode_ab.py` | LibRaw vs RAW9 解码 A/B |
| `hdr_ab.py` | SDR/HDR 对比图生成 |
| `benchmark_fast_backend.py` | 原生 kernel 基准 |
| `benchmark_realtime_preview.py` | 实时预览基准 |
| `benchmark_pipeline_completion.py` | 新进程 RAW 分析、AutoEV、SDR / float HDR / packed HDR 的精确身份与阶段时间对照 |
| `benchmark_phase_statistics.py` | 新进程 24/60MP uint16 CFA 噪声、SNR、health 的独立旧入口与共享有界 workspace；完整身份、阶段时间及 RSS |
| `benchmark_cli_delivery.py` | 从真实 RAW 运行完整 CLI；比较自动选参、最终 JPEG/HEIF 压缩内容、metadata 后文件与回读像素 |
| `benchmark_gainmap_search.py` | 固定 SDR/HDR 母版，记录 HEIF 候选、编码/回读次数，校验选参与压缩内容 |
| `benchmark_delivery_metrics.py` | 24MP base/coding 合并扫描及 HDR 多秩/workspace 的完整数值、时间与容量对照 |
| `benchmark_quantize_groups.py` | 固定完整 A/B 噪声下的 1M concat 与两个 500k slice，独立进程/外部 RSS 采样 |
| `benchmark_delivery_buffers.py` | HDR packing/readback 与 HEIF 输入平面的逐位及内存对照 |
| `benchmark_optional_render.py` | gated、非零 ChromaNR、RAW guidance 与 native 调用成本的独立测量 |
| `benchmark_pipeline_concurrency.py` | 真实 prepare/preview/isolated-export 并发，外部进程树采样、排队/执行时间及输出身份 |

GUI 白平衡工作集另见 `tests/benchmark_gui_cache_workset.py`，需显式运行，不随 unittest discovery 触发大图测量。非胶片效率优化的合同与实测口径见 [管线记录](../docs/performance-pipeline-completion.zh-CN.md)、[形成与交付](../docs/performance-render-delivery.zh-CN.md) 和 [GUI 缓存](../docs/performance-gui-cache.zh-CN.md)。
