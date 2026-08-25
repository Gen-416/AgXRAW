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
| `import_lens_transmittance.py` | 一手镜头/滤镜光谱透过率 → `data/lens_transmittance.json`(118 条,380–755nm@1nm) |

## 校准与拟合 / Calibration & Fitting

| 脚本 | 作用 |
|---|---|
| `fit_film_curve.py` | 特性曲线拟合 |
| `fit_skin_window.py` / `calibrate_skin_matrix.py` | 肤色前馈窗口/矩阵标定 |
| `calibrate_raw9_anchors.py` | RAW9 对齐锚点标定 |
| `grain_particle_oracle.py` | 颗粒粒子 oracle(多带频谱拟合依据) |
| `spectral_base.py` | 光谱底座共享库 |

## 冻结与门禁 / Freezes & Gates

| 脚本 | 作用 |
|---|---|
| `audit_digitization.py` | 图表数字化采样充分性审计(线性 vs PCHIP 歧义 ≤ 声明误差;测试门禁共用) |
| `regen_appearance_freeze.py` | 外观冻结再生成/校验(`--check`) |
| `regen_optics_freeze.py` | 光学冻结 + BASELINE 再生成/校验(`--check`) |
| `regen_sdr_freeze.py` | SDR 冻结再生成/校验 |
| `regen_golden.py` | golden 语料再生成 |
|  文档展示图整表重渲清单(NCC 裁切恢复、拼板重建) | 文档展示图整表重渲清单(NCC 裁切恢复、拼板重建) |

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
