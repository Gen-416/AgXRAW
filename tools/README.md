# 开发、验证与数据工具 / Tools

正常转换图片请使用 [GUI 或 CLI](../README.zh-CN.md)。这里是开发、分析、性能测量和资产维护入口；脚本继续保留原路径，方便测试和已有研究记录引用。

所有命令从仓库根目录运行，`python` 指已安装项目依赖的 Python 环境。原生性能工具需要先编译 Rust 扩展；Apple RAW 和 HDR 容器工具需要 macOS。光谱拟合、图表审计另需 [校准依赖](../requirements-calibration.txt)，完整测试依赖与版本见 [CI](../.github/workflows/ci.yml)。

## 先按任务选择入口

| 要做的事 | 入口 | 结果 |
|---|---|---|
| 修改或验证 Rust 计算核 | [build_native.sh](build_native.sh)、[开发检查](#开发检查) | 原位扩展、ABI/self-test、NumPy/native 回归 |
| 检查 RAW 分析、AutoEV、SDR/HDR 是否改变 | [benchmark_pipeline_completion.py](benchmark_pipeline_completion.py) | 完整母版及决策身份、分阶段耗时；不含编码 |
| 比较 JPEG/HEIF 质量、色度采样和体积 | [benchmark_delivery.py](benchmark_delivery.py) | 同一母版的编码矩阵、回读误差、文件大小 |
| 验证完整 CLI 导出的压缩内容与元数据 | [benchmark_cli_delivery.py](benchmark_cli_delivery.py) | 真实 RAW 到最终文件、压缩内容和回读身份 |
| 测 GUI 预览、缓存和并发 | [GUI 性能工具](#gui-性能) | 预览耗时、缓存工作集、排队与执行时间 |
| 分析一批 RAW 的自动处理选择 | [corpus_report.py](corpus_report.py)、[hdr_policy_probe.py](hdr_policy_probe.py) | CSV / 逐帧分析报告 |
| 复现解码或 HDR 差异 | [decode_ab.py](decode_ab.py)、[hdr_ab.py](hdr_ab.py) | 诊断数据、对比图 |
| 更新测试基准、胶片资产或数据来源 | [数据与资产维护](#数据与资产维护) | 明确写入的资产、清单或测试夹具 |

大图报告和 RAW 派生图片放在仓库外。以下示例共用一个新的输出目录；长期保留结果时换成自己的目录：

```sh
AGXRAW_RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/agxraw-tools.XXXXXX")"
```

带参数解析器的工具可以用 `--help` 查看选项。**不要对全部脚本批量执行 `--help`**：部分资产生成器没有参数解析器，传入 `--help` 仍会执行写入。下文分别列出只读检查、重建入口和历史工具。

## 开发检查

```sh
# 使用当前 Python 环境编译原生核，并检查 ABI 与 self-test
PYTHON="$(command -v python)" bash tools/build_native.sh

# 与 CI 相同的两条回归路径
DNGSCAN_FAST=0 python -m unittest discover -s tests -q
DNGSCAN_FAST=1 python -m unittest discover -s tests -q

# Rust 内部测试
cargo test --manifest-path rust/Cargo.toml
```

日常修改先运行受影响的测试模块；以上全量命令适用于提交前的完整验证。GUI 事件还应在浏览器中操作一次，服务层测试不能替代文件选择、版本切换和预览交互检查。原位编译产物位于 `dngscan/`，发行 wheel 使用干净 checkout 构建。

[build_libraw_master.sh](build_libraw_master.sh) 是修复旧环境中 rawpy/LibRaw 依赖的入口，版本由 [libraw-pin.env](libraw-pin.env) 钉扎。正常安装已使用该依赖，不需要每次开发都重建：

```sh
DNGSCAN_VENV=/path/to/venv sh tools/build_libraw_master.sh
```

## 性能测量

先选定输入、解码器、图像尺寸和输出模式，再在独立进程中交替运行参考与当前路径。`--reference` 通常只关闭该工具研究的优化，**不等于关闭全部 Rust 核**；以各脚本说明为准。多数比较工具会拒绝覆盖已有报告，重复测量应使用新的文件名。

以下是无需 RAW 的小尺寸统计对照，可用于确认工具链工作；真实 24/60MP 性能需要相应尺寸和重复测量：

```sh
python tools/benchmark_phase_statistics.py --size 1000 800 --reference \
  --out "$AGXRAW_RUN_DIR/phase-reference.json"
python tools/benchmark_phase_statistics.py --size 1000 800 \
  --out "$AGXRAW_RUN_DIR/phase-current.json" \
  --compare "$AGXRAW_RUN_DIR/phase-reference.json"
```

真实 RAW 的完整成像链对照（把路径替换为自己的样张）：

```sh
python tools/benchmark_pipeline_completion.py --source /path/to/photo.DNG \
  --decoder libraw --mode sdr --out "$AGXRAW_RUN_DIR/sdr-reference.json"
python tools/benchmark_pipeline_completion.py --source /path/to/photo.DNG \
  --decoder libraw --mode sdr --optimized --out "$AGXRAW_RUN_DIR/sdr-current.json" \
  --compare "$AGXRAW_RUN_DIR/sdr-reference.json"
```

`--mode hdr` 测 float HDR，`--mode hdr-packed` 测打包 HDR；`--repo` 可选择不同 checkout，双方均需与各自代码匹配的扩展。成像身份、编码身份和完整文件身份是不同检查，不能只凭耗时判断正确性。各工具对计时、哈希和 RSS 的包含范围写在文件头；GUI 缓存字节也不等同于进程 RSS。

### RAW 分析与成像

| 工具 | 测量范围 |
|---|---|
| [benchmark_pipeline_completion.py](benchmark_pipeline_completion.py) | RAW 分析、AutoEV、SDR / float HDR / packed HDR 的完整身份和阶段时间 |
| [benchmark_sensor_summary.py](benchmark_sensor_summary.py) | 传感器统计复用；其余 native 核保持启用 |
| [benchmark_sensor_rgb.py](benchmark_sensor_rgb.py) | Bayer、X-Trans、线性 RGB 的传感器 RGB 裁切计数 |
| [benchmark_sensor_channels.py](benchmark_sensor_channels.py) | 传感器顶值检测与逐通道裁切计数 |
| [benchmark_phase_statistics.py](benchmark_phase_statistics.py) | 噪声、SNR、health 的独立旧入口与共享有界工作区 |
| [benchmark_deferred_masks.py](benchmark_deferred_masks.py) | 完整成像中延迟构建 clip masks 的影响 |
| [benchmark_loss_pipeline.py](benchmark_loss_pipeline.py) | loss 核的裁切、原位合并及真实 RAW 成像 |
| [benchmark_optional_render.py](benchmark_optional_render.py) | gated、非零 ChromaNR、RAW guidance 与 native 调用成本 |
| [benchmark_fast_backend.py](benchmark_fast_backend.py) | AgX native / NumPy 合成核与可选真实 DNG 渲染 |
| [benchmark_native_memory.py](benchmark_native_memory.py) | 单核独立进程耗时/RSS；`DNGSCAN_FAST=0` 选择 NumPy |
| [benchmark_quantize_groups.py](benchmark_quantize_groups.py) | 同一完整噪声输入下，拼接与分组量化的身份、时间、进程采样 |

### 编码与交付

```sh
# SDR：同一母版比较可调 JPEG / HEIF 编码矩阵
python tools/benchmark_delivery.py --mode sdr --encoders tunable \
  --out "$AGXRAW_RUN_DIR/codecs" /path/to/photo.DNG

# 一次真实 RAW 到 HEIF 的完整 CLI 导出及验证
python tools/benchmark_cli_delivery.py --repo "$PWD" --source /path/to/photo.DNG \
  --decoder libraw --format sdr-heic --out "$AGXRAW_RUN_DIR/export.heic" \
  --report "$AGXRAW_RUN_DIR/export.json"
```

| 工具 | 测量范围 |
|---|---|
| [benchmark_delivery.py](benchmark_delivery.py) | `tunable` 为可调编码矩阵；`auto` 为生产 HDR 自动搜索；`system` 保留早期系统编码对照 |
| [benchmark_cli_delivery.py](benchmark_cli_delivery.py) | 自动选参、压缩内容、元数据后文件、回读像素；`--compare` 比较报告，`--require-file-match` 额外要求完整文件相同 |
| [benchmark_gainmap_search.py](benchmark_gainmap_search.py) | 固定 SDR/HDR `.npy` 母版上的 HEIF 自动搜索；记录候选、编码与回读次数，支持 `--dry-run` |
| [benchmark_delivery_metrics.py](benchmark_delivery_metrics.py) | base/coding 扫描及 HDR 多秩统计工作区 |
| [benchmark_delivery_buffers.py](benchmark_delivery_buffers.py) | HDR packing/readback、HEIF 8/10-bit 输入平面的逐位与内存对照 |

### GUI 性能

| 工具 | 用法与边界 |
|---|---|
| [benchmark_realtime_preview.py](benchmark_realtime_preview.py) | `python tools/benchmark_realtime_preview.py /path/to/photo.DNG --iterations 30`；测固定尺寸预览，可指定场景和输出后端 |
| [benchmark_pipeline_concurrency.py](benchmark_pipeline_concurrency.py) | 用 `--source-a`、`--source-b` 和 `--out` 指定两个不同 RAW 与报告；真实 prepare/preview/export 并发，外部进程树采样 |
| [benchmark_gui_cache_workset.py](../tests/benchmark_gui_cache_workset.py) | `python tests/benchmark_gui_cache_workset.py --repo "$PWD" --source /path/to/photo.DNG --out "$AGXRAW_RUN_DIR/cache.json"`；白平衡工作集与磁盘缓存复用，不随 unittest 自动运行 |

并发与缓存基准测服务路径，不包含 HTTP、浏览器绘制或鼠标交互延迟。测量合同与已完成的证据见 [管线记录](../docs/reports/performance/performance-pipeline-completion.zh-CN.md)、[成像与交付](../docs/reports/performance/performance-render-delivery.zh-CN.md)、[GUI 缓存](../docs/reports/performance/performance-gui-cache.zh-CN.md)。

## 质量诊断与研究

```sh
python tools/corpus_report.py --dir /path/to/raws --csv "$AGXRAW_RUN_DIR/corpus.csv"
python tools/hdr_policy_probe.py /path/to/photo.DNG
python tools/decode_ab.py /path/to/photo.DNG --full --write-jpegs "$AGXRAW_RUN_DIR/decode"
python tools/hdr_ab.py /path/to/photo.DNG --out "$AGXRAW_RUN_DIR/hdr"
```

| 工具 | 用途 / 输入 |
|---|---|
| [corpus_report.py](corpus_report.py) | 一批 RAW 的自动处理选择；半尺寸解码，不导出成片 |
| [hdr_policy_probe.py](hdr_policy_probe.py) | 逐帧查看 HDR 可靠高光、SNR 门控、白点和 headroom |
| [decode_ab.py](decode_ab.py) | LibRaw / Apple RAW 解码与分析计划交叉对照；几何差异须结合报告解释 |
| [hdr_ab.py](hdr_ab.py) | SDR/HDR 诊断拼图；拼图本身不是 HDR 交付文件 |
| [pipeline_impact.py](pipeline_impact.py) | 一张 RAW 上的单项处理影响矩阵；包括 native / NumPy 和胶片选择 |
| [scan_drt_geometry.py](scan_drt_geometry.py) | 合成 EV × 色相 × 色度的 DRT 扫描，输出 `--csv` |
| [validate_ideal_image.py](validate_ideal_image.py) | `python tools/validate_ideal_image.py /path/to/ideal-image-pair`；有已知黑白电平、增益、噪声的合成 DNG 对照 |
| [film_palette_probe.py](film_palette_probe.py) / [film_visibility_report.py](film_visibility_report.py) | 胶片调色板 / 真实照片可见性；后者默认使用本地样张矩阵 |
| [film_optics_report.py](film_optics_report.py) / [grain_particle_oracle.py](grain_particle_oracle.py) | 胶片光学算子 / 颗粒统计依据；`film_optics_report.py --perf` 包含大图测量 |
| [crosscheck_2383.py](crosscheck_2383.py) | 2383 印片资产与外部参考的交叉检验；输入要求见脚本头 |
| [fit_chroma_field.py](fit_chroma_field.py) / [fit_illuminant_tiers.py](fit_illuminant_tiers.py) | 胶片 Stage A 色度场与光源假设的交叉验证；重算选型证据，会写研究 JSON |

## 数据与资产维护

这一组会读取外部测量数据或重建仓库中的资产。先确认要变更的数据来源和模型，再同时检查生成文件、清单与受影响测试。冻结夹具记录既定行为，不能靠重生成来消除未解释的测试失败。

### 只读核验与冻结

```sh
python tools/audit_digitization.py
python tools/sync_film_optics_from_charts.py --check
python tools/gen_film_optics_manifest.py --check
python tools/regen_appearance_freeze.py --check
python tools/regen_optics_freeze.py --check
```

| 工具 | 默认行为 |
|---|---|
| [audit_digitization.py](audit_digitization.py) | 只读图表采样充分性审计；测试共用同一检查 |
| [sync_film_optics_from_charts.py](sync_film_optics_from_charts.py) | 重编译图表派生的光学资产；`--check` 只读 |
| [regen_appearance_freeze.py](regen_appearance_freeze.py) / [regen_optics_freeze.py](regen_optics_freeze.py) | 重写对应冻结与测量基线；`--check` 只读 |
| [regen_sdr_freeze.py](regen_sdr_freeze.py) | 重写 SDR 冻结；没有 `--check`，核验用 `python -m unittest tests.test_sdr_freeze` |
| [regen_golden.py](regen_golden.py) | 用 NumPy 参考路径重写 golden；可从 DNG 生成裁切夹具 |

### 可复现构建与导入

| 工具 | 输入 → 输出 |
|---|---|
| [build_film_v2_assets.py](build_film_v2_assets.py) | 光谱底座 → 当前 stock / print-state / B2 资产；可选 `--stocks` |
| [gen_film_v2_manifest.py](gen_film_v2_manifest.py) / [gen_film_optics_manifest.py](gen_film_optics_manifest.py) | 已审查资产 → 哈希清单；光学清单支持 `--check` |
| [build_film_appearance_recipes.py](build_film_appearance_recipes.py) | 当前外观层配方定义 → 配方资产及清单 |
| [fit_film_curve.py](fit_film_curve.py) / [export_film_ssf.py](export_film_ssf.py) | 胶片特性 / 光谱资料 → 曲线拟合与敏感度数据 |
| [calibrate_skin_matrix.py](calibrate_skin_matrix.py) / [fit_skin_window.py](fit_skin_window.py) | 光谱 / 真实照片 → 前馈矩阵 / 窗口；窗口拟合默认只打印，`--write` 才写回 |
| [regenerate_material_presets.py](regenerate_material_presets.py) | 各预设记录的目标 SSF → 材质前馈预设；可用 `--out` 指定新文件 |
| [calibrate_raw9_anchors.py](calibrate_raw9_anchors.py) | 声明的本地相机样张集 → 解码器窗口锚点；会更新 `decoder_anchor_transport.json` |
| [scan_chart_curves.py](scan_chart_curves.py) | 本地 Kodak PDF → [chart_scans/](chart_scans/) 数字化数据 |
| [import_kodak_granularity.py](import_kodak_granularity.py) / [import_kodak_mtf.py](import_kodak_mtf.py) | 图表扫描 → 颗粒 σ(D) / MTF 数据；随后用光学同步工具编译渲染资产 |
| [import_jptc.py](import_jptc.py) / [import_jptc_collect.py](import_jptc_collect.py) | 一手 JPTC 测量 → 传感器 priors；前者有 `--self-test` |
| [import_cbld.py](import_cbld.py) / [import_p2p_pdr.py](import_p2p_pdr.py) | 本地 CBLD / P2P 表 → priors；来源与分发状态见 [NOTICE](../NOTICE.md) |
| [import_lens_transmittance.py](import_lens_transmittance.py) | 一手镜头 / 滤镜光谱 → 镜头透过率库 |
| [make_evidence_shell.py](make_evidence_shell.py) / [import_dngshell.py](import_dngshell.py) | RAW / 上游壳 → 无像素的容器元数据测试语料；不用于解码测试 |
| [spectral_base.py](spectral_base.py) | 构建器共用的光谱基础库，不是独立操作入口 |

### 文档图片

[regen_showcases.py](regen_showcases.py) 是整表重新渲染入口；[showcase_manifests/](showcase_manifests/) 保存教程和 README 的输入与参数。先 `--list` 查看，再通过 `--samples`、`--scratch`、`--only` 选择输入、输出和范围；加 `--install` 才替换 `docs/assets/` 中的展示图。

[make_hdr_showcase.py](make_hdr_showcase.py) 生成网页尺寸的真实 gain-map JPEG；[plot_tone_curve_doc.py](plot_tone_curve_doc.py) 重算教程曲线图。[compose_plate.py](compose_plate.py) 为清单共用拼板器，直接入口是 `python tools/compose_plate.py OUT.jpg SPEC.json`。

## 历史工具与保留原因

| 工具 | 当前定位 |
|---|---|
| [build_film_appearance_identity.py](build_film_appearance_identity.py) | 早期 P1 布线验证的 identity 占位生成器。会覆盖同名当前配方；正常重建用 `build_film_appearance_recipes.py` |
| [build_full_lut.py](build_full_lut.py) | 独立执行生成历史 `full_lut` 资产；其观察者拟合与烘焙函数仍被当前 v2 构建器和交叉验证复用，因此保留原路径 |

历史工具用于追溯模型演进，不属于日常安装或更新步骤。现有性能与质量脚本即使带有阶段编号，仍可复现对应假设，保留在上述任务分类中。
