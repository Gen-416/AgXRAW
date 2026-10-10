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
| 验证原生 RAW 与 Rust/NumPy 成像 | [validate_native_raw.py](validate_native_raw.py) | 样本 hash、codec／读出、逐相位和局部区域指标；不冒充独立解码器准确度 |
| 检查 RAW 损坏／截断的明确失败行为 | [validate_raw_failures.py](validate_raw_failures.py) | 仅修改临时副本，记录正常解码控制、损坏类型、原件 hash、失败状态；不输出照片 |
| 复现近黑、固定偏置、二维色度与重复量化研究 | [validate_sensor_precision.py](validate_sensor_precision.py) | 无需个人文件的合成矩阵，可选真实 LibRaw 合成 DNG 到 SDR/HDR |
| 更新测试基准、标定或数据来源 | [数据与资产维护](#数据与资产维护) | 明确写入的资产、清单或测试夹具 |

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

# 同一 SDR/HDR 母版：测手动 HEIF 的辅助图重试与主图复用
python tools/benchmark_gainmap_search.py --base /path/to/base.npy \
  --hdr /path/to/hdr.npy --headroom 3 --delivery-profile share \
  --quality 90 --chroma 444 --heif-preset slow \
  --out "$AGXRAW_RUN_DIR/manual.heic"
```

| 工具 | 测量范围 |
|---|---|
| [benchmark_delivery.py](benchmark_delivery.py) | `tunable` 为可调编码矩阵；`auto` 为生产 HDR 自动搜索；`system` 保留早期系统编码对照 |
| [benchmark_cli_delivery.py](benchmark_cli_delivery.py) | 自动选参、压缩内容、元数据后文件、回读像素；`--compare` 比较报告，`--require-file-match` 额外要求完整文件相同 |
| [benchmark_gainmap_search.py](benchmark_gainmap_search.py) | 固定 SDR/HDR `.npy` 母版的 HEIF 自动搜索或手动重试；记录候选、编码/回读次数和进程 RSS；支持 `--dry-run`、参考报告/文件比较 |
| [benchmark_delivery_metrics.py](benchmark_delivery_metrics.py) | base/coding 扫描及 HDR 多秩统计工作区 |
| [benchmark_delivery_buffers.py](benchmark_delivery_buffers.py) | HDR packing/readback、HEIF 8/10-bit 输入平面的逐位与内存对照 |

`benchmark_gainmap_search.py` 默认为 `auto`，由生产策略选择质量和色度；显式 `--quality/--chroma` 要选 `--delivery-profile share`。`archive` 保留 q100/444，`--heif-preset` 只指定本次编码速度。可用 `--repo` 指向参考源码，以 `--reference-report/--reference-file --require-match` 检查同一母版的决策、指标和压缩内容；报告还单独记录完整文件 SHA-256 是否一致。计时不含导入、输入哈希、能力探测和结果比较，RSS 是包含输入准备的进程高水位。

两种导出基准也接受没有 `.git` 的源码快照，报告中的 commit 此时为 null；用对应源码和扩展的指纹记录版本，不能将 null 当作已验证提交。

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
| [pipeline_impact.py](pipeline_impact.py) | 一张 RAW 上的单项处理影响矩阵；包括 native / NumPy、解码、曝光与颜色几何 |
| [scan_drt_geometry.py](scan_drt_geometry.py) | 合成 EV × 色相 × 色度的 DRT 扫描，输出 `--csv` |
| [validate_ideal_image.py](validate_ideal_image.py) | `python tools/validate_ideal_image.py /path/to/ideal-image-pair`；有已知黑白电平、增益、噪声的合成 DNG 对照 |

## 数据与资产维护

这一组会读取外部测量数据或重建仓库中的资产。先确认要变更的数据来源和模型，再同时检查生成文件、清单与受影响测试。冻结夹具记录既定行为，不能靠重生成来消除未解释的测试失败。

### 只读核验与冻结

```sh
```

| 工具 | 默认行为 |
|---|---|
| [regen_sdr_freeze.py](regen_sdr_freeze.py) | 重写 SDR 冻结；没有 `--check`，核验用 `python -m unittest tests.test_sdr_freeze` |
| [regen_golden.py](regen_golden.py) | 用 NumPy 参考路径重写 golden；可从 DNG 生成裁切夹具 |

### 可复现构建与导入

| 工具 | 输入 → 输出 |
|---|---|
| [calibrate_skin_matrix.py](calibrate_skin_matrix.py) / [fit_skin_window.py](fit_skin_window.py) | 光谱 / 真实照片 → 前馈矩阵 / 窗口；窗口拟合默认只打印，`--write` 才写回 |
| [regenerate_material_presets.py](regenerate_material_presets.py) | 各预设记录的目标 SSF → 材质前馈预设；可用 `--out` 指定新文件 |
| [calibrate_raw9_anchors.py](calibrate_raw9_anchors.py) | 声明的本地相机样张集 → 解码器窗口锚点；会更新 `decoder_anchor_transport.json` |
| [import_jptc.py](import_jptc.py) / [import_jptc_collect.py](import_jptc_collect.py) | 一手 JPTC 测量 → 传感器 priors；前者有 `--self-test` |
| [import_cbld.py](import_cbld.py) / [import_p2p_pdr.py](import_p2p_pdr.py) | 本地 CBLD / P2P 表 → priors；来源与分发状态见 [NOTICE](../NOTICE.md) |
| [import_lens_transmittance.py](import_lens_transmittance.py) | 一手镜头 / 滤镜光谱 → 镜头透过率库 |
| [make_evidence_shell.py](make_evidence_shell.py) / [import_dngshell.py](import_dngshell.py) | RAW / 上游壳 → 无像素的容器元数据测试语料；不用于解码测试 |

### 文档图片

[regen_showcases.py](regen_showcases.py) 默认加载修图教程清单，是整表重新渲染入口；[showcase_manifests/](showcase_manifests/) 保存教程和 README 的输入与参数。先 `--list` 查看，再通过 `--samples`、`--scratch`、`--only` 选择输入、输出和范围；加 `--install` 才替换 `docs/assets/` 中的展示图。

[make_hdr_showcase.py](make_hdr_showcase.py) 生成网页尺寸的真实 gain-map JPEG；[plot_tone_curve_doc.py](plot_tone_curve_doc.py) 重算教程曲线图。[compose_plate.py](compose_plate.py) 为清单共用拼板器，直接入口是 `python tools/compose_plate.py OUT.jpg SPEC.json`。


胶片构建、图表采样、外观与光学冻结工具已移到 AgXFilm，入口见[拆分记录](../docs/FILM_SPLIT.zh-CN.md)。
