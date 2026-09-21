# 开发指南与仓库地图

[返回文档索引](README.md) · [技术架构](ARCHITECTURE.zh-CN.md) · [工具目录](../tools/README.md)

AgXRAW 是项目名，Python 包和命令行入口保留 `dngscan`。运行环境是 Apple Silicon macOS、Python 3.11+；GUI、Apple RAW 和 HDR 的平台依赖见 [根 README](../README.zh-CN.md)。

## 仓库怎么找

| 路径 | 职责 |
| --- | --- |
| [dngscan/](../dngscan) | 生产 Python 管线；CLI 与 GUI 共享解码、分析、渲染和交付 |
| [dngscan/gui/](../dngscan/gui) | 本地 HTTP 服务、页面、预览调度、缓存与导出任务 |
| [rust/src/](../rust/src) | 可选 Rust 计算核；Python 边界与 NumPy 回退仍在 `dngscan` |
| [dngscan/data/](../dngscan/data) | 随安装包分发的传感器、镜头和胶片资产；另有包根 JSON 预设 |
| [dngscan_assets/](../dngscan_assets) | 原始参考、光谱数据与出处；重新校准从这里追溯 |
| [tests/](../tests) | 正确性、边界、集成和冻结基线；`benchmark_*.py` 需显式运行 |
| [tools/](../tools/README.md) | 开发、校准和性能分析工具，不是用户启动入口 |
| [docs/](README.md) | 当前说明、设计合同和教程；历史测量分入 [reports/](reports/README.md) |
| [docs/assets/](assets) | 文档插图、公开样张和机器可读测量 |
| [.github/workflows/](../.github/workflows) | CI 环境、双执行模式测试与安装产物检查 |
| [pyproject.toml](../pyproject.toml)、[uv.lock](../uv.lock) | 包元数据、依赖与 CI 使用的冻结版本 |

运行时代码、包资源和冻结测试保持原路径，避免整理目录改变导入、资源查找或基线身份。个人 RAW、临时导出和完整 benchmark 输出放在仓库外；公共 `docs/assets` 只保留有说明、可追溯的展示和汇总。

## 按一次 RAW 导出的顺序读代码

先看 [产品架构](PRODUCT_ARCHITECTURE.zh-CN.md) 了解各层职责，再从下面的入口跟一次数据流。这里列的是阅读入口，不表示每个请求一定调用所有模块。

| 层 | 主要入口 | 要核对的边界 |
| --- | --- | --- |
| 用户参数 | [cli.py](../dngscan/cli.py)、[GUI service.py](../dngscan/gui/service.py) | 同一选项在 CLI、预览与导出中的含义 |
| RAW 与校正 | [raw_io.py](../dngscan/raw_io.py)、[coreimage_decode.py](../dngscan/coreimage_decode.py)、[evidence.py](../dngscan/evidence.py) | 解码图像、传感器证据、DNG 操作与几何如何对齐；能力不足怎样退回 |
| 分析 | [analysis.py](../dngscan/analysis.py)、[sensor_summary.py](../dngscan/sensor_summary.py)、[phase_statistics.py](../dngscan/phase_statistics.py) | 黑白电平、剪切、噪声和可靠高光的来源与资格 |
| 计划与曝光 | [tone.py](../dngscan/tone.py)、[auto_ev.py](../dngscan/auto_ev.py)、[models.py](../dngscan/models.py) | 分析如何生成不可变计划，手动意图如何覆盖默认值 |
| 成像 | [render.py](../dngscan/render.py)、[agx.py](../dngscan/agx.py)、[hdr_agx.py](../dngscan/hdr_agx.py) | SDR / HDR 分别形成，采样计划与全尺寸计算如何共享 |
| 编码与发布 | [export.py](../dngscan/export.py)、[auto_encode.py](../dngscan/auto_encode.py)、[delivery.py](../dngscan/delivery.py)、[delivery_transaction.py](../dngscan/delivery_transaction.py) | 固定图像母版、编码候选、实际回读、元数据与最终文件 |
| 原生加速 | [_fast.py](../dngscan/_fast.py)、[fast_plan.py](../dngscan/fast_plan.py)、[Rust lib.rs](../rust/src/lib.rs) | ABI、输入布局、所有权、失败政策与每个核的数值合同 |

胶片路径在 `film_*.py`、Rust `film_*` / `spatial.rs` 与包内校准资产之间展开；其设计合同和冻结记录由 [文档索引](README.md) 单独列出。

## 建立开发环境

日常安装和 GUI 启动按根 README。需要复现 CI 的依赖时，在仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip uv
uv export --frozen --all-groups --format requirements-txt --no-hashes \
  | grep -v '^-e' > /tmp/agxraw-requirements-lock.txt
python -m pip install -r /tmp/agxraw-requirements-lock.txt
python -m dngscan --help
python -m dngscan.gui
```

RAW 解码依赖包含固定 Git revision，需要 Git 和 Xcode Command Line Tools。可调 HEIF 编码另需带 x265 的 `libheif`，使用方法见 [导出说明](USER_GUIDE.zh-CN.md)。

## 验证改动

先跑受影响模块的测试，再按改动范围扩展。涉及图像计算、原生调度或发布前的完整回归时，和 [CI](../.github/workflows/ci.yml) 一样分别验证 NumPy 与严格原生模式：

```bash
DNGSCAN_FAST=0 python -m unittest discover -s tests -q
PYTHON=python bash tools/build_native.sh
DNGSCAN_FAST=1 python -m unittest discover -s tests -q
python -m unittest tests.test_digitization_precision -v
```

`DNGSCAN_FAST=0` 关闭可选原生核；`1` 要求核加载与执行成功；默认 `auto` 允许回退。构建脚本会检查 ABI、自测，并在 macOS 对生成的扩展签名。切换到不同 ABI 的 checkout 或更换 Python 后应重新构建，不能复用另一版本的 `.so`。原生核依各自合同验证；已有允许容差的路径不代表新优化可以引入额外像素变化。

GUI 页面事件还需要浏览器验证：选一张 RAW，等待预览，再修改受影响选项并导出。服务端单测通过不能证明 JavaScript 事件实际运行成功。`test_gui_page_runtime.py` 用 Node 执行关键事件回归；缺少 Node 时须注意相应跳过，不能记作已验证。

耗时和大图内存测量用 [工具索引](../tools/README.md) 中的显式 benchmark。记录输入身份、版本、native 模式、尺寸、wall/CPU/RSS 与输出等价性；计时范围应说明是否包含解码、编码和回读。不要把 synthetic 单核结果直接解释为全任务加速。已有测量见 [性能完成记录](reports/performance/performance-pipeline-completion.zh-CN.md)。

## 文档怎么维护

使用说明描述当前行为；架构说明描述当前边界；设计合同保留数学定义、批准范围与实施状态；报告保存当时证据。修改默认参数时同时更新中英文用户入口，历史测量保留原参数并标注版本。新增文档加入 [索引](README.md)，报告同时加入 [报告索引](reports/README.md)，移动文件时一起检查相对链接、图像和代码中的引用。
