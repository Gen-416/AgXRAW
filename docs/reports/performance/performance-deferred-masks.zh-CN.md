# 第三批第二步：分析完成后一次构建最终 sensor mask

本步将 CLI、GUI 冷预览及 GUI 导出的 LibRaw sensor mask 构建延后到 full-well 已确定的分析边界。此前 `load_raw` 先按 metadata white 建 mask；若 `analyze` 发现可信的实际饱和平台，`refresh_clip_masks_from_fullwell` 会重新做整套构建、几何对齐、羽化及 processing-loss 合并。现在这些内部调用只构建最终结果。

公开 `load_raw` 的默认行为保持原样；内部通过私有关键字 `_defer_clip_masks=True` 启用延迟。返回的 LibRaw bundle 此时 `clip_masks=None`、`_clip_masks_pending=True`，不得直接交给规划或渲染。CLI／冷预览立即进入 `analyze`；GUI 导出则进入 `analyze`，或复用已验证的完整 `Analysis` 后调用同一个 refresh。完成边界之前不构建 proxy、不持久化预览、不开始渲染。

与“在 load 中提前计算 full-well”的方案相比，延迟构建允许 GUI 缓存分析导出继续复用已有统计；不因省一次 mask 又额外执行传感器扫描。本步没有增加 Rust 核，仍使用 ABI 15 的现有 mask／loss 实现。

## 数值与生命周期合同

`_clip_masks_pending` 是 keyword-only dataclass 字段，因此 `replace` 与 hot-WB 保留状态，也不移动现有 `RawBundle` 的位置参数。它与 Apple 无空间 mask 的状态不同：正常 Apple 路径仍为 `clip_masks=None`、pending=false，独立 LibRaw 参考保持自己的几何与 mask。Apple auto 回退 LibRaw 时才传递延迟标志。

首次构建读取当时 bundle 的 RAW、CFA、颜色标签、白点、黑电平和空间黑电平，以及最终 scene shape、orientation、warp／crop 与 processing loss。仍按原顺序执行 soft confidence、2×2 降采样、几何搬运、soft resize、垂直／水平羽化、half 舍入，最后合并 processing loss。没有新增跨几何缓存，后续 full-well 改变仍通过原 refresh 更新。

尤其保留旧流程的浮点白点语义：refresh 比较的是整数 metadata，但原始 load 的 mask 使用浮点 `camera_white_levels`。当 resolved full-well 与整数 metadata 相同时，旧 refresh 不重建，因此新首次构建也必须使用原浮点 metadata，而不是直接使用整数 summary。只有原流程确实会重建时，才使用原 refresh 的逐通道 levels。这包含缺失、非正、正数小于 1、稀疏 channel ID 等原 fallback 规则。以 RAW=960、白点=1000.75 的小夹具为例，原 mask 为 0.136474609375，改成整数白点会变成 0.15625，不能忽略这一区别。

构建及 processing-loss 合并先在局部完成，成功后才发布 mask、清除 pending，并清空 resized-mask、RAW guidance、guidance resize 与两项 guidance 资格标记。任一操作失败都保留待构建状态，调用方不会得到被误标为完整的 mask。已有 mask 的后续 refresh 也使用完整结果发布。

## 实测结果

2026-09-20，macOS 27.2 arm64、10 个逻辑 CPU、Python 3.14.4、NumPy 2.5.2、ABI 15。先在未修改的 `081fd1a` 上生成四条路径的基线；再以当前代码交替运行 eager／deferred 模式。共 **16 次新进程测量的分析后 mask、分析／决策与 SDR／HDR master 全部逐位一致**。公开 eager load 刚返回时的 mask 也与旧版相同；内部 deferred load 此时尚无 mask，这是唯一有意不同的中间状态。

完整分轮计时、调用数、状态、图像与决策身份及源文件哈希见 [`deferred-masks.json`](../../assets/performance/deferred-masks.json)。计时不含编码。

| 路径 | 交替对照轮数 | mask 构建次数 | SDR 中位秒数：eager → deferred | HDR 中位秒数：eager → deferred |
|---|---:|---:|---:|---:|
| Sigma `_SDI0150.DNG`／LibRaw | 3 | 2 → 1 | 7.720 → 7.390 | 8.625 → 8.309 |
| Sony `DSC00225.ARW`／LibRaw | 3 | 2 → 1 | 11.233 → 10.795 | 12.414 → 11.954 |
| Sigma／Apple | 1 | 1 → 1 | 6.662 → 6.632 | 7.476 → 7.455 |
| Fuji `DSCF0214.RAF`／LibRaw | 1 | 1 → 1 | 23.661 → 23.513 | 24.575 → 24.469 |

Sigma 的 mask 构建累计墙钟中位数为 **0.557 → 0.287 s**，Sony 为 **0.796 → 0.422 s**；相关 processing-loss 合并也由两次变为一次。未编码管线的 SDR／HDR 中位数分别减少 **4.3%／3.7%**（Sigma）和 **3.9%／3.7%**（Sony），两种样片各自三对测量的两项管线耗时均下降。

Apple 正常路径仍只有独立参考的那一次 mask 构建，未受这次 deferral 影响。Fuji 的 full-well 不触发旧流程的重建，两侧都只建一次；其 HDR headroom 为 0 EV。两组各一对仅用于正确性和控制检查，不将时间差归因为本步收益，也不将 Fuji 的双路调用视为实际扩展动态范围交付。进程峰值 RSS 没有一致下降，不能据本步宣称整条导出管线的峰值内存降低。

## 验证与测量方法

本步新增 23 项测试，并强化既有缓存导出与冷预览测试。严格 native 完整 suite 运行 **1,691 项，1,685 通过、6 跳过**；NumPy 回退专项运行 **179 项，178 通过、1 跳过**，两者均无失败或错误。NumPy 唯一跳过项是实拍样片裁切高光不足，无法判断 Core Image 高光恢复；新增 23 项在两种模式下均通过。本步未修改 Rust 源码，沿用上一批已验证的 ABI 15 扩展。

[`test_deferred_clip_masks.py`](../../../tests/test_deferred_clip_masks.py) 覆盖旧 eager endpoint oracle 与延迟构建的逐位一致性，包括 Bayer、X-Trans、Linear RGB、稀疏通道、小数白点、空间黑电平、warp／crop／旋转、processing loss、失败重试、派生缓存失效、WB／replace、公开 loader 和 Apple fallback。

[`test_pipeline_corrections.py`](../../../tests/test_pipeline_corrections.py) 与 [`test_preview_cache.py`](../../../tests/test_preview_cache.py) 通过真正的 CLI／GUI 调用边界验证接入：冷预览与 CLI 在构建 proxy／渲染前完成分析；缓存导出禁止重新 `analyze`，直接用已有 full-well 完成 pending mask。metadata 相同和实际 full-well 改变两种情况均覆盖。

[`benchmark_deferred_masks.py`](../../../tools/benchmark_deferred_masks.py) 复用现有 loss pipeline 工具，每次新进程执行全分辨率解码、分析、自动曝光、默认 AgX／P3 SDR 与 800 nit HDR pair，不执行编码。两侧固定 `DNGSCAN_FAST=1`、`DNGSCAN_FAST_SKIP=""`；`--reference` 使用公开 eager load，普通模式只启用私有延迟标志，所有现有 Rust 核与 SensorSummary 均保持启用。

```sh
python tools/benchmark_deferred_masks.py \
  --source /path/to/photo.dng --decoder libraw --reference \
  --out /path/to/eager-1.json

python tools/benchmark_deferred_masks.py \
  --source /path/to/photo.dng --decoder libraw \
  --compare /path/to/eager-1.json --out /path/to/deferred-1.json
```

用 `--repo` 指向旧 checkout 并加 `--reference` 可验证改动前的 `081fd1a`；旧版本必须有匹配扩展和既有 loss pipeline 工具。真实样片只读，输出要求新路径；已有文件、悬空符号链接或运行期间出现的同名结果不会被覆盖。

延迟构建有意改变 **load 刚返回时** 的内部 mask 状态，所以比较只排除 `identity.masks_loaded`；报告仍如实保存该字段，以及每次 load 的 pending／mask 存在状态、每次 refresh 的前后状态。分析后的 mask、processing loss、独立参考、完整分析及自动决策、SDR／HDR master 的 shape／dtype／SHA-256 必须完全相等。若参考报告含输入文件 SHA，也必须一致；不匹配返回退出码 2。

`build_clip_masks` 与 refresh 计时存在嵌套，不能相加。构建由 load 移到 analyze，会使单独 analyze 看似变慢，判断收益应看整个 load＋analysis＋formation 的阶段总和。SDR 与 HDR 总和共享准备阶段，二者也不能相加。输出哈希、管线开始前的模块导入和 JPEG／HEIF 编码不在这些阶段计时中；阶段内的惰性导入仍会计入。进程峰值 RSS 包含解码与哈希，不等于活跃 mask 内存或 GPU 内存。
