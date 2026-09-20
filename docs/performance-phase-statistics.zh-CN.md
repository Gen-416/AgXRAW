# 有界 CFA 相位统计

噪声下限、SNR 曲线和 RAW health 现在可以共享一次任务内的相位统计。原先各入口分别展开完整 CFA 相位、计算差分或绿色相位差；新路径按最多 128 个相位行生成校正 DN，保留较小的 tile 统计，再交给原有消费者完成分箱和动态范围计算。health 的暗 tile 选择仍使用原来的全局排序，选定后再读取所需 tile；整数直方图保留原来的百分位插值规则。

这里限制的是中间校正平面的高度，整个任务的内存并非常数：原始图像、颜色索引和平铺统计仍随输入尺寸增长。非连续、非 uint16 或不能维持原有 tile 分组的布局继续使用原来的公开入口。统计只在当前 analyze 调用内共享，不跨文件或任务缓存。

## 测量方法

`tools/benchmark_phase_statistics.py` 每次启动一个新进程。reference 独立调用 `estimate_raw_noise_floor`、`compute_snr_curves` 和 `raw_health_metrics`；new 先构建 `PhaseStatistics`，再调用同一组消费者。两侧使用同一个工作树和 native 环境，只改变是否提供共享统计。这是计算段的消融对照，不是两个完整程序版本的端到端比较。

输入为 seed 731 的确定性 uint16 Bayer 数据，分别为 6000×4000 和 10000×6000，CFA 为 `[[0,1],[3,2]]`，每通道使用固定但不同的 black/fullwell。输入的均匀整数分布不是相机噪声模型；其用途是验证完整数组计算的时间、内存规模和精确等价性。此测量不包含 RAW 解码、场景 RGB、渲染或编码。

计时包含 workspace 构建及所有消费者；输入生成、输入 SHA256 和输出身份计算均在计时外。RSS 来自进程的 `ru_maxrss`，包含导入、输入生成和哈希阶段，是进程峰值而非 workspace 的独立分配量。报告同时记录进入计时前和计算后的峰值，不能用两者简单相减当作真实工作集。

环境为 macOS 27.2 arm64、10 个逻辑 CPU、Python 3.14.4、NumPy 2.5.2、ABI 18；`DNGSCAN_FAST=1` 且 skip 为空。24MP 使用 reference→new、new→reference、reference→new 的三对顺序，60MP 使用一对 reference→new。每份原始报告记录相关源码的 SHA256，八次正式运行的源码身份全部相同；基准的 HEAD 为 `9c8a447`，实际测量的改动由源码哈希标识。

## 结果

2026-09-20 的结果如下。24MP 为三次新进程测量的中位数；60MP 只有一对，主要用于检查内存随尺寸增长的情况，不据此判断计时稳定性。

| 输入 | 指标 | 独立旧入口 | 共享统计 | 变化 |
|---|---|---:|---:|---:|
| 24MP | 三项计算总时间 | 0.383807 s | 0.262907 s | −31.50% |
| 24MP | 进程峰值 RSS | 434.547 MiB | 146.297 MiB | −66.33% |
| 60MP | 三项计算总时间 | 0.975798 s | 0.651680 s | −33.22% |
| 60MP | 进程峰值 RSS | 1005.859 MiB | 275.359 MiB | −72.62% |

四对的输入和全部公开输出精确一致，包括 noise floor、各颜色 SNR 曲线的 stops/SNR/count 数组、SNR=1 动态范围及位置、health lag-1 相关性与空直方图比例。数组比较使用 shape、dtype 和完整字节 SHA256，浮点标量另存 float64 位模式，NaN 也没有从身份检查中删除。本输入的部分 SNR=1 结果为 NaN，两侧位模式一致。

完整输入声明、输出身份、逐对计时、阶段计时、RSS、环境和原始报告 SHA256 见 [phase-statistics.json](assets/performance/phase-statistics.json)。这组结果证明该合成工作集上的计算和内存收益；真实 RAW 的分析、AutoEV 和最终图像身份由完整管线验收另行覆盖，不把本表的百分比当作整个导出流程的提速。

复现单对的命令如下，输出文件必须尚不存在：

```sh
python tools/benchmark_phase_statistics.py --size 6000 4000 --reference --out /tmp/phase-before.json
python tools/benchmark_phase_statistics.py --size 6000 4000 --out /tmp/phase-after.json --compare /tmp/phase-before.json
```
