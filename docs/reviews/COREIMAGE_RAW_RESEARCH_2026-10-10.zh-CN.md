# Apple RAW 版本矩阵与解码后残差：受限验证

本记录对应 R9-05／R9-06／R9-07。版本矩阵验证实际执行路径、配方和数值输出；解码后残差模块只做离线研究。普通照片不是暗场、平场或 PTC，也不能证明解码器的噪声处理更准确、相机压缩无损或画质提升。

## 真实 RAW 版本与预览矩阵

[validate_coreimage_raw.py](../../tools/validate_coreimage_raw.py) 为每个配置启动独立进程，分别记录源 SHA256、实际 codec／存储依据、传感器证据、实际 decoder token、运行时、完整配方、统计模型来源、开启／跳过的处理、浮点负值与大于 1 的样本、SDR/HDR 有限性、阶段时间和进程峰值 RSS。不保存原片或图像缓冲。`scene_window_calibration` 独立保留窗口标定的实际资格状态、匹配原因、profile ID／hash 和资产 hash；传感器 profile 的诊断不能替代该记录。

```sh
python tools/validate_coreimage_raw.py \
  --files /path/to/sony.ARW /path/to/nikon.NEF \
  --matrix raw9 raw8 libraw auto \
  --sizes full native-preview resized-full \
  --scales native aligned --preview-max-edge 1280 \
  --out /path/to/new-matrix.json
```

`--files`、`--matrix`、`--sizes` 和 `--scales` 可缩小运行规模。`--preview-max-edge` 仅控制 `resized-full`，默认 1280；Apple 原生预览独立使用实际原生目标 1280，参数不能改变它。工具拒绝覆盖旧报告；实际 Apple 渲染需要可用的 macOS Quartz／Metal 上下文。执行上下文不可用单独记录，不能将沙箱渲染失败解释成相机文件不支持。

| 维度 | 具体语义 |
| --- | --- |
| `raw9`／`raw8` | 固定版本。未提供该版本记 `unavailable`；只接受 recipe 中实际 getter token 与 `parameter-state-verified` 的参数状态。缺少实际依据、token 不符或发生回退不能通过；参数状态核对不冒充实测线性或噪声资格。 |
| `auto` | 独立记录最终实际版本与失败／回退历史。成功回退到 LibRaw 不算 Apple 固定版本成功。 |
| `libraw` | 当前生产 LibRaw 参考路径，没有 Apple 的尺度轴；不冒充独立真值。 |
| `full` | 完整解码输出与对应完整 SDR/HDR。 |
| `native-preview` | 解码器自己的交互／半尺寸路径。Apple 的原生预览通常由原生 preview scale 决定，不保证恰为宽高各一半。 |
| `resized-full` | 先完整解码和显影，再另报告确定性面积块均值预览；边缘块按实际样本数求均值，不裁掉负值或高光。不将它称作原生预览。 |
| `native`／`aligned` | 分别使用 Apple 原生尺度和项目逐文件相对对齐尺度。对齐是相对尺度比较，不是绝对电子数或辐射标定。 |

原始输出计数和除以 `scene_scale` 的归一化场景统计分别保留；不同解码器的几何、重建与尺度不直接当作同一像素真值。完整 decoder recipe 保留 `configuration_id`、实际 token、运行时、几何、输出、控制支持／请求／有效状态以及项目 handoff。普通 LibRaw 参考没有 Apple 配置身份，明确记为没有该身份。

单张归一化场景另报告 Y 分布、零亮度平面 2×2 **空间变化**协方差和频带结构能量。这些量含有真实场景结构与噪声，不是噪声估计；在独立标明的诊断网格计算，频率单位不是传感器像素。跨配置分别比较原生预览／全尺寸后缩图、原生／对齐尺度以及 RAW9／8 的曝光分布、颜色和结构能量变化。没有空间配准或独立真值，分布变化不能解释成某些像素受损、噪声降低或真实纹理丢失。SDR/HDR 每格采用 `compute_auto_ev()` 的当格推荐值，属于生产自动管线执行基线；这些场景统计在曝光意图前归一化。自动曝光造成的显影差异不是降噪差异，固定 EV／WB 的画质实验仍需另行锁定条件。

RSS 是独立 worker 的完整高水位，包括导入、证据、解码、分析和显影，不是单个 Apple 核的净增内存。计时也不是实时预览延迟承诺。本轮与项目回归并行运行，阶段时间仅作执行记录，不用于受控性能排名。每格校验处理代码、工具、窗口标定资产、传感器标定及源文件身份；处理中或格间变化会使报告失败。当前工具关闭项目后级色度清理、风格和场景变换，记录 Apple 自身实际控制，不把控制值为 0 解释成 RAW9 的联合处理已完全关闭。

## 独立输出残差合同

[decoder_residual_research.py](../../dngscan/decoder_residual_research.py) 的 `repeat_residual_statistics()` 接受至少四张、偶数数量的独立输出帧，以及逐帧来源和采集合同。输入必须是有符号浮点 H×W×3 场景线性 Rec.2020，采集类型明确为暗场或平场；整数图片、普通照片和传感器 DN 不能冒充该输入。

合同固定机型、机身、ISO、曝光和读出标识，要求每帧不同的源 SHA256 与采集 ID。相同 RAW 反复解码会被拒绝；不同哈希和操作者声明本身仍不能证明真实采集独立，必须另行核对拍摄过程。每帧还须保留自己的完整 decoder recipe：除了配置身份，实际白平衡、BaselineExposure／对齐 handoff、输出几何和尺度也必须一致，不能让配置 ID 中有意排除的逐文件数据改变残差单位。实际浮点帧须精确匹配 recipe 的完整输出网格；裁短数组不能继续绑定原大网格，只能在完整输入上用 `regions` 显式选择统计区域。

```python
report = repeat_residual_statistics(
    output_frames, frame_records,
    contract=measurement_contract,
    regions={"center": [y0, x0, y1, x1]},
    reference_mean=independent_reference,
)
```

相邻独立帧形成互不重叠的 A−B 对。输出保留每对均值、原始差分二阶矩／2，以及减去区域内均值后的 RGB 3×3 差分协方差／2。均值移除避免将共同亮度漂移直接混入区域内噪声，移除的量仍单独报告，不宣称共同模式噪声消失。

零亮度平面由 Rec.2020 亮度权重 `[0.2627, 0.6780, 0.0593]` 的正交零空间定义，保留完整 2×2 色度协方差和径向频带功率。频率单位是 **cycles/output-pixel**，不是 CFA 或原生传感器像素；不借此移除生产流程的异常频谱门控。频带功率按 Parseval 的总体均方归一化，区域样本协方差使用 N−1 分母，二者的有限区域分母差异明确保留。

没有独立参考时，均值偏置和 FPN 资格为不可用，不能把平场照明不均或图像结构叫作固定图样。提供参考还需声明参考来源、输入域及其独立性；报告分别保存参考相对均值偏置、均值残差空间协方差和有限重复组的区域时域均值不确定度近似。扣除后出现负特征值时记为未分辨，不裁成零来伪造测量。参考误差、区域非平稳性和照明系统误差仍未完成完整预算。

这些输出没有 RAW shot/read 模型资格，没有 CFA 逐像素门控资格，也不会开启后级 NR 或用于自动 HDR 噪声阈值。

## 输出偏置候选的限制

`build_output_bias_candidate()` 只接受独立训练暗场、独立留出暗场及独立输出参考。训练和留出不能共享源哈希或采集 ID。候选保存为 float32 的有符号 RGB 输出偏置，并用这份实际候选在留出数据上比较偏置残差；减去固定偏置不被解释成时域随机噪声降低。

候选语义是「在同一解码及 handoff 条件下，离线减去解码后 RGB 偏置」。它**不等价于解码前的黑位、DSNU 或传感器暗场校正**：自适应解拜耳、降噪、高光处理和裁切已经执行，后减无法恢复它们丢弃的信号，也不能假定残差与信号无关。输出明确标记 `automatic_application=false`、`production_applicable=false` 和 `predecoder_raw_correction_equivalent=false`；没有生产应用入口。

## 验证与尚未测量

合成合同回归见 [test_decoder_residual_research.py](../../tests/test_decoder_residual_research.py) 和 [test_validate_coreimage_raw.py](../../tests/test_validate_coreimage_raw.py)。它们检查差分除 2、RGB／零亮度协方差、频带功率、共同偏置分离、同源重跑拒绝、逐帧配方变化、完整输出网格、留出独立性、负值预览、窗口资产身份和固定版本回退边界。最终工具冻结后的定向回归于 2026-10-10 20:58:58 UTC 完成，28 项通过。合成结果只证明这些数学和接口合同。

本轮实际样本为本地 Sony DSC00225.ARW 和 CC0 的 Nikon D750 NEF（raw.pixls.us 记录 897；[来源记录](../assets/delivery-quality/nikon-nef-20261010.json)，源哈希 `baae8dfc05a81a4a4034831a49ed99d129e797858ad876d31173a46d33b7d3ea`）。Sony 只提供 RAW7／8，RAW9 不可用；Nikon 提供 RAW6／7／8／9。首轮原生预览八格只是冒烟记录；最终结论以补齐配方、窗口资产身份后的 [42 格正式报告](../assets/pipeline/coreimage-raw-20261010.json) 为准。

正式矩阵于 2026-10-10 21:05:26 UTC 完成，在 macOS 27.2／26B5101f 的真实 Quartz／Metal 上下文中运行，退出码为 0。

| 样本 | 通过 | 版本不可用 | 执行失败 | `auto` 实际版本 |
| --- | ---: | ---: | ---: | --- |
| Sony DSC00225.ARW | 15 | 6（RAW9 全部配置） | 0 | RAW8 |
| Nikon D750 NEF | 21 | 0 | 0 | RAW9 |
| 合计 | 36 | 6 | 0 | 分别保留实际 token |

36 个成功格的解码输出、归一化场景、预览统计输出及 SDR/HDR 全部有限；18 格观察到负值，36 格观察到大于 1 的值。所有成功的 Apple 格都通过实际 token 和参数状态核对，固定版本没有以回退冒名。窗口状态为 30 格 `unmeasured` 和 6 格 LibRaw `identity`，没有任何格被声称已获得精确窗口标定。另保存 33 组描述性的跨配置分布对比，不能据此认定噪声改善或纹理损伤。

格内、格间及运行结束后的当前磁盘复核确认：处理代码、工具、窗口资产、传感器标定和原文件身份均一致。报告不含私有绝对路径或原像素，大小为 3,085,196 字节，SHA256 为 `2f3f692487b4b1538da673797ace3d62966343e5205bcdf2ca1363e956e72fba`。完整逐文件 codec 依据仍分别保存：Sony 的量化曲线记录不能被称作无损，Nikon 的压缩过程仍是未验证状态。

Nikon HE／HE*、独立真实暗场／平场／PTC 重复组、弱信号纹理留出和完整 Apple gain-map 容器编码矩阵仍未测。运行普通照片成功不能补上这些资格，也不能由输出残差研究宣称个人校正或 RAW9 画质收益。
