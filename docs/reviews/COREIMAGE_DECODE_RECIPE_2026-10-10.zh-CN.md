# Core Image 解码配方与能力状态

日期：2026-10-10。本轮实现 R9-04／R9-08 的解码审计；没有将 Core Image 参数读回解释为实测画质或完整传感器噪声传播资格。

## 官方语义与现有策略

Apple WWDC26 明确说明 RAW 9 使用联合去马赛克／降噪模型，`colorNoiseReductionAmount` 在 RAW 9 无效，`detailAmount`、`moireReductionAmount` 不再支持；实例的 `isSupported` 属性用于检查适用性。因此，legacy 参数读回为零，只能说明参数状态，不能声称内建降噪关闭。[Apple WWDC26 session 305](https://developer.apple.com/videos/play/wwdc2026/305/)

线性场景访问沿用 Apple WWDC21 代码示例：清除 `baselineExposure`、`shadowBias`、`boostAmount`、`localToneMapAmount`，关闭 gamut mapping，并渲染到扩展线性颜色空间和 RGBA half buffer。项目另明确请求解码曝光、EDR 参数与输出缩放。每项记录写入和读回结果，缺失 API 不自动等价于“成功清除”。[Apple WWDC21 session 10160](https://developer.apple.com/videos/play/wwdc2021/10160/)

现有附加锐化参数仍请求为零，以区分解码器的附加锐化政策和项目成像；空间算子不必然非线性，因此不再用“空间锐化必然破坏线性”作为论据。moire 保留解码器默认。镜头／高光恢复请求启用，但 getter 为 true 不证明具体每条 opcode 已执行，也不证明与已校正 LibRaw 输出具有精确像素对应。无完整样本 hash、配置和运行时支撑的历史固定百分比注释已移除。

## API 与配置身份

`decode_scene_rec2020()` 保持 `(float16_rgb, info)` 返回；增加 `info['recipe']`，其 `schema` 与 `COREIMAGE_RECIPE_VERSION` 均为 `coreimage-decode-recipe-1`。

配方分别记录：

- 原始请求、实际完整 decoder token（包括 `9.dng` 后缀）、文件提供的 tokens、失败尝试和 fallback。
- OS／build runtime 字符串及非唯一硬件事实；不把 Apple 对 Neural Engine 的一般说明当成这次调度的实测证明。
- 每个 control 的 requested、支持查询、适用性、setter 的存在／尝试／结果、getter 前后读值，以及有效语义。写入成功、参数读回确认与实际像素效果验证是三个不同状态。
- 完整源文件 SHA-256、文件大小及 hash 到解码结束的文件身份一致性；源文件发生变化时撤销 hash 资格。
- 真正参与 bitmap render 的 CIImage extent、渲染和最终 raster 尺寸、实际 scale、目标颜色空间、float16 输出、固定倍率、显式 resize 与 context options。

`configuration_id` 是规范化配置 JSON 的 SHA-256，包含实际 token、runtime／硬件、control 状态和有效配置值、WB 实现策略、实际缩放、固定倍率、context 和输出合同。固定项目热 WB 版本 `rec2020-conjugate-camera-matrix-approx-v1` 进入身份。显式 fixed-Kelvin 请求属于配置。

请求选择历史、offered tokens、fallback 链、源 hash、像素内容、原生 extent、文件 authored BaselineExposure 和 as-shot neutral 数字不进入配置身份。同一有效配方从预览的 `auto` 固定为导出的实际 token，身份保持；不同 decoder token、支持状态或缩放不能混用。完整逐文件证据仍保存在配方中，项目后续 `project_handoff` 由上层单独记录，不能误当成此 decoder 原生配置。

上层 `load_raw()` 保留原 ID 为 `coreimage_configuration_id`，再把项目尺度策略、参考解码版本／高光策略与热 WB 实现组合成最终 `configuration_id`。完整 handoff 另记录逐图 alignment、BaselineExposure 和实际热 WB 倍率／矩阵；这些连续值不进入可复用机型配置，却必须进入输出残差测量身份。`info['recipe']` 的原生配置与 bundle 的最终配置各有明确范围。

`linear_access.status` 只分 `parameter-state-verified` 和 `parameter-state-unverified`，并列出 `unverified_controls`。前者表示必需访问参数可确认或 API 明确表示不适用，仍不证明经验线性、无内建处理或输出域噪声模型成立。兼容字段 `color_noise_cleared` 保留，但 `color_noise_cleared_semantics` 明确为 `legacy-parameter-zero-only-not-internal-NR-off`。

## 重试与验收

`runtime_available()` 的失败仅缓存 60 秒；过期自动重建对应 workload context，`retry=True` 可立即显式重试，没有等待或重试 sleep。成功探测可在进程内复用。这是目标 bitmap 格式／context 探针，不保证任意相机的 RAW 9 模型都能成功。可选 memory-limit 常量不存在或提示被拒绝时，正确报告／回退，不再无条件访问缺失常量。

新增合成合同测试覆盖缺失 setter/getter、忽略写入、异常写入、明确 unsupported、实际 token 不符、配置跨文件稳定与有效配置变化、完整源 hash、fake-clock TTL 及可选 context 常量。另有单次 outputImage 测试：配方 extent 必须来自真正用于 render 的图像，不得在 render 后再次询问可能已变化的 graph。局部旧／新回归共 32 项全部通过。

macOS 实机 smoke 使用 `_SDI0016.DNG`，SHA-256 `1a501662a4ef4634b3e084e2b3f3a0d3ebe6a12bcd7ab73ba3867b7ff707a9ac`；实际 token `9`、runtime `版本27.2（版号26B5101f）`、Apple M4，输出为 854×1280 float16，记录的真实 extent 为 `[0,0,1280,854]`。必需访问参数读回通过；color/detail/moire 的 supported 为 false，luma NR 和 sharpness 的 supported 为 true、参数为零；镜头和高光恢复为 true。该结果不构成降噪强弱或画质收益结论。

此 smoke 必须在可访问系统 Core Image 模型缓存及服务的环境运行。进程沙盒内的 ANE 编译权限错误没有被计为相机或 decoder 不支持；放行正常系统访问后成功。完整 RAW 矩阵与全量发布 gate 由主任务单独记录。
