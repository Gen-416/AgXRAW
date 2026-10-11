# RAW9 窗口锚点的适用范围与重新标定

本轮只修复 scene transform 的解码器窗口传递与其离线标定合同。窗口传递移动的是颜色区域的中心与协方差，不修改原始 RAW，也不产生新的相机颜色矩阵、噪声模型或绝对曝光标尺。

## 运行时范围

新资产使用 `decoder-anchor-transport-3`。每份 profile 精确绑定 make、model、实际 Apple decoder token、OS/build runtime、配置 SHA256。配置身份由 Core Image 实际参数状态和项目 handoff 策略共同形成，必须包含实际 scale 策略与 hot-WB 实现；同一实现下的后级 WB 矩阵仍由既有窗口矩阵传递处理，不把逐文件连续 WB 数值或 BaselineExposure 混成机型配置。

`decoder_anchor_scope(bundle)` 还核对 bundle 中的实际 token/runtime 与 recipe 一致，拒绝缺失身份、fallback、RAW8、未知配置及未确认的 API 参数状态。`parameter-state-verified` 只表示规定参数已写入且回读符合预期，不证明 Apple 输出具有严格线性或独立原生噪声，也不表示 RAW9 联合解拜耳／降噪关闭。

`decoder_window_calibration(bundle)` 返回 `status`、`reason`、`profile_id`、`profile_sha256`、`asset_sha256` 和 scope。LibRaw 为 `identity`；Apple 缺少唯一合格匹配为 `unmeasured`，窗口额外解码器传递使用恒等。两者都不阻断基本显影，不影响显式选择 scene transform 本身。`decoder_anchor_fingerprint()` 和窗口 tag 都绑定资产内容 SHA256；同进程修改资产后，旧 tag 不能继续命中新数据，根缓存也使用该指纹。

旧默认 fp 和 iPhone 数值保存在 `archived_legacy`，状态为未合格。它们缺少实际配置身份、输入 hash 和独立空间／留出证据，不能迁移为新的合格测量。当前发布资产的 `profiles` 为空。未知机型不会借用 fp、iPhone 或 default；缺少逐类支持也不会冒充该类实测。只有该精确 scope 的独立全局传递本身合格时，未知类才可使用其明确的全局近似。

## 标定入口与证据

`tools/calibrate_raw9_anchors.py` 不再隐式扫描 Pictures，也不在测量结束后直接修改资产。当前支持专门拍摄的平坦颜色目标 ROI；普通自然图像的中心裁剪、粗块缩小或两路相同尺寸均不证明配准。工具不再采用“粗块不受畸变影响”或“LibRaw 不处理 GainMap”的假设。

清单使用 `raw9-anchor-manifest-1`，例如：

```json
{
  "schema": "raw9-anchor-manifest-1",
  "target_kind": "dedicated-flat-color-target",
  "actual_decoder_token": "9.dng",
  "scale_mode": "unity",
  "half_size": false,
  "spatial_support": {
    "kind": "flat-target-rois",
    "verified": true,
    "verification_note": "人工确认每对 ROI 覆盖同一物理平坦色块，且边界留有余量。"
  },
  "frames": [
    {"path":"train1.dng","split":"train","rois":[{"id":"patch1","libraw":[100,100,200,200],"coreimage":[105,102,205,202]}]},
    {"path":"train2.dng","split":"train","rois":[{"id":"patch1","libraw":[100,100,200,200],"coreimage":[105,102,205,202]}]},
    {"path":"holdout1.dng","split":"holdout","rois":[{"id":"patch1","libraw":[100,100,200,200],"coreimage":[105,102,205,202]}]},
    {"path":"holdout2.dng","split":"holdout","rois":[{"id":"patch1","libraw":[100,100,200,200],"coreimage":[105,102,205,202]}]}
  ]
}
```

坐标是各自实际输出 raster 的 `[y0,x0,y1,x1]`，示例数值不适用于任何现有样张。至少两个训练文件和两个独立留出文件；文件内容重复不能增加支持量。文件 hash 只能检查重复内容或内容变化，不能独自证明拍摄在统计上独立；采集与 ROI 对应仍需真实拍摄记录及人工核验。可选的帧 `sha256` 声明与实际文件不符时拒绝。

工具同时显式请求 LibRaw 与固定的实际 RAW9 token，拒绝任何实际 fallback、混合机身／runtime／配置、未知 LibRaw 参考版本及未实现的白平衡。当前标定限 `unity` scale；未来其他策略需要对应的独立测量，不能复用该 scope。对 Apple 不能从 RAW 掩码定位的损失，当前采用保守的整帧排除：RAW 已剪切或参考处理损失非零／未知时不将它当作合格支持。

每个 ROI 分别检查实际尺寸、内缩两像素后的有效支持、有限正通道、低信号与近饱和排除、局部色度变化和空间梯度。没有几何重采样。当前不实现通用图像配准，不能只填一个配准标记就启用自然场景标定。后续配准支持应有独立几何验证、残差与重叠掩码，再扩展合同。

全局与逐类分别报告训练／留出文件数、训练范围及留出误差。空间像素不当作独立拍摄重复。当前工程门槛为比例 0.65–1.30、留出相对误差中位数不超过 3%、95 百分位不超过 6%；这些是拒绝不稳定测量的工程门槛，不是普遍画质或物理精度保证。越界比例原样记为 rejected，绝不 clamp 成合格。逐类样本不足保持 unmeasured。

```sh
python tools/calibrate_raw9_anchors.py target-manifest.json --report raw9-candidate.json
python tools/calibrate_raw9_anchors.py --install-candidate raw9-candidate.json
```

第二步才更新资产。安装前重核输入文件 hash、候选 profile hash、空间与独立留出合同；候选失败或文件已变化均拒绝。测量失败仍输出原因明确的 unmeasured 报告。安装是显式本地动作，不包含 Git 提交或推送。

## 本轮验证边界

合成合同回归覆盖精确 scope 正例、机型／make 差异、RAW8／LibRaw／fallback、token 或 runtime 扁平字段冲突、缺身份／参数状态不明、旧默认资产、歧义 profile、资产变更、真实空间结构拒绝、越界 ROI、已知处理损失、留出失败、越界比例不 clamp，以及候选安装前输入变更。合成平坦色块仅验证工具计算与合同，不是 fp 或 iPhone 标定成果。

当前没有提供专门拍摄并核验的平坦颜色目标清单，也没有独立验证的自然图像配准证据。因此本轮不生成任何真实相机合格锚点；现有普通样张不能被冒充为该标定素材。实际 RAW9/RAW8 的能力与 recipe 验收由本轮独立解码验证工具记录，不能替代锚点标定的空间与留出验收。
