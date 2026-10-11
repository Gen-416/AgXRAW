# 镜头相对光谱：语义修复与离线颜色实验

本轮对应任务 D01–D04。D01 的数据迁移与 D02 的精确身份合同已实现；D03 的中性响应与 D04 的白约束矩阵／协方差是离线研究工具，未接入默认显影。`lens_transmittance.py` 从来不代表已经应用镜头颜色校正，本轮继续明确 `pixel_correction_applied=false`。

## 固定来源与迁移

来源固定到 [y-g-jiang.github.io@504e119](https://github.com/y-g-jiang/y-g-jiang.github.io/tree/504e119bcfb9ff2b58fc9e47c204ddb77eee5879)。同时读取每个测量 JSON 和 [JTJS 的实际 bundle 索引](https://github.com/y-g-jiang/y-g-jiang.github.io/blob/504e119bcfb9ff2b58fc9e47c204ddb77eee5879/assets/index-Du5lx1zq.js)，没有执行上游 JavaScript，也没有移植模拟器代码。本轮 NumPy 数学实现为原创；数据按项目现有授权与 NOTICE 保留作者来源。

118 条记录包含 84 条 Lens、34 条 Filter，是测量次数而不是独立镜头型号数。迁移核对确认 ID 集合和全部旧字段逐项相同，谱值没有重新归一化或改写。新字段包括 `measurement_id`、`quantity=relative_spectral_transmittance`、`semantic_schema`、源 commit、原始 JSON hash、JTJS bundle/index hash、归一化前相对最大／最小值、`optical_stack`、`is_bare_lens`、测量条件缺测状态。

例如 [Contax G35 原始测量](https://github.com/y-g-jiang/y-g-jiang.github.io/blob/504e119bcfb9ff2b58fc9e47c204ddb77eee5879/lens-transmittance-data/0612-contax-g35.json) 的 `relativeMaxBeforeScaling=1.12528465` 被保留，但不能据此恢复绝对透射率。曲线仍为峰值 100 的相对谱形。

[APO Lanthar 50](https://github.com/y-g-jiang/y-g-jiang.github.io/blob/504e119bcfb9ff2b58fc9e47c204ddb77eee5879/lens-transmittance-data/0703-3-apo-lanthar-50.json) 的 Hoya HMC UV(c) UV，以及 Loxia 25/2.4 第二条测量的 B+W UV，来自索引中的 `measurementNote`，不能仅从独立测量 JSON 发现。两条均保留 `is_bare_lens=false`。索引没有为其余 82 条 Lens 明确声明裸镜，所以它们保持 `unknown`，没有把字段缺失补成 true。合成的 `reference-coated-50mm` 不属于这 118 条实测资产；若未来出现 Reference 记录，精确匹配也禁止它冒充实测。

`format=dngscan-lens-transmittance-1`、`list_entries()`、`get_curve()` 保持兼容。`get_metadata()` 返回可独立修改的元数据副本；`match_exact()` 只匹配确切名称、类别、测量 ID 与光学组合，重复名称返回 ambiguous。默认要求明确裸镜；未知／附加 UV 不会匹配裸镜。高级调用者可指定 ID 并显式允许未知组合，用于诊断研究，这不会获得像素校正资格。

`mean_transmittance()` 仍返回所选波段的无权样本均值；文档不再把它描述为人眼加权透光率。`warmth_ratio()` 只是两波段相对谱形之比，不返回色温、WB 倍率、T-stop、电子数或噪声收益。

维护者可用以下离线流程重建资产，输入需要是固定版本的目录与 bundle：

```sh
python tools/import_lens_transmittance.py /absolute/path/to/lens-transmittance-data \
  --jtjs-bundle /absolute/path/to/index-Du5lx1zq.js \
  --source-commit 504e119bcfb9ff2b58fc9e47c204ddb77eee5879 \
  --out dngscan/data/lens_transmittance.json
```

bundle 解析只支持已审计布局，并将字面量转换为 JSON；不会 eval JS。新布局必须先审查或使用带来源/hash的 `--index` 数据型 JSON，不能默默使用其他版本。

## 离线光谱合同

实现为 `dngscan/spectral_color_research.py`，不会被 render 默认导入。输入要求：准确机身 ID／响应来源、严格递增波长、RGB SSF、energy/photon 约定、共同通道归一化或已知 prescale、SSF 是否包含参考镜头，以及准确参考光路身份。未知 prescale、未知参考光路、缺失响应约定均拒绝；同 CMOS 的其他机身不做自动替代。`match_camera_response()` 只能得到 matched-research，不授予正式校正资格。

完整相机 SSF 已包含参考光学组合时，计算 `SSF × T_target / T_reference`；仅传感器响应则乘目标谱一次。两个相对光学谱各自消除任意整体常数，这明确排除了以其整体倍率估计曝光或光子数。能量／光子输入转换明确使用波长因子。积分使用测量范围内的线性插值和梯形积分，不补零、不延长端值。380–755 nm 的镜头谱不能覆盖 380–780 nm 的 SSF：默认拒绝；只有调用者显式指定交集范围才可实验，并在结果记录被截取范围。

`neutral_white_balance_suggestion()` 的 WB 状态为 reference-optics 才给相对建议；AsShot、当前镜头灰卡及已包含目标光路的 profile 返回 already-compensated 和恒等倍率。unknown 返回 unavailable，避免二次补偿。即使给出建议，也不会自动应用到照片。

`fit_white_constrained_matrix()` 解带精确白点约束的最小二乘，拒绝病态／秩不足训练集。`propagate_color_covariance()` 执行 `M Σ Mᵀ`，拒绝非法矩阵与协方差。矩阵不是任意反射谱的精确逆变换；RGB 三值未包含完整反射谱信息。

## 可重复实验与结果

运行：

```sh
python tools/validate_spectral_color.py --output docs/assets/pipeline/lens-spectral-20261009.json
```

[机器报告](../assets/pipeline/lens-spectral-20261009.json)记录算法身份、输入内容 hash、源码 hash、矩阵、训练与留出 ID、每个样本 ΔE76／色相／色度／亮度误差及尾部。示范使用实测 Contax G35 相对谱、**合成**三高斯相机 SSF、合成光源及合成材料，明确限制到 380–755 nm。它不是 Sigma fp 标定，也不是真实相机画质成绩。

训练含 32 个合成反射谱及两个合成宽带光源；留出为 24 个独立反射谱、一个中性目标及两个未训练光源。两组形成都先按各自光路的中性目标 WB，矩阵用于剩余颜色响应差异。训练和留出 ID 及原始光谱样本均不能重复。

|留出光源|未加候选矩阵的 ΔE76 p95|候选矩阵 ΔE76 p95|原最大值|候选最大值|
|---|---:|---:|---:|---:|
|合成窄带 LED|1.17159|1.62034|1.70567|2.64602|
|合成弯曲宽带|2.02725|1.66534|2.96882|2.88774|

矩阵精确满足白点，最大白点误差为 0；中性样本误差约 `4.44e−14`。但 LED 留出明显变差，说明中性正确、某个光源平均改善也不能保证一般颜色更好。因此 D04 的自动矩阵路径保持关闭，不能把此示范称为生产 profile 验收。

对带交叉项的输入协方差，用 250,000 次独立高斯抽样核对矩阵传播；最大协方差误差相对预测最大方差约 0.003314（0.3314%）。这是线性数学合同的验证，不是完整自适应解拜耳噪声模型。

`--input` 可读取 `schema=agxraw-spectral-color-experiment-1` 的离线数据集。字段见 `demo_dataset()`：camera、target/reference_optics、training/validation_illuminants、training/validation_reflectances、reference_rgb_to_xyz 及来源、显式 wavelength_range_nm、white_balance_policy。每个 Spectrum 具有 measurement_id、wavelengths_nm、values、quantity。加载后仍强制全部资格和独立留出检查，不会安装生成的矩阵。

## 当前支持边界与验证

定向测试 `tests.test_lens_transmittance`、`tests.test_lens_spectral_contract` 共 21 项通过，覆盖相对谱缩放不改颜色、相同光路恒等、UV 组合不匹配裸镜、重复名称保留 ID、未知 SSF/prescale 拒绝、已补偿 WB 不重复处理、波长缺口不外推、能量／光子一致性、白约束矩阵及协方差 Monte Carlo、独立留出不得偷换训练样本。

Sigma fp 没有被本资产中的机身 SSF 测得，Sigma 35mm F2 也没有对应测量；默认成像继续使用原有 WB／颜色处理。没有借用 A7 III 的响应替代 fp，也没有由中心透射谱制造空间偏色图。真实灰卡／色卡、机身准确 SSF、独立照明和空间平场仍需采集；在这些证据缺失时交付的是语义准确的诊断与可重复研究工具。
