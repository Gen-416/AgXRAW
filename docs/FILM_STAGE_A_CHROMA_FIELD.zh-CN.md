# Stage A 色度场:测量记录与采纳判据

状态:**阶段 2(运行时接入)已实现**——schema 7 资产携带 CV 选型的色度场
LUT,`film_v2_math.stage_a_log_exposure` 按 stock 调度;24/25 stock 采纳,
pro400h 按边际规则保留 3×3。本文是路线 C(两路线纲领,2026-08-26)的决策
记录:Stage A 是否值得从固定 3×3 observer 升级为保持曝光齐次性的色度场。

## 问题

部署中的 Stage A 用受约束的 3×3 observer 逆(`build_full_lut.observer_matrix`,
NNLS + 白板锚)把场景 Rec.2020 映到三层胶片曝光。其 held-out 层曝光误差
`observer_p99_stop` 在各 stock 上为 0.47–1.13 档——这是任何**线性**三通道映射
的边界,其中混合了两种成分:三刺激→三层响应的同色异谱损失(原理性,不可消除),
以及线性模型本身的拟合不足(可消除)。本测量把两者分开。

## 候选形式

$$E_{\text{layer}} = Y \cdot 2^{P(x,y)}$$

每层一个 CIE 色度 $(x,y)$ 上的多项式 $P$(log2 域 ridge 最小二乘,白板锚点
精确)。曝光齐次性 $E(k\mathbf{X}) = kE(\mathbf{X})$ 由构造保证——$Y$ 外提,
$P$ 只见色度。折划分、训练集(rawtoaces 190 条反射光谱 + 白板)、D55 刺激
构造与基线**完全一致**;基线每折用生产版流程(NNLS + 锚)重拟合,对照测的
是模型族,不是别的。

## 结果(5-fold held-out,全部 20 支底片)

完整数字见 [chroma_field_cv.json](chroma_field_cv.json)(工具:
`tools/fit_chroma_field.py`,种子固定)。三次场(每层 10 系数)是稳健甜点
——四次增益边际且在低误差 stock 上出现过拟合迹象。

- **中位改善:p95 −39%,p99 −31%**。
- 高误差 stock 改善最大:ektachrome100 p99 1.078→0.609(−44%)、
  velvia100 1.022→0.604(−41%)、vision3250d 1.133→0.785(−31%)。
- 两个例外正是 3×3 下误差最低的两支:superia400 p99 持平(p95 仍 −27%),
  pro400h p99 0.485→0.500(+3%)——同色异谱压力小的 stock,线性模型已够。

结论:先前 0.47–1.13 档误差中约三分之一到近一半来自**线性假设**而非同色
异谱本身。剩余 ~0.46–0.79 档是三通道确定性映射的原理边界,不可再消除。

## 采纳判据与阶段 2 实现

Owner 判据(2026-08-26):held-out p95/p99 显著下降才采纳。**已满足并落地**:

1. **选型规则**(`build_film_v2_assets.CHROMA_P95_ADOPT_RATIO`):held-out
   p95 下降 ≥15% 且 p99 不劣于 3×3 才采纳。24/25 采纳;pro400h(p99 +3%)
   保留 3×3。决策连同 CV 数字烘进每个资产的 `chroma_cv_note`。
2. **域外守卫**:LUT 覆盖训练 XYZ 空间中 Rec.2020 原色三角形的包围盒;
   训练凸包内为三次场,包外为 observer 自身的色度响应,高斯混合带
   (σ=2 cell)连续过渡。运行时输入逐通道钳正,色度**不可能**离开原色
   三角形,因此只有退化输入(非有限)走精确 3×3 回退。烘焙后整表按
   白色度双线性采样重锚,灰轴与 observer 的 logE 轴位齐(实测 ≤5e-7)。
3. **oracle 同路**:stock 与 print-state 的 oracle_truth 经由与运行时
   完全相同的 `stage_a_log_exposure` 调度生成(`chain_eval` 的 stage_a
   注入),运行时-oracle 门在场化后全部保持(print p99 ≤0.023 stop)。
4. **声明式冻结重钉**:外观冻结 9 项、光学冻结 10/10 漂移
   (linear_max_abs 至 0.09,high-key portra400)——这正是升级要改变的
   逐像素色彩分离;中性轴、EV0 中灰、曝光齐次各 gate 无一移动。

光源分档(D55/A/高显色 LED)是其后的扩展(路线 D);按 WB 在档间插值
必须称"光源假设",不得称测量。
