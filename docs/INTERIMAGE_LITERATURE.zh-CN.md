# 层间效应文献锚 / Interimage Literature Anchors

数据文件:`dngscan/data/interimage_literature.json`(单文件可撤,不被渲染管线消费)。
本文档记录来源、数字含义、与现行 `INTERIMAGE_BETA` 表的对照分析,以及未来收窄 β 的路线。
侦察与转录:2026-08-25(多 agent 文献侦察 + 专利全文逐表转录)。

## 1. 来源与数字

三份免费专利全文给出负片体系的直接定量:

| 来源 | 体系 | 关键数字 |
|---|---|---|
| [US5942381](https://patents.google.com/patent/US5942381A/en) | C-41 双层实验(绿 causer→红 receiver) | 无 DIR:γ 1.00/0.99;商用对比剂压 receiver 13%(→0.86);发明剂压 34%(→0.65),causer 自层仅小降 |
| [US6004737](https://patents.google.com/patent/US6004737A/en) | C-41(蓝 causer→绿 receiver) | receiver/causer 比 R:无 DIR 0.84–0.85 → 商用剂 0.79–0.80 → 发明剂 0.70 |
| [US4830954](https://patents.google.com/patent/US4830954A/en) | C-41 整卷 | IIE 阈值 Y≥10%/M≥25%/C≥15%;实施例 5–35% 区间;感光度不付代价(DIN 表) |

IIE 的标准测量定义(US4830954 所引):**分色曝光相对白光曝光的色彩梯度百分比增量**,出处
Mees & James《The Theory of the Photographic Process》4th ed (1977) pp.574 & 614
([archive.org 借阅](https://archive.org/details/theoryofphotogra0004edmees));方法学原典
Hanson & Horton, JOSA 42, 663–669 (1952)(付费墙)。

反转片体系另录 [Fuji IS&T 1997](https://www.imaging.org/common/uploaded%20files/pdfs/Papers/1997/IST-0-4/62.pdf)
的机制结论与 Figure 2 文字结论(ASTIA 100 红层高光 IIE 大于 PROVIA 100);其曲线为两两重叠细噪线
且属 E-6 体系,按"并线区不猜"原则未数字化。

## 2. 与 INTERIMAGE_BETA 表的对照(诚实版)

现行 β 表(`dngscan/film_develop.py`,modelled 声明):Portra 族 0.62–0.67、Ektar 1.05、
消费负片 0.66–0.75、Pro 400H 0.32、电影负片 ~0.40。

**不能直接把 β 与文献 IIE% 画等号**,两者定义不同:

- 文献 IIE% 是"分色 vs 白光曝光的梯度增量"(在 D-logE 上测);
- 我们的 β 作用在有界映射 `t'=(1+β)t/(1+βt)` 的层间分离项 t 上,小分离处斜率为 (1+β),
  大分离处被轨道钳制——β 的名义值只在小振幅处兑现;
- datasheet 中性特性曲线已烘焙**中性 IIE**,β 补的是**分色增量**部分——与 IIE% 的测量对象
  一致,但换算须经过我们链条的 C(logĒ) 中性基准与分离度定义。

在此前提下的量级观察:β 名义值(0.32–1.05)高于专利 IIE 区间(0.05–0.35),但小振幅斜率
增量 β 在轨道钳制下的**有效**梯度增量随分离度衰减;两者是否矛盾取决于典型分离度落点,
这需要一次专门推导(把 IIE 测量协议在我们链条里数值复现:对模拟的分色/白光曝光对
测梯度增量,读出"等效 IIE%"),而不是拍脑袋换算。**该推导是收窄 β 的正确下一步**,
记为待办;做完之前,β 保持 modelled 声明不变,本文献集的作用是提供裁决时的外部区间。

## 3. 交叉验证点

spektrafilm 的 DIR 默认参数(同层 gamma 修正 0.27–0.34、层间矩阵 0.15–0.36、扩散 20µm+200µm 尾)
转录于数据文件——作者未标出处,视为手调/反推,只作模型间比对素材,不作文献锚。

## 4. 未来可解锁项

- **等效 IIE% 数值复现**(上文 §2):零成本纯计算,是 β 从 modelled 走向半实测的钥匙。
- Mees & James 4th ed pp.574/614 借阅翻拍:方法定义的权威原文。
- 90 年代 Kodak DIR 专利族(US6174662/US5989798/US5041367 等)按 US5942381 的模式
  大概率同样附 causer/receiver gamma 表,可继续扩充本数据集。
- DIR 空间扩散(µm 半径)见 Panning RIT 1978(https://repository.rit.edu/theses/4816/,免费 PDF),
  服务于 MTF 低频邻接峰缺口,属"色彩分离×感觉层"交界,oracle 后评估。
