# ISO 5-3 Status A / Status M 密度计光谱响应度

`status_{A,M}_responsivity_{r,g,b}.csv`：(波长 nm, 线性响应度) 两列。

出处：Giorgianni, Madden & Kriss, *Digital Color Management* (Wiley 2009)
p.335 的数字化，经 agx-emulsion 项目 v0.2.0-legacy
(`agx_emulsion/data/densitometer/`, GPL-3.0) 转载。

用途：`tools/crosscheck_2383.py --input divere-status` 的密度计量投影
（第五轮 DiVERE 交叉验证）——把染料组的光谱透过率投影为 Status A（印片）
/ Status M（负片）读数，使两条链在同一密度计量域内比较。不参与运行时渲染。
