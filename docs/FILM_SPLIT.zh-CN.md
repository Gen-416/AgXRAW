# 胶片研发拆分记录

[返回文档索引](README.md)

2026-10-09，胶片模拟从 AgXRAW 分离到独立的 **AgXFilm** 仓库，暂缓开发。AgXRAW 保留 RAW/DNG 解码与校正、传感器分析、独立噪声标定、AgX、SDR/HDR、编码与回读验证，以及通用白平衡、镜头滤镜和可选显示风格。

胶片型号、观察/冲印模式、色头、冲洗、印相曝光、层间效应、颗粒、halation、介质柔化和外观配方不再是 AgXRAW 的 GUI 或 CLI 选项。原有胶片参数会被拒绝，不能静默变成普通 AgX 导出；旧 GUI 设置中的胶片字段不再载入。胶片特性曲线、色头/光谱资产、Python/Rust 专属核、构建工具、教程、对比图和冻结测试均移入 AgXFilm。

## 数据与历史保全

来源提交是 `738da0a214c92244dc3edf42f369f691f0db144a`。拆分前尚未提交的噪声标定与色度核修改已单独提交保存在 AgXRAW；AgXFilm 的 `legacy-integration/PENDING_CHANGES.patch` 和 `untracked/` 另外保存了拆分前完整工作区差量，`WORKTREE_SNAPSHOT.json` 保存逐文件 SHA-256。混合文档和测试原文也保存在 `archive/reference/`，与清理后的当前主仓说明分开。

这是保留研究材料与集成接缝的拆仓，并未把旧集成自动改造成可直接 `pip install` 的独立插件。AgXFilm 顶层沿用 `dngscan/` 等历史路径以保存资产身份，恢复实验应使用它的恢复工具重建冻结的旧集成树。

```bash
cd ../AgXFilm
python tools/restore_legacy.py --source-repo ../AgXRAW --output ../AgXFilm-legacy
```

命令只读取本地 AgXRAW 的指定 Git 提交，将保存的差量应用到新的空目录，并核验快照清单；不会写回当前 AgXRAW。恢复后需要在新目录建立独立 Python 环境并重建与该源码一致的 Rust 扩展，不能复用拆分后的主仓扩展。

## 主仓阅读路线

当前操作见[使用说明](USER_GUIDE.zh-CN.md)，数学和数据流见[技术架构](ARCHITECTURE.zh-CN.md)，代码入口见[开发指南](DEVELOPMENT.zh-CN.md)。旧性能记录中的胶片数字保留为注明版本的历史证据，不表示主仓仍提供胶片功能。

拆分只移除当前树中的胶片负担，没有改写 AgXRAW 的 Git 历史；已有 clone 的历史对象体积不会随文件迁出自动缩小。

## 拆分验证

拆分前后 288 组非胶片 SDR 浮点、8-bit 与 HDR 配对数组逐像素一致，32 组自动曝光结果完全一致。Rust 扩展升至 ABI 19，删除胶片专属导出后重新编译、自检和内部测试通过。冻结集成实际重建的 976 个文件逐项 SHA-256 匹配；新的纯 Python wheel 安装验证不再携带胶片代码和资产。

本次移出的当前工作文件约 111.9 MiB。主仓当前工作文件约 35.7 MiB。本地 Git 对象经过正常压缩，`.git` 从拆分开始时约 1.4 GiB 降至约 574 MiB，所有历史提交仍保留，`git fsck` 通过。这是存储压缩，没有改写历史；若以后确需缩减远端历史，应另行评估已有 clone 的迁移。
