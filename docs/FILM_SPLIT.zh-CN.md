# 胶片研发拆分记录

[返回文档索引](README.md)

2026-10-09，胶片模拟从 AgXRAW 分离到独立的私有 [**AgXFilm** 仓库](https://github.com/Gen-416/AgXFilm)，暂缓开发。AgXRAW 保留 RAW/DNG 解码与校正、传感器分析、独立噪声标定、AgX、SDR/HDR、编码与回读验证，以及通用白平衡、镜头滤镜和可选显示风格。

胶片型号、观察/冲印模式、色头、冲洗、印相曝光、层间效应、颗粒、halation、介质柔化和外观配方不再是 AgXRAW 的 GUI 或 CLI 选项。原有胶片参数会被拒绝，不能静默变成普通 AgX 导出；旧 GUI 设置中的胶片字段不再载入。胶片特性曲线、色头/光谱资产、Python/Rust 专属核、构建工具、教程、对比图和冻结测试均移入 AgXFilm。

## 数据与历史保全

拆分前完整集成保存在 AgXRAW 提交 `522c7ed2d17d412b9d53615174a3a8a43df66eb9`，含当时尚未提交的噪声标定与色度核修改。AgXFilm 的 `legacy-integration/WORKTREE_SNAPSHOT.json` 保存 976 个文件的 SHA-256，恢复工具直接读取该冻结提交。早期的差量、共用源码和混合文档副本仍保存在 AgXFilm 提交 `37fc9f27eaeb07489117103b3c74db6501720a4f`；当前树移除重复副本，以 Git 保存历史。

AgXFilm 的 `archive/agxraw-history-2026-10-09` 分支还保存本地主仓的所有历史对象，包括旧分支和本地尚未被远端引用的提交。该归档包含原始引用映射与对象清单，不改写主仓历史。
这是保留研究材料与集成接缝的拆仓，并未把旧集成自动改造成可直接 `pip install` 的独立插件。AgXFilm 顶层沿用 `dngscan/` 等历史路径以保存资产身份，恢复实验应使用它的恢复工具重建冻结的旧集成树。

```bash
cd ../AgXFilm
python tools/restore_legacy.py --source-repo ../AgXRAW --output ../AgXFilm-legacy
```

命令从指定 Git 提交重建新的空目录并核验快照清单；不会写回当前 AgXRAW。部分克隆会按需联网下载缺失的 Git 对象。也可以先在 AgXFilm 中执行 `git fetch origin archive/agxraw-history-2026-10-09`，再以 `--source-repo .` 从私有历史归档恢复。恢复后需要在新目录建立独立 Python 环境并重建与该源码一致的 Rust 扩展，不能复用拆分后的主仓扩展。

## 主仓阅读路线

当前操作见[使用说明](USER_GUIDE.zh-CN.md)，数学和数据流见[技术架构](ARCHITECTURE.zh-CN.md)，代码入口见[开发指南](DEVELOPMENT.zh-CN.md)。旧性能记录中的胶片数字保留为注明版本的历史证据，不表示主仓仍提供胶片功能。

拆分只移除当前树中的胶片负担，没有改写 AgXRAW 的 Git 历史；已有 clone 的历史对象体积不会随文件迁出自动缩小。

## 拆分验证

拆分前后 288 组非胶片 SDR 浮点、8-bit 与 HDR 配对数组逐像素一致，32 组自动曝光结果完全一致。Rust 扩展升至 ABI 19，删除胶片专属导出后重新编译、自检和内部测试通过。冻结集成实际重建的 976 个文件逐项 SHA-256 匹配；新的纯 Python wheel 安装验证不再携带胶片代码和资产。

本次移出的当前工作文件约 111.9 MiB，主仓当前工作文件约 35.7 MiB。本地保留正在使用的 Python 环境、ABI 19 Rust 扩展、用户样张和标定数据；可重建的 Rust target、构建目录、缓存、已合并的旧测试工作树和本轮临时备份已经清理。

本地仓库采用 `blob:none` 部分克隆：提交和目录历史保留，旧版本文件按需从 Git 下载。AgXFilm 暂停工作区另采用稀疏检出，仅展开入口、恢复工具和清单；在其根目录运行 `git sparse-checkout disable` 可取回完整胶片工程。私有归档的访问需要 GitHub 凭据。这些设置不改写远端或本地提交历史。

拆分后的完整 Python 回归共 1464 项：NumPy 后端通过（跳过 66 项），严格 Rust 后端通过（跳过 35 项）；15 项 Rust 内部测试通过。平台或可选后端未满足的项目按既有条件跳过。
