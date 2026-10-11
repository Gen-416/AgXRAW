# PR177 管线合同复审修复

基线：`7004fafd60c79864e9939bf79b8ce74d82e85bac`。本轮落实用户提供的 PR177 管线复审，保留已验证的噪声模型、RAW9 独立 evidence、材质窗口资格和 SDR/HDR 形成结构。没有增加默认降噪、修改原片或放宽交付质量门限。

## 修复与执行分支

|复审项|本轮结果|
|---|---|
|PR177-01：必需线性化控制未确认仍接受 Apple scene|必要控件读回必须确认目标状态，才在 render 前取得 scene-input 资格。auto 继续尝试其他实际 offered token；固定版本明确失败。官方无效的 RAW9 旧 NR 参数不致命。setter 抛错但 getter 已确认目标时可以接受，异常仍记录。|
|PR177-02：固件／机身约束无文件入口|新增有边界的 TIFF/EXIF/DNG 与 Nikon type-2 MakerNote 适配器，独立保留值、来源和 verified/missing/invalid/conflict。接入正常 RAW evidence，不从标定复制身份，不放宽未知值匹配。|
|PR177-03：分组条件与 profile 不一致|保存逐表采集合同、条件覆盖及逐能力资格。唯一组关联不等于共同测量条件已证明；缺条件的组与运行时均保持 partial/unverified。独立 CSV 能力组合保留各自物理域、统计合同和来源。|
|PR177-04：源文件在读取期间变化仍成功|从 metadata 前到最终 bundle 提交使用同一源事务。确认变化则整次失败，不能带旧 evidence 换 decoder 重试。后期 I/O 导致无法验证与明确变化分开；初始身份不可得仍保持未知。|
|LEGACY-01：所有 LibRaw 错误被误报格式不支持|预检和 evidence 共用分类，区分不支持 codec、损坏数据、I/O 与运行失败。移除由 `.nef` 后缀推断 HE/HE* 和授权原因的说明。Apple-only 形成资格与传感器 evidence 资格分别判断。|

Collect 的详细合同、迁移及分包反例见 [同轮实施说明](PR177_COLLECT_ACQUISITION_2026-10-10.zh-CN.md)。当前 runtime 仍使用 profile-wide 的物理域；某张参与表缺必要条件时，不会只因为另一能力字段完整就放行整份统计模型。基础显影继续可用。

## 文件身份适配器范围

当前支持 EXIF `BodySerialNumber`（ExifIFD `0xa431`）、DNG `CameraSerialNumber`（IFD0 `0xc62f`）、Nikon type-2 MakerNote `SerialNumber`（`0x001d`）。不同来源的序列号冲突时返回不可验证，不静默挑一个。Nikon MakerNote `0x0001` 是 MakerNote 版本，不作机身固件。

固件适配器限定在原生文件的明确相机软件语法：Nikon NEF 的 `Ver.<major>.<minor>`，Sony ARW 的 `<与 Model 完全一致的型号> v<major>.<minor>`。转换 DNG、泛用编辑软件版本、不完整版本字符串不建立固件事实。其他厂商／布局暂保持 missing，而不是伪造支持。元数据是文件声明，不是密码学真实性证明。

Nikon TIFF 压缩码 `34713` 单独不能证明无损；只有原生 NEF 的 type-2 MakerNote `NEFCompression`（`0x0093`）明确为 `3` 时，才记录该已识别的无损存储声明。其他值和缺失布局保持未知，未把 HE/HE* 当成无损，也未从扩展名推断模式。Sony 的已识别有损编码继续拒绝外部传感器统计 prior。

适配器限制 IFD 数量、文本和 MakerNote 字节量，检查数据类型、偏移、重复标签与截断，支持 TIFF 和 MakerNote 各自的字节序／相对原点。匹配保留原始身份值；GUI 诊断和验证工具公开报告只输出机身序列号哈希及来源。

标签定义核对 [ExifTool 官方 EXIF 标签表](https://exiftool.org/TagNames/EXIF.html) 和 [Nikon 定义，固定版本 2200871d](https://github.com/exiftool/exiftool/blob/2200871d9cef988051d2a99d67df3bda6cbb30a8/lib/Image/ExifTool/Nikon.pm)。本项目使用原创的有界读取器，没有引入 ExifTool 运行时或移植其解密代码。Sony/Nikon 固件字符串适配另由本轮实际 ARW/NEF 字节验证，覆盖范围不外推到任意厂商文件。

## 缓存与迁移

Apple recipe 升至 `coreimage-decode-recipe-2`，包含 `scene_input_qualification`、必需控件及资格政策身份。资格仅表示必要 API 状态确认，不等于证明 Apple 形成算子的经验线性或噪声传播。

预览缓存由 33 升至 **34**，旧代理／Analysis 不复用，首次重新生成；这是对新增文件身份和条件合同的迁移。auto 导出继续固定已显示的实际版本，并核对有效配方和窗口身份。临时 fallback 不永久写盘，保留已有重试策略。

此前通过 manifest 安装的用户 Collect 包没有逐表合同，无法从 SHA 反推出缺测条件，因此明确显示 `measurement-acquisition-reimport-required`，保留记录供诊断，重新导入原包后获得新合同。传统手工 JSON 和历史公共资产保留既有合同，不凭空补造逐表证据。

## 验收范围

本轮机器记录见 [pr177-contract-fixes-20261010.json](../assets/pipeline/pr177-contract-fixes-20261010.json)。它记录冻结处理源码哈希、两后端完整回归、实际 RAW 矩阵、文件身份匹配与 GUI Analysis 复用、最终 HEIF 交付回读。旧 `coreimage-validation-20261010.json` 和 `coreimage-raw-20261010.json` 保留为原 PR 提交时的历史证据，不代替本轮验收。

身份匹配的实际文件探针使用隔离的合成 profile，仅检验从文件到生产 matcher 的连接，不把其任意 e⁻/DN 数字称作机身实测。Nikon 对正确固件／机身约束匹配，明确不同值拒绝；有损 Sony 即便固件正确也继续拒绝外部传感器统计模型。两者均检查 GUI auto → 固定实际 token 导出，以及 envelope／磁盘完整 Analysis 的实际命中。

真实 HEIF 检查采用原生预览长边 1280、默认 AgX、固定同一分析/曝光计划，使用生产写后回读门限，并独立再次打开核对 10-bit、P3 ICC、方向、尺寸、ISO gain-map 和展开 HDR。它证明指定尺度的编码接线，不代表全尺寸画质验收。

初次固定 share99/420 检查中，fp RAW9 HDR 被既有质量门限拒绝并丢弃文件；同一 scene 使用生产 auto 后通过。该负结果保留，不把所有固定参数记为通过，不为验收放宽阈值。HEIF 的质量值不等同于 JPEG 质量值。

尚缺 Nikon HE/HE* 原片、其他 Sony 编码模式、独立相机暗场/平场/弱色纹理及留出参考。RAW9 不提供的版本记 unavailable，不借其他版本计通过；同一 RAW 重渲染不代替随机噪声或独立解包参考。没有新增默认 RAW9 后级色度降噪，也没有把这些检查升级为真实纹理完全保真的结论。

本轮最终冻结代码在本机完整执行同一组 **2089** 项用例：

|计算后端|通过|跳过|失败／错误|耗时|
|---|---:|---:|---:|---:|
|Rust 严格路径，`DNGSCAN_FAST=1`|2086|3|0|262.142 秒|
|NumPy，`DNGSCAN_FAST=0`|2054|35|0|699.175 秒|

两次运行不相加为独立覆盖量。真实 Sony／Nikon 的 42 项解码矩阵另为 36 通过、6 版本不可用、0 失败；Sony auto 实际为 RAW8，Nikon auto 为 RAW9。最终六个 HEIF 出口均通过，身份／缓存及纯 Python wheel 检查也通过。具体运行时、源文件、配方与工具哈希保存在机器记录中。

## 复现入口

在项目锁定依赖及 ABI 22 native 已安装的 macOS 环境，从仓库根目录执行；两种后端运行的是同一组用例，不能相加为独立覆盖量。

```sh
DNGSCAN_FAST=1 MPLBACKEND=Agg python -m unittest discover -s tests -q
DNGSCAN_FAST=0 MPLBACKEND=Agg python -m unittest discover -s tests -q

python tools/validate_coreimage_raw.py \
  --files /path/to/sony.ARW /path/to/nikon.NEF \
  --matrix raw9 raw8 libraw auto \
  --sizes full native-preview resized-full \
  --scales native aligned --preview-max-edge 1280 \
  --out /path/to/coreimage-matrix.json
```

原生 Apple preview 与 full 后确定重采样分别计数，原生 preview 的实际 scaleFactor 由 recipe 记录，不能一律解释为 0.5。HEIF 机器记录另保存实际编码参数、生产门限与回读指标；这些文件交付检查和上述浮点 scene 矩阵分开计算。
