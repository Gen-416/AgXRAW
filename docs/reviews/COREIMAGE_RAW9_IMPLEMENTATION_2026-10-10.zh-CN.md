# Core Image RAW9 适配复审：实施与验收

在 `ad57ea2419484b2a2f61d9e2cbbc31b89a1df7ff` 工作区继续实施本轮任务书。远端 main 经获取仍为该提交；此前 Collect／标定／色度清理的未提交工作保留。本轮没有提交或推送。

后续状态：上述未提交状态是历史执行快照，基础改动已随 PR #177 合入 `c4e0c8e`。当前 recipe v2、窗口工具源事务及后续状态见[解码配方](COREIMAGE_DECODE_RECIPE_2026-10-10.zh-CN.md)和[PR179 收尾](CALIBRATION_SCOPE_LEGACY_2026-10-11.zh-CN.md#pr179-后续收尾2026-10-11)；原实机计数与当时配置身份不改写。

独立 RAW evidence 与 Apple 成像继续分离。没有移植 CFA 方差公式到 Apple 输出，没有取消 `opaque-decoder-noise-transfer` 保护，也没有默认开启第二层色度处理。普通 AgX 显影继续可用；缺标定只降级相应能力。

## 修复与当前能力

|任务|实现与验收口径|
|---|---|
|R9-01|保留上一轮按内容识别官方 Collect ZIP／目录、分组与逐能力报告；固定上游 writer 的实际命名夹具继续通过，不要求用户重命名 gain／PTC|
|R9-02|材质窗口绑定精确 make/model、实际 token、runtime、有效配置与资产 hash；缺身份、冲突、RAW8、回退、未确认必要参数或未知相机返回未测量；旧 fp default 数值归档，合格 profiles 当前为空|
|R9-03|标定改为专用平坦目标清单、明确 ROI、独立训练／留出、固定实际 RAW9；拒绝回退、混合版本／配置、未排除损失与异常比例；先出候选，再显式安装，不以 clamp 掩盖异常|
|R9-04|版本化实际 recipe、逐控件支持／设置／读回／语义、真实渲染 extent、实际尺度、源 hash 与项目 handoff；完整保存在 bundle、缓存、GUI 诊断与 CLI CSV 中|
|R9-05|专用实机工具分别测试 RAW9、RAW8、LibRaw、auto 与全尺寸、原生预览、全尺寸后缩图；固定版本缺实际读回不能计通过；不可用与失败分别登记|
|R9-06|独立的输出残差离线产品，要求不同拍摄帧／源 hash、固定完整逐帧变换与精确输出栅格；报告时域协方差、频带、均值与固定残差候选，不把普通照片纹理或同 RAW 重渲染当噪声测量|
|R9-07|输出偏置仅为独立训练／留出的离线候选，没有生产 hook；个人黑位／DSNU 未应用到 Apple CFA；镜头控制读回不冒充逐 opcode 执行历史或像素对应证明|
|R9-08|可选 context 常量访问已保护；无完整测量身份支撑的固定百分比注释已清理；运行时失败探测与自动回退可重试|

稳定配置 ID 包含 Apple 实际参数、token、runtime、硬件和实际解码尺度，以及项目尺度策略、参考解码版本／高光策略与热 WB 实现。逐图源 hash、AsShot、BaselineExposure、实际 alignment gain、用户 WB 倍率／矩阵另存，不把每张图变成一个“机型配置”。残差研究同时校验这些逐图变换，不能混用 camera 与 daylight 图像。

参数归零只表示参数状态。RAW9 联合去马赛克／降噪仍有效；`parameter-state-verified` 不证明整个形成算子是线性系统，也不表示传感器随机噪声透明。实际 render 使用的 CIImage extent 随同一次渲染收集，不再靠第二次 `outputImage()` 查询推断。

## 缓存迁移与导出

预览缓存升至 **33**，旧项自动重建。源缓存身份加入 recipe schema 和窗口资产 hash；分析／成像缓存保留实际 recipe 与所用标定。完整 Analysis 比较只忽略 auto／固定请求、offered tokens 与回退选择日志，仍核对所有有效参数、源、几何和项目 handoff。

瞬时 auto 回退不写磁盘；内存暂用 60 秒，下次准备照片重试。当前屏幕上的结果仍可被导出固定，不能在后台悄悄换算法。导出核对实际 token、runtime、配置和窗口标定，变化则要求刷新预览。失败的 runtime context 探测也有 60 秒重试及显式重试入口；没有定时后台任务。

## 实机和研究边界

实机系统为 macOS 27.2 / build 26B5101f，Python 3.14.4，rawpy 0.27.0+libraw.e419de08 / LibRaw 0.22.0。本地 Sony ILCE-7M5 ARW 实际提供 RAW7／RAW8；不能计作 RAW9 支持。CC0 Nikon D750 12-bit lossless NEF 提供 RAW6／7／8／9；该文件不能代表 Z 系列 HE／HE*。

普通照片的执行矩阵只建立固定输入／版本／配方基线，不证明解包 bit-exact、原始纹理无损或新增降噪收益。原生预览与全尺寸缩图的分布、协方差及频带结构统计包含真实场景和噪声，不是独立随机噪声测量；无配准时不据此判定逐像素损伤。SDR／HDR 此处指浮点成像结果，不冒充新一轮 gain-map 容器编码验收。

缺少匹配机身的独立暗场／平场、弱色纹理和留出参考，尚不能建立合格 DecoderResidualCalibration、新的真实相机窗口锚点或附加细色度处理。Nikon HE／HE*、其他 Sony codec／读出与组件迁移 sentinel 仍缺测。用户原始样张不修改、不提交；公共测试原片仅临时使用。

详细合同分别见[窗口锚点](RAW9_ANCHOR_SCOPE_2026-10-10.zh-CN.md)、[解码配方](COREIMAGE_DECODE_RECIPE_2026-10-10.zh-CN.md)和[实机工具／残差研究](COREIMAGE_RAW_RESEARCH_2026-10-10.zh-CN.md)。上一轮的 [LibRaw NumPy／Rust 实际对照](../assets/pipeline/calibrated-native-raw-20261010.json) 为当时源码的历史证据，不计入本轮 Apple 结果。

## 本轮最终验收

冻结处理代码后，两套完整回归均成功。它们是同一组 2050 个用例的两种配置，不能相加为独立覆盖量。

|配置|总数|通过|跳过|失败／错误|耗时|
|---|---:|---:|---:|---:|---:|
|Rust，`DNGSCAN_FAST=1`|2050|2047|3|0|301.168 s|
|NumPy，`DNGSCAN_FAST=0`|2050|2015|35|0|748.290 s|

最初的 Rust 全量回归发现两个旧 MagicMock 缓存夹具未显式提供合法配方字典。补齐夹具后重跑了以上两套完整回归，没有放宽生产缓存的证据检查。日志 hash、处理源码 hash、实际依赖、窗口资产和测试工具身份保存在[最终机器记录](../assets/pipeline/coreimage-validation-20261010.json)。纯 NumPy wheel 的隔离安装检查也通过：新模块、配方 schema、缓存版本 33 与窗口资产均正确打包；该检查没有重建 native wheel。

[真实 Sony／Nikon 矩阵](../assets/pipeline/coreimage-raw-20261010.json)共 42 格：36 通过、6 版本不可用、0 失败。Sony 为 15 通过和 6 RAW9 不可用，auto 实际选择 RAW8；Nikon 为 21 通过，auto 实际选择 RAW9。36 格源 hash 不变，scene／SDR／HDR 均有限，18 格保留了负值、36 格保留了大于 1 的值。代码、工具和标定身份在格内、格间及结束后复核一致。两份普通照片的结果不能扩展到全部机型、codec 或独立画质资格；并行运行时的耗时与 RSS 也不能作为受控性能排名。

另对这两个文件实际执行了完整尺寸 auto 解码、分析与预览代理建立，再通过 GUI 的导出加载入口固定实际 token。两份文件都保持相同有效配置；完整 Analysis 经 envelope 与隔离磁盘缓存均实际命中，临时缓存随后删除。fp 的独立 RAW9 原生预览配方检查确认实际 token 9、854×1280 float16 输出及同次渲染 extent，完整读回记录也保存在上述机器记录中。它们验证配置与接缝，不证明纹理或噪声改善。

本轮没有新增真实相机合格窗口 profile，也没有启用 Apple 后级色度 NR、个人黑位／DSNU 的 CFA 注入或研究偏置图自动应用。缺少独立暗场／平场／PTC、弱色纹理留出、HE／HE* 与系统组件迁移样本的项目仍按缺测列出。没有提交或推送；临时公共 Nikon 原片在验收后删除，用户原片保持原样。
