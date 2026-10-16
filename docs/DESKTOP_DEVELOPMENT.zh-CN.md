# 独立前端与 Tauri 开发入口

本轮基于 `main@bb513f9` 整理共享应用服务，再搭建独立前端和 Tauri 入口。
现有成像算法、默认效果、政策版本、预览几何及输出质量门限保持不变。

## 职责与入口

| 目录 | 职责 |
|---|---|
| `dngscan/application/` | 预览／导出用例、标定编排、会话代际、共享缓存和任务调度 |
| `dngscan/application/contracts.py` | 版本化请求、界面初始声明和 TypeScript 类型来源 |
| `dngscan/application/api.py` | 命令路由、参数边界和结构化错误；不重新解释显影规则 |
| `dngscan/transports/` | 可选 FastAPI 适配、浏览器上传、后端进程启动 |
| `dngscan/gui/` | 原有完整工作台，以及旧导入路径的兼容别名 |
| `desktop/src/` | TypeScript 接口、独立交互状态和 React 组件 |
| `desktop/src-tauri/` | 原生文件选择、桌面窗口及后端进程生命周期 |

旧 GUI 的七个应用模块按原始字节迁移。旧模块路径通过模块别名指向新实现，
不是复制一份处理逻辑，也不是建立第二个缓存／调度器。
传统批处理 CLI 保持既有流程；新增命令入口方便直接验证交互应用用例。

## 运行

先准备项目现有 Python 环境（包含 GUI 依赖）。在 `desktop/` 执行：

```sh
npm ci
cp .env.example .env.local
```

需要时在 `.env.local` 中填写 `AGXRAW_PYTHON=/绝对路径/venv/bin/python`。
未填写时优先使用仓库根目录的 `.venv/bin/python`，再尝试 `python3`。
该文件仅供本地使用，不提交到 Git。

```sh
npm run desktop   # Tauri 窗口
npm run dev       # 独立浏览器前端，支持热更新
```

可用 `npm run desktop -- -- --open /绝对路径/照片.dng` 从启动器打开原片。

Tauri 使用系统原生选择器取得原文件路径。浏览器入口保留受限上传临时副本的方式，
并继续使用原有 Host、Origin 与会话令牌校验。开发代理仅转发本机接口；
没有为外部页面放开跨域访问。

父进程通过管道持有后端的生命周期。窗口进程退出或开发启动器关闭后，管道关闭，
后端请求正常退出；桌面层设有有界的强制退出后备。长计算由已有调度和隔离导出管理。
后端标准输出只用于就绪握手，处理诊断写到标准错误。

## 接口合同

请求示例：

```json
{"version":1,"id":"example-1","command":"preview.render","params":{"input":"/absolute/photo.dng","evAuto":true}}
```

成功返回同一版本、同一 id、`ok=true` 和原应用结果；失败返回结构化
`error.code`／`error.message`。显影请求的输入类型和未知参数在边界拒绝；标定包继续
由既有导入合同校验。范围、资格和成像合法性继续由既有引擎判定。
诊断结果允许扩展；前端对实际消费的响应字段做检查。

```sh
python -m dngscan.application --schema
python -m dngscan.application --request request.json
python -m dngscan.application --serve --frontend desktop/dist --open
```

HTTP 与 Tauri 使用相同命令合同。前端通过 `EngineClient` 隔离传输；
Tauri 将请求转交自己启动的本机后端，不实现第二套曝光或输出算法。
标定管理命令也已经进入共享门面，完整界面逐步迁移。

## 状态与显示

- 用户意图、准备结果、当前预览和导出结果分别管理。
- 文件选择立即递增 selection epoch，上传未结束时也不会再接受旧图。
- 参数修改递增预览 generation；过期成功或错误都不能覆盖新请求。
- 自动 EV 留在结果中，不写回手动 EV；完整 RenderPlan 仍由原服务传到实际渲染。
- 导出冻结当前声明；导出中修改参数会明确提示成片使用开始导出时的参数。
- 不展示虚构进度百分比。当前报告读取、预览、导出／回读等实际操作阶段。
- 新前端继续使用现有 SDR JPEG 预览。HDR 文件输出并不表示界面已有 HDR 预览。

## 验证与当前边界

```sh
python tools/generate_application_types.py --check
DNGSCAN_FAST=0 python -m unittest tests.test_application_contract tests.test_gui_http tests.test_gui_pipeline_cache
cd desktop
npm test
npm run tauri -- build --debug --no-bundle
```

可选实拍端到端检查：先启动 `npm run dev`，再执行
`AGXRAW_TEST_RAW=/绝对路径/照片.dng npx playwright test`。
检查真实预览、全尺寸导出、原片哈希和浏览器错误；输出只写临时目录。

构建后还可执行 `python tools/validate_desktop_entry.py /绝对路径/照片.dng
--output /tmp/native-entry.json`。该工具打开自己的临时 Tauri 窗口，等待真实前端的
准备／渲染命令完成，检查共享帧缓存与原片哈希，再关闭自己启动的进程并检查后台回收。
它不替代显示色彩或 HDR 观感验收。开发跟踪 `AGXRAW_TRACE_COMMANDS=1` 仅输出命令、
请求 id 和状态，不记录原片路径、像素或会话令牌。

本轮实际结果与源文件身份见
[共享应用服务与桌面入口验收记录](assets/delivery-quality/shared-application-desktop-20261016.json)：
2246 项 NumPy 回归中 2175 通过／71 跳过，61 项严格 native 定向检查通过，
前端 7 项测试通过。浏览器端实际完成 iPhone DNG 和 Sony ARW 的默认预览及全尺寸
SDR JPEG 导出；实际 Tauri 前端完成 iPhone RAW 准备／预览、共享帧缓存核验与退出回收。
这些范围分别记录，不累加成独立覆盖，也不扩展为未执行的 HDR 画质矩阵。

本入口首先覆盖原文件打开、默认自动规划、基础影调调节、解码／白平衡以及
SDR/HDR 导出。高级标定管理、直方图和其余研究控制仍由“原有完整工作台”提供，
不会因新入口而删除。当前不是完全取代旧工作台的发布版本。

Python 引擎尚未随 Tauri 打包，独立安装包、签名／公证、自动更新和完整原生 UI
验收是后续交付工作。没有因此改变成像管线，也没有把浏览器验收算作 WKWebView
全部交互或真实 HDR 文件画质验收。
