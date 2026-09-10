# META Trans

中文、泰语、英语智能翻译 H5。文本与视觉翻译由多模态模型完成，并提供可增量维护的术语知识库。

## 首版能力

- 中文、泰语、英语自动识别与六方向互译
- 文本输入与可编辑译文
- JPG、PNG、WebP 图片识别和翻译
- DOCX、PPTX、TXT、Markdown 文字提取和翻译
- PDF 支持中、泰、英三语六方向互译：优先精确映射原生文字层，扫描页和轮廓文字走视觉识别
- 可选的“本次翻译背景”，用于当前请求中的专名和语义消歧
- 数字缺失、异常短译文等基础质量提示
- 每次翻译均可独立下载 UTF-8 TXT 纯文本译文；PDF 和 DOCX 可额外下载连续文本的未排版版本
- PDF、DOCX、PPTX、TXT、Markdown 按上传格式导出译文；PDF 按原文字块坐标覆盖，DOCX/PPTX 在原 OOXML 结构内替换文字并保留图片、表格、页眉页脚、母版和对象位置
- Excel/UTF-8 CSV 术语知识库和中英泰手动录入，支持泰语主键覆盖、整段命中直出和术语优先翻译
- 本地翻译历史、复制译文和格式化文件再次下载
- 桌面端与手机 H5 响应式界面
- Responses API 优先，网关不支持时回退到 Chat Completions
- 无有效密钥时可使用演示模式完成前后端联调

“本次翻译背景”不会写入数据库，也不会用于后续请求。知识库是单独上传和持久化的数据；翻译时只选择当前原文命中的词条，避免把整库加入模型请求。

## 技术结构

```text
浏览器 H5（Vue 3 + TypeScript）
        │
        ▼
FastAPI
  ├─ 文本：完整命中知识库时直出，否则调用翻译模型
  ├─ 图片：多模态识别并翻译，同时提供目标语言方向的知识库术语
  ├─ 文档：本地提取文字后按每 5 页分块并发调用翻译模型
  ├─ PDF：内容流精确替换中/泰/英原生文字，扫描与轮廓文字按坐标识别回写
  ├─ 导出：独立 TXT，或按源格式生成 PDF、DOCX、PPTX、TXT、Markdown
  └─ 历史与知识库：SQLite
        │
        ▼
OpenAI 兼容模型网关
```

## 本地启动

环境要求：Node.js 20 或更高版本、Python 3.9 或更高版本。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r api/requirements.txt
```

启动 API：

```bash
AI_PROVIDER=demo .venv/bin/uvicorn app.main:app --app-dir api --reload --port 8000
```

另开终端启动 H5：

```bash
cd web
npm install
npm run dev
```

访问 `http://127.0.0.1:5173/`，API 文档位于 `http://127.0.0.1:8000/docs`。

## 接入真实模型

在仓库根目录创建 `.env`，配置以下变量：

```dotenv
AI_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-openai-compatible-gateway.example/v1
OPENAI_MODEL=gpt-5.6-terra
OPENAI_VISION_MODEL=gpt-5.6-terra
OPENAI_API_MODE=auto
OPENAI_REASONING_EFFORT=none
OPENAI_MAX_CONCURRENCY=5
OPENAI_MAX_RETRIES=3
OPENAI_RETRY_BASE_SECONDS=1.0

APP_MAX_UPLOAD_MB=200
APP_MAX_DOCUMENT_CHARACTERS=1000000
APP_PDF_OCR_MAX_PAGES=50
APP_PDF_OCR_DESIRED_WIDTH=1600
APP_PDF_OCR_CONCURRENCY=2
APP_CAD_OCR_MODE=auto
APP_CAD_INDEXED_IMAGES_PER_REQUEST=3
APP_CAD_REVIEW_IMAGES_PER_REQUEST=4
APP_CAD_INDEXED_MODEL_CONCURRENCY=5
APP_CAD_VISUAL_PAGE_CONCURRENCY=4
APP_CAD_PADDLE_WORKERS=auto
APP_PADDLE_DEVICE=
APP_PADDLE_CPU_THREADS=auto
APP_PADDLE_ENABLE_MKLDNN=false
APP_CAD_INDEXED_HEDGE_DELAY_SECONDS=25
APP_CAD_INDEXED_HEDGE_CONCURRENCY=2
APP_TEXT_MODEL_HEDGE_DELAY_SECONDS=60
APP_EXPORT_FONT_ZH=
APP_EXPORT_FONT_TH=
APP_EXPORT_FONT_EN=
```

- 密钥只由后端读取，`.env` 已被 Git 忽略。
- `OPENAI_API_MODE=auto` 会优先请求 `/responses`，端点明确不受支持时改用 `/chat/completions`。
- 文本和图片默认使用同一模型；如网关的视觉模型名称不同，可单独设置 `OPENAI_VISION_MODEL`。
- `none` 以返回速度和成本为先。上线前应使用真实业务样本对准确率、延迟和费用做基准测试。
- 文案页每 5 页一个请求，默认最多同时处理 5 个批次；PDF 异步任务会边逐页解析、边提交已形成的批次，并始终按页码合并结果。模型调用也默认全局并发 5。遇到 429 会自动按 `Retry-After` 或指数退避重试，次数和基础等待时间可通过 `OPENAI_MAX_RETRIES`、`OPENAI_RETRY_BASE_SECONDS` 调整。
- 单个文件默认最大 200 MB。PDF、DOCX、PPTX 默认最多 1,000,000 个字符，并按最多 5 页、6,000 字符和 180 个文字块的组合阈值拆分请求；单次发送的 TXT、Markdown 仍限制为 50,000 字符。扫描 PDF 默认最多 50 页、页面渲染宽度 1600 像素、并发处理 2 页。限制可通过 `APP_MAX_DOCUMENT_CHARACTERS`、`APP_PDF_OCR_MAX_PAGES` 调整。
- CAD 主索引图默认以 3 张全分辨率图合并为一次视觉请求；每张仅 3 行的高清复核图则以 4 张合并，缺少 ID 时只补发对应原像素行，整组失败时先递归对半拆分，必要时才降级为单图。6400px 高清复核底图会与本地 OCR 提前并行生成，但仍保持原分辨率与裁切逻辑。Tesseract 与 Paddle 按容器可见 CPU 分配工作线，Paddle `auto` 只在至少 8 核、12 GiB 的运行环境启用两个常驻预测器，低配服务器自动保持 1。Paddle 与 Tesseract 命中同一行时，只有 Paddle 框的高清复核成功后才会替换原候选。同一文档内只有“字形像素指纹+本地泰文提示”均完全匹配的行才会复用已确认原文；并发页可合并正在进行的相同识别，剩余 PNG 行会以原像素重新紧凑打包。
- CAD 视觉请求和结构化文本请求分别在 25 秒、60 秒后启动延迟对冲，用于吸收网关长尾。CAD 默认保留 2 个专用对冲槽位，且仍受全局模型并发限制；可用 `APP_CAD_INDEXED_HEDGE_CONCURRENCY=0` 恢复保守的排队对冲，对应 `*_HEDGE_DELAY_SECONDS=0` 则完全关闭。
- 图片 PDF 页每页都会产生一次视觉模型调用，费用和耗时随图片页数增长。文案页每 5 页一个请求并发处理，前端按已完成页数展示实时进度，全部返回后按页码合并结果；个别图片页失败时会返回已完成页面并给出提示。
- PDF 内容流引擎按所选源语言翻译中文、泰文或英文文字层，保留其他语言、数字、图片、表格线和 CAD 线条；系统字体沿原基线单行回写，过长时只缩小字号，不自动换行。PDF 未排版译文在单个源页内容过长时会自动续页。DOCX 和 PPTX 直接修改源文件中的段落/文本框，保留源包内的版式资源。

## 知识库文件

推荐使用页面提供的 Excel 模板。工作表表头与业务提供的修订表一致：

```text
NO | Original Text（Thai） | Translated Text（English） | Revised Text（English） | Translated Text（Chinese） | Revised Text（Chinese）
```

泰语取 `Original Text（Thai）`。英文和中文优先使用各自的 `Revised Text`，修订内容为空时回退到 `Translated Text`。同一行至少需要英文或中文其中一种译文；最多导入 10,000 条，Excel 文件最大 10 MB。

同时兼容原有 UTF-8 CSV，必填表头如下。每行必须是包含泰语的中泰或泰英语言对：

```csv
source_language,target_language,source_text,translated_text
th,zh,มหาวิทยาลัยสงขลานครินทร์,宋卡王子大学
th,en,มหาวิทยาลัยสงขลานครินทร์,Prince of Songkla University
```

语言代码只允许 `zh`、`th`、`en`，源语言和目标语言不能相同。泰语规范化内容是知识条目的唯一键；后导入或后提交的中文、英文会覆盖同一泰语条目的旧值。手动录入要求泰语必填，中文和英文至少填写一项；三种语言齐全时会生成中泰、泰英和中英的双向匹配。

OpenAI 接口格式参考：[Responses API](https://developers.openai.com/api/reference/resources/responses) 与 [图片输入指南](https://developers.openai.com/api/docs/guides/images-vision)。第三方兼容网关支持的模型、参数与计费以网关实际返回为准。

## 测试与构建

```bash
cd api && ../.venv/bin/python -m pytest -q
cd ../web && npm run build
```

生产构建由 FastAPI 同端口托管：

```bash
cd web && npm ci && npm run build && cd ..
.venv/bin/uvicorn api.app.main:app --host 0.0.0.0 --port 8000
```

Docker 部署：

```bash
docker compose up --build
```

访问 `http://127.0.0.1:8000/`。容器端口默认只绑定到服务器回环地址，公网部署应通过 Nginx 或其他反向代理接入；可通过 `APP_PORT` 修改宿主机端口。SQLite 数据保存在 Docker volume 中。

### 文档诊断与原件留存

每次文档翻译在开始解析前都会将原始文件写入数据卷，并建立一条私有诊断记录。无论任务成功、PDF 预检失败还是页面解析失败，都可通过任务 ID 关联原件、文件 SHA-256、文件大小、处理阶段、异常类型和完整堆栈。普通翻译历史只展示成功结果，不暴露诊断信息。

- 原件目录：`/app/data/uploads/<任务ID>/source.<扩展名>`
- 结构化诊断记录：SQLite 表 `document_attempts`
- 运行日志：`/app/data/logs/document-jobs.log`，单文件 20 MB，保留 10 个滚动备份

这些目录含用户原件和内部错误信息，不能映射到 Nginx 静态目录，也不应开放下载接口。当前默认不自动清理原件；生产环境应监控 Docker 数据卷容量，并按已确认的留存政策由运维人员清理。

## 首版边界

- PDF 扫描页和转轮廓文字依赖视觉识别，极小字、模糊扫描、复杂 CAD 图线可能需要额外复核；原生文字层优先使用可精确映射的内容流引擎。DOCX 段落内存在多种字符样式时，译文沿用该段首个文字运行的样式，段落、表格和页面结构保持不变。
- 知识库首版是单租户术语库和精确翻译记忆；同义词、语义向量召回和审核版本流转尚未实现。
- 当前是单租户本地历史，正式上线前需要补充登录、租户隔离、限流和监控。
- 正式密钥接通后，需要建立中泰英业务评测集并记录准确率、P95 延迟和单次成本。
