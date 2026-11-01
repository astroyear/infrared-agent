# 无人机红外检测与知识管理助手

Vue 3 + FastAPI 应用。图片由本地 PWL-RTDETR 在 CPU 上检测。DeepSeek 通过 LangGraph 路由检测、资料问答和知识维护请求。公共 Markdown 按版本发布，私有图片、提议及对话按账号隔离。

## 本机快速启动（Windows）

在仓库根目录打开 PowerShell。使用已有的 `infrared-agent` Conda 环境（Python 3.11、CPU 模型依赖）。其他机器先按 [模型环境说明](RTDETR-main-code/环境安装与运行.md) 安装模型依赖。

1. 检查项目根目录的 `.env`。本机初次准备时已生成 JWT 和数据库随机密钥。在 `DEEPSEEK_API_KEY=` 后填写自己的 Key，再启动后端。Key 仅保存在后端。如果是新克隆，复制 `.env.example` 为 `.env`，分别生成 JWT_SECRET（32 字节以上）和 POSTGRES_PASSWORD（随机字母数字）。可以用 `python -c "import secrets; print(secrets.token_hex(32))"` 生成。
2. 启动后端。若本机已有 `.local/py`，直接使用下列命令。新克隆在执行前先运行 `python -m pip install --target .local/py -r backend/requirements.txt`。依赖安装在仓库的忽略目录，不修改模型环境里的已安装包。

```powershell
conda activate infrared-agent
$env:PYTHONPATH = (Resolve-Path .local/py).Path
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

3. 另开 PowerShell 终端，在仓库根目录启动前端。首次安装运行 `npm.cmd --prefix frontend ci`。

```powershell
npm.cmd --prefix frontend run dev
```

打开 Vite 打印的本地地址（默认 http://localhost:5173），注册测试账号。API 文档在 http://127.0.0.1:8000/docs。前端以同源 /api 路径访问后端。Windows 本地模式使用 storage/app.sqlite3，持久保存用户、检测历史、消息和文档。此模式下图的执行检查点和明确保存的偏好随进程重启清空。PostgreSQL 模式的两者均持久化。应用以单 worker 运行。

## 第一次操作

1. 图片检测页选择 RTDETR-main-code/test_images/hituav/1_60_50_0_07426.jpg，保留阈值 0.25。可对照已有结果（4 Car、12 Bicycle），点击“追问本次检测”询问类别数量。无 DeepSeek Key 时仍可注册、登录并进行按钮检测。
2. 在知识管理页附上仓库中的“环境安装与运行.md”，输入“把这份附件完整导入公共知识库”。核对差异后确认，再到问答页问“示例图用哪个 Python 版本验证过？”。引用按钮可打开对应版本正文。
3. 在同一会话要求“在刚才的公共文档末尾补充一条……”。只有提议人能确认或取消；确认时检查版本。换一个账号可以检索已发布文档。
4. 在问答页输入“记住偏好：用简短中文回答”可明确保存回答偏好。知识管理仅支持 UTF-8 Markdown。检测只接收 JPEG、PNG。

## DeepSeek 和检索配置

后端从 .env 读取 DEEPSEEK_API_KEY、DEEPSEEK_BASE_URL 和 DEEPSEEK_MODEL。默认 deepseek-flash，可按账号可用性调整。[DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)

初次只填 DeepSeek Key 时，问答使用中文分词和 BM25。要启用语义检索，在**首次发布文档前**配置支持 OpenAI embeddings 协议的 EMBEDDING_BASE_URL、EMBEDDING_API_KEY、EMBEDDING_MODEL 和模型实际输出维度 EMBEDDING_DIM。配置后发布时写入分块向量，问答用 BM25 与向量排名进行 RRF 融合。版本记录保留所用模型名称；之前仅有关键词索引的版本仍可按关键词检索。改变向量维度需要迁移 chunks 的向量列并重建索引数据，首版不自动迁移旧向量。

## WSL / Docker Compose

Docker 引擎启动后，在仓库根目录确认 .env 已填写、权重 RTDETR-main-code/pwl_rtdetr/best.pt 存在，然后运行：

```sh
docker compose up -d --build
docker compose ps
docker compose logs -f backend
```

打开 http://localhost:8080。前端 Nginx 反代后端，只有前端端口绑定宿主机回环地址。首次构建需要下载 Linux CPU 模型依赖。PostgreSQL 使用 pgvector，首次启动建表并初始化 LangGraph PostgresSaver 和 PostgresStore。数据库在 pg_data 卷，文件在 app_storage 卷。正常停止用 `docker compose down`。本地 SQLite 和 Compose 的数据库相互独立。

## 自检

```powershell
# 和启动本地后端一样，先激活环境并设置 PYTHONPATH
python -m backend.tests.smoke
npm.cmd --prefix frontend run build
docker compose config --quiet
```

冒烟测试使用临时数据库、真实模型、本地六张样图及可控的模型回答。它不消耗 DeepSeek 额度。真实问答质量和外部 Embedding 需用自己的接口验证。检测模型一次加载并限制同时一次推理；重启时未完成任务标记为 interrupted。密码使用带盐 PBKDF2 哈希。接口返回的 request_id 可用于定位服务端错误。

## 目录

- backend/app/api：认证、文件、检测、会话及知识接口。
- backend/app/services：推理、私有文件、检测任务、知识版本和检索。
- backend/app/agents.py、graph.py：DeepSeek 工具调用、编辑提议、状态及条件路由。
- frontend/src：登录及三个业务页。
- RTDETR-main-code：原有模型源码、权重、样图和 CLI。
- docs：设计、实施规划和验收进度。
