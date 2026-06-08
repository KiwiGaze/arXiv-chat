# File: tutorials/13-troubleshooting.md

# 第 13 章　常见问题排查

本章按"症状 → 原因 → 解决"组织，覆盖搭建与运行中最常见的坑。先掌握两条通用排查命令：

```bash
docker compose ps               # 看哪个服务没起来 / 不健康
docker compose logs -f <name>   # 看具体服务日志（如 rag-api / rag-opensearch / rag-langfuse-web）
```

---

## 13.1 服务启动类

### 症状：`docker compose up` 卡住或某服务一直 unhealthy

- **原因**：OpenSearch / Langfuse 栈较重，启动慢；或内存不足。
- **解决**：
  - 等 2–3 分钟，OpenSearch `start_period` 是 60s。
  - `docker compose logs opensearch` 看是否 OOM。
  - 提高 Docker Desktop 内存到 ≥8GB（推荐 12GB）。
  - 分步启动（见第 [12](12-run-build-deploy-rollback.md) 章 12.2），不要一次拉起全部。

### 症状：直接 `docker compose up` 时报 airflow 构建失败 / 找不到 `./airflow`

- **原因**：在还没创建 `airflow/` 目录（第 [05](05-week2-ingestion.md) 章）时就启动了全部服务。
- **解决**：Week 1 阶段只启动核心子集：
  ```bash
  docker compose up -d postgres opensearch opensearch-dashboards ollama redis
  docker compose up -d --build api
  ```

### 症状：端口冲突 `port is already allocated`

- **原因**：本机已有进程占用 8000 / 5432 / 9200 / 8080 / 6379 / 3001 等端口。
- **解决**：
  ```bash
  lsof -i :8000        # 找占用进程（macOS/Linux）
  # 停掉冲突进程，或改 compose.yml 里的宿主机端口映射（左边那个数字）
  ```

### 症状：OpenSearch 报 `max virtual memory areas vm.max_map_count [65530] is too low`

- **原因**：Linux 主机 `vm.max_map_count` 太小（OpenSearch 需要 ≥262144）。
- **解决**（Linux/WSL）：
  ```bash
  sudo sysctl -w vm.max_map_count=262144
  ```

### 症状（WSL/Ubuntu）：Airflow 容器权限错误 / 写日志失败

- **原因**：卷挂载的属主与容器内 `airflow`（UID 50000）不匹配。
- **解决**：在 `compose.yml` 的 `airflow` 服务取消注释 `user: "50000:0"`（Mac 上保持注释）。

---

## 13.2 健康检查类

### 症状：`make health` 显示 "API not responding"，但 API 其实在跑

- **原因**：`Makefile` 的 `make health` 用的是 `http://localhost:8000/health`，而真实端点在 `/api/v1/health`（见第 [12](12-run-build-deploy-rollback.md) 章说明）。
- **解决**：用正确路径验证：
  ```bash
  curl -s http://localhost:8000/api/v1/health | python -m json.tool
  ```

### 症状：`/api/v1/health` 返回 `status: degraded`

- **原因**：某个被检查的服务（database / opensearch / ollama）不健康。看 `services` 字段里哪个是 `unhealthy`。
- **解决**：按对应服务排查（DB 连接串、OpenSearch 是否起来、Ollama 是否拉了模型）。注意端点仍返回 200，这是设计如此（便于探针读取细节）。

---

## 13.3 数据摄取类（Week 2 / Airflow）

### 症状：抓取很慢，一篇接一篇

- **原因**：arXiv 要求请求间隔 ≥3s（`ARXIV__RATE_LIMIT_DELAY=3.0`），客户端会限速。这是**正常且必要**的（避免被封）。
- **解决**：耐心等；或调大 `ARXIV__MAX_RESULTS` 一次多抓些（但仍受限速）。

### 症状：PDF 解析很慢 / 内存飙升

- **原因**：Docling 解析是 CPU/内存密集，首次还要加载模型。
- **解决**：
  - 默认 `ARXIV__MAX_CONCURRENT_PARSING=1`（串行解析）已是保守值，别调太大。
  - `PDF_PARSER__MAX_PAGES=30` / `MAX_FILE_SIZE_MB=20` 限制大文件；超限会被优雅跳过（只存元数据）。
  - 保证 Docker 内存充足。

### 症状：PDF 解析全部失败，报 `TypeError: object ... can't be used in 'await' expression`

- **原因**：没有应用**修复 #6**（第 [05](05-week2-ingestion.md) 章）——`PDFParserService.parse_pdf` 里 `await self.docling_parser.parse_pdf(...)`，但 `DoclingParser.parse_pdf` 是同步方法。
- **解决**：去掉那个 `await`，改为 `result = self.docling_parser.parse_pdf(pdf_path)`。

### 症状：Airflow 日志报 `libGL.so.1: cannot open shared object file`；或 DAG 全绿但 `pdf_processed` 全为 false、OpenSearch count 为 0

- **原因**：Airflow 镜像基于 `python:3.12-slim`，默认不含 OpenGL 运行库；Docling 初始化/解析依赖 `libGL.so.1`。
- **解决**：
  1. 在 `airflow/Dockerfile` 加入 `libgl1`、`libglib2.0-0`（见第 [05](05-week2-ingestion.md) 章）。
  2. 重建并重启：
     ```bash
     docker compose build airflow && docker compose up -d airflow
     ```
  3. 在 Airflow UI 重新触发 `arxiv_paper_ingestion`（选有论文的日期；若自动日期返回 0 篇，可手动指定 `execution_date` 或等次日调度）。
  4. 验证解析与索引（见第 [07](07-week4-hybrid-search.md) 章 7.9 的 preflight 命令）。

### 症状：部分论文只有元数据，没有 `raw_text`/`sections`

- **原因**：PDF 下载失败、超大/超页数被跳过（`PDF_PARSER__MAX_PAGES=30` / `MAX_FILE_SIZE_MB=20`）、`libGL.so.1` 缺失导致 Docling 崩溃、或其他解析异常。`pdf_processed=False`。
- **解决**：这是预期的优雅降级。看日志里的 `Download failures` / `Parse failures` 统计；若报 `libGL.so.1`，按上一节修复 Dockerfile；可对失败的 arxiv_id 重试。

### 症状：Airflow DAG 里 `setup_environment` 或 `index_papers_hybrid` 失败

- **原因**：依赖 Week 3（OpenSearch 模块）/ Week 4（indexing 模块）。如果这些模块还没建，任务会 ImportError；或 `JINA_API_KEY` 没配导致嵌入失败。
- **解决**：确认已完成 Week 3–4 的代码；确认 `.env` 里 `JINA_API_KEY` 有效；OpenSearch 健康。

---

## 13.4 检索类（Week 3 / 4）

### 症状：检索总是返回空 `hits`

- **原因**：索引里没数据（还没灌）；或索引名不一致；或 PDF 从未解析成功（`pdf_processed=false`，索引任务会跳过空 `raw_text`）。
- **解决**：
  - 先灌数据（触发 Airflow DAG）。
  - 若 count 为 0，先查 `pdf_processed` 是否为 true，而不只是"有没有跑过 DAG"：
    ```bash
    docker exec rag-postgres psql -U rag_user -d rag_db -c \
      "SELECT COUNT(*) FILTER (WHERE pdf_processed) AS parsed, COUNT(*) AS total FROM papers;"
    ```
  - 确认索引存在且有文档：
    ```bash
    curl -s http://localhost:9200/arxiv-papers-chunks/_count | python -m json.tool
    ```
  - 索引名 = `OPENSEARCH__INDEX_NAME` + `-` + `OPENSEARCH__CHUNK_INDEX_SUFFIX` = `arxiv-papers-chunks`。

### 症状：混合检索退化成 BM25（`search_mode: bm25`，尽管请求了 `use_hybrid: true`）

- **原因**：嵌入生成失败（`JINA_API_KEY` 无效/额度用尽/网络问题），代码自动降级到 BM25。
- **解决**：看日志 "Failed to generate embeddings, falling back to BM25"；检查 `JINA_API_KEY`。这是设计的优雅降级，不影响可用性，只是少了语义召回。

### 症状：写入 chunk 报 `mapper_parsing_exception` 或维度不符

- **原因**：嵌入维度与索引映射的 `dimension: 1024` 不一致（如换了嵌入模型）。
- **解决**：保证嵌入维度 = `OPENSEARCH__VECTOR_DIMENSION`（默认 1024，对齐 Jina v3）。换模型需同步改映射并 `setup_indices(force=True)` 重建索引。

### 症状：`dynamic: "strict"` 导致写入被拒（unknown field）

- **原因**：索引映射是严格模式，写入了未声明字段。
- **解决**：只写映射里声明的字段（见第 [06](06-week3-opensearch-bm25.md) 章），或在映射中补充该字段后重建索引。

---

## 13.5 LLM / RAG 类（Week 5）

### 症状：`/ask` 报 Ollama 连接错误 / 超时

- **原因**：Ollama 没起、没拉模型、或模型推理超时。
- **解决**：
  ```bash
  docker compose up -d ollama
  docker exec -it rag-ollama ollama list           # 看有没有 gemma4:e2b
  docker exec -it rag-ollama ollama pull gemma4:e2b
  ```
  超时可调大 `OLLAMA_TIMEOUT`（默认 300s）。

### 症状：答案质量差 / 啰嗦 / 不按格式

- **原因**：`gemma4:e2b` 模型很小，能力有限，结构化输出不稳定。
- **解决**：换更大模型：`OLLAMA_MODEL=llama3.2:3b` 或 `qwen2.5:7b`（先 `ollama pull`），或在 Gradio 下拉里选。代码对结构化输出失败有多级兜底（见第 [08](08-week5-rag-llm.md) 章 `ResponseParser`），不会崩。

### 症状：Gradio 界面报 "Connection error" / 无响应

- **原因**：Gradio 调 `http://localhost:8000/api/v1/stream`，但 API 没起，或 `/stream` 端点尚未实现（Week 6 才有）。
- **解决**：先完成 Week 6（第 [09](09-week6-monitoring-caching.md) 章）并启动 api；确认 `curl -N -X POST .../stream` 能流式返回。

---

## 13.6 监控 / 缓存类（Week 6）

### 症状：`langfuse-web` 容器起不来

- **原因**：`LANGFUSE_ENCRYPTION_KEY` 是占位的全 0，或不是 64 位十六进制。
- **解决**：
  ```bash
  openssl rand -hex 32   # 把输出填入 .env 的 LANGFUSE_ENCRYPTION_KEY
  docker compose up -d langfuse-web
  ```

### 症状：`/ask` 不报错但 Langfuse 看不到 trace

- **原因**：`LANGFUSE_ENABLED=false` 或没填 `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY`，追踪被安全降级为 no-op。
- **解决**：在 Langfuse Web 创建项目密钥并填入 `.env`，设 `LANGFUSE_ENABLED=true`、`LANGFUSE_HOST=http://localhost:3001`（见第 [09](09-week6-monitoring-caching.md) 章 9.1）。

### 症状：缓存似乎没生效（每次都慢）

- **原因**：请求参数（query/model/top_k/use_hybrid/categories）有任一不同 → 精确缓存不命中；或 Redis 没起。
- **解决**：用**完全相同**的请求体连发两次对比；确认 Redis 健康：
  ```bash
  docker exec -it rag-redis redis-cli ping        # PONG
  docker exec -it rag-redis redis-cli KEYS 'exact_cache:*'
  ```

### 症状：API 启动时因 Redis 连接失败而崩

- **原因**：`main.py` 的 lifespan 里 `make_cache_client(settings)` 没有 try/except，Redis 不可达会让启动失败。
- **解决**：保证 `redis` 服务健康（它是 api 的 `depends_on: service_healthy`）。若在无 Redis 环境运行，按第 [10](10-week7-agentic-telegram.md) 章 10.18 的提示用 try/except 包裹该调用并把 `app.state.cache_client` 设为 `None`。

---

## 13.7 Agentic / Telegram 类（Week 7）

### 症状：`/ask-agentic` 报 `AttributeError: 'OllamaClient' object has no attribute 'get_langchain_model'`

- **原因**：没有应用**修复 #1**（第 [08](08-week5-rag-llm.md) 章）——`OllamaClient` 缺 `get_langchain_model`。
- **解决**：按第 [08](08-week5-rag-llm.md) 章在 `OllamaClient` 补上 `get_langchain_model`（返回 `ChatOllama`）。

### 症状：`/ask-agentic` 报 `AttributeError: ... create_span` / `end_span` / `trace_rag_request`

- **原因**：没有应用**修复 #2**（第 [09](09-week6-monitoring-caching.md) 章）——`LangfuseTracer` 缺这三个方法。
- **解决**：按第 [09](09-week6-monitoring-caching.md) 章在 `LangfuseTracer` 补上这三个方法（禁用时 no-op）。

### 症状：`/ask-agentic` 报 `TypeError: make_agentic_rag_service() got an unexpected keyword argument 'model'`

- **原因**：没有应用**修复 #5**（第 [10](10-week7-agentic-telegram.md) 章）——`get_agentic_rag_service` 给工厂传了不接受的 `model=`。
- **解决**：删去 `dependencies.py` 中 `get_agentic_rag_service` 里的 `model=settings.ollama_model` 这一行。

### 症状：Agentic 一直 "rewrite → retrieve" 但最终兜底拒答

- **原因**：检索不到相关内容（索引空/不匹配），评分判为不相关，重写后仍无果，达到 `max_retrieval_attempts`（默认 2）触发兜底。
- **解决**：确认索引里有相关论文；问更贴近已索引内容的问题；必要时调大 `GraphConfig.max_retrieval_attempts`。

### 症状：明显领域内的问题被护栏拦截（误判为 out_of_scope）

- **原因**：小模型护栏打分不稳，分数低于阈值（默认 60）。
- **解决**：换更强模型；或调低 `GraphConfig.guardrail_threshold`。

### 症状：Telegram 机器人没反应 / 启动日志没有 "Telegram bot started"

- **原因**：`TELEGRAM__ENABLED=false` 或 `TELEGRAM__BOT_TOKEN` 没配/无效。
- **解决**：向 @BotFather 申请 token，填入 `.env`，设 `TELEGRAM__ENABLED=true`，重启 api。看日志确认。

### 症状：智能体单元测试报 `fixture 'test_context' not found`

- **原因**：上游缺失智能体测试夹具（见第 [11](11-testing.md) 章问题 2）。
- **解决**：按第 [11](11-testing.md) 章新建 `tests/unit/services/agents/conftest.py` 并把测试里 `create_llm` 改为 `get_langchain_model`（修复 #4）。

---

## 13.8 依赖 / 环境类

### 症状：`uv sync` 失败 / 某依赖编译失败

- **原因**：Python 版本不对（需 3.12）、系统缺编译工具。
- **解决**：`uv run python --version` 确认 3.12.x；macOS 装 Xcode CLT，Linux 装 `build-essential`。

### 症状：容器内 `import src.*` 失败（Airflow）

- **原因**：`PYTHONPATH` / 挂载不对。
- **解决**：确认 `compose.yml` 的 airflow 服务有 `PYTHONPATH=/opt/airflow/src` 且挂了 `./src:/opt/airflow/src`；`common.py` 里有 `sys.path.insert(0, "/opt/airflow")`。

### 症状：IDE/静态分析器（如 Pyright/Pylance）报错 `Cannot find module 'src.db.factory'`（或其他以 `src.` 开头的导入错误），但程序实际运行一切正常

- **原因**：在包含 `src/` 目录的项目中，IDE 静态分析器默认会推导 `src/` 作为导入根目录（Import Root），因此会将绝对导入 `src.db.factory` 定位到 `src/src/db/factory.py` 导致报错。
- **解决**：在 `pyproject.toml` 的末尾添加 `[tool.pyright]` 与 `[tool.pyrefly]` 配置，把项目根目录（`.`）也指定为模块搜索路径：
  ```toml
  [tool.pyright]
  venvPath = "."
  venv = ".venv"
  extraPaths = ["."]

  [tool.pyrefly]
  search-path = ["."]
  ```

### 症状：配置没生效 / 改了 .env 没反应

- **原因**：`Settings` 是 `frozen=True` 且服务在 lifespan 启动时构建一次；改 `.env` 后需重启容器。
- **解决**：`docker compose up -d --build api`（或对应服务）重启。注意嵌套变量用双下划线（`ARXIV__MAX_RESULTS`）。

---

## 13.9 终极手段：完整重置

踩坑太深、想从干净状态重来：

```bash
docker compose down -v          # ⚠️ 删除所有数据卷
docker compose up --build -d
docker exec -it rag-ollama ollama pull gemma4:e2b
# 然后重新触发 Airflow DAG 灌数据
```

（详见第 [12](12-run-build-deploy-rollback.md) 章 12.5。）

---

## 13.10 求助前的自查清单

1. `docker compose ps` —— 服务都 healthy 吗？
2. `docker compose logs <name>` —— 有没有明确报错？
3. `.env` —— 必需密钥（`LANGFUSE_ENCRYPTION_KEY`、`JINA_API_KEY`、Telegram token）填了吗？
4. 六处上游修复（#1–#6）都应用了吗？（见第 [16](16-upstream-differences-and-fixes.md) 章）
5. 索引里有数据吗？（`curl .../arxiv-papers-chunks/_count`）
6. 模型拉了吗？（`ollama list`）

下一章 [`14-quality-performance-security.md`](14-quality-performance-security.md) 做性能、安全、可维护性的系统分析。
