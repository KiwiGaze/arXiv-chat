# File: tutorials/02-environment-and-dependencies.md

# 第 2 章　系统与环境需求、依赖与安装

本章把开发环境和所有依赖准备好。读完后你会有：一个初始化好的项目目录、`uv` 管理的虚拟环境、所有依赖锁定、以及一份填好必需密钥的 `.env`。

---

## 2.1 系统需求

| 项目 | 要求 | 说明 |
|------|------|------|
| 操作系统 | macOS / Linux / Windows(WSL2) | 本教程命令以 macOS/Linux 为准；Windows 请在 WSL2 中操作 |
| 内存 | **8GB 以上**（推荐 16GB） | OpenSearch + Ollama + Langfuse 全家桶较吃内存 |
| 磁盘 | **20GB 以上空闲** | Docker 镜像、大模型权重、PDF 与索引数据 |
| CPU | 4 核以上 | Docling 解析 PDF、Ollama 推理是 CPU 密集 |
| Docker | Docker Desktop（含 Compose v2） | 全部基础设施都跑在容器里 |
| Python | **3.12.x**（`>=3.12,<3.13`） | 项目锁定 3.12，详见 `pyproject.toml` |
| 包管理器 | **uv** | 比 pip 快很多，且用 `uv.lock` 精确锁版本 |

> **为什么锁定 Python 3.12 而不是更高/更低？**
> - **为什么这么选**：3.12 是依赖（docling、langgraph、opensearch-py、sentence-transformers 等）都成熟支持的版本，且性能优于 3.11。
> - **替代方案**：3.11（更保守）、3.13（更新）。
> - **优缺点**：3.13 部分 C 扩展依赖当时尚未提供 wheel，容易编译失败；3.11 缺少 3.12 的一些性能改进。
> - **影响**：固定单一小版本避免"在我机器上能跑"的环境漂移，可维护性更好。
> - **风险与缓解**：未来依赖要求更高版本时需统一升级；用 `uv.lock` + Docker 锁死版本降低风险。

---

## 2.2 安装 Docker

安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)，启动后在终端验证：

```bash
docker --version
docker compose version
```

在 Docker Desktop 的设置里，把内存上限调到 **至少 8GB**（推荐 12GB），否则 OpenSearch + Langfuse 可能因内存不足而启动失败。

---

## 2.3 安装 uv 包管理器

`uv` 是本项目的 Python 包管理器（来自 Astral，Ruff 的作者）。

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 验证
uv --version
```

> **为什么用 uv 而不是 pip/Poetry/conda？**
> - **为什么这么选**：`uv` 安装速度极快（Rust 实现），原生支持 `pyproject.toml` + `uv.lock` 锁定，命令简单（`uv sync` / `uv run`）。
> - **替代方案**：pip + venv、Poetry、PDM、conda。
> - **优缺点**：pip 需要手动管理 venv 和锁文件；Poetry 较慢且解析器偶有问题；conda 偏重、对纯 Python 项目过度。`uv` 的缺点是较新、生态仍在成长。
> - **影响（性能/可维护性）**：`uv.lock` 保证团队和 CI 安装到完全一致的版本，可复现性强；CI 安装时间大幅缩短。
> - **风险与缓解**：工具较新——但它生成的是标准 `pyproject.toml`，必要时可迁回 pip。

---

## 2.4 初始化项目目录

```bash
mkdir arxiv-paper-curator
cd arxiv-paper-curator
git init
```

接下来创建项目清单文件 `pyproject.toml`。**这是依赖与工具配置的唯一真相源**。

### 文件：`pyproject.toml`（逐字复制）

```toml
[project]
name = "arXiv-chat"
version = "0.1.0"
description = "arXiv Chat"
requires-python = ">=3.12,<3.13"
dependencies = [
    "fastapi[standard]>=0.115.12",
    "uvicorn>=0.34.0",
    "pydantic>=2.11.3",
    "pydantic-settings>=2.8.1",
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.10",
    "alembic>=1.13.3",
    "opensearch-py>=3.0.0",
    "requests>=2.32.3",
    "httpx>=0.28.1",
    "docling>=2.43.0",
    "python-dateutil>=2.9.0.post0",
    "sentence-transformers>=5.1.0",
    "gradio>=4.0.0",
    "langfuse>=3.0.0",
    "redis>=6.4.0",
    "python-telegram-bot>=21.0,<22.0",
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-community>=0.3.0",
    "langchain-ollama>=0.3.0",
]
readme = "README.md"

[dependency-groups]
dev = [
    "anyio[trio]>=4.9.0",
    "asgi-lifespan>=2.1.0",
    "jupyter>=1.1.1",
    "mypy>=1.15.0",
    "notebook>=7.4.4",
    "polyfactory>=2.21.0",
    "pre-commit>=4.2.0",
    "pytest>=8.3.5",
    "pytest-aiohttp>=1.1.0",
    "pytest-cov>=6.1.1",
    "pytest-dotenv>=0.5.2",
    "pytest-env>=1.1.5",
    "pytest-mock>=3.14.0",
    "ruff>=0.11.5",
    "testcontainers>=4.10.0",
    "types-sqlalchemy>=1.4.53.38",
]
viz = [
    "grandalf>=0.8",
]

[tool.ruff]
line-length = 130
exclude = ["notebooks/**", ".venv/**"]
src = ["src", "tests"]
lint.select = [
  "F401",
  "I",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
env_files = ".env.test"

[tool.mypy]
explicit_package_bases = true
ignore_errors = true

[tool.pyright]
venvPath = "."
venv = ".venv"
extraPaths = ["."]

[tool.pyrefly]
search-path = ["."]
```

> **`[tool.ruff] lint.select = ["F401", "I"]` 是怎么回事？**
> 这里只启用两条 lint 规则：`"F401"` 标记未使用的导入，`"I"` 负责导入排序（isort）。两者都能被 `ruff check --fix` 自动修复，保持导入整洁。

> **`[tool.mypy] ignore_errors = true` 是怎么回事？**
> 这是教学项目的现实折中：它让 `mypy` 只做导入/语法层面的检查而不因海量类型告警阻塞流程。生产项目应逐步收紧（见第 [15](15-cicd-and-maintenance.md) 章）。本教程如实保留上游配置。

> **`[tool.pyright]` 与 `[tool.pyrefly]` 是怎么回事？**
> 这是为了解决 IDE/静态分析器（如 Pyright/Pylance）的模块导入路径报错。在存在 `src` 目录时，IDE 默认会将 `src/` 当作导入根目录，导致其在解析 `src.db.factory` 时定位到不存在的 `src/src/db/factory.py`。`venvPath` / `venv` 指向项目的 `.venv`，`extraPaths = ["."]` 把项目根目录也加入模块搜索路径；`[tool.pyrefly] search-path` 同理，供 Pyrefly 解析 `src.*` 导入。

### 依赖清单逐项说明

| 依赖 | 作用 | 引入于 |
|------|------|--------|
| `fastapi[standard]`, `uvicorn` | Web 框架与 ASGI 服务器 | Week 1 |
| `pydantic`, `pydantic-settings` | 数据校验与 12-factor 配置 | Week 1 |
| `sqlalchemy`, `psycopg2-binary`, `alembic` | ORM、PostgreSQL 驱动、迁移工具 | Week 1–2 |
| `opensearch-py` | OpenSearch 客户端 | Week 3 |
| `requests`, `httpx` | HTTP 客户端（httpx 支持异步） | Week 2+ |
| `docling` | 科学 PDF 解析 | Week 2 |
| `python-dateutil` | 日期解析 | Week 2 |
| `sentence-transformers` | 嵌入相关（被 docling/生态间接使用） | Week 4 |
| `gradio` | Web 聊天界面 | Week 5 |
| `langfuse` | LLM 可观测性/追踪 | Week 6 |
| `redis` | 缓存客户端 | Week 6 |
| `python-telegram-bot` | Telegram 机器人 | Week 7 |
| `langgraph`, `langchain*`, `langchain-ollama` | 智能体编排与 LLM 适配 | Week 7（`langchain-ollama` 也用于 Week 5 修复，见下） |

> **依赖版本用 `>=` 下限而非 `==` 锁死，安全吗？**
> - `pyproject.toml` 声明的是"最低可接受版本"，**真正锁死的是 `uv.lock`**（含全部传递依赖的精确版本与哈希）。
> - **优缺点**：`>=` 让你能在不改清单的情况下小步升级；`uv.lock` 保证 CI/团队/Docker 装到完全一致的版本。
> - **安全影响**：`uv.lock` 记录哈希，可防止依赖被篡改（供应链安全）；定期 `uv lock --upgrade` + 测试是推荐节奏。

---

## 2.5 创建虚拟环境并安装依赖

```bash
# 让 uv 读取 pyproject.toml，创建 .venv 并安装全部依赖（含 dev 组）
uv sync
```

`uv sync` 会：
1. 创建 `.venv/`（基于 Python 3.12）。
2. 解析依赖并生成/更新 `uv.lock`（**请把 `uv.lock` 提交进 git**）。
3. 安装 `dependencies` + `dev` 组。

之后所有 Python 命令都用 `uv run <cmd>` 执行（自动用 `.venv`），例如：

```bash
uv run pytest
uv run ruff format
uv run python gradio_launcher.py
```

> **`viz` 组（`grandalf`）是可选的**：用于在终端打印 LangGraph 的 ASCII 图（Week 7）。需要时：`uv sync --group viz`。

---

## 2.6 关键配置文件：`.env`

项目用环境变量做配置（12-factor 原则）。先创建一份**示例**文件 `.env.example`，再复制为真正的 `.env` 并填入密钥。

### 文件：`.env.example`（逐字复制）

```bash
# arXiv Paper Curator - Environment Configuration (EXAMPLE)
# Copy this file to .env and adjust values as needed

# Application Settings
DEBUG=true
ENVIRONMENT=development

# PostgreSQL Database
POSTGRES_DATABASE_URL=postgresql+psycopg2://rag_user:rag_password@postgres:5432/rag_db

# External Services
OPENSEARCH_HOST=http://opensearch:9200
OPENSEARCH__HOST=http://opensearch:9200
OLLAMA_HOST=http://ollama:11434

# arXiv API Configuration  
ARXIV__MAX_RESULTS=15
ARXIV__BASE_URL=https://export.arxiv.org/api/query
ARXIV__PDF_CACHE_DIR=./data/arxiv_pdfs
ARXIV__RATE_LIMIT_DELAY=3.0
ARXIV__TIMEOUT_SECONDS=30
ARXIV__SEARCH_CATEGORY=cs.AI
ARXIV__DOWNLOAD_MAX_RETRIES=3
ARXIV__DOWNLOAD_RETRY_DELAY_BASE=5.0
ARXIV__MAX_CONCURRENT_DOWNLOADS=5
ARXIV__MAX_CONCURRENT_PARSING=1

# PDF Parser Configuration
PDF_PARSER__MAX_PAGES=30
PDF_PARSER__MAX_FILE_SIZE_MB=20
PDF_PARSER__DO_OCR=false
PDF_PARSER__DO_TABLE_STRUCTURE=true

# OpenSearch Configuration (Single hybrid index for all search types)
OPENSEARCH__INDEX_NAME=arxiv-papers
OPENSEARCH__CHUNK_INDEX_SUFFIX=chunks
OPENSEARCH__MAX_TEXT_SIZE=1000000

# Vector Search Settings
OPENSEARCH__VECTOR_DIMENSION=1024
OPENSEARCH__VECTOR_SPACE_TYPE=cosinesimil

# Hybrid Search Settings  
OPENSEARCH__RRF_PIPELINE_NAME=hybrid-rrf-pipeline
OPENSEARCH__HYBRID_SEARCH_SIZE_MULTIPLIER=2

# Text Chunking Configuration
CHUNKING__CHUNK_SIZE=600
CHUNKING__OVERLAP_SIZE=100
CHUNKING__MIN_CHUNK_SIZE=100
CHUNKING__SECTION_BASED=true

# Jina AI Embeddings (Required for hybrid search)
JINA_API_KEY=your_jina_api_key_here

# Ollama Configuration
OLLAMA_MODEL=gemma4:e2b
OLLAMA_TIMEOUT=300

# Langfuse v3 Tracing Configuration - Official SDK Standard (single underscore)
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE_FLUSH_AT=15
LANGFUSE_FLUSH_INTERVAL=1.0
LANGFUSE_DEBUG=true

# Langfuse Server Configuration (for docker-compose)
# WARNING: Change these values in production! These are development defaults.
LANGFUSE_NEXTAUTH_SECRET=changeme-v3-nextauth-secret-min-32-chars-recommended
LANGFUSE_SALT=changeme-v3-salt-min-32-chars-recommended-for-security
# REQUIRED: Generate a real key before starting: openssl rand -hex 32
LANGFUSE_ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000
LANGFUSE_REDIS_PASSWORD=langfuse_redis_password
LANGFUSE_MINIO_ACCESS_KEY=langfuse_minio
LANGFUSE_MINIO_SECRET_KEY=langfuse_minio_secret

# Redis Cache Configuration
REDIS__HOST=redis
REDIS__PORT=6379
# Leave empty since compose.yml Redis has no --requirepass configured
REDIS__PASSWORD=
REDIS__DB=0
REDIS__TTL_HOURS=6

# Telegram Bot Configuration (Week 7)
# Get your bot token from @BotFather on Telegram
TELEGRAM__ENABLED=true
TELEGRAM__BOT_TOKEN=your_telegram_bot_token_here

# Airflow Settings
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__LOAD_EXAMPLES=false
AIRFLOW__WEBSERVER__EXPOSE_CONFIG=true
AIRFLOW__HOME=/opt/airflow
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://rag_user:rag_password@postgres:5432/rag_db
```

### 复制为真实配置

```bash
cp .env.example .env
```

### 配置项命名规则（很重要）

注意上面有两种风格：`OPENSEARCH_HOST` 与 `OPENSEARCH__HOST`（双下划线）。这不是笔误：

- **双下划线 `__`** 是 `pydantic-settings` 的**嵌套分隔符**。`OPENSEARCH__HOST` 映射到 `Settings.opensearch.host`，`ARXIV__MAX_RESULTS` 映射到 `Settings.arxiv.max_results`。第 [04](04-week1-infrastructure.md) 章讲 `config.py` 时会看到对应的嵌套配置类。
- **单下划线**的若干变量（如 `OPENSEARCH_HOST`、`OLLAMA_HOST`、`OLLAMA_MODEL`、`JINA_API_KEY`、`POSTGRES_DATABASE_URL`）映射到顶层 `Settings` 字段或被 Docker/Airflow 直接读取。
- `compose.yml` 会对容器再注入一层覆盖（把 `localhost` 换成容器服务名，如 `opensearch`、`postgres`、`ollama`、`redis`），见第 [04](04-week1-infrastructure.md)、[09](09-week6-monitoring-caching.md) 章。

---

## 2.7 必须准备的密钥与凭据

### (1) Langfuse 加密密钥（Week 6 必需，否则 langfuse-web 起不来）

`LANGFUSE_ENCRYPTION_KEY` 必须是 **64 位十六进制**（32 字节）。示例文件里给的是全 0 占位符，**启动前务必换成真随机值**：

```bash
openssl rand -hex 32
```

把输出粘贴到 `.env` 的 `LANGFUSE_ENCRYPTION_KEY=`。同理建议把 `LANGFUSE_NEXTAUTH_SECRET` 与 `LANGFUSE_SALT` 也换成足够长的随机串（至少 32 字符）。

> **安全提醒**：`.env` 里 PostgreSQL 密码（`rag_password`）、MinIO/Redis 密码、Langfuse 管理员密码（`admin123`，见 `compose.yml`）都是**开发默认值**。**生产环境必须全部替换**。第 [14](14-quality-performance-security.md) 章有完整的安全清单。

### (2) Jina 嵌入 API Key（Week 4+ 必需）

去 [https://jina.ai/embeddings](https://jina.ai/embeddings) 免费注册，拿到 API Key，填入：

```bash
JINA_API_KEY=jina_xxxxxxxxxxxxxxxxxxxxxxxx
```

> 没有这个 Key，混合检索会优雅降级为纯 BM25（代码里有 try/except 兜底，见第 [07](07-week4-hybrid-search.md) 章），但你就用不上向量检索了。

### (3) Telegram Bot Token（Week 7 可选）

在 Telegram 里找 **@BotFather**，发送 `/newbot`，按提示创建机器人，拿到形如 `123456:ABC-DEF...` 的 Token，填入：

```bash
TELEGRAM__BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM__ENABLED=true
```

> 如果暂时不做 Telegram，把 `TELEGRAM__ENABLED=false` 即可，应用启动时会跳过机器人初始化。

### (4) Langfuse 项目密钥（Week 6 可选）

`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` 需要**先启动 Langfuse、在其 Web 界面创建项目**后才能得到。第 [09](09-week6-monitoring-caching.md) 章会一步步带你做。在此之前留占位符即可——追踪层在缺少密钥时会安全降级，不影响主流程。

---

## 2.8 测试用配置：`.env.test`

测试用一份独立配置（`pyproject.toml` 里 `env_files = ".env.test"` 指定）。

### 文件：`.env.test`（逐字复制）

```bash
# Test environment configuration
DEBUG=true
ENVIRONMENT=development

# Test Database URL (in-memory or localhost)
POSTGRES_DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/test_db

# Test External Services
OPENSEARCH__HOST=http://localhost:9200
OLLAMA_HOST=http://localhost:11434

# Test arXiv settings
ARXIV__MAX_RESULTS=15
```

测试默认 mock 掉所有外部服务（见第 [11](11-testing.md) 章），所以这里的连接串不需要真实可达。

---

## 2.9 `.gitignore`

创建 `.gitignore`，避免把虚拟环境、数据、密钥提交进仓库。

### 文件：`.gitignore`（关键条目）

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
.mypy_cache/
.ruff_cache/
.pytest_cache/

# Environment & secrets
.env

# Data & artifacts
data/
*.pdf

# Jupyter
.ipynb_checkpoints/

# Airflow
airflow/logs/
airflow/plugins/
airflow/*.pid
airflow/*.generated
airflow/simple_auth_manager_passwords.json.generated

# OS
.DS_Store
```

> **务必把 `.env` 加入 `.gitignore`**：它含密钥。只提交 `.env.example`（占位符版本）。

---

## 2.10 本章产出与验证

到这里你的目录应该是：

```
arxiv-paper-curator/
├── .env                # 真实配置（含密钥，不提交）
├── .env.example        # 示例配置（提交）
├── .env.test           # 测试配置（提交）
├── .gitignore
├── pyproject.toml
├── uv.lock             # uv sync 生成（提交）
└── .venv/              # 虚拟环境（不提交）
```

验证环境就绪：

```bash
# Python 版本正确
uv run python --version          # 期望 Python 3.12.x

# 依赖装好了（随便验证一个）
uv run python -c "import fastapi, sqlalchemy, opensearchpy, langgraph; print('deps OK')"

# Docker 可用
docker compose version
```

三条命令都正常输出，就可以进入下一章——开始写真正的代码。下一章 [`03-architecture-and-design.md`](03-architecture-and-design.md) 先讲架构与目录结构，让你在动手前理解每个文件的位置和职责。
