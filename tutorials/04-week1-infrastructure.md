# File: tutorials/04-week1-infrastructure.md

# 第 4 章　Week 1：基础设施与 FastAPI 骨架

**本周目标**：用 Docker Compose 把多服务栈编排起来，搭好 FastAPI 应用骨架、配置系统、PostgreSQL 数据层与健康检查端点。完成后你能 `docker compose up` 拉起服务，并用 `curl` 看到 `/api/v1/health` 返回各服务状态。

> 本章建立的"稳定文件"（`config.py`、数据库层、`models/paper.py`、异常、健康 schema）会一直用到最后。`main.py` / `dependencies.py` / `ping.py` 本周给出**引导版**（只接 DB），后续每周扩展，**最终完整版在第 [10](10-week7-agentic-telegram.md) 章逐字给出**。

---

## 4.1 容器镜像：`Dockerfile`

API 用多阶段构建：第一阶段用 `uv` 装依赖，第二阶段拷进精简的 Python 运行镜像。

### 文件：`Dockerfile`（逐字复制）

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

# Copy configuration files
COPY pyproject.toml uv.lock ./

# UV_COMPILE_BYTECODE for generating .pyc files -> faster application startup.
# UV_LINK_MODE=copy to silence warnings about not being able to use hard links
# since the cache and sync target are on separate file systems.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=/app/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/app/pyproject.toml \
    uv sync --frozen --no-dev

# Copy source code
COPY src /app/src

FROM python:3.12.8-slim AS final

EXPOSE 8000

# PYTHONUNBUFFERED=1 to disable output buffering
ENV PYTHONUNBUFFERED=1
ARG VERSION=0.1.0
ENV APP_VERSION=$VERSION

WORKDIR /app

# Copy the virtual environment from the base stage
COPY --from=base /app /app

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

> **为什么用多阶段构建 + `uv sync --frozen --no-dev`？**
> - **为什么这么选**：`base` 阶段含构建工具与缓存，`final` 阶段只保留运行所需，镜像更小、更安全（攻击面小）。`--frozen` 强制用 `uv.lock` 的精确版本（可复现）；`--no-dev` 不装测试/lint 依赖（更小）。
> - **替代方案**：单阶段构建（更大）、用 pip + requirements.txt（无哈希锁定）。
> - **影响**：启动快（`UV_COMPILE_BYTECODE` 预编译 .pyc）、镜像小（性能/安全）；版本锁定（可复现）。
> - **风险与缓解**：`--workers 4` 在内存紧张时可调小；构建缓存挂载需要 BuildKit（Docker 默认已启用）。

---

## 4.2 服务编排：`compose.yml`

这是**整个项目唯一的编排文件**。它定义了全部服务。为避免在各章之间产生"compose 漂移"，我们**一次性给出完整文件**，并标注每个服务属于哪一周。

> **如何按周启动**：
> - **Week 1–5 核心**：`postgres`、`opensearch`、`opensearch-dashboards`、`ollama`、`redis`、`api`。
> - **Week 2**：`airflow`（需要先按第 [05](05-week2-ingestion.md) 章创建 `airflow/` 构建上下文）。
> - **Week 6**：`langfuse-web`、`langfuse-worker`、`clickhouse`、`langfuse-postgres`、`langfuse-redis`、`langfuse-minio`（详见第 [09](09-week6-monitoring-caching.md) 章）。
>
> 本周只需启动核心子集（命令见 [4.13](#413-启动与验证)），**不要**直接 `docker compose up`（那会尝试构建尚不存在的 `airflow/`）。

### 文件：`compose.yml`（逐字复制）

```yaml
services:
  # API service
  api:
    build: .
    container_name: rag-api
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      opensearch:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')\""]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    env_file:
      - .env
    environment:
      # Container-specific overrides
      - OPENSEARCH_HOST=http://opensearch:9200
      - OPENSEARCH__HOST=http://opensearch:9200
      - OLLAMA_HOST=http://ollama:11434
      - POSTGRES_DATABASE_URL=postgresql+psycopg2://rag_user:rag_password@postgres:5432/rag_db
      - LANGFUSE_HOST=http://langfuse-web:3000 
      - LANGFUSE_DEBUG=true
      - REDIS__HOST=redis
    networks:
      - rag-network

  redis:
    image: redis:7-alpine
    container_name: rag-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    networks:
      - rag-network

  opensearch:
    image: opensearchproject/opensearch:2.19.0
    container_name: rag-opensearch
    environment:
      - discovery.type=single-node
      - OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m
      - DISABLE_SECURITY_PLUGIN=true
      - bootstrap.memory_lock=true
    ports:
      - "9200:9200"
      - "9600:9600"
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - opensearch_data:/usr/share/opensearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    restart: unless-stopped
    networks:
      - rag-network

  opensearch-dashboards:
    image: opensearchproject/opensearch-dashboards:2.19.0
    container_name: rag-dashboards
    ports:
      - "5601:5601"
    environment:
      - OPENSEARCH_HOSTS=http://opensearch:9200
      - DISABLE_SECURITY_DASHBOARDS_PLUGIN=true
    depends_on:
      - opensearch
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:5601/api/status || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    networks:
      - rag-network

  airflow:
    build: 
      context: ./airflow
      dockerfile: Dockerfile
    container_name: rag-airflow
    # user: "50000:0"  # Uncomment for WSL/Ubuntu, comment out for Mac
    depends_on:
      postgres:
        condition: service_healthy
    env_file:
      - .env
    environment:
      # Airflow-specific environment setup
      - AIRFLOW_HOME=/opt/airflow
      - PYTHONPATH=/opt/airflow/src
      # Container hostnames for Airflow
      - POSTGRES_DATABASE_URL=postgresql+psycopg2://rag_user:rag_password@postgres:5432/rag_db
      - OPENSEARCH_HOST=http://opensearch:9200
      - OPENSEARCH__HOST=http://opensearch:9200
      - OLLAMA_HOST=http://ollama:11434
      - REDIS__HOST=redis
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - airflow_logs:/opt/airflow/logs
      - ./airflow/plugins:/opt/airflow/plugins
      - ./src:/opt/airflow/src
    ports:
      - "8080:8080"
    # Command handled by entrypoint.sh
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
    networks:
      - rag-network

  ollama:
    image: ollama/ollama:0.30.4
    container_name: rag-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    networks:
      - rag-network

  postgres:
    image: postgres:16-alpine
    container_name: rag-postgres
    environment:
      - POSTGRES_DB=rag_db
      - POSTGRES_USER=rag_user
      - POSTGRES_PASSWORD=rag_password
      - POSTGRES_HOST_AUTH_METHOD=password
      - PGDATA=/var/lib/postgresql/data/pgdata
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag_user -d rag_db"]
      interval: 5s
      timeout: 5s
      retries: 10
      start_period: 30s
    restart: unless-stopped
    networks:
      - rag-network

  # ClickHouse for Langfuse analytics
  clickhouse:
    image: clickhouse/clickhouse-server:24.8-alpine
    container_name: rag-clickhouse
    environment:
      - CLICKHOUSE_DB=langfuse
      - CLICKHOUSE_USER=langfuse
      - CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1
      - CLICKHOUSE_PASSWORD=langfuse
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    healthcheck:
      test: ["CMD", "clickhouse-client", "--query", "SELECT 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    restart: unless-stopped
    networks:
      - rag-network

  langfuse-worker:
    image: docker.io/langfuse/langfuse-worker:3
    container_name: rag-langfuse-worker
    restart: unless-stopped
    depends_on:
      langfuse-postgres:
        condition: service_healthy
      langfuse-minio:
        condition: service_healthy
      langfuse-redis:
        condition: service_healthy
      clickhouse:
        condition: service_healthy
    ports:
      - "3030:3030"
    environment:
      NEXTAUTH_URL: http://localhost:3001
      DATABASE_URL: postgresql://langfuse:langfuse@langfuse-postgres:5432/langfuse
      # Security: Load sensitive values from .env file
      SALT: ${LANGFUSE_SALT}
      ENCRYPTION_KEY: ${LANGFUSE_ENCRYPTION_KEY}
      TELEMETRY_ENABLED: "false"
      LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: "true"
      CLICKHOUSE_MIGRATION_URL: clickhouse://clickhouse:9000
      CLICKHOUSE_URL: http://clickhouse:8123
      CLICKHOUSE_USER: langfuse
      CLICKHOUSE_PASSWORD: langfuse
      CLICKHOUSE_CLUSTER_ENABLED: "false"
      LANGFUSE_USE_AZURE_BLOB: "false"
      LANGFUSE_S3_EVENT_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_EVENT_UPLOAD_REGION: auto
      LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID: ${LANGFUSE_MINIO_ACCESS_KEY}
      LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY: ${LANGFUSE_MINIO_SECRET_KEY}
      LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT: http://langfuse-minio:9000
      LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE: "true"
      LANGFUSE_S3_EVENT_UPLOAD_PREFIX: events/
      LANGFUSE_S3_MEDIA_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_MEDIA_UPLOAD_REGION: auto
      LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID: ${LANGFUSE_MINIO_ACCESS_KEY}
      LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY: ${LANGFUSE_MINIO_SECRET_KEY}
      LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT: http://localhost:9090
      LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE: "true"
      LANGFUSE_S3_MEDIA_UPLOAD_PREFIX: media/
      REDIS_HOST: langfuse-redis
      REDIS_PORT: 6379
      REDIS_AUTH: ${LANGFUSE_REDIS_PASSWORD}
      REDIS_TLS_ENABLED: "false"
    networks:
      - rag-network

  langfuse-web:
    image: docker.io/langfuse/langfuse:3
    container_name: rag-langfuse-web
    restart: unless-stopped
    depends_on:
      langfuse-postgres:
        condition: service_healthy
      langfuse-minio:
        condition: service_healthy
      langfuse-redis:
        condition: service_healthy
      clickhouse:
        condition: service_healthy
    ports:
      - "3001:3000"
    environment:
      NEXTAUTH_URL: http://localhost:3001
      # Security: Load sensitive values from .env file
      NEXTAUTH_SECRET: ${LANGFUSE_NEXTAUTH_SECRET}
      DATABASE_URL: postgresql://langfuse:langfuse@langfuse-postgres:5432/langfuse
      SALT: ${LANGFUSE_SALT}
      ENCRYPTION_KEY: ${LANGFUSE_ENCRYPTION_KEY}
      TELEMETRY_ENABLED: "false"
      LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: "true"
      CLICKHOUSE_MIGRATION_URL: clickhouse://clickhouse:9000
      CLICKHOUSE_URL: http://clickhouse:8123
      CLICKHOUSE_USER: langfuse
      CLICKHOUSE_PASSWORD: langfuse
      CLICKHOUSE_CLUSTER_ENABLED: "false"
      LANGFUSE_USE_AZURE_BLOB: "false"
      LANGFUSE_S3_EVENT_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_EVENT_UPLOAD_REGION: auto
      LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID: ${LANGFUSE_MINIO_ACCESS_KEY}
      LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY: ${LANGFUSE_MINIO_SECRET_KEY}
      LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT: http://langfuse-minio:9000
      LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE: "true"
      LANGFUSE_S3_EVENT_UPLOAD_PREFIX: events/
      LANGFUSE_S3_MEDIA_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_MEDIA_UPLOAD_REGION: auto
      LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID: ${LANGFUSE_MINIO_ACCESS_KEY}
      LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY: ${LANGFUSE_MINIO_SECRET_KEY}
      LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT: http://localhost:9090
      LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE: "true"
      LANGFUSE_S3_MEDIA_UPLOAD_PREFIX: media/
      REDIS_HOST: langfuse-redis
      REDIS_PORT: 6379
      REDIS_AUTH: ${LANGFUSE_REDIS_PASSWORD}
      REDIS_TLS_ENABLED: "false"
      LANGFUSE_INIT_ORG_ID: "default-org"
      LANGFUSE_INIT_ORG_NAME: "RAG Organization"
      LANGFUSE_INIT_PROJECT_NAME: "Agentic RAG"
      LANGFUSE_INIT_USER_EMAIL: "admin@example.com"
      LANGFUSE_INIT_USER_NAME: "Admin User"
      LANGFUSE_INIT_USER_PASSWORD: "admin123"
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:3000/api/public/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    networks:
      - rag-network

  langfuse-postgres:
    image: postgres:17
    container_name: rag-langfuse-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=langfuse
      - POSTGRES_PASSWORD=langfuse
      - POSTGRES_DB=langfuse
      - POSTGRES_HOST_AUTH_METHOD=password
      - TZ=UTC
      - PGTZ=UTC
    ports:
      - "5433:5432"
    volumes:
      - langfuse_v3_postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse -d langfuse"]
      interval: 3s
      timeout: 3s
      retries: 10
      start_period: 30s
    networks:
      - rag-network

  langfuse-redis:
    image: docker.io/redis:7
    container_name: rag-langfuse-redis
    restart: unless-stopped
    command: --requirepass langfuse_redis_password
    ports:
      - "6380:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "langfuse_redis_password", "ping"]
      interval: 3s
      timeout: 10s
      retries: 10
    networks:
      - rag-network

  langfuse-minio:
    image: docker.io/minio/minio
    container_name: rag-langfuse-minio
    restart: unless-stopped
    entrypoint: sh
    command: -c 'mkdir -p /data/langfuse && minio server --address ":9000" --console-address ":9001" /data'
    environment:
      - MINIO_ROOT_USER=langfuse_minio
      - MINIO_ROOT_PASSWORD=langfuse_minio_secret
    ports:
      - "9090:9000"
      - "9091:9001"
    volumes:
      - langfuse_v3_minio_data:/data
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      timeout: 5s
      retries: 10
      start_period: 5s
    networks:
      - rag-network


volumes:
  postgres_data:
  opensearch_data:
  ollama_data:
  airflow_logs:
  clickhouse_data:
  redis_data:
  langfuse_v3_postgres_data:
  langfuse_v3_minio_data:

networks:
  rag-network:
    driver: bridge
```

### compose 关键点解读

- **`depends_on` + `condition: service_healthy`**：`api` 会等到 `postgres`/`opensearch`/`redis` 健康检查通过才启动，避免"DB 还没起来就连"的竞态。
- **每个服务都有 `healthcheck`**：这是生产实践——编排器据此判断服务可用性。
- **`environment` 覆盖**：`.env` 里用的是 `localhost`（便于你在宿主机直接连），但容器之间要用**服务名**互联。所以 `api` 容器里把 `OPENSEARCH__HOST`、`OLLAMA_HOST`、`POSTGRES_DATABASE_URL`、`REDIS__HOST` 覆盖为 `opensearch`/`ollama`/`postgres`/`redis`。
- **OpenSearch 关闭了安全插件**（`DISABLE_SECURITY_PLUGIN=true`）：仅适合本地开发，**生产必须开启**（见第 [14](14-quality-performance-security.md) 章）。
- **命名卷**（`volumes:` 段）：数据持久化，`docker compose down` 不会丢；`docker compose down -v` 才会删（回滚/重置用，见第 [12](12-run-build-deploy-rollback.md) 章）。

> **为什么所有东西都用 Docker Compose，而不是本机直接装 Postgres/OpenSearch？**
> - **为什么这么选**：一条命令拉起完全一致的环境，消除"在我机器上能跑"问题；版本由镜像锁定。
> - **替代方案**：本机各自安装、Kubernetes（生产级但本地过重）。
> - **影响**：可复现性极强、新人上手快（可维护）；本地资源占用较高（性能/成本）。
> - **风险与缓解**：内存不足导致 OpenSearch OOM——已设 `-Xms512m -Xmx512m` 限制堆；Docker Desktop 内存建议 ≥8GB。

---

## 4.3 配置系统：`src/config.py`

这是全项目配置的中枢。先创建包目录与空文件：

```bash
mkdir -p src/db/interfaces src/models src/repositories src/routers
mkdir -p src/schemas/api src/schemas/arxiv src/schemas/database src/schemas/pdf_parser
mkdir -p src/schemas/embeddings src/schemas/indexing src/schemas/telegram src/schemas/common
mkdir -p src/services
touch src/__init__.py
```

### 文件：`src/config.py`（逐字复制）

```python
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class BaseConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class ArxivSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="ARXIV__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    base_url: str = "https://export.arxiv.org/api/query"
    pdf_cache_dir: str = "./data/arxiv_pdfs"
    rate_limit_delay: float = 3.0
    timeout_seconds: int = 30
    max_results: int = 15
    search_category: str = "cs.AI"
    download_max_retries: int = 3
    download_retry_delay_base: float = 5.0
    max_concurrent_downloads: int = 5
    max_concurrent_parsing: int = 1

    namespaces: dict = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    @field_validator("pdf_cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v


class PDFParserSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="PDF_PARSER__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    max_pages: int = 30
    max_file_size_mb: int = 20
    do_ocr: bool = False
    do_table_structure: bool = True


class ChunkingSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="CHUNKING__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    chunk_size: int = 600  # Target words per chunk
    overlap_size: int = 100  # Words to overlap between chunks
    min_chunk_size: int = 100  # Minimum words for a valid chunk
    section_based: bool = True  # Use section-based chunking when available


class OpenSearchSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENSEARCH__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "http://localhost:9200"
    index_name: str = "arxiv-papers"
    chunk_index_suffix: str = "chunks"  # Creates single hybrid index: {index_name}-{suffix}
    max_text_size: int = 1000000

    # Vector search settings
    vector_dimension: int = 1024  # Jina embeddings dimension
    vector_space_type: str = "cosinesimil"  # cosinesimil, l2, innerproduct

    # Hybrid search settings
    rrf_pipeline_name: str = "hybrid-rrf-pipeline"
    hybrid_search_size_multiplier: int = 2  # Get k*multiplier for better recall


class LangfuseSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="LANGFUSE__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    public_key: str = ""
    secret_key: str = ""
    host: str = "http://localhost:3000"  # Self-hosted Langfuse URL
    enabled: bool = True
    flush_at: int = 15  # Number of events before flushing
    flush_interval: float = 1.0  # Seconds between flushes
    max_retries: int = 3
    timeout: int = 30
    debug: bool = False


class RedisSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="REDIS__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    decode_responses: bool = True
    socket_timeout: int = 30
    socket_connect_timeout: int = 30

    # Cache settings
    ttl_hours: int = 6  # Cache TTL in hours


class TelegramSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="TELEGRAM__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    bot_token: str = ""
    enabled: bool = False


class Settings(BaseConfigSettings):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "rag-api"

    postgres_database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 0

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e2b"
    ollama_timeout: int = 300

    # Jina AI embeddings configuration
    jina_api_key: str = ""

    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)

    @field_validator("postgres_database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not (v.startswith("postgresql://") or v.startswith("postgresql+psycopg2://")):
            raise ValueError("Database URL must start with 'postgresql://' or 'postgresql+psycopg2://'")
        return v


def get_settings() -> Settings:
    return Settings()
```

### config.py 关键点

- **`env_nested_delimiter="__"`**：让 `ARXIV__MAX_RESULTS` 自动映射到 `Settings().arxiv.max_results`。这就是第 [02](02-environment-and-dependencies.md) 章里双下划线的来历。
- **`frozen=True`**：配置对象不可变，避免运行时被意外改写（可维护/安全）。
- **`extra="ignore"`**：`.env` 里多余的变量不会报错（各子配置只取自己关心的前缀）。
- **`field_validator("pdf_cache_dir")`**：读取配置时顺手创建 PDF 缓存目录（副作用集中、可控）。
- **`validate_database_url`**：拒绝非 PostgreSQL 的连接串，提早失败。

---

## 4.4 数据库配置 schema：`src/schemas/database/config.py`

```bash
touch src/schemas/__init__.py src/schemas/database/__init__.py
touch src/schemas/api/__init__.py
```

### 文件：`src/schemas/database/config.py`（逐字复制）

```python
from pydantic import Field
from pydantic_settings import BaseSettings


class PostgreSQLSettings(BaseSettings):
    """PostgreSQL configuration settings."""

    database_url: str = Field(
        default="postgresql://rag_user:rag_password@localhost:5432/rag_db", description="PostgreSQL database URL"
    )
    echo_sql: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=20, description="Database connection pool size")
    max_overflow: int = Field(default=0, description="Maximum pool overflow")

    class Config:
        env_prefix = "POSTGRES_"
```

---

## 4.5 数据库抽象层：`src/db/interfaces/base.py`

```bash
touch src/db/__init__.py src/db/interfaces/__init__.py
```

> `src/db/__init__.py` 与 `src/db/interfaces/__init__.py` 都是**空文件**（仅作包标记）。

### 文件：`src/db/interfaces/base.py`（逐字复制）

```python
from abc import ABC, abstractmethod
from typing import Any, ContextManager, Dict, List, Optional

from sqlalchemy.orm import Session


class BaseDatabase(ABC):
    """Base class for database operations."""

    @abstractmethod
    def startup(self) -> None:
        """Initialize the database connection."""

    @abstractmethod
    def teardown(self) -> None:
        """Close the database connection."""

    @abstractmethod
    def get_session(self) -> ContextManager[Session]:
        """Get a database session."""


class BaseRepository(ABC):
    """Base repository pattern for data access."""

    def __init__(self, session: Session):
        self.session = session

    @abstractmethod
    def create(self, data: Dict[str, Any]) -> Any:
        """Create a new record."""

    @abstractmethod
    def get_by_id(self, record_id: Any) -> Optional[Any]:
        """Get a record by ID."""

    @abstractmethod
    def update(self, record_id: Any, data: Dict[str, Any]) -> Optional[Any]:
        """Update a record by ID."""

    @abstractmethod
    def delete(self, record_id: Any) -> bool:
        """Delete a record by ID."""

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """List records with pagination."""
```

> **为什么先定义抽象基类 `BaseDatabase`？**
> - **为什么这么选**：把"数据库能力"的契约与具体实现（PostgreSQL）解耦。
> - **替代方案**：直接写 PostgreSQL 类，不要抽象。
> - **优缺点**：抽象 ✅ 未来可换实现（如 SQLite 测试）、依赖注入更干净。❌ 单一实现时略显多余。
> - **影响**：对测试与未来扩展友好（可维护）。

---

## 4.6 PostgreSQL 实现：`src/db/interfaces/postgresql.py`

### 文件：`src/db/interfaces/postgresql.py`（逐字复制）

```python
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from src.db.interfaces.base import BaseDatabase
from src.schemas.database.config import PostgreSQLSettings

logger = logging.getLogger(__name__)


Base = declarative_base()


class PostgreSQLDatabase(BaseDatabase):
    """PostgreSQL database implementation."""

    def __init__(self, config: PostgreSQLSettings):
        self.config = config
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None

    def startup(self) -> None:
        """Initialize the database connection."""
        try:
            # Log connection attempt
            logger.info(
                f"Attempting to connect to PostgreSQL at: {self.config.database_url.split('@')[1] if '@' in self.config.database_url else 'localhost'}"
            )

            self.engine = create_engine(
                self.config.database_url,
                echo=self.config.echo_sql,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,  # Verify connections before use
            )

            self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

            # Test the connection
            assert self.engine is not None
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info("Database connection test successful")

            # Check which tables exist before creating
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()

            # Create tables if they don't exist (idempotent operation)
            Base.metadata.create_all(bind=self.engine)

            # Check if any new tables were created
            updated_tables = inspector.get_table_names()
            new_tables = set(updated_tables) - set(existing_tables)

            if new_tables:
                logger.info(f"Created new tables: {', '.join(new_tables)}")
            else:
                logger.info("All tables already exist - no new tables created")

            logger.info("PostgreSQL database initialized successfully")
            assert self.engine is not None
            logger.info(f"Database: {self.engine.url.database}")
            logger.info(f"Total tables: {', '.join(updated_tables) if updated_tables else 'None'}")
            logger.info("Database connection established")

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL database: {e}")
            raise

    def teardown(self) -> None:
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("PostgreSQL database connections closed")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call startup() first.")

        session = self.session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
```

### postgresql.py 关键点

- **`Base = declarative_base()`**：所有 ORM 模型继承它；`Base.metadata.create_all` 在启动时**幂等地**建表（表已存在则跳过）。这是本项目"无需手动迁移即可建表"的关键，方便教学。
- **`pool_pre_ping=True`**：每次取连接前先 ping，避免用到失效连接。
- **`get_session` 上下文管理器**：异常自动 `rollback`，无论如何都 `close`，杜绝连接泄漏。
- **`expire_on_commit=False`**：commit 后对象仍可访问其属性（避免懒加载报错）。

> **为什么用 `create_all` 而不是 Alembic 迁移？**
> - **为什么这么选**：教学项目里 `create_all` 零配置、幂等、即开即用。
> - **替代方案**：Alembic（项目已装，见 `pyproject.toml`）。
> - **优缺点**：`create_all` ✅ 简单。❌ 不能处理表结构**变更**（只建不改）。Alembic ✅ 可版本化迁移。❌ 需写迁移脚本。
> - **影响/建议**：生产应改用 Alembic 管理 schema 演进（见第 [15](15-cicd-and-maintenance.md) 章）。

---

## 4.7 数据库工厂：`src/db/factory.py`

### 文件：`src/db/factory.py`（逐字复制）

```python
from src.config import get_settings
from src.db.interfaces.base import BaseDatabase
from src.db.interfaces.postgresql import PostgreSQLDatabase
from src.schemas.database.config import PostgreSQLSettings


def make_database() -> BaseDatabase:
    """Factory function to create a database instance.

    :returns: An instance of the database
    :rtype: BaseDatabase
    """
    # Get settings from centralized config
    settings = get_settings()

    # Create PostgreSQL config from settings
    config = PostgreSQLSettings(
        database_url=settings.postgres_database_url,
        echo_sql=settings.postgres_echo_sql,
        pool_size=settings.postgres_pool_size,
        max_overflow=settings.postgres_max_overflow,
    )

    database = PostgreSQLDatabase(config=config)
    database.startup()
    return database
```

---

## 4.8 全局会话辅助：`src/database.py`

### 文件：`src/database.py`（逐字复制）

```python
from contextlib import contextmanager

from src.db.factory import make_database

# Global database instance
_database = None


def get_database():
    """Get or create database instance."""
    global _database
    if _database is None:
        _database = make_database()
    return _database


@contextmanager
def get_db_session():
    """Get a database session context manager."""
    database = get_database()
    with database.get_session() as session:
        yield session
```

> 这个模块主要给**脚本/笔记本**用（在 FastAPI 请求之外也能拿到会话）。FastAPI 内部走的是 `dependencies.py` 的注入路径。

---

## 4.9 ORM 模型：`src/models/paper.py`

```bash
touch src/models/__init__.py
```

### 文件：`src/models/paper.py`（逐字复制）

```python
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from src.db.interfaces.postgresql import Base


class Paper(Base):
    __tablename__ = "papers"

    # Core arXiv metadata
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    arxiv_id = Column(String, unique=True, nullable=False, index=True)
    title = Column(String, nullable=False)
    authors = Column(JSON, nullable=False)
    abstract = Column(Text, nullable=False)
    categories = Column(JSON, nullable=False)
    published_date = Column(DateTime, nullable=False)
    pdf_url = Column(String, nullable=False)

    # Parsed PDF content (added for comprehensive storage)
    raw_text = Column(Text, nullable=True)
    sections = Column(JSON, nullable=True)
    references = Column(JSON, nullable=True)

    # PDF processing metadata
    parser_used = Column(String, nullable=True)
    parser_metadata = Column(JSON, nullable=True)
    pdf_processed = Column(Boolean, default=False, nullable=False)
    pdf_processing_date = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
```

### Paper 模型设计要点

- **`id` 用 UUID**：避免自增整型 ID 的可枚举性（安全），分布式更友好。
- **`arxiv_id` 唯一 + 建索引**：用 arXiv ID 做业务主键去重（`upsert` 依赖它）。
- **`authors`/`categories`/`sections`/`references` 用 `JSON` 列**：这些是变长结构化数据，JSON 列省去额外关联表。
- **`raw_text`/`sections` 等可空**：论文先入库元数据，PDF 解析成功后再回填（解析可能失败或被跳过）。
- **`created_at`/`updated_at` 用带时区的 UTC**：时间统一为 UTC，避免时区混乱。

---

## 4.10 异常层级：`src/exceptions.py`

### 文件：`src/exceptions.py`（逐字复制）

```python
class RepositoryException(Exception):
    """Base exception for repository-related errors."""


class PaperNotFound(RepositoryException):
    """Exception raised when paper data is not found."""


class PaperNotSaved(RepositoryException):
    """Exception raised when paper data is not saved."""


class ParsingException(Exception):
    """Base exception for parsing-related errors."""


# Week 2: PDF parsing exceptions (implemented)
class PDFParsingException(ParsingException):
    """Base exception for PDF parsing-related errors."""


class PDFValidationError(PDFParsingException):
    """Exception raised when PDF file validation fails."""


class PDFDownloadException(Exception):
    """Base exception for PDF download-related errors."""


class PDFDownloadTimeoutError(PDFDownloadException):
    """Exception raised when PDF download times out."""


class PDFCacheException(Exception):
    """Exception raised for PDF cache-related errors."""


# Week 3+: OpenSearch exceptions (placeholders for Week 1)
class OpenSearchException(Exception):
    """Base exception for OpenSearch-related errors."""


# Week 2+: ArXiv API exceptions
class ArxivAPIException(Exception):
    """Base exception for arXiv API-related errors."""


class ArxivAPITimeoutError(ArxivAPIException):
    """Exception raised when arXiv API request times out."""


class ArxivAPIRateLimitError(ArxivAPIException):
    """Exception raised when arXiv API rate limit is exceeded."""


class ArxivParseError(ArxivAPIException):
    """Exception raised when arXiv API response parsing fails."""


# Week 2+: Metadata fetching exceptions
class MetadataFetchingException(Exception):
    """Base exception for metadata fetching pipeline errors."""


class PipelineException(MetadataFetchingException):
    """Exception raised during pipeline execution."""


class LLMException(Exception):
    """Base exception for LLM-related errors."""


class OllamaException(LLMException):
    """Exception raised for Ollama service errors."""


class OllamaConnectionError(OllamaException):
    """Exception raised when cannot connect to Ollama service."""


class OllamaTimeoutError(OllamaException):
    """Exception raised when Ollama service times out."""


# General application exceptions
class ConfigurationError(Exception):
    """Exception raised when configuration is invalid."""
```

> 我们**一次性定义全部异常**（含后续周才用到的），这样后面各章直接 `from src.exceptions import ...` 即可，不用反复改这个文件。分层的异常让调用方能按粒度捕获（如先捕 `PDFValidationError`，再兜底 `PDFParsingException`）。

---

## 4.11 简单日志中间件：`src/middlewares.py`

### 文件：`src/middlewares.py`（逐字复制）

```python
import logging

logger = logging.getLogger(__name__)


# What's missing and why:
# - These functions are not integrated into FastAPI middleware system
# - No automatic request/response logging or timing
# - No error handling or request tracing
# - Functions exist for future use when middleware integration is needed
#
# In production, these would be used via FastAPI middleware decorators
# or integrated into a proper BaseHTTPMiddleware class for automatic
# request logging, performance monitoring, and error tracking.


def log_request(method: str, path: str) -> None:
    """Simple request logging for Week 1."""
    logger.info(f"{method} {path}")


def log_error(error: str, method: str, path: str) -> None:
    """Simple error logging for Week 1."""
    logger.error(f"Error in {method} {path}: {error}")
```

---

## 4.12 健康检查 schema 与端点

### 文件：`src/schemas/api/health.py`（逐字复制）

```python
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """Individual service status."""

    status: str = Field(..., description="Service status", examples=["healthy"])
    message: Optional[str] = Field(None, description="Status message", examples=["Connected successfully"])


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Overall health status", examples=["ok"])
    version: str = Field(..., description="Application version", examples=["0.1.0"])
    environment: str = Field(..., description="Deployment environment", examples=["development"])
    service_name: str = Field(..., description="Service identifier", examples=["rag-api"])
    services: Optional[Dict[str, ServiceStatus]] = Field(None, description="Individual service statuses")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "status": "ok",
                "version": "0.1.0",
                "environment": "development",
                "service_name": "rag-api",
                "services": {
                    "database": {"status": "healthy", "message": "Connected successfully"},
                    "pdf_parser": {"status": "healthy", "message": "Docling parser ready"},
                },
            }
        }
```

### Week 1 引导版依赖注入：`src/dependencies.py`

```bash
touch src/routers/__init__.py
```

> ⚠️ **演进说明**：下面是 **Week 1 引导版**，只包含 DB 相关注入，保证本周可独立运行。随着各服务加入，本文件会逐步扩展；**最终完整版在第 [10](10-week7-agentic-telegram.md) 章逐字给出**。请先用这个版本。

### 文件：`src/dependencies.py`（Week 1 引导版）

```python
from functools import lru_cache
from typing import Annotated, Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from src.config import Settings
from src.db.interfaces.base import BaseDatabase


@lru_cache
def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def get_request_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    """Get database from the request state."""
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    """Get database session dependency."""
    with database.get_session() as session:
        yield session


# Dependency annotations
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
```

### Week 1 引导版健康端点：`src/routers/ping.py`

> ⚠️ **演进说明**：本周的 `/health` 只检查数据库。Week 3 加入 OpenSearch 检查、Week 5 加入 Ollama 检查；**最终完整版（DB + OpenSearch + Ollama）在第 [08](08-week5-rag-llm.md) 章逐字给出**。

### 文件：`src/routers/ping.py`（Week 1 引导版）

```python
from fastapi import APIRouter
from sqlalchemy import text

from ..dependencies import DatabaseDep, SettingsDep
from ..schemas.api.health import HealthResponse, ServiceStatus

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(settings: SettingsDep, database: DatabaseDep) -> HealthResponse:
    """Week 1 health check: verifies the database connection."""
    services = {}
    overall_status = "ok"

    try:
        with database.get_session() as session:
            session.execute(text("SELECT 1"))
        services["database"] = ServiceStatus(status="healthy", message="Connected successfully")
    except Exception as e:
        services["database"] = ServiceStatus(status="unhealthy", message=str(e))
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
    )
```

### Week 1 引导版应用入口：`src/main.py`

> ⚠️ **演进说明**：这是 **Week 1 引导版**，只接 DB 与 health。每周会往 `lifespan` 加服务、往 `app.include_router` 加路由；**最终完整版在第 [10](10-week7-agentic-telegram.md) 章逐字给出**。

### 文件：`src/main.py`（Week 1 引导版）

```python
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from src.config import get_settings
from src.db.factory import make_database
from src.routers import ping

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan for the API (Week 1: database only)."""
    logger.info("Starting RAG API...")

    settings = get_settings()
    app.state.settings = settings

    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    logger.info("API ready")
    yield

    database.teardown()
    logger.info("API shutdown complete")


app = FastAPI(
    title="arXiv Paper Curator API",
    description="Personal arXiv CS.AI paper curator with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# Include routers
app.include_router(ping.router, prefix="/api/v1")  # Health check endpoint


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
```

---

## 4.13 启动与验证

### 准备 .env 中的容器主机名

回顾：`.env` 里默认是 `localhost`，但容器之间用服务名。`compose.yml` 已经为 `api` 容器注入了正确的覆盖（`postgres`/`opensearch`/`ollama`/`redis`），所以 `.env` 保持默认即可。

### 启动 Week 1 子集

```bash
# 1. 先起基础数据服务（不含 airflow / langfuse）
docker compose up -d postgres opensearch opensearch-dashboards ollama redis

# 2. 等它们健康（postgres/opensearch 需要几十秒）
docker compose ps

# 3. 构建并启动 API
docker compose up -d --build api

# 4. 看日志确认启动成功
docker compose logs -f api
```

### 验证健康检查

```bash
curl -s http://localhost:8000/api/v1/health | python -m json.tool
```

期望输出类似：

```json
{
    "status": "ok",
    "version": "0.1.0",
    "environment": "development",
    "service_name": "rag-api",
    "services": {
        "database": {
            "status": "healthy",
            "message": "Connected successfully"
        }
    }
}
```

打开交互式文档：浏览器访问 **http://localhost:8000/docs**，应看到 `/api/v1/health` 端点，可直接点 "Try it out"。

### 本地（非容器）运行 API（可选，便于调试）

如果你想在宿主机直接跑 API（数据服务仍用 Docker），需要把 `.env` 里的 `localhost` 用上——但 `.env` 默认是容器名。最简单的方式是在容器里跑（上面的方式）。若要本机跑，临时导出覆盖：

```bash
# 让本机进程连到映射到宿主机的端口
export POSTGRES_DATABASE_URL="postgresql+psycopg2://rag_user:rag_password@localhost:5432/rag_db"
export OPENSEARCH__HOST="http://localhost:9200"
export OLLAMA_HOST="http://localhost:11434"
uv run uvicorn src.main:app --reload --port 8000
```

---

## 4.14 本章小结

你已经有了：

- ✅ 一键启动的多服务栈骨架（`compose.yml` + `Dockerfile`）。
- ✅ 集中式配置系统（`config.py` + 嵌套子配置）。
- ✅ PostgreSQL 数据层（抽象基类 + 实现 + 工厂 + 自动建表）。
- ✅ `Paper` ORM 模型与完整异常层级。
- ✅ 可工作的 `/api/v1/health` 健康检查。

**Week 1 里程碑达成**：`curl` 能看到数据库健康。下一章 [`05-week2-ingestion.md`](05-week2-ingestion.md) 会让系统"活起来"——自动从 arXiv 抓论文、解析 PDF、入库，并用 Airflow 编排成每日管道。
