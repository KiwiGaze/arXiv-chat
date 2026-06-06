# File: tutorials/12-run-build-deploy-rollback.md

# 第 12 章　本地运行、构建、部署与回滚

本章汇总把系统跑起来、构建镜像、部署到服务器、以及出问题时安全回滚的全部操作。

---

## 12.1 Makefile：常用命令快捷方式

### 文件：`Makefile`（项目根目录，逐字复制）

```makefile
.PHONY: help start stop restart status logs health setup format lint test test-cov clean

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Service management
start: ## Start all services
	docker compose up --build -d

stop: ## Stop all services
	docker compose down

restart: ## Restart all services
	docker compose restart

status: ## Show service status
	docker compose ps

logs: ## Show service logs
	docker compose logs -f

# Health checks
health: ## Check all services health
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health | jq . || echo "API not responding"
	@curl -s http://localhost:9200/_cluster/health | jq . || echo "OpenSearch not responding"
	@curl -s http://localhost:8080/api/v2/monitor/health || echo "Airflow not responding"
	@curl -s http://localhost:11434/api/version | jq . || echo "Ollama not responding"

# Development
setup: ## Install Python dependencies
	uv sync

format: ## Format code
	uv run ruff format

lint: ## Lint and type check
	uv run ruff check --fix
	uv run mypy src/

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=html

# Cleanup
clean: ## Clean up everything
	docker compose down -v
	docker system prune -f
```

> **`make health` 里的 `/health` 注意**：Makefile 用的是 `http://localhost:8000/health`，但本项目的健康端点实际在 `http://localhost:8000/api/v1/health`（见 `main.py` 的 `prefix="/api/v1"`）。所以 `make health` 的 API 那行可能报 "API not responding"。**正确的健康检查命令是 `curl http://localhost:8000/api/v1/health`**（这也是 `compose.yml` 里 api 容器 healthcheck 用的路径）。这属于上游 Makefile 的一处小瑕疵，本教程逐字保留并在此提醒。

### 命令速查

| 命令 | 作用 |
|------|------|
| `make help` | 列出所有命令 |
| `make start` | 构建并启动全部服务（`docker compose up --build -d`） |
| `make stop` | 停止服务（保留数据卷） |
| `make restart` | 重启服务 |
| `make status` | 查看服务状态 |
| `make logs` | 跟随查看日志 |
| `make health` | 检查各服务健康 |
| `make setup` | `uv sync` 安装依赖 |
| `make format` | `ruff format` 格式化 |
| `make lint` | `ruff check --fix` + `mypy src/` |
| `make test` | 跑测试 |
| `make test-cov` | 跑测试 + 覆盖率 |
| `make clean` | **删除所有数据卷** + 清理 Docker（危险，见 12.5） |

---

## 12.2 本地完整启动顺序

第一次从零启动，推荐分步进行（避免一次性拉起十几个容器导致内存峰值）：

```bash
# 0. 前置：已 cp .env.example .env，并填好 LANGFUSE_ENCRYPTION_KEY / JINA_API_KEY（见第 02 章）
uv sync

# 1. 基础数据层
docker compose up -d postgres opensearch opensearch-dashboards redis ollama
docker compose ps     # 等 postgres/opensearch 变 healthy（约 30–60s）

# 2. 拉取本地大模型（首次，约 1.3GB）
docker exec -it rag-ollama ollama pull gemma4:e2b

# 3. 构建并启动 API
docker compose up -d --build api

# 4.（可选）数据管道
docker compose up -d --build airflow      # http://localhost:8080  admin/admin

# 5.（可选）监控栈
docker compose up -d clickhouse langfuse-postgres langfuse-redis langfuse-minio
docker compose up -d langfuse-web langfuse-worker    # http://localhost:3001

# 6. 验证
curl -s http://localhost:8000/api/v1/health | python -m json.tool

# 7.（可选）Gradio 界面（本机进程，非容器）
uv run python gradio_launcher.py          # http://localhost:7861
```

一键启动（所有服务，需机器内存充足且 .env 已就绪）：

```bash
make start        # docker compose up --build -d
```

> 首次 `make start` 会构建 api 与 airflow 镜像、拉取所有基础镜像，耗时较长且吃内存。机器内存紧张时建议用上面的分步方式。

### 数据初始化（让系统"有东西可搜"）

新系统索引是空的。两种方式灌数据：

1. **Airflow（推荐）**：打开 http://localhost:8080，手动触发 `arxiv_paper_ingestion` DAG。它会抓取→解析→入库→分块→嵌入→索引。
2. **脚本**：参考第 [05](05-week2-ingestion.md)、[07](07-week4-hybrid-search.md) 章的验证脚本，调用 `MetadataFetcher` + `HybridIndexingService`。

---

## 12.3 构建

### 构建 API 镜像

```bash
# 通过 compose（推荐）
docker compose build api

# 或直接 docker build（带版本号）
docker build --build-arg VERSION=0.1.0 -t arxiv-curator-api:0.1.0 .
```

`Dockerfile`（第 [04](04-week1-infrastructure.md) 章）是多阶段构建：`uv sync --frozen --no-dev` 用锁文件装运行依赖，最终镜像基于 `python:3.12.8-slim`。

### 构建 Airflow 镜像

```bash
docker compose build airflow
```

### 依赖锁定与更新

```bash
uv sync                    # 按 uv.lock 安装（可复现）
uv lock --upgrade          # 升级依赖并更新 uv.lock（之后跑测试验证）
```

> **务必提交 `uv.lock`**：它保证团队、CI、镜像装到完全一致的版本（含哈希，供应链安全）。

---

## 12.4 部署

本项目以本地/单机 Docker Compose 为主。部署到服务器的关键步骤与加固点：

### 部署前检查清单

1. **替换所有开发默认密钥/密码**（详见第 [14](14-quality-performance-security.md) 章安全清单）：
   - PostgreSQL：`rag_user`/`rag_password` → 强密码
   - Langfuse：`LANGFUSE_ENCRYPTION_KEY`（`openssl rand -hex 32`）、`LANGFUSE_NEXTAUTH_SECRET`、`LANGFUSE_SALT`、管理员 `admin123`
   - MinIO / Langfuse-Redis 密码
   - Airflow `admin`/`admin`
2. **开启 OpenSearch 安全插件**（生产不要 `DISABLE_SECURITY_PLUGIN=true`）。
3. **不要把端口直接暴露公网**：用反向代理（Nginx/Caddy）+ TLS，只暴露必要端口（如 API 的 8000）。
4. **环境变量**：`ENVIRONMENT=production`、`DEBUG=false`、`LANGFUSE_DEBUG=false`。
5. **资源**：给 OpenSearch/Ollama 足够内存；考虑把 Ollama 换成 GPU 或外部 LLM。
6. **数据卷**：用持久化卷或外部托管数据库；定期备份。

### 单机部署（最简）

```bash
# 在服务器上
git clone <your-repo> && cd arxiv-paper-curator
cp .env.example .env          # 然后编辑 .env，替换全部密钥
docker compose up --build -d
```

### 生产分层建议

| 组件 | 生产建议 |
|------|----------|
| API | 多副本 + 反向代理 + TLS；`uvicorn --workers` 已设 4，可按 CPU 调 |
| PostgreSQL | 用托管数据库（RDS/Cloud SQL）或独立实例 + 备份；用 Alembic 管理迁移（见第 [15](15-cicd-and-maintenance.md) 章） |
| OpenSearch | 多节点集群、副本≥1、开启安全；按数据量调分片 |
| Ollama | GPU 实例或换云端 LLM；按并发调 |
| Redis | 托管 Redis；评估持久化策略 |
| Langfuse | 自托管栈较重，生产可用 Langfuse Cloud 或独立部署 |
| Airflow | 独立调度器/执行器；生产用 `CeleryExecutor`/`KubernetesExecutor` 而非 `LocalExecutor` |

### 滚动更新（零停机思路）

```bash
git pull
docker compose build api
docker compose up -d --no-deps api     # 只重建/重启 api，不动数据层
docker compose logs -f api             # 观察健康
```

`api` 容器有 healthcheck（`/api/v1/health`），编排器据此判断新实例是否就绪。

---

## 12.5 回滚

回滚分三个层面：代码、镜像/容器、数据。

### (1) 回滚代码到某一周的稳定版本

上游为每周打了 release tag（`week1.0` … `week7.0`）。如果你基于上游仓库：

```bash
# 查看可用 tag
git tag

# 回滚到某一周（只读检出）
git checkout week5.0

# 或回滚到上一个提交
git checkout <commit-sha>

# 重新同步依赖并重建
uv sync
docker compose up -d --build
```

> 如果你是自己的仓库，请确保发布时打 tag（如 `v1.0.0`），回滚时 `git checkout <tag>`。

### (2) 回滚容器/镜像

```bash
# 停止当前服务（保留数据卷）
docker compose down

# 切回旧代码/旧镜像 tag 后重启
docker compose up -d

# 若用具名镜像 tag 部署，直接指定旧 tag 重启该服务
docker compose up -d --no-deps api
```

### (3) 重置或回滚数据

```bash
# 仅停止（数据保留）——最常用，可安全反复
docker compose down

# ⚠️ 危险：删除全部数据卷（PostgreSQL、OpenSearch、Ollama 模型、Redis、Langfuse 数据全没）
docker compose down -v

# make clean 同样会 down -v 并 prune，谨慎
make clean
```

> **`down` vs `down -v`**：`down` 只删容器与网络，**数据卷保留**，重启后数据还在；`down -v` 会**连数据卷一起删**——这是"彻底重置"用的（如索引映射改坏了、想从干净状态重来），但会丢失所有论文、索引、模型权重。请确认后再用。

### 完整重置流程（开发环境踩坑后从头来）

```bash
docker compose down -v          # 清空所有数据
docker compose up --build -d    # 全新启动
# 重新拉模型 + 重新灌数据（触发 Airflow DAG）
```

### 索引层的"软回滚"

如果只是 OpenSearch 索引坏了（不想动其它数据），可强制重建索引（会丢索引数据，但 PostgreSQL 里的论文还在，可重新索引）：

```python
# 临时脚本
from src.services.opensearch.factory import make_opensearch_client_fresh
client = make_opensearch_client_fresh()
print(client.setup_indices(force=True))   # force=True 删旧索引重建
```

之后再触发 Airflow 索引任务，从 PostgreSQL 重新分块嵌入入索引。

---

## 12.6 本章小结

- ✅ Makefile 全部命令（及 `make health` 路径小瑕疵的提醒）。
- ✅ 分步与一键启动、数据初始化。
- ✅ 镜像构建与依赖锁定。
- ✅ 部署清单、生产分层建议、滚动更新。
- ✅ 三层回滚（代码 tag / 容器镜像 / 数据卷）与完整重置。

下一章 [`13-troubleshooting.md`](13-troubleshooting.md) 是常见问题排查手册。
