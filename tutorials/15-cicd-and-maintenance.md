# File: tutorials/15-cicd-and-maintenance.md

# 第 15 章　CI/CD 与长期维护

本章给出代码质量自动化（pre-commit）、一套完整可用的 CI 流水线（GitHub Actions）、以及长期维护建议。

---

## 15.1 提交前钩子：`.pre-commit-config.yaml`

`pre-commit` 在每次 `git commit` 前自动跑格式化与类型检查，把问题挡在提交之前。

### 文件：`.pre-commit-config.yaml`（项目根目录，逐字复制）

```yaml
repos:
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.11.5
  hooks:
    - id: ruff
      args: [
        "--select=I",
        "--fix"
      ]
    - id: ruff-format
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: 'v1.15.0'
  hooks:
  -   id: mypy
      args: [
        --ignore-missing-imports,
        --disable-error-code=import-untyped
      ]
```

### 启用

```bash
# pre-commit 已是 dev 依赖（pyproject）
uv run pre-commit install        # 装 git 钩子
uv run pre-commit run --all-files  # 手动对全仓库跑一遍
```

之后每次 `git commit` 会自动：
- `ruff`（`--select=I` 排序 import + `--fix` 自动修）
- `ruff-format`（格式化）
- `mypy`（忽略缺失导入、忽略未标注类型的第三方）

> 钩子版本（`rev`）与 `pyproject.toml` 的 `ruff>=0.11.5`、`mypy>=1.15.0` 对齐。升级工具时同步更新两处。

---

## 15.2 完整 CI 流水线（GitHub Actions）

上游仓库未自带 CI；下面给出一套**完整可用**的流水线，跑 lint、类型检查、离线测试，并演示用服务容器跑集成测试。把它保存为 `.github/workflows/ci.yml`。

### 文件：`.github/workflows/ci.yml`（完整可用，可直接保存）

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint-and-test:
    name: Lint, type-check, unit & API tests
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Set up Python 3.12
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync --frozen

      - name: Ruff format check
        run: uv run ruff format --check

      - name: Ruff lint
        run: uv run ruff check

      - name: Mypy type check
        run: uv run mypy src/

      - name: Run unit & API tests (offline, services mocked)
        run: uv run pytest --ignore=tests/integration -q

  integration-tests:
    name: Integration tests (real OpenSearch)
    runs-on: ubuntu-latest
    services:
      opensearch:
        image: opensearchproject/opensearch:2.19.0
        env:
          discovery.type: single-node
          DISABLE_SECURITY_PLUGIN: "true"
          OPENSEARCH_JAVA_OPTS: "-Xms512m -Xmx512m"
        ports:
          - 9200:9200
        options: >-
          --health-cmd "curl -f http://localhost:9200/_cluster/health || exit 1"
          --health-interval 30s
          --health-timeout 10s
          --health-retries 10
          --health-start-period 60s
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Set up Python 3.12
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync --frozen

      - name: Wait for OpenSearch
        run: |
          for i in $(seq 1 30); do
            if curl -sf http://localhost:9200/_cluster/health >/dev/null; then
              echo "OpenSearch is up"; break
            fi
            echo "waiting for opensearch ($i)..."; sleep 5
          done

      - name: Run integration tests
        env:
          OPENSEARCH__HOST: http://localhost:9200
        run: uv run pytest tests/integration -q

  docker-build:
    name: Build API image
    runs-on: ubuntu-latest
    needs: lint-and-test
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build API image (no push)
        uses: docker/build-push-action@v6
        with:
          context: .
          push: false
          tags: arxiv-curator-api:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### 流水线说明

- **`lint-and-test`**：格式化检查 + lint + mypy + **离线测试**（`--ignore=tests/integration`，服务全 mock，快且确定）。
- **`integration-tests`**：用 GitHub Actions 的 **service containers** 起一个真实 OpenSearch，跑 `tests/integration`。注意 `test_arxiv_client_basic` 会真连 arXiv（需网络）；如不希望 CI 依赖外网，可在该测试加跳过标记或拆分。
- **`docker-build`**：验证 `Dockerfile` 能构建成功（用 GHA 缓存加速），`push: false` 只构建不推送。
- **`concurrency`**：同一分支新提交自动取消旧运行，省资源。
- **`--frozen`**：用 `uv.lock` 精确安装，CI 与本地一致。

> **扩展为 CD（部署）**：在 `docker-build` 后加一个 `deploy` job（仅 `main` 分支），登录镜像仓库、`push` 带版本标签的镜像，再 SSH/调度到服务器 `docker compose pull && up -d`，或推送到 K8s。部署 job 用 GitHub Environments + Secrets 管理凭据。

---

## 15.3 数据库迁移（Alembic）

当前用 `Base.metadata.create_all`（只建表、不改表）。生产应改用 **Alembic**（已是依赖）管理 schema 演进。最小接入：

```bash
# 初始化 alembic（在项目根）
uv run alembic init alembic
```

在 `alembic/env.py` 里把 `target_metadata` 指向项目的 `Base`：

```python
# alembic/env.py 关键改动（节选）
from src.db.interfaces.postgresql import Base
import src.models.paper  # noqa: F401  确保模型被导入注册到 Base.metadata

target_metadata = Base.metadata
```

并在 `alembic.ini` 把 `sqlalchemy.url` 指向你的数据库（或在 `env.py` 里从 `get_settings().postgres_database_url` 读取）。然后：

```bash
uv run alembic revision --autogenerate -m "create papers table"
uv run alembic upgrade head
```

> 接入 Alembic 后，应去掉/弱化 `PostgreSQLDatabase.startup` 里的 `create_all`（避免与迁移冲突），改由迁移负责建表与改表。

---

## 15.4 依赖管理与升级

```bash
# 升级所有依赖到允许范围内的最新，并更新 uv.lock
uv lock --upgrade
uv sync
uv run pytest --ignore=tests/integration   # 验证

# 升级单个依赖
uv lock --upgrade-package langgraph
```

维护节奏建议：
- **每月**：`uv lock --upgrade` + 跑测试，小步升级。
- **安全告警**：接入 GitHub Dependabot / `pip-audit`（可加一个 CI job 跑 `uvx pip-audit`）。
- **锁文件**：始终提交 `uv.lock`（含哈希，供应链安全）。
- **基础镜像**：定期更新 `compose.yml` 里的镜像 tag（OpenSearch 2.19、Postgres 16、Ollama 0.30.4、Langfuse 3 等），升级前在测试环境验证兼容性。

---

## 15.5 模型与提示词维护

- **LLM 模型**：`OLLAMA_MODEL` 可切换；升级模型后回归测试答案质量。新模型先 `ollama pull`。
- **嵌入模型**：换嵌入模型需保证维度与索引 `dimension` 一致，否则要改映射并 `setup_indices(force=True)` 重建 + 全量重索引。
- **提示词**：系统提示词放在 `src/services/ollama/prompts/rag_system.txt` 与 `src/services/agents/prompts.py`，改提示词无需改逻辑代码；建议用 Langfuse 观察改动对质量的影响。

---

## 15.6 监控与运维

- **Langfuse**：观察每次请求的检索/生成/耗时/token、用户反馈（`/feedback`），定位质量回退。
- **健康探针**：`/api/v1/health` 供负载均衡/编排器探活（返回各依赖状态）。
- **日志**：统一 `logging`（`%(asctime)s - %(name)s - %(levelname)s - %(message)s`）；生产接入集中日志（ELK/Cloud Logging）并脱敏。
- **备份**：定期备份 PostgreSQL（论文与解析内容是事实源，索引可从它重建）。OpenSearch 索引可重建，优先级低于 PostgreSQL。

---

## 15.7 分支与发布流程建议

- **分支**：`main` 保护分支；功能走 feature 分支 + PR + CI 通过 + Review。
- **PR 前**：本地跑 `make lint && make test`（呼应工程规范：开 PR 前本地过 CI 检查）。
- **发布**：打语义化版本 tag（`v1.2.0`），CI 据 tag 构建并推送带版本的镜像；保留每个版本镜像以便回滚（第 [12](12-run-build-deploy-rollback.md) 章）。
- **变更记录**：维护 CHANGELOG，记录破坏性变更（尤其是索引映射/数据库 schema 变更，因为它们影响数据迁移）。

---

## 15.8 长期维护检查表（季度）

1. 依赖升级 + 安全扫描 + 回归测试。
2. 基础镜像 tag 更新 + 兼容性验证。
3. 复查并清偿技术债（第 [16](16-upstream-differences-and-fixes.md) 章清单）：收紧 mypy、补测试、统一模型配置、接入 Alembic。
4. 复查安全加固清单（第 [14](14-quality-performance-security.md) 章）：轮换密钥、复查暴露面。
5. 评估检索质量（用 Langfuse 数据 + 人工抽检），按需调 HNSW/RRF/分块参数。
6. 备份演练（确认能从备份恢复）。

下一章 [`16-upstream-differences-and-fixes.md`](16-upstream-differences-and-fixes.md) 是附录：上游差异与全部修复清单、源文件覆盖清单、以及所有包 `__init__.py` 的精确内容。
