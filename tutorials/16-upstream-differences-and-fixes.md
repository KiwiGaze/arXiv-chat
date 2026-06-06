# File: tutorials/16-upstream-differences-and-fixes.md

# 第 16 章　附录：上游差异、修复清单、`__init__` 内容与源文件覆盖

本章是教程的"完整性背书"：汇总本教程对上游 `arxiv-paper-curator`（`main` 分支）所做的全部改动、所有包 `__init__.py` 的精确内容、以及"每个源文件由哪一章复现"的覆盖清单。配合前面各章，你可以**逐文件**还原整个项目。

---

## A. 六处"让上游可运行"的修复（运行时崩溃点）

本教程的硬性要求是"所有代码必须完整、可运行"。审计 `main` 时发现以下 6 处**已提交但会在运行时崩溃**的缺陷。每处都用 `⚠️ 上游修复(#n)` 在正文标出，原则是"**只修崩溃点，不做额外改进**"。

### 修复 #1 —— `OllamaClient.get_langchain_model` 缺失（第 [08](08-week5-rag-llm.md) 章）

- **症状**：调用 `/ask-agentic` 时 `AttributeError: 'OllamaClient' object has no attribute 'get_langchain_model'`。
- **原因**：Week 7 的全部智能体节点（guardrail/grade/rewrite/generate）调用 `runtime.context.ollama_client.get_langchain_model(...)`，但该方法在 `OllamaClient` 中从未实现（git 历史里也从未存在）。
- **修复**：在 `OllamaClient` 增加：
  ```python
  def get_langchain_model(self, model: str, temperature: float = 0.0):
      from langchain_ollama import ChatOllama
      return ChatOllama(base_url=self.base_url, model=model, temperature=temperature)
  ```
  `langchain-ollama` 已在 `pyproject.toml` 依赖中；`ChatOllama` 支持节点所需的 `.with_structured_output()` 与 `await .ainvoke()`（已在 `.venv` 核对其构造参数与方法面）。

### 修复 #2 —— `LangfuseTracer` 缺 3 个方法（第 [09](09-week6-monitoring-caching.md) 章）

- **症状**：调用 `/ask`、`/stream`、`/ask-agentic` 时 `AttributeError`（`trace_rag_request` / `create_span` / `end_span`）。
- **原因**：`RAGTracer`（Week 6）与智能体节点（Week 7）调用这三个方法，但 Week 7 把 `LangfuseTracer` 重构为 v3 后删除了它们，未更新调用方。
- **修复**：在 `LangfuseTracer` 末尾补回 `trace_rag_request`（上下文管理器）、`create_span`、`end_span`，**未启用 Langfuse 时返回 None / no-op，启用时全程 try/except**，保证追踪绝不中断主流程。

### 修复 #3 —— `TextChunker._reconstruct_text` 参数数不符（第 [07](07-week4-hybrid-search.md) 章）

- **症状**：当文本词数 < `min_chunk_size` 时 `TypeError`（传了 2 个参数给只接收 1 个的方法）。
- **原因**：`chunk_text` 里 `self._reconstruct_text(words, text)`，而定义是 `def _reconstruct_text(self, words)`。
- **修复**：改调用处为 `self._reconstruct_text(words)`。

### 修复 #4 —— 智能体测试命名不一致 + 缺失夹具（第 [11](11-testing.md) 章）

- **症状**：智能体测试无法运行（`fixture 'test_context' not found`；mock 的方法名也不对）。
- **原因**：(1) 测试 mock `create_llm`，节点实际调用 `get_langchain_model`；(2) `test_context`、`sample_*` 夹具在任何 `conftest.py` 中都未定义。
- **修复**：把测试里的 `create_llm` 改为 `get_langchain_model`；新建 `tests/unit/services/agents/conftest.py` 提供缺失夹具（第 [11](11-testing.md) 章已给出完整代码）。

### 修复 #5 —— `get_agentic_rag_service` 传非法 `model=`（第 [10](10-week7-agentic-telegram.md) 章）

- **症状**：每次 `/ask-agentic` 请求 `TypeError: make_agentic_rag_service() got an unexpected keyword argument 'model'`。
- **原因**：`dependencies.py` 调 `make_agentic_rag_service(..., model=settings.ollama_model)`，但该工厂没有 `model` 形参。
- **修复**：删去 `model=settings.ollama_model` 这一参数（智能体使用 `GraphConfig` 默认模型 `gemma4:e2b`）。
- **替代方案**：若要让智能体跟随 `OLLAMA_MODEL`，给工厂加 `model: Optional[str] = None` 形参并传入 `GraphConfig(model=model)`，同时保留 `dependencies.py` 原样。

### 修复 #6 —— `PDFParserService.parse_pdf` 误用 `await`（第 [05](05-week2-ingestion.md) 章）

- **症状**：任何 PDF 解析（Week 2 摄取主路径）抛 `TypeError: object ... can't be used in 'await' expression`；`test_parse_pdf_success` 也失败。
- **原因**：近期提交 "remove unnecessary async from parse_pdf" 把 `DoclingParser.parse_pdf` 改为**同步**方法，但 `PDFParserService.parse_pdf` 仍 `result = await self.docling_parser.parse_pdf(pdf_path)`——`await` 一个同步调用的返回值（`PdfContent`/`None`）会崩。
- **修复**：去掉那个 `await`，改为 `result = self.docling_parser.parse_pdf(pdf_path)`。

---

## B. 死代码 / 不完整脚手架（不影响运行，但需处理）

### B.1 `schemas/common` 与 `schemas/telegram` 的 `__init__` 引用了不存在的模块

上游这两个 `__init__.py` 从以下**不存在的模块**导入：

- `src/schemas/common/__init__.py` → `from src.schemas.search.hybrid import ChunkResult, HybridSearchRequest, HybridSearchResponse`（**`src/schemas/search/` 目录不存在**）。
- `src/schemas/telegram/__init__.py` → `from .commands import TelegramCommand` 等（**`commands.py`/`messages.py`/`user_settings.py` 不存在**）。

**这两个包从未被运行代码导入**（Telegram 机器人用的是 `src.schemas.api.ask` / `src.schemas.api.search`），所以它们不影响应用运行；但若有人 `import src.schemas.common`，就会 `ModuleNotFoundError`。

**本教程处置**：把这两个 `__init__.py` 创建为**空文件**（包标记），避免误导与潜在导入错误：

```bash
mkdir -p src/schemas/common src/schemas/telegram
: > src/schemas/common/__init__.py     # 空文件
: > src/schemas/telegram/__init__.py   # 空文件
```

### B.2 `src/services/cache/` 没有 `__init__.py`

上游 `cache/` 目录无 `__init__.py`，作为命名空间包工作，导入 `src.services.cache.client` / `.factory` 正常。本教程保持一致，**不创建** `cache/__init__.py`。

---

## C. 非崩溃的小瑕疵（本教程逐字保留并标注）

| 项 | 位置 | 说明 | 处置 |
|----|------|------|------|
| `make health` 路径 | `Makefile` | 用 `http://localhost:8000/health`，真实端点在 `/api/v1/health` | 保留；第 [12](12-run-build-deploy-rollback.md) 章提醒用正确路径 |
| `'\\n\\n'` 字面量 | `TextChunker._create_combined_chunk` | 合并小章节时分隔符是字面 `\n\n`（反斜杠+n）而非换行 | 逐字保留（罕见分支、非崩溃，未纳入修复集） |
| `mypy ignore_errors=true` | `pyproject.toml` | 全局忽略类型错误，仅做导入级检查 | 保留；第 [14](14-quality-performance-security.md)/[15](15-cicd-and-maintenance.md) 章建议逐步收紧 |
| `update_span` 后再 `span.end()` | `RAGTracer.trace_embedding` | `update_span` 已 end，外层又 end 一次（启用 Langfuse 时双 end，通常无害） | 逐字保留 |

---

## D. 所有包 `__init__.py` 的精确内容

为便于**逐文件精确还原**，下面列出全部 `__init__.py`。前面各章为保证"按周可独立运行"，对部分 re-export 型 `__init__` 暂用空文件——若你想与上游仓库**完全一致**，用下面的内容覆盖即可（功能上空文件也能运行，因为代码都用子模块路径导入）。

### D.1 空文件（包标记，内容为空）

```
src/__init__.py
src/db/__init__.py
src/db/interfaces/__init__.py
src/services/__init__.py
src/services/arxiv/__init__.py
src/services/pdf_parser/__init__.py
src/services/opensearch/__init__.py
src/services/embeddings/__init__.py
src/services/indexing/__init__.py
src/services/langfuse/__init__.py
src/schemas/api/__init__.py
src/schemas/arxiv/__init__.py
src/schemas/database/__init__.py
src/schemas/pdf_parser/__init__.py
src/schemas/embeddings/__init__.py
src/schemas/indexing/__init__.py
src/schemas/common/__init__.py     # 见 B.1：上游有内容但引用不存在模块，本教程置空
src/schemas/telegram/__init__.py   # 见 B.1：同上，置空
tests/__init__.py
tests/api/__init__.py
tests/api/routers/__init__.py
tests/integration/__init__.py
tests/unit/__init__.py
tests/unit/schemas/__init__.py
tests/unit/services/__init__.py
tests/unit/services/agents/__init__.py
```

> `src/services/cache/` 无 `__init__.py`（命名空间包，见 B.2）。

### D.2 `src/models/__init__.py`（逐字复制）

```python
from .paper import Paper

__all__ = [
    "Paper",
]
```

### D.3 `src/repositories/__init__.py`（逐字复制）

```python
from .paper import PaperRepository

__all__ = [
    "PaperRepository",
]
```

### D.4 `src/schemas/__init__.py`（逐字复制）

```python
from .api.health import HealthResponse
from .api.search import SearchHit, SearchRequest, SearchResponse
from .arxiv.paper import ArxivPaper, PaperCreate, PaperResponse, PaperSearchResponse
from .pdf_parser.models import PaperFigure, PaperSection, PaperTable, ParsedPaper, ParserType

__all__ = [
    "HealthResponse",
    "SearchRequest",
    "SearchHit",
    "SearchResponse",
    "ArxivPaper",
    "PaperCreate",
    "PaperResponse",
    "PaperSearchResponse",
    "ParsedPaper",
    "PaperSection",
    "PaperFigure",
    "PaperTable",
    "ParserType",
]
```

### D.5 `src/routers/__init__.py`（逐字复制，第 [09](09-week6-monitoring-caching.md) 章已给）

```python
"""Router modules for the RAG API."""

# Import all available routers
from . import ask, hybrid_search, ping

__all__ = ["ask", "ping", "hybrid_search"]
```

### D.6 `src/services/ollama/__init__.py`（逐字复制，第 [08](08-week5-rag-llm.md) 章已给）

```python
from .client import OllamaClient

__all__ = ["OllamaClient"]
```

### D.7 `src/services/agents/__init__.py`（逐字复制，第 [10](10-week7-agentic-telegram.md) 章已给）

```python
from .agentic_rag import AgenticRAGService
from .config import GraphConfig
from .context import Context
from .factory import make_agentic_rag_service
from .state import AgentState

__all__ = [
    "AgenticRAGService",
    "GraphConfig",
    "Context",
    "AgentState",
    "make_agentic_rag_service",
]
```

### D.8 `src/services/agents/nodes/__init__.py`（逐字复制，第 [10](10-week7-agentic-telegram.md) 章已给）

```python
from .generate_answer_node import ainvoke_generate_answer_step
from .grade_documents_node import ainvoke_grade_documents_step
from .guardrail_node import ainvoke_guardrail_step, continue_after_guardrail
from .out_of_scope_node import ainvoke_out_of_scope_step
from .retrieve_node import ainvoke_retrieve_step
from .rewrite_query_node import ainvoke_rewrite_query_step

__all__ = [
    "ainvoke_guardrail_step",
    "continue_after_guardrail",
    "ainvoke_out_of_scope_step",
    "ainvoke_retrieve_step",
    "ainvoke_grade_documents_step",
    "ainvoke_rewrite_query_step",
    "ainvoke_generate_answer_step",
]
```

### D.9 `src/services/telegram/__init__.py`（逐字复制，第 [10](10-week7-agentic-telegram.md) 章已给）

```python
from .bot import TelegramBot
from .factory import make_telegram_service

__all__ = ["TelegramBot", "make_telegram_service"]
```

---

## E. 源文件 → 教程章节覆盖清单

下表把仓库里每个**应用/配置/管道/测试文件**映射到复现它的章节，确保无遗漏。

### E.1 根目录与配置

| 源文件 | 章节 |
|--------|------|
| `pyproject.toml` | [02](02-environment-and-dependencies.md) |
| `.env.example` / `.env.test` | [02](02-environment-and-dependencies.md) |
| `.gitignore` | [02](02-environment-and-dependencies.md) |
| `.pre-commit-config.yaml` | [15](15-cicd-and-maintenance.md) |
| `Dockerfile` | [04](04-week1-infrastructure.md) |
| `compose.yml` | [04](04-week1-infrastructure.md)（完整）、[09](09-week6-monitoring-caching.md)（Langfuse 部分） |
| `Makefile` | [12](12-run-build-deploy-rollback.md) |
| `gradio_launcher.py` | [08](08-week5-rag-llm.md) |

### E.2 `src/` 核心

| 源文件 | 章节 |
|--------|------|
| `src/config.py` | [04](04-week1-infrastructure.md) |
| `src/main.py` | [04](04-week1-infrastructure.md)（引导版）→ [10](10-week7-agentic-telegram.md)（最终版） |
| `src/dependencies.py` | [04](04-week1-infrastructure.md)→各周增量→ [10](10-week7-agentic-telegram.md)（最终版，含修复 #5） |
| `src/database.py` | [04](04-week1-infrastructure.md) |
| `src/exceptions.py` | [04](04-week1-infrastructure.md) |
| `src/middlewares.py` | [04](04-week1-infrastructure.md) |
| `src/gradio_app.py` | [08](08-week5-rag-llm.md) |
| `src/db/factory.py` / `interfaces/base.py` / `interfaces/postgresql.py` | [04](04-week1-infrastructure.md) |
| `src/models/paper.py` | [04](04-week1-infrastructure.md) |
| `src/repositories/paper.py` | [05](05-week2-ingestion.md) |

### E.3 `src/routers/`

| 源文件 | 章节 |
|--------|------|
| `routers/ping.py` | [04](04-week1-infrastructure.md)（引导版）→ [08](08-week5-rag-llm.md)（最终版） |
| `routers/hybrid_search.py` | [07](07-week4-hybrid-search.md) |
| `routers/ask.py` | [09](09-week6-monitoring-caching.md) |
| `routers/agentic_ask.py` | [10](10-week7-agentic-telegram.md) |

### E.4 `src/schemas/`

| 源文件 | 章节 |
|--------|------|
| `schemas/database/config.py` | [04](04-week1-infrastructure.md) |
| `schemas/api/health.py` | [04](04-week1-infrastructure.md) |
| `schemas/arxiv/paper.py` | [05](05-week2-ingestion.md) |
| `schemas/pdf_parser/models.py` | [05](05-week2-ingestion.md) |
| `schemas/api/search.py` | [06](06-week3-opensearch-bm25.md) |
| `schemas/indexing/models.py` | [07](07-week4-hybrid-search.md) |
| `schemas/embeddings/jina.py` | [07](07-week4-hybrid-search.md) |
| `schemas/ollama.py` | [08](08-week5-rag-llm.md) |
| `schemas/api/ask.py` | [08](08-week5-rag-llm.md) |
| `schemas/common/__init__.py` / `schemas/telegram/__init__.py` | 本章 B.1（置空） |

### E.5 `src/services/`

| 源文件 | 章节 |
|--------|------|
| `services/arxiv/client.py` / `factory.py` | [05](05-week2-ingestion.md) |
| `services/pdf_parser/docling.py` / `parser.py` / `factory.py` | [05](05-week2-ingestion.md) |
| `services/metadata_fetcher.py` | [05](05-week2-ingestion.md) |
| `services/opensearch/index_config_hybrid.py` / `query_builder.py` / `client.py` / `factory.py` | [06](06-week3-opensearch-bm25.md) |
| `services/embeddings/jina_client.py` / `factory.py` | [07](07-week4-hybrid-search.md) |
| `services/indexing/text_chunker.py`（修复 #3）/ `hybrid_indexer.py` / `factory.py` | [07](07-week4-hybrid-search.md) |
| `services/ollama/client.py`（修复 #1）/ `prompts.py` / `factory.py` / `prompts/rag_system.txt` | [08](08-week5-rag-llm.md) |
| `services/langfuse/client.py`（修复 #2）/ `tracer.py` / `factory.py` | [09](09-week6-monitoring-caching.md) |
| `services/cache/client.py` / `factory.py` | [09](09-week6-monitoring-caching.md) |
| `services/agents/*`（models/state/context/config/prompts/tools/nodes/agentic_rag/factory） | [10](10-week7-agentic-telegram.md) |
| `services/telegram/bot.py` / `factory.py` | [10](10-week7-agentic-telegram.md) |

### E.6 `airflow/`

| 源文件 | 章节 |
|--------|------|
| `airflow/Dockerfile` / `entrypoint.sh` / `requirements-airflow.txt` | [05](05-week2-ingestion.md) |
| `airflow/dags/arxiv_paper_ingestion.py` | [05](05-week2-ingestion.md) |
| `airflow/dags/arxiv_ingestion/common.py` / `setup.py` / `fetching.py` / `indexing.py` / `reporting.py` | [05](05-week2-ingestion.md) |

### E.7 `tests/`

| 源文件 | 章节 |
|--------|------|
| `tests/conftest.py` / `tests/api/conftest.py` | [11](11-testing.md) |
| `tests/api/routers/test_ping.py` / `test_ask.py` / `test_hybrid_search.py` | [11](11-testing.md)（逐字） |
| `tests/api/routers/test_agentic_ask.py` | [11](11-testing.md)（清单） |
| `tests/integration/test_services.py` | [11](11-testing.md)（逐字） |
| `tests/unit/test_config.py` / `schemas/test_search.py` / `services/test_opensearch_query_builder.py` | [11](11-testing.md)（逐字） |
| `tests/unit/services/test_arxiv_client.py` / `test_pdf_parser.py` / `test_metadata_fetcher.py` / `test_telegram.py` | [11](11-testing.md)（清单 + 模式说明） |
| `tests/unit/services/agents/test_models.py` / `test_tools.py` / `test_nodes.py` / `test_agentic_rag.py` | [11](11-testing.md)（清单 + 修复 #4 + 夹具） |
| `tests/unit/services/agents/conftest.py` | [11](11-testing.md)（本教程补充，使智能体测试可运行） |

> **说明**：`tests/` 下**全部测试文件均在第 [11](11-testing.md) 章逐字复现**（`test_nodes.py` 含修复 #4 的命名改动）。唯一的新增文件是 `tests/unit/services/agents/conftest.py`（上游缺失，本教程补充以提供智能体测试夹具，使其可运行）。应用层、配置、管道、测试的全部代码均为**逐字复现**（仅 6 处崩溃点按 A 节修复）。

### E.8 静态资源与笔记本

| 源 | 处置 |
|----|------|
| `static/*.png` / `*.gif` | 架构示意图，非代码；可选，不影响运行 |
| `notebooks/week1..7/*.ipynb` + `README.md` | 上游交互式学习材料；本教程已把每步**可运行操作写进正文**，不复刻 `.ipynb`（见第 [03](03-architecture-and-design.md) 章说明） |

---

## F. 必备内容 → 章节映射（按需求核对）

| 需求项 | 章节 |
|--------|------|
| 项目概览 | [01](01-project-overview.md) |
| 系统与环境需求 | [02](02-environment-and-dependencies.md) |
| 依赖版本与安装步骤 | [02](02-environment-and-dependencies.md) |
| 项目目录结构说明 | [03](03-architecture-and-design.md)（+ 各周顶部） |
| 架构设计与技术选型理由 | [03](03-architecture-and-design.md)（+ 各周决策框） |
| 从零开始的完整实现步骤 | [04](04-week1-infrastructure.md)–[10](10-week7-agentic-telegram.md) |
| 可直接复制运行的完整代码 | [04](04-week1-infrastructure.md)–[10](10-week7-agentic-telegram.md) |
| 关键配置文件与参数说明 | [02](02-environment-and-dependencies.md)、[03](03-architecture-and-design.md)、[04](04-week1-infrastructure.md) 及各周 |
| 测试用例、运行命令和验证方法 | [11](11-testing.md) |
| 本地运行、构建、部署和回滚步骤 | [12](12-run-build-deploy-rollback.md) |
| 常见问题排查 | [13](13-troubleshooting.md) |
| 性能、安全性、可维护性分析 | [14](14-quality-performance-security.md) |
| CI/CD 与长期维护建议 | [15](15-cicd-and-maintenance.md) |
| 设计决策（为什么/替代/优缺点/影响/风险） | [03](03-architecture-and-design.md) 集中 + 各周决策框 |

---

## G. 一句话收尾

照着第 [01](01-project-overview.md)→[16](16-upstream-differences-and-fixes.md) 顺序，从空目录逐文件创建本教程给出的代码（应用层与测试全部逐字复现、6 处上游崩溃点已修复并标注），你就能得到一个**完整、可运行、生产风格**的 Agentic RAG 系统。祝构建顺利。🎉
