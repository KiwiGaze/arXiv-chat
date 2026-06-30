# File: tutorials/11-testing.md

# 第 11 章　测试：用例、运行命令与验证方法

本章覆盖测试套件的组织、可直接运行的测试用例、运行命令、覆盖率与验证方法。

> ⚠️ **本章含上游修复 #4 的说明**：智能体单元测试（`tests/unit/services/agents/*`）在上游存在两个问题：(1) mock 的方法名是 `create_llm`，而节点实际调用的是 `get_langchain_model`；(2) 它们使用的 `test_context`、`sample_*` 等 fixture **在仓库任何 `conftest.py` 中都没有定义**。因此这些测试**上游无法运行**。本章给出可运行的测试为主，并说明如何让智能体测试跑起来。

---

## 11.1 测试技术栈与配置

测试用 `pytest` + `pytest-asyncio`（异步）+ `asgi-lifespan` + `httpx`（API 测试）。配置在 `pyproject.toml`（第 [02](02-environment-and-dependencies.md) 章已给出）：

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"                         # async def 测试自动识别，无需逐个加 @pytest.mark.asyncio
asyncio_default_fixture_loop_scope = "function"
env_files = ".env.test"                       # 测试用独立环境变量
```

- **`asyncio_mode = "auto"`**：所有 `async def test_*` 自动当作异步测试运行。
- **`env_files = ".env.test"`**：测试读取 `.env.test`（第 [02](02-environment-and-dependencies.md) 章），与开发配置隔离。

---

## 11.2 测试目录结构

```
tests/
├── __init__.py
├── conftest.py                 # 顶层（仅注释）
├── api/
│   ├── __init__.py
│   ├── conftest.py             # API 测试夹具：mock 全部外部服务的 client
│   └── routers/
│       ├── test_ping.py
│       ├── test_ask.py
│       ├── test_hybrid_search.py
│       └── test_agentic_ask.py
├── integration/
│   ├── __init__.py
│   └── test_services.py        # 需要真实服务（arXiv/OpenSearch）
└── unit/
    ├── __init__.py
    ├── test_config.py
    ├── schemas/
    │   ├── __init__.py
    │   └── test_search.py
    └── services/
        ├── test_arxiv_client.py
        ├── test_metadata_fetcher.py
        ├── test_opensearch_query_builder.py
        ├── test_pdf_parser.py
        ├── test_telegram.py
        └── agents/
            ├── __init__.py
            ├── test_agentic_rag.py
            ├── test_models.py
            ├── test_nodes.py
            └── test_tools.py
```

> 每个测试子目录都需要 `__init__.py`（包标记，内容为空）。请相应创建。

---

## 11.3 测试夹具

### 文件：`tests/conftest.py`（逐字复制）

```python
# Test configuration and shared fixtures
```

> 顶层 conftest 目前仅是占位注释。

### 文件：`tests/api/conftest.py`（逐字复制）

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from src.main import app


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Async backend for testing."""
    return "asyncio"


@pytest.fixture
async def client():
    """HTTP client for API testing with mocked services."""
    # Mock database startup and session to prevent real connections
    with (
        patch("src.db.interfaces.postgresql.PostgreSQLDatabase.startup") as mock_startup,
        patch("src.db.interfaces.postgresql.PostgreSQLDatabase.get_session") as mock_get_session,
        patch("src.services.opensearch.factory.make_opensearch_client") as mock_os,
        patch("src.services.arxiv.factory.make_arxiv_client") as mock_arxiv,
        patch("src.services.pdf_parser.factory.make_pdf_parser_service") as mock_pdf,
        patch("src.services.ollama.client.OllamaClient") as mock_ollama,
        patch("src.repositories.paper.PaperRepository.get_by_arxiv_id") as mock_get_by_id,
    ):
        # Mock startup to do nothing
        mock_startup.return_value = None

        # Mock get_session to return a mock session
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_get_session.return_value.__exit__.return_value = None

        # Mock repository methods to return None (not found) by default
        mock_get_by_id.return_value = None

        # Set up other mock return values
        mock_os.return_value = AsyncMock()
        mock_arxiv.return_value = AsyncMock()
        mock_pdf.return_value = AsyncMock()
        mock_ollama.return_value = AsyncMock()

        async with LifespanManager(app) as manager:
            async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
                yield client
```

### 夹具设计要点

- **`client` 夹具 mock 掉全部外部服务**：用 `patch` 替换数据库启动、会话、OpenSearch/arXiv/PDF/Ollama 工厂。这样 API 测试**不需要任何真实服务**就能跑（快、可在 CI 离线运行）。
- **`LifespanManager(app)`**：手动驱动 FastAPI 的 lifespan（启动/关闭），但因为外部服务被 mock，不会真的连数据库/OpenSearch。
- **`ASGITransport`**：让 `httpx` 直接打 ASGI app，无需起真实服务器。

> **为什么 API 测试用 mock 而不连真实服务？**
> - **为什么这么选**：单测/CI 要快、要确定、要离线可跑。
> - **替代方案**：用 `testcontainers`（项目 dev 依赖已含）起真实容器做集成测试。
> - **影响**：mock 测试快但只验证"接线"和契约，不验证真实检索质量；两者互补——集成测试见 11.7。

---

## 11.4 API 端点测试

### 文件：`tests/api/routers/test_ping.py`（逐字复制）

```python
import pytest


async def test_health_check(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service_name" in data
    assert "version" in data
    assert "services" in data
```

### 文件：`tests/api/routers/test_ask.py`（逐字复制）

```python
import pytest


async def test_ask_endpoint_basic(client):
    response = await client.post("/api/v1/ask", json={"query": "What is machine learning?", "model": "llama3.2:3b"})

    assert response.status_code in [200, 500, 503]

    if response.status_code == 200:
        data = response.json()

        assert "query" in data
        assert "answer" in data
        assert "sources" in data
        assert "chunks_used" in data
        assert "search_mode" in data

        assert data["query"] == "What is machine learning?"
        assert isinstance(data["sources"], list)
        assert isinstance(data["chunks_used"], int)


async def test_ask_endpoint_with_hybrid_search(client):
    response = await client.post(
        "/api/v1/ask", json={"query": "neural networks", "model": "llama3.2:3b", "use_hybrid": True, "top_k": 5}
    )

    assert response.status_code in [200, 500, 503]

    if response.status_code == 200:
        data = response.json()
        assert data["query"] == "neural networks"


async def test_ask_endpoint_with_categories(client):
    response = await client.post(
        "/api/v1/ask", json={"query": "computer vision", "model": "llama3.2:3b", "categories": ["cs.CV", "cs.AI"], "top_k": 3}
    )

    assert response.status_code in [200, 500, 503]


async def test_ask_endpoint_validation_errors(client):
    response = await client.post("/api/v1/ask", json={"query": "", "model": "llama3.2:3b"})
    assert response.status_code == 422

    response = await client.post("/api/v1/ask", json={"model": "llama3.2:3b"})
    assert response.status_code == 422

    response = await client.post("/api/v1/ask", json={"query": "test", "model": "llama3.2:3b", "top_k": 0})
    assert response.status_code == 422


async def test_stream_endpoint_basic(client):
    response = await client.post("/api/v1/stream", json={"query": "What is deep learning?", "model": "llama3.2:3b"})

    assert response.status_code in [200, 500, 503]

    if response.status_code == 200:
        assert "text/plain" in response.headers.get("content-type", "")


async def test_stream_endpoint_validation_errors(client):
    response = await client.post("/api/v1/stream", json={"query": "", "model": "llama3.2:3b"})
    assert response.status_code == 422
```

### 文件：`tests/api/routers/test_hybrid_search.py`（逐字复制）

```python
import pytest


async def test_search_endpoint_basic(client):
    response = await client.post("/api/v1/hybrid-search/", json={"query": "neural networks", "size": 5})

    assert response.status_code == 200
    data = response.json()

    assert "query" in data
    assert "total" in data
    assert "hits" in data
    assert "size" in data
    assert "from" in data

    assert data["query"] == "neural networks"
    assert isinstance(data["total"], int)
    assert isinstance(data["hits"], list)


async def test_search_endpoint_with_latest_papers(client):
    response = await client.post(
        "/api/v1/hybrid-search/", json={"query": "machine learning", "size": 3, "latest_papers": True, "use_hybrid": False}
    )

    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "machine learning"


async def test_search_endpoint_with_categories(client):
    response = await client.post(
        "/api/v1/hybrid-search/",
        json={"query": "deep learning", "size": 5, "categories": ["cs.AI", "cs.LG"], "latest_papers": False, "use_hybrid": False},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "deep learning"


async def test_search_endpoint_validation_errors(client):
    response = await client.post("/api/v1/hybrid-search/", json={"query": ""})
    assert response.status_code == 422

    response = await client.post("/api/v1/hybrid-search/", json={"query": "test", "size": 0})
    assert response.status_code == 422

    response = await client.post("/api/v1/hybrid-search/", json={"size": 10})
    assert response.status_code == 422


async def test_search_endpoint_pagination(client):
    response = await client.post("/api/v1/hybrid-search/", json={"query": "artificial intelligence", "size": 5, "from": 10})

    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "artificial intelligence"


async def test_search_endpoint_all_parameters(client):
    response = await client.post(
        "/api/v1/hybrid-search/",
        json={
            "query": "transformers attention mechanism",
            "size": 8,
            "from": 5,
            "categories": ["cs.AI"],
            "latest_papers": True,
            "use_hybrid": False,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "transformers attention mechanism"
    assert isinstance(data["total"], int)
    assert isinstance(data["hits"], list)

    for hit in data["hits"]:
        assert "arxiv_id" in hit
        assert "title" in hit
        assert "score" in hit
```

> API 测试的典型模式：**断言契约而非具体内容**。例如 `assert response.status_code in [200, 500, 503]`——因为外部服务被 mock，实际可能返回多种状态；测试关注"结构正确、校验生效"，而不是答案内容。校验错误（空 query、超界 size）必须返回 422。

---

## 11.5 单元测试（纯逻辑，无需服务）

### 文件：`tests/unit/services/test_opensearch_query_builder.py`（逐字复制）

```python
import pytest
from src.services.opensearch.query_builder import QueryBuilder


def test_query_builder_basic_query():
    builder = QueryBuilder(query="machine learning", size=5)

    query = builder.build()

    assert query["size"] == 5
    assert query["from"] == 0
    assert query["track_total_hits"] is True

    bool_query = query["query"]["bool"]
    assert len(bool_query["must"]) == 1

    multi_match = bool_query["must"][0]["multi_match"]
    assert multi_match["query"] == "machine learning"
    assert "title^3" in multi_match["fields"]
    assert "abstract^2" in multi_match["fields"]
    assert "authors^1" in multi_match["fields"]


def test_query_builder_with_categories():
    builder = QueryBuilder(query="deep learning", categories=["cs.AI", "cs.LG"])

    query = builder.build()

    bool_query = query["query"]["bool"]
    assert "filter" in bool_query

    filters = bool_query["filter"]
    assert len(filters) == 1
    assert filters[0]["terms"]["categories"] == ["cs.AI", "cs.LG"]


def test_query_builder_latest_papers_sorting():
    builder = QueryBuilder(query="neural networks", latest_papers=True)

    query = builder.build()

    assert "sort" in query
    sort_config = query["sort"]
    assert len(sort_config) == 2
    assert sort_config[0]["published_date"]["order"] == "desc"
    assert sort_config[1] == "_score"


def test_query_builder_relevance_sorting():
    builder = QueryBuilder(query="transformers attention", latest_papers=False)

    query = builder.build()

    assert "sort" not in query


def test_query_builder_empty_query_sorting():
    builder = QueryBuilder(query="", latest_papers=False)

    query = builder.build()

    assert "sort" in query
    sort_config = query["sort"]
    assert sort_config[0]["published_date"]["order"] == "desc"


def test_query_builder_highlighting():
    builder = QueryBuilder(query="test query")

    query = builder.build()

    highlight = query["highlight"]
    assert "fields" in highlight

    fields = highlight["fields"]
    assert "title" in fields
    assert "abstract" in fields
    assert "authors" in fields

    assert fields["title"]["fragment_size"] == 0
    assert fields["abstract"]["fragment_size"] == 150
    assert fields["authors"]["pre_tags"] == ["<mark>"]


def test_query_builder_source_fields():
    builder = QueryBuilder(query="test query")

    query = builder.build()

    source_fields = query["_source"]
    expected_fields = ["arxiv_id", "title", "authors", "abstract", "categories", "published_date", "pdf_url"]

    for field in expected_fields:
        assert field in source_fields


def test_query_builder_custom_fields():
    custom_fields = ["title^5", "abstract^1"]
    builder = QueryBuilder(query="test", fields=custom_fields)

    query = builder.build()

    multi_match = query["query"]["bool"]["must"][0]["multi_match"]
    assert multi_match["fields"] == custom_fields
```

### 文件：`tests/unit/test_config.py`（逐字复制）

```python
import os

import pytest
from src.config import Settings


def test_settings_initialization():
    """Test settings can be initialized."""
    settings = Settings()

    assert settings.app_version == "0.1.0"
    assert settings.debug is True
    assert settings.environment == "development"
    assert settings.service_name == "rag-api"


def test_settings_postgres_defaults():
    """Test PostgreSQL default configuration."""
    settings = Settings()

    assert "postgresql://" in settings.postgres_database_url
    assert settings.postgres_echo_sql is False
    assert settings.postgres_pool_size == 20
    assert settings.postgres_max_overflow == 0


def test_settings_opensearch_defaults():
    """Test OpenSearch default configuration."""
    settings = Settings()

    assert settings.opensearch.host == "http://localhost:9200"
    assert settings.opensearch.index_name == "arxiv-papers"


def test_settings_ollama_defaults():
    """Test Ollama default configuration."""
    settings = Settings()

    # In Docker environment, this should be ollama service host
    expected_host = "http://ollama:11434" if "OLLAMA_HOST" not in os.environ else settings.ollama_host
    assert settings.ollama_host in ["http://localhost:11434", "http://ollama:11434"]
```

### 文件：`tests/unit/schemas/test_search.py`（逐字复制）

```python
import pytest
from pydantic import ValidationError
from src.schemas.api.search import SearchHit, SearchRequest, SearchResponse


def test_search_request_valid():
    """Test valid SearchRequest creation."""
    request = SearchRequest(query="neural networks", size=10, latest_papers=True, categories=["cs.AI", "cs.LG"])

    assert request.query == "neural networks"
    assert request.size == 10
    assert request.from_ == 0  # Default value
    assert request.latest_papers is True
    assert request.categories == ["cs.AI", "cs.LG"]


def test_search_request_defaults():
    """Test SearchRequest with default values."""
    request = SearchRequest(query="test query")

    assert request.query == "test query"
    assert request.size == 10
    assert request.from_ == 0
    assert request.latest_papers is False
    assert request.categories is None


def test_search_request_validation_errors():
    """Test SearchRequest validation errors."""

    # Empty query should fail
    with pytest.raises(ValidationError):
        SearchRequest(query="")

    # Query too long should fail
    with pytest.raises(ValidationError):
        SearchRequest(query="a" * 501)

    # Invalid size should fail
    with pytest.raises(ValidationError):
        SearchRequest(query="test", size=0)

    with pytest.raises(ValidationError):
        SearchRequest(query="test", size=51)

    # Invalid from_ gets coerced to 0 due to ge=0 constraint
    request = SearchRequest(query="test", from_=-1)
    assert request.from_ == 0  # Pydantic coerces negative values to minimum


def test_search_hit_creation():
    """Test SearchHit creation."""
    hit = SearchHit(
        arxiv_id="2024.12345v1",
        title="Test Paper",
        authors="John Doe, Jane Smith",
        abstract="This is a test paper about machine learning.",
        published_date="2024-01-01T00:00:00Z",
        pdf_url="https://arxiv.org/pdf/2024.12345v1.pdf",
        score=1.5,
        highlights={"title": ["<mark>Test</mark> Paper"]},
    )

    assert hit.arxiv_id == "2024.12345v1"
    assert hit.title == "Test Paper"
    assert hit.score == 1.5
    assert hit.highlights == {"title": ["<mark>Test</mark> Paper"]}


def test_search_response_creation():
    """Test SearchResponse creation."""
    hits = [
        SearchHit(
            arxiv_id="2024.12345v1",
            title="Test Paper",
            authors="John Doe",
            abstract="Test abstract",
            published_date="2024-01-01",
            pdf_url="https://test.pdf",
            score=1.0,
        )
    ]

    response = SearchResponse(query="test query", total=1, hits=hits, size=10, **{"from": 0})

    assert response.query == "test query"
    assert response.total == 1
    assert len(response.hits) == 1
    assert response.error is None
```

---

## 11.6 集成测试（需要真实服务）

### 文件：`tests/integration/test_services.py`（逐字复制）

```python
import pytest
from src.config import get_settings
from src.services.arxiv.factory import make_arxiv_client
from src.services.opensearch.factory import make_opensearch_client


async def test_arxiv_client_basic():
    client = make_arxiv_client()

    papers = await client.fetch_papers_with_query("cat:cs.AI", max_results=1)

    assert isinstance(papers, list)


def test_opensearch_client_health():
    client = make_opensearch_client()

    health = client.health_check()
    assert isinstance(health, bool)


def test_settings_loading():
    settings = get_settings()

    assert hasattr(settings, "app_version")
    assert hasattr(settings, "service_name")
    assert hasattr(settings, "environment")
```

> **集成测试会真的连外部服务**：`test_arxiv_client_basic` 真的请求 arXiv API；`test_opensearch_client_health` 需要 OpenSearch 在 `localhost:9200`。运行集成测试前请先 `docker compose up -d opensearch`，并保证能联网访问 arXiv。

---

## 11.7 其余测试文件（全部逐字复现）与修复 #4

下面把其余测试文件**全部逐字给出**（覆盖摄取、PDF 解析、Telegram、Agentic 端点与智能体层）。其覆盖内容一览：

| 文件 | 覆盖内容 |
|------|----------|
| `tests/unit/services/test_arxiv_client.py` | arXiv 客户端：XML 解析、日期过滤、超时/HTTP 错误、按 ID 取、缓存、限速（mock httpx） |
| `tests/unit/services/test_pdf_parser.py` | Docling 校验（空/非 PDF 头/不存在）、解析成功/无结果/异常、工厂缓存（与修复 #6 相关） |
| `tests/unit/services/test_metadata_fetcher.py` | 摄取编排：初始化、空列表、限速（mock 依赖） |
| `tests/unit/services/test_telegram.py` | Telegram 机器人创建、设置、工厂启用/禁用 |
| `tests/api/routers/test_agentic_ask.py` | `/ask-agentic` 端点契约（覆盖依赖、模型参数、错误、推理步骤） |
| `tests/unit/services/agents/test_models.py` | 智能体全部 Pydantic 模型校验（纯逻辑） |
| `tests/unit/services/agents/test_tools.py` | 检索工具：SearchHit→Document、空结果、自定义 top_k、元数据 |
| `tests/unit/services/agents/test_nodes.py` | 5 个决策节点行为（含修复 #4 的命名改动） |
| `tests/unit/services/agents/test_agentic_rag.py` | 图编排服务：初始化、ask()、可视化、错误处理 |

### ⚠️ 智能体测试的两个上游问题与修复 #4

**问题 1（修复 #4 — 命名不一致）**：`test_nodes.py` 等用 `Mock` 配置的是 `create_llm`：

```python
# 上游写法（错误）：节点其实调用的是 get_langchain_model，不是 create_llm
test_context.ollama_client.create_llm = Mock(return_value=mock_llm)
```

而节点代码（第 [10](10-week7-agentic-telegram.md) 章）调用的是 `runtime.context.ollama_client.get_langchain_model(...)`。**修复 #4**：把测试里所有 `create_llm` 改成 `get_langchain_model`：

```python
# 修复后：与节点实际调用的方法名一致
test_context.ollama_client.get_langchain_model = Mock(return_value=mock_llm)
```

**问题 2（缺失夹具）**：`test_nodes.py` 等使用了 `test_context`、`sample_human_message`、`sample_tool_message`、`sample_ai_message` 等 fixture，但**仓库里没有任何 `conftest.py` 定义它们**。因此这些测试在上游会报 `fixture 'test_context' not found`。

要让智能体节点测试真正可运行，需要新建 `tests/unit/services/agents/conftest.py` 提供这些夹具。下面给出一份可运行的实现（基于第 [10](10-week7-agentic-telegram.md) 章的 `Context` 与消息类型）：

### 文件：`tests/unit/services/agents/conftest.py`（本教程补充，使智能体测试可运行）

这份 `conftest.py` 提供智能体测试用到的全部夹具：`test_context`（节点测试）、`sample_*` 消息、以及 `mock_opensearch_client` / `mock_jina_embeddings_client` / `mock_ollama_client`（工具与图编排测试）。

```python
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.services.agents.context import Context


@pytest.fixture
def mock_opensearch_client():
    """Mock OpenSearch client returning two sample chunk hits."""
    client = MagicMock()
    client.search_unified = Mock(
        return_value={
            "total": 2,
            "hits": [
                {
                    "chunk_text": "Transformers are neural network architectures based on self-attention mechanisms.",
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "authors": "Vaswani et al.",
                    "score": 0.95,
                    "section_name": "Introduction",
                },
                {
                    "chunk_text": "BERT uses bidirectional training of transformers for language understanding.",
                    "arxiv_id": "1810.04805",
                    "title": "BERT",
                    "authors": "Devlin et al.",
                    "score": 0.88,
                    "section_name": "Method",
                },
            ],
        }
    )
    return client


@pytest.fixture
def mock_jina_embeddings_client():
    """Mock Jina embeddings client (async embed_query)."""
    client = MagicMock()
    client.embed_query = AsyncMock(return_value=[0.1] * 1024)
    return client


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client."""
    return MagicMock()


@pytest.fixture
def test_context() -> Context:
    """A Context with mocked clients for node unit tests."""
    return Context(
        ollama_client=MagicMock(),
        opensearch_client=MagicMock(),
        embeddings_client=MagicMock(),
        langfuse_tracer=None,
        trace=None,
        langfuse_enabled=False,
        model_name="gemma4:e2b",
        temperature=0.0,
        top_k=3,
        max_retrieval_attempts=2,
        guardrail_threshold=60,
    )


@pytest.fixture
def sample_human_message() -> HumanMessage:
    return HumanMessage(content="What is machine learning?")


@pytest.fixture
def sample_ai_message() -> AIMessage:
    return AIMessage(content="Machine learning is a field of AI.")


@pytest.fixture
def sample_tool_message() -> ToolMessage:
    return ToolMessage(
        content="Transformers are a neural network architecture based on attention.",
        tool_call_id="retrieve_1",
        name="retrieve_papers",
    )
```

> 有了这份 conftest，`langfuse_enabled=False` 让节点跳过所有 span 创建分支（与"禁用 Langfuse"默认运行态一致）；`search_unified` 是同步 `Mock`、`embed_query` 是 `AsyncMock`（与真实客户端的同步/异步一致）。

### 文件：`tests/unit/services/test_arxiv_client.py`（逐字复制）

```python
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import httpx
import pytest
from src.exceptions import ArxivAPIException, ArxivAPITimeoutError, ArxivParseError, PDFDownloadException, PDFDownloadTimeoutError
from src.schemas.arxiv.paper import ArxivPaper
from src.services.arxiv.client import ArxivClient
from src.services.arxiv.factory import make_arxiv_client


class TestArxivClient:
    """Test ArxivClient functionality."""

    @pytest.fixture
    def arxiv_client(self):
        """Create ArxivClient instance for testing."""
        from src.config import ArxivSettings

        settings = ArxivSettings(
            base_url="https://export.arxiv.org/api/query",
            search_category="cs.AI",
            max_results=10,
            rate_limit_delay=0.1,  # Faster for tests
            timeout_seconds=5,
            pdf_cache_dir="/tmp/test_arxiv_cache",
        )
        return ArxivClient(settings)

    @pytest.fixture
    def mock_arxiv_response(self):
        """Mock arXiv API XML response."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2024.0001v1</id>
            <updated>2024-01-01T00:00:00Z</updated>
            <published>2024-01-01T00:00:00Z</published>
            <title>Test Paper Title</title>
            <summary>Test abstract content</summary>
            <author><name>Test Author</name></author>
            <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
            <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
            <link title="pdf" href="http://arxiv.org/pdf/2024.0001v1" rel="alternate" type="application/pdf"/>
          </entry>
        </feed>"""

    def test_factory_creates_client(self):
        """Test that factory creates ArxivClient instance."""
        client = make_arxiv_client()
        assert isinstance(client, ArxivClient)
        assert client.search_category == "cs.AI"
        assert client.max_results == 15  # Default from ArxivSettings

    @pytest.mark.asyncio
    async def test_fetch_papers_success(self, arxiv_client, mock_arxiv_response):
        """Test successful paper fetching."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.text = mock_arxiv_response
            mock_response.raise_for_status.return_value = None

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            papers = await arxiv_client.fetch_papers(max_results=1)

            assert len(papers) == 1
            assert papers[0].arxiv_id == "2024.0001v1"
            assert papers[0].title == "Test Paper Title"
            assert papers[0].abstract == "Test abstract content"
            assert papers[0].authors == ["Test Author"]
            assert papers[0].categories == ["cs.AI"]

    @pytest.mark.asyncio
    async def test_fetch_papers_with_date_filters(self, arxiv_client, mock_arxiv_response):
        """Test paper fetching with date filters."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.text = mock_arxiv_response
            mock_response.raise_for_status.return_value = None

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            papers = await arxiv_client.fetch_papers(max_results=1, from_date="20240101", to_date="20240131")

            assert len(papers) == 1
            # Verify the URL includes date filters
            call_args = mock_client.return_value.__aenter__.return_value.get.call_args[0][0]
            assert "submittedDate:[202401010000+TO+202401312359]" in call_args

    @pytest.mark.asyncio
    async def test_fetch_papers_http_timeout(self, arxiv_client):
        """Test handling of HTTP timeout errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.TimeoutException("Request timeout")
            )

            with pytest.raises(ArxivAPITimeoutError) as exc_info:
                await arxiv_client.fetch_papers(max_results=1)

            assert "arXiv API request timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_papers_http_error(self, arxiv_client):
        """Test handling of HTTP status errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.HTTPStatusError("Server error", request=MagicMock(), response=mock_response)
            )

            with pytest.raises(ArxivAPIException) as exc_info:
                await arxiv_client.fetch_papers(max_results=1)

            assert "arXiv API returned error 500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_paper_by_id_success(self, arxiv_client, mock_arxiv_response):
        """Test fetching a single paper by ID."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.text = mock_arxiv_response
            mock_response.raise_for_status.return_value = None

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            paper = await arxiv_client.fetch_paper_by_id("2024.0001v1")

            assert paper is not None
            assert paper.arxiv_id == "2024.0001v1"
            assert paper.title == "Test Paper Title"

    @pytest.mark.asyncio
    async def test_fetch_paper_by_id_not_found(self, arxiv_client):
        """Test handling when single paper is not found."""
        empty_response = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.text = empty_response
            mock_response.raise_for_status.return_value = None

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            paper = await arxiv_client.fetch_paper_by_id("nonexistent")

            assert paper is None

    def test_parse_response_invalid_xml(self, arxiv_client):
        """Test handling of invalid XML response."""
        invalid_xml = "This is not valid XML"

        with pytest.raises(ArxivParseError) as exc_info:
            arxiv_client._parse_response(invalid_xml)

        assert "Failed to parse arXiv XML response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_download_pdf_cached(self, arxiv_client):
        """Test that cached PDFs are returned without downloading."""
        paper = ArxivPaper(
            arxiv_id="2024.0001v1",
            title="Test Paper",
            authors=["Test Author"],
            abstract="Test abstract",
            categories=["cs.AI"],
            published_date="2024-01-01T00:00:00Z",
            pdf_url="http://arxiv.org/pdf/2024.0001v1",
        )

        with patch("pathlib.Path.exists", return_value=True):
            pdf_path = await arxiv_client.download_pdf(paper)

            assert pdf_path is not None
            assert pdf_path.name == "2024.0001v1.pdf"

    def test_rate_limiting(self, arxiv_client):
        """Test rate limiting delay calculation."""
        import time

        # Mock the last request time
        arxiv_client._last_request_time = time.time() - 1.0  # 1 second ago

        # This would normally cause a delay in real usage
        # In tests, we just verify the logic exists
        assert arxiv_client.rate_limit_delay == 0.1  # Our test value
        assert arxiv_client._last_request_time is not None
```

### 文件：`tests/unit/services/test_pdf_parser.py`（逐字复制）

> 其中 `test_parse_pdf_success` 断言 `mock_parse.assert_called_once_with(valid_pdf_path)` 且对返回值做 `await`——这正要求 `PDFParserService.parse_pdf` 内部**同步**调用 `DoclingParser.parse_pdf`（即修复 #6）。应用修复 #6 后该测试通过。

```python
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from src.exceptions import PDFParsingException, PDFValidationError
from src.schemas.pdf_parser.models import PaperSection, ParserType, PdfContent
from src.services.pdf_parser.docling import DoclingParser
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.pdf_parser.parser import PDFParserService


class TestDoclingParser:
    """Test DoclingParser functionality."""

    @pytest.fixture
    def docling_parser(self):
        """Create DoclingParser instance for testing."""
        return DoclingParser(max_pages=20, max_file_size_mb=10, do_ocr=False)

    @pytest.fixture
    def valid_pdf_path(self, tmp_path):
        """Create a mock valid PDF file path."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")
        return pdf_file

    @pytest.fixture
    def empty_pdf_path(self, tmp_path):
        """Create an empty PDF file path."""
        pdf_file = tmp_path / "empty.pdf"
        pdf_file.write_bytes(b"")
        return pdf_file

    @pytest.fixture
    def invalid_pdf_path(self, tmp_path):
        """Create an invalid PDF file path."""
        pdf_file = tmp_path / "invalid.pdf"
        pdf_file.write_bytes(b"Not a PDF file")
        return pdf_file

    def test_docling_parser_initialization(self, docling_parser):
        """Test DoclingParser initialization."""
        assert docling_parser.max_pages == 20
        assert docling_parser.max_file_size_bytes == 10 * 1024 * 1024
        assert docling_parser._warmed_up is False

    def test_validate_pdf_valid_file(self, docling_parser, valid_pdf_path):
        """Test PDF validation with valid file."""
        # This test is complex due to pypdfium2 dependency, skip for now
        pass

    def test_validate_pdf_empty_file(self, docling_parser, empty_pdf_path):
        """Test PDF validation with empty file."""
        with pytest.raises(PDFValidationError) as exc_info:
            docling_parser._validate_pdf(empty_pdf_path)

        assert "PDF file is empty" in str(exc_info.value)

    def test_validate_pdf_invalid_header(self, docling_parser, invalid_pdf_path):
        """Test PDF validation with invalid header."""
        with pytest.raises(PDFValidationError) as exc_info:
            docling_parser._validate_pdf(invalid_pdf_path)

        assert "File does not have PDF header" in str(exc_info.value)

    def test_validate_pdf_nonexistent_file(self, docling_parser):
        """Test PDF validation with nonexistent file."""
        nonexistent_path = Path("/nonexistent/file.pdf")

        with pytest.raises(PDFValidationError) as exc_info:
            docling_parser._validate_pdf(nonexistent_path)

        assert "Error validating PDF" in str(exc_info.value)

    # Complex PDF parsing tests removed - too dependent on external libraries


class TestPDFParserService:
    """Test PDFParserService functionality."""

    @pytest.fixture
    def pdf_parser_service(self):
        """Create PDFParserService instance for testing."""
        return PDFParserService(max_pages=20, max_file_size_mb=10)

    @pytest.fixture
    def valid_pdf_path(self, tmp_path):
        """Create a mock valid PDF file path."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")
        return pdf_file

    def test_pdf_parser_service_initialization(self, pdf_parser_service):
        """Test PDFParserService initialization."""
        assert isinstance(pdf_parser_service.docling_parser, DoclingParser)
        assert pdf_parser_service.docling_parser.max_pages == 20

    @pytest.mark.asyncio
    async def test_parse_pdf_file_not_found(self, pdf_parser_service):
        """Test parsing with non-existent file."""
        nonexistent_path = Path("/nonexistent/file.pdf")

        with pytest.raises(PDFValidationError) as exc_info:
            await pdf_parser_service.parse_pdf(nonexistent_path)

        assert "PDF file not found" in str(exc_info.value)

    @patch("src.services.pdf_parser.parser.DoclingParser.parse_pdf")
    @pytest.mark.asyncio
    async def test_parse_pdf_success(self, mock_parse, pdf_parser_service, valid_pdf_path):
        """Test successful PDF parsing."""
        mock_content = PdfContent(
            raw_text="Test content", sections=[], tables=[], figures=[], parser_used=ParserType.DOCLING, metadata={}
        )
        mock_parse.return_value = mock_content

        result = await pdf_parser_service.parse_pdf(valid_pdf_path)

        assert result == mock_content
        mock_parse.assert_called_once_with(valid_pdf_path)

    @patch("src.services.pdf_parser.parser.DoclingParser.parse_pdf")
    @pytest.mark.asyncio
    async def test_parse_pdf_no_result(self, mock_parse, pdf_parser_service, valid_pdf_path):
        """Test PDF parsing when no result is returned."""
        mock_parse.return_value = None

        with pytest.raises(PDFParsingException) as exc_info:
            await pdf_parser_service.parse_pdf(valid_pdf_path)

        assert "Docling parsing returned no result" in str(exc_info.value)

    @patch("src.services.pdf_parser.parser.DoclingParser.parse_pdf")
    @pytest.mark.asyncio
    async def test_parse_pdf_docling_error(self, mock_parse, pdf_parser_service, valid_pdf_path):
        """Test PDF parsing when Docling raises an error."""
        mock_parse.side_effect = Exception("Docling error")

        with pytest.raises(PDFParsingException) as exc_info:
            await pdf_parser_service.parse_pdf(valid_pdf_path)

        assert "Docling parsing error" in str(exc_info.value)

    def test_factory_creates_service(self):
        """Test that factory creates PDFParserService instance."""
        service = make_pdf_parser_service()
        assert isinstance(service, PDFParserService)
        assert isinstance(service.docling_parser, DoclingParser)

    def test_factory_caching(self):
        """Test that factory uses caching."""
        service1 = make_pdf_parser_service()
        service2 = make_pdf_parser_service()
        # Should be the same instance due to @lru_cache
        assert service1 is service2
```

### 文件：`tests/unit/services/test_metadata_fetcher.py`（逐字复制）

```python
import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.exceptions import MetadataFetchingException, PipelineException
from src.schemas.arxiv.paper import ArxivPaper
from src.schemas.pdf_parser.models import ParserType, PdfContent
from src.services.arxiv.client import ArxivClient
from src.services.metadata_fetcher import MetadataFetcher, make_metadata_fetcher
from src.services.pdf_parser.parser import PDFParserService


class TestMetadataFetcher:
    """Test MetadataFetcher functionality."""

    @pytest.fixture
    def mock_arxiv_client(self):
        """Create mock ArxivClient."""
        client = MagicMock(spec=ArxivClient)
        return client

    @pytest.fixture
    def mock_pdf_parser(self):
        """Create mock PDFParserService."""
        parser = MagicMock(spec=PDFParserService)
        return parser

    @pytest.fixture
    def metadata_fetcher(self, mock_arxiv_client, mock_pdf_parser, tmp_path):
        """Create MetadataFetcher instance for testing."""
        return MetadataFetcher(
            arxiv_client=mock_arxiv_client,
            pdf_parser=mock_pdf_parser,
            pdf_cache_dir=tmp_path,
            max_concurrent_downloads=2,
            max_concurrent_parsing=1,
        )

    @pytest.fixture
    def sample_arxiv_papers(self):
        """Create sample ArxivPaper objects."""
        return [
            ArxivPaper(
                arxiv_id="2024.0001v1",
                title="Test Paper 1",
                authors=["Author 1"],
                abstract="Abstract 1",
                categories=["cs.AI"],
                published_date="2024-01-01T00:00:00Z",
                pdf_url="http://arxiv.org/pdf/2024.0001v1",
            ),
            ArxivPaper(
                arxiv_id="2024.0002v1",
                title="Test Paper 2",
                authors=["Author 2"],
                abstract="Abstract 2",
                categories=["cs.AI"],
                published_date="2024-01-02T00:00:00Z",
                pdf_url="http://arxiv.org/pdf/2024.0002v1",
            ),
        ]

    @pytest.fixture
    def sample_pdf_content(self):
        """Create sample PdfContent."""
        return PdfContent(
            raw_text="Sample PDF content", sections=[], tables=[], figures=[], parser_used=ParserType.DOCLING, metadata={}
        )

    def test_metadata_fetcher_initialization(self, metadata_fetcher, tmp_path):
        """Test MetadataFetcher initialization."""
        assert metadata_fetcher.pdf_cache_dir == tmp_path
        assert metadata_fetcher.max_concurrent_downloads == 2
        assert metadata_fetcher.max_concurrent_parsing == 1

    # Complex integration tests removed for simplicity

    # Most complex tests removed - keeping only simple ones

    @pytest.mark.asyncio
    async def test_empty_papers_list(self, metadata_fetcher):
        """Test handling of empty papers list."""
        result = await metadata_fetcher.fetch_and_process_papers(max_results=0, process_pdfs=False, store_to_db=False)

        assert result["papers_fetched"] == 0
        assert result["pdfs_downloaded"] == 0
        assert result["pdfs_parsed"] == 0
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_rate_limiting_respected(self, metadata_fetcher):
        """Test that rate limiting delays are respected."""
        # This is a basic test to ensure the rate limiting logic exists
        # More comprehensive testing would require timing analysis
        metadata_fetcher.arxiv_client.fetch_papers = AsyncMock(return_value=[])

        start_time = time.time()
        await metadata_fetcher.fetch_and_process_papers(max_results=1)
        end_time = time.time()

        # Should complete quickly for empty result
        assert end_time - start_time < 1.0
```

### 文件：`tests/unit/services/test_telegram.py`（逐字复制）

```python
from unittest.mock import MagicMock, patch

from src.config import TelegramSettings
from src.services.telegram.bot import TelegramBot
from src.services.telegram.factory import make_telegram_service


class TestTelegramBot:
    """Test Telegram bot."""

    def test_bot_creation(self):
        """Test creating bot instance."""
        bot = TelegramBot(
            bot_token="test_token",
            opensearch_client=MagicMock(),
            embeddings_client=MagicMock(),
            ollama_client=MagicMock(),
        )

        assert bot.bot_token == "test_token"
        assert bot.opensearch is not None
        assert bot.embeddings is not None
        assert bot.ollama is not None


class TestTelegramSettings:
    """Test Telegram settings."""

    def test_default_settings(self):
        """Test default settings."""
        # Explicitly set default values to test the schema, ignoring .env
        settings = TelegramSettings(bot_token="", enabled=False)
        assert settings.enabled is False
        assert settings.bot_token == ""

    def test_custom_settings(self):
        """Test custom settings."""
        settings = TelegramSettings(bot_token="test", enabled=True)
        assert settings.enabled is True
        assert settings.bot_token == "test"


class TestTelegramFactory:
    """Test factory."""

    @patch("src.services.telegram.factory.get_settings")
    def test_factory_disabled(self, mock_settings):
        """Test factory returns None when disabled."""
        mock_settings.return_value.telegram.enabled = False
        bot = make_telegram_service(
            opensearch_client=MagicMock(),
            embeddings_client=MagicMock(),
            ollama_client=MagicMock(),
        )
        assert bot is None

    @patch("src.services.telegram.factory.get_settings")
    def test_factory_no_token(self, mock_settings):
        """Test factory returns None without token."""
        mock_settings.return_value.telegram.enabled = True
        mock_settings.return_value.telegram.bot_token = ""
        bot = make_telegram_service(
            opensearch_client=MagicMock(),
            embeddings_client=MagicMock(),
            ollama_client=MagicMock(),
        )
        assert bot is None

    @patch("src.services.telegram.factory.get_settings")
    def test_factory_success(self, mock_settings):
        """Test factory creates bot."""
        mock_settings.return_value.telegram.enabled = True
        mock_settings.return_value.telegram.bot_token = "test_token"
        bot = make_telegram_service(
            opensearch_client=MagicMock(),
            embeddings_client=MagicMock(),
            ollama_client=MagicMock(),
        )
        assert bot is not None
        assert bot.bot_token == "test_token"
```

### 文件：`tests/api/routers/test_agentic_ask.py`（逐字复制）

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock

from src.main import app
from src.services.agents.agentic_rag import AgenticRAGService
from src import dependencies


@pytest.fixture
def mock_agentic_rag_service():
    """Mock AgenticRAGService for API testing."""
    service = Mock(spec=AgenticRAGService)
    service.ask = AsyncMock(return_value={
        "query": "What is machine learning?",
        "answer": "Machine learning is a subset of AI that enables systems to learn from data.",
        "sources": ["https://arxiv.org/pdf/2301.00001.pdf"],
        "reasoning_steps": [
            "Validated query is about AI research",
            "Retrieved 3 relevant papers",
            "Generated answer from sources"
        ],
        "retrieval_attempts": 1,
        "rewritten_query": None,
    })
    return service


@pytest.fixture
def client(mock_agentic_rag_service):
    """FastAPI test client with mocked dependencies."""
    # Override the dependency to return our mock service
    def override_get_agentic_rag_service():
        return mock_agentic_rag_service

    app.dependency_overrides[dependencies.get_agentic_rag_service] = override_get_agentic_rag_service

    yield TestClient(app)

    # Clean up after test
    app.dependency_overrides.clear()


class TestAgenticAskEndpoint:
    """Tests for POST /api/v1/ask-agentic endpoint."""

    def test_ask_agentic_success(self, client, mock_agentic_rag_service):
        """Test successful agentic RAG request."""
        response = client.post(
            "/api/v1/ask-agentic",
            json={
                "query": "What is machine learning?",
                "model": "gemma4:e2b",
                "top_k": 3,
                "use_hybrid": True
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "query" in data
        assert "answer" in data
        assert "sources" in data
        assert "reasoning_steps" in data
        assert "retrieval_attempts" in data
        assert "chunks_used" in data
        assert "search_mode" in data

        # Verify content
        assert data["query"] == "What is machine learning?"
        assert "machine learning" in data["answer"].lower()
        assert len(data["sources"]) > 0
        assert len(data["reasoning_steps"]) > 0
        assert data["retrieval_attempts"] == 1

    def test_ask_agentic_minimal_request(self, client, mock_agentic_rag_service):
        """Test agentic RAG with minimal required fields."""
        response = client.post(
            "/api/v1/ask-agentic",
            json={"query": "What is neural network?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_ask_agentic_empty_query(self, client, mock_agentic_rag_service):
        """Test agentic RAG with empty query returns 422."""
        mock_agentic_rag_service.ask = AsyncMock(side_effect=ValueError("Query cannot be empty"))

        response = client.post(
            "/api/v1/ask-agentic",
            json={"query": ""}
        )

        assert response.status_code == 422

    def test_ask_agentic_missing_query(self, client):
        """Test agentic RAG without query field returns 422."""
        response = client.post(
            "/api/v1/ask-agentic",
            json={"model": "gemma4:e2b"}
        )

        assert response.status_code == 422

    def test_ask_agentic_service_error(self, client, mock_agentic_rag_service):
        """Test agentic RAG when service raises exception."""
        mock_agentic_rag_service.ask = AsyncMock(side_effect=Exception("Service error"))

        response = client.post(
            "/api/v1/ask-agentic",
            json={"query": "Test query"}
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    def test_ask_agentic_with_sources(self, client, mock_agentic_rag_service):
        """Test that sources are properly returned in response."""
        mock_agentic_rag_service.ask = AsyncMock(return_value={
            "query": "What is transformer architecture?",
            "answer": "Transformers use self-attention mechanisms.",
            "sources": ["https://arxiv.org/pdf/1706.03762.pdf"],
            "reasoning_steps": ["Retrieved papers", "Generated answer"],
            "retrieval_attempts": 1,
            "rewritten_query": None,
        })

        response = client.post(
            "/api/v1/ask-agentic",
            json={"query": "What is transformer architecture?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 1
        assert "1706.03762" in data["sources"][0]

    def test_ask_agentic_reasoning_steps(self, client, mock_agentic_rag_service):
        """Test that reasoning steps are included in response."""
        mock_agentic_rag_service.ask = AsyncMock(return_value={
            "query": "What is deep learning?",
            "answer": "Deep learning is...",
            "sources": [],
            "reasoning_steps": [
                "Query validation passed",
                "Retrieved 3 papers",
                "Graded documents as relevant",
                "Generated final answer"
            ],
            "retrieval_attempts": 1,
            "rewritten_query": None,
        })

        response = client.post(
            "/api/v1/ask-agentic",
            json={"query": "What is deep learning?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["reasoning_steps"]) == 4
        assert "Query validation passed" in data["reasoning_steps"]

    def test_ask_agentic_with_rewritten_query(self, client, mock_agentic_rag_service):
        """Test response when query was rewritten."""
        mock_agentic_rag_service.ask = AsyncMock(return_value={
            "query": "ML stuff",
            "answer": "Machine learning...",
            "sources": [],
            "reasoning_steps": ["Query rewritten", "Retrieved papers"],
            "retrieval_attempts": 2,
            "rewritten_query": "What are the key concepts in machine learning?",
        })

        response = client.post(
            "/api/v1/ask-agentic",
            json={"query": "ML stuff"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["rewritten_query"] == "What are the key concepts in machine learning?"
        assert data["retrieval_attempts"] == 2

    def test_ask_agentic_custom_model(self, client, mock_agentic_rag_service):
        """Test agentic RAG with custom model parameter."""
        response = client.post(
            "/api/v1/ask-agentic",
            json={
                "query": "What is AI?",
                "model": "llama3.2:3b"
            }
        )

        assert response.status_code == 200
        # Verify the service was called with the custom model
        mock_agentic_rag_service.ask.assert_called_once()
        call_kwargs = mock_agentic_rag_service.ask.call_args.kwargs
        assert call_kwargs["model"] == "llama3.2:3b"

    def test_ask_agentic_search_mode_hybrid(self, client, mock_agentic_rag_service):
        """Test that search_mode is set correctly for hybrid search."""
        response = client.post(
            "/api/v1/ask-agentic",
            json={
                "query": "What is AI?",
                "use_hybrid": True
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["search_mode"] == "hybrid"

    def test_ask_agentic_search_mode_bm25(self, client, mock_agentic_rag_service):
        """Test that search_mode is set correctly for BM25 search."""
        response = client.post(
            "/api/v1/ask-agentic",
            json={
                "query": "What is AI?",
                "use_hybrid": False
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["search_mode"] == "bm25"
```

> **注意（呼应修复 #5）**：`test_agentic_ask.py` 用 `app.dependency_overrides[dependencies.get_agentic_rag_service]` **覆盖了**依赖，注入 mock 服务——所以这些 API 测试**绕开了** `get_agentic_rag_service` 的真实实现，即使不打修复 #5 也能过。但真实运行 `/ask-agentic`（不覆盖依赖）必须有修复 #5，否则 `TypeError`。

### 文件：`tests/unit/services/agents/test_models.py`（逐字复制）

```python
import pytest
from pydantic import ValidationError

from src.services.agents.models import (
    GuardrailScoring,
    GradeDocuments,
    SourceItem,
    ToolArtifact,
    RoutingDecision,
    GradingResult,
    ReasoningStep,
)


class TestGuardrailScoring:
    """Tests for GuardrailScoring model."""

    def test_valid_scoring(self):
        """Test creating valid guardrail scoring."""
        scoring = GuardrailScoring(score=75, reason="Query is relevant to AI research papers")
        assert scoring.score == 75
        assert scoring.reason == "Query is relevant to AI research papers"

    def test_score_boundaries(self):
        """Test score boundary validation."""
        # Valid boundaries
        GuardrailScoring(score=0, reason="Minimum score")
        GuardrailScoring(score=100, reason="Maximum score")
        GuardrailScoring(score=50, reason="Middle score")

    def test_invalid_score_too_low(self):
        """Test score below minimum."""
        with pytest.raises(ValidationError):
            GuardrailScoring(score=-1, reason="Invalid")

    def test_invalid_score_too_high(self):
        """Test score above maximum."""
        with pytest.raises(ValidationError):
            GuardrailScoring(score=101, reason="Invalid")


class TestGradeDocuments:
    """Tests for GradeDocuments model."""

    def test_valid_yes_grade(self):
        """Test creating valid 'yes' grade."""
        grade = GradeDocuments(binary_score="yes", reasoning="Document is highly relevant")
        assert grade.binary_score == "yes"
        assert grade.reasoning == "Document is highly relevant"

    def test_valid_no_grade(self):
        """Test creating valid 'no' grade."""
        grade = GradeDocuments(binary_score="no", reasoning="Document is off-topic")
        assert grade.binary_score == "no"
        assert grade.reasoning == "Document is off-topic"

    def test_default_reasoning(self):
        """Test default empty reasoning."""
        grade = GradeDocuments(binary_score="yes")
        assert grade.reasoning == ""

    def test_invalid_binary_score(self):
        """Test invalid binary score value."""
        with pytest.raises(ValidationError):
            GradeDocuments(binary_score="maybe")


class TestSourceItem:
    """Tests for SourceItem model."""

    def test_valid_source_item(self):
        """Test creating valid source item."""
        source = SourceItem(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Vaswani, A.", "Shazeer, N."],
            url="https://arxiv.org/abs/1706.03762",
            relevance_score=0.95
        )
        assert source.arxiv_id == "1706.03762"
        assert source.title == "Attention Is All You Need"
        assert len(source.authors) == 2
        assert source.url == "https://arxiv.org/abs/1706.03762"
        assert source.relevance_score == 0.95

    def test_default_values(self):
        """Test default field values."""
        source = SourceItem(
            arxiv_id="1234.5678",
            title="Test Paper",
            url="https://arxiv.org/abs/1234.5678"
        )
        assert source.authors == []
        assert source.relevance_score == 0.0

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        source = SourceItem(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Vaswani, A."],
            url="https://arxiv.org/abs/1706.03762",
            relevance_score=0.95
        )
        source_dict = source.to_dict()

        assert isinstance(source_dict, dict)
        assert source_dict["arxiv_id"] == "1706.03762"
        assert source_dict["title"] == "Attention Is All You Need"
        assert source_dict["authors"] == ["Vaswani, A."]
        assert source_dict["url"] == "https://arxiv.org/abs/1706.03762"
        assert source_dict["relevance_score"] == 0.95


class TestToolArtifact:
    """Tests for ToolArtifact model."""

    def test_valid_tool_artifact(self):
        """Test creating valid tool artifact."""
        artifact = ToolArtifact(
            tool_name="retrieve_papers",
            tool_call_id="call_123",
            content="Retrieved 3 papers",
            metadata={"count": 3, "source": "opensearch"}
        )
        assert artifact.tool_name == "retrieve_papers"
        assert artifact.tool_call_id == "call_123"
        assert artifact.content == "Retrieved 3 papers"
        assert artifact.metadata["count"] == 3

    def test_default_metadata(self):
        """Test default empty metadata."""
        artifact = ToolArtifact(
            tool_name="test_tool",
            tool_call_id="call_456",
            content="Test content"
        )
        assert artifact.metadata == {}


class TestRoutingDecision:
    """Tests for RoutingDecision model."""

    def test_valid_routing_decisions(self):
        """Test all valid routing options."""
        routes = ["retrieve", "out_of_scope", "generate_answer", "rewrite_query"]

        for route in routes:
            decision = RoutingDecision(route=route, reason=f"Testing {route}")
            assert decision.route == route
            assert decision.reason == f"Testing {route}"

    def test_default_reason(self):
        """Test default empty reason."""
        decision = RoutingDecision(route="retrieve")
        assert decision.reason == ""

    def test_invalid_route(self):
        """Test invalid routing option."""
        with pytest.raises(ValidationError):
            RoutingDecision(route="invalid_route")


class TestGradingResult:
    """Tests for GradingResult model."""

    def test_valid_grading_result(self):
        """Test creating valid grading result."""
        result = GradingResult(
            document_id="doc_123",
            is_relevant=True,
            score=0.87,
            reasoning="Contains relevant information about transformers"
        )
        assert result.document_id == "doc_123"
        assert result.is_relevant is True
        assert result.score == 0.87
        assert "transformers" in result.reasoning

    def test_default_values(self):
        """Test default field values."""
        result = GradingResult(
            document_id="doc_456",
            is_relevant=False
        )
        assert result.score == 0.0
        assert result.reasoning == ""


class TestReasoningStep:
    """Tests for ReasoningStep model."""

    def test_valid_reasoning_step(self):
        """Test creating valid reasoning step."""
        step = ReasoningStep(
            step_name="retrieve",
            description="Retrieved 3 relevant papers from OpenSearch",
            metadata={"num_docs": 3, "retrieval_time_ms": 150}
        )
        assert step.step_name == "retrieve"
        assert step.description == "Retrieved 3 relevant papers from OpenSearch"
        assert step.metadata["num_docs"] == 3
        assert step.metadata["retrieval_time_ms"] == 150

    def test_default_metadata(self):
        """Test default empty metadata."""
        step = ReasoningStep(
            step_name="generate",
            description="Generated final answer"
        )
        assert step.metadata == {}
```

### 文件：`tests/unit/services/agents/test_tools.py`（逐字复制）

```python
import pytest
from unittest.mock import AsyncMock
from langchain_core.documents import Document

from src.services.agents.tools import create_retriever_tool


@pytest.mark.asyncio
async def test_create_retriever_tool_basic(mock_opensearch_client, mock_jina_embeddings_client):
    """Test basic retriever tool creation and invocation."""
    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        top_k=2,
        use_hybrid=True,
    )

    # Verify tool properties
    assert tool.name == "retrieve_papers"
    assert "Search and return relevant arXiv research papers" in tool.description

    # Invoke tool
    result = await tool.ainvoke({"query": "machine learning"})

    # Verify result
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(doc, Document) for doc in result)

    # Verify first document
    first_doc = result[0]
    assert first_doc.page_content == "Transformers are neural network architectures based on self-attention mechanisms."
    assert first_doc.metadata["arxiv_id"] == "1706.03762"
    assert first_doc.metadata["title"] == "Attention Is All You Need"
    assert first_doc.metadata["score"] == 0.95

    # Verify embeddings were generated
    mock_jina_embeddings_client.embed_query.assert_called_once_with("machine learning")

    # Verify search was called correctly
    mock_opensearch_client.search_unified.assert_called_once()
    call_args = mock_opensearch_client.search_unified.call_args
    assert call_args.kwargs["query"] == "machine learning"
    assert call_args.kwargs["size"] == 2  # search_unified uses 'size', not 'top_k'
    assert call_args.kwargs["use_hybrid"] is True


@pytest.mark.asyncio
async def test_retriever_tool_empty_results(mock_opensearch_client, mock_jina_embeddings_client):
    """Test retriever tool with no results."""
    from unittest.mock import Mock
    mock_opensearch_client.search_unified = Mock(return_value={"hits": []})

    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
    )

    result = await tool.ainvoke({"query": "nonexistent topic"})

    assert isinstance(result, list)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_retriever_tool_custom_top_k(mock_opensearch_client, mock_jina_embeddings_client):
    """Test retriever tool with custom top_k parameter."""
    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        top_k=5,
        use_hybrid=False,
    )

    await tool.ainvoke({"query": "test query"})

    call_args = mock_opensearch_client.search_unified.call_args
    # search_unified uses 'size' parameter, not 'top_k'
    assert call_args.kwargs["size"] == 5
    assert call_args.kwargs["use_hybrid"] is False


@pytest.mark.asyncio
async def test_retriever_tool_metadata_fields(mock_opensearch_client, mock_jina_embeddings_client):
    """Test that all expected metadata fields are present."""
    from unittest.mock import Mock
    mock_opensearch_client.search_unified = Mock(return_value={
        "hits": [
            {
                "chunk_text": "Test content",
                "arxiv_id": "2301.00001",
                "title": "Test Paper",
                "authors": "Author One, Author Two",
                "score": 0.95,
                "section_name": "Introduction",
            }
        ]
    })

    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
    )

    result = await tool.ainvoke({"query": "test"})

    doc = result[0]
    assert "arxiv_id" in doc.metadata
    assert "title" in doc.metadata
    assert "authors" in doc.metadata
    assert "score" in doc.metadata
    assert "source" in doc.metadata
    assert "section" in doc.metadata
```

### 文件：`tests/unit/services/agents/test_nodes.py`（含修复 #4）

> ⚠️ **修复 #4 已应用**：上游本文件把 mock 配置在 `create_llm` 上（节点实际调用的是 `get_langchain_model`）。下面已把全部 5 处 `test_context.ollama_client.create_llm = Mock(...)` 改为 `test_context.ollama_client.get_langchain_model = Mock(...)`，与节点实现一致。

```python
"""Tests for agentic RAG node functions using Runtime[Context] pattern."""

import pytest
from unittest.mock import AsyncMock, Mock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from src.services.agents.nodes import (
    ainvoke_retrieve_step,
    ainvoke_grade_documents_step,
    ainvoke_rewrite_query_step,
    ainvoke_generate_answer_step,
    ainvoke_out_of_scope_step,
    continue_after_guardrail,
)
from src.services.agents.nodes.utils import get_latest_query, get_latest_context
from src.services.agents.models import GuardrailScoring, GradeDocuments
from src.services.agents.state import AgentState


class TestGuardrailNode:
    """Tests for guardrail validation node."""

    def test_continue_after_guardrail_pass(self, test_context):
        """Test routing decision after guardrail pass."""
        state: AgentState = {
            "messages": [],
            "retrieval_attempts": 0,
            "guardrail_result": GuardrailScoring(score=75, reason="Pass"),
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = continue_after_guardrail(state, runtime)

        assert result == "continue"

    def test_continue_after_guardrail_fail(self, test_context):
        """Test routing decision after guardrail fail."""
        state: AgentState = {
            "messages": [],
            "retrieval_attempts": 0,
            "guardrail_result": GuardrailScoring(score=30, reason="Fail"),
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = continue_after_guardrail(state, runtime)

        assert result == "out_of_scope"


class TestRetrieveNode:
    """Tests for document retrieval node."""

    @pytest.mark.asyncio
    async def test_retrieve_creates_tool_call(self, test_context, sample_human_message):
        """Test retrieve node creates tool call."""
        state: AgentState = {
            "messages": [sample_human_message],
            "retrieval_attempts": 0,
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_retrieve_step(state, runtime)

        assert "retrieval_attempts" in result
        assert result["retrieval_attempts"] == 1
        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)
        assert len(result["messages"][0].tool_calls) > 0
        assert result["messages"][0].tool_calls[0]["name"] == "retrieve_papers"

    @pytest.mark.asyncio
    async def test_retrieve_max_attempts_reached(self, test_context, sample_human_message):
        """Test retrieve node when max attempts reached."""
        state: AgentState = {
            "messages": [sample_human_message],
            "retrieval_attempts": 2,  # Already at max
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_retrieve_step(state, runtime)

        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)
        # Check that message indicates failure to find papers
        content_lower = result["messages"][0].content.lower()
        assert "apologize" in content_lower or "unable" in content_lower or "couldn't find" in content_lower


class TestGradeDocumentsNode:
    """Tests for document grading node."""

    @pytest.mark.asyncio
    async def test_grade_documents_relevant(self, test_context, sample_human_message, sample_tool_message):
        """Test grading node with relevant documents."""
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=GradeDocuments(
            binary_score="yes",
            reasoning="Document discusses transformers which is relevant"
        ))
        test_context.ollama_client.get_langchain_model = Mock(return_value=mock_llm)

        state: AgentState = {
            "messages": [sample_human_message, sample_tool_message],
            "retrieval_attempts": 1,
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_grade_documents_step(state, runtime)

        assert "grading_results" in result

    @pytest.mark.asyncio
    async def test_grade_documents_not_relevant(self, test_context, sample_human_message, sample_tool_message):
        """Test grading node with irrelevant documents."""
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=GradeDocuments(
            binary_score="no",
            reasoning="Document is not relevant to the query"
        ))
        test_context.ollama_client.get_langchain_model = Mock(return_value=mock_llm)

        state: AgentState = {
            "messages": [sample_human_message, sample_tool_message],
            "retrieval_attempts": 1,
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_grade_documents_step(state, runtime)

        assert "grading_results" in result


class TestRewriteQueryNode:
    """Tests for query rewriting node."""

    @pytest.mark.asyncio
    async def test_rewrite_query_success(self, test_context, sample_human_message):
        """Test query rewriting with LLM."""
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content="What are the key concepts in transformer neural network architectures?"
        ))
        test_context.ollama_client.get_langchain_model = Mock(return_value=mock_llm)

        state: AgentState = {
            "messages": [sample_human_message],
            "retrieval_attempts": 1,
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_rewrite_query_step(state, runtime)

        assert "messages" in result
        assert isinstance(result["messages"][0], HumanMessage)
        assert len(result["messages"][0].content) > 0
        assert "rewritten_query" in result


class TestGenerateAnswerNode:
    """Tests for answer generation node."""

    @pytest.mark.asyncio
    async def test_generate_answer_success(self, test_context, sample_human_message, sample_tool_message):
        """Test answer generation with context."""
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content="Based on the papers, transformers are neural network architectures."
        ))
        test_context.ollama_client.get_langchain_model = Mock(return_value=mock_llm)

        state: AgentState = {
            "messages": [sample_human_message, sample_tool_message],
            "retrieval_attempts": 1,
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_generate_answer_step(state, runtime)

        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)
        assert len(result["messages"][0].content) > 0


class TestOutOfScopeNode:
    """Tests for out-of-scope handling node."""

    @pytest.mark.asyncio
    async def test_out_of_scope_response(self, test_context, sample_human_message):
        """Test out-of-scope helpful rejection."""
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content="I'm designed to help with AI research papers."
        ))
        test_context.ollama_client.get_langchain_model = Mock(return_value=mock_llm)

        state: AgentState = {
            "messages": [sample_human_message],
            "retrieval_attempts": 0,
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_out_of_scope_step(state, runtime)

        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)


class TestNodeUtils:
    """Tests for node utility functions."""

    def test_get_latest_query(self, sample_human_message, sample_ai_message):
        """Test extracting latest query from messages."""
        messages = [sample_human_message, sample_ai_message]
        query = get_latest_query(messages)

        assert query == "What is machine learning?"

    def test_get_latest_query_with_multiple_human_messages(self):
        """Test extracting latest query with multiple human messages."""
        messages = [
            HumanMessage(content="First query"),
            AIMessage(content="First response"),
            HumanMessage(content="Second query"),
        ]
        query = get_latest_query(messages)

        assert query == "Second query"

    def test_get_latest_context(self, sample_tool_message):
        """Test extracting tool message context."""
        messages = [HumanMessage(content="Query"), sample_tool_message]
        context = get_latest_context(messages)

        assert context is not None
        assert "Transformers" in context

    def test_get_latest_context_no_tool_messages(self, sample_human_message):
        """Test extracting context when no tool messages exist."""
        messages = [sample_human_message]
        context = get_latest_context(messages)

        assert context == ""
```

### 文件：`tests/unit/services/agents/test_agentic_rag.py`（逐字复制）

```python
"""Tests for AgenticRAGService using LangGraph 2.0 Runtime pattern."""

import pytest
from unittest.mock import AsyncMock, Mock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.services.agents.agentic_rag import AgenticRAGService
from src.services.agents.config import GraphConfig
from src.services.agents.models import GuardrailScoring


@pytest.fixture
def test_service(mock_opensearch_client, mock_ollama_client, mock_jina_embeddings_client):
    """Create AgenticRAGService with mocked dependencies."""
    config = GraphConfig(
        model="gemma4:e2b",
        temperature=0.0,
        top_k=3,
        use_hybrid=True,
        max_retrieval_attempts=2,
        guardrail_threshold=60,
    )
    return AgenticRAGService(
        opensearch_client=mock_opensearch_client,
        ollama_client=mock_ollama_client,
        embeddings_client=mock_jina_embeddings_client,
        langfuse_tracer=None,
        graph_config=config,
    )


class TestAgenticRAGServiceInitialization:
    """Tests for service initialization."""

    def test_service_initialization(self, test_service):
        """Test that service initializes correctly."""
        assert test_service.opensearch is not None
        assert test_service.ollama is not None
        assert test_service.embeddings is not None
        assert test_service.graph is not None
        assert test_service.graph_config is not None

    def test_graph_config_values(self, test_service):
        """Test graph configuration values."""
        assert test_service.graph_config.model == "gemma4:e2b"
        assert test_service.graph_config.top_k == 3
        assert test_service.graph_config.use_hybrid is True
        assert test_service.graph_config.max_retrieval_attempts == 2
        assert test_service.graph_config.guardrail_threshold == 60


class TestAgenticRAGAskMethod:
    """Tests for the ask() method."""

    @pytest.mark.asyncio
    async def test_ask_empty_query_validation(self, test_service):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await test_service.ask(query="")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await test_service.ask(query="   ")

    @pytest.mark.asyncio
    async def test_ask_with_model_override(self, test_service):
        """Test ask method with model parameter override."""
        mock_final_state = {
            "messages": [
                HumanMessage(content="Test query"),
                AIMessage(content="Test answer"),
            ],
            "retrieval_attempts": 0,
            "guardrail_result": GuardrailScoring(score=85, reason="Relevant"),
            "sources": [],
            "relevant_sources": [],
            "grading_results": [],
            "metadata": {},
            "original_query": "Test query",
            "rewritten_query": None,
            "routing_decision": "generate_answer",
            "relevant_tool_artifacts": None,
        }

        test_service.graph.ainvoke = AsyncMock(return_value=mock_final_state)

        result = await test_service.ask(query="Test query", model="llama3.2:3b")

        assert result is not None
        # Verify graph was called
        test_service.graph.ainvoke.assert_called_once()


class TestAgenticRAGGraphVisualization:
    """Tests for graph visualization methods."""

    def test_get_graph_mermaid(self, test_service):
        """Test Mermaid diagram generation."""
        mermaid = test_service.get_graph_mermaid()

        assert isinstance(mermaid, str)
        assert len(mermaid) > 0
        assert "graph" in mermaid.lower() or "flowchart" in mermaid.lower()


class TestAgenticRAGErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_ask_with_graph_execution_error(self, test_service):
        """Test error handling when graph execution fails."""
        # Mock graph to raise an exception
        test_service.graph.ainvoke = AsyncMock(side_effect=Exception("Graph execution failed"))

        with pytest.raises(Exception, match="Graph execution failed"):
            await test_service.ask(query="Test query")
```

> 加上 conftest（提供全部夹具）+ 修复 #4（命名）+ 修复 #1/#2（让节点不在导入期崩溃）后，整套智能体测试即可 `uv run pytest tests/unit/services/agents/ -q` 运行。

---

## 11.8 运行命令

```bash
# 运行全部测试
make test
# 等价于：
uv run pytest

# 只跑某个目录/文件
uv run pytest tests/unit/
uv run pytest tests/api/routers/test_ping.py

# 只跑某个测试函数
uv run pytest tests/unit/test_config.py::test_settings_initialization

# 详细输出 + 第一处失败即停
uv run pytest -v -x

# 带覆盖率（生成 HTML 报告到 htmlcov/）
make test-cov
# 等价于：
uv run pytest --cov=src --cov-report=html
```

> **跑测试前的环境**：API/单元测试默认 mock 外部服务，**无需启动 Docker**。**集成测试**（`tests/integration/`）需要先 `docker compose up -d opensearch` 且能联网。如果只想跑离线测试，可排除集成目录：`uv run pytest --ignore=tests/integration`。

---

## 11.9 验证方法清单

| 验证项 | 命令 / 方法 | 期望 |
|--------|-------------|------|
| 单元/API 测试通过 | `uv run pytest --ignore=tests/integration` | 全绿 |
| 覆盖率 | `make test-cov` → 打开 `htmlcov/index.html` | 查看各模块覆盖 |
| 代码风格 | `uv run ruff format --check` 与 `uv run ruff check` | 无问题 |
| 类型检查 | `uv run mypy src/` | 通过（项目 `ignore_errors=true`，主要查导入） |
| 健康检查 | `curl localhost:8000/api/v1/health` | `status: ok` |
| 检索 | `POST /api/v1/hybrid-search/` | 返回 hits |
| RAG | `POST /api/v1/ask` | 返回带来源的 answer |
| Agentic | `POST /api/v1/ask-agentic` | 返回 answer + reasoning_steps |

> **诚实提醒**：本教程的代码经过逐文件审阅源码、在 `.venv` 核对关键第三方 API、并修正了上游会崩溃的缺陷而写成。但**整套系统未在本编写环境里端到端跑通**（需 Docker/OpenSearch/Ollama/Jina/Telegram）。请在你的机器上按上表逐项实际验证。

---

## 11.10 本章小结

- ✅ 理解了测试技术栈（pytest + asyncio + mock 外部服务）。
- ✅ 拿到了可运行的 API 测试、单元测试、集成测试。
- ✅ 知道了智能体测试的上游问题（命名 #4 + 缺失夹具）及修复方法。
- ✅ 掌握了运行命令、覆盖率与完整验证清单。

下一章 [`12-run-build-deploy-rollback.md`](12-run-build-deploy-rollback.md) 讲本地运行、构建、部署与回滚。
