# File: tutorials/03-architecture-and-design.md

# 第 3 章　架构设计、技术选型与目录结构

动手写代码前，先建立全局架构认知。本章讲清楚：**系统由哪些部分组成、为什么这么选型、目录怎么组织、数据如何端到端流动**。每个关键决策都按"为什么 / 替代方案 / 优缺点 / 对性能·维护·安全的影响 / 风险与缓解"展开。

---

## 3.1 高层架构

```
                         ┌──────────────────────────────────────────────┐
                         │                 用户入口                       │
                         │  Gradio UI(:7861)  /  Telegram Bot  /  curl    │
                         └───────────────────────┬──────────────────────┘
                                                 │ HTTP / SSE
                         ┌───────────────────────▼──────────────────────┐
                         │              FastAPI 应用 (rag-api :8000)       │
                         │  routers: ping / hybrid_search / ask / stream  │
                         │          / agentic_ask / feedback              │
                         │  lifespan: 启动时构建所有 service 单例          │
                         └───┬───────────┬───────────┬──────────┬────────┘
                             │           │           │          │
              ┌──────────────▼─┐ ┌───────▼──────┐ ┌──▼───────┐ ┌▼─────────────┐
              │ PostgreSQL     │ │ OpenSearch   │ │ Ollama   │ │ Redis        │
              │ (元数据/全文)  │ │ (BM25+向量)  │ │ (本地LLM)│ │ (精确缓存)   │
              │  :5432         │ │  :9200       │ │  :11434  │ │  :6379       │
              └────────────────┘ └──────────────┘ └──────────┘ └──────────────┘
                             ▲                          ▲
                             │                          │ embeddings
              ┌──────────────┴───────────┐   ┌──────────┴──────────┐
              │ Airflow (:8080)          │   │ Jina Embeddings API │
              │ 每日抓取→解析→入库→索引  │   │ (外部, jina-v3)     │
              └──────────────────────────┘   └─────────────────────┘

              ┌──────────────────────────────────────────────────────┐
              │ Langfuse 全家桶 (web:3001 + clickhouse/minio/pg/redis)│
              │ 接收 RAG 全链路追踪：检索/提示词/生成/耗时/token       │
              └──────────────────────────────────────────────────────┘
```

整套系统分四个平面：

1. **入口平面**：Gradio、Telegram、HTTP 客户端。
2. **应用平面**：FastAPI，按路由组织端点，启动时（lifespan）构建所有服务单例并挂到 `app.state`。
3. **数据/能力平面**：PostgreSQL（事实存储）、OpenSearch（检索）、Ollama（生成）、Redis（缓存）、Jina（嵌入）。
4. **编排与观测平面**：Airflow（定时数据管道）、Langfuse（追踪）。

---

## 3.2 三个贯穿全局的代码模式

理解这三个模式，后面所有代码都会变得很好读。

### 模式一：Factory（工厂函数）

每个服务都有一个 `make_xxx()` 工厂函数（如 `make_database`、`make_opensearch_client`、`make_ollama_client`）。工厂负责"读配置 → 构造实例"。

```python
# 例：src/services/ollama/factory.py 的形态
@lru_cache(maxsize=1)
def make_ollama_client() -> OllamaClient:
    settings = get_settings()
    return OllamaClient(settings)
```

> **为什么用工厂 + `lru_cache` 单例？**
> - **为什么这么选**：把"如何构造"与"如何使用"解耦；`lru_cache` 让无参工厂天然变成进程内单例，避免重复建连接池/重复加载。
> - **替代方案**：直接 `OllamaClient()` 到处 new；用依赖注入框架（如 `dependency-injector`）；用全局变量。
> - **优缺点**：工厂模式无需额外框架、易测试（可在测试里 patch 工厂）；缺点是单例的生命周期需谨慎（见下文 `make_embeddings_service` 故意**不**缓存的反例）。
> - **影响**：连接池/模型只初始化一次，启动后请求路径无额外开销（性能）；构造逻辑集中，易改（可维护）。
> - **风险与缓解**：`lru_cache` 单例在配置变更后不会自动刷新——本项目在 lifespan 启动时一次性构建，符合预期。

### 模式二：依赖注入（FastAPI Depends + app.state）

应用启动时在 `lifespan` 里构建所有服务，挂到 `app.state`；路由用 `Depends` 从 `request.app.state` 取出。`src/dependencies.py` 集中定义这些"取出函数"和类型别名（如 `OpenSearchDep`、`OllamaDep`）。

> **为什么"启动时构建一次"而不是"每请求构建"？**
> - **为什么这么选**：OpenSearch/DB 连接池、Ollama 客户端等构造有成本，复用单例性能最好。
> - **替代方案**：每请求 new（简单但慢）；模块级全局（难测试）。
> - **影响**：消除了"每请求建连接"的开销（性能）；`Depends` 让测试可轻松注入 mock（可维护/可测试）。
> - **风险**：单例需线程/协程安全——这些客户端都设计为可并发使用。

### 模式三：Schema 与 Model 分离

- `src/models/`：**SQLAlchemy ORM 模型**（数据库表结构），如 `Paper`。
- `src/schemas/`：**Pydantic 模型**（API 出入参、内部 DTO、配置子类），如 `PaperCreate`、`AskRequest`、`SearchResponse`。

> **为什么把"数据库模型"和"API 模型"分开？**
> - **为什么这么选**：数据库结构与对外契约的变更节奏不同；分开避免把内部字段（如内部 ID、未脱敏字段）泄露到 API。
> - **替代方案**：直接把 ORM 对象序列化返回（耦合、易泄露）。
> - **影响（安全/可维护）**：对外只暴露明确定义的字段；表结构演进不必同步破坏 API 契约。
> - **风险与缓解**：需要在两层之间做映射（略增样板代码）——用 Pydantic 的 `from_attributes=True` 减轻。

---

## 3.3 完整目录结构

下面是 7 周全部完成后的最终目录（已标注每个文件的职责与所属章节）。**带 `(空)` 的 `__init__.py` 是包标记文件，内容为空或仅含 re-export，但为保证包结构完整你也要创建它们。**

```
arxiv-paper-curator/
├── pyproject.toml              # 依赖与工具配置                      [第02章]
├── uv.lock                     # 锁定版本（uv sync 生成）            [第02章]
├── .env / .env.example / .env.test                                  [第02章]
├── .gitignore / .pre-commit-config.yaml                             [第02、15章]
├── Dockerfile                  # API 镜像（多阶段构建）              [第04章]
├── compose.yml                 # 全部服务编排                        [第04、09章]
├── Makefile                    # 常用命令快捷方式                    [第12章]
├── gradio_launcher.py          # 启动 Gradio 的入口脚本              [第08章]
│
├── src/
│   ├── config.py               # Pydantic Settings（全部配置）       [第04章]
│   ├── main.py                 # FastAPI 入口 + lifespan             [第04→10章]
│   ├── dependencies.py         # 依赖注入定义                        [第04→10章]
│   ├── database.py             # 全局 DB 会话辅助                    [第04章]
│   ├── exceptions.py           # 自定义异常层级                      [第04章]
│   ├── middlewares.py          # 简单日志中间件                      [第04章]
│   ├── gradio_app.py           # Gradio 界面实现                     [第08章]
│   │
│   ├── db/
│   │   ├── __init__.py         (空)
│   │   ├── factory.py          # make_database()                    [第04章]
│   │   └── interfaces/
│   │       ├── __init__.py     (空)
│   │       ├── base.py         # BaseDatabase / BaseRepository 抽象  [第04章]
│   │       └── postgresql.py   # PostgreSQLDatabase 实现 + Base      [第04章]
│   │
│   ├── models/
│   │   ├── __init__.py         (空)
│   │   └── paper.py            # Paper ORM 模型                      [第04章]
│   │
│   ├── repositories/
│   │   ├── __init__.py         (空)
│   │   └── paper.py            # PaperRepository（数据访问）         [第05章]
│   │
│   ├── routers/
│   │   ├── __init__.py         (空)
│   │   ├── ping.py             # /health                            [第04章]
│   │   ├── hybrid_search.py    # /hybrid-search/                    [第06、07章]
│   │   ├── ask.py              # /ask 与 /stream                    [第08章]
│   │   └── agentic_ask.py      # /ask-agentic 与 /feedback          [第10章]
│   │
│   ├── schemas/                # 全部 Pydantic 模型
│   │   ├── api/                # health / search / ask              [第04、06、08章]
│   │   ├── arxiv/              # ArxivPaper / PaperCreate / ...      [第05章]
│   │   ├── database/           # PostgreSQLSettings                 [第04章]
│   │   ├── pdf_parser/         # PdfContent / ParsedPaper / ...      [第05章]
│   │   ├── embeddings/         # Jina 请求/响应模型                  [第07章]
│   │   ├── indexing/           # TextChunk / ChunkMetadata          [第07章]
│   │   ├── ollama.py           # RAGResponse                        [第08章]
│   │   ├── telegram/           # (空 __init__)                      [第10章]
│   │   └── common/             # (空 __init__)
│   │
│   └── services/               # 业务逻辑（每个子包一个 factory.py）
│       ├── arxiv/              # client.py / factory.py             [第05章]
│       ├── pdf_parser/         # docling.py / parser.py / factory   [第05章]
│       ├── metadata_fetcher.py # 摄取编排器                          [第05章]
│       ├── opensearch/         # client / query_builder / 索引配置  [第06、07章]
│       ├── embeddings/         # jina_client / factory              [第07章]
│       ├── indexing/           # text_chunker / hybrid_indexer      [第07章]
│       ├── ollama/             # client / prompts / prompts/*.txt   [第08章]
│       ├── cache/              # client / factory（Redis）          [第09章]
│       ├── langfuse/           # client / tracer / factory          [第09章]
│       ├── telegram/           # bot / factory                      [第10章]
│       └── agents/             # LangGraph 智能体全部                [第10章]
│           ├── agentic_rag.py  # 图编排与服务
│           ├── state.py models.py context.py config.py
│           ├── prompts.py tools.py
│           └── nodes/          # guardrail/retrieve/grade/rewrite/generate/out_of_scope
│
├── airflow/                    # 数据管道                            [第05章]
│   ├── Dockerfile entrypoint.sh requirements-airflow.txt
│   └── dags/
│       ├── arxiv_paper_ingestion.py    # DAG 定义
│       └── arxiv_ingestion/            # 模块化任务函数
│           ├── common.py setup.py fetching.py indexing.py reporting.py
│
├── tests/                      # 测试套件                            [第11章]
│   ├── conftest.py api/ unit/ integration/
│
└── notebooks/                  # 每周配套 Jupyter 学习材料（可选阅读）
    └── week1 ... week7/
```

> **关于 notebooks**：上游每周配有一个 `.ipynb` 学习笔记本。本教程**不复刻笔记本内容**，但**把每一步可运行的操作都直接写进了正文**——你不需要打开任何笔记本就能复现完整项目。笔记本仅作为额外的交互式练习场。

---

## 3.4 关键技术选型决策

下面逐一说明最重要的选型。每个都用统一格式。

### 决策 1：OpenSearch 作为检索引擎（而非专用向量数据库）

- **为什么这么选**：OpenSearch 一个引擎同时提供**成熟的 BM25 关键词检索**和 **kNN 向量检索**，还原生支持 **RRF（Reciprocal Rank Fusion）混合检索管道**。一个组件覆盖关键词 + 向量 + 混合，运维面小。
- **替代方案**：Pinecone / Weaviate / Qdrant / Milvus（专用向量库）+ Elasticsearch（关键词）；或 pgvector（在 PostgreSQL 里做向量）。
- **优缺点**：
  - OpenSearch：✅ 一栈多能、关键词检索强、可本地自托管、Apache 2.0 许可。❌ 比纯向量库重，向量召回/延迟在超大规模下不及专用库。
  - 专用向量库：✅ 向量性能极佳。❌ 还得另配关键词引擎、混合检索要自己拼，组件更多。
  - pgvector：✅ 复用 PostgreSQL。❌ 关键词检索与混合融合较弱。
- **对性能/维护/安全的影响**：单引擎降低运维与依赖复杂度（可维护）；本地自托管，数据不出境（安全/合规）；学习曲线集中。
- **风险与缓解**：超大规模下向量性能瓶颈——本项目用 HNSW 近似最近邻 + 单分片配置满足教学/中小规模；规模上来后可调分片、副本与 HNSW 参数。

### 决策 2：先 BM25 后向量（检索的演进顺序）

- **为什么这么选**：见第 [01](01-project-overview.md) 章的设计哲学。关键词检索是可解释、可靠、零模型依赖的底座；向量是增强。
- **替代方案**：直接上纯向量检索。
- **优缺点**：纯向量对精确词（人名、缩写、算法名）召回差且难调试；BM25 + 向量混合兼顾精确与语义。
- **影响**：先有可用的检索（Week 3 即可交付），降低风险（工程）；混合检索显著提升真实查询的相关性（效果）。
- **风险与缓解**：混合需要嵌入服务可用——代码对嵌入失败做了降级到 BM25 的 try/except。

### 决策 3：RRF（Reciprocal Rank Fusion）做融合（而非加权平均）

- **为什么这么选**：RRF 只看每路结果的**排名**而非分数，天然免去 BM25 分数与向量相似度**量纲不同**的归一化难题，且无需手调权重。
- **替代方案**：归一化后加权平均（如 30% BM25 + 70% 向量，需手调权重）。
- **优缺点**：RRF ✅ 无量纲问题、无需调参、鲁棒。❌ 无法显式偏向某一路。加权平均 ✅ 可精细控制。❌ 归一化与权重难调、易过拟合某类查询。
- **影响**：少调参、结果稳定（可维护/效果）。
- **风险与缓解**：默认 `rank_constant=60`（RRF 公式 `1/(k+rank)` 的 k）适用面广；本项目把加权平均管道作为注释保留以备需要（见第 [07](07-week4-hybrid-search.md) 章）。

### 决策 4：Docling 解析科学 PDF

- **为什么这么选**：Docling 专门针对科学文档，能抽取**结构化的章节标题与正文**（而不仅是裸文本），这对后续"按章节分块"至关重要。
- **替代方案**：PyMuPDF / pdfminer / Unstructured / GROBID。
- **优缺点**：Docling ✅ 结构化输出好、对学术 PDF 友好。❌ 较重、CPU 密集、冷启动慢。
- **影响**：结构化章节让分块质量更高（效果）；但解析慢——本项目用页数/体积上限 + 并发控制（默认 `MAX_CONCURRENT_PARSING=1`）来约束资源（性能）。
- **风险与缓解**：超大/扫描件 PDF 解析失败——代码做了体积/页数校验并优雅跳过（返回 None，仅存元数据）。

### 决策 5：Jina Embeddings v3（外部嵌入 API）

- **为什么这么选**：Jina v3 提供 1024 维、面向检索优化的嵌入，且**区分 `retrieval.passage`（建库）与 `retrieval.query`（查询）两种任务模式**，对非对称检索效果更好；有免费额度，省去本地 GPU。
- **替代方案**：本地 `sentence-transformers` 模型、OpenAI embeddings、Cohere。
- **优缺点**：Jina ✅ 免费额度、质量高、无需本地算力。❌ 依赖外部网络、有速率/额度限制、数据需出境到 Jina。
- **影响（安全/性能）**：数据出境需评估合规；网络往返增加延迟——代码批量化（`batch_size`）并对失败降级。
- **风险与缓解**：API 不可用 → 降级到 BM25；如对数据出境敏感，可替换为本地嵌入模型（维度需与索引一致，见第 [07](07-week4-hybrid-search.md) 章）。

### 决策 6：Ollama 本地大模型（默认 `gemma4:e2b`）

- **为什么这么选**：完全本地、零 API 成本、数据不出境；`gemma4:e2b` 极小、CPU 也能跑，适合教学与快速迭代。
- **替代方案**：OpenAI/Anthropic 等云端 LLM；vLLM 自托管大模型。
- **优缺点**：Ollama+1b ✅ 免费、私密、快。❌ 1b 模型能力弱，复杂问题/结构化输出可靠性有限。
- **影响**：本地推理保护隐私（安全）；小模型对**结构化输出**不稳定——这正是第 [10](10-week7-agentic-telegram.md) 章里每个智能体节点都包了 try/except 兜底的原因。
- **风险与缓解**：需要更好质量时，把 `OLLAMA_MODEL` 换成 `llama3.2:3b` / `qwen2.5:7b` 等（Gradio 界面也提供下拉切换）。

### 决策 7：LangGraph 编排智能体（Week 7）

- **为什么这么选**：Agentic RAG 需要**有状态的、带条件分支的工作流**（护栏→检索→评分→重写→生成）。LangGraph 用"状态图 + 节点 + 条件边"精确表达这种控制流，比一连串 if/else 更清晰、可视化、可追踪。
- **替代方案**：手写状态机、LangChain 的 AgentExecutor、自研编排。
- **优缺点**：LangGraph ✅ 显式状态、条件路由、可导出图、与 Langfuse 集成好。❌ 概念较多、学习曲线、版本演进快。
- **影响**：决策流程显式可见、易调试与追踪（可维护/可观测）。
- **风险与缓解**：API 仍在演进——本项目用 `context_schema` + `Runtime[Context]` 的较新模式；锁定 `langgraph>=0.2.0` 并用 `uv.lock` 固定。

### 决策 8：Redis 精确匹配缓存（Week 6）

- **为什么这么选**：完全相同的问题（含参数）应直接命中缓存，把 RAG 的数秒延迟降到毫秒。精确匹配实现简单、可预测。
- **替代方案**：语义缓存（按问题相似度命中）、应用内内存缓存、无缓存。
- **优缺点**：精确缓存 ✅ 简单、O(1)、命中即对。❌ 措辞略不同就不命中（命中率有限）。语义缓存命中率高但可能返回不完全匹配的旧答案，且更复杂。
- **影响**：重复查询延迟大降、LLM 调用成本下降（性能/成本）。
- **风险与缓解**：缓存键含 query+model+top_k+use_hybrid+categories 的哈希，参数不同不会串答案；TTL 默认 6 小时防止陈旧；Redis 不可用时代码 try/except 降级为正常流程。

### 决策 9：Langfuse 可观测性（Week 6，自托管）

- **为什么这么选**：RAG 是多步流水线（嵌入→检索→提示词→生成），出问题要能看到每一步的输入输出、耗时、token。Langfuse 专为 LLM 应用追踪而生，且可自托管。
- **替代方案**：纯日志、OpenTelemetry + 通用 APM、LangSmith（云）。
- **优缺点**：Langfuse ✅ LLM 语义的追踪/评分/数据集、可自托管。❌ 自托管要拉起一整套支撑服务（ClickHouse/MinIO/PG/Redis），较重。
- **影响**：可观测性强（运维）；但增加本地资源占用（性能/成本）。
- **风险与缓解**：追踪是**可选**的——禁用 Langfuse（不填密钥）时全链路追踪安全降级为 no-op，不影响主流程（这正是第 [16](16-upstream-differences-and-fixes.md) 章修复要保证的不变量）。

### 决策 10：Airflow 编排数据管道（Week 2）

- **为什么这么选**：每日抓取是典型的有依赖、需重试、需可观测的批处理工作流，Airflow 是行业标准。
- **替代方案**：cron + 脚本、Prefect、Dagster、Celery beat。
- **优缺点**：Airflow ✅ 成熟、UI 强、重试/调度/依赖完善。❌ 重、配置多。
- **影响**：管道可观测、可重跑、有依赖编排（运维）。
- **风险与缓解**：本地用 `LocalExecutor` + 单库即可；DAG 任务被拆成模块化函数（`arxiv_ingestion/*`）便于测试与维护。

---

## 3.5 端到端数据流

### 流 A：数据摄取（Airflow，离线）

```
Airflow DAG (每工作日 06:00 UTC)
  setup_environment      → 校验 DB/OpenSearch 连接，创建混合索引 + RRF 管道
  fetch_daily_papers     → ArxivClient 抓当天 cs.AI 论文(XML)
                         → 限速 3s/请求，下载 PDF（并发受控）
                         → DoclingParser 解析 PDF → 章节 + 全文
                         → MetadataFetcher 组装 → PaperRepository.upsert → PostgreSQL
  index_papers_hybrid    → 取最近入库论文 → TextChunker 分块(600词/100重叠)
                         → JinaEmbeddingsClient 批量嵌入(1024维)
                         → OpenSearchClient.bulk_index_chunks → OpenSearch
  generate_daily_report  → 汇总统计（XCom 串联各任务结果）
  cleanup_temp_files     → 清理 30 天前的临时 PDF
```

### 流 B：检索（在线，`/hybrid-search/`）

```
POST /api/v1/hybrid-search/ {query, use_hybrid, ...}
  → 若 use_hybrid: JinaEmbeddingsClient.embed_query(query) → 1024维向量
  → OpenSearchClient.search_unified
       → 有向量且 use_hybrid: _search_hybrid_native（BM25 + kNN → RRF 管道融合）
       → 否则: _search_bm25_only（QueryBuilder 构造 multi_match）
  → 返回 SearchResponse(hits[], total, search_mode)
```

### 流 C：RAG 问答（在线，`/ask`）

```
POST /api/v1/ask {query, top_k, use_hybrid, model}
  → RAGTracer.trace_request（Langfuse 顶层 trace；禁用时 no-op）
  → CacheClient.find_cached_response（命中则直接返回）
  → _prepare_chunks_and_sources：embed_query → search_unified → 取 top_k chunks
  → RAGPromptBuilder：system_prompt + 各 chunk(含 arxiv_id) + question
  → OllamaClient.generate_rag_answer：调用本地 LLM → answer
  → 组装 AskResponse(answer, sources, chunks_used, search_mode)
  → CacheClient.store_response（写缓存，TTL 6h）
```

### 流 D：Agentic RAG（在线，`/ask-agentic`，LangGraph）

```
POST /api/v1/ask-agentic {query}
  → AgenticRAGService.ask → 编译好的 LangGraph 图.ainvoke
     START → guardrail（LLM 打分：是否 CS/AI/ML 领域内）
           → [score < 阈值] → out_of_scope → END
           → [score ≥ 阈值] → retrieve（产生工具调用）
                            → tool_retrieve（embed+search→Documents）
                            → grade_documents（LLM 判定相关性）
                               → [相关] → generate_answer → END
                               → [不相关] → rewrite_query（LLM 重写）→ retrieve（再试，受 max_attempts 限制）
  → 返回 AgenticAskResponse(answer, sources, reasoning_steps, retrieval_attempts, trace_id)
```

---

## 3.6 配置系统总览（`config.py` 预览）

所有配置集中在 `src/config.py` 的 `Settings`，它聚合多个**嵌套子配置类**（每个对应 `.env` 里一组带前缀的变量）：

| 子配置类 | env 前缀 | 控制什么 |
|----------|----------|----------|
| `ArxivSettings` | `ARXIV__` | 抓取速率、分类、并发、缓存目录 |
| `PDFParserSettings` | `PDF_PARSER__` | 页数/体积上限、OCR、表格结构 |
| `ChunkingSettings` | `CHUNKING__` | 分块大小、重叠、最小块 |
| `OpenSearchSettings` | `OPENSEARCH__` | 主机、索引名、向量维度、RRF 管道名 |
| `LangfuseSettings` | `LANGFUSE__` | 追踪开关、密钥、刷新策略 |
| `RedisSettings` | `REDIS__` | 主机/端口/密码、TTL |
| `TelegramSettings` | `TELEGRAM__` | 开关、Bot Token |
| `Settings`（顶层） | （无前缀） | DB URL、Ollama、Jina key、环境 |

完整实现见下一章。

---

至此你已经有了全局地图。从下一章开始，我们按周逐文件构建。先从 [`04-week1-infrastructure.md`](04-week1-infrastructure.md)：基础设施与 FastAPI 骨架。
