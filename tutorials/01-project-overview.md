# File: tutorials/01-project-overview.md

# 第 1 章　项目概览

本章回答三个问题：**这是什么？最终能做什么？整体怎么演进？** 读完本章你会对全局有清晰的心智模型，后面每一章都能对号入座。

---

## 1.1 一句话定义

**arXiv Paper Curator 是一个本地优先的 RAG 研究助手**：它自动抓取 arXiv 的计算机科学（默认 `cs.AI`）论文，解析全文，建立可检索的索引，然后让你用自然语言提问，由本地大模型基于检索到的论文片段生成**带引用来源**的答案。

它和"直接调用 ChatGPT 问论文"的本质区别：

- **答案有据可依**：回答只基于检索到的真实论文片段，并附 arXiv 链接，可追溯、可验证。
- **数据不出本地**：嵌入向量、检索、生成都可在本机完成（嵌入用 Jina 免费 API，也可替换为本地模型）。
- **工程化**：有数据管道、健康检查、监控追踪、缓存、测试、CI 配置——按真实项目的方式组织。

---

## 1.2 设计哲学：先打好搜索地基，再加 AI

这是本项目最重要的理念，也贯穿全部章节：

> **专业的 RAG 系统不是"AI 优先"，而是"搜索优先"。** 先把关键词检索（BM25）做扎实，再用向量嵌入增强为混合检索，最后才接大模型。

为什么不一上来就做向量检索？

- 纯向量检索对**精确关键词**（人名、专有名词、算法名、缩写）召回很差。问 "BERT" 时，向量可能给你一堆"语言模型"相关却没提到 BERT 的内容。
- BM25 这类关键词检索**可解释、可调试、零模型依赖、快**，是生产系统的可靠底座。
- 两者结合（混合检索）才能既抓住精确词，又理解语义。这正是 Week 3 → Week 4 的演进逻辑。

---

## 1.3 最终系统能做什么（交付能力清单）

完成全部 7 周后，你将拥有：

1. **一键启动的多服务栈**：`docker compose up` 拉起 API、PostgreSQL、OpenSearch（含仪表盘）、Ollama、Redis、Airflow、Langfuse 全家桶。
2. **自动数据管道**：Airflow DAG 按工作日定时抓取当天 arXiv 论文 → 解析 PDF → 入库 → 分块 → 嵌入 → 索引。
3. **统一检索 API**：单个 `/api/v1/hybrid-search/` 端点支持 BM25、向量、混合三种模式，带分类过滤、分页、高亮。
4. **RAG 问答 API**：`/api/v1/ask`（一次性）与 `/api/v1/stream`（流式 SSE），带来源引用与精确缓存。
5. **Agentic RAG API**：`/api/v1/ask-agentic`，智能体会先判断问题是否在领域内（护栏）、检索、给文档打分、必要时重写查询再检索、最后生成答案，并返回**推理步骤**。
6. **Web 聊天界面**：Gradio 界面（端口 7861），带高级参数控制与流式输出。
7. **Telegram 机器人**：在手机上随时向你的论文助手提问。
8. **生产可观测性**：Langfuse 仪表盘看到每一次请求的检索、提示词、生成、耗时、token 用量；Redis 缓存把重复问题的延迟从数秒降到毫秒级。

---

## 1.4 七周演进路线（每周的目标与产出）

下表是全局地图。每一周对应本教程的一章（Week N → 第 N+3 章）。

| 周 | 主题 | 核心目标 | 新增的关键组件 | 对应章节 |
|----|------|----------|----------------|----------|
| **1** | 基础设施 | 让多服务栈能一键启动，API 能自检健康 | `compose.yml`、FastAPI `main.py`、`config.py`、PostgreSQL、`/health` | [04](04-week1-infrastructure.md) |
| **2** | 数据摄取 | 自动抓论文、解析 PDF、入库 | `ArxivClient`、`DoclingParser`、`MetadataFetcher`、Airflow DAG | [05](05-week2-ingestion.md) |
| **3** | 关键词检索 | 把内容索引进 OpenSearch，实现 BM25 | 索引映射、`QueryBuilder`、`OpenSearchClient`（BM25） | [06](06-week3-opensearch-bm25.md) |
| **4** | 混合检索 | 分块 + 嵌入 + RRF 融合 | `TextChunker`、`JinaEmbeddingsClient`、`HybridIndexingService`、RRF 管道 | [07](07-week4-hybrid-search.md) |
| **5** | 完整 RAG | 接本地大模型，生成带引用的答案 | `OllamaClient`、提示词构建、`/ask` + `/stream`、Gradio | [08](08-week5-rag-llm.md) |
| **6** | 监控与缓存 | 全链路追踪 + 精确缓存 | `LangfuseTracer`、`RAGTracer`、`CacheClient`（Redis） | [09](09-week6-monitoring-caching.md) |
| **7** | Agentic RAG | 智能体决策 + 移动端 | LangGraph 图、5 个决策节点、`TelegramBot`、`/ask-agentic` | [10](10-week7-agentic-telegram.md) |

每一周都是**可独立运行的里程碑**：你完成 Week 3 后就有一个能用的关键词搜索引擎，完成 Week 5 后就有一个完整的 RAG 问答系统。

---

## 1.5 服务与端口一览

完整启动后，本机会运行以下服务（端口来自 `compose.yml`）：

| 服务 | 容器名 | 端口 | 作用 |
|------|--------|------|------|
| FastAPI 应用 | `rag-api` | `8000` | 主 API，交互式文档在 `/docs` |
| PostgreSQL | `rag-postgres` | `5432` | 论文元数据与解析内容存储 |
| OpenSearch | `rag-opensearch` | `9200`, `9600` | 混合检索引擎（BM25 + 向量） |
| OpenSearch Dashboards | `rag-dashboards` | `5601` | 检索引擎的可视化界面 |
| Ollama | `rag-ollama` | `11434` | 本地大模型服务 |
| Redis | `rag-redis` | `6379` | RAG 精确匹配缓存 |
| Airflow | `rag-airflow` | `8080` | 数据摄取工作流编排 |
| Langfuse Web | `rag-langfuse-web` | `3001`（容器内 3000） | RAG 全链路追踪仪表盘 |
| Gradio UI | （本机进程，非容器） | `7861` | 用户聊天界面 |

> Langfuse 自身还会拉起一套支撑服务（ClickHouse、MinIO、独立的 Postgres 与 Redis），第 [09](09-week6-monitoring-caching.md) 章会详解。

### 各界面访问地址

| 界面 | URL | 用途 |
|------|-----|------|
| API 交互文档 | http://localhost:8000/docs | 在浏览器里直接试 API |
| Gradio 聊天 | http://localhost:7861 | 友好的问答界面 |
| Langfuse 仪表盘 | http://localhost:3001 | 监控与追踪 |
| Airflow | http://localhost:8080 | 管道管理（默认账号 `admin`/`admin`） |
| OpenSearch Dashboards | http://localhost:5601 | 检索引擎 UI |

---

## 1.6 API 端点总览

| 端点 | 方法 | 说明 | 引入于 |
|------|------|------|--------|
| `/api/v1/health` | GET | 服务健康检查（DB / OpenSearch / Ollama） | Week 1 |
| `/api/v1/hybrid-search/` | POST | 混合检索（BM25 / 向量 / 混合三合一） | Week 3–4 |
| `/api/v1/ask` | POST | RAG 问答（一次性返回） | Week 5 |
| `/api/v1/stream` | POST | RAG 问答（SSE 流式） | Week 5 |
| `/api/v1/ask-agentic` | POST | Agentic RAG（带推理步骤） | Week 7 |
| `/api/v1/feedback` | POST | 对某次回答提交反馈分数（写入 Langfuse） | Week 7 |

---

## 1.7 成本结构

**这套课程几乎零成本**：

- **本地开发**：$0。Docker、PostgreSQL、OpenSearch、Ollama、Redis、Airflow、Langfuse 全部本地运行。
- **Jina 嵌入 API**：有免费额度，足够学习与中小规模使用（Week 4 起需要）。
- **可选**：如果你想换用云端大模型（而非本地 Ollama），会有少量 API 费用（约 $2–5 量级，纯属可选）。

需要准备的外部凭据（详见第 [02](02-environment-and-dependencies.md) 章）：

- **Jina API Key**（Week 4+ 必需）：免费注册获取。
- **Telegram Bot Token**（Week 7 可选）：向 Telegram 的 @BotFather 申请。
- **Langfuse 密钥**（Week 6 可选）：本地自建 Langfuse 后在其界面创建项目得到。

---

## 1.8 你需要具备/准备的

- **硬件**：8GB+ 内存、20GB+ 空闲磁盘（Docker 镜像 + 模型 + 数据）。
- **软件**：Docker Desktop（含 Compose）、Python 3.12、`uv` 包管理器。
- **心态**：按章节顺序、动手敲完每一个文件。本教程不省略任何代码，你照着做就能得到一个完整项目。

下一章 [`02-environment-and-dependencies.md`](02-environment-and-dependencies.md) 会带你把开发环境和依赖准备好。
