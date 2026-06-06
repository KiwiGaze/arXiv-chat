# File: tutorials/README.md

# arXiv Paper Curator —— 从零构建生产级 RAG 系统（完整教程）

> **语言约定（请先读这一句）**：本教程**正文使用中文**，但所有**代码、命令、文件路径、配置键名、日志输出保持英文原样**。这样做的原因：代码必须能直接复制运行，任何对标识符的翻译都会让程序无法工作。如果你更需要英文正文，可在阅读后让作者重新生成。

本教程教你**从一个空目录开始**，一步步构建出一个**可运行的、生产风格的 RAG（检索增强生成）系统**：它能自动抓取 arXiv 上的计算机科学论文、解析 PDF、做关键词 + 向量混合检索、用本地大模型回答问题、带监控和缓存，最后升级为**带 LangGraph 智能体和 Telegram 机器人**的 Agentic RAG。

整个系统**完全可在本地运行，核心功能零云成本**（仅嵌入和可选监控用到免费额度的外部服务）。

---

## 这套教程适合谁

| 读者 | 你会得到什么 |
|------|--------------|
| AI / ML 工程师 | 超越玩具教程的、真实公司在用的 RAG 架构 |
| 后端 / 全栈工程师 | 一个端到端的 AI 应用：API、数据管道、检索、LLM、可观测性 |
| 想系统学习 RAG 的人 | 从关键词检索打地基，再加向量与智能体，循序渐进 |

**前置知识**：会用命令行、懂 Python 基础、了解 Docker 的基本概念即可。不需要机器学习背景。

---

## 教程文件索引（建议按顺序阅读）

每个文件都是独立主题，但后面的章节依赖前面的成果。

| 文件 | 主题 | 对应原项目阶段 |
|------|------|----------------|
| [`README.md`](README.md) | 本文件：总览、约定、学习路径 | — |
| [`01-project-overview.md`](01-project-overview.md) | 项目概览：你将构建什么、最终能力、7 周路线 | 全局 |
| [`02-environment-and-dependencies.md`](02-environment-and-dependencies.md) | 系统与环境需求、依赖版本、安装步骤、密钥准备 | 准备工作 |
| [`03-architecture-and-design.md`](03-architecture-and-design.md) | 架构设计、技术选型理由、目录结构、端到端数据流 | 全局 |
| [`04-week1-infrastructure.md`](04-week1-infrastructure.md) | Week 1：Docker 编排、FastAPI、配置、PostgreSQL、健康检查 | Week 1 |
| [`05-week2-ingestion.md`](05-week2-ingestion.md) | Week 2：arXiv 抓取、Docling 解析 PDF、入库、Airflow 管道 | Week 2 |
| [`06-week3-opensearch-bm25.md`](06-week3-opensearch-bm25.md) | Week 3：OpenSearch 索引、BM25 关键词检索、查询构建器 | Week 3 |
| [`07-week4-hybrid-search.md`](07-week4-hybrid-search.md) | Week 4：文本分块、Jina 嵌入、向量检索、RRF 混合检索 | Week 4 |
| [`08-week5-rag-llm.md`](08-week5-rag-llm.md) | Week 5：Ollama 本地大模型、RAG 问答、流式响应、Gradio 界面 | Week 5 |
| [`09-week6-monitoring-caching.md`](09-week6-monitoring-caching.md) | Week 6：Langfuse 全链路追踪、Redis 缓存、生产可观测性 | Week 6 |
| [`10-week7-agentic-telegram.md`](10-week7-agentic-telegram.md) | Week 7：LangGraph 智能体、文档评分、查询重写、Telegram 机器人 | Week 7 |
| [`11-testing.md`](11-testing.md) | 测试套件、conftest、运行命令、验证方法 | 全局 |
| [`12-run-build-deploy-rollback.md`](12-run-build-deploy-rollback.md) | 本地运行、构建、部署、回滚步骤 | 运维 |
| [`13-troubleshooting.md`](13-troubleshooting.md) | 常见问题排查手册 | 运维 |
| [`14-quality-performance-security.md`](14-quality-performance-security.md) | 性能、安全性、可维护性分析与已知缺陷 | 分析 |
| [`15-cicd-and-maintenance.md`](15-cicd-and-maintenance.md) | CI/CD 流水线、长期维护、依赖管理建议 | 工程 |
| [`16-upstream-differences-and-fixes.md`](16-upstream-differences-and-fixes.md) | 附录：与上游仓库的差异、本教程修复的缺陷、源文件覆盖清单 | 附录 |

---

## ⚠️ 关于"可运行"与本教程对上游缺陷的修复

本教程对照的是真实开源项目 `arxiv-paper-curator`。在编写时，我们**逐文件审计了 `main` 分支的源码**，发现 `main` 上有几处**已提交但无法运行**的缺陷（主要在 Week 5–7 引入的追踪层和智能体层）。

由于本教程的硬性要求是"**所有代码必须完整、可运行**"，我们采取如下原则（在第 [16](16-upstream-differences-and-fixes.md) 章有完整清单）：

1. **绝大多数代码逐字复刻上游**，保证你复现的就是这个项目本身。
2. **凡是上游缺失或写错、导致无法运行的地方，本教程补全为可运行版本**，并在出现处用如下标记清楚说明：

> ⚠️ **上游修复**：上游 `main` 在此处缺少 `Xxx` 方法/写错了签名，运行时会抛 `AttributeError`/`TypeError`。下面给出补全后的可运行实现，原因见行内说明。

本教程锁定的修复集合（**仅此 6 类"让上游可运行"的修复，不做任何额外"改进"**）：

| # | 位置 | 问题 | 修复 |
|---|------|------|------|
| 1 | `OllamaClient`（第 [08](08-week5-rag-llm.md) 章） | 智能体节点调用 `get_langchain_model()`，但该方法从未实现 | 补全：返回一个 `ChatOllama` 实例 |
| 2 | `LangfuseTracer`（第 [09](09-week6-monitoring-caching.md) 章） | `RAGTracer` 与智能体节点调用 `trace_rag_request()` / `create_span()` / `end_span()`，v3 重构后这些方法已不存在 | 补全这 3 个方法，且在未启用 Langfuse 时安全降级为 no-op |
| 3 | `TextChunker._reconstruct_text`（第 [07](07-week4-hybrid-search.md) 章） | 定义只接收 1 个参数，却被以 2 个参数调用，短文本分支会抛 `TypeError` | 修正调用处为单参数 |
| 4 | 智能体单元测试（第 [11](11-testing.md) 章） | 测试 mock 的是 `create_llm`，而节点实际调用 `get_langchain_model`，命名不一致；且缺失夹具 | 统一为 `get_langchain_model`，并补全 `conftest.py` 夹具 |
| 5 | `dependencies.py`（第 [10](10-week7-agentic-telegram.md) 章） | `get_agentic_rag_service` 给 `make_agentic_rag_service` 传了它不接受的 `model=` 参数，`/ask-agentic` 每次请求都会抛 `TypeError` | 删去多余的 `model=` 参数 |
| 6 | `PDFParserService.parse_pdf`（第 [05](05-week2-ingestion.md) 章） | `await self.docling_parser.parse_pdf(...)`，但 `DoclingParser.parse_pdf` 是同步方法，`await` 同步返回值抛 `TypeError`，整条 PDF 解析链路崩溃 | 去掉那个多余的 `await` |

此外，`src/schemas/common/__init__.py` 与 `src/schemas/telegram/__init__.py` 在上游引用了**不存在的子模块**（`src.schemas.search.hybrid`、`.commands` 等），但它们从未被运行代码导入；本教程将其创建为空包标记（详见第 [16](16-upstream-differences-and-fixes.md) 章）。

> **诚实声明**：本教程的代码是**逐文件审阅源码 + 在 `.venv` 中核对了关键第三方 API（如 `ChatOllama` 的构造参数）后写成的，并修正了上游会导致崩溃的缺陷**。但由于本系统需要 Docker、OpenSearch、Ollama、PostgreSQL、外部 Jina 密钥与 Telegram Token 才能整体跑起来，**这些不是在编写环境里逐一端到端跑通后才写出的**。换句话说：代码是"完整 + 按可运行标准编写 + 修正了上游崩溃点"，而不是"已在生产中端到端验证"。每一步我们都给出了**你自己可执行的验证命令**，请按第 [11](11-testing.md)、[12](12-run-build-deploy-rollback.md) 章实际验证。

---

## 最快上手路径（先跑起来，再读原理）

如果你想先看到系统跑起来，再回头精读：

```bash
# 1. 准备一个空目录并初始化（详见第 02、04 章）
mkdir arxiv-paper-curator && cd arxiv-paper-curator
git init

# 2. 按第 02 章安装 uv、Docker，并准备 .env（含必需密钥）
#    然后按第 04–10 章逐文件创建源码

# 3. 安装依赖
uv sync

# 4. 启动全部服务
docker compose up --build -d

# 5. 验证 API 健康
curl http://localhost:8000/api/v1/health
```

但**强烈建议按 01 → 16 的顺序通读**，因为每一周都建立在前一周之上，跳读会缺少上下文。

---

## 学习路线图（一句话看懂每周在做什么）

```
Week 1  基础设施     →  Docker 把 5 个服务编排起来，FastAPI 提供健康检查
Week 2  数据摄取     →  从 arXiv 抓论文，用 Docling 解析 PDF，存进 PostgreSQL
Week 3  关键词检索   →  把内容索引进 OpenSearch，实现 BM25 关键词搜索（打地基）
Week 4  混合检索     →  分块 + Jina 向量嵌入 + RRF 融合，关键词与语义结合
Week 5  完整 RAG     →  本地 Ollama 大模型 + 检索拼提示词 + 流式回答 + Gradio UI
Week 6  监控与缓存   →  Langfuse 全链路追踪 + Redis 精确缓存，迈向生产
Week 7  Agentic RAG  →  LangGraph 智能体（护栏/评分/重写）+ Telegram 机器人
```

准备好了吗？从 [`01-project-overview.md`](01-project-overview.md) 开始。
