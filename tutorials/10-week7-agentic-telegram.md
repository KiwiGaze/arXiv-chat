# File: tutorials/10-week7-agentic-telegram.md

# 第 10 章　Week 7：Agentic RAG（LangGraph）与 Telegram 机器人

**本周目标**：用 **LangGraph** 把 RAG 升级为会"思考"的智能体——它会先判断问题是否在领域内（护栏）、检索、给文档打分、必要时重写查询再检索、最后生成答案，并返回**推理步骤**。再接上 **Telegram 机器人**实现移动端访问。最后给出 `main.py` 与 `dependencies.py` 的**最终完整版**。

> **依赖前置修复**：本章的智能体节点会调用 `ollama_client.get_langchain_model(...)`（修复 #1，第 [08](08-week5-rag-llm.md) 章）和 `langfuse_tracer.create_span/end_span(...)`（修复 #2，第 [09](09-week6-monitoring-caching.md) 章）。请确保这两处已就位，否则 Agentic RAG 会抛 `AttributeError`。

---

## 10.1 LangGraph 工作流总览

```
        START
          │
          ▼
     ┌─────────┐  guardrail：LLM 给问题打分(0-100)，是否 CS/AI/ML 研究领域
     │guardrail│
     └────┬────┘
   score≥阈值│ score<阈值
     ┌──────┴───────┐
     ▼              ▼
 ┌────────┐    ┌────────────┐
 │retrieve│    │out_of_scope│→ END（礼貌拒答）
 └───┬────┘    └────────────┘
     │ 产生 tool_call（或达最大次数→END 兜底）
     ▼
 ┌─────────────┐  tool_retrieve：embed+search→Documents
 │tool_retrieve│
 └──────┬──────┘
        ▼
 ┌──────────────┐  grade_documents：LLM 判定相关性
 │grade_documents│
 └──────┬───────┘
 相关   │   不相关
  ┌─────┴──────┐
  ▼            ▼
┌──────────────┐  ┌─────────────┐
│generate_answer│  │rewrite_query│→ 回到 retrieve（受 max_retrieval_attempts 限制）
└──────┬───────┘  └─────────────┘
       ▼
      END
```

**五个决策能力**：护栏（领域边界）、检索（带次数上限）、文档评分（相关性）、查询重写（自适应）、答案生成。每个节点都是一个轻量异步函数，通过 `Runtime[Context]` 拿到依赖（LLM、检索、嵌入、追踪）。

> **为什么用"状态图 + 节点 + 条件边"而不是一串 if/else？**（呼应第 [03](03-architecture-and-design.md) 章决策 7）：决策流程显式、可视化（能导出 mermaid/PNG）、与 Langfuse 追踪天然集成、易扩展新节点。

---

## 10.2 智能体数据模型：`src/services/agents/models.py`

```bash
mkdir -p src/services/agents/nodes
touch src/services/agents/nodes/__init__.py   # 稍后写入内容
```

### 文件：`src/services/agents/models.py`（逐字复制）

```python
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class GuardrailScoring(BaseModel):
    """Scoring result of a user query for guardrail validation.

    :param score: Relevance score between 0 and 100
    :param reason: Brief explanation for the score
    """

    score: int = Field(ge=0, le=100, description="Relevance score between 0 and 100")
    reason: str = Field(description="Brief reason for the score")


class GradeDocuments(BaseModel):
    """Binary score for document relevance check.

    :param binary_score: Relevance score: 'yes' or 'no'
    :param reasoning: Explanation for the relevance decision
    """

    binary_score: Literal["yes", "no"] = Field(description="Document relevance: 'yes' or 'no'")
    reasoning: str = Field(default="", description="Explanation for the decision")


class SourceItem(BaseModel):
    """Source item from retrieved documents.

    :param arxiv_id: arXiv paper ID
    :param title: Paper title
    :param authors: List of authors
    :param url: Link to the paper
    :param relevance_score: Relevance score from retrieval
    """

    arxiv_id: str = Field(description="arXiv paper ID")
    title: str = Field(description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    url: str = Field(description="Link to paper")
    relevance_score: float = Field(default=0.0, description="Relevance score from search")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "relevance_score": self.relevance_score,
        }


class ToolArtefact(BaseModel):
    """Artifact returned by tool calls with metadata.

    :param tool_name: Name of the tool that generated this artifact
    :param tool_call_id: Unique ID of the tool call
    :param content: The actual content/result from the tool
    :param metadata: Additional metadata about the tool execution
    """

    tool_name: str = Field(description="Name of the tool")
    tool_call_id: str = Field(description="Unique tool call ID")
    content: Any = Field(description="Tool result content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RoutingDecision(BaseModel):
    """Routing decision for graph navigation.

    :param route: The next node to route to
    :param reason: Explanation for the routing decision
    """

    route: Literal["retrieve", "out_of_scope", "generate_answer", "rewrite_query"] = Field(
        description="Next node to route to"
    )
    reason: str = Field(default="", description="Reason for routing decision")


class GradingResult(BaseModel):
    """Result of document grading with details.

    :param document_id: Identifier for the graded document
    :param is_relevant: Whether document is relevant
    :param score: Relevance score
    :param reasoning: Explanation for the grade
    """

    document_id: str = Field(description="Document identifier")
    is_relevant: bool = Field(description="Relevance flag")
    score: float = Field(default=0.0, description="Relevance score")
    reasoning: str = Field(default="", description="Grading reasoning")


class ReasoningStep(BaseModel):
    """A reasoning step in the agent workflow.

    :param step_name: Name of the step/node
    :param description: Human-readable description
    :param metadata: Additional step metadata
    """

    step_name: str = Field(description="Name of the reasoning step")
    description: str = Field(description="Human-readable description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Step metadata")
```

> `GuardrailScoring`、`GradeDocuments` 等会作为 LLM 的**结构化输出 schema**（`llm.with_structured_output(GuardrailScoring)`），强制小模型按字段产出。

---

## 10.3 图状态：`src/services/agents/state.py`

### 文件：`src/services/agents/state.py`（逐字复制）

```python
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .models import GradingResult, GuardrailScoring, RoutingDecision, SourceItem, ToolArtefact


class AgentState(TypedDict):
    """State class for the Agentic RAG workflow.

    TypedDict-based state following LangGraph 2025 best practices.
    Tracks all data that needs to be passed between nodes.

    :cvar messages:
        List of messages in the conversation. Uses add_messages reducer
        to append new messages rather than overwrite.
    :type messages: Annotated[list[AnyMessage], add_messages]

    :cvar original_query:
        The original user query before any rewrites.
    :type original_query: Optional[str]

    :cvar rewritten_query:
        The rewritten query after optimization for better retrieval.
    :type rewritten_query: Optional[str]

    :cvar retrieval_attempts:
        Number of retrieval attempts made (for max attempt tracking).
    :type retrieval_attempts: int

    :cvar guardrail_result:
        Result from guardrail validation with score and reasoning.
    :type guardrail_result: Optional[GuardrailScoring]

    :cvar routing_decision:
        The routing decision determining the next node in the graph.
    :type routing_decision: Optional[RoutingDecision]

    :cvar sources:
        Dictionary mapping tool_call_id to their output sources.
    :type sources: Optional[Dict[str, Any]]

    :cvar relevant_sources:
        List of relevant sources to display to the user.
    :type relevant_sources: List[SourceItem]

    :cvar relevant_tool_artefacts:
        List of tool artifacts with metadata from tool executions.
    :type relevant_tool_artefacts: Optional[List[ToolArtefact]]

    :cvar grading_results:
        List of grading results for each retrieved document.
    :type grading_results: List[GradingResult]

    :cvar metadata:
        Runtime metadata for tracing and analytics.
    :type metadata: Dict[str, Any]
    """

    messages: Annotated[list[AnyMessage], add_messages]
    original_query: Optional[str]
    rewritten_query: Optional[str]
    retrieval_attempts: int
    guardrail_result: Optional[GuardrailScoring]
    routing_decision: Optional[RoutingDecision]
    sources: Optional[Dict[str, Any]]
    relevant_sources: List[SourceItem]
    relevant_tool_artefacts: Optional[List[ToolArtefact]]
    grading_results: List[GradingResult]
    metadata: Dict[str, Any]
```

> **`messages: Annotated[list[AnyMessage], add_messages]`** 是 LangGraph 的精髓：`add_messages` 是一个 reducer，节点返回 `{"messages": [新消息]}` 时会**追加**而不是覆盖。这样多轮检索/重写的消息历史能自然累积。

---

## 10.4 运行时上下文与配置

### 文件：`src/services/agents/context.py`（逐字复制）

```python
from dataclasses import dataclass
from langfuse._client.span import LangfuseSpan
from typing import TYPE_CHECKING, Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient


@dataclass
class Context:
    """Runtime context for agent dependencies.

    This contains immutable dependencies that nodes need but don't modify.

    :param ollama_client: Client for LLM generation
    :param opensearch_client: Client for document search
    :param embeddings_client: Client for embeddings
    :param langfuse_tracer: Optional tracer for observability
    :param trace: Current Langfuse trace object (if enabled)
    :param langfuse_enabled: Whether Langfuse tracing is enabled
    :param model_name: Model to use for LLM calls
    :param temperature: Temperature for generation
    :param top_k: Number of documents to retrieve
    :param max_retrieval_attempts: Maximum retrieval attempts
    :param guardrail_threshold: Threshold for guardrail validation (0-100)
    """

    ollama_client: OllamaClient
    opensearch_client: OpenSearchClient
    embeddings_client: JinaEmbeddingsClient
    langfuse_tracer: Optional[LangfuseTracer]
    trace: Optional["LangfuseSpan"] = None
    langfuse_enabled: bool = False
    model_name: str = "gemma4:e2b"
    temperature: float = 0.0
    top_k: int = 3
    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 60
```

> `Context` 是**依赖注入容器**：节点通过 `runtime.context.xxx` 拿到客户端，但不修改它们（不可变依赖）。这与 `AgentState`（可变状态）分离——清晰区分"依赖"与"数据"。

### 文件：`src/services/agents/config.py`（逐字复制）

```python
from typing import Any, Dict

from pydantic import BaseModel, Field

from src.config import Settings, get_settings


class GraphConfig(BaseModel):
    """Configuration for the entire graph execution.

    This is the configuration used by AgenticRAGService for controlling
    graph behavior, retrieval settings, and execution parameters.

    :param max_retrieval_attempts: Maximum number of retrieval attempts before fallback
    :param guardrail_threshold: Threshold score for guardrail validation (0-100)
    :param model: Default model to use for LLM calls (e.g., "gemma4:e2b")
    :param temperature: Temperature for LLM generation (0.0 = deterministic)
    :param top_k: Number of documents to retrieve from search
    :param use_hybrid: Whether to use hybrid search (BM25 + vector)
    :param enable_tracing: Whether to enable Langfuse tracing
    :param metadata: Additional runtime metadata for tracking and analytics
    :param settings: Application settings instance for environment and service config
    """

    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 60
    model: str = "gemma4:e2b"
    temperature: float = 0.0
    top_k: int = 3
    use_hybrid: bool = True
    enable_tracing: bool = True
    metadata: Dict[str, Any] = {}
    settings: Settings = Field(default_factory=get_settings)
```

---

## 10.5 提示词：`src/services/agents/prompts.py`

### 文件：`src/services/agents/prompts.py`（逐字复制）

```python
# Grade documents for relevance (used in grade_documents_node)
GRADE_DOCUMENTS_PROMPT = """You are a grader assessing relevance of retrieved documents to a user question.

Retrieved Documents:
{context}

User Question: {question}

If the documents contain keywords or semantic meaning related to the question, grade them as relevant.
Give a binary score 'yes' or 'no' to indicate whether the documents are relevant to the question.
Also provide brief reasoning for your decision.

Respond in JSON format with 'binary_score' (yes/no) and 'reasoning' fields."""

# Rewrite query for better retrieval
REWRITE_PROMPT = """You are a question re-writer that converts an input question to a better version that is optimized for retrieving relevant documents.

Look at the initial question and try to reason about the underlying semantic intent or meaning.

Here is the initial question:
{question}

Formulate an improved question that will retrieve more relevant documents.
Provide only the improved question without any preamble or explanation."""

# System message for query generation/response
SYSTEM_MESSAGE = """You are an AI assistant specializing in academic research papers from arXiv.
Your domain of expertise is: Computer Science, Machine Learning, AI, and related technical research.

You have access to a tool to retrieve relevant research papers. Use this tool when:
- The user asks about specific research topics in CS/AI/ML
- The question requires knowledge from academic papers (e.g., "What are transformer architectures?")
- You need context from scientific literature (e.g., "How does BERT work?")

Do NOT use the tool when:
- The question is about general knowledge unrelated to research (e.g., "What is the meaning of dog?")
- The question is simple factual or mathematical (e.g., "what is 2+2?")
- The question is conversational, greeting, or personal
- The question is about topics outside CS/AI/ML research (e.g., cooking, history, medicine)

When you use the retrieval tool, you will receive relevant paper excerpts to help answer the question."""

# Decision prompt for routing
DECISION_PROMPT = """You are an AI assistant that ONLY helps with academic research papers from arXiv in Computer Science, AI, and Machine Learning.

Question: "{question}"

Is this question about CS/AI/ML research that requires academic papers?

CRITICAL RULES:
- RETRIEVE: ONLY if the question is specifically about AI/ML/CS research topics (neural networks, algorithms, models, techniques)
- RESPOND: For EVERYTHING else (general knowledge, definitions, greetings, non-research questions)

Examples:
- "What are transformer architectures in deep learning?" -> RETRIEVE
- "Explain BERT model" -> RETRIEVE
- "What is the meaning of dog?" -> RESPOND (general dictionary definition)
- "What is a dog?" -> RESPOND (not about research)
- "Hello" -> RESPOND (greeting)
- "What is 2+2?" -> RESPOND (math, not research)

Answer with ONLY ONE WORD: "RETRIEVE" or "RESPOND"

Your answer:"""

# Direct response prompt (no retrieval)
DIRECT_RESPONSE_PROMPT = """You are an AI assistant specializing in academic research papers from arXiv (Computer Science, AI, ML).

The following question appears to be outside the scope of academic research papers or doesn't require retrieval from research literature:

Question: {question}

Explain that this question is outside your domain of expertise (arXiv research papers in CS/AI/ML) and that you cannot answer it accurately. Be helpful by suggesting what kind of resource would be more appropriate for this question.

Answer:"""

# Guardrail validation prompt (used in guardrail_node)
GUARDRAIL_PROMPT = """You are a guardrail evaluator assessing whether a user query is within the scope of academic research papers from arXiv in Computer Science, AI, and Machine Learning.

User Query: {question}

Evaluate whether this query is:
- About CS/AI/ML research topics (neural networks, algorithms, models, architectures, techniques, etc.)
- Requires academic paper knowledge to answer
- Within the domain of Computer Science research

Assign a relevance score (0-100):
- 80-100: Clearly about CS/AI/ML research (e.g., "What are transformer architectures?", "How does BERT work?")
- 60-79: Potentially research-related but unclear (e.g., "Tell me about attention mechanisms")
- 40-59: Borderline or ambiguous (e.g., "What is machine learning?")
- 0-39: NOT about research papers (e.g., "What is a dog?", "Hello", "What is 2+2?")

Provide:
1. A score between 0 and 100
2. A brief reason explaining why you gave this score

Respond in JSON format with 'score' (integer 0-100) and 'reason' (string) fields."""

# Answer generation prompt (used in generate_answer_node)
GENERATE_ANSWER_PROMPT = """You are an AI research assistant specializing in academic papers from arXiv in Computer Science, AI, and Machine Learning.

Your task is to answer the user's question using ONLY the information from the retrieved research papers provided below.

Retrieved Research Papers:
{context}

User Question: {question}

Instructions:
- Provide a comprehensive, accurate answer based ONLY on the retrieved papers
- Cite specific papers when making claims (use paper titles or arxiv IDs)
- If the papers don't contain enough information to fully answer the question, acknowledge this
- Structure your answer clearly and professionally
- Focus on the key insights and findings from the papers
- Do NOT make up information or cite papers not in the retrieved context

Answer:"""
```

---

## 10.6 检索工具：`src/services/agents/tools.py`

### 文件：`src/services/agents/tools.py`（逐字复制）

```python
import logging

from langchain_core.documents import Document
from langchain_core.tools import tool

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


def create_retriever_tool(
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    top_k: int = 3,
    use_hybrid: bool = True,
):
    """Create a retriever tool that wraps OpenSearch service.

    :param opensearch_client: Existing OpenSearch service
    :param embeddings_client: Existing Jina embeddings service
    :param top_k: Number of chunks to retrieve
    :param use_hybrid: Use hybrid search (BM25 + vector)
    :returns: LangChain tool for retrieving papers
    """

    @tool
    async def retrieve_papers(query: str) -> list[Document]:
        """Search and return relevant arXiv research papers.

        Use this tool when the user asks about:
        - Machine learning concepts or techniques
        - Deep learning architectures
        - Natural language processing
        - Computer vision methods
        - AI research topics
        - Specific algorithms or models

        :param query: The search query describing what papers to find
        :returns: List of relevant paper excerpts with metadata
        """
        logger.info(f"Retrieving papers for query: {query[:100]}...")
        logger.debug(f"Search mode: {'hybrid' if use_hybrid else 'bm25'}, top_k: {top_k}")

        # Generate query embedding
        logger.debug("Generating query embedding")
        query_embedding = await embeddings_client.embed_query(query)
        logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")

        # Search using OpenSearch
        logger.debug("Searching OpenSearch")
        search_results = opensearch_client.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=top_k,
            use_hybrid=use_hybrid,
        )

        # Convert SearchHit to LangChain Document
        documents = []
        hits = search_results.get("hits", [])
        logger.info(f"Found {len(hits)} documents from OpenSearch")

        for hit in hits:
            doc = Document(
                page_content=hit["chunk_text"],
                metadata={
                    "arxiv_id": hit["arxiv_id"],
                    "title": hit.get("title", ""),
                    "authors": hit.get("authors", ""),
                    "score": hit.get("score", 0.0),
                    "source": f"https://arxiv.org/pdf/{hit['arxiv_id']}.pdf",
                    "section": hit.get("section_name", ""),
                    "search_mode": "hybrid" if use_hybrid else "bm25",
                    "top_k": top_k,
                },
            )
            documents.append(doc)

        logger.debug(f"Converted {len(documents)} hits to LangChain Documents")
        logger.info(f"✓ Retrieved {len(documents)} papers successfully")

        return documents

    return retrieve_papers
```

> 这个工具用 `@tool` 装饰，被 `ToolNode` 执行。它复用 Week 4 的混合检索，把 `SearchHit` 转成 LangChain `Document`，供后续节点（评分、生成）读取。

---

## 10.7 节点工具函数：`src/services/agents/nodes/utils.py`

### 文件：`src/services/agents/nodes/utils.py`（逐字复制）

```python
import logging
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..models import ReasoningStep, SourceItem, ToolArtefact

logger = logging.getLogger(__name__)


def extract_sources_from_tool_messages(messages: List) -> List[SourceItem]:
    """Extract sources from tool messages in conversation.

    :param messages: List of messages from graph state
    :returns: List of SourceItem objects
    """
    sources = []

    for msg in messages:
        if isinstance(msg, ToolMessage) and hasattr(msg, "name"):
            if msg.name == "retrieve_papers":
                # Parse tool response for sources
                # This would need to parse the actual document metadata
                # For now, return empty list
                pass

    return sources


def extract_tool_artefacts(messages: List) -> List[ToolArtefact]:
    """Extract tool artifacts from messages.

    :param messages: List of messages from graph state
    :returns: List of ToolArtefact objects
    """
    artefacts = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            artefact = ToolArtefact(
                tool_name=getattr(msg, "name", "unknown"),
                tool_call_id=getattr(msg, "tool_call_id", ""),
                content=msg.content,
                metadata={},
            )
            artefacts.append(artefact)

    return artefacts


def create_reasoning_step(
    step_name: str,
    description: str,
    metadata: Optional[Dict] = None,
) -> ReasoningStep:
    """Create a reasoning step record.

    :param step_name: Name of the step/node
    :param description: Human-readable description
    :param metadata: Additional metadata
    :returns: ReasoningStep object
    """
    return ReasoningStep(
        step_name=step_name,
        description=description,
        metadata=metadata or {},
    )


def filter_messages(messages: List) -> List[AIMessage | HumanMessage]:
    """Filter messages to include only HumanMessage and AIMessage types.

    Excludes tool messages and other internal message types.

    :param messages: List of messages to filter
    :returns: Filtered list of messages
    """
    return [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]


def get_latest_query(messages: List) -> str:
    """Get the latest user query from messages.

    :param messages: List of messages
    :returns: Latest query text
    :raises ValueError: If no user query found
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content

    raise ValueError("No user query found in messages")


def get_latest_context(messages: List) -> str:
    """Get the latest context from tool messages.

    :param messages: List of messages
    :returns: Latest context text or empty string
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg.content if hasattr(msg, "content") else ""

    return ""
```

---

## 10.8 节点：护栏 `src/services/agents/nodes/guardrail_node.py`

### 文件：`src/services/agents/nodes/guardrail_node.py`（逐字复制）

```python
import logging
import time
from typing import Dict, Literal

from langgraph.runtime import Runtime

from ..context import Context
from ..models import GuardrailScoring
from ..prompts import GUARDRAIL_PROMPT
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


def continue_after_guardrail(state: AgentState, runtime: Runtime[Context]) -> Literal["continue", "out_of_scope"]:
    """Determine whether to continue or reject based on guardrail results.

    This function checks the guardrail_result score against a threshold.
    If the score is above threshold, continue; otherwise route to out_of_scope.

    :param state: Current agent state with guardrail results
    :param runtime: Runtime context containing guardrail threshold
    :returns: "continue" if score >= threshold, "out_of_scope" otherwise
    """
    guardrail_result = state.get("guardrail_result")
    if not guardrail_result:
        logger.warning("No guardrail result found, defaulting to continue")
        return "continue"

    score = guardrail_result.score
    threshold = runtime.context.guardrail_threshold

    logger.info(f"Guardrail score: {score}, threshold: {threshold}")

    return "continue" if score >= threshold else "out_of_scope"


async def ainvoke_guardrail_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, GuardrailScoring]:
    """Asynchronously invoke the guardrail validation step using LLM.

    This function evaluates whether the user query is within scope
    (CS/AI/ML research papers) and assigns a score using an LLM.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with guardrail_result
    """
    logger.info("NODE: guardrail_validation")
    start_time = time.time()

    # Get the latest user query
    query = get_latest_query(state["messages"])
    logger.debug(f"Evaluating query: {query[:100]}...")

    # Create span for guardrail validation (v2 SDK)
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="guardrail_validation",
                input_data={
                    "query": query,
                    "threshold": runtime.context.guardrail_threshold,
                },
                metadata={
                    "node": "guardrail",
                    "model": runtime.context.model_name,
                },
            )
            logger.debug("Created Langfuse span for guardrail validation (v2 SDK)")
        except Exception as e:
            logger.warning(f"Failed to create span for guardrail validation: {e}")

    try:
        # Create guardrail prompt from template
        guardrail_prompt = GUARDRAIL_PROMPT.format(question=query)

        # Get LLM from runtime context
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.0,
        )

        # Create structured output LLM for guardrail scoring
        structured_llm = llm.with_structured_output(GuardrailScoring)

        # Invoke LLM for guardrail evaluation
        logger.info("Invoking LLM for guardrail validation")
        response = await structured_llm.ainvoke(guardrail_prompt)

        logger.info(f"Guardrail result - Score: {response.score}, Reason: {response.reason}")

        # Update span with successful result
        if span:
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            runtime.context.langfuse_tracer.end_span(
                span,
                output={
                    "score": response.score,
                    "reason": response.reason,
                    "decision": "continue" if response.score >= runtime.context.guardrail_threshold else "out_of_scope",
                },
                metadata={
                    "execution_time_ms": execution_time,
                    "threshold": runtime.context.guardrail_threshold,
                },
            )

    except Exception as e:
        logger.error(f"LLM guardrail validation failed: {e}, falling back to default")

        # Fallback to a conservative default if LLM fails
        response = GuardrailScoring(
            score=50,
            reason=f"LLM validation failed, using conservative default: {str(e)}"
        )

        # Update span with error
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_span(
                span,
                output={"score": response.score, "reason": response.reason, "error": str(e)},
                metadata={"execution_time_ms": execution_time, "fallback": True},
                level="WARNING",
            )
            runtime.context.langfuse_tracer.end_span(span)

    return {"guardrail_result": response}
```

> **护栏 + 兜底**：用 LLM 给问题打 0–100 分。`continue_after_guardrail` 按阈值（默认 60）路由：≥60 进检索，否则去 `out_of_scope`。LLM 失败时回退到保守默认分 50（兜底，不崩）。`get_langchain_model`（修复 #1）+ `create_span/end_span`（修复 #2）在此被调用。

---

## 10.9 节点：领域外回复 `out_of_scope_node.py`

### 文件：`src/services/agents/nodes/out_of_scope_node.py`（逐字复制）

```python
import logging
from typing import Dict, List

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_out_of_scope_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Handle out-of-scope queries with a helpful message.

    This node responds to queries that are outside the domain of
    CS/AI/ML research papers with a polite, informative message.

    :param state: Current agent state
    :param runtime: Runtime context (not used in this node)
    :returns: Dictionary with messages containing the out-of-scope response
    """
    logger.info("NODE: out_of_scope")

    question = get_latest_query(state["messages"])

    # Generate helpful response message
    response_text = (
        "I apologize, but I can only help with questions about academic research papers "
        "in Computer Science, Artificial Intelligence, and Machine Learning from arXiv.\n\n"
        f"Your question: '{question}'\n\n"
        "This appears to be outside my domain of expertise. For questions like this, you might want to try:\n"
        "- General-purpose AI assistants for broad knowledge questions\n"
        "- Domain-specific resources for topics outside CS/AI/ML\n"
        "- Technical documentation if asking about specific software/tools\n\n"
        "If you have a question about AI/ML research papers, I'd be happy to help!"
    )

    logger.info("Responding with out-of-scope message")

    return {"messages": [AIMessage(content=response_text)]}
```

---

## 10.10 节点：检索发起 `retrieve_node.py`

### 文件：`src/services/agents/nodes/retrieve_node.py`（逐字复制）

```python
import logging
import time
from typing import Dict, Union

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_retrieve_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Union[int, str, list]]:
    """Initiate retrieval or return fallback if max attempts reached.

    This node creates a tool call to retrieve documents, or returns a fallback
    message if the maximum number of retrieval attempts has been reached.

    :param state: Current agent state
    :param runtime: Runtime context containing max_retrieval_attempts
    :returns: Dictionary with updated state (retrieval_attempts, messages, original_query)
    """
    logger.info("NODE: retrieve")
    start_time = time.time()

    messages = state["messages"]
    question = get_latest_query(messages)
    current_attempts = state.get("retrieval_attempts", 0)

    # Get max attempts from context
    max_attempts = runtime.context.max_retrieval_attempts

    # Store original query if not set
    updates = {}
    if state.get("original_query") is None:
        updates["original_query"] = question
        logger.debug(f"Stored original query: {question[:100]}...")

    # Create span for retrieval initiation
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="document_retrieval_initiation",
                input_data={
                    "query": question,
                    "attempt": current_attempts + 1,
                    "max_attempts": max_attempts,
                },
                metadata={
                    "node": "retrieve",
                    "top_k": runtime.context.top_k,
                },
            )
            logger.debug(f"Created Langfuse span for retrieval attempt {current_attempts + 1}")
        except Exception as e:
            logger.warning(f"Failed to create span for retrieve node: {e}")

    # Check if max attempts reached
    if current_attempts >= max_attempts:
        logger.warning(f"Max retrieval attempts ({max_attempts}) reached")
        fallback_msg = (
            f"I apologize, but I couldn't find relevant research papers after {max_attempts} attempts.\n"
            "This may be because:\n"
            "1. No papers in the database contain relevant information\n"
            "2. The query terms don't match the indexed content\n\n"
            "Please try rephrasing your question with more specific technical terms."
        )

        # Update span with max attempts reached
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"status": "max_attempts_reached", "fallback": True},
                metadata={"execution_time_ms": execution_time},
            )

        return {**updates, "messages": [AIMessage(content=fallback_msg)]}

    # Increment retrieval attempts
    new_attempt_count = current_attempts + 1
    updates["retrieval_attempts"] = new_attempt_count
    logger.info(f"Retrieval attempt {new_attempt_count}/{max_attempts}")

    # Create tool call for retrieval
    updates["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"retrieve_{new_attempt_count}",
                    "name": "retrieve_papers",
                    "args": {"query": question},
                }
            ],
        )
    ]

    logger.debug(f"Created tool call for query: {question[:100]}...")

    # Update span with successful tool call creation
    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "status": "tool_call_created",
                "query": question,
                "attempt": new_attempt_count,
            },
            metadata={"execution_time_ms": execution_time},
        )

    return updates
```

> **检索次数上限**：`retrieve` 节点产生一个 `tool_calls`（让 `tool_retrieve` 去执行检索）。每进一次自增计数；超过 `max_retrieval_attempts`（默认 2）就返回兜底消息并结束，**防止"重写→检索"无限循环**（这是 Agentic 系统的关键安全阀，呼应全局 CLAUDE.md 的"无限递归风险"原则）。

---

## 10.11 节点：文档评分 `grade_documents_node.py`

### 文件：`src/services/agents/nodes/grade_documents_node.py`（逐字复制）

```python
import logging
import time
from typing import Dict

from langgraph.runtime import Runtime

from ..context import Context
from ..models import GradeDocuments, GradingResult
from ..prompts import GRADE_DOCUMENTS_PROMPT
from ..state import AgentState
from .utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_grade_documents_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, str | list]:
    """Grade retrieved documents for relevance using LLM.

    This function uses an LLM to evaluate whether the retrieved documents
    are relevant to the user's query and decides whether to generate an
    answer or rewrite the query for better results.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with routing_decision and grading_results
    """
    logger.info("NODE: grade_documents")
    start_time = time.time()

    # Get query and context
    question = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])

    # Extract document chunks from context for logging
    chunks_preview = []
    if context:
        # Context is a string containing all documents concatenated
        # Let's show a preview of what was retrieved
        context_preview = context[:500] + "..." if len(context) > 500 else context
        chunks_preview = [{"text_preview": context_preview, "length": len(context)}]

    # Create span for document grading
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="document_grading",
                input_data={
                    "query": question,
                    "context_length": len(context) if context else 0,
                    "has_context": context is not None,
                    "chunks_received": chunks_preview,
                },
                metadata={
                    "node": "grade_documents",
                    "model": runtime.context.model_name,
                },
            )
            logger.debug("Created Langfuse span for document grading")
        except Exception as e:
            logger.warning(f"Failed to create span for grade_documents node: {e}")

    if not context:
        logger.warning("No context found, routing to rewrite_query")

        # Update span with no context result
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"routing_decision": "rewrite_query", "reason": "no_context"},
                metadata={"execution_time_ms": execution_time},
            )

        return {"routing_decision": "rewrite_query", "grading_results": []}

    logger.debug(f"Grading context of length {len(context)} characters")

    # Use LLM to grade document relevance
    try:
        # Create grading prompt from template
        grading_prompt = GRADE_DOCUMENTS_PROMPT.format(
            context=context,
            question=question,
        )

        # Get LLM from runtime context
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.0,
        )

        # Create structured output LLM for grading
        structured_llm = llm.with_structured_output(GradeDocuments)

        # Invoke LLM grading
        logger.info("Invoking LLM for document grading")
        grading_response = await structured_llm.ainvoke(grading_prompt)

        is_relevant = grading_response.binary_score == "yes"
        score = 1.0 if is_relevant else 0.0

        logger.info(f"LLM grading: score={grading_response.binary_score}, reasoning={grading_response.reasoning}")

        # Create grading result record
        grading_result = GradingResult(
            document_id="retrieved_docs",
            is_relevant=is_relevant,
            score=score,
            reasoning=grading_response.reasoning,
        )

    except Exception as e:
        logger.error(f"LLM grading failed: {e}, falling back to heuristic")
        # Fallback to simple heuristic if LLM fails
        is_relevant = len(context.strip()) > 50
        grading_result = GradingResult(
            document_id="retrieved_docs",
            is_relevant=is_relevant,
            score=1.0 if is_relevant else 0.0,
            reasoning=f"Fallback heuristic (LLM failed): {'sufficient content' if is_relevant else 'insufficient content'}",
        )

    # Determine routing
    route = "generate_answer" if is_relevant else "rewrite_query"

    logger.info(f"Grading result: {'relevant' if is_relevant else 'not relevant'}, routing to: {route}")

    # Update span with grading result
    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "routing_decision": route,
                "is_relevant": is_relevant,
                "score": score,
                "reasoning": grading_result.reasoning,
            },
            metadata={
                "execution_time_ms": execution_time,
                "context_length": len(context),
            },
        )

    return {
        "routing_decision": route,
        "grading_results": [grading_result],
    }
```

> **文档评分决定路由**：LLM 判定检索到的内容是否与问题相关（`yes`/`no`）。相关 → `generate_answer`；不相关 → `rewrite_query`。LLM 失败时用"内容长度>50"的启发式兜底。

---

## 10.12 节点：查询重写 `rewrite_query_node.py`

### 文件：`src/services/agents/nodes/rewrite_query_node.py`（逐字复制）

```python
import logging
import time
from typing import Dict, List

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from ..context import Context
from ..prompts import REWRITE_PROMPT
from ..state import AgentState

logger = logging.getLogger(__name__)


class QueryRewriteOutput(BaseModel):
    """Structured output for query rewriting."""

    rewritten_query: str = Field(
        description="The improved query optimized for document retrieval"
    )
    reasoning: str = Field(
        description="Brief explanation of how the query was improved"
    )


async def ainvoke_rewrite_query_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, str | List]:
    """Rewrite the original query for better document retrieval using LLM.

    This node uses an LLM to intelligently rewrite the user's query
    to improve the chances of finding relevant documents.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with rewritten_query and updated messages
    """
    logger.info("NODE: rewrite_query")
    start_time = time.time()

    # Get original query
    original_question = state.get("original_query") or state["messages"][0].content
    current_attempt = state.get("retrieval_attempts", 0)

    logger.debug(f"Rewriting query using LLM: {original_question[:100]}...")

    # Create span for query rewriting
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="query_rewriting",
                input_data={
                    "original_query": original_question,
                    "attempt": current_attempt,
                },
                metadata={
                    "node": "rewrite_query",
                    "strategy": "llm_based_expansion",
                    "model": runtime.context.model_name,
                },
            )
            logger.debug("Created Langfuse span for query rewriting")
        except Exception as e:
            logger.warning(f"Failed to create span for rewrite_query node: {e}")

    # Use LLM to rewrite the query intelligently
    try:
        # Create structured LLM for query rewriting
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.3,  # Lower temperature for more focused rewriting
        )
        structured_llm = llm.with_structured_output(QueryRewriteOutput)

        # Format prompt with original question
        prompt = REWRITE_PROMPT.format(question=original_question)

        logger.debug(f"Invoking LLM for query rewriting (model: {runtime.context.model_name})")
        llm_start = time.time()

        # Get rewritten query from LLM
        result: QueryRewriteOutput = await structured_llm.ainvoke(prompt)

        # Validate LLM output
        if not result or not result.rewritten_query:
            raise ValueError("LLM failed to return valid structured output for query rewriting")

        rewritten_query = result.rewritten_query.strip()
        if not rewritten_query:
            raise ValueError("LLM returned empty rewritten query")

        reasoning = result.reasoning

        llm_duration = time.time() - llm_start
        logger.info(
            f"Query rewritten in {llm_duration:.2f}s: "
            f"'{original_question[:50]}...' -> '{rewritten_query[:50]}...'"
        )
        logger.debug(f"Rewriting reasoning: {reasoning}")

    except Exception as e:
        logger.error(f"Failed to rewrite query using LLM: {e}")
        logger.warning("Falling back to simple keyword expansion")
        # Fallback to simple expansion if LLM fails
        rewritten_query = f"{original_question} research paper arxiv machine learning"
        reasoning = "Fallback: Simple keyword expansion due to LLM error"

    # Update span with rewriting result
    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "rewritten_query": rewritten_query,
                "reasoning": reasoning,
                "original_query": original_question,
            },
            metadata={
                "execution_time_ms": execution_time,
                "original_length": len(original_question),
                "rewritten_length": len(rewritten_query),
                "llm_duration_seconds": llm_duration if 'llm_duration' in locals() else None,
            },
        )

    return {
        "messages": [HumanMessage(content=rewritten_query)],
        "rewritten_query": rewritten_query,
    }
```

> **自适应重写**：评分不相关时，LLM 把原问题改写成更利于检索的版本（温度 0.3 更聚焦），作为新的 `HumanMessage` 加入消息流，回到 `retrieve` 再试一次。失败时用简单关键词扩展兜底。

---

## 10.13 节点：答案生成 `generate_answer_node.py`

### 文件：`src/services/agents/nodes/generate_answer_node.py`（逐字复制）

```python
import logging
import time
from typing import Dict, List

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..prompts import GENERATE_ANSWER_PROMPT
from ..state import AgentState
from .utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_generate_answer_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Generate final answer using retrieved documents.

    This node generates a comprehensive answer to the
    user's question based on the retrieved context using an LLM.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with messages containing the generated answer
    """
    logger.info("NODE: generate_answer")
    start_time = time.time()

    # Get question and context
    question = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])

    # Count sources from relevant_sources
    sources_count = len(state.get("relevant_sources", []))

    if not context:
        context = "No relevant documents found."
        logger.warning("No context available for answer generation")

    logger.debug(f"Generating answer for query: {question[:100]}...")
    logger.debug(f"Using context of length: {len(context)} characters")

    # Extract document chunks preview for logging
    chunks_preview = []
    if context:
        context_preview = context[:1000] + "..." if len(context) > 1000 else context
        chunks_preview = [{"text_preview": context_preview, "length": len(context)}]

    # Create span for answer generation
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="answer_generation",
                input_data={
                    "query": question,
                    "context_length": len(context),
                    "sources_count": sources_count,
                    "chunks_used": chunks_preview,
                },
                metadata={
                    "node": "generate_answer",
                    "model": runtime.context.model_name,
                    "temperature": runtime.context.temperature,
                },
            )
            logger.debug("Created Langfuse span for answer generation")
        except Exception as e:
            logger.warning(f"Failed to create span for generate_answer node: {e}")

    try:
        # Create answer generation prompt from template
        answer_prompt = GENERATE_ANSWER_PROMPT.format(
            context=context,
            question=question,
        )

        # Get LLM from runtime context
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=runtime.context.temperature,
        )

        # Invoke LLM for answer generation
        logger.info("Invoking LLM for answer generation")
        response = await llm.ainvoke(answer_prompt)

        # Extract content from response
        answer = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Generated answer of length: {len(answer)} characters")

        # Update span with successful result
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={
                    "answer_length": len(answer),
                    "sources_used": sources_count,
                },
                metadata={
                    "execution_time_ms": execution_time,
                    "context_length": len(context),
                },
            )

    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}, falling back to error message")

        # Fallback to error message if LLM fails
        answer = f"I apologize, but I encountered an error while generating the answer: {str(e)}\n\nPlease try again or rephrase your question."

        # Update span with error
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_span(
                span,
                output={"error": str(e), "fallback": True},
                metadata={"execution_time_ms": execution_time},
                level="ERROR",
            )
            runtime.context.langfuse_tracer.end_span(span)

    return {"messages": [AIMessage(content=answer)]}
```

### 节点包导出：`src/services/agents/nodes/__init__.py`

### 文件：`src/services/agents/nodes/__init__.py`（逐字复制）

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

---

## 10.14 图编排服务：`src/services/agents/agentic_rag.py`

### 文件：`src/services/agents/agentic_rag.py`（逐字复制）

```python
import logging
import time
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient

from .config import GraphConfig
from .context import Context
from .nodes import (
    ainvoke_generate_answer_step,
    ainvoke_grade_documents_step,
    ainvoke_guardrail_step,
    ainvoke_out_of_scope_step,
    ainvoke_retrieve_step,
    ainvoke_rewrite_query_step,
    continue_after_guardrail,
)
from .state import AgentState
from .tools import create_retriever_tool

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """Agentic RAG service 

    This implementation uses:
    - context_schema for dependency injection
    - Runtime[Context] for type-safe access in nodes
    - Direct client invocation (no pre-built runnables)
    - Lightweight nodes as pure functions
    """

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        ollama_client: OllamaClient,
        embeddings_client: JinaEmbeddingsClient,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        graph_config: Optional[GraphConfig] = None,
    ):
        """Initialize agentic RAG service.

        :param opensearch_client: Client for document search
        :param ollama_client: Client for LLM generation
        :param embeddings_client: Client for embeddings
        :param langfuse_tracer: Optional Langfuse tracer
        :param graph_config: Configuration for graph execution
        """
        self.opensearch = opensearch_client
        self.ollama = ollama_client
        self.embeddings = embeddings_client
        self.langfuse_tracer = langfuse_tracer
        self.graph_config = graph_config or GraphConfig()

        logger.info("Initializing AgenticRAGService with configuration:")
        logger.info(f"  Model: {self.graph_config.model}")
        logger.info(f"  Top-k: {self.graph_config.top_k}")
        logger.info(f"  Hybrid search: {self.graph_config.use_hybrid}")
        logger.info(f"  Max retrieval attempts: {self.graph_config.max_retrieval_attempts}")
        logger.info(f"  Guardrail threshold: {self.graph_config.guardrail_threshold}")

        # Build graph once (no runnables needed!)
        self.graph = self._build_graph()
        logger.info("✓ AgenticRAGService initialized successfully")

    def _build_graph(self):
        """Build and compile the LangGraph workflow.

        Uses context_schema for type-safe dependency injection.
        Nodes are lightweight functions that receive Runtime[Context].

        :returns: Compiled graph ready for invocation
        """
        logger.info("Building LangGraph workflow with context_schema")

        # Create workflow with AgentState and Context schema
        workflow = StateGraph(AgentState, context_schema=Context)

        # Create tools (these still need to be created upfront for ToolNode)
        retriever_tool = create_retriever_tool(
            opensearch_client=self.opensearch,
            embeddings_client=self.embeddings,
            top_k=self.graph_config.top_k,
            use_hybrid=self.graph_config.use_hybrid,
        )
        tools = [retriever_tool]

        # Add nodes (just function references - no closures needed!)
        logger.info("Adding nodes to workflow graph")
        workflow.add_node("guardrail", ainvoke_guardrail_step)
        workflow.add_node("out_of_scope", ainvoke_out_of_scope_step)
        workflow.add_node("retrieve", ainvoke_retrieve_step)
        workflow.add_node("tool_retrieve", ToolNode(tools))
        workflow.add_node("grade_documents", ainvoke_grade_documents_step)
        workflow.add_node("rewrite_query", ainvoke_rewrite_query_step)
        workflow.add_node("generate_answer", ainvoke_generate_answer_step)

        # Add edges
        logger.info("Configuring graph edges and routing logic")

        # Start → guardrail validation
        workflow.add_edge(START, "guardrail")

        # Guardrail → route based on score
        workflow.add_conditional_edges(
            "guardrail",
            continue_after_guardrail,
            {
                "continue": "retrieve",
                "out_of_scope": "out_of_scope",
            },
        )

        # Out of scope → END
        workflow.add_edge("out_of_scope", END)

        # Retrieve node creates tool call
        workflow.add_conditional_edges(
            "retrieve",
            tools_condition,
            {
                "tools": "tool_retrieve",
                END: END,
            },
        )

        # After tool retrieval → grade documents
        workflow.add_edge("tool_retrieve", "grade_documents")

        # After grading → route based on relevance
        workflow.add_conditional_edges(
            "grade_documents",
            lambda state: state.get("routing_decision", "generate_answer"),
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
            },
        )

        # After rewriting → try retrieve again
        workflow.add_edge("rewrite_query", "retrieve")

        # After answer generation → done
        workflow.add_edge("generate_answer", END)

        # Compile graph
        logger.info("Compiling LangGraph workflow")
        compiled_graph = workflow.compile()
        logger.info("✓ Graph compilation successful")

        return compiled_graph

    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
        model: Optional[str] = None,
    ) -> dict:
        """Ask a question using agentic RAG.

        :param query: User question
        :param user_id: User identifier for tracing
        :param model: Optional model override
        :returns: Dictionary with answer, sources, reasoning steps, and metadata
        :raises ValueError: If query is empty
        """
        model_to_use = model or self.graph_config.model

        logger.info("=" * 80)
        logger.info("Starting Agentic RAG Request")
        logger.info(f"Query: {query}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Model: {model_to_use}")
        logger.info("=" * 80)

        # Validate input
        if not query or len(query.strip()) == 0:
            logger.error("Empty query received")
            raise ValueError("Query cannot be empty")

        # Create trace if Langfuse is enabled (v3 SDK)
        trace = None
        if self.langfuse_tracer and self.langfuse_tracer.client:
            logger.info("Creating Langfuse trace (v3 SDK)")
            metadata = {
                "env": self.graph_config.settings.environment,
                "service": "agentic_rag",
                "top_k": self.graph_config.top_k,
                "use_hybrid": self.graph_config.use_hybrid,
                "model": model_to_use,
            }
            # V3 SDK: Use start_as_current_span - will be used with 'with' statement
            trace = self.langfuse_tracer.client.start_as_current_span(
                name="agentic_rag_request",
            )

        # Use proper context manager pattern
        async def _execute_with_trace():
            """Execute the workflow with or without tracing context."""
            if trace is not None:
                with trace as trace_obj:
                    trace_obj.update(
                        input={"query": query},
                        metadata=metadata,
                        user_id=user_id,
                        session_id=f"session_{user_id}",
                    )
                    logger.debug(f"Trace created: {trace_obj}")
                    return await self._run_workflow(query, model_to_use, user_id, trace_obj)
            else:
                return await self._run_workflow(query, model_to_use, user_id, None)

        try:
            return await _execute_with_trace()
        except Exception as e:
            logger.error(f"Error in Agentic RAG execution: {str(e)}")
            logger.exception("Full traceback:")
            raise

    async def _run_workflow(self, query: str, model_to_use: str, user_id: str, trace) -> dict:
        """Execute the workflow with the given trace context."""
        try:
            start_time = time.time()

            logger.info("Invoking LangGraph workflow")

            # State initialization
            state_input = {
                "messages": [HumanMessage(content=query)],
                "retrieval_attempts": 0,
                "guardrail_result": None,
                "routing_decision": None,
                "sources": None,
                "relevant_sources": [],
                "relevant_tool_artefacts": None,
                "grading_results": [],
                "metadata": {},
                "original_query": None,
                "rewritten_query": None,
            }

            # Runtime context (dependencies)
            runtime_context = Context(
                ollama_client=self.ollama,
                opensearch_client=self.opensearch,
                embeddings_client=self.embeddings,
                langfuse_tracer=self.langfuse_tracer,
                trace=trace,
                langfuse_enabled=self.langfuse_tracer is not None and self.langfuse_tracer.client is not None,
                model_name=model_to_use,
                temperature=self.graph_config.temperature,
                top_k=self.graph_config.top_k,
                max_retrieval_attempts=self.graph_config.max_retrieval_attempts,
                guardrail_threshold=self.graph_config.guardrail_threshold,
            )

            # Create config with CallbackHandler if Langfuse is enabled (v3 SDK)
            config = {"thread_id": f"user_{user_id}_session_{int(time.time())}"}

            # Add CallbackHandler for automatic LLM tracing
            # IMPORTANT: CallbackHandler automatically inherits the current span context
            # Since we're inside start_as_current_span, it will be linked automatically
            if self.langfuse_tracer and trace:
                try:
                    # V3 SDK: CallbackHandler() automatically uses current trace context
                    # No need to pass trace explicitly - it's handled by context propagation
                    callback_handler = CallbackHandler()
                    config["callbacks"] = [callback_handler]
                    logger.info("✓ CallbackHandler added (will auto-link to current trace)")
                except Exception as e:
                    logger.warning(f"Failed to create CallbackHandler: {e}")

            result = await self.graph.ainvoke(
                state_input,
                config=config,
                context=runtime_context,
            )

            execution_time = time.time() - start_time
            logger.info(f"✓ Graph execution completed in {execution_time:.2f}s")

            # Extract results
            answer = self._extract_answer(result)
            sources = self._extract_sources(result)
            retrieval_attempts = result.get("retrieval_attempts", 0)
            reasoning_steps = self._extract_reasoning_steps(result)

            # Update trace (cleanup handled by context manager)
            if trace:
                trace.update(
                    output={
                        "answer": answer,
                        "sources_count": len(sources),
                        "retrieval_attempts": retrieval_attempts,
                        "reasoning_steps": reasoning_steps,
                        "execution_time": execution_time,
                    }
                )
                trace.end()
                self.langfuse_tracer.flush()

            logger.info("=" * 80)
            logger.info("Agentic RAG Request Completed Successfully")
            logger.info(f"Answer length: {len(answer)} characters")
            logger.info(f"Sources found: {len(sources)}")
            logger.info(f"Retrieval attempts: {retrieval_attempts}")
            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info("=" * 80)

            return {
                "query": query,
                "answer": answer,
                "sources": sources,
                "reasoning_steps": reasoning_steps,
                "retrieval_attempts": retrieval_attempts,
                "rewritten_query": result.get("rewritten_query"),
                "execution_time": execution_time,
                "guardrail_score": result.get("guardrail_result").score if result.get("guardrail_result") else None,
            }

        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            logger.exception("Full traceback:")

            # Update trace with error (cleanup handled by context manager)
            if trace:
                trace.update(output={"error": str(e)}, level="ERROR")
                trace.end()
                self.langfuse_tracer.flush()

            raise

    def _extract_answer(self, result: dict) -> str:
        """Extract final answer from graph result."""
        messages = result.get("messages", [])
        if not messages:
            return "No answer generated."

        final_message = messages[-1]
        return final_message.content if hasattr(final_message, "content") else str(final_message)

    def _extract_sources(self, result: dict) -> List[dict]:
        """Extract sources from graph result."""
        sources = []
        relevant_sources = result.get("relevant_sources", [])

        for source in relevant_sources:
            if hasattr(source, "to_dict"):
                sources.append(source.to_dict())
            elif isinstance(source, dict):
                sources.append(source)

        return sources

    def _extract_reasoning_steps(self, result: dict) -> List[str]:
        """Extract reasoning steps from graph result."""
        steps = []
        retrieval_attempts = result.get("retrieval_attempts", 0)
        guardrail_result = result.get("guardrail_result")
        grading_results = result.get("grading_results", [])

        if guardrail_result:
            steps.append(f"Validated query scope (score: {guardrail_result.score}/100)")

        if retrieval_attempts > 0:
            steps.append(f"Retrieved documents ({retrieval_attempts} attempt(s))")

        if grading_results:
            relevant_count = sum(1 for g in grading_results if g.is_relevant)
            steps.append(f"Graded documents ({relevant_count} relevant)")

        if result.get("rewritten_query"):
            steps.append("Rewritten query for better results")

        steps.append("Generated answer from context")

        return steps

    def get_graph_visualization(self) -> bytes:
        """Get the LangGraph workflow visualization as PNG.

        This method generates a visual representation of the graph workflow
        using mermaid diagram format, then converts it to PNG.

        :returns: PNG image bytes
        :raises ImportError: If required dependencies (pygraphviz/graphviz) are not installed
        :raises Exception: If graph visualization generation fails

        Example:
            >>> service = AgenticRAGService(...)
            >>> png_bytes = service.get_graph_visualization()
            >>> with open("graph.png", "wb") as f:
            ...     f.write(png_bytes)
        """
        try:
            logger.info("Generating graph visualization as PNG")
            png_bytes = self.graph.get_graph().draw_mermaid_png()
            logger.info(f"✓ Generated PNG visualization ({len(png_bytes)} bytes)")
            return png_bytes
        except ImportError as e:
            logger.error(f"Failed to generate visualization - missing dependencies: {e}")
            logger.error("Install with: pip install pygraphviz or apt-get install graphviz")
            raise ImportError(
                "Graph visualization requires pygraphviz. "
                "Install with: pip install pygraphviz (requires graphviz system package)"
            ) from e
        except Exception as e:
            logger.error(f"Failed to generate graph visualization: {e}")
            raise

    def get_graph_mermaid(self) -> str:
        """Get the LangGraph workflow as a mermaid diagram string.

        This method generates the graph workflow representation in mermaid
        diagram syntax, which can be rendered in markdown or mermaid viewers.

        :returns: Mermaid diagram syntax as string

        Example:
            >>> service = AgenticRAGService(...)
            >>> mermaid = service.get_graph_mermaid()
            >>> print(mermaid)
            graph TD
                __start__ --> guardrail
                ...
        """
        try:
            logger.info("Generating graph as mermaid diagram")
            mermaid_str = self.graph.get_graph().draw_mermaid()
            logger.info(f"✓ Generated mermaid diagram ({len(mermaid_str)} characters)")
            return mermaid_str
        except Exception as e:
            logger.error(f"Failed to generate mermaid diagram: {e}")
            raise

    def get_graph_ascii(self) -> str:
        """Get ASCII representation of the graph.

        This method generates a simple ASCII art representation of the
        graph structure, useful for quick inspection in terminals.

        :returns: ASCII art representation of the graph

        Example:
            >>> service = AgenticRAGService(...)
            >>> print(service.get_graph_ascii())
        """
        try:
            logger.info("Generating ASCII graph representation")
            ascii_str = self.graph.get_graph().print_ascii()
            logger.info("✓ Generated ASCII graph representation")
            return ascii_str
        except Exception as e:
            logger.error(f"Failed to generate ASCII graph: {e}")
            raise
```

### 编排服务要点

- **`StateGraph(AgentState, context_schema=Context)`**：状态用 `AgentState`，依赖用 `Context`（通过 `context=` 在 `ainvoke` 时注入）。
- **节点即纯函数**：`workflow.add_node("guardrail", ainvoke_guardrail_step)` 直接挂函数引用，无需闭包。
- **条件边**：`add_conditional_edges("guardrail", continue_after_guardrail, {...})` 把路由函数的返回值映射到下一节点。
- **`tools_condition`**：LangGraph 预置的条件——检测最新 AIMessage 是否带 `tool_calls`，有则去 `tool_retrieve`。
- **可视化**：`get_graph_mermaid()` / `get_graph_ascii()` 导出工作流图（`ascii` 需要 `viz` 组的 `grandalf`）。

### 工厂与包导出

### 文件：`src/services/agents/factory.py`（逐字复制）

```python
from typing import Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient

from .agentic_rag import AgenticRAGService
from .config import GraphConfig


def make_agentic_rag_service(
    opensearch_client: OpenSearchClient,
    ollama_client: OllamaClient,
    embeddings_client: JinaEmbeddingsClient,
    langfuse_tracer: Optional[LangfuseTracer] = None,
    top_k: int = 3,
    use_hybrid: bool = True,
) -> AgenticRAGService:
    """
    Create AgenticRAGService with dependency injection.

    Args:
        opensearch_client: Client for document search
        ollama_client: Client for LLM generation
        embeddings_client: Client for embeddings
        langfuse_tracer: Optional Langfuse tracer for observability
        top_k: Number of documents to retrieve (default: 3)
        use_hybrid: Use hybrid search (default: True)

    Returns:
        Configured AgenticRAGService instance
    """
    # Create graph configuration with the provided parameters
    graph_config = GraphConfig(
        top_k=top_k,
        use_hybrid=use_hybrid,
    )

    return AgenticRAGService(
        opensearch_client=opensearch_client,
        ollama_client=ollama_client,
        embeddings_client=embeddings_client,
        langfuse_tracer=langfuse_tracer,
        graph_config=graph_config,
    )
```

### 文件：`src/services/agents/__init__.py`（逐字复制）

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

---

## 10.15 Agentic 端点：`src/routers/agentic_ask.py`

### 文件：`src/routers/agentic_ask.py`（逐字复制）

```python
from fastapi import APIRouter, HTTPException
from src.dependencies import AgenticRAGDep, LangfuseDep
from src.schemas.api.ask import AgenticAskResponse, AskRequest, FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api/v1", tags=["agentic-rag"])


@router.post("/ask-agentic", response_model=AgenticAskResponse)
async def ask_agentic(
    request: AskRequest,
    agentic_rag: AgenticRAGDep,
) -> AgenticAskResponse:
    """
    Agentic RAG endpoint with intelligent retrieval and query refinement.

    Features:
    - Decides if retrieval is needed
    - Grades document relevance
    - Rewrites queries if needed
    - Provides reasoning transparency

    The agent will automatically:
    1. Determine if the question requires research paper retrieval
    2. If needed, search for relevant papers
    3. Grade retrieved documents for relevance
    4. Rewrite the query if documents aren't relevant
    5. Generate an answer with citations

    Args:
        request: Question and parameters
        agentic_rag: Injected agentic RAG service

    Returns:
        Answer with sources and reasoning steps

    Raises:
        HTTPException: If processing fails
    """
    try:
        result = await agentic_rag.ask(
            query=request.query,
        )

        return AgenticAskResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result.get("sources", []),
            chunks_used=request.top_k,
            search_mode="hybrid" if request.use_hybrid else "bm25",
            reasoning_steps=result.get("reasoning_steps", []),
            retrieval_attempts=result.get("retrieval_attempts", 0),
            trace_id=result.get("trace_id"),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    langfuse_tracer: LangfuseDep,
) -> FeedbackResponse:
    """
    Submit user feedback for an agentic RAG response.

    This endpoint allows users to rate the quality of answers and provide
    optional comments. Feedback is tracked in Langfuse for continuous improvement.

    Args:
        request: Feedback data including trace_id, score, and optional comment
        langfuse_tracer: Injected Langfuse tracer service

    Returns:
        FeedbackResponse indicating success or failure

    Raises:
        HTTPException: If feedback submission fails
    """
    try:
        if not langfuse_tracer:
            raise HTTPException(
                status_code=503,
                detail="Langfuse tracing is disabled. Cannot submit feedback."
            )

        success = langfuse_tracer.submit_feedback(
            trace_id=request.trace_id,
            score=request.score,
            comment=request.comment,
        )

        if success:
            # Flush to ensure feedback is sent immediately
            langfuse_tracer.flush()

            return FeedbackResponse(
                success=True,
                message="Feedback recorded successfully"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to submit feedback to Langfuse"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting feedback: {str(e)}"
        )
```

> 注意 `agentic_ask.router` 自带 `prefix="/api/v1"`，所以在 `main.py` 里注册时**不再加前缀**（`app.include_router(agentic_ask.router)`）。

---

## 10.16 Telegram 机器人：`src/services/telegram/`

### 文件：`src/services/telegram/bot.py`（逐字复制）

```python
import logging
from typing import Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from src.schemas.api.ask import AskRequest, AskResponse
from src.schemas.api.search import HybridSearchRequest

logger = logging.getLogger(__name__)


class TelegramBot:
    """Simple Telegram bot for Q&A."""

    def __init__(
        self,
        bot_token: str,
        opensearch_client,
        embeddings_client,
        ollama_client,
        cache_client=None,
    ):
        """Initialize bot with required services."""
        self.bot_token = bot_token
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.ollama = ollama_client
        self.cache = cache_client
        self.application: Optional[Application] = None

    async def start(self) -> None:
        """Start the bot."""
        logger.info("Starting Telegram bot...")
        self.application = Application.builder().token(self.bot_token).build()

        # Register handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("search", self._search_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_question))

        # Start polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        logger.info("Telegram bot started successfully")

    async def stop(self) -> None:
        """Stop the bot."""
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram bot stopped")

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        await update.message.reply_text(
            "Welcome to arXiv Paper Curator!\n\n"
            "Ask me questions about CS papers and I'll provide answers with sources.\n\n"
            "Commands:\n"
            "/help - Show this help\n"
            "/search <keywords> - Search papers"
        )

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        await update.message.reply_text(
            "Send me any question about computer science research papers.\n\n"
            "Examples:\n"
            "- What are transformer architectures?\n"
            "- How does BERT work?\n"
            "- Explain attention mechanisms\n\n"
            "Use /search to find specific papers."
        )

    async def _search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /search command."""
        if not context.args:
            await update.message.reply_text("Usage: /search <keywords>\nExample: /search neural networks")
            return

        query = " ".join(context.args)
        await update.message.chat.send_action("typing")

        try:
            # Generate embedding
            query_embedding = await self.embeddings.embed_query(query)

            # Search
            results = self.opensearch.search_unified(
                query=query,
                query_embedding=query_embedding,
                size=10,
                use_hybrid=True,
            )

            hits = results.get("hits", [])
            if not hits:
                await update.message.reply_text("No papers found. Try different keywords.")
                return

            # Deduplicate by arxiv_id (since chunks may have same paper)
            seen_ids = set()
            unique_papers = []
            for hit in hits:
                arxiv_id = hit.get("arxiv_id", "")
                if arxiv_id and arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    unique_papers.append(hit)
                if len(unique_papers) >= 5:
                    break

            # Format results
            response = f"Found {len(unique_papers)} papers:\n\n"
            for idx, hit in enumerate(unique_papers, 1):
                title = hit.get("title", "Untitled")
                arxiv_id = hit.get("arxiv_id", "")
                url = f"https://arxiv.org/abs/{arxiv_id}"
                response += f"{idx}. {title}\n{url}\n\n"

            await update.message.reply_text(response, disable_web_page_preview=True)

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            await update.message.reply_text(f"Search failed: {str(e)}")

    async def _handle_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle user questions."""
        query = update.message.text
        await update.message.chat.send_action("typing")

        try:
            # Build request
            ask_request = AskRequest(query=query, top_k=3, use_hybrid=True)

            # Check cache
            if self.cache:
                try:
                    cached_response = await self.cache.find_cached_response(ask_request)
                    if cached_response:
                        await self._send_answer(update, cached_response)
                        return
                except Exception as e:
                    logger.warning(f"Cache lookup failed: {e}")

            # RAG pipeline
            from src.services.ollama.prompts import RAGPromptBuilder

            # Get embeddings if hybrid
            query_embedding = None
            if ask_request.use_hybrid:
                try:
                    query_embedding = await self.embeddings.embed_query(query)
                    logger.info("Generated query embedding")
                except Exception as e:
                    logger.warning(f"Failed to generate embeddings: {e}")

            # Search OpenSearch
            search_results = self.opensearch.search_unified(
                query=query,
                query_embedding=query_embedding,
                size=ask_request.top_k,
                use_hybrid=ask_request.use_hybrid and query_embedding is not None,
            )

            # Extract chunks and sources
            chunks = []
            sources_set = set()
            for hit in search_results.get("hits", []):
                arxiv_id = hit.get("arxiv_id", "")
                chunks.append(
                    {
                        "arxiv_id": arxiv_id,
                        "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                    }
                )
                if arxiv_id:
                    arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                    sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

            sources = list(sources_set)

            if not chunks:
                await update.message.reply_text("No relevant papers found. Try rephrasing your question.")
                return

            # Generate answer
            prompt = RAGPromptBuilder().create_rag_prompt(query=query, chunks=chunks)
            ollama_response = await self.ollama.generate(model="gemma4:e2b", prompt=prompt, stream=False)
            answer = ollama_response.get("response", "") if ollama_response else ""

            # Build response
            response = AskResponse(
                query=query, answer=answer, sources=sources, chunks_used=len(chunks), search_mode="hybrid"
            )

            # Cache it
            if self.cache:
                try:
                    await self.cache.store_response(ask_request, response)
                except Exception:
                    pass

            # Send to user
            await self._send_answer(update, response)

        except Exception as e:
            logger.error(f"Question handling failed: {e}", exc_info=True)
            await update.message.reply_text(f"Error: {str(e)}")

    async def _send_answer(self, update: Update, response: AskResponse) -> None:
        """Send formatted answer to user."""
        # Answer
        message = f"*Answer:*\n{response.answer}\n"

        # Sources
        if response.sources:
            message += "\n*Sources:*\n"
            for idx, source_url in enumerate(response.sources[:5], 1):
                arxiv_id = source_url.split("/")[-1].replace(".pdf", "")
                message += f"{idx}. https://arxiv.org/abs/{arxiv_id}\n"

        # Send (try markdown, fallback to plain)
        try:
            await update.message.reply_text(message, parse_mode="Markdown", disable_web_page_preview=True)
        except Exception:
            await update.message.reply_text(message, disable_web_page_preview=True)
```

### 文件：`src/services/telegram/factory.py`（逐字复制）

```python
import logging
from typing import Optional

from src.config import get_settings
from src.services.telegram.bot import TelegramBot

logger = logging.getLogger(__name__)


def make_telegram_service(
    opensearch_client,
    embeddings_client,
    ollama_client,
    cache_client=None,
    langfuse_tracer=None,
) -> Optional[TelegramBot]:
    """
    Create Telegram bot if enabled.

    Args:
        opensearch_client: OpenSearch client
        embeddings_client: Embeddings service client
        ollama_client: Ollama LLM client
        cache_client: Optional cache client
        langfuse_tracer: Optional Langfuse tracer (not used)

    Returns:
        TelegramBot instance or None if disabled
    """
    settings = get_settings()

    if not settings.telegram.enabled:
        logger.info("Telegram bot is disabled")
        return None

    if not settings.telegram.bot_token:
        logger.warning("Telegram bot token not configured")
        return None

    bot = TelegramBot(
        bot_token=settings.telegram.bot_token,
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        ollama_client=ollama_client,
        cache_client=cache_client,
    )

    logger.info("Telegram bot created successfully")
    return bot
```

### 文件：`src/services/telegram/__init__.py`（逐字复制）

```python
from .bot import TelegramBot
from .factory import make_telegram_service

__all__ = ["TelegramBot", "make_telegram_service"]
```

> 机器人用**长轮询**（`updater.start_polling()`），在 FastAPI 的 lifespan 里启动/停止。三个命令：`/start`、`/help`、`/search <kw>`；普通文本消息走完整 RAG（检索 + 生成 + 缓存）。`make_telegram_service` 在 `TELEGRAM__ENABLED=false` 或无 token 时返回 `None`，应用据此跳过启动。

---

## 10.17 最终版 `src/dependencies.py`（含修复 #5）

现在所有服务齐备，给出**依赖注入的最终完整版**（替换之前所有增量）。

> ⚠️ **上游修复（修复集 #5）**：上游 `get_agentic_rag_service` 调用 `make_agentic_rag_service(..., model=settings.ollama_model)`，但该工厂**没有 `model` 形参**（只有 `top_k`、`use_hybrid`）。由于 `/ask-agentic` 端点每次请求都会通过 `AgenticRAGDep` 解析这个依赖，**上游每次调用 `/ask-agentic` 都会抛 `TypeError`**。下面的版本**删去了多余的 `model=` 参数**（用行内注释标出），智能体改用 `GraphConfig` 的默认模型 `gemma4:e2b`（与项目默认 `OLLAMA_MODEL` 一致）。若你想让智能体也跟随 `OLLAMA_MODEL`，可改为给工厂增加 `model` 形参——见第 [16](16-upstream-differences-and-fixes.md) 章。

### 文件：`src/dependencies.py`（含修复 #5，可直接运行）

```python
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Generator, Optional

if TYPE_CHECKING:
    from fastapi import Depends, Request
    from sqlalchemy.orm import Session
else:
    try:
        from fastapi import Depends, Request
        from sqlalchemy.orm import Session
    except ImportError:
        pass

from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.arxiv.client import ArxivClient
from src.services.cache.client import CacheClient
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient
from src.services.pdf_parser.parser import PDFParserService
from src.services.telegram.bot import TelegramBot
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.agents.factory import make_agentic_rag_service


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


def get_opensearch_client(request: Request) -> OpenSearchClient:
    """Get OpenSearch client from the request state."""
    return request.app.state.opensearch_client


def get_arxiv_client(request: Request) -> ArxivClient:
    """Get arXiv client from the request state."""
    return request.app.state.arxiv_client


def get_pdf_parser(request: Request) -> PDFParserService:
    """Get PDF parser service from the request state."""
    return request.app.state.pdf_parser


def get_embeddings_service(request: Request) -> JinaEmbeddingsClient:
    """Get embeddings service from the request state."""
    return request.app.state.embeddings_service


def get_ollama_client(request: Request) -> OllamaClient:
    """Get Ollama client from the request state."""
    return request.app.state.ollama_client


def get_langfuse_tracer(request: Request) -> LangfuseTracer:
    """Get Langfuse tracer from the request state."""
    return request.app.state.langfuse_tracer


def get_cache_client(request: Request) -> CacheClient | None:
    """Get cache client from the request state."""
    return getattr(request.app.state, "cache_client", None)


def get_telegram_service(request: Request) -> Optional[TelegramBot]:
    """Get Telegram service from the request state."""
    return getattr(request.app.state, "telegram_service", None)


# Dependency annotations
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
ArxivDep = Annotated[ArxivClient, Depends(get_arxiv_client)]
PDFParserDep = Annotated[PDFParserService, Depends(get_pdf_parser)]
EmbeddingsDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_service)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama_client)]
LangfuseDep = Annotated[LangfuseTracer, Depends(get_langfuse_tracer)]
CacheDep = Annotated[CacheClient | None, Depends(get_cache_client)]
TelegramDep = Annotated[Optional[TelegramBot], Depends(get_telegram_service)]


def get_agentic_rag_service(
    opensearch: OpenSearchDep,
    ollama: OllamaDep,
    embeddings: EmbeddingsDep,
    langfuse: LangfuseDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgenticRAGService:
    """Get agentic RAG service."""
    # ⚠️ 上游修复(#5): 上游此处多传了 model=settings.ollama_model，但
    # make_agentic_rag_service 没有 model 形参，会在每次 /ask-agentic 请求时抛 TypeError。
    # 删去该参数即可；智能体使用 GraphConfig 默认模型 gemma4:e2b（= 项目默认 OLLAMA_MODEL）。
    return make_agentic_rag_service(
        opensearch_client=opensearch,
        ollama_client=ollama,
        embeddings_client=embeddings,
        langfuse_tracer=langfuse,
    )


AgenticRAGDep = Annotated[AgenticRAGService, Depends(get_agentic_rag_service)]
```

> **修复 #5 说明**：`/ask-agentic` 端点通过 `AgenticRAGDep` → `get_agentic_rag_service` **每次请求**构建智能体服务。上游传入工厂不接受的 `model=` 关键字，会让该端点 100% 抛 `TypeError`。删去这一个参数即可让端点跑通。`settings` 形参仍保留（签名不变），只是不再用它传 model——这是让 Agentic RAG 真正可运行的关键一步。

---

## 10.18 最终版 `src/main.py`

### 文件：`src/main.py`（逐字复制，最终版）

```python
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from src.config import get_settings
from src.db.factory import make_database
from src.routers import agentic_ask, hybrid_search, ping
from src.routers.ask import ask_router, stream_router
from src.services.arxiv.factory import make_arxiv_client
from src.services.cache.factory import make_cache_client
from src.services.embeddings.factory import make_embeddings_service
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.ollama.factory import make_ollama_client
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.telegram.factory import make_telegram_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan for the API.
    """
    logger.info("Starting RAG API...")

    settings = get_settings()
    app.state.settings = settings

    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    # Initialize search service
    opensearch_client = make_opensearch_client()
    app.state.opensearch_client = opensearch_client

    # Verify OpenSearch connectivity and create index if needed
    if opensearch_client.health_check():
        logger.info("OpenSearch connected successfully")

        # Setup hybrid index (supports all search types)
        setup_results = opensearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("Hybrid index created")
        else:
            logger.info("Hybrid index already exists")

        # Get simple statistics
        try:
            stats = opensearch_client.client.count(index=opensearch_client.index_name)
            logger.info(f"OpenSearch ready: {stats['count']} documents indexed")
        except Exception:
            logger.info("OpenSearch index ready (stats unavailable)")
    else:
        logger.warning("OpenSearch connection failed - search features will be limited")

    # Initialize other services (kept for future endpoints and notebook demos)
    app.state.arxiv_client = make_arxiv_client()
    app.state.pdf_parser = make_pdf_parser_service()
    app.state.embeddings_service = make_embeddings_service()
    app.state.ollama_client = make_ollama_client()
    app.state.langfuse_tracer = make_langfuse_tracer()
    app.state.cache_client = make_cache_client(settings)
    logger.info("Services initialized: arXiv API client, PDF parser, OpenSearch, Embeddings, Ollama, Langfuse, Cache")

    # Initialize Telegram bot (Week 7)
    telegram_service = make_telegram_service(
        opensearch_client=app.state.opensearch_client,
        embeddings_client=app.state.embeddings_service,
        ollama_client=app.state.ollama_client,
        cache_client=app.state.cache_client,
        langfuse_tracer=app.state.langfuse_tracer,
    )

    if telegram_service:
        app.state.telegram_service = telegram_service
        try:
            await telegram_service.start()
            logger.info("Telegram bot started successfully")
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
    else:
        logger.info("Telegram bot not configured - skipping initialization")

    logger.info("API ready")
    yield

    # Cleanup
    if hasattr(app.state, "telegram_service") and app.state.telegram_service:
        await app.state.telegram_service.stop()
        logger.info("Telegram bot stopped")

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
app.include_router(hybrid_search.router, prefix="/api/v1")  # Search chunks with BM25/hybrid
app.include_router(ask_router, prefix="/api/v1")  # RAG question answering with LLM
app.include_router(stream_router, prefix="/api/v1")  # Streaming RAG responses
app.include_router(agentic_ask.router)  # Agentic RAG with intelligent retrieval


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
```

> **`make_cache_client(settings)` 在 lifespan 中无 try/except**：若 Redis 不可达会让启动失败。开发栈里 `redis` 是 `api` 的 `depends_on: service_healthy`，正常可达。若你在无 Redis 环境运行，可把这行用 try/except 包裹并将 `app.state.cache_client` 设为 `None`（`CacheDep` 已支持 None）。

---

## 10.19 本周验证

### 准备

```bash
# .env: TELEGRAM__ENABLED=true 且填好 TELEGRAM__BOT_TOKEN（可选；不做 TG 就设 false）
# 确保 Ollama 模型已拉取、JINA_API_KEY 已填、已有论文索引
docker compose up -d --build api postgres opensearch redis ollama
```

### 验证 Agentic 端点

```bash
# 领域内问题：应触发检索 + 评分 + 生成
curl -s -X POST http://localhost:8000/api/v1/ask-agentic \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformer architectures in deep learning?"}' | python -m json.tool

# 领域外问题：应被护栏拦截，礼貌拒答
curl -s -X POST http://localhost:8000/api/v1/ask-agentic \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the best recipe for pasta?"}' | python -m json.tool
```

期望：领域内返回 `answer` + `reasoning_steps`（如 "Validated query scope..."、"Retrieved documents..."、"Generated answer..."）+ `retrieval_attempts`；领域外的 `answer` 是礼貌拒答，`reasoning_steps` 显示护栏判定。

### 验证 Telegram（可选）

若配置了 Token，应用启动日志会显示 "Telegram bot started successfully"。在 Telegram 里找到你的机器人，发 `/start`、`/help`、`/search neural networks`，或直接发一个问题。

### 可视化工作流图（可选）

```bash
uv sync --group viz   # 安装 grandalf（ASCII 图需要）
```

```python
# 临时脚本
from src.services.opensearch.factory import make_opensearch_client_fresh
from src.services.ollama.factory import make_ollama_client
from src.services.embeddings.factory import make_embeddings_client
from src.services.agents.factory import make_agentic_rag_service

svc = make_agentic_rag_service(
    opensearch_client=make_opensearch_client_fresh(),
    ollama_client=make_ollama_client(),
    embeddings_client=make_embeddings_client(),
)
print(svc.get_graph_ascii())
```

---

## 10.20 本章小结

你已经有了**完整的 7 周系统**：

- ✅ LangGraph Agentic RAG：护栏 → 检索 → 评分 → 重写 → 生成，带次数上限与全节点兜底。
- ✅ 5 个决策节点 + 工具节点 + 图编排服务（可导出工作流图）。
- ✅ `/api/v1/ask-agentic` 与 `/api/v1/feedback` 端点。
- ✅ Telegram 机器人（命令 + RAG 问答 + 缓存）。
- ✅ `main.py` 与 `dependencies.py` 最终完整版。

**🎉 至此，从空目录到生产风格 Agentic RAG 的全部代码已就绪。**

接下来：第 [11](11-testing.md) 章覆盖测试与验证；第 [12](12-run-build-deploy-rollback.md) 章讲运行/构建/部署/回滚；第 [13](13-troubleshooting.md)–[15](15-cicd-and-maintenance.md) 章分别是排错、质量分析、CI/CD 与维护；第 [16](16-upstream-differences-and-fixes.md) 章汇总全部上游修复与源文件覆盖清单。
