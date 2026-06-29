# File: tutorials/09-week6-monitoring-caching.md

# 第 9 章　Week 6：生产监控（Langfuse）与缓存（Redis）

**本周目标**：把 RAG 升级到"生产可用"。加上 **Langfuse 全链路追踪**（看到每次请求的检索、提示词、生成、耗时、token）和 **Redis 精确缓存**（重复问题毫秒级返回）。本章同时给出依赖追踪/缓存的 `/api/v1/ask` 与 `/api/v1/stream` 端点完整代码——至此 Gradio 可端到端运行。

> ⚠️ **本章包含上游修复 #2**：`LangfuseTracer` 在 v3 重构后缺失了 `trace_rag_request()` / `create_span()` / `end_span()` 三个方法，而 `RAGTracer`（本章）和 Week 7 智能体节点都在调用它们，运行时会抛 `AttributeError`。本章在 `LangfuseTracer` 中**补全这三个方法**，并保证**未启用 Langfuse 时安全降级为 no-op**——这是让 `/ask`、`/stream`、`/ask-agentic` 在"无 Langfuse 密钥"默认配置下也能跑通的关键。

---

## 9.1 Langfuse 服务端启动与取密钥

`compose.yml`（第 [04](04-week1-infrastructure.md) 章已完整给出）里包含整套 Langfuse v3 自托管栈：`langfuse-web`、`langfuse-worker`、`clickhouse`、`langfuse-postgres`、`langfuse-redis`、`langfuse-minio`。

### 第一步：生成加密密钥（必需）

`langfuse-web` 没有合法的 `LANGFUSE_ENCRYPTION_KEY` 不会启动。生成并写入 `.env`：

```bash
openssl rand -hex 32
# 把输出粘到 .env 的 LANGFUSE_ENCRYPTION_KEY=
# 建议同时把 LANGFUSE_NEXTAUTH_SECRET 与 LANGFUSE_SALT 换成足够长的随机串
```

### 第二步：启动 Langfuse 栈

```bash
docker compose up -d clickhouse langfuse-postgres langfuse-redis langfuse-minio
# 等它们 healthy（约 30–60s）
docker compose up -d langfuse-web langfuse-worker
docker compose logs -f langfuse-web
```

### 第三步：登录并取项目密钥

打开 **http://localhost:3001**。`compose.yml` 通过 `LANGFUSE_INIT_*` 预置了管理员账号：

- 邮箱：`admin@example.com`
- 密码：`admin123`

> **安全**：这是开发默认账号，**生产必须改**（见第 [14](14-quality-performance-security.md) 章）。

登录后进入预置项目 **"Agentic RAG"** → Settings → API Keys → 创建一对密钥，把它们填入 `.env`：

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_ENABLED=true
```

> 注意配置键是**单下划线**（`LANGFUSE_PUBLIC_KEY`），但代码里 `LangfuseSettings` 用前缀 `LANGFUSE__`（双下划线）。`.env` 同时提供了两种形式，且 `compose.yml` 给 `api` 容器注入了 `LANGFUSE_HOST=http://langfuse-web:3000`。若密钥留空或 `LANGFUSE_ENABLED=false`，追踪会**安全降级**，主流程照常工作。

---

## 9.2 Langfuse 客户端：`src/services/langfuse/client.py`（含修复 #2）

```bash
mkdir -p src/services/langfuse
touch src/services/langfuse/__init__.py
```

> `src/services/langfuse/__init__.py` 为空（包标记）。

> ⚠️ **上游修复（修复集 #2）**：下面的 `LangfuseTracer` 在上游基础上**追加了 `trace_rag_request`、`create_span`、`end_span` 三个方法**（在文件末尾、类内，用 `# ⚠️ 上游修复(#2)` 标出）。它们在未启用 Langfuse 时返回 `None` / no-op；启用时尽力创建真实 span 且全程 try/except 包裹，**绝不因追踪问题中断请求**。其余方法逐字复刻上游。

### 文件：`src/services/langfuse/client.py`（含修复，可直接运行）

```python
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

from langfuse import Langfuse
from src.config import Settings

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Wrapper for Langfuse v3 tracing client with CallbackHandler support."""

    def __init__(self, settings: Settings):
        self.settings = settings.langfuse
        self.client: Optional[Langfuse] = None

        if self.settings.enabled and self.settings.public_key and self.settings.secret_key:
            try:
                # Initialize Langfuse v3 singleton client
                # Configuration moved to client initialization (not handler)
                self.client = Langfuse(
                    public_key=self.settings.public_key,
                    secret_key=self.settings.secret_key,
                    host=self.settings.host,
                    flush_at=self.settings.flush_at,
                    flush_interval=self.settings.flush_interval,
                    debug=self.settings.debug,
                )
                logger.info(f"Langfuse v3 tracing initialized (host: {self.settings.host})")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self.client = None
        else:
            logger.info("Langfuse tracing disabled or missing credentials")

    def get_callback_handler(
        self,
        trace_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ):
        """
        Get a CallbackHandler for LangChain/LangGraph integration.

        This is the v3 recommended approach - all LLM calls are automatically traced.

        Args:
            trace_name: Optional name for the trace
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Additional metadata to attach to the trace
            tags: Optional tags for the trace

        Returns:
            CallbackHandler instance if Langfuse is enabled, None otherwise
        """
        if not self.client:
            return None

        try:
            # Import v3 CallbackHandler (new path)
            from langfuse.langchain import CallbackHandler

            # Create handler with trace metadata
            # Note: flush settings are now on the client, not the handler
            handler = CallbackHandler(
                trace_name=trace_name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata,
                tags=tags,
            )
            return handler
        except Exception as e:
            logger.error(f"Error creating CallbackHandler: {e}")
            return None

    @contextmanager
    def trace_langgraph_agent(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ):
        """
        Context manager to wrap LangGraph agent execution with a top-level trace span.

        This follows the Langfuse LangGraph cookbook pattern of wrapping the entire
        graph invocation in a span for better observability.

        Usage:
            with tracer.trace_langgraph_agent(name="agentic_rag", ...) as (trace_ctx, handler):
                result = graph.invoke(input, config={"callbacks": [handler]})
                trace_ctx.update(output=result)

        Args:
            name: Name for the trace span (e.g., "agentic_rag_graph")
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Additional metadata to attach
            tags: Optional tags for the trace

        Yields:
            Tuple of (trace_context, callback_handler) for graph execution
        """
        if not self.client:
            # Return no-op context if Langfuse is disabled
            yield (None, None)
            return

        # Create callback handler for LangChain/LangGraph integration
        # The handler will automatically create traces
        handler = self.get_callback_handler(
            trace_name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags,
        )

        # In Langfuse v3, the CallbackHandler manages tracing automatically
        # We just need to return the handler and a placeholder trace context
        # The actual trace will be created by the handler
        yield (None, handler)

    def get_trace_id(self, trace=None) -> Optional[str]:
        """
        Get the current trace ID from Langfuse context.

        In Langfuse v3, the CallbackHandler manages traces automatically.
        We can get the current trace ID using get_current_trace_id().

        Args:
            trace: Deprecated, not used in v3

        Returns:
            Trace ID string or None if trace is disabled
        """
        if not self.client:
            return None

        try:
            # In Langfuse v3, use get_current_trace_id()
            trace_id = self.client.get_current_trace_id()
            return trace_id
        except Exception as e:
            logger.error(f"Error getting trace ID: {e}")
            return None

    def submit_feedback(
        self,
        trace_id: str,
        score: float,
        name: str = "user-feedback",
        comment: Optional[str] = None,
    ) -> bool:
        """
        Submit user feedback for a trace (following Langfuse cookbook pattern).

        Args:
            trace_id: Trace ID from get_trace_id()
            score: Feedback score (0-1 or -1 to 1)
            name: Name of the score (default: "user-feedback")
            comment: Optional feedback comment

        Returns:
            True if feedback was submitted successfully, False otherwise
        """
        if not self.client:
            logger.warning("Cannot submit feedback: Langfuse is disabled")
            return False

        try:
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=score,
                comment=comment,
            )
            logger.info(f"Submitted feedback for trace {trace_id}: score={score}")
            return True
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return False

    def flush(self):
        """Flush any pending traces."""
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse: {e}")

    def shutdown(self):
        """Shutdown the Langfuse client."""
        if self.client:
            try:
                self.client.flush()
                self.client.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down Langfuse: {e}")

    @contextmanager
    def start_generation(
        self,
        name: str,
        model: str,
        input_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Start a generation span for LLM calls (following Langfuse cookbook pattern).

        This creates a generation observation that tracks:
        - Model name and parameters
        - Input prompt/messages
        - Output completion
        - Token usage
        - Latency

        Usage:
            with tracer.start_generation(name="decision_llm", model="llama3.2", input_data=prompt) as gen:
                response = await llm.generate(...)
                gen.update(output=response, usage_metadata={...})

        Args:
            name: Name for this generation (e.g., "decision_llm", "grading_llm")
            model: Model identifier (e.g., "gemma4:e2b", "gpt-4o")
            input_data: Input to the LLM (prompt or messages)
            metadata: Additional metadata (temperature, max_tokens, etc.)

        Yields:
            Generation context object for updates
        """
        if not self.client:
            # No-op context when disabled
            yield None
            return

        try:
            generation = self.client.generation(
                name=name,
                model=model,
                input=input_data,
                metadata=metadata or {},
            )
            yield generation
        except Exception as e:
            logger.error(f"Error creating generation span: {e}")
            yield None

    @contextmanager
    def start_span(
        self,
        name: str,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Start a generic span for non-LLM operations (following Langfuse cookbook pattern).

        Use this for operations like:
        - Document retrieval
        - Query rewriting logic
        - Document grading logic
        - Any other processing step

        Usage:
            with tracer.start_span(name="retrieve_papers", input_data={"query": q}) as span:
                docs = retrieve(...)
                span.update(output={"docs_count": len(docs)})

        Args:
            name: Name for this span (e.g., "retrieve_papers", "grade_documents")
            input_data: Input to this operation
            metadata: Additional metadata

        Yields:
            Span context object for updates
        """
        if not self.client:
            # No-op context when disabled
            yield None
            return

        try:
            span = self.client.span(
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
            yield span
        except Exception as e:
            logger.error(f"Error creating span: {e}")
            yield None

    def update_generation(
        self,
        generation,
        output: Any,
        usage_metadata: Optional[Dict[str, Any]] = None,
        completion_start_time: Optional[float] = None,
    ):
        """
        Update a generation span with output and usage metrics.

        Args:
            generation: Generation object from start_generation()
            output: LLM output/response
            usage_metadata: Token usage and timing info
                - prompt_tokens: int
                - completion_tokens: int
                - total_tokens: int
                - latency_ms: float
            completion_start_time: Optional start time for latency calculation
        """
        if not generation:
            return

        try:
            update_data = {"output": output}

            if usage_metadata:
                # Add usage metadata following Langfuse format
                if "prompt_tokens" in usage_metadata:
                    update_data["usage"] = {
                        "input": usage_metadata.get("prompt_tokens", 0),
                        "output": usage_metadata.get("completion_tokens", 0),
                        "total": usage_metadata.get("total_tokens", 0),
                    }

                # Add timing metadata
                if "latency_ms" in usage_metadata:
                    update_data["metadata"] = update_data.get("metadata", {})
                    update_data["metadata"]["latency_ms"] = usage_metadata["latency_ms"]

            generation.update(**update_data)
            generation.end()
        except Exception as e:
            logger.error(f"Error updating generation: {e}")

    def update_span(
        self,
        span,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ):
        """
        Update a span with output and metadata. Does not end the span; callers own the
        lifecycle via end_span() or the trace_* context managers.

        Args:
            span: Span object from start_span()
            output: Operation output
            metadata: Additional metadata to attach
            level: Log level (e.g., "ERROR", "WARNING") for error tracking
            status_message: Status or error message
        """
        if not span:
            return

        try:
            update_data = {}
            if output is not None:
                update_data["output"] = output
            if metadata:
                update_data["metadata"] = metadata
            if level:
                update_data["level"] = level
            if status_message:
                update_data["status_message"] = status_message

            if update_data:
                span.update(**update_data)
        except Exception as e:
            logger.error(f"Error updating span: {e}")

    # ⚠️ 上游修复(#2): 以下三个方法在上游 v3 重构后缺失，但 RAGTracer 与 Week 7
    # 智能体节点都在调用，缺失会抛 AttributeError。这里补全；未启用 Langfuse 时安全降级。

    @contextmanager
    def trace_rag_request(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """⚠️ 上游修复(#2): RAGTracer 用它作为顶层 trace 的上下文管理器。

        启用时创建一个根 span 并 yield；禁用或出错时 yield None（调用方均做了 None 判断）。
        """
        if not self.client:
            yield None
            return

        trace = None
        try:
            trace = self.client.span(
                name="rag_request",
                input={"query": query},
                metadata=metadata or {},
            )
            yield trace
        except Exception as e:
            logger.error(f"Error in trace_rag_request: {e}")
            yield None
        finally:
            if trace is not None:
                try:
                    trace.end()
                except Exception:
                    pass

    def create_span(
        self,
        trace=None,
        name: str = "span",
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """⚠️ 上游修复(#2): 创建一个 span 并返回（供调用方稍后 update/end）。

        在 v3 中父子关系由当前上下文自动管理，故 trace 参数仅为兼容旧签名。
        禁用或出错时返回 None。
        """
        if not self.client:
            return None
        try:
            return self.client.span(name=name, input=input_data, metadata=metadata or {})
        except Exception as e:
            logger.error(f"Error creating span: {e}")
            return None

    def end_span(
        self,
        span,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ):
        """⚠️ 上游修复(#2): 给 span 写入 output/metadata 并结束；span 为 None 时 no-op。"""
        if not span:
            return
        try:
            update_data = {}
            if output is not None:
                update_data["output"] = output
            if metadata:
                update_data["metadata"] = metadata
            if level:
                update_data["level"] = level
            if status_message:
                update_data["status_message"] = status_message
            if update_data:
                span.update(**update_data)
            span.end()
        except Exception as e:
            logger.error(f"Error ending span: {e}")
```

> **设计取舍（修复 #2）**：上游历史上 `LangfuseTracer` 是有 `create_span`/`end_span`/`trace_rag_request` 的，v3 重构（Week 7 提交）把它们删了却没更新调用方。本教程补回**与调用方签名匹配**、且**禁用时 no-op、启用时全程 try/except** 的版本。这样无论是否配置 Langfuse，`/ask`、`/stream`、`/ask-agentic` 都能跑——追踪是增益而非阻断（呼应第 [03](03-architecture-and-design.md) 章决策 9）。

---

## 9.3 RAG 追踪器：`src/services/langfuse/tracer.py`

`RAGTracer` 是面向 RAG 流程的便捷封装，基于上面的 `LangfuseTracer`。

### 文件：`src/services/langfuse/tracer.py`（逐字复制）

```python
"""Simple, efficient Langfuse tracing utility for RAG pipeline."""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .client import LangfuseTracer


class RAGTracer:
    """Clean, purpose-built tracer for RAG operations."""

    def __init__(self, tracer: LangfuseTracer):
        self.tracer = tracer

    @contextmanager
    def trace_request(self, user_id: str, query: str):
        """Main request trace context manager."""
        trace = None
        try:
            with self.tracer.trace_rag_request(
                query=query, user_id=user_id, session_id=f"session_{user_id}", metadata={"simplified_tracing": True}
            ) as trace:
                yield trace
        finally:
            if trace:
                self.tracer.flush()

    @contextmanager
    def trace_embedding(self, trace, query: str):
        """Query embedding operation with timing."""
        start_time = time.time()
        span = self.tracer.create_span(
            trace=trace, name="query_embedding", input_data={"query": query, "query_length": len(query)}
        )
        try:
            yield span
        finally:
            duration = time.time() - start_time
            if span:
                self.tracer.update_span(span=span, output={"embedding_duration_ms": round(duration * 1000, 2), "success": True})
                span.end()

    @contextmanager
    def trace_search(self, trace, query: str, top_k: int):
        """Search operation with timing."""
        span = self.tracer.create_span(trace=trace, name="search_retrieval", input_data={"query": query, "top_k": top_k})
        try:
            yield span
        finally:
            if span:
                span.end()

    def end_search(self, span, chunks: List[Dict], arxiv_ids: List[str], total_hits: int):
        """End search span with essential results."""
        if not span:
            return

        self.tracer.update_span(
            span=span,
            output={
                "chunks_returned": len(chunks),
                "unique_papers": len(set(arxiv_ids)),
                "total_hits": total_hits,
                "arxiv_ids": list(set(arxiv_ids)),
            },
        )

    @contextmanager
    def trace_prompt_construction(self, trace, chunks: List[Dict]):
        """Prompt building with timing."""
        span = self.tracer.create_span(trace=trace, name="prompt_construction", input_data={"chunk_count": len(chunks)})
        try:
            yield span
        finally:
            if span:
                span.end()

    def end_prompt(self, span, prompt: str):
        """End prompt span with final prompt."""
        if not span:
            return

        self.tracer.update_span(
            span=span,
            output={
                "prompt_length": len(prompt),
                # Don't duplicate the full prompt here since it's in llm_generation input
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            },
        )

    @contextmanager
    def trace_generation(self, trace, model: str, prompt: str):
        """LLM generation with timing."""
        span = self.tracer.create_span(
            trace=trace, name="llm_generation", input_data={"model": model, "prompt_length": len(prompt), "prompt": prompt}
        )
        try:
            yield span
        finally:
            if span:
                span.end()

    def end_generation(self, span, response: str, model: str):
        """End generation span with response."""
        if not span:
            return

        self.tracer.update_span(span=span, output={"response": response, "response_length": len(response), "model_used": model})

    def end_request(self, trace, response: str, total_duration: float):
        """End main request trace."""
        if not trace:
            return

        try:
            trace.update(
                output={"answer": response, "total_duration_seconds": round(total_duration, 3), "response_length": len(response)}
            )
        except Exception:
            # Silently fail - don't break the request for tracing issues
            pass
```

> `RAGTracer` 把 RAG 的每个阶段（embedding / search / prompt / generation）都封成上下文管理器，调用方只需 `with rag_tracer.trace_xxx(...) as span:`。所有方法对 `span/trace 为 None` 都安全（禁用 Langfuse 时全是 None）。

### 文件：`src/services/langfuse/factory.py`（逐字复制）

```python
from functools import lru_cache

from src.config import get_settings
from src.services.langfuse.client import LangfuseTracer


@lru_cache(maxsize=1)
def make_langfuse_tracer() -> LangfuseTracer:
    """
    Create and return a singleton Langfuse tracer instance.

    Returns:
        LangfuseTracer: Configured Langfuse tracer
    """
    settings = get_settings()
    return LangfuseTracer(settings)
```

---

## 9.4 Redis 精确缓存：`src/services/cache/`

```bash
mkdir -p src/services/cache
```

> **注意**：上游 `src/services/cache/` **没有 `__init__.py`**（作为命名空间包使用，导入仍正常）。本教程保持一致，不创建该文件。

### 文件：`src/services/cache/client.py`（逐字复制）

```python
import hashlib
import json
import logging
from datetime import timedelta
from typing import Optional

import redis
from src.config import RedisSettings
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)


class CacheClient:
    """Redis-based exact match cache for RAG queries."""

    def __init__(self, redis_client: redis.Redis, settings: RedisSettings):
        self.redis = redis_client
        self.settings = settings
        self.ttl = timedelta(hours=settings.ttl_hours)

    def _generate_cache_key(self, request: AskRequest) -> str:
        """Generate exact cache key based on request parameters."""
        key_data = {
            "query": request.query,
            "model": request.model,
            "top_k": request.top_k,
            "use_hybrid": request.use_hybrid,
            "categories": sorted(request.categories) if request.categories else [],
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"exact_cache:{key_hash}"

    async def find_cached_response(self, request: AskRequest) -> Optional[AskResponse]:
        """Find cached response for exact query match."""
        try:
            cache_key = self._generate_cache_key(request)

            # Simple Redis GET operation - O(1)
            cached_response = self.redis.get(cache_key)

            if cached_response:
                try:
                    response_data = json.loads(cached_response)
                    logger.info(f"Cache hit for exact query match")
                    return AskResponse(**response_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to deserialize cached response: {e}")
                    return None

            return None

        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None

    async def store_response(self, request: AskRequest, response: AskResponse) -> bool:
        """Store response for exact query matching."""
        try:
            cache_key = self._generate_cache_key(request)

            # Simple Redis SET operation with TTL
            success = self.redis.set(cache_key, response.model_dump_json(), ex=self.ttl)

            if success:
                logger.info(f"Stored response in exact cache with key {cache_key[:16]}...")
                return True
            else:
                logger.warning(f"Failed to store response in cache")
                return False

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False
```

### 缓存设计要点

- **精确匹配键**：把 `query`+`model`+`top_k`+`use_hybrid`+`categories` 规范化后做 SHA-256，取前 16 位做 key（`exact_cache:<hash>`）。**参数不同绝不会串答案**（正确性）。`categories` 先排序，保证顺序无关。
- **O(1) GET/SET**：精确缓存就是一次 Redis GET/SET，极快。
- **TTL**：默认 6 小时（`REDIS__TTL_HOURS`），防止陈旧答案。
- **全程 try/except**：缓存任何异常都不影响主流程（降级）。

### 文件：`src/services/cache/factory.py`（逐字复制）

```python
import logging

import redis
from src.config import Settings
from src.services.cache.client import CacheClient

logger = logging.getLogger(__name__)


def make_redis_client(settings: Settings) -> redis.Redis:
    """Create Redis client with connection pooling."""
    redis_settings = settings.redis

    try:
        client = redis.Redis(
            host=redis_settings.host,
            port=redis_settings.port,
            password=redis_settings.password if redis_settings.password else None,
            db=redis_settings.db,
            decode_responses=redis_settings.decode_responses,
            socket_timeout=redis_settings.socket_timeout,
            socket_connect_timeout=redis_settings.socket_connect_timeout,
            retry_on_timeout=True,
            retry_on_error=[redis.ConnectionError, redis.TimeoutError],
        )

        # Test connection
        client.ping()
        logger.info(f"Connected to Redis at {redis_settings.host}:{redis_settings.port}")
        return client

    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating Redis client: {e}")
        raise


def make_cache_client(settings: Settings) -> CacheClient:
    """Create exact match cache client."""
    try:
        redis_client = make_redis_client(settings)
        cache_client = CacheClient(redis_client, settings.redis)
        logger.info("Exact match cache client created successfully")
        return cache_client
    except Exception as e:
        logger.error(f"Failed to create cache client: {e}")
        raise
```

---

## 9.5 RAG 端点（含追踪与缓存）：`src/routers/ask.py`

这是真正的 `/api/v1/ask`（一次性）与 `/api/v1/stream`（SSE 流式）端点，已集成追踪与精确缓存。

### 文件：`src/routers/ask.py`（逐字复制）

```python
import json
import logging
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.dependencies import CacheDep, EmbeddingsDep, LangfuseDep, OllamaDep, OpenSearchDep
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.langfuse.tracer import RAGTracer

logger = logging.getLogger(__name__)

# Two separate routers - one for regular ask, one for streaming
ask_router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])


async def _prepare_chunks_and_sources(
    request: AskRequest,
    opensearch_client,
    embeddings_service,
    rag_tracer: RAGTracer,
    trace=None,
) -> tuple[List[Dict], List[str], List[str]]:
    """Retrieve and prepare chunks for RAG with clean tracing."""

    # Handle embeddings for hybrid search
    query_embedding = None
    if request.use_hybrid:
        with rag_tracer.trace_embedding(trace, request.query) as embedding_span:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                if embedding_span:
                    rag_tracer.tracer.update_span(embedding_span, output={"success": False, "error": str(e)})

    # Search with tracing
    with rag_tracer.trace_search(trace, request.query, request.top_k) as search_span:
        search_results = opensearch_client.search_unified(
            query=request.query,
            query_embedding=query_embedding,
            size=request.top_k,
            from_=0,
            categories=request.categories,
            use_hybrid=request.use_hybrid and query_embedding is not None,
            min_score=0.0,
        )

        # Extract essential data for LLM
        chunks = []
        arxiv_ids = []
        sources_set = set()

        for hit in search_results.get("hits", []):
            arxiv_id = hit.get("arxiv_id", "")

            # Minimal chunk data for LLM
            chunks.append(
                {
                    "arxiv_id": arxiv_id,
                    "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                }
            )

            if arxiv_id:
                arxiv_ids.append(arxiv_id)
                arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

        # End search span with essential metadata
        rag_tracer.end_search(search_span, chunks, arxiv_ids, search_results.get("total", 0))

    return chunks, list(sources_set), arxiv_ids


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> AskResponse:
    """Clean RAG endpoint with essential tracing and exact match caching."""

    rag_tracer = RAGTracer(langfuse_tracer)
    start_time = time.time()

    with rag_tracer.trace_request("api_user", request.query) as trace:
        try:
            # Check exact cache first
            cached_response = None
            if cache_client:
                try:
                    cached_response = await cache_client.find_cached_response(request)
                    if cached_response:
                        logger.info("Returning cached response for exact query match")
                        return cached_response
                except Exception as e:
                    logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

            # Generate query embedding for hybrid search if needed
            query_embedding = None

            # Retrieve chunks
            chunks, sources, _ = await _prepare_chunks_and_sources(
                request, opensearch_client, embeddings_service, rag_tracer, trace
            )

            if not chunks:
                response = AskResponse(
                    query=request.query,
                    answer="I couldn't find any relevant information in the papers to answer your question.",
                    sources=[],
                    chunks_used=0,
                    search_mode="bm25" if not request.use_hybrid else "hybrid",
                )
                rag_tracer.end_request(trace, response.answer, time.time() - start_time)
                return response

            # Build prompt
            with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                from src.services.ollama.prompts import RAGPromptBuilder

                prompt_builder = RAGPromptBuilder()

                try:
                    prompt_data = prompt_builder.create_structured_prompt(request.query, chunks)
                    final_prompt = prompt_data["prompt"]
                except Exception:
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)

                rag_tracer.end_prompt(prompt_span, final_prompt)

            # Generate answer
            with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                rag_response = await ollama_client.generate_rag_answer(query=request.query, chunks=chunks, model=request.model)
                answer = rag_response.get("answer", "Unable to generate answer")
                rag_tracer.end_generation(gen_span, answer, request.model)

            # Prepare response
            response = AskResponse(
                query=request.query,
                answer=answer,
                sources=sources,
                chunks_used=len(chunks),
                search_mode="bm25" if not request.use_hybrid else "hybrid",
            )

            rag_tracer.end_request(trace, answer, time.time() - start_time)

            # Store response in exact match cache
            if cache_client:
                try:
                    await cache_client.store_response(request, response)
                except Exception as e:
                    logger.warning(f"Failed to store response in cache: {e}")

            return response

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@stream_router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> StreamingResponse:
    """Clean streaming RAG endpoint."""

    async def generate_stream():
        rag_tracer = RAGTracer(langfuse_tracer)
        start_time = time.time()

        with rag_tracer.trace_request("api_user", request.query) as trace:
            try:
                # Check exact cache first
                if cache_client:
                    try:
                        cached_response = await cache_client.find_cached_response(request)
                        if cached_response:
                            logger.info("Returning cached response for exact streaming query match")

                            # Send metadata first (same format as non-cached)
                            metadata_response = {
                                "sources": cached_response.sources,
                                "chunks_used": cached_response.chunks_used,
                                "search_mode": cached_response.search_mode,
                            }
                            yield f"data: {json.dumps(metadata_response)}\n\n"

                            # Stream the cached response in chunks
                            for chunk in cached_response.answer.split():
                                yield f"data: {json.dumps({'chunk': chunk + ' '})}\n\n"

                            # Send completion signal with just the final answer
                            yield f"data: {json.dumps({'answer': cached_response.answer, 'done': True})}\n\n"
                            return
                    except Exception as e:
                        logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

                # Retrieve chunks
                chunks, sources, _ = await _prepare_chunks_and_sources(
                    request, opensearch_client, embeddings_service, rag_tracer, trace
                )

                if not chunks:
                    yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': [], 'done': True})}\n\n"
                    return

                # Send metadata first
                search_mode = "bm25" if not request.use_hybrid else "hybrid"
                metadata_response = {"sources": sources, "chunks_used": len(chunks), "search_mode": search_mode}
                yield f"data: {json.dumps(metadata_response)}\n\n"

                # Build prompt
                with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                    from src.services.ollama.prompts import RAGPromptBuilder

                    prompt_builder = RAGPromptBuilder()
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)
                    rag_tracer.end_prompt(prompt_span, final_prompt)

                # Stream generation
                with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                    full_response = ""
                    async for chunk in ollama_client.generate_rag_answer_stream(
                        query=request.query, chunks=chunks, model=request.model
                    ):
                        if chunk.get("response"):
                            text_chunk = chunk["response"]
                            full_response += text_chunk
                            yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                        if chunk.get("done", False):
                            rag_tracer.end_generation(gen_span, full_response, request.model)
                            yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
                            break

                rag_tracer.end_request(trace, full_response, time.time() - start_time)

                # Store response in exact match cache
                if cache_client and full_response:
                    try:
                        search_mode = "bm25" if not request.use_hybrid else "hybrid"
                        response_to_cache = AskResponse(
                            query=request.query,
                            answer=full_response,
                            sources=sources,
                            chunks_used=len(chunks),
                            search_mode=search_mode,
                        )
                        await cache_client.store_response(request, response_to_cache)
                    except Exception as e:
                        logger.warning(f"Failed to store streaming response in cache: {e}")

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(), media_type="text/plain", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
```

### 端点设计要点

- **两个 router**：`ask_router`（`/ask`）和 `stream_router`（`/stream`），在 `main.py` 分别注册。
- **缓存优先**：先查精确缓存，命中直接返回（流式时把缓存答案按词切片"伪流式"输出，保持前端体验一致）。
- **追踪贯穿全程**：`trace_request` 包住整个请求，内部再分 embedding / search / prompt / generation 子 span——Langfuse 上能看到完整瀑布图。
- **SSE 协议**：流式用 `data: {json}\n\n` 帧，先发元数据（sources/chunks_used/search_mode），再逐块发 `chunk`，最后发 `{answer, done:true}`。
- **缓存优雅降级**：`if cache_client:` —— 若未配置 Redis（`CacheDep` 返回 None），整段缓存逻辑跳过，RAG 照常工作。

### 文件：`src/routers/__init__.py`（逐字复制，最终版）

现在 `ask` 模块已存在，给出最终的路由包初始化：

```python
"""Router modules for the RAG API."""

# Import all available routers
from . import ask, hybrid_search, ping

__all__ = ["ask", "ping", "hybrid_search"]
```

---

## 9.6 接入 FastAPI（main.py / dependencies.py 增量）

> 增量改动；最终完整版见第 [10](10-week7-agentic-telegram.md) 章。

`src/dependencies.py` 加入：

```python
# import 区：
from src.services.cache.client import CacheClient
from src.services.langfuse.client import LangfuseTracer

# 函数：
def get_langfuse_tracer(request: Request) -> LangfuseTracer:
    """Get Langfuse tracer from the request state."""
    return request.app.state.langfuse_tracer


def get_cache_client(request: Request) -> CacheClient | None:
    """Get cache client from the request state."""
    return getattr(request.app.state, "cache_client", None)

# 类型别名：
LangfuseDep = Annotated[LangfuseTracer, Depends(get_langfuse_tracer)]
CacheDep = Annotated[CacheClient | None, Depends(get_cache_client)]
```

`src/main.py` 的 `lifespan` 里加入：

```python
    from src.services.langfuse.factory import make_langfuse_tracer
    from src.services.cache.factory import make_cache_client

    app.state.langfuse_tracer = make_langfuse_tracer()
    app.state.cache_client = make_cache_client(settings)
```

`src/main.py` 注册路由处加入：

```python
    from src.routers.ask import ask_router, stream_router
    app.include_router(ask_router, prefix="/api/v1")     # RAG question answering with LLM
    app.include_router(stream_router, prefix="/api/v1")  # Streaming RAG responses
```

> `CacheDep` 用 `getattr(..., None)`：即使 `make_cache_client` 在 Redis 不可用时抛错（lifespan 里可按需 try/except），端点也能拿到 `None` 而非崩溃。本项目 lifespan 直接赋值；若你担心 Redis 不可用，可在 lifespan 用 try/except 包裹 `make_cache_client`。

---

## 9.7 本周验证

### 启动全栈

```bash
# 确保模型已拉取（Week 5）、JINA_API_KEY 已填（Week 4）、Langfuse 密钥已填（9.1）
docker compose up -d --build api postgres opensearch redis ollama
# 如需追踪，另启 Langfuse 栈（见 9.1）
```

### 验证 /ask

```bash
curl -s -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is self-attention in transformers?", "top_k": 3, "use_hybrid": true, "model": "gemma4:e2b"}' \
  | python -m json.tool
```

期望：返回 `answer`、`sources`（PDF 链接）、`chunks_used`、`search_mode`。

### 验证缓存（第二次应秒回）

```bash
# 连发两次同样的请求，比较耗时
time curl -s -X POST http://localhost:8000/api/v1/ask -H "Content-Type: application/json" \
  -d '{"query": "What is self-attention in transformers?", "top_k": 3, "use_hybrid": true, "model": "gemma4:e2b"}' >/dev/null
time curl -s -X POST http://localhost:8000/api/v1/ask -H "Content-Type: application/json" \
  -d '{"query": "What is self-attention in transformers?", "top_k": 3, "use_hybrid": true, "model": "gemma4:e2b"}' >/dev/null
```

期望：第二次明显更快（命中缓存）。也可进 Redis 看 key：

```bash
docker exec -it rag-redis redis-cli KEYS 'exact_cache:*'
```

### 验证 /stream（SSE）

```bash
curl -N -X POST http://localhost:8000/api/v1/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain attention mechanism", "top_k": 3, "use_hybrid": true, "model": "gemma4:e2b"}'
```

期望：看到一连串 `data: {...}` 帧，逐步吐字，最后一帧含 `"done": true`。

### 验证 Gradio（现在 /stream 已就绪）

```bash
uv run python gradio_launcher.py
# 浏览器打开 http://localhost:7861，提问试试
```

### 验证 Langfuse 追踪

打开 **http://localhost:3001** → Traces，应能看到每次 `/ask` 的瀑布图：`rag_request` → `query_embedding` → `search_retrieval` → `prompt_construction` → `llm_generation`，含各阶段耗时。

> 若 Langfuse 未配置密钥，以上 `/ask`、`/stream`、Gradio **仍然全部可用**（追踪 no-op）——这正是修复 #2 要保证的。

---

## 9.8 本章小结

你已经有了：

- ✅ Langfuse 全链路追踪（含修复 #2：补全 `trace_rag_request`/`create_span`/`end_span`，禁用时安全降级）。
- ✅ `RAGTracer` 阶段化追踪封装。
- ✅ Redis 精确匹配缓存（O(1)、TTL、参数安全、优雅降级）。
- ✅ `/api/v1/ask` 与 `/api/v1/stream` 完整端点。
- ✅ Gradio 端到端可用。

**Week 6 里程碑**：生产级可观测性与缓存就绪。下一章 [`10-week7-agentic-telegram.md`](10-week7-agentic-telegram.md) 是收官——用 LangGraph 把系统升级为会"思考"的 Agentic RAG（护栏、文档评分、查询重写、自适应检索），并接上 Telegram 机器人，最后给出 `main.py` 与 `dependencies.py` 的最终完整版。
