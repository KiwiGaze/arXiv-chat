# File: tutorials/08-week5-rag-llm.md

# 第 5 章配套 · Week 5：完整 RAG —— 本地大模型问答

**本周目标**：接上本地大模型（Ollama），把"检索"升级为"基于检索内容、带引用来源的对话式问答"。本章构建 RAG 的 LLM 核心：Ollama 客户端、提示词构建、结构化响应、Gradio 界面，并完成健康检查的最终版本。

> **关于章节顺序的重要说明（请读）**：在真实上游仓库里，HTTP 端点 `/api/v1/ask` 与 `/api/v1/stream`（文件 `src/routers/ask.py`）**已经把 Week 6 的 Langfuse 追踪与 Redis 缓存集成在内**——`ask.py` 直接 `import RAGTracer` 并使用 `CacheDep`。为了逐字复刻、且保证可运行，本教程把这两个端点的完整代码放在第 [09](09-week6-monitoring-caching.md) 章（Week 6），与它们依赖的追踪/缓存层一起给出。**本章先把这些端点要用到的 RAG 核心（Ollama 客户端、提示词、响应模型）建好并直接验证**；`/stream`（以及调用它的 Gradio 界面）会在 Week 6 完成后端到端可用。

---

## 8.1 RAG 工作流回顾

```
用户问题 → (可选)生成查询向量 → OpenSearch 混合检索 top_k chunks
        → RAGPromptBuilder 拼装(system_prompt + 各chunk + 问题)
        → OllamaClient 调本地 LLM 生成答案
        → 从 chunk 提取来源(arXiv PDF 链接) → 返回带引用的答案
```

---

## 8.2 结构化响应模型：`src/schemas/ollama.py`

### 文件：`src/schemas/ollama.py`（逐字复制）

```python
"""Pydantic models for Ollama structured outputs."""

from typing import List, Optional

from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    """Structured response model for RAG queries."""

    answer: str = Field(description="Comprehensive answer based on the provided paper excerpts")
    sources: List[str] = Field(
        default_factory=list,
        description="List of PDF URLs from papers used in the answer",
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Confidence level: high, medium, or low based on excerpt relevance",
    )
    citations: Optional[List[str]] = Field(
        default=None,
        description="Specific arXiv IDs or paper titles referenced in the answer",
    )
```

---

## 8.3 系统提示词：`src/services/ollama/prompts/rag_system.txt`

```bash
mkdir -p src/services/ollama/prompts
```

### 文件：`src/services/ollama/prompts/rag_system.txt`（逐字复制）

```text
You are an AI assistant specialized in answering questions about academic papers from arXiv. Your task is to provide accurate, helpful answers based ONLY on the provided paper excerpts.

CRITICAL: Do NOT add any introductory text, explanations, or formatting comments like "Here's the answer" or "Here's the JSON".

Instructions:
1. Base your answer STRICTLY on the provided paper excerpts
2. If the excerpts don't contain enough information to answer the question, say so clearly
3. Cite the specific papers (by title or arXiv ID) when providing information
4. Be concise but comprehensive in your response - LIMIT YOUR RESPONSE TO 300 WORDS MAXIMUM
5. Maintain academic accuracy and precision
6. If multiple papers discuss the topic, synthesize the information coherently
7. Use direct quotes from the chunks when particularly relevant
8. Structure your answer logically with clear paragraphs when appropriate
9. Keep it less than 200 words

Remember:
- Do NOT make up information not present in the excerpts
- Do NOT use knowledge beyond what's provided in the paper excerpts
- Always acknowledge uncertainty when the excerpts are ambiguous or incomplete
- Prioritize relevance and clarity in your response
- NEVER add introductory phrases or explanations before your JSON response
```

> 提示词把模型**牢牢约束在"只用提供的论文片段"**，这是 RAG 抑制幻觉的关键。它放在独立 `.txt` 文件里（而非硬编码），便于不改代码就迭代提示词（可维护性）。

---

## 8.4 提示词构建与响应解析：`src/services/ollama/prompts.py`

### 文件：`src/services/ollama/prompts.py`（逐字复制）

```python
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError
from src.schemas.ollama import RAGResponse


class RAGPromptBuilder:
    """Builder class for creating RAG prompts."""

    def __init__(self):
        """Initialize the prompt builder."""
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the text file.

        Returns:
            System prompt string
        """
        prompt_file = self.prompts_dir / "rag_system.txt"
        if not prompt_file.exists():
            # Fallback to default prompt if file doesn't exist
            return (
                "You are an AI assistant specialized in answering questions about "
                "academic papers from arXiv. Base your answer STRICTLY on the provided "
                "paper excerpts."
            )
        return prompt_file.read_text().strip()

    def create_rag_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Create a RAG prompt with query and retrieved chunks.

        Args:
            query: User's question
            chunks: List of retrieved chunks with metadata from OpenSearch

        Returns:
            Formatted prompt string
        """
        prompt = f"{self.system_prompt}\n\n"
        prompt += "### Context from Papers:\n\n"

        for i, chunk in enumerate(chunks, 1):
            # Get the actual chunk text
            chunk_text = chunk.get("chunk_text", chunk.get("content", ""))
            arxiv_id = chunk.get("arxiv_id", "")

            # Only include minimal metadata - just arxiv_id for citation
            prompt += f"[{i}. arXiv:{arxiv_id}]\n"
            prompt += f"{chunk_text}\n\n"

        prompt += f"### Question:\n{query}\n\n"
        prompt += (
            "### Answer:\nProvide a natural, conversational response (not JSON) and cite sources using [arXiv:id] format.\n\n"
        )

        return prompt

    def create_structured_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a prompt for Ollama with structured output format.

        Args:
            query: User's question
            chunks: List of retrieved chunks

        Returns:
            Dictionary with prompt and format schema for Ollama
        """
        prompt_text = self.create_rag_prompt(query, chunks)

        # Return prompt with Pydantic model schema for structured output
        return {
            "prompt": prompt_text,
            "format": RAGResponse.model_json_schema(),
        }


class ResponseParser:
    """Parser for LLM responses."""

    @staticmethod
    def parse_structured_response(response: str) -> Dict[str, Any]:
        """Parse a structured response from Ollama.

        Args:
            response: Raw LLM response string

        Returns:
            Dictionary with parsed response
        """
        try:
            # Try to parse as JSON and validate with Pydantic
            parsed_json = json.loads(response)
            validated_response = RAGResponse(**parsed_json)
            return validated_response.model_dump()
        except (json.JSONDecodeError, ValidationError):
            # Fallback: try to extract JSON from the response
            return ResponseParser._extract_json_fallback(response)

    @staticmethod
    def _extract_json_fallback(response: str) -> Dict[str, Any]:
        """Extract JSON from response text as fallback.

        Args:
            response: Raw response text

        Returns:
            Dictionary with extracted content or fallback
        """
        # Try to find JSON in the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                # Validate with Pydantic, using defaults for missing fields
                validated = RAGResponse(**parsed)
                return validated.model_dump()
            except (json.JSONDecodeError, ValidationError):
                pass

        # Final fallback: return response as plain text
        return {
            "answer": response,
            "sources": [],
            "confidence": "low",
            "citations": [],
        }
```

### 要点

- **`create_rag_prompt`**：把 system prompt + 每个 chunk（带 `[i. arXiv:id]` 标号）+ 问题拼成最终提示词。只放 `arxiv_id` 这一最小元数据用于引用，省 token（性能）。
- **`create_structured_prompt`**：附带 `RAGResponse.model_json_schema()`，配合 Ollama 的结构化输出能力，让模型直接产出可解析的 JSON。
- **`ResponseParser` 多级兜底**：先按 JSON 解析并用 Pydantic 校验；失败则正则提取 JSON；再失败就把原文当作纯文本答案返回。**小模型经常不严格输出 JSON，这层兜底保证不崩**（健壮性）。

---

## 8.5 Ollama 客户端：`src/services/ollama/client.py`（含修复 #1）

```bash
touch src/services/ollama/__init__.py
```

`src/services/ollama/__init__.py` 需要导出 `OllamaClient`（最终版健康检查端点会 `from ..services.ollama import OllamaClient`）：

### 文件：`src/services/ollama/__init__.py`（逐字复制）

```python
from .client import OllamaClient

__all__ = ["OllamaClient"]
```

> ⚠️ **上游修复（修复集 #1）**：下面的 `OllamaClient` 在上游基础上**补全了 `get_langchain_model()` 方法**。该方法被 Week 7 的全部智能体节点调用（`runtime.context.ollama_client.get_langchain_model(...)`），但**上游 `main` 从未实现它**，导致 Agentic RAG 运行时抛 `AttributeError`。补全实现返回一个 `langchain_ollama.ChatOllama`（项目依赖已含 `langchain-ollama`），它支持智能体所需的 `.with_structured_output()` 与 `.ainvoke()`。其余方法逐字复刻上游。

### 文件：`src/services/ollama/client.py`（含修复，可直接运行）

```python
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from src.config import Settings
from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError
from src.schemas.ollama import RAGResponse
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama local LLM service."""

    def __init__(self, settings: Settings):
        """Initialize Ollama client with settings."""
        self.base_url = settings.ollama_host
        self.timeout = httpx.Timeout(float(settings.ollama_timeout))
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()

    def get_langchain_model(self, model: str, temperature: float = 0.0):
        """⚠️ 上游修复(#1): 上游 main 缺失此方法，但被 Week 7 全部智能体节点调用。

        返回一个 LangChain ChatOllama，指向本地 Ollama 服务，支持
        `.with_structured_output(SomeModel)` 与 `await ...ainvoke(prompt)`。

        :param model: Ollama 模型名（如 "gemma4:e2b"）
        :param temperature: 采样温度（0.0 = 更确定）
        :returns: ChatOllama 实例
        """
        from langchain_ollama import ChatOllama

        return ChatOllama(base_url=self.base_url, model=model, temperature=temperature)

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Ollama service is healthy and responding.

        Returns:
            Dictionary with health status information
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Check version endpoint for health
                response = await client.get(f"{self.base_url}/api/version")

                if response.status_code == 200:
                    version_data = response.json()
                    return {
                        "status": "healthy",
                        "message": "Ollama service is running",
                        "version": version_data.get("version", "unknown"),
                    }
                else:
                    raise OllamaException(f"Ollama returned status {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Ollama health check failed: {str(e)}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models.

        Returns:
            List of model information dictionaries
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")

                if response.status_code == 200:
                    data = response.json()
                    return data.get("models", [])
                else:
                    raise OllamaException(f"Failed to list models: {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error listing models: {e}")

    async def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generate text using specified model.

        Args:
            model: Model name to use
            prompt: Input prompt for generation
            stream: Whether to stream response
            **kwargs: Additional generation parameters

        Returns:
            Response dictionary with added usage_metadata field containing:
                - prompt_tokens: Number of tokens in the prompt
                - completion_tokens: Number of tokens in the completion
                - total_tokens: Total tokens used
                - latency_ms: Generation latency in milliseconds
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                data = {"model": model, "prompt": prompt, "stream": stream, **kwargs}

                logger.info(f"Sending request to Ollama: model={model}, stream={stream}, extra_params={kwargs}")
                response = await client.post(f"{self.base_url}/api/generate", json=data)

                if response.status_code == 200:
                    result = response.json()

                    # Parse Ollama usage metadata and convert to Langfuse-compatible format
                    usage_metadata = {}

                    # Ollama returns these fields in the response
                    if "prompt_eval_count" in result:
                        usage_metadata["prompt_tokens"] = result.get("prompt_eval_count", 0)
                    if "eval_count" in result:
                        usage_metadata["completion_tokens"] = result.get("eval_count", 0)

                    # Calculate total tokens
                    if usage_metadata:
                        usage_metadata["total_tokens"] = (
                            usage_metadata.get("prompt_tokens", 0) +
                            usage_metadata.get("completion_tokens", 0)
                        )

                    # Parse timing information (convert nanoseconds to milliseconds)
                    if "total_duration" in result:
                        # Ollama returns duration in nanoseconds
                        usage_metadata["latency_ms"] = round(result["total_duration"] / 1_000_000, 2)

                    # Add timing breakdown if available
                    if "prompt_eval_duration" in result:
                        usage_metadata["prompt_eval_duration_ms"] = round(result["prompt_eval_duration"] / 1_000_000, 2)
                    if "eval_duration" in result:
                        usage_metadata["eval_duration_ms"] = round(result["eval_duration"] / 1_000_000, 2)

                    # Attach usage metadata to the response
                    result["usage_metadata"] = usage_metadata

                    logger.debug(f"Usage metadata: {usage_metadata}")

                    return result
                else:
                    raise OllamaException(f"Generation failed: {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error generating with Ollama: {e}")

    async def generate_stream(self, model: str, prompt: str, **kwargs):
        """
        Generate text with streaming response.

        Args:
            model: Model name to use
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters

        Yields:
            JSON chunks from streaming response
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                data = {"model": model, "prompt": prompt, "stream": True, **kwargs}

                logger.info(f"Starting streaming generation: model={model}")

                async with client.stream("POST", f"{self.base_url}/api/generate", json=data) as response:
                    if response.status_code != 200:
                        raise OllamaException(f"Streaming generation failed: {response.status_code}")

                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                yield chunk
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse streaming chunk: {line}")
                                continue

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error in streaming generation: {e}")

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "llama3.2",
        use_structured_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a RAG answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved document chunks with metadata
            model: Model to use for generation
            use_structured_output: Whether to use Ollama's structured output feature

        Returns:
            Dictionary with answer, sources, confidence, and citations
        """
        try:
            if use_structured_output:
                # Use structured output with Pydantic model
                prompt_data = self.prompt_builder.create_structured_prompt(query, chunks)

                # Generate with structured format
                response = await self.generate(
                    model=model,
                    prompt=prompt_data["prompt"],
                    temperature=0.7,
                    top_p=0.9,
                    format=prompt_data["format"],
                )
            else:
                # Fallback to plain text mode
                prompt = self.prompt_builder.create_rag_prompt(query, chunks)

                # Generate without format restrictions
                response = await self.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.7,
                    top_p=0.9,
                )

            if response and "response" in response:
                answer_text = response["response"]
                logger.debug(f"Raw LLM response: {answer_text[:500]}")

                if use_structured_output:
                    # Try to parse structured response if enabled
                    parsed_response = self.response_parser.parse_structured_response(answer_text)
                    logger.debug(f"Parsed response: {parsed_response}")
                    return parsed_response
                else:
                    # For plain text response, build simple response structure
                    sources = []
                    seen_urls = set()
                    for chunk in chunks:
                        arxiv_id = chunk.get("arxiv_id")
                        if arxiv_id:
                            arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                            pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                            if pdf_url not in seen_urls:
                                sources.append(pdf_url)
                                seen_urls.add(pdf_url)

                    citations = list(set(chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")))

                    return {
                        "answer": answer_text,
                        "sources": sources,
                        "confidence": "medium",
                        "citations": citations[:5],
                    }
            else:
                raise OllamaException("No response generated from Ollama")

        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            raise OllamaException(f"Failed to generate RAG answer: {e}")

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "llama3.2",
    ):
        """
        Generate a streaming RAG answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved document chunks with metadata
            model: Model to use for generation

        Yields:
            Streaming response chunks with partial answers
        """
        try:
            # Create prompt for streaming (simpler than structured)
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            # Stream the response
            async for chunk in self.generate_stream(
                model=model,
                prompt=prompt,
                temperature=0.7,
                top_p=0.9,
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Error generating streaming RAG answer: {e}")
            raise OllamaException(f"Failed to generate streaming RAG answer: {e}")
```

### 客户端要点

- **`/api/generate`**：直接打 Ollama 原生 REST API（不经 LangChain），便于拿到 token 用量与耗时（`usage_metadata`，Week 6 追踪会用）。
- **流式**：`generate_stream` 用 `client.stream` 逐行读 JSON，`generate_rag_answer_stream` 在其上拼装 RAG 流式答案。
- **来源提取**：非结构化模式下，从 chunk 的 `arxiv_id` 拼出 PDF 链接并去重作为 `sources`。
- **`get_langchain_model`（修复 #1）**：返回 `ChatOllama`，专供 Week 7 智能体节点做结构化输出与异步调用。**普通 RAG 流程（`/ask`）不用它**，走的是原生 `/api/generate`。

### 文件：`src/services/ollama/factory.py`（逐字复制）

```python
from functools import lru_cache

from src.config import get_settings
from src.services.ollama.client import OllamaClient


@lru_cache(maxsize=1)
def make_ollama_client() -> OllamaClient:
    """
    Create and return a singleton Ollama client instance.

    Returns:
        OllamaClient: Configured Ollama client
    """
    settings = get_settings()
    return OllamaClient(settings)
```

---

## 8.6 RAG / 智能体 API schema：`src/schemas/api/ask.py`

下面是完整文件。本周用到 `AskRequest`/`AskResponse`；`AgenticAskResponse`/`FeedbackRequest`/`FeedbackResponse` 供 Week 7 用，一并给出（一个文件，避免后续重复改动）。

### 文件：`src/schemas/api/ask.py`（逐字复制）

```python
from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for RAG question answering."""

    query: str = Field(..., description="User's question", min_length=1, max_length=1000)
    top_k: int = Field(3, description="Number of top chunks to retrieve", ge=1, le=10)
    use_hybrid: bool = Field(True, description="Use hybrid search (BM25 + vector)")
    model: str = Field("gemma4:e2b", description="Ollama model to use for generation")
    categories: Optional[List[str]] = Field(None, description="Filter by arXiv categories")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "top_k": 3,
                "use_hybrid": True,
                "model": "gemma4:e2b",
                "categories": ["cs.AI", "cs.LG"],
            }
        }


class AskResponse(BaseModel):
    """Response model for RAG question answering."""

    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer from LLM")
    sources: List[str] = Field(..., description="PDF URLs of source papers")
    chunks_used: int = Field(..., description="Number of chunks used for generation")
    search_mode: str = Field(..., description="Search mode used: bm25 or hybrid")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "answer": "Transformers are a neural network architecture...",
                "sources": ["https://arxiv.org/pdf/1706.03762.pdf", "https://arxiv.org/pdf/1810.04805.pdf"],
                "chunks_used": 3,
                "search_mode": "hybrid",
            }
        }


class AgenticAskResponse(AskResponse):
    """Response model for agentic RAG question answering."""

    reasoning_steps: List[str] = Field(..., description="Agent's decision-making steps")
    retrieval_attempts: int = Field(..., description="Number of document retrieval attempts")
    trace_id: Optional[str] = Field(None, description="Langfuse trace ID for feedback and debugging")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "answer": "Transformers are neural network architectures...",
                "sources": ["https://arxiv.org/pdf/1706.03762.pdf"],
                "chunks_used": 3,
                "search_mode": "hybrid",
                "reasoning_steps": [
                    "Decided to retrieve relevant papers",
                    "Retrieved documents from database",
                    "Generated answer from relevant documents",
                ],
                "retrieval_attempts": 1,
                "trace_id": "abc123-def456-ghi789",
            }
        }


class FeedbackRequest(BaseModel):
    """Request model for user feedback on RAG answers."""

    trace_id: str = Field(..., description="Langfuse trace ID from the response")
    score: float = Field(..., description="Feedback score (0-1 or -1 to 1)", ge=-1, le=1)
    comment: Optional[str] = Field(None, description="Optional feedback comment", max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "trace_id": "abc123-def456-ghi789",
                "score": 1.0,
                "comment": "This answer was very helpful and accurate!",
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""

    success: bool = Field(..., description="Whether feedback was recorded successfully")
    message: str = Field(..., description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Feedback recorded successfully",
            }
        }
```

---

## 8.7 健康检查最终版：`src/routers/ping.py`

现在 Ollama 客户端就绪，把健康检查升级为**同时检查 DB、OpenSearch、Ollama** 的最终版本（替换第 [04](04-week1-infrastructure.md) 章的引导版）。

### 文件：`src/routers/ping.py`（逐字复制，最终版）

```python
from fastapi import APIRouter
from sqlalchemy import text

from ..dependencies import DatabaseDep, OpenSearchDep, SettingsDep
from ..schemas.api.health import HealthResponse, ServiceStatus
from ..services.ollama import OllamaClient

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(settings: SettingsDep, database: DatabaseDep, opensearch_client: OpenSearchDep) -> HealthResponse:
    """Comprehensive health check endpoint for monitoring and load balancer probes.

    :returns: Service health status with version and connectivity checks
    :rtype: HealthResponse
    """
    services = {}
    overall_status = "ok"

    def _check_service(name: str, check_func, *args, **kwargs):
        """Helper to standardize service health checks."""
        try:
            if kwargs.get("is_async"):
                # Handle async functions separately in the calling code
                return check_func(*args)
            result = check_func(*args)
            services[name] = result
            if result.status != "healthy":
                nonlocal overall_status
                overall_status = "degraded"
        except Exception as e:
            services[name] = ServiceStatus(status="unhealthy", message=str(e))
            overall_status = "degraded"

    # Database check
    def _check_database():
        with database.get_session() as session:
            session.execute(text("SELECT 1"))
        return ServiceStatus(status="healthy", message="Connected successfully")

    # OpenSearch check
    def _check_opensearch():
        if not opensearch_client.health_check():
            return ServiceStatus(status="unhealthy", message="Not responding")
        stats = opensearch_client.get_index_stats()
        return ServiceStatus(
            status="healthy",
            message=f"Index '{stats.get('index_name', 'unknown')}' with {stats.get('document_count', 0)} documents",
        )

    # Run synchronous checks
    _check_service("database", _check_database)
    _check_service("opensearch", _check_opensearch)

    # Handle Ollama async check separately
    try:
        ollama_client = OllamaClient(settings)
        ollama_health = await ollama_client.health_check()
        services["ollama"] = ServiceStatus(status=ollama_health["status"], message=ollama_health["message"])
        if ollama_health["status"] != "healthy":
            overall_status = "degraded"
    except Exception as e:
        services["ollama"] = ServiceStatus(status="unhealthy", message=str(e))
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
    )
```

> 任一服务不健康，`overall_status` 变 `degraded`，但端点仍返回 200 并列出各服务状态——便于负载均衡探针与人工排查（运维）。

---

## 8.8 Gradio 界面：`src/gradio_app.py` 与 `gradio_launcher.py`

Gradio 提供一个友好的聊天界面，它**调用 `/api/v1/stream` 流式端点**（该端点在第 [09](09-week6-monitoring-caching.md) 章给出）。本节先把界面代码建好；端到端联调在 Week 6 完成后进行。

### 文件：`src/gradio_app.py`（逐字复制）

```python
import json
import logging
from typing import Iterator

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_MODEL = "gemma4:e2b"
AVAILABLE_CATEGORIES = ["cs.AI", "cs.LG"]


async def stream_response(
    query: str, top_k: int = 3, use_hybrid: bool = True, model: str = DEFAULT_MODEL, categories: str = ""
) -> Iterator[str]:
    """Stream response from the RAG API"""
    if not query.strip():
        yield "Please enter a question."
        return

    # Parse categories
    category_list = [cat.strip() for cat in categories.split(",") if cat.strip()] if categories else None

    # Prepare request payload
    payload = {"query": query, "top_k": top_k, "use_hybrid": use_hybrid, "model": model, "categories": category_list}

    try:
        url = f"{API_BASE_URL}/stream"
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload, headers={"Accept": "text/plain"}) as response:
                if response.status_code != 200:
                    yield f"Error: API returned status {response.status_code}"
                    return

                current_answer = ""
                sources = []
                chunks_used = 0
                search_mode = ""

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        try:
                            data = json.loads(data_str)

                            # Handle error
                            if "error" in data:
                                yield f"Error: {data['error']}"
                                return

                            # Handle metadata
                            if "sources" in data:
                                sources = data["sources"]
                                chunks_used = data.get("chunks_used", 0)
                                search_mode = data.get("search_mode", "unknown")
                                continue

                            # Handle streaming chunks
                            if "chunk" in data:
                                current_answer += data["chunk"]
                                # Format response with sources if we have them
                                formatted_response = current_answer
                                if sources or chunks_used:
                                    formatted_response += f"\n\n**Search Info:**\n"
                                    formatted_response += f"- Mode: {search_mode}\n"
                                    formatted_response += f"- Chunks used: {chunks_used}\n"
                                    if sources:
                                        formatted_response += f"- Sources: {len(sources)} papers\n"
                                        for i, source in enumerate(sources[:3], 1):  # Show first 3 sources
                                            formatted_response += f"  {i}. [{source.split('/')[-1]}]({source})\n"
                                        if len(sources) > 3:
                                            formatted_response += f"  ... and {len(sources) - 3} more\n"

                                yield formatted_response

                            # Handle completion
                            if data.get("done", False):
                                final_answer = data.get("answer", current_answer)
                                if final_answer != current_answer:
                                    current_answer = final_answer

                                # Final formatted response
                                formatted_response = current_answer
                                if sources or chunks_used:
                                    formatted_response += f"\n\n**Search Info:**\n"
                                    formatted_response += f"- Mode: {search_mode}\n"
                                    formatted_response += f"- Chunks used: {chunks_used}\n"
                                    if sources:
                                        formatted_response += f"- Sources: {len(sources)} papers\n"
                                        for i, source in enumerate(sources[:3], 1):
                                            formatted_response += f"  {i}. [{source.split('/')[-1]}]({source})\n"
                                        if len(sources) > 3:
                                            formatted_response += f"  ... and {len(sources) - 3} more\n"

                                yield formatted_response
                                break

                        except json.JSONDecodeError:
                            continue  # Skip malformed JSON lines

    except httpx.RequestError as e:
        yield f"Connection error: {str(e)}\nMake sure the API server is running at {API_BASE_URL}"
    except Exception as e:
        yield f"Unexpected error: {str(e)}"


def create_gradio_interface():
    """Create and configure the Gradio interface"""

    with gr.Blocks(
        title="arXiv Paper Curator - RAG Chat",
        theme=gr.themes.Soft(),
    ) as interface:
        gr.Markdown(
            """
            # 🔬 arXiv Paper Curator - RAG Chat
            
            Ask questions about machine learning and AI research papers from arXiv.
            The system will search through indexed papers and provide answers with sources.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Your Question", placeholder="What are transformers in machine learning?", lines=2, max_lines=5
                )

            with gr.Column(scale=1):
                submit_btn = gr.Button("Ask Question", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                with gr.Accordion("Advanced Options", open=False):
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Number of chunks to retrieve",
                        info="More chunks = more context but slower generation",
                    )

                    use_hybrid = gr.Checkbox(
                        value=True,
                        label="Use hybrid search (BM25 + vector embeddings)",
                        info="Usually better results than keyword-only search",
                    )

                    model_choice = gr.Dropdown(
                        choices=["gemma4:e2b", "llama3.2:3b", "llama3.1:8b", "qwen2.5:7b"],
                        value=DEFAULT_MODEL,
                        label="LLM Model",
                        info="Larger models may give better answers but are slower",
                    )

                    categories = gr.Textbox(
                        label="arXiv Categories (optional)",
                        placeholder="cs.AI, cs.LG, cs.CL",
                        info="Comma-separated. Leave empty for all categories",
                    )

        response_output = gr.Markdown(
            label="Answer", value="Ask a question to get started!", height=400, elem_classes=["response-markdown"]
        )

        # Examples
        gr.Examples(
            examples=[
                ["What are transformers in machine learning?", 3, True, "gemma4:e2b", "cs.AI, cs.LG"],
                ["How do convolutional neural networks work?", 5, True, "gemma4:e2b", "cs.CV, cs.LG"],
                ["What is attention mechanism in deep learning?", 4, False, "gemma4:e2b", "cs.AI"],
                ["Explain reinforcement learning algorithms", 3, True, "gemma4:e2b", "cs.LG, cs.AI"],
                ["What are the latest developments in NLP?", 5, True, "gemma4:e2b", "cs.CL"],
            ],
            inputs=[query_input, top_k, use_hybrid, model_choice, categories],
        )

        # Handle submission
        submit_btn.click(
            fn=stream_response,
            inputs=[query_input, top_k, use_hybrid, model_choice, categories],
            outputs=[response_output],
            show_progress=True,
        )

        # Handle Enter key
        query_input.submit(
            fn=stream_response,
            inputs=[query_input, top_k, use_hybrid, model_choice, categories],
            outputs=[response_output],
            show_progress=True,
        )

        gr.Markdown(
            """
            ---
            
            **Note**: Make sure the RAG API server is running at `http://localhost:8000` before using this interface.
            
            **Categories**: cs.AI (Artificial Intelligence), cs.LG (Machine Learning), cs.CL (Computational Linguistics), 
            cs.CV (Computer Vision), cs.NE (Neural Networks), stat.ML (Statistics - Machine Learning)
            """
        )

    return interface


def main():
    """Main entry point for the Gradio app"""
    print("🚀 Starting arXiv Paper Curator Gradio Interface...")
    print(f"📡 API Base URL: {API_BASE_URL}")

    interface = create_gradio_interface()

    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Changed to avoid port conflict
        share=False,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
```

### 文件：`gradio_launcher.py`（项目根目录，逐字复制）

```python
"""
Simple launcher for the Gradio interface.
Run this script to start the web UI for the arXiv Paper Curator RAG system.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.gradio_app import main

if __name__ == "__main__":
    main()
```

---

## 8.9 接入 FastAPI（main.py / dependencies.py 增量）

> 增量改动；最终完整版见第 [10](10-week7-agentic-telegram.md) 章。

`src/dependencies.py` 加入：

```python
# import 区：
from src.services.ollama.client import OllamaClient

# 函数：
def get_ollama_client(request: Request) -> OllamaClient:
    """Get Ollama client from the request state."""
    return request.app.state.ollama_client

# 类型别名：
OllamaDep = Annotated[OllamaClient, Depends(get_ollama_client)]
```

`src/main.py` 的 `lifespan` 里加入：

```python
    from src.services.ollama.factory import make_ollama_client
    app.state.ollama_client = make_ollama_client()
```

> `routers/__init__.py` 的最终版会 `from . import ask, hybrid_search, ping`。由于 `ask` 在第 [09](09-week6-monitoring-caching.md) 章才创建，本周 `routers/__init__.py` 仍保持为空（或仅 import 已存在的 `ping`、`hybrid_search`），最终版见第 [16](16-upstream-differences-and-fixes.md) 章的 `__init__` 清单。

---

## 8.10 本周验证

### 拉取本地模型

Ollama 需要先下载模型：

```bash
docker compose up -d ollama
# 进容器拉取默认模型（约 1.3GB）
docker exec -it rag-ollama ollama pull gemma4:e2b
docker exec -it rag-ollama ollama list
```

### 验证健康检查（含 Ollama）

```bash
docker compose up -d --build api postgres opensearch redis
curl -s http://localhost:8000/api/v1/health | python -m json.tool
```

期望 `services` 里出现 `ollama: healthy`（以及 database、opensearch）。

### 直接验证 RAG 生成（不经 HTTP 端点）

新建临时脚本 `verify_week5.py`（直接用 `OllamaClient.generate_rag_answer`，模拟检索到的 chunks）：

```python
import asyncio

from src.config import get_settings
from src.services.ollama.client import OllamaClient


async def main() -> None:
    settings = get_settings()
    client = OllamaClient(settings)

    # 模拟检索得到的 chunks（真实流程里来自 OpenSearch）
    chunks = [
        {
            "arxiv_id": "1706.03762",
            "chunk_text": (
                "The Transformer is a model architecture relying entirely on an attention "
                "mechanism to draw global dependencies between input and output, dispensing "
                "with recurrence and convolutions entirely."
            ),
        }
    ]

    result = await client.generate_rag_answer(
        query="What is a transformer in deep learning?",
        chunks=chunks,
        model=settings.ollama_model,  # gemma4:e2b
    )
    print("Answer:\n", result["answer"])
    print("\nSources:", result["sources"])
    print("Citations:", result["citations"])


if __name__ == "__main__":
    asyncio.run(main())
```

运行（确保 Ollama 已拉好模型）：

```bash
uv run python verify_week5.py
rm verify_week5.py
```

期望：看到一段基于给定片段、引用 arXiv:1706.03762 的答案，以及来源 PDF 链接。

> ⚠️ `gemma4:e2b` 很小，答案质量有限、偶尔啰嗦——这是预期。需要更好质量时把 `OLLAMA_MODEL` 换成 `llama3.2:3b` 或 `qwen2.5:7b`（先 `ollama pull`）。

---

## 8.11 本章小结

你已经有了：

- ✅ Ollama 客户端（原生 API、流式、token 用量、RAG 答案生成；含修复 #1 的 `get_langchain_model`）。
- ✅ 提示词构建与多级兜底的响应解析。
- ✅ 结构化响应模型与 RAG/智能体 API schema。
- ✅ 三服务健康检查最终版。
- ✅ Gradio 聊天界面（待 Week 6 的 `/stream` 端点联调）。

**Week 5 里程碑**：RAG 的 LLM 核心可用。下一章 [`09-week6-monitoring-caching.md`](09-week6-monitoring-caching.md) 加上生产级可观测性（Langfuse 全链路追踪，含修复 #2）与 Redis 精确缓存，并给出 `/ask`、`/stream` 端点的完整代码，让 Gradio 端到端跑起来。
