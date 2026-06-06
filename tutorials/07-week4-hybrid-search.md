# File: tutorials/07-week4-hybrid-search.md

# 第 7 章　Week 4：智能分块、向量嵌入与 RRF 混合检索

**本周目标**：给检索加上"语义层"。包括：按章节智能分块、用 Jina 生成 1024 维向量、把"分块 + 嵌入"索引进 OpenSearch，并启用 **RRF 混合检索**（BM25 + 向量）。最后把统一检索端点 `/api/v1/hybrid-search/` 接上 HTTP。

> **本章需要 Jina API Key**（见第 [02](02-environment-and-dependencies.md) 章）。没有 Key 时混合检索会优雅降级为纯 BM25。

---

## 7.1 为什么要分块？

大模型上下文有限，且检索的"颗粒度"直接影响相关性。把整篇论文切成**带重叠的小块（chunk）**，每块单独嵌入和检索，能：

- 让检索命中"最相关的那一段"，而不是整篇。
- 控制送进 LLM 的上下文长度。
- 用重叠避免把一个完整语义切断在边界。

本项目用**混合分块策略**（见 `TextChunker`）：优先按章节切；章节太小就合并，太大就再按词数切，并给每块都带上"标题 + 摘要"作为上下文锚点。

---

## 7.2 分块数据模型：`src/schemas/indexing/models.py`

```bash
touch src/schemas/indexing/__init__.py
```

### 文件：`src/schemas/indexing/models.py`（逐字复制）

```python
from typing import Optional

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""

    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    overlap_with_previous: int
    overlap_with_next: int
    section_title: Optional[str] = None


class TextChunk(BaseModel):
    """A chunk of text with metadata."""

    text: str
    metadata: ChunkMetadata
    arxiv_id: str
    paper_id: str
```

---

## 7.3 文本分块器：`src/services/indexing/text_chunker.py`

```bash
mkdir -p src/services/indexing
touch src/services/indexing/__init__.py
```

> ⚠️ **上游修复（修复集 #3）**：上游 `text_chunker.py` 第 114 行调用 `self._reconstruct_text(words, text)`（**两个参数**），但 `_reconstruct_text` 只定义了一个参数（`words`），在"文本词数少于 `min_chunk_size`"的短文本分支会抛 `TypeError`。下面已修正为单参数 `self._reconstruct_text(words)`（在代码中用注释标出）。这是本教程对该文件做的**唯一**改动，其余逐字复刻上游（包括 `'\\n\\n'` 这类原样保留的写法）。

### 文件：`src/services/indexing/text_chunker.py`（含修复，可直接运行）

```python
import json
import logging
import re
from typing import Dict, List, Optional, Union

from src.schemas.indexing.models import ChunkMetadata, TextChunk

logger = logging.getLogger(__name__)


class TextChunker:
    """Service for chunking text into overlapping segments.

    Uses word-based chunking with configurable chunk size and overlap.
    Default: 600 words per chunk with 100 word overlap.
    """

    def __init__(self, chunk_size: int = 600, overlap_size: int = 100, min_chunk_size: int = 100):
        """Initialize text chunker.

        :param chunk_size: Target number of words per chunk
        :param overlap_size: Number of overlapping words between chunks
        :param min_chunk_size: Minimum words for a chunk to be valid
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size

        if overlap_size >= chunk_size:
            raise ValueError("Overlap size must be less than chunk size")

        logger.info(
            f"Text chunker initialized: chunk_size={chunk_size}, overlap_size={overlap_size}, min_chunk_size={min_chunk_size}"
        )

    def _split_into_words(self, text: str) -> List[str]:
        """Split text into words while preserving whitespace information.

        :param text: Input text
        :returns: List of words
        """
        # Split on whitespace while keeping the words
        words = re.findall(r"\S+", text)
        return words

    def _reconstruct_text(self, words: List[str]) -> str:
        """Reconstruct text from words.

        :param words: List of words
        :returns: Reconstructed text
        """
        return " ".join(words)

    def chunk_paper(
        self,
        title: str,
        abstract: str,
        full_text: str,
        arxiv_id: str,
        paper_id: str,
        sections: Optional[Union[Dict[str, str], str, list]] = None,
    ) -> List[TextChunk]:
        """Chunk a paper using hybrid section-based approach.

        Strategy:
        - For sections 100-800 words: Use as single chunk with title+abstract
        - For sections <100 words: Combine with adjacent sections
        - For sections >800 words: Split using traditional word-based chunking
        - Fallback to traditional chunking if no sections available

        :param title: Paper title
        :param abstract: Paper abstract
        :param full_text: Full text content
        :param arxiv_id: ArXiv ID of the paper
        :param paper_id: Database ID of the paper
        :param sections: Dictionary or JSON string of sections
        :returns: List of text chunks with metadata
        """
        # Try section-based chunking first
        if sections:
            try:
                section_chunks = self._chunk_by_sections(title, abstract, arxiv_id, paper_id, sections)
                if section_chunks:
                    logger.info(f"Created {len(section_chunks)} section-based chunks for {arxiv_id}")
                    return section_chunks
            except Exception as e:
                logger.warning(f"Section-based chunking failed for {arxiv_id}: {e}")

        # Fallback to traditional word-based chunking
        logger.info(f"Using traditional word-based chunking for {arxiv_id}")
        return self.chunk_text(full_text, arxiv_id, paper_id)

    def chunk_text(self, text: str, arxiv_id: str, paper_id: str) -> List[TextChunk]:
        """Chunk text into overlapping segments.

        :param text: Full text to chunk
        :param arxiv_id: ArXiv ID of the paper
        :param paper_id: Database ID of the paper
        :returns: List of text chunks with metadata
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for paper {arxiv_id}")
            return []

        # Split text into words
        words = self._split_into_words(text)

        if len(words) < self.min_chunk_size:
            logger.warning(f"Text for paper {arxiv_id} has only {len(words)} words, less than minimum {self.min_chunk_size}")
            # Return single chunk if text is too small
            if words:
                return [
                    TextChunk(
                        # ⚠️ 上游修复(#3): 原为 self._reconstruct_text(words, text)，多传了一个参数会抛 TypeError
                        text=self._reconstruct_text(words),
                        metadata=ChunkMetadata(
                            chunk_index=0,
                            start_char=0,
                            end_char=len(text),
                            word_count=len(words),
                            overlap_with_previous=0,
                            overlap_with_next=0,
                        ),
                        arxiv_id=arxiv_id,
                        paper_id=paper_id,
                    )
                ]
            return []

        chunks = []
        chunk_index = 0
        current_position = 0

        while current_position < len(words):
            # Calculate chunk boundaries
            chunk_start = current_position
            chunk_end = min(current_position + self.chunk_size, len(words))

            # Extract chunk words
            chunk_words = words[chunk_start:chunk_end]
            chunk_text = self._reconstruct_text(chunk_words)

            # Calculate character offsets (approximate)
            start_char = len(" ".join(words[:chunk_start])) if chunk_start > 0 else 0
            end_char = len(" ".join(words[:chunk_end]))

            # Calculate overlaps
            overlap_with_previous = min(self.overlap_size, chunk_start) if chunk_start > 0 else 0
            overlap_with_next = self.overlap_size if chunk_end < len(words) else 0

            # Create chunk
            chunk = TextChunk(
                text=chunk_text,
                metadata=ChunkMetadata(
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    word_count=len(chunk_words),
                    overlap_with_previous=overlap_with_previous,
                    overlap_with_next=overlap_with_next,
                    section_title=None,  # Could be enhanced with section detection
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            chunks.append(chunk)

            # Move to next chunk position (with overlap)
            current_position += self.chunk_size - self.overlap_size
            chunk_index += 1

            # Break if we've processed all words
            if chunk_end >= len(words):
                break

        logger.info(f"Chunked paper {arxiv_id}: {len(words)} words -> {len(chunks)} chunks")

        return chunks

    def _chunk_by_sections(
        self, title: str, abstract: str, arxiv_id: str, paper_id: str, sections: Union[Dict[str, str], str, list]
    ) -> List[TextChunk]:
        """Implement hybrid section-based chunking strategy.

        :param title: Paper title
        :param abstract: Paper abstract
        :param arxiv_id: ArXiv ID
        :param paper_id: Database ID
        :param sections: Sections data
        :returns: List of text chunks
        """
        # Parse sections data
        sections_dict = self._parse_sections(sections)
        if not sections_dict:
            return []

        # Filter and clean sections
        sections_dict = self._filter_sections(sections_dict, abstract)
        if not sections_dict:
            logger.warning(f"No meaningful sections found after filtering for {arxiv_id}")
            return []

        # Create header (title + abstract)
        header = f"{title}\n\nAbstract: {abstract}\n\n"

        # Process sections using hybrid strategy
        chunks = []
        small_sections = []  # Buffer for combining small sections

        section_items = list(sections_dict.items())

        for i, (section_title, section_content) in enumerate(section_items):
            content_str = str(section_content) if section_content else ""
            section_words = len(content_str.split())

            if section_words < 100:
                # Collect small sections to combine later
                small_sections.append((section_title, content_str, section_words))

                # If this is the last section or next section is large, process accumulated small sections
                if i == len(section_items) - 1 or len(str(section_items[i + 1][1]).split()) >= 100:
                    chunks.extend(self._create_combined_chunk(header, small_sections, chunks, arxiv_id, paper_id))
                    small_sections = []

            elif 100 <= section_words <= 800:
                # Perfect size - create single chunk
                chunk_text = f"{header}Section: {section_title}\n\n{content_str}"
                chunk = self._create_section_chunk(chunk_text, section_title, len(chunks), arxiv_id, paper_id)
                chunks.append(chunk)

            else:
                # Large section - split using traditional chunking
                section_text = f"Section: {section_title}\n\n{content_str}"
                full_section_text = f"{header}{section_text}"

                # Use traditional chunking but with section context
                section_chunks = self._split_large_section(
                    full_section_text, header, section_title, len(chunks), arxiv_id, paper_id
                )
                chunks.extend(section_chunks)

        return chunks

    def _parse_sections(self, sections: Union[Dict[str, str], str, list]) -> Dict[str, str]:
        """Parse sections data into a dictionary."""
        if isinstance(sections, dict):
            return sections
        elif isinstance(sections, list):
            # Handle list of sections directly
            result = {}
            for i, section in enumerate(sections):
                if isinstance(section, dict):
                    title = section.get("title", section.get("heading", f"Section {i + 1}"))
                    content = section.get("content", section.get("text", ""))
                    result[title] = content
                else:
                    result[f"Section {i + 1}"] = str(section)
            return result
        elif isinstance(sections, str):
            try:
                parsed = json.loads(sections)
                if isinstance(parsed, dict):
                    return parsed
                elif isinstance(parsed, list):
                    # Convert list to dict with enumerated keys
                    result = {}
                    for i, section in enumerate(parsed):
                        if isinstance(section, dict):
                            title = section.get("title", section.get("heading", f"Section {i + 1}"))
                            content = section.get("content", section.get("text", ""))
                            result[title] = content
                        else:
                            result[f"Section {i + 1}"] = str(section)
                    return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse sections JSON")
        return {}

    def _filter_sections(self, sections_dict: Dict[str, str], abstract: str) -> Dict[str, str]:
        """Filter out unwanted sections and avoid duplication.

        :param sections_dict: Dictionary of sections
        :param abstract: Paper abstract for duplication check
        :returns: Filtered dictionary of sections
        """
        filtered = {}
        abstract_words = set(abstract.lower().split())

        for section_title, section_content in sections_dict.items():
            content_str = str(section_content).strip()

            # Skip empty sections
            if not content_str:
                continue

            # Skip metadata/header sections based on title
            if self._is_metadata_section(section_title):
                continue

            # Skip sections that are duplicates of the abstract
            if self._is_duplicate_abstract(content_str, abstract, abstract_words):
                logger.debug(f"Skipping duplicate abstract section: {section_title}")
                continue

            # Skip sections that are too small and contain only metadata
            if len(content_str.split()) < 20 and self._is_metadata_content(content_str):
                logger.debug(f"Skipping metadata section: {section_title}")
                continue

            filtered[section_title] = content_str

        return filtered

    def _is_metadata_section(self, section_title: str) -> bool:
        """Check if a section title indicates metadata/header content."""
        title_lower = section_title.lower().strip()

        metadata_indicators = [
            "content",
            "header",
            "authors",
            "author",
            "affiliation",
            "email",
            "arxiv",
            "preprint",
            "submitted",
            "received",
            "accepted",
        ]

        # Exact matches or very short titles that are likely metadata
        if title_lower in metadata_indicators or len(title_lower) < 5:
            return True

        # Check if title contains only metadata indicators
        for indicator in metadata_indicators:
            if indicator in title_lower and len(title_lower) < 20:
                return True

        return False

    def _is_duplicate_abstract(self, content: str, abstract: str, abstract_words: set) -> bool:
        """Check if section content is a duplicate of the abstract."""
        content_lower = content.lower().strip()
        abstract_lower = abstract.lower().strip()

        # Direct string match (allowing for minor formatting differences)
        if abstract_lower in content_lower or content_lower in abstract_lower:
            return True

        # Word overlap check - if >80% of words overlap, likely duplicate
        content_words = set(content_lower.split())

        if len(abstract_words) > 10:  # Only check for substantial abstracts
            overlap = len(abstract_words.intersection(content_words))
            overlap_ratio = overlap / len(abstract_words)

            if overlap_ratio > 0.8:
                return True

        return False

    def _is_metadata_content(self, content: str) -> bool:
        """Check if content contains only metadata (emails, arxiv IDs, etc.)."""
        content_lower = content.lower()

        # Check for common metadata patterns
        metadata_patterns = [
            "@",  # Email addresses
            "arxiv:",  # ArXiv IDs
            "university",
            "institute",
            "department",
            "college",
            "gmail.com",
            "edu",
            "ac.uk",
            "preprint",
        ]

        # If content is mostly metadata patterns
        word_count = len(content.split())
        if word_count < 30:  # Short content
            metadata_word_count = sum(1 for pattern in metadata_patterns if pattern in content_lower)
            if metadata_word_count >= 2:  # Contains multiple metadata indicators
                return True

        return False

    def _create_combined_chunk(
        self, header: str, small_sections: List, existing_chunks: List, arxiv_id: str, paper_id: str
    ) -> List[TextChunk]:
        """Create chunks by combining small sections."""
        if not small_sections:
            return []

        # Combine all small sections
        combined_content = []
        total_words = 0

        for section_title, content, word_count in small_sections:
            combined_content.append(f"Section: {section_title}\n\n{content}")
            total_words += word_count

        combined_text = f"{header}{'\\n\\n'.join(combined_content)}"

        # If still too small, combine with previous chunk if possible
        if total_words + len(header.split()) < 200 and existing_chunks:
            # Try to merge with previous chunk
            prev_chunk = existing_chunks[-1]
            merged_text = f"{prev_chunk.text}\\n\\n{'\\n\\n'.join(combined_content)}"

            # Update the previous chunk
            existing_chunks[-1] = TextChunk(
                text=merged_text,
                metadata=ChunkMetadata(
                    chunk_index=prev_chunk.metadata.chunk_index,
                    start_char=0,
                    end_char=len(merged_text),
                    word_count=len(merged_text.split()),
                    overlap_with_previous=0,
                    overlap_with_next=0,
                    section_title=f"{prev_chunk.metadata.section_title} + Combined",
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            return []

        # Create new chunk with combined content
        sections_titles = [title for title, _, _ in small_sections]
        combined_title = " + ".join(sections_titles[:3])  # Limit title length
        if len(sections_titles) > 3:
            combined_title += f" + {len(sections_titles) - 3} more"

        chunk = self._create_section_chunk(combined_text, combined_title, len(existing_chunks), arxiv_id, paper_id)
        return [chunk]

    def _create_section_chunk(
        self, chunk_text: str, section_title: str, chunk_index: int, arxiv_id: str, paper_id: str
    ) -> TextChunk:
        """Create a single section-based chunk."""
        return TextChunk(
            text=chunk_text,
            metadata=ChunkMetadata(
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(chunk_text),
                word_count=len(chunk_text.split()),
                overlap_with_previous=0,
                overlap_with_next=0,
                section_title=section_title,
            ),
            arxiv_id=arxiv_id,
            paper_id=paper_id,
        )

    def _split_large_section(
        self, full_section_text: str, header: str, section_title: str, base_chunk_index: int, arxiv_id: str, paper_id: str
    ) -> List[TextChunk]:
        """Split large sections using traditional word-based chunking."""
        # Remove header from section text for chunking, then add back to each chunk
        section_only = full_section_text[len(header) :]

        # Use traditional chunking on section content
        traditional_chunks = self.chunk_text(section_only, arxiv_id, paper_id)

        # Add header to each chunk and update metadata
        enhanced_chunks = []
        for i, chunk in enumerate(traditional_chunks):
            enhanced_text = f"{header}{chunk.text}"

            enhanced_chunk = TextChunk(
                text=enhanced_text,
                metadata=ChunkMetadata(
                    chunk_index=base_chunk_index + i,
                    start_char=chunk.metadata.start_char,
                    end_char=chunk.metadata.end_char + len(header),
                    word_count=len(enhanced_text.split()),
                    overlap_with_previous=chunk.metadata.overlap_with_previous,
                    overlap_with_next=chunk.metadata.overlap_with_next,
                    section_title=f"{section_title} (Part {i + 1})",
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks
```

### 分块策略要点

- **滑动窗口 + 重叠**：`chunk_text` 用 `chunk_size=600` 词的窗口，每次前移 `chunk_size - overlap_size = 500` 词，相邻块重叠 100 词（避免切断语义）。
- **混合章节策略**（`chunk_paper` → `_chunk_by_sections`）：
  - 章节 100–800 词：单独成块（理想大小）。
  - 章节 <100 词：缓冲合并相邻小章节。
  - 章节 >800 词：再按词数切，每块带回 header。
  - 无章节时回退到纯滑窗 `chunk_text`。
- **过滤噪声**（`_filter_sections`）：跳过空章节、与摘要重复的章节、纯元数据章节（作者/邮箱/机构）。
- **每块都带 header（标题 + 摘要）**：给每个块一个"它属于哪篇论文"的锚点，提升检索与生成质量。

> **为什么用"词数"而不是"token 数"或"字符数"分块？**
> - **为什么这么选**：词数实现简单、与人类直觉一致、跨模型稳定（不绑定某个 tokenizer）。
> - **替代方案**：按 token（更精确对齐 LLM 上下文）、按字符、按句子/语义边界。
> - **优缺点**：词数 ✅ 简单稳定。❌ 与真实 token 数有偏差（英文约 1 词≈1.3 token）。
> - **影响**：足够好且无 tokenizer 依赖（可维护）；如需严格控长可换 token 分块。

---

## 7.4 Jina 嵌入：schema 与客户端

### 文件：`src/schemas/embeddings/jina.py`（逐字复制）

```bash
touch src/schemas/embeddings/__init__.py
```

```python
from typing import Dict, List

from pydantic import BaseModel


class JinaEmbeddingRequest(BaseModel):
    """Request model for Jina embeddings API."""

    model: str = "jina-embeddings-v3"
    task: str = "retrieval.passage"  # or "retrieval.query" for queries
    dimensions: int = 1024
    late_chunking: bool = False
    embedding_type: str = "float"
    input: List[str]


class JinaEmbeddingResponse(BaseModel):
    """Response model from Jina embeddings API."""

    model: str
    object: str = "list"
    usage: Dict[str, int]
    data: List[Dict]
```

### 文件：`src/services/embeddings/jina_client.py`（逐字复制）

```bash
mkdir -p src/services/embeddings
touch src/services/embeddings/__init__.py
```

```python
import logging
from typing import List

import httpx
from src.schemas.embeddings.jina import JinaEmbeddingRequest, JinaEmbeddingResponse

logger = logging.getLogger(__name__)


class JinaEmbeddingsClient:
    """Client for Jina AI embeddings API.

    Uses Jina embeddings v3 model with 1024 dimensions optimized for retrieval.
    Documentation: https://jina.ai/embeddings
    """

    def __init__(self, api_key: str, base_url: str = "https://api.jina.ai/v1"):
        """Initialize Jina embeddings client.

        :param api_key: Jina API key
        :param base_url: API base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("Jina embeddings client initialized")

    async def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embed text passages for indexing.

        :param texts: List of text passages to embed
        :param batch_size: Number of texts to process in each API call
        :returns: List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            request_data = JinaEmbeddingRequest(
                model="jina-embeddings-v3", task="retrieval.passage", dimensions=1024, input=batch
            )

            try:
                response = await self.client.post(
                    f"{self.base_url}/embeddings", headers=self.headers, json=request_data.model_dump()
                )
                response.raise_for_status()

                result = JinaEmbeddingResponse(**response.json())
                batch_embeddings = [item["embedding"] for item in result.data]
                embeddings.extend(batch_embeddings)

                logger.debug(f"Embedded batch of {len(batch)} passages")

            except httpx.HTTPError as e:
                logger.error(f"Error embedding passages: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in embed_passages: {e}")
                raise

        logger.info(f"Successfully embedded {len(texts)} passages")
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a search query.

        :param query: Query text to embed
        :returns: Embedding vector for the query
        """
        request_data = JinaEmbeddingRequest(model="jina-embeddings-v3", task="retrieval.query", dimensions=1024, input=[query])

        try:
            response = await self.client.post(f"{self.base_url}/embeddings", headers=self.headers, json=request_data.model_dump())
            response.raise_for_status()

            result = JinaEmbeddingResponse(**response.json())
            embedding = result.data[0]["embedding"]

            logger.debug(f"Embedded query: '{query[:50]}...'")
            return embedding

        except httpx.HTTPError as e:
            logger.error(f"Error embedding query: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in embed_query: {e}")
            raise

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
```

> **关键设计：建库与查询用不同 task**。`embed_passages` 用 `task="retrieval.passage"`（建库），`embed_query` 用 `task="retrieval.query"`（查询）。Jina v3 为"非对称检索"分别优化这两端，比用同一种嵌入效果更好。建库时按 `batch_size` 批量调用，降低请求次数（性能/成本）。

### 文件：`src/services/embeddings/factory.py`（逐字复制）

```python
from typing import Optional

from src.config import Settings, get_settings

from .jina_client import JinaEmbeddingsClient


def make_embeddings_service(settings: Optional[Settings] = None) -> JinaEmbeddingsClient:
    """Factory function to create embeddings service.

    Creates a new client instance each time to avoid closed client issues.

    :param settings: Optional settings instance
    :returns: JinaEmbeddingsClient instance
    """
    if settings is None:
        settings = get_settings()

    # Get API key from settings
    api_key = settings.jina_api_key

    return JinaEmbeddingsClient(api_key=api_key)


def make_embeddings_client(settings: Optional[Settings] = None) -> JinaEmbeddingsClient:
    """Factory function to create embeddings client.

    Creates a new client instance each time to avoid closed client issues.

    :param settings: Optional settings instance
    :returns: JinaEmbeddingsClient instance
    """
    if settings is None:
        settings = get_settings()

    # Get API key from settings
    api_key = settings.jina_api_key

    return JinaEmbeddingsClient(api_key=api_key)
```

> **注意：嵌入工厂故意不加 `lru_cache`**。`JinaEmbeddingsClient` 内部持有一个 `httpx.AsyncClient`，若做成单例并被某处 `close()`，后续请求会用到已关闭的客户端。每次新建可规避这个陷阱（这是对"工厂=单例"惯例的有意例外，呼应第 [03](03-architecture-and-design.md) 章模式一）。

---

## 7.5 混合索引服务：`src/services/indexing/hybrid_indexer.py`

### 文件：`src/services/indexing/hybrid_indexer.py`（逐字复制）

```python
import logging
from typing import Dict, List, Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient

from .text_chunker import TextChunker

logger = logging.getLogger(__name__)


class HybridIndexingService:
    """Service for indexing papers with chunking and embeddings for hybrid search.

    Orchestrates the process of:
    1. Chunking papers into overlapping segments
    2. Generating embeddings for each chunk
    3. Indexing chunks with embeddings into OpenSearch
    """

    def __init__(self, chunker: TextChunker, embeddings_client: JinaEmbeddingsClient, opensearch_client: OpenSearchClient):
        """Initialize hybrid indexing service.

        :param chunker: Text chunking service
        :param embeddings_client: Embeddings generation client
        :param opensearch_client: OpenSearch client
        """
        self.chunker = chunker
        self.embeddings_client = embeddings_client
        self.opensearch_client = opensearch_client

        logger.info("Hybrid indexing service initialized")

    async def index_paper(self, paper_data: Dict) -> Dict[str, int]:
        """Index a single paper with chunking and embeddings.

        :param paper_data: Paper data from database
        :returns: Dictionary with indexing statistics
        """
        arxiv_id = paper_data.get("arxiv_id")
        paper_id = str(paper_data.get("id", ""))

        if not arxiv_id:
            logger.error("Paper missing arxiv_id")
            return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 1}

        try:
            # Step 1: Chunk the paper using hybrid section-based approach
            chunks = self.chunker.chunk_paper(
                title=paper_data.get("title", ""),
                abstract=paper_data.get("abstract", ""),
                full_text=paper_data.get("raw_text", paper_data.get("full_text", "")),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
                sections=paper_data.get("sections"),
            )

            if not chunks:
                logger.warning(f"No chunks created for paper {arxiv_id}")
                return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 0}

            logger.info(f"Created {len(chunks)} chunks for paper {arxiv_id}")

            # Step 2: Generate embeddings for chunks
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await self.embeddings_client.embed_passages(
                texts=chunk_texts,
                batch_size=50,  # Process in batches
            )

            if len(embeddings) != len(chunks):
                logger.error(f"Embedding count mismatch: {len(embeddings)} != {len(chunks)}")
                return {"chunks_created": len(chunks), "chunks_indexed": 0, "embeddings_generated": len(embeddings), "errors": 1}

            # Step 3: Prepare chunks with embeddings for indexing
            chunks_with_embeddings = []

            for chunk, embedding in zip(chunks, embeddings):
                # Prepare chunk data for OpenSearch
                chunk_data = {
                    "arxiv_id": chunk.arxiv_id,
                    "paper_id": chunk.paper_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "chunk_text": chunk.text,
                    "chunk_word_count": chunk.metadata.word_count,
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char,
                    "section_title": chunk.metadata.section_title,
                    "embedding_model": "jina-embeddings-v3",
                    # Denormalized paper metadata for efficient search
                    "title": paper_data.get("title", ""),
                    "authors": ", ".join(paper_data.get("authors", []))
                    if isinstance(paper_data.get("authors"), list)
                    else paper_data.get("authors", ""),
                    "abstract": paper_data.get("abstract", ""),
                    "categories": paper_data.get("categories", []),
                    "published_date": paper_data.get("published_date"),
                }

                chunks_with_embeddings.append({"chunk_data": chunk_data, "embedding": embedding})

            # Step 4: Index chunks into OpenSearch
            results = self.opensearch_client.bulk_index_chunks(chunks_with_embeddings)

            logger.info(f"Indexed paper {arxiv_id}: {results['success']} chunks successful, {results['failed']} failed")

            return {
                "chunks_created": len(chunks),
                "chunks_indexed": results["success"],
                "embeddings_generated": len(embeddings),
                "errors": results["failed"],
            }

        except Exception as e:
            logger.error(f"Error indexing paper {arxiv_id}: {e}")
            return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 1}

    async def index_papers_batch(self, papers: List[Dict], replace_existing: bool = False) -> Dict[str, int]:
        """Index multiple papers in batch.

        :param papers: List of paper data
        :param replace_existing: If True, delete existing chunks before indexing
        :returns: Aggregated statistics
        """
        total_stats = {
            "papers_processed": 0,
            "total_chunks_created": 0,
            "total_chunks_indexed": 0,
            "total_embeddings_generated": 0,
            "total_errors": 0,
        }

        for paper in papers:
            arxiv_id = paper.get("arxiv_id")

            # Optionally delete existing chunks
            if replace_existing and arxiv_id:
                self.opensearch_client.delete_paper_chunks(arxiv_id)

            # Index the paper
            stats = await self.index_paper(paper)

            # Update totals
            total_stats["papers_processed"] += 1
            total_stats["total_chunks_created"] += stats["chunks_created"]
            total_stats["total_chunks_indexed"] += stats["chunks_indexed"]
            total_stats["total_embeddings_generated"] += stats["embeddings_generated"]
            total_stats["total_errors"] += stats["errors"]

        logger.info(
            f"Batch indexing complete: {total_stats['papers_processed']} papers, "
            f"{total_stats['total_chunks_indexed']} chunks indexed"
        )

        return total_stats

    async def reindex_paper(self, arxiv_id: str, paper_data: Dict) -> Dict[str, int]:
        """Reindex a paper by deleting old chunks and creating new ones.

        :param arxiv_id: ArXiv ID of the paper
        :param paper_data: Updated paper data
        :returns: Indexing statistics
        """
        # Delete existing chunks
        deleted = self.opensearch_client.delete_paper_chunks(arxiv_id)
        if deleted:
            logger.info(f"Deleted existing chunks for paper {arxiv_id}")

        # Index with new data
        return await self.index_paper(paper_data)
```

### 索引服务要点

- **四步流水线**：分块 → 嵌入 → 组装文档（含反范式化的论文元数据）→ 批量写入。
- **反范式化（denormalization）**：把 `title`/`authors`/`abstract`/`categories` 复制进**每个 chunk 文档**。这样 BM25 既能匹配块正文也能匹配标题/摘要，且检索结果直接带论文信息，无需再回查 PostgreSQL（性能：避免二次查询）。代价是存储冗余（可接受）。
- **`replace_existing`**：重新索引前先删旧块，避免重复（Airflow 每日索引用 `replace_existing=True`）。
- **数量校验**：嵌入数与块数不一致直接判失败，防止错位。

### 文件：`src/services/indexing/factory.py`（逐字复制）

```python
from typing import Optional

from src.config import Settings, get_settings
from src.services.embeddings.factory import make_embeddings_client
from src.services.opensearch.factory import make_opensearch_client_fresh

from .hybrid_indexer import HybridIndexingService
from .text_chunker import TextChunker


def make_hybrid_indexing_service(
    settings: Optional[Settings] = None, opensearch_host: Optional[str] = None
) -> HybridIndexingService:
    """Factory function to create hybrid indexing service.

    Creates a new service instance each time.

    :param settings: Optional settings instance
    :param opensearch_host: Optional OpenSearch host override
    :returns: HybridIndexingService instance
    """
    if settings is None:
        settings = get_settings()

    # Create dependencies using configuration
    chunker = TextChunker(
        chunk_size=settings.chunking.chunk_size,
        overlap_size=settings.chunking.overlap_size,
        min_chunk_size=settings.chunking.min_chunk_size,
    )
    embeddings_client = make_embeddings_client(settings)
    opensearch_client = make_opensearch_client_fresh(settings, host=opensearch_host)

    # Create indexing service
    return HybridIndexingService(chunker=chunker, embeddings_client=embeddings_client, opensearch_client=opensearch_client)
```

> 这就是 Week 2 里 Airflow `indexing.py` 调用的 `make_hybrid_indexing_service`。现在它的依赖（chunker / embeddings / opensearch）都齐了，**Airflow 的索引任务从此可端到端运行**。

---

## 7.6 RRF 混合检索是怎么工作的

回顾 `OpenSearchClient._search_hybrid_native`（第 [06](06-week3-opensearch-bm25.md) 章已给出完整代码）：

```python
hybrid_query = {"hybrid": {"queries": [bm25_query, {"knn": {"embedding": {"vector": query_embedding, "k": size * 2}}}]}}
response = self.client.search(index=..., body=search_body, params={"search_pipeline": HYBRID_RRF_PIPELINE["id"]})
```

流程：
1. 同时跑 **BM25 查询**（关键词）和 **kNN 查询**（向量）。
2. 各自得到一个**排名列表**。
3. RRF 管道按 `1/(k + rank)`（默认 `k=60`）给每个文档在两个列表中的排名打分并相加，得到融合排名。
4. 取 `size * 2` 再截断，提升召回。

**RRF 的妙处**：它只用"排名"不用"原始分数"，所以 BM25 分数（无上界）和余弦相似度（0–1）的量纲差异完全不影响融合——无需归一化、无需调权重。

---

## 7.7 统一检索端点：`src/routers/hybrid_search.py`

现在嵌入服务齐备，可以把统一检索接上 HTTP 了。**一个端点支持 BM25 / 向量 / 混合三种模式**（由 `use_hybrid` 与是否成功生成向量决定）。

### 文件：`src/routers/hybrid_search.py`（逐字复制）

```python
import logging

from fastapi import APIRouter, HTTPException
from src.dependencies import EmbeddingsDep, OpenSearchDep
from src.schemas.api.search import HybridSearchRequest, SearchHit, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hybrid-search", tags=["hybrid-search"])


@router.post("/", response_model=SearchResponse)
async def hybrid_search(
    request: HybridSearchRequest, opensearch_client: OpenSearchDep, embeddings_service: EmbeddingsDep
) -> SearchResponse:
    """
    Hybrid search endpoint supporting multiple search modes.
    """
    try:
        if not opensearch_client.health_check():
            raise HTTPException(status_code=503, detail="Search service is currently unavailable")

        query_embedding = None
        if request.use_hybrid:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                query_embedding = None

        logger.info(f"Hybrid search: '{request.query}' (hybrid: {request.use_hybrid and query_embedding is not None})")

        results = opensearch_client.search_unified(
            query=request.query,
            query_embedding=query_embedding,
            size=request.size,
            from_=request.from_,
            categories=request.categories,
            latest=request.latest_papers,
            use_hybrid=request.use_hybrid,
            min_score=request.min_score,
        )

        hits = []
        for hit in results.get("hits", []):
            hits.append(
                SearchHit(
                    arxiv_id=hit.get("arxiv_id", ""),
                    title=hit.get("title", ""),
                    authors=hit.get("authors"),
                    abstract=hit.get("abstract"),
                    published_date=hit.get("published_date"),
                    pdf_url=hit.get("pdf_url"),
                    score=hit.get("score", 0.0),
                    highlights=hit.get("highlights"),
                    chunk_text=hit.get("chunk_text"),
                    chunk_id=hit.get("chunk_id"),
                    section_name=hit.get("section_name"),
                )
            )

        search_response = SearchResponse(
            query=request.query,
            total=results.get("total", 0),
            hits=hits,
            size=request.size,
            **{"from": request.from_},
            search_mode="hybrid" if (request.use_hybrid and query_embedding) else "bm25",
        )

        logger.info(f"Search completed: {search_response.total} results returned")
        return search_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
```

> **优雅降级**：如果生成向量失败（如 Jina 不可用），`query_embedding` 为 `None`，`search_unified` 自动退回纯 BM25——检索仍可用。`search_mode` 字段如实反映实际用了哪种模式。

---

## 7.8 接入 FastAPI（main.py / dependencies.py 增量）

> 增量改动；最终完整版见第 [10](10-week7-agentic-telegram.md) 章。

`src/dependencies.py` 加入：

```python
# import 区：
from src.services.embeddings.jina_client import JinaEmbeddingsClient

# 函数：
def get_embeddings_service(request: Request) -> JinaEmbeddingsClient:
    """Get embeddings service from the request state."""
    return request.app.state.embeddings_service

# 类型别名：
EmbeddingsDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_service)]
```

`src/main.py` 的 `lifespan` 里加入：

```python
    from src.services.embeddings.factory import make_embeddings_service
    app.state.embeddings_service = make_embeddings_service()
```

`src/main.py` 注册路由处加入：

```python
    from src.routers import hybrid_search
    app.include_router(hybrid_search.router, prefix="/api/v1")  # Search chunks with BM25/hybrid
```

---

## 7.9 本周验证

确保 `.env` 里填了真实的 `JINA_API_KEY`，并已有论文入库（Week 2）。

### 方式 A：触发 Airflow DAG（推荐，端到端）

```bash
docker compose up -d --build api airflow opensearch postgres
```

打开 http://localhost:8080，手动触发 `arxiv_paper_ingestion`。完成后 `index_papers_hybrid` 会把论文分块、嵌入并写入 OpenSearch。

### 方式 B：直接调用统一检索端点

```bash
# 混合检索
curl -s -X POST http://localhost:8000/api/v1/hybrid-search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanism", "size": 5, "use_hybrid": true}' | python -m json.tool

# 纯 BM25（关掉 hybrid）
curl -s -X POST http://localhost:8000/api/v1/hybrid-search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanism", "size": 5, "use_hybrid": false}' | python -m json.tool
```

期望：返回 `hits` 列表，每条含 `arxiv_id`、`chunk_text`、`score`；`search_mode` 显示 `hybrid` 或 `bm25`。对比两种模式的结果，体会向量带来的语义召回差异。

> 你也可以验证降级：把 `.env` 的 `JINA_API_KEY` 改错再请求 `use_hybrid:true`，日志会出现 "falling back to BM25"，`search_mode` 返回 `bm25`。

---

## 7.10 本章小结

你已经有了：

- ✅ 智能混合分块（按章节 + 滑窗 + 过滤噪声 + 每块带 header）。
- ✅ Jina v3 嵌入（passage/query 双模式、批量、异步）。
- ✅ `HybridIndexingService`（分块→嵌入→反范式化→批量索引）。
- ✅ RRF 混合检索（BM25 + 向量，免调参融合）。
- ✅ 统一检索端点 `/api/v1/hybrid-search/`（三模式 + 优雅降级）。
- ✅ Airflow 索引任务从此端到端可用。

**Week 4 里程碑**：智能的混合检索引擎就绪。下一章 [`08-week5-rag-llm.md`](08-week5-rag-llm.md) 接上本地大模型，把"检索"升级为"带引用的对话式问答"，并提供流式响应与 Gradio 界面。
