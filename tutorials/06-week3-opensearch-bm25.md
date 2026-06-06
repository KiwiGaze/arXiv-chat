# File: tutorials/06-week3-opensearch-bm25.md

# 第 6 章　Week 3：OpenSearch 索引与 BM25 关键词检索

**本周目标**：把论文内容索引进 OpenSearch，实现 **BM25 关键词检索**——这是整个 RAG 的检索地基（回顾第 [01](01-project-overview.md)、[03](03-architecture-and-design.md) 章的"搜索优先"哲学）。

> **一个索引，服务所有检索模式**：本项目用**单个混合索引**（`arxiv-papers-chunks`）同时承载 BM25 文本字段和 kNN 向量字段。本章创建这个索引并打通 BM25；Week 4 再往里写入向量、启用混合检索。因此本章会**完整给出 OpenSearch 客户端**（含 Week 4 才用到的向量/混合方法），你一次创建到位即可。

---

## 6.1 索引映射与 RRF 管道：`src/services/opensearch/index_config_hybrid.py`

```bash
mkdir -p src/services/opensearch
touch src/services/opensearch/__init__.py
```

### 文件：`src/services/opensearch/index_config_hybrid.py`（逐字复制）

```python
"""OpenSearch index configuration for hybrid search (BM25 + Vector).

This configuration supports both keyword search (BM25) and vector similarity search
using HNSW algorithm for approximate nearest neighbor search.
"""

ARXIV_PAPERS_CHUNKS_INDEX = "arxiv-papers-chunks"

# Index mapping for chunked papers with vector embeddings
ARXIV_PAPERS_CHUNKS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,
        "index.knn.space_type": "cosinesimil",
        "analysis": {
            "analyzer": {
                "standard_analyzer": {"type": "standard", "stopwords": "_english_"},
                "text_analyzer": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "stop", "snowball"]},
            }
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "chunk_id": {"type": "keyword"},
            "arxiv_id": {"type": "keyword"},
            "paper_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "chunk_text": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "chunk_word_count": {"type": "integer"},
            "start_char": {"type": "integer"},
            "end_char": {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,  # Jina v3 embeddings dimension
                "method": {
                    "name": "hnsw",  # Hierarchical Navigable Small World
                    "space_type": "cosinesimil",  # Cosine similarity
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 512,  # Higher value = better recall, slower indexing
                        "m": 16,  # Number of bi-directional links
                    },
                },
            },
            "title": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "authors": {
                "type": "text",
                "analyzer": "standard_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "abstract": {"type": "text", "analyzer": "text_analyzer"},
            "categories": {"type": "keyword"},
            "published_date": {"type": "date"},
            "section_title": {"type": "keyword"},
            "embedding_model": {"type": "keyword"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        },
    },
}

HYBRID_RRF_PIPELINE = {
    "id": "hybrid-rrf-pipeline",
    "description": "Post processor for hybrid RRF search",
    "phase_results_processors": [
        {
            "score-ranker-processor": {
                "combination": {
                    "technique": "rrf",  # Reciprocal Rank Fusion
                    "rank_constant": 60,  # Default k=60 for RRF formula: 1/(k+rank)
                }
            }
        }
    ],
}

# Alternative: Weighted average pipeline (commented out - not used by default)
# This could be used if you need explicit control over BM25 vs vector weights
# However, RRF generally provides better results without manual weight tuning
"""
HYBRID_SEARCH_PIPELINE = {
    "id": "hybrid-ranking-pipeline",
    "description": "Hybrid search pipeline using weighted average for BM25 and vector similarity",
    "phase_results_processors": [
        {
            "normalization-processor": {
                "normalization": {
                    "technique": "l2"  # L2 normalization for better score distribution
                },
                "combination": {
                    "technique": "harmonic_mean",  # Harmonic mean often works better than arithmetic
                    "parameters": {
                        "weights": [0.3, 0.7]  # 30% BM25, 70% vector similarity
                    }
                }
            }
        }
    ]
}
"""
```

### 索引映射逐项解读

- **`number_of_shards: 1`, `number_of_replicas: 0`**：单节点开发配置。生产应增加副本（高可用）并按数据量调分片。
- **`index.knn: True`**：开启 kNN 向量检索能力（Week 4 用）。
- **两个分析器**：
  - `text_analyzer`（自定义）：`lowercase` + `stop`（停用词）+ `snowball`（词干化）。用于 `chunk_text`/`title`/`abstract`——让 "learning" 能匹配 "learn"、"learns"（提升召回）。
  - `standard_analyzer`：标准 + 英文停用词。用于 `authors`。
- **`dynamic: "strict"`**：禁止写入未声明的字段，防止索引被脏数据污染（数据质量/安全）。
- **`embedding` 字段**：`knn_vector`，1024 维（对齐 Jina v3），用 **HNSW** 近似最近邻：
  - `ef_construction: 512`：建图时的候选数，越大召回越好但建索引越慢。
  - `m: 16`：每个节点的双向连接数，影响图质量与内存。
  - `engine: nmslib`、`space_type: cosinesimil`：余弦相似度。
- **`chunk_text.fields.keyword`**：同时保留可精确匹配/聚合的 keyword 子字段。
- **`categories`/`arxiv_id` 用 `keyword`**：精确过滤/聚合用（不分词）。

> **为什么 BM25 索引也用 `snowball` 词干 + 停用词？**
> - **为什么这么选**：学术查询常有词形变化（"optimization" vs "optimize"），词干化提升召回；停用词去掉 "the/of/a" 这类噪声，提升精度。
> - **替代方案**：只用 `standard` 分析器（不词干化）。
> - **优缺点**：词干化 ✅ 召回高。❌ 偶有过度词干（"university"→"univers"）。
> - **影响**：对真实查询的相关性明显更好（效果）。

> **为什么默认 RRF 而不是注释里的加权平均管道？**（呼应第 [03](03-architecture-and-design.md) 章决策 3）RRF 免去 BM25 分数与向量相似度的量纲归一化与权重调参，鲁棒且少维护。加权平均管道作为注释保留，供需要精细控制权重时启用。

---

## 6.2 查询构建器：`src/services/opensearch/query_builder.py`

### 文件：`src/services/opensearch/query_builder.py`（逐字复制）

```python
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryBuilder:
    """
    Unified query builder for OpenSearch supporting both paper-level and chunk-level search.

    Builds complex OpenSearch queries with proper scoring, filtering, and highlighting.
    """

    def __init__(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        track_total_hits: bool = True,
        latest_papers: bool = False,
        search_chunks: bool = False,
    ):
        """Initialize query builder.

        :param query: Search query text
        :param size: Number of results to return
        :param from_: Offset for pagination
        :param fields: Fields to search in (if None, auto-determined based on search_chunks)
        :param categories: Filter by categories
        :param track_total_hits: Whether to track total hits accurately
        :param latest_papers: Sort by publication date instead of relevance
        :param search_chunks: Whether searching chunks (True) or papers (False)
        """
        self.query = query
        self.size = size
        self.from_ = from_
        self.categories = categories
        self.track_total_hits = track_total_hits
        self.latest_papers = latest_papers
        self.search_chunks = search_chunks

        if fields is None:
            if search_chunks:
                self.fields = ["chunk_text^3", "title^2", "abstract^1"]
            else:
                self.fields = ["title^3", "abstract^2", "authors^1"]
        else:
            self.fields = fields

    def build(self) -> Dict[str, Any]:
        """Build the complete OpenSearch query.

        :returns: Complete query dictionary ready for OpenSearch
        """
        query_body = {
            "query": self._build_query(),
            "size": self.size,
            "from": self.from_,
            "track_total_hits": self.track_total_hits,
            "_source": self._build_source_fields(),
            "highlight": self._build_highlight(),
        }

        sort = self._build_sort()
        if sort:
            query_body["sort"] = sort

        return query_body

    def _build_query(self) -> Dict[str, Any]:
        """Build the main query with filters.

        :returns: Query dictionary with bool structure
        """
        must_clauses = []

        if self.query.strip():
            must_clauses.append(self._build_text_query())

        filter_clauses = self._build_filters()

        bool_query = {}

        if must_clauses:
            bool_query["must"] = must_clauses
        else:
            bool_query["must"] = [{"match_all": {}}]

        if filter_clauses:
            bool_query["filter"] = filter_clauses

        return {"bool": bool_query}

    def _build_text_query(self) -> Dict[str, Any]:
        """Build the main text search query.

        :returns: Multi-match query for text search
        """
        return {
            "multi_match": {
                "query": self.query,
                "fields": self.fields,
                "type": "best_fields",
                "operator": "or",
                "fuzziness": "AUTO",
                "prefix_length": 2,
            }
        }

    def _build_filters(self) -> List[Dict[str, Any]]:
        """Build filter clauses for the query.

        :returns: List of filter clauses
        """
        filters = []

        if self.categories:
            filters.append({"terms": {"categories": self.categories}})

        return filters

    def _build_source_fields(self) -> Any:
        """Define which fields to return in results.

        :returns: Source field configuration (list for papers, dict for chunks)
        """
        if self.search_chunks:
            return {"excludes": ["embedding"]}
        else:
            return ["arxiv_id", "title", "authors", "abstract", "categories", "published_date", "pdf_url"]

    def _build_highlight(self) -> Dict[str, Any]:
        """Build highlighting configuration.

        :returns: Highlight configuration dictionary
        """
        if self.search_chunks:
            return {
                "fields": {
                    "chunk_text": {
                        "fragment_size": 150,
                        "number_of_fragments": 2,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                    "title": {"fragment_size": 0, "number_of_fragments": 0, "pre_tags": ["<mark>"], "post_tags": ["</mark>"]},
                    "abstract": {
                        "fragment_size": 150,
                        "number_of_fragments": 1,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                },
                "require_field_match": False,
            }
        else:
            # Paper-specific highlighting
            return {
                "fields": {
                    "title": {
                        "fragment_size": 0,
                        "number_of_fragments": 0,
                    },
                    "abstract": {
                        "fragment_size": 150,
                        "number_of_fragments": 3,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                    "authors": {
                        "fragment_size": 0,
                        "number_of_fragments": 0,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                },
                "require_field_match": False,
            }

    def _build_sort(self) -> Optional[List[Dict[str, Any]]]:
        """Build sorting configuration.

        :returns: Sort configuration or None for relevance scoring
        """
        if self.latest_papers:
            return [{"published_date": {"order": "desc"}}, "_score"]

        if self.query.strip():
            return None

        return [{"published_date": {"order": "desc"}}, "_score"]
```

### QueryBuilder 设计要点

- **字段加权（boosting）**：`chunk_text^3, title^2, abstract^1` 表示匹配在 chunk 正文里权重最高。这是 BM25 相关性调优的核心手段。
- **`multi_match` + `best_fields`**：跨多个字段查询，取最佳字段的分数。
- **`fuzziness: "AUTO"` + `prefix_length: 2`**：容忍拼写错误（"transfomer"→"transformer"），但前 2 个字符必须精确（避免过度模糊）。
- **`operator: "or"`**：查询词之间是 OR（任一匹配即可），提升召回。
- **过滤 vs 查询**：`categories` 走 `filter`（不算分、可缓存），文本走 `must`（算相关性分）。
- **高亮**：返回 `<mark>` 包裹的命中片段，便于前端展示。
- **排序**：默认按相关性（`_score`）；`latest_papers=True` 或空查询时按发布日期倒序。

---

## 6.3 OpenSearch 客户端：`src/services/opensearch/client.py`

下面是**完整客户端**。本周关注 BM25 相关方法（`search_papers`、`_search_bm25_only`、`setup_indices`、`bulk_index_chunks` 等）；向量/混合方法（`search_chunks_vector`、`_search_hybrid_native`）在 Week 4 启用。

### 文件：`src/services/opensearch/client.py`（逐字复制）

```python
"""Unified OpenSearch client supporting both simple BM25 and hybrid search."""

import logging
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch
from src.config import Settings

from .index_config_hybrid import ARXIV_PAPERS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE
from .query_builder import QueryBuilder

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """OpenSearch client supporting BM25 and hybrid search with native RRF."""

    def __init__(self, host: str, settings: Settings):
        self.host = host
        self.settings = settings
        self.index_name = f"{settings.opensearch.index_name}-{settings.opensearch.chunk_index_suffix}"

        self.client = OpenSearch(
            hosts=[host],
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )

        logger.info(f"OpenSearch client initialized with host: {host}")

    def health_check(self) -> bool:
        """Check if OpenSearch cluster is healthy."""
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for the hybrid index."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                return {"index_name": self.index_name, "exists": False, "document_count": 0}

            stats_response = self.client.indices.stats(index=self.index_name)
            index_stats = stats_response["indices"][self.index_name]["total"]

            return {
                "index_name": self.index_name,
                "exists": True,
                "document_count": index_stats["docs"]["count"],
                "deleted_count": index_stats["docs"]["deleted"],
                "size_in_bytes": index_stats["store"]["size_in_bytes"],
            }

        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"index_name": self.index_name, "exists": False, "document_count": 0, "error": str(e)}

    def setup_indices(self, force: bool = False) -> Dict[str, bool]:
        """Setup the hybrid search index and RRF pipeline."""
        results = {}
        results["hybrid_index"] = self._create_hybrid_index(force)
        results["rrf_pipeline"] = self._create_rrf_pipeline(force)
        return results

    def _create_hybrid_index(self, force: bool = False) -> bool:
        """Create hybrid index for all search types (BM25, vector, hybrid).

        :param force: If True, recreate index even if it exists
        :returns: True if created, False if already exists
        """
        try:
            if force and self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing hybrid index: {self.index_name}")

            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(index=self.index_name, body=ARXIV_PAPERS_CHUNKS_MAPPING)
                logger.info(f"Created hybrid index: {self.index_name}")
                return True

            logger.info(f"Hybrid index already exists: {self.index_name}")
            return False

        except Exception as e:
            # Handle race condition when multiple workers start simultaneously:
            # all check exists() -> False, all try to create, only one succeeds.
            if "resource_already_exists_exception" in str(e):
                logger.info(f"Hybrid index already exists (created by another worker): {self.index_name}")
                return False
            logger.error(f"Error creating hybrid index: {e}")
            raise

    def _create_rrf_pipeline(self, force: bool = False) -> bool:
        """Create RRF search pipeline for native hybrid search.

        :param force: If True, recreate pipeline even if it exists
        :returns: True if created, False if already exists
        """
        try:
            pipeline_id = HYBRID_RRF_PIPELINE["id"]

            if force:
                try:
                    self.client.ingest.get_pipeline(id=pipeline_id)
                    self.client.ingest.delete_pipeline(id=pipeline_id)
                    logger.info(f"Deleted existing RRF pipeline: {pipeline_id}")
                except Exception:
                    pass

            try:
                self.client.ingest.get_pipeline(id=pipeline_id)
                logger.info(f"RRF pipeline already exists: {pipeline_id}")
                return False
            except Exception:
                pass
            pipeline_body = {
                "description": HYBRID_RRF_PIPELINE["description"],
                "phase_results_processors": HYBRID_RRF_PIPELINE["phase_results_processors"],
            }

            self.client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_id}", body=pipeline_body)

            logger.info(f"Created RRF search pipeline: {pipeline_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating RRF pipeline: {e}")
            raise

    def search_papers(
        self, query: str, size: int = 10, from_: int = 0, categories: Optional[List[str]] = None, latest: bool = True
    ) -> Dict[str, Any]:
        """BM25 search for papers."""
        return self._search_bm25_only(query=query, size=size, from_=from_, categories=categories, latest=latest)

    def search_chunks_vector(
        self, query_embedding: List[float], size: int = 10, categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Pure vector search on chunks.

        :param query_embedding: Query embedding vector
        :param size: Number of results
        :param categories: Optional category filter
        :returns: Search results
        """
        try:
            # Build filter
            filter_clause = []
            if categories:
                filter_clause.append({"terms": {"categories": categories}})

            search_body = {
                "size": size,
                "query": {"knn": {"embedding": {"vector": query_embedding, "k": size}}},
                "_source": {"excludes": ["embedding"]},
            }

            if filter_clause:
                search_body["query"] = {"bool": {"must": [search_body["query"]], "filter": filter_clause}}

            response = self.client.search(index=self.index_name, body=search_body)

            results = {"total": response["hits"]["total"]["value"], "hits": []}

            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                chunk["score"] = hit["_score"]
                chunk["chunk_id"] = hit["_id"]
                results["hits"].append(chunk)

            return results

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return {"total": 0, "hits": []}

    def search_unified(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        size: int = 10,
        from_: int = 0,
        categories: Optional[List[str]] = None,
        latest: bool = False,
        use_hybrid: bool = True,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Unified search method supporting BM25, vector, and hybrid modes.

        :param query: Text query for search
        :param query_embedding: Optional embedding for vector/hybrid search
        :param size: Number of results to return
        :param from_: Offset for pagination
        :param categories: Optional category filter
        :param latest: Sort by date instead of relevance
        :param use_hybrid: If True and embedding provided, use hybrid search
        :param min_score: Minimum score threshold
        :returns: Search results
        """
        try:
            # If no embedding provided or hybrid disabled, use BM25 only
            if not query_embedding or not use_hybrid:
                return self._search_bm25_only(query=query, size=size, from_=from_, categories=categories, latest=latest)

            # Use native OpenSearch hybrid search with RRF pipeline
            return self._search_hybrid_native(
                query=query, query_embedding=query_embedding, size=size, categories=categories, min_score=min_score
            )

        except Exception as e:
            logger.error(f"Unified search error: {e}")
            return {"total": 0, "hits": []}

    def _search_bm25_only(
        self, query: str, size: int, from_: int, categories: Optional[List[str]], latest: bool
    ) -> Dict[str, Any]:
        """Pure BM25 search implementation."""
        builder = QueryBuilder(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            latest_papers=latest,
            search_chunks=True,  # Enable chunk search mode
        )
        search_body = builder.build()

        response = self.client.search(index=self.index_name, body=search_body)

        results = {"total": response["hits"]["total"]["value"], "hits": []}

        for hit in response["hits"]["hits"]:
            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]

            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]

            results["hits"].append(chunk)

        logger.info(f"BM25 search for '{query[:50]}...' returned {results['total']} results")
        return results

    def _search_hybrid_native(
        self, query: str, query_embedding: List[float], size: int, categories: Optional[List[str]], min_score: float
    ) -> Dict[str, Any]:
        """Native OpenSearch hybrid search with RRF pipeline."""
        builder = QueryBuilder(
            query=query, size=size * 2, from_=0, categories=categories, latest_papers=False, search_chunks=True
        )
        bm25_search_body = builder.build()

        bm25_query = bm25_search_body["query"]

        hybrid_query = {"hybrid": {"queries": [bm25_query, {"knn": {"embedding": {"vector": query_embedding, "k": size * 2}}}]}}

        search_body = {
            "size": size,
            "query": hybrid_query,
            "_source": bm25_search_body["_source"],
            "highlight": bm25_search_body["highlight"],
        }

        # Execute search with RRF pipeline
        response = self.client.search(
            index=self.index_name, body=search_body, params={"search_pipeline": HYBRID_RRF_PIPELINE["id"]}
        )

        results = {"total": response["hits"]["total"]["value"], "hits": []}

        for hit in response["hits"]["hits"]:
            if hit["_score"] < min_score:
                continue

            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]

            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]

            results["hits"].append(chunk)

        results["total"] = len(results["hits"])
        logger.info(f"Native hybrid search for '{query[:50]}...' returned {results['total']} results")
        return results

    def search_chunks_hybrid(
        self,
        query: str,
        query_embedding: List[float],
        size: int = 10,
        categories: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Hybrid search combining BM25 and vector similarity using native RRF."""
        return self._search_hybrid_native(
            query=query, query_embedding=query_embedding, size=size, categories=categories, min_score=min_score
        )

    def index_chunk(self, chunk_data: Dict[str, Any], embedding: List[float]) -> bool:
        """Index a single chunk with its embedding.

        :param chunk_data: Chunk data dictionary
        :param embedding: Embedding vector
        :returns: True if successful
        """
        try:
            chunk_data["embedding"] = embedding

            response = self.client.index(index=self.index_name, body=chunk_data, refresh=True)

            return response["result"] in ["created", "updated"]

        except Exception as e:
            logger.error(f"Error indexing chunk: {e}")
            return False

    def bulk_index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Bulk index multiple chunks with embeddings.

        :param chunks: List of dicts with 'chunk_data' and 'embedding'
        :returns: Statistics
        """
        from opensearchpy import helpers

        try:
            actions = []
            for chunk in chunks:
                chunk_data = chunk["chunk_data"].copy()
                chunk_data["embedding"] = chunk["embedding"]

                action = {"_index": self.index_name, "_source": chunk_data}
                actions.append(action)

            success, failed = helpers.bulk(self.client, actions, refresh=True)

            logger.info(f"Bulk indexed {success} chunks, {len(failed)} failed")
            return {"success": success, "failed": len(failed)}

        except Exception as e:
            logger.error(f"Bulk chunk indexing error: {e}")
            raise

    def delete_paper_chunks(self, arxiv_id: str) -> bool:
        """Delete all chunks for a specific paper.

        :param arxiv_id: ArXiv ID of the paper
        :returns: True if deletion was successful
        """
        try:
            response = self.client.delete_by_query(
                index=self.index_name, body={"query": {"term": {"arxiv_id": arxiv_id}}}, refresh=True
            )

            deleted = response.get("deleted", 0)
            logger.info(f"Deleted {deleted} chunks for paper {arxiv_id}")
            return deleted > 0

        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False

    def get_chunks_by_paper(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific paper.

        :param arxiv_id: ArXiv ID of the paper
        :returns: List of chunks sorted by chunk_index
        """
        try:
            search_body = {
                "query": {"term": {"arxiv_id": arxiv_id}},
                "size": 1000,
                "sort": [{"chunk_index": "asc"}],
                "_source": {"excludes": ["embedding"]},
            }

            response = self.client.search(index=self.index_name, body=search_body)

            chunks = []
            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                chunk["chunk_id"] = hit["_id"]
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return []
```

### 客户端关键点

- **索引名**：`{index_name}-{chunk_index_suffix}` = `arxiv-papers-chunks`（来自配置）。
- **`setup_indices`**：建索引 + 建 RRF 管道，幂等。处理了多 worker 同时启动的竞态（`resource_already_exists_exception`）。
- **`_search_bm25_only`**：用 `QueryBuilder` 构造 `multi_match` 查询，返回带高亮的 chunk。
- **`bulk_index_chunks`**：用 `opensearchpy.helpers.bulk` 批量写入（Week 4 索引会用）。`refresh=True` 让写入立即可搜（教学方便；生产高吞吐场景应权衡 refresh 频率）。
- **`_search_hybrid_native`**（Week 4）：把 BM25 query 与 kNN query 组成 `hybrid` 查询，走 RRF 管道融合。

### 工厂：`src/services/opensearch/factory.py`（逐字复制）

```python
"""Unified factory for OpenSearch client."""

from functools import lru_cache
from typing import Optional

from src.config import Settings, get_settings

from .client import OpenSearchClient


@lru_cache(maxsize=1)
def make_opensearch_client(settings: Optional[Settings] = None) -> OpenSearchClient:
    """Factory function to create cached OpenSearch client.

    Uses lru_cache to maintain a singleton instance for efficiency.

    :param settings: Optional settings instance
    :returns: Cached OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()

    return OpenSearchClient(host=settings.opensearch.host, settings=settings)


def make_opensearch_client_fresh(settings: Optional[Settings] = None, host: Optional[str] = None) -> OpenSearchClient:
    """Factory function to create a fresh OpenSearch client (not cached).

    Use this when you need a new client instance (e.g., for testing
    or when connection issues occur).

    :param settings: Optional settings instance
    :param host: Optional host override
    :returns: New OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()

    # Use provided host or settings host
    opensearch_host = host or settings.opensearch.host

    return OpenSearchClient(host=opensearch_host, settings=settings)
```

> 提供两个工厂：`make_opensearch_client`（缓存单例，API 用）和 `make_opensearch_client_fresh`（每次新建，Airflow 任务/测试用，避免缓存的客户端在跨进程/跨事件循环时出问题）。

---

## 6.4 检索 API schema：`src/schemas/api/search.py`

### 文件：`src/schemas/api/search.py`（逐字复制）

```python
from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query across title, abstract, and authors")
    size: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    from_: int = Field(default=0, ge=0, alias="from", description="Offset for pagination")
    categories: Optional[List[str]] = Field(default=None, description="Filter by categories")
    latest_papers: bool = Field(default=False, description="Sort by publication date (newest first) instead of relevance")


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search supporting all search modes."""

    query: str = Field(..., description="Search query text", min_length=1, max_length=500)
    size: int = Field(10, description="Number of results to return", ge=1, le=100)
    from_: int = Field(0, description="Offset for pagination", ge=0, alias="from")
    categories: Optional[List[str]] = Field(None, description="Filter by arXiv categories (e.g., ['cs.AI', 'cs.LG'])")
    latest_papers: bool = Field(False, description="Sort by publication date instead of relevance")
    use_hybrid: bool = Field(True, description="Enable hybrid search (BM25 + vector) with automatic embedding generation")
    min_score: float = Field(0.0, description="Minimum score threshold for results", ge=0.0)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "query": "machine learning neural networks",
                "size": 10,
                "categories": ["cs.AI", "cs.LG"],
                "latest_papers": False,
                "use_hybrid": True,
            }
        }


class SearchHit(BaseModel):
    """Individual search result."""

    arxiv_id: str
    title: str
    authors: Optional[str]
    abstract: Optional[str]
    published_date: Optional[str]
    pdf_url: Optional[str]
    score: float
    highlights: Optional[dict] = None

    # Chunk-specific fields (for unified search)
    chunk_text: Optional[str] = Field(None, description="Text content of the matching chunk")
    chunk_id: Optional[str] = Field(None, description="Unique identifier of the chunk")
    section_name: Optional[str] = Field(None, description="Section name where the chunk was found")


class SearchResponse(BaseModel):
    """Search response model."""

    query: str
    total: int
    hits: List[SearchHit]
    size: int = Field(description="Number of results requested")
    from_: int = Field(alias="from", description="Offset used for pagination")
    search_mode: Optional[str] = Field(None, description="Search mode used: bm25, vector, or hybrid")
    error: Optional[str] = None

    class Config:
        populate_by_name = True
```

> `from_` 字段用 `alias="from"`（因为 `from` 是 Python 关键字，不能直接做字段名）。`populate_by_name = True` 让代码里能用 `from_`、JSON 里能用 `from`。

> **关于 HTTP 端点**：真实仓库把 BM25 与混合检索**统一到一个端点** `/api/v1/hybrid-search/`（用 `use_hybrid=false` 即为纯 BM25）。该端点依赖 Week 4 的嵌入服务，因此其完整代码放在第 [07](07-week4-hybrid-search.md) 章给出。本周我们在**客户端层**直接验证 BM25（见 6.6）。

---

## 6.5 接入 FastAPI（main.py / dependencies.py 增量）

> 以下是**对前面版本的增量**。最终完整版见第 [10](10-week7-agentic-telegram.md) 章。

在 `src/dependencies.py` 加入：

```python
# import 区：
from src.services.opensearch.client import OpenSearchClient

# 函数：
def get_opensearch_client(request: Request) -> OpenSearchClient:
    """Get OpenSearch client from the request state."""
    return request.app.state.opensearch_client

# 类型别名：
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
```

在 `src/main.py` 的 `lifespan` 里加入（构建客户端 + 启动时建索引）：

```python
    from src.services.opensearch.factory import make_opensearch_client

    opensearch_client = make_opensearch_client()
    app.state.opensearch_client = opensearch_client

    if opensearch_client.health_check():
        logger.info("OpenSearch connected successfully")
        setup_results = opensearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("Hybrid index created")
        else:
            logger.info("Hybrid index already exists")
    else:
        logger.warning("OpenSearch connection failed - search features will be limited")
```

> 健康检查端点 `/health` 也会扩展为包含 OpenSearch（以及 Week 5 的 Ollama）。其**最终完整版在第 [08](08-week5-rag-llm.md) 章逐字给出**——现在先保留 Week 1 引导版即可，OpenSearch 已经在 lifespan 里初始化。

---

## 6.6 本周验证：直接验证 BM25

启动 OpenSearch 后，用下面的脚本验证"建索引 → 写入 → BM25 检索"全链路（**不需要 Jina**，用全 0 占位向量满足索引的向量字段）。

新建临时脚本 `verify_week3.py`：

```python
from src.config import get_settings
from src.services.opensearch.factory import make_opensearch_client_fresh


def main() -> None:
    settings = get_settings()
    client = make_opensearch_client_fresh(settings)

    assert client.health_check(), "OpenSearch is not healthy"

    # 1) 建索引 + RRF 管道（幂等）
    print("setup:", client.setup_indices(force=True))

    # 2) 写入几条 chunk（embedding 用 1024 维全 0 占位，仅为满足映射）
    zero_vec = [0.0] * settings.opensearch.vector_dimension
    samples = [
        {
            "arxiv_id": "0001.0001",
            "paper_id": "p1",
            "chunk_index": 0,
            "chunk_text": "Transformers are a neural network architecture based on self-attention mechanisms.",
            "title": "Attention Is All You Need",
            "authors": "Vaswani et al.",
            "abstract": "We propose the Transformer, based solely on attention.",
            "categories": ["cs.AI", "cs.LG"],
        },
        {
            "arxiv_id": "0002.0002",
            "paper_id": "p2",
            "chunk_index": 0,
            "chunk_text": "Convolutional neural networks excel at image classification tasks.",
            "title": "Deep Residual Learning",
            "authors": "He et al.",
            "abstract": "We present a residual learning framework.",
            "categories": ["cs.CV"],
        },
    ]
    for s in samples:
        ok = client.index_chunk(s, zero_vec)
        print("indexed", s["arxiv_id"], ok)

    # 3) BM25 检索
    results = client.search_papers(query="transformer attention", size=5, latest=False)
    print(f"\nBM25 results: total={results['total']}")
    for hit in results["hits"]:
        print(f"  score={hit['score']:.3f}  {hit['arxiv_id']}  {hit.get('chunk_text','')[:60]}")


if __name__ == "__main__":
    main()
```

运行：

```bash
docker compose up -d opensearch
# 等 opensearch 健康（约 30–60s）
uv run python verify_week3.py
rm verify_week3.py
```

期望：第一条（含 "transformer"/"attention"）排在前面，分数更高；第二条（CNN）分数低或不出现。你也可以打开 **http://localhost:5601**（OpenSearch Dashboards）在 Dev Tools 里直接发查询。

> ⚠️ 上面用 `index_chunk` 手动塞了占位向量只是为了**本周演示 BM25**。真正的内容索引（真实分块 + 真实 Jina 向量）在 Week 4 由 `HybridIndexingService` 完成。

---

## 6.7 本章小结

你已经有了：

- ✅ 一个混合索引映射（BM25 文本字段 + HNSW 向量字段，一索引多用）。
- ✅ 强大的 `QueryBuilder`（字段加权、模糊、过滤、高亮、排序）。
- ✅ 统一 `OpenSearchClient`（BM25 已通，向量/混合待 Week 4 启用）。
- ✅ RRF 管道（已创建，Week 4 启用）。
- ✅ 检索 API schema。

**Week 3 里程碑**：BM25 关键词检索可用。下一章 [`07-week4-hybrid-search.md`](07-week4-hybrid-search.md) 加上"语义层"——智能分块、Jina 向量嵌入、RRF 混合检索，并把统一检索端点接上 HTTP。
