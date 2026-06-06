# File: tutorials/05-week2-ingestion.md

# 第 5 章　Week 2：数据摄取管道（arXiv → 解析 → 入库 → Airflow）

**本周目标**：让系统"活起来"——自动从 arXiv 抓取论文元数据、下载并用 Docling 解析 PDF、把结果存进 PostgreSQL，最后用 Apache Airflow 编排成可定时运行的每日管道。

> **跨周依赖提示**：本周的 `metadata_fetcher.py` 顶部 `import` 了 `src.services.opensearch.client`（一个 Week 3 模块），Airflow 的 `setup.py` / `indexing.py` 还会用到 Week 3（OpenSearch）和 Week 4（混合索引）的模块。因此：
> - **Week 2 可独立验证的部分**：`ArxivClient` 抓取、`DoclingParser` 解析、`PaperRepository` 入库（用 [5.10](#510-本周验证) 的脚本，它**不**导入 `metadata_fetcher`，避免触发尚不存在的 opensearch 模块）。
> - **完整的 `MetadataFetcher` 与 Airflow DAG 端到端运行**：在你完成 Week 3–4 的 opensearch / indexing 模块后即可。本章把全部 Airflow 文件一次给齐，便于集中管理。

---

## 5.1 摄取流程总览

```
ArxivClient.fetch_papers()      → 调 arXiv API(XML) → 解析为 ArxivPaper 列表（限速 3s）
        ↓
ArxivClient.download_pdf()      → 流式下载 PDF 到本地缓存（带重试）
        ↓
PDFParserService.parse_pdf()    → DoclingParser 解析 → PdfContent(章节 + 全文)
        ↓
MetadataFetcher                 → 组装 ArxivPaper + PdfContent → PaperCreate
        ↓
PaperRepository.upsert()        → 写入/更新 PostgreSQL（按 arxiv_id 去重）
```

`MetadataFetcher` 是**主编排器**：它用异步信号量控制"下载"和"解析"的并发，做到边下边解的流水线。

---

## 5.2 arXiv 数据 schema：`src/schemas/arxiv/paper.py`

```bash
touch src/schemas/arxiv/__init__.py src/schemas/pdf_parser/__init__.py
```

### 文件：`src/schemas/arxiv/paper.py`（逐字复制）

```python
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ArxivPaper(BaseModel):
    """Schema for arXiv API response data."""

    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of author names")
    abstract: str = Field(..., description="Paper abstract")
    categories: List[str] = Field(..., description="Paper categories")
    published_date: str = Field(..., description="Date published on arXiv (ISO format)")
    pdf_url: str = Field(..., description="URL to PDF")


class PaperBase(BaseModel):
    # Core arXiv metadata
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of author names")
    abstract: str = Field(..., description="Paper abstract")
    categories: List[str] = Field(..., description="Paper categories")
    published_date: datetime = Field(..., description="Date published on arXiv")
    pdf_url: str = Field(..., description="URL to PDF")


class PaperCreate(PaperBase):
    """Schema for creating a paper with optional parsed content."""

    # Parsed PDF content (optional - added when PDF is processed)
    raw_text: Optional[str] = Field(None, description="Full raw text extracted from PDF")
    sections: Optional[List[Dict[str, Any]]] = Field(None, description="List of sections with titles and content")
    references: Optional[List[Dict[str, Any]]] = Field(None, description="List of references if extracted")

    # PDF processing metadata (optional)
    parser_used: Optional[str] = Field(None, description="Which parser was used (DOCLING, etc.)")
    parser_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional parser metadata")
    pdf_processed: Optional[bool] = Field(False, description="Whether PDF was successfully processed")
    pdf_processing_date: Optional[datetime] = Field(None, description="When PDF was processed")


class PaperResponse(PaperBase):
    """Schema for paper API responses with all content."""

    id: UUID

    # Parsed PDF content (optional fields)
    raw_text: Optional[str] = Field(None, description="Full raw text extracted from PDF")
    sections: Optional[List[Dict[str, Any]]] = Field(None, description="List of sections with titles and content")
    references: Optional[List[Dict[str, Any]]] = Field(None, description="List of references if extracted")

    # PDF processing metadata
    parser_used: Optional[str] = Field(None, description="Which parser was used")
    parser_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional parser metadata")
    pdf_processed: bool = Field(False, description="Whether PDF was successfully processed")
    pdf_processing_date: Optional[datetime] = Field(None, description="When PDF was processed")

    # Timestamps
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
```

> **注意 `ArxivPaper.published_date` 是 `str`，而 `PaperBase.published_date` 是 `datetime`**。这是有意的：arXiv API 返回的是 ISO 字符串，入库前由 `MetadataFetcher` 用 `dateutil` 解析为 `datetime`。`PaperResponse.Config.from_attributes = True` 让它能直接从 ORM 对象构造。

补一个尾部小 schema（上游同文件里有，便于 API 返回列表）：

```python
class PaperSearchResponse(BaseModel):
    papers: List[PaperResponse]
    total: int
```

> 把上面这段也加到 `src/schemas/arxiv/paper.py` 末尾（它与上面同属一个文件）。

---

## 5.3 PDF 内容 schema：`src/schemas/pdf_parser/models.py`

### 文件：`src/schemas/pdf_parser/models.py`（逐字复制）

```python
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ParserType(str, Enum):
    """PDF parser types."""

    DOCLING = "docling"


class PaperSection(BaseModel):
    """Represents a section of a paper."""

    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    level: int = Field(default=1, description="Section hierarchy level")


class PaperFigure(BaseModel):
    """Represents a figure in a paper."""

    caption: str = Field(..., description="Figure caption")
    id: str = Field(..., description="Figure identifier")


class PaperTable(BaseModel):
    """Represents a table in a paper."""

    caption: str = Field(..., description="Table caption")
    id: str = Field(..., description="Table identifier")


class PdfContent(BaseModel):
    """PDF-specific content extracted by parsers like Docling."""

    sections: List[PaperSection] = Field(default_factory=list, description="Paper sections")
    figures: List[PaperFigure] = Field(default_factory=list, description="Figures")
    tables: List[PaperTable] = Field(default_factory=list, description="Tables")
    raw_text: str = Field(..., description="Full extracted text")
    references: List[str] = Field(default_factory=list, description="References")
    parser_used: ParserType = Field(..., description="Parser used for extraction")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Parser metadata")


class ArxivMetadata(BaseModel):
    """Paper metadata from arXiv API."""

    title: str = Field(..., description="Paper title from arXiv")
    authors: List[str] = Field(..., description="Authors from arXiv")
    abstract: str = Field(..., description="Abstract from arXiv")
    arxiv_id: str = Field(..., description="arXiv identifier")
    categories: List[str] = Field(default_factory=list, description="arXiv categories")
    published_date: str = Field(..., description="Publication date")
    pdf_url: str = Field(..., description="PDF download URL")


class ParsedPaper(BaseModel):
    """Complete paper data combining arXiv metadata and PDF content."""

    arxiv_metadata: ArxivMetadata = Field(..., description="Metadata from arXiv API")
    pdf_content: Optional[PdfContent] = Field(None, description="Content extracted from PDF")
```

---

## 5.4 arXiv 客户端：`src/services/arxiv/client.py`

```bash
mkdir -p src/services/arxiv
touch src/services/arxiv/__init__.py
```

### 文件：`src/services/arxiv/client.py`（逐字复制）

```python
import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote, urlencode

import httpx
from src.config import ArxivSettings
from src.exceptions import ArxivAPIException, ArxivAPITimeoutError, ArxivParseError, PDFDownloadException, PDFDownloadTimeoutError
from src.schemas.arxiv.paper import ArxivPaper

logger = logging.getLogger(__name__)


class ArxivClient:
    """Client for fetching papers from arXiv API."""

    def __init__(self, settings: ArxivSettings):
        self._settings = settings
        self._last_request_time: Optional[float] = None

    @cached_property
    def pdf_cache_dir(self) -> Path:
        """PDF cache directory."""
        cache_dir = Path(self._settings.pdf_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    @property
    def namespaces(self) -> dict:
        return self._settings.namespaces

    @property
    def rate_limit_delay(self) -> float:
        return self._settings.rate_limit_delay

    @property
    def timeout_seconds(self) -> int:
        return self._settings.timeout_seconds

    @property
    def max_results(self) -> int:
        return self._settings.max_results

    @property
    def search_category(self) -> str:
        return self._settings.search_category

    async def fetch_papers(
        self,
        max_results: Optional[int] = None,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[ArxivPaper]:
        """
        Fetch papers from arXiv for the configured category.

        Args:
            max_results: Maximum number of papers to fetch (uses settings default if None)
            start: Starting index for pagination
            sort_by: Sort criteria (submittedDate, lastUpdatedDate, relevance)
            sort_order: Sort order (ascending, descending)
            from_date: Filter papers submitted after this date (format: YYYYMMDD)
            to_date: Filter papers submitted before this date (format: YYYYMMDD)

        Returns:
            List of ArxivPaper objects for the configured category
        """
        if max_results is None:
            max_results = self.max_results

        # Build search query
        search_query = f"cat:{self.search_category}"

        # Add date filtering if provided
        if from_date or to_date:
            # Convert dates to arXiv format (YYYYMMDDHHMM) - use 0000 for start of day, 2359 for end
            date_from = f"{from_date}0000" if from_date else "*"
            date_to = f"{to_date}2359" if to_date else "*"
            # Use correct arXiv API syntax with + symbols
            search_query += f" AND submittedDate:[{date_from}+TO+{date_to}]"

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": min(max_results, 2000),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        safe = ":+[]"  # Don't encode :, +, [, ] characters needed for arXiv queries
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        try:
            logger.info(f"Fetching {max_results} {self.search_category} papers from arXiv")

            # Add rate limiting delay between all requests (arXiv recommends 3 seconds)
            if self._last_request_time is not None:
                time_since_last = time.time() - self._last_request_time
                if time_since_last < self.rate_limit_delay:
                    sleep_time = self.rate_limit_delay - time_since_last
                    await asyncio.sleep(sleep_time)

            self._last_request_time = time.time()

            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url)
                response.raise_for_status()
                xml_data = response.text

            papers = self._parse_response(xml_data)
            logger.info(f"Fetched {len(papers)} papers")

            return papers

        except httpx.TimeoutException as e:
            logger.error(f"arXiv API timeout: {e}")
            raise ArxivAPITimeoutError(f"arXiv API request timed out: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"arXiv API HTTP error: {e}")
            raise ArxivAPIException(f"arXiv API returned error {e.response.status_code}: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch papers from arXiv: {e}")
            raise ArxivAPIException(f"Unexpected error fetching papers from arXiv: {e}")

    async def fetch_papers_with_query(
        self,
        search_query: str,
        max_results: Optional[int] = None,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> List[ArxivPaper]:
        """
        Fetch papers from arXiv using a custom search query.

        Args:
            search_query: Custom arXiv search query (e.g., "cat:cs.AI AND submittedDate:[20240101 TO 20241231]")
            max_results: Maximum number of papers to fetch (uses settings default if None)
            start: Starting index for pagination
            sort_by: Sort criteria (submittedDate, lastUpdatedDate, relevance)
            sort_order: Sort order (ascending, descending)

        Returns:
            List of ArxivPaper objects matching the search query

        Examples:
            # Papers from last 30 days
            "cat:cs.AI AND submittedDate:[20240101 TO *]"

            # Papers by specific author
            "au:LeCun AND cat:cs.AI"

            # Papers with specific keywords in title
            "ti:transformer AND cat:cs.AI"
        """
        if max_results is None:
            max_results = self.max_results

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": min(max_results, 2000),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        safe = ":+[]*"  # Don't encode :, +, [, ], *, characters needed for arXiv queries
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        try:
            # Add rate limiting delay between all requests (arXiv recommends 3 seconds)
            if self._last_request_time is not None:
                time_since_last = time.time() - self._last_request_time
                if time_since_last < self.rate_limit_delay:
                    sleep_time = self.rate_limit_delay - time_since_last
                    await asyncio.sleep(sleep_time)

            self._last_request_time = time.time()

            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url)
                response.raise_for_status()
                xml_data = response.text

            papers = self._parse_response(xml_data)
            logger.info(f"Query returned {len(papers)} papers")

            return papers

        except httpx.TimeoutException as e:
            logger.error(f"arXiv API timeout: {e}")
            raise ArxivAPITimeoutError(f"arXiv API request timed out: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"arXiv API HTTP error: {e}")
            raise ArxivAPIException(f"arXiv API returned error {e.response.status_code}: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch papers from arXiv: {e}")
            raise ArxivAPIException(f"Unexpected error fetching papers from arXiv: {e}")

    async def fetch_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Fetch a specific paper by its arXiv ID.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2507.17748v1" or "2507.17748")

        Returns:
            ArxivPaper object or None if not found
        """
        # Clean the arXiv ID (remove version if needed for search)
        clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
        params = {"id_list": clean_id, "max_results": 1}

        safe = ":+[]*"  # Don't encode :, +, [, ], *, characters needed for arXiv queries
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                xml_data = response.text

            papers = self._parse_response(xml_data)

            if papers:
                return papers[0]
            else:
                logger.warning(f"Paper {arxiv_id} not found")
                return None

        except httpx.TimeoutException as e:
            logger.error(f"arXiv API timeout for paper {arxiv_id}: {e}")
            raise ArxivAPITimeoutError(f"arXiv API request timed out for paper {arxiv_id}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"arXiv API HTTP error for paper {arxiv_id}: {e}")
            raise ArxivAPIException(f"arXiv API returned error {e.response.status_code} for paper {arxiv_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch paper {arxiv_id} from arXiv: {e}")
            raise ArxivAPIException(f"Unexpected error fetching paper {arxiv_id} from arXiv: {e}")

    def _parse_response(self, xml_data: str) -> List[ArxivPaper]:
        """
        Parse arXiv API XML response into ArxivPaper objects.

        Args:
            xml_data: Raw XML response from arXiv API

        Returns:
            List of parsed ArxivPaper objects
        """
        try:
            root = ET.fromstring(xml_data)
            entries = root.findall("atom:entry", self.namespaces)

            papers = []
            for entry in entries:
                paper = self._parse_single_entry(entry)
                if paper:
                    papers.append(paper)

            return papers

        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML response: {e}")
            raise ArxivParseError(f"Failed to parse arXiv XML response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing arXiv response: {e}")
            raise ArxivParseError(f"Unexpected error parsing arXiv response: {e}")

    def _parse_single_entry(self, entry: ET.Element) -> Optional[ArxivPaper]:
        """
        Parse a single entry from arXiv XML response.

        Args:
            entry: XML entry element

        Returns:
            ArxivPaper object or None if parsing fails
        """
        try:
            # Extract basic metadata
            arxiv_id = self._get_arxiv_id(entry)
            if not arxiv_id:
                return None

            title = self._get_text(entry, "atom:title", clean_newlines=True)
            authors = self._get_authors(entry)
            abstract = self._get_text(entry, "atom:summary", clean_newlines=True)
            published = self._get_text(entry, "atom:published")
            categories = self._get_categories(entry)
            pdf_url = self._get_pdf_url(entry)

            return ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published_date=published,
                categories=categories,
                pdf_url=pdf_url,
            )

        except Exception as e:
            logger.error(f"Failed to parse entry: {e}")
            return None

    def _get_text(self, element: ET.Element, path: str, clean_newlines: bool = False) -> str:
        """
        Extract text from XML element safely.

        Args:
            element: Parent XML element
            path: XPath to find the text element
            clean_newlines: Whether to replace newlines with spaces

        Returns:
            Extracted text or empty string
        """
        elem = element.find(path, self.namespaces)
        if elem is None or elem.text is None:
            return ""

        text = elem.text.strip()
        return text.replace("\n", " ") if clean_newlines else text

    def _get_arxiv_id(self, entry: ET.Element) -> Optional[str]:
        """
        Extract arXiv ID from entry.

        Args:
            entry: XML entry element

        Returns:
            arXiv ID or None
        """
        id_elem = entry.find("atom:id", self.namespaces)
        if id_elem is None or id_elem.text is None:
            return None
        return id_elem.text.split("/")[-1]

    def _get_authors(self, entry: ET.Element) -> List[str]:
        """
        Extract author names from entry.

        Args:
            entry: XML entry element

        Returns:
            List of author names
        """
        authors = []
        for author in entry.findall("atom:author", self.namespaces):
            name = self._get_text(author, "atom:name")
            if name:
                authors.append(name)
        return authors

    def _get_categories(self, entry: ET.Element) -> List[str]:
        """
        Extract categories from entry.

        Args:
            entry: XML entry element

        Returns:
            List of category terms
        """
        categories = []
        for category in entry.findall("atom:category", self.namespaces):
            term = category.get("term")
            if term:
                categories.append(term)
        return categories

    def _get_pdf_url(self, entry: ET.Element) -> str:
        """
        Extract PDF URL from entry links.

        Args:
            entry: XML entry element

        Returns:
            PDF URL or empty string (always HTTPS)
        """
        for link in entry.findall("atom:link", self.namespaces):
            if link.get("type") == "application/pdf":
                url = link.get("href", "")
                # Convert HTTP to HTTPS for arXiv URLs
                if url.startswith("http://arxiv.org/"):
                    url = url.replace("http://arxiv.org/", "https://arxiv.org/")
                return url
        return ""

    async def download_pdf(self, paper: ArxivPaper, force_download: bool = False) -> Optional[Path]:
        """
        Download PDF for a given paper to local cache.

        Args:
            paper: ArxivPaper object containing PDF URL
            force_download: Force re-download even if file exists

        Returns:
            Path to downloaded PDF file or None if download failed
        """
        if not paper.pdf_url:
            logger.error(f"No PDF URL for paper {paper.arxiv_id}")
            return None

        pdf_path = self._get_pdf_path(paper.arxiv_id)

        # Return cached PDF if exists
        if pdf_path.exists() and not force_download:
            logger.info(f"Using cached PDF: {pdf_path.name}")
            return pdf_path

        # Download with retry
        if await self._download_with_retry(paper.pdf_url, pdf_path):
            return pdf_path
        else:
            return None

    def _get_pdf_path(self, arxiv_id: str) -> Path:
        """
        Get the local path for a PDF file.

        Args:
            arxiv_id: arXiv paper ID

        Returns:
            Path object for the PDF file
        """
        safe_filename = arxiv_id.replace("/", "_") + ".pdf"
        return self.pdf_cache_dir / safe_filename

    async def _download_with_retry(self, url: str, path: Path, max_retries: Optional[int] = None) -> bool:
        """Download a file with retry logic."""
        if max_retries is None:
            max_retries = self._settings.download_max_retries

        logger.info(f"Downloading PDF from {url}")

        # Respect rate limits
        await asyncio.sleep(self.rate_limit_delay)

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=float(self.timeout_seconds)) as client:
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()
                        with open(path, "wb") as f:
                            async for chunk in response.aiter_bytes():
                                f.write(chunk)
                logger.info(f"Successfully downloaded to {path.name}")
                return True

            except httpx.TimeoutException as e:
                if attempt < max_retries - 1:
                    wait_time = self._settings.download_retry_delay_base * (attempt + 1)
                    logger.warning(f"PDF download timeout (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"PDF download failed after {max_retries} attempts due to timeout: {e}")
                    raise PDFDownloadTimeoutError(f"PDF download timed out after {max_retries} attempts: {e}")
            except httpx.HTTPError as e:
                if attempt < max_retries - 1:
                    wait_time = self._settings.download_retry_delay_base * (attempt + 1)  # Exponential backoff
                    logger.warning(f"Download failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise PDFDownloadException(f"PDF download failed after {max_retries} attempts: {e}")
            except Exception as e:
                logger.error(f"Unexpected download error: {e}")
                raise PDFDownloadException(f"Unexpected error during PDF download: {e}")

        # Clean up partial download
        if path.exists():
            path.unlink()

        return False
```

### ArxivClient 设计要点

- **限速 3 秒**：arXiv 官方要求请求间隔 ≥3s。客户端用 `_last_request_time` 记录上次请求时间，不足 3s 就 `await asyncio.sleep` 补齐。**这是对外部服务负责，也避免被封禁**（安全/稳定）。
- **`safe = ":+[]"`**：arXiv 查询语法里 `:`、`+`、`[`、`]` 不能被 URL 编码，否则查询失效。
- **流式下载 + 重试 + 指数退避**：大 PDF 不会一次性读进内存；失败按 `download_retry_delay_base * (attempt+1)` 退避重试；最终失败清理半成品文件。
- **HTTP→HTTPS 修正**：arXiv 有时给 `http://` 链接，统一改成 `https://`（安全）。
- **缓存命中**：PDF 已下载则直接复用，省带宽和时间。

> **为什么用 `httpx` 异步而不是 `requests`？**
> - **为什么这么选**：摄取要并发下载多个 PDF，异步 I/O 吞吐高；`httpx` 同时支持同步/异步、流式下载。
> - **替代方案**：`requests`（同步）+ 线程池。
> - **影响**：单进程内高并发下载（性能）；与 FastAPI 的 async 生态一致（可维护）。

### 文件：`src/services/arxiv/factory.py`（逐字复制）

```python
from src.config import get_settings

from .client import ArxivClient


def make_arxiv_client() -> ArxivClient:
    """Factory function to create an arXiv client instance.

    :returns: An instance of the arXiv client
    :rtype: ArxivClient
    """
    # Get settings from centralized config
    settings = get_settings()

    # Create arXiv client with explicit settings
    client = ArxivClient(settings=settings.arxiv)

    return client
```

---

## 5.5 PDF 解析（Docling）：`src/services/pdf_parser/`

```bash
mkdir -p src/services/pdf_parser
touch src/services/pdf_parser/__init__.py
```

### 文件：`src/services/pdf_parser/docling.py`（逐字复制）

```python
import logging
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from src.exceptions import PDFParsingException, PDFValidationError
from src.schemas.pdf_parser.models import PaperSection, ParserType, PdfContent

logger = logging.getLogger(__name__)


class DoclingParser:
    """Docling PDF parser for scientific document processing."""

    def __init__(self, max_pages: int, max_file_size_mb: int, do_ocr: bool = False, do_table_structure: bool = True):
        """Initialize DocumentConverter with optimized pipeline options.

        :param max_pages: Maximum number of pages to process
        :param max_file_size_mb: Maximum file size in MB
        :param do_ocr: Enable OCR for scanned PDFs (default: False, very slow)
        :param do_table_structure: Extract table structures (default: True)
        """
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions(
            do_table_structure=do_table_structure,
            do_ocr=do_ocr,  # Usually disabled for speed
        )

        self._converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
        self._warmed_up = False
        self.max_pages = max_pages
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _warm_up_models(self):
        """Pre-warm the models with a small dummy document to avoid cold start."""
        if not self._warmed_up:
            # This happens only once per DoclingParser instance
            self._warmed_up = True

    def _validate_pdf(self, pdf_path: Path) -> bool:
        """Comprehensive PDF validation including size and page limits.

        :param pdf_path: Path to PDF file
        :returns: True if PDF appears valid and within limits, False otherwise
        """
        try:
            # Check file exists and is not empty
            if pdf_path.stat().st_size == 0:
                logger.error(f"PDF file is empty: {pdf_path}")
                raise PDFValidationError(f"PDF file is empty: {pdf_path}")

            # Check file size limit
            file_size = pdf_path.stat().st_size
            if file_size > self.max_file_size_bytes:
                logger.warning(
                    f"PDF file size ({file_size / 1024 / 1024:.1f}MB) exceeds limit ({self.max_file_size_bytes / 1024 / 1024:.1f}MB), skipping processing"
                )
                raise PDFValidationError(
                    f"PDF file too large: {file_size / 1024 / 1024:.1f}MB > {self.max_file_size_bytes / 1024 / 1024:.1f}MB"
                )

            # Check if file starts with PDF header
            with open(pdf_path, "rb") as f:
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    logger.error(f"File does not have PDF header: {pdf_path}")
                    raise PDFValidationError(f"File does not have PDF header: {pdf_path}")

            # Check page count limit
            pdf_doc = pdfium.PdfDocument(str(pdf_path))
            actual_pages = len(pdf_doc)
            pdf_doc.close()

            if actual_pages > self.max_pages:
                logger.warning(
                    f"PDF has {actual_pages} pages, exceeding limit of {self.max_pages} pages. Skipping processing to avoid performance issues."
                )
                raise PDFValidationError(f"PDF has too many pages: {actual_pages} > {self.max_pages}")

            return True

        except PDFValidationError:
            raise
        except Exception as e:
            logger.error(f"Error validating PDF {pdf_path}: {e}")
            raise PDFValidationError(f"Error validating PDF {pdf_path}: {e}")

    def parse_pdf(self, pdf_path: Path) -> Optional[PdfContent]:
        """Parse PDF using Docling parser.
        Limited to 20 pages to avoid memory issues with large papers.

        :param pdf_path: Path to PDF file
        :returns: PdfContent object or None if parsing failed
        """
        try:
            # Validate PDF first (includes size and page limits)
            self._validate_pdf(pdf_path)

            # Warm up models on first use
            self._warm_up_models()

            # Convert PDF using the modern API
            # Limit processing to avoid memory issues with large papers
            result = self._converter.convert(str(pdf_path), max_num_pages=self.max_pages, max_file_size=self.max_file_size_bytes)

            # Extract structured content
            doc = result.document

            # Extract sections from document structure
            sections = []
            current_section = {"title": "Content", "content": ""}

            for element in doc.texts:
                if hasattr(element, "label") and element.label in ["title", "section_header"]:
                    # Save previous section if it has content
                    if current_section["content"].strip():
                        sections.append(PaperSection(title=current_section["title"], content=current_section["content"].strip()))
                    # Start new section
                    current_section = {"title": element.text.strip(), "content": ""}
                else:
                    # Add content to current section
                    if hasattr(element, "text") and element.text:
                        current_section["content"] += element.text + "\n"

            # Add final section
            if current_section["content"].strip():
                sections.append(PaperSection(title=current_section["title"], content=current_section["content"].strip()))

            # Focus on what arXiv API doesn't provide: structured full text content only
            return PdfContent(
                sections=sections,
                figures=[],  # Removed: basic metadata not useful
                tables=[],  # Removed: basic metadata not useful
                raw_text=doc.export_to_text(),
                references=[],
                parser_used=ParserType.DOCLING,
                metadata={"source": "docling", "note": "Content extracted from PDF, metadata comes from arXiv API"},
            )

        except PDFValidationError as e:
            # Handle size/page limit validation errors gracefully by returning None
            error_msg = str(e).lower()
            if "too large" in error_msg or "too many pages" in error_msg:
                logger.info(f"Skipping PDF processing due to size/page limits: {e}")
                return None
            else:
                # Re-raise other validation errors (corrupted files, etc.)
                raise
        except Exception as e:
            logger.error(f"Failed to parse PDF with Docling: {e}")
            logger.error(f"PDF path: {pdf_path}")
            logger.error(f"PDF size: {pdf_path.stat().st_size} bytes")
            logger.error(f"Error type: {type(e).__name__}")

            # Add specific handling for common issues
            error_msg = str(e).lower()

            # Note: Page and size limit checks are now handled in _validate_pdf method

            if "not valid" in error_msg:
                logger.error("PDF appears to be corrupted or not a valid PDF file")
                raise PDFParsingException(f"PDF appears to be corrupted or invalid: {pdf_path}")
            elif "timeout" in error_msg:
                logger.error("PDF processing timed out - file may be too complex")
                raise PDFParsingException(f"PDF processing timed out: {pdf_path}")
            elif "memory" in error_msg or "ram" in error_msg:
                logger.error("Out of memory - PDF may be too large or complex")
                raise PDFParsingException(f"Out of memory processing PDF: {pdf_path}")
            elif "max_num_pages" in error_msg or "page" in error_msg:
                logger.error(f"PDF processing issue likely related to page limits (current limit: {self.max_pages} pages)")
                raise PDFParsingException(
                    f"PDF processing failed, possibly due to page limit ({self.max_pages} pages). Error: {e}"
                )
            else:
                raise PDFParsingException(f"Failed to parse PDF with Docling: {e}")
```

### DoclingParser 设计要点

- **先校验再解析**：空文件、超大、超页数、非 PDF 头都在 `_validate_pdf` 里拦截。超大/超页数**优雅跳过**（返回 `None`，只存元数据），而不是让整个管道崩。
- **页数/体积上限**（默认 30 页 / 20MB）：Docling 解析很重，限制资源占用（性能）。
- **章节抽取**：遍历 `doc.texts`，遇到 `title`/`section_header` 标签就开新章节——这正是 Week 4 "按章节分块"的输入。
- **`do_ocr=False`**：OCR 极慢，默认关闭；只处理"文本型"PDF。
- **`figures`/`tables` 留空**：上游有意只保留"arXiv API 不提供的"结构化全文，避免存无用元数据。

> ⚠️ **上游修复（修复集 #6）**：上游 `DoclingParser.parse_pdf` 是**同步**方法（近期一次提交 "remove unnecessary async from parse_pdf" 移除了它的 `async`），但 `PDFParserService.parse_pdf` 仍 `await self.docling_parser.parse_pdf(...)`——`await` 一个同步调用的返回值会抛 `TypeError`，**导致整条 PDF 解析链路（Week 2 摄取）崩溃**，对应的 `test_parse_pdf_success` 也会失败。下面**去掉了那个多余的 `await`**（用注释标出）。其余逐字复刻上游。

### 文件：`src/services/pdf_parser/parser.py`（含修复 #6，可直接运行）

```python
import logging
from pathlib import Path
from typing import Optional

from src.exceptions import PDFParsingException, PDFValidationError
from src.schemas.pdf_parser.models import PdfContent

from .docling import DoclingParser

logger = logging.getLogger(__name__)


class PDFParserService:
    """Main PDF parsing service using Docling only."""

    def __init__(self, max_pages: int, max_file_size_mb: int, do_ocr: bool = False, do_table_structure: bool = True):
        """Initialize PDF parser service with configurable limits."""
        self.docling_parser = DoclingParser(
            max_pages=max_pages, max_file_size_mb=max_file_size_mb, do_ocr=do_ocr, do_table_structure=do_table_structure
        )

    async def parse_pdf(self, pdf_path: Path) -> Optional[PdfContent]:
        """Parse PDF using Docling parser only.

        :param pdf_path: Path to PDF file
        :returns: PdfContent object or None if parsing failed
        """
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise PDFValidationError(f"PDF file not found: {pdf_path}")

        try:
            # ⚠️ 上游修复(#6): DoclingParser.parse_pdf 是同步方法，上游此处误用 await 会抛 TypeError。
            # 去掉 await，直接调用同步方法。
            result = self.docling_parser.parse_pdf(pdf_path)
            if result:
                logger.info(f"Parsed {pdf_path.name}")
                return result
            else:
                logger.error(f"Docling parsing returned no result for {pdf_path.name}")
                raise PDFParsingException(f"Docling parsing returned no result for {pdf_path.name}")

        except (PDFValidationError, PDFParsingException):
            raise
        except Exception as e:
            logger.error(f"Docling parsing error for {pdf_path.name}: {e}")
            raise PDFParsingException(f"Docling parsing error for {pdf_path.name}: {e}")
```

> `PDFParserService.parse_pdf` 本身是 `async def`（外层 `MetadataFetcher` 用 `await self.pdf_parser.parse_pdf(pdf_path)` 调用它，并用信号量限制并发）。但它**内部**调用的 `DoclingParser.parse_pdf` 是**同步**的——所以内部那一行**不能再 `await`**（已按修复 #6 去掉）。这种"外层 async、内层调用同步 CPU 密集函数"的写法是上游的真实结构。

### 文件：`src/services/pdf_parser/factory.py`（逐字复制）

```python
from functools import lru_cache

from src.config import get_settings

from .parser import PDFParserService


@lru_cache(maxsize=1)
def make_pdf_parser_service() -> PDFParserService:
    """Create cached PDF parser service using Docling."""
    settings = get_settings()
    return PDFParserService(
        max_pages=settings.pdf_parser.max_pages,
        max_file_size_mb=settings.pdf_parser.max_file_size_mb,
        do_ocr=settings.pdf_parser.do_ocr,
        do_table_structure=settings.pdf_parser.do_table_structure,
    )
```

> `@lru_cache(maxsize=1)` 让解析器成为单例：Docling 模型只加载一次（模型加载很贵），后续复用（性能）。

---

## 5.6 数据访问层：`src/repositories/paper.py`

```bash
touch src/repositories/__init__.py
```

### 文件：`src/repositories/paper.py`（逐字复制）

```python
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from src.models.paper import Paper
from src.schemas.arxiv.paper import PaperCreate


class PaperRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, paper: PaperCreate) -> Paper:
        db_paper = Paper(**paper.model_dump())
        self.session.add(db_paper)
        self.session.commit()
        self.session.refresh(db_paper)
        return db_paper

    def get_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        stmt = select(Paper).where(Paper.arxiv_id == arxiv_id)
        return self.session.scalar(stmt)

    def get_by_id(self, paper_id: UUID) -> Optional[Paper]:
        stmt = select(Paper).where(Paper.id == paper_id)
        return self.session.scalar(stmt)

    def get_all(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        stmt = select(Paper).order_by(Paper.published_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_count(self) -> int:
        stmt = select(func.count(Paper.id))
        return self.session.scalar(stmt) or 0

    def get_processed_papers(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """Get papers that have been successfully processed with PDF content."""
        stmt = (
            select(Paper)
            .where(Paper.pdf_processed == True)
            .order_by(Paper.pdf_processing_date.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.scalars(stmt))

    def get_unprocessed_papers(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """Get papers that haven't been processed for PDF content yet."""
        stmt = select(Paper).where(Paper.pdf_processed == False).order_by(Paper.published_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_papers_with_raw_text(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """Get papers that have raw text content stored."""
        stmt = select(Paper).where(Paper.raw_text != None).order_by(Paper.pdf_processing_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_processing_stats(self) -> dict:
        """Get statistics about PDF processing status."""
        total_papers = self.get_count()

        # Count processed papers
        processed_stmt = select(func.count(Paper.id)).where(Paper.pdf_processed == True)
        processed_papers = self.session.scalar(processed_stmt) or 0

        # Count papers with text
        text_stmt = select(func.count(Paper.id)).where(Paper.raw_text != None)
        papers_with_text = self.session.scalar(text_stmt) or 0

        return {
            "total_papers": total_papers,
            "processed_papers": processed_papers,
            "papers_with_text": papers_with_text,
            "processing_rate": (processed_papers / total_papers * 100) if total_papers > 0 else 0,
            "text_extraction_rate": (papers_with_text / processed_papers * 100) if processed_papers > 0 else 0,
        }

    def update(self, paper: Paper) -> Paper:
        self.session.add(paper)
        self.session.commit()
        self.session.refresh(paper)
        return paper

    def upsert(self, paper_create: PaperCreate) -> Paper:
        # Check if paper already exists
        existing_paper = self.get_by_arxiv_id(paper_create.arxiv_id)
        if existing_paper:
            # Update existing paper with new content
            for key, value in paper_create.model_dump(exclude_unset=True).items():
                setattr(existing_paper, key, value)
            return self.update(existing_paper)
        else:
            # Create new paper
            return self.create(paper_create)
```

### PaperRepository 设计要点

- **`upsert` 按 `arxiv_id` 去重**：重复抓到的论文会更新而非重复插入。`exclude_unset=True` 只更新本次提供的字段（避免把已有内容覆盖成 None）。
- **`get_unprocessed_papers` / `get_processed_papers`**：用 `pdf_processed` 标志区分，支持"先存元数据，后补解析内容"的两阶段流程。
- **仓储模式**：所有 DB 访问集中在这里，业务层不直接写 SQL（可维护/可测试）。

> 注：`Paper.pdf_processed == True` / `!= None` 是 SQLAlchemy 的表达式写法（生成 SQL），不是 Python 的真值判断，**这里必须这么写**，不能改成 `is True` / `is not None`。

---

## 5.7 摄取编排器：`src/services/metadata_fetcher.py`

### 文件：`src/services/metadata_fetcher.py`（逐字复制）

```python
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dateutil import parser as date_parser
from sqlalchemy.orm import Session
from src.config import Settings
from src.exceptions import MetadataFetchingException, PipelineException
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import ArxivPaper, PaperCreate
from src.schemas.pdf_parser.models import ArxivMetadata, ParsedPaper
from src.services.arxiv.client import ArxivClient
from src.services.pdf_parser.parser import PDFParserService

logger = logging.getLogger(__name__)


class MetadataFetcher:
    """Service for fetching arXiv papers with PDF processing and database storage."""

    def __init__(
        self,
        arxiv_client: ArxivClient,
        pdf_parser: PDFParserService,
        pdf_cache_dir: Optional[Path] = None,
        max_concurrent_downloads: int = 5,
        max_concurrent_parsing: int = 3,
        settings: Optional[Settings] = None,
    ):
        """Initialize metadata fetcher with services and settings.

        :param arxiv_client: Client for arXiv API operations
        :param pdf_parser: Service for parsing PDF documents
        :param pdf_cache_dir: Directory for caching downloaded PDFs
        :param max_concurrent_downloads: Maximum concurrent PDF downloads
        :param max_concurrent_parsing: Maximum concurrent PDF parsing operations
        :param settings: Application settings instance
        :type arxiv_client: ArxivClient
        :type pdf_parser: PDFParserService
        :type pdf_cache_dir: Optional[Path]
        :type max_concurrent_downloads: int
        :type max_concurrent_parsing: int
        :type settings: Optional[Settings]
        """
        from src.config import get_settings

        self.arxiv_client = arxiv_client
        self.pdf_parser = pdf_parser
        self.pdf_cache_dir = pdf_cache_dir or self.arxiv_client.pdf_cache_dir
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_parsing = max_concurrent_parsing
        self.settings = settings or get_settings()

    async def fetch_and_process_papers(
        self,
        max_results: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        process_pdfs: bool = True,
        store_to_db: bool = True,
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Fetch papers from arXiv, process PDFs, and store to database.

        :param max_results: Maximum papers to fetch
        :param from_date: Filter papers from this date (YYYYMMDD)
        :param to_date: Filter papers to this date (YYYYMMDD)
        :param process_pdfs: Whether to download and parse PDFs
        :param store_to_db: Whether to store results in database
        :param db_session: Database session (required if store_to_db=True)
        :type max_results: Optional[int]
        :type from_date: Optional[str]
        :type to_date: Optional[str]
        :type process_pdfs: bool
        :type store_to_db: bool
        :type db_session: Optional[Session]
        :returns: Dictionary with processing results and statistics
        :rtype: Dict[str, Any]
        """

        results = {
            "papers_fetched": 0,
            "pdfs_downloaded": 0,
            "pdfs_parsed": 0,
            "papers_stored": 0,
            "errors": [],
            "processing_time": 0,
        }

        start_time = datetime.now()

        try:
            # Step 1: Fetch paper metadata from arXiv
            papers = await self.arxiv_client.fetch_papers(
                max_results=max_results, from_date=from_date, to_date=to_date, sort_by="submittedDate", sort_order="descending"
            )

            results["papers_fetched"] = len(papers)

            if not papers:
                logger.warning("No papers found")
                return results

            # Step 2: Process PDFs if requested
            pdf_results = {}
            if process_pdfs:
                pdf_results = await self._process_pdfs_batch(papers)
                results["pdfs_downloaded"] = pdf_results["downloaded"]
                results["pdfs_parsed"] = pdf_results["parsed"]
                results["errors"].extend(pdf_results["errors"])

            # Step 3: Store to database if requested
            if store_to_db and db_session:
                logger.info("Step 3: Storing papers to database...")
                stored_count = self._store_papers_to_db(papers, pdf_results.get("parsed_papers", {}), db_session)
                results["papers_stored"] = stored_count
            elif store_to_db:
                logger.warning("Database storage requested but no session provided")
                results["errors"].append("Database session not provided for storage")

            # Calculate total processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            results["processing_time"] = processing_time

            # Simple logging summary
            logger.info(
                f"Pipeline completed in {processing_time:.1f}s: {results['papers_fetched']} papers, {results['pdfs_downloaded']} PDFs, {len(results['errors'])} errors"
            )

            if results["errors"]:
                logger.warning("Errors summary:")
                for i, error in enumerate(results["errors"][:5], 1):  # Show first 5 errors
                    logger.warning(f"  {i}. {error}")
                if len(results["errors"]) > 5:
                    logger.warning(f"  ... and {len(results['errors']) - 5} more errors")

            return results

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            results["errors"].append(f"Pipeline error: {str(e)}")
            raise PipelineException(f"Pipeline execution failed: {e}") from e

    async def _process_pdfs_batch(self, papers: List[ArxivPaper]) -> Dict[str, Any]:
        """
        Process PDFs for a batch of papers with async concurrency.

        Uses overlapping download+parse pipeline:
        - Downloads happen concurrently (up to max_concurrent_downloads)
        - As each download completes, parsing starts immediately
        - Multiple PDFs can be parsing while others are still downloading

        This is optimal for production workloads like 100 papers/day.

        Args:
            papers: List of ArxivPaper objects

        Returns:
            Dictionary with processing results and statistics
        """
        results = {
            "downloaded": 0,
            "parsed": 0,
            "parsed_papers": {},
            "errors": [],
            "download_failures": [],
            "parse_failures": [],
        }

        logger.info(f"Starting async pipeline for {len(papers)} PDFs...")
        logger.info(f"Concurrent downloads: {self.max_concurrent_downloads}")
        logger.info(f"Concurrent parsing: {self.max_concurrent_parsing}")

        # Create semaphores for controlled concurrency
        download_semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        parse_semaphore = asyncio.Semaphore(self.max_concurrent_parsing)

        # Start all download+parse pipelines concurrently
        pipeline_tasks = [self._download_and_parse_pipeline(paper, download_semaphore, parse_semaphore) for paper in papers]

        # Wait for all pipelines to complete
        pipeline_results = await asyncio.gather(*pipeline_tasks, return_exceptions=True)

        # Process results with detailed error tracking
        for paper, result in zip(papers, pipeline_results):
            if isinstance(result, Exception):
                error_msg = f"Pipeline error for {paper.arxiv_id}: {str(result)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
            elif result:
                # Check if result is a tuple before unpacking
                # Handle AirflowTaskTerminated and other non-tuple results
                if isinstance(result, tuple) and len(result) == 2:
                    # Result is tuple: (download_success, parsed_paper)
                    download_success, parsed_paper = result
                else:
                    # Result is not a tuple (could be AirflowTaskTerminated or other error)
                    error_msg = f"Pipeline error for {paper.arxiv_id}: Unexpected result type {type(result).__name__}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    continue

                if download_success:
                    results["downloaded"] += 1

                    if parsed_paper:
                        results["parsed"] += 1
                        results["parsed_papers"][paper.arxiv_id] = parsed_paper
                    else:
                        # Download succeeded but parsing failed
                        results["parse_failures"].append(paper.arxiv_id)
                else:
                    # Download failed
                    results["download_failures"].append(paper.arxiv_id)
            else:
                # No result returned (shouldn't happen but handle gracefully)
                results["download_failures"].append(paper.arxiv_id)

        # Simple processing summary
        logger.info(f"PDF processing: {results['downloaded']}/{len(papers)} downloaded, {results['parsed']} parsed")

        if results["download_failures"]:
            logger.warning(f"Download failures: {len(results['download_failures'])}")

        if results["parse_failures"]:
            logger.warning(f"Parse failures: {len(results['parse_failures'])}")

        # Add specific failure info to general errors list for backward compatibility
        if results["download_failures"]:
            results["errors"].extend([f"Download failed: {arxiv_id}" for arxiv_id in results["download_failures"]])
        if results["parse_failures"]:
            results["errors"].extend([f"PDF parse failed: {arxiv_id}" for arxiv_id in results["parse_failures"]])

        return results

    async def _download_and_parse_pipeline(
        self, paper: ArxivPaper, download_semaphore: asyncio.Semaphore, parse_semaphore: asyncio.Semaphore
    ) -> tuple:
        """
        Complete download+parse pipeline for a single paper with true parallelism.
        Downloads PDF, then immediately starts parsing while other downloads continue.

        Returns:
            Tuple of (download_success: bool, parsed_paper: Optional[ParsedPaper])
        """
        download_success = False
        parsed_paper = None

        try:
            # Step 1: Download PDF with download concurrency control
            async with download_semaphore:
                logger.debug(f"Starting download: {paper.arxiv_id}")
                pdf_path = await self.arxiv_client.download_pdf(paper, False)

                if pdf_path:
                    download_success = True
                    logger.debug(f"Download complete: {paper.arxiv_id}")
                else:
                    logger.error(f"Download failed: {paper.arxiv_id}")
                    return (False, None)

            # Step 2: Parse PDF with parse concurrency control (happens AFTER download completes)
            # This allows other downloads to continue while this PDF is being parsed
            async with parse_semaphore:
                logger.debug(f"Starting parse: {paper.arxiv_id}")
                pdf_content = await self.pdf_parser.parse_pdf(pdf_path)

                if pdf_content:
                    # Create ArxivMetadata from the paper
                    arxiv_metadata = ArxivMetadata(
                        title=paper.title,
                        authors=paper.authors,
                        abstract=paper.abstract,
                        arxiv_id=paper.arxiv_id,
                        categories=paper.categories,
                        published_date=paper.published_date,
                        pdf_url=paper.pdf_url,
                    )

                    # Combine into ParsedPaper
                    parsed_paper = ParsedPaper(arxiv_metadata=arxiv_metadata, pdf_content=pdf_content)
                    logger.debug(f"Parse complete: {paper.arxiv_id} - {len(pdf_content.raw_text)} chars extracted")
                else:
                    # PDF parsing failed, but this is not critical - we can continue with metadata only
                    logger.warning(f"PDF parsing failed for {paper.arxiv_id}, continuing with metadata only")

        except Exception as e:
            logger.error(f"Pipeline error for {paper.arxiv_id}: {e}")
            raise MetadataFetchingException(f"Pipeline error for {paper.arxiv_id}: {e}") from e

        return (download_success, parsed_paper)

    def _serialize_parsed_content(self, parsed_paper: ParsedPaper) -> Dict[str, Any]:
        """Serialize ParsedPaper content for database storage.

        :param parsed_paper: ParsedPaper object with PDF content
        :type parsed_paper: ParsedPaper
        :returns: Dictionary with serialized content for database storage
        :rtype: Dict[str, Any]
        """
        try:
            pdf_content = parsed_paper.pdf_content
            if pdf_content is None:
                return {"pdf_processed": False, "parser_metadata": {"note": "No PDF content to serialize"}}

            # Serialize sections
            sections = [{"title": section.title, "content": section.content} for section in pdf_content.sections]

            # Serialize references
            references = list(pdf_content.references)  #

            return {
                "raw_text": pdf_content.raw_text,
                "sections": sections,
                "references": references,
                "parser_used": pdf_content.parser_used.value if pdf_content.parser_used else None,
                "parser_metadata": pdf_content.metadata or {},
                "pdf_processed": True,
                "pdf_processing_date": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Failed to serialize parsed content: {e}")
            return {"pdf_processed": False, "parser_metadata": {"error": str(e)}}

    def _store_papers_to_db(
        self,
        papers: List[ArxivPaper],
        parsed_papers: Dict[str, ParsedPaper],
        db_session: Session,
    ) -> int:
        """
        Store papers and parsed content to database with comprehensive content storage.

        Args:
            papers: List of ArxivPaper metadata
            parsed_papers: Dictionary of parsed PDF content by arxiv_id
            db_session: Database session

        Returns:
            Number of papers stored successfully
        """
        paper_repo = PaperRepository(db_session)
        stored_count = 0

        for paper in papers:
            try:
                # Get parsed content if available
                parsed_paper = parsed_papers.get(paper.arxiv_id)

                # Base paper data
                published_date = (
                    date_parser.parse(paper.published_date) if isinstance(paper.published_date, str) else paper.published_date
                )
                paper_data = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "published_date": published_date,
                    "pdf_url": paper.pdf_url,
                }

                # Add parsed content if available
                if parsed_paper:
                    parsed_content = self._serialize_parsed_content(parsed_paper)
                    paper_data.update(parsed_content)
                    logger.debug(
                        f"Storing paper {paper.arxiv_id} with parsed content ({len(parsed_content.get('raw_text', '')) if parsed_content.get('raw_text') else 0} chars)"
                    )
                else:
                    # No parsed content - just store metadata
                    paper_data.update(
                        {"pdf_processed": False, "parser_metadata": {"note": "PDF processing not available or failed"}}
                    )
                    logger.debug(f"Storing paper {paper.arxiv_id} with metadata only")

                paper_create = PaperCreate(**paper_data)
                stored_paper = paper_repo.upsert(paper_create)

                if stored_paper:
                    stored_count += 1
                    content_info = "with parsed content" if parsed_paper else "metadata only"
                    logger.debug(f"Stored paper {paper.arxiv_id} to database ({content_info})")

            except Exception as e:
                logger.error(f"Failed to store paper {paper.arxiv_id}: {e}")

        # Commit all changes
        try:
            db_session.commit()
            logger.info(f"Committed {stored_count} papers to database with full content storage")
        except Exception as e:
            logger.error(f"Failed to commit papers to database: {e}")
            db_session.rollback()
            stored_count = 0

        return stored_count


def make_metadata_fetcher(
    arxiv_client: ArxivClient,
    pdf_parser: PDFParserService,
    pdf_cache_dir: Optional[Path] = None,
    settings: Optional[Settings] = None,
) -> MetadataFetcher:
    """Create MetadataFetcher instance with configuration settings.

    :param arxiv_client: Client for arXiv API operations
    :param pdf_parser: Service for parsing PDF documents
    :param pdf_cache_dir: Directory for caching downloaded PDFs
    :param settings: Application settings instance (uses default if None)
    :type arxiv_client: ArxivClient
    :type pdf_parser: PDFParserService
    :type pdf_cache_dir: Optional[Path]
    :type settings: Optional[Settings]
    :returns: Configured MetadataFetcher instance
    :rtype: MetadataFetcher
    """
    from src.config import get_settings

    if settings is None:
        settings = get_settings()

    return MetadataFetcher(
        arxiv_client=arxiv_client,
        pdf_parser=pdf_parser,
        pdf_cache_dir=pdf_cache_dir,
        max_concurrent_downloads=settings.arxiv.max_concurrent_downloads,
        max_concurrent_parsing=settings.arxiv.max_concurrent_parsing,
        settings=settings,
    )
```

### MetadataFetcher 设计要点

- **边下边解的重叠流水线**：用两个信号量分别限制下载并发（默认 5）和解析并发（默认 1）。下载完一篇就立刻进入解析，其它论文继续下载——比"全下完再全解析"快很多（性能）。
- **`asyncio.gather(..., return_exceptions=True)`**：单篇失败不影响整批；逐篇统计 `download_failures` / `parse_failures`。
- **解析失败不致命**：解析失败仍存元数据（`pdf_processed=False`），保证"至少有元数据可检索"。
- **日期解析**：arXiv 的 ISO 字符串用 `dateutil.parser.parse` 转成 `datetime` 再入库。
- **职责单一**：`MetadataFetcher` 只做"抓取 → 解析 PDF → 写库"。建索引（OpenSearch）是 Week 4 之后独立出去的 `HybridIndexingService` 的事，由 Airflow DAG 单独编排，不在本服务内。

---

## 5.8 Airflow 数据管道

Airflow 用独立镜像运行（与 API 镜像分开），通过卷把 `src/` 挂进容器，复用上面的服务代码。

```bash
mkdir -p airflow/dags/arxiv_ingestion
touch airflow/dags/arxiv_ingestion/__init__.py
```

### 文件：`airflow/Dockerfile`（逐字复制）

```dockerfile
FROM python:3.12-slim

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW_VERSION=2.10.3
ENV PYTHON_VERSION=3.12
ENV CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
        poppler-utils \
        tesseract-ocr \
        && rm -rf /var/lib/apt/lists/*

# Create airflow user with UID/GID 50000 for cross-platform compatibility
RUN groupadd -r -g 50000 airflow && useradd -r -u 50000 -g airflow -d ${AIRFLOW_HOME} -s /bin/bash airflow

# Create airflow directories with proper ownership
RUN mkdir -p ${AIRFLOW_HOME} && \
    mkdir -p ${AIRFLOW_HOME}/dags && \
    mkdir -p ${AIRFLOW_HOME}/logs && \
    mkdir -p ${AIRFLOW_HOME}/plugins && \
    chown -R 50000:50000 ${AIRFLOW_HOME} && \
    chmod -R 755 ${AIRFLOW_HOME}

# Install Airflow with PostgreSQL support
RUN pip install --no-cache-dir \
    "apache-airflow[postgres]==${AIRFLOW_VERSION}" \
    --constraint "${CONSTRAINT_URL}" \
    psycopg2-binary

# Copy requirements and install project dependencies
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

# Copy and set up entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to airflow user and set working directory
USER airflow
WORKDIR ${AIRFLOW_HOME}

# Expose port
EXPOSE 8080

CMD ["/entrypoint.sh"]
```

> **为什么 Airflow 用独立镜像 + 独立依赖（`requirements-airflow.txt`）而不复用 API 镜像？**
> - Airflow 有自己的版本约束（用官方 constraints 文件锁定），与应用依赖混装容易冲突。
> - 任务只需要"摄取/索引"相关依赖子集，镜像更聚焦。
> - `poppler-utils`/`tesseract-ocr` 是 PDF/OCR 的系统依赖。

### 文件：`airflow/requirements-airflow.txt`（逐字复制）

```text
# Core dependencies needed for Airflow tasks
httpx>=0.27.0
sqlalchemy>=1.4.36,<2.0.0
pydantic>=2.0.0,<3.0.0
python-dateutil>=2.8.0

# PDF processing dependencies  
docling>=2.0.0

# Search engine dependencies
opensearch-py>=2.4.0

# Database drivers
psycopg2-binary>=2.9.0
```

### 文件：`airflow/entrypoint.sh`（逐字复制）

```bash
#!/bin/bash
set -e

# Clean up any existing PID files and processes
echo "Cleaning up any existing Airflow processes..."
pkill -f "airflow webserver" || true
pkill -f "airflow scheduler" || true
rm -f /opt/airflow/airflow-webserver.pid
rm -f /opt/airflow/airflow-scheduler.pid

# Wait a moment for processes to fully terminate
sleep 2

# Initialize Airflow database
echo "Initializing Airflow database..."
airflow db init

# Create admin user with admin/admin credentials
echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || echo "Admin user already exists"

# Start webserver and scheduler
echo "Starting Airflow webserver and scheduler..."
airflow webserver --port 8080 --daemon &
airflow scheduler
```

> **安全提醒**：入口脚本创建了 `admin`/`admin` 账户，仅用于本地。**生产环境必须改密并启用正式鉴权**（见第 [14](14-quality-performance-security.md) 章）。

### 共享服务工厂：`airflow/dags/arxiv_ingestion/common.py`

```python
import logging
import sys
from functools import lru_cache
from typing import Any, Tuple

sys.path.insert(0, "/opt/airflow")

from src.db.factory import make_database
from src.services.arxiv.factory import make_arxiv_client
from src.services.metadata_fetcher import make_metadata_fetcher
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cached_services() -> Tuple[Any, Any, Any, Any, Any]:
    """Get cached service instances using lru_cache for automatic memoization.

    :returns: Tuple of (arxiv_client, pdf_parser, database, metadata_fetcher, opensearch_client)
    """
    logger.info("Initializing services (cached with lru_cache)")

    # Initialize core services
    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()
    database = make_database()
    opensearch_client = make_opensearch_client()

    # Create metadata fetcher with dependencies
    metadata_fetcher = make_metadata_fetcher(arxiv_client, pdf_parser)

    logger.info("All services initialized and cached with lru_cache")
    return arxiv_client, pdf_parser, database, metadata_fetcher, opensearch_client
```

> `sys.path.insert(0, "/opt/airflow")` 让容器内能 `import src.*`（`src/` 被挂载到 `/opt/airflow/src`，且 `PYTHONPATH=/opt/airflow/src`）。`get_cached_services` 用 `make_opensearch_client`（Week 3），所以该任务在 Week 3 之后才完整可用。

### 抓取任务：`airflow/dags/arxiv_ingestion/fetching.py`

```python
import asyncio
import logging
from datetime import datetime, timedelta

from .common import get_cached_services

logger = logging.getLogger(__name__)


async def run_paper_ingestion_pipeline(
    target_date: str,
    process_pdfs: bool = True,
) -> dict:
    """Async wrapper for the paper ingestion pipeline.

    :param target_date: Date to fetch papers for (YYYYMMDD format)
    :param process_pdfs: Whether to download and process PDFs
    :returns: Dictionary with ingestion statistics
    """
    arxiv_client, _, database, metadata_fetcher, _ = get_cached_services()

    max_results = arxiv_client.max_results
    logger.info(f"Using default max_results from config: {max_results}")

    with database.get_session() as session:
        return await metadata_fetcher.fetch_and_process_papers(
            max_results=max_results,
            from_date=target_date,
            to_date=target_date,
            process_pdfs=process_pdfs,
            store_to_db=True,
            db_session=session,
        )


def fetch_daily_papers(**context):
    """Fetch daily papers from arXiv and store in PostgreSQL.

    This task:
    1. Determines the target date (defaults to yesterday)
    2. Fetches papers from arXiv API
    3. Downloads and processes PDFs using Docling
    4. Stores metadata and parsed content in PostgreSQL

    Note: OpenSearch indexing is handled by a separate dedicated task
    """
    logger.info("Starting daily paper fetching task")

    execution_date = context.get("execution_date")
    if execution_date:
        target_dt = execution_date - timedelta(days=1)
        target_date = target_dt.strftime("%Y%m%d")
    else:
        yesterday = datetime.now() - timedelta(days=1)
        target_date = yesterday.strftime("%Y%m%d")

    logger.info(f"Fetching papers for date: {target_date}")

    results = asyncio.run(
        run_paper_ingestion_pipeline(
            target_date=target_date,
            process_pdfs=True,
        )
    )

    logger.info(f"Daily fetch complete: {results['papers_fetched']} papers for {target_date}")

    results["date"] = target_date
    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="fetch_results", value=results)

    return results
```

### 索引任务：`airflow/dags/arxiv_ingestion/indexing.py`

> ⚠️ 本文件用到 **Week 4** 的 `make_hybrid_indexing_service`（第 [07](07-week4-hybrid-search.md) 章创建）。现在先放进项目，等 Week 4 完成后该任务即可端到端运行。

```python
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from src.db.factory import make_database
from src.services.indexing.factory import make_hybrid_indexing_service
from src.services.opensearch.factory import make_opensearch_client_fresh

logger = logging.getLogger(__name__)


async def _index_papers_with_chunks(papers):
    """Async helper to index papers with chunking and embeddings."""
    indexing_service = make_hybrid_indexing_service()

    papers_data = []
    for paper in papers:
        if hasattr(paper, "__dict__"):
            paper_dict = {
                "id": str(paper.id),
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "categories": paper.categories,
                "published_date": paper.published_date,
                "raw_text": paper.raw_text,
                "sections": paper.sections,
            }
        else:
            paper_dict = paper
        papers_data.append(paper_dict)

    stats = await indexing_service.index_papers_batch(papers=papers_data, replace_existing=True)

    return stats


def index_papers_hybrid(**context):
    """Index papers with chunking and vector embeddings for hybrid search.

    This task:
    1. Fetches recently processed papers from PostgreSQL
    2. Chunks them into overlapping segments (600 words, 100 overlap)
    3. Generates embeddings using Jina AI
    4. Indexes chunks with embeddings into OpenSearch
    """
    try:
        database = make_database()

        ti = context.get("ti")

        fetch_results = None
        if ti:
            fetch_results = ti.xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")

        with database.get_session() as session:
            from src.models.paper import Paper

            if fetch_results and fetch_results.get("papers_stored", 0) > 0:
                from sqlalchemy import desc

                papers = session.query(Paper).order_by(desc(Paper.created_at)).limit(fetch_results["papers_stored"]).all()
            else:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=1)
                papers = session.query(Paper).filter(Paper.created_at >= cutoff_date).all()

            if not papers:
                logger.info("No papers to index for hybrid search")
                return {"papers_indexed": 0, "chunks_created": 0}

            logger.info(f"Indexing {len(papers)} papers for hybrid search")

            stats = asyncio.run(_index_papers_with_chunks(papers))

            logger.info(
                f"Hybrid indexing complete: {stats['papers_processed']} papers, "
                f"{stats['total_chunks_created']} chunks created, "
                f"{stats['total_chunks_indexed']} chunks indexed"
            )

            if ti:
                ti.xcom_push(key="hybrid_index_stats", value=stats)

            return stats

    except Exception as e:
        logger.error(f"Failed to index papers for hybrid search: {e}")
        raise


def verify_hybrid_index(**context):
    """Verify hybrid index health and get statistics."""
    try:
        opensearch_client = make_opensearch_client_fresh()

        stats = opensearch_client.client.indices.stats(index=opensearch_client.index_name)

        count = opensearch_client.client.count(index=opensearch_client.index_name)

        paper_count_query = {"aggs": {"unique_papers": {"cardinality": {"field": "arxiv_id"}}}, "size": 0}

        paper_count_response = opensearch_client.client.search(index=opensearch_client.index_name, body=paper_count_query)

        unique_papers = paper_count_response["aggregations"]["unique_papers"]["value"]

        result = {
            "index_name": opensearch_client.index_name,
            "total_chunks": count["count"],
            "unique_papers": unique_papers,
            "avg_chunks_per_paper": (count["count"] / unique_papers if unique_papers > 0 else 0),
            "index_size_mb": stats["indices"][opensearch_client.index_name]["total"]["store"]["size_in_bytes"] / (1024 * 1024),
        }

        logger.info(
            f"Hybrid index stats: {result['total_chunks']} chunks, "
            f"{result['unique_papers']} papers, "
            f"{result['avg_chunks_per_paper']:.1f} chunks/paper"
        )

        return result

    except Exception as e:
        logger.error(f"Failed to verify hybrid index: {e}")
        raise
```

### 环境校验任务：`airflow/dags/arxiv_ingestion/setup.py`

```python
import logging

from sqlalchemy import text

from .common import get_cached_services

logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment and verify dependencies.

    Creates hybrid search index with RRF pipeline.
    """
    logger.info("Setting up environment for arXiv paper ingestion")

    try:
        arxiv_client, _pdf_parser, database, _metadata_fetcher, opensearch_client = get_cached_services()

        with database.get_session() as session:
            session.execute(text("SELECT 1"))
            logger.info("Database connection verified")

        try:
            health = opensearch_client.client.cluster.health()
            if health["status"] in ["green", "yellow", "red"]:
                logger.info(f"OpenSearch hybrid client connected (cluster status: {health['status']})")
            else:
                raise Exception(f"OpenSearch cluster unhealthy: {health['status']}")
        except Exception as e:
            raise Exception(f"OpenSearch hybrid client connection failed: {e}")

        setup_results = opensearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("Hybrid search index created with vector support")
        else:
            logger.info("Hybrid search index already exists")

        if setup_results.get("rrf_pipeline"):
            logger.info("RRF pipeline created successfully")
        else:
            logger.info("RRF pipeline already exists")

        logger.info("Hybrid search setup completed")

        logger.info(f"arXiv client ready: {arxiv_client.base_url}")
        logger.info("PDF parser service ready (Docling models cached)")

        return {"status": "success", "message": "Environment setup completed"}

    except Exception as e:
        error_msg = f"Environment setup failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
```

### 报告任务：`airflow/dags/arxiv_ingestion/reporting.py`

```python
import json
import logging
from datetime import datetime

from .common import get_cached_services

logger = logging.getLogger(__name__)


def generate_daily_report(**context):
    """Generate a daily report of the ingestion pipeline results.

    Collects statistics from all previous tasks and generates a summary report.
    """
    logger.info("Generating daily ingestion report")

    ti = context.get("ti")
    if not ti:
        logger.warning("No task instance available, generating basic report")
        return {"status": "basic_report", "message": "No task instance for XCom data"}

    fetch_stats = ti.xcom_pull(task_ids="fetch_daily_papers", key="fetch_results") or {}
    hybrid_stats = ti.xcom_pull(task_ids="index_papers_hybrid", key="hybrid_index_stats") or {}

    report = {
        "execution_date": context.get("execution_date", datetime.now()).isoformat(),
        "fetch_statistics": {
            "papers_fetched": fetch_stats.get("papers_fetched", 0),
            "papers_stored": fetch_stats.get("papers_stored", 0),
            "target_date": fetch_stats.get("date", "unknown"),
        },
        "indexing_statistics": {
            "papers_processed": hybrid_stats.get("papers_processed", 0),
            "chunks_created": hybrid_stats.get("total_chunks_created", 0),
            "chunks_indexed": hybrid_stats.get("total_chunks_indexed", 0),
            "embeddings_generated": hybrid_stats.get("total_embeddings_generated", 0),
        },
        "pipeline_status": "success" if fetch_stats and hybrid_stats else "partial",
    }

    try:
        _arxiv_client, _pdf_parser, database, _metadata_fetcher, opensearch_client = get_cached_services()

        with database.get_session() as session:
            from sqlalchemy import func
            from src.models.paper import Paper

            total_papers = session.query(func.count(Paper.id)).scalar()
            report["database_statistics"] = {"total_papers": total_papers}

        if opensearch_client.health_check():
            try:
                stats_response = opensearch_client.client.indices.stats(index=opensearch_client.index_name)

                count_response = opensearch_client.client.count(index=opensearch_client.index_name)

                index_stats = stats_response["indices"][opensearch_client.index_name]["total"]

                report["opensearch_statistics"] = {
                    "index_name": opensearch_client.index_name,
                    "document_count": count_response["count"],
                    "index_size_mb": round(index_stats["store"]["size_in_bytes"] / (1024 * 1024), 2),
                }
            except Exception as stats_error:
                logger.error(f"Failed to get OpenSearch statistics: {stats_error}")
                report["opensearch_statistics"] = {"index_name": opensearch_client.index_name, "error": str(stats_error)}
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        report["error"] = str(e)

    logger.info("Daily Ingestion Report:")
    logger.info(json.dumps(report, indent=2))

    ti.xcom_push(key="daily_report", value=report)

    return report
```

### DAG 定义：`airflow/dags/arxiv_paper_ingestion.py`

```python
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from arxiv_ingestion.fetching import fetch_daily_papers
from arxiv_ingestion.indexing import index_papers_hybrid, verify_hybrid_index
from arxiv_ingestion.reporting import generate_daily_report

# Import task functions from modular structure
from arxiv_ingestion.setup import setup_environment

# Default DAG arguments
default_args = {
    "owner": "arxiv-curator",
    "depends_on_past": False,
    "start_date": datetime(2025, 8, 8),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=30),
    "catchup": False,
}

# Create the DAG
dag = DAG(
    "arxiv_paper_ingestion",
    default_args=default_args,
    description="Daily arXiv CS.AI paper pipeline: fetch → store to PostgreSQL → chunk & embed → hybrid OpenSearch indexing",
    schedule="0 6 * * 1-5",  # Monday-Friday at 6 AM UTC
    max_active_runs=1,
    catchup=False,
    tags=["arxiv", "papers", "ingestion", "hybrid-search", "embeddings", "chunks"],
)

# Task definitions
setup_task = PythonOperator(
    task_id="setup_environment",
    python_callable=setup_environment,
    dag=dag,
)

fetch_task = PythonOperator(
    task_id="fetch_daily_papers",
    python_callable=fetch_daily_papers,
    dag=dag,
)

# Hybrid search indexing task (replaces old OpenSearch task)
index_hybrid_task = PythonOperator(
    task_id="index_papers_hybrid",
    python_callable=index_papers_hybrid,
    dag=dag,
)

report_task = PythonOperator(
    task_id="generate_daily_report",
    python_callable=generate_daily_report,
    dag=dag,
)

cleanup_task = BashOperator(
    task_id="cleanup_temp_files",
    bash_command="""
    echo "Cleaning up temporary files..."
    # Remove PDFs older than 30 days to manage disk space
    find /tmp -name "*.pdf" -type f -mtime +30 -delete 2>/dev/null || true
    echo "Cleanup completed"
    """,
    dag=dag,
)

# Task dependencies
# Simplified pipeline: setup -> fetch -> hybrid index -> report -> cleanup
setup_task >> fetch_task >> index_hybrid_task >> report_task >> cleanup_task
```

### DAG 设计要点

- **任务依赖链**：`setup → fetch → index → report → cleanup`，用 `>>` 表达。
- **`schedule="0 6 * * 1-5"`**：每周一至周五 06:00 UTC（arXiv 周末不更新）。
- **`retries=2` + `retry_delay=30min`**：抓取/网络任务天然不稳定，自动重试。
- **`catchup=False`**：不补跑历史日期，只跑当前。
- **XCom 传参**：`fetch` 把结果推到 XCom，`index`/`report` 再拉取——任务间解耦传递数据。
- **模块化任务函数**：逻辑放在 `arxiv_ingestion/*.py`，DAG 文件只做编排，便于单测与维护。

---

## 5.9 把摄取服务接入 FastAPI（main.py / dependencies.py 增量）

虽然摄取主要由 Airflow 跑，但应用启动时也会构建 arXiv 客户端与 PDF 解析器（供未来端点与笔记本演示用）。

> 以下是**对 Week 1 引导版的增量改动**。最终完整版见第 [10](10-week7-agentic-telegram.md) 章。

在 `src/dependencies.py` 顶部 import 区加入，并补充对应的 getter 与类型别名：

```python
# 在 import 区加入：
from src.services.arxiv.client import ArxivClient
from src.services.pdf_parser.parser import PDFParserService

# 在文件中加入：
def get_arxiv_client(request: Request) -> ArxivClient:
    """Get arXiv client from the request state."""
    return request.app.state.arxiv_client


def get_pdf_parser(request: Request) -> PDFParserService:
    """Get PDF parser service from the request state."""
    return request.app.state.pdf_parser


ArxivDep = Annotated[ArxivClient, Depends(get_arxiv_client)]
PDFParserDep = Annotated[PDFParserService, Depends(get_pdf_parser)]
```

在 `src/main.py` 的 `lifespan` 里（`yield` 之前）加入：

```python
    from src.services.arxiv.factory import make_arxiv_client
    from src.services.pdf_parser.factory import make_pdf_parser_service

    app.state.arxiv_client = make_arxiv_client()
    app.state.pdf_parser = make_pdf_parser_service()
    logger.info("Services initialized: arXiv API client, PDF parser")
```

---

## 5.10 本周验证

### 启动 Airflow

```bash
# 确保 postgres 已在运行
docker compose up -d postgres

# 构建并启动 Airflow（首次构建较慢）
docker compose up -d --build airflow

# 看日志，等 webserver + scheduler 起来
docker compose logs -f airflow
```

打开 **http://localhost:8080**，用 `admin`/`admin` 登录，应能看到 `arxiv_paper_ingestion` DAG。

> 完整触发 DAG 需要 Week 3–4 的 opensearch / indexing 模块就绪。现在你可以单独验证"抓取 + 解析"。

### 不依赖 OpenSearch 的"抓取 + 解析"验证脚本

在项目根目录新建临时脚本 `verify_week2.py`（**仅导入 Week 1/2 模块，不导入 metadata_fetcher**）：

```python
import asyncio

from src.services.arxiv.factory import make_arxiv_client
from src.services.pdf_parser.factory import make_pdf_parser_service


async def main() -> None:
    arxiv = make_arxiv_client()
    parser = make_pdf_parser_service()

    # 抓取最近 3 篇 cs.AI 论文（限速 3s，请耐心）
    papers = await arxiv.fetch_papers(max_results=3)
    print(f"Fetched {len(papers)} papers")
    for p in papers:
        print(f"  - {p.arxiv_id}: {p.title[:70]}")

    if not papers:
        return

    # 下载并解析第一篇
    pdf_path = await arxiv.download_pdf(papers[0])
    print(f"Downloaded PDF to: {pdf_path}")

    if pdf_path:
        content = await parser.parse_pdf(pdf_path)
        if content:
            print(f"Parsed sections: {len(content.sections)}")
            print(f"Raw text length: {len(content.raw_text)} chars")
            for s in content.sections[:3]:
                print(f"  section: {s.title[:60]}")


if __name__ == "__main__":
    asyncio.run(main())
```

运行（PDF 缓存目录默认 `./data/arxiv_pdfs`）：

```bash
uv run python verify_week2.py
```

期望：看到抓取的论文标题、下载路径、解析出的章节数与全文字符数。验证完可删除该脚本：

```bash
rm verify_week2.py
```

> 第一次解析 PDF 时 Docling 会下载/加载模型，**较慢且较吃内存**，属正常现象。

---

## 5.11 本章小结

你已经有了：

- ✅ arXiv 抓取客户端（限速、重试、流式下载、XML 解析）。
- ✅ Docling PDF 解析（校验、章节抽取、优雅降级）。
- ✅ `PaperRepository`（按 arxiv_id upsert 去重）。
- ✅ `MetadataFetcher` 异步重叠流水线编排器。
- ✅ 完整的 Airflow 每日管道（5 个任务 + 模块化任务函数）。

**Week 2 里程碑**：能从 arXiv 抓论文、解析 PDF、入库。但论文还"搜不了"——下一章 [`06-week3-opensearch-bm25.md`](06-week3-opensearch-bm25.md) 把内容索引进 OpenSearch，实现 BM25 关键词检索，给整个 RAG 打下检索地基。
