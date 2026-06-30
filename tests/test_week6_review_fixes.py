from contextlib import contextmanager

from src.services.langfuse.client import LangfuseTracer
from src.services.langfuse.tracer import RAGTracer


class _FakeSpan:
    def __init__(self):
        self.outputs = []
        self.ended = 0

    def update(self, **kwargs):
        if "output" in kwargs:
            self.outputs.append(kwargs["output"])

    def end(self):
        self.ended += 1


class _FakeLangfuseTracer:
    """Minimal stand-in for LangfuseTracer that records span lifecycle."""

    def __init__(self):
        self.client = object()

    def create_span(self, trace=None, name="span", input_data=None, metadata=None):
        return _FakeSpan()

    def update_span(self, span, output=None, metadata=None, level=None, status_message=None):
        if span and output is not None:
            span.update(output=output)

    @contextmanager
    def trace_rag_request(self, query, user_id=None, session_id=None, metadata=None):
        yield None

    def flush(self):
        pass


def _tracer_without_init():
    """Construct LangfuseTracer skipping __init__ (settings not needed for span-lifecycle tests)."""
    tracer = LangfuseTracer.__new__(LangfuseTracer)
    tracer.client = object()
    return tracer


def test_update_span_does_not_end_span():
    tracer = _tracer_without_init()
    span = _FakeSpan()

    tracer.update_span(span=span, output={"embedding_duration_ms": 1.2})

    assert span.ended == 0
    assert {"embedding_duration_ms": 1.2} in span.outputs


def test_end_span_still_ends_span():
    """Sibling method keeps its end; only update_span loses it."""
    tracer = _tracer_without_init()
    span = _FakeSpan()

    tracer.end_span(span=span, output={"x": 1})

    assert span.ended == 1


def test_trace_search_ends_span_once():
    """trace_search's context manager ends its span exactly once (regression guard)."""
    rag_tracer = RAGTracer(_FakeLangfuseTracer())

    with rag_tracer.trace_search(None, "q", 5) as span:
        rag_tracer.end_search(span, [], [], 0)

    assert span.ended == 1


def test_trace_embedding_does_not_emit_success_true():
    rag_tracer = RAGTracer(_FakeLangfuseTracer())

    with rag_tracer.trace_embedding(None, "query") as span:
        pass  # success path

    assert span.outputs, "expected at least one update"
    assert "success" not in span.outputs[-1]
    assert "embedding_duration_ms" in span.outputs[-1]


def test_trace_embedding_failure_not_overwritten_with_true():
    """Caller writes success=False; finally must not clobber with True."""
    rag_tracer = RAGTracer(_FakeLangfuseTracer())

    with rag_tracer.trace_embedding(None, "query") as span:
        rag_tracer.tracer.update_span(span, output={"success": False, "error": "boom"})

    assert span.outputs[-1].get("success") is not True


# --- Fix #2: search_mode must reflect actual retrieval mode ---


class _FakeEmbeddings:
    def __init__(self, fail=False):
        self.fail = fail

    async def embed_query(self, query):
        if self.fail:
            raise RuntimeError("embed failed")
        return [0.1, 0.2]


class _FakeOpenSearch:
    def __init__(self):
        self.last_use_hybrid = None

    def search_unified(self, *, query, query_embedding, size, from_, categories, use_hybrid, min_score):
        self.last_use_hybrid = use_hybrid
        return {"hits": [{"arxiv_id": "2401.00001", "chunk_text": "x"}], "total": 1}


def test_search_mode_is_bm25_when_embedding_fails():
    import asyncio

    from src.routers.ask import _prepare_chunks_and_sources
    from src.schemas.api.ask import AskRequest

    opensearch = _FakeOpenSearch()
    request = AskRequest(query="q", use_hybrid=True)
    chunks, sources, ids, mode = asyncio.run(
        _prepare_chunks_and_sources(request, opensearch, _FakeEmbeddings(fail=True), RAGTracer(_FakeLangfuseTracer()), None)
    )
    assert mode == "bm25"
    assert opensearch.last_use_hybrid is False


def test_search_mode_is_hybrid_when_embedding_succeeds():
    import asyncio

    from src.routers.ask import _prepare_chunks_and_sources
    from src.schemas.api.ask import AskRequest

    opensearch = _FakeOpenSearch()
    request = AskRequest(query="q", use_hybrid=True)
    chunks, sources, ids, mode = asyncio.run(
        _prepare_chunks_and_sources(request, opensearch, _FakeEmbeddings(fail=False), RAGTracer(_FakeLangfuseTracer()), None)
    )
    assert mode == "hybrid"
    assert opensearch.last_use_hybrid is True


def test_search_mode_is_bm25_when_hybrid_not_requested():
    import asyncio

    from src.routers.ask import _prepare_chunks_and_sources
    from src.schemas.api.ask import AskRequest

    opensearch = _FakeOpenSearch()
    request = AskRequest(query="q", use_hybrid=False)
    chunks, sources, ids, mode = asyncio.run(
        _prepare_chunks_and_sources(request, opensearch, _FakeEmbeddings(fail=False), RAGTracer(_FakeLangfuseTracer()), None)
    )
    assert mode == "bm25"
    assert opensearch.last_use_hybrid is False
