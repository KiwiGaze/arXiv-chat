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
