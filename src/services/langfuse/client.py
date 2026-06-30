import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

from src.config import Settings

from langfuse import Langfuse

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Wrapper for Langfuse v3 tracing client with CallbackHandler support."""

    def __init__(self, settings: Settings):
        self.settings = settings.langfuse
        self.client: Optional[Langfuse] = None

        if self.settings.enabled and self.settings.public_key and self.settings.secret_key:
            try:
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
            from langfuse.langchain import CallbackHandler

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
            yield (None, None)
            return

        handler = self.get_callback_handler(
            trace_name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags,
        )

        yield (None, handler)

    def get_trace_id(self, trace=None) -> Optional[str]:
        """
        Get the current trace ID from Langfuse context.

        Args:
            trace: Deprecated, not used in v3

        Returns:
            Trace ID string or None if trace is disabled
        """
        if not self.client:
            return None

        try:
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
            completion_start_time: Optional start time for latency calculation
        """
        if not generation:
            return

        try:
            update_data = {"output": output}

            if usage_metadata:
                if "prompt_tokens" in usage_metadata:
                    update_data["usage"] = {
                        "input": usage_metadata.get("prompt_tokens", 0),
                        "output": usage_metadata.get("completion_tokens", 0),
                        "total": usage_metadata.get("total_tokens", 0),
                    }

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

    @contextmanager
    def trace_rag_request(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """RAGTracer uses this as the top-level trace context manager.

        When enabled, creates a root span and yields it; when disabled or on error, yields None.
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
        """Create a span and return it for later update/end.

        The trace parameter is kept for backward-compatible call signatures.
        Returns None when Langfuse is disabled or on error.
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
        """Write output/metadata to a span and end it; no-op when span is None."""
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
