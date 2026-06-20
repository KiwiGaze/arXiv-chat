import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from src.exceptions import OllamaException
from src.schemas.api.ask import AskResponse
from src.services.ollama.client import OllamaClient


def test_ask_response_rejects_unknown_search_mode() -> None:
    with pytest.raises(ValidationError):
        AskResponse(
            query="What are transformers?",
            answer="Transformers are neural network architectures.",
            sources=[],
            chunks_used=0,
            search_mode="semantic",
        )


def test_generate_rejects_streaming_mode() -> None:
    settings = SimpleNamespace(ollama_host="http://localhost:11434", ollama_timeout=1)
    client = OllamaClient(settings)

    with pytest.raises(OllamaException, match="generate_stream"):
        asyncio.run(client.generate("gemma4:e2b", "hello", stream=True))
