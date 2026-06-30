import asyncio
from types import SimpleNamespace

from src import main as main_module


class FakeDatabase:
    def __init__(self) -> None:
        self.torn_down = False

    def teardown(self) -> None:
        self.torn_down = True


class FakeOpenSearchClient:
    def health_check(self) -> bool:
        return False


def test_week6_ask_and_stream_routes_are_registered() -> None:
    paths = {route.path for route in main_module.app.routes}

    assert "/api/v1/ask" in paths
    assert "/api/v1/stream" in paths


def test_week6_lifespan_initializes_langfuse_and_cache(monkeypatch) -> None:
    fake_database = FakeDatabase()
    fake_settings = SimpleNamespace()
    fake_langfuse_tracer = object()
    fake_cache_client = object()

    monkeypatch.setattr(main_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(main_module, "make_database", lambda: fake_database)

    import src.services.arxiv.factory as arxiv_factory
    import src.services.cache.factory as cache_factory
    import src.services.embeddings.factory as embeddings_factory
    import src.services.langfuse.factory as langfuse_factory
    import src.services.ollama.factory as ollama_factory
    import src.services.opensearch.factory as opensearch_factory
    import src.services.pdf_parser.factory as pdf_parser_factory

    monkeypatch.setattr(arxiv_factory, "make_arxiv_client", object)
    monkeypatch.setattr(pdf_parser_factory, "make_pdf_parser_service", object)
    monkeypatch.setattr(embeddings_factory, "make_embeddings_service", object)
    monkeypatch.setattr(opensearch_factory, "make_opensearch_client", FakeOpenSearchClient)
    monkeypatch.setattr(ollama_factory, "make_ollama_client", object)
    monkeypatch.setattr(langfuse_factory, "make_langfuse_tracer", lambda: fake_langfuse_tracer)
    monkeypatch.setattr(cache_factory, "make_cache_client", lambda settings: fake_cache_client)

    async def run_lifespan() -> None:
        async with main_module.lifespan(main_module.app):
            assert main_module.app.state.langfuse_tracer is fake_langfuse_tracer
            assert main_module.app.state.cache_client is fake_cache_client

    asyncio.run(run_lifespan())

    assert fake_database.torn_down is True
