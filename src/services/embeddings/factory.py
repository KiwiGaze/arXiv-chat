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

    api_key = settings.jina_api_key.strip()
    if not api_key:
        raise ValueError("Jina API key is not configured. Set JINA_API_KEY in the environment or .env.")

    return JinaEmbeddingsClient(api_key=api_key)


make_embeddings_client = make_embeddings_service
