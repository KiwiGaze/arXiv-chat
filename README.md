# arXiv Chat

A local-first RAG research assistant that ingests arXiv CS.AI papers, indexes them for hybrid search, and answers questions with cited sources.

## Status

This repository is built week-by-week following the tutorial series in `tutorials/`.

| Week | Topic | Status |
|------|-------|--------|
| 1 | Infrastructure (Docker, FastAPI, PostgreSQL) | Done |
| 2 | Data ingestion (arXiv, Docling, Airflow) | Done |
| 3 | BM25 keyword search (OpenSearch) | Done |
| 4 | Hybrid search (chunking, embeddings, RRF) | Done |
| 5 | RAG + local LLM (Ollama, Gradio) | Done |
| 6 | Monitoring & caching (Langfuse, Redis) | In progress |
| 7 | Agentic RAG (LangGraph, Telegram) | Planned |

## Quick start

```bash
# Install dependencies
uv sync

# Copy environment template and fill in secrets
cp .env.example .env

# Start core services (Week 1)
docker compose up -d postgres opensearch opensearch-dashboards ollama redis api

# Health check
curl http://localhost:8000/api/v1/health
```

## Tutorials

Step-by-step build guide: see [`tutorials/README.md`](tutorials/README.md).

## License

MIT
