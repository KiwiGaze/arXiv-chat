# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

**arXiv Chat** is a local-first RAG research assistant (Week 1–2 implemented at this commit). The FastAPI API exposes `/api/v1/health`; full search/RAG features are added in later tutorial weeks.

### Prerequisites

- **Python 3.12** and **uv** (dependency manager; see `pyproject.toml` + `uv.lock`)
- **Docker** with Compose v2 for the backing services

### Dependency refresh (automatic)

On VM startup, `uv sync --frozen` runs via the update script. Ensure `~/.local/bin` is on `PATH` for `uv` (installed via https://astral.sh/uv/install.sh if missing).

### Docker daemon (cloud VMs)

Docker is not managed by systemd in this environment. If `docker` commands fail with permission or connection errors:

```bash
sudo dockerd > /tmp/dockerd.log 2>&1 &
sleep 2
sudo chmod 666 /var/run/docker.sock
```

Storage driver is `fuse-overlayfs` (`/etc/docker/daemon.json`).

### Environment file

Copy `.env.example` to `.env` before starting Compose. Default values work for local dev; `JINA_API_KEY` and `TELEGRAM__BOT_TOKEN` are only needed for later weeks.

### Running services

Core stack (matches README quick start):

```bash
docker compose up -d postgres opensearch opensearch-dashboards ollama redis api
curl http://localhost:8000/api/v1/health
```

Optional: `airflow` (port 8080) for ingestion DAGs; Langfuse stack for Week 6 monitoring.

### Local API (hot reload, without rebuilding the container)

```bash
uv run uvicorn src.main:app --reload --port 8000
```

Use host URLs in `.env` (`localhost`) when running the API on the host while DB/search run in Docker.

### Lint, test, build

See `.github/workflows/ci.yml` and `README.md`:

| Task | Command |
|------|---------|
| Format check | `uv run ruff format --check` |
| Lint | `uv run ruff check` |
| Type check | `uv run mypy src/` |
| Unit tests | `uv run pytest --ignore=tests/integration -q` (no `tests/` dir yet at early commits) |
| API image | `docker compose build api` |

### Useful URLs

| Service | URL |
|---------|-----|
| API docs (Swagger) | http://localhost:8000/docs |
| Health | http://localhost:8000/api/v1/health |
| OpenSearch | http://localhost:9200 |
| OpenSearch Dashboards | http://localhost:5601 |
| Ollama | http://localhost:11434 |
| Airflow | http://localhost:8080 (admin/admin) |

### Gotchas

- The `api` service `depends_on` postgres, opensearch, and redis — all three must be healthy before the API container starts.
- OpenSearch can take ~60s on first boot; the API retries connection at startup.
- `tests/`, `Makefile`, and `gradio_launcher.py` are referenced in tutorials but may not exist until later weeks.
- Integration tests (`tests/integration`) require a live OpenSearch on `:9200`; the CI job is disabled (`if: false`).
