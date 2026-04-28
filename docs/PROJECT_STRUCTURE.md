# CP_RAG Project Structure

## Entry Layer

- `main.py`: FastAPI application entrypoint, lifespan management, and homepage mount.
- `app_run.py`: Local development launcher.
- `start.bat`: One-click Windows startup script.

## API Layer

- `api/router.py`: Aggregates route modules.
- `api/routes/health.py`: Infra health checks.
- `api/routes/query.py`: Query endpoint orchestration (session policy + pipeline + LLM call).

## Core Layer

- `core/config.py`: Environment-driven typed settings.
- `core/router.py`: Query routing pipeline (Tier1 cache, Tier2 MySQL, Tier3 Milvus).
- `core/llm_client.py`: OpenAI-compatible LLM HTTP client.
- `core/session_manager.py`: Session state/history utilities and completion heuristic.

## Data Access Layer

- `db/redis_client.py`: Redis singleton client.
- `db/mysql_pool.py`: Async MySQL connection pool singleton.
- `db/milvus_client.py`: Milvus singleton client with async wrappers.

## Data Ingestion Utilities

- `utils/insert_data.py`: Legacy MySQL insertion script.
- `utils/ingest_raw_to_milvus.py`: Scheme-B document ingestion for Milvus
  (text extraction first, OCR fallback for low-text files).

## Frontend

- `web/index.html`: Lightweight single-page chat UI.
