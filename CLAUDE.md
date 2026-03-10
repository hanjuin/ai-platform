# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment setup
make venv        # Create .venv
make install     # Install requirements.txt into .venv

# Run the app
make run         # uvicorn with --reload at http://localhost:8000

# Docker (PostgreSQL 15 + pgvector on port 5433, Redis on 6379)
make docker-up
make docker-down
make docker-reset   # Full wipe: down + remove volumes + up
make docker-logs

# DB shell
make db-shell

# Update requirements
make freeze
```

No test or lint commands are configured yet.

## Environment Variables

Required at runtime (not committed):
- `DATABASE_URL` — defaults to `postgresql://postgres:1234@localhost:5433/aidoc`
- `OPENAI_API_KEY` — for the `/chat/` endpoint (GPT-4o-mini)
- `S3_BUCKET_NAME` — set to `han-ai-platform` in `.env`
- `AWS_REGION` — required for S3 operations

## Architecture

This is a multi-tenant RAG (Retrieval-Augmented Generation) platform built with FastAPI. Users upload markdown documents, which are chunked, embedded, and stored in PostgreSQL with pgvector. A chat endpoint answers questions using semantic retrieval + GPT-4o-mini.

### Request Flow

1. **Upload** (`POST /documents/`): File → S3 → background job → `md_chunk.py` parses markdown by headings → `semantic_chunk.py` splits large sections by cosine similarity → `embedding_service.py` encodes chunks → stored in `document_chunks` with 384-dim vectors.

2. **Search** (`GET /search/`): Query → embed → pgvector `<=>` L2 distance → top-k chunks. Admin users see all documents; regular users see only their own. Results cached in Redis (5-min TTL).

3. **Chat** (`POST /chat/`): Same retrieval as search → chunks formatted with `[Source X | filename]` attribution → passed to GPT-4o-mini via `llm_services.py`.

### Key Files

| File | Role |
|------|------|
| [app/main.py](app/main.py) | FastAPI app init, router registration, startup (creates tables + enables pgvector extension) |
| [app/models/db_models.py](app/models/db_models.py) | SQLAlchemy models: `User`, `Document`, `DocumentChunk` |
| [app/models/schemes.py](app/models/schemes.py) | Pydantic request/response schemas |
| [app/routers/chat.py](app/routers/chat.py) | RAG chat endpoint — retrieves chunks, formats context, calls LLM |
| [app/routers/documents.py](app/routers/documents.py) | Upload handler: S3 + background embedding job |
| [app/routers/search.py](app/routers/search.py) | Semantic search with role-based filtering |
| [app/services/embedding_service.py](app/services/embedding_service.py) | Wraps `all-MiniLM-L6-v2` (384-dim vectors) |
| [app/services/llm_services.py](app/services/llm_services.py) | GPT-4o-mini calls via OpenAI SDK |
| [app/services/cache_service.py](app/services/cache_service.py) | Redis client, 5-min TTL |
| [app/services/security.py](app/services/security.py) | bcrypt password hashing, HS256 JWT (30-min expiry) |
| [app/modules/chunk_content/md_chunk.py](app/modules/chunk_content/md_chunk.py) | Markdown → sections by heading, token-aware splitting (max 200 tokens via tiktoken CL100K) |
| [app/modules/chunk_content/semantic_chunk.py](app/modules/chunk_content/semantic_chunk.py) | Splits large sections by sentence-level cosine similarity (threshold: 0.6) |
| [app/core/logging_config.py](app/core/logging_config.py) | Centralized logging setup |
| [docker-compose.yml](docker-compose.yml) | PostgreSQL 15 (ankane/pgvector image) + Redis 7 |

### Auth & Roles

- `POST /auth/register`: First user named `"admin"` auto-gets admin role; others get `"user"`.
- `POST /auth/login`: Returns JWT. Token required for all document/search/chat endpoints.
- Admin endpoints (`GET/DELETE /users/`) require `role == "admin"`.

### Data Models

- **User**: `user_id`, `username`, `hashed_password`, `is_active`, `role`
- **Document**: `document_id`, `filename`, `s3_key`, `owner_id`
- **DocumentChunk**: `chunk_id`, `document_id`, `content`, `embedding` (Vector 384)

Vector similarity query pattern used in search and chat:
```python
func.cast(1 - (DocumentChunk.embedding.op("<=>")(query_vector)), Float)
```
