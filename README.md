# AI Platform

A RAG (Retrieval-Augmented Generation) platform built with FastAPI. Designed to answer questions grounded in uploaded documents, with multi-turn conversation support, hybrid retrieval, and an admin pipeline for handling unanswered questions.

## Features

- **Document ingestion** — Upload files to S3; chunked, embedded, and indexed automatically
- **Hybrid retrieval** — Vector similarity (pgvector HNSW) + full-text search (PostgreSQL `tsvector`), fused via Reciprocal Rank Fusion
- **Multi-query expansion** — GPT-4o-mini generates 4 query variants + a hypothetical document (HyDE) to improve recall
- **Cross-encoder reranking** — Cohere `rerank-v3.5` re-scores candidates after initial retrieval
- **Streaming chat** — Single-turn and multi-turn conversation endpoints with SSE streaming
- **Greeting detection** — Rule-based + LLM classifier short-circuits the RAG pipeline for greetings
- **Unanswered question pipeline** — Fallback responses are auto-flagged; admins can answer them and the answers are embedded for future retrieval
- **Role-based access** — `admin` and `user` roles; non-admin users only query their own documents
- **Redis caching** — Search results cached 5 min; query expansions cached 1 hour

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Reranker | Cohere `rerank-v3.5` |
| Database | PostgreSQL + pgvector |
| Cache | Redis |
| Storage | AWS S3 |
| Auth | JWT (HS256) + bcrypt |

## Getting Started

### Prerequisites

- Python 3.10+
- Docker (for PostgreSQL and Redis)
- AWS account (S3 bucket)
- OpenAI and Cohere API keys

### Setup

```bash
# 1. Create virtual environment and install dependencies
make venv
make install

# 2. Copy and fill in environment variables
cp .env.example .env

# 3. Start PostgreSQL and Redis
make docker-up

# 4. Start the dev server
make run
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=
COHERE_API_KEY=

# Redis
REDIS_URL=redis://localhost:6379

# AWS S3
S3_BUCKET_NAME=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=

# Admin user (seeded on startup)
ADMIN_USERNAME=
ADMIN_PASSWORD=

# JWT signing key — change this in production
SECRET_KEY=supersecretkey

# Optional: override default DB URL
# DATABASE_URL=postgresql://postgres:1234@localhost:5433/aidoc
```

## Request Flow

**Document ingestion** (background task):
```
POST /documents/ → S3 upload → Markdown/semantic chunking → OpenAI embeddings → PostgreSQL (pgvector)
```

**Query/chat**:
```
Query → Greeting check (short-circuit) → Query expansion (4 variants + HyDE)
      → Hybrid retrieval (vector + FTS, RRF fusion) → Cohere reranking
      → GPT-4o-mini answer generation → Response with source chunks
               ↓ (if fallback phrase triggered)
          Auto-flag → FlaggedQuestion table → Admin review/answer → Embedded as Q&A chunk
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/auth/token` | Login, returns JWT |
| `POST` | `/documents/` | Upload a document |
| `GET` | `/documents/` | List documents |
| `POST` | `/search/` | Semantic/hybrid search |
| `POST` | `/chat/` | Single-turn chat |
| `POST` | `/conversation/` | Multi-turn chat (non-streaming) |
| `POST` | `/conversation/stream` | Multi-turn chat (SSE streaming) |
| `GET` | `/flagged-questions/` | List unanswered questions (admin) |
| `POST` | `/flagged-questions/{id}/answer` | Answer a flagged question (admin) |
| `GET` | `/health` | Health check |
| `GET` | `/health/db` | Database health check |

## Project Structure

```
app/
├── main.py                         # App entry point, router registration, admin seeding
├── config.py                       # Settings and environment variables
├── dependencies.py                 # FastAPI dependency injection
├── routers/                        # Endpoint handlers
│   ├── auth.py
│   ├── users.py
│   ├── admin.py
│   ├── documents.py
│   ├── search.py
│   ├── chat.py
│   ├── conversation.py
│   └── flagged_question.py
├── services/                       # Business logic
│   ├── embedding_service.py        # OpenAI embeddings
│   ├── reranker_service.py         # Cohere reranking
│   ├── query_expansion.py          # Multi-query + HyDE
│   ├── llm_services.py             # Answer generation
│   ├── greeting_service.py         # Greeting classification
│   ├── search_service.py           # Hybrid retrieval + RRF
│   ├── cache_service.py            # Redis helpers
│   ├── storage_service.py          # S3 upload/download
│   └── security.py                 # JWT + bcrypt
├── modules/
│   └── chunk_content/              # Markdown + semantic chunking
├── models/
│   ├── db_models.py                # SQLAlchemy ORM models
│   └── schemes.py                  # Pydantic request/response schemas
├── db/                             # Database session
├── core/                           # Logging
└── evaluation/                     # Benchmark dataset generation and scoring
```

## Database Schema

PostgreSQL with `pgvector`. Default: `postgresql://postgres:1234@localhost:5433/aidoc`

| Table | Description |
|-------|-------------|
| `documents` | File metadata and S3 keys |
| `document_chunks` | Child chunks with `embedding vector(1536)`, HNSW index, `tsvector` FTS index |
| `users` | Auth with role (`admin` / `user`) |
| `conversation_session` | Chat sessions |
| `conversation_message` | Per-message history within a session |
| `flagged_questions` | Questions that triggered the fallback; tracks answer and `answered_at` |

## Docker Commands

```bash
make docker-up      # Start PostgreSQL + Redis
make docker-down    # Stop services
make docker-reset   # Wipe volumes and restart
make docker-logs    # Follow service logs
make db-shell       # Open PostgreSQL shell
```

## Evaluation

The evaluation suite lives in `app/evaluation/` and is run directly with Python (no test runner configured).
