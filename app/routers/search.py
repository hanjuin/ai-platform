from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.dependencies import get_db
from app.services.embedding_service import embedding_service
from app.services.reranker_service import reranker_service
from app.services.query_expansion import expand_query
from app.models.schemes import SearchResponse
from app.services.cache_service import get_cache, set_cache
from app.services.security import get_current_user
from app.services.llm_services import generate_hypothetical
from app.models.db_models import User, UserRole

SIMILARITY_THRESHOLD = 0.3
OVERFETCH_LIMIT = 20
RRF_K = 60

router = APIRouter(prefix="/search", tags=["Search"])


def _run_retrieval(db: Session, embedding_str: str, q: str, owner_filter: str, params: dict) -> dict[int, dict]:
    """
    Run vector + FTS queries for one query variant. Returns chunk_id → chunk dict with RRF scores.
    Only retrieves child chunks (parent_chunk_id IS NOT NULL).
    """
    vector_sql = text(f"""
        SELECT dc.chunk_id, dc.content, dc.chunk_header, d.document_id, d.filename, d.owner_id,
            1 - (dc.embedding <=> (:query_embedding)::vector) AS similarity,
            ROW_NUMBER() OVER (ORDER BY dc.embedding <=> (:query_embedding)::vector) AS vector_rank
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.document_id
        WHERE dc.parent_chunk_id IS NOT NULL
            AND 1 - (dc.embedding <=> (:query_embedding)::vector) > :threshold
            {owner_filter}
        ORDER BY dc.embedding <=> (:query_embedding)::vector
        LIMIT :overfetch
    """)

    fts_sql = text(f"""
        SELECT dc.chunk_id,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank(to_tsvector('english', dc.content), plainto_tsquery('english', :q)) DESC
            ) AS fts_rank
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.document_id
        WHERE dc.parent_chunk_id IS NOT NULL
            AND to_tsvector('english', dc.content) @@ plainto_tsquery('english', :q)
            {owner_filter}
        LIMIT :overfetch
    """)

    vector_rows = db.execute(vector_sql, {**params, "query_embedding": embedding_str}).fetchall()
    fts_rows = db.execute(fts_sql, {**params, "q": q}).fetchall()

    scores: dict[int, dict] = {}
    for row in vector_rows:
        scores[row.chunk_id] = {
            "chunk_id": row.chunk_id,
            "content": row.content,
            "chunk_header": row.chunk_header,
            "document_id": row.document_id,
            "filename": row.filename,
            "owner_id": row.owner_id,
            "similarity": row.similarity,
            "rrf": 1 / (RRF_K + row.vector_rank),
        }
    for row in fts_rows:
        if row.chunk_id in scores:
            scores[row.chunk_id]["rrf"] += 1 / (RRF_K + row.fts_rank)

    return scores


@router.get("/", response_model=list[SearchResponse])
async def search_documents(
    q: str,
    limit: int = 5,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    cache_key = f"search:{current_user.user_id}:{current_user.role}:{q}:{limit}:{offset}"
    cached = get_cache(cache_key)
    if cached:
        return cached

    # Multi-query expansion
    queries = await run_in_threadpool(expand_query, q)
    hypothetical = await run_in_threadpool(generate_hypothetical, q)

    owner_filter = "" if current_user.role == UserRole.admin else "AND d.owner_id = :user_id"
    base_params: dict = {
        "threshold": SIMILARITY_THRESHOLD,
        "overfetch": OVERFETCH_LIMIT,
    }
    if current_user.role != UserRole.admin:
        base_params["user_id"] = current_user.user_id

    # Retrieve for each query variant, merge with RRF
    merged: dict[int, dict] = {}
    for variant in queries + [hypothetical]:
        embedding = await run_in_threadpool(embedding_service.generate_embedding, variant)
        embedding_str = f"[{','.join(map(str, embedding[0]))}]"
        variant_scores = await run_in_threadpool(
            _run_retrieval, db, embedding_str, variant, owner_filter, base_params
        )
        for chunk_id, chunk in variant_scores.items():
            if chunk_id in merged:
                merged[chunk_id]["rrf"] += chunk["rrf"]
            else:
                merged[chunk_id] = dict(chunk)

    # Cross-encoder rerank using the original query
    candidates = list(merged.values())
    reranked = await run_in_threadpool(reranker_service.rerank, q, candidates)

    page = reranked[offset: offset + limit]

    response = [
        SearchResponse(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            filename=item["filename"],
            chunk_header=item["chunk_header"],
            content=item["content"],
            similarity=item["similarity"],
            owner_id=item["owner_id"],
        )
        for item in page
    ]

    set_cache(cache_key, [item.model_dump() for item in response])
    return response
