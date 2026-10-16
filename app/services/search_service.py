from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi.concurrency import run_in_threadpool

from app.models.db_models import User, UserRole
from app.services.embedding_service import embedding_service
from app.services.reranker_service import reranker_service
from app.services.query_expansion import expand_query
from app.services.llm_services import generate_hypothetical

SIMILARITY_THRESHOLD = 0.3
OVERFETCH_LIMIT = 20
RRF_K = 60

CHUNK_COLUMNS = """
    dc.chunk_id,
    dc.content,
    dc.chunk_header,
    dc.parent_chunk_id,
    d.document_id,
    d.filename,
    d.owner_id,
    1 - (dc.embedding <=> (:embedding)::vector) AS similarity
"""


def owner_scope(current_user: User | None) -> tuple[str, dict]:
    """
    SQL filter + params restricting retrieval to documents the user may see.
    Admins (and unauthenticated callers, which apply no filter) see everything.
    Documents with no owner (admin-answered Q&A entries) are shared with all users.
    """
    if current_user is None or current_user.role == UserRole.admin:
        return "", {}
    return "AND (d.owner_id = :user_id OR d.owner_id IS NULL)", {"user_id": current_user.user_id}


def _run_retrieval(db: Session, embedding_str: str, q: str, owner_filter: str, params: dict) -> dict[int, dict]:
    """
    Vector + FTS retrieval for one query variant, fused with RRF. Only retrieves child chunks
    (parent_chunk_id IS NOT NULL). Chunks found by either method are kept.
    """
    vector_sql = text(f"""
        SELECT {CHUNK_COLUMNS},
            ROW_NUMBER() OVER (ORDER BY dc.embedding <=> (:embedding)::vector) AS vector_rank
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.document_id
        WHERE dc.parent_chunk_id IS NOT NULL
            AND 1 - (dc.embedding <=> (:embedding)::vector) > :threshold
            {owner_filter}
        ORDER BY dc.embedding <=> (:embedding)::vector
        LIMIT :overfetch
    """)

    fts_sql = text(f"""
        SELECT {CHUNK_COLUMNS},
            ROW_NUMBER() OVER (
                ORDER BY ts_rank(to_tsvector('english', dc.content), plainto_tsquery('english', :q)) DESC
            ) AS fts_rank
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.document_id
        WHERE dc.parent_chunk_id IS NOT NULL
            AND to_tsvector('english', dc.content) @@ plainto_tsquery('english', :q)
            {owner_filter}
        ORDER BY fts_rank
        LIMIT :overfetch
    """)

    query_params = {**params, "embedding": embedding_str, "q": q}
    vector_rows = db.execute(vector_sql, query_params).fetchall()
    fts_rows = db.execute(fts_sql, query_params).fetchall()

    scores: dict[int, dict] = {}
    for rows, rank_attr in ((vector_rows, "vector_rank"), (fts_rows, "fts_rank")):
        for row in rows:
            rrf = 1 / (RRF_K + getattr(row, rank_attr))
            if row.chunk_id in scores:
                scores[row.chunk_id]["rrf"] += rrf
            else:
                scores[row.chunk_id] = {
                    "chunk_id": row.chunk_id,
                    "content": row.content,
                    "chunk_header": row.chunk_header,
                    "parent_chunk_id": row.parent_chunk_id,
                    "document_id": row.document_id,
                    "filename": row.filename,
                    "owner_id": row.owner_id,
                    "similarity": row.similarity,
                    "rrf": rrf,
                }

    return scores


async def hybrid_search(db: Session, query: str, current_user: User | None) -> list[dict]:
    """
    Multi-query expansion + HyDE → hybrid retrieval per variant → RRF merge →
    cross-encoder rerank on the original query. Returns child chunks, best first.
    """
    queries = await run_in_threadpool(expand_query, query)
    hypothetical = await run_in_threadpool(generate_hypothetical, query)

    owner_filter, owner_params = owner_scope(current_user)
    base_params: dict = {
        "threshold": SIMILARITY_THRESHOLD,
        "overfetch": OVERFETCH_LIMIT,
        **owner_params,
    }

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

    candidates = list(merged.values())
    return await run_in_threadpool(reranker_service.rerank, query, candidates)


def _fetch_parent_contents(db: Session, parent_ids: list[int]) -> dict[int, str]:
    if not parent_ids:
        return {}
    rows = db.execute(
        text("SELECT chunk_id, content FROM document_chunks WHERE chunk_id = ANY(:ids)"),
        {"ids": parent_ids}
    ).fetchall()
    return {row.chunk_id: row.content for row in rows}


async def build_context(db: Session, reranked: list[dict], limit: int) -> tuple[str, list[str]]:
    """
    Parent-child: take the best `limit` distinct parent sections (several child hits from the
    same section count once) and return the LLM context plus deduplicated source filenames.
    """
    top_chunks: list[dict] = []
    seen_parents: set[int] = set()
    for chunk in reranked:
        parent_key = chunk.get("parent_chunk_id") or chunk["chunk_id"]
        if parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)
        top_chunks.append(chunk)
        if len(top_chunks) == limit:
            break

    parent_ids = [c["parent_chunk_id"] for c in top_chunks if c.get("parent_chunk_id")]
    parents = await run_in_threadpool(_fetch_parent_contents, db, parent_ids)

    context_parts = []
    for i, chunk in enumerate(top_chunks):
        context_text = parents.get(chunk.get("parent_chunk_id")) or chunk["content"]
        header = f" | {chunk['chunk_header']}" if chunk.get("chunk_header") else ""
        context_parts.append(
            f"[Source {i+1} | {chunk['filename']}{header}]\n{context_text}"
        )

    sources = list(dict.fromkeys(c["filename"] for c in top_chunks))
    return "\n\n".join(context_parts), sources
