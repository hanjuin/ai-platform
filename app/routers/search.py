from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.dependencies import get_db
from app.services.embedding_service import embedding_service
from app.models.schemes import SearchResponse
from app.services.cache_service import get_cache, set_cache
from app.services.security import get_current_user
from app.models.db_models import User, UserRole
router = APIRouter(prefix="/search", tags=["Search"])

@router.get("/", response_model=list[SearchResponse])
async def search_documents(
    q: str,
    limit: int = 5, 
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    query_embedding = await run_in_threadpool(
        embedding_service.generate_embedding,
        q
    )
    
    cache_key = f"search:{current_user.user_id}:{current_user.role}:{q}:{limit}:{offset}"
    cached = get_cache(cache_key)
    if cached:
        return cached

    embedding_str = f"[{','.join(map(str, query_embedding))}]"

    if current_user.role == UserRole.admin:
        sql = text("""
            SELECT document_id, filename, content, owner_id,
                1 - (embedding <=> (:query_embedding)::vector) AS similarity
            FROM documents
            WHERE
                1 - (embedding <=> (:query_embedding)::vector) > 0.6
            ORDER BY embedding <=> (:query_embedding)::vector
            LIMIT :limit OFFSET :offset
            """)
        params={"query_embedding": embedding_str,
                "limit": limit,
                "offset": offset,
                }
    else:
        sql = text("""
            SELECT document_id, filename, content, owner_id,
                1 - (embedding <=> (:query_embedding)::vector) AS similarity
            FROM documents
            WHERE owner_id = :user_id
                AND 1 - (embedding <=> (:query_embedding)::vector) > 0.6
            ORDER BY embedding <=> (:query_embedding)::vector
            LIMIT :limit OFFSET :offset
            """)
        params={"query_embedding": embedding_str,
                "limit": limit,
                "offset": offset,
                "user_id": current_user.user_id
                }
    
    results = db.execute(
        sql, params
    ).fetchall()

    response = [
        SearchResponse(
            document_id=row.document_id,
            filename=row.filename,
            content=row.content,
            similarity=row.similarity,
            owner_id=row.owner_id
        ) for row in results
    ]

    set_cache(cache_key, [item.model_dump() for item in response])

    return response