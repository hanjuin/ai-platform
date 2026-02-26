from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.dependencies import get_db
from app.services.embedding_service import embedding_service
from app.models.schemes import SearchResponse
from app.services.cache_service import get_cache, set_cache
from app.services.security import get_current_user
router = APIRouter(prefix="/search", tags=["Search"])

@router.get("/", response_model=list[SearchResponse])
async def search_documents(
    q: str,
    limit: int = 5, 
    offset: int = 0,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    cache_key = f"search:{q}"
    cached = get_cache(cache_key)
    if cached:
        return cached
    
    query_embedding = await run_in_threadpool(
        embedding_service.generate_embedding,
        q
    )
    
    embedding_str = f"[{','.join(map(str, query_embedding))}]"
    sql = text("""
        SELECT id, filename, content,
               1 - (embedding <=> (:query_embedding)::vector) AS similarity
        FROM documents
        WHERE 1 - (embedding <=> (:query_embedding)::vector) > 0.6
        ORDER BY embedding <=> (:query_embedding)::vector
        LIMIT :limit OFFSET :offset
        """)
    
    results = db.execute(
        sql,
        {"query_embedding": embedding_str,
         "limit": limit,
         "offset": offset}
    ).fetchall()

    response = [
        SearchResponse(
            id=row.id,
            filename=row.filename,
            content=row.content,
            similarity=row.similarity
        ) for row in results
    ]

    set_cache(cache_key, [item.model_dump() for item in response])

    return response