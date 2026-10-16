from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.services.search_service import hybrid_search
from app.models.schemes import SearchResponse
from app.services.cache_service import get_cache, set_cache
from app.services.security import get_current_user
from app.models.db_models import User

router = APIRouter(prefix="/search", tags=["Search"])


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

    reranked = await hybrid_search(db, q, current_user)

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
