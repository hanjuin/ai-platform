from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi.concurrency import run_in_threadpool

from app.dependencies import get_db
from app.services.security import get_current_user
from app.services.embedding_service import embedding_service
from app.services.llm_services import generate_answer
from app.models.schemes import ChatRequest, ChatResponse
from app.models.db_models import User, UserRole

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query_embedding = await run_in_threadpool(
        embedding_service.generate_embedding,
        request.message
    )

    embedding_str = f"[{','.join(map(str,query_embedding))}]"

    if current_user.role == UserRole.admin:
        sql = text("""
            SELECT content, filename
            FROM documents
            ORDER BY embedding <=> (:embedding)::vector
            LIMIT 5 
        """)
        params = {"embedding": embedding_str}
    else:
        sql = text("""
            SELECT content, filename
            FROM documents
            WHERE owner_id = :user_id
            ORDER BY embedding <=> (:embedding)::vector
            LIMIT 5 
        """)
        params = {
            "embedding": embedding_str,
            "user_id": current_user.user_id
        }
    results = db.execute(sql, params).fetchall()

    context = "\n\n".join([row.content for row in results])

    answer = await run_in_threadpool(
        generate_answer,
        context,
        request.message
    )

    return ChatResponse(
        answer=answer,
        sources=[row.filename for row in results]
    )