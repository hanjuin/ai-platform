from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from fastapi.concurrency import run_in_threadpool

from app.dependencies import get_db
from app.services.security import get_current_user
from app.services.search_service import hybrid_search, build_context
from app.services.llm_services import generate_answer
from app.models.schemes import ChatRequest, ChatResponse
from app.models.db_models import User, FlaggedQuestion
from app.services.greeting_service import classify_message, GREETING_RESPONSE

CONTEXT_CHUNKS = 8
FALLBACK_PHRASE = "Oops! Looks like Han forgot to document this one. I've added it to his ever-growing list of things to do — right after his coffee break."

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Greeting detection — short-circuit before retrieval
    msg_type = await run_in_threadpool(classify_message, request.message)
    if msg_type == "greeting":
        return ChatResponse(answer=GREETING_RESPONSE, sources=[])

    reranked = await hybrid_search(db, request.message, current_user)
    context, sources = await build_context(db, reranked, CONTEXT_CHUNKS)

    answer = await run_in_threadpool(
        generate_answer,
        context,
        request.message
    )

    if FALLBACK_PHRASE in answer:
        db.add(FlaggedQuestion(question=request.message, session_id=None))
    
    db.commit()

    return ChatResponse(
        answer=answer,
        sources=sources
    )
