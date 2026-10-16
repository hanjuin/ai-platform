from fastapi import APIRouter, Depends,HTTPException
from fastapi.responses import StreamingResponse
from app.db.session import SessionLocal
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.models.db_models import ConversationSession, ConversationMessage, FlaggedQuestion  # no auth imports needed
from app.models.schemes import SessionResponse, MessageResponse, ChatRequest,ChatResponse
from app.dependencies import get_db
from app.services.search_service import hybrid_search, build_context
from app.services.llm_services import generate_answer, generate_answer_stream
from app.services.greeting_service import classify_message, GREETING_RESPONSE

CONTEXT_CHUNKS = 8
FALLBACK_PHRASE = "Oops! Looks like Han forgot to document this one. I've added it to his ever-growing list of things to do — right after his coffee break."

router = APIRouter(prefix="/conversation", tags=["conversation"])


@router.post("/chat/sessions/", response_model= SessionResponse)
def create_sessions(
    db: Session = Depends(get_db),
):
    conversation_session = ConversationSession(
        user_id = None
    )
    
    db.add(conversation_session)
    db.commit()
    db.refresh(conversation_session)
    
    return conversation_session
    
    
@router.get("/chat/{session_id}/", response_model= list[MessageResponse])
def get_message_responses(
    session_id: int,
    db: Session = Depends(get_db),
):
    message = db.query(ConversationMessage).filter(ConversationMessage.session_id == session_id).order_by(ConversationMessage.created_at).all()

    return message

@router.post("/{session_id}", response_model=ChatResponse)
async def chat_with_history(
    session_id: int,
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    session = db.query(ConversationSession).filter(ConversationSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    past_messages = (
        db.query(ConversationMessage).filter(ConversationMessage.session_id == session_id).order_by(ConversationMessage.created_at).all()
    )

    history = [{"role":m.role,"content":m.content} for m in past_messages]

    # Greeting detection — short-circuit before retrieval
    msg_type = await run_in_threadpool(classify_message, request.message)
    if msg_type == "greeting":
        db.add(ConversationMessage(session_id=session_id, role="user", content=request.message))
        db.add(ConversationMessage(session_id=session_id, role="assistant", content=GREETING_RESPONSE))
        db.commit()
        return ChatResponse(answer=GREETING_RESPONSE, sources=[])

    db.add(ConversationMessage(
        session_id=session_id,
        role="user",
        content=request.message
    ))
    db.commit()

    # Conversation endpoints are unauthenticated, so retrieval is not scoped to an owner
    reranked = await hybrid_search(db, request.message, None)
    context, sources = await build_context(db, reranked, CONTEXT_CHUNKS)
    
    answer = await run_in_threadpool(generate_answer, context, request.message, history)
    
    db.add(ConversationMessage(
        session_id = session_id,
        role="assistant",
        content=answer
    ))
    
    if FALLBACK_PHRASE in answer:
        db.add(FlaggedQuestion(question=request.message, session_id=session_id))
    
    db.commit()
    
    return ChatResponse(answer=answer,sources=sources)


@router.post("/{session_id}/stream")
async def chat_stream(session_id: int,
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    session = db.query(ConversationSession).filter(ConversationSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    past_messages = (
        db.query(ConversationMessage).filter(ConversationMessage.session_id == session_id).order_by(ConversationMessage.created_at).all()
    )

    history = [{"role":m.role,"content":m.content} for m in past_messages]

    # Greeting detection — short-circuit before retrieval
    msg_type = await run_in_threadpool(classify_message, request.message)
    if msg_type == "greeting":
        save_db = SessionLocal()
        try:
            save_db.add(ConversationMessage(session_id=session_id, role="user", content=request.message))
            save_db.add(ConversationMessage(session_id=session_id, role="assistant", content=GREETING_RESPONSE))
            save_db.commit()
        finally:
            save_db.close()

        def greeting_stream():
            for char in GREETING_RESPONSE:
                yield char

        return StreamingResponse(greeting_stream(), media_type="text/plain")

    db.add(ConversationMessage(
        session_id=session_id,
        role="user",
        content=request.message
    ))
    db.commit()

    # Conversation endpoints are unauthenticated, so retrieval is not scoped to an owner
    reranked = await hybrid_search(db, request.message, None)
    context, sources = await build_context(db, reranked, CONTEXT_CHUNKS)
    
    collected = []
    
    def generate():
        for token in generate_answer_stream(context, request.message, history):
            collected.append(token)
            yield token
        full_response= "".join(collected)
        save_db = SessionLocal()
        try:
            save_db.add(ConversationMessage(
                session_id=session_id,
                role="assistant",
                content=full_response
            ))
            
            if FALLBACK_PHRASE in full_response:
                save_db.add(FlaggedQuestion(question=request.message, session_id=session_id))
        
            save_db.commit()
        finally:
            save_db.close()

    return StreamingResponse(generate(), media_type="text/plain")