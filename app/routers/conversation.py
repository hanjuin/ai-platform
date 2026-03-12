from fastapi import APIRouter, Depends,HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.db_models import ConversationSession, ConversationMessage, User, UserRole
from app.models.schemes import SessionResponse, MessageResponse, ChatRequest,ChatResponse
from app.dependencies import get_db
from app.services.security import get_current_user
from app.services.embedding_service import embedding_service
from app.services.reranker_service import reranker_service
from app.services.query_expansion import expand_query
from app.services.llm_services import generate_answer, generate_hypothetical

SIMILARITY_THRESHOLD = 0.3
OVERFETCH_LIMIT = 20
CONTEXT_CHUNKS = 8
RRF_K = 60

router = APIRouter(prefix="/conversation", tags=["conversation"])


def _run_retrieval(db: Session, embedding_str: str, q: str, owner_filter: str, params: dict) -> dict[int, dict]:
    """
    Vector + FTS retrieval for one query variant. Only retrieves child chunks
    (parent_chunk_id IS NOT NULL).
    """
    vector_sql = text(f"""
        SELECT
            dc.chunk_id,
            dc.content,
            dc.chunk_header,
            dc.parent_chunk_id,
            d.filename,
            1 - (dc.embedding <=> (:embedding)::vector) AS similarity,
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

    vector_rows = db.execute(vector_sql, {**params, "embedding": embedding_str}).fetchall()
    fts_rows = db.execute(fts_sql, {**params, "q": q}).fetchall()

    scores: dict[int, dict] = {}
    for row in vector_rows:
        scores[row.chunk_id] = {
            "chunk_id": row.chunk_id,
            "content": row.content,
            "chunk_header": row.chunk_header,
            "parent_chunk_id": row.parent_chunk_id,
            "filename": row.filename,
            "rrf": 1 / (RRF_K + row.vector_rank),
        }
    for row in fts_rows:
        if row.chunk_id in scores:
            scores[row.chunk_id]["rrf"] += 1 / (RRF_K + row.fts_rank)

    return scores


def _fetch_parent_content(db: Session, parent_chunk_id: int) -> str | None:
    row = db.execute(
        text("SELECT content FROM document_chunks WHERE chunk_id = :id"),
        {"id": parent_chunk_id}
    ).fetchone()
    return row.content if row else None


@router.post("/chat/sessions/", response_model= SessionResponse)
def create_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    conversation_session = ConversationSession(
        user_id = current_user.user_id
    )
    
    db.add(conversation_session)
    db.commit()
    db.refresh(conversation_session)
    
    return conversation_session
    
    
@router.get("/chat/{session_id}/", response_model= list[MessageResponse])
def get_message_responses(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    session = db.query(ConversationSession).filter(ConversationSession.session_id == session_id).first()
    
    if not session or session.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    message = db.query(ConversationMessage).filter(ConversationMessage.session_id == session_id).order_by(ConversationMessage.created_at).all()
    
    return message

@router.post("/{session_id}", response_model=ChatResponse)
async def chat_with_history(
    session_id: int,
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    session = db.query(ConversationSession).filter(ConversationSession.session_id == session_id).first()
    if not session or session.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    past_messages = (
        db.query(ConversationMessage).filter(ConversationMessage.session_id == session_id).order_by(ConversationMessage.created_at).all()
    )
    
    history = [{"role":m.role,"content":m.content} for m in past_messages]
    
    db.add(ConversationMessage(
        session_id=session_id,
        role="user",
        content=request.message
    ))
    db.commit()
    
    queries = await run_in_threadpool(expand_query, request.message)
    hypothetical = await run_in_threadpool(generate_hypothetical, request.message)
    
    owner_filter = "" if current_user.role == UserRole.admin else "AND d.owner_id = :user_id"
    base_params: dict = {
        "threshold": SIMILARITY_THRESHOLD,
        "overfetch": OVERFETCH_LIMIT,
    }
    
    if current_user.role != UserRole.admin:
        base_params["user_id"] = current_user.user_id
        
    
    merged: dict[int, dict] = {}
    for variant in queries + [hypothetical]:
        embedding = await run_in_threadpool(embedding_service.generate_embedding, variant)
        vec = embedding[0] if isinstance(embedding[0], list) else embedding
        embedding_str = f"[{','.join(map(str,vec))}]"
        variant_scores = await run_in_threadpool(
            _run_retrieval,db,embedding_str,variant,owner_filter,base_params
        )
        
        for chunk_id, chunk in variant_scores.items():
            if chunk_id in merged:
                merged[chunk_id]["rrf"] += chunk["rrf"]
            else:
                merged[chunk_id] = dict(chunk)
                
    candidates = list(merged.values())
    reranked = await run_in_threadpool(reranker_service.rerank, request.message,candidates)
    top_chunks = reranked[:CONTEXT_CHUNKS]
    
    context_parts = []
    
    for i, chunk in enumerate(top_chunks):
        parent_id = chunk.get("parent_chunk_id")
        if parent_id:
            parent_content = await run_in_threadpool(_fetch_parent_content, db, parent_id)
            context_text = parent_content or chunk["content"]
        else:
            context_text = chunk["content"]
        
        header = f" | {chunk['chunk_header']}" if chunk.get("chunk_header") else ""
        context_parts.append(f"[Source {i+1} | {chunk['filename']}{header}]\n{context_text}")
    context = "\n\n".join(context_parts)
    
    answer = await run_in_threadpool(generate_answer, context, request.message, history)
    
    db.add(ConversationMessage(
        session_id = session_id,
        role="assistant",
        content=answer
    ))
    db.commit()
    
    return ChatResponse(answer=answer,sources=[c["filename"] for c in top_chunks])