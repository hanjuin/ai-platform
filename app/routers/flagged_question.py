from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from datetime import datetime

from app.dependencies import get_db
from app.models.db_models import FlaggedQuestion, Document, DocumentChunk, User
from app.models.schemes import FlaggedQuestionResponse, AdminAnswerRequest
from app.services.security import get_current_admin
from app.services.embedding_service import embedding_service

router = APIRouter(prefix="/flagged-questions", tags=["Flagged Questions"])

@router.get("/", response_model=list[FlaggedQuestionResponse])
def list_flagged_question(
    include_answered: bool=False,
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    query = db.query(FlaggedQuestion)
    
    if not include_answered:
        query = query.filter(FlaggedQuestion.answered.is_(False))
    return query.order_by(FlaggedQuestion.created_at.desc()).all()

@router.post("/{question_id}/answer", response_model=FlaggedQuestionResponse)
async def answer_flagged_question(
    question_id: int,
    payload: AdminAnswerRequest,
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    flagged = db.query(FlaggedQuestion).filter(FlaggedQuestion.id == question_id).first()
    
    if not flagged:
        raise HTTPException(status_code=404, detail="Flagged question not found")
    
    if flagged.answered:
        raise HTTPException(status_code=400, detail="This question is already been answered")
    
    flagged.answered = True
    flagged.answer = payload.answer
    flagged.answered_at = datetime.now()
    
    qa_text = f"Q: {flagged.question}\n\nA: {payload.answer}"
    
    doc = Document(
        filename=f"qa_entry_{question_id}.txt",
        s3_key=f"manual_qa_entry/{question_id}",
        owner_id=None,
        embedding=None
    )
    
    db.add(doc)
    db.flush()
    
    parent_chunk = DocumentChunk(
        document_id=doc.document_id,
        content=qa_text,
        chunk_header="Q&A Manual Entry",
        parent_chunk_id=None,
        embedding=None
    )
    
    db.add(parent_chunk)
    db.flush()
    
    vector = await run_in_threadpool(embedding_service.generate_embedding, qa_text)
    vec = vector[0]
    
    child_chunk = DocumentChunk(
        document_id=doc.document_id,
        content=qa_text,
        chunk_header="Q&A Manual Entry",
        parent_chunk_id=parent_chunk.chunk_id,
        embedding=vec
    )
    
    db.add(child_chunk)
    db.commit()
    db.refresh(flagged)
    
    return flagged