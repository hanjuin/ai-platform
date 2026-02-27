from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.models.db_models import Document
from app.models.schemes import DocumentCreate, DocumentResponse
from app.services.embedding_service import embedding_service
from app.services.security import get_current_user
from app.models.db_models import User


router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/", response_model=DocumentResponse)
def create_document(
    document: DocumentCreate,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == current_user).first()
    db_document = Document(
        filename=document.filename,
        content=document.content,
        owner_id=user.id
    )

    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    background_tasks.add_task(
        run_in_threadpool,
        generate_and_store_embedding,
        db_document.id,
        document.content
    )

    return db_document

def generate_and_store_embedding(doc_id: int, content: str):
    from app.db.session import SessionLocal
    from app.models.db_models import Document
    from app.services.embedding_service import embedding_service

    db = SessionLocal()

    try:
        embedding = embedding_service.generate_embedding(content)
        document = db.query(Document).filter(Document.id == doc_id).first()
        document.embedding = embedding
        db.commit()
    finally:
        db.close()