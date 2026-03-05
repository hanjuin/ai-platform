import boto3
import uuid
import os
from fastapi import APIRouter, Depends, BackgroundTasks, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from typing import cast
from app.dependencies import get_db
from app.models.db_models import Document, DocumentChunk
from app.models.schemes import DocumentResponse
from app.services.security import get_current_user
from app.models.db_models import User
from app.modules.chunk_content.md_chunk import chunk_doc_by_headings
from app.core.logging_config import logger
load_dotenv()
s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
BUCKET = os.getenv("S3_BUCKET_NAME")
if BUCKET and BUCKET.startswith("s3://"):
    BUCKET = BUCKET.replace("s3://", "")

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/", response_model=DocumentResponse)
def create_document(
    background_tasks: BackgroundTasks,
    document: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    key = f"document/{uuid.uuid4()}_{document.filename}"
    
    s3.upload_fileobj(document.file, BUCKET, key)
    
    db_document = Document(
        filename=document.filename,
        s3_key=key,
        owner_id=current_user.user_id
    )

    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    background_tasks.add_task(
        run_in_threadpool,
        generate_and_store_embedding,
        db_document.document_id,
    )
    logger.info("Upload Successful!")
    return db_document

def generate_and_store_embedding(doc_id: int):
    from app.db.session import SessionLocal
    from app.models.db_models import Document
    from app.services.embedding_service import embedding_service

    db = SessionLocal()

    try:
        document = db.query(Document).filter(Document.document_id == doc_id).first()
        
        response = s3.get_object(Bucket= BUCKET, Key=document.s3_key)
        object_bytes = cast(bytes, response["Body"].read())
        
        text = object_bytes.decode("utf-8")
        
        chunks = chunk_doc_by_headings(text)
        
        embeddings = embedding_service.generate_embedding(chunks)

        for chunk_text, vector in zip(chunks, embeddings):
            chunk = DocumentChunk(
                document_id=doc_id,
                content=chunk_text,
                embedding=vector
            )
            db.add(chunk)
        
        db.commit()
        logger.info("Embed Successful!")
    finally:
        db.close()