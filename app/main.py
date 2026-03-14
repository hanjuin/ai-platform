import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.models.db_models import Base, User, UserRole
from app.db.session import engine, SessionLocal
from app.routers import documents, auth, users, search, chat, conversation
from app.services.security import hash_password
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.dependencies import get_db
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Document Intelligence API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://www.hanjuin.com/"],
    allow_methods=["GET","POST"],
    allow_headers=["Content-Type"],
)

@app.on_event("startup")
def startup():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(bind=engine)

    with engine.connect() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding "
            "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
        ))
        conn.commit()

    _seed_admin()


def _seed_admin():
    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "admin")

    db = SessionLocal()
    try:
        if not db.query(User).filter(User.username == username).first():
            db.add(User(
                username=username,
                hashed_password=hash_password(password),
                role=UserRole.admin,
                is_active=True,
            ))
            db.commit()
    finally:
        db.close()

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(chat.router)
app.include_router(conversation.router)

@app.get("/health")
def health_check():
    return {"status" : "ok"}

@app.get("/health/db")
def db_health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"db" : "ok"}

