from fastapi import FastAPI, Depends
from app.models.db_models import Base
from app.db.session import engine
from app.routers import documents, auth, users
from app.services.embedding_service import embedding_service
from app.routers import search
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.dependencies import get_db

app = FastAPI(title="AI Document Intelligence API")

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    _ = embedding_service.model

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(auth.router)
app.include_router(users.router)

@app.get("/health")
def health_check():
    return {"status" : "ok"}

@app.get("/health/db")
def db_health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"db" : "ok"}

