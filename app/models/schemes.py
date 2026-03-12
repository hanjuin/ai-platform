from pydantic import BaseModel
from datetime import datetime


class DocumentCreate(BaseModel):
    filename:str
    content:str

class DocumentResponse(BaseModel):
    document_id:int
    filename:str

    class Config:
        from_attributes = True

class SearchResponse(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_header: str | None
    content: str
    similarity: float
    owner_id: int
    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    user_id: int
    username: str
    role: str
    is_active: bool

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    message: str
    session_id: int | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    
class SessionResponse(BaseModel):
    session_id: int
    class Config:
        from_attributes = True
    
class MessageResponse(BaseModel):
    role: str
    content: str
    created_at: datetime
    
    class Config:
        from_attributes = True