from pydantic import BaseModel

class DocumentCreate(BaseModel):
    filename:str
    content:str

class DocumentResponse(BaseModel):
    document_id:int
    filename:str
    content:str

    class Config:
        from_attributes = True

class SearchResponse(BaseModel):
    document_id: int
    filename: str
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

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]