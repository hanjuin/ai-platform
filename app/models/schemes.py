from pydantic import BaseModel

class DocumentCreate(BaseModel):
    filename:str
    content:str

class DocumentResponse(BaseModel):
    id:int
    filename:str
    content:str

    class Config:
        from_attributes = True

class SearchResponse(BaseModel):
    id: int
    filename: str
    content: str
    similarity: float

    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str