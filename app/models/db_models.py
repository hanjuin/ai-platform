from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, Enum
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector
import enum

Base = declarative_base()

class UserRole(str, enum.Enum):
    admin = "admin"
    user = "user"

class Document(Base):
    __tablename__ = "documents"

    document_id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    embedding = Column(Vector(384)) # 384-dim embedding
    s3_key = Column(String, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.user_id"))
    owner = relationship("User", back_populates="documents")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    chunk_id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.document_id"))
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384)) # 384-dim embedding

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

    role = Column(Enum(UserRole), default=UserRole.user)

    documents = relationship("Document", back_populates="owner")
    