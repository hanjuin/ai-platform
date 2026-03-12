from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, Enum, DateTime
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector
import enum
from datetime import datetime

Base = declarative_base()

class UserRole(str, enum.Enum):
    admin = "admin"
    user = "user"

class Document(Base):
    __tablename__ = "documents"

    document_id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    embedding = Column(Vector(1536)) # 384-dim embedding
    s3_key = Column(String, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.user_id"))
    owner = relationship("User", back_populates="documents")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    chunk_id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.document_id"))
    content = Column(Text, nullable=False)
    chunk_header = Column(Text, nullable=True)   # breadcrumb path, not embedded
    parent_chunk_id = Column(Integer, ForeignKey("document_chunks.chunk_id"), nullable=True)
    # NULL embedding = parent chunk (full section, used as LLM context)
    # Non-null embedding = child chunk (subchunk, used for retrieval)
    embedding = Column(Vector(1536))

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

    role = Column(Enum(UserRole), default=UserRole.user)

    documents = relationship("Document", back_populates="owner")
    
class ConversationSession(Base):
    __tablename__ = "conversation_session"
    
    session_id = Column(Integer,primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
class ConversationMessage(Base):
    __tablename__ = "conversation_message"
    
    message_id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("conversation_session.session_id"))
    role = Column(String)
    content = Column(String)
    created_at = Column(DateTime, default=datetime.now)
    