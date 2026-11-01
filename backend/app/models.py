import os
from datetime import datetime
from sqlalchemy import JSON, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from .core import now, uid

DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

class Base(DeclarativeBase):
    pass

class Record(Base):
    __abstract__ = True
    id: Mapped[str] = mapped_column(primary_key=True, default=uid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

class User(Record):
    __tablename__ = "users"
    username: Mapped[str] = mapped_column(unique=True)
    password: Mapped[str]

class File(Record):
    __tablename__ = "files"
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    purpose: Mapped[str]
    name: Mapped[str]
    key: Mapped[str]
    sha256: Mapped[str]

class Task(Record):
    __tablename__ = "detection_tasks"
    __table_args__ = (UniqueConstraint("owner_id", "key"),)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    file_id: Mapped[str] = mapped_column(ForeignKey("files.id"))
    status: Mapped[str]
    conf: Mapped[float]
    key: Mapped[str | None]
    request_hash: Mapped[str]
    result: Mapped[dict | None] = mapped_column(JSON)

class Thread(Record):
    __tablename__ = "conversations"
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str]

class Turn(Record):
    __tablename__ = "chat_messages"
    __table_args__ = (UniqueConstraint("thread_id", "client_id"),)
    thread_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    client_id: Mapped[str]
    question: Mapped[str]
    attachments: Mapped[list] = mapped_column(JSON)
    reply: Mapped[dict] = mapped_column(JSON)

class Document(Record):
    __tablename__ = "documents"
    title: Mapped[str] = mapped_column(unique=True)
    active: Mapped[int]
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))

class Version(Record):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("doc_id", "number"),)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    number: Mapped[int]
    body: Mapped[str]
    digest: Mapped[str] = mapped_column(index=True)
    author: Mapped[str] = mapped_column(ForeignKey("users.id"))
    embedding_model: Mapped[str]

class Chunk(Record):
    __tablename__ = "chunks"
    version_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id"), index=True)
    location: Mapped[str]
    body: Mapped[str]
    vector: Mapped[list | None] = mapped_column(JSON().with_variant(Vector(DIM), "postgresql"), nullable=True)

class Operation(Record):
    __tablename__ = "knowledge_operations"
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    doc_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"))
    title: Mapped[str]
    expected: Mapped[int]
    candidate: Mapped[str]
    status: Mapped[str]
    version_id: Mapped[str | None] = mapped_column(ForeignKey("document_versions.id"))
    request_id: Mapped[str | None]
