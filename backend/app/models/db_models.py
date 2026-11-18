"""
SQLAlchemy ORM models.

Two tables: Document (one row per uploaded file, tracks pipeline status)
and Clause (one row per segmented clause, gradually filled in as the
pipeline progresses: text -> deontic category/entity -> nothing else,
since embeddings live in Qdrant and graph structure lives in Neo4j —
Postgres is the source of truth for "what clauses exist and what stage
have they reached", not a duplicate of the vector/graph stores).
"""
from __future__ import annotations

import enum
import uuid

from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, Text, JSON, Enum as SAEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.db import Base


class DocumentStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    SEGMENTED = "segmented"
    INDEXED = "indexed"
    CLASSIFIED = "classified"
    GRAPH_BUILT = "graph_built"
    AUDITED = "audited"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    status = Column(SAEnum(DocumentStatus), nullable=False, default=DocumentStatus.UPLOADED)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    report_json = Column(JSON, nullable=True)  # final compiled report, written by report.py

    clauses = relationship("Clause", back_populates="document", cascade="all, delete-orphan")


class Clause(Base):
    __tablename__ = "clauses"

    id = Column(String, primary_key=True)  # matches pipeline.segmentation.Clause.id, e.g. "clause_0007"
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    order = Column(Integer, nullable=False)
    label = Column(String, nullable=True)  # detected numbering, e.g. "3.2", "(a)"
    text = Column(Text, nullable=False)

    # Filled in by the deontic classifier (pipeline/deontic.py)
    deontic_category = Column(String, nullable=True)  # Obligation | Permission | Prohibition | Penalty
    deontic_entity = Column(String, nullable=True)     # who/what the clause binds
    deontic_confidence = Column(String, nullable=True)  # kept as string; local-LLM confidence is qualitative

    document = relationship("Document", back_populates="clauses")
