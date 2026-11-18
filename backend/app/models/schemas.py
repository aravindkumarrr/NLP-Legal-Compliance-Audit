"""
Pydantic response/request schemas for the API layer. Kept separate from
db_models.py (SQLAlchemy) on purpose -- API shape and storage shape are
allowed to drift (e.g. the report nests contradictions/compliance the
way a reviewer wants to read them, not the way they're stored).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ClauseOut(BaseModel):
    id: str
    order: int
    label: Optional[str] = None
    text: str
    deontic_category: Optional[str] = None
    deontic_entity: Optional[str] = None

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    clause_count: int
    status: str


class IndexResponse(BaseModel):
    document_id: str
    indexed_clauses: int
    status: str


class ClassifyResponse(BaseModel):
    document_id: str
    classified_clauses: int
    unclassified_clauses: int
    status: str


class GraphResponse(BaseModel):
    document_id: str
    nodes_created: int
    edges_created: int
    status: str


class ContradictionOut(BaseModel):
    clause_id_a: str
    clause_id_b: str
    entity: str
    category_a: str
    category_b: str
    confidence: float
    explanation: str


class ComplianceOut(BaseModel):
    clause_id: str
    status: str  # "compliant" | "gap" | "unclear" | "not_applicable"
    matched_reference: Optional[str] = None
    rationale: str


class AuditResponse(BaseModel):
    document_id: str
    contradictions: list[ContradictionOut]
    compliance_results: list[ComplianceOut]
    status: str


class ReportOut(BaseModel):
    document_id: str
    filename: str
    clause_count: int
    contradiction_count: int
    compliance_gap_count: int
    compliance_not_applicable_count: int
    clauses: list[ClauseOut]
    contradictions: list[ContradictionOut]
    compliance_results: list[ComplianceOut]
