"""
Orchestration layer: ties the pipeline stages (extraction -> segmentation
-> embedding -> deontic classification -> graph build -> audit -> report)
to Postgres persistence, so the API layer (app/api/documents.py) stays
thin and each stage function keeps taking plain data rather than DB
sessions where it doesn't need to.

Each function here is meant to be called from its own API endpoint (so a
long document can be indexed, classified, and audited as separate
requests with visible progress), but run_full_pipeline chains all of
them for convenience / for the "just process this document" case.
"""
from __future__ import annotations

import uuid

from neo4j import Driver
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.db_models import Clause as ClauseRow
from app.models.db_models import Document, DocumentStatus
from app.pipeline import compliance as compliance_pipeline
from app.pipeline import contradiction as contradiction_pipeline
from app.pipeline.deontic import DeonticCategory, classify_clauses
from app.pipeline.embedding import index_clauses
from app.pipeline.extraction import extract_text
from app.pipeline.graph_builder import build_graph_for_document
from app.pipeline.report import compile_report
from app.pipeline.segmentation import Clause, segment_clauses


def ingest_document(db: Session, file_bytes: bytes, filename: str) -> Document:
    """Stage 1: extract text, segment into clauses, persist both."""
    text = extract_text(file_bytes, filename)
    clauses = segment_clauses(text)

    document = Document(id=str(uuid.uuid4()), filename=filename, status=DocumentStatus.SEGMENTED)
    db.add(document)
    db.flush()  # populate document.id relationships before adding clauses

    for clause in clauses:
        db.add(
            ClauseRow(
                id=f"{document.id}_{clause.id}",
                document_id=document.id,
                order=clause.order,
                label=clause.label,
                text=clause.text,
            )
        )
    db.commit()
    db.refresh(document)
    return document


def _clause_rows_to_domain(rows: list[ClauseRow]) -> list[Clause]:
    return [Clause(id=r.id, label=r.label, text=r.text, order=r.order) for r in sorted(rows, key=lambda r: r.order)]


def run_indexing(db: Session, qdrant_client: QdrantClient, document_id: str) -> int:
    """Stage 2: embed clauses and index them in Qdrant."""
    document = db.get(Document, document_id)
    clauses = _clause_rows_to_domain(document.clauses)

    count = index_clauses(qdrant_client, settings.qdrant_collection, document_id, clauses)

    document.status = DocumentStatus.INDEXED
    db.commit()
    return count


def run_classification(db: Session, document_id: str) -> tuple[int, int]:
    """Stage 3: deontic classification, persisted back onto each Clause row."""
    document = db.get(Document, document_id)
    clauses = _clause_rows_to_domain(document.clauses)

    labels = classify_clauses(clauses)
    labels_by_id = {l.clause_id: l for l in labels}

    classified, unclassified = 0, 0
    for row in document.clauses:
        label = labels_by_id.get(row.id)
        if label is None or label.category == DeonticCategory.UNKNOWN:
            unclassified += 1
            continue
        row.deontic_category = label.category.value
        row.deontic_entity = label.entity or "unspecified"
        row.deontic_confidence = label.confidence
        classified += 1

    document.status = DocumentStatus.CLASSIFIED
    db.commit()
    return classified, unclassified


def run_graph_build(db: Session, neo4j_driver: Driver, document_id: str):
    """Stage 4: build the Deontic Logic Knowledge Graph from classified clauses."""
    from app.pipeline.deontic import DeonticLabel  # local import: avoids a cycle with classify_clauses' return type

    document = db.get(Document, document_id)
    clauses = _clause_rows_to_domain(document.clauses)
    labels = [
        DeonticLabel(
            clause_id=row.id,
            category=DeonticCategory(row.deontic_category) if row.deontic_category else DeonticCategory.UNKNOWN,
            entity=row.deontic_entity,
            confidence=row.deontic_confidence or "low",
            raw_response="",
        )
        for row in document.clauses
    ]

    result = build_graph_for_document(neo4j_driver, document_id, clauses, labels)

    document.status = DocumentStatus.GRAPH_BUILT
    db.commit()
    return result


def run_audit(db: Session, qdrant_client: QdrantClient, neo4j_driver: Driver, document_id: str):
    """Stages 6-7: rerank-backed contradiction detection + EURLEX compliance assessment."""
    document = db.get(Document, document_id)

    clause_texts_by_id = {row.id: row.text for row in document.clauses}
    category_by_clause_id = {
        row.id: DeonticCategory(row.deontic_category) if row.deontic_category else DeonticCategory.UNKNOWN
        for row in document.clauses
    }
    entity_by_clause_id = {row.id: row.deontic_entity or "unspecified" for row in document.clauses}

    contradictions = contradiction_pipeline.detect_contradictions_for_document(
        qdrant_client=qdrant_client,
        neo4j_driver=neo4j_driver,
        collection_name=settings.qdrant_collection,
        document_id=document_id,
        clause_texts_by_id=clause_texts_by_id,
        category_by_clause_id=category_by_clause_id,
        entity_by_clause_id=entity_by_clause_id,
    )

    compliance_results = compliance_pipeline.assess_document_compliance(
        qdrant_client=qdrant_client,
        clause_texts_by_id=clause_texts_by_id,
    )

    document.status = DocumentStatus.AUDITED
    db.commit()
    return contradictions, compliance_results


def generate_report(
    db: Session,
    document_id: str,
    contradictions: list[contradiction_pipeline.ContradictionFlag],
    compliance_results: list[compliance_pipeline.ComplianceResult],
):
    """Stage 8: compile and persist the final report."""
    document = db.get(Document, document_id)
    report = compile_report(document, document.clauses, contradictions, compliance_results)

    document.report_json = report.model_dump()
    db.commit()
    return report


def run_full_pipeline(
    db: Session,
    qdrant_client: QdrantClient,
    neo4j_driver: Driver,
    file_bytes: bytes,
    filename: str,
):
    """Convenience wrapper chaining every stage for a fresh document.
    Prefer the individual stage functions (above) when you want visible
    per-stage progress in the UI, e.g. for a larger document where the
    user shouldn't stare at one spinner for the whole pipeline."""
    document = ingest_document(db, file_bytes, filename)
    run_indexing(db, qdrant_client, document.id)
    run_classification(db, document.id)
    run_graph_build(db, neo4j_driver, document.id)
    contradictions, compliance_results = run_audit(db, qdrant_client, neo4j_driver, document.id)
    return generate_report(db, document.id, contradictions, compliance_results)
