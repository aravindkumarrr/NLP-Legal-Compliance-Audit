from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.db import get_db, get_neo4j_driver, get_qdrant_client
from app.models.db_models import Document
from app.models.schemas import (
    AuditResponse,
    ClassifyResponse,
    ComplianceOut,
    ContradictionOut,
    GraphResponse,
    IndexResponse,
    ReportOut,
    UploadResponse,
)
from app.pipeline.extraction import UnsupportedFileType, extract_text
from app.pipeline.segmentation import segment_clauses
from app.services import pipeline_service

router = APIRouter(prefix="/documents", tags=["documents"])


def _get_document_or_404(db: Session, document_id: str) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"No document with id '{document_id}'.")
    return document


@router.post("/preview-clauses")
async def preview_clauses(file: UploadFile = File(...)):
    """
    Stage 1 sanity check: upload a PDF/DOCX and get back the segmented
    clauses without persisting anything. Useful for eyeballing clause
    boundaries on a new document type before committing it to the
    pipeline via /upload.
    """
    file_bytes = await file.read()
    try:
        text = extract_text(file_bytes, file.filename)
    except UnsupportedFileType as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text:
        raise HTTPException(status_code=422, detail="No extractable text found in the uploaded file.")

    clauses = segment_clauses(text)
    return {
        "filename": file.filename,
        "clause_count": len(clauses),
        "clauses": [{"id": c.id, "label": c.label, "order": c.order, "text": c.text} for c in clauses],
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Stage 1, persisted: extract + segment, store the document and its clauses."""
    file_bytes = await file.read()
    try:
        document = pipeline_service.ingest_document(db, file_bytes, file.filename)
    except UnsupportedFileType as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not document.clauses:
        raise HTTPException(status_code=422, detail="No extractable text found in the uploaded file.")

    return UploadResponse(
        document_id=document.id,
        filename=document.filename,
        clause_count=len(document.clauses),
        status=document.status.value,
    )


@router.post("/{document_id}/index", response_model=IndexResponse)
def index_document(document_id: str, db: Session = Depends(get_db)):
    """Stage 2: embed the document's clauses and index them in Qdrant."""
    document = _get_document_or_404(db, document_id)
    count = pipeline_service.run_indexing(db, get_qdrant_client(), document_id)
    return IndexResponse(document_id=document_id, indexed_clauses=count, status=document.status.value)


@router.post("/{document_id}/classify", response_model=ClassifyResponse)
def classify_document(document_id: str, db: Session = Depends(get_db)):
    """Stage 3: deontic classification (Obligation/Permission/Prohibition/Penalty) via the local LLM."""
    document = _get_document_or_404(db, document_id)
    classified, unclassified = pipeline_service.run_classification(db, document_id)
    return ClassifyResponse(
        document_id=document_id,
        classified_clauses=classified,
        unclassified_clauses=unclassified,
        status=document.status.value,
    )


@router.post("/{document_id}/graph", response_model=GraphResponse)
def build_graph(document_id: str, db: Session = Depends(get_db)):
    """Stage 4: build the Deontic Logic Knowledge Graph in Neo4j from classified clauses."""
    document = _get_document_or_404(db, document_id)
    result = pipeline_service.run_graph_build(db, get_neo4j_driver(), document_id)
    return GraphResponse(
        document_id=document_id,
        nodes_created=result.nodes_created,
        edges_created=result.edges_created,
        status=document.status.value,
    )


@router.post("/{document_id}/audit", response_model=AuditResponse)
def audit_document(document_id: str, db: Session = Depends(get_db)):
    """Stages 5-7: GraphRAG retrieval + cross-encoder re-ranking feeding contradiction
    detection and EURLEX57K compliance assessment."""
    document = _get_document_or_404(db, document_id)
    contradictions, compliance_results = pipeline_service.run_audit(
        db, get_qdrant_client(), get_neo4j_driver(), document_id
    )
    return AuditResponse(
        document_id=document_id,
        contradictions=[ContradictionOut(**vars(f)) for f in contradictions],
        compliance_results=[ComplianceOut(**vars(r)) for r in compliance_results],
        status=document.status.value,
    )


@router.get("/{document_id}/report", response_model=ReportOut)
def get_report(document_id: str, db: Session = Depends(get_db)):
    """Stage 8: fetch the compiled report. Runs audit first if it hasn't happened yet,
    so this endpoint always returns something even if the caller skipped straight here
    after /graph."""
    document = _get_document_or_404(db, document_id)
    if document.report_json is not None:
        return ReportOut(**document.report_json)

    contradictions, compliance_results = pipeline_service.run_audit(
        db, get_qdrant_client(), get_neo4j_driver(), document_id
    )
    return pipeline_service.generate_report(db, document_id, contradictions, compliance_results)


@router.post("/run-pipeline", response_model=ReportOut)
async def run_full_pipeline(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Convenience endpoint: runs every stage (upload -> index -> classify ->
    graph -> audit -> report) in one call. Prefer the individual endpoints
    above when you want to show per-stage progress in a UI; this is for
    quick end-to-end testing of the pipeline on a single document.
    """
    file_bytes = await file.read()
    try:
        report = pipeline_service.run_full_pipeline(
            db, get_qdrant_client(), get_neo4j_driver(), file_bytes, file.filename
        )
    except UnsupportedFileType as e:
        raise HTTPException(status_code=400, detail=str(e))
    return report
