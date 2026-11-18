"""
Stage 8: Reporting Module.

Compiles the outputs of every prior stage -- clause classifications,
contradiction flags, compliance results -- into the structured report a
reviewer actually reads. Deliberately just assembly, no new analysis:
every judgment call has already been made by an earlier stage, so this
module's only job is presenting them coherently and computing simple
rollup counts.
"""
from __future__ import annotations

from app.models.db_models import Clause, Document
from app.models.schemas import ClauseOut, ComplianceOut, ContradictionOut, ReportOut
from app.pipeline.compliance import ComplianceResult
from app.pipeline.contradiction import ContradictionFlag


def compile_report(
    document: Document,
    clauses: list[Clause],
    contradictions: list[ContradictionFlag],
    compliance_results: list[ComplianceResult],
) -> ReportOut:
    clause_outs = [ClauseOut.model_validate(c) for c in clauses]

    contradiction_outs = [
        ContradictionOut(
            clause_id_a=f.clause_id_a,
            clause_id_b=f.clause_id_b,
            entity=f.entity,
            category_a=f.category_a,
            category_b=f.category_b,
            confidence=f.confidence,
            explanation=f.explanation,
        )
        for f in contradictions
    ]

    compliance_outs = [
        ComplianceOut(
            clause_id=r.clause_id,
            status=r.status,
            matched_reference=r.matched_reference,
            rationale=r.rationale,
        )
        for r in compliance_results
    ]

    # Only a genuine LLM-judged "gap" counts here. Clauses the relevance
    # gate marked "not_applicable" (no reference in the corpus actually
    # bears on them) are tallied separately below -- lumping them into
    # gap_count is exactly the inflated-false-positive bug this gate fixes.
    gap_count = sum(1 for r in compliance_results if r.status == "gap")
    not_applicable_count = sum(1 for r in compliance_results if r.status == "not_applicable")

    return ReportOut(
        document_id=document.id,
        filename=document.filename,
        clause_count=len(clauses),
        contradiction_count=len(contradiction_outs),
        compliance_gap_count=gap_count,
        compliance_not_applicable_count=not_applicable_count,
        clauses=clause_outs,
        contradictions=contradiction_outs,
        compliance_results=compliance_outs,
    )
