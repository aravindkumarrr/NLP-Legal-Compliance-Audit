"""
Stage 7a: Contradiction Detection.

Flags clause pairs that assign conflicting obligations to the same
entity -- the specific capability the Deontic Logic Knowledge Graph
restructuring was meant to enable (see PDF: "two clauses that assign
conflicting obligations to the same entity are now something we can
detect structurally instead of just semantically").

Detection is rule-based on top of the graph + reranker, not a trained
NLI model -- ContractNLI fine-tuning is the documented upgrade path
once this baseline's false-positive rate is understood on real
documents. The rule is:
  1. Same Entity (from the graph)
  2. Opposing deontic category (Obligation vs Prohibition on the same
     entity is the classic conflict shape -- "shall" vs "shall not")
  3. High textual similarity (cross-encoder score above threshold) --
     otherwise an Obligation and an unrelated Prohibition about a
     different topic would falsely flag just for sharing an entity
  4. Negation-cue mismatch as a secondary confidence signal, not a hard
     gate -- modal phrasing varies too much in real contracts to gate on

`has_conflicting_modality` and `entity_matches` are kept as pure,
unit-testable functions since they're the actual detection logic --
everything else in this module is plumbing to fetch pairs and call them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from neo4j import Driver
from qdrant_client import QdrantClient

from app.pipeline.deontic import DeonticCategory
from app.pipeline.embedding import embed_texts
from app.pipeline.graphrag import retrieve
from app.pipeline.reranker import RerankedCandidate, rerank

# Category pairs that represent a plausible structural conflict when
# bound to the same entity. Obligation-vs-Obligation is included because
# two different obligations on the same entity can still directly
# contradict (e.g. conflicting deadlines), just without the modal
# negation cue that Obligation-vs-Prohibition has.
_CONFLICT_CATEGORY_PAIRS = {
    frozenset({DeonticCategory.OBLIGATION, DeonticCategory.PROHIBITION}),
    frozenset({DeonticCategory.OBLIGATION, DeonticCategory.OBLIGATION}),
}

_NEGATION_CUES = re.compile(r"\b(shall not|must not|may not|is not (?:required|permitted)|prohibited)\b", re.IGNORECASE)

_RERANK_MIN_SCORE = 0.3  # cross-encoder scores below this are treated as unrelated, not conflicting


@dataclass
class ContradictionFlag:
    clause_id_a: str
    clause_id_b: str
    entity: str
    category_a: str
    category_b: str
    confidence: float
    explanation: str


def has_conflicting_modality(text_a: str, text_b: str) -> bool:
    """True if exactly one of the two clauses carries a negation cue --
    the textbook "shall" vs "shall not" pattern. Used only to strengthen
    the explanation/confidence, never as the sole trigger, since a lot of
    real contradictions don't use an explicit negation word at all."""
    return bool(_NEGATION_CUES.search(text_a)) != bool(_NEGATION_CUES.search(text_b))


def _categories_conflict(cat_a: DeonticCategory, cat_b: DeonticCategory) -> bool:
    return frozenset({cat_a, cat_b}) in _CONFLICT_CATEGORY_PAIRS


def detect_contradictions_for_clause(
    qdrant_client: QdrantClient,
    neo4j_driver: Driver,
    collection_name: str,
    document_id: str,
    clause_id: str,
    clause_text: str,
    clause_category: DeonticCategory,
    clause_entity: str,
    category_by_clause_id: dict[str, DeonticCategory],
    entity_by_clause_id: dict[str, str],
) -> list[ContradictionFlag]:
    """
    Runs GraphRAG retrieval + reranking around a single clause, then
    applies the conflict rule to each reranked candidate. Designed to be
    called once per Obligation/Prohibition clause in the document (see
    detect_contradictions_for_document).
    """
    if clause_category not in (DeonticCategory.OBLIGATION, DeonticCategory.PROHIBITION):
        return []

    query_vector = embed_texts([clause_text])[0]
    candidates = retrieve(
        qdrant_client=qdrant_client,
        neo4j_driver=neo4j_driver,
        collection_name=collection_name,
        document_id=document_id,
        query_clause_id=clause_id,
        query_vector=query_vector,
    )
    reranked = rerank(clause_text, candidates, top_k=5, min_score=_RERANK_MIN_SCORE)

    flags: list[ContradictionFlag] = []
    for candidate in reranked:
        other_category = category_by_clause_id.get(candidate.clause_id, DeonticCategory.UNKNOWN)
        other_entity = entity_by_clause_id.get(candidate.clause_id, "unspecified")

        if other_entity != clause_entity:
            continue
        if not _categories_conflict(clause_category, other_category):
            continue

        modality_conflict = has_conflicting_modality(clause_text, candidate.text)
        confidence = candidate.rerank_score * (1.15 if modality_conflict else 1.0)

        flags.append(
            ContradictionFlag(
                clause_id_a=clause_id,
                clause_id_b=candidate.clause_id,
                entity=clause_entity,
                category_a=clause_category.value,
                category_b=other_category.value,
                confidence=round(min(confidence, 1.0), 3),
                explanation=_build_explanation(clause_category, other_category, clause_entity, modality_conflict),
            )
        )
    return flags


def _build_explanation(cat_a: DeonticCategory, cat_b: DeonticCategory, entity: str, modality_conflict: bool) -> str:
    base = f"Both clauses bind '{entity}' as {cat_a.value} and {cat_b.value} on closely related subject matter."
    if modality_conflict:
        base += " One clause uses an affirmative modal ('shall'/'may') while the other uses a negated one ('shall not'/'must not')."
    return base


def detect_contradictions_for_document(
    qdrant_client: QdrantClient,
    neo4j_driver: Driver,
    collection_name: str,
    document_id: str,
    clause_texts_by_id: dict[str, str],
    category_by_clause_id: dict[str, DeonticCategory],
    entity_by_clause_id: dict[str, str],
) -> list[ContradictionFlag]:
    """
    Runs detection for every Obligation/Prohibition clause in the
    document and deduplicates symmetric pairs (A-vs-B and B-vs-A) since
    each clause's own retrieval pass can independently surface the other.
    """
    seen_pairs: set[frozenset[str]] = set()
    all_flags: list[ContradictionFlag] = []

    for clause_id, category in category_by_clause_id.items():
        if category not in (DeonticCategory.OBLIGATION, DeonticCategory.PROHIBITION):
            continue

        flags = detect_contradictions_for_clause(
            qdrant_client=qdrant_client,
            neo4j_driver=neo4j_driver,
            collection_name=collection_name,
            document_id=document_id,
            clause_id=clause_id,
            clause_text=clause_texts_by_id[clause_id],
            clause_category=category,
            clause_entity=entity_by_clause_id.get(clause_id, "unspecified"),
            category_by_clause_id=category_by_clause_id,
            entity_by_clause_id=entity_by_clause_id,
        )

        for flag in flags:
            pair_key = frozenset({flag.clause_id_a, flag.clause_id_b})
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            all_flags.append(flag)

    return all_flags
