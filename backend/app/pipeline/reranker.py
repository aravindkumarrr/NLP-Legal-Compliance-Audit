"""
Stage 6: Cross-Encoder Re-ranking.

Takes the (over-retrieved) candidate set from graphrag.py and scores
each (query clause, candidate clause) pair jointly with a cross-encoder,
which is slower but substantially more precise than the bi-encoder
similarity scores from stage 2 -- exactly the trade-off described in the
project's method section. This is the last filtering step before a
clause pair reaches contradiction/compliance logic, so precision here
matters more than recall (weak matches that slip through become false
positives a reviewer has to manually dismiss).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.pipeline.graphrag import Candidate


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(settings.cross_encoder_model_name)


@dataclass
class RerankedCandidate:
    clause_id: str
    text: str
    label: str | None
    sources: set[str]
    rerank_score: float


def rerank(
    query_text: str,
    candidates: list[Candidate],
    top_k: int = 5,
    min_score: float | None = None,
) -> list[RerankedCandidate]:
    """
    Scores every candidate against the query clause and returns the
    top_k, sorted descending, optionally filtered by min_score. Returns
    an empty list (not an error) when there are no candidates -- a
    clause with nothing related to it is a valid, common outcome, not a
    failure.
    """
    if not candidates:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [(query_text, c.text) for c in candidates]
    scores = cross_encoder.predict(pairs)

    reranked = [
        RerankedCandidate(
            clause_id=c.clause_id,
            text=c.text,
            label=c.label,
            sources=c.sources,
            rerank_score=float(score),
        )
        for c, score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda r: r.rerank_score, reverse=True)

    if min_score is not None:
        reranked = [r for r in reranked if r.rerank_score >= min_score]

    return reranked[:top_k]
